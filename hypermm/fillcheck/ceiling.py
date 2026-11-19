"""How much of a fill is knowable: the ceiling of predictability, and the scale it gives.

A fill model's score means little by itself: 0.76 of what? Part of an order's fate
is decided by flow that has not happened when the order is placed, so 1.0 is out
of reach for anything that reads only the state at placement. The ceiling is the
score of the strongest model we can fit on that state; a model's score is then
read as a share of the distance from the uninformative floor to the ceiling.

Two tiers of information give two scales, and they must not be mixed:

    ex ante   the state at placement. Ceiling: gradient boosting of the discrete-time
              fill hazard (below). The exponential clock and the Cox model are
              scored against it, with censoring-aware metrics: an order cancelled
              before it filled is not a miss, its fill time is unknown beyond that.
              (Boosting and a Cox model on log-scaled features are close; boosting
              leads on every metric once concordance is taken time-dependent.)
    ex post   the market record after placement as well. Under strict time priority
              that record *determines* whether a resting order was filled, so the
              ceiling is the truth itself, and the touch, volume-through and
              queue-aware rules are scored against the realised outcome.

The boosting model works on person-period rows: each order contributes one row
per time bin it was at risk in, labelled by whether it was filled in that bin. Any
classifier of those rows is a survival model, S(t|x) = prod (1 - hazard_k(x)), with
no proportional-hazards assumption.

Censoring-aware metrics weight by the inverse probability of censoring (IPCW),
which takes cancellation to be independent of the fill time. It is not: makers
cancel when a fill is about to hurt (journal, 5 November). The ex ante scores are
comparable between models, which share that bias, but are not unbiased estimates.
"""

from __future__ import annotations

import numpy as np

# right edges of the time bins, seconds since placement; block times are ~0.1 s apart
EDGES = np.array(
    [0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 20, 30, 45, 60, 120, 300, 600, 1800]
)
HORIZONS = (1.0, 5.0, 30.0)
FLOOR = 0.5  # the concordance and the AUC of an uninformative score


def bin_of(durations: np.ndarray) -> np.ndarray:
    """The bin a duration falls in; len(EDGES) for one past the last edge."""
    return np.searchsorted(EDGES, durations, side="left")


def person_period(x: np.ndarray, durations: np.ndarray, events: np.ndarray):
    """Rows (features + bin index), labels and weights of the discrete-time hazard.

    An order is at risk in every bin before the one its life ended in, and in that
    one too: with a fill it is the positive row; with a cancellation it was at risk
    for part of the bin only and counts half (the actuarial convention).
    """
    last = bin_of(durations)
    inside = last < len(EDGES)
    rows = last + inside
    order = np.repeat(np.arange(len(durations)), rows)
    k = np.arange(rows.sum()) - np.repeat(np.cumsum(rows) - rows, rows)
    final = k == last[order]
    y = (final & events[order]).astype(np.int8)
    weight = np.where(final & ~events[order], 0.5, 1.0).astype(np.float32)
    return np.column_stack([x[order], k.astype(np.float32)]), y, weight


class HazardBoosting:
    """Gradient boosting of the discrete-time fill hazard on person-period rows."""

    def __init__(self, max_iter: int = 400, seed: int = 0):
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=0.1,
            max_leaf_nodes=31,
            min_samples_leaf=200,
            l2_regularization=1.0,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=seed,
        )
        self.seed = seed

    def fit(self, x: np.ndarray, durations: np.ndarray, events: np.ndarray) -> HazardBoosting:
        x = np.asarray(x, dtype=np.float32)
        held_out = np.random.default_rng(self.seed).random(len(x)) < 0.1  # whole orders
        train = person_period(x[~held_out], durations[~held_out], events[~held_out])
        valid = person_period(x[held_out], durations[held_out], events[held_out])
        self.model.fit(
            train[0], train[1], sample_weight=train[2],
            X_val=valid[0], y_val=valid[1], sample_weight_val=valid[2],
        )  # fmt: skip
        return self

    def survival(self, x: np.ndarray, chunk: int = 100_000) -> np.ndarray:
        """S(t|x) at each of EDGES, one row per order."""
        x = np.asarray(x, dtype=np.float32)
        bins = len(EDGES)
        out = np.empty((len(x), bins), dtype=np.float32)
        k = np.tile(np.arange(bins, dtype=np.float32), chunk)
        for start in range(0, len(x), chunk):
            part = x[start : start + chunk]
            rows = np.column_stack([np.repeat(part, bins, axis=0), k[: len(part) * bins]])
            hazard = self.model.predict_proba(rows)[:, 1].reshape(len(part), bins)
            out[start : start + chunk] = np.cumprod(1 - hazard, axis=1)
        return out


# --- survival curves of the other ex ante models, on the same grid --------------------


def clock_survival(rate: np.ndarray) -> np.ndarray:
    """The exponential clock: S(t) = exp(-rate · t), rate = A·exp(-k·δ) per second."""
    return np.exp(-np.outer(rate, EDGES))


def cox_survival(cox, x: np.ndarray, features: list[str]) -> np.ndarray:
    """S(t|x) = exp(-H0(t) · exp((x - mean) · beta)) of a fitted lifelines Cox model."""
    beta = cox.params_[features].to_numpy()
    mean = cox._norm_mean[features].to_numpy()  # lifelines centres the covariates
    cumulative = cox.baseline_cumulative_hazard_.iloc[:, 0]
    at = np.searchsorted(cumulative.index.to_numpy(), EDGES, side="right") - 1
    baseline = np.where(at >= 0, cumulative.to_numpy()[np.maximum(at, 0)], 0.0)
    return np.exp(-np.outer(np.exp((x - mean) @ beta), baseline))


def mortality(survival: np.ndarray) -> np.ndarray:
    """One risk score per order from a survival curve: the cumulative hazard summed
    over the grid. Larger means sooner."""
    return -np.log(np.clip(survival, 1e-12, 1)).sum(axis=1)


# --- censoring-aware metrics ------------------------------------------------------------


class KaplanMeier:
    """The Kaplan-Meier survival curve of `events` (true: the event; false: censored)."""

    def __init__(self, durations: np.ndarray, events: np.ndarray):
        order = np.argsort(durations, kind="stable")
        t, e = durations[order], events[order]
        self.times, first = np.unique(t, return_index=True)
        died = np.add.reduceat(e.astype(np.float64), first)
        at_risk = len(t) - first
        self.curve = np.cumprod(1 - died / at_risk)

    def __call__(self, t, *, before: bool = False) -> np.ndarray:
        """S(t), or with `before` its left limit S(t-)."""
        i = np.searchsorted(self.times, t, side="left" if before else "right") - 1
        return np.where(i >= 0, self.curve[np.maximum(i, 0)], 1.0)


def censoring(durations: np.ndarray, events: np.ndarray) -> KaplanMeier:
    """The distribution of the censoring time: the event is being censored."""
    return KaplanMeier(durations, ~events)


def auc_at(durations, events, risk, horizon: float, g: KaplanMeier) -> float:
    """Cumulative/dynamic AUC at `horizon`: orders filled by then against orders still
    resting, the filled ones weighted by 1/G(T-). Orders cancelled before the horizon
    are unknown at it and take no part."""
    from sklearn.metrics import roc_auc_score

    cases = events & (durations <= horizon)
    controls = durations > horizon
    keep = cases | controls
    weight = np.where(cases, 1 / np.clip(g(durations, before=True), 1e-3, None), 1.0)
    return float(roc_auc_score(cases[keep], risk[keep], sample_weight=weight[keep]))


def brier_at(durations, events, survival_at, horizon: float, g: KaplanMeier) -> float:
    """IPCW Brier score at `horizon` of predicted survival probabilities."""
    filled = events & (durations <= horizon)
    resting = durations > horizon
    g_then = np.clip(g(durations, before=True), 1e-3, None)
    g_horizon = max(float(g(horizon)), 1e-3)
    loss = filled * survival_at**2 / g_then + resting * (1 - survival_at) ** 2 / g_horizon
    return float(loss.mean())


def log_likelihood(durations, events, survival: np.ndarray) -> float:
    """Mean right-censored log-likelihood on the grid: a fill in bin k has probability
    S(k-1) - S(k); an order censored in bin k is only known to have reached it."""
    last = bin_of(durations)
    padded = np.column_stack([np.ones(len(survival)), survival])
    before = padded[np.arange(len(last)), last]
    inside = last < len(EDGES)
    after = padded[np.arange(len(last)), np.minimum(last + 1, len(EDGES))]
    filled = events & inside
    probability = np.where(filled, before - after, before)
    return float(np.log(np.clip(probability, 1e-12, 1)).mean())


def concordance(durations, events, risk) -> float:
    """Harrell's C of a risk score that should be larger for orders filled sooner."""
    from lifelines.utils import concordance_index

    return float(concordance_index(durations, -np.asarray(risk), events))


def concordance_td(durations, events, survival: np.ndarray) -> float:
    """Time-dependent concordance (Antolini et al.) on the grid: an order filled in a
    bin against every order that outlived that bin, compared by the survival predicted
    for the end of that bin. Pairs ending in the same bin are tied at the grid's
    resolution and left out.

    Harrell's C needs one risk score per order, which a model whose ranking changes
    with the horizon does not have; for a proportional-hazards model the two agree.
    """
    from sklearn.metrics import roc_auc_score

    last = bin_of(durations)
    agreeing = pairs = 0.0
    for k in range(len(EDGES)):
        filled, outlived = events & (last == k), last > k
        weight = float(filled.sum()) * float(outlived.sum())
        if weight == 0:
            continue
        labels = np.r_[np.ones(filled.sum(), dtype=bool), np.zeros(outlived.sum(), dtype=bool)]
        risk = np.r_[1 - survival[filled, k], 1 - survival[outlived, k]]
        agreeing += weight * roc_auc_score(labels, risk)
        pairs += weight
    return agreeing / pairs


def share(score: float, ceiling: float, floor: float = FLOOR) -> float:
    """The share of the attainable signal a score recovers: 0 at the floor, 1 at the ceiling."""
    return (score - floor) / (ceiling - floor)


# --- ex post: a rule against the realised outcome ---------------------------------------


def realised_auc(filled: np.ndarray, predicted: np.ndarray) -> float:
    """AUC of a prediction of the realised outcome; for a yes/no rule this is its
    balanced accuracy, (true positive rate + true negative rate) / 2."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(filled, predicted))


def brier_skill(filled: np.ndarray, predicted: np.ndarray, base_rate: float) -> float:
    """1 - Brier / Brier of always answering `base_rate`: 1 is perfect, 0 no better
    than the base rate, negative worse than it. Unlike the AUC it charges a rule for
    saying "filled" too often."""
    y = filled.astype(np.float64)
    reference = np.mean((base_rate - y) ** 2)
    return float(1 - np.mean((np.asarray(predicted, dtype=np.float64) - y) ** 2) / reference)
