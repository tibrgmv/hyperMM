"""The outcome table recast for survival analysis: features known at placement, and
time-to-event targets for the two stages of a fill.

A passive order fills only after reaching the front of its queue (journal, 4
November: strict time priority holds without exception), so

    P(fill) = P(reach the front) · P(meet an aggressor while there),

and each factor is a time-to-event question of its own, censored by the owner's
cancellation or the end of observation:

    front   placement -> front of the queue      every order
    fill    front -> first fill                  orders that reached the front
    total   placement -> first fill              every order

Features describe the market as it stood just before the order arrived -- the book
state one change earlier, trades strictly before placement -- so nothing in them
can know the order's future.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from hypermm.fillsim.models import NS, level_trades, tick

STAGES = ("front", "fill", "total")

WINDOWS = (1, 10)  # seconds of recent trading and price movement
# What is measured, in its own units: ticks and counts. Distances reach thousands of
# ticks, so a model linear in them is ruled by a few far-away orders; the features
# below are their signed logarithms, sign(x)·log(1 + |x|). (On the day of 1 December
# 2025 a Cox model on the raw measurements ranked BTC fills worse than the distance
# from the mid alone; on the logarithms it is within a few points of the ceiling.)
SCALED = {
    "logBehind": "ticksBehind",
    "logFromMid": "ticksFromMid",
    "logSpread": "spreadTicks",
    **{f"logTrades{w}s": f"trades{w}s" for w in WINDOWS},
    **{f"logMove{w}s": f"move{w}s" for w in WINDOWS},
}
RAW = tuple(SCALED.values())
FEATURES = (
    "isAsk",
    "logSize",
    "logAhead",
    "emptyLevel",
    "logBehind",
    "logFromMid",
    "logSpread",
    "imbalance",
    *(f"logTrades{w}s" for w in WINDOWS),
    *(f"logMove{w}s" for w in WINDOWS),
)
# Further book-state features, read only by the ceiling model (fillcheck/ceiling.py):
# depth at the best prices, and activity and price range over the last minute.
EXTRA = ("logBestOwn", "logBestOther", "logTrades60s", "logAbsMove60s")


def signed_log(x: pl.Expr) -> pl.Expr:
    return x.sign() * x.abs().log1p()


SCALE = 1e8  # prices and sizes are integers in units of 1e-8


def _before_placement(top: pl.DataFrame) -> pl.DataFrame:
    """The book as it stood one change before each placement."""
    previous = [
        pl.col(c).shift(1) for c in ("bestBid", "bestAsk", "bestBidVolume", "bestAskVolume")
    ]
    return (
        top.sort("seq")
        .with_columns(*(p.alias(f"{p.meta.output_name()}Before") for p in previous))
        .filter(pl.col("kind") == "new")
        .select(
            "oid", "bestBidBefore", "bestAskBefore", "bestBidVolumeBefore", "bestAskVolumeBefore"
        )
    )


def _recent_trades(table: pl.DataFrame, trades: pl.DataFrame, seconds: int) -> pl.Series:
    """Trade blocks that hit the order's side in [placed - seconds, placed)."""
    counted = (
        trades.select(t=pl.col("ts").cast(pl.Int64), isAsk="isAsk")
        .unique()
        .sort("t")
        .with_columns(n=pl.int_range(1, pl.len() + 1).over("isAsk"))
    )

    def count_until(at: pl.Expr) -> pl.DataFrame:
        probe = table.select("isAsk", t=at).with_row_index().sort("t")
        # both frames are sorted on t, hence within each side too
        found = probe.join_asof(
            counted,
            on="t",
            by="isAsk",
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,
        )
        return found.sort("index")["n"].fill_null(0)

    placed = pl.col("tPlaced").cast(pl.Int64)
    return count_until(placed) - count_until(placed - seconds * NS)


def _recent_move(table: pl.DataFrame, top: pl.DataFrame, seconds: int, step: int) -> pl.Series:
    """How far the mid moved towards the order's side over the `seconds` before it, in
    ticks: positive when the price was coming to the order (falling for a bid)."""
    mids = top.select(
        t=pl.col("ts").cast(pl.Int64), mid=(pl.col("bestBid") + pl.col("bestAsk")) / 2
    ).drop_nulls()

    def mid_at(at: pl.Expr) -> pl.Series:
        probe = table.select(t=at).with_row_index().sort("t")
        found = probe.join_asof(mids, on="t", strategy="backward", allow_exact_matches=False)
        return found.sort("index")["mid"]

    placed = pl.col("tPlaced").cast(pl.Int64)
    change = (mid_at(placed) - mid_at(placed - seconds * NS)) / step
    towards = pl.Series(np.where(table["isAsk"].to_numpy(), 1.0, -1.0))
    return change * towards


def survival_table(table: pl.DataFrame, top: pl.DataFrame, fills: pl.DataFrame) -> pl.DataFrame:
    """One row per order of the outcome table: features at placement and targets.

    Durations are in seconds. `<stage>Event` is true when the stage ended in its
    event, false when it was censored. `censoredBy` tells the two kinds of
    censoring apart: the owner's cancellation and the end of the window.
    """
    step = tick(top)
    trades = level_trades(fills)
    placed = pl.col("tPlaced").cast(pl.Int64)
    end = placed + pl.col("duration")
    front = pl.col("tFront").cast(pl.Int64)
    reached = front.is_not_null() & (front <= end)

    own_best = pl.when(pl.col("isAsk")).then("bestAskBefore").otherwise("bestBidBefore")
    own_volume = (
        pl.when(pl.col("isAsk")).then("bestAskVolumeBefore").otherwise("bestBidVolumeBefore")
    )
    other_volume = (
        pl.when(pl.col("isAsk")).then("bestBidVolumeBefore").otherwise("bestAskVolumeBefore")
    )
    # ticks behind the best price on the order's own side: 0 joins it, < 0 improves it
    behind = (
        pl.when(pl.col("isAsk")).then(pl.col("px") - own_best).otherwise(own_best - pl.col("px"))
    )

    out = (
        table.with_row_index()
        .join(_before_placement(top), on="oid", how="left")
        .sort("index")
        .with_columns(
            logSize=(pl.col("restingSize") / SCALE).log(),
            logAhead=(pl.col("volumeAhead") / SCALE).log1p(),
            emptyLevel=pl.col("volumeAhead") == 0,
            ticksBehind=behind / step,
            # the distance the exponential clock prices a fill by
            ticksFromMid=(
                pl.col("px") - (pl.col("bestBidBefore") + pl.col("bestAskBefore")) / 2
            ).abs()
            / step,
            spreadTicks=(pl.col("bestAskBefore") - pl.col("bestBidBefore")) / step,
            imbalance=(own_volume - other_volume) / (own_volume + other_volume),
            logBestOwn=(own_volume / SCALE).log1p(),
            logBestOther=(other_volume / SCALE).log1p(),
        )
    )
    out = out.with_columns(
        **{f"trades{w}s": _recent_trades(out, trades, w) for w in WINDOWS},
        **{f"move{w}s": _recent_move(out, top, w, step) for w in WINDOWS},
        logTrades60s=_recent_trades(out, trades, 60).log1p(),
        logAbsMove60s=_recent_move(out, top, 60, step).abs().log1p(),
    )
    out = out.with_columns(**{name: signed_log(pl.col(raw)) for name, raw in SCALED.items()})
    return out.select(
        "oid",
        "tPlaced",
        *FEATURES,
        *EXTRA,
        *RAW,
        frontDuration=pl.when(reached).then(front - placed).otherwise(pl.col("duration")) / NS,
        frontEvent=reached,
        fillDuration=pl.when(reached).then(end - front) / NS,
        fillEvent=pl.when(reached).then(pl.col("filled")),
        totalDuration=pl.col("duration") / NS,
        totalEvent=pl.col("filled"),
        censoredBy=pl.when(pl.col("filled"))
        .then(None)
        .when(pl.col("tRemoved").is_not_null())
        .then(pl.lit("cancelled"))
        .otherwise(pl.lit("window")),
    )


def stage_frame(survival: pl.DataFrame, stage: str):
    """The rows of one stage as a pandas frame for a survival model: features, T, E.

    Orders that end in their own block have a duration of zero, which proportional
    hazards cannot take; they are moved a nanosecond on.
    """
    rows = survival.filter(pl.col(f"{stage}Duration").is_not_null())
    return (
        rows.select(
            *FEATURES,
            T=pl.col(f"{stage}Duration").clip(lower_bound=1e-9),
            E=pl.col(f"{stage}Event").cast(pl.Int8),
        )
        .with_columns(pl.col(pl.Boolean).cast(pl.Int8))
        .to_pandas()
    )


def fit_cox(survival: pl.DataFrame, stage: str, penalizer: float = 0.01):
    """Cox proportional hazards on the features at placement, for one stage."""
    from lifelines import CoxPHFitter

    return CoxPHFitter(penalizer=penalizer).fit(stage_frame(survival, stage), "T", "E")


def concordance(survival: pl.DataFrame, stage: str, later) -> float:
    """Harrell's C of a score that should be larger for orders whose event comes later.

    `later` is a column name of the survival table or a fitted Cox model (whose
    partial hazard is larger for orders whose event comes sooner).
    """
    from lifelines.utils import concordance_index

    frame = stage_frame(survival, stage)
    if isinstance(later, str):
        score = frame[later]
    else:
        score = -later.predict_partial_hazard(frame)
    return float(concordance_index(frame["T"], score, frame["E"]))


@dataclass
class FillModel:
    """A fitted Cox model reduced to what a strategy needs at every decision: the
    probability that an order placed now is filled within `horizon` seconds,

        P = 1 - exp(-H0(horizon) · exp((x - mean) · beta)),

    and the probabilities that the training orders joining the best price had, by
    quantile, as thresholds."""

    beta: np.ndarray
    mean: np.ndarray
    baseline: float  # cumulative baseline hazard at the horizon
    horizon: float
    at_best: np.ndarray  # features of training orders that joined the best price
    thresholds: dict[float, float] = None

    def for_size(self, size: float) -> FillModel:
        """Thresholds for orders of `size` coins: the training orders at the best price,
        as if each had been of that size -- size is a strong feature, and a strategy's
        lot is far smaller than the typical order."""
        features = self.at_best.copy()
        features[:, FEATURES.index("logSize")] = np.log(size)
        hazard = np.exp((features - self.mean) @ self.beta)
        probabilities = 1 - np.exp(-self.baseline * hazard)
        thresholds = {q: float(np.quantile(probabilities, q)) for q in (0.25, 0.5, 0.75)}
        return FillModel(
            self.beta, self.mean, self.baseline, self.horizon, self.at_best, thresholds
        )

    def probability(self, features: np.ndarray) -> float:
        return float(1 - np.exp(-self.baseline * np.exp((features - self.mean) @ self.beta)))

    def features_at(self, view, is_ask: bool, px: int, params) -> np.ndarray:
        """The FEATURES of an order placed at `px` now, as survival_table defines them."""
        book, market, t, step = view.book, view.market, view.t, params.tick
        own_best, own, other = (
            (book.ask, book.ask_volume, book.bid_volume)
            if is_ask
            else (book.bid, book.bid_volume, book.ask_volume)
        )
        ahead = market.level_volume(is_ask, px, t)
        behind = (px - own_best) if is_ask else (own_best - px)
        towards = 1.0 if is_ask else -1.0

        def move(seconds: int) -> float:
            before = market.mid(t - seconds * NS)
            return 0.0 if before is None else (book.mid - before) / step * towards

        raw = {
            "ticksBehind": behind / step,
            "ticksFromMid": abs(px - book.mid) / step,
            "spreadTicks": (book.ask - book.bid) / step,
            **{f"trades{w}s": float(market.recent_blocks(is_ask, t, w)) for w in WINDOWS},
            **{f"move{w}s": move(w) for w in WINDOWS},
        }
        values = {
            "isAsk": float(is_ask),
            "logSize": np.log(params.lot_size),
            "logAhead": np.log1p(ahead / SCALE),
            "emptyLevel": float(ahead == 0),
            "imbalance": (own - other) / (own + other) if own + other else 0.0,
            **{name: np.sign(raw[r]) * np.log1p(abs(raw[r])) for name, r in SCALED.items()},
        }
        return np.array([values[f] for f in FEATURES])


def fit_fill_model(
    survival: pl.DataFrame, horizon: float = 5.0, sample: int = 200_000, seed: int = 0
) -> FillModel:
    """Fit the total stage on (a sample of) `survival` and reduce it to a FillModel;
    `for_size` then sets the thresholds for a given order size."""
    drawn = survival.sample(min(sample, survival.height), seed=seed)
    cox = fit_cox(drawn, "total")
    beta = cox.params_[list(FEATURES)].to_numpy()
    mean = cox._norm_mean[list(FEATURES)].to_numpy()  # lifelines centres the covariates
    cumulative = cox.baseline_cumulative_hazard_.iloc[:, 0]
    baseline = float(cumulative[cumulative.index <= horizon].iloc[-1])
    at_best = stage_frame(drawn.filter(pl.col("ticksBehind") == 0), "total")[list(FEATURES)]
    return FillModel(beta, mean, baseline, horizon, at_best.to_numpy(dtype=float))
