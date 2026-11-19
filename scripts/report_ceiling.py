"""RQ2: the ceiling of predictability of a fill, and each model as a share of it.

Two tiers of information, two scales (hypermm/fillcheck/ceiling.py):

  ex ante   what is known at placement. The exponential clock, a Cox model and
            gradient boosting of the fill hazard on the same features; the boosting
            is the empirical ceiling. Censoring-aware metrics on held-out hours:
            Harrell's C, the IPCW AUC and Brier skill at fixed horizons, the
            right-censored log-likelihood. A second boosting on more features shows
            how far the ceiling moves with the feature set it is conditional on.
  ex post   the realised outcome within the order's observed life. The ceiling is 1:
            given the order-level record, matching is deterministic. The touch,
            volume-through and queue-aware rules, the clock used as a rule, and --
            the bridge between the tiers -- the best that placement-time features
            alone can say about the realised outcome.

Training hours come first in the day, a buffer hour is left out, test hours follow;
orders are drawn at random from each side (seed 0). The clock is calibrated on the
hour before the one it is applied to. Intervals are bootstrap intervals over test
hours (the hour is the cluster).

Usage: python scripts/report_ceiling.py [--date 20251201] [--train 0-10] [--test 12-23]
                                        [--coins SOL ETH BTC] [--sample 1000000]
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from _report import hour_list, section

from hypermm import config
from hypermm.fillcheck import ceiling
from hypermm.fillcheck.ceiling import EDGES, HORIZONS
from hypermm.fillcheck.dataset import hour_path, load_orders, load_sample
from hypermm.fillcheck.survival import EXTRA, FEATURES, fit_cox
from hypermm.pipeline import Reader

RULES = {
    "touch": "touchWithin",
    "volume-through": "volumeWithin",
    "queue-aware": "queueWithin",
    "queue-aware, block": "queueWithinBlock",
}
RESAMPLES = 2_000


def orders(reader: Reader, date: str, hours: list[int], coin: str, sample: int) -> pl.DataFrame:
    """The hours' orders, each with the rate of the clock calibrated on the hour before."""

    def with_clock(frame: pl.DataFrame, hour: int) -> pl.DataFrame:
        before = hour_path(date, hour - 1, coin)
        if hour > 0 and not before.exists():
            load_orders(reader, date, hour - 1, coin)
        source = pl.read_parquet(before, columns=["clockA", "clockK"], n_rows=1) if hour > 0 else frame
        a, k = source.select("clockA", "clockK").row(0)
        return frame.with_columns(clockRate=a * (-k * pl.col("ticksFromMid")).exp())

    return load_sample(reader, date, hours, coin, sample, prepare=with_clock)


def matrix(frame: pl.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    return frame.select(pl.col(c).cast(pl.Float64) for c in columns).to_numpy()


def over_hours(hours: np.ndarray, values: dict[str, np.ndarray], reference: str):
    """Mean over test hours of each model's hourly score and of its share of the
    reference's, with 95% bootstrap intervals over hours."""
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(hours), size=(RESAMPLES, len(hours)))
    out = {}
    for name, v in values.items():
        means = v[draws].mean(axis=1)
        shares = ceiling.share(means, values[reference][draws].mean(axis=1))
        out[name] = (
            v.mean(), *np.percentile(means, [2.5, 97.5]),
            ceiling.share(v.mean(), values[reference].mean()), *np.percentile(shares, [2.5, 97.5]),
        )  # fmt: skip
    return out


def report(reader: Reader, args: argparse.Namespace, coin: str) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier

    train = orders(reader, args.date, args.train, coin, args.sample)
    test = orders(reader, args.date, args.test, coin, args.sample)
    t_train, e_train = train["totalDuration"].to_numpy(), train["totalEvent"].to_numpy()
    t, e = test["totalDuration"].to_numpy(), test["totalEvent"].to_numpy()
    hour = test["hour"].to_numpy()
    base, wide = tuple(FEATURES), tuple(FEATURES) + tuple(EXTRA)

    # --- ex ante: survival curves on the grid, one per model ---------------------
    cox = fit_cox(train, "total")
    boosted = ceiling.HazardBoosting(max_iter=600).fit(matrix(train, base), t_train, e_train)
    widened = ceiling.HazardBoosting(max_iter=600).fit(matrix(train, wide), t_train, e_train)
    curves = {
        "exp. clock": ceiling.clock_survival(test["clockRate"].to_numpy()),
        "Cox": ceiling.cox_survival(cox, matrix(test, base), list(base)),
        "boosting (ceiling)": boosted.survival(matrix(test, base)),
        "boosting, more features": widened.survival(matrix(test, wide)),
    }
    reference = "boosting (ceiling)"
    marginal = ceiling.KaplanMeier(t_train, e_train)
    flat = np.tile(marginal(EDGES), (len(t), 1))  # the same curve for every order
    g = ceiling.censoring(t, e)
    at = {h: int(np.searchsorted(EDGES, h)) for h in HORIZONS}

    section(f"{coin}: ex ante -- what is knowable at placement")
    print(f"    train hours {args.train[0]}-{args.train[-1]}: {train.height:,} orders,"
          f" {100 * e_train.mean():.2f}% filled; test hours {args.test[0]}-{args.test[-1]}:"
          f" {test.height:,} orders, {100 * e.mean():.2f}% filled")
    print(f"    features: {len(base)} ({len(wide)} for 'more features');"
          f" boosting stopped at {boosted.model.n_iter_} / {widened.model.n_iter_} trees")
    print("    model                         C     AUC " + "  AUC ".join(f"{h:g}s" for h in HORIZONS)
          + "    Brier skill " + " / ".join(f"{h:g}s" for h in HORIZONS) + "    log-lik skill")
    ll_flat = ceiling.log_likelihood(t, e, flat)
    for name, s in curves.items():
        c = ceiling.concordance_td(t, e, s)
        aucs = [ceiling.auc_at(t, e, 1 - s[:, at[h]], h, g) for h in HORIZONS]
        skills = [
            1 - ceiling.brier_at(t, e, s[:, at[h]], h, g) / ceiling.brier_at(t, e, flat[:, at[h]], h, g)
            for h in HORIZONS
        ]
        ll = 1 - ceiling.log_likelihood(t, e, s) / ll_flat
        print(f"    {name:<25} {c:>7.3f}   " + "   ".join(f"{a:.3f}" for a in aucs)
              + "       " + " / ".join(f"{k:+.3f}" for k in skills) + f"       {ll:+.4f}")

    section(f"{coin}: ex ante -- share of the ceiling, by test hour")
    hours = np.unique(hour)
    scores = [("C", lambda s, m: ceiling.concordance_td(t[m], e[m], s[m]))]
    for h in HORIZONS:
        scores.append(
            (f"AUC {h:g}s", lambda s, m, h=h: ceiling.auc_at(t[m], e[m], 1 - s[m][:, at[h]], h, g))
        )
    for label, score in scores:
        hourly = {
            name: np.array([score(s, hour == h) for h in hours]) for name, s in curves.items()
        }
        print(f"    {label}: mean over {len(hours)} test hours [95% interval]"
              "        share of the ceiling [95% interval]")
        for name, (m, lo, hi, sh, slo, shi) in over_hours(hours, hourly, reference).items():
            print(f"    {name:<25} {m:.3f} [{lo:.3f}, {hi:.3f}]"
                  f"            {100 * sh:>5.1f}% [{100 * slo:.1f}%, {100 * shi:.1f}%]")

    # --- ex post: the realised outcome within the order's life --------------------
    section(f"{coin}: ex post -- the realised outcome within the order's life (ceiling: 1)")
    base_rate = float(e_train.mean())
    classifier = HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=True, n_iter_no_change=15, random_state=0,
    )  # fmt: skip
    classifier.fit(matrix(train, base), e_train)
    rate = test["clockRate"].to_numpy()
    answers = {
        "placement only, boosting": classifier.predict_proba(matrix(test, base))[:, 1],
        "exp. clock, given the life": 1 - np.exp(-rate * t),
        **{name: test[column].to_numpy().astype(float) for name, column in RULES.items()},
    }
    print("    rule                           × real      AUC   share of attainable   Brier skill")
    for name, p in answers.items():
        auc = ceiling.realised_auc(e, p)
        print(f"    {name:<28} {p.sum() / e.sum():>8.2f} {auc:>8.4f} {100 * ceiling.share(auc, 1.0):>18.1f}%"
              f" {ceiling.brier_skill(e, p, base_rate):>13.3f}")



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--raw", type=Path, default=config.DATA_RAW_DIR)
    parser.add_argument("--date", default="20251201")
    parser.add_argument("--train", type=hour_list, default=hour_list("0-10"))
    parser.add_argument("--test", type=hour_list, default=hour_list("12-23"))
    parser.add_argument("--coins", nargs="+", default=["SOL", "ETH", "BTC"])
    parser.add_argument("--sample", type=int, default=1_000_000)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    reader = Reader(args.raw)

    for coin in args.coins:
        report(reader, args, coin)


if __name__ == "__main__":
    main()
