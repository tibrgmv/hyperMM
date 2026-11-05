"""Four fill rules against what real passive orders actually did.

Backs the journal entry of 4 November 2026, «Шаг 3». Each rule is asked whether an
order would have been filled within its observed life -- until its first real fill,
its cancellation, or the end of the hour. Orders placed in the first five minutes
are left out: the book is still warming up (journal, 24 August).

Usage: python scripts/report_fill_models.py [--date 20251201] [--hour 12] [--coins SOL BTC]
"""

from __future__ import annotations

import numpy as np
import polars as pl
from _report import arguments, section, share

from hypermm.fillsim import models
from hypermm.pipeline import load_hour, read_changes, read_fills

WARM_UP = 5 * 60 * models.NS
RULES = {"touch": "tTouch", "volume-through": "tVolume", "queue-aware": "tQueue"}
DECILES = 10


def calibration(top: pl.DataFrame, trades: pl.DataFrame) -> dict[str, tuple[float, float]]:
    fits = {
        "weighted": models.clock_fit(top, trades),
        "unweighted": models.clock_fit(top, trades, weighted=False),
    }
    seconds = (top["ts"].max() - top["ts"].min()) / models.NS
    depths = models.reach(top, trades)
    print("    reaching δ = 0.5 / 1.5 / 2.5 ticks from the mid, per side per hour:")
    print("      observed                                 "
          + " / ".join(f"{(depths >= d).sum() / 2:,.0f}" for d in (0.5, 1.5, 2.5)))
    for name, (a, k) in fits.items():
        implied = [a * np.exp(-k * d) * seconds for d in (0.5, 1.5, 2.5)]
        print(f"      {name:<10} A {a:.4f}/s  k {k:.4f}/tick   "
              + " / ".join(f"{v:,.0f}" for v in implied))
    return fits


def rule_table(judged: pl.DataFrame, clocks: dict[str, pl.Series]) -> None:
    real = judged["filled"]
    print(f"    orders {judged.height:,}; filled {share(real.sum(), judged.height)}")
    print("    rule              says filled   × real   false, of unfilled   missed, of filled"
          "   timing vs real fill, ms (p10 / median / p90)")
    for name, column in RULES.items():
        says = judged[f"{column}Within"]
        wrong = judged.filter(~pl.col("filled"))[f"{column}Within"].sum()
        missed = (~judged.filter("filled")[f"{column}Within"]).sum()
        late = (
            judged.filter(pl.col("filled") & pl.col(column).is_not_null())
            .select((pl.col(column) - pl.col("tFirstFill")) / 1e6)
            .to_series()
        )
        print(f"    {name:<16} {share(says.sum(), judged.height):>12} {says.mean() / real.mean():>8.2f}"
              f" {share(wrong, judged.height - real.sum()):>20} {share(missed, real.sum()):>19}"
              f"   {late.quantile(0.1):>8,.0f} / {late.median():,.0f} / {late.quantile(0.9):,.0f}")
    for name, p in clocks.items():
        print(f"    clock, {name:<9} {100 * p.mean():>11.2f}% {p.mean() / real.mean():>8.2f}"
              "   (a probability: no single order is right or wrong)")


def by_decile(judged: pl.DataFrame) -> None:
    labels = [str(i) for i in range(DECILES)]
    grouped = (
        judged.with_columns(
            decile=pl.col("volumeAhead").qcut(DECILES, labels=labels).cast(pl.String).cast(pl.Int32)
        )
        .group_by("decile")
        .agg(
            real=pl.col("filled").mean(),
            touch=pl.col("tTouchWithin").mean(),
            volume=pl.col("tVolumeWithin").mean(),
            queue=pl.col("tQueueWithin").mean(),
            clock=pl.col("pClock").mean(),
        )
        .sort("decile")
    )
    print("    filled, % by decile of volume ahead at placement")
    print("    decile    real   touch  volume   queue   clock")
    for row in grouped.iter_rows():
        print(f"    {row[0]:>6} " + " ".join(f"{100 * v:>7.2f}" for v in row[1:]))


def cancelled_ahead(judged: pl.DataFrame, trades: pl.DataFrame) -> None:
    """Of the volume ahead of an order that reached the front, how much left by
    cancellation rather than by trading? That part is what volume-through cannot see."""
    front = judged.filter(pl.col("tFront").is_not_null() & (pl.col("volumeAhead") > 0))
    level = trades.select(*models.KEY, t="ts", traded="traded")

    def traded_by(at: str, exact: bool) -> pl.Series:
        return (
            front.select(*models.KEY, t=pl.col(at).cast(pl.Int64))
            .with_row_index()
            .sort("t")
            .join_asof(level, on="t", by=models.KEY, allow_exact_matches=exact,
                       check_sortedness=False)
            .sort("index")["traded"]
            .fill_null(0)
        )

    ahead = front["volumeAhead"]
    traded = (traded_by("tFront", True) - traded_by("tPlaced", False)).clip(0, None)
    cancelled = (ahead - traded).clip(0, None)
    print(f"    orders that queued behind someone and reached the front: {front.height:,}")
    print(f"    volume ahead that left by cancellation: {share(cancelled.sum(), ahead.sum())} of all;"
          f" median per order {share((cancelled / ahead).median(), 1)}")


def queue_errors(judged: pl.DataFrame) -> None:
    wrong = judged.filter(~pl.col("filled") & pl.col("tQueueWithin"))
    missed = judged.filter(pl.col("filled") & ~pl.col("tQueueWithin"))
    at_front = wrong["tQueue"] == wrong["tFront"].cast(pl.Int64)
    at_cancel = wrong["tQueue"] == wrong["tRemoved"].cast(pl.Int64)
    print(f"    false fills {wrong.height}: the trade fell in the block where the order"
          f" was cancelled {at_cancel.sum()}, reached the front {at_front.sum()},"
          f" either {(at_front | at_cancel).sum()}")
    print(f"    missed fills {missed.height}: never at the front {missed['tFront'].null_count()}")


def main() -> None:
    args = arguments(__doc__)
    changes = read_changes(args.raw, args.date, args.hour)
    fills = read_fills(args.raw, args.date, args.hour)
    for coin in args.coins:
        hour = load_hour(args.raw, args.date, args.hour, coin, changes=changes, fills=fills)
        trades = models.level_trades(hour.fills)

        section(f"{coin}: Avellaneda–Stoikov clock, calibrated on the hour")
        fits = calibration(hour.top, trades)

        end = pl.col("tPlaced").cast(pl.Int64) + pl.col("duration")
        judged = (
            models.predict(hour.table, hour.top, hour.fills)
            .filter(pl.col("tPlaced") >= hour.top["ts"].min() + WARM_UP)
            .with_columns(
                **{f"{c}Within": (pl.col(c) <= end).fill_null(False) for c in RULES.values()}
            )
        )
        a, k = fits["unweighted"]
        clocks = {
            "weighted": judged["pClock"],
            "unweighted": models.clock(judged, hour.top, a, k),
        }

        section(f"{coin}: each rule against the real outcome")
        rule_table(judged, clocks)
        by_decile(judged)

        section(f"{coin}: where the rules go wrong")
        cancelled_ahead(judged, trades)
        queue_errors(judged)


if __name__ == "__main__":
    main()
