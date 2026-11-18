"""RQ5: how often a cancellation in the aggressor's own block saves a resting order,
and what the saved fills would have been worth.

Backs the journal entry of 5 November 2026, «Шаг 4: RQ5». An encounter is the first
trade at an order's price after it reached the front of its queue, within its
life. Under strict time priority that trade fills it -- unless the order was
cancelled in the same block, which the exchange processes first (journal, 23
August). Orders placed in the first five minutes are left out, as in report_fill_models.

Usage: python scripts/report_block_protection.py [--date 20251201] [--hour 12] [--coins SOL BTC]
"""

from __future__ import annotations

import polars as pl
from _report import arguments, section, share

from hypermm.fillcheck.judge import RULES, judge
from hypermm.fillsim import models
from hypermm.pipeline import load_hour, read_changes, read_fills
from hypermm.toxicity.markout import markout

WARM_UP = 5 * 60 * models.NS
HORIZONS = (1, 10, 60)


def encounters(predicted: pl.DataFrame) -> pl.DataFrame:
    """Orders that met a trade at their price while at the front, and how that ended."""
    removed = pl.col("tRemoved").cast(pl.Int64)
    return judge(predicted).filter("tQueueWithin").with_columns(
        ending=pl.when(pl.col("filled") & (pl.col("tFirstFill") == pl.col("tQueue")))
        .then(pl.lit("filled"))
        .when(~pl.col("filled") & (removed == pl.col("tQueue")))
        .then(pl.lit("saved"))
        .otherwise(pl.lit("other")),
        # reached the front in an earlier block, so it surely led the level when the trade came
        settled=pl.col("tFront").cast(pl.Int64) < pl.col("tQueue"),
    )


def ending_table(met: pl.DataFrame) -> None:
    print("                          encounters     filled      saved      other")
    for name, group in (("all", met), ("front in an earlier block", met.filter("settled"))):
        counts = dict(group.group_by("ending").len().iter_rows())
        print(f"    {name:<25} {group.height:>8,}" + "".join(
            f" {share(counts.get(e, 0), group.height):>10}" for e in ("filled", "saved", "other")
        ))


def block_rule(predicted: pl.DataFrame) -> None:
    """Each rule with and without the exchange's in-block order of processing."""
    unfilled = (~predicted["filled"]).sum()
    real = predicted["filled"].sum()
    print("    rule              × real: ignoring blocks / cancels first"
          "    false fills: ignoring / cancels first")
    plain, ordered = judge(predicted), judge(predicted, cancel_first=True)
    for name, column in RULES.items():
        a, b = plain[f"{column}Within"], ordered[f"{column}Within"]
        wrong_a = (a & ~plain["filled"]).sum()
        wrong_b = (b & ~ordered["filled"]).sum()
        print(f"    {name:<16} {a.sum() / real:>17.3f} / {b.sum() / real:<14.3f}"
              f" {wrong_a:>17,} / {wrong_b:,}   (of {unfilled:,} unfilled)")


def markouts(met: pl.DataFrame, top: pl.DataFrame) -> None:
    """Real fills against the fills the cancellations prevented, at the encounter."""
    groups = {
        "filled": met.filter(pl.col("ending") == "filled"),
        "saved": met.filter(pl.col("ending") == "saved"),
    }
    print("    markout, bps (mean ± s.e.)        filled               saved     saved − filled")
    for seconds in HORIZONS:
        stats = {}
        for name, group in groups.items():
            gain = markout(group.select("px", "isAsk", t="tQueue"), top, seconds).drop_nulls()
            stats[name] = (gain.mean(), gain.std() / gain.len() ** 0.5)
        (fm, fs), (sm, ss) = stats["filled"], stats["saved"]
        gap, gap_se = sm - fm, (fs**2 + ss**2) ** 0.5
        print(f"    after {seconds:>2} s {'':<19} {fm:>+7.3f} ± {fs:.3f}   {sm:>+7.3f} ± {ss:.3f}"
              f"   {gap:>+7.3f}  ({gap / gap_se:+.1f} s.e.)")
    print(f"    (filled {groups['filled'].height:,}, saved {groups['saved'].height:,})")


def main() -> None:
    args = arguments(__doc__)
    changes = read_changes(args.raw, args.date, args.hour)
    fills = read_fills(args.raw, args.date, args.hour)
    for coin in args.coins:
        hour = load_hour(args.raw, args.date, args.hour, coin, changes=changes, fills=fills)
        predicted = models.predict(hour.table, hour.top, hour.fills).filter(
            pl.col("tPlaced") >= hour.top["ts"].min() + WARM_UP
        )
        met = encounters(predicted)

        section(f"{coin}: orders at the front meeting a trade at their price")
        ending_table(met)

        section(f"{coin}: the fill rules with the exchange's in-block order")
        block_rule(predicted)

        section(f"{coin}: what the saved fills would have been worth")
        markouts(met, hour.top)


if __name__ == "__main__":
    main()
