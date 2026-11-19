"""RQ5: how often a cancellation in the aggressor's own block saves a resting order,
and what the saved fills would have been worth.

Backs the journal entry of 5 November 2026, «Шаг 4: RQ5». An encounter is the first
trade at an order's price after it reached the front of its queue, within its
life. Under strict time priority that trade fills it -- unless the order was
cancelled in the same block, which the exchange processes first (journal, 23
August). Orders placed in the first five minutes are left out, as in report_fill_models.

Usage: python scripts/report_block_protection.py [--date 20251201] [--hour 12 | --hours 0-23]
                                                [--coins SOL BTC]

A single --hour is rebuilt on its own, as the journal of 5 November did: the book
starts cold, the first five minutes are left out, outcomes stop at the hour's end.
With --hours (or --chained) each hour is rebuilt between its neighbours where their
files are at hand (pipeline.load_window), and the hours are pooled.
"""

from __future__ import annotations

from collections import Counter

import polars as pl
from _report import arguments, section, share

from hypermm.fillcheck.judge import RULES, judge
from hypermm.fillsim import models
from hypermm.pipeline import WARM_UP, Reader, load_hour, load_window
from hypermm.toxicity.markout import markout

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


def block_counts(predicted: pl.DataFrame) -> dict[str, int]:
    """Fills each rule gives, with and without the exchange's in-block order, and how
    many of them are false."""
    counts = {"real": int(predicted["filled"].sum()), "unfilled": int((~predicted["filled"]).sum())}
    for order, judged in (
        ("plain", judge(predicted)),
        ("ordered", judge(predicted, cancel_first=True)),
    ):
        for name, column in RULES.items():
            within = judged[f"{column}Within"]
            counts[f"{name} {order}"] = int(within.sum())
            counts[f"{name} {order} false"] = int((within & ~judged["filled"]).sum())
    return counts


def block_rule(counts: dict[str, int]) -> None:
    real, unfilled = counts["real"], counts["unfilled"]
    print(
        "    rule              × real: ignoring blocks / cancels first"
        "    false fills: ignoring / cancels first"
    )
    for name in RULES:
        a, b = counts[f"{name} plain"], counts[f"{name} ordered"]
        wrong_a, wrong_b = counts[f"{name} plain false"], counts[f"{name} ordered false"]
        print(
            f"    {name:<16} {a / real:>17.3f} / {b / real:<14.3f}"
            f" {wrong_a:>17,} / {wrong_b:,}   (of {unfilled:,} unfilled)"
        )


def with_markouts(met: pl.DataFrame, top: pl.DataFrame) -> pl.DataFrame:
    """Markout of every encounter, as if it filled at the trade that met it.

    Taken within the hour's own book, so a horizon past the end of the hour is null.
    """
    at = met.select("px", "isAsk", t="tQueue")
    end = int(top["ts"].max())
    return met.with_columns(
        **{f"markout{seconds}": markout(at, top, seconds, until=end) for seconds in HORIZONS}
    )


def markouts(met: pl.DataFrame) -> None:
    """Real fills against the fills the cancellations prevented, at the encounter."""
    groups = {
        "filled": met.filter(pl.col("ending") == "filled"),
        "saved": met.filter(pl.col("ending") == "saved"),
    }
    print("    markout, bps (mean ± s.e.)        filled               saved     saved − filled")
    for seconds in HORIZONS:
        stats = {}
        for name, group in groups.items():
            gain = group[f"markout{seconds}"].drop_nulls()
            stats[name] = (gain.mean(), gain.std() / gain.len() ** 0.5)
        (fm, fs), (sm, ss) = stats["filled"], stats["saved"]
        gap, gap_se = sm - fm, (fs**2 + ss**2) ** 0.5
        print(
            f"    after {seconds:>2} s {'':<19} {fm:>+7.3f} ± {fs:.3f}   {sm:>+7.3f} ± {ss:.3f}"
            f"   {gap:>+7.3f}  ({gap / gap_se:+.1f} s.e.)"
        )
    print(f"    (filled {groups['filled'].height:,}, saved {groups['saved'].height:,})")


def by_hour(met: pl.DataFrame) -> None:
    """The saved share hour by hour: is it one hour's quirk or a steady rate."""
    rows = (
        met.group_by("hour")
        .agg(
            n=pl.len(),
            saved=(pl.col("ending") == "saved").mean(),
            gap=pl.col("markout1").filter(pl.col("ending") == "saved").mean()
            - pl.col("markout1").filter(pl.col("ending") == "filled").mean(),
        )
        .sort("hour")
    )
    print("    hour   encounters    saved    saved − filled, 1 s markout (bps)")
    for hour, n, saved, gap in rows.iter_rows():
        print(f"    {hour:>4} {n:>12,} {100 * saved:>7.2f}%    {gap:>+7.3f}")
    saved = rows["saved"]
    print(
        f"    saved share across hours: min {100 * saved.min():.2f}%,"
        f" median {100 * saved.median():.2f}%, max {100 * saved.max():.2f}%"
    )


def clustered(met: pl.DataFrame) -> None:
    """The gap saved − filled with the hour as the cluster: orders within an hour share
    its market, so the hourly gaps, not the orders, are the independent observations."""
    print("    saved − filled markout, bps: mean of hourly gaps ± s.e. over hours")
    for seconds in HORIZONS:
        column = pl.col(f"markout{seconds}")
        gaps = (
            met.group_by("hour")
            .agg(
                gap=column.filter(pl.col("ending") == "saved").mean()
                - column.filter(pl.col("ending") == "filled").mean()
            )["gap"]
            .drop_nulls()
        )
        se = gaps.std() / gaps.len() ** 0.5
        print(f"    after {seconds:>2} s    {gaps.mean():>+7.3f} ± {se:.3f}  ({gaps.mean() / se:+.1f} s.e.,"
              f" {gaps.len()} hours, negative in {(gaps < 0).sum()})")
    saved = met.group_by("hour").agg(share=(pl.col("ending") == "saved").mean())["share"]
    se = saved.std() / saved.len() ** 0.5
    print(f"    saved share: mean of hours {100 * saved.mean():.2f}% ± {100 * se:.2f}%")


def main() -> None:
    args = arguments(__doc__)
    reader = Reader(args.raw)
    met: dict[str, list[pl.DataFrame]] = {coin: [] for coin in args.coins}
    counts: dict[str, Counter] = {coin: Counter() for coin in args.coins}
    for hour_of_day in args.hours:
        for coin in args.coins:
            if args.hours == [args.hour] and not args.chained:
                hour = load_hour(args.raw, args.date, hour_of_day, coin)
                table = hour.table.filter(pl.col("tPlaced") >= hour.top["ts"].min() + WARM_UP)
            else:
                hour = load_window(reader, args.date, hour_of_day, coin)
                table = hour.table
            predicted = models.predict(table, hour.top, hour.fills)
            met[coin].append(
                with_markouts(encounters(predicted), hour.top).with_columns(
                    hour=pl.lit(hour_of_day)
                )
            )
            counts[coin].update(block_counts(predicted))

    hours = f"{args.date}, hours {args.hours[0]}-{args.hours[-1]}" if len(args.hours) > 1 else ""
    for coin in args.coins:
        pooled = pl.concat(met[coin])
        section(f"{coin}: orders at the front meeting a trade at their price {hours}".rstrip())
        ending_table(pooled)

        section(f"{coin}: the fill rules with the exchange's in-block order")
        block_rule(counts[coin])

        section(f"{coin}: what the saved fills would have been worth")
        markouts(pooled)

        if len(args.hours) > 1:
            section(f"{coin}: hour by hour")
            by_hour(pooled)
            section(f"{coin}: with the hour as the cluster")
            clustered(pooled)


if __name__ == "__main__":
    main()
