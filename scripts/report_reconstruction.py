"""Rebuilding the book: level volumes, queue position, time, top of book.

Backs the journal entries of 24 August 2026: «Этап 2, часть первая» (queue
position, warm-up, the blocking checks), «Этап 2, часть вторая» (attaching time)
and «Этап 2, часть третья» (top of book and its by-products).

Usage: python scripts/report_reconstruction.py [--date 20251201] [--hour 12] [--coins SOL BTC]
"""

from __future__ import annotations

import numpy as np
import polars as pl
from _report import Stopwatch, arguments, section, share

from hypermm.ingest.book_diffs import NEW, UPDATE
from hypermm.ingest.codes import Status
from hypermm.ingest.schema import SCALE
from hypermm.pipeline import read_changes, read_statuses
from hypermm.reconstruct import book, queue, timing

TWELFTHS = 12


def queue_position(changes: pl.DataFrame) -> None:
    clock = Stopwatch()
    rebuilt = queue.with_queue_position(changes)
    print(f"all coins, {changes.height:,} changes, rebuilt in {clock}")
    for name, n in queue.check(rebuilt).items():
        print(f"    {name:<22} {n:>8,}")
    placements = rebuilt.filter(pl.col("kind") == NEW).height
    cold = pl.col("firstSeen") & (pl.col("kind") != NEW)
    print(f"    partial fills (updates)   {rebuilt.filter(pl.col('kind') == UPDATE).height:,}")
    print(f"    already resting at open   {rebuilt.filter(cold).height:,}"
          f"   ({share(rebuilt.filter(cold).height, placements)} of placements)")

    print("\n    warm-up: share of changes that are the first sight of an order already")
    print("    resting at open, by twelfth of the file (file order, all coins):")
    by_part = (
        rebuilt.with_columns(part=pl.col("seq") * TWELFTHS // changes.height)
        .group_by("part")
        .agg((cold.mean() * 100).alias("pct"))
        .sort("part")
    )
    print("    " + "  ".join(f"{p:.3f}" for p in by_part["pct"]))

    print("\n    queue position at placement")
    print(f"    {'coin':<5} {'placements':>11} {'empty level':>12} {'median ahead':>14} {'p90 ahead':>12}")
    for coin in ("BTC", "ETH", "SOL"):
        placed = rebuilt.filter((pl.col("coin") == coin) & (pl.col("kind") == NEW))
        ahead = placed["volumeAhead"]
        print(f"    {coin:<5} {placed.height:>11,} {share((ahead == 0).sum(), placed.height):>12}"
              f" {ahead.median() / SCALE:>14,.3f} {ahead.quantile(0.9) / SCALE:>12,.3f}")


def attach_time(rebuilt: pl.DataFrame, statuses: pl.DataFrame, coin: str) -> pl.DataFrame:
    clock = Stopwatch()
    timed = timing.with_time(rebuilt, statuses)
    counts = timing.check(timed)
    print(f"\n{coin}: {timed.height:,} changes, timed in {clock}")
    for name, n in counts.items():
        print(f"    {name:<10} {n:>10,}   {share(n, timed.height):>8}")

    ts = timed["ts"].to_numpy().astype(np.int64)
    print(f"    non-decreasing {bool((np.diff(ts) >= 0).all())}, span {(ts[-1] - ts[0]) / 60e9:.2f} min")

    rejected = statuses.filter(
        pl.col("statusId").is_in([int(s) for s in timing.REJECTIONS if s != Status.TRIGGERED])
    )
    seen = rejected.filter(pl.col("oid").is_in(rebuilt["oid"].implode()))["oid"].n_unique()
    print(f"    orders with a rejection status that appear in book diffs: {seen}")

    # Accuracy of inherited time: the k-th size reduction of an order against the
    # k-th `filled` status of the same order, where the status stream has one.
    filled = (
        statuses.filter(pl.col("statusId") == Status.FILLED)
        .sort("ts")
        .select("oid", pl.col("ts").alias("tsFilled"))
        .with_columns(k=pl.int_range(pl.len()).over("oid"))
    )
    carried = (
        timed.filter((pl.col("kind") == UPDATE) & ~pl.col("tsExact"))
        .sort("seq")
        .select("oid", "ts")
        .with_columns(k=pl.int_range(pl.len()).over("oid"))
    )
    error = (
        carried.join(filled, on=["oid", "k"])
        .select((pl.col("ts").cast(pl.Int64) - pl.col("tsFilled").cast(pl.Int64)).abs() / 1e6)
        .to_series()
    )
    print(f"    inherited time vs k-th `filled` status: n {error.len():,},"
          f" median error {error.median():.1f} ms, within 100 ms {share((error < 100).sum(), error.len())}")
    return timed


def top_of_book(timed: pl.DataFrame, coin: str) -> None:
    clock = Stopwatch()
    top = book.with_top_of_book(timed)
    print(f"\n{coin}: top of book over {top.height:,} changes in {clock}")
    for name, n in book.check(top).items():
        print(f"    {name:<18} {n:>6,}")

    known = top.drop_nulls(["bestBid", "bestAsk"]).with_columns(
        spread=pl.col("bestAsk") - pl.col("bestBid"),
        mid=(pl.col("bestAsk") + pl.col("bestBid")) / 2,
        imbalance=(pl.col("bestBidVolume") - pl.col("bestAskVolume"))
        / (pl.col("bestBidVolume") + pl.col("bestAskVolume")),
    )
    tick = known.filter(pl.col("spread") > 0)["spread"].min()
    one_tick = (known["spread"] == tick).to_numpy()
    lopsided = (known["imbalance"].abs() > 0.5).to_numpy()
    # time each state lasted, for the time-weighted versions of the same shares
    ts = known["ts"].to_numpy().astype(np.int64)
    lasted = np.diff(np.append(ts, ts[-1]))

    print(f"    tick (smallest positive spread)  {tick / SCALE}")
    print(f"    median spread                    {known['spread'].median() / SCALE}")
    print(f"    spread of one tick               {100 * one_tick.mean():.1f}% of changes,"
          f" {100 * (lasted * one_tick).sum() / lasted.sum():.1f}% of time")
    print(f"    mid range                        {known['mid'].min() / SCALE:,.3f} – {known['mid'].max() / SCALE:,.3f}")
    print(f"    median imbalance at the touch    {known['imbalance'].median():+.3f}")
    print(f"    |imbalance| above 0.5            {100 * lopsided.mean():.1f}% of changes,"
          f" {100 * (lasted * lopsided).sum() / lasted.sum():.1f}% of time")


def main() -> None:
    args = arguments(__doc__)
    changes = read_changes(args.raw, args.date, args.hour)

    section("Level volume and queue position at placement")
    queue_position(changes)

    section("Attaching time to the book diff stream")
    timed = {}
    for coin in args.coins:
        statuses = read_statuses(args.raw, args.date, args.hour, coin)
        # rebuilt per coin, as the pipeline does
        own = queue.with_queue_position(changes.filter(pl.col("coin") == coin))
        timed[coin] = attach_time(own, statuses, coin)

    section("Top of book")
    for coin, frame in timed.items():
        top_of_book(frame, coin)


if __name__ == "__main__":
    main()
