"""Are blocks recoverable from `ts`, and in what order is a block processed?

RQ5 asks what it costs to ignore Hyperliquid's block semantics, which requires
knowing the block a message landed in -- but no stream in the dataset carries a
block number. The only candidate is `ts`. If the exchange stamps every action in
a block with that block's own time, blocks are recoverable by grouping on equal
timestamps; if `ts` is a per-event clock, they are not, and RQ5 needs rethinking.

Passing a second instrument's file for the same hour settles it: blocks belong to
the chain, not to a market, so a block clock must produce the same stamps in both.

The ordering section then measures the sequence events take inside a recovered
block, which is what the market maker's free in-block protection rests on.

Usage: python scripts/check_block_semantics.py <hourly .data.gz> [<same hour, other coin>]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

from hypermm.ingest.codes import Status
from hypermm.ingest.orders import read_hour

NS_PER_MS = 1_000_000
SECONDS_PER_HOUR = 3_600
BLOCK_WINDOW_MS = (50, 200)


def describe_clock(ts: np.ndarray) -> None:
    stamps, per_stamp = np.unique(ts, return_counts=True)
    gaps = np.diff(stamps) / NS_PER_MS
    low, high = BLOCK_WINDOW_MS

    print(f"records              {len(ts):,}")
    print(f"distinct timestamps  {len(stamps):,}"
          f"   -> {SECONDS_PER_HOUR / len(stamps) * 1000:.1f} ms apart on average")
    print(f"records per stamp    median {np.median(per_stamp):,.0f}"
          f"   p90 {np.quantile(per_stamp, 0.9):,.0f}   max {per_stamp.max():,}")
    print(f"gap between stamps   median {np.median(gaps):.1f} ms"
          f"   p10 {np.quantile(gaps, 0.1):.1f}   p90 {np.quantile(gaps, 0.9):.1f}")
    print(f"gaps within {low}-{high}ms"
          f"  {100 * np.mean((gaps >= low) & (gaps <= high)):.1f}%")
    print(f"file order sorted    {bool((np.diff(ts) >= 0).all())}")
    print(f"covered span         {(stamps[-1] - stamps[0]) / NS_PER_MS / 60_000:.1f} min")


def describe_ordering(frame: pl.DataFrame) -> None:
    """How often does one status precede another inside the same block?

    Counted over pairs, not over blocks: a "no violation anywhere in the block"
    test grows harder to pass as blocks grow, which would read a busy instrument
    as obeying a different rule.

    File order is taken as the order the exchange emitted the events, so the
    obvious alternative -- a file merely sorted by order id -- is measured too.
    """
    for earlier, later in ((Status.OPEN, Status.CANCELED), (Status.CANCELED, Status.FILLED)):
        pairs = (
            frame.with_columns(
                is_earlier=(pl.col("statusId") == earlier).cast(pl.Int64),
                is_later=(pl.col("statusId") == later).cast(pl.Int64),
            )
            .with_columns(
                preceding=pl.col("is_earlier").cum_sum().over("ts") - pl.col("is_earlier"),
                in_block=pl.col("is_earlier").sum().over("ts"),
            )
            .filter(pl.col("is_later") == 1)
        )
        total = pairs["in_block"].sum()
        ordered = pairs["preceding"].sum()
        print(f"{earlier.name} before {later.name}"
              f"   {100 * ordered / total:.1f}%   over {total:,} same-block pairs")

    ascending = frame.group_by("ts").agg(
        pl.col("oid").diff().drop_nulls().ge(0).all().alias("ascending")
    )["ascending"].mean()
    print(f"blocks whose oids ascend (an order-id sort would give ~100%): {100 * ascending:.1f}%")


def compare_stamps(left: np.ndarray, right: np.ndarray, names: tuple[str, str]) -> None:
    """Do two instruments of the same hour share their timestamps?"""
    left, right = np.unique(left), np.unique(right)
    shared = len(np.intersect1d(left, right, assume_unique=True))
    union = np.union1d(left, right)
    span = union[-1] - union[0]

    print(f"stamps in {names[0]}      {len(left):,}")
    print(f"stamps in {names[1]}      {len(right):,}")
    print(f"shared                 {shared:,}"
          f"   ({100 * shared / len(left):.2f}% of {names[0]},"
          f" {100 * shared / len(right):.2f}% of {names[1]})")
    print(f"union                  {len(union):,}"
          f"   -> {SECONDS_PER_HOUR / len(union) * 1000:.1f} ms apart on average")
    print(f"shared expected by chance under independent ns clocks:"
          f" {len(left) * len(right) / span:.1e}")


if __name__ == "__main__":
    if not 2 <= len(sys.argv) <= 3:
        sys.exit(__doc__)

    path = Path(sys.argv[1])
    frame = read_hour(path, columns=["ts", "statusId", "oid"])
    print(f"file                 {path.name}\n")
    describe_clock(frame["ts"].to_numpy().astype(np.int64))
    print()
    describe_ordering(frame)

    if len(sys.argv) == 3:
        other = Path(sys.argv[2])
        print()
        compare_stamps(
            frame["ts"].to_numpy(),
            read_hour(other, columns=["ts"])["ts"].to_numpy(),
            (path.stem.split("_")[0], other.stem.split("_")[0]),
        )
