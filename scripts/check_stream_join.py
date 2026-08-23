"""Do the order status stream and the book diff stream describe the same orders?

The two streams are written in different formats -- packed binary against JSON
text -- and are parsed by different code paths that share nothing but the target
representation. Every order accepted onto the book therefore appears twice, and
the two descriptions must agree exactly: same side, same price to the last digit,
same size. Disagreement means one of the two readers is wrong.

This is also the only evidence that queue reconstruction is possible at all, since
the book diff stream carries no time of its own and has to borrow `oid` from here.

Usage: python scripts/check_stream_join.py <hourly order .data.gz> <hourly ex<H>.gz> <coin>
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

from hypermm.ingest import book_diffs, orders
from hypermm.ingest.codes import Status


def report(orders_path: Path, diffs_path: Path, coin: str) -> None:
    placed = (
        orders.read_hour(orders_path, columns=["oid", "statusId", "isAsk", "limitPx", "sz"])
        .filter(pl.col("statusId") == Status.OPEN)
        .unique(subset="oid", keep="first")
        .select(
            "oid",
            pl.col("isAsk").alias("isAsk_order"),
            pl.col("limitPx").alias("px_order"),
            pl.col("sz").alias("sz_order"),
        )
    )
    arrived = book_diffs.read_hour(diffs_path, coins=[coin]).filter(
        pl.col("kind") == book_diffs.NEW
    )

    print(f"orders accepted into the book   {placed.height:,}")
    print(f"new orders seen in book diffs   {arrived.height:,}")

    matched = arrived.join(placed, on="oid", how="left").drop_nulls("px_order")
    print(f"book diffs whose oid is known   {matched.height:,}"
          f"   ({100 * matched.height / arrived.height:.2f}%)")
    if matched.is_empty():
        return

    for label, left, right in (
        ("price", "px", "px_order"),
        ("side", "isAsk", "isAsk_order"),
        ("size", "size", "sz_order"),
    ):
        agree = (matched[left] == matched[right]).sum()
        print(f"    {label:<6} agrees            {agree:,}"
              f"   ({100 * agree / matched.height:.2f}%)")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    report(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
