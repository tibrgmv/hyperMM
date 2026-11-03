"""What the three streams contain, and whether they agree with one another.

Backs the journal entry of 23 August 2026: «Находка: состав потока», «Проверка
согласованности потоков», «Находка: reduce-only заявки попадают в стакан урезанными».
Block semantics are covered separately by scripts/check_block_semantics.py.

Usage: python scripts/report_streams.py [--date 20251201] [--hour 12] [--coins SOL BTC]
"""

from __future__ import annotations

import polars as pl
from _report import arguments, section, share

from hypermm.ingest.book_diffs import NEW
from hypermm.ingest.codes import Status, Tif
from hypermm.pipeline import read_changes, read_statuses


def composition(statuses: pl.DataFrame, coin: str) -> None:
    """Shares are of events in the accepted-orders file, not of distinct orders."""
    print(f"\n{coin}: {statuses.height:,} events in the order status file (rejected file not read)")
    for column, codes in (("statusId", Status), ("tifId", Tif)):
        counts = statuses.group_by(column).len().sort("len", descending=True)
        for code, n in counts.iter_rows():
            print(f"    {codes(code).name:<26} {n:>10,}   {share(n, statuses.height):>8}")
        print()


def agreement(changes: pl.DataFrame, statuses: pl.DataFrame, coin: str) -> None:
    placed = (
        statuses.filter(pl.col("statusId") == Status.OPEN)
        .unique(subset="oid", keep="first")
        .select(
            "oid",
            "reduceOnly",
            pl.col("isAsk").alias("isAskOrder"),
            pl.col("limitPx").alias("pxOrder"),
            pl.col("sz").alias("szOrder"),
        )
    )
    arrived = changes.filter(pl.col("kind") == NEW)
    matched = arrived.join(placed, on="oid", how="inner")

    print(f"\n{coin}")
    print(f"    new orders in book diffs        {arrived.height:,}")
    print(f"    their oid found in statuses     {matched.height:,} ({share(matched.height, arrived.height)})")
    for label, left, right in (
        ("price", "px", "pxOrder"),
        ("side", "isAsk", "isAskOrder"),
        ("size", "size", "szOrder"),
    ):
        print(f"    {label:<6} agrees                    "
              f"{share((matched[left] == matched[right]).sum(), matched.height)}")

    differs = matched.with_columns(differs=pl.col("size") != pl.col("szOrder"))
    print("    size disagreement by reduce-only flag:")
    for flag in (True, False):
        group = differs.filter(pl.col("reduceOnly") == flag)
        print(f"        reduceOnly={flag!s:<5}  {group['differs'].sum():>7,} of {group.height:>9,}"
              f"   ({share(group['differs'].sum(), group.height)})")
    smaller = differs.filter("differs")
    print(f"    disagreeing sizes smaller in the book: {share((smaller['size'] < smaller['szOrder']).sum(), smaller.height)}")

    if smaller.is_empty():
        return
    user, n = smaller.group_by("user").len().sort("len", descending=True).row(0)
    own = differs.filter(pl.col("user") == user)
    print(f"    largest source of disagreement: {user}")
    print(f"        {n:,} of {smaller.height:,} disagreements"
          f"; its disagreeing orders reduce-only: {share(own.filter('differs')['reduceOnly'].sum(), n)}"
          f"; its agreeing orders reduce-only: {own.filter(~pl.col('differs'))['reduceOnly'].sum()}"
          f" of {own.filter(~pl.col('differs')).height}")
    events = changes.filter(pl.col("user") == user).height
    print(f"        share of the coin's book diff events: {share(events, changes.height)}")


def main() -> None:
    args = arguments(__doc__)
    changes = read_changes(args.raw, args.date, args.hour)

    section("Book diff stream: kinds of change and coins")
    for column in ("kind", "coin"):
        counts = changes.group_by(column).len().sort("len", descending=True).head(5)
        for value, n in counts.iter_rows():
            print(f"    {value:<8} {n:>10,}   {share(n, changes.height):>8}")
        print()

    statuses = {coin: read_statuses(args.raw, args.date, args.hour, coin) for coin in args.coins}

    section("Order status stream: statuses and time in force (shares of events)")
    for coin, frame in statuses.items():
        composition(frame, coin)

    section("Do the order status and book diff streams describe the same orders?")
    for coin, frame in statuses.items():
        agreement(changes.filter(pl.col("coin") == coin), frame, coin)


if __name__ == "__main__":
    main()
