"""Who was the maker, what the status stream misses, and how passive orders ended.

Backs the journal entries of 24 August 2026: «Открытый вопрос: сокращения размера
без события об исполнении», «Разрешение вопроса: поток статусов не сообщает о трети
исполнений мейкера», «Этап 3: разметка исходов», «Находка: позиция в очереди
предсказывает исполнение монотонно», «Находка: цензурирование не второстепенно»,
and the entry of 4 November 2026 on the front of the queue.

Usage: python scripts/report_fills.py [--date 20251201] [--hour 12] [--coins SOL BTC]
"""

from __future__ import annotations

import polars as pl
from _report import arguments, section, share

from hypermm.ingest.book_diffs import NEW, UPDATE
from hypermm.ingest.codes import Status
from hypermm.ingest.schema import SCALE
from hypermm.pipeline import load_hour, read_changes, read_fills
from hypermm.reconstruct.outcomes import CANCELLED, check, maker_fills
from hypermm.reconstruct.queue import LEVEL

NS = 1e9
DECILES = 10


def maker_rules(top: pl.DataFrame, fills: pl.DataFrame) -> None:
    """Which counterparty of a trade was resting? Two rules, the earlier and the current."""
    placed = top.filter(pl.col("kind") == NEW)["oid"].unique().implode()
    present = top["oid"].unique().implode()
    print("    maker identification     one side   both sides   neither")
    for name, known in (("placed in the file", placed), ("present in the file", present)):
        a, b = fills["oid0"].is_in(known), fills["oid1"].is_in(known)
        print(f"    {name:<22} {share((a ^ b).sum(), fills.height):>10}"
              f" {share((a & b).sum(), fills.height):>12} {share((~a & ~b).sum(), fills.height):>9}")


def silent_reductions(top: pl.DataFrame, statuses: pl.DataFrame, fills: pl.DataFrame) -> None:
    """Size reductions of resting orders that have no `filled` status at all."""
    reduced = top.filter(pl.col("kind") == UPDATE)
    reported = statuses.filter(pl.col("statusId") == Status.FILLED)["oid"].unique().implode()
    reduce_only = statuses.filter("reduceOnly")["oid"].unique().implode()
    traded = pl.concat([fills["oid0"], fills["oid1"]]).unique().implode()

    print("    size reductions without any `filled` status:")
    for label, subset in (
        ("all orders", reduced),
        ("not reduce-only", reduced.filter(~pl.col("oid").is_in(reduce_only))),
    ):
        silent = subset.filter(~pl.col("oid").is_in(reported))
        orders = subset["oid"].n_unique()
        silent_orders = silent["oid"].unique()
        print(f"      {label:<16} orders reduced {orders:>7,};"
              f" without `filled` {share(len(silent_orders), orders):>7};"
              f" their share of reduced volume {share(silent['delta'].sum(), subset['delta'].sum()):>7};"
              f" found in trades {share(silent_orders.is_in(traded).sum(), len(silent_orders)):>7}")


def status_coverage(top: pl.DataFrame, statuses: pl.DataFrame, fills: pl.DataFrame) -> None:
    """Trades per maker order against `filled` statuses per order."""
    events = (
        statuses.filter(pl.col("statusId") == Status.FILLED)
        .group_by("oid")
        .agg(pl.len().alias("events"))
    )
    placed = top.filter(pl.col("kind") == NEW)["oid"].unique().implode()
    a, b = fills["oid0"].is_in(placed), fills["oid1"].is_in(placed)
    earlier_rule = fills.select(
        pl.when(a & ~b).then("oid0").when(b & ~a).then("oid1").alias("oid")
    ).drop_nulls()
    current_rule = maker_fills(top, fills).select("oid")

    print("    maker orders vs `filled` statuses   orders   equal  more trades  more events  coverage")
    for name, makers in (("earlier rule (placed)", earlier_rule), ("current rule (present)", current_rule)):
        per = (
            makers.group_by("oid")
            .agg(pl.len().alias("trades"))
            .join(events, on="oid", how="left")
            .with_columns(pl.col("events").fill_null(0))
        )
        print(f"    {name:<34} {per.height:>7,} {share((per['trades'] == per['events']).sum(), per.height):>7}"
              f" {share((per['trades'] > per['events']).sum(), per.height):>12}"
              f" {share((per['trades'] < per['events']).sum(), per.height):>12}"
              f" {share(per['events'].sum(), per['trades'].sum()):>9}")


def same_block(top: pl.DataFrame, statuses: pl.DataFrame, fills: pl.DataFrame) -> None:
    """Does each side of a trade have a `filled` status in the trade's own block?"""
    reported = statuses.filter(pl.col("statusId") == Status.FILLED)
    seen = reported.select("oid", "ts").unique().with_columns(seen=pl.lit(True))
    trades = maker_fills(top, fills).with_columns(pl.col("ts").cast(pl.UInt64))

    print(f"    trades {fills.height:,}; `filled` statuses {reported.height:,}")
    for role, column in (("taker", "taker"), ("maker", "oid")):
        hit = trades.join(seen.rename({"oid": column}), on=[column, "ts"], how="left")["seen"]
        print(f"    {role} has a `filled` status in the trade's block:"
              f" {share(hit.is_not_null().sum(), trades.height)}")
    pairs = trades.select("oid", "ts").unique()
    hit = pairs.join(seen, on=["oid", "ts"], how="left")["seen"]
    print(f"    distinct (maker order, block) pairs {pairs.height:,};"
          f" with a `filled` status {share(hit.is_not_null().sum(), pairs.height)}")


def outcome_table(table: pl.DataFrame) -> None:
    print(f"    passive orders placed within the hour: {table.height:,}")
    for outcome, n in table.group_by("outcome").len().sort("outcome").iter_rows():
        print(f"      {outcome:<10} {n:>10,}   {share(n, table.height):>8}")
    print(f"      at least one fill   {share(table['filled'].sum(), table.height):>8}")
    print("    checks: " + ", ".join(f"{k} {v}" for k, v in check(table).items()))


def queue_deciles(table: pl.DataFrame) -> None:
    labels = [str(i) for i in range(DECILES)]
    grouped = (
        table.with_columns(
            decile=pl.col("volumeAhead").qcut(DECILES, labels=labels).cast(pl.String).cast(pl.Int32)
        )
        .group_by("decile")
        .agg(
            ahead=pl.col("volumeAhead").median() / SCALE,
            filled=pl.col("filled").mean() * 100,
            life=pl.col("duration").median() / NS,
            size=pl.col("restingSize").median() / SCALE,
        )
        .sort("decile")
    )
    print("    decile  median ahead   filled   median life   median size")
    for d, ahead, filled, life, size in grouped.iter_rows():
        print(f"    {d:>6} {ahead:>13,.4f} {filled:>7.2f}% {life:>11.2f} s {size:>13,.4f}")
    empty = table.filter(pl.col("volumeAhead") == 0)["filled"].mean()
    queued = table.filter(pl.col("volumeAhead") > 0)["filled"].mean()
    print(f"    filled when joining an empty level {100 * empty:.2f}%, a non-empty one {100 * queued:.2f}%")


def censoring(table: pl.DataFrame) -> None:
    to_fill = table.filter("filled")["duration"].median() / NS
    to_cancel = table.filter(~pl.col("filled") & (pl.col("outcome") == CANCELLED))["duration"].median() / NS
    print(f"    median time to first fill {to_fill:.2f} s; to cancellation (never filled) {to_cancel:.2f} s")
    print("    orders living longer than: " + ", ".join(
        f"{t} s {share((table['duration'] > t * NS).sum(), table.height)}" for t in (0.1, 0.5, 1, 5, 30)
    ))


def still_ahead(top: pl.DataFrame, oid: int) -> pl.DataFrame:
    """Orders that arrived before `oid` and still rested when it was first reduced."""
    own = top.filter(pl.col("oid") == oid)
    placed = own.filter(pl.col("kind") == NEW).row(0, named=True)
    reduced = own.filter(pl.col("kind") != NEW)["seq"].min()
    return (
        top.filter(*(pl.col(c) == placed[c] for c in LEVEL))
        .with_columns(arrival=pl.col("seq").min().over("oid"))
        .filter((pl.col("arrival") < placed["seq"]) & (pl.col("seq") <= reduced))
        .sort("seq")
        .group_by("oid")
        .agg(pl.col("size").last(), pl.col("ts").last())
        .filter(pl.col("size") > 0)
    )


def queue_front(top: pl.DataFrame, statuses: pl.DataFrame, table: pl.DataFrame) -> None:
    """Strict time priority: an order can only be filled once nothing rests ahead of it."""
    front = table.filter(pl.col("tFront").is_not_null())
    filled = table.filter("filled")
    print(f"    reached the front of its queue     {share(front.height, table.height)}")
    print(f"    filled, having reached the front   {share(front['filled'].sum(), front.height)}")
    print(f"    filled, never at the front         {table.filter(pl.col('tFront').is_null())['filled'].sum()}")
    print(f"    median time from placement to front {(front['tFront'] - front['tPlaced']).median() / NS:.2f} s")
    print(f"    filled orders at the front no later than their first fill: "
          f"{share((filled['tFront'] <= filled['tFirstFill']).sum(), filled.height)}")

    jumped = filled.filter(pl.col("tFront").is_null())["oid"].to_list()
    if not jumped:
        return
    blockers = pl.concat([still_ahead(top, oid) for oid in jumped]).unique("oid")
    last_status = (
        statuses.filter(pl.col("oid").is_in(blockers["oid"].implode()))
        .sort("ts")
        .group_by("oid")
        .agg(pl.col("statusId").last())
    )
    print(f"    the {len(jumped)} filled without reaching it were behind {blockers.height} orders; how those ended:")
    for status, n in last_status.group_by("statusId").len().sort("len", descending=True).iter_rows():
        print(f"      {Status(status).name:<22} {n}")


def main() -> None:
    args = arguments(__doc__)
    changes = read_changes(args.raw, args.date, args.hour)
    fills = read_fills(args.raw, args.date, args.hour)

    for coin in args.coins:
        hour = load_hour(args.raw, args.date, args.hour, coin, changes=changes, fills=fills)
        section(f"{coin}: makers and what the status stream reports")
        maker_rules(hour.top, hour.fills)
        silent_reductions(hour.top, hour.statuses, hour.fills)
        status_coverage(hour.top, hour.statuses, hour.fills)
        same_block(hour.top, hour.statuses, hour.fills)

        section(f"{coin}: outcomes of passive orders")
        outcome_table(hour.table)
        queue_deciles(hour.table)
        censoring(hour.table)

        section(f"{coin}: front of the queue")
        queue_front(hour.top, hour.statuses, hour.table)


if __name__ == "__main__":
    main()
