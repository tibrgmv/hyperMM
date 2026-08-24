"""Attaching time to the book diff stream, which carries none of its own."""

from __future__ import annotations

import polars as pl

from hypermm.ingest.book_diffs import NEW, REMOVE
from hypermm.ingest.codes import Status

# A rejected order is turned away before it can affect the book, so its event
# never has a counterpart among the book diffs.
REJECTIONS = frozenset(
    {
        Status.BAD_ALO_PX_REJECTED,
        Status.PERP_MARGIN_REJECTED,
        Status.IOC_CANCEL_REJECTED,
        Status.MIN_TRADE_NTL_REJECTED,
        Status.REDUCE_ONLY_REJECTED,
        Status.PERP_MAX_POSITION_REJECTED,
        Status.ORACLE_REJECTED,
        Status.TRIGGERED,
    }
)


def with_time(diffs: pl.DataFrame, statuses: pl.DataFrame) -> pl.DataFrame:
    """Give every book change a timestamp, borrowed from the order status stream.

    Two kinds of change carry their own time exactly. An order entering the book
    is the same event as its acceptance, so it takes the time of that acceptance;
    an order leaving the book is the same event as whatever ended it, so it takes
    the time of its last status. Together these are about 99% of the stream.

    A size reduction has no such counterpart -- the status stream does not report
    every partial fill of a resting order -- so it inherits the last exact time
    before it in file order. `tsExact` says which of the two a row received.

    Time is then forced to be non-decreasing along the file. It is already so
    except in a handful of orders per hour, and a queue that runs backwards in
    time cannot be replayed at all.
    """
    touching = statuses.filter(~pl.col("statusId").is_in([int(s) for s in REJECTIONS]))
    accepted = (
        touching.filter(pl.col("statusId") == Status.OPEN)
        .group_by("oid")
        .agg(pl.col("ts").min().alias("tsOpen"))
    )
    ended = touching.group_by("oid").agg(pl.col("ts").max().alias("tsLast"))

    anchored = pl.when(pl.col("kind") == NEW).then("tsOpen").when(pl.col("kind") == REMOVE).then(
        "tsLast"
    )
    return (
        diffs.sort("seq")
        .join(accepted, on="oid", how="left")
        .join(ended, on="oid", how="left")
        .with_columns(ts=anchored)
        .with_columns(tsExact=pl.col("ts").is_not_null())
        .with_columns(ts=pl.col("ts").forward_fill().cum_max())
        .drop("tsOpen", "tsLast")
    )


def check(timed: pl.DataFrame) -> dict[str, int]:
    """Count rows whose time could not be established, and rows that had to be moved."""
    return {
        "exact": timed.filter("tsExact").height,
        "carried": timed.filter(~pl.col("tsExact") & pl.col("ts").is_not_null()).height,
        "unknown": timed.filter(pl.col("ts").is_null()).height,
    }
