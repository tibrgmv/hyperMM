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

# The book diff stream has no timestamps of its own, 
# so the only way to attach time is to join on the order ID against the order status stream
def with_time(diffs: pl.DataFrame, statuses: pl.DataFrame) -> pl.DataFrame:
    
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
        .with_columns(tsExact=pl.col("ts").is_not_null(), ts=pl.col("ts").forward_fill())
        # an exact time behind one already seen is pulled forward, and marked as such
        .with_columns(tsMoved=pl.col("ts") < pl.col("ts").cum_max())
        .with_columns(ts=pl.col("ts").cum_max())
        .drop("tsOpen", "tsLast")
    )


def check(timed: pl.DataFrame) -> dict[str, int]:
    """Count rows whose time could not be established, and rows that had to be moved."""
    return {
        "exact": timed.filter("tsExact").height,
        "carried": timed.filter(~pl.col("tsExact") & pl.col("ts").is_not_null()).height,
        "unknown": timed.filter(pl.col("ts").is_null()).height,
        "movedForward": timed.filter(pl.col("tsMoved") & pl.col("tsExact")).height,
    }
