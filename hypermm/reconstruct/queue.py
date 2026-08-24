"""Rebuilding resting volume per price level, and queue position at placement."""

from __future__ import annotations

import polars as pl

from hypermm.ingest.book_diffs import NEW, REMOVE, UPDATE

LEVEL = ("coin", "isAsk", "px")


# Add the volume of all orders resting at the same price level, and the queue position of each order when it was placed.
def with_queue_position(diffs: pl.DataFrame) -> pl.DataFrame:    

    already_resting = pl.col("firstSeen") & (pl.col("kind") != NEW)

    return (
        diffs.sort("seq")
        # the shift is the costly step here, so it is materialised once and reused
        .with_columns(carried=pl.col("size").shift(1).over("oid"))
        .with_columns(
            firstSeen=pl.col("carried").is_null(),
            sizeBefore=pl.coalesce("carried", "prevSize", pl.lit(0, dtype=pl.Int64)),
        )
        .with_columns(
            delta=pl.col("size") - pl.col("sizeBefore"),
            opening=pl.when(already_resting).then("sizeBefore").otherwise(0),
        )
        .with_columns(
            levelVolume=(pl.col("opening") + pl.col("delta")).cum_sum().over(LEVEL)
            - pl.col("delta")
        )
        .with_columns(volumeAhead=pl.when(pl.col("kind") == NEW).then(pl.col("levelVolume")))
        .drop("carried")
    )


def check(rebuilt: pl.DataFrame) -> dict[str, int]:
    """Count the ways a rebuilt book can be wrong."""
    settled = ~pl.col("firstSeen")
    started_cold = pl.col("firstSeen") & (pl.col("kind") != NEW)

    def count(condition: pl.Expr) -> int:
        return rebuilt.filter(condition).height

    return {
        "coldStartRecovered": count(started_cold & (pl.col("kind") == UPDATE)),
        "coldStartLost": count(started_cold & (pl.col("kind") == REMOVE)),
        "negativeLevel": count(pl.col("levelVolume") + pl.col("delta") < 0),
        "reArrival": count(settled & (pl.col("kind") == NEW)),
        "sizeBeforeDisagrees": count(
            settled & (pl.col("kind") == UPDATE) & (pl.col("sizeBefore") != pl.col("prevSize"))
        ),
        "growingUpdate": count(settled & (pl.col("kind") == UPDATE) & (pl.col("delta") >= 0)),
        "nonEmptyRemoval": count((pl.col("kind") == REMOVE) & (pl.col("size") != 0)),
    }
