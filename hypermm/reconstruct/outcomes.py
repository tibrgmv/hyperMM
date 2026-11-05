"""One row per passive order: how it was placed, and how it ended."""

from __future__ import annotations

import polars as pl

from hypermm.ingest.book_diffs import NEW, REMOVE

FILLED, CANCELLED, RESTING = "filled", "cancelled", "resting"


# detect who is a maker in a trade
def maker_fills(diffs: pl.DataFrame, trades: pl.DataFrame) -> pl.DataFrame:

    in_book = diffs.group_by("oid").agg(pl.col("ts").min().alias("since"))

    paired = trades.join(
        in_book.rename({"oid": "oid0", "since": "since0"}), on="oid0", how="left"
    ).join(in_book.rename({"oid": "oid1", "since": "since1"}), on="oid1", how="left")

    # The maker of a trade is the counterparty whose order was already resting on the book
    waiting = (
        pl.when(pl.col("since1").is_null())
        .then("oid0")
        .when(pl.col("since0").is_null())
        .then("oid1")
        .when(pl.col("since0") <= pl.col("since1"))
        .then("oid0")
        .otherwise("oid1")
    )
    return (
        paired.filter(pl.col("since0").is_not_null() | pl.col("since1").is_not_null())
        .with_columns(oid=waiting)
        .select(
            "ts",
            "oid",
            pl.when(pl.col("oid") == pl.col("oid0")).then("oid1").otherwise("oid0").alias("taker"),
            "px",
            "sz",
            "isAggressorAsk",
        )
    )


# 
def outcomes(diffs: pl.DataFrame, trades: pl.DataFrame) -> pl.DataFrame:
    """
    Orders already resting when the file opened are left out: their placement is not
    in it, so neither their queue position nor their age can be established.
    """

    window_end = diffs["ts"].max()

    placed = diffs.filter(pl.col("kind") == NEW).select(
        "oid",
        "coin",
        "isAsk",
        "px",
        pl.col("ts").alias("tPlaced"),
        pl.col("size").alias("restingSize"),
        "volumeAhead",
        "frontSeq",
    )
    fronts = diffs.select(pl.col("seq").alias("frontSeq"), pl.col("ts").alias("tFront"))
    removed = diffs.filter(pl.col("kind") == REMOVE).group_by("oid").agg(
        pl.col("ts").max().alias("tRemoved")
    )
    filled = maker_fills(diffs, trades).group_by("oid").agg(
        pl.col("ts").min().alias("tFirstFill"),
        pl.col("sz").sum().alias("filledSize"),
        pl.len().alias("fills"),
    )

    # Trade times are Int64 and book times UInt64; mixed, they would meet as Float64 and
    # lose the last nanoseconds, so every time is taken as Int64 before they meet.
    ended = pl.coalesce(
        "tFirstFill", pl.col("tRemoved").cast(pl.Int64), pl.lit(window_end).cast(pl.Int64)
    )
    
    return (
        placed.join(fronts, on="frontSeq", how="left")
        .join(removed, on="oid", how="left")
        .join(filled, on="oid", how="left")
        .with_columns(
            filledSize=pl.col("filledSize").fill_null(0),
            fills=pl.col("fills").fill_null(0),
        )
        .with_columns(
            filled=pl.col("tFirstFill").is_not_null(),
            duration=ended - pl.col("tPlaced").cast(pl.Int64),
            outcome=pl.when(pl.col("filledSize") >= pl.col("restingSize"))
            .then(pl.lit(FILLED))
            .when(pl.col("tRemoved").is_not_null())
            .then(pl.lit(CANCELLED))
            .otherwise(pl.lit(RESTING)),
        )
    )


def check(table: pl.DataFrame) -> dict[str, int]:
    """Count rows that contradict the way an order can behave. All must be zero."""

    def count(condition: pl.Expr) -> int:
        return table.filter(condition).height

    return {
        "negativeDuration": count(pl.col("duration") < 0),
        "fillWithoutTime": count((pl.col("fills") > 0) & pl.col("tFirstFill").is_null()),
        "fillBeforePlacement": count(pl.col("tFirstFill") < pl.col("tPlaced")),
        "removalBeforePlacement": count(pl.col("tRemoved") < pl.col("tPlaced")),
        "overfilled": count(pl.col("filledSize") > pl.col("restingSize")),
        "restingButRemoved": count(
            (pl.col("outcome") == RESTING) & pl.col("tRemoved").is_not_null()
        ),
        "duplicated": table.height - table["oid"].n_unique(),
    }
