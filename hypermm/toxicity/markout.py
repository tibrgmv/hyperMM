"""Markout: what a passive fill was worth once the price had time to move."""

from __future__ import annotations

import polars as pl

NS = 1_000_000_000


def markout(
    fills: pl.DataFrame, top: pl.DataFrame, seconds: float, *, until: int | None = None
) -> pl.Series:
    """Gain of each passive fill, in basis points of its price, against the mid
    `seconds` later. Positive means the fill paid; negative means the price went on
    through it, the mark of an informed counterparty.

    `fills` needs `t` (Int64 ns), `px` and `isAsk` -- the resting side: a bid that
    is filled has bought, an ask has sold.

    `until` (Int64 ns) is the end of what `top` observed, such as the end of its
    hour: a horizon past it is null rather than the last mid before it.
    """
    mids = top.select(
        at=pl.col("ts").cast(pl.Int64), mid=(pl.col("bestBid") + pl.col("bestAsk")) / 2
    ).drop_nulls()
    later = (
        fills.select("px", "isAsk", at=pl.col("t") + int(seconds * NS))
        .with_row_index()
        .sort("at")
        .join_asof(mids, on="at", strategy="backward")
        .sort("index")
    )
    gain = pl.when(pl.col("isAsk")).then(pl.col("px") - pl.col("mid")).otherwise(
        pl.col("mid") - pl.col("px")
    )
    value = 10_000 * gain / pl.col("px")
    if until is not None:
        value = pl.when(pl.col("at") <= until).then(value)
    return later.select(value).to_series()
