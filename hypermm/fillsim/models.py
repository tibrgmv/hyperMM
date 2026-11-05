"""Four rules for deciding when a resting order would have been filled.

They are applied to real orders, whose outcome is known, so that the error of each
rule is measured rather than assumed. The rules read only the market around an
order -- the trades at its price and the queue ahead of it -- never its own fills.

    touch           the first trade at the order's price fills it
    volume-through  filled once trades at its price exceed the volume ahead at placement
    queue-aware     the first trade at its price after it reached the front fills it
    clock           Avellaneda–Stoikov: fills arrive at rate A·exp(−k·δ), δ ticks from mid

The first three give a time; the clock gives a probability of a fill within the
order's observed life.
"""

from __future__ import annotations

import numpy as np
import polars as pl

KEY = ("px", "isAsk")  # a price level of one coin: the trades there hit its resting side
NEVER = np.iinfo(np.int64).max
NS = 1_000_000_000


def level_trades(fills: pl.DataFrame) -> pl.DataFrame:
    """Trades merged per block and level, with the volume traded at the level so far.

    A trade hits the side opposite its aggressor. Trades within one block are merged,
    since the block is the finest time the data resolves.
    """
    return (
        fills.group_by("ts", "px", isAsk=~pl.col("isAggressorAsk"))
        .agg(volume=pl.col("sz").sum())
        .sort("ts")
        .with_columns(traded=pl.col("volume").cum_sum().over(KEY))
    )


def _first_trade(table: pl.DataFrame, trades: pl.DataFrame, since: str) -> pl.Series:
    """Time of the first trade at each order's level at or after column `since`."""
    left = table.select(*KEY, t=pl.col(since).cast(pl.Int64).fill_null(NEVER)).with_row_index()
    right = trades.select(*KEY, t="ts", hit="ts")
    return (
        left.sort("t")
        .join_asof(right, on="t", by=KEY, strategy="forward", check_sortedness=False)
        .sort("index")["hit"]
    )


def touch(table: pl.DataFrame, trades: pl.DataFrame) -> pl.Series:
    return _first_trade(table, trades, "tPlaced")


def queue_aware(table: pl.DataFrame, trades: pl.DataFrame) -> pl.Series:
    """Under strict time priority an order at the front is filled by the next trade at
    its price. Within the block in which it reached the front the order of trades and
    departures is not known, so a trade there counts as filling it."""
    return _first_trade(table, trades, "tFront")


def volume_through(table: pl.DataFrame, trades: pl.DataFrame) -> pl.Series:
    """Filled at the first trade that takes the volume traded at the level since
    placement past the volume that was ahead. Cancellations ahead are not seen."""
    left = table.select(*KEY, t=pl.col("tPlaced").cast(pl.Int64), ahead="volumeAhead")
    before = (
        left.with_row_index()
        .sort("t")
        .join_asof(
            trades.select(*KEY, t="ts", base="traded"),
            on="t",
            by=KEY,
            strategy="backward",
            allow_exact_matches=False,  # a trade in the placement block comes after it
            check_sortedness=False,
        )
        .with_columns(target=pl.col("base").fill_null(0) + pl.col("ahead") + 1)
    )
    return (
        before.sort("target")
        .join_asof(
            trades.select(*KEY, target="traded", hit="ts").sort("target"),
            on="target",
            by=KEY,
            strategy="forward",
            check_sortedness=False,
        )
        .sort("index")["hit"]
    )


def _mid(top: pl.DataFrame) -> pl.Expr:
    return (pl.col("bestBid") + pl.col("bestAsk")) / 2


def tick(top: pl.DataFrame) -> int:
    """The price step: the smallest positive spread the hour showed."""
    spread = (top["bestAsk"] - top["bestBid"]).drop_nulls()
    return int(spread.filter(spread > 0).min())


def reach(top: pl.DataFrame, trades: pl.DataFrame) -> np.ndarray:
    """How far from the mid, in ticks, trading reached: once per block and side, at
    the deepest level hit, against the mid as it stood before the block."""
    mids = top.select(t=pl.col("ts").cast(pl.Int64), mid=_mid(top)).drop_nulls()
    return (
        trades.select("isAsk", "px", t="ts")
        .join_asof(mids, on="t", strategy="backward", allow_exact_matches=False)
        .group_by("t", "isAsk")
        .agg(depth=((pl.col("px") - pl.col("mid")).abs() / tick(top)).max())
        .drop_nulls()["depth"]
        .to_numpy()
    )


def clock_fit(
    top: pl.DataFrame, trades: pl.DataFrame, min_count: int = 5, *, weighted: bool = True
) -> tuple[float, float]:
    """Fit λ(δ) = A·exp(−k·δ): how often per second, per side, trading reaches δ ticks
    from the mid. Depths reached fewer than `min_count` times are left out. Unweighted,
    the fit follows the sparse deep levels and misses the touch, where nearly all
    orders rest; it is kept for comparison."""
    depths = reach(top, trades)
    seconds = (top["ts"].max() - top["ts"].min()) / NS
    grid = 0.5 + np.arange(int(depths.max()) + 1)
    counts = np.array([(depths >= d - 1e-9).sum() for d in grid])
    keep = counts >= min_count
    # a log count has variance about 1/count, hence the weights
    rate = np.log(counts[keep] / seconds / 2)
    weights = np.sqrt(counts[keep]) if weighted else None
    slope, intercept = np.polyfit(grid[keep], rate, 1, w=weights)
    return float(np.exp(intercept)), float(-slope)


def clock(table: pl.DataFrame, top: pl.DataFrame, a: float, k: float) -> pl.Series:
    """Probability of at least one fill within the order's observed life."""
    before = (
        top.sort("seq")
        .with_columns(midBefore=_mid(top).shift(1))
        .filter(pl.col("kind") == "new")
        .select("oid", "midBefore")
    )
    delta = (pl.col("px") - pl.col("midBefore")).abs() / tick(top)
    rate = a * (-k * delta).exp()
    return (
        table.with_row_index()
        .join(before, on="oid", how="left")
        .sort("index")
        .select(1 - (-rate * pl.col("duration") / NS).exp())
        .to_series()
    )


def predict(table: pl.DataFrame, top: pl.DataFrame, fills: pl.DataFrame) -> pl.DataFrame:
    """The outcome table with each rule's answer added."""
    trades = level_trades(fills)
    a, k = clock_fit(top, trades)
    return table.with_columns(
        tTouch=touch(table, trades),
        tVolume=volume_through(table, trades),
        tQueue=queue_aware(table, trades),
        pClock=clock(table, top, a, k),
    )
