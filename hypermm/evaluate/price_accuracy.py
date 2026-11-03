"""What a bar-based backtest gets wrong about the price of a market order.

A charting platform decides on the close of one bar and fills on the open of the
next, at a single OHLC number, with neither spread nor depth in sight. The book is
reconstructed here, so the same order can be priced properly -- by walking it -- and
the two answers compared.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from hypermm.reconstruct.book import sweep

NS_PER_S = 1_000_000_000
BAR = 60 * NS_PER_S

# Ordered from the crudest assumption to the one closest to the truth. `mid` and
# `touch` need a book and so are not available to a bar-based backtest at all; they
# are here to split the error into what the spread costs and what the walk adds.
APPROXIMATIONS = ("mid", "touch", "vwap", "close", "nextOpen")

# A walk that stops at the touch should return the touch price exactly, and misses it
# by one rounding step of the accumulated cost -- some 10^-11 of a dollar on bitcoin.
# Anything a backtest could act on is nine orders of magnitude above that.
TOLERANCE_BPS = 1e-3


def bars(fills: pl.DataFrame, span: int = BAR) -> pl.DataFrame:
    """Cut the trade stream into bars, the way a chart would.

    Turnover is accumulated in floating point: a scaled price times a scaled size
    carries scale 10^16 and overflows int64.
    """
    return (
        fills.sort("ts")
        .with_columns(bar=pl.col("ts") // span)
        .group_by("bar")
        .agg(
            open=pl.col("px").first(),
            close=pl.col("px").last(),
            vwap=(pl.col("px").cast(pl.Float64) * pl.col("sz")).sum() / pl.col("sz").sum(),
        )
        .sort("bar")
        .with_columns(nextOpen=pl.col("open").shift(-1))
    )


def _at_or_before(top: pl.DataFrame, edge: np.ndarray) -> np.ndarray:
    """Row of the last book event that had happened by each instant; -1 if none had."""
    return np.searchsorted(top["ts"].to_numpy(), edge, side="left") - 1


def _flattery(name: str, *, buying: bool) -> pl.Expr:
    """How much better than reality the approximation makes the trade look, in bps.

    Positive means the backtest hands the strategy a price it could not have got.
    """
    edge = pl.col("price") - pl.col(name) if buying else pl.col(name) - pl.col("price")
    return (10_000 * edge / pl.col("mid")).alias(f"{name}Bps")


def compare(
    top: pl.DataFrame, fills: pl.DataFrame, want: int, *, buying: bool, span: int = BAR
) -> pl.DataFrame:
    """Price one market order per bar both ways: by the chart and by the book.

    `top` is the rebuilt hour in event order, as `with_top_of_book` leaves it. The
    order is sent at the close of each bar -- the moment a chart strategy decides --
    and the book is taken as it stood at that instant.
    """
    table = bars(fills, span).filter(pl.col("nextOpen").is_not_null())
    at = _at_or_before(top, (table["bar"].to_numpy() + 1) * span)

    table = table.filter(pl.Series(at >= 0))
    at = at[at >= 0]

    walked = sweep(top, at, want, buying=buying).drop("at")
    state = top[at].select(
        touch=pl.col("bestAsk") if buying else pl.col("bestBid"),
        mid=(pl.col("bestBid") + pl.col("bestAsk")) / 2,
        spread=pl.col("bestAsk") - pl.col("bestBid"),
    )

    return table.with_columns(at=at, **walked.to_dict(), **state.to_dict()).with_columns(
        *(_flattery(name, buying=buying) for name in APPROXIMATIONS),
        spreadBps=10_000 * pl.col("spread") / pl.col("mid"),
    )


def summary(compared: pl.DataFrame) -> pl.DataFrame:
    """Reduce the per-bar comparison to one row per approximation."""
    return pl.DataFrame(
        [
            {
                "approximation": name,
                "meanBps": compared[f"{name}Bps"].mean(),
                "medianAbsBps": compared[f"{name}Bps"].abs().median(),
                "flattersShare": 100 * (compared[f"{name}Bps"] > 0).mean(),
            }
            for name in APPROXIMATIONS
        ]
    )


def check(compared: pl.DataFrame) -> dict[str, int]:
    """Count states the comparison cannot be in. All must be zero.

    `touchBps` already carries the sign of the side, so beating the best price on
    offer reads as a negative number whether the order was buying or selling.
    """
    return {
        "beatTheTouch": compared.filter(pl.col("touchBps") < -TOLERANCE_BPS).height,
        "noBook": compared.filter(pl.col("mid").is_null()).height,
        "exhaustedBook": compared.filter(pl.col("short") > 0).height,
    }
