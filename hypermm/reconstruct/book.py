"""Tracking the top of the book: best prices and the volume standing at them."""

from __future__ import annotations

import numpy as np
import polars as pl
from numba import njit

EMPTY = -1


def _ranked(rebuilt: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray]:
    """Replace prices by their position among the prices the hour actually used.

    Both sides share one scale, so a bid and an ask at the same price get the same
    index and the whole book fits in two small arrays.
    """
    prices = rebuilt["px"].unique().sort()
    ladder = pl.DataFrame({"px": prices, "rank": np.arange(len(prices), dtype=np.int64)})
    return rebuilt.sort("seq").join(ladder, on="px", how="left"), prices.to_numpy()


@njit(cache=True)
def _walk(rank, is_ask, volume, levels):  # pragma: no cover - compiled
    """Replay level volumes in order, carrying the best rank on each side along.

    The best price moves by one level at a time far more often than it jumps, so
    it is nudged from where it was rather than searched for: raised when an order
    lands in front of it, and walked back only when the level it names empties.
    """
    bid_at = np.zeros(levels, dtype=np.int64)
    ask_at = np.zeros(levels, dtype=np.int64)
    best_bid, best_ask = EMPTY, levels

    n = rank.shape[0]
    bid_rank = np.empty(n, dtype=np.int64)
    ask_rank = np.empty(n, dtype=np.int64)
    bid_volume = np.empty(n, dtype=np.int64)
    ask_volume = np.empty(n, dtype=np.int64)

    for i in range(n):
        at, resting = rank[i], volume[i]
        if is_ask[i]:
            ask_at[at] = resting
            if resting > 0:
                if at < best_ask:
                    best_ask = at
            elif at == best_ask:
                while best_ask < levels and ask_at[best_ask] <= 0:
                    best_ask += 1
        else:
            bid_at[at] = resting
            if resting > 0:
                if at > best_bid:
                    best_bid = at
            elif at == best_bid:
                while best_bid >= 0 and bid_at[best_bid] <= 0:
                    best_bid -= 1

        bid_rank[i] = best_bid
        ask_rank[i] = best_ask if best_ask < levels else EMPTY
        bid_volume[i] = bid_at[best_bid] if best_bid >= 0 else 0
        ask_volume[i] = ask_at[best_ask] if best_ask < levels else 0

    return bid_rank, ask_rank, bid_volume, ask_volume


@njit(cache=True)
def _snapshots(rank, is_ask, volume, levels, at, depth):  # pragma: no cover - compiled
    """Replay again, pausing at the given events to copy the nearest levels each side."""
    bid_at = np.zeros(levels, dtype=np.int64)
    ask_at = np.zeros(levels, dtype=np.int64)
    taken = 0
    bid_rank = np.full((at.shape[0], depth), EMPTY, dtype=np.int64)
    ask_rank = np.full((at.shape[0], depth), EMPTY, dtype=np.int64)
    bid_volume = np.zeros((at.shape[0], depth), dtype=np.int64)
    ask_volume = np.zeros((at.shape[0], depth), dtype=np.int64)

    for i in range(rank.shape[0]):
        if is_ask[i]:
            ask_at[rank[i]] = volume[i]
        else:
            bid_at[rank[i]] = volume[i]

        while taken < at.shape[0] and at[taken] == i:
            found, level = 0, levels - 1
            while level >= 0 and found < depth:
                if bid_at[level] > 0:
                    bid_rank[taken, found] = level
                    bid_volume[taken, found] = bid_at[level]
                    found += 1
                level -= 1
            found, level = 0, 0
            while level < levels and found < depth:
                if ask_at[level] > 0:
                    ask_rank[taken, found] = level
                    ask_volume[taken, found] = ask_at[level]
                    found += 1
                level += 1
            taken += 1

    return bid_rank, bid_volume, ask_rank, ask_volume


@njit(cache=True)
def _sweep(rank, is_ask, volume, price, levels, at, want, buying):  # pragma: no cover
    """Replay again, pausing at the given events to eat `want` off one side."""
    bid_at = np.zeros(levels, dtype=np.int64)
    ask_at = np.zeros(levels, dtype=np.int64)
    taken = 0
    paid = np.zeros(at.shape[0], dtype=np.float64)
    short = np.zeros(at.shape[0], dtype=np.int64)
    touched = np.zeros(at.shape[0], dtype=np.int64)

    for i in range(rank.shape[0]):
        if is_ask[i]:
            ask_at[rank[i]] = volume[i]
        else:
            bid_at[rank[i]] = volume[i]

        while taken < at.shape[0] and at[taken] == i:
            left, spent, used = want, 0.0, 0
            level = 0 if buying else levels - 1
            step = 1 if buying else -1
            while 0 <= level < levels and left > 0:
                resting = ask_at[level] if buying else bid_at[level]
                if resting > 0:
                    eaten = min(resting, left)
                    spent += float(price[level]) * float(eaten)
                    left -= eaten
                    used += 1
                level += step
            paid[taken], short[taken], touched[taken] = spent, left, used
            taken += 1

    return paid, short, touched


def sweep(rebuilt: pl.DataFrame, at: np.ndarray, want: int, *, buying: bool) -> pl.DataFrame:
    """Price a market order of size `want` sent at each of the given events.

    Walks the book outward from the touch, eating whole levels until the order is
    filled, and returns the volume-weighted average of the prices reached -- what a
    taker would actually pay. A bar-based backtest replaces this with a single OHLC
    number, and the gap between the two is what the measurement is after.

    Cost accumulates in floating point rather than the project's exact integers: a
    price times a size carries scale 10^16 and overflows int64, and unlike a queue
    volume an average price has no exact value worth preserving.

    Events are returned in ascending order regardless of the order given.
    """
    ordered, at_rank = _ranked(rebuilt)
    at = np.sort(np.asarray(at, dtype=np.int64))

    paid, short, touched = _sweep(
        ordered["rank"].to_numpy(),
        ordered["isAsk"].to_numpy(),
        (ordered["levelVolume"] + ordered["delta"]).to_numpy(),
        at_rank,
        len(at_rank),
        at,
        int(want),
        buying,
    )

    got = want - short
    return pl.DataFrame(
        {
            "at": at,
            "price": np.where(got > 0, paid / np.maximum(got, 1), np.nan),
            "short": short,
            "levels": touched,
        }
    )


def check_sweep(small: pl.DataFrame, large: pl.DataFrame, *, buying: bool) -> dict[str, int]:
    """Count impossible walks, plus orders the book was too thin to fill.

    Only the last two are defects; a book too thin for the size is a fact about the
    hour, and the count is there to keep it from passing unnoticed.
    """
    filled = (pl.col("short") == 0) & (pl.col("shortBig") == 0)
    both = small.join(large, on="at", suffix="Big").filter(filled)
    cheaper = (
        pl.col("priceBig") < pl.col("price") if buying else pl.col("priceBig") > pl.col("price")
    )

    return {
        "exhaustedBook": small.filter(pl.col("short") > 0).height,
        "cheaperInBulk": both.filter(cheaper).height,
        "shallowerInBulk": both.filter(pl.col("levelsBig") < pl.col("levels")).height,
    }


def snapshots(rebuilt: pl.DataFrame, at: np.ndarray, depth: int = 12) -> list[dict]:
    """Copy the book as it stood after each of the given events, nearest levels first.

    Scanning outward from the touch is cheap only because the levels an hour uses
    number in the thousands; the scan would have to be replaced by an index if the
    ladder ever grew.
    """
    ordered, at_rank = _ranked(rebuilt)

    bid_rank, bid_volume, ask_rank, ask_volume = _snapshots(
        ordered["rank"].to_numpy(),
        ordered["isAsk"].to_numpy(),
        (ordered["levelVolume"] + ordered["delta"]).to_numpy(),
        len(at_rank),
        np.sort(np.asarray(at, dtype=np.int64)),
        depth,
    )

    def side(ranks: np.ndarray, volumes: np.ndarray, row: int) -> list[list[int]]:
        keep = ranks[row] != EMPTY
        return [
            [int(at_rank[r]), int(v)]
            for r, v in zip(ranks[row][keep], volumes[row][keep], strict=True)
        ]

    return [
        {"bids": side(bid_rank, bid_volume, row), "asks": side(ask_rank, ask_volume, row)}
        for row in range(len(at))
    ]


def with_top_of_book(rebuilt: pl.DataFrame) -> pl.DataFrame:
    """Add the best bid and ask, and the volume resting at each, after every change.

    Takes the frame `with_queue_position` produced: the volume a level holds once a
    change is applied is already there as `levelVolume` plus `delta`, so this only
    has to remember which level is currently in front on each side.

    Ranking prices turns "the highest bid" into "the largest index still occupied",
    which is what makes the walk cheap.

    Where a side is empty the price is null rather than zero -- a book with no bids
    has no best bid, and zero would be read as a price of zero.
    """
    ordered, at_rank = _ranked(rebuilt)

    bid_rank, ask_rank, bid_volume, ask_volume = _walk(
        ordered["rank"].to_numpy(),
        ordered["isAsk"].to_numpy(),
        (ordered["levelVolume"] + ordered["delta"]).to_numpy(),
        len(at_rank),
    )

    def price_of(ranks: np.ndarray) -> pl.Expr:
        occupied = pl.Series(ranks != EMPTY)
        return pl.when(occupied).then(pl.Series(at_rank[np.maximum(ranks, 0)]))

    return ordered.drop("rank").with_columns(
        bestBid=price_of(bid_rank),
        bestAsk=price_of(ask_rank),
        bestBidVolume=pl.Series(bid_volume),
        bestAskVolume=pl.Series(ask_volume),
    )


def check(top: pl.DataFrame) -> dict[str, int]:
    """Count states a book cannot be in. All must be zero once the book has filled."""
    known = pl.col("bestBid").is_not_null() & pl.col("bestAsk").is_not_null()

    def count(condition: pl.Expr) -> int:
        return top.filter(condition).height

    return {
        "crossed": count(known & (pl.col("bestBid") >= pl.col("bestAsk"))),
        "emptyBid": count(pl.col("bestBid").is_null()),
        "emptyAsk": count(pl.col("bestAsk").is_null()),
        "bestWithoutVolume": count(known & (pl.col("bestBidVolume") <= 0)),
    }
