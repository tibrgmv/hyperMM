"""The rebuilt market as a backtest reads it: the book a strategy sees when it
decides, the trades that may fill its orders, and the queue its orders would join.

A backtested order was never in the book. It is placed at a decision time `t` and
lands in the first block after it, behind every real order then resting at its
level; it does not change the market (the limitation of every replay backtest).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import polars as pl

from hypermm.fillsim.models import NS, clock_fit, level_trades, tick

NEVER = np.iinfo(np.int64).max


@dataclass(frozen=True)
class Book:
    """The top of the book as it stood at a decision."""

    t: int
    bid: int | None
    ask: int | None
    bid_volume: int
    ask_volume: int

    @property
    def mid(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return (self.bid + self.ask) / 2


@dataclass
class Side:
    """Trades that hit one side of the book: time, price, volume, in time order."""

    ts: np.ndarray
    px: np.ndarray
    volume: np.ndarray


class Market:
    """One coin over a span of rebuilt hours (pipeline.load_window's `top` and `fills`)."""

    def __init__(self, top: pl.DataFrame, fills: pl.DataFrame):
        top = top.sort("seq")
        self.tick = tick(top)
        self.ts = top["ts"].cast(pl.Int64).to_numpy()
        # prices up to 1e13 are exact in float64; NaN marks an empty side
        self._bid = top["bestBid"].cast(pl.Float64).to_numpy()
        self._ask = top["bestAsk"].cast(pl.Float64).to_numpy()
        self._bid_volume = top["bestBidVolume"].fill_null(0).to_numpy()
        self._ask_volume = top["bestAskVolume"].fill_null(0).to_numpy()
        self.start, self.end = int(self.ts[0]), int(self.ts[-1])

        self._trades = trades = level_trades(fills)
        self._top = top
        self.sides = {
            hit: Side(
                part["ts"].cast(pl.Int64).to_numpy(),
                part["px"].to_numpy(),
                part["volume"].to_numpy(),
            )
            for hit, part in (
                (False, trades.filter(~pl.col("isAsk"))),
                (True, trades.filter("isAsk")),
            )
        }
        self._levels = self._queues(top)
        self._volumes = self._level_volumes(top)
        # the blocks in which each side was hit, for "how busy was it lately"
        self._hit_blocks = {hit: np.unique(side.ts) for hit, side in self.sides.items()}

    @cached_property
    def clock(self) -> tuple[float, float]:
        """A and k of the exponential clock fitted on this market's own trading."""
        return clock_fit(self._top, self._trades)

    @property
    def a(self) -> float:
        return self.clock[0]

    @property
    def k(self) -> float:
        return self.clock[1]

    # --- what a strategy sees -------------------------------------------------------

    def book(self, t: int) -> Book:
        """The book after the last change at or before `t`."""
        i = int(np.searchsorted(self.ts, t, side="right")) - 1
        if i < 0:
            return Book(t, None, None, 0, 0)

        def price(values: np.ndarray) -> int | None:
            return None if np.isnan(values[i]) else int(values[i])

        return Book(
            t,
            price(self._bid),
            price(self._ask),
            int(self._bid_volume[i]),
            int(self._ask_volume[i]),
        )

    def mid(self, t: int) -> float | None:
        return self.book(t).mid

    def recent_blocks(self, is_ask: bool, t: int, seconds: float) -> int:
        """Trade blocks that hit one side of the book in [t - seconds, t)."""
        blocks = self._hit_blocks[is_ask]
        return int(
            np.searchsorted(blocks, t, side="left")
            - np.searchsorted(blocks, t - int(seconds * NS), side="left")
        )

    def next_block(self, t: int) -> int:
        """The time of the first block after `t`: where an action decided at `t` lands."""
        i = int(np.searchsorted(self.ts, t, side="right"))
        return int(self.ts[i]) if i < len(self.ts) else NEVER

    # --- the queue a backtested order would join -------------------------------------

    @staticmethod
    def _queues(top: pl.DataFrame) -> dict[tuple[bool, int], tuple[np.ndarray, np.ndarray]]:
        """For each level, when each real order arrived and left, as row positions."""
        orders = (
            top.with_row_index("pos")
            .group_by("oid")
            .agg(
                pl.col("isAsk").first(),
                pl.col("px").first(),
                arrived=pl.col("pos").min(),
                left=pl.col("pos").filter(pl.col("size") == 0).min(),
            )
            .with_columns(pl.col("left").fill_null(np.iinfo(np.int64).max).cast(pl.Int64))
        )
        return {
            (is_ask, px): (group["arrived"].cast(pl.Int64).to_numpy(), group["left"].to_numpy())
            for (is_ask, px), group in orders.group_by("isAsk", "px")
        }

    @staticmethod
    def _level_volumes(top: pl.DataFrame) -> dict[tuple[bool, int], tuple[np.ndarray, np.ndarray]]:
        """For each level, the row positions of its changes and its volume after each."""
        rows = top.with_row_index("pos").select(
            "isAsk", "px", "pos", after=pl.col("levelVolume") + pl.col("delta")
        )
        return {
            (is_ask, px): (group["pos"].cast(pl.Int64).to_numpy(), group["after"].to_numpy())
            for (is_ask, px), group in rows.group_by("isAsk", "px", maintain_order=True)
        }

    def level_volume(self, is_ask: bool, px: int, t: int) -> int:
        """Volume resting at a level after the last change at or before `t`: what an
        order placed at `t` sees and queues behind."""
        landed = int(np.searchsorted(self.ts, t, side="right"))
        positions, after = self._volumes.get((is_ask, px), (np.empty(0, np.int64), None))
        i = int(np.searchsorted(positions, landed, side="left")) - 1
        return int(after[i]) if i >= 0 else 0

    def front(self, is_ask: bool, px: int, t: int) -> int:
        """When an order placed at `t` would lead its level: once every real order
        resting there when it landed has left. NEVER if one of them outlives the data."""
        landed = int(np.searchsorted(self.ts, t, side="right"))
        if landed >= len(self.ts):
            return NEVER
        arrived, left = self._levels.get(
            (is_ask, px), (np.empty(0, np.int64), np.empty(0, np.int64))
        )
        ahead = left[(arrived < landed) & (left >= landed)]
        if ahead.size == 0:
            return int(self.ts[landed])
        last = int(ahead.max())
        return NEVER if last >= len(self.ts) else int(self.ts[last])

    # --- calibration ------------------------------------------------------------------

    @cached_property
    def sigma(self) -> float:
        """Standard deviation of one-second mid changes, in ticks per sqrt(second)."""
        grid = np.arange(self.start, self.end, NS)
        i = np.searchsorted(self.ts, grid, side="right") - 1
        mid = (self._bid[i] + self._ask[i]) / 2
        changes = np.diff(mid[~np.isnan(mid)]) / self.tick
        return float(np.std(changes)) if changes.size else 0.0
