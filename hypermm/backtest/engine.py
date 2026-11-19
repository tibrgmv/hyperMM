"""Replaying a quoting strategy over the rebuilt market under one fill rule.

At every decision time the engine first books the fills that have happened since
the last decision, then asks the strategy for its quotes. A quote at an unchanged
price keeps its order, and with it its place in the queue; a new price cancels the
order and places another at the back. Each order is one lot and fills whole.

The fill rule decides when an order would be filled (hypermm/backtest/fills.py);
everything downstream -- inventory, the next quotes, the PnL -- follows from it, so
each rule is a separate run of the same strategy on the same data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hypermm.backtest import fills
from hypermm.backtest.market import NEVER, Book, Market
from hypermm.fillsim.models import NS
from hypermm.strategies.quoters import Strategy

SCALE = 1e8


@dataclass
class Order:
    is_ask: bool
    px: int
    placed: int
    trigger: int  # when the rule says it would fill, cancellation aside


@dataclass
class Run:
    """What one strategy did under one rule."""

    strategy: str
    rule: str
    lot: float  # in coins
    fills: list[tuple[int, bool, int]] = field(default_factory=list)  # time, is_ask, px
    marks: list[tuple[int, float]] = field(default_factory=list)  # time, PnL in quote currency
    orders: int = 0

    @property
    def pnl(self) -> float:
        return self.marks[-1][1] if self.marks else 0.0


@dataclass
class View:
    """What a strategy knows when it decides: the book, its inventory, where its own
    orders rest, and the market up to now (strategies read it only at or before `t`)."""

    t: int
    book: Book
    lots: int
    elapsed: float  # seconds since the run started
    resting: dict[bool, tuple[int, ...]]  # prices of its live orders, per side
    market: Market


def run(
    market: Market,
    strategy: Strategy,
    rule: str,
    *,
    lot: float,
    start: int,
    end: int,
    step: int = NS,
    seed: int = 0,
) -> Run:
    """Run `strategy` with decisions every `step` ns over [start, end).

    A strategy may rest several orders on a side (a ladder), one lot each, at
    distinct prices. An order at a price it quotes again is kept, with its place in
    the queue; an order at a price it no longer quotes is cancelled.

    PnL is marked to the mid at every decision: cash plus inventory at the mid, in
    the quote currency, fees left out.
    """
    rng = np.random.default_rng(seed)
    out = Run(strategy.name, rule, lot)
    live: dict[tuple[bool, int], Order] = {}
    lots, cash = 0, 0.0

    def fill(order: Order) -> None:
        nonlocal lots, cash
        out.fills.append((order.trigger, order.is_ask, order.px))
        sign = -1 if order.is_ask else 1  # a filled bid has bought
        lots += sign
        cash -= sign * order.px / SCALE * lot

    for t in range(start, end, step):
        for key in [key for key, order in live.items() if order.trigger <= t]:
            fill(live.pop(key))

        book = market.book(t)
        if book.mid is not None:
            out.marks.append((t, cash + lots * book.mid / SCALE * lot))

        resting = {side: tuple(sorted(px for s, px in live if s == side)) for side in (False, True)}
        view = View(t, book, lots, (t - start) / NS, resting, market)
        bids, asks = strategy.quote(view)
        wanted = {(False, px) for px in bids} | {(True, px) for px in asks}

        for key in [key for key in live if key not in wanted]:
            order = live.pop(key)
            if order.trigger <= fills.cancelled_at(rule, market, t):
                fill(order)  # the trade got there before the cancellation did
        for side, px in sorted(wanted - live.keys()):
            live[(side, px)] = Order(side, px, t, fills.trigger(rule, market, side, px, t, rng))
            out.orders += 1

    for order in live.values():  # what is still resting at the end is filled if its
        if order.trigger <= end:  # trade came before the end
            fill(order)
    final = market.book(end)
    if final.mid is not None:
        out.marks.append((end, cash + lots * final.mid / SCALE * lot))
    return out


def inventory_path(run: Run) -> np.ndarray:
    """Inventory in lots after each fill, in time order."""
    signs = np.array([-1 if is_ask else 1 for _, is_ask, _ in sorted(run.fills)])
    return np.cumsum(signs) if signs.size else np.zeros(0)


__all__ = ["NEVER", "Order", "Run", "View", "inventory_path", "run"]
