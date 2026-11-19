"""Quoting strategies: given what is known now, where to rest bids and asks.

A strategy returns the prices at which it wants orders resting, per side; the
engine keeps orders at prices quoted again (with their place in the queue) and
cancels the rest. Every order is one lot, post-only: a price that would cross the
book is pulled back to one tick off the other side. Inventory is counted in lots
and capped at `max_lots`: a strategy never rests more orders on a side than would
take it past the cap.

Families (one line each; `VARIANTS` below lists the parameter values):

    offset        rest k ticks behind the best price on both sides (k = 0: join it)
    penny         improve the best price by one tick where the spread allows
    sticky        keep an order while it stays within n ticks of the best price,
                  instead of chasing every move -- and so keep its queue place
    skew          join the best price, shifting both quotes s ticks per lot of
                  inventory to lean against it
    imbalance     join the best price, but stand aside on the thin side when the
                  volume at the best prices leans by more than c
    trade filter  stand aside on a side for w seconds after it was hit by a trade
    vol filter    stand aside on both sides after the mid moved v ticks in a second
    short queue   join the best price only where the queue there is short
    survival      join the best price only where a survival model fitted on the
                  previous hour gives a fill within 5 s a high enough probability
    ladder        rest one lot on each of the n best levels of each side
    A-S           Avellaneda--Stoikov reservation price and spread, rolling horizon tau,
                  optionally shifted by an order-book imbalance signal
    GLFT          Guéant--Lehalle--Fernandez-Tapia closed-form quotes with inventory skew

Prices are integers in units of 1e-8, as everywhere in the project; the formulas
work in ticks. Everything a strategy reads is at or before the decision time.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hypermm.backtest.engine import View
    from hypermm.fillcheck.survival import FillModel

Quotes = tuple[list[int], list[int]]  # bid prices, ask prices
NS = 1_000_000_000


@dataclass
class Params:
    """What a strategy knows about the market, calibrated on the previous hour."""

    tick: int
    sigma: float = 1.0  # ticks per sqrt(second)
    a: float = 1.0  # exponential clock: fills per second at the mid
    k: float = 1.0  # per tick
    queue_limits: dict[float, int] = field(default_factory=dict)  # quantile -> volume
    lot_size: float = 1.0  # coins per lot (a feature of the survival model)
    survival: FillModel | None = None


class Strategy:
    name = "strategy"

    def __init__(self, params: Params, max_lots: int = 5):
        self.p, self.max_lots = params, max_lots

    def quote(self, view: View) -> Quotes:
        bids, asks = self.prices(view)
        return self._clean(view, bids, asks)

    def prices(self, view: View) -> tuple[list[int | None], list[int | None]]:
        raise NotImplementedError

    def _clean(self, view: View, bids, asks) -> Quotes:
        """Post-only, on distinct prices, nearest first, within the inventory cap."""
        book, tick = view.book, self.p.tick
        bids = sorted(
            {min(b, book.ask - tick) if book.ask is not None else b for b in bids if b is not None},
            reverse=True,
        )
        asks = sorted(
            {max(a, book.bid + tick) if book.bid is not None else a for a in asks if a is not None}
        )
        return bids[: max(self.max_lots - view.lots, 0)], asks[: max(self.max_lots + view.lots, 0)]

    def _grid(self, bid_ticks: float, ask_ticks: float) -> tuple[int, int]:
        """Prices in ticks, rounded away from the mid onto the price grid."""
        return math.floor(bid_ticks) * self.p.tick, math.ceil(ask_ticks) * self.p.tick


def _imbalance(view: View) -> float:
    """(bid volume - ask volume) / their sum at the best prices; 0 when both are empty."""
    total = view.book.bid_volume + view.book.ask_volume
    return (view.book.bid_volume - view.book.ask_volume) / total if total else 0.0


# --- families --------------------------------------------------------------------


class Offset(Strategy):
    def __init__(self, params: Params, k: int = 0, max_lots: int = 5):
        super().__init__(params, max_lots)
        self.k = k
        self.name = "symmetric" if k == 0 else f"offset {k}"

    def prices(self, view):
        b, tick = view.book, self.p.tick
        return [b.bid and b.bid - self.k * tick], [b.ask and b.ask + self.k * tick]


class Penny(Strategy):
    name = "penny"

    def prices(self, view):
        b, tick = view.book, self.p.tick
        if b.bid is None or b.ask is None:
            return [b.bid], [b.ask]
        spread = (b.ask - b.bid) // tick
        bid = b.bid + tick if spread >= 2 else b.bid
        ask = b.ask - tick if spread >= 3 else b.ask
        return [bid], [ask]


class Sticky(Strategy):
    def __init__(self, params: Params, n: int = 2, max_lots: int = 5):
        super().__init__(params, max_lots)
        self.n, self.name = n, f"sticky {n}"

    def prices(self, view):
        b, reach = view.book, self.n * self.p.tick
        bids = [px for px in view.resting[False] if b.bid is not None and b.bid - reach < px]
        asks = [px for px in view.resting[True] if b.ask is not None and px < b.ask + reach]
        return bids[:1] or [b.bid], asks[:1] or [b.ask]


class Skew(Strategy):
    def __init__(self, params: Params, s: float = 1.0, max_lots: int = 5):
        super().__init__(params, max_lots)
        self.s, self.name = s, f"skew {s:g}"

    def prices(self, view):
        b = view.book
        shift = round(self.s * view.lots) * self.p.tick
        return [b.bid and b.bid - shift], [b.ask and b.ask - shift]


class Imbalance(Strategy):
    def __init__(self, params: Params, c: float = 0.5, max_lots: int = 5):
        super().__init__(params, max_lots)
        self.c, self.name = c, f"imbalance {c:g}"

    def prices(self, view):
        b, i = view.book, _imbalance(view)
        return [b.bid if i >= -self.c else None], [b.ask if i <= self.c else None]


class TradeFilter(Strategy):
    def __init__(self, params: Params, w: float = 1.0, max_lots: int = 5):
        super().__init__(params, max_lots)
        self.w, self.name = w, f"trade filter {w:g}s"

    def prices(self, view):
        b, m, t = view.book, view.market, view.t
        bid = None if m.recent_blocks(False, t, self.w) else b.bid
        ask = None if m.recent_blocks(True, t, self.w) else b.ask
        return [bid], [ask]


class VolFilter(Strategy):
    def __init__(self, params: Params, v: float = 1.0, max_lots: int = 5):
        super().__init__(params, max_lots)
        self.v, self.name = v, f"vol filter {v:g}"

    def prices(self, view):
        b = view.book
        before = view.market.mid(view.t - NS)
        moved = (
            b.mid is not None and before is not None and abs(b.mid - before) >= self.v * self.p.tick
        )
        return ([None], [None]) if moved else ([b.bid], [b.ask])


class ShortQueue(Strategy):
    def __init__(self, params: Params, q: float = 0.5, max_lots: int = 5):
        super().__init__(params, max_lots)
        self.q, self.name = q, f"short queue {q:g}"

    def prices(self, view):
        b, limit = view.book, self.p.queue_limits.get(self.q, 0)
        return [b.bid if b.bid_volume <= limit else None], [
            b.ask if b.ask_volume <= limit else None
        ]


class Survival(Strategy):
    def __init__(self, params: Params, q: float = 0.5, max_lots: int = 5):
        super().__init__(params, max_lots)
        self.q, self.name = q, f"survival {q:g}"

    def prices(self, view):
        model, b = self.p.survival, view.book
        if model is None or b.mid is None:
            return [None], [None]
        floor = model.thresholds[self.q]
        quotes = []
        for is_ask, px in ((False, b.bid), (True, b.ask)):
            features = model.features_at(view, is_ask, px, self.p)
            quotes.append([px if model.probability(features) >= floor else None])
        return quotes[0], quotes[1]


class Ladder(Strategy):
    def __init__(self, params: Params, n: int = 3, max_lots: int = 5):
        super().__init__(params, max_lots)
        self.n, self.name = n, f"ladder {n}"

    def prices(self, view):
        b, tick = view.book, self.p.tick
        bids = [b.bid - i * tick for i in range(self.n)] if b.bid is not None else []
        asks = [b.ask + i * tick for i in range(self.n)] if b.ask is not None else []
        return bids, asks


class AvellanedaStoikov(Strategy):
    """Reservation price m - q·γσ²τ, half-spread γσ²τ/2 + ln(1+γ/k)/γ, over a rolling
    horizon τ: over the remaining hour the finite-horizon form quotes hundreds of
    ticks wide. `alpha` shifts the reservation price by alpha·imbalance ticks, a
    short-term drift signal in the spirit of Cartea and Jaimungal."""

    def __init__(self, params: Params, gamma=0.1, tau=10.0, alpha=0.0, max_lots: int = 5):
        super().__init__(params, max_lots)
        self.gamma, self.tau, self.alpha = gamma, tau, alpha
        self.name = f"A-S γ{gamma:g} τ{tau:g}" + (f" α{alpha:g}" if alpha else "")

    def prices(self, view):
        b = view.book
        if b.mid is None:
            return [None], [None]
        g, s, k = self.gamma, self.p.sigma, self.p.k
        mid = b.mid / self.p.tick + self.alpha * _imbalance(view)
        reservation = mid - view.lots * g * s**2 * self.tau
        half = g * s**2 * self.tau / 2 + math.log(1 + g / k) / g
        bid, ask = self._grid(reservation - half, reservation + half)
        return [bid], [ask]


class GLFT(Strategy):
    """δ = ln(1+γ/k)/γ ± (2q ± 1)/2 · ω, ω = sqrt(σ²γ/(2kA) · (1+γ/k)^(1+k/γ))."""

    def __init__(self, params: Params, gamma: float = 0.1, max_lots: int = 5):
        super().__init__(params, max_lots)
        self.gamma = gamma
        self.name = f"GLFT γ{gamma:g}" + (f" cap {max_lots}" if max_lots != 5 else "")

    def prices(self, view):
        b = view.book
        if b.mid is None:
            return [None], [None]
        g, s, k, a, q = self.gamma, self.p.sigma, self.p.k, self.p.a, view.lots
        base = math.log(1 + g / k) / g
        skew = math.sqrt(s**2 * g / (2 * k * a) * (1 + g / k) ** (1 + k / g))
        mid = b.mid / self.p.tick
        bid, ask = self._grid(
            mid - (base + (2 * q + 1) / 2 * skew), mid + (base - (2 * q - 1) / 2 * skew)
        )
        return [bid], [ask]


# --- the variants backtested ----------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    family: str
    make: Callable[[Params], Strategy]
    step: float = 1.0  # seconds between decisions

    def build(self, params: Params) -> Strategy:
        strategy = self.make(params)
        if self.step != 1.0:
            strategy.name += f" @{self.step:g}s"
        return strategy


def _variants() -> list[Variant]:
    v: list[Variant] = []
    v += [Variant("offset", lambda p, k=k: Offset(p, k)) for k in (0, 1, 2, 5)]
    v += [Variant("penny", Penny)]
    v += [Variant("sticky", lambda p, n=n: Sticky(p, n)) for n in (1, 2, 3)]
    v += [Variant("skew", lambda p, s=s: Skew(p, s)) for s in (0.5, 1, 2)]
    v += [Variant("imbalance", lambda p, c=c: Imbalance(p, c)) for c in (0.3, 0.6)]
    v += [Variant("trade filter", lambda p, w=w: TradeFilter(p, w)) for w in (1, 5)]
    v += [Variant("vol filter", lambda p, x=x: VolFilter(p, x)) for x in (1, 2)]
    v += [Variant("short queue", lambda p, q=q: ShortQueue(p, q)) for q in (0.25, 0.5, 0.75)]
    v += [Variant("survival", lambda p, q=q: Survival(p, q)) for q in (0.5, 0.75)]
    v += [Variant("ladder", lambda p, n=n: Ladder(p, n)) for n in (3, 5)]
    v += [
        Variant("A-S", lambda p, g=g, t=t: AvellanedaStoikov(p, g, t))
        for g, t in ((0.1, 10), (0.05, 10), (0.5, 10), (0.1, 3), (0.1, 30))
    ]
    v += [Variant("A-S", lambda p, a=a: AvellanedaStoikov(p, 0.1, 10, a)) for a in (0.5, 1.0)]
    v += [Variant("GLFT", lambda p, g=g: GLFT(p, g)) for g in (0.05, 0.1, 0.5)]
    v += [Variant("offset", lambda p, m=m: _capped(Offset(p, 0, m), m)) for m in (2, 10)]
    v += [Variant("GLFT", lambda p, m=m: GLFT(p, 0.1, m)) for m in (2, 10)]
    # reaction speed: the same strategies deciding ten times as often, and five times as rarely
    fast_and_slow = [
        ("offset", lambda p: Offset(p, 0)),
        ("A-S", lambda p: AvellanedaStoikov(p)),
        ("GLFT", lambda p: GLFT(p)),
        ("imbalance", lambda p: Imbalance(p, 0.3)),
        ("trade filter", lambda p: TradeFilter(p, 1)),
    ]
    v += [Variant(family, make, step) for family, make in fast_and_slow for step in (0.1, 5.0)]
    return v


def _capped(strategy: Strategy, cap: int) -> Strategy:
    strategy.name += f" cap {cap}"
    return strategy


VARIANTS = _variants()
# the four strategies of the first experiment, kept for comparison
BASELINE = ("symmetric", "short queue 0.5", "A-S γ0.1 τ10", "GLFT γ0.1")
__all__ = ["BASELINE", "VARIANTS", "Params", "Strategy", "Variant"]
