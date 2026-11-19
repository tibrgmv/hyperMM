"""When a backtested order would be filled, under each fill rule.

The rules are those of hypermm/fillsim/models.py, restated for an order that was
never in the book. One difference: a real order rests at a level that holds it,
so trades at its price are enough; a backtested order may sit at a level with no
real volume, so a trade *through* its price -- below a bid, above an ask -- fills
it as well: the aggressor went past it.

Each rule returns the time of the trade that would fill an order placed at `t`,
ignoring cancellation, or NEVER. The engine compares it with the time the order
was cancelled.

    touch           the first trade at or through the price after placement
    volume-through  ... once the volume traded at the price exceeds the volume
                    resting there at placement (a trade through it always fills)
    queue-aware     the first trade at or through the price once every real order
                    that was ahead has left
    clock           an exponential waiting time at rate A·exp(−k·δ), δ the
                    distance from the mid in ticks at placement
"""

from __future__ import annotations

import numpy as np

from hypermm.backtest.market import NEVER, Market
from hypermm.fillsim.models import NS

RULES = ("touch", "volume-through", "queue-aware", "queue-aware, block", "clock")


def _hits(market: Market, is_ask: bool, px: int, since: int, inclusive: bool):
    """Trades on the order's side from `since` on, and which of them reach its price."""
    side = market.sides[is_ask]  # trades are keyed by the side they hit
    start = int(np.searchsorted(side.ts, since, side="left" if inclusive else "right"))
    ts, prices, volume = side.ts[start:], side.px[start:], side.volume[start:]
    at = prices == px
    through = prices > px if is_ask else prices < px
    return ts, at, through, volume


def _first(ts: np.ndarray, mask: np.ndarray) -> int:
    i = int(np.argmax(mask)) if mask.size else 0
    return int(ts[i]) if mask.size and mask[i] else NEVER


def touch(market: Market, is_ask: bool, px: int, t: int) -> int:
    ts, at, through, _ = _hits(market, is_ask, px, t, inclusive=False)
    return _first(ts, at | through)


def volume_through(market: Market, is_ask: bool, px: int, t: int) -> int:
    ahead = market.level_volume(is_ask, px, t)
    ts, at, through, volume = _hits(market, is_ask, px, t, inclusive=False)
    traded = np.cumsum(np.where(at, volume, 0))
    return _first(ts, through | (traded > ahead))


def queue_aware(market: Market, is_ask: bool, px: int, t: int) -> int:
    front = market.front(is_ask, px, t)
    if front == NEVER:
        return NEVER
    # trades from the block in which it reached the front, and never before it landed
    ts, at, through, _ = _hits(market, is_ask, px, max(front, market.next_block(t)), inclusive=True)
    return _first(ts, at | through)


def clock(market: Market, is_ask: bool, px: int, t: int, rng: np.random.Generator) -> int:
    mid = market.book(t).mid
    if mid is None:
        return NEVER
    delta = abs(px - mid) / market.tick
    rate = market.a * np.exp(-market.k * delta)  # fills per second
    return t + int(rng.exponential(1 / rate) * NS) if rate > 0 else NEVER


def trigger(
    rule: str, market: Market, is_ask: bool, px: int, t: int, rng: np.random.Generator
) -> int:
    if rule == "touch":
        return touch(market, is_ask, px, t)
    if rule == "volume-through":
        return volume_through(market, is_ask, px, t)
    if rule in ("queue-aware", "queue-aware, block"):
        return queue_aware(market, is_ask, px, t)
    if rule == "clock":
        return clock(market, is_ask, px, t, rng)
    raise ValueError(rule)


def cancelled_at(rule: str, market: Market, t: int) -> int:
    """The last moment a trade can still fill an order cancelled at decision `t`.

    The cancellation lands in the next block together with whatever trades that block
    holds. Only the block rule knows that the exchange takes cancellations first;
    every other rule lets that block's trades fill the order.
    """
    if rule == "queue-aware, block":
        return t
    if rule == "clock":
        return t
    return market.next_block(t)
