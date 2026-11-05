import math

import polars as pl
import pytest

from hypermm.fillsim.models import (
    clock,
    clock_fit,
    level_trades,
    queue_aware,
    touch,
    volume_through,
)

S = 1_000_000_000
BID = False


def trades(*events):
    """events: (second, price, size, isAggressorAsk)."""
    return level_trades(
        pl.DataFrame(
            {
                "ts": [e[0] * S for e in events],
                "px": [e[1] for e in events],
                "sz": [e[2] for e in events],
                "isAggressorAsk": [e[3] for e in events],
            },
            schema={"ts": pl.Int64, "px": pl.Int64, "sz": pl.Int64, "isAggressorAsk": pl.Boolean},
        )
    )


def orders(*rows):
    """rows: (second placed, price, isAsk, volumeAhead, second at the front or None)."""
    return pl.DataFrame(
        {
            "tPlaced": [r[0] * S for r in rows],
            "px": [r[1] for r in rows],
            "isAsk": [r[2] for r in rows],
            "volumeAhead": [r[3] for r in rows],
            "tFront": [None if r[4] is None else r[4] * S for r in rows],
        },
        schema={
            "tPlaced": pl.UInt64,
            "px": pl.Int64,
            "isAsk": pl.Boolean,
            "volumeAhead": pl.Int64,
            "tFront": pl.UInt64,
        },
    )


def seconds(series):
    return [None if v is None else v // S for v in series.to_list()]


def test_trades_of_one_block_at_one_level_are_merged_and_hit_the_passive_side():
    merged = trades((1, 100, 3, True), (1, 100, 2, True), (2, 100, 4, True))
    assert merged["isAsk"].to_list() == [BID, BID]  # a selling aggressor hits the bids
    assert merged["volume"].to_list() == [5, 4]
    assert merged["traded"].to_list() == [5, 9]


def test_touch_fills_on_the_first_trade_at_the_price_on_the_resting_side():
    hits = trades(
        (1, 100, 5, True),  # before placement
        (3, 99, 5, True),  # another price
        (4, 100, 5, False),  # buyers lifting asks at 100: not the bids
        (6, 100, 1, True),
    )
    assert seconds(touch(orders((2, 100, BID, 500, None)), hits)) == [6]


def test_a_trade_in_the_placement_block_counts_after_placement():
    assert seconds(touch(orders((2, 100, BID, 0, None)), trades((2, 100, 1, True)))) == [2]


def test_volume_through_waits_until_more_than_the_volume_ahead_has_traded():
    hits = trades((1, 100, 9, True), (3, 100, 6, True), (5, 100, 4, True), (7, 100, 1, True))
    # 10 ahead at second 2: 6 + 4 only reaches it, the next trade goes past
    assert seconds(volume_through(orders((2, 100, BID, 10, None)), hits)) == [7]


def test_volume_through_counts_trades_in_the_placement_block():
    hits = trades((2, 100, 3, True), (4, 100, 1, True))
    assert seconds(volume_through(orders((2, 100, BID, 2, None)), hits)) == [2]


def test_queue_aware_fills_on_the_first_trade_once_at_the_front():
    hits = trades((3, 100, 5, True), (6, 100, 1, True))
    table = orders((2, 100, BID, 5, 4), (2, 100, BID, 5, None))
    assert seconds(queue_aware(table, hits)) == [6, None]


def test_an_order_never_traded_at_is_never_filled():
    table = orders((2, 100, BID, 0, 2))
    hits = trades((3, 101, 5, True))
    assert touch(table, hits).to_list() == [None]
    assert volume_through(table, hits).to_list() == [None]
    assert queue_aware(table, hits).to_list() == [None]


def book(span_seconds):
    """A book quoting 100/101 from second 0 to `span_seconds`: tick 1, mid 100.5."""
    return pl.DataFrame(
        {
            "seq": [0, 1, 2],
            "ts": [0, 0, span_seconds * S],
            "oid": [1, 2, 3],
            "kind": ["new", "new", "new"],
            "px": [100, 101, 100],
            "bestBid": [100, 100, 100],
            "bestAsk": [None, 101, 101],
        },
        schema_overrides={"ts": pl.UInt64, "bestAsk": pl.Int64},
    )


def test_the_clock_recovers_an_exponential_decay_of_reach():
    # blocks reaching 0.5, 1.5, 2.5, 3.5, 4.5 ticks above the mid: 32, 16, 8, 4, 4 times,
    # so 64, 32, 16, 8, 4 blocks reach at least each depth -- halving per tick
    deepest = [0] * 32 + [1] * 16 + [2] * 8 + [3] * 4 + [4] * 4
    hits = trades(*((i + 1, 101 + d, 1, False) for i, d in enumerate(deepest)))
    a, k = clock_fit(book(64), hits)  # 64 blocks over 64 s on one of two sides
    assert k == pytest.approx(math.log(2))
    assert a == pytest.approx(0.5 * math.sqrt(2))


def test_the_clock_turns_rate_and_life_into_a_probability():
    table = pl.DataFrame({"oid": [3], "px": [100], "duration": [2 * S]})
    p = clock(table, book(64), a=1.0, k=0.0)  # rate 1 per second, alive 2 s
    assert p.to_list() == [pytest.approx(1 - math.exp(-2))]
