import polars as pl
import pytest

from hypermm.backtest import engine, fills
from hypermm.backtest.market import NEVER, Market
from hypermm.ingest.book_diffs import NEW, REMOVE, UPDATE
from hypermm.reconstruct.book import with_top_of_book
from hypermm.reconstruct.queue import with_queue_position
from hypermm.strategies.quoters import Ladder, Params, Skew, Sticky, Strategy, TradeFilter

S = 1_000_000_000
BID, ASK = False, True

# second, oid, isAsk, px, kind, size, prevSize
EVENTS = [
    (1, 1, BID, 100, NEW, 5, None),
    (1, 2, ASK, 101, NEW, 5, None),  # rests for good
    (2, 3, BID, 100, NEW, 3, None),
    (4, 1, BID, 100, REMOVE, 0, None),  # cancelled
    (6, 3, BID, 100, UPDATE, 1, 3),  # 2 traded
    (8, 3, BID, 100, REMOVE, 0, None),  # its last 1 traded
    (10, 4, BID, 99, NEW, 1, None),
    (10, 4, BID, 99, REMOVE, 0, None),  # taken by a trade below 100
    (12, 5, BID, 100, NEW, 2, None),
    (12, 5, BID, 100, REMOVE, 0, None),
    (14, 6, ASK, 102, NEW, 1, None),
]
# second, px, size, isAggressorAsk (sellers hit bids)
TRADES = [(6, 100, 2, True), (8, 100, 1, True), (10, 99, 1, True), (12, 100, 2, True)]


@pytest.fixture(scope="module")
def market():
    diffs = pl.DataFrame(
        {
            "seq": list(range(len(EVENTS))),
            "oid": [e[1] for e in EVENTS],
            "coin": ["SOL"] * len(EVENTS),
            "isAsk": [e[2] for e in EVENTS],
            "px": [e[3] for e in EVENTS],
            "kind": [e[4] for e in EVENTS],
            "size": [e[5] for e in EVENTS],
            "prevSize": [e[6] for e in EVENTS],
        },
        schema_overrides={"seq": pl.UInt32, "prevSize": pl.Int64},
    )
    top = with_top_of_book(
        with_queue_position(diffs).with_columns(
            ts=pl.Series([e[0] * S for e in EVENTS], dtype=pl.UInt64)
        )
    )
    trades = pl.DataFrame(
        {
            "ts": [t[0] * S for t in TRADES],
            "px": [t[1] for t in TRADES],
            "sz": [t[2] for t in TRADES],
            "isAggressorAsk": [t[3] for t in TRADES],
        },
        schema={"ts": pl.Int64, "px": pl.Int64, "sz": pl.Int64, "isAggressorAsk": pl.Boolean},
    )
    return Market(top, trades)


def test_touch_fills_on_the_first_trade_at_the_price_after_placement(market):
    assert fills.touch(market, BID, 100, 3 * S) == 6 * S


def test_queue_aware_waits_until_every_order_ahead_has_left(market):
    # orders 1 and 3 were resting when it landed; order 3 leaves at second 8
    assert market.front(BID, 100, 3 * S) == 8 * S
    assert fills.queue_aware(market, BID, 100, 3 * S) == 8 * S


def test_volume_through_waits_for_the_volume_resting_at_placement_or_a_trade_through(market):
    # 8 resting at landing; 3 trade at 100, never more than 8: the trade at 99 goes through
    assert market.level_volume(BID, 100, 3 * S) == 8
    assert fills.volume_through(market, BID, 100, 3 * S) == 10 * S


def test_an_order_at_an_empty_level_leads_it_and_a_trade_through_fills_it(market):
    assert market.front(BID, 100, 9 * S) == 10 * S
    assert fills.queue_aware(market, BID, 100, 9 * S) == 10 * S


def test_an_order_behind_one_that_never_leaves_never_reaches_the_front(market):
    assert market.front(ASK, 101, 3 * S) == NEVER
    assert fills.queue_aware(market, ASK, 101, 3 * S) == NEVER


def test_only_the_block_rule_lets_a_cancellation_beat_the_trade_in_its_block(market):
    # cancelled at second 7: the cancellation lands in the block of second 8,
    # which also holds the trade that would fill the order
    assert fills.cancelled_at("queue-aware, block", market, 7 * S) == 7 * S
    assert fills.cancelled_at("queue-aware", market, 7 * S) == 8 * S


class Bids(Strategy):
    """Rests one bid at 100, nothing on the ask."""

    name = "bids"

    def prices(self, view):
        return [100], []


def test_the_engine_books_a_fill_and_marks_the_inventory_to_the_mid(market):
    lot = 1e8  # one lot is worth its price in units of 1e-8
    run = engine.run(
        market, Bids(Params(tick=1), max_lots=1), "touch", lot=lot, start=3 * S, end=7 * S, step=S
    )
    assert run.fills == [(6 * S, BID, 100)]
    # bought at 100; the mid at second 7 is (100 + 101) / 2
    assert run.pnl == pytest.approx(0.5)


def test_a_rule_that_never_fills_leaves_the_strategy_flat(market):
    run = engine.run(
        market, Bids(Params(tick=1)), "queue-aware, block", lot=1e8, start=3 * S, end=7 * S, step=S
    )
    assert run.fills == []
    assert run.pnl == 0


def view(market, t, lots=0, resting=None):
    return engine.View(t, market.book(t), lots, 0.0, resting or {BID: (), ASK: ()}, market)


def test_a_ladder_rests_one_lot_on_each_of_the_best_levels(market):
    bids, asks = Ladder(Params(tick=1), 3).quote(view(market, 3 * S))
    assert bids == [100, 99, 98]
    assert asks == [101, 102, 103]


def test_the_inventory_cap_limits_how_many_orders_a_side_may_rest(market):
    bids, asks = Ladder(Params(tick=1), 3, max_lots=5).quote(view(market, 3 * S, lots=4))
    assert bids == [100]  # one more lot would reach the cap
    assert len(asks) == 3


def test_sticky_keeps_an_order_within_reach_and_chases_one_left_behind(market):
    # at second 3 the best bid is 100
    near = view(market, 3 * S, resting={BID: (99,), ASK: ()})
    assert Sticky(Params(tick=1), 2).quote(near)[0] == [99]  # one tick behind: kept
    far = view(market, 3 * S, resting={BID: (97,), ASK: ()})
    assert Sticky(Params(tick=1), 2).quote(far)[0] == [100]  # three behind: back to the best


def test_skew_leans_both_quotes_against_the_inventory(market):
    bids, asks = Skew(Params(tick=1), 1).quote(view(market, 3 * S, lots=2))
    # long two lots: both quotes two ticks lower; the ask is pulled back off the bid
    assert bids == [98]
    assert asks == [101]


def test_the_trade_filter_stands_aside_on_a_side_just_hit(market):
    # sellers hit the bids at second 6
    bids, asks = TradeFilter(Params(tick=1), 1).quote(view(market, 6 * S + S // 2))
    assert bids == []
    assert asks == [101]
