import polars as pl
import pytest

from hypermm.fillcheck.survival import survival_table

S = 1_000_000_000
TICK = 100


def book(*rows):
    """rows: (second, kind, oid, best bid, best ask, bid volume, ask volume) after the change."""
    return pl.DataFrame(
        {
            "seq": list(range(len(rows))),
            "ts": [r[0] * S for r in rows],
            "kind": [r[1] for r in rows],
            "oid": [r[2] for r in rows],
            "bestBid": [r[3] for r in rows],
            "bestAsk": [r[4] for r in rows],
            "bestBidVolume": [r[5] for r in rows],
            "bestAskVolume": [r[6] for r in rows],
        },
        schema_overrides={"ts": pl.UInt64},
    )


def orders(*rows):
    """rows: (oid, second placed, px, isAsk, volume ahead, second at front, seconds lived,
    filled, second removed)."""
    return pl.DataFrame(
        {
            "oid": [r[0] for r in rows],
            "tPlaced": [r[1] * S for r in rows],
            "px": [r[2] for r in rows],
            "isAsk": [r[3] for r in rows],
            "restingSize": [100_000_000] * len(rows),
            "volumeAhead": [r[4] for r in rows],
            "tFront": [None if r[5] is None else r[5] * S for r in rows],
            "duration": [r[6] * S for r in rows],
            "filled": [r[7] for r in rows],
            "tRemoved": [None if r[8] is None else r[8] * S for r in rows],
        },
        schema_overrides={"tPlaced": pl.UInt64, "tFront": pl.UInt64, "tRemoved": pl.UInt64},
    )


def trades(*rows):
    """rows: (second, px, isAggressorAsk)."""
    return pl.DataFrame(
        {
            "ts": [r[0] * S for r in rows],
            "px": [r[1] for r in rows],
            "sz": [1] * len(rows),
            "isAggressorAsk": [r[2] for r in rows],
        },
        schema={"ts": pl.Int64, "px": pl.Int64, "sz": pl.Int64, "isAggressorAsk": pl.Boolean},
    )


# bid 10 000, ask 10 100 (one tick). Order 7 joins the best bid; once the bid has fallen
# to 9 900, order 8 improves it back to 10 000; order 9 is an ask behind the best ask.
BOOK = book(
    (0, "new", 1, 10_000, 10_100, 300, 100),
    (5, "new", 2, 10_000, 10_100, 300, 200),
    (10, "new", 7, 10_000, 10_100, 500, 200),  # order 7 joins the best bid
    (12, "remove", 1, 9_900, 10_100, 50, 200),
    (20, "new", 8, 10_000, 10_100, 10, 200),  # order 8 improves the bid back to 10_000
    (25, "new", 9, 9_900, 10_300, 50, 200),
)


def test_features_describe_the_book_before_the_order_arrived():
    table = orders(
        (7, 10, 10_000, False, 300, 15, 10, True, None),
        (8, 20, 10_000, False, 0, 20, 3, False, 23),
    )
    out = survival_table(table, BOOK, trades((1, 10_000, False)))
    # order 7 joined the best bid; order 8 came in one tick above the best bid of 9 900
    assert out["ticksBehind"].to_list() == [0, -1]
    # imbalance on the order's own side, before it: (300 - 200) / 500 and (50 - 200) / 250
    assert out["imbalance"].to_list() == [pytest.approx(0.2), pytest.approx(-0.6)]
    assert out["emptyLevel"].to_list() == [False, True]
    # mids before them: 10 050 and 10 000 -- half a tick and no distance
    assert out["ticksFromMid"].to_list() == [pytest.approx(0.5), pytest.approx(0)]


def test_the_two_stages_split_the_life_of_an_order():
    table = orders(
        (7, 10, 10_000, False, 300, 15, 10, True, None),  # front after 5 s, filled 5 s later
        (8, 20, 10_000, False, 0, None, 3, False, 23),  # cancelled before the front
    )
    out = survival_table(table, BOOK, trades((1, 10_000, False)))
    assert out["frontDuration"].to_list() == [5, 3]
    assert out["frontEvent"].to_list() == [True, False]
    assert out["fillDuration"].to_list() == [5, None]
    assert out["fillEvent"].to_list() == [True, None]
    assert out["totalEvent"].to_list() == [True, False]
    assert out["censoredBy"].to_list() == [None, "cancelled"]


def test_an_order_still_resting_when_observation_ends_is_censored_by_the_window():
    table = orders((7, 10, 10_000, False, 300, None, 50, False, None))
    out = survival_table(table, BOOK, trades((1, 10_000, False)))
    assert out["censoredBy"].to_list() == ["window"]


def test_recent_trades_count_only_earlier_blocks_hitting_the_order_side():
    table = orders((9, 25, 10_300, True, 0, None, 1, False, 26))
    hits = trades(
        (16, 10_100, False),  # buyers lifting asks, 9 s before: in the 10 s window only
        (24, 10_100, False),  # 1 s before: windows are [placed - w, placed), so it counts
        (24, 10_100, False),  # the same block again: one block, not two
        (24, 9_900, True),  # sellers hitting bids: the other side
        (25, 10_100, False),  # the placement block itself: not before the order
    )
    out = survival_table(table, BOOK, hits)
    assert out["trades10s"].to_list() == [2]
    assert out["trades1s"].to_list() == [1]


def test_a_price_coming_towards_the_order_counts_as_a_positive_move():
    # the mid fell from 10 050 (second 10) to 10 000 (second 12): 0.5 tick, towards a bid
    table = orders((8, 20, 10_000, False, 0, 20, 1, False, 21))
    out = survival_table(table, BOOK, trades((1, 10_000, False)))
    assert out["move10s"].to_list() == [pytest.approx(0.5)]
