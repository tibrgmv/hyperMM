import polars as pl

from hypermm.fillcheck.judge import judge

S = 1_000_000_000


def orders(*rows):
    """rows: (rule fires at second or None, filled, ended at second, removed at second or None)."""
    return pl.DataFrame(
        {
            "tTouch": [None if r[0] is None else r[0] * S for r in rows],
            "tVolume": [None] * len(rows),
            "tQueue": [None if r[0] is None else r[0] * S for r in rows],
            "filled": [r[1] for r in rows],
            "tPlaced": [0] * len(rows),
            "duration": [r[2] * S for r in rows],
            "tRemoved": [None if r[3] is None else r[3] * S for r in rows],
        },
        schema_overrides={
            "tTouch": pl.Int64,
            "tVolume": pl.Int64,
            "tQueue": pl.Int64,
            "tPlaced": pl.UInt64,
            "tRemoved": pl.UInt64,
        },
    )


def test_a_rule_fills_only_within_the_life_the_order_had():
    table = judge(orders((3, False, 5, 5), (7, False, 5, 5), (None, False, 5, 5)))
    assert table["tTouchWithin"].to_list() == [True, False, False]


def test_a_trade_in_the_block_of_the_cancellation_fills_unless_cancels_come_first():
    table = orders((5, False, 5, 5))
    assert judge(table)["tQueueWithin"].to_list() == [True]
    assert judge(table, cancel_first=True)["tQueueWithin"].to_list() == [False]


def test_cancels_first_does_not_touch_orders_that_were_filled_or_still_rest():
    table = orders((5, True, 5, 5), (3, False, 9, None))
    assert judge(table, cancel_first=True)["tQueueWithin"].to_list() == [True, True]
