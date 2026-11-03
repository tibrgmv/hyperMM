import polars as pl

from hypermm.ingest.book_diffs import NEW, REMOVE, UPDATE
from hypermm.reconstruct.outcomes import CANCELLED, FILLED, RESTING, check, maker_fills, outcomes

T0 = 1_764_593_360_000_000_000
S = 1_000_000_000  # секунда


def diffs(*events):
    """events: (oid, kind, size, volumeAhead, second)."""
    return pl.DataFrame(
        {
            "oid": [e[0] for e in events],
            "kind": [e[1] for e in events],
            "size": [e[2] for e in events],
            "volumeAhead": [e[3] for e in events],
            "ts": [T0 + e[4] * S for e in events],
            "coin": ["SOL"] * len(events),
            "isAsk": [False] * len(events),
            "px": [12629000000] * len(events),
        },
        schema_overrides={"ts": pl.UInt64, "volumeAhead": pl.Int64},
    )


def trades(*fills):
    """fills: (oid0, oid1, size, second)."""
    return pl.DataFrame(
        {
            "oid0": [f[0] for f in fills],
            "oid1": [f[1] for f in fills],
            "sz": [f[2] for f in fills],
            "ts": [T0 + f[3] * S for f in fills],
            "px": [12629000000] * len(fills),
            "isAggressorAsk": [True] * len(fills),
        },
        schema={
            "oid0": pl.Int64,
            "oid1": pl.Int64,
            "sz": pl.Int64,
            "ts": pl.Int64,
            "px": pl.Int64,
            "isAggressorAsk": pl.Boolean,
        },
    )


def test_an_order_filled_in_full_is_an_observed_event():
    table = outcomes(
        diffs((1, NEW, 100, 500, 0), (1, REMOVE, 0, None, 7)),
        trades((1, 99, 100, 7)),
    )
    assert table["outcome"].to_list() == [FILLED]
    assert table["filled"].to_list() == [True]
    assert table["duration"].to_list() == [7 * S]


def test_an_order_cancelled_untouched_is_censored_by_its_owner():
    table = outcomes(diffs((1, NEW, 100, 500, 0), (1, REMOVE, 0, None, 4)), trades())
    assert table["outcome"].to_list() == [CANCELLED]
    assert table["filled"].to_list() == [False]
    assert table["duration"].to_list() == [4 * S]


def test_an_order_still_resting_is_censored_by_the_window():
    table = outcomes(diffs((1, NEW, 100, 500, 0), (2, NEW, 10, 600, 9)), trades())
    resting = table.filter(pl.col("oid") == 1)
    assert resting["outcome"].to_list() == [RESTING]
    assert resting["duration"].to_list() == [9 * S]  # до конца окна, а не до отмены


def test_a_partial_fill_stops_the_clock_even_if_the_rest_is_cancelled():
    table = outcomes(
        diffs((1, NEW, 100, 500, 0), (1, UPDATE, 60, None, 3), (1, REMOVE, 0, None, 8)),
        trades((1, 99, 40, 3)),
    )
    # событие наступило на первом исполнении, отмена остатка на него не влияет
    assert table["filled"].to_list() == [True]
    assert table["duration"].to_list() == [3 * S]
    assert table["outcome"].to_list() == [CANCELLED]
    assert table["filledSize"].to_list() == [40]


def test_every_fill_of_an_order_is_counted():
    table = outcomes(
        diffs((1, NEW, 100, 0, 0), (1, UPDATE, 70, None, 2), (1, UPDATE, 40, None, 5)),
        trades((1, 98, 30, 2), (99, 1, 30, 5)),
    )
    assert table["fills"].to_list() == [2]
    assert table["filledSize"].to_list() == [60]
    assert table["duration"].to_list() == [2 * S]


def test_the_resting_side_is_recognised_wherever_it_sits_in_the_record():
    fills = maker_fills(diffs((1, NEW, 100, 0, 0)), trades((99, 1, 10, 1), (1, 98, 10, 2)))
    assert fills["oid"].to_list() == [1, 1]


def test_an_order_already_resting_when_the_file_opened_can_still_be_the_maker():
    # заявка 7 не имеет постановки в файле, но присутствует в нём
    fills = maker_fills(diffs((7, UPDATE, 40, None, 1)), trades((7, 99, 10, 1)))
    assert fills["oid"].to_list() == [7]


def test_when_both_sides_are_in_the_book_the_earlier_one_was_waiting():
    # заявка 1 стояла, заявка 2 пересекла спред и осела остатком
    fills = maker_fills(
        diffs((1, NEW, 100, 0, 0), (2, NEW, 50, 100, 5)),
        trades((2, 1, 10, 5)),
    )
    assert fills["oid"].to_list() == [1]


def test_orders_already_resting_are_left_out_of_the_table():
    table = outcomes(diffs((7, REMOVE, 0, None, 1), (1, NEW, 100, 0, 2)), trades())
    assert table["oid"].to_list() == [1]


def test_a_well_formed_table_passes_every_check():
    table = outcomes(
        diffs((1, NEW, 100, 500, 0), (1, REMOVE, 0, None, 7), (2, NEW, 20, 0, 1)),
        trades((1, 99, 100, 7)),
    )
    assert check(table) == {
        "negativeDuration": 0,
        "fillWithoutTime": 0,
        "fillBeforePlacement": 0,
        "removalBeforePlacement": 0,
        "overfilled": 0,
        "restingButRemoved": 0,
        "duplicated": 0,
    }
