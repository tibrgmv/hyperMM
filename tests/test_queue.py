import polars as pl
import pytest

from hypermm.ingest.book_diffs import NEW, REMOVE, UPDATE
from hypermm.reconstruct.queue import check, with_queue_position


def book(*events):
    """events: (oid, kind, size, prevSize) at one price level, in file order."""
    return pl.DataFrame(
        {
            "seq": list(range(len(events))),
            "oid": [e[0] for e in events],
            "coin": ["SOL"] * len(events),
            "isAsk": [False] * len(events),
            "px": [12560000000] * len(events),
            "kind": [e[1] for e in events],
            "size": [e[2] for e in events],
            "prevSize": [e[3] for e in events],
        },
        schema_overrides={"seq": pl.UInt32, "prevSize": pl.Int64},
    )


def test_first_order_at_an_empty_level_queues_behind_nothing():
    rebuilt = with_queue_position(book((1, NEW, 100, None)))
    assert rebuilt["volumeAhead"].to_list() == [0]


def test_each_arrival_queues_behind_everything_already_resting():
    rebuilt = with_queue_position(
        book((1, NEW, 100, None), (2, NEW, 250, None), (3, NEW, 40, None))
    )
    assert rebuilt["volumeAhead"].to_list() == [0, 100, 350]


def test_a_removal_frees_the_size_the_order_last_had():
    rebuilt = with_queue_position(
        book((1, NEW, 100, None), (2, NEW, 250, None), (1, REMOVE, 0, None), (3, NEW, 7, None))
    )
    # 350 rested, order 1 left with 100, so the newcomer queues behind 250
    assert rebuilt["volumeAhead"].to_list() == [0, 100, None, 250]


def test_a_partial_fill_frees_only_what_it_consumed():
    rebuilt = with_queue_position(
        book((1, NEW, 100, None), (1, UPDATE, 60, 100), (2, NEW, 5, None))
    )
    assert rebuilt["volumeAhead"].to_list() == [0, None, 60]
    assert rebuilt["delta"].to_list() == [100, -40, 5]


def test_levels_are_kept_apart():
    frame = book((1, NEW, 100, None), (2, NEW, 250, None)).with_columns(
        px=pl.Series([12560000000, 12570000000])
    )
    assert with_queue_position(frame)["volumeAhead"].to_list() == [0, 0]


def test_sides_of_the_same_price_are_kept_apart():
    frame = book((1, NEW, 100, None), (2, NEW, 250, None)).with_columns(
        isAsk=pl.Series([False, True])
    )
    assert with_queue_position(frame)["volumeAhead"].to_list() == [0, 0]


def test_a_clean_book_passes_every_check():
    rebuilt = with_queue_position(
        book((1, NEW, 100, None), (2, NEW, 250, None), (1, UPDATE, 60, 100), (1, REMOVE, 0, None))
    )
    assert check(rebuilt) == {
        "coldStartRecovered": 0,
        "coldStartLost": 0,
        "negativeLevel": 0,
        "reArrival": 0,
        "sizeBeforeDisagrees": 0,
        "growingUpdate": 0,
        "nonEmptyRemoval": 0,
    }


def test_an_order_removed_without_ever_arriving_leaves_its_size_unknown():
    rebuilt = with_queue_position(book((99, REMOVE, 0, None), (1, NEW, 100, None)))
    report = check(rebuilt)
    assert report["coldStartLost"] == 1
    # nothing says how much order 99 was resting with, so the newcomer sees an empty level
    assert rebuilt["volumeAhead"].to_list() == [None, 0]


def test_an_order_first_seen_being_filled_has_its_resting_size_recovered():
    # order 99 was resting with 500 before the file opened; the update reveals it
    rebuilt = with_queue_position(book((99, UPDATE, 320, 500), (1, NEW, 10, None)))
    report = check(rebuilt)
    assert report["coldStartRecovered"] == 1
    assert report["growingUpdate"] == 0
    assert report["sizeBeforeDisagrees"] == 0
    assert rebuilt["volumeAhead"].to_list() == [None, 320]


def test_an_update_disagreeing_with_its_own_previous_size_is_caught():
    rebuilt = with_queue_position(book((1, NEW, 100, None), (1, UPDATE, 60, 90)))
    assert check(rebuilt)["sizeBeforeDisagrees"] == 1


@pytest.mark.parametrize(
    "events, flag",
    [
        ([(1, NEW, 100, None), (1, NEW, 50, None)], "reArrival"),
        ([(1, NEW, 100, None), (1, UPDATE, 140, 100)], "growingUpdate"),
        ([(1, NEW, 100, None), (1, REMOVE, 30, None)], "nonEmptyRemoval"),
    ],
)
def test_malformed_histories_are_caught(events, flag):
    assert check(with_queue_position(book(*events)))[flag] == 1
