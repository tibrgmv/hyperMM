import polars as pl
import pytest

from hypermm.ingest.book_diffs import NEW, REMOVE, UPDATE
from hypermm.reconstruct.queue import ahead_path, check, with_front, with_queue_position


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


def front(*events):
    """frontSeq of each row, for events at one level as in `book`."""
    return with_front(with_queue_position(book(*events)))["frontSeq"].to_list()


def test_an_order_joining_an_empty_level_is_at_the_front_at_once():
    assert front((1, NEW, 100, None)) == [0]


def test_an_order_reaches_the_front_when_the_one_ahead_leaves():
    assert front((1, NEW, 100, None), (2, NEW, 50, None), (1, REMOVE, 0, None)) == [0, 2, 0]


def test_a_partial_fill_ahead_does_not_bring_the_front():
    rows = front((1, NEW, 100, None), (2, NEW, 50, None), (1, UPDATE, 60, 100))
    assert rows == [0, None, 0]


def test_the_front_passes_down_the_queue_in_arrival_order():
    rows = front(
        (1, NEW, 100, None),
        (2, NEW, 50, None),
        (3, NEW, 20, None),
        (2, REMOVE, 0, None),  # 3 still waits behind 1
        (1, REMOVE, 0, None),  # 3 is now first
    )
    assert rows == [0, None, 4, None, 0]


def test_an_order_cancelled_while_still_queued_never_reaches_the_front():
    rows = front((1, NEW, 100, None), (2, NEW, 50, None), (2, REMOVE, 0, None))
    assert rows == [0, None, None]


def test_an_order_arriving_after_the_level_emptied_is_first_at_once():
    rows = front((1, NEW, 100, None), (1, REMOVE, 0, None), (2, NEW, 50, None))
    assert rows == [0, 0, 2]


def test_other_levels_do_not_hold_an_order_back():
    frame = book((1, NEW, 100, None), (2, NEW, 50, None)).with_columns(
        px=pl.Series([12560000000, 12570000000])
    )
    assert with_front(with_queue_position(frame))["frontSeq"].to_list() == [0, 1]


def test_the_path_ahead_falls_by_what_earlier_orders_lose_and_ignores_later_ones():
    rebuilt = with_queue_position(
        book(
            (1, NEW, 100, None),
            (2, NEW, 50, None),
            (3, NEW, 20, None),  # the order followed
            (4, NEW, 70, None),  # behind it: never counts
            (1, UPDATE, 60, 100),
            (4, REMOVE, 0, None),
            (2, REMOVE, 0, None),
            (1, REMOVE, 0, None),
            (3, REMOVE, 0, None),
        )
    )
    path = ahead_path(rebuilt, 3)
    assert path["seq"].to_list() == [2, 4, 6, 7]
    assert path["ahead"].to_list() == [150, 110, 60, 0]
    # the path reaches zero exactly where with_front says the order led the level
    assert with_front(rebuilt).filter(pl.col("oid") == 3)["frontSeq"][0] == 7
