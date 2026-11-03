import polars as pl

from hypermm.ingest.book_diffs import NEW, REMOVE, UPDATE
from hypermm.ingest.codes import Status
from hypermm.reconstruct.timing import check, with_time

T0 = 1_764_593_360_000_000_000
BLOCK = 100_000_000  # 100 ms


def diffs(*events):
    """events: (oid, kind) in file order."""
    return pl.DataFrame(
        {
            "seq": list(range(len(events))),
            "oid": [e[0] for e in events],
            "kind": [e[1] for e in events],
        },
        schema_overrides={"seq": pl.UInt32},
    )


def statuses(*events):
    """events: (oid, statusId, block index)."""
    return pl.DataFrame(
        {
            "oid": [e[0] for e in events],
            "statusId": [int(e[1]) for e in events],
            "ts": [T0 + e[2] * BLOCK for e in events],
        },
        schema_overrides={"statusId": pl.UInt8, "ts": pl.UInt64},
    )


def test_an_arrival_takes_the_time_its_order_was_accepted():
    timed = with_time(diffs((1, NEW)), statuses((1, Status.OPEN, 0)))
    assert timed["ts"].to_list() == [T0]
    assert timed["tsExact"].to_list() == [True]


def test_a_removal_takes_the_time_of_whatever_ended_the_order():
    timed = with_time(
        diffs((1, NEW), (1, REMOVE)),
        statuses((1, Status.OPEN, 0), (1, Status.CANCELED, 7)),
    )
    assert timed["ts"].to_list() == [T0, T0 + 7 * BLOCK]
    assert timed["tsExact"].to_list() == [True, True]


def test_a_size_reduction_inherits_the_last_exact_time_before_it():
    timed = with_time(
        diffs((1, NEW), (2, NEW), (1, UPDATE), (1, REMOVE)),
        statuses(
            (1, Status.OPEN, 0),
            (2, Status.OPEN, 3),
            (1, Status.CANCELED, 9),
        ),
    )
    assert timed["ts"].to_list() == [T0, T0 + 3 * BLOCK, T0 + 3 * BLOCK, T0 + 9 * BLOCK]
    assert timed["tsExact"].to_list() == [True, True, False, True]


def test_rejected_orders_never_supply_a_time():
    # a rejection has no book event of its own and must not become another one's anchor
    timed = with_time(
        diffs((1, NEW)),
        statuses((1, Status.OPEN, 4), (2, Status.BAD_ALO_PX_REJECTED, 0)),
    )
    assert timed["ts"].to_list() == [T0 + 4 * BLOCK]


def test_a_rejection_does_not_end_an_order_that_also_rested():
    # the order opened, was later rejected on a re-submission path, then cancelled
    timed = with_time(
        diffs((1, NEW), (1, REMOVE)),
        statuses(
            (1, Status.OPEN, 0),
            (1, Status.CANCELED, 5),
            (1, Status.PERP_MARGIN_REJECTED, 20),
        ),
    )
    assert timed["ts"].to_list() == [T0, T0 + 5 * BLOCK]


def test_time_is_never_allowed_to_run_backwards():
    timed = with_time(
        diffs((1, NEW), (2, NEW), (3, NEW)),
        statuses((1, Status.OPEN, 5), (2, Status.OPEN, 2), (3, Status.OPEN, 8)),
    )
    assert timed["ts"].to_list() == [T0 + 5 * BLOCK, T0 + 5 * BLOCK, T0 + 8 * BLOCK]
    assert timed["tsMoved"].to_list() == [False, True, False]
    assert check(timed)["movedForward"] == 1


def test_an_order_with_no_status_at_all_leaves_the_time_unknown():
    timed = with_time(diffs((9, REMOVE), (1, NEW)), statuses((1, Status.OPEN, 3)))
    assert timed["ts"].to_list() == [None, T0 + 3 * BLOCK]
    assert check(timed) == {"exact": 1, "carried": 0, "unknown": 1, "movedForward": 0}


def test_the_report_counts_each_kind_of_row():
    timed = with_time(
        diffs((1, NEW), (1, UPDATE), (1, REMOVE)),
        statuses((1, Status.OPEN, 0), (1, Status.CANCELED, 4)),
    )
    assert check(timed) == {"exact": 2, "carried": 1, "unknown": 0, "movedForward": 0}
