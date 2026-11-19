import polars as pl

from hypermm.ingest.book_diffs import NEW, REMOVE, UPDATE
from hypermm.pipeline import Reader, chain
from hypermm.reconstruct.queue import check, with_front, with_queue_position


def book(*events):
    """One hourly file at one price level. events: (oid, kind, size, prevSize)."""
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


EARLIER = book((1, NEW, 100, None), (2, NEW, 50, None))
LATER = book((1, UPDATE, 60, 100), (3, NEW, 10, None), (2, REMOVE, 0, None))


def test_seq_runs_on_across_files():
    assert chain([EARLIER, LATER])["seq"].to_list() == [0, 1, 2, 3, 4]


def test_a_later_file_read_alone_starts_cold():
    # order 2 rested from before the file and its removal reveals no size: it is lost
    assert check(with_queue_position(LATER))["coldStartLost"] == 1


def test_chained_after_the_earlier_file_the_book_is_already_built():
    rebuilt = with_queue_position(chain([EARLIER, LATER]))
    counts = check(rebuilt)
    assert counts["coldStartLost"] == counts["coldStartRecovered"] == 0
    # order 3 arrives behind order 1 (60 after its reduction) and order 2 (50)
    assert rebuilt.filter(pl.col("oid") == 3)["volumeAhead"].to_list() == [110]


def test_priority_carries_across_files():
    # order 1 arrived first in the earlier file and keeps the front through its
    # reduction in the later one, so order 3 never leads
    rebuilt = with_front(with_queue_position(chain([EARLIER, LATER])))
    front = dict(rebuilt.group_by("oid").agg(pl.col("frontSeq").first()).iter_rows())
    assert front[1] == 0
    assert front[3] is None  # order 1 still rests ahead of it


def test_a_missing_hourly_file_reads_as_none(tmp_path):
    from datetime import datetime

    reader = Reader(tmp_path)
    assert reader.changes(datetime(2025, 12, 1, 12)) is None
    assert not reader.complete(datetime(2025, 12, 1, 12), "SOL")
