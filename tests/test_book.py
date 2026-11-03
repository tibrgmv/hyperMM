import polars as pl

from hypermm.ingest.book_diffs import NEW, REMOVE, UPDATE
from hypermm.reconstruct.book import check, check_sweep, sweep, with_top_of_book
from hypermm.reconstruct.queue import with_queue_position

P = 100_000_000  # один «рубль» в единицах 1e-8


def book(*events):
    """events: (oid, isAsk, price, kind, size)."""
    return with_top_of_book(
        with_queue_position(
            pl.DataFrame(
                {
                    "seq": list(range(len(events))),
                    "oid": [e[0] for e in events],
                    "isAsk": [e[1] for e in events],
                    "px": [e[2] * P for e in events],
                    "kind": [e[3] for e in events],
                    "size": [e[4] for e in events],
                    "coin": ["SOL"] * len(events),
                    "prevSize": [None] * len(events),
                },
                schema_overrides={"seq": pl.UInt32, "prevSize": pl.Int64},
            )
        )
    )


def prices(frame, column):
    return [None if v is None else v // P for v in frame[column].to_list()]


def test_the_only_order_is_the_best_on_its_side():
    top = book((1, False, 100, NEW, 50))
    assert prices(top, "bestBid") == [100]
    assert prices(top, "bestAsk") == [None]
    assert top["bestBidVolume"].to_list() == [50]


def test_the_highest_bid_and_the_lowest_ask_are_the_best():
    top = book(
        (1, False, 100, NEW, 50),
        (2, False, 101, NEW, 20),
        (3, True, 105, NEW, 30),
        (4, True, 103, NEW, 10),
    )
    assert prices(top, "bestBid") == [100, 101, 101, 101]
    assert prices(top, "bestAsk") == [None, None, 105, 103]


def test_emptying_the_best_level_falls_back_to_the_next_one():
    top = book(
        (1, False, 100, NEW, 50),
        (2, False, 101, NEW, 20),
        (2, False, 101, REMOVE, 0),
    )
    assert prices(top, "bestBid") == [100, 101, 100]
    assert top["bestBidVolume"].to_list() == [50, 20, 50]


def test_the_best_price_holds_while_the_level_still_has_volume():
    top = book(
        (1, False, 100, NEW, 50),
        (2, False, 100, NEW, 20),
        (1, False, 100, REMOVE, 0),
    )
    assert prices(top, "bestBid") == [100, 100, 100]
    assert top["bestBidVolume"].to_list() == [50, 70, 20]


def test_a_partial_fill_shrinks_the_level_without_moving_the_price():
    top = book((1, False, 100, NEW, 50), (1, False, 100, UPDATE, 30))
    assert prices(top, "bestBid") == [100, 100]
    assert top["bestBidVolume"].to_list() == [50, 30]


def test_a_side_that_empties_completely_has_no_best_price():
    top = book((1, True, 105, NEW, 10), (1, True, 105, REMOVE, 0))
    assert prices(top, "bestAsk") == [105, None]
    assert top["bestAskVolume"].to_list() == [10, 0]


def test_the_two_sides_do_not_see_each_other_at_the_same_price():
    top = book((1, False, 100, NEW, 50), (2, True, 100, NEW, 40))
    assert prices(top, "bestBid") == [100, 100]
    assert prices(top, "bestAsk") == [None, 100]


def test_the_best_price_can_jump_several_levels_at_once():
    top = book(
        (1, False, 100, NEW, 50),
        (2, False, 101, NEW, 20),
        (3, False, 102, NEW, 20),
        (3, False, 102, REMOVE, 0),
        (2, False, 101, REMOVE, 0),
    )
    assert prices(top, "bestBid") == [100, 101, 102, 101, 100]


LADDER = (
    (1, False, 100, NEW, 30),
    (2, False, 98, NEW, 30),
    (3, True, 103, NEW, 10),
    (4, True, 105, NEW, 30),
)


def test_a_small_buy_pays_the_touch_and_ignores_the_bids():
    swept = sweep(book(*LADDER), [3], 10, buying=True)
    assert swept["price"].to_list() == [103 * P]
    assert swept["levels"].to_list() == [1]
    assert swept["short"].to_list() == [0]


def test_a_buy_too_large_for_the_touch_walks_to_the_next_level():
    swept = sweep(book(*LADDER), [3], 25, buying=True)
    assert swept["price"].to_list() == [10_420_000_000]  # (10×103 + 15×105) / 25
    assert swept["levels"].to_list() == [2]


def test_a_sell_walks_down_the_bids():
    swept = sweep(book(*LADDER), [3], 60, buying=False)
    assert swept["price"].to_list() == [99 * P]  # (30×100 + 30×98) / 60
    assert swept["levels"].to_list() == [2]


def test_an_order_bigger_than_the_book_reports_what_it_could_not_get():
    swept = sweep(book(*LADDER), [3], 100, buying=True)
    assert swept["price"].to_list() == [10_450_000_000]  # (10×103 + 30×105) / 40
    assert swept["short"].to_list() == [60]


def test_buying_in_bulk_is_never_cheaper():
    rebuilt = book(*LADDER)
    report = check_sweep(
        sweep(rebuilt, [3], 10, buying=True),
        sweep(rebuilt, [3], 25, buying=True),
        buying=True,
    )
    assert report == {"exhaustedBook": 0, "cheaperInBulk": 0, "shallowerInBulk": 0}


def test_a_healthy_book_passes_the_checks():
    top = book(
        (1, False, 100, NEW, 50),
        (2, True, 103, NEW, 30),
        (3, False, 101, NEW, 10),
    )
    report = check(top)
    assert report["crossed"] == 0
    assert report["bestWithoutVolume"] == 0
    assert report["emptyAsk"] == 1  # пока не пришла первая заявка на продажу
