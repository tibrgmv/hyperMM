import polars as pl

from hypermm.evaluate.price_accuracy import bars, check, compare
from hypermm.ingest.book_diffs import NEW
from tests.test_book import P, book

S = 1_000_000_000
MINUTE = 60 * S


def trades(*events):
    """events: (second, price, size)."""
    return pl.DataFrame(
        {
            "ts": [e[0] * S for e in events],
            "px": [e[1] * P for e in events],
            "sz": [e[2] * P for e in events],
        }
    )


def timed(rebuilt, *seconds):
    return rebuilt.with_columns(ts=pl.Series([s * S for s in seconds]))


def test_a_bar_opens_on_its_first_trade_and_closes_on_its_last():
    table = bars(trades((10, 100, 1), (20, 102, 1), (70, 105, 1)))
    assert table["bar"].to_list() == [0, 1]
    assert [v // P for v in table["open"].to_list()] == [100, 105]
    assert [v // P for v in table["close"].to_list()] == [102, 105]
    assert [None if v is None else v // P for v in table["nextOpen"].to_list()] == [105, None]


def test_the_bar_vwap_leans_towards_the_larger_trade():
    table = bars(trades((10, 100, 3), (20, 108, 1)))
    assert table["vwap"].to_list() == [102 * P]  # (3×100 + 1×108) / 4


def test_the_next_bar_open_flatters_a_buy_that_had_to_cross_the_spread():
    rebuilt = timed(
        book(
            (1, False, 100, NEW, 30),
            (2, True, 102, NEW, 10),
            (3, True, 104, NEW, 10),
        ),
        1,
        2,
        3,
    )
    compared = compare(rebuilt, trades((10, 101, 1), (70, 101, 1)), 20, buying=True)

    assert compared["price"].to_list() == [103 * P]  # (10×102 + 10×104) / 20
    assert compared["levels"].to_list() == [2]
    # Реально заплачено 103, бэктест насчитал бы 101 — подарок в 2 при середине 101.
    assert compared["nextOpenBps"].to_list() == [10_000 * 2 / 101]
    assert compared["midBps"].to_list() == [10_000 * 2 / 101]
    assert compared["touchBps"].to_list() == [10_000 * 1 / 101]


def test_the_same_gift_runs_the_other_way_for_a_sell():
    rebuilt = timed(
        book(
            (1, True, 102, NEW, 30),
            (2, False, 100, NEW, 10),
            (3, False, 98, NEW, 10),
        ),
        1,
        2,
        3,
    )
    compared = compare(rebuilt, trades((10, 101, 1), (70, 101, 1)), 20, buying=False)

    assert compared["price"].to_list() == [99 * P]  # (10×100 + 10×98) / 20
    assert compared["nextOpenBps"].to_list() == [10_000 * 2 / 101]


def test_an_honest_comparison_passes_the_checks():
    rebuilt = timed(
        book((1, False, 100, NEW, 30), (2, True, 102, NEW, 30)),
        1,
        2,
    )
    compared = compare(rebuilt, trades((10, 101, 1), (70, 101, 1)), 10, buying=True)
    assert check(compared) == {
        "beatTheTouch": 0,
        "noBook": 0,
        "exhaustedBook": 0,
    }
