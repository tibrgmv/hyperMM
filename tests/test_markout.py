import polars as pl
import pytest

from hypermm.toxicity.markout import markout

S = 1_000_000_000


def mids(*rows):
    """rows: (second, best bid, best ask)."""
    return pl.DataFrame(
        {
            "ts": [r[0] * S for r in rows],
            "bestBid": [r[1] for r in rows],
            "bestAsk": [r[2] for r in rows],
        },
        schema={"ts": pl.UInt64, "bestBid": pl.Int64, "bestAsk": pl.Int64},
    )


def fill(second, px, is_ask):
    return pl.DataFrame({"t": [second * S], "px": [px], "isAsk": [is_ask]})


BOOK = mids((0, 9_999, 10_001), (5, 9_989, 9_991))  # mid 10 000, then 9 990 from second 5


def test_a_bid_filled_before_the_price_falls_loses():
    # bought at 10 000; ten seconds on the mid is 9 990: 10 bps lost
    assert markout(fill(1, 10_000, False), BOOK, 10).to_list() == [pytest.approx(-10)]


def test_an_ask_filled_before_the_price_falls_gains():
    assert markout(fill(1, 10_000, True), BOOK, 10).to_list() == [pytest.approx(10)]


def test_the_mid_is_taken_as_it_stood_at_the_horizon():
    # one second on, the mid has not moved yet
    assert markout(fill(1, 10_000, False), BOOK, 1).to_list() == [pytest.approx(0)]


def test_a_horizon_past_the_observed_window_is_unknown():
    fills = pl.concat([fill(1, 10_000, False), fill(8, 10_000, False)])
    # the window ends at second 12: the first horizon (11 s) is inside it, the second (18 s) not
    assert markout(fills, BOOK, 10, until=12 * S).to_list() == [pytest.approx(-10), None]
