from decimal import Decimal

import polars as pl
import pytest

from hypermm.ingest.decimals import parse_decimal
from hypermm.ingest.schema import SCALE

EXACT = ["2964.0", "2962.3", "0.2274", "5.0632", "96543.21", "2964", "0.00000001"]


def parsed(values):
    return pl.DataFrame({"v": values}).select(parse_decimal("v"))["v"].to_list()


@pytest.mark.parametrize("text", EXACT)
def test_matches_exact_decimal_arithmetic(text):
    assert parsed([text])[0] == int(Decimal(text) * SCALE)


def test_survives_the_top_of_the_range_where_float_does_not():
    text = "99999999.99999999"
    assert parsed([text])[0] == int(Decimal(text) * SCALE)
    assert int(float(text) * SCALE) != int(Decimal(text) * SCALE)


def test_handles_negative_values():
    # trade side_info carries signed positions
    assert parsed(["-2.0", "-0.5"]) == [-2 * SCALE, -SCALE // 2]


def test_agrees_with_the_binary_stream_encoding():
    # the same price reached from the packed uint32 and from the JSON text
    from hypermm.ingest.schema import decode_fixed

    assert parsed(["2964.3"])[0] == decode_fixed([(1 << 29) | 29643])[0]


def test_too_many_places_yields_null_not_a_wrong_number():
    assert parsed(["1.123456789"]) == [None]
