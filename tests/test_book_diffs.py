import gzip
import json

import pytest

from hypermm.ingest.book_diffs import NEW, REMOVE, UPDATE, hour_path, read_hour
from hypermm.ingest.schema import SCALE

# The three record shapes exactly as published in SCHEMA.md.
RECORDS = [
    {
        "user": "0x31ca8395cf837de08b24da3f660e77761dfb974b",
        "oid": 273024455731,
        "coin": "ETH",
        "side": "A",
        "px": "2964.0",
        "raw_book_diff": {"new": {"sz": "0.2274"}},
    },
    {
        "user": "0x6ba889db7f923622d3548f621ecc2054b80c1817",
        "oid": 273024450830,
        "coin": "ETH",
        "side": "B",
        "px": "2962.3",
        "raw_book_diff": "remove",
    },
    {
        "user": "0xecb63caa47c7c4e77f60f1ce858cf28dc2b82b00",
        "oid": 273024455174,
        "coin": "BTC",
        "side": "B",
        "px": "2962.5",
        "raw_book_diff": {"update": {"origSz": "5.0632", "newSz": "4.7287"}},
    },
]


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / "20251215" / "ex14.gz"
    path.parent.mkdir()
    body = "\n".join(json.dumps(record) for record in RECORDS)
    path.write_bytes(gzip.compress(body.encode()))
    return path


def test_hour_path_leaves_the_hour_unpadded(tmp_path):
    assert hour_path(tmp_path, "2025-12-15", 7).name == "ex7.gz"
    assert hour_path(tmp_path, "20251215", 14).name == "ex14.gz"
    assert hour_path(tmp_path, "2025-12-15", 0).parent.name == "20251215"


def test_classifies_all_three_kinds_of_change(sample):
    assert read_hour(sample)["kind"].to_list() == [NEW, REMOVE, UPDATE]


def test_size_is_what_rests_on_the_book_after_the_change(sample):
    assert read_hour(sample)["size"].to_list() == [
        round(0.2274 * SCALE),
        0,
        round(4.7287 * SCALE),
    ]


def test_previous_size_is_kept_only_where_it_exists(sample):
    assert read_hour(sample)["prevSize"].to_list() == [None, None, round(5.0632 * SCALE)]


def test_side_becomes_the_same_flag_the_binary_stream_uses(sample):
    assert read_hour(sample)["isAsk"].to_list() == [True, False, False]


def test_prices_parse_exactly(sample):
    assert read_hour(sample)["px"].to_list() == [
        round(2964.0 * SCALE),
        round(2962.3 * SCALE),
        round(2962.5 * SCALE),
    ]


def test_file_position_is_preserved_because_nothing_else_orders_this_stream(sample):
    assert read_hour(sample)["seq"].to_list() == [0, 1, 2]


def test_filtering_by_coin_keeps_the_original_positions(sample):
    frame = read_hour(sample, coins=["btc"])
    assert frame["coin"].to_list() == ["BTC"]
    assert frame["seq"].to_list() == [2]


def test_identifiers_keep_full_width(sample):
    assert read_hour(sample)["oid"].to_list() == [273024455731, 273024450830, 273024455174]


def test_rejects_a_value_it_cannot_represent_exactly(tmp_path):
    path = tmp_path / "20251215" / "ex1.gz"
    path.parent.mkdir()
    broken = dict(RECORDS[0], px="2964.123456789")
    path.write_bytes(gzip.compress(json.dumps(broken).encode()))
    with pytest.raises(ValueError, match="could not be parsed"):
        read_hour(path)


def test_a_plain_string_path_works_too(sample):
    assert read_hour(str(sample)).height == len(RECORDS)
