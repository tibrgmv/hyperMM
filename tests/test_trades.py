import gzip
import json

import pytest

from hypermm.ingest.schema import SCALE
from hypermm.ingest.trades import SAME_BLOCK_HASH, hour_path, read_hour

# Shaped exactly as published in SCHEMA.md.
TRADE = {
    "coin": "BTC",
    "side": "A",
    "time": "2025-12-01T12:34:56.789012345",
    "px": "96543.0",
    "sz": "0.125",
    "hash": "0x1234abcd",
    "trade_dir_override": "Na",
    "side_info": [
        {"user": "0xabc", "start_pos": "1.5", "oid": 110237687900, "twap_id": None, "cloid": None},
        {"user": "0xdef", "start_pos": "-2.0", "oid": 110237699140, "twap_id": 857226,
         "cloid": None},
    ],
}
OTHER = dict(TRADE, coin="HYPE", side="B", hash=SAME_BLOCK_HASH)


def write(path, trades):
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(gzip.compress("\n".join(json.dumps(t) for t in trades).encode()))
    return path


@pytest.fixture
def sample(tmp_path):
    return write(tmp_path / "20251201" / "12.gz", [TRADE, OTHER])


def test_hour_path_leaves_the_hour_unpadded(tmp_path):
    assert hour_path(tmp_path, "2025-12-01", 7).name == "7.gz"
    assert hour_path(tmp_path, "20251201", 14).parent.name == "20251201"


def test_time_lands_on_the_same_clock_as_the_order_stream(sample):
    # nanoseconds since the epoch, not a calendar type, and not rounded
    assert read_hour(sample)["ts"].to_list() == [1_764_592_496_789_012_345] * 2


def test_price_and_size_parse_exactly(sample):
    frame = read_hour(sample)
    assert frame["px"].to_list() == [round(96543.0 * SCALE)] * 2
    assert frame["sz"].to_list() == [round(0.125 * SCALE)] * 2


def test_aggressor_side_becomes_a_flag(sample):
    assert read_hour(sample)["isAggressorAsk"].to_list() == [True, False]


def test_both_counterparties_are_kept(sample):
    frame = read_hour(sample)
    assert frame["user0"].to_list() == ["0xabc", "0xabc"]
    assert frame["user1"].to_list() == ["0xdef", "0xdef"]
    assert frame["oid0"].to_list() == [110237687900] * 2
    assert frame["oid1"].to_list() == [110237699140] * 2


def test_signed_starting_position_survives(sample):
    frame = read_hour(sample)
    assert frame["startPos0"].to_list() == [round(1.5 * SCALE)] * 2
    assert frame["startPos1"].to_list() == [-2 * SCALE] * 2


def test_twap_membership_is_kept_and_may_be_absent(sample):
    frame = read_hour(sample)
    assert frame["twapId0"].to_list() == [None, None]
    assert frame["twapId1"].to_list() == [857226, 857226]


def test_the_empty_hash_is_read_as_a_same_block_trade(sample):
    assert read_hour(sample)["sameBlock"].to_list() == [False, True]


def test_filtering_selects_one_of_the_many_listed_coins(sample):
    assert read_hour(sample, coins=["hype"])["coin"].to_list() == ["HYPE"]


def test_rejects_a_trade_without_two_counterparties(tmp_path):
    broken = dict(TRADE, side_info=TRADE["side_info"][:1])
    path = write(tmp_path / "20251201" / "1.gz", [broken])
    with pytest.raises(ValueError, match="two counterparties"):
        read_hour(path)
