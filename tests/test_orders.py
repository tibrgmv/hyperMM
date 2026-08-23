import gzip

import numpy as np
import polars as pl
import pytest

from hypermm.ingest.codes import Status, Tif
from hypermm.ingest.orders import hour_path, read_hour
from hypermm.ingest.schema import RECORD_DTYPE, SCALE


def encode(value: float, decimals: int) -> int:
    return (decimals << 29) | round(value * 10**decimals)


@pytest.fixture
def sample(tmp_path):
    """Two records: an ALO order resting in the book, then the same order filled."""
    records = np.zeros(2, dtype=RECORD_DTYPE)
    records["ts"] = [1_765_000_000_000_000_000, 1_765_000_000_250_000_000]
    records["oid"] = 273_024_455_731
    records["userId"] = [7, 7]
    records["statusId"] = [Status.OPEN, Status.FILLED]
    records["tifId"] = Tif.ALO
    records["isAsk"] = [False, False]
    records["limitPx"] = encode(2964.3, 1)
    records["origSz"] = encode(0.2274, 4)
    records["sz"] = [encode(0.2274, 4), encode(0.0, 0)]
    records["timestampDiff"] = [0, 250]

    path = tmp_path / "20251215" / "btc_12.data.gz"
    path.parent.mkdir()
    path.write_bytes(gzip.compress(records.tobytes()))
    return path


def test_hour_path_matches_the_published_naming(tmp_path):
    assert hour_path(tmp_path, "2025-12-15", 12, "BTC").name == "btc_12.data.gz"
    assert hour_path(tmp_path, "20251215", 7, "eth").name == "eth_07.data.gz"
    assert hour_path(tmp_path, "2025-12-15", 0, "sol", rejected=True).name == (
        "sol_00_rejected.data.gz"
    )
    assert hour_path(tmp_path, "2025-12-15", 12, "btc").parent.name == "20251215"


def test_reads_every_field_of_every_record(sample):
    frame = read_hour(sample)
    assert frame.height == 2
    assert set(frame.columns) == set(RECORD_DTYPE.names)


def test_decodes_prices_and_sizes_but_leaves_plain_fields_alone(sample):
    frame = read_hour(sample)
    assert frame["limitPx"].to_list() == [round(2964.3 * SCALE)] * 2
    assert frame["origSz"].to_list() == [round(0.2274 * SCALE)] * 2
    assert frame["sz"].to_list() == [round(0.2274 * SCALE), 0]
    assert frame["oid"].to_list() == [273_024_455_731] * 2
    assert frame["timestampDiff"].to_list() == [0, 250]


def test_keeps_timestamps_as_exact_nanoseconds(sample):
    frame = read_hour(sample)
    assert frame["ts"].dtype == pl.UInt64
    assert frame["ts"].to_list() == [1_765_000_000_000_000_000, 1_765_000_000_250_000_000]


def test_codes_survive_the_trip(sample):
    frame = read_hour(sample)
    assert frame["statusId"].to_list() == [Status.OPEN, Status.FILLED]
    assert frame["tifId"].to_list() == [Tif.ALO, Tif.ALO]
    assert frame["isAsk"].to_list() == [False, False]


def test_selecting_columns_returns_only_those(sample):
    frame = read_hour(sample, columns=["ts", "limitPx", "statusId"])
    assert frame.columns == ["ts", "limitPx", "statusId"]


def test_rejects_an_unknown_column_name(sample):
    with pytest.raises(ValueError, match="limit_px"):
        read_hour(sample, columns=["ts", "limit_px"])


def test_rejects_a_truncated_file(sample, tmp_path):
    broken = tmp_path / "broken.data.gz"
    broken.write_bytes(gzip.compress(gzip.decompress(sample.read_bytes())[:-3]))
    with pytest.raises(ValueError, match="whole number"):
        read_hour(broken)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_hour(tmp_path / "20251215" / "btc_13.data.gz")
