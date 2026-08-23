import numpy as np
import pytest

from hypermm.ingest.schema import (
    RECORD_DTYPE,
    RECORD_SIZE,
    SCALE,
    decode_fixed,
    to_float,
)

# Offsets as published in SCHEMA.md; guards against silent numpy padding.
EXPECTED_OFFSETS = {
    "ts": 0,
    "userId": 8,
    "isBuilder": 12,
    "statusId": 13,
    "isAsk": 14,
    "limitPx": 15,
    "sz": 19,
    "oid": 23,
    "timestampDiff": 31,
    "triggerCondition": 35,
    "triggered": 39,
    "isTrigger": 40,
    "hasChildren": 41,
    "isPositionTpsl": 42,
    "reduceOnly": 43,
    "orderTypeId": 44,
    "tifId": 45,
    "triggerPx": 46,
    "origSz": 50,
}


def test_record_is_54_bytes():
    assert RECORD_SIZE == 54


def test_field_offsets_match_published_schema():
    offsets = {name: RECORD_DTYPE.fields[name][1] for name in RECORD_DTYPE.names}
    assert offsets == EXPECTED_OFFSETS


def test_decodes_the_worked_example_from_schema():
    # SCHEMA.md: BTC at $96,543.21 encodes as (2 << 29) | 9654321 == 0x40935031
    encoded = np.array([0x40935031], dtype=np.uint32)
    assert decode_fixed(encoded)[0] == 9654321 * 10**6
    assert to_float(decode_fixed(encoded))[0] == pytest.approx(96543.21)


@pytest.mark.parametrize("decimals", range(8))
def test_roundtrip_across_every_decimal_exponent(decimals):
    value = 123456
    encoded = np.array([(decimals << 29) | value], dtype=np.uint32)
    assert to_float(decode_fixed(encoded))[0] == pytest.approx(value / 10**decimals)


def test_equal_prices_written_at_different_exponents_decode_identically():
    # 2964.3 as (decimals=1, 29643) and as (decimals=3, 2964300)
    encoded = np.array([(1 << 29) | 29643, (3 << 29) | 2964300], dtype=np.uint32)
    decoded = decode_fixed(encoded)
    assert decoded[0] == decoded[1]


def test_accumulating_sizes_stays_exact():
    # 0.1 repeated 1e6 times: exact in int64, drifts in float
    encoded = np.full(1_000_000, (1 << 29) | 1, dtype=np.uint32)
    assert decode_fixed(encoded).sum() == 1_000_000 * SCALE // 10


def test_max_representable_value_does_not_overflow():
    encoded = np.array([(7 << 29) | 0x1FFFFFFF], dtype=np.uint32)
    assert decode_fixed(encoded)[0] == 0x1FFFFFFF * 10
