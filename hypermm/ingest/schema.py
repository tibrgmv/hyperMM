"""Binary layout and fixed-point decoding for the order status stream."""

from __future__ import annotations

import numpy as np

RECORD_DTYPE = np.dtype(
    [
        ("ts", "<u8"),
        ("userId", "<u4"),
        ("isBuilder", "?"),
        ("statusId", "<u1"),
        ("isAsk", "?"),
        ("limitPx", "<u4"),
        ("sz", "<u4"),
        ("oid", "<u8"),
        ("timestampDiff", "<u4"),
        ("triggerCondition", "<i4"),
        ("triggered", "?"),
        ("isTrigger", "?"),
        ("hasChildren", "?"),
        ("isPositionTpsl", "?"),
        ("reduceOnly", "?"),
        ("orderTypeId", "<u1"),
        ("tifId", "<u1"),
        ("triggerPx", "<u4"),
        ("origSz", "<u4"),
    ]
)

RECORD_SIZE = RECORD_DTYPE.itemsize


# scale factor for converting between int64 units of 1/SCALE and float64 units of 1.0
SCALE = 10**8

# for decoding where the decimal part is
_DECIMALS_SHIFT = 29
_VALUE_MASK = 0x1FFFFFFF
_RESCALE = np.array([10**8, 10**7, 10**6, 10**5, 10**4, 10**3, 10**2, 10], dtype=np.int64)


def decode_fixed(encoded: np.ndarray) -> np.ndarray:
    encoded = np.asarray(encoded, dtype=np.uint32)
    return (encoded & _VALUE_MASK).astype(np.int64) * _RESCALE[encoded >> _DECIMALS_SHIFT]


def to_float(scaled: np.ndarray) -> np.ndarray:
    return np.asarray(scaled, dtype=np.float64) / SCALE
