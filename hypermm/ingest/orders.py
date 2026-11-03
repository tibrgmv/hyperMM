"""Reading the hourly files of the order status stream."""

from __future__ import annotations

import gzip
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from hypermm.ingest.schema import RECORD_DTYPE, RECORD_SIZE, decode_fixed

ENCODED_FIELDS = frozenset({"limitPx", "sz", "origSz", "triggerPx"})


def hour_path(root: str | Path, date: str, hour: int, coin: str, *, rejected: bool = False) -> Path:
    """Build the path to one hourly file: <root>/<YYYYMMDD>/<coin>_<HH>[_rejected].data.gz."""
    root = Path(root)
    return (
        root
        / date.replace("-", "")
        / f"{coin.lower()}_{hour:02d}{'_rejected' if rejected else ''}.data.gz"
    )


def read_hour(path: str | Path, columns: Sequence[str] | None = None) -> pl.DataFrame:
    """Read one hourly file into a frame, with prices and sizes already decoded.

    `ts` stays a raw uint64 nanosecond count: block reconstruction turns on exact
    equality between timestamps, so no calendar type is imposed here.
    """
    names = tuple(columns) if columns is not None else RECORD_DTYPE.names
    unknown = set(names) - set(RECORD_DTYPE.names)
    if unknown:
        raise ValueError(f"no such field(s) in the record: {sorted(unknown)}")

    path = Path(path)
    raw = gzip.decompress(path.read_bytes())
    if len(raw) % RECORD_SIZE:
        raise ValueError(
            f"{path}: {len(raw)} bytes is not a whole number of {RECORD_SIZE}-byte records"
        )

    records = np.frombuffer(raw, dtype=RECORD_DTYPE)
    return pl.DataFrame(
        {
            name: decode_fixed(records[name]) if name in ENCODED_FIELDS else records[name]
            for name in names
        }
    )
