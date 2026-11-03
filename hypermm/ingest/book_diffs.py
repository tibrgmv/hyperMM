"""Reading the hourly files of the raw book diff stream."""

from __future__ import annotations

import gzip
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from hypermm.ingest.decimals import parse_decimal

NEW, UPDATE, REMOVE = "new", "update", "remove"

_SIZE = r'"sz":\s*"([^"]*)"'
_NEW_SIZE = r'"newSz":\s*"([^"]*)"'
_PREV_SIZE = r'"origSz":\s*"([^"]*)"'


def hour_path(root: str | Path, date: str, hour: int) -> Path:
    """Build the path to one hourly file: <root>/<YYYYMMDD>/ex<H>.gz (hour not padded)."""
    return Path(root) / date.replace("-", "") / f"ex{hour}.gz"


def read_hour(path: str | Path, coins: Sequence[str] | None = None) -> pl.DataFrame:
    """Read one hourly file into a frame carrying the book state after each change.

    This stream has no timestamp of any kind, so position in the file is the only
    ordering that exists; it is kept as `seq` rather than left implicit. Time comes
    later, by joining `oid` against the order status stream.

    `size` is the order's size resting on the book once the change is applied --
    the new size, the reduced size, or zero once the order is gone -- so that
    reconstruction reads one column instead of branching on the kind of change.
    """
    path = Path(path)
    frame = pl.read_ndjson(
        gzip.decompress(path.read_bytes()),
        schema_overrides={"raw_book_diff": pl.String},
    ).with_row_index("seq")
    if coins is not None:
        frame = frame.filter(pl.col("coin").is_in([coin.upper() for coin in coins]))

    diff = pl.col("raw_book_diff")
    kind = (
        pl.when(diff == REMOVE)
        .then(pl.lit(REMOVE))
        .when(diff.str.starts_with('{"' + NEW))
        .then(pl.lit(NEW))
        .otherwise(pl.lit(UPDATE))
    )
    resting = (
        pl.when(diff == REMOVE)
        .then(pl.lit(0, dtype=pl.Int64))
        .when(diff.str.starts_with('{"' + NEW))
        .then(parse_decimal(diff.str.extract(_SIZE, 1)))
        .otherwise(parse_decimal(diff.str.extract(_NEW_SIZE, 1)))
    )

    parsed = frame.select(
        "seq",
        "user",
        "oid",
        "coin",
        (pl.col("side") == "A").alias("isAsk"),
        parse_decimal("px").alias("px"),
        kind.alias("kind"),
        resting.alias("size"),
        parse_decimal(diff.str.extract(_PREV_SIZE, 1)).alias("prevSize"),
    )

    unparsed = parsed.select(
        (pl.col("px").is_null().sum() + pl.col("size").is_null().sum()).alias("n")
    ).item()
    if unparsed:
        raise ValueError(f"{path}: {unparsed} price/size values could not be parsed exactly")
    return parsed
