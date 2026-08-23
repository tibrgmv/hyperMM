"""Reading the hourly files of the trade stream."""

from __future__ import annotations

import gzip
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from hypermm.ingest.decimals import parse_decimal

COUNTERPARTIES = 2

# A trade whose order arrived in the block that matched it carries no L1 hash.
SAME_BLOCK_HASH = "0x" + "0" * 64


def hour_path(root: Path, date: str, hour: int) -> Path:
    """Build the path to one hourly file: <root>/<YYYYMMDD>/<H>.gz (hour not padded)."""
    return root / date.replace("-", "") / f"{hour}.gz"


def read_hour(path: Path, coins: Sequence[str] | None = None) -> pl.DataFrame:
    """Read one hourly file, keeping both counterparties of every trade.

    `ts` is nanoseconds since the epoch, the same clock and unit the order status
    stream uses, so trades group into blocks alongside order events.

    Which of the two counterparties was the maker is not marked in this stream and
    is not guessed here: it follows from whether the order was already resting,
    which only the book diff stream can say.
    """
    frame = pl.read_ndjson(gzip.decompress(path.read_bytes()))
    if coins is not None:
        frame = frame.filter(pl.col("coin").is_in([coin.upper() for coin in coins]))

    malformed = frame.select(
        (pl.col("side_info").list.len() != COUNTERPARTIES).sum().alias("n")
    ).item()
    if malformed:
        raise ValueError(f"{path}: {malformed} trades do not have exactly two counterparties")

    parsed = frame.select(
        pl.col("time").str.to_datetime(time_unit="ns").cast(pl.Int64).alias("ts"),
        "coin",
        (pl.col("side") == "A").alias("isAggressorAsk"),
        parse_decimal("px").alias("px"),
        parse_decimal("sz").alias("sz"),
        (pl.col("hash") == SAME_BLOCK_HASH).alias("sameBlock"),
        *_counterparty(0),
        *_counterparty(1),
    )

    unparsed = parsed.select(
        (pl.col("px").is_null().sum() + pl.col("sz").is_null().sum()).alias("n")
    ).item()
    if unparsed:
        raise ValueError(f"{path}: {unparsed} price/size values could not be parsed exactly")
    return parsed


def _counterparty(index: int) -> list[pl.Expr]:
    entry = pl.col("side_info").list.get(index)
    return [
        entry.struct.field("user").alias(f"user{index}"),
        entry.struct.field("oid").alias(f"oid{index}"),
        parse_decimal(entry.struct.field("start_pos")).alias(f"startPos{index}"),
        entry.struct.field("twap_id").alias(f"twapId{index}"),
    ]
