"""One hour of one instrument, taken from raw files to the rebuilt book and the outcome table.

The steps are the same wherever an hour is needed -- the page export, the reports
that back the journal, and later the fill models -- so they live in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from hypermm.ingest import book_diffs, orders, trades
from hypermm.reconstruct.book import with_top_of_book
from hypermm.reconstruct.outcomes import outcomes
from hypermm.reconstruct.queue import with_queue_position
from hypermm.reconstruct.timing import with_time

STATUS_COLUMNS = (
    "ts", "oid", "statusId", "isAsk", "limitPx", "sz", "reduceOnly", "tifId", "userId",
)


@dataclass(frozen=True)
class Hour:
    coin: str
    statuses: pl.DataFrame  # order status stream, this coin
    fills: pl.DataFrame  # trade stream, this coin
    top: pl.DataFrame  # book diffs with queue position, time and top of book
    table: pl.DataFrame  # one row per passive order placed within the hour


def read_changes(raw: str | Path, date: str, hour: int, coins=None) -> pl.DataFrame:
    """The book diff file is shared by all coins and is the slow one to parse: read it once."""
    return book_diffs.read_hour(book_diffs.hour_path(Path(raw) / "book_diffs", date, hour), coins)


def read_fills(raw: str | Path, date: str, hour: int, coins=None) -> pl.DataFrame:
    return trades.read_hour(trades.hour_path(Path(raw) / "trades", date, hour), coins)


def read_statuses(raw: str | Path, date: str, hour: int, coin: str) -> pl.DataFrame:
    path = orders.hour_path(Path(raw) / "order_statuses", date, hour, coin)
    return orders.read_hour(path, columns=list(STATUS_COLUMNS))


def rebuild(changes: pl.DataFrame, statuses: pl.DataFrame) -> pl.DataFrame:
    """Queue position, then time, then top of book -- each step needs the one before."""
    return with_top_of_book(with_time(with_queue_position(changes), statuses))


def load_hour(
    raw: str | Path,
    date: str,
    hour: int,
    coin: str,
    *,
    changes: pl.DataFrame | None = None,
    fills: pl.DataFrame | None = None,
) -> Hour:
    """Everything known about one coin in one hour.

    `changes` and `fills` may be passed in already read (for all coins) to avoid
    parsing the shared files once per coin.
    """
    coin = coin.upper()
    changes = read_changes(raw, date, hour) if changes is None else changes
    fills = read_fills(raw, date, hour) if fills is None else fills
    changes = changes.filter(pl.col("coin") == coin)
    fills = fills.filter(pl.col("coin") == coin)

    statuses = read_statuses(raw, date, hour, coin)
    top = rebuild(changes, statuses)
    return Hour(coin, statuses, fills, top, outcomes(top, fills))
