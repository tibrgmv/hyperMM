"""One hour of one instrument, taken from raw files to the rebuilt book and the outcome table.

The steps are the same wherever an hour is needed -- the page export, the reports
that back the journal, and later the fill models -- so they live in one place.

An hour can be rebuilt on its own (`load_hour`) or inside a window of consecutive
hourly files (`load_window`). The book diff stream carries no snapshot, only
changes, so a file read on its own starts with an empty book: orders resting from
before it have no placement, and its first minutes are a warm-up (journal, 24
August). Chained after the previous hour, the book is already built when the hour
opens, and chained before the next one, an order placed late in the hour is
watched past the hour's end instead of being cut off at it.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from hypermm import config
from hypermm.ingest import book_diffs, orders, trades
from hypermm.reconstruct.book import with_top_of_book
from hypermm.reconstruct.outcomes import outcomes
from hypermm.reconstruct.queue import with_front, with_queue_position
from hypermm.reconstruct.timing import with_time

NS = 1_000_000_000
WARM_UP = 5 * 60 * NS  # orders placed this soon after a cold start are left out

STATUS_COLUMNS = (
    "ts", "oid", "statusId", "isAsk", "limitPx", "sz", "reduceOnly", "tifId",
)


@dataclass(frozen=True)
class Hour:
    coin: str
    statuses: pl.DataFrame  # order status stream, this coin
    fills: pl.DataFrame  # trade stream, this coin
    top: pl.DataFrame  # book diffs with queue position, time and top of book
    table: pl.DataFrame  # one row per passive order placed within the hour
    after_previous: bool = False  # the book came in built from the previous hour
    before_next: bool = False  # orders are watched on into the next hour


def read_changes(raw: str | Path, date: str, hour: int, coins=None) -> pl.DataFrame:
    """The book diff file is shared by all coins and is the slow one to parse: read it once."""
    return book_diffs.read_hour(book_diffs.hour_path(Path(raw) / "book_diffs", date, hour), coins)


def read_fills(raw: str | Path, date: str, hour: int, coins=None) -> pl.DataFrame:
    return trades.read_hour(trades.hour_path(Path(raw) / "trades", date, hour), coins)


def read_statuses(raw: str | Path, date: str, hour: int, coin: str) -> pl.DataFrame:
    path = orders.hour_path(Path(raw) / "order_statuses", date, hour, coin)
    return orders.read_hour(path, columns=list(STATUS_COLUMNS))


def rebuild(changes: pl.DataFrame, statuses: pl.DataFrame) -> pl.DataFrame:
    """Queue position, front of queue, time, top of book -- each needs the frame before."""
    return with_top_of_book(with_time(with_front(with_queue_position(changes)), statuses))


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


def hour_bounds(date: str, hour: int) -> tuple[int, int]:
    """Start and end of an hourly file's hour, in ns since the epoch (UTC)."""
    start = datetime.strptime(date, "%Y%m%d").replace(tzinfo=UTC) + timedelta(hours=hour)
    return int(start.timestamp()) * NS, int((start + timedelta(hours=1)).timestamp()) * NS


def chain(frames: Sequence[pl.DataFrame]) -> pl.DataFrame:
    """Consecutive hourly book diff files as one stream: `seq` runs on across files.

    Each frame must be a whole file (all coins), so that its height is its length.
    """
    parts, offset = [], 0
    for frame in frames:
        parts.append(frame.with_columns((pl.col("seq").cast(pl.Int64) + offset).alias("seq")))
        offset += frame.height
    return pl.concat(parts)


class Reader:
    """Parsed hourly files, kept for the few hours a sliding window revisits.

    Book diffs and trades are shared by all coins and parsed once per hour; a
    missing file reads as None, so a window simply stops where the data stops.
    """

    def __init__(self, raw: str | Path, keep: int = 3):  # a window is three hours
        self.raw, self.keep = Path(raw), keep
        self._cache: dict[tuple, pl.DataFrame | None] = {}
        self._hours: OrderedDict[datetime, None] = OrderedDict()  # least recently used first

    def _get(self, key: tuple, path: Path, read):
        """Keys are (kind, hour, ...): everything of the `keep` hours used last is kept."""
        at = key[1]
        if key not in self._cache:
            self._cache[key] = read() if path.exists() else None
        self._hours[at] = None
        self._hours.move_to_end(at)
        while len(self._hours) > self.keep:
            stale, _ = self._hours.popitem(last=False)
            for old in [k for k in self._cache if k[1] == stale]:
                del self._cache[old]
        return self._cache[key]

    def changes(self, at: datetime) -> pl.DataFrame | None:
        date, hour = at.strftime("%Y%m%d"), at.hour
        path = book_diffs.hour_path(self.raw / "book_diffs", date, hour)
        return self._get(("changes", at), path, lambda: read_changes(self.raw, date, hour))

    def fills(self, at: datetime) -> pl.DataFrame | None:
        date, hour = at.strftime("%Y%m%d"), at.hour
        path = trades.hour_path(self.raw / "trades", date, hour)
        # the trade file covers every listed instrument; one of them (a spot pair) has
        # been seen with a size that has more than eight decimals
        return self._get(
            ("fills", at), path, lambda: read_fills(self.raw, date, hour, config.INSTRUMENTS)
        )

    def statuses(self, at: datetime, coin: str) -> pl.DataFrame | None:
        date, hour = at.strftime("%Y%m%d"), at.hour
        path = orders.hour_path(self.raw / "order_statuses", date, hour, coin)
        return self._get(
            ("statuses", at, coin), path, lambda: read_statuses(self.raw, date, hour, coin)
        )

    def complete(self, at: datetime, coin: str) -> bool:
        return all(
            frame is not None
            for frame in (self.changes(at), self.fills(at), self.statuses(at, coin))
        )


def load_window(reader: Reader, date: str, hour: int, coin: str) -> Hour:
    """One hour rebuilt between its neighbours, where their files are at hand.

    The book, queue and outcomes are rebuilt over previous + this + next hour; the
    table keeps the orders placed in this hour's file. Without the previous hour the
    book starts cold and the first five minutes are left out, as `load_hour` users
    do by hand; without the next one, outcomes are cut off at the end of the hour.
    """
    coin = coin.upper()
    centre = datetime.strptime(date, "%Y%m%d") + timedelta(hours=hour)
    if not reader.complete(centre, coin):
        raise FileNotFoundError(f"{coin} {date} {hour:02d}: some hourly file is missing")
    previous, following = centre - timedelta(hours=1), centre + timedelta(hours=1)
    after_previous = reader.complete(previous, coin)
    before_next = reader.complete(following, coin)
    span = [
        at
        for at, use in ((previous, after_previous), (centre, True), (following, before_next))
        if use
    ]

    files = [reader.changes(at) for at in span]
    changes = chain(files).filter(pl.col("coin") == coin)
    fills = pl.concat([reader.fills(at) for at in span]).filter(pl.col("coin") == coin)
    statuses = pl.concat([reader.statuses(at, coin) for at in span])

    top = rebuild(changes, statuses)
    first = sum(f.height for f in files[: span.index(centre)])
    table = outcomes(top, fills).filter(
        pl.col("seqPlaced").is_between(first, first + files[span.index(centre)].height - 1)
    )
    if not after_previous:
        table = table.filter(pl.col("tPlaced") >= top["ts"].min() + WARM_UP)
    return Hour(coin, statuses, fills, top, table, after_previous, before_next)
