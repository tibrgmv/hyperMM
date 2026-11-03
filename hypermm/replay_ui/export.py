"""Reducing one hour of rebuilt book to a small file the page can load at once."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from hypermm.ingest import book_diffs, orders, trades
from hypermm.ingest.schema import SCALE
from hypermm.reconstruct.book import snapshots, with_top_of_book
from hypermm.reconstruct.outcomes import outcomes
from hypermm.reconstruct.queue import with_queue_position
from hypermm.reconstruct.timing import with_time

NS_PER_S = 1_000_000_000
FRAMES = 720
DEPTH = 12
DECILES = 10


def build(raw: str | Path, coin: str, date: str, hour: int) -> dict:
    """Turn one hour into the frames, series and summaries the page draws."""
    raw = Path(raw)
    changes = book_diffs.read_hour(book_diffs.hour_path(raw / "book_diffs", date, hour), [coin])
    order_events = orders.read_hour(
        orders.hour_path(raw / "order_statuses", date, hour, coin),
        columns=["ts", "oid", "statusId"],
    )
    fills = trades.read_hour(trades.hour_path(raw / "trades", date, hour), [coin])

    top = with_top_of_book(with_time(with_queue_position(changes), order_events))
    table = outcomes(top, fills)

    at = np.linspace(0, top.height - 1, FRAMES, dtype=np.int64)
    frames = top[at].with_columns(
        second=(pl.col("ts") - top["ts"].min()) / NS_PER_S,
        mid=(pl.col("bestBid") + pl.col("bestAsk")) / 2,
        spread=pl.col("bestAsk") - pl.col("bestBid"),
    )

    return {
        "meta": {"coin": coin, "date": date, "hour": hour, "scale": SCALE},
        "stats": _stats(top, table),
        "series": {
            "second": [round(s, 2) for s in frames["second"].to_list()],
            "mid": [_price(v) for v in frames["mid"].to_list()],
            "spread": [_price(v) for v in frames["spread"].to_list()],
        },
        "frames": snapshots(top, at, DEPTH),
        "queue": _by_queue_position(table),
        "outcomes": _by_outcome(table),
    }


def _price(value: float | None) -> float | None:
    return None if value is None else round(value / SCALE, 8)


def _stats(top: pl.DataFrame, table: pl.DataFrame) -> dict:
    both = top.drop_nulls(["bestBid", "bestAsk"])
    spread = (both["bestAsk"] - both["bestBid"]).median()
    filled = table.filter("filled")["duration"].median()
    censored = table.filter(~pl.col("filled"))["duration"].median()
    return {
        "changes": top.height,
        "orders": table.height,
        "fillRate": round(100 * table["filled"].mean(), 3),
        "medianSpread": _price(spread),
        "secondsToFill": round(filled / NS_PER_S, 2) if filled else None,
        "secondsToCancel": round(censored / NS_PER_S, 2) if censored else None,
        "medianAhead": _price(table["volumeAhead"].median()),
    }


def _by_queue_position(table: pl.DataFrame) -> list[dict]:
    labels = [str(i) for i in range(DECILES)]
    grouped = (
        table.with_columns(bucket=pl.col("volumeAhead").qcut(DECILES, labels=labels))
        .group_by("bucket")
        .agg(
            orders=pl.len(),
            ahead=pl.col("volumeAhead").median(),
            fillRate=pl.col("filled").mean() * 100,
        )
        .sort("bucket")
    )
    return [
        {"bucket": int(b), "orders": n, "ahead": _price(a), "fillRate": round(r, 3)}
        for b, n, a, r in grouped.iter_rows()
    ]


def _by_outcome(table: pl.DataFrame) -> list[dict]:
    grouped = table.group_by("outcome").len().sort("len", descending=True)
    return [{"outcome": o, "orders": n} for o, n in grouped.iter_rows()]


def write(destination: str | Path, **kwargs) -> Path:
    destination = Path(destination)
    destination.write_text(json.dumps(build(**kwargs), separators=(",", ":")))
    return destination


if __name__ == "__main__":
    from hypermm import config

    out = write(
        Path(__file__).resolve().parents[2] / "web" / "data.json",
        raw=config.DATA_RAW_DIR,
        coin="SOL",
        date="20251201",
        hour=12,
    )
    print(f"{out}  {out.stat().st_size / 1e6:.2f} МБ")
