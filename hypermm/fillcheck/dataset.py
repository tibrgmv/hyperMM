"""One hour of real orders as a modelling table, cached on disk.

Rebuilding an hour between its neighbours takes tens of seconds; every model and
metric of the fill-time analysis reads the same rows, so they are built once and
kept as Parquet under data/processed/orders/. One row per passive order placed in
the hour:

    features at placement and survival targets    fillcheck/survival.survival_table
    what each fill rule says, within the order's observed life, with and without
    the exchange's in-block order                 fillsim/models, fillcheck/judge
    the exponential clock fitted on this hour     clockA, clockK (constants)

A change to what the rows hold needs a new VERSION, which starts a fresh cache.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from hypermm import config
from hypermm.fillcheck.judge import RULES, judge
from hypermm.fillcheck.survival import survival_table
from hypermm.fillsim import models
from hypermm.pipeline import Reader, hour_bounds, load_window

VERSION = 2  # 2: features on a logarithmic scale
WITHIN = {"touch": "touchWithin", "volume-through": "volumeWithin", "queue-aware": "queueWithin"}


def build_hour(reader: Reader, date: str, hour: int, coin: str) -> pl.DataFrame:
    window = load_window(reader, date, hour, coin)
    trades = models.level_trades(window.fills)
    table = window.table.with_columns(
        tTouch=models.touch(window.table, trades),
        tVolume=models.volume_through(window.table, trades),
        tQueue=models.queue_aware(window.table, trades),
    )
    plain, ordered = judge(table), judge(table, cancel_first=True)

    # the clock as a practitioner would calibrate it: on this hour's own trading
    start, end = hour_bounds(date, hour)
    in_hour = pl.col("ts").cast(pl.Int64).is_between(start, end, closed="left")
    a, k = models.clock_fit(window.top.filter(in_hour), trades.filter(in_hour))

    return survival_table(window.table, window.top, window.fills).with_columns(
        hour=pl.lit(hour, dtype=pl.Int8),
        **{WITHIN[rule]: plain[f"{column}Within"] for rule, column in RULES.items()},
        **{f"{WITHIN[rule]}Block": ordered[f"{column}Within"] for rule, column in RULES.items()},
        clockA=pl.lit(a),
        clockK=pl.lit(k),
    )


def hour_path(date: str, hour: int, coin: str, root: Path | None = None) -> Path:
    root = config.DATA_PROCESSED_DIR / "orders" if root is None else root
    return root / f"v{VERSION}" / date / f"{coin.lower()}_{hour:02d}.parquet"


def load_orders(
    reader: Reader, date: str, hour: int, coin: str, root: Path | None = None
) -> pl.DataFrame:
    """The hour's table, from the cache if it is there, built and cached if not."""
    path = hour_path(date, hour, coin, root)
    if path.exists():
        return pl.read_parquet(path)
    frame = build_hour(reader, date, hour, coin)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    frame.write_parquet(partial)
    partial.rename(path)
    return frame


def load_sample(
    reader: Reader,
    date: str,
    hours: list[int],
    coin: str,
    sample: int,
    prepare=None,
    root: Path | None = None,
) -> pl.DataFrame:
    """Orders of several hours, drawn down to about `sample` rows (0 keeps all).

    The draw is made hour by hour at one common rate, so the pooled table -- several
    gigabytes for a day of BTC -- is never held in memory. `prepare(frame, hour)`
    may add columns to an hour's table before it is drawn from.
    """
    for hour in hours:
        if not hour_path(date, hour, coin, root).exists():
            load_orders(reader, date, hour, coin, root)
    total = sum(
        pl.scan_parquet(hour_path(date, hour, coin, root)).select(pl.len()).collect().item()
        for hour in hours
    )
    rate = min(1.0, sample / total) if sample else 1.0
    parts = []
    for hour in hours:
        frame = pl.read_parquet(hour_path(date, hour, coin, root))
        if prepare is not None:
            frame = prepare(frame, hour)
        parts.append(frame if rate >= 1 else frame.sample(fraction=rate, seed=hour))
    return pl.concat(parts)
