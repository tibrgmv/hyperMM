"""A fill rule learned from what an ordinary data feed shows, scored against L4 truth.

The queue-aware rule is right but needs order-level (L4) data, which almost no one
has; touch and volume-through need only trades and are wrong by multiples. How
much of that gap can a model close using only what L1 and L2 data show?

For every real order and a fixed horizon h, the features describe the market over
[placement, placement + h] as a feed without order identities would show it:

    L1   trades at the order's price (volume, number of blocks, time of the first),
         where the mid stands at the horizon relative to the price
    L2   also the volume resting at the level at placement and at the horizon

The window is fixed, not the order's own life, so nothing in it can tell when the
order itself ended. The target is whether the order was filled within h, known for
orders filled by then or still resting then; orders cancelled before the horizon
are left out, for every rule alike.
"""

from __future__ import annotations

import polars as pl

from hypermm.fillsim import models
from hypermm.fillsim.models import KEY, NS
from hypermm.pipeline import Hour

HORIZONS = (1, 5, 30)  # seconds
SCALE = 1e8
L1 = ("tradedRatio", "logTraded", "logBlocks", "sinceTouch", "ticksThrough")
L2 = ("logLevelThen", "levelChange")


def _asof(left: pl.DataFrame, right: pl.DataFrame, value: str, *, exact: bool) -> pl.Series:
    """`value` of the last `right` row of the order's level at or before `t`."""
    found = (
        left.with_row_index()
        .sort("t")
        .join_asof(
            right.sort("t"),
            on="t",
            by=KEY,
            strategy="backward",
            allow_exact_matches=exact,
            check_sortedness=False,
        )  # fmt: skip
        .sort("index")
    )
    return found[value]


def l2_table(window: Hour) -> pl.DataFrame:
    """One row per order: per horizon, the L1/L2 view of its level, each rule's
    answer, and the truth."""
    table, top = window.table, window.top
    step = models.tick(top)
    trades = models.level_trades(window.fills).with_columns(
        blocks=pl.int_range(1, pl.len() + 1).over(KEY)
    )
    traded = trades.select(*KEY, "traded", "blocks", t="ts")
    level = top.select(
        *KEY, t=pl.col("ts").cast(pl.Int64), level=pl.col("levelVolume") + pl.col("delta")
    )
    mids = top.select(
        t=pl.col("ts").cast(pl.Int64), mid=(pl.col("bestBid") + pl.col("bestAsk")) / 2
    ).drop_nulls()

    placed = table["tPlaced"].cast(pl.Int64)
    at = table.select(*KEY, t=placed)
    before = {v: _asof(at, traded, v, exact=False).fill_null(0) for v in ("traded", "blocks")}
    ruled = table.with_columns(
        tTouch=models.touch(table, trades),
        tVolume=models.volume_through(table, trades),
        tQueue=models.queue_aware(table, trades),
    )
    end = placed + table["duration"]
    saved = ~table["filled"] & (ruled["tQueue"] == table["tRemoved"].cast(pl.Int64)).fill_null(
        False
    )
    ahead = table["volumeAhead"]

    out = table.select("oid")
    for h in HORIZONS:
        then = placed + h * NS
        at_h = table.select(*KEY, t=then)
        volume = _asof(at_h, traded, "traded", exact=True).fill_null(0) - before["traded"]
        blocks = _asof(at_h, traded, "blocks", exact=True).fill_null(0) - before["blocks"]
        level_then = _asof(at_h, level, "level", exact=True).fill_null(0)
        mid = (
            at_h.select("t").with_row_index().sort("t")
            .join_asof(mids, on="t", strategy="backward").sort("index")["mid"]
        )  # fmt: skip
        # how far the mid has gone through the price, in ticks: positive is past it
        through = pl.Series(
            ((table["px"] - mid) / step).to_numpy() * (1 - 2 * table["isAsk"].to_numpy())
        )
        since = ((ruled["tTouch"] - placed) / NS).clip(upper_bound=h).fill_null(h)
        fires = {
            r: (ruled[c] <= then).fill_null(False)
            for r, c in (("touch", "tTouch"), ("volume", "tVolume"), ("queue", "tQueue"))
        }
        out = out.with_columns(
            **{
                f"known{h}": (table["filled"] & (table["tFirstFill"] <= then)) | (end > then),
                f"filled{h}": table["filled"] & (table["tFirstFill"] <= then),
                f"touch{h}": fires["touch"],
                f"volume{h}": fires["volume"],
                f"queue{h}": fires["queue"],
                f"queueBlock{h}": fires["queue"] & ~saved,
                f"tradedRatio{h}": volume / (ahead + table["restingSize"]),
                f"logTraded{h}": (volume / SCALE).log1p(),
                f"logBlocks{h}": blocks.cast(pl.Float64).log1p(),
                f"sinceTouch{h}": since,
                f"ticksThrough{h}": through.sign() * through.abs().log1p(),
                f"logLevelThen{h}": (level_then / SCALE).log1p(),
                f"levelChange{h}": (level_then - ahead - table["restingSize"])
                / (ahead + table["restingSize"]),
            }
        )
    return out
