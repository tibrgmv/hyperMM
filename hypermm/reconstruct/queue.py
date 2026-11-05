"""Rebuilding resting volume per price level, queue position at placement, and the
moment each order reaches the front of its queue."""

from __future__ import annotations

import numpy as np
import polars as pl
from numba import njit

from hypermm.ingest.book_diffs import NEW, REMOVE, UPDATE

LEVEL = ("coin", "isAsk", "px")


# Add the volume of all orders resting at the same price level, and the queue position of each order when it was placed.
def with_queue_position(diffs: pl.DataFrame) -> pl.DataFrame:    

    already_resting = pl.col("firstSeen") & (pl.col("kind") != NEW)

    return (
        diffs.sort("seq")
        # the shift is the costly step here, so it is materialised once and reused
        .with_columns(carried=pl.col("size").shift(1).over("oid"))
        .with_columns(
            firstSeen=pl.col("carried").is_null(),
            sizeBefore=pl.coalesce("carried", "prevSize", pl.lit(0, dtype=pl.Int64)),
        )
        .with_columns(
            delta=pl.col("size") - pl.col("sizeBefore"),
            opening=pl.when(already_resting).then("sizeBefore").otherwise(0),
        )
        .with_columns(
            levelVolume=(pl.col("opening") + pl.col("delta")).cum_sum().over(LEVEL)
            - pl.col("delta")
        )
        .with_columns(volumeAhead=pl.when(pl.col("kind") == NEW).then(pl.col("levelVolume")))
        .drop("carried")
    )


def check(rebuilt: pl.DataFrame) -> dict[str, int]:
    """Count the ways a rebuilt book can be wrong."""
    settled = ~pl.col("firstSeen")
    started_cold = pl.col("firstSeen") & (pl.col("kind") != NEW)

    def count(condition: pl.Expr) -> int:
        return rebuilt.filter(condition).height

    return {
        "coldStartRecovered": count(started_cold & (pl.col("kind") == UPDATE)),
        "coldStartLost": count(started_cold & (pl.col("kind") == REMOVE)),
        "negativeLevel": count(pl.col("levelVolume") + pl.col("delta") < 0),
        "reArrival": count(settled & (pl.col("kind") == NEW)),
        "sizeBeforeDisagrees": count(
            settled & (pl.col("kind") == UPDATE) & (pl.col("sizeBefore") != pl.col("prevSize"))
        ),
        "growingUpdate": count(settled & (pl.col("kind") == UPDATE) & (pl.col("delta") >= 0)),
        "nonEmptyRemoval": count((pl.col("kind") == REMOVE) & (pl.col("size") != 0)),
    }


NOT_YET, RESTING, GONE = 0, 1, 2  # states of a queue slot: not arrived, resting, gone


@njit(cache=True)
def _front(level, slot, size, first):  # pragma: no cover - compiled
    """Replay arrivals and departures, noting the change at which each slot leads its level.

    Slots of a level are numbered in arrival order, so the front only ever moves back:
    past orders that have gone, never past one still to come. That keeps the walk
    linear in the number of changes plus orders.
    """
    head = first[:-1].copy()
    state = np.full(first[-1], NOT_YET, dtype=np.int8)
    front = np.full(first[-1], -1, dtype=np.int64)

    for i in range(slot.shape[0]):
        lv = level[i]
        state[slot[i]] = RESTING if size[i] > 0 else GONE
        h = head[lv]
        while h < first[lv + 1] and state[h] == GONE:
            h += 1
        head[lv] = h
        if h < first[lv + 1] and state[h] == RESTING and front[h] < 0:
            front[h] = i
    return front


def with_front(rebuilt: pl.DataFrame) -> pl.DataFrame:
    """Add `frontSeq`: the change after which nothing rested ahead of the order.

    Takes the frame `with_queue_position` produced. Priority is the order of arrival
    at the level, an arrival being the order's first change in the file -- the same
    convention `volumeAhead` follows, so an order already resting when the file
    opened queues from where it is first seen. Size reductions keep priority.

    Null for an order that never reached the front.
    """
    ordered = rebuilt.sort("seq")
    slots = (
        ordered.group_by("oid")
        .agg(pl.col("seq").min().alias("arrival"), *(pl.col(c).first() for c in LEVEL))
        .sort(*LEVEL, "arrival")
        .with_row_index("slot")
        .with_columns(level=pl.struct(LEVEL).rle_id())
    )
    starts = slots.group_by("level").agg(pl.col("slot").min()).sort("level")["slot"]
    first = np.append(starts.to_numpy(), slots.height).astype(np.int64)

    ordered = ordered.join(slots.select("oid", "slot", "level"), on="oid", how="left")
    front = _front(
        ordered["level"].to_numpy().astype(np.int64),
        ordered["slot"].to_numpy().astype(np.int64),
        ordered["size"].to_numpy(),
        first,
    )[ordered["slot"].to_numpy()]

    seq = ordered["seq"].to_numpy().astype(np.int64)
    return ordered.drop("slot", "level").with_columns(
        frontSeq=pl.when(pl.Series(front >= 0)).then(pl.Series(seq[np.maximum(front, 0)]))
    )


def ahead_path(rebuilt: pl.DataFrame, oid: int) -> pl.DataFrame:
    """Volume resting ahead of one order: at placement, then after each change ahead of it.

    Only orders that arrived earlier are ahead, and an order can only shrink once it
    rests, so the path starts at `volumeAhead` and falls by whatever they lose.
    For an order placed within the file; one resting before it opened has no start.
    """
    own = rebuilt.filter(pl.col("oid") == oid)
    placed = own.filter(pl.col("kind") == NEW).row(0, named=True)
    losses = (
        rebuilt.filter(*(pl.col(c) == placed[c] for c in LEVEL))
        .with_columns(arrival=pl.col("seq").min().over("oid"))
        .filter(
            pl.col("seq").is_between(placed["seq"], own["seq"].max())
            & (pl.col("arrival") < placed["seq"])
        )
        .sort("seq")
    )
    start = pl.DataFrame({"seq": [placed["seq"]], "ahead": [placed["volumeAhead"]]})
    path = losses.select("seq", ahead=placed["volumeAhead"] + pl.col("delta").cum_sum())
    return pl.concat([start, path.cast(start.schema)])
