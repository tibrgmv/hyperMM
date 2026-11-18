"""Scoring a fill rule against what a real order actually did."""

from __future__ import annotations

import polars as pl

RULES = {"touch": "tTouch", "volume-through": "tVolume", "queue-aware": "tQueue"}


def judge(predicted: pl.DataFrame, *, cancel_first: bool = False) -> pl.DataFrame:
    """Add `<rule>Within`: did the rule fill the order within its observed life --
    until its first real fill, its cancellation, or the end of the window?

    With `cancel_first`, a trade in the very block in which an unfilled order was
    cancelled does not fill it: inside a block the exchange takes cancellations
    before aggressors (journal, 23 August 2026).
    """
    end = pl.col("tPlaced").cast(pl.Int64) + pl.col("duration")
    removed = pl.col("tRemoved").cast(pl.Int64)

    def within(column: str) -> pl.Expr:
        fired = (pl.col(column) <= end).fill_null(False)
        if not cancel_first:
            return fired
        saved = (~pl.col("filled") & (pl.col(column) == removed)).fill_null(False)
        return fired & ~saved

    return predicted.with_columns(**{f"{c}Within": within(c) for c in RULES.values()})
