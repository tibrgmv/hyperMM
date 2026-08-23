"""Exact parsing of the decimal strings carried by the JSON streams."""

from __future__ import annotations

import polars as pl

DECIMALS = 8


def parse_decimal(column: str | pl.Expr) -> pl.Expr:
    """Parse a decimal string into int64 units of 1/SCALE by shifting digits as text."""

    source = pl.col(column) if isinstance(column, str) else column
    parts = source.str.split(".")
    whole = parts.list.get(0)
    frac = parts.list.get(1, null_on_oob=True).fill_null("")
    
    return (
        pl.when(frac.str.len_chars() <= DECIMALS)
        .then((whole + frac.str.pad_end(DECIMALS, "0")).cast(pl.Int64))
        .otherwise(None)
    )
