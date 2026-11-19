"""Shared plumbing for the report scripts: argument parsing and printing.

Every report regenerates the numbers of one part of docs/JOURNAL.md from the raw
files, so that a figure in the text can always be traced back to code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hypermm import config

COINS_WITH_STATUSES = ("SOL", "BTC")  # the order status files downloaded so far


def arguments(doc: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--raw", type=Path, default=config.DATA_RAW_DIR)
    parser.add_argument("--date", default="20251201")
    parser.add_argument("--hour", type=int, default=12)
    parser.add_argument(
        "--hours",
        type=hour_list,
        default=None,
        help="several hours, e.g. 0-23 or 10,12-14 (reports that pool hours)",
    )
    parser.add_argument("--coins", nargs="+", default=list(COINS_WITH_STATUSES))
    parser.add_argument(
        "--chained",
        action="store_true",
        help="rebuild each hour between its neighbours (reports that support it)",
    )
    parser.add_argument(
        "--grid", action="store_true",
        help="the full grid of strategy configurations (report_backtest.py)",
    )
    args = parser.parse_args()
    if args.hours is None:
        args.hours = [args.hour]
    return args


def hour_list(text: str) -> list[int]:
    """Parse "0-23" or "10,12-14" into a list of hours."""
    out: list[int] = []
    for part in text.split(","):
        first, _, last = part.partition("-")
        out.extend(range(int(first), int(last or first) + 1))
    return out


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def share(part: float, whole: float) -> str:
    return f"{100 * part / whole:.2f}%" if whole else "—"
