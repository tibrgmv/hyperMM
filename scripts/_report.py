"""Shared plumbing for the report scripts: argument parsing and printing.

Every report regenerates the numbers of one part of docs/JOURNAL.md from the raw
files, so that a figure in the text can always be traced back to code.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from hypermm import config

COINS_WITH_STATUSES = ("SOL", "BTC")  # the order status files downloaded so far


def arguments(doc: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--raw", type=Path, default=config.DATA_RAW_DIR)
    parser.add_argument("--date", default="20251201")
    parser.add_argument("--hour", type=int, default=12)
    parser.add_argument("--coins", nargs="+", default=list(COINS_WITH_STATUSES))
    return parser.parse_args()


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def share(part: float, whole: float) -> str:
    return f"{100 * part / whole:.2f}%" if whole else "—"


class Stopwatch:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def __str__(self) -> str:
        return f"{time.perf_counter() - self.start:.1f} s"
