"""Build the cached per-hour order tables (hypermm/fillcheck/dataset.py).

The reports of the fill-time analysis build them on demand; running this first
does it for all coins in one pass, parsing each hour's shared files once.

Usage: python scripts/build_orders.py [--date 20251201] [--hours 0-23] [--coins SOL ETH BTC]
"""

from __future__ import annotations

import time

from _report import arguments

from hypermm.fillcheck.dataset import hour_path, load_orders
from hypermm.pipeline import Reader


def main() -> None:
    args = arguments(__doc__)
    reader = Reader(args.raw)
    for hour in args.hours:
        for coin in args.coins:
            if hour_path(args.date, hour, coin).exists():
                continue
            started = time.time()
            frame = load_orders(reader, args.date, hour, coin)
            filled = frame["totalEvent"].mean()
            print(
                f"{args.date} {hour:02d} {coin}: {frame.height:,} orders, {100 * filled:.2f}%"
                f" filled, {time.time() - started:.0f} s",
                flush=True,
            )


if __name__ == "__main__":
    main()
