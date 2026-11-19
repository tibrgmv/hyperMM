"""The candle backtest's fill rule against the real outcome of real orders.

The most common backtest of all runs on bars: a buy limit order counts as filled
once a bar's low reaches its price (a sell order, once the high does). The bar does
not know *when* within it the price was there, so a touch minutes before the order
was placed fills it just the same. Here the rule is applied to every real passive
order: it fires if any bar overlapping the order's life reached its price. Two
readings in use are both scored -- the bar's extreme must reach the price ("touch"),
or go strictly through it ("through", the cautious setting some platforms offer).

Bars are built from the trade stream at 1, 5 and 15 minutes, aligned to the hour.
The last rows repeat, from the cached order tables, the trade-level touch rule and
the queue-aware rule with the block rule, on the same orders.

Most real orders live under a second, far less than a bar, and a bar backtest
cannot hold an order for less than a bar; so the rule is also scored on the
orders that rested for at least one bar's length, the ones a bar backtest could
have placed.

Usage: python scripts/report_candles.py [--date 20251201] [--hours 0-23] [--coins SOL ETH BTC]
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
from _report import arguments, section

from hypermm.fillcheck import ceiling
from hypermm.fillcheck.dataset import load_orders
from hypermm.pipeline import NS, Reader, hour_bounds

MINUTES = (1, 5, 15)
SPAN = 2 * 3600  # seconds of trades a bar grid covers: the hour and the next


def extremes(trades: pl.DataFrame, start: int, minutes: int) -> tuple[np.ndarray, np.ndarray]:
    """Lowest low and highest high over every range of bars [first, last]."""
    bars = SPAN // (60 * minutes)
    index = ((trades["ts"].to_numpy() - start) // (60 * minutes * NS)).astype(np.int64)
    px = trades["px"].to_numpy().astype(np.float64)
    keep = (index >= 0) & (index < bars)
    low, high = np.full(bars, np.inf), np.full(bars, -np.inf)
    np.minimum.at(low, index[keep], px[keep])
    np.maximum.at(high, index[keep], px[keep])
    lows = np.full((bars, bars), np.inf)
    highs = np.full((bars, bars), -np.inf)
    for first in range(bars):
        lows[first, first:] = np.minimum.accumulate(low[first:])
        highs[first, first:] = np.maximum.accumulate(high[first:])
    return lows, highs


def main() -> None:
    args = arguments(__doc__)
    reader = Reader(args.raw, keep=3)
    answers: dict[str, dict[str, list[np.ndarray]]] = {coin: {} for coin in args.coins}
    for hour in args.hours:
        at = datetime.strptime(args.date, "%Y%m%d") + timedelta(hours=hour)
        start, _ = hour_bounds(args.date, hour)
        prices = reader.changes(at).filter(pl.col("kind") == "new").select("oid", "coin", "px")
        later = reader.fills(at + timedelta(hours=1))
        fills = reader.fills(at) if later is None else pl.concat([reader.fills(at), later])
        for coin in args.coins:
            orders = load_orders(reader, args.date, hour, coin).join(
                prices.filter(pl.col("coin") == coin).select("oid", "px"), on="oid"
            )
            placed = orders["tPlaced"].cast(pl.Int64).to_numpy()
            ended = placed + (orders["totalDuration"].to_numpy() * NS).astype(np.int64)
            px, is_ask = orders["px"].to_numpy().astype(np.float64), orders["isAsk"].to_numpy()
            out = answers[coin]
            out.setdefault("real", []).append(orders["totalEvent"].to_numpy())
            trades = fills.filter(pl.col("coin") == coin)
            for minutes in MINUTES:
                lows, highs = extremes(trades, start, minutes)
                width, bars = 60 * minutes * NS, lows.shape[0]
                first = np.clip((placed - start) // width, 0, bars - 1)
                last = np.clip((ended - start) // width, 0, bars - 1)
                low, high = lows[first, last], highs[first, last]
                out.setdefault(f"candle {minutes:>2} min, touch", []).append(
                    np.where(is_ask, high >= px, low <= px)
                )
                out.setdefault(f"candle {minutes:>2} min, through", []).append(
                    np.where(is_ask, high > px, low < px)
                )
            out.setdefault("life", []).append(orders["totalDuration"].to_numpy())
            out.setdefault("touch, trade by trade", []).append(orders["touchWithin"].to_numpy())
            out.setdefault("queue-aware, block", []).append(orders["queueWithinBlock"].to_numpy())

    for coin in args.coins:
        pooled = {name: np.concatenate(parts) for name, parts in answers[coin].items()}
        real, life = pooled.pop("real"), pooled.pop("life")
        section(f"{coin}: the candle rule against the real outcome, {args.date} hours"
                f" {args.hours[0]}-{args.hours[-1]}")
        print(f"    orders {len(real):,}; filled {100 * real.mean():.2f}%")
        print("    rule                        says filled   × real   false, of unfilled"
              "   missed, of filled      AUC   Brier skill")
        for name, said in pooled.items():
            print(f"    {name:<26} {100 * said.mean():>11.2f}% {said.sum() / real.sum():>8.2f}"
                  f" {100 * (said & ~real).sum() / (~real).sum():>18.2f}%"
                  f" {100 * (~said & real).sum() / real.sum():>18.2f}%"
                  f" {ceiling.realised_auc(real, said):>8.4f}"
                  f" {ceiling.brier_skill(real, said, real.mean()):>13.2f}")


        # a bar backtest cannot hold an order for less than a bar: the same rule on the
        # orders that rested for at least one bar's length
        print("    orders that rested at least one bar:")
        print("    bar          orders   really filled   candle says   × real   false, of unfilled"
              "   touch by trade × real   queue-aware, block × real")
        for minutes in MINUTES:
            long = life >= 60 * minutes
            truth, said = real[long], pooled[f"candle {minutes:>2} min, touch"][long]
            print(f"    {minutes:>2} min {long.sum():>12,} {100 * truth.mean():>14.2f}% {100 * said.mean():>12.2f}%"
                  f" {said.sum() / truth.sum():>8.2f} {100 * (said & ~truth).sum() / (~truth).sum():>18.2f}%"
                  f" {pooled['touch, trade by trade'][long].sum() / truth.sum():>22.2f}"
                  f" {pooled['queue-aware, block'][long].sum() / truth.sum():>27.2f}")


if __name__ == "__main__":
    main()
