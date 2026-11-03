"""What a bar-based backtest gets wrong about the price of a market order.

Backs docs/PRICE_ACCURACY.md (sections 3–5) and the journal entries of 24 August
2026 «Переориентация записки о цене» and «Ошибка цены рыночной заявки разложена».

For each size, one market order per bar is priced twice: as a chart backtest does
(the next bar's open) and by walking the rebuilt book. Sizes are in dollars and
converted to coins at the hour's median mid.

The two identities the note rests on are printed with their residuals:
    gift over a round trip = mean spread + depth paid buying + depth paid selling
    sell gift − buy gift   = 2 × drift + (depth selling − depth buying)

Usage: python scripts/report_price_accuracy.py [--date 20251201] [--hour 12] [--coins SOL BTC]
"""

from __future__ import annotations

import polars as pl
from _report import arguments, section

from hypermm.evaluate.price_accuracy import BAR, check, compare, summary
from hypermm.ingest.schema import SCALE
from hypermm.pipeline import load_hour, read_changes, read_fills

SIZES_USD = (1_000, 10_000, 100_000)
FOCUS_USD = 10_000
BAR_MINUTES = (1, 2, 5, 10)


def coins_for(usd: int, top: pl.DataFrame) -> int:
    known = top.drop_nulls(["bestBid", "bestAsk"])
    mid = ((known["bestBid"] + known["bestAsk"]) / 2).median()
    return int(usd * SCALE / (mid / SCALE))


def round_trip(top: pl.DataFrame, fills: pl.DataFrame, want: int, span: int = BAR):
    return (
        compare(top, fills, want, buying=True, span=span),
        compare(top, fills, want, buying=False, span=span),
    )


def report(top: pl.DataFrame, fills: pl.DataFrame, coin: str) -> None:
    first_bar = fills["ts"].min() // BAR

    print("    size       buy    sell   round trip   median spread   in spreads")
    rows = {}
    for usd in SIZES_USD:
        buy, sell = round_trip(top, fills, coins_for(usd, top))
        rows[usd] = (buy, sell)
        gift_buy, gift_sell = buy["nextOpenBps"].mean(), sell["nextOpenBps"].mean()
        median_spread = buy["spreadBps"].median()
        print(f"    ${usd:<8,} {gift_buy:>6.3f} {gift_sell:>7.3f} {gift_buy + gift_sell:>12.3f}"
              f" {median_spread:>15.3f} {(gift_buy + gift_sell) / median_spread:>12.2f}")

    print("\n    identity 1: round trip = mean spread + depth buying + depth selling")
    print("    size     mean spread  depth buy  depth sell     sum   measured   residual")
    for usd, (buy, sell) in rows.items():
        spread, depth_buy, depth_sell = buy["spreadBps"].mean(), buy["touchBps"].mean(), sell["touchBps"].mean()
        measured = buy["nextOpenBps"].mean() + sell["nextOpenBps"].mean()
        total = spread + depth_buy + depth_sell
        print(f"    ${usd:<8,} {spread:>10.3f} {depth_buy:>10.3f} {depth_sell:>11.3f}"
              f" {total:>7.3f} {measured:>10.3f} {measured - total:>10.1e}")

    print("\n    identity 2: sell − buy = 2 × drift + (depth selling − depth buying)")
    for usd, (buy, sell) in rows.items():
        drift = (10_000 * (buy["nextOpen"] - buy["mid"]) / buy["mid"]).mean()
        gap = sell["nextOpenBps"].mean() - buy["nextOpenBps"].mean()
        depth_gap = sell["touchBps"].mean() - buy["touchBps"].mean()
        print(f"    ${usd:<8,} sell − buy {gap:+.3f} = 2 × {drift:+.3f} {depth_gap:+.3f}"
              f"   residual {gap - 2 * drift - depth_gap:.1e}")

    buy, sell = rows[FOCUS_USD]
    median_spread = buy["spreadBps"].median()
    print(f"\n    pricing at the touch (knows the spread, not the depth), ${FOCUS_USD:,}:"
          f" buy {buy['touchBps'].mean() / median_spread:.2f}, sell {sell['touchBps'].mean() / median_spread:.2f}"
          " median spreads per side")
    print(f"\n    approximations, ${FOCUS_USD:,} buy (bps; positive flatters the strategy)")
    for row in summary(buy).iter_rows(named=True):
        print(f"      {row['approximation']:<9} mean {row['meanBps']:>7.3f}"
              f"   median |error| {row['medianAbsBps']:>7.3f}   flatters {row['flattersShare']:>5.1f}% of bars")

    print("\n    depth of the walk (levels touched)")
    for usd, (buy, sell) in rows.items():
        for side, frame in (("buy", buy), ("sell", sell)):
            deepest = frame.sort("levels", descending=True).row(0, named=True)
            print(f"      ${usd:<8,} {side:<4} median {frame['levels'].median():.0f},"
                  f" max {deepest['levels']} at minute {deepest['bar'] - first_bar}")

    print(f"\n    bar length, ${FOCUS_USD:,} round trip")
    for minutes in BAR_MINUTES:
        b, s = round_trip(top, fills, coins_for(FOCUS_USD, top), span=minutes * BAR)
        print(f"      {minutes:>2} min   bars {b.height:>3}   round trip {b['nextOpenBps'].mean() + s['nextOpenBps'].mean():.3f}")

    print("\n    checks (all must be zero):")
    for usd, (buy, sell) in rows.items():
        print(f"      ${usd:<8,} buy {check(buy)}  sell {check(sell)}")


def main() -> None:
    args = arguments(__doc__)
    changes = read_changes(args.raw, args.date, args.hour)
    fills = read_fills(args.raw, args.date, args.hour)
    for coin in args.coins:
        hour = load_hour(args.raw, args.date, args.hour, coin, changes=changes, fills=fills)
        section(f"{coin}: market order priced by the chart and by the book")
        report(hour.top, hour.fills, coin)


if __name__ == "__main__":
    main()
