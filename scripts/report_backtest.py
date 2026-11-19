"""What the fill rule does to a market-making backtest.

Every strategy variant of hypermm/strategies/quoters.py (VARIANTS: twelve families,
their parameters, three reaction speeds) is replayed on the rebuilt market under
each fill rule of hypermm/backtest/fills.py. Data, strategies and parameters are
the same in every run; only the fill rule changes.

Each hour is a separate one-hour run, rebuilt between its neighbours. Strategies
are calibrated on the hour before -- volatility, the exponential clock's A and k,
quantiles of the volume at the best price, and the survival model of fill time --
so none sees the hour it trades; the first hour of a span calibrates on itself
and is marked. One lot is worth $1 000 at the start of the hour. PnL is marked to
the mid, fees left out.

Usage: python scripts/report_backtest.py [--date 20251201] [--hours 0-23] [--coins SOL]
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
from _report import arguments, section
from scipy.stats import kendalltau, spearmanr

from hypermm import config
from hypermm.backtest import engine
from hypermm.backtest.fills import RULES
from hypermm.backtest.market import Market
from hypermm.fillcheck.survival import fit_fill_model, survival_table
from hypermm.fillsim.models import NS
from hypermm.pipeline import Reader, hour_bounds, load_window
from hypermm.strategies.quoters import BASELINE, GRID, VARIANTS, Params
from hypermm.toxicity.markout import markout

NOTIONAL = 1_000.0  # dollars per lot
REFERENCE = "queue-aware, block"  # the rule that reproduces real fills (Section 7)
TOP = 5


def calibrate(market: Market, start: int, end: int, survival: pl.DataFrame, lot: float) -> Params:
    grid = np.arange(start, end, 10 * NS)
    books = [market.book(int(t)) for t in grid]
    volumes = [v for b in books for v in (b.bid_volume, b.ask_volume)]
    return Params(
        tick=market.tick,
        sigma=market.sigma,
        a=market.a,
        k=market.k,
        queue_limits={q: int(np.quantile(volumes, q)) for q in (0.25, 0.5, 0.75)},
        lot_size=lot,
        survival=fit_fill_model(survival).for_size(lot),
    )


def fill_markout(run: engine.Run, top: pl.DataFrame, end: int) -> pl.Series:
    if not run.fills:
        return pl.Series([], dtype=pl.Float64)
    frame = pl.DataFrame(run.fills, schema=["t", "isAsk", "px"], orient="row")
    return markout(frame, top, 1, until=end).drop_nulls()


def per_variant(table: pl.DataFrame) -> pl.DataFrame:
    return table.group_by("strategy", "family", "rule", maintain_order=True).agg(
        fills=pl.col("fills").sum(),
        pnl=pl.col("pnl").sum(),
        mk=pl.col("mkSum").sum() / pl.col("mkN").sum(),
        sharpe=pl.col("pnl").mean() / pl.col("pnl").std(),  # hourly, not annualised
    )


def print_variants(summary: pl.DataFrame) -> None:
    print("    strategy                  rule                  fills      PnL $"
          "   per fill, bp   markout 1 s, bp    Sharpe, hourly")
    nan = float("nan")
    for strategy, _, rule, n, pnl, mk, sharpe in summary.iter_rows():
        per_fill = pnl / (n * NOTIONAL) * 1e4 if n else nan
        print(f"    {strategy:<25} {rule:<20} {n:>7,} {pnl:>10.2f} {per_fill:>14.3f}"
              f" {nan if mk is None else mk:>17.3f} {nan if sharpe is None else sharpe:>17.2f}")


def print_ranking(summary: pl.DataFrame, names: list[str], label: str) -> None:
    pnl = {(s, r): p for s, _, r, _, p, *_ in summary.iter_rows()}
    reference = np.array([pnl[(s, REFERENCE)] for s in names])
    top_n = min(TOP, len(names) - 1)
    best = set(np.array(names)[np.argsort(-reference)[:top_n]])
    print(f"    {label}: {len(names)} strategies")
    print("    rule                  Kendall tau  Spearman rho   sign differs   flattered"
          f"   top {top_n} kept   best strategy")
    for rule in RULES:
        values = np.array([pnl[(s, rule)] for s in names])
        tau = kendalltau(values, reference).statistic
        rho = spearmanr(values, reference).statistic
        flips = np.mean(np.sign(values) != np.sign(reference))
        flattered = np.mean(values > reference)
        top = set(np.array(names)[np.argsort(-values)[:top_n]])
        print(f"    {rule:<20} {tau:>12.2f} {rho:>13.2f} {100 * flips:>13.0f}%"
              f" {100 * flattered:>10.0f}% {len(top & best):>10} / {top_n}"
              f"   {names[int(np.argmax(values))]}")


def main() -> None:
    args = arguments(__doc__)
    warnings.filterwarnings("ignore")
    reader = Reader(args.raw)

    variants, tag = (GRID, "_grid") if args.grid else (VARIANTS, "")
    for coin in args.coins:
        rows = []
        previous = None  # (market, start, end, survival table) of the hour before
        own = []
        for hour in args.hours:
            start, end = hour_bounds(args.date, hour)
            window = load_window(reader, args.date, hour, coin)
            market = Market(window.top, window.fills)
            if market.book(start).mid is None:
                start = market.next_block(start)
            lot = NOTIONAL / (market.book(start).mid / 1e8)
            survival = survival_table(window.table, window.top, window.fills)
            if previous is None:
                own.append(hour)
                previous = (market, start, end, survival)
            params = calibrate(*previous, lot)
            for variant in variants:
                for rule in RULES:
                    strategy = variant.build(params)
                    run = engine.run(
                        market, strategy, rule, lot=lot, start=start, end=end,
                        step=int(variant.step * NS),
                    )
                    gains = fill_markout(run, window.top, end)
                    rows.append((hour, strategy.name, variant.family, rule, len(run.fills),
                                 run.pnl, float(gains.sum()), gains.len(),
                                 run.orders, run.quoted, run.depth, variant.step))
            previous = (market, start, end, survival)

        table = pl.DataFrame(
            rows,
            schema=["hour", "strategy", "family", "rule", "fills", "pnl", "mkSum", "mkN",
                    "orders", "quoted", "depth", "step"],
            orient="row",
        )
        raw = config.DATA_PROCESSED_DIR / "backtest"  # hourly rows, for later statistics
        raw.mkdir(parents=True, exist_ok=True)
        table.write_parquet(raw / f"{args.date}_{coin.lower()}{tag}.parquet")
        summary = per_variant(table)
        names = list(dict.fromkeys(summary["strategy"]))
        hours = f"{args.date}, hours {args.hours[0]}-{args.hours[-1]}"

        section(f"{coin}: {len(names)} strategies under five fill rules, {hours}")
        if own:
            print(f"    calibrated on their own hour (no hour before): {own}")
        print_variants(summary)

        section(f"{coin}: does the fill rule change which strategy looks best?")
        print_ranking(summary, names, "all variants")
        print()
        print_ranking(summary, [n for n in names if n in BASELINE], "the four of the first experiment")
        print(f"    (reference: {REFERENCE}, the rule that reproduces real fills)")


if __name__ == "__main__":
    main()
