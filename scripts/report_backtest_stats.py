"""What a backtest under the wrong fill rule costs: statistics over the hourly rows
that report_backtest.py saves (data/processed/backtest/<date>_<coin>.parquet).

For each fee level and each fill rule, against the reference rule (queue-aware with
the block rule, the one that reproduces real fills):

  ranking     Kendall tau of the strategies' net PnL and of their hourly Sharpe ratio;
              the share of pairs of strategies the rule puts in the wrong order
  leaders     where the rule's five best strategies stand under the reference
  regret      choose the strategy the rule ranks first: its PnL under the reference,
              against the best strategy under the reference and against not trading
  overfitting the deflated Sharpe ratio of the rule's best strategy (Bailey and López
              de Prado 2014) and the probability of backtest overfitting by
              combinatorially symmetric cross-validation over the hours (Bailey et
              al. 2017), within the rule itself

A fill is one lot of $1 000; the maker fee is charged on that notional. Hyperliquid's
maker fee runs from 1.5 bp at the base tier through zero to a rebate for the largest
makers, hence the three levels.

Usage: python scripts/report_backtest_stats.py [--date 20251201] [--coins SOL ETH BTC]
"""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import polars as pl
from _report import section
from scipy.stats import kendalltau, kurtosis, norm, skew

from hypermm import config

REFERENCE = "queue-aware, block"
RULES = ("touch", "volume-through", "queue-aware", "queue-aware, block", "clock")
NOTIONAL = 1_000.0
FEES_BP = (1.5, 0.0, -0.3)
TOP = 5
BLOCKS = 8  # CSCV: the 24 hours in 8 blocks of 3, 70 train/test splits


def hourly(table: pl.DataFrame, rule: str, fee_bp: float) -> tuple[list[str], np.ndarray]:
    """Net PnL per hour (rows) and strategy (columns) under one rule."""
    net = pl.col("pnl") - pl.col("fills") * NOTIONAL * fee_bp / 1e4
    wide = (
        table.filter(pl.col("rule") == rule)
        .with_columns(net=net)
        .pivot(on="strategy", index="hour", values="net")
        .sort("hour")
    )
    names = [c for c in wide.columns if c != "hour"]
    return names, wide.select(names).to_numpy()


def sharpe(pnl: np.ndarray) -> np.ndarray:
    return pnl.mean(axis=0) / pnl.std(axis=0, ddof=1)


def deflated_sharpe(returns: np.ndarray, trials: np.ndarray) -> float:
    """Probability that the chosen strategy's true Sharpe ratio is positive, given
    that it is the best of `trials` (their Sharpe ratios) and how non-normal it is."""
    n = len(returns)
    sr = returns.mean() / returns.std(ddof=1)
    euler = 0.5772156649
    k = len(trials)
    expected_max = np.std(trials, ddof=1) * (
        (1 - euler) * norm.ppf(1 - 1 / k) + euler * norm.ppf(1 - 1 / (k * np.e))
    )
    g3, g4 = skew(returns), kurtosis(returns, fisher=False)
    spread = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr**2, 1e-12))
    return float(norm.cdf((sr - expected_max) * np.sqrt(n - 1) / spread))


def overfitting(pnl: np.ndarray) -> float:
    """PBO: over all splits of the hours into two halves, how often the strategy best
    in one half is below the median of all strategies in the other."""
    blocks = np.array_split(np.arange(len(pnl)), BLOCKS)
    below = total = 0
    for chosen in combinations(range(BLOCKS), BLOCKS // 2):
        inside = np.concatenate([blocks[i] for i in chosen])
        outside = np.setdiff1d(np.arange(len(pnl)), inside)
        best = int(np.argmax(sharpe(pnl[inside])))
        rank = (sharpe(pnl[outside]) < sharpe(pnl[outside])[best]).mean()
        below += rank < 0.5
        total += 1
    return below / total


def report(table: pl.DataFrame, coin: str) -> None:
    for fee in FEES_BP:
        section(f"{coin}: maker fee {fee:+.1f} bp")
        names, reference = hourly(table, REFERENCE, fee)
        ref_total, ref_sharpe = reference.sum(axis=0), sharpe(reference)
        best_ref = int(np.argmax(ref_total))
        print(f"    under the reference: {int((ref_total > 0).sum())} of {len(names)} strategies"
              f" profitable; best {names[best_ref]} ${ref_total[best_ref]:,.0f};"
              f" median ${np.median(ref_total):,.0f}")
        print("    rule                 tau PnL  tau Sharpe  pairs misordered  profitable"
              "   its best strategy          says $    really $   regret $   rank really")
        leaders = {}
        for rule in RULES:
            _, pnl = hourly(table, rule, fee)
            total = pnl.sum(axis=0)
            tau = kendalltau(total, ref_total).statistic
            tau_sharpe = kendalltau(sharpe(pnl), ref_sharpe).statistic
            best = int(np.argmax(total))
            rank = int((ref_total > ref_total[best]).sum()) + 1
            top = np.argsort(-total)[:TOP]
            leaders[rule] = [int((ref_total > ref_total[i]).sum()) + 1 for i in top]
            print(f"    {rule:<19} {tau:>8.2f} {tau_sharpe:>11.2f} {100 * (1 - tau) / 2:>16.1f}%"
                  f" {int((total > 0).sum()):>8} / {len(names)}   {names[best]:<22} {total[best]:>9,.0f}"
                  f" {ref_total[best]:>11,.0f} {ref_total[best_ref] - ref_total[best]:>10,.0f} {rank:>10} / {len(names)}")
        print(f"    where each rule's top {TOP} stand under the reference (rank of {len(names)}):")
        for rule in RULES:
            print(f"    {rule:<19} {leaders[rule]}")
        print("    overfitting within each rule: deflated Sharpe of its best strategy, PBO")
        for rule in RULES:
            _, pnl = hourly(table, rule, fee)
            ratios = sharpe(pnl)
            best = int(np.argmax(ratios))
            print(f"    {rule:<19} best Sharpe {ratios[best]:>6.2f} ({names[best]})"
                  f"   DSR {deflated_sharpe(pnl[:, best], ratios):.3f}   PBO {overfitting(pnl):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--date", default="20251201")
    parser.add_argument("--coins", nargs="+", default=["SOL", "ETH", "BTC"])
    args = parser.parse_args()
    for coin in args.coins:
        path = config.DATA_PROCESSED_DIR / "backtest" / f"{args.date}_{coin.lower()}.parquet"
        report(pl.read_parquet(path), coin)


if __name__ == "__main__":
    main()
