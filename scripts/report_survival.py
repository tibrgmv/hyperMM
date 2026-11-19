"""Fill time as a survival problem: how well features known at placement rank
orders by how soon they reach the front of their queue and are filled.

Cox proportional hazards is fitted on the training hours and scored on the test
hours by Harrell's concordance, for each stage of a fill (see
hypermm/fillcheck/survival.py). For comparison, single features ranked on their
own: the volume ahead, the distance behind the best price, and the distance from
the mid-price -- the only thing the exponential clock of Avellaneda--Stoikov
prices a fill by, so its concordance is the clock's.

Each hour is rebuilt between its neighbours (pipeline.load_window).

Usage: python scripts/report_survival.py [--date 20251201] [--train 0-11] [--test 12-23]
                                         [--coins SOL BTC] [--sample 1000000]

Over a full day an instrument has millions of orders; a random sample of each of
the training and the test hours keeps the Cox fit to minutes.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import polars as pl
from _report import COINS_WITH_STATUSES, hour_list, section, share

from hypermm import config
from hypermm.fillcheck.survival import STAGES, concordance, fit_cox, survival_table
from hypermm.pipeline import Reader, load_window

BASELINES = {
    "volume ahead": "logAhead",
    "ticks behind best": "ticksBehind",
    "ticks from mid (clock)": "ticksFromMid",
}


def table(reader: Reader, date: str, hours: list[int], coin: str) -> pl.DataFrame:
    parts = []
    for hour in hours:
        rebuilt = load_window(reader, date, hour, coin)
        parts.append(survival_table(rebuilt.table, rebuilt.top, rebuilt.fills))
    return pl.concat(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--raw", type=Path, default=config.DATA_RAW_DIR)
    parser.add_argument("--date", default="20251201")
    parser.add_argument("--train", type=hour_list, default=hour_list("12"))
    parser.add_argument("--test", type=hour_list, default=hour_list("13"))
    parser.add_argument("--coins", nargs="+", default=list(COINS_WITH_STATUSES))
    parser.add_argument(
        "--sample", type=int, default=1_000_000,
        help="orders drawn at random from each of train and test (seed 0); 0 keeps all",
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore", module="lifelines")
    reader = Reader(args.raw)

    for coin in args.coins:
        train = table(reader, args.date, args.train, coin)
        test = table(reader, args.date, args.test, coin)
        drawn = ""
        if args.sample:
            drawn = f"; {args.sample:,} orders drawn from each of {train.height:,} / {test.height:,}"
            train = train.sample(min(args.sample, train.height), seed=0)
            test = test.sample(min(args.sample, test.height), seed=0)
        section(f"{coin}: train hours {args.train}, test hours {args.test}")
        if drawn:
            print(f"    {drawn[2:]}")
        print("    stage   orders train / test     event share train / test"
              "    C, Cox: train / test")
        models = {}
        for stage in STAGES:
            models[stage] = model = fit_cox(train, stage)
            n_train = train[f"{stage}Duration"].drop_nulls().len()
            n_test = test[f"{stage}Duration"].drop_nulls().len()
            print(f"    {stage:<7} {n_train:>12,} / {n_test:<10,}"
                  f" {share(train[f'{stage}Event'].sum(), n_train):>12} / {share(test[f'{stage}Event'].sum(), n_test):<12}"
                  f" {concordance(train, stage, model):>8.3f} / {concordance(test, stage, model):.3f}")

        section(f"{coin}: single features on the test hours, C")
        print("    stage   " + "".join(f"{name:>24}" for name in BASELINES))
        for stage in STAGES:
            print(f"    {stage:<7} " + "".join(
                f"{concordance(test, stage, column):>24.3f}" for column in BASELINES.values()
            ))

        section(f"{coin}: Cox hazard ratios on the training hours (total stage)")
        summary = models["total"].summary
        print("    feature           hazard ratio      p")
        for name, row in summary.iterrows():
            print(f"    {name:<16} {row['exp(coef)']:>13.3f}  {row['p']:>7.3g}")


if __name__ == "__main__":
    main()
