"""A fill rule learned from L1 and L2 data, against the rules and the L4 truth.

For each horizon (1, 5, 30 s) and each order whose state at the horizon is known,
gradient boosting predicts whether the order was filled within it, from the
placement-time features (all visible in L2) plus the L1 or the L1 + L2 view of its
level over the window (hypermm/fillcheck/l2rule.py). Trained on the first hours,
scored on the later ones, against touch, volume-through and queue-aware evaluated
at the same horizon on the same orders. The learned rules answer with a
probability; "× real" is the sum of their probabilities over the true fills.

Usage: python scripts/report_l2_rule.py [--date 20251201] [--train 0-10] [--test 12-23]
                                        [--coins SOL ETH BTC] [--sample 1000000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from _report import hour_list, section

from hypermm import config
from hypermm.fillcheck import ceiling, l2rule
from hypermm.fillcheck.dataset import load_orders
from hypermm.fillcheck.survival import FEATURES
from hypermm.pipeline import Reader, load_window

CACHE = config.DATA_PROCESSED_DIR / "l2" / "v1"
RULES = {"touch": "touch", "volume-through": "volume", "queue-aware": "queue",
         "queue-aware, block": "queueBlock"}


def hour_frame(reader: Reader, date: str, hour: int, coin: str) -> pl.DataFrame:
    path = CACHE / date / f"{coin.lower()}_{hour:02d}.parquet"
    if not path.exists():
        frame = l2rule.l2_table(load_window(reader, date, hour, coin))
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
    placement = load_orders(reader, date, hour, coin).select("oid", *FEATURES)
    return pl.read_parquet(path).join(placement, on="oid")


def pooled(reader, date, hours, coin, sample) -> pl.DataFrame:
    frame = pl.concat([hour_frame(reader, date, h, coin) for h in hours])
    return frame.sample(min(sample, frame.height), seed=0)


def matrix(frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    return frame.select(pl.col(c).cast(pl.Float64) for c in columns).to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--raw", type=Path, default=config.DATA_RAW_DIR)
    parser.add_argument("--date", default="20251201")
    parser.add_argument("--train", type=hour_list, default=hour_list("0-10"))
    parser.add_argument("--test", type=hour_list, default=hour_list("12-23"))
    parser.add_argument("--coins", nargs="+", default=["SOL", "ETH", "BTC"])
    parser.add_argument("--sample", type=int, default=1_000_000)
    args = parser.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier

    reader = Reader(args.raw)
    for coin in args.coins:
        train = pooled(reader, args.date, args.train, coin, args.sample)
        test = pooled(reader, args.date, args.test, coin, args.sample)
        section(f"{coin}: fill within a horizon -- rules, learned rules, truth")
        for h in l2rule.HORIZONS:
            a, b = train.filter(f"known{h}"), test.filter(f"known{h}")
            y_train, y = a[f"filled{h}"].to_numpy(), b[f"filled{h}"].to_numpy()
            base = float(y_train.mean())
            answers = {name: b[f"{column}{h}"].to_numpy().astype(float) for name, column in RULES.items()}
            sets = {
                "learned, L1": [*FEATURES, *(f"{f}{h}" for f in l2rule.L1)],
                "learned, L1 + L2": [*FEATURES, *(f"{f}{h}" for f in (*l2rule.L1, *l2rule.L2))],
            }
            for name, columns in sets.items():
                model = HistGradientBoostingClassifier(
                    max_iter=400, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=200,
                    l2_regularization=1.0, early_stopping=True, n_iter_no_change=15, random_state=0,
                )  # fmt: skip
                model.fit(matrix(a, columns), y_train)
                answers[name] = model.predict_proba(matrix(b, columns))[:, 1]
            print(f"    horizon {h} s: {len(y):,} test orders with a known state, {100 * y.mean():.2f}% filled")
            print("    rule                    × real      AUC   Brier skill   false, of unfilled   missed, of filled")
            for name, p in answers.items():
                said = p >= 0.5
                print(f"    {name:<21} {p.sum() / y.sum():>8.2f} {ceiling.realised_auc(y, p):>8.4f}"
                      f" {ceiling.brier_skill(y, p, base):>13.3f}"
                      f" {100 * (said & ~y).sum() / (~y).sum():>19.3f}% {100 * (~said & y).sum() / y.sum():>18.2f}%")


if __name__ == "__main__":
    main()
