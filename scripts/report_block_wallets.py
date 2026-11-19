"""RQ5, by participant: who is rescued by cancelling inside the aggressor's block.

An encounter is an order at the front of its queue meeting a trade at its price
within its life (the queue-aware rule fires); it is rescued when the order was
cancelled in that very block and not filled (the block rule withholds the fill).
Both are read from the cached order tables; the owner of each order comes from
the book-diff stream. If rescue is a property of the venue, every maker gets it
at a similar rate; if it is a skill, a few makers do.

Addresses are shortened in the output.

Usage: python scripts/report_block_wallets.py [--date 20251201] [--hours 0-23] [--coins SOL ETH BTC]
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
from _report import arguments, section, share

from hypermm.fillcheck.dataset import load_orders
from hypermm.pipeline import Reader

TOP = 10
MIN_ENCOUNTERS = 200  # makers with fewer are too thin to give a rate


def main() -> None:
    args = arguments(__doc__)
    reader = Reader(args.raw, keep=2)
    met = {coin: [] for coin in args.coins}
    for hour in args.hours:
        at = datetime.strptime(args.date, "%Y%m%d") + timedelta(hours=hour)
        owners = reader.changes(at).filter(pl.col("kind") == "new").select("oid", "coin", "user")
        for coin in args.coins:
            orders = load_orders(reader, args.date, hour, coin).filter("queueWithin")
            met[coin].append(
                orders.select(
                    "oid",
                    rescued=pl.col("queueWithin") & ~pl.col("queueWithinBlock"),
                    filled="totalEvent",
                ).join(owners.filter(pl.col("coin") == coin).select("oid", "user"), on="oid")
            )

    for coin in args.coins:
        pooled = pl.concat(met[coin])
        makers = (
            pooled.group_by("user")
            .agg(encounters=pl.len(), rescued=pl.col("rescued").sum(), filled=pl.col("filled").sum())
            .with_columns(rate=pl.col("rescued") / pl.col("encounters"))
            .sort("encounters", descending=True)
        )
        total, saved = pooled.height, int(pooled["rescued"].sum())
        section(f"{coin}: who meets the aggressor, and who gets away")
        print(f"    encounters {total:,} by {makers.height:,} makers; rescued {saved:,}"
              f" ({share(saved, total)})")
        by_rescued = makers.sort("rescued", descending=True)
        for n in (1, 5, TOP):
            print(f"    top {n:>2} makers by encounters hold {share(makers['encounters'].head(n).sum(), total)}"
                  f" of encounters; top {n:>2} by rescues hold {share(by_rescued['rescued'].head(n).sum(), saved)} of rescues")
        never = makers.filter(pl.col("rescued") == 0)
        print(f"    makers never rescued: {never.height:,} of {makers.height:,},"
              f" with {share(never['encounters'].sum(), total)} of encounters")

        print(f"    the {TOP} makers with most encounters:")
        print("    maker        encounters    rescued    rate")
        for user, n, r, _, rate in makers.head(TOP).iter_rows():
            print(f"    {user[:6]}..{user[-4:]} {n:>10,} {r:>10,} {100 * rate:>6.2f}%")
        thick = makers.filter(pl.col("encounters") >= MIN_ENCOUNTERS)
        rates = thick["rate"]
        print(f"    rescue rate among the {thick.height} makers with at least {MIN_ENCOUNTERS} encounters:"
              f" min {100 * rates.min():.2f}%, quartiles {100 * rates.quantile(0.25):.2f}% /"
              f" {100 * rates.median():.2f}% / {100 * rates.quantile(0.75):.2f}%, max {100 * rates.max():.2f}%")


if __name__ == "__main__":
    main()
