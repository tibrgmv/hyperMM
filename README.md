# HyperMM

Queue-aware replay and fill-model risk measurement for market-making backtests
on Hyperliquid's on-chain CLOB, using the public L4 dataset (*An Open Book*,
Albers, Cucuringu, Howison, Shestopaloff, 2026).

Central question: naive MM backtests assume a fill rule that cannot be
observed historically. This project measures how much that unobserved
assumption changes backtest conclusions (fill-rate, PnL, Sharpe, strategy
ranking), using real order outcomes from Hyperliquid's L4 data as ground
truth.

See `docs/` (or the project's research proposal) for the full research
design, hypotheses (H1-H5) and methodology.

## Status

Data: one full day, 2025-12-01, BTC, ETH and SOL (`scripts/fetch_data.py`).

| research question | state |
|---|---|
| RQ1 fill rules against real outcomes | done on the pilot hour; censoring-aware metric panel still to do |
| RQ2 ceiling of predictability, normalised scores | done on the day: `fillcheck/ceiling.py`, `report_ceiling.py` |
| RQ3 what the fill rule does to a backtest | 48 strategy variants x 5 rules on the day; fees, Sharpe-based ranking, deflated Sharpe and the ~150-configuration grid still to do |
| RQ4 robustness | not started beyond three instruments and reaction speed |
| RQ5 in-block cancellation | done on the pilot hour; the day still to do |

The research proposal is `RP_ВКР_HMM.pdf`; the paper is in `paper/`. Progress and
decisions: `docs/JOURNAL.md`.

## Setup

```bash
uv venv
uv pip install -e ".[dev]"
# later, as needed:
uv pip install -e ".[ml,viz]"
```

## Layout

```
hypermm/
├── pipeline.py    # raw files -> rebuilt book -> outcome table; one hour alone or chained to its neighbours
├── ingest/        # raw An Open Book streams -> typed frames
├── reconstruct/   # L4 book + exact queue position reconstruction
├── fillsim/       # 4 fill models: naive touch / exponential clock / volume-through / queue-aware
├── fillcheck/     # fill rules against real outcomes; survival model; ceiling of predictability; cached order tables
├── toxicity/      # markout
├── strategies/    # 48 quoting variants in 12 families (A-S, GLFT, ladders, filters, ...)
├── backtest/      # the rebuilt market as a strategy sees it, fill rules for orders never in the book, engine
├── evaluate/      # market-order price accuracy
└── replay_ui/     # export of one hour for the web page in web/
scripts/           # fetch_data, check_* diagnostics, report_* behind every number, figures and tables of the paper
docs/reports/      # reference output of the report scripts: 20251201_12/ (pilot hour), 20251201_day/
paper/             # the paper (LaTeX): `tectonic -X compile main.tex`
web/               # static replay page
```

## Reproducing the numbers

Every number in `docs/JOURNAL.md` and in the paper is printed by one of these;
reference output is in `docs/reports/`.

```bash
python scripts/fetch_data.py --date 20251201 --hours 0-23 --coins btc sol eth   # ~4.4 GB

# the pilot hour (2025-12-01 12:00), docs/reports/20251201_12/
python scripts/check_block_semantics.py data/raw/order_statuses/20251201/btc_12.data.gz \
                                        data/raw/order_statuses/20251201/sol_12.data.gz
python scripts/report_streams.py          # stream contents and cross-stream agreement
python scripts/report_reconstruction.py   # queue, time, top of book
python scripts/report_fills.py            # makers, status-stream gaps, outcomes
python scripts/report_fill_models.py      # four fill models against real outcomes
python scripts/report_block_protection.py # RQ5: cancellation inside the aggressor's block
python scripts/report_price_accuracy.py   # docs/PRICE_ACCURACY.md

# the day, docs/reports/20251201_day/ (one file per coin: redirect the output there)
python scripts/build_orders.py --hours 0-23 --coins SOL ETH BTC   # cached order tables
python scripts/report_survival.py --coins SOL    # fill time: Cox model by stage
python scripts/report_ceiling.py --coins SOL     # RQ2: ceiling of predictability
python scripts/report_backtest.py --hours 0-23 --coins SOL   # RQ3: strategies x fill rules

# the paper
python scripts/paper_tables.py && python scripts/figures.py
```

## Data

Source: *An Open Book* L4 dataset on Zenodo (record 18184441; BTC/ETH/SOL order
lifecycle events incl. rejected orders, book diffs, trades). Not committed.
Expected layout under `data/raw/`:

```
book_diffs/20251201/ex12.gz
order_statuses/20251201/{btc,sol}_12.data.gz
trades/20251201/12.gz
```

Download notes (rate limits, single-file extraction) are in the journal entry of
23 August 2026.
