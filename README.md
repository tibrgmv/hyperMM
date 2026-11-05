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

Done: ingest, book and queue reconstruction, outcomes of passive orders, the
front of the queue, and the four fill models measured against real outcomes
(one hour, 2025-12-01 12:00 UTC, BTC and SOL). Next: RQ5 (in-block cancellation)
and survival analysis. Progress and decisions: `docs/JOURNAL.md`.

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
├── pipeline.py    # one hour of one coin: raw files -> rebuilt book -> outcome table
├── ingest/        # raw An Open Book streams -> typed frames
├── reconstruct/   # L4 book + exact queue position reconstruction
├── fillsim/       # 4 fill models: naive touch / exponential clock / volume-through / queue-aware
├── fillcheck/     # validates fill models against real order outcomes (survival analysis)
├── toxicity/      # lookahead-free wallet-resolved toxicity signal + markout
├── strategies/    # A-S / GLFT / symmetric quoters / queue-aware heuristics
├── backtest/      # backtest engine, PnL accounting (interval-valued)
├── evaluate/      # fill-gap, ranking statistics, robustness cuts; market-order price accuracy
└── replay_ui/     # export of one hour for the web page in web/
scripts/           # check_* diagnostics and report_* scripts behind the journal's numbers
docs/reports/      # reference output of the report scripts, per hour
web/               # static replay page
```

## Reproducing the journal

Every number in `docs/JOURNAL.md` comes from one of these, run on the hour in
`data/raw/`; the reference output is in `docs/reports/20251201_12/`.

```bash
python scripts/check_block_semantics.py data/raw/order_statuses/20251201/btc_12.data.gz \
                                        data/raw/order_statuses/20251201/sol_12.data.gz
python scripts/report_streams.py          # stream contents and cross-stream agreement
python scripts/report_reconstruction.py   # queue, time, top of book
python scripts/report_fills.py            # makers, status-stream gaps, outcomes
python scripts/report_fill_models.py      # four fill models against real outcomes
python scripts/report_price_accuracy.py   # docs/PRICE_ACCURACY.md
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
