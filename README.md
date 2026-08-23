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

Stage 1 of 10 (see calendar plan): data ingestion. Not yet functional —
under active development.

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
├── ingest/        # raw An Open Book streams -> normalized Parquet
├── reconstruct/   # L4 book + exact queue position reconstruction
├── fillsim/       # 4 fill models: naive touch / exponential clock / volume-through / queue-aware
├── fillcheck/     # validates fill models against real order outcomes (survival analysis)
├── toxicity/      # lookahead-free wallet-resolved toxicity signal + markout
├── strategies/    # A-S / GLFT / symmetric quoters / queue-aware heuristics
├── backtest/      # backtest engine, PnL accounting (interval-valued)
├── evaluate/      # fill-gap, ranking statistics, robustness cuts
└── replay_ui/     # replay/analysis tool (queue, fill-path, PnL decomposition)
```

## Data

Source: *An Open Book* L4 dataset on Zenodo (BTC/ETH/SOL order lifecycle
events incl. rejected orders, book diffs, trades). Not committed to this
repo — see `hypermm/ingest/` once the download recipe is written.
