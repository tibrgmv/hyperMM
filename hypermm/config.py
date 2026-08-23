"""Central configuration: filesystem paths and dataset constants."""

from __future__ import annotations

from pathlib import Path

# Filesystem layout
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = REPO_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = REPO_ROOT / "data" / "processed"


# Instruments covered by the L4 dataset
INSTRUMENTS: tuple[str, ...] = ("BTC", "ETH", "SOL")

# Data coverage windows (An Open Book, Zenodo record 18184441)
ORDER_STATUS_MONTH = "2025-12"
TRADES_START_MONTH = "2025-10"
TRADES_END_MONTH = "2026-01"


def ensure_data_dirs() -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
