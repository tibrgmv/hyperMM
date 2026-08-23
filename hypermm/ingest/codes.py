"""Integer codes carried by the order status record, and a guard on their meaning."""

from __future__ import annotations

import re
from csv import reader
from enum import IntEnum
from pathlib import Path


class Status(IntEnum):
    BAD_ALO_PX_REJECTED = 0
    OPEN = 1
    CANCELED = 2
    PERP_MARGIN_REJECTED = 3
    IOC_CANCEL_REJECTED = 4
    FILLED = 5
    MIN_TRADE_NTL_REJECTED = 6
    REDUCE_ONLY_CANCELED = 7
    REDUCE_ONLY_REJECTED = 8
    TRIGGERED = 9
    SCHEDULED_CANCEL = 10
    SIBLING_FILLED_CANCELED = 11
    SELF_TRADE_CANCELED = 12
    MARGIN_CANCELED = 13
    VAULT_WITHDRAWAL_CANCELED = 14
    PERP_MAX_POSITION_REJECTED = 15
    LIQUIDATED_CANCELED = 16
    ORACLE_REJECTED = 17


class Tif(IntEnum):
    ALO = 0     # add liquidity only (post-only)
    GTC = 1     # good till cancel
    IOC = 2     # immediate or cancel
    NULL = 3
    FRONTEND_MARKET = 4    
    LIQUIDATION_MARKET = 5


def check_codes(mapdir: Path) -> None:
    """Raise if the lookup tables shipped with the data disagree with the constants above."""
    for filename, codes in (("statuses.csv", Status), ("tifs.csv", Tif)):
        shipped = _read_code_table(mapdir / filename)
        ours = {member.value: member.name for member in codes}
        mismatched = {
            code: (ours.get(code), shipped.get(code))
            for code in sorted(ours.keys() | shipped.keys())
            if ours.get(code) != shipped.get(code)
        }
        if mismatched:
            raise ValueError(
                f"{filename} disagrees with hypermm.ingest.codes "
                f"(code: expected, shipped): {mismatched}"
            )


def _read_code_table(path: Path) -> dict[int, str]:
    with path.open(newline="") as handle:
        rows = [row for row in reader(handle) if row]
    return {int(code): _normalize(label) for label, code in rows}


# badAloPxRejected -> BAD_ALO_PX_REJECTED
def _normalize(label: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", label).upper()
