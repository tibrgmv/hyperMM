"""Table rows of the paper, written to paper/tables/ as LaTeX.

Rows are read from the reference output of the report scripts, never typed by
hand, so a number in the paper can always be traced to a script. Whatever a report
does not (yet) hold is left as an empty red slot (\\tbd).

Usage: python scripts/paper_tables.py [--reports docs/reports/20251201_day]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from hypermm import config

COINS = ("BTC", "ETH", "SOL")
RULES = ("touch", "volume-through", "queue-aware", "queue-aware, block", "clock")
RULE_TEX = {
    "touch": r"\touch",
    "volume-through": r"\vt",
    "queue-aware": r"\qa",
    "queue-aware, block": r"\qa, block",
    "clock": r"\clock",
}
# longest first, so that "queue-aware, block" is not read as "queue-aware"
RULE_PATTERN = "|".join(sorted(map(re.escape, RULES), key=len, reverse=True))
TBD = r"\tbd"

STAGE_TEX = {"front": "Front", "fill": r"Fill $\mid$ front", "total": "Total"}
FEATURE_TEX = {
    "isAsk": "Ask side",
    "logSize": "log size",
    "logAhead": "log volume ahead",
    "emptyLevel": "Empty level",
    "ticksBehind": "Ticks behind best",
    "ticksFromMid": "Ticks from mid",
    "spreadTicks": "Spread (ticks)",
    "imbalance": "Imbalance, own side",
    "trades1s": r"Trades, last 1\,s",
    "trades10s": r"Trades, last 10\,s",
    "move1s": r"Move to order, 1\,s",
    "move10s": r"Move to order, 10\,s",
}


def blocks(path: Path):
    """(coin, title, body) of each section of a report."""
    parts = re.split(r"^={78}\n(\w+): (.+)\n={78}\n", path.read_text(), flags=re.M)
    for i in range(1, len(parts) - 2, 3):
        yield parts[i], parts[i + 1], parts[i + 2]


def num(value: float, fmt: str) -> str:
    return f"${value:{fmt}}$"


# --- backtest ---------------------------------------------------------------------

ROW = re.compile(
    rf"^ {{4}}(\S.*?)\s+({RULE_PATTERN})\s+([\d,]+)\s+(-?[\d.]+)\s+(-?[\d.]+|nan)"
    r"\s+(-?[\d.]+|nan)\s+(-?[\d.]+|nan)$",
    re.M,
)
RANK = re.compile(
    rf"^ {{4}}({RULE_PATTERN})\s+(-?[\d.]+|nan)\s+(-?[\d.]+|nan)\s+(\d+)%\s+(\d+)%"
    r"\s+(\d+) / (\d+)",
    re.M,
)


def read_backtest(reports: Path) -> tuple[dict, dict, list[str]]:
    """(coin, strategy, rule) -> (fills, pnl, markout, hourly Sharpe or None);
    (coin, rule) -> (tau, rho, sign differs %, flattered %, kept, of), over all
    variants; and the strategy names in report order."""
    cells, ranks, names = {}, {}, []
    for path in sorted(reports.glob("backtest*.txt")):
        for coin, title, body in blocks(path):
            if "strategies under" in title:
                hourly = "Sharpe, hourly" in body  # older reports annualised it
                for m in ROW.finditer(body):
                    name, rule = m.group(1).strip(), m.group(2)
                    if name not in names:
                        names.append(name)
                    cells[(coin, name, rule)] = (
                        int(m.group(3).replace(",", "")),
                        float(m.group(4)),
                        float(m.group(6)),
                        float(m.group(7)) if hourly else None,
                    )
            elif title.startswith("does the fill rule"):
                over_all = body.split("\n\n")[0]  # the first ranking is over all variants
                for m in RANK.finditer(over_all):
                    tau, rho, flips, flattered, kept, of = m.groups()[1:]
                    ranks[(coin, m.group(1))] = (
                        float(tau),
                        float(rho),
                        int(flips),
                        int(flattered),
                        int(kept),
                        int(of),
                    )
    return cells, ranks, names


def ranking_rows(ranks: dict) -> str:
    lines = []
    for rule in RULES:
        parts = [RULE_TEX[rule]]
        for j, coin in enumerate(COINS):
            if j:
                parts.append("")
            found = ranks.get((coin, rule))
            if found is None:
                parts += [TBD] * 4
            else:
                tau, _, flips, flattered, kept, of = found
                parts += [num(tau, ".2f"), rf"{flips}\%", rf"{flattered}\%", f"{kept}/{of}"]
        lines.append(" & ".join(parts) + r" \\")
    return "\n".join(lines) + "\n"


def tex_name(name: str) -> str:
    for greek, tex in (("γ", r"$\gamma$"), ("τ", r" $\tau$"), ("α", r" $\alpha$"), ("@", r"@\,")):
        name = name.replace(greek, tex)
    return name


def variant_rows(cells: dict, names: list[str]) -> str:
    lines = []
    for name in names:
        parts = [tex_name(name)]
        for j, coin in enumerate(COINS):
            if j:
                parts.append("")
            for rule in RULES:
                cell = cells.get((coin, name, rule))
                parts.append(TBD if cell is None else num(cell[1], ".0f"))
        lines.append(" & ".join(parts) + r" \\")
    return "\n".join(lines) + "\n"


# --- survival ---------------------------------------------------------------------


def read_survival(reports: Path) -> tuple[dict, dict]:
    """(coin, stage) -> (Cox test C, volume-ahead C, clock C); (coin, feature) -> hazard ratio."""
    test_c, single, hazard = {}, {}, {}
    stage_c = re.compile(r"^\s+(front|fill|total)\s+.*?([\d.]+) / ([\d.]+)$", re.M)
    stage_single = re.compile(r"^\s+(front|fill|total)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$", re.M)
    for path in sorted(reports.glob("survival*.txt")):
        for coin, title, body in blocks(path):
            if title.startswith("train hours"):
                for m in stage_c.finditer(body):
                    test_c[(coin, m.group(1))] = float(m.group(3))
            elif title.startswith("single features"):
                for m in stage_single.finditer(body):
                    single[(coin, m.group(1))] = (float(m.group(2)), float(m.group(4)))
            elif title.startswith("Cox hazard"):
                for m in re.finditer(r"^\s+(\w+)\s+([\d.]+)\s+\S+$", body, re.M):
                    if m.group(1) in FEATURE_TEX:
                        hazard[(coin, m.group(1))] = float(m.group(2))
    concord = {key: (c, *single.get(key, (None, None))) for key, c in test_c.items()}
    return concord, hazard


def survival_rows(concord: dict) -> str:
    lines = []
    for coin in COINS:
        for i, stage in enumerate(STAGE_TEX):
            values = concord.get((coin, stage), (None, None, None))
            cells = [TBD if v is None else f"{v:.3f}" for v in values]
            label = f"\\{coin}" if i == 0 else ""
            lines.append(" & ".join([label, STAGE_TEX[stage], *cells]) + r" \\")
        if coin != COINS[-1]:
            lines.append(r"\addlinespace")
    return "\n".join(lines) + "\n"


def hazard_rows(hazard: dict) -> str:
    lines = []
    for feature, name in FEATURE_TEX.items():
        cells = [f"{hazard[(c, feature)]:.3f}" if (c, feature) in hazard else TBD for c in COINS]
        lines.append(" & ".join([name, *cells]) + r" \\")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--reports", type=Path, default=config.REPO_ROOT / "docs" / "reports" / "20251201_day"
    )
    parser.add_argument("--out", type=Path, default=config.REPO_ROOT / "paper" / "tables")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cells, ranks, names = read_backtest(args.reports)
    concord, hazard = read_survival(args.reports)
    (args.out / "ranking_rows.tex").write_text(ranking_rows(ranks))
    (args.out / "variant_rows.tex").write_text(variant_rows(cells, names))
    (args.out / "survival_rows.tex").write_text(survival_rows(concord))
    (args.out / "hazard_rows.tex").write_text(hazard_rows(hazard))
    print(
        f"tables written to {args.out}: {len(names)} strategies, {len(cells)} backtest cells,"
        f" {len(ranks)} rankings, {len(concord)} survival stages, {len(hazard)} hazard ratios"
    )


if __name__ == "__main__":
    main()
