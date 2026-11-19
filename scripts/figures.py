"""Figures of the paper, written to paper/figures/ as PDF.

Numbers that a report script already prints are read from its reference output in
docs/reports/<date>_<hour>/, so a figure and the table beside it can never
disagree; the rest (order lifetimes, the clock calibration) is computed from the
same hour through the same pipeline.

Usage: python scripts/figures.py [--date 20251201] [--hour 12] [--out paper/figures]

Colours: the categorical slots of a palette validated for colour-vision
deficiency, assigned in fixed order; the observed outcome is drawn in ink, not in
a series colour. Each series also has its own marker, so the figures survive
greyscale print.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from paper_tables import read_backtest  # noqa: E402

from hypermm import config  # noqa: E402
from hypermm.fillsim import models  # noqa: E402
from hypermm.pipeline import WARM_UP, load_hour, read_changes, read_fills  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
COIN_COLOUR = {"BTC": SLOT[0], "ETH": SLOT[1], "SOL": SLOT[2]}  # the colour follows the coin
FULL, COLUMN = 6.85, 3.3  # inches: text width and column width of the paper

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)


# --- reading the reference reports ----------------------------------------------


def coin_sections(text: str) -> dict[tuple[str, str], str]:
    """Split a report into its sections, keyed by (coin, title)."""
    parts = re.split(r"^={78}\n(\w+): (.+)\n={78}\n", text, flags=re.M)
    return {(parts[i], parts[i + 1]): parts[i + 2] for i in range(1, len(parts) - 2, 3)}


def deciles(report: Path) -> dict[str, np.ndarray]:
    """Fill rate by decile of volume ahead, % : columns real, touch, volume, queue, clock."""
    out = {}
    for (coin, title), body in coin_sections(report.read_text()).items():
        if title != "each rule against the real outcome":
            continue
        rows = re.findall(
            r"^\s+(\d)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$", body, re.M
        )
        out[coin] = np.array(rows, dtype=float)
    return out


def markouts(report: Path) -> dict[str, list[tuple]]:
    """(seconds, filled mean, filled s.e., saved mean, saved s.e.) per horizon."""
    out = {}
    number = r"([+-]?[\d.]+)"
    for (coin, title), body in coin_sections(report.read_text()).items():
        if title != "what the saved fills would have been worth":
            continue
        rows = re.findall(rf"after\s+(\d+) s\s+{number} ± {number}\s+{number} ± {number}", body)
        out[coin] = [tuple(float(v) for v in row) for row in rows]
    return out


# --- figures ----------------------------------------------------------------------


def label_end(ax, x, y, text, color, dy=0.0):
    ax.annotate(
        text, (x, y), xytext=(4, dy), textcoords="offset points", va="center", fontsize=7, color=INK
    )


def fig_deciles(data: dict[str, np.ndarray], out: Path) -> None:
    """Fill rate by queue position: the real one and what each rule says."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4), sharey=True)
    series = [
        (1, "Observed", INK, "o", "-", 1.8),
        (2, "Touch", SLOT[0], "^", "-", 1.2),
        (3, "Volume-through", SLOT[1], "v", "-", 1.2),
        (4, "Queue-aware", SLOT[2], "s", "--", 1.2),
        (5, "Exp. clock", SLOT[3], "D", "-", 1.2),
    ]
    for ax, coin in zip(axes, ("SOL", "BTC"), strict=True):
        rows = data[coin]
        for col, name, color, marker, style, width in series:
            ax.plot(
                rows[:, 0],
                rows[:, col],
                color=color,
                marker=marker,
                ls=style,
                lw=width,
                mfc="white" if name == "Queue-aware" else color,
                label=name,
                zorder=3 if col == 1 else 2,
            )
        ax.set_yscale("log")
        ax.set_title(coin, loc="left", fontweight="bold")
        ax.set_xlabel("Decile of volume ahead at placement (0 = shortest queue)")
        ax.set_xticks(range(10))
    axes[0].set_ylabel("Orders filled (%)")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fill_by_decile.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_markout(data: dict[str, list[tuple]], out: Path) -> None:
    """Markout of real fills and of fills rescued by in-block cancellation."""
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 2.0), sharey=False)
    for ax, coin in zip(axes, ("SOL", "BTC"), strict=True):
        rows = data[coin]
        x = np.arange(len(rows))
        for shift, (name, mean_col, se_col, color, marker) in zip(
            (-0.12, 0.12),
            (("Filled", 1, 2, SLOT[0], "o"), ("Rescued", 3, 4, SLOT[1], "s")),
            strict=True,
        ):
            mean = np.array([r[mean_col] for r in rows])
            se = np.array([r[se_col] for r in rows])
            ax.errorbar(
                x + shift,
                mean,
                yerr=1.96 * se,
                fmt=marker,
                color=color,
                ms=4,
                elinewidth=1.0,
                capsize=0,
                label=name,
            )
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xticks(x, [f"{int(r[0])} s" for r in rows])
        ax.set_xlim(-0.5, len(rows) - 0.5)
        ax.set_title(coin, loc="left", fontweight="bold")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Markout (bp), 95% CI")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "markout.pdf", bbox_inches="tight")
    plt.close(fig)


def survival(durations: np.ndarray, grid: np.ndarray) -> np.ndarray:
    ordered = np.sort(durations)
    return 1 - np.searchsorted(ordered, grid, side="right") / len(ordered)


def fig_lifetimes(tables: dict[str, pl.DataFrame], out: Path) -> None:
    """How long orders live before their first fill and before cancellation."""
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 2.0), sharey=True)
    grid = np.logspace(-3, 3, 400)  # seconds
    for ax, coin in zip(axes, ("SOL", "BTC"), strict=True):
        table = tables[coin]
        seconds = (table["duration"] / models.NS).to_numpy()
        filled = table["filled"].to_numpy()
        cancelled = (table["outcome"] == "cancelled").to_numpy() & ~filled
        for name, mask, color, style in (
            ("Time to first fill", filled, SLOT[0], "-"),
            ("Time to cancellation", cancelled, SLOT[1], "--"),
        ):
            ax.plot(grid, survival(seconds[mask], grid), color=color, ls=style, label=name)
            median = np.median(seconds[mask])
            ax.plot([median], [0.5], marker="o" if style == "-" else "s", color=color, ms=3.5)
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1e3)
        ax.set_xticks([1e-3, 1e-1, 1e1, 1e3])
        ax.set_title(coin, loc="left", fontweight="bold")
        ax.set_xlabel("Seconds since placement")
    axes[0].set_ylabel("Share still resting")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "lifetimes.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_clock(hours: dict, out: Path) -> None:
    """How often trading reaches δ ticks from the mid, against the exponential fits."""
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 2.0))
    for ax, coin in zip(axes, ("SOL", "BTC"), strict=True):
        top, trades = hours[coin]
        seconds = (top["ts"].max() - top["ts"].min()) / models.NS
        depths = models.reach(top, trades)
        grid = 0.5 + np.arange(int(min(depths.max(), 40)) + 1)
        observed = np.array([(depths >= d - 1e-9).sum() / 2 for d in grid])
        keep = observed > 0
        ax.plot(grid[keep], observed[keep], "o", color=INK, ms=3, label="Observed", zorder=3)
        for (name, weighted), color, style in zip(
            (("Weighted fit", True), ("Unweighted fit", False)), SLOT[:2], ("-", "--"), strict=True
        ):
            a, k = models.clock_fit(top, trades, weighted=weighted)
            ax.plot(grid, a * np.exp(-k * grid) * seconds, color=color, ls=style, label=name)
        ax.set_yscale("log")
        ax.set_title(coin, loc="left", fontweight="bold")
        ax.set_xlabel(r"Distance from mid $\delta$ (ticks)")
    axes[0].set_ylabel("Times reached per side per hour")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "clock_fit.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_pnl_by_rule(cells: dict, out: Path) -> None:
    """Each variant's PnL under a rule against its PnL under the reference rule."""
    coins = sorted({coin for coin, _, _ in cells}, key=("BTC", "ETH", "SOL").index)
    rules = [
        ("touch", "Touch"),
        ("volume-through", "Volume-through"),
        ("queue-aware", "Queue-aware, no block rule"),
        ("clock", "Exp. clock"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 2.1), sharex=True, sharey=True)
    markers = {"BTC": "o", "ETH": "s", "SOL": "^"}
    limits = [0.0, 0.0]
    for ax, (rule, title) in zip(axes, rules, strict=True):
        for coin in coins:
            names = sorted({name for c, name, r in cells if c == coin and r == rule})
            x = np.array([cells[(coin, n, "queue-aware, block")][1] for n in names]) / 1e3
            y = np.array([cells[(coin, n, rule)][1] for n in names]) / 1e3
            ax.scatter(
                x,
                y,
                s=9,
                color=COIN_COLOUR[coin],
                marker=markers[coin],
                label=coin,
                linewidths=0,
                alpha=0.85,
                zorder=3,
            )
            limits = [min(limits[0], x.min(), y.min()), max(limits[1], x.max(), y.max())]
        ax.set_title(title, loc="left", fontweight="bold")
        ax.axhline(0, color=GRID, lw=0.8, zorder=1)
        ax.axvline(0, color=GRID, lw=0.8, zorder=1)
    pad = 0.05 * (limits[1] - limits[0])
    for ax in axes:
        ax.plot(
            [limits[0] - pad, limits[1] + pad],
            [limits[0] - pad, limits[1] + pad],
            color=MUTED,
            lw=0.8,
            ls="--",
            zorder=2,
        )
        ax.set_xlim(limits[0] - pad, limits[1] + pad)
        ax.set_ylim(limits[0] - pad, limits[1] + pad)
        ax.set_xlabel("PnL, reference rule ($k)")
    axes[0].set_ylabel("PnL under the rule ($k)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(coins), bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "pnl_by_rule.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--raw", type=Path, default=config.DATA_RAW_DIR)
    parser.add_argument("--date", default="20251201")
    parser.add_argument("--hour", type=int, default=12)
    parser.add_argument("--out", type=Path, default=config.REPO_ROOT / "paper" / "figures")
    parser.add_argument(
        "--day",
        type=Path,
        default=config.REPO_ROOT / "docs" / "reports" / "20251201_day",
        help="reports over the whole day (backtest)",
    )
    parser.add_argument(
        "--only-day", action="store_true", help="redraw only the figures of the day reports"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reports = config.REPO_ROOT / "docs" / "reports" / f"{args.date}_{args.hour:02d}"

    cells, _, _ = read_backtest(args.day)
    if cells:
        fig_pnl_by_rule(cells, args.out)
    if args.only_day:
        print(f"figures written to {args.out}")
        return

    fig_deciles(deciles(reports / "fill_models.txt"), args.out)
    fig_markout(markouts(reports / "block_protection.txt"), args.out)

    changes = read_changes(args.raw, args.date, args.hour)
    fills = read_fills(args.raw, args.date, args.hour)
    tables, hours = {}, {}
    for coin in ("SOL", "BTC"):
        hour = load_hour(args.raw, args.date, args.hour, coin, changes=changes, fills=fills)
        tables[coin] = hour.table.filter(pl.col("tPlaced") >= hour.top["ts"].min() + WARM_UP)
        hours[coin] = (hour.top, models.level_trades(hour.fills))
    fig_lifetimes(tables, args.out)
    fig_clock(hours, args.out)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
