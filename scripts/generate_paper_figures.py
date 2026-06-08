"""Generate paper figures from frozen paper table CSV files.

The script intentionally reads only `results/paper_tables/*.csv`, so paper
figures remain tied to the evidence-freeze layer rather than scattered logs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "results" / "paper_tables"
FIGURE_DIR = ROOT / "results" / "paper_figures"


def savefig(name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        path = FIGURE_DIR / f"{name}.{suffix}"
        plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (9.5, 5.4),
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
        }
    )


def figure_01_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.axis("off")
    steps = [
        "Open-source\nPolymarket data",
        "Strict resolution\nand live features",
        "Wallet-level\nML ranking",
        "Fold ensemble\nor recency decay",
        "Consensus\ntrading signal",
        "Robustness and\nmicrostructure tests",
    ]
    x = np.linspace(0.08, 0.92, len(steps))
    y = 0.55
    for i, (xi, label) in enumerate(zip(x, steps)):
        ax.text(
            xi,
            y,
            label,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.45", fc="#F7F7F7", ec="#2F4858", lw=1.25),
            transform=ax.transAxes,
        )
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x[i + 1] - 0.065, y),
                xytext=(xi + 0.065, y),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#2F4858"),
            )
    ax.set_title("Research Pipeline: From Blockchain Traces to Tested Alpha Pockets", pad=18)
    savefig("figure_01_research_pipeline")


def figure_02_alpha_decay() -> None:
    df = pd.read_csv(TABLE_DIR / "table_03_threshold_alpha_decay.csv")
    ensemble = df[df["fold_policy"] == "ensemble"].copy()
    ensemble = ensemble.sort_values("threshold_pct")

    fig, ax1 = plt.subplots()
    x = np.arange(len(ensemble))
    labels = [f"{v:.2f}%" for v in ensemble["threshold_pct"]]

    bars = ax1.bar(x, ensemble["paper_pnl_usd"], color="#4C78A8", alpha=0.82, label="Paper PnL")
    ax1.axhline(0, color="#333333", lw=0.9)
    ax1.set_ylabel("Paper PnL (USD)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Top wallet threshold")

    ax2 = ax1.twinx()
    ax2.plot(x, ensemble["settlement_win_rate"] * 100, color="#F58518", marker="o", lw=2.2, label="Win rate")
    ax2.set_ylabel("Settlement win rate (%)")
    ax2.set_ylim(0, max(85, ensemble["settlement_win_rate"].max() * 115))
    ax2.spines["right"].set_visible(True)

    for bar in bars:
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:,.0f}",
            ha="center",
            va=va,
            fontsize=8,
        )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    ax1.set_title("Monotonic Alpha Decay Across Model-Ranked Wallet Thresholds")
    savefig("figure_02_alpha_decay")


def figure_03_snapshot_policy() -> None:
    df = pd.read_csv(TABLE_DIR / "table_04_snapshot_policy.csv")
    df = df[df["fold_policy"].isin(["all", "latest", "ensemble", "recency_decay"])].copy()
    order = ["all", "latest", "ensemble", "recency_decay"]
    df["fold_policy"] = pd.Categorical(df["fold_policy"], order, ordered=True)
    df = df.sort_values("fold_policy")

    fig, ax1 = plt.subplots()
    x = np.arange(len(df))
    ax1.bar(x, df["paper_position_pnl"], color="#72B7B2", alpha=0.85, label="Paper-position PnL")
    ax1.axhline(0, color="#333333", lw=0.9)
    ax1.set_ylabel("Paper-position PnL (USD)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["fold_policy"], rotation=0)

    ax2 = ax1.twinx()
    ax2.plot(x, df["position_win_rate"] * 100, color="#E45756", marker="o", lw=2.2, label="Win rate")
    ax2.set_ylabel("Position win rate (%)")
    ax2.set_ylim(0, 75)
    ax2.spines["right"].set_visible(True)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="lower right")
    ax1.set_title("Consensus Snapshots Improve Signal Quality vs Latest-Fold Deployment")
    savefig("figure_03_snapshot_policy")


def figure_04_semantic_filter() -> None:
    df = pd.read_csv(TABLE_DIR / "table_05_semantic_filter.csv")
    df = df[df["panel"] == "65_day_top_0_30"].copy()
    order = ["all", "exclude_crypto_finance", "other_ai", "other"]
    df["policy_or_filter"] = pd.Categorical(df["policy_or_filter"], order, ordered=True)
    df = df.sort_values("policy_or_filter")

    fig, ax1 = plt.subplots()
    x = np.arange(len(df))
    colors = ["#B279A2" if pnl >= 0 else "#D95F02" for pnl in df["paper_pnl_usd"]]
    ax1.bar(x, df["paper_pnl_usd"], color=colors, alpha=0.86, label="Paper PnL")
    ax1.axhline(0, color="#333333", lw=0.9)
    ax1.set_ylabel("Paper PnL (USD)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["all", "exclude\ncrypto/finance", "other + AI", "other"], rotation=0)

    ax2 = ax1.twinx()
    ax2.plot(x, df["win_rate"] * 100, color="#1B9E77", marker="o", lw=2.2, label="Win rate")
    ax2.set_ylabel("Win rate (%)")
    ax2.set_ylim(50, 75)
    ax2.spines["right"].set_visible(True)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    ax1.set_title("Semantic Filtering Turns Broad Noisy Signals into Positive Alpha Pockets")
    savefig("figure_04_semantic_filter")


def figure_05_size_aware_reversal() -> None:
    df = pd.read_csv(TABLE_DIR / "table_08_matched_event_study.csv")
    order = ["5m", "60m", "24h"]
    df["horizon"] = pd.Categorical(df["horizon"], order, ordered=True)
    df = df.sort_values(["matching_type", "horizon"])

    fig, ax = plt.subplots()
    width = 0.34
    x = np.arange(len(order))
    styles = [
        ("time_only_3_controls", -width / 2, "#4C78A8", "Time-only controls"),
        ("size_aware_3_controls", width / 2, "#F58518", "Size-aware controls"),
    ]
    for matching, offset, color, label in styles:
        sub = df[df["matching_type"] == matching].set_index("horizon").loc[order]
        y = sub["delta"].to_numpy()
        yerr = np.vstack([y - sub["ci_95_low"].to_numpy(), sub["ci_95_high"].to_numpy() - y])
        ax.bar(x + offset, y, width=width, color=color, alpha=0.82, label=label)
        ax.errorbar(x + offset, y, yerr=yerr, fmt="none", ecolor="#333333", elinewidth=1.0, capsize=3)
    ax.axhline(0, color="#333333", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Direction-adjusted delta")
    ax.set_xlabel("Post-trade horizon")
    ax.legend(loc="lower right")
    ax.set_title("Trade-Size Controls Reverse the Apparent Price-Impact Advantage")
    savefig("figure_05_size_aware_reversal")


def figure_06_information_flow_mechanism() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.3))
    ax.axis("off")

    nodes = [
        (0.08, 0.72, "Public wallet\ntrading traces", "#E8F1F2"),
        (0.28, 0.72, "Live feature\nconstruction", "#E8F1F2"),
        (0.48, 0.72, "ML wallet\nranking", "#E8F1F2"),
        (0.68, 0.72, "Fold-ensemble\nconsensus", "#E8F1F2"),
        (0.88, 0.72, "Market-level\nsignal", "#E8F1F2"),
        (0.28, 0.28, "Semantic\nconditioning", "#FFF3D6"),
        (0.48, 0.28, "Liquidity and\ntrade-size boundary", "#FCE6E0"),
        (0.68, 0.28, "Cap / stop-loss\nrisk translation", "#E7F4E4"),
        (0.88, 0.28, "Tested alpha\npocket", "#E7F4E4"),
    ]
    for x, y, label, color in nodes:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.5", fc=color, ec="#2F4858", lw=1.25),
            transform=ax.transAxes,
        )

    arrows = [
        ((0.15, 0.72), (0.21, 0.72)),
        ((0.35, 0.72), (0.41, 0.72)),
        ((0.55, 0.72), (0.61, 0.72)),
        ((0.75, 0.72), (0.81, 0.72)),
        ((0.28, 0.62), (0.28, 0.39)),
        ((0.48, 0.62), (0.48, 0.39)),
        ((0.68, 0.62), (0.68, 0.39)),
        ((0.35, 0.28), (0.41, 0.28)),
        ((0.55, 0.28), (0.61, 0.28)),
        ((0.75, 0.28), (0.81, 0.28)),
    ]
    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", lw=1.25, color="#2F4858"),
        )

    ax.text(
        0.5,
        0.06,
        "Core mechanism: alpha emerges from ranked information-flow consensus only after category conditioning, liquidity-aware interpretation, and risk translation.",
        ha="center",
        va="center",
        fontsize=10,
        transform=ax.transAxes,
    )
    ax.set_title("Information-Flow Consensus Mechanism", pad=14)
    savefig("figure_06_information_flow_mechanism")


def main() -> None:
    set_style()
    figure_01_pipeline()
    figure_02_alpha_decay()
    figure_03_snapshot_policy()
    figure_04_semantic_filter()
    figure_05_size_aware_reversal()
    figure_06_information_flow_mechanism()
    print(f"Wrote figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
