#!/usr/bin/env python3
"""Build capacity-frontier and wallet-turnover diagnostics from GPU artifacts."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GPU_DIR = ROOT / "research" / "strategy_track" / "gpu_results"
PRED_PATH = ROOT / "results" / "new_data_sources" / "full_data_step1_true_live" / "full_data_step1_predictions.csv"
REGIME = "live_plus_static_event_shape"


def md_table(df: pd.DataFrame) -> str:
    tmp = df.copy()
    for col in tmp.columns:
        if pd.api.types.is_float_dtype(tmp[col]):
            tmp[col] = tmp[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    lines = ["| " + " | ".join(tmp.columns) + " |", "| " + " | ".join(["---"] * len(tmp.columns)) + " |"]
    for _, row in tmp.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in tmp.columns) + " |")
    return "\n".join(lines)


def build_capacity_frontier() -> pd.DataFrame:
    sweep = pd.read_csv(GPU_DIR / "step_a_full_universe_sweep.csv")
    sweep = sweep[sweep["regime"].eq(REGIME)].copy()
    keep = sweep[
        [
            "full_universe_threshold",
            "full_universe_threshold_pct",
            "wallet_cap_pct",
            "wallets_per_fold",
            "precision",
            "capped_profit",
            "positive_profit_folds",
            "profit_sharpe",
            "total_invested",
            "actual_capital_roi",
        ]
    ].copy()
    keep["profit_per_wallet_per_fold"] = keep["capped_profit"] / keep["wallets_per_fold"].replace(0, np.nan)
    keep["capacity_bucket"] = np.select(
        [
            keep["wallets_per_fold"] <= 150,
            keep["wallets_per_fold"].between(151, 400),
            keep["wallets_per_fold"] > 400,
        ],
        ["scarce_tail", "balanced_tail", "capacity_push"],
        default="unknown",
    )
    return keep.sort_values(["wallet_cap_pct", "full_universe_threshold"])


def build_wallet_turnover(threshold: float = 0.003) -> pd.DataFrame:
    pred = pd.read_csv(PRED_PATH, usecols=["regime", "fold", "address", "score"])
    pred = pred[pred["regime"].eq(REGIME)].copy()
    sets: dict[int, set[str]] = {}
    rows = []
    for fold, group in pred.groupby("fold"):
        group = group.sort_values("score", ascending=False)
        k = max(1, int(np.ceil(len(group) * threshold)))
        wallets = set(group.head(k)["address"].str.lower())
        sets[int(fold)] = wallets
        rows.append({"fold": int(fold), "selected_wallets": len(wallets), "threshold": threshold})
    for a, b in combinations(sorted(sets), 2):
        inter = len(sets[a] & sets[b])
        union = len(sets[a] | sets[b])
        rows.append(
            {
                "fold": f"{a}-{b}",
                "selected_wallets": np.nan,
                "threshold": threshold,
                "comparison": "pairwise_overlap",
                "fold_a": a,
                "fold_b": b,
                "intersection": inter,
                "union": union,
                "jaccard": inter / union if union else np.nan,
                "overlap_vs_smaller": inter / min(len(sets[a]), len(sets[b])) if min(len(sets[a]), len(sets[b])) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    GPU_DIR.mkdir(parents=True, exist_ok=True)
    cap = build_capacity_frontier()
    cap_path = GPU_DIR / "table_21_capacity_frontier.csv"
    cap.to_csv(cap_path, index=False)

    turnover = build_wallet_turnover()
    turnover_path = GPU_DIR / "table_19_wallet_turnover_decay.csv"
    turnover.to_csv(turnover_path, index=False)

    pairwise = turnover[turnover.get("comparison", pd.Series(dtype=str)).eq("pairwise_overlap")]
    report = [
        "# Tier 2 Capacity and Wallet-Turnover Diagnostics",
        "",
        "## Capacity Frontier: Top Rows by Sharpe",
        "",
        md_table(cap.sort_values("profit_sharpe", ascending=False).head(12)),
        "",
        "## Wallet Turnover",
        "",
        f"Pairwise mean Jaccard: {pairwise['jaccard'].mean():.4f}" if not pairwise.empty else "No pairwise rows.",
        f"Pairwise median Jaccard: {pairwise['jaccard'].median():.4f}" if not pairwise.empty else "",
        f"Pairwise mean overlap vs smaller set: {pairwise['overlap_vs_smaller'].mean():.4f}" if not pairwise.empty else "",
        "",
        "Interpretation: low fold-to-fold overlap supports the paper's dynamic-consensus framing rather than a permanent smart-wallet list.",
    ]
    (GPU_DIR / "tier2_capacity_turnover_report.md").write_text("\n".join(report))
    print(f"Wrote {cap_path}")
    print(f"Wrote {turnover_path}")
    print("\n".join(report))


if __name__ == "__main__":
    main()
