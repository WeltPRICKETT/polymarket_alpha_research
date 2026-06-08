#!/usr/bin/env python3
"""Focused diagnostics for live vs topic-diversification rich candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "results" / "new_data_sources" / "event_ablation_rich_paper_fixed_runs"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "new_data_sources"
BASELINE = "live"
CANDIDATE = "live_plus_topic_diversification"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _bootstrap_summary(values: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            "mean": None,
            "median": None,
            "p025": None,
            "p975": None,
            "prob_gt_zero": None,
            "two_sided_p_around_zero": None,
        }
    prob_gt_zero = float((clean > 0).mean())
    return {
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p025": float(clean.quantile(0.025)),
        "p975": float(clean.quantile(0.975)),
        "prob_gt_zero": prob_gt_zero,
        "two_sided_p_around_zero": float(2 * min(prob_gt_zero, 1 - prob_gt_zero)),
    }


def _top_wallets(predictions: pd.DataFrame, pct: float = 0.10) -> pd.DataFrame:
    rows = []
    for fold, group in predictions.groupby("fold"):
        ordered = group.sort_values("predicted_probability", ascending=False)
        n = max(1, int(np.ceil(len(ordered) * pct)))
        part = ordered.head(n).copy()
        part["selected_top_pct"] = pct
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _wallet_overlap(base_pred: pd.DataFrame, cand_pred: pd.DataFrame) -> pd.DataFrame:
    base_top = _top_wallets(base_pred)
    cand_top = _top_wallets(cand_pred)
    rows = []
    folds = sorted(set(base_top["fold"]).union(set(cand_top["fold"])))
    for fold in folds:
        b = set(base_top.loc[base_top["fold"] == fold, "address"])
        c = set(cand_top.loc[cand_top["fold"] == fold, "address"])
        union = b | c
        rows.append(
            {
                "fold": fold,
                "baseline_top_wallets": len(b),
                "candidate_top_wallets": len(c),
                "overlap_wallets": len(b & c),
                "jaccard_overlap": len(b & c) / len(union) if union else None,
            }
        )
    return pd.DataFrame(rows)


def _load_event_map() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / "sii_market_event_map.csv"
    event_map = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame({"market_id": event_map["market_id"].astype(str)})
    out["event_key"] = event_map["event_id"].astype("string").fillna(out["market_id"]).astype(str)
    out["event_title"] = event_map["event_title"].astype("string").fillna(out["market_id"]).astype(str)
    return out.drop_duplicates("market_id")


def _event_profit(ledger: pd.DataFrame, event_map: pd.DataFrame) -> pd.DataFrame:
    merged = ledger.copy()
    merged["market_id"] = merged["market_id"].astype(str)
    merged = merged.merge(event_map, on="market_id", how="left")
    merged["event_key"] = merged["event_key"].fillna(merged["market_id"]).astype(str)
    merged["event_title"] = merged["event_title"].fillna(merged["market_id"]).astype(str)
    grouped = (
        merged.groupby("event_key", dropna=False)
        .agg(
            event_title=("event_title", "first"),
            net_profit=("net_profit", "sum"),
            filled_notional=("filled_notional", "sum"),
            trades=("net_profit", "size"),
            markets=("market_id", "nunique"),
        )
        .reset_index()
    )
    return grouped.sort_values("net_profit", ascending=False)


def _event_overlap(base_ledger: pd.DataFrame, cand_ledger: pd.DataFrame, event_map: pd.DataFrame) -> dict[str, Any]:
    base_events = _event_profit(base_ledger, event_map)
    cand_events = _event_profit(cand_ledger, event_map)
    top_base = set(base_events.head(10)["event_key"])
    top_cand = set(cand_events.head(10)["event_key"])
    all_base = set(base_events["event_key"])
    all_cand = set(cand_events["event_key"])
    return {
        "all_traded_event_jaccard": len(all_base & all_cand) / len(all_base | all_cand)
        if (all_base | all_cand)
        else None,
        "top10_profit_event_jaccard": len(top_base & top_cand) / len(top_base | top_cand)
        if (top_base | top_cand)
        else None,
        "baseline_top_event": base_events.head(1).to_dict(orient="records")[0] if not base_events.empty else {},
        "candidate_top_event": cand_events.head(1).to_dict(orient="records")[0] if not cand_events.empty else {},
    }


def _fold_comparison(run_root: Path) -> pd.DataFrame:
    base_roll = pd.read_csv(run_root / BASELINE / "rolling_origin_results.csv")
    cand_roll = pd.read_csv(run_root / CANDIDATE / "rolling_origin_results.csv")
    base_wf = pd.read_csv(run_root / BASELINE / "model_informed_walk_forward_results.csv")
    cand_wf = pd.read_csv(run_root / CANDIDATE / "model_informed_walk_forward_results.csv")

    base_wf = base_wf[base_wf["strategy"] == "model_top_pct_0.10"].copy()
    cand_wf = cand_wf[cand_wf["strategy"] == "model_top_pct_0.10"].copy()
    roll_cols = ["fold", "auc_roc", "avg_precision", "precision_at_k"]
    wf_cols = ["fold", "total_net_profit", "equal_capital_roi", "unique_markets", "top_market_share"]
    base = base_roll[roll_cols].merge(base_wf[wf_cols], on="fold", how="left")
    cand = cand_roll[roll_cols].merge(cand_wf[wf_cols], on="fold", how="left")
    merged = base.merge(cand, on="fold", suffixes=("_live", "_topic"))
    for col in ["auc_roc", "avg_precision", "precision_at_k", "total_net_profit", "equal_capital_roi", "unique_markets", "top_market_share"]:
        merged[f"delta_{col}"] = merged[f"{col}_topic"] - merged[f"{col}_live"]
    return merged


def _write_markdown(path: Path, summary: dict[str, Any], fold: pd.DataFrame, wallet: pd.DataFrame) -> None:
    lines = [
        "# Focused Topic Diversification Diagnostics",
        "",
        "Comparison: `live` vs `live_plus_topic_diversification`.",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Fold Deltas",
            "",
            _markdown_table(
                fold[
                    [
                        "fold",
                        "delta_auc_roc",
                        "delta_avg_precision",
                        "delta_precision_at_k",
                        "delta_total_net_profit",
                        "delta_equal_capital_roi",
                        "delta_top_market_share",
                    ]
                ]
            ),
            "",
            "## Top Wallet Overlap",
            "",
            _markdown_table(wallet),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _markdown_table(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[col]) for col in df.columns) + " |")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fold = _fold_comparison(run_root)
    base_pred = pd.read_csv(run_root / BASELINE / "rolling_origin_predictions.csv")
    cand_pred = pd.read_csv(run_root / CANDIDATE / "rolling_origin_predictions.csv")
    wallet = _wallet_overlap(base_pred, cand_pred)

    base_ledger = pd.read_csv(run_root / BASELINE / "model_top_pct_0.10_walk_forward_trades.csv")
    cand_ledger = pd.read_csv(run_root / CANDIDATE / "model_top_pct_0.10_walk_forward_trades.csv")
    event_overlap = _event_overlap(base_ledger, cand_ledger, _load_event_map())

    base_boot = pd.read_csv(run_root / BASELINE / "fold_matched_random_bootstrap.csv")
    cand_boot = pd.read_csv(run_root / CANDIDATE / "fold_matched_random_bootstrap.csv")
    boot = pd.DataFrame(
        {
            "bootstrap_sample": cand_boot["bootstrap_sample"],
            "delta_roi_model_minus_random": cand_boot[
                "mean_equal_capital_roi_model_minus_random"
            ]
            - base_boot["mean_equal_capital_roi_model_minus_random"],
            "delta_profit_model_minus_random": cand_boot[
                "total_net_profit_model_minus_random"
            ]
            - base_boot["total_net_profit_model_minus_random"],
        }
    )

    base_summary = _read_json(run_root / BASELINE / "event_ablation_summary.json")
    cand_summary = _read_json(run_root / CANDIDATE / "event_ablation_summary.json")
    summary = {
        "baseline": BASELINE,
        "candidate": CANDIDATE,
        "folds": int(len(fold)),
        "mean_delta_auc": float(fold["delta_auc_roc"].mean()),
        "mean_delta_avg_precision": float(fold["delta_avg_precision"].mean()),
        "mean_delta_precision_at_k": float(fold["delta_precision_at_k"].mean()),
        "total_delta_net_profit": float(
            cand_summary["metrics"]["walk_forward_total_net_profit"]
            - base_summary["metrics"]["walk_forward_total_net_profit"]
        ),
        "delta_equal_capital_roi": float(
            cand_summary["metrics"]["walk_forward_mean_equal_capital_roi"]
            - base_summary["metrics"]["walk_forward_mean_equal_capital_roi"]
        ),
        "folds_profit_improved": int((fold["delta_total_net_profit"] > 0).sum()),
        "folds_ap_improved": int((fold["delta_avg_precision"] > 0).sum()),
        "mean_top_wallet_jaccard": float(wallet["jaccard_overlap"].mean()),
        **{f"event_{k}": v for k, v in event_overlap.items()},
        "bootstrap_delta_roi": _bootstrap_summary(boot["delta_roi_model_minus_random"]),
        "bootstrap_delta_profit": _bootstrap_summary(boot["delta_profit_model_minus_random"]),
    }

    fold_path = out_dir / "focused_topic_fold_diagnostics.csv"
    wallet_path = out_dir / "focused_topic_wallet_overlap.csv"
    boot_path = out_dir / "focused_topic_bootstrap_delta.csv"
    summary_path = out_dir / "focused_topic_diagnostics_summary.json"
    report_path = out_dir / "focused_topic_diagnostics_report.md"
    fold.to_csv(fold_path, index=False)
    wallet.to_csv(wallet_path, index=False)
    boot.to_csv(boot_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _write_markdown(report_path, summary, fold, wallet)

    print(f"Wrote {fold_path}")
    print(f"Wrote {wallet_path}")
    print(f"Wrote {boot_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")
    print(json.dumps(summary, indent=2, default=str))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
