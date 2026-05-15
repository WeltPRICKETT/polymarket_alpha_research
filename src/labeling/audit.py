"""
Author: AI Assistant
Date: 2026-05-12
Description: Label audit exports for reproducible academic review.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
LABEL_AUDIT_PATH = RESULTS_DIR / "label_audit.csv"
LABEL_SUMMARY_PATH = RESULTS_DIR / "label_summary.json"

COMPOSITE_BASIS_FEATURES = ["total_roi", "win_rate", "profit_loss_ratio"]
RESOLUTION_BASIS_FIELDS = ["n_resolved", "n_correct", "accuracy"]
FUTURE_RETURN_BASIS_FIELDS = [
    "feature_cutoff",
    "label_window_end",
    "future_n_resolved_trades",
    "future_net_profit",
    "future_roi",
    "future_return_threshold",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _counts(series: pd.Series) -> Dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).to_dict().items()}


def _counts_by_split(df: pd.DataFrame, col: str) -> Dict[str, int]:
    if "split" not in df.columns or col not in df.columns:
        return {}
    grouped = df.groupby("split")[col].value_counts(dropna=False)
    return {f"{split}:{value}": int(count) for (split, value), count in grouped.to_dict().items()}


def _single_class_splits(df: pd.DataFrame) -> Iterable[str]:
    if "split" not in df.columns or "Trader_Success_Rate" not in df.columns:
        return []
    bad = []
    for split, subset in df.groupby("split"):
        if subset["Trader_Success_Rate"].nunique(dropna=False) < 2:
            bad.append(str(split))
    return bad


def _label_independence(label_strategy: str) -> str:
    if label_strategy == "resolution":
        return "independent_resolution"
    if label_strategy == "future_return":
        return "independent_future_return"
    if label_strategy == "composite":
        return "feature_derived_exploratory"
    return "mixed_or_unknown"


def build_label_audit(df: pd.DataFrame, label_strategy: str, requested_label_mode: str) -> pd.DataFrame:
    """Return a row-level label audit table."""
    audit = pd.DataFrame({
        "address": df["address"],
        "split": df.get("split", pd.Series(["unknown"] * len(df), index=df.index)),
        "label": df["Trader_Success_Rate"],
        "label_source": df.get("label_source", pd.Series(["unknown"] * len(df), index=df.index)),
        "label_strategy": label_strategy,
        "requested_label_mode": requested_label_mode,
        "label_independence": _label_independence(label_strategy),
    })

    for col in ["composite_score", *RESOLUTION_BASIS_FIELDS, *FUTURE_RETURN_BASIS_FIELDS]:
        if col in df.columns:
            audit[col] = df[col]

    audit["label_basis_features"] = ""
    composite_mask = audit["label_source"].astype(str).str.startswith("composite")
    audit.loc[composite_mask, "label_basis_features"] = ",".join(COMPOSITE_BASIS_FEATURES)
    resolution_mask = audit["label_source"].astype(str).str.startswith("resolution")
    audit.loc[resolution_mask, "label_basis_features"] = ",".join(RESOLUTION_BASIS_FIELDS)
    future_mask = audit["label_source"].astype(str).str.startswith("future_return")
    audit.loc[future_mask, "label_basis_features"] = ",".join(FUTURE_RETURN_BASIS_FIELDS)

    return audit


def build_label_summary(df: pd.DataFrame, label_strategy: str, requested_label_mode: str) -> Dict:
    """Return summary metrics for label coverage and split health."""
    single_class = list(_single_class_splits(df))
    summary = {
        "generated_at": _utc_now(),
        "requested_label_mode": requested_label_mode,
        "label_strategy": label_strategy,
        "label_independence": _label_independence(label_strategy),
        "row_count": int(len(df)),
        "label_counts": _counts(df["Trader_Success_Rate"]) if "Trader_Success_Rate" in df else {},
        "label_counts_by_split": _counts_by_split(df, "Trader_Success_Rate"),
        "label_source_counts": _counts(df["label_source"]) if "label_source" in df else {},
        "single_class_splits": single_class,
        "is_trainable_for_binary_eval": len(single_class) == 0,
    }
    if label_strategy == "composite":
        summary["warning"] = (
            "Composite labels are feature-derived exploratory labels. "
            "Use resolution labels for primary academic claims when coverage is sufficient."
        )
        summary["composite_basis_features"] = COMPOSITE_BASIS_FEATURES
    if label_strategy == "future_return":
        summary["future_return_basis_fields"] = FUTURE_RETURN_BASIS_FIELDS
        if "future_return_threshold" in df.columns and not df.empty:
            summary["future_return_threshold"] = float(df["future_return_threshold"].iloc[0])
        if "future_n_resolved_trades" in df.columns:
            summary["future_label_coverage"] = {
                "eligible_rows": int((df["label_source"] == "future_return").sum()),
                "mean_future_resolved_trades": float(df["future_n_resolved_trades"].mean()),
            }
    return summary


def write_label_audit(df: pd.DataFrame, label_strategy: str, requested_label_mode: str) -> Dict:
    """Write row-level and summary label audit artifacts."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    audit_df = build_label_audit(df, label_strategy, requested_label_mode)
    summary = build_label_summary(df, label_strategy, requested_label_mode)

    audit_df.to_csv(LABEL_AUDIT_PATH, index=False)
    LABEL_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(f"Wrote label audit → {LABEL_AUDIT_PATH}")
    logger.info(f"Wrote label summary → {LABEL_SUMMARY_PATH}")
    if summary["single_class_splits"]:
        logger.warning(f"Single-class label splits detected: {summary['single_class_splits']}")
    return {"audit_path": str(LABEL_AUDIT_PATH), "summary": summary}
