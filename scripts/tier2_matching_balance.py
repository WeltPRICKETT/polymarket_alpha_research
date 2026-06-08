#!/usr/bin/env python3
"""Build matching-quality diagnostics for matched event-study designs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "new_data_sources" / "full_data_step1_true_live"
OUT_DIR = ROOT / "research" / "strategy_track" / "gpu_results"

PAIR_FILES = {
    "time_only_3_controls_15m": BASE
    / "matched_event_multi_control_timeonly_15m"
    / "full_data_matched_event_study_pairs.csv",
    "size_aware_3_controls_15m": BASE
    / "matched_event_multi_control_15m"
    / "full_data_matched_event_study_pairs.csv",
    "size_aware_1_control_15m": BASE
    / "matched_event_sizeaware_single_15m"
    / "full_data_matched_event_study_pairs.csv",
}


def md_table(df: pd.DataFrame) -> str:
    tmp = df.copy()
    for col in tmp.columns:
        if pd.api.types.is_float_dtype(tmp[col]):
            tmp[col] = tmp[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    lines = ["| " + " | ".join(tmp.columns) + " |", "| " + " | ".join(["---"] * len(tmp.columns)) + " |"]
    for _, row in tmp.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in tmp.columns) + " |")
    return "\n".join(lines)


def smd(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    y = pd.to_numeric(y, errors="coerce").dropna()
    if x.empty or y.empty:
        return float("nan")
    pooled = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    return float((x.mean() - y.mean()) / pooled) if pooled and np.isfinite(pooled) else 0.0


def summarize_design(name: str, path: Path) -> dict[str, float | str | int]:
    df = pd.read_csv(path)
    if "horizon" in df.columns:
        df = df[df["horizon"].eq("5m")].copy()
    df = df.drop_duplicates(["pair_id", "rn"]) if "rn" in df.columns else df.drop_duplicates("pair_id")
    df["top_log_usd"] = np.log1p(pd.to_numeric(df["top_usd_amount"], errors="coerce"))
    df["control_log_usd"] = np.log1p(pd.to_numeric(df["control_usd_amount"], errors="coerce"))
    df["top_entry_price"] = pd.to_numeric(df["top_entry_price"], errors="coerce")
    df["control_entry_price"] = pd.to_numeric(df["control_entry_price"], errors="coerce")
    df["abs_time_diff"] = pd.to_numeric(df["abs_time_diff"], errors="coerce")
    df["abs_log_usd_diff"] = pd.to_numeric(df["abs_log_usd_diff"], errors="coerce")
    size_ratio = pd.to_numeric(df["control_usd_amount"], errors="coerce") / pd.to_numeric(
        df["top_usd_amount"], errors="coerce"
    ).replace(0, np.nan)
    return {
        "design": name,
        "matched_rows_5m": int(len(df)),
        "unique_top_trades": int(df["pair_id"].nunique()) if "pair_id" in df.columns else int(len(df)),
        "controls_per_top_mean": float(len(df) / df["pair_id"].nunique()) if "pair_id" in df.columns else 1.0,
        "same_event_share": float((df["top_event_id"] == df["control_event_id"]).mean())
        if {"top_event_id", "control_event_id"}.issubset(df.columns)
        else float("nan"),
        "mean_abs_time_diff_sec": float(df["abs_time_diff"].mean()),
        "median_abs_time_diff_sec": float(df["abs_time_diff"].median()),
        "p95_abs_time_diff_sec": float(df["abs_time_diff"].quantile(0.95)),
        "mean_abs_log_usd_diff": float(df["abs_log_usd_diff"].mean()),
        "median_abs_log_usd_diff": float(df["abs_log_usd_diff"].median()),
        "p95_abs_log_usd_diff": float(df["abs_log_usd_diff"].quantile(0.95)),
        "median_control_to_top_size_ratio": float(size_ratio.median()),
        "p10_control_to_top_size_ratio": float(size_ratio.quantile(0.10)),
        "p90_control_to_top_size_ratio": float(size_ratio.quantile(0.90)),
        "smd_log_usd": smd(df["top_log_usd"], df["control_log_usd"]),
        "smd_entry_price": smd(df["top_entry_price"], df["control_entry_price"]),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [summarize_design(name, path) for name, path in PAIR_FILES.items() if path.exists()]
    out = pd.DataFrame(rows)
    out_path = OUT_DIR / "table_20_matching_balance.csv"
    out.to_csv(out_path, index=False)
    report = [
        "# Tier 2 Matching Balance Diagnostics",
        "",
        "This table audits whether matched event-study controls are balanced on time, size, event identity, and entry price.",
        "",
        md_table(out),
        "",
        "Interpretation: size-aware designs should sharply reduce `abs_log_usd_diff` and `smd_log_usd`; time-only designs prioritize timestamp proximity.",
    ]
    (OUT_DIR / "tier2_matching_balance_report.md").write_text("\n".join(report))
    print(out.to_string(index=False))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
