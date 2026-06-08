#!/usr/bin/env python3
"""Signal-level rerun of consensus after collapsing repeated wallet clusters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tier2_sybil_pseudo_consensus import (
    OUT_DIR,
    annotate_clusters,
    build_pair_table,
    load_signals,
    md_table,
)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def summarize_slice(name: str, df: pd.DataFrame) -> dict[str, float | int | str]:
    pnl = numeric(df.get("total_pnl", pd.Series(dtype=float)))
    delta = numeric(df.get("mean_delta", pd.Series(dtype=float)))
    notional = numeric(df.get("trigger_notional", pd.Series(dtype=float)))
    return {
        "slice": name,
        "signals": int(len(df)),
        "unique_markets": int(df["market_id"].nunique()) if "market_id" in df else 0,
        "unique_events": int(df["event_id"].nunique()) if "event_id" in df else 0,
        "unique_wallets": int(len(set(w for wallets in df["wallet_list"] for w in wallets))) if len(df) else 0,
        "total_pnl": float(pnl.sum(skipna=True)),
        "mean_pnl_per_signal": float(pnl.mean(skipna=True)),
        "median_pnl_per_signal": float(pnl.median(skipna=True)),
        "positive_signal_share": float((pnl > 0).mean()) if len(pnl) else np.nan,
        "mean_delta": float(delta.mean(skipna=True)),
        "median_delta": float(delta.median(skipna=True)),
        "trigger_notional_sum": float(notional.sum(skipna=True)),
        "mean_trigger_notional": float(notional.mean(skipna=True)),
    }


def build_policy_comparison(annotated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant_name, group in [("all_variants", annotated)] + [
        (str(k), v) for k, v in annotated.groupby("policy_variant")
    ]:
        raw = summarize_slice(f"{variant_name}: raw_wallet_consensus", group)
        retained_group = group[group["cluster_level_consensus_retained"]].copy()
        retained = summarize_slice(f"{variant_name}: cluster_aware_consensus", retained_group)
        flagged = summarize_slice(f"{variant_name}: pseudo_consensus_flagged", group[group["pseudo_consensus_flag"]])
        retained["retention_rate_vs_raw"] = retained["signals"] / raw["signals"] if raw["signals"] else np.nan
        retained["pnl_change_vs_raw"] = retained["total_pnl"] - raw["total_pnl"]
        retained["pnl_change_pct_of_abs_raw"] = retained["pnl_change_vs_raw"] / abs(raw["total_pnl"]) if raw["total_pnl"] else np.nan
        raw["retention_rate_vs_raw"] = 1.0
        raw["pnl_change_vs_raw"] = 0.0
        raw["pnl_change_pct_of_abs_raw"] = 0.0
        flagged["retention_rate_vs_raw"] = flagged["signals"] / raw["signals"] if raw["signals"] else np.nan
        flagged["pnl_change_vs_raw"] = flagged["total_pnl"] - raw["total_pnl"]
        flagged["pnl_change_pct_of_abs_raw"] = flagged["pnl_change_vs_raw"] / abs(raw["total_pnl"]) if raw["total_pnl"] else np.nan
        rows.extend([raw, retained, flagged])
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    signals = load_signals()
    pairs = build_pair_table(signals)
    annotated = annotate_clusters(signals, pairs)
    comparison = build_policy_comparison(annotated)

    out_path = OUT_DIR / "table_22_cluster_aware_consensus_signal_replay.csv"
    report_path = OUT_DIR / "tier2_cluster_aware_consensus_report.md"
    comparison.to_csv(out_path, index=False)

    focus = comparison[
        comparison["slice"].str.contains("raw_wallet_consensus|cluster_aware_consensus", regex=True)
    ].copy()
    report = [
        "# Tier 2 Cluster-Aware Consensus Signal Replay",
        "",
        "This is a signal-level replay, not a full execution simulator. It asks what happens if strategy signals require at least two repeated-pair wallet clusters rather than at least two wallet addresses.",
        "",
        "## Raw vs Cluster-Aware",
        "",
        md_table(focus),
        "",
        "## Interpretation",
        "",
        "- If cluster-aware consensus improves PnL while reducing signal count, pseudo-consensus was not merely a theoretical Sybil concern; it was an economically material noise source.",
        "- This table should be described as a diagnostic rerun. A full paper-trading rerun should rebuild position ledgers from the cluster-aware signal set.",
    ]
    report_path.write_text("\n".join(report))
    print(md_table(focus))
    print(f"Wrote {out_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
