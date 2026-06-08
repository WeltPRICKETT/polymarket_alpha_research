#!/usr/bin/env python3
"""Position-PnL diagnostic after collapsing repeated wallet clusters."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from tier2_sybil_pseudo_consensus import (
    GRID_DIR,
    OUT_DIR,
    annotate_clusters,
    build_pair_table,
    md_table,
    parse_wallets,
)


def parse_position_source(path: Path) -> tuple[str, str]:
    match = re.match(r"daily_position_pnl_(\d{8})(?:_(.+))?\.csv$", path.name)
    if not match:
        return "", "unknown"
    return match.group(1), match.group(2) or "base"


def load_positions() -> pd.DataFrame:
    frames = []
    for path in sorted(GRID_DIR.glob("daily_position_pnl_*.csv")):
        date, variant = parse_position_source(path)
        try:
            df = pd.read_csv(path)
        except EmptyDataError:
            continue
        if df.empty or "top_wallets" not in df.columns:
            continue
        df = df.copy()
        df["source_file"] = path.name
        df["signal_date"] = date
        df["policy_variant"] = variant
        df["signal_row"] = np.arange(len(df))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No daily_position_pnl files found in {GRID_DIR}")
    positions = pd.concat(frames, ignore_index=True)
    positions["wallet_list"] = positions["top_wallets"].map(parse_wallets)
    positions["wallet_count_parsed"] = positions["wallet_list"].map(len)
    positions = positions[positions["wallet_count_parsed"].ge(2)].copy()
    positions["signal_id"] = (
        positions["source_file"].astype(str)
        + "#"
        + positions["signal_row"].astype(str)
        + ":"
        + positions.get("market_id", "").astype(str)
        + ":"
        + positions.get("dominant_side", "").astype(str)
    )
    return positions


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def summarize_slice(name: str, df: pd.DataFrame) -> dict[str, float | int | str]:
    resolved = df[df.get("resolved", pd.Series(False, index=df.index)).eq(True)].copy()
    pnl = num(resolved.get("paper_position_pnl", pd.Series(dtype=float)))
    all_pnl = num(df.get("paper_position_pnl", pd.Series(dtype=float)))
    notional = num(df.get("paper_notional", pd.Series(dtype=float)))
    return {
        "slice": name,
        "positions": int(len(df)),
        "resolved_positions": int(len(resolved)),
        "unique_markets": int(df["market_id"].nunique()) if "market_id" in df else 0,
        "unique_events": int(df["event_id"].nunique()) if "event_id" in df else 0,
        "unique_wallets": int(len(set(w for wallets in df["wallet_list"] for w in wallets))) if len(df) else 0,
        "paper_position_pnl_resolved": float(pnl.sum(skipna=True)),
        "paper_position_pnl_all_non_na": float(all_pnl.sum(skipna=True)),
        "mean_resolved_pnl": float(pnl.mean(skipna=True)),
        "median_resolved_pnl": float(pnl.median(skipna=True)),
        "positive_resolved_share": float((pnl > 0).mean()) if len(pnl) else np.nan,
        "paper_notional_sum": float(notional.sum(skipna=True)),
        "mean_paper_notional": float(notional.mean(skipna=True)),
    }


def build_comparison(annotated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant_name, group in [("all_variants", annotated)] + [
        (str(k), v) for k, v in annotated.groupby("policy_variant")
    ]:
        raw = summarize_slice(f"{variant_name}: raw_wallet_consensus", group)
        retained_group = group[group["cluster_level_consensus_retained"]].copy()
        retained = summarize_slice(f"{variant_name}: cluster_aware_consensus", retained_group)
        flagged = summarize_slice(f"{variant_name}: pseudo_consensus_flagged", group[group["pseudo_consensus_flag"]])
        raw["retention_rate_vs_raw"] = 1.0
        raw["pnl_change_vs_raw"] = 0.0
        retained["retention_rate_vs_raw"] = retained["positions"] / raw["positions"] if raw["positions"] else np.nan
        retained["pnl_change_vs_raw"] = retained["paper_position_pnl_resolved"] - raw["paper_position_pnl_resolved"]
        flagged["retention_rate_vs_raw"] = flagged["positions"] / raw["positions"] if raw["positions"] else np.nan
        flagged["pnl_change_vs_raw"] = flagged["paper_position_pnl_resolved"] - raw["paper_position_pnl_resolved"]
        rows.extend([raw, retained, flagged])
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    positions = load_positions()
    pair_table = build_pair_table(positions)
    annotated = annotate_clusters(positions, pair_table)
    comparison = build_comparison(annotated)

    out_path = OUT_DIR / "table_23_cluster_aware_position_pnl.csv"
    report_path = OUT_DIR / "tier2_cluster_aware_position_pnl_report.md"
    comparison.to_csv(out_path, index=False)

    focus = comparison[
        comparison["slice"].str.contains("raw_wallet_consensus|cluster_aware_consensus", regex=True)
    ].copy()
    report = [
        "# Tier 2 Cluster-Aware Position-PnL Diagnostic",
        "",
        "This filters existing daily position-PnL ledgers by the repeated-wallet cluster rule. It is stronger than signal-level replay because it uses resolved paper-position PnL where available, but it still reuses existing ledgers rather than regenerating the trading monitor from scratch.",
        "",
        md_table(focus),
        "",
        "Interpretation: if cluster-aware filtering improves resolved position PnL, pseudo-consensus is an economically material execution risk rather than only a theoretical identification threat.",
    ]
    report_path.write_text("\n".join(report))
    print(md_table(focus))
    print(f"Wrote {out_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
