#!/usr/bin/env python3
"""Stress-test consensus signals against repeated-wallet pseudo-consensus.

This is not identity attribution. It is a conservative robustness audit: if
two or more wallets repeatedly co-appear in model-triggered signals, collapse
that repeated co-appearance into one heuristic cluster and ask whether the
original multi-wallet signal remains multi-cluster.
"""

from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


ROOT = Path(__file__).resolve().parents[1]
GRID_DIR = ROOT / "research" / "strategy_track" / "paper_trading_policy_grid"
OUT_DIR = ROOT / "research" / "strategy_track" / "gpu_results"

PAIR_REPEAT_THRESHOLD = 3


def md_table(df: pd.DataFrame) -> str:
    tmp = df.copy()
    for col in tmp.columns:
        if pd.api.types.is_float_dtype(tmp[col]):
            tmp[col] = tmp[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    lines = ["| " + " | ".join(tmp.columns) + " |", "| " + " | ".join(["---"] * len(tmp.columns)) + " |"]
    for _, row in tmp.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in tmp.columns) + " |")
    return "\n".join(lines)


def parse_source(path: Path) -> tuple[str, str]:
    match = re.match(r"daily_signal_ledger_(\d{8})(?:_(.+))?\.csv$", path.name)
    if not match:
        return "", "unknown"
    date = match.group(1)
    variant = match.group(2) or "base"
    return date, variant


def parse_wallets(value: object) -> list[str]:
    if pd.isna(value):
        return []
    wallets = []
    for part in str(value).replace(";", ",").split(","):
        part = part.strip().lower()
        if part.startswith("0x") and len(part) >= 10:
            wallets.append(part)
    return sorted(set(wallets))


class DSU:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def load_signals() -> pd.DataFrame:
    frames = []
    for path in sorted(GRID_DIR.glob("daily_signal_ledger_*.csv")):
        date, variant = parse_source(path)
        try:
            df = pd.read_csv(path)
        except EmptyDataError:
            continue
        if "top_wallets" not in df.columns:
            continue
        if df.empty:
            continue
        df = df.copy()
        df["source_file"] = path.name
        df["signal_date"] = date
        df["policy_variant"] = variant
        df["signal_row"] = np.arange(len(df))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No daily_signal_ledger files found in {GRID_DIR}")
    signals = pd.concat(frames, ignore_index=True)
    signals["wallet_list"] = signals["top_wallets"].map(parse_wallets)
    signals["wallet_count_parsed"] = signals["wallet_list"].map(len)
    signals = signals[signals["wallet_count_parsed"].ge(2)].copy()
    signals["signal_id"] = (
        signals["source_file"].astype(str)
        + "#"
        + signals["signal_row"].astype(str)
        + ":"
        + signals.get("market_id", "").astype(str)
        + ":"
        + signals.get("dominant_side", "").astype(str)
    )
    return signals


def build_pair_table(signals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in signals.iterrows():
        wallets = row["wallet_list"]
        for a, b in combinations(wallets, 2):
            rows.append(
                {
                    "wallet_a": a,
                    "wallet_b": b,
                    "signal_id": row["signal_id"],
                    "policy_variant": row["policy_variant"],
                    "market_id": row.get("market_id"),
                    "event_id": row.get("event_id"),
                    "signal_date": row.get("signal_date"),
                }
            )
    pair_rows = pd.DataFrame(rows)
    if pair_rows.empty:
        return pair_rows
    return (
        pair_rows.groupby(["wallet_a", "wallet_b"], dropna=False)
        .agg(
            pair_signal_count=("signal_id", "nunique"),
            unique_markets=("market_id", "nunique"),
            unique_events=("event_id", "nunique"),
            unique_dates=("signal_date", "nunique"),
            policy_variants=("policy_variant", lambda x: ",".join(sorted(set(map(str, x))))),
        )
        .reset_index()
        .sort_values(["pair_signal_count", "unique_events", "unique_markets"], ascending=False)
    )


def annotate_clusters(signals: pd.DataFrame, pair_table: pd.DataFrame) -> pd.DataFrame:
    dsu = DSU()
    risky_pairs = pair_table[pair_table["pair_signal_count"].ge(PAIR_REPEAT_THRESHOLD)]
    for _, row in risky_pairs.iterrows():
        dsu.union(row["wallet_a"], row["wallet_b"])

    pair_count_lookup = {
        tuple(sorted((row.wallet_a, row.wallet_b))): int(row.pair_signal_count)
        for row in pair_table.itertuples(index=False)
    }

    annotated = signals.copy()
    cluster_counts = []
    max_pair_repeats = []
    risky_pair_counts = []
    hub_wallet_shares = []
    cluster_ids_col = []
    for wallets in annotated["wallet_list"]:
        clusters = [dsu.find(w) for w in wallets]
        cluster_counts.append(len(set(clusters)))
        cluster_ids_col.append(",".join(sorted(set(clusters))))
        counts = []
        risky = 0
        for a, b in combinations(wallets, 2):
            c = pair_count_lookup.get(tuple(sorted((a, b))), 0)
            counts.append(c)
            if c >= PAIR_REPEAT_THRESHOLD:
                risky += 1
        max_pair_repeats.append(max(counts) if counts else 0)
        risky_pair_counts.append(risky)
        per_cluster = pd.Series(clusters).value_counts(normalize=True)
        hub_wallet_shares.append(float(per_cluster.max()) if not per_cluster.empty else np.nan)

    annotated["cluster_count_after_repeat_collapse"] = cluster_counts
    annotated["cluster_ids_after_repeat_collapse"] = cluster_ids_col
    annotated["max_pair_repeat_count"] = max_pair_repeats
    annotated["risky_repeated_pair_count"] = risky_pair_counts
    annotated["largest_cluster_wallet_share"] = hub_wallet_shares
    annotated["pseudo_consensus_flag"] = annotated["cluster_count_after_repeat_collapse"].lt(2)
    annotated["cluster_level_consensus_retained"] = annotated["cluster_count_after_repeat_collapse"].ge(2)
    return annotated


def summarize(annotated: pd.DataFrame, pair_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("all_variants", annotated)]
    groups.extend((str(k), v) for k, v in annotated.groupby("policy_variant"))
    for name, group in groups:
        pnl = pd.to_numeric(group.get("total_pnl", pd.Series(dtype=float)), errors="coerce")
        mean_delta = pd.to_numeric(group.get("mean_delta", pd.Series(dtype=float)), errors="coerce")
        rows.append(
            {
                "policy_variant": name,
                "signals": len(group),
                "unique_markets": group["market_id"].nunique() if "market_id" in group else np.nan,
                "unique_events": group["event_id"].nunique() if "event_id" in group else np.nan,
                "unique_wallets": len(set(w for wallets in group["wallet_list"] for w in wallets)),
                "mean_wallets_per_signal": group["wallet_count_parsed"].mean(),
                "median_wallets_per_signal": group["wallet_count_parsed"].median(),
                "cluster_consensus_retention": group["cluster_level_consensus_retained"].mean(),
                "pseudo_consensus_flag_share": group["pseudo_consensus_flag"].mean(),
                "mean_max_pair_repeat": group["max_pair_repeat_count"].mean(),
                "p95_max_pair_repeat": group["max_pair_repeat_count"].quantile(0.95),
                "total_pnl_all_signals": pnl.sum(skipna=True),
                "total_pnl_cluster_retained": pnl[group["cluster_level_consensus_retained"]].sum(skipna=True),
                "total_pnl_pseudo_flagged": pnl[group["pseudo_consensus_flag"]].sum(skipna=True),
                "mean_delta_all_signals": mean_delta.mean(skipna=True),
                "mean_delta_cluster_retained": mean_delta[group["cluster_level_consensus_retained"]].mean(skipna=True),
            }
        )
    out = pd.DataFrame(rows)
    out["retained_pnl_share"] = out["total_pnl_cluster_retained"] / out["total_pnl_all_signals"].replace(0, np.nan)
    out["pseudo_flagged_pnl_share"] = out["total_pnl_pseudo_flagged"] / out["total_pnl_all_signals"].replace(0, np.nan)
    return out


def wallet_hub_table(signals: pd.DataFrame) -> pd.DataFrame:
    appearances: defaultdict[str, list[str]] = defaultdict(list)
    for _, row in signals.iterrows():
        for wallet in row["wallet_list"]:
            appearances[wallet].append(row["signal_id"])
    rows = [
        {"wallet": wallet, "signal_appearances": len(ids), "unique_signal_appearances": len(set(ids))}
        for wallet, ids in appearances.items()
    ]
    return pd.DataFrame(rows).sort_values("signal_appearances", ascending=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    signals = load_signals()
    pair_table = build_pair_table(signals)
    annotated = annotate_clusters(signals, pair_table)
    summary = summarize(annotated, pair_table)
    hubs = wallet_hub_table(signals)

    summary_path = OUT_DIR / "table_16_pseudo_consensus_stress.csv"
    pair_path = OUT_DIR / "table_16_pseudo_consensus_repeated_pairs.csv"
    signal_path = OUT_DIR / "table_16_pseudo_consensus_signal_level.csv"
    hub_path = OUT_DIR / "table_16_pseudo_consensus_wallet_hubs.csv"
    report_path = OUT_DIR / "tier2_sybil_pseudo_consensus_report.md"

    summary.to_csv(summary_path, index=False)
    pair_table.head(500).to_csv(pair_path, index=False)
    annotated.drop(columns=["wallet_list"]).to_csv(signal_path, index=False)
    hubs.head(500).to_csv(hub_path, index=False)

    high_repeat_pairs = pair_table[pair_table["pair_signal_count"].ge(PAIR_REPEAT_THRESHOLD)]
    report = [
        "# Tier 2 Pseudo-Consensus / Sybil Stress Test",
        "",
        "This audit does not identify real-world ownership. It asks whether strategy consensus survives after repeatedly co-appearing wallet pairs are collapsed into heuristic clusters.",
        "",
        f"Pair repeat threshold: >= {PAIR_REPEAT_THRESHOLD} signal co-appearances.",
        f"Signal ledgers analyzed: {signals['source_file'].nunique()} files.",
        f"Signals with at least two parsed wallets: {len(signals)}.",
        f"Repeated wallet pairs above threshold: {len(high_repeat_pairs)}.",
        "",
        "## Summary",
        "",
        md_table(summary),
        "",
        "## Top Repeated Pairs",
        "",
        md_table(pair_table.head(15)) if not pair_table.empty else "No repeated pairs.",
        "",
        "## Top Wallet Hubs",
        "",
        md_table(hubs.head(15)) if not hubs.empty else "No wallet appearances.",
        "",
        "Interpretation: a low cluster-consensus retention rate would mean apparent multi-wallet consensus is fragile to pseudo-consensus risk. A high retention rate supports the claim that consensus is not mechanically driven by the same repeated wallet pair.",
    ]
    report_path.write_text("\n".join(report))
    print(md_table(summary))
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
