#!/usr/bin/env python
"""Clustered bootstrap for matched event-study pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def cluster_bootstrap(pairs: pd.DataFrame, cluster_col: str, samples: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    pairs = pairs.dropna(subset=["top_signed_price_move", "control_signed_price_move"]).copy()
    pairs["delta"] = pairs["top_signed_price_move"] - pairs["control_signed_price_move"]
    summary_rows = []
    boot_rows = []
    for horizon, group in pairs.groupby("horizon", sort=True):
        clusters = group[cluster_col].dropna().unique()
        if len(clusters) == 0:
            continue
        cluster_groups = {cluster: data["delta"].to_numpy() for cluster, data in group.groupby(cluster_col)}
        boot_values = []
        for i in range(samples):
            sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
            values = np.concatenate([cluster_groups[c] for c in sampled_clusters])
            mean_delta = float(np.mean(values)) if len(values) else np.nan
            boot_values.append(mean_delta)
            boot_rows.append(
                {
                    "horizon": horizon,
                    "bootstrap_sample": i,
                    "cluster_col": cluster_col,
                    "mean_delta": mean_delta,
                }
            )
        arr = np.array(boot_values)
        summary_rows.append(
            {
                "horizon": horizon,
                "cluster_col": cluster_col,
                "pairs": int(len(group)),
                "clusters": int(len(clusters)),
                "mean_delta": float(group["delta"].mean()),
                "cluster_bootstrap_p025": float(np.nanquantile(arr, 0.025)),
                "cluster_bootstrap_p975": float(np.nanquantile(arr, 0.975)),
                "cluster_prob_delta_positive": float((arr > 0).mean()),
                "cluster_one_sided_p_delta_le_0": float((arr <= 0).mean()),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(boot_rows)


def render_report(summary: pd.DataFrame, source: str) -> str:
    def table(df: pd.DataFrame) -> str:
        df = df.copy()
        for col in df.columns:
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
        return "\n".join(lines)

    supported = summary[summary["cluster_bootstrap_p025"] > 0]
    lines = [
        "# Matched Event Study Cluster Bootstrap",
        "",
        "Generated: 2026-05-22",
        "",
        f"Source pairs: `{source}`",
        "",
        table(summary),
        "",
        "## Interpretation",
        "",
    ]
    if supported.empty:
        lines.append("- No horizon has a market-clustered bootstrap interval fully above zero.")
    else:
        lines.append("- The following horizons survive market-clustered bootstrap with the 2.5% bound above zero: " + ", ".join(supported["horizon"].astype(str)) + ".")
    lines.append("- Market clustering is stricter than row-level bootstrap because repeated trades in the same market no longer count as fully independent evidence.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cluster-col", default="market_id")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(args.pairs)
    summary, boot = cluster_bootstrap(pairs, args.cluster_col, args.samples, args.seed)
    summary.to_csv(out_dir / "matched_event_cluster_bootstrap_summary.csv", index=False)
    boot.to_csv(out_dir / "matched_event_cluster_bootstrap_samples.csv", index=False)
    (out_dir / "matched_event_cluster_bootstrap_report.md").write_text(render_report(summary, args.pairs))
    print(summary.to_string(index=False))
    print(f"Wrote cluster bootstrap outputs to {out_dir}")


if __name__ == "__main__":
    main()
