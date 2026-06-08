#!/usr/bin/env python
"""Price-impact smoke test for full-data top-model wallets.

This is intentionally a bounded diagnostic, not a final causal event study. It
asks whether trades by the strongest true-live wallet candidate set are followed
by directionally favorable same-token price moves.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


HORIZONS = {
    "5m": (5 * 60, 15 * 60),
    "60m": (60 * 60, 60 * 60),
    "24h": (24 * 60 * 60, 6 * 60 * 60),
}


def build_top_wallets(predictions: Path, out_path: Path, regime: str, threshold: float) -> pd.DataFrame:
    pred = pd.read_csv(predictions, usecols=["regime", "fold", "address", "score"])
    parts = []
    for fold, group in pred[pred["regime"].eq(regime)].groupby("fold"):
        k = max(1, int(math.ceil(len(group) * threshold)))
        parts.append(group.nlargest(k, "score"))
    wallets = pd.concat(parts, ignore_index=True)
    wallets[["address", "fold", "score"]].to_csv(out_path, index=False)
    return wallets


def summarize_impacts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group_name, horizon), group in df.groupby(["sample_group", "horizon"], sort=True):
        impact = group["signed_price_move"].dropna()
        weighted = group.dropna(subset=["signed_price_move"])
        weight_sum = weighted["usd_amount"].clip(lower=0).sum()
        mean = float(impact.mean()) if len(impact) else np.nan
        std = float(impact.std(ddof=1)) if len(impact) > 1 else np.nan
        rows.append(
            {
                "sample_group": group_name,
                "horizon": horizon,
                "events": int(len(group)),
                "resolved_events": int(impact.notna().sum()),
                "coverage": float(len(impact) / len(group)) if len(group) else np.nan,
                "mean_signed_price_move": mean,
                "median_signed_price_move": float(impact.median()) if len(impact) else np.nan,
                "weighted_mean_signed_price_move": float((weighted["signed_price_move"] * weighted["usd_amount"].clip(lower=0)).sum() / weight_sum)
                if weight_sum > 0
                else np.nan,
                "positive_move_share": float((impact > 0).mean()) if len(impact) else np.nan,
                "t_stat_vs_zero": mean / (std / math.sqrt(len(impact))) if len(impact) > 1 and std > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def render_report(summary: pd.DataFrame, meta: dict) -> str:
    def table(df: pd.DataFrame) -> str:
        df = df.copy()
        for col in df.columns:
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
        return "\n".join(lines)

    pivot_rows = []
    top = summary[summary["sample_group"].eq("top_model")].set_index("horizon")
    control = summary[summary["sample_group"].eq("control")].set_index("horizon")
    for horizon in HORIZONS:
        if horizon not in top.index or horizon not in control.index:
            continue
        pivot_rows.append(
            {
                "horizon": horizon,
                "top_mean_move": top.loc[horizon, "mean_signed_price_move"],
                "control_mean_move": control.loc[horizon, "mean_signed_price_move"],
                "delta": top.loc[horizon, "mean_signed_price_move"] - control.loc[horizon, "mean_signed_price_move"],
                "top_positive_share": top.loc[horizon, "positive_move_share"],
                "control_positive_share": control.loc[horizon, "positive_move_share"],
                "top_t_stat": top.loc[horizon, "t_stat_vs_zero"],
            }
        )
    comparison = pd.DataFrame(pivot_rows)
    lines = [
        "# Full-Data Price-Impact Smoke Test",
        "",
        "Generated: 2026-05-22",
        "",
        "This diagnostic tests whether top-model wallet trades are followed by directionally favorable same-token price movement.",
        "",
        "## Metadata",
        "",
        "```json",
        json.dumps(meta, indent=2),
        "```",
        "",
        "## Summary",
        "",
        table(summary),
        "",
        "## Top Model vs Control",
        "",
        table(comparison) if not comparison.empty else "No comparable rows.",
        "",
        "## Interpretation",
        "",
    ]
    if not comparison.empty and (comparison["delta"] > 0).all():
        lines.append("- The top-model trade set has higher direction-adjusted price movement than the control set at all tested horizons.")
    elif not comparison.empty and (comparison["delta"] > 0).any():
        lines.append("- The top-model trade set shows selective price-impact evidence, but the effect is horizon-dependent.")
    else:
        lines.append("- This smoke test does not yet support a broad post-trade price-impact claim.")
    lines.append("- This is a bounded smoke test; a publishable study should add same-market/time matching, fee/slippage assumptions, and bootstrap confidence intervals.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users-parquet", default="data/external/sii_polymarket_data/users.parquet")
    parser.add_argument("--predictions", default="results/new_data_sources/full_data_step1_true_live/full_data_step1_predictions.csv")
    parser.add_argument("--output-dir", default="results/new_data_sources/full_data_step1_true_live")
    parser.add_argument("--regime", default="live_plus_static_event_shape")
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--max-events-per-group", type=int, default=100000)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    top_wallets_path = out_dir / "price_impact_top_wallets.csv"
    wallets = build_top_wallets(Path(args.predictions), top_wallets_path, args.regime, args.threshold)

    con = duckdb.connect(str(out_dir / "price_impact_smoke.duckdb"))
    con.execute("PRAGMA threads=8")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE top_wallets AS
        SELECT * FROM read_csv_auto('{top_wallets_path}')
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE candidate_trades AS
        SELECT
          'top_model' AS sample_group,
          row_number() OVER () AS trade_id,
          u.timestamp,
          u.address,
          u.direction,
          u.usd_amount,
          u.price AS entry_price,
          u.market_id,
          u.nonusdc_side
        FROM read_parquet('{args.users_parquet}') u
        JOIN top_wallets w USING(address)
        WHERE u.price BETWEEN 0.001 AND 0.999
          AND u.usd_amount > 0
          AND u.direction IN ('BUY', 'SELL')
        ORDER BY hash(u.address, u.timestamp, u.market_id, u.nonusdc_side)
        LIMIT {args.max_events_per_group}
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE candidate_keys AS
        SELECT DISTINCT market_id, nonusdc_side FROM candidate_trades
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE control_trades AS
        SELECT
          'control' AS sample_group,
          row_number() OVER () AS trade_id,
          u.timestamp,
          u.address,
          u.direction,
          u.usd_amount,
          u.price AS entry_price,
          u.market_id,
          u.nonusdc_side
        FROM read_parquet('{args.users_parquet}') u
        JOIN candidate_keys k USING(market_id, nonusdc_side)
        LEFT JOIN top_wallets w USING(address)
        WHERE w.address IS NULL
          AND u.price BETWEEN 0.001 AND 0.999
          AND u.usd_amount > 0
          AND u.direction IN ('BUY', 'SELL')
          AND random() < 0.01
        LIMIT {args.max_events_per_group}
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE event_trades AS
        SELECT * FROM candidate_trades
        UNION ALL
        SELECT * FROM control_trades
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE price_points AS
        SELECT u.timestamp, u.market_id, u.nonusdc_side, u.price
        FROM read_parquet('{args.users_parquet}') u
        JOIN candidate_keys k USING(market_id, nonusdc_side)
        WHERE u.price BETWEEN 0.001 AND 0.999
        """
    )

    impact_parts = []
    for horizon, (offset, tolerance) in HORIZONS.items():
        table_name = f"impact_{horizon}"
        con.execute(
            f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT
              e.sample_group,
              '{horizon}' AS horizon,
              e.trade_id,
              e.timestamp,
              e.address,
              e.direction,
              e.usd_amount,
              e.entry_price,
              e.market_id,
              e.nonusdc_side,
              min(p.timestamp) AS future_timestamp,
              min_by(p.price, p.timestamp) AS future_price
            FROM event_trades e
            LEFT JOIN price_points p
              ON p.market_id = e.market_id
             AND p.nonusdc_side = e.nonusdc_side
             AND p.timestamp >= e.timestamp + {offset}
             AND p.timestamp <= e.timestamp + {offset + tolerance}
            GROUP BY
              e.sample_group, e.trade_id, e.timestamp, e.address, e.direction,
              e.usd_amount, e.entry_price, e.market_id, e.nonusdc_side
            """
        )
        part = con.execute(
            f"""
            SELECT *,
              CASE
                WHEN future_price IS NULL THEN NULL
                WHEN direction = 'BUY' THEN future_price - entry_price
                WHEN direction = 'SELL' THEN entry_price - future_price
                ELSE NULL
              END AS signed_price_move
            FROM {table_name}
            """
        ).fetchdf()
        impact_parts.append(part)

    impacts = pd.concat(impact_parts, ignore_index=True)
    summary = summarize_impacts(impacts)
    impacts.to_csv(out_dir / "full_data_price_impact_smoke_events.csv", index=False)
    summary.to_csv(out_dir / "full_data_price_impact_smoke_summary.csv", index=False)
    meta = {
        "regime": args.regime,
        "threshold": args.threshold,
        "top_wallet_rows": int(len(wallets)),
        "unique_top_wallets": int(wallets["address"].nunique()),
        "max_events_per_group": args.max_events_per_group,
        "horizons": HORIZONS,
        "candidate_trade_count": int(con.execute("SELECT count(*) FROM candidate_trades").fetchone()[0]),
        "control_trade_count": int(con.execute("SELECT count(*) FROM control_trades").fetchone()[0]),
    }
    (out_dir / "full_data_price_impact_smoke_summary.json").write_text(json.dumps(meta, indent=2, default=str))
    (out_dir / "full_data_price_impact_smoke_report.md").write_text(render_report(summary, meta))
    print(summary.to_string(index=False))
    print(f"Wrote price-impact smoke outputs to {out_dir}")
    con.close()


if __name__ == "__main__":
    main()
