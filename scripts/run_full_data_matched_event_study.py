#!/usr/bin/env python
"""Matched price-impact event study for full-data top-model wallets."""

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


def summarize_pairs(pairs: pd.DataFrame, bootstrap_samples: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    boot_rows = []
    for horizon, group in pairs.groupby("horizon", sort=True):
        valid = group.dropna(subset=["top_signed_price_move", "control_signed_price_move"]).copy()
        if valid.empty:
            rows.append(
                {
                    "horizon": horizon,
                    "matched_pairs": 0,
                    "mean_top_move": np.nan,
                    "mean_control_move": np.nan,
                    "mean_delta": np.nan,
                    "median_delta": np.nan,
                    "positive_delta_share": np.nan,
                    "bootstrap_delta_p025": np.nan,
                    "bootstrap_delta_p975": np.nan,
                    "bootstrap_prob_delta_positive": np.nan,
                    "one_sided_p_delta_le_0": np.nan,
                }
            )
            continue
        delta = valid["top_signed_price_move"] - valid["control_signed_price_move"]
        boot = []
        n = len(delta)
        delta_values = delta.to_numpy()
        for i in range(bootstrap_samples):
            sample = delta_values[rng.integers(0, n, size=n)]
            value = float(np.mean(sample))
            boot.append(value)
            boot_rows.append({"horizon": horizon, "bootstrap_sample": i, "mean_delta": value})
        boot_arr = np.array(boot)
        rows.append(
            {
                "horizon": horizon,
                "matched_pairs": int(n),
                "mean_top_move": float(valid["top_signed_price_move"].mean()),
                "mean_control_move": float(valid["control_signed_price_move"].mean()),
                "mean_delta": float(delta.mean()),
                "median_delta": float(delta.median()),
                "positive_delta_share": float((delta > 0).mean()),
                "bootstrap_delta_p025": float(np.quantile(boot_arr, 0.025)),
                "bootstrap_delta_p975": float(np.quantile(boot_arr, 0.975)),
                "bootstrap_prob_delta_positive": float((boot_arr > 0).mean()),
                "one_sided_p_delta_le_0": float((boot_arr <= 0).mean()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(boot_rows)


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

    supported = summary[(summary["horizon"].isin(["60m", "24h"])) & (summary["bootstrap_delta_p025"] > 0)]
    lines = [
        "# Full-Data Matched Event Study",
        "",
        "Generated: 2026-05-22",
        "",
        "This study matches each top-model wallet trade to a non-top-wallet trade in the same market, same token side, same trade direction, and nearby time window.",
        "",
        "## Metadata",
        "",
        "```json",
        json.dumps(meta, indent=2),
        "```",
        "",
        "## Matched Price-Impact Summary",
        "",
        table(summary),
        "",
        "## Interpretation",
        "",
    ]
    if len(supported) >= 2:
        lines.append("- The 60m and 24h horizons both have bootstrap confidence intervals above zero. This supports a medium-horizon informed-trader interpretation.")
    elif not supported.empty:
        lines.append("- At least one medium-horizon window survives matched controls; the signal is promising but horizon-specific.")
    else:
        lines.append("- The matched study does not yet show a robust positive medium-horizon delta after same-market/time controls.")
    lines.append("- The 5m window should be interpreted separately: a negative or weak 5m result means this is not an immediate microstructure edge.")
    lines.append("- Remaining academic upgrades: stricter time matching, multiple controls per top trade, event-level clustered bootstrap, and fee/slippage sensitivity.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users-parquet", default="data/external/sii_polymarket_data/users.parquet")
    parser.add_argument("--predictions", default="results/new_data_sources/full_data_step1_true_live/full_data_step1_predictions.csv")
    parser.add_argument("--output-dir", default="results/new_data_sources/full_data_step1_true_live")
    parser.add_argument("--regime", default="live_plus_static_event_shape")
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--max-top-events", type=int, default=30000)
    parser.add_argument("--match-window-seconds", type=int, default=3600)
    parser.add_argument("--controls-per-top", type=int, default=1)
    parser.add_argument("--size-match-weight", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    top_wallets_path = out_dir / "matched_event_top_wallets.csv"
    wallets = build_top_wallets(Path(args.predictions), top_wallets_path, args.regime, args.threshold)

    con = duckdb.connect(str(out_dir / "matched_event_study.duckdb"))
    con.execute("PRAGMA threads=8")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE top_wallets AS
        SELECT * FROM read_csv_auto('{top_wallets_path}')
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE top_trades AS
        SELECT
          row_number() OVER () AS pair_id,
          u.timestamp AS top_timestamp,
          u.address AS top_address,
          u.direction,
          u.usd_amount AS top_usd_amount,
          u.price AS top_entry_price,
          u.market_id,
          u.event_id AS top_event_id,
          u.nonusdc_side
        FROM read_parquet('{args.users_parquet}') u
        JOIN top_wallets w USING(address)
        WHERE u.price BETWEEN 0.001 AND 0.999
          AND u.usd_amount > 0
          AND u.direction IN ('BUY', 'SELL')
        ORDER BY hash(u.address, u.timestamp, u.market_id, u.nonusdc_side)
        LIMIT {args.max_top_events}
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE top_keys AS
        SELECT
          market_id,
          nonusdc_side,
          direction,
          min(top_timestamp) AS min_ts,
          max(top_timestamp) AS max_ts
        FROM top_trades
        GROUP BY 1, 2, 3
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE control_pool AS
        SELECT
          u.timestamp,
          u.address,
          u.direction,
          u.usd_amount,
          u.price,
          u.market_id,
          u.event_id,
          u.nonusdc_side
        FROM read_parquet('{args.users_parquet}') u
        JOIN top_keys k USING(market_id, nonusdc_side, direction)
        LEFT JOIN top_wallets w USING(address)
        WHERE w.address IS NULL
          AND u.price BETWEEN 0.001 AND 0.999
          AND u.usd_amount > 0
          AND u.timestamp BETWEEN k.min_ts - {args.match_window_seconds}
                              AND k.max_ts + {args.match_window_seconds}
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE matched_pairs AS
        SELECT *
        FROM (
          SELECT
            t.pair_id,
            t.top_timestamp,
            t.top_address,
            c.timestamp AS control_timestamp,
            c.address AS control_address,
            t.direction,
            t.market_id,
            t.top_event_id,
            c.event_id AS control_event_id,
            t.nonusdc_side,
            t.top_usd_amount,
            c.usd_amount AS control_usd_amount,
            t.top_entry_price,
            c.price AS control_entry_price,
            abs(c.timestamp::BIGINT - t.top_timestamp::BIGINT) AS abs_time_diff,
            abs(ln(1 + c.usd_amount) - ln(1 + t.top_usd_amount)) AS abs_log_usd_diff,
            row_number() OVER (
              PARTITION BY t.pair_id
              ORDER BY
                abs(c.timestamp::BIGINT - t.top_timestamp::BIGINT)
                  + ({args.size_match_weight} * 3600 * abs(ln(1 + c.usd_amount) - ln(1 + t.top_usd_amount))),
                hash(c.address, c.timestamp, c.market_id)
            ) AS rn
          FROM top_trades t
          JOIN control_pool c
            ON c.market_id = t.market_id
           AND c.nonusdc_side = t.nonusdc_side
           AND c.direction = t.direction
           AND c.timestamp BETWEEN t.top_timestamp - {args.match_window_seconds}
                               AND t.top_timestamp + {args.match_window_seconds}
        )
        WHERE rn <= {args.controls_per_top}
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE pair_keys AS
        SELECT DISTINCT market_id, nonusdc_side FROM matched_pairs
        """
    )
    min_ts, max_ts = con.execute(
        """
        SELECT min(least(top_timestamp, control_timestamp)), max(greatest(top_timestamp, control_timestamp))
        FROM matched_pairs
        """
    ).fetchone()
    max_horizon = max(offset + tolerance for offset, tolerance in HORIZONS.values())
    con.execute(
        f"""
        CREATE OR REPLACE TABLE price_points AS
        SELECT u.timestamp, u.market_id, u.nonusdc_side, u.price
        FROM read_parquet('{args.users_parquet}') u
        JOIN pair_keys k USING(market_id, nonusdc_side)
        WHERE u.price BETWEEN 0.001 AND 0.999
          AND u.timestamp BETWEEN {int(min_ts or 0)} AND {int((max_ts or 0) + max_horizon)}
        """
    )

    impact_parts = []
    for horizon, (offset, tolerance) in HORIZONS.items():
        con.execute(
            f"""
            CREATE OR REPLACE TABLE pair_impact_{horizon} AS
            WITH top_future AS (
              SELECT
                m.pair_id,
                min_by(p.price, p.timestamp) AS top_future_price
              FROM matched_pairs m
              LEFT JOIN price_points p
                ON p.market_id = m.market_id
               AND p.nonusdc_side = m.nonusdc_side
               AND p.timestamp BETWEEN m.top_timestamp + {offset}
                                   AND m.top_timestamp + {offset + tolerance}
              GROUP BY m.pair_id
            ),
            control_future AS (
              SELECT
                m.pair_id,
                min_by(p.price, p.timestamp) AS control_future_price
              FROM matched_pairs m
              LEFT JOIN price_points p
                ON p.market_id = m.market_id
               AND p.nonusdc_side = m.nonusdc_side
               AND p.timestamp BETWEEN m.control_timestamp + {offset}
                                   AND m.control_timestamp + {offset + tolerance}
              GROUP BY m.pair_id
            )
            SELECT
              m.*,
              '{horizon}' AS horizon,
              tf.top_future_price,
              cf.control_future_price,
              CASE
                WHEN tf.top_future_price IS NULL THEN NULL
                WHEN m.direction = 'BUY' THEN tf.top_future_price - m.top_entry_price
                WHEN m.direction = 'SELL' THEN m.top_entry_price - tf.top_future_price
                ELSE NULL
              END AS top_signed_price_move,
              CASE
                WHEN cf.control_future_price IS NULL THEN NULL
                WHEN m.direction = 'BUY' THEN cf.control_future_price - m.control_entry_price
                WHEN m.direction = 'SELL' THEN m.control_entry_price - cf.control_future_price
                ELSE NULL
              END AS control_signed_price_move
            FROM matched_pairs m
            LEFT JOIN top_future tf USING(pair_id)
            LEFT JOIN control_future cf USING(pair_id)
            """
        )
        impact_parts.append(con.execute(f"SELECT * FROM pair_impact_{horizon}").fetchdf())

    pairs = pd.concat(impact_parts, ignore_index=True)
    summary, bootstrap = summarize_pairs(pairs, args.bootstrap_samples, seed=42)
    pairs.to_csv(out_dir / "full_data_matched_event_study_pairs.csv", index=False)
    summary.to_csv(out_dir / "full_data_matched_event_study_summary.csv", index=False)
    bootstrap.to_csv(out_dir / "full_data_matched_event_study_bootstrap.csv", index=False)
    meta = {
        "regime": args.regime,
        "threshold": args.threshold,
        "top_wallet_rows": int(len(wallets)),
        "unique_top_wallets": int(wallets["address"].nunique()),
        "max_top_events": args.max_top_events,
        "match_window_seconds": args.match_window_seconds,
        "controls_per_top": args.controls_per_top,
        "size_match_weight": args.size_match_weight,
        "bootstrap_samples": args.bootstrap_samples,
        "top_trades": int(con.execute("SELECT count(*) FROM top_trades").fetchone()[0]),
        "matched_pairs": int(con.execute("SELECT count(*) FROM matched_pairs").fetchone()[0]),
        "control_pool": int(con.execute("SELECT count(*) FROM control_pool").fetchone()[0]),
        "horizons": HORIZONS,
    }
    (out_dir / "full_data_matched_event_study_summary.json").write_text(json.dumps(meta, indent=2, default=str))
    (out_dir / "full_data_matched_event_study_report.md").write_text(render_report(summary, meta))
    print(summary.to_string(index=False))
    print(f"Wrote matched event study outputs to {out_dir}")
    con.close()


if __name__ == "__main__":
    main()
