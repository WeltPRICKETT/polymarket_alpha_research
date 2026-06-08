#!/usr/bin/env python3
"""Summarize market/event concentration robustness from event ablation runs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.engine import StrategyBacktester

RESULTS_DIR = PROJECT_ROOT / "results" / "new_data_sources"
DEFAULT_RUN_ROOT = RESULTS_DIR / "event_ablation_paper_runs"
DEFAULT_OUTPUT_TABLE = RESULTS_DIR / "concentration_robustness_table.csv"
DEFAULT_OUTPUT_REPORT = RESULTS_DIR / "concentration_robustness_report.md"
DEFAULT_CANDIDATES = ["live", "live_plus_event_diversification"]


@dataclass(frozen=True)
class ConcentrationInput:
    ablation: str
    run_dir: Path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _load_event_map() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / "sii_market_event_map.csv"
    if not path.exists():
        return pd.DataFrame(columns=["market_id", "event_key", "event_title"])
    df = pd.read_csv(path)
    if "market_id" not in df.columns:
        return pd.DataFrame(columns=["market_id", "event_key", "event_title"])
    out = pd.DataFrame({"market_id": df["market_id"].astype(str)})
    out["event_key"] = (
        df["event_id"].astype("string")
        if "event_id" in df.columns
        else out["market_id"].astype("string")
    )
    out["event_title"] = (
        df["event_title"].astype("string")
        if "event_title" in df.columns
        else out["market_id"].astype("string")
    )
    out["event_key"] = out["event_key"].fillna(out["market_id"]).astype(str)
    out["event_title"] = out["event_title"].fillna(out["market_id"]).astype(str)
    return out.drop_duplicates("market_id")


def rebuild_model_top_pct_ledger(run_dir: Path, top_percentile: float = 0.10) -> pd.DataFrame:
    """Replay the archived rolling predictions for one run into a trade ledger."""
    predictions_path = run_dir / "rolling_origin_predictions.csv"
    if not predictions_path.exists():
        return pd.DataFrame()
    pred_df = pd.read_csv(predictions_path)
    if pred_df.empty:
        return pd.DataFrame()

    backtester = StrategyBacktester(latency_minutes=5, trade_size=100.0, fees_pct=0.001)
    backtester.load_data()
    frames: list[pd.DataFrame] = []
    for fold, group in pred_df.groupby("fold"):
        group = group.sort_values("predicted_probability", ascending=False)
        start = pd.Timestamp(group["test_start"].iloc[0])
        end = pd.Timestamp(group["test_end"].iloc[0])
        n = max(1, int(np.ceil(len(group) * float(top_percentile))))
        targets = group.head(n)["address"].tolist()
        prefix = f"concentration_replay_{run_dir.name}_fold_{int(fold)}"
        backtester.simulate_for_addresses(
            target_addresses=targets,
            strategy_name=f"model_top_pct_{top_percentile:.2f}",
            window_start=start,
            window_end=end,
            output_prefix=prefix,
        )
        path = PROJECT_ROOT / "results" / f"{prefix}_trades.csv"
        if path.exists():
            fold_df = pd.read_csv(path)
            if not fold_df.empty:
                fold_df["fold"] = int(fold)
                frames.append(fold_df)
            path.unlink()

    ledger = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not ledger.empty:
        ledger.to_csv(run_dir / "model_top_pct_0.10_walk_forward_trades.csv", index=False)
    return ledger


def _hhi(values: pd.Series) -> float | None:
    values = pd.to_numeric(values, errors="coerce").fillna(0).abs()
    denom = float(values.sum())
    if denom <= 0:
        return None
    shares = values / denom
    return float((shares**2).sum())


def _drop_group_metrics(
    ledger: pd.DataFrame,
    group_col: str,
    label_col: str,
    prefix: str,
    total_profit: float,
) -> dict[str, Any]:
    grouped = (
        ledger.groupby(group_col, dropna=False)
        .agg(
            label=(label_col, "first"),
            net_profit=("net_profit", "sum"),
            filled_notional=("filled_notional", "sum"),
            trades=("net_profit", "size"),
        )
        .reset_index()
        .sort_values("net_profit", ascending=False)
    )
    if grouped.empty:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_hhi_abs_profit": None,
            f"top_{prefix}_profit_share": None,
        }

    top1_profit = float(grouped.head(1)["net_profit"].sum())
    top3_profit = float(grouped.head(3)["net_profit"].sum())
    top5_profit = float(grouped.head(5)["net_profit"].sum())
    max_notional = grouped.sort_values("filled_notional", ascending=False).head(1)
    max_notional_profit = float(max_notional["net_profit"].sum())
    max_notional_filled = float(max_notional["filled_notional"].sum())
    total_filled = float(pd.to_numeric(ledger["filled_notional"], errors="coerce").sum())
    remaining_notional_profit = total_profit - max_notional_profit
    remaining_notional_filled = total_filled - max_notional_filled
    remaining_notional_roi = (
        remaining_notional_profit / remaining_notional_filled
        if remaining_notional_filled > 0
        else None
    )
    return {
        f"{prefix}_count": int(grouped[group_col].nunique(dropna=False)),
        f"{prefix}_hhi_abs_profit": _hhi(grouped["net_profit"]),
        f"top_{prefix}_profit_key": str(grouped[group_col].iloc[0]),
        f"top_{prefix}_profit_label": str(grouped["label"].iloc[0]),
        f"top_{prefix}_profit_share": top1_profit / total_profit if total_profit else None,
        f"drop_top_1_profit_{prefix}_remaining_net_profit": total_profit - top1_profit,
        f"drop_top_3_profit_{prefix}_remaining_net_profit": total_profit - top3_profit,
        f"drop_top_5_profit_{prefix}_remaining_net_profit": total_profit - top5_profit,
        f"top_notional_{prefix}_key": str(max_notional[group_col].iloc[0]),
        f"top_notional_{prefix}_label": str(max_notional["label"].iloc[0]),
        f"drop_top_notional_{prefix}_remaining_roi": remaining_notional_roi,
    }


def summarize_run(
    item: ConcentrationInput,
    event_map: pd.DataFrame,
    *,
    rebuild_ledgers: bool,
) -> dict[str, Any]:
    ledger_path = item.run_dir / "model_top_pct_0.10_walk_forward_trades.csv"
    summary = _read_json(item.run_dir / "event_ablation_summary.json")
    summary_metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
    if rebuild_ledgers:
        rebuilt = rebuild_model_top_pct_ledger(item.run_dir)
        if not rebuilt.empty:
            ledger = rebuilt
        elif ledger_path.exists():
            ledger = pd.read_csv(ledger_path)
        else:
            ledger = pd.DataFrame()
    elif ledger_path.exists():
        ledger = pd.read_csv(ledger_path)
    else:
        ledger = pd.DataFrame()
    if not ledger_path.exists():
        return {
            "ablation": item.ablation,
            "status": "missing_ledger",
            "ledger_path": str(ledger_path),
        }
    if ledger.empty:
        return {"ablation": item.ablation, "status": "empty_ledger", "ledger_path": str(ledger_path)}

    ledger["market_id"] = ledger["market_id"].astype(str)
    ledger["net_profit"] = pd.to_numeric(ledger["net_profit"], errors="coerce").fillna(0.0)
    ledger["filled_notional"] = pd.to_numeric(
        ledger["filled_notional"], errors="coerce"
    ).fillna(0.0)
    merged = ledger.merge(event_map, on="market_id", how="left")
    merged["event_key"] = merged["event_key"].fillna(merged["market_id"]).astype(str)
    merged["event_title"] = merged["event_title"].fillna(merged["market_id"]).astype(str)

    total_profit = float(merged["net_profit"].sum())
    summary_profit = summary_metrics.get("walk_forward_total_net_profit")
    ledger_summary_profit_gap = (
        float(total_profit - summary_profit)
        if summary_profit is not None
        else None
    )
    total_filled = float(merged["filled_notional"].sum())
    random_summary = _read_json(item.run_dir / "fold_matched_random_baseline_summary.json")
    matched_path = item.run_dir / "fold_matched_random_baseline_matches.csv"
    matched_df = pd.read_csv(matched_path) if matched_path.exists() else pd.DataFrame()
    matched_random_roi = (
        float(pd.to_numeric(matched_df["equal_capital_roi"], errors="coerce").mean())
        if "equal_capital_roi" in matched_df.columns and not matched_df.empty
        else None
    )
    random_roi = _nested(
        random_summary,
        "matched_random_aggregate_mean",
        "mean_equal_capital_roi",
        default=matched_random_roi,
    )
    bootstrap_prob = _nested(
        random_summary,
        "bootstrap_model_minus_matched_random",
        "intervals",
        "mean_equal_capital_roi",
        "probability_model_beats_matched_random",
        default=None,
    )

    row: dict[str, Any] = {
        "ablation": item.ablation,
        "status": "ok",
        "ledger_path": str(ledger_path),
        "trades": int(len(merged)),
        "total_net_profit": total_profit,
        "summary_total_net_profit": summary_profit,
        "ledger_summary_profit_gap": ledger_summary_profit_gap,
        "total_filled_notional": total_filled,
        "raw_roi": total_profit / total_filled if total_filled > 0 else None,
        "matched_random_mean_equal_capital_roi": random_roi,
        "bootstrap_prob_beats_matched_random": bootstrap_prob,
    }
    row.update(
        _drop_group_metrics(
            merged,
            group_col="market_id",
            label_col="market_id",
            prefix="market",
            total_profit=total_profit,
        )
    )
    row.update(
        _drop_group_metrics(
            merged,
            group_col="event_key",
            label_col="event_title",
            prefix="event",
            total_profit=total_profit,
        )
    )

    for scope in ("market", "event"):
        remaining = row.get(f"drop_top_notional_{scope}_remaining_roi")
        row[f"drop_top_notional_{scope}_roi_above_matched_random"] = (
            bool(remaining > random_roi)
            if remaining is not None and random_roi is not None
            else None
        )
    row["concentration_flag"] = (
        "material_concentration"
        if (
            (row.get("top_event_profit_share") or 0) >= 0.5
            or row.get("drop_top_3_profit_event_remaining_net_profit", 0) <= 0
        )
        else "less_concentrated"
    )
    return row


def write_report(df: pd.DataFrame, path: Path) -> None:
    display_cols = [
        "ablation",
        "total_net_profit",
        "raw_roi",
        "matched_random_mean_equal_capital_roi",
        "bootstrap_prob_beats_matched_random",
        "drop_top_1_profit_market_remaining_net_profit",
        "drop_top_3_profit_market_remaining_net_profit",
        "drop_top_1_profit_event_remaining_net_profit",
        "drop_top_3_profit_event_remaining_net_profit",
        "top_event_profit_share",
        "event_hhi_abs_profit",
        "drop_top_notional_market_roi_above_matched_random",
        "drop_top_notional_event_roi_above_matched_random",
        "concentration_flag",
    ]
    available = [col for col in display_cols if col in df.columns]
    table = _to_markdown_table(df[available])
    lines = [
        "# Concentration Robustness Report",
        "",
        "This report reuses paper event-ablation walk-forward trade ledgers. It does not retrain models.",
        "",
        table,
        "",
        "Interpretation rule: if profit remains dominated by one event or disappears after dropping the top few events, treat the economic evidence as concentration-sensitive rather than broad alpha.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _to_markdown_table(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[col]) for col in df.columns) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-table", type=Path, default=DEFAULT_OUTPUT_TABLE)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_OUTPUT_REPORT)
    parser.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES)
    parser.add_argument(
        "--use-existing-ledgers",
        action="store_true",
        help="Do not replay archived rolling predictions before concentration analysis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event_map = _load_event_map()
    rows = [
        summarize_run(
            ConcentrationInput(name, args.run_root / name),
            event_map,
            rebuild_ledgers=not args.use_existing_ledgers,
        )
        for name in args.candidates
    ]
    df = pd.DataFrame(rows)
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_table, index=False)
    write_report(df, args.output_report)
    print(f"Wrote {args.output_table}")
    print(f"Wrote {args.output_report}")
    print(df[["ablation", "status", "total_net_profit", "concentration_flag"]].to_string(index=False))


if __name__ == "__main__":
    main()
