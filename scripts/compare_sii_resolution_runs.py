#!/usr/bin/env python3
"""Compare strict CLOB and CLOB+secondary resolution runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.engine import StrategyBacktester
from src.data_ingestion.external_sources import build_resolution_coverage_summary


RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "new_data_sources"
STRICT_DIR = RUNS_DIR / "sii_strict_clob_run"
PLUS_DIR = RUNS_DIR / "sii_clob_plus_secondary_run"
COMPARISON_PATH = RUNS_DIR / "sii_resolution_run_comparison.csv"
RISK_PATH = RUNS_DIR / "sii_secondary_fallback_risk_checks.json"
PLUS_LEDGER_PATH = PLUS_DIR / "model_top_pct_0.10_walk_forward_trades.csv"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _best_direct_metric(run_dir: Path, metric: str) -> float | None:
    df = pd.read_csv(run_dir / "model_comparison.csv")
    model_df = df[~df["Model Type"].eq("baseline")].copy()
    values = pd.to_numeric(model_df[metric], errors="coerce")
    if values.dropna().empty:
        return None
    return float(values.max())


def _run_values(run_dir: Path, coverage: float) -> dict[str, Any]:
    label = _read_json(run_dir / "label_summary.json")
    rolling = _read_json(run_dir / "rolling_origin_summary.json")
    wf = _read_json(run_dir / "model_informed_walk_forward_summary.json")
    matched = _read_json(run_dir / "fold_matched_random_baseline_summary.json")
    model_strategy = matched.get("model_strategy", "model_top_pct_0.10")
    wf_agg = _nested(wf, "aggregate", model_strategy, default={}) or {}
    bootstrap_roi = _nested(
        matched,
        "bootstrap_model_minus_matched_random",
        "intervals",
        "mean_equal_capital_roi",
        default={},
    ) or {}
    random_position = _nested(
        _read_json(run_dir / "random_baseline_stability_summary.json"),
        "model_vs_random",
        "mean_equal_capital_roi",
        default={},
    ) or {}
    return {
        "resolution_coverage": coverage,
        "eligible_wallet_rows": _nested(label, "future_label_coverage", "eligible_rows"),
        "test_positives": _nested(label, "label_counts_by_split", "test:1"),
        "direct_split_auc": _best_direct_metric(run_dir, "AUC-ROC"),
        "direct_split_average_precision": _best_direct_metric(run_dir, "Avg Precision"),
        "rolling_auc": _nested(rolling, "aggregate", "mean_auc_roc"),
        "rolling_average_precision": _nested(rolling, "aggregate", "mean_avg_precision"),
        "precision_at_20": _nested(rolling, "threshold_policies", "top_k_20", "mean_precision"),
        "top_5_percent_precision": _nested(rolling, "threshold_policies", "top_pct_0.05", "mean_precision"),
        "top_10_percent_precision": _nested(rolling, "threshold_policies", "top_pct_0.10", "mean_precision"),
        "walk_forward_total_net_profit": wf_agg.get("total_net_profit"),
        "walk_forward_mean_equal_capital_roi": wf_agg.get("mean_equal_capital_roi"),
        "model_vs_random_percentile": random_position.get("random_percentile"),
        "bootstrap_prob_beats_matched_random": bootstrap_roi.get(
            "probability_model_beats_matched_random"
        ),
        "mean_unique_markets": wf_agg.get("mean_unique_markets"),
        "top_market_filled_notional_share": wf_agg.get("mean_top_market_share"),
    }


def write_strict_coverage_summary() -> dict[str, Any]:
    strict_resolutions = pd.read_csv(STRICT_DIR / "sii_clob_market_resolutions.csv")
    summary = build_resolution_coverage_summary(strict_resolutions)
    _write_json(STRICT_DIR / "sii_resolution_coverage_summary.json", summary)
    return summary


def write_comparison() -> pd.DataFrame:
    strict_summary = write_strict_coverage_summary()
    plus_summary = _read_json(PLUS_DIR / "sii_resolution_coverage_summary.json")
    strict_values = _run_values(
        STRICT_DIR, strict_summary["primary_clob_winner_field_coverage"]
    )
    plus_values = _run_values(
        PLUS_DIR, plus_summary["combined_usable_resolution_coverage"]
    )
    rows = [
        {
            "metric": metric,
            "strict_clob_only": strict_values.get(metric),
            "clob_plus_secondary": plus_values.get(metric),
        }
        for metric in strict_values
    ]
    out = pd.DataFrame(rows)
    out.to_csv(COMPARISON_PATH, index=False)
    return out


def _select_model_top_pct_targets(pred_group: pd.DataFrame, top_percentile: float) -> list[str]:
    group = pred_group.sort_values("predicted_probability", ascending=False)
    n = max(1, int(np.ceil(len(group) * top_percentile)))
    return group.head(n)["address"].tolist()


def collect_plus_secondary_ledger(top_percentile: float = 0.10) -> pd.DataFrame:
    predictions_path = RESULTS_DIR / "rolling_origin_predictions.csv"
    if not predictions_path.exists():
        return pd.DataFrame()

    pred_df = pd.read_csv(predictions_path)
    backtester = StrategyBacktester(latency_minutes=5, trade_size=100.0, fees_pct=0.001)
    backtester.load_data()

    frames: list[pd.DataFrame] = []
    for fold, group in pred_df.groupby("fold"):
        start = pd.Timestamp(group["test_start"].iloc[0])
        end = pd.Timestamp(group["test_end"].iloc[0])
        targets = _select_model_top_pct_targets(group, top_percentile)
        prefix = f"sii_plus_model_top_pct_0.10_fold_{int(fold)}"
        backtester.simulate_for_addresses(
            target_addresses=targets,
            strategy_name="model_top_pct_0.10",
            window_start=start,
            window_end=end,
            output_prefix=prefix,
        )
        path = RESULTS_DIR / f"{prefix}_trades.csv"
        if path.exists():
            fold_df = pd.read_csv(path)
            if not fold_df.empty:
                fold_df["fold"] = int(fold)
                frames.append(fold_df)

    ledger = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not ledger.empty:
        PLUS_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        ledger.to_csv(PLUS_LEDGER_PATH, index=False)
    return ledger


def write_risk_checks() -> dict[str, Any]:
    ledger = collect_plus_secondary_ledger()
    plus_resolutions = pd.read_csv(PLUS_DIR / "sii_clob_plus_secondary_market_resolutions.csv")
    source_by_market = plus_resolutions.set_index("market_id")["resolution_source"].to_dict()
    if ledger.empty:
        risk = {"status": "insufficient_data", "reason": "No plus-secondary ledger rows."}
        _write_json(RISK_PATH, risk)
        return risk

    ledger["resolution_source"] = ledger["market_id"].map(source_by_market).fillna("unknown")
    market = (
        ledger.groupby(["market_id", "resolution_source"], dropna=False)
        .agg(net_profit=("net_profit", "sum"), filled_notional=("filled_notional", "sum"))
        .reset_index()
    )
    total_profit = float(ledger["net_profit"].sum())
    total_notional = float(ledger["filled_notional"].sum())
    max_profit_market = market.sort_values("net_profit", ascending=False).head(1)
    top3_profit_markets = market.sort_values("net_profit", ascending=False).head(3)
    max_notional_market = market.sort_values("filled_notional", ascending=False).head(1)

    without_max_profit = total_profit - float(max_profit_market["net_profit"].sum())
    without_top3_profit = total_profit - float(top3_profit_markets["net_profit"].sum())
    without_max_notional_profit = total_profit - float(max_notional_market["net_profit"].sum())
    without_max_notional_filled = total_notional - float(max_notional_market["filled_notional"].sum())
    without_max_notional_roi = (
        without_max_notional_profit / without_max_notional_filled
        if without_max_notional_filled > 0
        else None
    )

    by_source = (
        ledger.groupby("resolution_source")
        .agg(
            trades=("net_profit", "size"),
            net_profit=("net_profit", "sum"),
            filled_notional=("filled_notional", "sum"),
        )
        .reset_index()
    )
    by_source["roi"] = by_source["net_profit"] / by_source["filled_notional"]
    source_rows = by_source.set_index("resolution_source").to_dict(orient="index")
    primary_profit = float(
        source_rows.get("clob_winner_field", {}).get("net_profit", 0.0)
    )
    secondary_profit = float(
        source_rows.get("clob_terminal_price_argmax_secondary", {}).get("net_profit", 0.0)
    )
    secondary_share = secondary_profit / total_profit if total_profit else None

    matched = _read_json(PLUS_DIR / "fold_matched_random_baseline_summary.json")
    random_summary = _read_json(PLUS_DIR / "random_baseline_stability_summary.json")
    random_roi_reference = _nested(
        random_summary,
        "distributions",
        "mean_equal_capital_roi",
        "mean",
        default=None,
    )
    if random_roi_reference is None:
        random_roi_reference = _nested(
            matched,
            "matched_random_aggregate_mean",
            "mean_equal_capital_roi",
            default=None,
        )

    conclusion = "secondary fallback sensitivity acceptable"
    if (
        secondary_share is not None
        and secondary_share > 0.5
        and (without_top3_profit <= 0 or without_max_profit <= 0)
    ):
        conclusion = "secondary fallback introduces material sensitivity"

    risk = {
        "status": "ok",
        "strategy": "model_top_pct_0.10",
        "ledger_csv": str(PLUS_LEDGER_PATH),
        "total_net_profit": total_profit,
        "total_filled_notional": total_notional,
        "drop_max_profit_market": {
            "market_id": max_profit_market["market_id"].iloc[0],
            "source": max_profit_market["resolution_source"].iloc[0],
            "removed_net_profit": float(max_profit_market["net_profit"].iloc[0]),
            "remaining_net_profit": without_max_profit,
            "still_positive": bool(without_max_profit > 0),
        },
        "drop_top_3_profit_markets": {
            "market_ids": top3_profit_markets["market_id"].tolist(),
            "sources": top3_profit_markets["resolution_source"].tolist(),
            "removed_net_profit": float(top3_profit_markets["net_profit"].sum()),
            "remaining_net_profit": without_top3_profit,
            "still_positive": bool(without_top3_profit > 0),
        },
        "drop_max_notional_market": {
            "market_id": max_notional_market["market_id"].iloc[0],
            "source": max_notional_market["resolution_source"].iloc[0],
            "removed_filled_notional": float(max_notional_market["filled_notional"].iloc[0]),
            "remaining_roi": without_max_notional_roi,
            "random_roi_reference": random_roi_reference,
            "roi_still_above_random": (
                bool(without_max_notional_roi > random_roi_reference)
                if random_roi_reference is not None and without_max_notional_roi is not None
                else None
            ),
        },
        "source_contribution": source_rows,
        "secondary_profit_share_of_total": secondary_share,
        "strict_only_vs_secondary_only_direction_consistent": bool(
            primary_profit == 0
            or secondary_profit == 0
            or np.sign(primary_profit) == np.sign(secondary_profit)
        ),
        "bootstrap_prob_beats_matched_random": _nested(
            matched,
            "bootstrap_model_minus_matched_random",
            "intervals",
            "mean_equal_capital_roi",
            "probability_model_beats_matched_random",
        ),
        "conclusion": conclusion,
    }
    _write_json(RISK_PATH, risk)
    return risk


def main() -> None:
    comparison = write_comparison()
    risk = write_risk_checks()
    print(f"Wrote {COMPARISON_PATH}")
    print(comparison.to_string(index=False))
    print(f"Wrote {RISK_PATH}")
    print(json.dumps(risk, indent=2, default=str))


if __name__ == "__main__":
    main()
