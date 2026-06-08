#!/usr/bin/env python3
"""Run controlled event-feature ablations on top of the live feature set."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.models.feature_sets as feature_sets
from src.backtesting.engine import StrategyBacktester
from src.experiments.future_label_experiments import (
    run_fold_matched_random_baseline_analysis,
    run_model_informed_walk_forward,
    run_random_baseline_stability,
    run_rolling_origin_evaluation,
)

RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = RESULTS_DIR / "new_data_sources"
EVENT_ABLATION_DIR = OUTPUT_DIR / "event_ablation_runs"
TABLE_PATH = OUTPUT_DIR / "event_ablation_table.csv"
REPORT_PATH = OUTPUT_DIR / "event_ablation_report.md"
PAPER_CANDIDATES = [
    "live",
    "live_plus_event_diversification",
    "live_plus_mean_markets_per_event_traded",
]
RICH_CANDIDATES = [
    "live",
    "live_plus_event_diversification",
    "live_plus_series_diversification",
    "live_plus_topic_diversification",
    "live_plus_tag_topic_concentration",
    "live_plus_neg_risk_exposure",
    "live_plus_static_event_shape",
]


@dataclass(frozen=True)
class AblationSpec:
    name: str
    extra_features: list[str]


@dataclass
class AblationResult:
    name: str
    feature_set: str
    extra_features: list[str]
    run_dir: Path
    metrics: dict[str, Any]


EVENT_ABLATION_SPECS = [
    AblationSpec("live", []),
    AblationSpec("live_plus_unique_events", ["unique_events"]),
    AblationSpec("live_plus_event_diversification", ["event_diversification"]),
    AblationSpec("live_plus_event_notional_hhi", ["event_notional_hhi"]),
    AblationSpec("live_plus_top_event_notional_share", ["top_event_notional_share"]),
    AblationSpec("live_plus_same_event_multi_market_share", ["same_event_multi_market_share"]),
    AblationSpec("live_plus_mean_markets_per_event_traded", ["mean_markets_per_event_traded"]),
    AblationSpec(
        "live_plus_unique_events_event_diversification",
        ["unique_events", "event_diversification"],
    ),
    AblationSpec(
        "live_plus_event_notional_hhi_top_share",
        ["event_notional_hhi", "top_event_notional_share"],
    ),
    AblationSpec(
        "live_plus_series_diversification",
        ["unique_series", "series_diversification", "series_notional_hhi", "top_series_notional_share"],
    ),
    AblationSpec(
        "live_plus_topic_diversification",
        ["unique_topics", "topic_diversification", "topic_notional_hhi", "top_topic_notional_share"],
    ),
    AblationSpec(
        "live_plus_tag_topic_concentration",
        [
            "unique_primary_tags",
            "primary_tag_notional_hhi",
            "top_primary_tag_notional_share",
            "unique_topics",
            "topic_notional_hhi",
            "top_topic_notional_share",
        ],
    ),
    AblationSpec(
        "live_plus_neg_risk_exposure",
        ["neg_risk_trade_share", "neg_risk_notional_share"],
    ),
    AblationSpec(
        "live_plus_static_event_shape",
        [
            "avg_event_market_count",
            "avg_event_duration_days",
            "avg_creation_to_start_days",
            "neg_risk_trade_share",
            "order_book_trade_share",
        ],
    ),
]

ARTIFACTS_TO_ARCHIVE = [
    "rolling_origin_results.csv",
    "rolling_origin_predictions.csv",
    "rolling_origin_summary.json",
    "threshold_policy_results.csv",
    "model_informed_walk_forward_results.csv",
    "model_informed_walk_forward_summary.json",
    "random_baseline_stability.csv",
    "random_baseline_seed_summary.csv",
    "random_baseline_stability_summary.json",
    "fold_matched_random_baseline_matches.csv",
    "fold_matched_random_baseline_rank_summary.csv",
    "fold_matched_random_bootstrap.csv",
    "fold_matched_random_baseline_summary.json",
]
ROLLING_ARTIFACTS = [
    "rolling_origin_results.csv",
    "rolling_origin_predictions.csv",
    "rolling_origin_summary.json",
    "threshold_policy_results.csv",
]
WALK_FORWARD_ARTIFACTS = [
    "model_informed_walk_forward_results.csv",
    "model_informed_walk_forward_summary.json",
]
RANDOM_ARTIFACTS = [
    "random_baseline_stability.csv",
    "random_baseline_seed_summary.csv",
    "random_baseline_stability_summary.json",
]
FOLD_MATCH_ARTIFACTS = [
    "fold_matched_random_baseline_matches.csv",
    "fold_matched_random_baseline_rank_summary.csv",
    "fold_matched_random_bootstrap.csv",
    "fold_matched_random_baseline_summary.json",
]


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


def register_ablation_feature_set(name: str, extra_features: Iterable[str]) -> str:
    """Register a temporary live-plus-event feature set for this process."""
    extras = list(extra_features)
    unknown = [col for col in extras if col not in feature_sets.EVENT_LIVE_FEATURE_COLS]
    if unknown:
        raise ValueError(f"Unknown event live feature(s): {unknown}")
    feature_set = f"event_ablation__{name}"
    feature_sets.FEATURE_SETS[feature_set] = feature_sets.LIVE_FEATURE_COLS + extras
    feature_sets.FEATURE_LEAKAGE_CLASS.update(
        {col: "pre_resolution_event_context" for col in extras}
    )
    return feature_set


def run_direct_split_lr(feature_set: str, feature_path: Path | None = None) -> dict[str, Any]:
    """Fast direct split check using a balanced logistic regression."""
    feature_path = feature_path or DATA_DIR / "features" / "model_input.csv"
    df = pd.read_csv(feature_path)
    feature_cols = feature_sets.get_feature_columns(feature_set)
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns for {feature_set}: {missing}")

    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].copy()
    if train.empty or test.empty:
        return {"status": "empty_split"}
    y_train = train["Trader_Success_Rate"].astype(int).to_numpy()
    y_test = test["Trader_Success_Rate"].astype(int).to_numpy()
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {"status": "single_class_split"}

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )
    model.fit(train[feature_cols].to_numpy(), y_train)
    y_prob = model.predict_proba(test[feature_cols].to_numpy())[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "status": "ok",
        "test_rows": int(len(test)),
        "test_positives": int((y_test == 1).sum()),
        "auc": float(roc_auc_score(y_test, y_prob)),
        "average_precision": float(average_precision_score(y_test, y_prob)),
        "precision_0_5": float(precision_score(y_test, y_pred, zero_division=0)),
    }


def collect_model_top_pct_ledger(
    run_dir: Path,
    top_percentile: float = 0.10,
    predictions_path: Path | None = None,
) -> pd.DataFrame:
    """Replay the model top-percentile strategy to capture trade-level rows."""
    predictions_path = predictions_path or RESULTS_DIR / "rolling_origin_predictions.csv"
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
        prefix = f"event_ablation_{run_dir.name}_fold_{int(fold)}"
        backtester.simulate_for_addresses(
            target_addresses=targets,
            strategy_name=f"model_top_pct_{top_percentile:.2f}",
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
            path.unlink()

    ledger = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not ledger.empty:
        run_dir.mkdir(parents=True, exist_ok=True)
        ledger.to_csv(run_dir / "model_top_pct_0.10_walk_forward_trades.csv", index=False)
    return ledger


def drop_top_event_metrics(ledger: pd.DataFrame) -> dict[str, Any]:
    """Summarize how much walk-forward profit depends on the largest events."""
    if ledger.empty or "market_id" not in ledger.columns or "net_profit" not in ledger.columns:
        return {"drop_top_event_status": "insufficient_data"}

    event_map_path = DATA_DIR / "processed" / "sii_market_event_map.csv"
    if event_map_path.exists():
        event_map = pd.read_csv(event_map_path)
        map_cols = ["market_id", "event_id", "event_title"]
        event_map = event_map[[col for col in map_cols if col in event_map.columns]].drop_duplicates("market_id")
        merged = ledger.merge(event_map, on="market_id", how="left")
    else:
        merged = ledger.copy()
        merged["event_id"] = merged["market_id"]
        merged["event_title"] = merged["market_id"]

    merged["event_key"] = merged["event_id"].fillna(merged["market_id"]).astype(str)
    by_event = (
        merged.groupby("event_key", dropna=False)
        .agg(
            event_title=("event_title", "first") if "event_title" in merged.columns else ("market_id", "first"),
            net_profit=("net_profit", "sum"),
            filled_notional=("filled_notional", "sum"),
            markets=("market_id", "nunique"),
        )
        .reset_index()
        .sort_values("net_profit", ascending=False)
    )
    total_profit = float(pd.to_numeric(merged["net_profit"], errors="coerce").sum())
    if by_event.empty:
        return {"drop_top_event_status": "insufficient_data"}
    top1 = by_event.head(1)
    top3 = by_event.head(3)
    profit_abs = by_event["net_profit"].abs()
    denom = float(profit_abs.sum())
    hhi = float(((profit_abs / denom) ** 2).sum()) if denom else None
    return {
        "drop_top_event_status": "ok",
        "top_profit_event": str(top1["event_title"].iloc[0]),
        "top_profit_event_key": str(top1["event_key"].iloc[0]),
        "total_net_profit": total_profit,
        "drop_top_1_profit_event_removed_net_profit": float(top1["net_profit"].sum()),
        "drop_top_1_profit_event_remaining_net_profit": total_profit - float(top1["net_profit"].sum()),
        "drop_top_3_profit_events_removed_net_profit": float(top3["net_profit"].sum()),
        "drop_top_3_profit_events_remaining_net_profit": total_profit - float(top3["net_profit"].sum()),
        "top_event_profit_share": (
            float(top1["net_profit"].sum()) / total_profit if total_profit else None
        ),
        "event_hhi_of_abs_profit": hhi,
    }


def _summarise_current_run(
    direct: dict[str, Any],
    rolling: dict[str, Any],
    walk_forward: dict[str, Any],
    random_summary: dict[str, Any],
    fold_matched: dict[str, Any],
    event_risk: dict[str, Any],
    top_percentile: float = 0.10,
) -> dict[str, Any]:
    model_strategy = f"model_top_pct_{top_percentile:.2f}"
    wf_agg = _nested(walk_forward, "aggregate", model_strategy, default={}) or {}
    bootstrap_roi = _nested(
        fold_matched,
        "bootstrap_model_minus_matched_random",
        "intervals",
        "mean_equal_capital_roi",
        default={},
    ) or {}
    random_position = _nested(
        random_summary,
        "model_vs_random",
        "mean_equal_capital_roi",
        default={},
    ) or {}
    return {
        "direct_split_auc": direct.get("auc"),
        "direct_split_average_precision": direct.get("average_precision"),
        "direct_split_precision_0_5": direct.get("precision_0_5"),
        "eligible_wallet_rows": direct.get("test_rows"),
        "test_positives": direct.get("test_positives"),
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
        "bootstrap_roi_ci_p025": bootstrap_roi.get("p025"),
        "bootstrap_roi_ci_p975": bootstrap_roi.get("p975"),
        "mean_unique_markets": wf_agg.get("mean_unique_markets"),
        "top_market_filled_notional_share": wf_agg.get("mean_top_market_share"),
        **event_risk,
    }


def archive_current_artifacts(run_dir: Path) -> None:
    archive_artifacts(run_dir, ARTIFACTS_TO_ARCHIVE)


def archive_artifacts(run_dir: Path, artifact_names: Iterable[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in artifact_names:
        src = RESULTS_DIR / name
        if src.exists():
            shutil.copy2(src, run_dir / name)


def run_one_ablation(
    spec: AblationSpec,
    rolling_folds: int,
    baseline_seed_count: int,
    fold_match_count_per_fold: int,
    fold_match_bootstrap_samples: int,
    top_percentile: float,
    run_root: Path = EVENT_ABLATION_DIR,
) -> AblationResult:
    feature_set = register_ablation_feature_set(spec.name, spec.extra_features)
    run_dir = run_root / spec.name
    direct = run_direct_split_lr(feature_set)
    rolling = run_rolling_origin_evaluation(
        feature_set=feature_set,
        n_folds=rolling_folds,
    )
    archive_artifacts(run_dir, ROLLING_ARTIFACTS)
    walk_forward = run_model_informed_walk_forward(top_percentile=top_percentile)
    archive_artifacts(run_dir, WALK_FORWARD_ARTIFACTS)
    random_summary = run_random_baseline_stability(
        seed_count=baseline_seed_count,
        top_percentile=top_percentile,
    )
    archive_artifacts(run_dir, RANDOM_ARTIFACTS)
    fold_matched = run_fold_matched_random_baseline_analysis(
        top_percentile=top_percentile,
        match_count_per_fold=fold_match_count_per_fold,
        bootstrap_samples=fold_match_bootstrap_samples,
    )
    archive_artifacts(run_dir, FOLD_MATCH_ARTIFACTS)
    ledger = collect_model_top_pct_ledger(
        run_dir=run_dir,
        top_percentile=top_percentile,
        predictions_path=run_dir / "rolling_origin_predictions.csv",
    )
    event_risk = drop_top_event_metrics(ledger)
    metrics = _summarise_current_run(
        direct=direct,
        rolling=rolling,
        walk_forward=walk_forward,
        random_summary=random_summary,
        fold_matched=fold_matched,
        event_risk=event_risk,
        top_percentile=top_percentile,
    )
    (run_dir / "event_ablation_summary.json").write_text(
        json.dumps(
            {
                "name": spec.name,
                "feature_set": feature_set,
                "extra_features": spec.extra_features,
                "metrics": metrics,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return AblationResult(spec.name, feature_set, spec.extra_features, run_dir, metrics)


def write_outputs(
    results: list[AblationResult],
    output_dir: Path = OUTPUT_DIR,
    run_name: str = "event_ablation",
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        rows.append(
            {
                "ablation": result.name,
                "feature_set": result.feature_set,
                "extra_features": ",".join(result.extra_features),
                "run_dir": str(result.run_dir),
                **result.metrics,
            }
        )
    table = pd.DataFrame(rows)
    table_path = output_dir / f"{run_name}_table.csv"
    table.to_csv(table_path, index=False)

    metric_cols = [
        "direct_split_auc",
        "direct_split_average_precision",
        "rolling_auc",
        "rolling_average_precision",
        "precision_at_20",
        "top_10_percent_precision",
        "walk_forward_total_net_profit",
        "walk_forward_mean_equal_capital_roi",
        "model_vs_random_percentile",
        "bootstrap_prob_beats_matched_random",
        "drop_top_1_profit_event_remaining_net_profit",
        "drop_top_3_profit_events_remaining_net_profit",
        "top_event_profit_share",
    ]
    available = [col for col in metric_cols if col in table.columns]
    markdown_table = _markdown_table(table[["ablation", "extra_features", *available]])
    report = "\n".join(
        [
            "# Event Ablation Report",
            "",
            "This event ablation keeps `live` as the primary baseline and adds only small event-feature subsets.",
            "Direct split metrics are diagnostic; rolling-origin, walk-forward, matched-random, bootstrap, and drop-top-event robustness carry the academic weight.",
            "",
            markdown_table,
            "",
            "Acceptance rule: keep an event feature only if rolling AP, precision@k, walk-forward robustness, matched-random position, bootstrap probability, and drop-top-event sensitivity do not materially degrade versus `live`.",
        ]
    )
    report_path = output_dir / f"{run_name}_report.md"
    report_path.write_text(report, encoding="utf-8")
    return table_path, report_path


def _markdown_table(df: pd.DataFrame) -> str:
    """Render a small Markdown table without requiring pandas' tabulate extra."""
    if df.empty:
        return "_No ablation rows._"
    headers = [str(col) for col in df.columns]
    rows = []
    for _, row in df.iterrows():
        rows.append([_format_markdown_cell(row[col]) for col in df.columns])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _format_markdown_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live-plus-event feature ablations")
    parser.add_argument("--rolling-folds", type=int, default=5)
    parser.add_argument("--baseline-seed-count", type=int, default=10)
    parser.add_argument("--fold-match-count-per-fold", type=int, default=3)
    parser.add_argument("--fold-match-bootstrap-samples", type=int, default=500)
    parser.add_argument("--top-percentile", type=float, default=0.10)
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated ablation names to run. Default runs the full matrix.",
    )
    parser.add_argument(
        "--paper-candidates",
        action="store_true",
        help="Run only live, event_diversification, and mean_markets_per_event_traded.",
    )
    parser.add_argument(
        "--rich-candidates",
        action="store_true",
        help="Run richer metadata candidates from SII markets and local Polymarket events.",
    )
    parser.add_argument(
        "--run-name",
        default="event_ablation",
        help="Output prefix and run directory stem. Default: event_ablation.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose INFO logs from backtesting internals.",
    )
    args = parser.parse_args()

    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    if args.paper_candidates and args.rich_candidates:
        raise SystemExit("--paper-candidates and --rich-candidates are mutually exclusive")

    selected = EVENT_ABLATION_SPECS
    if args.paper_candidates:
        names = set(PAPER_CANDIDATES)
        selected = [spec for spec in EVENT_ABLATION_SPECS if spec.name in names]
    if args.rich_candidates:
        names = set(RICH_CANDIDATES)
        selected = [spec for spec in EVENT_ABLATION_SPECS if spec.name in names]
    if args.only.strip():
        names = {name.strip() for name in args.only.split(",") if name.strip()}
        selected = [spec for spec in EVENT_ABLATION_SPECS if spec.name in names]
        missing = sorted(names - {spec.name for spec in selected})
        if missing:
            raise ValueError(f"Unknown ablation name(s): {missing}")

    run_root = OUTPUT_DIR / f"{args.run_name}_runs"
    results = [
        run_one_ablation(
            spec,
            rolling_folds=args.rolling_folds,
            baseline_seed_count=args.baseline_seed_count,
            fold_match_count_per_fold=args.fold_match_count_per_fold,
            fold_match_bootstrap_samples=args.fold_match_bootstrap_samples,
            top_percentile=args.top_percentile,
            run_root=run_root,
        )
        for spec in selected
    ]
    table_path, report_path = write_outputs(results, run_name=args.run_name)
    print(f"Wrote {table_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
