"""
Author: AI Assistant
Date: 2026-05-12
Description: Phase 0 baseline audit for project reproducibility and phase validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
AUDIT_DIR = PROJECT_ROOT / "docs" / "audit"
BASELINE_JSON = RESULTS_DIR / "phase0_baseline.json"
BASELINE_MD = AUDIT_DIR / "phase0_baseline.md"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(cmd: List[str], timeout: int = 30) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {"cmd": cmd, "returncode": -1, "stdout": "", "stderr": str(exc)}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc)}


def _file_info(path: Path) -> Dict[str, Any]:
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        if path.exists()
        else None,
    }


def _csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            return sum(1 for _ in csv.DictReader(fh))
    except Exception:
        return 0


def _csv_has_columns(path: Path, columns: List[str]) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = set(reader.fieldnames or [])
        return all(column in fieldnames for column in columns)
    except Exception:
        return False


def _phase_checks() -> Dict[str, Any]:
    checks: Dict[str, Any] = {}

    try:
        from src.backtesting.run_backtest import main as backtest_main
        checks["phase1_backtest_entrypoint_imports"] = callable(backtest_main)
    except Exception as exc:
        checks["phase1_backtest_entrypoint_imports"] = False
        checks["phase1_backtest_entrypoint_error"] = str(exc)

    try:
        from src.data_ingestion.polymarket_client import DataFetchError
        checks["phase1_data_fetch_error_defined"] = issubclass(DataFetchError, RuntimeError)
    except Exception as exc:
        checks["phase1_data_fetch_error_defined"] = False
        checks["phase1_data_fetch_error"] = str(exc)

    try:
        from src.visualization.api import app
        paths = {route.path for route in app.routes}
        checks["phase1_backtest_api_routes"] = all(
            path in paths for path in ["/api/backtest/status", "/api/backtest/results"]
        )
        checks["phase6_walk_forward_api_route"] = "/api/backtest/walk-forward" in paths
    except Exception as exc:
        checks["phase1_backtest_api_routes"] = False
        checks["phase1_backtest_api_error"] = str(exc)
        checks["phase6_walk_forward_api_route"] = False

    checks["phase2_data_quality_report_exists"] = (RESULTS_DIR / "data_quality_report.json").exists()
    checks["phase2_manifest_exists"] = (PROJECT_ROOT / "data" / "manifest.json").exists()
    checks["phase2_data_contract_exists"] = (PROJECT_ROOT / "docs" / "data_contract.md").exists()

    label_summary = _read_json(RESULTS_DIR / "label_summary.json")
    dataset_completeness_summary = _read_json(RESULTS_DIR / "dataset_completeness_report.json")
    checks["phase3_label_summary_exists"] = bool(label_summary)
    checks["phase3_label_independence"] = label_summary.get("label_independence")
    checks["phase3_single_class_splits"] = label_summary.get("single_class_splits", [])
    checks["phase7_future_return_label_active"] = (
        label_summary.get("label_strategy") == "future_return"
        and label_summary.get("label_independence") == "independent_future_return"
    )
    checks["phase7_future_return_trainable"] = bool(label_summary.get("is_trainable_for_binary_eval"))
    checks["phase7_future_return_coverage"] = label_summary.get("future_label_coverage", {})

    try:
        from src.models.feature_sets import FEATURE_LEAKAGE_CLASS, get_feature_columns
        live = get_feature_columns("live")
        checks["phase4_live_feature_set_default_safe"] = (
            "total_roi" not in live
            and "win_rate" not in live
            and all(FEATURE_LEAKAGE_CLASS[col] == "pre_resolution_live" for col in live)
        )
        checks["phase4_live_features"] = live
    except Exception as exc:
        checks["phase4_live_feature_set_default_safe"] = False
        checks["phase4_feature_set_error"] = str(exc)

    try:
        from src.models.trainer import ModelTrainer
        trainer = ModelTrainer()
        try:
            trainer.load_data()
            checks["phase5_training_gate_validates_current_data"] = bool(
                label_summary.get("is_trainable_for_binary_eval")
            )
            checks["phase5_training_gate_mode"] = "accepted_trainable_data"
        except ValueError as exc:
            checks["phase5_training_gate_validates_current_data"] = (
                "Single-class splits" in str(exc)
                and not label_summary.get("is_trainable_for_binary_eval", False)
            )
            checks["phase5_training_gate_mode"] = "rejected_untrainable_data"
            checks["phase5_training_gate_error"] = str(exc)
    except Exception as exc:
        checks["phase5_training_gate_validates_current_data"] = False
        checks["phase5_training_gate_error"] = str(exc)

    try:
        from src.backtesting.engine import BacktestAssumptions, LEDGER_COLUMNS
        from src.backtesting.walk_forward import run_walk_forward

        assumptions = BacktestAssumptions()
        checks["phase6_backtest_assumptions_defined"] = (
            assumptions.latency_minutes > 0
            and assumptions.trade_size > 0
            and assumptions.settlement == "binary_payout_at_resolution"
        )
        checks["phase6_ledger_schema_defined"] = all(
            col in LEDGER_COLUMNS
            for col in ["strategy", "expert_bet", "final_resolution", "fee", "net_profit"]
        )
        checks["phase6_walk_forward_imports"] = callable(run_walk_forward)
        checks["phase13_execution_constraints_defined"] = (
            assumptions.max_trade_fraction_of_balance > 0
            and assumptions.max_market_exposure_fraction > 0
            and assumptions.max_participation_rate > 0
            and assumptions.liquidity_lookback_minutes > 0
            and assumptions.price_impact_bps >= 0
        )
        checks["phase13_execution_ledger_defined"] = all(
            col in LEDGER_COLUMNS
            for col in [
                "filled_notional",
                "recent_market_volume",
                "participation_rate",
                "liquidity_cap",
                "capital_cap",
                "market_exposure_after",
            ]
        )
    except Exception as exc:
        checks["phase6_backtest_assumptions_defined"] = False
        checks["phase6_ledger_schema_defined"] = False
        checks["phase6_walk_forward_imports"] = False
        checks["phase13_execution_constraints_defined"] = False
        checks["phase13_execution_ledger_defined"] = False
        checks["phase6_import_error"] = str(exc)

    walk_forward_summary = _read_json(RESULTS_DIR / "walk_forward_summary.json")
    checks["phase6_walk_forward_summary_exists"] = bool(walk_forward_summary)
    checks["phase6_temporal_guard_present"] = bool(walk_forward_summary.get("temporal_guard"))
    checks["phase6_walk_forward_formal_validity"] = (
        walk_forward_summary.get("temporal_guard", {}).get("formal_validity")
    )
    checks["phase6_backtesting_doc_exists"] = (PROJECT_ROOT / "docs" / "backtesting.md").exists()
    checks["phase14_walk_forward_equal_capital"] = any(
        "mean_equal_capital_roi" in strategy_summary
        and "total_equal_capital_net_profit" in strategy_summary
        for strategy_summary in walk_forward_summary.get("aggregate", {}).values()
    )

    sensitivity_summary = _read_json(RESULTS_DIR / "future_label_sensitivity_summary.json")
    rolling_summary = _read_json(RESULTS_DIR / "rolling_origin_summary.json")
    phase8_10_summary = _read_json(RESULTS_DIR / "phase8_10_summary.json")
    model_walk_forward_summary = _read_json(RESULTS_DIR / "model_informed_walk_forward_summary.json")
    random_stability_summary = _read_json(RESULTS_DIR / "random_baseline_stability_summary.json")
    exposure_concentration_summary = _read_json(RESULTS_DIR / "exposure_concentration_summary.json")
    stratified_random_summary = _read_json(RESULTS_DIR / "stratified_random_baseline_summary.json")
    fold_matched_random_summary = _read_json(RESULTS_DIR / "fold_matched_random_baseline_summary.json")
    academic_evidence_summary = _read_json(RESULTS_DIR / "academic_evidence_summary.json")
    dataset_completeness_summary = _read_json(RESULTS_DIR / "dataset_completeness_report.json")
    threshold_policy_csv = RESULTS_DIR / "threshold_policy_results.csv"
    model_walk_forward_results_csv = RESULTS_DIR / "model_informed_walk_forward_results.csv"
    random_stability_csv = RESULTS_DIR / "random_baseline_stability.csv"
    random_seed_summary_csv = RESULTS_DIR / "random_baseline_seed_summary.csv"
    stratified_random_matches_csv = RESULTS_DIR / "stratified_random_baseline_matches.csv"
    fold_matched_random_matches_csv = RESULTS_DIR / "fold_matched_random_baseline_matches.csv"
    fold_matched_random_rank_csv = RESULTS_DIR / "fold_matched_random_baseline_rank_summary.csv"
    fold_matched_random_bootstrap_csv = RESULTS_DIR / "fold_matched_random_bootstrap.csv"
    academic_evidence_csv = RESULTS_DIR / "academic_evidence_table.csv"
    academic_evidence_md = PROJECT_ROOT / "docs" / "audit" / "phase20_academic_evidence.md"
    dataset_completeness_md = PROJECT_ROOT / "docs" / "audit" / "phase24_dataset_completeness.md"
    checks["phase8_label_sensitivity_exists"] = bool(sensitivity_summary)
    checks["phase8_recommended_config_present"] = bool(sensitivity_summary.get("recommended_config"))
    checks["phase9_rolling_origin_exists"] = bool(rolling_summary)
    checks["phase9_rolling_origin_valid_folds"] = int(rolling_summary.get("valid_fold_count", 0) or 0)
    checks["phase10_sensitivity_summary_exists"] = bool(phase8_10_summary)
    checks["phase10_docs_exist"] = (PROJECT_ROOT / "docs" / "audit" / "phase8_10_revalidation.md").exists()
    threshold_policies = rolling_summary.get("threshold_policies", {})
    checks["phase11_threshold_policies_exist"] = bool(threshold_policies) and threshold_policy_csv.exists()
    checks["phase11_threshold_policy_count"] = len(threshold_policies)
    checks["phase11_threshold_policy_rows"] = _csv_row_count(threshold_policy_csv)
    checks["phase12_model_walk_forward_exists"] = bool(model_walk_forward_summary) and model_walk_forward_results_csv.exists()
    checks["phase12_model_walk_forward_status"] = model_walk_forward_summary.get("status")
    checks["phase12_model_strategy_present"] = any(
        str(strategy).startswith("model_top_")
        for strategy in model_walk_forward_summary.get("aggregate", {})
    )
    checks["phase13_model_walk_forward_fill_metrics"] = any(
        "mean_fill_rate" in strategy_summary
        and "total_filled_notional" in strategy_summary
        for strategy_summary in model_walk_forward_summary.get("aggregate", {}).values()
    )
    checks["phase14_model_walk_forward_equal_capital"] = any(
        "mean_equal_capital_roi" in strategy_summary
        and "total_equal_capital_net_profit" in strategy_summary
        for strategy_summary in model_walk_forward_summary.get("aggregate", {}).values()
    )
    checks["phase15_random_baseline_stability_exists"] = (
        bool(random_stability_summary)
        and random_stability_csv.exists()
        and random_seed_summary_csv.exists()
    )
    checks["phase15_random_baseline_seed_count"] = int(random_stability_summary.get("seed_count", 0) or 0)
    checks["phase15_random_baseline_comparison"] = (
        random_stability_summary.get("model_vs_random", {})
        .get("mean_equal_capital_roi", {})
        .get("status") == "ok"
    )
    checks["phase16_model_walk_forward_concentration"] = any(
        "mean_market_hhi" in strategy_summary
        and "mean_top_market_share" in strategy_summary
        and "mean_effective_markets" in strategy_summary
        for strategy_summary in model_walk_forward_summary.get("aggregate", {}).values()
    )
    checks["phase16_random_seed_concentration"] = (
        random_seed_summary_csv.exists()
        and _csv_has_columns(
            random_seed_summary_csv,
            ["mean_market_hhi", "mean_top_market_share", "mean_effective_markets"],
        )
    )
    checks["phase16_exposure_concentration_summary"] = (
        exposure_concentration_summary.get("status") == "ok"
        and exposure_concentration_summary.get("model_vs_random", {})
        .get("mean_market_hhi", {})
        .get("status") == "ok"
    )
    checks["phase17_stratified_random_summary"] = (
        stratified_random_summary.get("status") == "ok"
        and stratified_random_matches_csv.exists()
    )
    checks["phase17_stratified_random_match_count"] = int(
        stratified_random_summary.get("matched_seed_count", 0) or 0
    )
    checks["phase17_stratified_random_comparison"] = (
        stratified_random_summary.get("model_vs_matched_random", {})
        .get("mean_equal_capital_roi", {})
        .get("status") == "ok"
    )
    checks["phase18_fold_matched_random_summary"] = (
        fold_matched_random_summary.get("status") == "ok"
        and fold_matched_random_matches_csv.exists()
        and fold_matched_random_rank_csv.exists()
    )
    checks["phase18_fold_matched_rank_count"] = int(
        fold_matched_random_summary.get("matched_rank_count", 0) or 0
    )
    checks["phase18_fold_matched_random_comparison"] = (
        fold_matched_random_summary.get("model_vs_fold_matched_random", {})
        .get("mean_equal_capital_roi", {})
        .get("status") == "ok"
    )
    checks["phase19_expanded_random_seed_pool"] = int(random_stability_summary.get("seed_count", 0) or 0) >= 20
    checks["phase19_fold_matched_bootstrap"] = (
        fold_matched_random_summary.get("bootstrap_model_minus_matched_random", {})
        .get("status") == "ok"
        and fold_matched_random_bootstrap_csv.exists()
    )
    checks["phase19_bootstrap_ci_present"] = (
        fold_matched_random_summary.get("bootstrap_model_minus_matched_random", {})
        .get("intervals", {})
        .get("total_net_profit", {})
        .get("status") == "ok"
    )
    checks["phase19_caliper_tracking"] = "caliper_passed_folds" in fold_matched_random_summary
    checks["phase20_academic_evidence_summary"] = academic_evidence_summary.get("status") == "ok"
    checks["phase20_academic_evidence_rows"] = int(
        academic_evidence_summary.get("evidence_row_count", 0) or 0
    )
    checks["phase20_academic_evidence_table"] = (
        academic_evidence_csv.exists()
        and _csv_row_count(academic_evidence_csv) >= 8
        and _csv_has_columns(
            academic_evidence_csv,
            [
                "evidence_family",
                "research_question",
                "estimand",
                "design",
                "metric",
                "interpretation",
                "caveat",
            ],
        )
    )
    checks["phase20_academic_guardrails"] = all(
        guardrail in academic_evidence_summary.get("academic_guardrails", [])
        for guardrail in [
            "future_outcome_label",
            "pre_resolution_live_features",
            "rolling_origin_validation",
            "fold_matched_bootstrap_intervals",
        ]
    )
    phase21_24_gates = dataset_completeness_summary.get("phase21_24_gates", {})
    checks["phase21_collection_provenance_tables"] = bool(
        phase21_24_gates.get("collection_provenance_tables_exist")
    )
    checks["phase22_outcome_preserved"] = bool(
        phase21_24_gates.get("db_outcome_preserved")
        and phase21_24_gates.get("cleaned_outcome_preserved")
    )
    checks["phase23_resolution_schema_extended"] = bool(
        phase21_24_gates.get("resolution_schema_extended")
    )
    checks["phase24_dataset_completeness_report"] = (
        dataset_completeness_summary.get("status") == "ok"
        and (RESULTS_DIR / "dataset_completeness_report.json").exists()
        and dataset_completeness_md.exists()
    )
    checks["phase12_docs_exist"] = (PROJECT_ROOT / "docs" / "audit" / "phase11_12_revalidation.md").exists()
    checks["phase8_12_summary_exists"] = (RESULTS_DIR / "phase8_12_summary.json").exists()
    checks["phase13_docs_exist"] = (PROJECT_ROOT / "docs" / "audit" / "phase13_revalidation.md").exists()
    checks["phase14_docs_exist"] = (PROJECT_ROOT / "docs" / "audit" / "phase14_revalidation.md").exists()
    checks["phase15_docs_exist"] = (PROJECT_ROOT / "docs" / "audit" / "phase15_revalidation.md").exists()
    checks["phase16_docs_exist"] = (PROJECT_ROOT / "docs" / "audit" / "phase16_revalidation.md").exists()
    checks["phase17_docs_exist"] = (PROJECT_ROOT / "docs" / "audit" / "phase17_revalidation.md").exists()
    checks["phase18_docs_exist"] = (PROJECT_ROOT / "docs" / "audit" / "phase18_revalidation.md").exists()
    checks["phase19_docs_exist"] = (PROJECT_ROOT / "docs" / "audit" / "phase19_revalidation.md").exists()
    checks["phase20_docs_exist"] = academic_evidence_md.exists()
    checks["phase24_docs_exist"] = dataset_completeness_md.exists()

    return checks


def build_baseline() -> Dict[str, Any]:
    from src.data_quality.audit import build_data_quality_report, build_manifest

    git_status = _run(["git", "status", "--short"])
    git_branch = _run(["git", "branch", "--show-current"])
    git_head = _run(["git", "rev-parse", "--short", "HEAD"])

    data_quality = build_data_quality_report()
    manifest = build_manifest()
    label_summary = _read_json(RESULTS_DIR / "label_summary.json")
    dataset_completeness_summary = _read_json(RESULTS_DIR / "dataset_completeness_report.json")

    key_artifacts = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "data" / "research.db",
        PROJECT_ROOT / "data" / "features" / "model_input.csv",
        PROJECT_ROOT / "results" / "label_summary.json",
        PROJECT_ROOT / "results" / "data_quality_report.json",
        PROJECT_ROOT / "results" / "training_report.json",
        PROJECT_ROOT / "results" / "backtest_metrics.json",
        PROJECT_ROOT / "results" / "walk_forward_summary.json",
        PROJECT_ROOT / "results" / "walk_forward_results.csv",
        PROJECT_ROOT / "results" / "future_label_sensitivity_summary.json",
        PROJECT_ROOT / "results" / "rolling_origin_summary.json",
        PROJECT_ROOT / "results" / "rolling_origin_predictions.csv",
        PROJECT_ROOT / "results" / "threshold_policy_results.csv",
        PROJECT_ROOT / "results" / "model_informed_walk_forward_summary.json",
        PROJECT_ROOT / "results" / "model_informed_walk_forward_results.csv",
        PROJECT_ROOT / "results" / "random_baseline_stability_summary.json",
        PROJECT_ROOT / "results" / "random_baseline_stability.csv",
        PROJECT_ROOT / "results" / "random_baseline_seed_summary.csv",
        PROJECT_ROOT / "results" / "exposure_concentration_summary.json",
        PROJECT_ROOT / "results" / "stratified_random_baseline_summary.json",
        PROJECT_ROOT / "results" / "stratified_random_baseline_matches.csv",
        PROJECT_ROOT / "results" / "fold_matched_random_baseline_summary.json",
        PROJECT_ROOT / "results" / "fold_matched_random_baseline_matches.csv",
        PROJECT_ROOT / "results" / "fold_matched_random_baseline_rank_summary.csv",
        PROJECT_ROOT / "results" / "fold_matched_random_bootstrap.csv",
        PROJECT_ROOT / "results" / "academic_evidence_table.csv",
        PROJECT_ROOT / "results" / "academic_evidence_summary.json",
        PROJECT_ROOT / "results" / "dataset_completeness_report.json",
        PROJECT_ROOT / "results" / "phase8_10_summary.json",
        PROJECT_ROOT / "results" / "phase8_12_summary.json",
    ]

    return {
        "generated_at": _utc_now(),
        "project_root": str(PROJECT_ROOT),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "git": {
            "branch": git_branch.get("stdout"),
            "head": git_head.get("stdout"),
            "status_short": git_status.get("stdout", "").splitlines(),
            "is_dirty": bool(git_status.get("stdout", "").strip()),
        },
        "key_artifacts": [_file_info(path) for path in key_artifacts],
        "data_quality": data_quality,
        "label_summary": label_summary,
        "dataset_completeness": dataset_completeness_summary,
        "manifest_artifact_count": len(manifest.get("artifacts", [])),
        "phase_checks": _phase_checks(),
    }


def _md_bool(value: Any) -> str:
    return "PASS" if value else "FAIL"


def render_markdown(baseline: Dict[str, Any]) -> str:
    checks = baseline["phase_checks"]
    dq = baseline.get("data_quality", {})
    dataset_completeness_summary = baseline.get("dataset_completeness", {})
    tx = dq.get("transactions", {})
    label = baseline.get("label_summary", {})

    lines = [
        "# Phase 0 Baseline Audit",
        "",
        f"Generated at: `{baseline['generated_at']}`",
        f"Project root: `{baseline['project_root']}`",
        "",
        "## Environment",
        "",
        f"- Python: `{baseline['environment']['python'].split()[0]}`",
        f"- Platform: `{baseline['environment']['platform']}`",
        f"- Git branch: `{baseline['git']['branch']}`",
        f"- Git HEAD: `{baseline['git']['head']}`",
        f"- Dirty worktree: `{baseline['git']['is_dirty']}`",
        "",
        "## Data Baseline",
        "",
        f"- Transactions: `{tx.get('row_count')}`",
        f"- Unique transaction ids: `{tx.get('unique_transaction_ids')}`",
        f"- Duplicate transaction ids: `{tx.get('duplicate_transaction_ids')}`",
        f"- Unique wallets: `{tx.get('unique_wallets')}`",
        f"- Unique markets: `{tx.get('unique_markets')}`",
        f"- Time range: `{tx.get('min_timestamp')}` to `{tx.get('max_timestamp')}`",
        f"- Invalid prices: `{tx.get('invalid_price_count')}`",
        f"- Invalid amounts: `{tx.get('invalid_amount_count')}`",
        "",
        "## Label Baseline",
        "",
        f"- Label strategy: `{label.get('label_strategy')}`",
        f"- Label independence: `{label.get('label_independence')}`",
        f"- Single-class splits: `{label.get('single_class_splits')}`",
        f"- Trainable for binary evaluation: `{label.get('is_trainable_for_binary_eval')}`",
        "",
        "## Phase 1-24 Gate Snapshot",
        "",
        "| Gate | Status |",
        "|---|---|",
        f"| Phase 1 backtest entrypoint imports | {_md_bool(checks.get('phase1_backtest_entrypoint_imports'))} |",
        f"| Phase 1 DataFetchError defined | {_md_bool(checks.get('phase1_data_fetch_error_defined'))} |",
        f"| Phase 1 backtest API routes | {_md_bool(checks.get('phase1_backtest_api_routes'))} |",
        f"| Phase 2 data quality report | {_md_bool(checks.get('phase2_data_quality_report_exists'))} |",
        f"| Phase 2 manifest | {_md_bool(checks.get('phase2_manifest_exists'))} |",
        f"| Phase 2 data contract | {_md_bool(checks.get('phase2_data_contract_exists'))} |",
        f"| Phase 3 label summary | {_md_bool(checks.get('phase3_label_summary_exists'))} |",
        f"| Phase 4 live feature set safe | {_md_bool(checks.get('phase4_live_feature_set_default_safe'))} |",
        f"| Phase 5 current-data training gate | {_md_bool(checks.get('phase5_training_gate_validates_current_data'))} |",
        f"| Phase 6 backtest assumptions | {_md_bool(checks.get('phase6_backtest_assumptions_defined'))} |",
        f"| Phase 6 ledger schema | {_md_bool(checks.get('phase6_ledger_schema_defined'))} |",
        f"| Phase 6 walk-forward import | {_md_bool(checks.get('phase6_walk_forward_imports'))} |",
        f"| Phase 6 walk-forward API route | {_md_bool(checks.get('phase6_walk_forward_api_route'))} |",
        f"| Phase 6 walk-forward summary | {_md_bool(checks.get('phase6_walk_forward_summary_exists'))} |",
        f"| Phase 6 temporal guard | {_md_bool(checks.get('phase6_temporal_guard_present'))} |",
        f"| Phase 6 backtesting docs | {_md_bool(checks.get('phase6_backtesting_doc_exists'))} |",
        f"| Phase 7 future-return label active | {_md_bool(checks.get('phase7_future_return_label_active'))} |",
        f"| Phase 7 future-return trainable | {_md_bool(checks.get('phase7_future_return_trainable'))} |",
        f"| Phase 8 label sensitivity | {_md_bool(checks.get('phase8_label_sensitivity_exists'))} |",
        f"| Phase 8 recommended config | {_md_bool(checks.get('phase8_recommended_config_present'))} |",
        f"| Phase 9 rolling-origin folds | {_md_bool(checks.get('phase9_rolling_origin_valid_folds', 0) > 0)} |",
        f"| Phase 10 sensitivity summary | {_md_bool(checks.get('phase10_sensitivity_summary_exists'))} |",
        f"| Phase 8-10 docs | {_md_bool(checks.get('phase10_docs_exist'))} |",
        f"| Phase 11 threshold policies | {_md_bool(checks.get('phase11_threshold_policies_exist'))} |",
        f"| Phase 12 model-informed walk-forward | {_md_bool(checks.get('phase12_model_walk_forward_exists'))} |",
        f"| Phase 12 model strategy present | {_md_bool(checks.get('phase12_model_strategy_present'))} |",
        f"| Phase 11-12 docs | {_md_bool(checks.get('phase12_docs_exist'))} |",
        f"| Phase 8-12 summary | {_md_bool(checks.get('phase8_12_summary_exists'))} |",
        f"| Phase 13 execution constraints | {_md_bool(checks.get('phase13_execution_constraints_defined'))} |",
        f"| Phase 13 execution ledger | {_md_bool(checks.get('phase13_execution_ledger_defined'))} |",
        f"| Phase 13 model walk-forward fill metrics | {_md_bool(checks.get('phase13_model_walk_forward_fill_metrics'))} |",
        f"| Phase 13 docs | {_md_bool(checks.get('phase13_docs_exist'))} |",
        f"| Phase 14 walk-forward equal-capital metrics | {_md_bool(checks.get('phase14_walk_forward_equal_capital'))} |",
        f"| Phase 14 model walk-forward equal-capital metrics | {_md_bool(checks.get('phase14_model_walk_forward_equal_capital'))} |",
        f"| Phase 14 docs | {_md_bool(checks.get('phase14_docs_exist'))} |",
        f"| Phase 15 random baseline stability | {_md_bool(checks.get('phase15_random_baseline_stability_exists'))} |",
        f"| Phase 15 model-vs-random comparison | {_md_bool(checks.get('phase15_random_baseline_comparison'))} |",
        f"| Phase 15 docs | {_md_bool(checks.get('phase15_docs_exist'))} |",
        f"| Phase 16 model concentration metrics | {_md_bool(checks.get('phase16_model_walk_forward_concentration'))} |",
        f"| Phase 16 random seed concentration metrics | {_md_bool(checks.get('phase16_random_seed_concentration'))} |",
        f"| Phase 16 exposure concentration summary | {_md_bool(checks.get('phase16_exposure_concentration_summary'))} |",
        f"| Phase 16 docs | {_md_bool(checks.get('phase16_docs_exist'))} |",
        f"| Phase 17 stratified random summary | {_md_bool(checks.get('phase17_stratified_random_summary'))} |",
        f"| Phase 17 matched-random comparison | {_md_bool(checks.get('phase17_stratified_random_comparison'))} |",
        f"| Phase 17 docs | {_md_bool(checks.get('phase17_docs_exist'))} |",
        f"| Phase 18 fold-matched random summary | {_md_bool(checks.get('phase18_fold_matched_random_summary'))} |",
        f"| Phase 18 fold-matched comparison | {_md_bool(checks.get('phase18_fold_matched_random_comparison'))} |",
        f"| Phase 18 docs | {_md_bool(checks.get('phase18_docs_exist'))} |",
        f"| Phase 19 expanded random seed pool | {_md_bool(checks.get('phase19_expanded_random_seed_pool'))} |",
        f"| Phase 19 fold-matched bootstrap | {_md_bool(checks.get('phase19_fold_matched_bootstrap'))} |",
        f"| Phase 19 bootstrap CI | {_md_bool(checks.get('phase19_bootstrap_ci_present'))} |",
        f"| Phase 19 caliper tracking | {_md_bool(checks.get('phase19_caliper_tracking'))} |",
        f"| Phase 19 docs | {_md_bool(checks.get('phase19_docs_exist'))} |",
        f"| Phase 20 academic evidence summary | {_md_bool(checks.get('phase20_academic_evidence_summary'))} |",
        f"| Phase 20 academic evidence table | {_md_bool(checks.get('phase20_academic_evidence_table'))} |",
        f"| Phase 20 academic guardrails | {_md_bool(checks.get('phase20_academic_guardrails'))} |",
        f"| Phase 20 docs | {_md_bool(checks.get('phase20_docs_exist'))} |",
        f"| Phase 21 collection provenance tables | {_md_bool(checks.get('phase21_collection_provenance_tables'))} |",
        f"| Phase 22 outcome preservation | {_md_bool(checks.get('phase22_outcome_preserved'))} |",
        f"| Phase 23 extended resolution schema | {_md_bool(checks.get('phase23_resolution_schema_extended'))} |",
        f"| Phase 24 dataset completeness report | {_md_bool(checks.get('phase24_dataset_completeness_report'))} |",
        f"| Phase 24 docs | {_md_bool(checks.get('phase24_docs_exist'))} |",
        "",
        "## Notes",
        "",
        f"- Phase 5 gate mode: `{checks.get('phase5_training_gate_mode')}`.",
        f"- Phase 11 threshold policies tracked: `{checks.get('phase11_threshold_policy_count')}` policies, `{checks.get('phase11_threshold_policy_rows')}` fold-policy rows.",
        f"- Phase 12 model-informed walk-forward status: `{checks.get('phase12_model_walk_forward_status')}`.",
        f"- Phase 13 model-informed walk-forward fill metrics present: `{checks.get('phase13_model_walk_forward_fill_metrics')}`.",
        f"- Phase 14 model-informed equal-capital metrics present: `{checks.get('phase14_model_walk_forward_equal_capital')}`.",
        f"- Phase 15 random baseline seed count: `{checks.get('phase15_random_baseline_seed_count')}`.",
        f"- Phase 16 concentration summary present: `{checks.get('phase16_exposure_concentration_summary')}`.",
        f"- Phase 17 matched random seed count: `{checks.get('phase17_stratified_random_match_count')}`.",
        f"- Phase 18 fold-matched rank count: `{checks.get('phase18_fold_matched_rank_count')}`.",
        f"- Phase 19 expanded random seed pool: `{checks.get('phase19_expanded_random_seed_pool')}`.",
        f"- Phase 19 bootstrap CI present: `{checks.get('phase19_bootstrap_ci_present')}`.",
        f"- Phase 20 academic evidence rows: `{checks.get('phase20_academic_evidence_rows')}`.",
        f"- Phase 24 known dataset gaps: `{len(dataset_completeness_summary.get('known_gaps', []))}`.",
        "- Phase 6 walk-forward formal validity is expected to pass only when labels are independent resolution or future-return labels.",
        "- To run diagnostics on current artifacts, use `POLYMARKET_ALLOW_SINGLE_CLASS_SPLITS=true`.",
    ]
    return "\n".join(lines) + "\n"


def write_baseline() -> Dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    baseline = build_baseline()
    BASELINE_JSON.write_text(json.dumps(baseline, indent=2, default=str), encoding="utf-8")
    BASELINE_MD.write_text(render_markdown(baseline), encoding="utf-8")
    return baseline


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 0 project baseline audit")
    parser.add_argument("--json-only", action="store_true", help="Print JSON to stdout without writing artifacts")
    args = parser.parse_args()

    if args.json_only:
        print(json.dumps(build_baseline(), indent=2, default=str))
    else:
        baseline = write_baseline()
        print(f"Wrote {BASELINE_JSON.relative_to(PROJECT_ROOT)}")
        print(f"Wrote {BASELINE_MD.relative_to(PROJECT_ROOT)}")
        print(json.dumps(baseline["phase_checks"], indent=2, default=str))


if __name__ == "__main__":
    main()
