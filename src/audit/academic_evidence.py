"""
Author: AI Assistant
Date: 2026-05-14
Description: Phase 20 academic evidence pack generator.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
AUDIT_DIR = PROJECT_ROOT / "docs" / "audit"
EVIDENCE_CSV = RESULTS_DIR / "academic_evidence_table.csv"
EVIDENCE_JSON = RESULTS_DIR / "academic_evidence_summary.json"
EVIDENCE_MD = AUDIT_DIR / "phase20_academic_evidence.md"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 4) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return ""
    return f"{numeric:.{digits}f}"


def _comparison_row(
    rows: List[Dict[str, Any]],
    *,
    evidence_family: str,
    research_question: str,
    estimand: str,
    design: str,
    metric: str,
    comparison: Dict[str, Any],
    comparator: str,
    artifact: str,
    interpretation: str,
    caveat: str,
    bootstrap_interval: Dict[str, Any] | None = None,
) -> None:
    rows.append({
        "evidence_family": evidence_family,
        "research_question": research_question,
        "estimand": estimand,
        "design": design,
        "metric": metric,
        "observed": _as_float(comparison.get("observed")),
        "comparator": comparator,
        "comparator_n": comparison.get("random_seed_count"),
        "percentile_vs_comparator": _as_float(comparison.get("random_percentile")),
        "right_tail_p_value": _as_float(comparison.get("right_tail_p_value")),
        "ci_lower_2_5": _as_float((bootstrap_interval or {}).get("p025")),
        "ci_upper_97_5": _as_float((bootstrap_interval or {}).get("p975")),
        "probability_model_beats_comparator": _as_float(
            (bootstrap_interval or {}).get("probability_model_beats_matched_random")
        ),
        "artifact": artifact,
        "interpretation": interpretation,
        "caveat": caveat,
    })


def build_evidence_rows() -> List[Dict[str, Any]]:
    random_summary = _read_json(RESULTS_DIR / "random_baseline_stability_summary.json")
    stratified_summary = _read_json(RESULTS_DIR / "stratified_random_baseline_summary.json")
    fold_summary = _read_json(RESULTS_DIR / "fold_matched_random_baseline_summary.json")
    rolling_summary = _read_json(RESULTS_DIR / "rolling_origin_summary.json")
    label_summary = _read_json(RESULTS_DIR / "label_summary.json")
    phase0_summary = _read_json(RESULTS_DIR / "phase0_baseline.json")

    rows: List[Dict[str, Any]] = []
    performance_metrics = [
        ("total_net_profit", "Total execution-aware net profit"),
        ("mean_equal_capital_roi", "Mean equal-capital ROI"),
        ("total_equal_capital_net_profit", "Total equal-capital net profit"),
    ]

    for metric, estimand in performance_metrics:
        comparison = random_summary.get("model_vs_random", {}).get(metric, {})
        if comparison.get("status") == "ok":
            _comparison_row(
                rows,
                evidence_family="random_baseline_stability",
                research_question="Does the model-ranked wallet strategy outperform multi-seed random wallet selection?",
                estimand=estimand,
                design="Rolling-origin out-of-sample backtest; same execution assumptions; 20 random seeds.",
                metric=metric,
                comparison=comparison,
                comparator="multi_seed_random_wallets",
                artifact="results/random_baseline_stability_summary.json",
                interpretation="Primary non-parametric randomization benchmark.",
                caveat="Random-wallet selection is not matched fold-by-fold on execution profile.",
            )

    for metric, estimand in performance_metrics:
        comparison = stratified_summary.get("model_vs_matched_random", {}).get(metric, {})
        if comparison.get("status") == "ok":
            _comparison_row(
                rows,
                evidence_family="stratified_random_matching",
                research_question="Does the model advantage persist after matching random seeds on execution profile?",
                estimand=estimand,
                design="Seed-level nearest-neighbor matching on trades, fill rate, and market concentration.",
                metric=metric,
                comparison=comparison,
                comparator="execution_profile_matched_random_seeds",
                artifact="results/stratified_random_baseline_summary.json",
                interpretation="Checks whether alpha is explained by broad execution-profile differences.",
                caveat="Seed-level matching can still hide fold-local mismatch.",
            )

    bootstrap = fold_summary.get("bootstrap_model_minus_matched_random", {})
    intervals = bootstrap.get("intervals", {})
    for metric, estimand in performance_metrics:
        comparison = fold_summary.get("model_vs_fold_matched_random", {}).get(metric, {})
        if comparison.get("status") == "ok":
            _comparison_row(
                rows,
                evidence_family="fold_matched_bootstrap",
                research_question="Does the model advantage persist under fold-local matched random baselines?",
                estimand=f"Model-minus-matched-random difference in {estimand}",
                design="Within-fold nearest-neighbor matching; folds resampled with replacement for bootstrap intervals.",
                metric=metric,
                comparison=comparison,
                comparator="fold_local_matched_random_wallets",
                artifact="results/fold_matched_random_baseline_summary.json",
                interpretation="Strongest current backtest evidence because comparison is local to each out-of-sample fold.",
                caveat="Only five chronological folds are available; bootstrap intervals remain fold-sample sensitive.",
                bootstrap_interval=intervals.get(metric, {}),
            )

    aggregate = rolling_summary.get("aggregate", {})
    if aggregate:
        rows.append({
            "evidence_family": "rolling_origin_prediction",
            "research_question": "Does the classifier produce temporally out-of-sample ranking signal?",
            "estimand": "Mean precision at k and ranking quality over chronological folds",
            "design": "Cumulative train / next-block test rolling-origin evaluation.",
            "metric": "mean_precision_at_k",
            "observed": _as_float(aggregate.get("mean_precision_at_k")),
            "comparator": "class_balance_and_threshold_policy_diagnostics",
            "comparator_n": rolling_summary.get("valid_fold_count"),
            "percentile_vs_comparator": None,
            "right_tail_p_value": None,
            "ci_lower_2_5": None,
            "ci_upper_97_5": None,
            "probability_model_beats_comparator": None,
            "artifact": "results/rolling_origin_summary.json",
            "interpretation": "Predictive validity is evaluated before economic backtesting.",
            "caveat": "Precision-at-k is a ranking diagnostic, not a trading-profit estimand.",
        })

    data_quality = phase0_summary.get("data_quality", {}).get("transactions", {})
    if label_summary:
        rows.append({
            "evidence_family": "label_independence",
            "research_question": "Is the supervised label independent from contemporaneous feature leakage?",
            "estimand": "Future-return label validity and trainability",
            "design": "Features are pre-resolution live features; labels use future resolved trades after the observation window.",
            "metric": "eligible_labeled_rows",
            "observed": _as_float(label_summary.get("future_label_coverage", {}).get("eligible_rows")),
            "comparator": "phase0_data_quality_and_training_gate",
            "comparator_n": data_quality.get("row_count"),
            "percentile_vs_comparator": None,
            "right_tail_p_value": None,
            "ci_lower_2_5": None,
            "ci_upper_97_5": None,
            "probability_model_beats_comparator": None,
            "artifact": "results/label_summary.json",
            "interpretation": "The target is suitable for academic model evaluation because it is temporally separated from live features.",
            "caveat": "This supports predictive validity, not causal identification of trader skill.",
        })

    return rows


def render_markdown(summary: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    label = summary.get("label_strategy", {})
    random = summary.get("random_baseline", {})
    fold = summary.get("fold_matched_bootstrap", {})

    lines = [
        "# Phase 20 Academic Evidence Pack",
        "",
        f"Generated at: `{summary['generated_at']}`",
        "",
        "## Academic Scope",
        "",
        "This pack converts the project artifacts into an auditable research evidence table.",
        "The claim supported here is predictive and economic: model-ranked wallets show",
        "out-of-sample signal under explicit execution assumptions. It is not a causal",
        "claim about trader skill.",
        "",
        "## Study Design",
        "",
        "- Unit of prediction: wallet/address.",
        "- Label: independent future-return outcome after the observation window.",
        "- Feature set: pre-resolution live features only.",
        "- Validation design: chronological splits plus rolling-origin folds.",
        "- Economic evaluation: execution-aware walk-forward backtests.",
        "- Inference checks: multi-seed random baselines, execution-profile matching, fold-local matching, and bootstrap intervals.",
        "",
        "## Reproducibility Anchors",
        "",
        f"- Label strategy: `{label.get('label_strategy')}`.",
        f"- Label independence: `{label.get('label_independence')}`.",
        f"- Trainable: `{label.get('is_trainable_for_binary_eval')}`.",
        f"- Random baseline seeds: `{random.get('seed_count')}`.",
        f"- Fold-matched bootstrap samples: `{fold.get('bootstrap_samples')}`.",
        f"- Evidence rows: `{summary['evidence_row_count']}`.",
        "",
        "## Primary Evidence Table",
        "",
        "| Evidence | Metric | Observed | Comparator | p-value | 95% CI | P(model > comparator) |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        interval = ""
        if row.get("ci_lower_2_5") is not None and row.get("ci_upper_97_5") is not None:
            interval = f"[{_fmt(row['ci_lower_2_5'])}, {_fmt(row['ci_upper_97_5'])}]"
        lines.append(
            "| "
            f"{row['evidence_family']} | "
            f"{row['metric']} | "
            f"{_fmt(row.get('observed'))} | "
            f"{row['comparator']} | "
            f"{_fmt(row.get('right_tail_p_value'))} | "
            f"{interval} | "
            f"{_fmt(row.get('probability_model_beats_comparator'))} |"
        )

    lines.extend([
        "",
        "## Methodological Guardrails",
        "",
        "- Temporal validity: training data precedes each rolling-origin test block.",
        "- Leakage control: default features are restricted to pre-resolution live signals.",
        "- Label independence: the active target is an independent future-outcome label, not a composite contemporaneous success proxy.",
        "- Execution realism: backtests include latency, fees, liquidity caps, capital caps, and market exposure diagnostics.",
        "- Baseline discipline: model results are read against random, stratified random, and fold-matched random controls.",
        "- Uncertainty reporting: Phase 19 bootstrap intervals are reported as model-minus-matched-random differences.",
        "",
        "## Limitations",
        "",
        "- The current evidence is observational and predictive; it should not be written as causal identification.",
        "- Only five rolling-origin folds are available in the current artifact set.",
        "- Bootstrap intervals are useful for fold-level uncertainty but do not replace a larger independent holdout period.",
        "- Backtest profitability depends on the stated execution model and should be stress-tested before deployment.",
        "",
        "## Artifacts",
        "",
        "- `results/academic_evidence_table.csv`",
        "- `results/academic_evidence_summary.json`",
        "- `docs/audit/phase20_academic_evidence.md`",
    ])
    return "\n".join(lines) + "\n"


def build_academic_evidence_pack() -> Dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    rows = build_evidence_rows()
    evidence_df = pd.DataFrame(rows)
    evidence_df.to_csv(EVIDENCE_CSV, index=False)

    random_summary = _read_json(RESULTS_DIR / "random_baseline_stability_summary.json")
    fold_summary = _read_json(RESULTS_DIR / "fold_matched_random_baseline_summary.json")
    label_summary = _read_json(RESULTS_DIR / "label_summary.json")
    summary = {
        "status": "ok" if rows else "no_evidence_rows",
        "generated_at": _utc_now(),
        "evidence_row_count": int(len(rows)),
        "evidence_csv": str(EVIDENCE_CSV),
        "evidence_markdown": str(EVIDENCE_MD),
        "label_strategy": {
            "label_strategy": label_summary.get("label_strategy"),
            "label_independence": label_summary.get("label_independence"),
            "is_trainable_for_binary_eval": label_summary.get("is_trainable_for_binary_eval"),
        },
        "random_baseline": {
            "seed_count": random_summary.get("seed_count"),
            "status": random_summary.get("status"),
        },
        "fold_matched_bootstrap": {
            "status": fold_summary.get("bootstrap_model_minus_matched_random", {}).get("status"),
            "bootstrap_samples": fold_summary.get("bootstrap_model_minus_matched_random", {}).get("bootstrap_samples"),
            "matched_rows": fold_summary.get("matched_rows"),
            "matched_rank_count": fold_summary.get("matched_rank_count"),
        },
        "academic_guardrails": [
            "future_outcome_label",
            "pre_resolution_live_features",
            "rolling_origin_validation",
            "execution_aware_backtest",
            "multi_seed_random_baseline",
            "fold_matched_bootstrap_intervals",
        ],
    }
    EVIDENCE_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    EVIDENCE_MD.write_text(render_markdown(summary, rows), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 20 academic evidence pack")
    parser.add_argument("--json", action="store_true", help="Print the generated summary JSON")
    args = parser.parse_args()
    summary = build_academic_evidence_pack()
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print(f"Wrote {EVIDENCE_CSV.relative_to(PROJECT_ROOT)}")
        print(f"Wrote {EVIDENCE_JSON.relative_to(PROJECT_ROOT)}")
        print(f"Wrote {EVIDENCE_MD.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
