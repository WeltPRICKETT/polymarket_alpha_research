#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    if [ -x "./conda-env/bin/python" ]; then
        PYTHON_BIN="./conda-env/bin/python"
    elif [ -x "./venv/bin/python" ]; then
        PYTHON_BIN="./venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    else
        PYTHON_BIN="$(command -v python)"
    fi
fi

MODE="${1:-quick}"
ROLLING_FOLDS="${ROLLING_FOLDS:-5}"
MAX_TARGETS="${MAX_TARGETS:-50}"

if [ "$MODE" = "paper" ]; then
    BASELINE_SEED_COUNT="${BASELINE_SEED_COUNT:-100}"
    FOLD_MATCH_COUNT_PER_FOLD="${FOLD_MATCH_COUNT_PER_FOLD:-5}"
    FOLD_MATCH_BOOTSTRAP_SAMPLES="${FOLD_MATCH_BOOTSTRAP_SAMPLES:-5000}"
else
    BASELINE_SEED_COUNT="${BASELINE_SEED_COUNT:-20}"
    FOLD_MATCH_COUNT_PER_FOLD="${FOLD_MATCH_COUNT_PER_FOLD:-3}"
    FOLD_MATCH_BOOTSTRAP_SAMPLES="${FOLD_MATCH_BOOTSTRAP_SAMPLES:-1000}"
fi

echo "========================================"
echo "  Academic Analysis Workflow"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo "Python:       $PYTHON_BIN"
echo "Mode:         $MODE"
echo "Started at:   $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo ""

if [ ! -f "data/research.db" ]; then
    echo "ERROR: data/research.db not found."
    echo "This workflow requires the full research database, not only the CSV sample."
    exit 1
fi

run_step() {
    local title="$1"
    shift
    echo ""
    echo "----------------------------------------"
    echo "$title"
    echo "----------------------------------------"
    "$@"
}

run_step "1/10 Data quality audit" \
    "$PYTHON_BIN" -m src.data_quality.audit

run_step "2/10 Dataset completeness and resolution gates" \
    "$PYTHON_BIN" -m src.data_quality.dataset_completeness

run_step "3/10 Preprocessing with independent future-return labels" \
    "$PYTHON_BIN" src/preprocessing/pipeline.py --label-mode future_return

run_step "4/10 Live-feature model training and calibrated probabilities" \
    env POLYMARKET_FEATURE_SET=live "$PYTHON_BIN" -m src.models.trainer

run_step "5/10 Future-label sensitivity, rolling-origin prediction, random baselines, bootstrap" \
    "$PYTHON_BIN" -m src.experiments.future_label_experiments \
        --rolling-folds "$ROLLING_FOLDS" \
        --baseline-seed-count "$BASELINE_SEED_COUNT" \
        --fold-match-count-per-fold "$FOLD_MATCH_COUNT_PER_FOLD" \
        --fold-match-bootstrap-samples "$FOLD_MATCH_BOOTSTRAP_SAMPLES"

run_step "6/10 Execution-aware walk-forward backtest" \
    "$PYTHON_BIN" -m src.backtesting.run_backtest \
        --walk-forward \
        --folds "$ROLLING_FOLDS" \
        --max-targets "$MAX_TARGETS"

run_step "7/10 Academic evidence table and caveat pack" \
    "$PYTHON_BIN" -m src.audit.academic_evidence

run_step "8/10 End-to-end phase audit" \
    "$PYTHON_BIN" -m src.audit.phase0

run_step "9/10 Test suite" \
    "$PYTHON_BIN" -m pytest tests/ -v

echo ""
echo "----------------------------------------"
echo "10/10 Artifact checklist"
echo "----------------------------------------"
ARTIFACTS=(
    "results/data_quality_report.json"
    "results/dataset_completeness_report.json"
    "data/features/model_input.csv"
    "results/label_summary.json"
    "results/label_audit.csv"
    "results/training_report.json"
    "results/model_comparison.csv"
    "results/rolling_origin_summary.json"
    "results/threshold_policy_results.csv"
    "results/model_informed_walk_forward_summary.json"
    "results/random_baseline_stability_summary.json"
    "results/fold_matched_random_baseline_summary.json"
    "results/fold_matched_random_bootstrap.csv"
    "results/walk_forward_summary.json"
    "results/academic_evidence_table.csv"
    "results/academic_evidence_summary.json"
    "docs/audit/phase0_baseline.md"
    "docs/audit/phase20_academic_evidence.md"
    "docs/audit/phase24_dataset_completeness.md"
)

ALL_EXIST=true
for artifact in "${ARTIFACTS[@]}"; do
    if [ -f "$artifact" ]; then
        echo "OK       $artifact"
    else
        echo "MISSING  $artifact"
        ALL_EXIST=false
    fi
done

echo ""
echo "========================================"
if [ "$ALL_EXIST" = true ]; then
    echo "ALL REQUIRED ARTIFACTS EXIST"
else
    echo "SOME ARTIFACTS ARE MISSING"
    exit 1
fi
echo "Finished at: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "========================================"
