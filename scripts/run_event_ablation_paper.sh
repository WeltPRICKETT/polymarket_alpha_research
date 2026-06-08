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

ROLLING_FOLDS="${ROLLING_FOLDS:-8}"
BASELINE_SEED_COUNT="${BASELINE_SEED_COUNT:-200}"
FOLD_MATCH_COUNT_PER_FOLD="${FOLD_MATCH_COUNT_PER_FOLD:-5}"
FOLD_MATCH_BOOTSTRAP_SAMPLES="${FOLD_MATCH_BOOTSTRAP_SAMPLES:-10000}"

echo "========================================"
echo "  Event Ablation Paper Run"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo "Python:       $PYTHON_BIN"
echo "Started at:   $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Candidates:   live, live_plus_event_diversification, live_plus_mean_markets_per_event_traded"
echo "Rolling folds:                 $ROLLING_FOLDS"
echo "Baseline seed count:           $BASELINE_SEED_COUNT"
echo "Fold-match count per fold:     $FOLD_MATCH_COUNT_PER_FOLD"
echo "Fold-match bootstrap samples:  $FOLD_MATCH_BOOTSTRAP_SAMPLES"
echo ""

"$PYTHON_BIN" scripts/run_event_ablation.py \
    --paper-candidates \
    --run-name event_ablation_paper \
    --rolling-folds "$ROLLING_FOLDS" \
    --baseline-seed-count "$BASELINE_SEED_COUNT" \
    --fold-match-count-per-fold "$FOLD_MATCH_COUNT_PER_FOLD" \
    --fold-match-bootstrap-samples "$FOLD_MATCH_BOOTSTRAP_SAMPLES" \
    --quiet

echo ""
echo "Outputs:"
echo "  results/new_data_sources/event_ablation_paper_table.csv"
echo "  results/new_data_sources/event_ablation_paper_report.md"
echo "  results/new_data_sources/event_ablation_paper_runs/"
echo "Finished at: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
