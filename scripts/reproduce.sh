#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "========================================"
echo "  Polymarket Alpha — Reproducibility"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo "Started at:   $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo ""

if [ ! -f "data/research.db" ]; then
    echo "ERROR: data/research.db not found. Cannot reproduce without the dataset."
    exit 1
fi

if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "Using active virtualenv: $VIRTUAL_ENV"
else
    echo "WARNING: No virtualenv found. Using system Python."
fi

echo ""
echo "Step 1/6: Data quality audit"
echo "----------------------------------------"
python -m src.data_quality.audit

echo ""
echo "Step 2/6: Dataset completeness report"
echo "----------------------------------------"
python -m src.data_quality.dataset_completeness

echo ""
echo "Step 3/6: Phase 0 baseline audit"
echo "----------------------------------------"
python -m src.audit.phase0

echo ""
echo "Step 4/6: Preprocessing pipeline (future-return labels)"
echo "----------------------------------------"
python src/preprocessing/pipeline.py --label-mode future_return

echo ""
echo "Step 5/6: Test suite"
echo "----------------------------------------"
python -m pytest tests/ -v

echo ""
echo "Step 6/6: Verify key artifacts exist"
echo "----------------------------------------"
ARTIFACTS=(
    "results/phase0_baseline.json"
    "results/dataset_completeness_report.json"
    "results/data_quality_report.json"
    "results/label_summary.json"
    "results/label_audit.csv"
    "results/academic_evidence_table.csv"
    "results/academic_evidence_summary.json"
    "data/features/model_input.csv"
    "data/processed/market_resolutions.csv"
    "data/processed/cleaned_transactions.csv"
    "docs/audit/phase0_baseline.md"
    "docs/audit/phase24_dataset_completeness.md"
)

ALL_EXIST=true
for f in "${ARTIFACTS[@]}"; do
    if [ -f "$f" ]; then
        echo "  OK  $f"
    else
        echo "  MISSING  $f"
        ALL_EXIST=false
    fi
done

echo ""
echo "========================================"
if [ "$ALL_EXIST" = true ]; then
    echo "  ALL CHECKS PASSED"
else
    echo "  SOME ARTIFACTS MISSING — see above"
    exit 1
fi
echo "  Finished at: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "========================================"
