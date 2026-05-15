# Reproducibility Guide

This document describes how to verify all research results from scratch, given access to the dataset.

## Quick Start

```bash
bash scripts/reproduce.sh
```

This single command runs the full verification pipeline (Steps 1–6 below) and exits non-zero if anything fails.

## Prerequisites

- Python 3.11+
- `data/research.db` (the SQLite dataset, ~834 MB — not committed to the repo due to size)
- Dependencies: `pip install -r requirements.txt`

## Verification Steps

### Step 1: Data Quality Audit

```bash
python -m src.data_quality.audit
```

Checks raw data integrity: timestamp validity, outcome preservation, price ranges, duplicate detection. Produces `results/data_quality_report.json` and `data/manifest.json`.

### Step 2: Dataset Completeness Report

```bash
python -m src.data_quality.dataset_completeness
```

Validates 10 gates covering provenance, outcome preservation, resolution schema, and resolution coverage. Produces `results/dataset_completeness_report.json` and `docs/audit/phase24_dataset_completeness.md`.

Key gates:
| Gate | Requirement |
|------|-------------|
| `resolution_coverage_ge_99pct` | ≥99% of DB markets have resolution records |
| `resolution_no_legacy_in_closed` | Zero closed markets use the legacy (broken) method |
| `resolution_fetch_failed_lt_1pct` | <1% of markets have fetch failures |
| `resolution_clob_dominates_closed` | ≥90% of closed markets use CLOB winner field |

### Step 3: Phase 0 Baseline Audit

```bash
python -m src.audit.phase0
```

End-to-end audit across all validated phases (0–26). Produces `results/phase0_baseline.json` and `docs/audit/phase0_baseline.md`. Verifies label independence, walk-forward integrity, feature sets, random baselines, bootstrap CIs, and academic evidence tables.

### Step 4: Preprocessing Pipeline

```bash
python src/preprocessing/pipeline.py --label-mode future_return
```

Rebuilds the full pipeline: raw transactions → cleaned trades → features → temporal split → future-return labels. Produces `data/features/model_input.csv`, `results/label_summary.json`, `results/label_audit.csv`.

### Step 5: Test Suite

```bash
python -m pytest tests/ -v
```

60 tests across 8 test files covering:
- Phase 3–4: Feature set isolation, label audit
- Phase 5: Chronological splits, baseline models
- Phase 6: Walk-forward, backtesting, concentration
- Phase 7: Future-return label construction
- Phase 8–10: Sensitivity, rolling origin, random baselines
- Phase 25: Provenance integrity (SHA-256 verification)
- Phase 26: Resolution verification (CLOB correctness, hard gates)

### Step 6: Artifact Check

Verifies all expected output files exist after the pipeline run.

## Evidence Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| Phase 0 report | `results/phase0_baseline.json` | Master audit across all phases |
| Completeness report | `results/dataset_completeness_report.json` | Data quality gates |
| Label summary | `results/label_summary.json` | Label strategy, counts, balance |
| Academic evidence | `results/academic_evidence_table.csv` | Summary statistics for paper |
| Walk-forward results | `results/walk_forward_results.csv` | Temporal validation folds |
| Random baselines | `results/random_baseline_stability_summary.json` | Model vs random comparison |
| Bootstrap CIs | `results/fold_matched_random_bootstrap.csv` | Confidence intervals |
| Feature importance | `results/feature_importance.csv` | Model feature rankings |
| Backtest metrics | `results/backtest_metrics.json` | Execution realism results |

## Audit Trail

Phase-by-phase audit reports are in `docs/audit/`:

| Phase | File | Focus |
|-------|------|-------|
| 0 | `phase0_baseline.md` | End-to-end baseline |
| 1–5 | `phase1_5_revalidation.md` | Core pipeline gates |
| 6 | `phase6_revalidation.md` | Walk-forward formal validity |
| 7 | `phase7_revalidation.md` | Future-return label |
| 8–10 | `phase8_10_revalidation.md` | Sensitivity, rolling origin |
| 11–12 | `phase11_12_revalidation.md` | Threshold policy, model walk-forward |
| 13–19 | `phase13–19_revalidation.md` | Execution, baselines, bootstrap |
| 20 | `phase20_academic_evidence.md` | Evidence table, guardrails |
| 24 | `phase24_dataset_completeness.md` | Completeness gates |

## Label Strategy

The primary label uses **future-return independence**: each trader's label is based on their ROI in a forward-looking window that starts *after* the feature observation period. This prevents label leakage and ensures the model predicts future performance, not past accuracy.

A resolution-based label (accuracy on resolved markets) is available as a robustness check. Both routes produce test splits >1,000 wallets.

## Last Verified Run

**Date**: 2026-05-15T19:57+08:00 (Phase 30 freeze)

```
Step 1/6: Data quality audit              PASS
Step 2/6: Dataset completeness report     PASS (10/10 gates)
Step 3/6: Phase 0 baseline audit          PASS (all phases green)
Step 4/6: Preprocessing pipeline          PASS (15,799 eligible, train=13,654/val=1,119/test=1,026)
Step 5/6: Test suite                      60 passed, 3 warnings
Step 6/6: Artifact check                  12/12 OK
Result:                                   ALL CHECKS PASSED
```

**Frozen artifact hashes**:
- `model_input.csv` snapshot: `b1debd9bae99721e`
- Best model: XGBoost (AUC=0.593)
- Label strategy: `future_return` (independent)
- Resolution coverage: 99.8% (CLOB-verified)

## Data Contract

See `docs/data_contract.md` for the formal data schema and `docs/features.md` for feature definitions.
