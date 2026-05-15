# Phase 1-5 Revalidation After Phase 0

Generated after adding the Phase 0 baseline audit.

## Commands Run

```bash
python -m src.audit.phase0
python -m compileall -q src run.py run_backtest.py verify_chapter_1.py tests
python -m pytest -q tests/test_phase1_phase2.py tests/test_phase3_phase4.py tests/test_phase5.py
python -m src.data_quality.audit
python -m src.backtesting.run_backtest
python -c "from src.models.trainer import ModelTrainer; t=ModelTrainer(); t.load_data()"
POLYMARKET_ALLOW_SINGLE_CLASS_SPLITS=true python -c "from src.models.trainer import ModelTrainer; t=ModelTrainer(); t.load_data(); t._train_baselines(); print(sorted(t.results.keys()))"
```

## Results

| Area | Result | Notes |
|---|---|---|
| Phase 0 baseline | PASS | Wrote `results/phase0_baseline.json` and `docs/audit/phase0_baseline.md`. |
| Compile check | PASS | `compileall` completed without errors. |
| Regression tests | PASS | `13 passed`; warnings are expected baseline precision warnings from dummy models. |
| Phase 1 hard failures | PASS | Backtest entrypoint imports, `DataFetchError` exists, backtest API routes exist. |
| Phase 2 data audit | PASS | `results/data_quality_report.json` and `data/manifest.json` generated. |
| Phase 3 label audit | PASS with caveat | Label audit exists and correctly marks current labels as `feature_derived_exploratory`. |
| Phase 4 leakage control | PASS | Default trainer feature set is `live` and excludes post-resolution ROI/win-rate features. |
| Phase 5 training gate | PASS | Default training rejects current artifact because test split is single-class. |
| Diagnostic baseline mode | PASS | With explicit diagnostic override, majority and stratified baselines run. |
| Backtest CLI | PASS with caveat | CLI completes and writes `no_valid_trades`, reflecting current label/model artifact limitations. |

## Current Baseline Findings

- Current label strategy: `composite`.
- Current label independence: `feature_derived_exploratory`.
- Current single-class split: `test`.
- Current data is not valid for formal binary model evaluation.
- This is an expected blocker, not a regression: Phase 5 correctly prevents misleading model reports.

## Decision

Phases 1-5 are structurally valid after Phase 0.

The remaining blocker is research/data validity, not engineering plumbing:

1. Regenerate labels with higher `resolution` coverage or a better independent future-return label.
2. Rebuild splits so train, validation, and test all contain both classes.
3. Re-run full training without `POLYMARKET_ALLOW_SINGLE_CLASS_SPLITS`.
4. Only then interpret model metrics or backtest performance as formal results.
