# Model Training Contract

Phase 5 hardens the model training pipeline.

## Data Gates

Training refuses to continue unless train, validation, and test splits each contain
both binary classes. This prevents misleading reports such as perfect accuracy on a
single-class test set.

After Phase 8-12, current artifacts are expected to pass this gate when generated
with the default `future_return` settings: `1` observation day, `14` future-label
days, and at least `3` future resolved BUY trades. Older composite artifacts may
still be rejected, which is the correct behavior.

Diagnostic override:

```bash
POLYMARKET_ALLOW_SINGLE_CLASS_SPLITS=true python -m src.models.trainer
```

Use this only for debugging old artifacts, not for academic or commercial reporting.

## Chronological Ordering

Before model arrays are built, each split is sorted by `first_trade_date` when the
column exists. This makes `TimeSeriesSplit` consume samples in temporal order.

## Baselines

Every training run evaluates these baselines before tuned models:

- `Baseline: Majority Class`
- `Baseline: Stratified Random`

Production models should be interpreted only relative to these baselines.

Hyperparameter search runs with `POLYMARKET_SKLEARN_N_JOBS=1` and
`POLYMARKET_SEARCH_N_ITER=8` by default so the workflow is reliable in restricted
local environments. Increase `POLYMARKET_SEARCH_N_ITER` for full experiments.

Rolling-origin evidence is generated separately:

```bash
python -m src.experiments.future_label_experiments
```

That workflow writes:

- `results/rolling_origin_results.csv`
- `results/rolling_origin_predictions.csv`
- `results/threshold_policy_results.csv`
- `results/rolling_origin_summary.json`
- `results/phase8_12_summary.json`

Academic reporting evidence is generated separately:

```bash
python -m src.audit.academic_evidence
```

That workflow writes a research-ready evidence table and methodology appendix:

- `results/academic_evidence_table.csv`
- `results/academic_evidence_summary.json`
- `docs/audit/phase20_academic_evidence.md`

## Threshold Policies

Phase 11 evaluates model outputs as ranked selection policies, not only as a
fixed probability cutoff. The current audit records:

- `fixed_0_5`
- `top_k_10`, `top_k_20`, `top_k_50`
- `top_pct_0.05`, `top_pct_0.10`, `top_pct_0.20`

Use the top-k or top-percentile policy table for research decisions when
probability calibration is still immature. The fixed `0.5` threshold remains a
diagnostic reference, not the only decision rule.

## Calibration

Probability calibration is fitted on the validation split only. The base model is
trained on the training split; validation probabilities are then used to learn a
probability mapping:

- `isotonic` when validation has at least 30 samples in each class
- `sigmoid` for smaller validation splits

The method and sample count are written to `results/training_report.json`.

## Metadata

The training report records:

- data path
- split counts
- label counts by split
- chronological sort column
- feature set and feature leakage classes
- label audit summary when available
- baseline names
