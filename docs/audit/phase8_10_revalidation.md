# Phase 8-10 Revalidation

Generated at: `2026-05-13`

## Scope

Phase 8-10 improves the evidence workflow around independent future-return
labels:

- Phase 8 expands and audits candidate out-of-sample label configurations.
- Phase 9 adds rolling-origin evaluation to avoid relying on one tiny final test
  window.
- Phase 10 records a sensitivity grid across future-label horizons, minimum
  resolved-trade counts, and positive-label percentiles.

## Implemented Gates

| Gate | Result |
|---|---|
| Future-label sensitivity grid runs and writes CSV/JSON artifacts | PASS |
| Recommended future-label config is selected from measured configs | PASS |
| Current default future-label config uses the best available single-split config | PASS |
| Rolling-origin evaluation writes fold-level OOS metrics | PASS |
| All rolling-origin folds are valid binary folds | PASS |
| Phase 8-10 artifacts are included in Phase 0 audit | PASS |

## Current Default Label Config

- Observation window: `1` day
- Future horizon: `14` days
- Minimum future resolved BUY trades: `3`
- Positive cutoff: top `20%` by train-split future ROI

This config was selected because it produced the strongest available single
holdout coverage among the tested grid:

- Rows: `11,022`
- Train: `10,366`, positives `2,074`
- Val: `620`, positives `161`
- Test: `36`, positives `15`

It still does **not** pass the stricter single-split gate of `test_rows >= 50`.
That is now explicitly tracked in
`results/future_label_sensitivity_summary.json`.

## Rolling-Origin Result

Rolling-origin evaluation is currently the stronger out-of-sample evidence:

- Valid folds: `5`
- Total OOS rows: `6,614`
- Total OOS positives: `1,489`
- Mean AUC-ROC: `0.5625`
- Mean average precision: `0.2596`
- Mean precision@20: `0.19`

These numbers are modest, but they are based on much broader OOS support than
the final holdout split.

## Model Snapshot

The small-budget model run completed all enabled model families. On the final
36-row holdout:

- Best F1: LightGBM, `0.4211`
- Best AUC-ROC: Gaussian NB, `0.6333`
- LightGBM recall: `0.2667`
- XGBoost recall: `0.0667`

The single holdout is still too small for strong performance claims. Treat it as
a sanity check; use rolling-origin metrics for primary model evidence.

## Extended By Phase 11-12

The original remaining items around threshold policies and model-informed
walk-forward backtesting are now implemented in Phase 11-12. See
`docs/audit/phase11_12_revalidation.md`.

## Remaining Work

1. Extend the data window or add more rolling folds before making commercial or
   academic performance claims.
