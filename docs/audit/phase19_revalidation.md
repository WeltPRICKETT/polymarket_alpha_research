# Phase 19 Revalidation

Generated at: `2026-05-14`

## Scope

Phase 19 strengthens the fold-matched random baseline by adding three formal
reporting safeguards:

1. A larger default random seed pool for baseline stability.
2. Optional caliper tracking for match distances.
3. Fold-level bootstrap confidence intervals for model-minus-matched-random
   performance differences.

## Method

The fold matcher still compares `model_top_pct_0.10` against random baseline
rows from the same rolling-origin fold. Phase 19 adds:

- `--baseline-seed-count`, now defaulting to `20`.
- `--fold-match-max-distance`, which filters poor matches when set.
- `--fold-match-bootstrap-samples`, defaulting to `1000`.
- `--fold-match-bootstrap-seed`, defaulting to `42`.

Bootstrap samples resample folds with replacement and draw one matched random
candidate per sampled fold. The reported intervals are model-minus-matched-random
differences, so positive values favor the model strategy.

## Implemented Gates

| Gate | Result |
|---|---|
| Expanded random seed pool is supported | PASS |
| Caliper pass/fail tracking is written | PASS |
| Bootstrap sample table is written | PASS |
| Bootstrap confidence intervals are written | PASS |
| Phase 0 audit includes Phase 19 gates | PASS |

## Artifacts

- `results/fold_matched_random_bootstrap.csv`
- `results/fold_matched_random_baseline_summary.json`
- `results/random_baseline_stability_summary.json`
- `results/phase8_12_summary.json`
- `results/phase0_baseline.json`

## Current Run

The validation run used 20 random seeds, 5 folds, 3 fold-local matches per fold,
and 1000 bootstrap samples.

| Metric | Model-minus-matched mean | 2.5% | 97.5% | P(model > matched) |
|---|---:|---:|---:|---:|
| Total net profit | 238102.2246 | -89686.8006 | 649158.9881 | 0.9130 |
| Mean equal-capital ROI | 2.2358 | -0.9738 | 5.2110 | 0.8990 |
| Total equal-capital net profit | 21771.3104 | -8514.7195 | 60778.0098 | 0.8870 |

The rank-based fold-matched comparison places the model above all three matched
rank portfolios for the tracked performance metrics. Because that rank
comparison has only three portfolios, its right-tail p-value remains coarse at
`0.25`; the bootstrap intervals provide the richer Phase 19 uncertainty view.

## Interpretation

Phase 19 moves the random comparison closer to formal inference. The empirical
matched-rank p-values remain useful for a quick rank-based comparison, while the
bootstrap intervals show whether model advantage persists when out-of-sample
folds are resampled. Calipers make weak local matches visible instead of letting
them blend into aggregate results.
