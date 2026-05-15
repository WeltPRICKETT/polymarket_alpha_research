# Phase 18 Revalidation

Generated at: `2026-05-14`

## Scope

Phase 18 adds fold-level nearest-neighbor random baseline matching. This makes
the matched random comparison local to each out-of-sample window instead of only
matching seed-level aggregates.

## Method

For each rolling-origin fold:

1. Select the `model_top_pct_0.10` row from
   `results/model_informed_walk_forward_results.csv`.
2. Compare it against random baseline rows from the same fold in
   `results/random_baseline_stability.csv`.
3. Compute standardized execution-profile distance using:
   - `total_trades`
   - `fill_rate`
   - `market_hhi`
   - `top_market_share`
   - `effective_markets`
4. Keep the nearest random rows inside each fold.
5. Aggregate matched rows by match rank across folds and compare model
   performance against those fold-matched rank portfolios.

## Implemented Gates

| Gate | Result |
|---|---|
| Fold-local random matching is computed | PASS |
| Fold-level match table is written | PASS |
| Matched-rank summary is written | PASS |
| Model-vs-fold-matched comparison is written | PASS |
| Phase 0 audit includes Phase 18 gates | PASS |

## Current Run

The local validation run now uses 20 random seeds, giving 20 candidate random rows per
fold. The matcher keeps 3 nearest random rows per fold, producing 3 matched-rank
portfolios across 5 folds.

| Metric | Model observed | Fold-matched mean | Model percentile | Right-tail p-value |
|---|---:|---:|---:|---:|
| Total net profit | 451407.6875 | 216963.4338 | 1.0000 | 0.2500 |
| Mean equal-capital ROI | 6.7529 | 4.6548 | 1.0000 | 0.2500 |
| Total equal-capital net profit | 39139.2150 | 17293.9001 | 1.0000 | 0.2500 |

Match quality improved with the larger random pool. The weakest local match is
still fold 5, where the best standardized distance is `4.4499`; this remains a
candidate for stricter caliper review before formal reporting.

## Artifacts

- `results/fold_matched_random_baseline_matches.csv`
- `results/fold_matched_random_baseline_rank_summary.csv`
- `results/fold_matched_random_baseline_summary.json`
- `results/phase8_12_summary.json`
- `results/phase0_baseline.json`

## Interpretation

Fold-level matching is stricter than seed-level matching because each time
window must find its own comparable random rows. The current local run is still
limited by the random seed count, so unmatched execution-profile differences can
remain. For formal evidence, increase `--baseline-seed-count` and rerun the
workflow.

## Remaining Work

These items are addressed by Phase 19:

1. Increase random seed count to improve fold-local candidate pools.
2. Add caliper thresholds so poor matches are flagged rather than silently used.
3. Add fold-level bootstrap confidence intervals over matched rank portfolios.
