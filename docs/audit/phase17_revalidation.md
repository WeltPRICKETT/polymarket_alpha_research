# Phase 17 Revalidation

Generated at: `2026-05-14`

## Scope

Phase 17 adds stratified random baseline matching. This reduces the risk that a
model-vs-random comparison is merely comparing different execution profiles.

## Method

The analysis reads:

- `results/model_informed_walk_forward_summary.json`
- `results/random_baseline_seed_summary.csv`

For `model_top_pct_0.10`, each random seed receives a standardized distance to
the model execution profile using:

- `total_trades`
- `mean_fill_rate`
- `mean_market_hhi`
- `mean_top_market_share`
- `mean_effective_markets`

The nearest random seeds become the matched baseline set. Performance is then
compared against this matched subset using total profit, equal-capital ROI, and
equal-capital profit.

## Implemented Gates

| Gate | Result |
|---|---|
| Random seed execution profile distance is computed | PASS |
| Matched random seed table is written | PASS |
| Matched-random performance comparison is written | PASS |
| Phase 0 audit includes Phase 17 gates | PASS |

## Current Run

The local validation run had 5 available random seeds. The matcher selected the
3 nearest seeds by standardized execution-profile distance: `0`, `4`, and `2`.

| Metric | Model observed | Matched random mean | Model percentile | Right-tail p-value |
|---|---:|---:|---:|---:|
| Total net profit | 451407.6875 | 124597.4648 | 1.0000 | 0.2500 |
| Mean equal-capital ROI | 6.7529 | 4.6892 | 1.0000 | 0.2500 |
| Total equal-capital net profit | 39139.2150 | 19718.7360 | 1.0000 | 0.2500 |

The selected random seeds are still not perfectly balanced against the model:
the model has more trades and broader market exposure. This reinforces the need
for a larger random seed pool before treating the matched comparison as formal
evidence.

## Artifacts

- `results/stratified_random_baseline_matches.csv`
- `results/stratified_random_baseline_summary.json`
- `results/phase8_12_summary.json`
- `results/phase0_baseline.json`

## Interpretation

This is a matching diagnostic, not a causal proof. It improves over raw random
seed comparison by asking whether the model still wins against random baselines
with similar execution size, fill rate, and market concentration. Small local
seed counts still limit tail probability resolution.

## Remaining Work

1. Run with substantially more random seeds before publication.
2. Add fold-level nearest-neighbor matching, not only seed-level matching.
3. Add exact stratification bins once event/category metadata is reliable.
