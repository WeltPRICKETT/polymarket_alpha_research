# Phase 16 Revalidation

Generated at: `2026-05-14`

## Scope

Phase 16 adds market exposure concentration diagnostics. This checks whether
model-vs-random comparisons are being driven by unusually concentrated exposure
to a small number of markets.

## Method

Each backtest now reports filled-notional concentration across markets:

- `Unique_Markets`
- `Top_Market_Filled_Notional_Share`
- `Market_HHI`
- `Effective_Markets`

The model-informed walk-forward summary aggregates these metrics by strategy.
The random baseline stability summary aggregates them by random seed. A
dedicated `exposure_concentration_summary.json` compares the selected model
strategy against the random-seed concentration distribution.

## Implemented Gates

| Gate | Result |
|---|---|
| Backtest metrics include market concentration fields | PASS |
| Model-informed walk-forward aggregates concentration fields | PASS |
| Random baseline seed summary aggregates concentration fields | PASS |
| Exposure concentration summary compares model against random seeds | PASS |
| Phase 0 audit includes Phase 16 gates | PASS |

## Current Run

The local validation run used `--baseline-seed-count 5`.

| Metric | Model observed | Random mean | Random p95 | Model percentile |
|---|---:|---:|---:|---:|
| Mean unique markets | 210.6000 | 31.4400 | 45.7600 | 1.0000 |
| Mean top-market share | 0.0579 | 0.1467 | 0.1595 | 0.0000 |
| Mean market HHI | 0.0197 | 0.0769 | 0.0895 | 0.0000 |
| Mean effective markets | 69.8285 | 18.4305 | 26.7644 | 1.0000 |

Interpretation: in this small seed run, `model_top_pct_0.10` is less
concentrated than all sampled random baselines by HHI and top-market share, and
it spans more effective markets. The conclusion is diagnostic because five
random seeds produce coarse tail probabilities.

## Artifacts

- `results/model_informed_walk_forward_results.csv`
- `results/model_informed_walk_forward_summary.json`
- `results/random_baseline_seed_summary.csv`
- `results/exposure_concentration_summary.json`
- `results/phase0_baseline.json`

## Interpretation

Lower `Market_HHI` and lower `Top_Market_Filled_Notional_Share` indicate less
dependence on one market. Higher `Effective_Markets` indicates broader market
exposure. These diagnostics do not eliminate concentration risk, but they make
it visible and auditable before treating model profit as portable alpha.

## Remaining Work

1. Add stratified random baselines matched on concentration buckets.
2. Add market-category or event-level grouping when reliable metadata is
   available.
3. Add confidence intervals over exposure-adjusted performance.
