# Phase 15 Revalidation

Generated at: `2026-05-14`

## Scope

Phase 15 adds multi-seed random baseline stability checks. This addresses the
Phase 14 warning that one random-wallet baseline seed looked unusually strong.

## Method

For each rolling-origin test fold:

1. Draw `top_k` random wallets from addresses known before the fold start.
2. Repeat for multiple random seeds.
3. Run the same execution-aware backtest assumptions as Phase 13.
4. Aggregate seed-level raw profit, fill rate, and equal-capital metrics.
5. Compare the selected model strategy against the random-seed distribution.

The summary reports percentile rank and right-tail p-value with a small-sample
correction:

`p = (count(random_scores >= observed_score) + 1) / (seed_count + 1)`

## Current Run

The local validation run used `--baseline-seed-count 5` to keep runtime bounded.

| Metric | Model observed | Random mean | Random p95 | Model percentile | Right-tail p-value |
|---|---:|---:|---:|---:|---:|
| Total net profit | 451407.6875 | 93010.3823 | 190350.8175 | 1.0000 | 0.1667 |
| Mean equal-capital ROI | 6.7529 | 4.3328 | 6.5859 | 1.0000 | 0.1667 |
| Total equal-capital net profit | 39139.2150 | 19897.9471 | 27858.6250 | 1.0000 | 0.1667 |

Interpretation: the model strategy beats every sampled random seed in this
small run, including on equal-capital metrics. The p-value is still coarse
because only five random seeds were sampled.

## Implemented Gates

| Gate | Result |
|---|---|
| Random stability runner writes fold-level rows | PASS |
| Random stability runner writes seed-level summary | PASS |
| Summary includes distributions for profit, fill rate, and equal-capital ROI | PASS |
| Summary compares model strategy against random distribution | PASS |
| Phase 0 audit includes Phase 15 gates | PASS |

## Artifacts

- `results/random_baseline_stability.csv`
- `results/random_baseline_seed_summary.csv`
- `results/random_baseline_stability_summary.json`
- `results/phase8_12_summary.json`
- `results/phase0_baseline.json`

## Interpretation

This phase does not prove the model is commercially robust. It answers a narrower
question: whether the model-informed strategy beats a distribution of random
wallet selections under the same rolling windows and execution assumptions.

Because the default seed count is intentionally small for local runtime, p-values
should be treated as coarse diagnostics. Increase `--baseline-seed-count` before
making formal academic claims.

## Remaining Work

1. Increase random seed count for final research runs.
2. Add bootstrap confidence intervals for non-random strategies.
3. Add event/market exposure clustering so random baselines are compared under
   similar concentration risk.
