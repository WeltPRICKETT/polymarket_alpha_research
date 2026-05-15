# Phase 14 Revalidation

Generated at: `2026-05-14`

## Scope

Phase 14 adds equal-capital evaluation to walk-forward backtests. This makes
strategy comparisons less sensitive to one strategy producing far more fills or
far more deployed notional than another.

## Method

For each fold:

1. Find the minimum positive `total_filled_notional` across strategies.
2. Treat that as the shared equal-capital budget for the fold.
3. Compute each strategy's realized `profit_per_filled_notional`.
4. Scale each strategy's profit to the shared fold budget.

The method is recorded as:

`per_fold_min_positive_filled_notional_budget; strategy_profit_scaled_by_realized_profit_per_filled_notional`

## Implemented Gates

| Gate | Result |
|---|---|
| Equal-capital helper is unit tested | PASS |
| Standard walk-forward CSV includes equal-capital columns | PASS |
| Standard walk-forward summary aggregates equal-capital metrics | PASS |
| Model-informed walk-forward CSV includes equal-capital columns | PASS |
| Model-informed walk-forward summary aggregates equal-capital metrics | PASS |
| Phase 0 audit includes Phase 14 gates | PASS |

## Interpretation

Equal-capital results answer a narrower question than raw net profit:

> If each strategy were evaluated on the same fold-level filled-notional budget,
> which strategy produced more profit per unit of deployed capital?

This does not solve portfolio construction. It does not assume skipped trades
can be redeployed perfectly, and it does not yet equalize market, sector, or
event concentration risk.

## Current Model-Informed Equal-Capital Result

The model-informed walk-forward was rerun with Phase 13 execution constraints
and Phase 14 equal-capital normalization:

| Strategy | Supported folds | Equal budget | Equal-capital net profit | Mean equal-capital ROI |
|---|---:|---:|---:|---:|
| `model_top_k_20` | `5` | `7059.63` | `20801.97` | `0.8080` |
| `model_top_pct_0.10` | `5` | `7059.63` | `39139.22` | `6.7529` |
| `baseline_top_volume` | `5` | `7059.63` | `5197.84` | `1.4602` |
| `baseline_random_wallets` | `5` | `7059.63` | `71404.54` | `15.8439` |

Under this normalization, `model_top_pct_0.10` remains stronger than
`baseline_top_volume` on profit per deployed notional, but the random-wallet
baseline remains unusually strong on the current sample. That is a useful
warning: the next phase should quantify confidence intervals and baseline
stability rather than treating a single random seed as decisive.

## Artifacts

- `results/walk_forward_results.csv`
- `results/walk_forward_summary.json`
- `results/model_informed_walk_forward_results.csv`
- `results/model_informed_walk_forward_summary.json`
- `results/phase0_baseline.json`

## Remaining Work

1. Add exposure reports by market and event cluster.
2. Add equal-risk comparison using drawdown or volatility targets.
3. Add confidence intervals around equal-capital fold metrics.
