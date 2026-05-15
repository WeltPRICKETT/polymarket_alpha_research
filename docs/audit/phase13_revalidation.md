# Phase 13 Revalidation

Generated at: `2026-05-14`

## Scope

Phase 13 hardens the backtesting layer from idealized copy trading toward
execution-aware research simulation.

Implemented constraints:

- single-trade notional cap from current balance
- cumulative market exposure cap from current balance
- liquidity cap from recent observed market volume
- max participation-rate cap
- minimum fill threshold
- adverse price impact after latency

## Implemented Gates

| Gate | Result |
|---|---|
| Backtest assumptions include capital, liquidity, and slippage constraints | PASS |
| Trade ledger records requested and filled notional | PASS |
| Trade ledger records recent market volume and participation rate | PASS |
| Metrics record fill rate and cap counters | PASS |
| Walk-forward outputs expose fill metrics | PASS |
| Model-informed walk-forward outputs expose fill metrics | PASS |
| Phase 0 audit includes Phase 13 gates | PASS |

## Interpretation

Phase 13 changes the default meaning of backtest profitability. Results are now
conditioned on explicit execution constraints rather than full fills at observed
prices.

The simulation is still not a full exchange execution model. It does not model
order-book queue position, maker/taker order placement, hidden liquidity,
cancel/replace behavior, or adverse selection from the copied trader seeing the
same market.

## Current Model-Informed Walk-Forward

The Phase 12 model-informed walk-forward was rerun after enabling Phase 13
constraints:

| Strategy | Requested trades | Filled trades | Filled notional | Mean fill rate | Net profit | Mean ROI ratio |
|---|---:|---:|---:|---:|---:|---:|
| `model_top_k_20` | `254` | `193` | `7263.85` | `0.2345` | `21661.01` | `0.8080` |
| `model_top_pct_0.10` | `1681` | `1430` | `77898.87` | `0.4053` | `451407.69` | `6.7529` |
| `baseline_top_volume` | `7352` | `6621` | `339897.35` | `0.4637` | `502898.88` | `1.4602` |
| `baseline_random_wallets` | `232` | `223` | `19295.05` | `0.8106` | `295696.11` | `15.8439` |

The constrained results remain positive for the wider `model_top_pct_0.10`
selection, while `model_top_k_20` is much more liquidity constrained and less
stable. This is a healthier signal than the unconstrained results because the
fill-rate column now exposes execution pressure directly.

## Artifacts

- `results/backtest_trades.csv`
- `results/backtest_metrics.json`
- `results/walk_forward_results.csv`
- `results/walk_forward_summary.json`
- `results/model_informed_walk_forward_results.csv`
- `results/model_informed_walk_forward_summary.json`
- `results/phase0_baseline.json`

## Remaining Work

1. Add equal-capital portfolio accounting across strategy and baseline groups.
2. Add market-level exposure reports so concentrated-event risk is visible.
3. Add bootstrap confidence intervals for fold-level strategy metrics.
