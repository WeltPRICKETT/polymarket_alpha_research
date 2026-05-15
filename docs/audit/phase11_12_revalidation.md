# Phase 11-12 Revalidation

Generated at: `2026-05-14`

## Scope

Phase 11-12 turns the Phase 8-10 research evidence into more actionable model
selection and backtest evidence:

- Phase 11 evaluates ranked threshold policies instead of relying only on a
  fixed `0.5` probability cutoff.
- Phase 12 uses rolling-origin out-of-sample predictions to choose target
  wallets in walk-forward backtests.

## Implemented Gates

| Gate | Result |
|---|---|
| Rolling-origin predictions are persisted | PASS |
| Top-k and top-percentile policies are evaluated per fold | PASS |
| Threshold policy summary is included in rolling-origin summary | PASS |
| Model-informed wallet targets are backtested per rolling fold | PASS |
| Baselines are backtested on the same fold windows | PASS |
| Phase 0 audit includes Phase 11-12 artifacts | PASS |

## Threshold Policy Result

Current rolling-origin policy summary:

- `fixed_0_5`: mean precision `0.2573`, mean recall `0.5987`
- `top_k_20`: mean precision `0.1900`, mean recall `0.0125`
- `top_pct_0.10`: mean precision `0.2782`, mean recall `0.1230`
- `top_pct_0.20`: mean precision `0.2770`, mean recall `0.2446`

The fixed threshold has high recall because it selects many wallets. The
top-percentile policies are more suitable for constrained research portfolios
because they make the selection budget explicit.

## Model-Informed Walk-Forward Result

Current aggregate backtest summary:

| Strategy | Total trades | Total net profit | Mean ROI % | Mean Sharpe |
|---|---:|---:|---:|---:|
| `model_top_k_20` | `254` | `119010.06` | `272.58` | `-0.41` |
| `model_top_pct_0.10` | `1681` | `1126886.28` | `1043.34` | `20.10` |
| `baseline_top_volume` | `7352` | `1913858.50` | `419.52` | `16.47` |
| `baseline_random_wallets` | `232` | `517512.32` | `6159.47` | `14.01` |

These numbers show that model-ranked selection is wired into the backtest
system. They were generated before Phase 13 execution-realism constraints became
the default, so use the current Phase 13 artifacts for execution-aware claims.

## Artifacts

- `results/rolling_origin_predictions.csv`
- `results/threshold_policy_results.csv`
- `results/rolling_origin_summary.json`
- `results/model_informed_walk_forward_results.csv`
- `results/model_informed_walk_forward_summary.json`
- `results/phase8_12_summary.json`

## Remaining Work

1. Separate strategy evaluation from wallet-discovery evaluation so baselines
   compare equal target budgets and equal capital budgets.
2. Add confidence intervals or bootstrap stability for fold-level metrics.
