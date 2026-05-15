# Phase 7 Revalidation

Generated at: `2026-05-13`

## Scope

Phase 7 replaces the previous composite exploratory label path with an
independent future-return label workflow. The new workflow computes model
features from each wallet's initial observation window and computes labels from
resolved BUY trades in a later future window.

## Implemented Gates

| Gate | Result |
|---|---|
| `future_return` label mode is available in preprocessing CLI | PASS |
| Features are generated from an observation window before the label window | PASS |
| Future labels are derived from resolved future-window outcomes, not feature ranks | PASS |
| Label threshold is learned from the final train split and applied to val/test | PASS |
| Train/val/test all contain both classes | PASS |
| Trainer no longer rejects current data as single-class | PASS |
| Walk-forward temporal guard accepts `independent_future_return` | PASS |
| Model search runs in single-process mode by default for sandbox compatibility | PASS |

## Current Label State

- Label strategy: `future_return`
- Label independence: `independent_future_return`
- Rows: `14,819`
- Train: `14,166` rows, `2,834` positive
- Val: `644` rows, `166` positive
- Test: `9` rows, `1` positive
- Future-return threshold: `0.3358390228524193`
- Mean future resolved trades per labeled wallet: `14.12`

## Current Model State

A small-budget verification run completed all enabled model families:

- Logistic Regression
- Random Forest
- XGBoost
- Gaussian NB
- LightGBM

The current test split is very small. AUC values are therefore not publishable
performance evidence yet, even though the pipeline is now formally trainable.
All tested models predicted zero positive wallets on the 9-row test split, so
precision/recall/F1 for the positive class remain `0.0`.

## Current Backtest State

Walk-forward now reports formal temporal validity because the label summary is
`independent_future_return`. The latest run produced label-informed trades, but
the observed trade counts remain small and must be treated as preliminary.

## Next Required Research Step

Increase the out-of-sample test population. The most direct options are:

1. Use a shorter horizon or alternative split policy with enough complete
   future-label windows.
2. Extend the collection window so recent wallets have complete future outcomes.
3. Add rolling-origin evaluation so every eligible historical period can serve
   as an out-of-sample fold.
