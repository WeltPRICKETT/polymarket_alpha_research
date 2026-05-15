# Phase 6 Revalidation

Generated at: `2026-05-13`

Superseded note: Phase 7 later replaced composite labels with
`independent_future_return` labels. The diagnostic limitations below describe
the Phase 6 state before that label upgrade.

## Scope

Phase 6 upgrades the project backtesting layer from an executable script into an
auditable research component. The work covers explicit simulation assumptions,
ledger-grade outputs, walk-forward evaluation, baseline comparisons, API
surface, and Phase 0 audit integration.

## Implemented Gates

| Gate | Result |
|---|---|
| Backtest assumptions are explicit in code and output JSON | PASS |
| Empty/no-trade backtests produce auditable artifacts | PASS |
| Trade ledger includes strategy, side inference, resolution, fees, slippage, and balance | PASS |
| Walk-forward target selection is chronological | PASS |
| Walk-forward compares label-informed wallets with top-volume and random baselines | PASS |
| Walk-forward summary includes temporal guard and formal-validity flag | PASS |
| API exposes walk-forward result endpoint | PASS |
| Phase 0 baseline validates Phase 6 artifacts and routes | PASS |

## Current Diagnostic Result

- Standard backtest status: `no_valid_trades`
- Standard backtest reason: `No target addresses supplied.`
- Walk-forward fold count: `3`
- Walk-forward formal validity: `false`
- Label independence: `feature_derived_exploratory`
- Label-informed walk-forward trades: `0`
- Baseline top-volume walk-forward trades: `2790`
- Baseline random-wallet walk-forward trades: `61`

## Interpretation

The Phase 6 workflow is now reproducible and auditable, but the current strategy
result remains diagnostic only. The temporal split is chronological, yet labels
are still composite feature-derived labels rather than independently resolved
future-outcome labels. That means the framework is valid for engineering
verification, baseline stress testing, and workflow hardening, but not yet for
commercial or academic performance claims.

## Verification Commands

```bash
venv/bin/python -m compileall src tests
venv/bin/python -m pytest tests/test_phase1_phase2.py tests/test_phase3_phase4.py tests/test_phase5.py tests/test_phase6.py
venv/bin/python -m src.backtesting.run_backtest --latency-minutes 5 --trade-size 100 --fees-pct 0.001
venv/bin/python -m src.backtesting.run_backtest --walk-forward --folds 3 --max-targets 20
venv/bin/python -m src.audit.phase0
```

## Next Required Research Step

Replace the composite exploratory label with an independent target, such as
future realized profit, resolved-market edge, or post-signal return over a fixed
horizon. After that, rerun Phase 3 through Phase 6 and require
`phase6_walk_forward_formal_validity=true` before treating backtest output as
publishable evidence.
