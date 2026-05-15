# Phase 0 Baseline Audit

Generated at: `2026-05-15T11:54:25.567578+00:00`
Project root: `/Users/mac/project 2/poly/polymarket_alpha_research`

## Environment

- Python: `3.13.11`
- Platform: `macOS-26.3-arm64-arm-64bit-Mach-O`
- Git branch: `main`
- Git HEAD: `e71383b`
- Dirty worktree: `True`

## Data Baseline

- Transactions: `1730663`
- Unique transaction ids: `1730663`
- Duplicate transaction ids: `0`
- Unique wallets: `205289`
- Unique markets: `98120`
- Time range: `2026-03-18 00:57:21.000000` to `2026-05-15 11:34:40.000000`
- Invalid prices: `1`
- Invalid amounts: `0`

## Label Baseline

- Label strategy: `future_return`
- Label independence: `independent_future_return`
- Single-class splits: `[]`
- Trainable for binary evaluation: `True`

## Phase 1-24 Gate Snapshot

| Gate | Status |
|---|---|
| Phase 1 backtest entrypoint imports | PASS |
| Phase 1 DataFetchError defined | PASS |
| Phase 1 backtest API routes | PASS |
| Phase 2 data quality report | PASS |
| Phase 2 manifest | PASS |
| Phase 2 data contract | PASS |
| Phase 3 label summary | PASS |
| Phase 4 live feature set safe | PASS |
| Phase 5 current-data training gate | PASS |
| Phase 6 backtest assumptions | PASS |
| Phase 6 ledger schema | PASS |
| Phase 6 walk-forward import | PASS |
| Phase 6 walk-forward API route | PASS |
| Phase 6 walk-forward summary | PASS |
| Phase 6 temporal guard | PASS |
| Phase 6 backtesting docs | PASS |
| Phase 7 future-return label active | PASS |
| Phase 7 future-return trainable | PASS |
| Phase 8 label sensitivity | PASS |
| Phase 8 recommended config | PASS |
| Phase 9 rolling-origin folds | PASS |
| Phase 10 sensitivity summary | PASS |
| Phase 8-10 docs | PASS |
| Phase 11 threshold policies | PASS |
| Phase 12 model-informed walk-forward | PASS |
| Phase 12 model strategy present | PASS |
| Phase 11-12 docs | PASS |
| Phase 8-12 summary | PASS |
| Phase 13 execution constraints | PASS |
| Phase 13 execution ledger | PASS |
| Phase 13 model walk-forward fill metrics | PASS |
| Phase 13 docs | PASS |
| Phase 14 walk-forward equal-capital metrics | PASS |
| Phase 14 model walk-forward equal-capital metrics | PASS |
| Phase 14 docs | PASS |
| Phase 15 random baseline stability | PASS |
| Phase 15 model-vs-random comparison | PASS |
| Phase 15 docs | PASS |
| Phase 16 model concentration metrics | PASS |
| Phase 16 random seed concentration metrics | PASS |
| Phase 16 exposure concentration summary | PASS |
| Phase 16 docs | PASS |
| Phase 17 stratified random summary | PASS |
| Phase 17 matched-random comparison | PASS |
| Phase 17 docs | PASS |
| Phase 18 fold-matched random summary | PASS |
| Phase 18 fold-matched comparison | PASS |
| Phase 18 docs | PASS |
| Phase 19 expanded random seed pool | PASS |
| Phase 19 fold-matched bootstrap | PASS |
| Phase 19 bootstrap CI | PASS |
| Phase 19 caliper tracking | PASS |
| Phase 19 docs | PASS |
| Phase 20 academic evidence summary | PASS |
| Phase 20 academic evidence table | PASS |
| Phase 20 academic guardrails | PASS |
| Phase 20 docs | PASS |
| Phase 21 collection provenance tables | PASS |
| Phase 22 outcome preservation | PASS |
| Phase 23 extended resolution schema | PASS |
| Phase 24 dataset completeness report | PASS |
| Phase 24 docs | PASS |

## Notes

- Phase 5 gate mode: `accepted_trainable_data`.
- Phase 11 threshold policies tracked: `7` policies, `35` fold-policy rows.
- Phase 12 model-informed walk-forward status: `ok`.
- Phase 13 model-informed walk-forward fill metrics present: `True`.
- Phase 14 model-informed equal-capital metrics present: `True`.
- Phase 15 random baseline seed count: `20`.
- Phase 16 concentration summary present: `True`.
- Phase 17 matched random seed count: `10`.
- Phase 18 fold-matched rank count: `3`.
- Phase 19 expanded random seed pool: `True`.
- Phase 19 bootstrap CI present: `True`.
- Phase 20 academic evidence rows: `11`.
- Phase 24 known dataset gaps: `0`.
- Phase 6 walk-forward formal validity is expected to pass only when labels are independent resolution or future-return labels.
- To run diagnostics on current artifacts, use `POLYMARKET_ALLOW_SINGLE_CLASS_SPLITS=true`.
