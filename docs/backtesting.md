# Backtesting Contract

Phase 6 makes backtests auditable rather than merely executable.

## Standard Backtest

```bash
python -m src.backtesting.run_backtest
```

Configurable assumptions:

```bash
python -m src.backtesting.run_backtest \
  --latency-minutes 5 \
  --trade-size 100 \
  --fees-pct 0.001 \
  --max-trade-fraction-of-balance 0.02 \
  --max-market-exposure-fraction 0.10 \
  --liquidity-lookback-minutes 60 \
  --max-participation-rate 0.10 \
  --price-impact-bps 10
```

Outputs:

- `results/backtest_trades.csv`
- `results/backtest_metrics.json`

The metrics file includes an `assumptions` block with latency, trade size, fee,
execution-price rule, settlement rule, liquidity model, capital model, slippage
model, and market filter.

## Execution Realism

Phase 13 enables conservative execution constraints by default:

- single-trade notional capped by current balance
- cumulative notional capped per market
- filled notional capped by recent observed market volume and participation rate
- adverse price impact applied after latency
- trades below `min_fill_notional` are skipped

The trade ledger records requested notional, filled notional, recent market
volume, participation rate, capital cap, liquidity cap, and market exposure.
Metrics include requested trades, fill rate, skipped trades, and cap counters.

## Equal-Capital Evaluation

Phase 14 adds fold-local equal-capital normalization for strategy comparisons.
For each fold, the common comparison budget is the minimum positive filled
notional among strategies in that fold. Each strategy's realized profit per
filled notional is then scaled to that shared budget.

New output columns include:

- `profit_per_filled_notional`
- `equal_capital_budget`
- `equal_capital_net_profit`
- `equal_capital_roi`
- `equal_capital_supported`

Summary files also include `mean_equal_capital_roi` and
`total_equal_capital_net_profit` per strategy. This does not assume the strategy
could redeploy rejected liquidity; it is a conservative comparability metric,
not an optimizer.

## Random Baseline Stability

Phase 15 evaluates the random-wallet baseline across multiple random seeds:

```bash
python -m src.experiments.future_label_experiments --baseline-seed-count 25
```

Outputs:

- `results/random_baseline_stability.csv`
- `results/random_baseline_seed_summary.csv`
- `results/random_baseline_stability_summary.json`

The summary compares `model_top_pct_0.10` against the random-seed distribution
using raw profit, equal-capital profit, and equal-capital ROI. The default seed
count is small for local runtime; increase it for publication-grade evidence.

## Exposure Concentration

Phase 16 adds market concentration diagnostics to execution-aware backtests:

- `Unique_Markets`
- `Top_Market_Filled_Notional_Share`
- `Market_HHI`
- `Effective_Markets`

The experiment workflow writes:

- `results/exposure_concentration_summary.json`

Lower HHI and top-market share mean less dependence on one market. Higher
effective market count means broader exposure. These diagnostics should be read
alongside equal-capital and random-baseline metrics.

## Stratified Random Baseline

Phase 17 matches random baseline seeds to the model strategy by execution
profile before comparing performance. The matching distance uses total trades,
mean fill rate, market HHI, top-market share, and effective markets.

Outputs:

- `results/stratified_random_baseline_matches.csv`
- `results/stratified_random_baseline_summary.json`

This is still limited by the number of random seeds. Use a larger
`--baseline-seed-count` for formal evidence.

## Fold-Matched Random Baseline

Phase 18 performs nearest-neighbor matching inside each rolling-origin test fold.
It matches random baseline rows to the model row from the same fold using total
trades, fill rate, market HHI, top-market share, and effective markets.

Outputs:

- `results/fold_matched_random_baseline_matches.csv`
- `results/fold_matched_random_baseline_rank_summary.csv`
- `results/fold_matched_random_bootstrap.csv`
- `results/fold_matched_random_baseline_summary.json`

This is the preferred diagnostic over seed-level matching when enough random
seeds are available, because it keeps comparisons local to the same
out-of-sample window.

Phase 19 extends this diagnostic with optional match-distance calipers and
fold-level bootstrap confidence intervals. Use `--fold-match-max-distance` to
filter weak local matches and `--fold-match-bootstrap-samples` to control the
bootstrap sample count.

## Academic Evidence Pack

Phase 20 converts the validation and backtest artifacts into a research evidence
table. The pack separates predictive/economic claims from causal claims and
records the estimand, design, comparator, interpretation, and caveat for each
reported result.

```bash
python -m src.audit.academic_evidence
```

Outputs:

- `results/academic_evidence_table.csv`
- `results/academic_evidence_summary.json`
- `docs/audit/phase20_academic_evidence.md`

## Walk-Forward Backtest

```bash
python -m src.backtesting.run_backtest --walk-forward --folds 3 --max-targets 50
```

Outputs:

- `results/walk_forward_results.csv`
- `results/walk_forward_summary.json`

Each fold selects target wallets using only addresses whose `first_trade_date` is
before the fold start. The same fold is compared against:

- `walk_forward_label_informed`
- `baseline_top_volume`
- `baseline_random_wallets`

## Temporal Guard

The walk-forward runner writes a `temporal_guard` section. If labels are still
`feature_derived_exploratory`, the result is diagnostic only even though target
selection is chronological. `independent_resolution` and
`independent_future_return` labels pass the formal-validity guard.

Formal backtest interpretation requires:

1. Independent labels such as `resolution` or future-return labels.
2. Train/validation/test splits with both classes.
3. A model artifact generated without `POLYMARKET_ALLOW_SINGLE_CLASS_SPLITS`.
4. Walk-forward results compared against baselines.

## API

Trigger regular backtest:

```http
POST /api/run-backtest
```

Trigger walk-forward:

```http
POST /api/run-backtest?walk_forward=true
```

Read latest walk-forward summary:

```http
GET /api/backtest/walk-forward
```

## Model-Informed Walk-Forward

Phase 12 connects rolling-origin model predictions to walk-forward backtesting.
It uses `results/rolling_origin_predictions.csv` to select wallets inside each
out-of-sample fold, then compares those selections against chronological
baselines.

Run through the experiment workflow:

```bash
python -m src.experiments.future_label_experiments
```

Outputs:

- `results/model_informed_walk_forward_results.csv`
- `results/model_informed_walk_forward_summary.json`

Current compared strategies:

- `model_top_k_20`
- `model_top_pct_0.10`
- `baseline_top_volume`
- `baseline_random_wallets`

Interpretation caveat: this is a model-informed research backtest with explicit
latency, fee, binary settlement, liquidity, slippage, and capital constraints.
It is still not a full order-book simulator with queue position, partial maker
fills, adverse selection, or exchange-level order placement.
