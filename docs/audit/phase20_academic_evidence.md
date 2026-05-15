# Phase 20 Academic Evidence Pack

Generated at: `2026-05-14T09:56:30.448204+00:00`

## Academic Scope

This pack converts the project artifacts into an auditable research evidence table.
The claim supported here is predictive and economic: model-ranked wallets show
out-of-sample signal under explicit execution assumptions. It is not a causal
claim about trader skill.

## Study Design

- Unit of prediction: wallet/address.
- Label: independent future-return outcome after the observation window.
- Feature set: pre-resolution live features only.
- Validation design: chronological splits plus rolling-origin folds.
- Economic evaluation: execution-aware walk-forward backtests.
- Inference checks: multi-seed random baselines, execution-profile matching, fold-local matching, and bootstrap intervals.

## Reproducibility Anchors

- Label strategy: `future_return`.
- Label independence: `independent_future_return`.
- Trainable: `True`.
- Random baseline seeds: `20`.
- Fold-matched bootstrap samples: `1000`.
- Evidence rows: `11`.

## Primary Evidence Table

| Evidence | Metric | Observed | Comparator | p-value | 95% CI | P(model > comparator) |
|---|---|---:|---|---:|---:|---:|
| random_baseline_stability | total_net_profit | 451407.6875 | multi_seed_random_wallets | 0.0952 |  |  |
| random_baseline_stability | mean_equal_capital_roi | 6.7529 | multi_seed_random_wallets | 0.2857 |  |  |
| random_baseline_stability | total_equal_capital_net_profit | 39139.2150 | multi_seed_random_wallets | 0.0952 |  |  |
| stratified_random_matching | total_net_profit | 451407.6875 | execution_profile_matched_random_seeds | 0.1818 |  |  |
| stratified_random_matching | mean_equal_capital_roi | 6.7529 | execution_profile_matched_random_seeds | 0.1818 |  |  |
| stratified_random_matching | total_equal_capital_net_profit | 39139.2150 | execution_profile_matched_random_seeds | 0.0909 |  |  |
| fold_matched_bootstrap | total_net_profit | 451407.6875 | fold_local_matched_random_wallets | 0.2500 | [-89686.8006, 649158.9881] | 0.9130 |
| fold_matched_bootstrap | mean_equal_capital_roi | 6.7529 | fold_local_matched_random_wallets | 0.2500 | [-0.9738, 5.2110] | 0.8990 |
| fold_matched_bootstrap | total_equal_capital_net_profit | 39139.2150 | fold_local_matched_random_wallets | 0.2500 | [-8514.7195, 60778.0098] | 0.8870 |
| rolling_origin_prediction | mean_precision_at_k | 0.1900 | class_balance_and_threshold_policy_diagnostics |  |  |  |
| label_independence | eligible_labeled_rows | 11022.0000 | phase0_data_quality_and_training_gate |  |  |  |

## Methodological Guardrails

- Temporal validity: training data precedes each rolling-origin test block.
- Leakage control: default features are restricted to pre-resolution live signals.
- Label independence: the active target is an independent future-outcome label, not a composite contemporaneous success proxy.
- Execution realism: backtests include latency, fees, liquidity caps, capital caps, and market exposure diagnostics.
- Baseline discipline: model results are read against random, stratified random, and fold-matched random controls.
- Uncertainty reporting: Phase 19 bootstrap intervals are reported as model-minus-matched-random differences.

## Limitations

- The current evidence is observational and predictive; it should not be written as causal identification.
- Only five rolling-origin folds are available in the current artifact set.
- Bootstrap intervals are useful for fold-level uncertainty but do not replace a larger independent holdout period.
- Backtest profitability depends on the stated execution model and should be stress-tested before deployment.

## Artifacts

- `results/academic_evidence_table.csv`
- `results/academic_evidence_summary.json`
- `docs/audit/phase20_academic_evidence.md`
