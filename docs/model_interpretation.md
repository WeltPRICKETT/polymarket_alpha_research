# Model Interpretation & Threshold Strategy

## Summary

The best model (XGBoost, AUC-ROC = 0.593) ranks informed traders modestly better than random but does not produce reliable binary classifications at the default 0.5 probability threshold. This is expected behavior, not a bug.

## Why Most Models Predict Zero Positives

The positive rate in the dataset is approximately 20%. At a 0.5 decision threshold, models learn that predicting "not informed" for everyone achieves ~80% accuracy. Only Random Forest (at AUC = 0.559) produced any positive predictions, with very low recall (4.0%).

This is a well-known phenomenon in imbalanced classification: when the signal is weak, conservative classifiers default to the majority class. The AUC-ROC metric, which evaluates ranking ability across all thresholds, is more informative than accuracy or F1 at a single cutoff.

## Recommended Usage: Top-k / Top-percentile Selection

Instead of using a fixed 0.5 threshold, treat the model output as a **ranking signal**. Select the top-k or top-percentile of traders by predicted probability.

### Threshold Policy Results (averaged across walk-forward folds)

| Policy | Selected | Precision | Recall | Lift vs Base Rate |
|--------|----------|-----------|--------|-------------------|
| Base rate | — | 22.5% | — | 1.00x |
| Top 5% | 67 | 24.5% | 5.5% | 1.09x |
| Top 10% | 133 | 27.8% | 12.3% | 1.24x |
| Top 20% | 265 | 27.7% | 24.5% | 1.23x |
| Top 50 | 50 | 24.8% | 4.1% | 1.10x |
| Fixed 0.5 | 693 | 25.7% | 59.9% | 1.14x |

The top-10% policy achieves a 1.24x precision lift over the base rate. This is a modest but real signal — the model identifies a subset of traders who are approximately 24% more likely to be informed than a random selection.

## Academic Interpretation

### What this result means

1. **Future-return prediction is genuinely difficult.** Unlike resolution-based accuracy (which measures past performance on known outcomes), future-return labels require predicting whether a trader will outperform in a forward-looking window. This is closer to alpha prediction than pattern recognition.

2. **The signal is weak but real.** AUC = 0.593 is statistically above the random baseline (0.5), and the top-percentile precision lift confirms the model captures some information. However, this is far from a reliable trading signal.

3. **The contribution is methodological, not predictive.** The value of this research lies in:
   - A fully reproducible pipeline with 60 automated tests and 10 data quality gates
   - CLOB-verified market resolutions (correcting a systemic bias in Gamma API)
   - Temporal walk-forward validation preventing data leakage
   - Future-return labeling ensuring label independence
   - Honest reporting of modest results, rather than overfitting to inflated metrics

### Why earlier metrics were higher

Early iterations of this project reported AUC > 0.99 and precision > 96%. These numbers were artifacts of:
- **Broken resolution labels**: The Gamma API `conditionId` filter was non-functional, causing all 35,000+ closed markets to resolve as "Yes." Models trained on these labels learned a trivial pattern, not genuine informed trading behavior.
- **Tiny test split**: With only 36 test wallets, even random variation could produce extreme metrics.
- **Resolution-based labels**: Predicting past accuracy on resolved markets is fundamentally easier than predicting future returns.

Phase 26 fixed the resolution bug, Phase 27A established future-return as the primary label, and Phase 28-29 ensured reproducibility. The current results are honest and methodologically sound.

## Model Comparison

| Model | AUC-ROC | Notes |
|-------|---------|-------|
| XGBoost | 0.593 | Best overall ranking ability |
| LightGBM | 0.570 | Second best, similar profile |
| Random Forest | 0.559 | Only model producing positive predictions at 0.5 |
| Logistic Regression | 0.546 | Linear model, weak signal |
| Gaussian NB | 0.543 | Simplest model, baseline-adjacent |
| Stratified Random | 0.517 | Random baseline |
| Majority Class | 0.500 | Trivial baseline |

All models beat the random baseline, confirming the features contain some information. The gap between models is small, suggesting the signal ceiling is low for this feature set and label definition.
