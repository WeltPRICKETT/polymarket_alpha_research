# Feature Sets

The project now separates predictive features from explanatory research features.

## `live`

Use this feature set for commercial-style prediction and no-leakage model training.
These fields can be computed from transaction behavior before market resolution.

| Feature | Timing | Notes |
|---|---|---|
| `early_entry_score` | pre-resolution | Relative entry timing inside observed market activity |
| `contrarian_score` | pre-resolution | Distance from market-level buy ratio |
| `information_ratio` | pre-resolution | Price-path stability proxy from observed trades |
| `cross_market_diversification` | pre-resolution | Unique markets divided by trader activity |
| `avg_holding_period` | pre-resolution | Average gap between trader events |
| `trading_frequency` | pre-resolution | Trades per observed active day |
| `capital_flow_centrality` | pre-resolution | Trader volume share in current dataset |

## `research`

Use this feature set for explanatory analysis only. These fields use post-resolution or
outcome-derived information and should not be used for live prediction claims.

| Feature | Timing | Notes |
|---|---|---|
| `total_roi` | post-resolution | Requires resolved markets or return proxy |
| `max_drawdown` | post-resolution | Built from realized/proxy PnL path |
| `win_rate` | post-resolution | Requires realized/proxy win classification |
| `profit_loss_ratio` | post-resolution | Requires realized/proxy trade PnL |

## Running Models

Default:

```bash
python -m src.models.trainer
```

This uses `POLYMARKET_FEATURE_SET=live` by default.

Training refuses single-class train/validation/test splits by default. For diagnostics
only, set:

```bash
POLYMARKET_ALLOW_SINGLE_CLASS_SPLITS=true python -m src.models.trainer
```

Research-only comparison:

```bash
POLYMARKET_FEATURE_SET=research python -m src.models.trainer
```

Extended research mode with `Risk_Adjusted_Return`:

```bash
POLYMARKET_FEATURE_SET=research_with_rar python -m src.models.trainer
```

## Label Caveat

`resolution` labels are suitable for primary academic claims when coverage is sufficient.
`composite` labels are exploratory because they are derived from `total_roi`, `win_rate`,
and `profit_loss_ratio`; they are exported with `label_independence=feature_derived_exploratory`
in `results/label_audit.csv`.

`future_return` labels are independent future-outcome labels. In this mode,
features are computed from each wallet's initial observation window and labels
come from resolved BUY trades in the later label window. The threshold is learned
from the train split and then applied to validation/test.

The current Phase 8-10 default is `1` observation day, `14` future-label days,
and at least `3` future resolved BUY trades. Sensitivity results are written to
`results/future_label_sensitivity.csv`.
