"""
Author: AI Assistant
Date: 2026-05-14
Description: Equal-capital normalization helpers for walk-forward backtests.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


EQUAL_CAPITAL_METHOD = (
    "per_fold_min_positive_filled_notional_budget; "
    "strategy_profit_scaled_by_realized_profit_per_filled_notional"
)


def add_equal_capital_metrics(df: pd.DataFrame, fold_col: str = "fold") -> pd.DataFrame:
    """Add fold-local equal-capital profit metrics to strategy rows."""
    out = df.copy()
    required = {fold_col, "total_filled_notional", "total_net_profit"}
    if out.empty or not required.issubset(out.columns):
        out["profit_per_filled_notional"] = []
        out["equal_capital_budget"] = []
        out["equal_capital_net_profit"] = []
        out["equal_capital_roi"] = []
        out["equal_capital_supported"] = []
        return out

    filled = pd.to_numeric(out["total_filled_notional"], errors="coerce").fillna(0.0)
    profit = pd.to_numeric(out["total_net_profit"], errors="coerce").fillna(0.0)
    out["profit_per_filled_notional"] = np.where(filled > 0, profit / filled, 0.0)

    positive_budgets = out[filled > 0].groupby(fold_col)["total_filled_notional"].min()
    out["equal_capital_budget"] = out[fold_col].map(positive_budgets).fillna(0.0).astype(float)
    out["equal_capital_supported"] = (filled > 0) & (out["equal_capital_budget"] > 0)
    out["equal_capital_net_profit"] = np.where(
        out["equal_capital_supported"],
        out["profit_per_filled_notional"] * out["equal_capital_budget"],
        0.0,
    )
    out["equal_capital_roi"] = np.where(
        out["equal_capital_budget"] > 0,
        out["equal_capital_net_profit"] / out["equal_capital_budget"],
        0.0,
    )
    return out


def summarize_equal_capital(df: pd.DataFrame) -> Dict[str, Dict]:
    """Aggregate equal-capital metrics by strategy."""
    if df.empty or "strategy" not in df.columns:
        return {}
    enriched = add_equal_capital_metrics(df)
    return {
        strategy: {
            "folds_supported": int(group["equal_capital_supported"].sum()),
            "total_equal_capital_budget": float(group["equal_capital_budget"].sum()),
            "total_equal_capital_net_profit": float(group["equal_capital_net_profit"].sum()),
            "mean_equal_capital_roi": float(group["equal_capital_roi"].mean()),
        }
        for strategy, group in enriched.groupby("strategy")
    }
