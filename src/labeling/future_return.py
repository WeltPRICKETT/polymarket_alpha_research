"""
Author: AI Assistant
Date: 2026-05-13
Description: Independent future-return labels with observation/horizon separation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger

from src.labeling.resolution_based import _normalise_outcome, load_resolutions

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class FutureReturnLabelConfig:
    observation_days: int = 1
    horizon_days: int = 7
    min_future_resolved_trades: int = 1
    top_percentile: float = 0.20
    threshold_source: str = "train_split_quantile"
    trade_filter: str = "BUY_only"
    prediction_rule: str = "price>=0.5 implies YES else NO"
    right_censoring_rule: str = "drop_addresses_without_complete_label_window"


def _normalise_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None)


def build_observation_transactions(
    trades_df: pd.DataFrame,
    observation_days: int,
) -> pd.DataFrame:
    """Keep only each address's first observation window for feature generation."""
    tx = trades_df.copy()
    tx["timestamp"] = _normalise_time(tx["timestamp"])
    tx = tx.dropna(subset=["address", "timestamp"])
    first_trade = tx.groupby("address")["timestamp"].transform("min")
    cutoff = first_trade + pd.to_timedelta(observation_days, unit="D")
    return tx[tx["timestamp"] <= cutoff].copy()


def compute_future_return_labels(
    trades_df: pd.DataFrame,
    resolutions: Dict[str, str],
    config: FutureReturnLabelConfig,
) -> pd.DataFrame:
    """
    Compute per-address labels from resolved BUY trades after the observation window.

    The label basis is intentionally independent from the feature matrix. It uses
    only future-window settlement outcomes and a fixed prediction heuristic.
    """
    tx = trades_df.copy()
    tx["timestamp"] = _normalise_time(tx["timestamp"])
    tx = tx.dropna(subset=["address", "market_id", "timestamp"])
    tx["price"] = pd.to_numeric(tx["price"], errors="coerce")
    tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce")
    tx = tx.dropna(subset=["price", "amount"])
    tx = tx[(tx["price"] > 0) & (tx["price"] < 1) & (tx["amount"] > 0)]

    first_trade = tx.groupby("address")["timestamp"].min().rename("first_trade_date")
    labels = first_trade.reset_index()
    labels["feature_cutoff"] = labels["first_trade_date"] + pd.to_timedelta(
        config.observation_days, unit="D"
    )
    labels["label_window_end"] = labels["feature_cutoff"] + pd.to_timedelta(
        config.horizon_days, unit="D"
    )
    labels["label_window_complete"] = labels["label_window_end"] <= tx["timestamp"].max()

    windows = labels[["address", "feature_cutoff", "label_window_end", "label_window_complete"]]
    future = tx.merge(windows, on="address", how="inner")
    future = future[
        (future["label_window_complete"])
        & (future["timestamp"] > future["feature_cutoff"])
        & (future["timestamp"] <= future["label_window_end"])
        & (future["side"].astype(str).str.upper() == "BUY")
    ].copy()

    res_map = {mid: _normalise_outcome(res) for mid, res in resolutions.items()}
    future["final_resolution"] = future["market_id"].map(res_map).fillna("OPEN")
    future = future[future["final_resolution"].isin(["YES", "NO"])].copy()

    if not future.empty:
        future["predicted_outcome"] = np.where(future["price"] >= 0.5, "YES", "NO")
        future["is_correct"] = future["predicted_outcome"] == future["final_resolution"]
        future["future_invested"] = future["amount"] * future["price"]
        future["future_net_profit"] = np.where(
            future["is_correct"],
            future["amount"] * (1.0 - future["price"]),
            -future["future_invested"],
        )
        agg = future.groupby("address").agg(
            future_n_resolved_trades=("market_id", "size"),
            future_n_correct=("is_correct", "sum"),
            future_invested=("future_invested", "sum"),
            future_net_profit=("future_net_profit", "sum"),
        )
    else:
        agg = pd.DataFrame(
            columns=[
                "address",
                "future_n_resolved_trades",
                "future_n_correct",
                "future_invested",
                "future_net_profit",
            ]
        )

    labels = labels.merge(agg, on="address", how="left")
    metric_cols = [
        "future_n_resolved_trades",
        "future_n_correct",
        "future_invested",
        "future_net_profit",
    ]
    labels[metric_cols] = labels[metric_cols].fillna(0)
    labels["future_n_resolved_trades"] = labels["future_n_resolved_trades"].astype(int)
    labels["future_n_correct"] = labels["future_n_correct"].astype(int)
    labels["future_accuracy"] = labels["future_n_correct"] / labels["future_n_resolved_trades"].replace(0, np.nan)
    labels["future_accuracy"] = labels["future_accuracy"].fillna(0.0)
    labels["future_roi"] = labels["future_net_profit"] / labels["future_invested"].replace(0, np.nan)
    labels["future_roi"] = labels["future_roi"].fillna(0.0)
    labels["Trader_Success_Rate"] = -1
    labels["label_source"] = "future_return_insufficient"
    labels.loc[~labels["label_window_complete"], "label_source"] = "future_return_right_censored"

    eligible = labels["label_window_complete"] & (
        labels["future_n_resolved_trades"] >= config.min_future_resolved_trades
    )
    labels.loc[eligible, "label_source"] = "future_return_pending_threshold"
    return labels


def apply_future_return_labels(
    features_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    observation_days: int = 1,
    horizon_days: int = 7,
    min_future_resolved_trades: int = 1,
    top_percentile: float = 0.20,
) -> pd.DataFrame:
    """Merge independent future-return labels into an already split feature frame."""
    config = FutureReturnLabelConfig(
        observation_days=observation_days,
        horizon_days=horizon_days,
        min_future_resolved_trades=min_future_resolved_trades,
        top_percentile=top_percentile,
    )
    label_df = compute_future_return_labels(trades_df, load_resolutions(), config)
    merge_cols = [
        "address",
        "feature_cutoff",
        "label_window_end",
        "label_window_complete",
        "future_n_resolved_trades",
        "future_n_correct",
        "future_accuracy",
        "future_invested",
        "future_net_profit",
        "future_roi",
        "Trader_Success_Rate",
        "label_source",
    ]
    merged = features_df.merge(label_df[merge_cols], on="address", how="left")

    merged = assign_future_return_threshold(
        merged,
        top_percentile=top_percentile,
        config=config,
    )

    before = len(merged)
    merged = merged[merged["Trader_Success_Rate"] != -1].copy()
    threshold = merged["future_return_threshold"].iloc[0] if not merged.empty else np.inf
    logger.info(
        "Future-return labels: kept {:,}/{:,} traders | threshold={:.6f}".format(
            len(merged), before, threshold
        )
    )
    return merged


def assign_future_return_threshold(
    df: pd.DataFrame,
    top_percentile: float = 0.20,
    config: FutureReturnLabelConfig | None = None,
) -> pd.DataFrame:
    """Assign binary labels from future ROI using the train split threshold when present."""
    labeled = df.copy()
    eligible = labeled["label_source"].isin(["future_return_pending_threshold", "future_return"])
    if "split" in labeled.columns and (eligible & labeled["split"].eq("train")).any():
        threshold_basis = labeled.loc[eligible & labeled["split"].eq("train"), "future_roi"]
    else:
        threshold_basis = labeled.loc[eligible, "future_roi"]

    if threshold_basis.empty:
        threshold = np.inf
        logger.warning("No eligible future-return labels found.")
    else:
        threshold = float(threshold_basis.quantile(1 - top_percentile))

    labeled.loc[eligible, "Trader_Success_Rate"] = (
        labeled.loc[eligible, "future_roi"] >= threshold
    ).astype(int)
    labeled.loc[eligible, "label_source"] = "future_return"
    labeled["future_return_threshold"] = threshold
    label_config = config or FutureReturnLabelConfig(top_percentile=top_percentile)
    labeled["future_label_config"] = str(asdict(label_config))
    return labeled
