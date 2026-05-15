"""
Author: AI Assistant
Date: 2026-05-13
Description: Walk-forward backtesting with temporal guards and baselines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.backtesting.equal_capital import EQUAL_CAPITAL_METHOD, add_equal_capital_metrics
from src.backtesting.engine import StrategyBacktester

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _normalise_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None)


def _make_time_folds(tx_df: pd.DataFrame, n_folds: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    timestamps = _normalise_time(tx_df["timestamp"]).dropna().sort_values()
    if timestamps.empty:
        return []

    boundaries = pd.date_range(timestamps.min(), timestamps.max(), periods=n_folds + 1)
    folds = []
    for i in range(n_folds):
        start = boundaries[i]
        end = boundaries[i + 1]
        if start < end:
            folds.append((start, end))
    return folds


def _select_label_targets(feature_df: pd.DataFrame, fold_start: pd.Timestamp, max_targets: int) -> List[str]:
    if "first_trade_date" not in feature_df.columns:
        return []
    eligible = feature_df.copy()
    eligible["first_trade_date"] = _normalise_time(eligible["first_trade_date"])
    eligible = eligible[
        (eligible["first_trade_date"] < fold_start)
        & (eligible["Trader_Success_Rate"] == 1)
    ].copy()
    if eligible.empty:
        return []

    rank_col = "predicted_probability" if "predicted_probability" in eligible.columns else "composite_score"
    if rank_col in eligible.columns:
        eligible = eligible.sort_values(rank_col, ascending=False)
    else:
        eligible = eligible.sort_values("address")
    return eligible["address"].head(max_targets).tolist()


def _select_top_volume_targets(tx_df: pd.DataFrame, fold_start: pd.Timestamp, max_targets: int) -> List[str]:
    history = tx_df[_normalise_time(tx_df["timestamp"]) < fold_start].copy()
    if history.empty:
        return []
    history["volume"] = history.get("volume", history["amount"] * history["price"])
    ranked = history.groupby("address")["volume"].sum().sort_values(ascending=False)
    return ranked.head(max_targets).index.tolist()


def _select_random_targets(tx_df: pd.DataFrame, fold_start: pd.Timestamp, max_targets: int, seed: int) -> List[str]:
    history = tx_df[_normalise_time(tx_df["timestamp"]) < fold_start]
    addresses = sorted(history["address"].dropna().unique().tolist())
    if not addresses:
        return []
    rng = np.random.default_rng(seed)
    size = min(max_targets, len(addresses))
    return rng.choice(addresses, size=size, replace=False).tolist()


def run_walk_forward(
    n_folds: int = 3,
    max_targets: int = 50,
    latency_minutes: int = 5,
    trade_size: float = 100.0,
    fees_pct: float = 0.001,
    max_trade_fraction_of_balance: float = 0.02,
    max_market_exposure_fraction: float = 0.10,
    liquidity_lookback_minutes: int = 60,
    max_participation_rate: float = 0.10,
    min_fill_notional: float = 1.0,
    price_impact_bps: float = 10.0,
    random_seed: int = 42,
) -> Dict:
    """Run fold-by-fold backtests using only addresses known before each fold."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    backtester = StrategyBacktester(
        latency_minutes=latency_minutes,
        trade_size=trade_size,
        fees_pct=fees_pct,
        max_trade_fraction_of_balance=max_trade_fraction_of_balance,
        max_market_exposure_fraction=max_market_exposure_fraction,
        liquidity_lookback_minutes=liquidity_lookback_minutes,
        max_participation_rate=max_participation_rate,
        min_fill_notional=min_fill_notional,
        price_impact_bps=price_impact_bps,
    )
    backtester.load_data()

    feature_path = PROJECT_ROOT / "data" / "features" / "model_input.csv"
    feature_df = pd.read_csv(feature_path) if feature_path.exists() else pd.DataFrame()
    label_summary = _load_json(RESULTS_DIR / "label_summary.json")
    independent_label_modes = {"independent_resolution", "independent_future_return"}
    temporal_guard = {
        "target_selection": "addresses must have first_trade_date before fold_start",
        "label_independence": label_summary.get("label_independence", "unknown"),
        "label_strategy": label_summary.get("label_strategy", "unknown"),
        "formal_validity": label_summary.get("label_independence") in independent_label_modes,
        "warning": None,
    }
    if not temporal_guard["formal_validity"]:
        temporal_guard["warning"] = (
            "Walk-forward target selection is chronological, but labels are not independent. "
            "Treat results as diagnostic until labels are resolution/future-return based."
        )

    folds = _make_time_folds(backtester.tx_df, n_folds=n_folds)
    rows = []
    fold_details = []

    for fold_idx, (start, end) in enumerate(folds, start=1):
        label_targets = _select_label_targets(feature_df, start, max_targets)
        top_volume_targets = _select_top_volume_targets(backtester.tx_df, start, max(len(label_targets), 1))
        random_targets = _select_random_targets(backtester.tx_df, start, max(len(label_targets), 1), random_seed + fold_idx)

        strategies = {
            "walk_forward_label_informed": label_targets,
            "baseline_top_volume": top_volume_targets,
            "baseline_random_wallets": random_targets,
        }

        fold_detail = {
            "fold": fold_idx,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "target_counts": {name: len(targets) for name, targets in strategies.items()},
        }
        fold_details.append(fold_detail)

        for strategy_name, targets in strategies.items():
            metrics = backtester.simulate_for_addresses(
                target_addresses=targets,
                strategy_name=strategy_name,
                window_start=start,
                window_end=end,
                output_prefix=None,
            )
            rows.append({
                "fold": fold_idx,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "strategy": strategy_name,
                "target_count": len(targets),
                "status": metrics.get("status"),
                "total_trades": metrics.get("Total_Trades", 0),
                "requested_trades": metrics.get("Requested_Trades", 0),
                "total_requested_notional": metrics.get("Total_Requested_Notional", 0.0),
                "total_filled_notional": metrics.get("Total_Filled_Notional", 0.0),
                "fill_rate": metrics.get("Fill_Rate", 0.0),
                "skipped_liquidity": metrics.get("Skipped_Liquidity", 0),
                "skipped_capital": metrics.get("Skipped_Capital", 0),
                "capital_capped_trades": metrics.get("Capital_Capped_Trades", 0),
                "liquidity_capped_trades": metrics.get("Liquidity_Capped_Trades", 0),
                "market_exposure_capped_trades": metrics.get("Market_Exposure_Capped_Trades", 0),
                "win_rate": metrics.get("Win_Rate", 0.0),
                "total_net_profit": metrics.get("Total_Net_Profit", 0.0),
                "roi_pct": metrics.get("ROI_Pct", 0.0),
                "max_drawdown": metrics.get("Max_Drawdown", 0.0),
                "sharpe": metrics.get("Sharpe_Ratio (Annualized)", 0.0),
                "reason": metrics.get("reason", ""),
            })

    fold_df = add_equal_capital_metrics(pd.DataFrame(rows))
    fold_path = RESULTS_DIR / "walk_forward_results.csv"
    fold_df.to_csv(fold_path, index=False)

    summary = {
        "status": "ok" if rows else "no_folds",
        "fold_count": len(folds),
        "max_targets": max_targets,
        "temporal_guard": temporal_guard,
        "assumptions": backtester.assumptions.__dict__,
        "equal_capital_method": EQUAL_CAPITAL_METHOD,
        "folds": fold_details,
        "aggregate": {},
    }
    if not fold_df.empty:
        summary["aggregate"] = {
            strategy: {
                "folds": int(len(g)),
                "total_trades": int(g["total_trades"].sum()),
                "requested_trades": int(g["requested_trades"].sum()),
                "total_filled_notional": float(g["total_filled_notional"].sum()),
                "mean_fill_rate": float(g["fill_rate"].mean()),
                "total_net_profit": float(g["total_net_profit"].sum()),
                "mean_roi_pct": float(g["roi_pct"].mean()),
                "total_equal_capital_budget": float(g["equal_capital_budget"].sum()),
                "total_equal_capital_net_profit": float(g["equal_capital_net_profit"].sum()),
                "mean_equal_capital_roi": float(g["equal_capital_roi"].mean()),
                "equal_capital_supported_folds": int(g["equal_capital_supported"].sum()),
                "mean_sharpe": float(g["sharpe"].mean()),
            }
            for strategy, g in fold_df.groupby("strategy")
        }

    summary_path = RESULTS_DIR / "walk_forward_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info(f"Wrote walk-forward results → {fold_path}")
    logger.info(f"Wrote walk-forward summary → {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run walk-forward Polymarket backtest")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-targets", type=int, default=50)
    parser.add_argument("--latency-minutes", type=int, default=5)
    parser.add_argument("--trade-size", type=float, default=100.0)
    parser.add_argument("--fees-pct", type=float, default=0.001)
    parser.add_argument("--max-trade-fraction-of-balance", type=float, default=0.02)
    parser.add_argument("--max-market-exposure-fraction", type=float, default=0.10)
    parser.add_argument("--liquidity-lookback-minutes", type=int, default=60)
    parser.add_argument("--max-participation-rate", type=float, default=0.10)
    parser.add_argument("--min-fill-notional", type=float, default=1.0)
    parser.add_argument("--price-impact-bps", type=float, default=10.0)
    args = parser.parse_args()
    run_walk_forward(
        n_folds=args.folds,
        max_targets=args.max_targets,
        latency_minutes=args.latency_minutes,
        trade_size=args.trade_size,
        fees_pct=args.fees_pct,
        max_trade_fraction_of_balance=args.max_trade_fraction_of_balance,
        max_market_exposure_fraction=args.max_market_exposure_fraction,
        liquidity_lookback_minutes=args.liquidity_lookback_minutes,
        max_participation_rate=args.max_participation_rate,
        min_fill_notional=args.min_fill_notional,
        price_impact_bps=args.price_impact_bps,
    )


if __name__ == "__main__":
    main()
