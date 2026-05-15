"""
Author: AI Assistant
Date: 2026-05-12
Description: Backtest CLI entrypoint using the current StrategyBacktester engine.
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config.settings import LOG_LEVEL
from src.backtesting.engine import StrategyBacktester
from src.backtesting.walk_forward import run_walk_forward


def main(
    latency_minutes: int = 5,
    trade_size: float = 100.0,
    fees_pct: float = 0.001,
    max_trade_fraction_of_balance: float = 0.02,
    max_market_exposure_fraction: float = 0.10,
    liquidity_lookback_minutes: int = 60,
    max_participation_rate: float = 0.10,
    min_fill_notional: float = 1.0,
    price_impact_bps: float = 10.0,
):
    """Run the copy-trading backtest and write standard result artifacts."""
    logger.remove()
    logger.add(sys.stdout, level=LOG_LEVEL)

    logger.info("=" * 60)
    logger.info("PHASE 6: STRATEGY BACKTEST")
    logger.info("=" * 60)

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
    metrics = backtester.simulate()

    if metrics is None:
        raise RuntimeError("Backtest finished without valid trades. Check informed traders and resolved markets.")

    logger.info("=" * 60)
    logger.info("BACKTEST COMPLETE")
    logger.info("Outputs: results/backtest_trades.csv, results/backtest_metrics.json")
    if metrics.get("status") != "no_valid_trades":
        logger.info("Plot: results/plots/backtest_equity.png")
    logger.info("=" * 60)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Polymarket copy-trading backtest")
    parser.add_argument("--latency-minutes", type=int, default=5)
    parser.add_argument("--trade-size", type=float, default=100.0)
    parser.add_argument("--fees-pct", type=float, default=0.001)
    parser.add_argument("--max-trade-fraction-of-balance", type=float, default=0.02)
    parser.add_argument("--max-market-exposure-fraction", type=float, default=0.10)
    parser.add_argument("--liquidity-lookback-minutes", type=int, default=60)
    parser.add_argument("--max-participation-rate", type=float, default=0.10)
    parser.add_argument("--min-fill-notional", type=float, default=1.0)
    parser.add_argument("--price-impact-bps", type=float, default=10.0)
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward backtest with baselines")
    parser.add_argument("--folds", type=int, default=3, help="Number of walk-forward folds")
    parser.add_argument("--max-targets", type=int, default=50, help="Max target wallets per fold")
    args = parser.parse_args()
    if args.walk_forward:
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
    else:
        main(
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
