"""
Author: AI Assistant
Date: 2026-03-18
Description: Orchestrator script to run the backtest and print results.
"""

import sys
from pathlib import Path
import pandas as pd
from loguru import logger

# Add project root to sys path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.backtesting.engine import BacktestEngine
from src.backtesting.strategies import ShadowWhaleStrategy, RandomBaselineStrategy
from src.backtesting.metrics import calculate_metrics

def main():
    logger.info("Starting Phase 4: Strategy Backtesting")
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    
    # 1. Load Data
    trades_path = BASE_DIR / "data" / "processed" / "real_trades_enriched.csv"
    resolutions_path = BASE_DIR / "data" / "processed" / "market_resolutions.csv"
    informed_path = BASE_DIR / "results" / "informed_traders.csv"
    
    if not all(p.exists() for p in [trades_path, resolutions_path, informed_path]):
        logger.error("Missing required data files. Please run earlier pipeline steps.")
        return
        
    trades_df = pd.read_csv(trades_path)
    if 'timestamp' in trades_df.columns:
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'], errors='coerce')
    trades_df = trades_df.dropna(subset=['timestamp']).sort_values('timestamp')
    
    resolutions_df = pd.read_csv(resolutions_path)
    informed_df = pd.read_csv(informed_path)
    
    # Extract only the addresses classified as informed (predicted_label == 1.0)
    # If the file includes top 10% directly, we might just use them all.
    # Let's use any address that is in informed_df and actually labeled informed
    if 'predicted_label' in informed_df.columns:
        target_wallets = informed_df[informed_df['predicted_label'] == 1.0]['address'].tolist()
    else:
        target_wallets = informed_df['address'].tolist()
        
    logger.info(f"Loaded {len(target_wallets)} target Whale addresses to shadow.")
    
    # 2. Setup Backtest Environment Variables
    initial_cap = 10000.0
    out_dir = BASE_DIR / "results" / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ==============================================================
    # RUN 1: Shadow Whale Strategy
    # ==============================================================
    engine_whale = BacktestEngine(initial_capital=initial_cap)
    strategy_whale = ShadowWhaleStrategy(
        informed_addresses=target_wallets, 
        max_position_size=1000.0, 
        copy_fraction=0.1
    )
    
    eq_whale, hist_whale = engine_whale.run(trades_df, resolutions_df, strategy_whale)
    metrics_whale = calculate_metrics(eq_whale, initial_cap)
    
    eq_whale.to_csv(out_dir / "whale_equity.csv", index=False)
    hist_whale.to_csv(out_dir / "whale_trades.csv", index=False)
    
    # ==============================================================
    # RUN 2: Random Baseline Strategy
    # ==============================================================
    engine_random = BacktestEngine(initial_capital=initial_cap)
    strategy_random = RandomBaselineStrategy(trade_probability=0.03, fixed_trade_size=50.0)
    
    eq_random, hist_random = engine_random.run(trades_df, resolutions_df, strategy_random)
    metrics_random = calculate_metrics(eq_random, initial_cap)
    
    # ==============================================================
    # Summary Report
    # ==============================================================
    logger.info("==============================================================")
    logger.info("🎯 BACKTEST PERFORMANCE REPORT")
    logger.info("==============================================================")
    
    logger.info(f"[WHALE SHADOWING STRATEGY]")
    for k, v in metrics_whale.items():
        logger.info(f"  {k}: {v:,.2f}")
    logger.info(f"  Total Trades Executed: {len(hist_whale[hist_whale['action']=='BUY']) if not hist_whale.empty else 0}")
        
    logger.info("-" * 40)
    logger.info(f"[RANDOM BASELINE STRATEGY]")
    for k, v in metrics_random.items():
        logger.info(f"  {k}: {v:,.2f}")
    logger.info(f"  Total Trades Executed: {len(hist_random[hist_random['action']=='BUY']) if not hist_random.empty else 0}")
        
    logger.info("==============================================================")
    logger.info(f"Alpha Captured (Returns Diff): {(metrics_whale.get('Total Return (%)', 0) - metrics_random.get('Total Return (%)', 0)):.2f}%")
    logger.info(f"Check /results/backtest/ for full equity curves and trade ledgers.")

if __name__ == "__main__":
    main()
