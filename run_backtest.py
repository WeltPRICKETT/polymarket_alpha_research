"""
Author: AI Assistant
Date: 2026-04-11
Description: Phase 3 Orchestrator
Executes the Event Study and the Backtesting Engine.
"""

import sys
from pathlib import Path
from loguru import logger

from src.config.settings import LOG_LEVEL
from src.backtesting.event_study import EventStudyEngine
from src.backtesting.engine import StrategyBacktester

logger.remove()
logger.add(sys.stdout, level=LOG_LEVEL)

def main():
    logger.info("="*60)
    logger.info("PHASE 3: STRATEGY BACKTESTING & EVENT STUDY")
    logger.info("="*60)
    
    # 1. Event Study 
    logger.info("\n>>> Running Event Study (Analyzing Informed Price Trajectories)...")
    es_engine = EventStudyEngine()
    es_engine.load_data()
    es_df = es_engine.run_event_study(time_windows_min=[1, 5, 15, 30, 60, 240, 1440])
    es_summary = es_engine.evaluate_and_save(es_df)
    
    # 2. Backtest Engine
    logger.info("\n>>> Running Backtest Engine (Copy Trading Simulation)...")
    bt_engine = StrategyBacktester(latency_minutes=5, trade_size=100.0, fees_pct=0.001)
    bt_engine.load_data()
    bt_engine.simulate()
    
    logger.info("="*60)
    logger.info("PHASE 3 COMPLETION: SUCCESS")
    logger.info("Outputs generated in /results/ and /results/plots/")
    logger.info("="*60)


if __name__ == "__main__":
    main()
