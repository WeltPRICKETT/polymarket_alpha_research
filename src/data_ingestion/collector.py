"""
Author: AI Assistant
Date: 2026-03-26
Description: Unified data collection CLI — supports full/incremental scrape and full pipeline.
"""

import sys
import argparse
from loguru import logger
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config.settings import LOG_LEVEL
from src.data_ingestion.storage import Storage
from src.data_ingestion.public_scraper import PublicScraper

# Set up logging
logger.remove()
logger.add(sys.stdout, level=LOG_LEVEL)
log_path = Path(__file__).resolve().parent.parent.parent / "logs" / "app.log"
log_path.parent.mkdir(parents=True, exist_ok=True)
logger.add(log_path, level=LOG_LEVEL, rotation="10 MB")


def collect_data(mode: str = "full", max_trades: int = 5000):
    """Main data collection routine."""
    logger.info(f"Starting data collection pipeline (mode={mode})...")

    storage = Storage()
    scraper = PublicScraper(storage)

    if mode == "incremental":
        scraper.harvest_incremental(max_trades=max_trades)
    else:
        scraper.harvest_all(max_trades=max_trades)

    # Verify
    df = storage.load_transactions_df()
    unique_wallets = df["address"].nunique() if "address" in df.columns else 0
    logger.info(f"Final DB stats: {len(df)} trades, {unique_wallets} unique wallets")

    if len(df) >= 10000:
        logger.info("✅ Validation passed: 10,000+ real trades harvested.")
    elif len(df) >= 100:
        logger.info("⚠️  Partial success: 100+ trades, but less than 10,000.")
    else:
        logger.warning(f"❌ Only {len(df)} trades — check API connectivity.")

    return len(df)


def run_pipeline(max_trades: int = 5000, top_percentile: float = 0.1):
    """Full pipeline: scrape → preprocess → feature engineering."""
    logger.info("=" * 60)
    logger.info("Running FULL PIPELINE: Scrape → Clean → Feature Engineering")
    logger.info("=" * 60)

    # Step 1: Collect data
    collect_data(mode="incremental", max_trades=max_trades)

    # Step 2: Run preprocessing pipeline
    try:
        from src.preprocessing.pipeline import main as run_preprocessing
        run_preprocessing(top_percentile=top_percentile)
        logger.info("✅ Full pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Pipeline failed during preprocessing: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polymarket Data Collection CLI")
    parser.add_argument("--mode", choices=["full", "incremental", "pipeline"],
                       default="full",
                       help="'full' = fresh scrape, 'incremental' = append only, 'pipeline' = scrape + ML processing")
    parser.add_argument("--max-trades", type=int, default=5000, help="Max trades to collect")
    parser.add_argument("--top-percentile", type=float, default=0.1,
                       help="Top percentile filter for pipeline mode (default: top 10%%)")
    args = parser.parse_args()

    if args.mode == "pipeline":
        run_pipeline(max_trades=args.max_trades, top_percentile=args.top_percentile)
    else:
        collect_data(mode=args.mode, max_trades=args.max_trades)
