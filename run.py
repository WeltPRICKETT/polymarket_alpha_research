#!/usr/bin/env python3
"""
Polymarket Alpha Research — One-Command Launcher

Usage:
    python run.py                    # Start dashboard server
    python run.py --scrape           # Scrape data first, then start server
    python run.py --pipeline         # Run full pipeline (scrape + ML), then start server
    python run.py --port 8080        # Custom port
"""

import sys
import os
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VENV_PYTHON = BASE_DIR / "venv" / "bin" / "python"

# ── Auto-relaunch with venv Python if not already using it ──────
def _is_in_venv():
    """Check if we're running inside the project venv."""
    return hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )

if not _is_in_venv() and VENV_PYTHON.exists():
    # Re-exec this script with the venv Python
    print(f"[AUTO] Detected system Python. Re-launching with venv: {VENV_PYTHON}")
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

# ── From here on, we're inside the venv ─────────────────────────
import argparse
from loguru import logger

sys.path.insert(0, str(BASE_DIR))

# Setup logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
log_path = BASE_DIR / "logs" / "app.log"
log_path.parent.mkdir(parents=True, exist_ok=True)
logger.add(log_path, level="INFO", rotation="10 MB")


def main():
    parser = argparse.ArgumentParser(
        description="Polymarket Alpha Research — Unified Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                    Start the dashboard server
  python run.py --scrape           Scrape data then start server
  python run.py --pipeline         Full pipeline then start server
  python run.py --port 8080        Use custom port
        """
    )
    parser.add_argument("--scrape", action="store_true", help="Run data scraping before starting server")
    parser.add_argument("--pipeline", action="store_true", help="Run full pipeline (scrape + ML) before starting server")
    parser.add_argument("--max-trades", type=int, default=2900, help="Max trades to collect (default: 2900)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    args = parser.parse_args()

    # Pre-flight: run scraping or pipeline if requested
    if args.pipeline:
        logger.info("🚀 Running full pipeline before starting server...")
        from src.data_ingestion.collector import run_pipeline
        run_pipeline(max_trades=args.max_trades)
        logger.info("✅ Pipeline complete. Starting dashboard server...")
    elif args.scrape:
        logger.info("🔄 Running incremental scrape before starting server...")
        from src.data_ingestion.collector import collect_data
        collect_data(mode="incremental", max_trades=args.max_trades)
        logger.info("✅ Scrape complete. Starting dashboard server...")

    # Start the FastAPI server
    logger.info(f"🌐 Starting Polymarket Alpha Dashboard at http://localhost:{args.port}")
    logger.info(f"   Open your browser to access the dashboard")

    import uvicorn
    uvicorn.run(
        "src.visualization.api:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
