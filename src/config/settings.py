"""
Author: AI Assistant
Date: 2026-03-17 (updated 2026-04-11)
Description: Configuration settings loaded from environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Define base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEV_ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=DEV_ENV_PATH)

POLYMARKET_API_KEY = os.getenv("POLYMARKET_API_KEY", "")

# For Clob Client, support comma-separated list of private keys or direct API credentials
POLYMARKET_PRIVATE_KEYS = [k.strip() for k in os.getenv("POLYMARKET_PRIVATE_KEYS", "").split(",") if k.strip()]
POLYMARKET_CLOB_API_KEYS = [k.strip() for k in os.getenv("POLYMARKET_CLOB_API_KEYS", "").split(",") if k.strip()]
POLYMARKET_CLOB_SECRETS = [k.strip() for k in os.getenv("POLYMARKET_CLOB_SECRETS", "").split(",") if k.strip()]
POLYMARKET_CLOB_PASSPHRASES = [k.strip() for k in os.getenv("POLYMARKET_CLOB_PASSPHRASES", "").split(",") if k.strip()]

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/data/research.db")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

# Data Collection Settings
ALLOW_MOCK_DATA = os.getenv("ALLOW_MOCK_DATA", "false").lower() == "true"

# ── Data Cleaning Parameters (P0.2: unified) ────────────────────────────────
# Min trades per trader to be included in the active pool
CLEANER_MIN_TRADES: int = int(os.getenv("CLEANER_MIN_TRADES", "3"))
# Min total volume (USDC) per trader
CLEANER_MIN_VOLUME: float = float(os.getenv("CLEANER_MIN_VOLUME", "50.0"))
# Optional top-percentile whale filter (None = all active traders)
CLEANER_TOP_PERCENTILE = None

# ── Feature Engineering Parameters ──────────────────────────────────────────
# Temporal split: fraction of timeline used for training and validation
TRAIN_RATIO: float = float(os.getenv("TRAIN_RATIO", "0.7"))
VAL_RATIO: float = float(os.getenv("VAL_RATIO", "0.15"))
# Test ratio is implicitly (1 - TRAIN_RATIO - VAL_RATIO)
# Outlier clipping percentiles for numeric features
FEATURE_CLIP_LO: float = 0.01
FEATURE_CLIP_HI: float = 0.99

# ── Label Configuration ──────────────────────────────────────────────────────
# Minimum number of resolved-market trades for a trader to be eligible for labeling
MIN_RESOLVED_TRADES: int = int(os.getenv("MIN_RESOLVED_TRADES", "3"))
# Prediction accuracy threshold to be "Informed" (resolution-based label)
INFORMED_ACCURACY_THRESHOLD: float = float(os.getenv("INFORMED_ACCURACY_THRESHOLD", "0.60"))
# Percentile cutoff for composite-rank fallback label
LABEL_TOP_PERCENTILE: float = 0.20

# ── ML Training Parameters ──────────────────────────────────────────────────
CV_FOLDS: int = 5
RANDOM_SEED: int = 42
# Overfitting alert threshold (train_auc - test_auc)
OVERFIT_THRESHOLD: float = 0.05

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "data" / "features").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "results" / "plots").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "models" / "artifacts").mkdir(parents=True, exist_ok=True)

