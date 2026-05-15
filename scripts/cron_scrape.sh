#!/bin/bash
# ============================================================================
# Polymarket 增量数据爬取定时脚本 (Incremental Scrape Cron Script)
# 每小时自动爬取 ~2900 条新交易数据并追加到数据库
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="${SCRIPT_DIR}/logs/cron_scrape.log"
VENV_PATH="${SCRIPT_DIR}/venv/bin/activate"

# Ensure log directory exists
mkdir -p "${SCRIPT_DIR}/logs"

echo "========================================" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting incremental scrape..." >> "$LOG_FILE"

# Activate virtualenv and run incremental scraper
cd "$SCRIPT_DIR"
source "$VENV_PATH"
python src/data_ingestion/public_scraper.py --mode incremental --max-trades 2900 >> "$LOG_FILE" 2>&1

EXIT_CODE=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Scrape finished with exit code: $EXIT_CODE" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

exit $EXIT_CODE
