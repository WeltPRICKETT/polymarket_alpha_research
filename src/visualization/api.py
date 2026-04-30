"""
Author: AI Assistant
Date: 2026-03-26
Description: Enhanced FastAPI backend with rich data APIs and static frontend hosting.
"""

import os
import json
import subprocess
import threading
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from loguru import logger
from typing import List, Dict, Any
import datetime

app = FastAPI(title="Polymarket Alpha Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "data" / "research.db"
LOG_PATH = BASE_DIR / "logs" / "cron_scrape.log"
APP_LOG_PATH = BASE_DIR / "logs" / "app.log"
PLOT_DIR = BASE_DIR / "results" / "plots"
STATIC_DIR = BASE_DIR / "static"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Pipeline execution state (shared across requests)
_pipeline_state = {"running": False, "status": "idle", "message": "", "progress": 0}
_pipeline_lock = threading.Lock()

# ── Helper: lazy-load Storage to avoid import issues ────────────────────────

def _get_storage():
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from src.data_ingestion.storage import Storage
    return Storage()

# ── Mount static assets ────────────────────────────────────────────────────

if PLOT_DIR.exists():
    app.mount("/plots", StaticFiles(directory=str(PLOT_DIR)), name="plots")

# ── Core Statistics API ────────────────────────────────────────────────────

@app.get("/api/stats")
def get_stats():
    """Get database statistics."""
    try:
        storage = _get_storage()
        return {
            "total_transactions": storage.get_total_count(),
            "total_markets": storage.get_unique_market_count(),
            "total_wallets": storage.get_unique_wallet_count(),
            "last_update": storage.get_latest_timestamp(),
        }
    except Exception as e:
        logger.error(f"Error querying stats: {e}")
        return {
            "total_transactions": 0,
            "total_markets": 0,
            "total_wallets": 0,
            "last_update": None,
            "error": str(e),
        }

# ── Trade Timeline API ────────────────────────────────────────────────────

@app.get("/api/trades/timeline")
def get_trade_timeline(interval: str = "day"):
    """Get trade volume aggregated by time interval."""
    if interval not in ("hour", "day", "month"):
        raise HTTPException(400, "interval must be 'hour', 'day', or 'month'")
    try:
        storage = _get_storage()
        return {"data": storage.get_trade_timeline(interval=interval)}
    except Exception as e:
        logger.error(f"Error querying timeline: {e}")
        raise HTTPException(500, str(e))

# ── Top Wallets API ────────────────────────────────────────────────────────

@app.get("/api/wallets/top")
def get_top_wallets(limit: int = 20):
    """Get top wallets by volume."""
    try:
        storage = _get_storage()
        return {"data": storage.get_wallet_stats(limit=limit)}
    except Exception as e:
        logger.error(f"Error querying wallets: {e}")
        raise HTTPException(500, str(e))

# ── Markets Overview API ───────────────────────────────────────────────────

@app.get("/api/markets/overview")
def get_markets_overview(limit: int = 20):
    """Get market-level statistics."""
    try:
        storage = _get_storage()
        return {"data": storage.get_market_overview(limit=limit)}
    except Exception as e:
        logger.error(f"Error querying markets: {e}")
        raise HTTPException(500, str(e))

# ── Logs API ───────────────────────────────────────────────────────────────

@app.get("/api/logs")
def get_logs(lines: int = 80):
    """Get recent log lines."""
    log_lines = []
    for lp in [APP_LOG_PATH, LOG_PATH]:
        if lp.exists():
            try:
                with open(lp, "r") as f:
                    log_lines.extend(f.readlines()[-lines:])
            except:
                pass
    if not log_lines:
        log_lines = ["No log files found."]
    return {"logs": log_lines[-lines:]}

# ── Results API ────────────────────────────────────────────────────────────

@app.get("/api/results")
def list_results():
    """List generated plot files."""
    if not PLOT_DIR.exists():
        return {"files": []}
    files = [f.name for f in PLOT_DIR.glob("*.png")]
    return {"files": sorted(files)}

# ── Comparison Data API ────────────────────────────────────────────────────

@app.get("/api/comparison-data")
def get_comparison_data():
    """Get model comparison metrics — reads from model_comparison.csv if available."""
    import pandas as pd
    comp_path = BASE_DIR / "results" / "model_comparison.csv"
    if comp_path.exists():
        try:
            df = pd.read_csv(comp_path)
            models = []
            for _, row in df.iterrows():
                models.append({
                    "name": row.get("Model", "Unknown"),
                    "accuracy": row.get("Accuracy", 0),
                    "precision": row.get("Precision", 0),
                    "recall": row.get("Recall", 0),
                    "f1": row.get("F1-Score", 0),
                    "auc_roc": row.get("AUC-ROC", 0),
                    "avg_precision": row.get("Avg Precision", 0),
                    "log_loss": row.get("Log Loss", 0),
                    "brier_score": row.get("Brier Score", 0),
                    "cv_auc": row.get("CV AUC (5-fold)", 0),
                    "train_time": row.get("Train Time (s)", 0),
                })
            return {"models": models, "source": "model_comparison.csv"}
        except Exception as e:
            logger.error(f"Error reading model_comparison.csv: {e}")

    # Fallback
    return {"models": [], "source": "none"}

# ── ML Training API ────────────────────────────────────────────────────────

_ml_state = {"running": False, "status": "idle", "message": "", "progress": 0}
_ml_lock = threading.Lock()
_ml_results = {}  # Cached latest results

def _run_ml_background():
    """Background task for ML training."""
    global _ml_state, _ml_results
    with _ml_lock:
        _ml_state = {"running": True, "status": "training", "message": "Starting ML pipeline", "progress": 5}
    try:
        import sys
        sys.path.insert(0, str(BASE_DIR))
        from src.models.trainer import ModelTrainer
        trainer = ModelTrainer()

        # Proxy status updates
        def poll_trainer():
            import time
            while _ml_state["running"]:
                s = trainer.status
                with _ml_lock:
                    if s["state"] != "idle":
                        _ml_state["message"] = s["message"]
                        _ml_state["progress"] = s["progress"]
                time.sleep(1)

        import threading as _thr
        poll_thread = _thr.Thread(target=poll_trainer, daemon=True)
        poll_thread.start()

        summary = trainer.run()

        with _ml_lock:
            _ml_results = summary
            _ml_state = {"running": False, "status": "done", "message": "ML training complete", "progress": 100}
    except Exception as e:
        with _ml_lock:
            _ml_state = {"running": False, "status": "error", "message": str(e), "progress": 0}
        logger.error(f"ML training failed: {e}")

@app.post("/api/ml/train")
async def trigger_ml_train(background_tasks: BackgroundTasks):
    """Trigger ML model training in background."""
    with _ml_lock:
        if _ml_state["running"]:
            raise HTTPException(409, "ML training is already running")
    with _pipeline_lock:
        if _pipeline_state["running"]:
            raise HTTPException(409, "A data pipeline is running, wait for it to finish")
    background_tasks.add_task(_run_ml_background)
    return {"message": "ML training triggered in background."}

@app.get("/api/ml/status")
def get_ml_status():
    """Get ML training status."""
    with _ml_lock:
        return _ml_state.copy()

@app.get("/api/ml/results")
def get_ml_results():
    """Get latest ML training results."""
    # Try reading from file first
    report_path = BASE_DIR / "results" / "training_report.json"
    if report_path.exists():
        try:
            with open(report_path) as f:
                return json.load(f)
        except:
            pass
    with _ml_lock:
        if _ml_results:
            return _ml_results
    return {"message": "No ML results available. Run training first."}

# ── Scrape Trigger API ─────────────────────────────────────────────────────

def _run_scrape_background(mode: str, max_trades: int):
    """Background task for scraping."""
    global _pipeline_state
    with _pipeline_lock:
        _pipeline_state = {"running": True, "status": "scraping", "message": f"Mode: {mode}", "progress": 10}
    try:
        import sys
        sys.path.insert(0, str(BASE_DIR))
        from src.data_ingestion.collector import collect_data
        collect_data(mode=mode, max_trades=max_trades)
        with _pipeline_lock:
            _pipeline_state = {"running": False, "status": "done", "message": "Scrape completed", "progress": 100}
    except Exception as e:
        with _pipeline_lock:
            _pipeline_state = {"running": False, "status": "error", "message": str(e), "progress": 0}

@app.post("/api/scrape")
async def trigger_scrape(background_tasks: BackgroundTasks, mode: str = "incremental", max_trades: int = 2900):
    """Trigger data scraping in background."""
    with _pipeline_lock:
        if _pipeline_state["running"]:
            raise HTTPException(409, "A pipeline is already running")
    background_tasks.add_task(_run_scrape_background, mode, max_trades)
    return {"message": f"Scrape ({mode}) triggered in background."}

# ── Pipeline Trigger API ───────────────────────────────────────────────────

def _run_pipeline_background(max_trades: int):
    """Background task for full pipeline."""
    global _pipeline_state
    with _pipeline_lock:
        _pipeline_state = {"running": True, "status": "pipeline", "message": "Running full pipeline", "progress": 5}
    try:
        import sys
        sys.path.insert(0, str(BASE_DIR))
        from src.data_ingestion.collector import run_pipeline
        run_pipeline(max_trades=max_trades)

        # Also generate visualizations
        with _pipeline_lock:
            _pipeline_state["message"] = "Generating visualizations..."
            _pipeline_state["progress"] = 85
        try:
            from src.visualization.dashboard import main as generate_plots
            generate_plots()
        except:
            pass

        with _pipeline_lock:
            _pipeline_state = {"running": False, "status": "done", "message": "Full pipeline completed", "progress": 100}
    except Exception as e:
        with _pipeline_lock:
            _pipeline_state = {"running": False, "status": "error", "message": str(e), "progress": 0}

@app.post("/api/pipeline")
async def trigger_pipeline(background_tasks: BackgroundTasks, max_trades: int = 2900):
    """Trigger full pipeline (scrape → preprocess → visualize) in background."""
    with _pipeline_lock:
        if _pipeline_state["running"]:
            raise HTTPException(409, "A pipeline is already running")
    background_tasks.add_task(_run_pipeline_background, max_trades)
    return {"message": "Full pipeline triggered in background."}

@app.get("/api/pipeline/status")
def get_pipeline_status():
    """Get current pipeline execution status."""
    with _pipeline_lock:
        return _pipeline_state.copy()

# ── Backtest Trigger (legacy) ──────────────────────────────────────────────

@app.post("/api/run-backtest")
async def trigger_backtest(background_tasks: BackgroundTasks, model: str = "xgboost"):
    """Trigger backtest pipeline."""
    venv_python = BASE_DIR / "venv" / "bin" / "python"
    backtest_script = str(BASE_DIR / "src" / "backtesting" / "run_backtest.py")
    viz_script = str(BASE_DIR / "src" / "visualization" / "dashboard.py")

    def run_script(script_path):
        try:
            subprocess.run([str(venv_python), script_path], capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Script failed: {script_path}\nError: {e.stderr}")

    background_tasks.add_task(run_script, backtest_script)
    background_tasks.add_task(run_script, viz_script)
    return {"message": f"Backtest pipeline for {model} triggered."}

# ── Frontend Hosting ───────────────────────────────────────────────────────

# Serve static frontend files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def serve_frontend():
    """Serve the main frontend page."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Frontend not found. Place index.html in /static/ directory."}

# ── Startup ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
