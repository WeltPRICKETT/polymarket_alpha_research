"""Rebuild market_resolutions.csv using CLOB API for accurate winner determination.

Supports --resume to continue from where a previous run left off.
"""

import sys
import sqlite3
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_ingestion.storage import Storage
from src.data_ingestion.public_scraper import PublicScraper
from loguru import logger
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "research.db"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "market_resolutions.csv"
BATCH_SIZE = 1000


def get_all_market_ids():
    db = sqlite3.connect(str(DB_PATH))
    cids = [r[0] for r in db.execute("SELECT DISTINCT market_id FROM transactions").fetchall()]
    db.close()
    return cids


def get_already_resolved():
    """Load market_ids that have already been successfully resolved via CLOB (for resume).

    fetch_failed and gamma_outcome_prices_argmax are NOT considered done — they need retry.
    """
    if not OUT_PATH.exists():
        return set()
    df = pd.read_csv(OUT_PATH, usecols=["market_id", "resolution_method"], low_memory=False)
    done = df[df["resolution_method"].isin(["clob_winner_field", "clob_no_winner"])]
    return set(done["market_id"].dropna().unique())


def rebuild(resume: bool = False):
    all_cids = get_all_market_ids()
    logger.info(f"Total unique markets in DB: {len(all_cids)}")

    # Resume support: skip already-resolved markets
    existing_records = []
    if resume and OUT_PATH.exists():
        already_done = get_already_resolved()
        logger.info(f"Resume mode: {len(already_done)} markets already resolved via CLOB")
        # Load existing records to preserve them
        existing_df = pd.read_csv(OUT_PATH, low_memory=False)
        done_df = existing_df[existing_df["market_id"].isin(already_done)]
        existing_records = done_df.to_dict("records")
        # Only process remaining
        remaining_cids = [cid for cid in all_cids if cid not in already_done]
        logger.info(f"Remaining to resolve: {len(remaining_cids)}")
    else:
        remaining_cids = all_cids
        logger.info(f"Full rebuild: {len(remaining_cids)} markets to resolve")

    storage = Storage()
    scraper = PublicScraper(storage, rate_limit_delay=0.1)

    new_records = []
    total_batches = (len(remaining_cids) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(remaining_cids))
        batch_cids = remaining_cids[start:end]

        resolutions = scraper.fetch_market_resolutions(batch_cids, max_workers=5)

        for cid in batch_cids:
            rec = resolutions.get(cid)
            if rec and isinstance(rec, dict):
                new_records.append(rec)
            else:
                new_records.append({
                    "market_id": cid,
                    "question": "",
                    "resolution": "__UNFETCHED__",
                    "closed": False,
                    "winning_outcome": "",
                    "winning_outcome_index": None,
                    "winning_outcome_price": 0.0,
                    "outcomes_json": "",
                    "outcome_prices_json": "",
                    "slug": "",
                    "resolution_method": "fetch_failed",
                    "resolved_at": "",
                })

        logger.info(f"  Batch {batch_idx+1}/{total_batches}: {len(resolutions)} resolved")

        # Checkpoint every 10 batches
        if (batch_idx + 1) % 10 == 0:
            _save_checkpoint(existing_records + new_records)

    _save_checkpoint(existing_records + new_records)
    total = len(existing_records) + len(new_records)
    logger.info(f"Done. Total records: {total} (existing: {len(existing_records)}, new: {len(new_records)})")

    from collections import Counter
    all_recs = existing_records + new_records
    outcomes = Counter(r.get("winning_outcome", "") for r in all_recs if r.get("winning_outcome"))
    logger.info(f"Top outcomes: {outcomes.most_common(10)}")
    methods = Counter(r.get("resolution_method", "") for r in all_recs)
    logger.info(f"Methods: {dict(methods)}")


def _save_checkpoint(records):
    df = pd.DataFrame(records)
    df.to_csv(OUT_PATH, index=False)
    logger.info(f"  Checkpoint: saved {len(records)} records to {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild market resolutions via CLOB API")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous run, skipping already-resolved markets")
    args = parser.parse_args()
    rebuild(resume=args.resume)
