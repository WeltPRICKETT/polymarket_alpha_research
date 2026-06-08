#!/usr/bin/env python
"""Fetch CLOB-only market resolutions for SII canonical transactions.

This script intentionally does not fall back to Gamma or terminal-price argmax.
Rows are suitable for replacing diagnostic SII resolutions when enough CLOB
winner-field coverage is available.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_ingestion.external_sources import (  # noqa: E402
    MARKET_RESOLUTION_COLUMNS,
    build_failed_clob_resolution_record,
    parse_clob_market_resolution,
)


CLOB_MARKET_URL = "https://clob.polymarket.com/markets/{condition_id}"


def load_market_ids(transactions_path: str | Path, limit: int | None = None) -> list[str]:
    tx = pd.read_csv(transactions_path, usecols=["market_id"])
    ids = (
        tx["market_id"]
        .dropna()
        .astype(str)
        .loc[lambda s: s.str.len() > 0]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    return ids[:limit] if limit else ids


def load_existing(output_path: Path) -> tuple[list[dict], set[str]]:
    if not output_path.exists():
        return [], set()
    existing = pd.read_csv(output_path, low_memory=False)
    if "market_id" not in existing.columns:
        return [], set()
    done_methods = {"clob_winner_field", "clob_no_winner"}
    done = set(
        existing.loc[existing.get("resolution_method", "").isin(done_methods), "market_id"]
        .dropna()
        .astype(str)
    )
    return existing.to_dict("records"), done


def fetch_one(condition_id: str, timeout: float, retries: int, delay: float) -> dict:
    session = requests.Session()
    last_status = None
    for attempt in range(retries + 1):
        try:
            resp = session.get(CLOB_MARKET_URL.format(condition_id=condition_id), timeout=timeout)
            last_status = resp.status_code
            if resp.status_code == 200:
                return parse_clob_market_resolution(condition_id, resp.json())
            if resp.status_code == 429:
                time.sleep(delay * (attempt + 1))
                continue
            break
        except requests.RequestException:
            time.sleep(delay * (attempt + 1))
    return build_failed_clob_resolution_record(condition_id, method=f"fetch_failed:{last_status or 'request_error'}")


def save_records(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    for column in MARKET_RESOLUTION_COLUMNS:
        if column not in df.columns:
            df[column] = None
    df = df[MARKET_RESOLUTION_COLUMNS].drop_duplicates(subset=["market_id"], keep="last")
    df.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transactions", default="data/processed/sii_sample_transactions.csv")
    parser.add_argument("--output", default="data/processed/sii_clob_market_resolutions.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()

    output_path = Path(args.output)
    market_ids = load_market_ids(args.transactions, limit=args.limit)
    existing_records, done_ids = load_existing(output_path) if args.resume else ([], set())
    pending = [market_id for market_id in market_ids if market_id not in done_ids]

    print(f"Markets requested: {len(market_ids):,}")
    print(f"Already CLOB-resolved: {len(done_ids):,}")
    print(f"Pending fetches: {len(pending):,}")

    records = existing_records
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_one, market_id, args.timeout, args.retries, args.delay): market_id
            for market_id in pending
        }
        for future in as_completed(futures):
            records.append(future.result())
            completed += 1
            if completed % args.checkpoint_every == 0:
                save_records(records, output_path)
                print(f"Checkpoint: {completed:,}/{len(pending):,} fetched")

    save_records(records, output_path)
    result = pd.read_csv(output_path, low_memory=False)
    print(f"Wrote {len(result):,} rows to {output_path}")
    print(result["resolution_method"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
