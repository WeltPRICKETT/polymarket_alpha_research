#!/usr/bin/env python
"""Create a separate SQLite research DB from canonical SII sample transactions."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transactions", default="data/processed/sii_sample_transactions.csv")
    parser.add_argument("--output-db", default="data/sii_sample_research.db")
    args = parser.parse_args()

    tx_path = Path(args.transactions)
    if not tx_path.exists():
        raise SystemExit(f"Transactions CSV not found: {tx_path}")

    df = pd.read_csv(tx_path)
    out = Path(args.output_db)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    with sqlite3.connect(out) as conn:
        df.to_sql("transactions", conn, index_label="id", if_exists="replace")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_transactions_transaction_id ON transactions(transaction_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_transactions_address ON transactions(address)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_transactions_market_id ON transactions(market_id)")
        conn.execute(
            """
            CREATE TABLE positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                address TEXT,
                market_id TEXT,
                token_balance REAL,
                avg_price REAL,
                last_updated DATETIME
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE markets (
                market_id TEXT PRIMARY KEY,
                question TEXT,
                resolved BOOLEAN,
                resolution_outcome TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE collection_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                mode TEXT,
                source TEXT,
                started_at DATETIME,
                finished_at DATETIME,
                status TEXT,
                max_trades INTEGER,
                scanned_trades INTEGER,
                new_trades INTEGER,
                error_count INTEGER,
                raw_pages INTEGER,
                first_trade_timestamp DATETIME,
                last_trade_timestamp DATETIME,
                notes TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE raw_api_pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                endpoint TEXT,
                params_json TEXT,
                status_code INTEGER,
                response_sha256 TEXT,
                row_count INTEGER,
                fetched_at DATETIME,
                raw_path TEXT
            )
            """
        )
    print(f"Wrote {len(df):,} transactions to {out}")


if __name__ == "__main__":
    main()
