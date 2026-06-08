#!/usr/bin/env python
"""Build a canonical transaction sample from SII users/trades parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_ingestion.external_sources import (
    convert_sii_trades_to_transactions,
    convert_sii_users_to_transactions,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to SII users.parquet/trades.parquet or hf://users.parquet")
    parser.add_argument("--kind", choices=["users", "trades"], default="users")
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--output", default="data/processed/sii_sample_transactions.csv")
    args = parser.parse_args()

    if args.input.startswith("hf://"):
        repo_path = args.input.removeprefix("hf://").lstrip("/")
        fs = HfFileSystem()
        parquet_path = f"datasets/SII-WANGZJ/Polymarket_data/{repo_path}"
        with fs.open(parquet_path, "rb") as handle:
            pf = pq.ParquetFile(handle)
            batches = []
            remaining = args.rows
            for row_group in range(pf.num_row_groups):
                table = pf.read_row_group(row_group)
                part = table.to_pandas()
                batches.append(part)
                remaining -= len(part)
                if remaining <= 0:
                    break
        df = pd.concat(batches, ignore_index=True).head(args.rows)
    else:
        source = Path(args.input)
        if not source.exists():
            raise SystemExit(f"Input parquet not found: {source}")
        df = pd.read_parquet(source)
        if args.rows and len(df) > args.rows:
            df = df.head(args.rows)

    if args.kind == "users":
        converted = convert_sii_users_to_transactions(df)
    else:
        converted = convert_sii_trades_to_transactions(df)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    converted.to_csv(out, index=False)
    print(f"Wrote {len(converted):,} canonical transactions to {out}")
    print(converted.head().to_string(index=False))


if __name__ == "__main__":
    main()
