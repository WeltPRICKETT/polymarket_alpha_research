#!/usr/bin/env python
"""Assess local Polymarket events CSV and optional SII markets metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_ingestion.external_sources import (
    build_event_context_features,
    compare_event_market_ids,
    summarize_event_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-csv", default="polymarket_events.csv")
    parser.add_argument("--sii-markets", default="data/external/sii_polymarket/markets.parquet")
    parser.add_argument("--out-dir", default="results/new_data_sources")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    event_features = build_event_context_features(args.events_csv)
    event_features_path = Path("data/features/event_context_features.csv")
    event_features_path.parent.mkdir(parents=True, exist_ok=True)
    event_features.to_csv(event_features_path, index=False)

    assessment = {
        "events_csv": str(args.events_csv),
        "event_context_features": str(event_features_path),
        "events": summarize_event_metadata(event_features),
    }

    markets_path = Path(args.sii_markets)
    if markets_path.exists():
        markets = pd.read_parquet(markets_path)
        assessment["sii_markets"] = {
            "path": str(markets_path),
            "rows": int(len(markets)),
            "columns": list(markets.columns),
        }
        assessment["id_mapping"] = compare_event_market_ids(event_features, markets)
        markets.head(50).to_csv(out_dir / "sii_markets_head.csv", index=False)
    else:
        assessment["sii_markets"] = {
            "path": str(markets_path),
            "available": False,
            "next_step": "Run ./conda-env/bin/python scripts/download_sii_markets.py",
        }

    with (out_dir / "assessment.json").open("w") as f:
        json.dump(assessment, f, indent=2, default=str)

    print(json.dumps(assessment, indent=2, default=str))


if __name__ == "__main__":
    main()
