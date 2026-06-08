#!/usr/bin/env python
"""Build CLOB + secondary terminal-price fallback resolutions for SII sample data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_ingestion.external_sources import (  # noqa: E402
    MARKET_RESOLUTION_COLUMNS,
    apply_secondary_terminal_price_fallback,
    build_resolution_coverage_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-input", default="data/processed/sii_clob_market_resolutions.csv")
    parser.add_argument("--output", default="data/processed/sii_clob_plus_secondary_market_resolutions.csv")
    parser.add_argument("--summary-output", default="data/processed/sii_resolution_coverage_summary.json")
    args = parser.parse_args()

    strict = pd.read_csv(args.strict_input, low_memory=False)
    records = [apply_secondary_terminal_price_fallback(record) for record in strict.to_dict("records")]
    out = pd.DataFrame(records)
    for column in MARKET_RESOLUTION_COLUMNS:
        if column not in out.columns:
            out[column] = None
    out = out[MARKET_RESOLUTION_COLUMNS].drop_duplicates(subset=["market_id"], keep="last")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    summary = build_resolution_coverage_summary(out)
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {len(out):,} rows to {output_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
