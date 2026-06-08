#!/usr/bin/env python
"""Build diagnostic market_resolutions.csv from SII terminal outcome prices.

This is intentionally marked as untrusted for academic claims. It exists to
exercise the pipeline before CLOB winner-field resolutions are fetched.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd


def _parse_prices(raw: object) -> list[float]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return []
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transactions", default="data/processed/sii_sample_transactions.csv")
    parser.add_argument("--markets", default="data/external/sii_polymarket/markets.parquet")
    parser.add_argument("--output", default="data/processed/market_resolutions.csv")
    args = parser.parse_args()

    tx = pd.read_csv(args.transactions, usecols=["market_id"])
    market_ids = set(tx["market_id"].astype(str).dropna())
    markets = pd.read_parquet(
        args.markets,
        columns=[
            "condition_id",
            "question",
            "answer1",
            "answer2",
            "outcome_prices",
            "closed",
            "slug",
            "end_date",
        ],
    )
    markets = markets[markets["condition_id"].astype(str).isin(market_ids)].copy()

    rows = []
    for _, row in markets.iterrows():
        outcomes = [row.get("answer1"), row.get("answer2")]
        prices = _parse_prices(row.get("outcome_prices"))
        if len(prices) >= 2 and max(prices) >= 0.99:
            winner_idx = int(pd.Series(prices).idxmax())
            resolution = outcomes[winner_idx]
            winning_price = prices[winner_idx]
        else:
            winner_idx = -1
            resolution = "__OPEN__"
            winning_price = None
        rows.append(
            {
                "market_id": row["condition_id"],
                "question": row.get("question"),
                "resolution": resolution,
                "closed": bool(row.get("closed")),
                "winning_outcome": resolution if resolution != "__OPEN__" else None,
                "winning_outcome_index": winner_idx,
                "winning_outcome_price": winning_price,
                "outcomes_json": json.dumps(outcomes),
                "outcome_prices_json": json.dumps(prices),
                "slug": row.get("slug"),
                "resolution_method": "sii_terminal_outcome_prices_argmax_untrusted",
                "resolved_at": row.get("end_date"),
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows).drop_duplicates(subset=["market_id"])
    result.to_csv(out, index=False)
    print(f"Wrote {len(result):,} diagnostic resolutions to {out}")
    print(result["resolution"].value_counts(dropna=False).head().to_string())


if __name__ == "__main__":
    main()
