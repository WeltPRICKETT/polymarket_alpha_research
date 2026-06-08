#!/usr/bin/env python3
"""Audit and materialize the new SII markets + Polymarket events data sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_ingestion.external_sources import (
    build_event_context_features,
    build_sii_market_event_map,
    build_wallet_event_features,
    compare_event_market_ids,
    summarize_event_metadata,
    summarize_market_event_map,
)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _top_counts(series: pd.Series, n: int = 15) -> list[dict[str, Any]]:
    counts = series.fillna("__missing__").astype(str).value_counts().head(n)
    return [{"value": idx, "count": int(value)} for idx, value in counts.items()]


def _write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    coverage = summary["market_event_map"]
    wallet = summary["wallet_event_features"]
    lines = [
        "# New Data Sources Audit",
        "",
        "Generated: 2026-05-20",
        "",
        "## Sources",
        "",
        f"- SII markets parquet: `{summary['inputs']['markets_parquet']}`",
        f"- Polymarket events CSV: `{summary['inputs']['events_csv']}`",
        f"- Sample transactions: `{summary['inputs']['transactions_csv']}`",
        "",
        "## Coverage",
        "",
        f"- Sample markets: {coverage['sample_markets']:,}",
        f"- SII market metadata matched: {coverage['sii_market_metadata_matched']:,} "
        f"({coverage['sii_market_metadata_coverage']:.2%})",
        f"- Local event metadata matched: {coverage['local_event_metadata_matched']:,} "
        f"({coverage['local_event_metadata_coverage']:.2%})",
        f"- Unique events in sample: {coverage['unique_events']:,}",
        f"- Unique event slugs in sample: {coverage['unique_event_slugs']:,}",
        "",
        "## Wallet Event Features",
        "",
        f"- Wallet rows: {wallet['rows']:,}",
        f"- Mean unique events per wallet: {wallet['mean_unique_events']:.3f}",
        f"- Mean event diversification: {wallet['mean_event_diversification']:.4f}",
        f"- Mean same-event multi-market share: {wallet['mean_same_event_multi_market_share']:.4f}",
        "",
        "## Top Sample Events",
        "",
    ]
    for row in summary["top_sample_events"]:
        lines.append(f"- {row['value']}: {row['count']}")
    lines.extend(["", "## Top Sample Topics", ""])
    for row in summary["top_sample_topics"]:
        lines.append(f"- {row['value']}: {row['count']}")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Event context features: `{summary['outputs']['event_context_features']}`",
            f"- Market-event map: `{summary['outputs']['market_event_map']}`",
            f"- Wallet event features: `{summary['outputs']['wallet_event_features']}`",
            f"- JSON summary: `{summary['outputs']['audit_json']}`",
            "",
            "## Recommendation",
            "",
            "Use the wallet event features as an additional live feature family for the next "
            "strict-vs-secondary experiment. The current join coverage is high enough to make "
            "event-level specialization and concentration analysis meaningful.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    events_csv = Path(args.events_csv)
    markets_parquet = Path(args.markets_parquet)
    transactions_csv = Path(args.transactions_csv)

    event_features = build_event_context_features(events_csv)
    event_features_path = Path(args.event_features_output)
    event_features_path.parent.mkdir(parents=True, exist_ok=True)
    event_features.to_csv(event_features_path, index=False)

    markets = pd.read_parquet(markets_parquet)
    transactions = pd.read_csv(transactions_csv, low_memory=False)
    market_event_map = build_sii_market_event_map(transactions, markets, event_features)
    market_event_map_path = Path(args.market_event_map_output)
    market_event_map_path.parent.mkdir(parents=True, exist_ok=True)
    market_event_map.to_csv(market_event_map_path, index=False)

    wallet_event_features = build_wallet_event_features(transactions, market_event_map)
    wallet_features_path = Path(args.wallet_features_output)
    wallet_features_path.parent.mkdir(parents=True, exist_ok=True)
    wallet_event_features.to_csv(wallet_features_path, index=False)

    wallet_summary = {
        "rows": int(len(wallet_event_features)),
        "mean_unique_events": float(wallet_event_features["unique_events"].mean())
        if not wallet_event_features.empty
        else 0.0,
        "mean_event_diversification": float(wallet_event_features["event_diversification"].mean())
        if not wallet_event_features.empty
        else 0.0,
        "mean_event_notional_hhi": float(wallet_event_features["event_notional_hhi"].mean())
        if not wallet_event_features.empty
        else 0.0,
        "mean_top_event_notional_share": float(wallet_event_features["top_event_notional_share"].mean())
        if not wallet_event_features.empty
        else 0.0,
        "mean_same_event_multi_market_share": float(
            wallet_event_features["same_event_multi_market_share"].mean()
        )
        if not wallet_event_features.empty
        else 0.0,
    }

    summary = {
        "inputs": {
            "events_csv": str(events_csv),
            "markets_parquet": str(markets_parquet),
            "transactions_csv": str(transactions_csv),
        },
        "outputs": {
            "event_context_features": str(event_features_path),
            "market_event_map": str(market_event_map_path),
            "wallet_event_features": str(wallet_features_path),
            "audit_json": args.audit_json_output,
            "audit_report": args.audit_report_output,
        },
        "event_metadata": summarize_event_metadata(event_features),
        "event_market_join_probe": compare_event_market_ids(event_features, markets),
        "market_event_map": summarize_market_event_map(market_event_map),
        "wallet_event_features": wallet_summary,
        "top_sample_events": _top_counts(market_event_map.get("event_title", pd.Series(dtype=str))),
        "top_sample_categories": _top_counts(market_event_map.get("category", pd.Series(dtype=str))),
        "top_sample_tags": _top_counts(market_event_map.get("primary_tag_slug", pd.Series(dtype=str))),
        "top_sample_topics": _top_counts(market_event_map.get("topic_slug", pd.Series(dtype=str))),
    }
    _write_json(Path(args.audit_json_output), summary)
    _write_markdown_report(Path(args.audit_report_output), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-csv", default="polymarket_events.csv")
    parser.add_argument(
        "--markets-parquet",
        default="data/external/sii_polymarket_data/markets.parquet",
    )
    parser.add_argument(
        "--transactions-csv",
        default="data/processed/sii_sample_transactions.csv",
    )
    parser.add_argument(
        "--event-features-output",
        default="data/features/event_context_features.csv",
    )
    parser.add_argument(
        "--market-event-map-output",
        default="data/processed/sii_market_event_map.csv",
    )
    parser.add_argument(
        "--wallet-features-output",
        default="data/features/sii_wallet_event_features.csv",
    )
    parser.add_argument(
        "--audit-json-output",
        default="results/new_data_sources/sii_new_data_source_audit.json",
    )
    parser.add_argument(
        "--audit-report-output",
        default="results/new_data_sources/sii_new_data_source_audit.md",
    )
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
