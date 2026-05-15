"""
Author: AI Assistant
Date: 2026-05-12
Description: Data quality reporting and artifact manifest generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "research.db"
RESULTS_DIR = PROJECT_ROOT / "results"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"
QUALITY_REPORT_PATH = RESULTS_DIR / "data_quality_report.json"

CRITICAL_COLUMNS = ("transaction_id", "address", "market_id", "timestamp")
ARTIFACT_PATTERNS = (
    "data/research.db",
    "data/raw/api_pages/*.json",
    "data/processed/*.csv",
    "data/features/*.csv",
    "results/*.csv",
    "results/*.json",
    "models/artifacts/*",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _query_scalar(conn: sqlite3.Connection, sql: str, default: Any = None) -> Any:
    try:
        return conn.execute(sql).fetchone()[0]
    except Exception as exc:
        logger.warning(f"Data quality query failed: {sql} | {exc}")
        return default


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _sqlite_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _hash_file(path: Path, full_hash_threshold_mb: int = 128) -> Dict[str, Any]:
    """
    Hash small artifacts fully. For very large files, hash only head/tail by default
    and mark the hash as partial unless FULL_MANIFEST_HASH=true is set.
    """
    size = path.stat().st_size
    full_hash = os.getenv("FULL_MANIFEST_HASH", "false").lower() == "true"
    threshold = full_hash_threshold_mb * 1024 * 1024
    digest = hashlib.sha256()

    if full_hash or size <= threshold:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return {"sha256": digest.hexdigest(), "hash_mode": "full"}

    sample_size = 1024 * 1024
    with path.open("rb") as f:
        digest.update(f.read(sample_size))
        if size > sample_size:
            f.seek(max(0, size - sample_size))
            digest.update(f.read(sample_size))
    digest.update(str(size).encode("utf-8"))
    return {"sha256": digest.hexdigest(), "hash_mode": "sample_head_tail_size"}


def _artifact_paths(patterns: Iterable[str]) -> Iterable[Path]:
    seen = set()
    for pattern in patterns:
        for path in PROJECT_ROOT.glob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path


def build_manifest(patterns: Iterable[str] = ARTIFACT_PATTERNS) -> Dict[str, Any]:
    """Create a manifest for standard data, model, and result artifacts."""
    artifacts = []
    for path in sorted(_artifact_paths(patterns)):
        rel_path = path.relative_to(PROJECT_ROOT).as_posix()
        hash_info = _hash_file(path)
        artifacts.append({
            "path": rel_path,
            "bytes": path.stat().st_size,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            **hash_info,
        })

    return {
        "generated_at": _utc_now(),
        "project_root": str(PROJECT_ROOT),
        "artifacts": artifacts,
    }


def build_data_quality_report(db_path: Path = DB_PATH) -> Dict[str, Any]:
    """Inspect transaction DB and key CSV artifacts without loading the full DB into memory."""
    report: Dict[str, Any] = {
        "generated_at": _utc_now(),
        "database": {
            "path": _display_path(db_path),
            "exists": db_path.exists(),
        },
        "transactions": {},
        "market_resolutions": {},
        "feature_matrix": {},
        "label_audit": {},
        "collection_provenance": {},
    }

    if db_path.exists():
        report["database"]["bytes"] = db_path.stat().st_size
        with sqlite3.connect(db_path) as conn:
            if _sqlite_table_exists(conn, "transactions"):
                total = _query_scalar(conn, "SELECT COUNT(*) FROM transactions", 0) or 0
                distinct_tx = _query_scalar(conn, "SELECT COUNT(DISTINCT transaction_id) FROM transactions", 0) or 0
                report["transactions"] = {
                    "row_count": total,
                    "unique_transaction_ids": distinct_tx,
                    "duplicate_transaction_ids": max(0, total - distinct_tx),
                    "unique_wallets": _query_scalar(conn, "SELECT COUNT(DISTINCT address) FROM transactions", 0),
                    "unique_markets": _query_scalar(conn, "SELECT COUNT(DISTINCT market_id) FROM transactions", 0),
                    "min_timestamp": _query_scalar(conn, "SELECT MIN(timestamp) FROM transactions"),
                    "max_timestamp": _query_scalar(conn, "SELECT MAX(timestamp) FROM transactions"),
                    "missing_critical": {
                        col: _query_scalar(
                            conn,
                            f"SELECT COUNT(*) FROM transactions WHERE {col} IS NULL OR TRIM(CAST({col} AS TEXT)) = ''",
                            0,
                        )
                        for col in CRITICAL_COLUMNS
                    },
                    "invalid_price_count": _query_scalar(
                        conn,
                        "SELECT COUNT(*) FROM transactions WHERE price IS NULL OR price < 0 OR price > 1",
                        0,
                    ),
                    "invalid_amount_count": _query_scalar(
                        conn,
                        "SELECT COUNT(*) FROM transactions WHERE amount IS NULL OR amount < 0",
                        0,
                    ),
                    "missing_outcome_count": _query_scalar(
                        conn,
                        "SELECT COUNT(*) FROM transactions WHERE outcome IS NULL OR TRIM(CAST(outcome AS TEXT)) = ''",
                        0,
                    ),
                }
            else:
                report["transactions"] = {"error": "transactions table not found"}
            if _sqlite_table_exists(conn, "collection_runs"):
                report["collection_provenance"]["collection_runs"] = {
                    "row_count": _query_scalar(conn, "SELECT COUNT(*) FROM collection_runs", 0),
                    "latest_started_at": _query_scalar(conn, "SELECT MAX(started_at) FROM collection_runs"),
                    "total_new_trades": _query_scalar(conn, "SELECT SUM(COALESCE(new_trades, 0)) FROM collection_runs", 0),
                }
            if _sqlite_table_exists(conn, "raw_api_pages"):
                report["collection_provenance"]["raw_api_pages"] = {
                    "row_count": _query_scalar(conn, "SELECT COUNT(*) FROM raw_api_pages", 0),
                    "run_count": _query_scalar(conn, "SELECT COUNT(DISTINCT run_id) FROM raw_api_pages", 0),
                    "payload_rows": _query_scalar(conn, "SELECT SUM(COALESCE(row_count, 0)) FROM raw_api_pages", 0),
                }

    res_path = PROJECT_ROOT / "data" / "processed" / "market_resolutions.csv"
    if res_path.exists():
        res_df = pd.read_csv(res_path, low_memory=False)
        resolution = res_df.get("resolution", pd.Series(dtype="object")).astype(str)
        report["market_resolutions"] = {
            "path": res_path.relative_to(PROJECT_ROOT).as_posix(),
            "row_count": int(len(res_df)),
            "unique_markets": int(res_df["market_id"].nunique()) if "market_id" in res_df else 0,
            "open_count": int((resolution == "__OPEN__").sum()),
            "missing_resolution_count": int(resolution.isna().sum()),
            "schema_columns": list(res_df.columns),
            "has_extended_resolution_schema": all(
                col in res_df.columns
                for col in ["closed", "winning_outcome", "outcomes_json", "outcome_prices_json", "resolution_method"]
            ),
        }

    feature_path = PROJECT_ROOT / "data" / "features" / "model_input.csv"
    if feature_path.exists():
        feature_df = pd.read_csv(feature_path)
        feature_report = {
            "path": feature_path.relative_to(PROJECT_ROOT).as_posix(),
            "row_count": int(len(feature_df)),
            "column_count": int(len(feature_df.columns)),
            "missing_values": int(feature_df.isna().sum().sum()),
        }
        if "split" in feature_df.columns:
            feature_report["split_counts"] = {
                str(k): int(v) for k, v in feature_df["split"].value_counts(dropna=False).to_dict().items()
            }
        if "Trader_Success_Rate" in feature_df.columns:
            feature_report["label_counts"] = {
                str(k): int(v) for k, v in feature_df["Trader_Success_Rate"].value_counts(dropna=False).to_dict().items()
            }
            if "split" in feature_df.columns:
                grouped = feature_df.groupby("split")["Trader_Success_Rate"].value_counts(dropna=False)
                feature_report["label_counts_by_split"] = {
                    f"{split}:{label}": int(count)
                    for (split, label), count in grouped.to_dict().items()
                }
        report["feature_matrix"] = feature_report

    label_summary_path = PROJECT_ROOT / "results" / "label_summary.json"
    if label_summary_path.exists():
        try:
            report["label_audit"] = json.loads(label_summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            report["label_audit"] = {"error": str(exc)}

    return report


def write_audit_artifacts() -> Dict[str, Any]:
    """Write the standard quality report and manifest files."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    report = build_data_quality_report()
    manifest = build_manifest()

    QUALITY_REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    logger.info(f"Wrote data quality report → {QUALITY_REPORT_PATH}")
    logger.info(f"Wrote artifact manifest → {MANIFEST_PATH}")
    return {"quality_report": report, "manifest": manifest}


def main():
    parser = argparse.ArgumentParser(description="Generate data quality report and artifact manifest")
    parser.add_argument("--report-only", action="store_true", help="Print report JSON without writing artifacts")
    args = parser.parse_args()

    if args.report_only:
        print(json.dumps(build_data_quality_report(), indent=2, default=str))
    else:
        write_audit_artifacts()


if __name__ == "__main__":
    main()
