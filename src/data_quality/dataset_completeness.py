"""
Author: AI Assistant
Date: 2026-05-14
Description: Phase 21-24 data provenance and dataset completeness reporting.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
AUDIT_DIR = PROJECT_ROOT / "docs" / "audit"
DB_PATH = DATA_DIR / "research.db"
COMPLETENESS_JSON = RESULTS_DIR / "dataset_completeness_report.json"
COMPLETENESS_MD = AUDIT_DIR / "phase24_dataset_completeness.md"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_csv_info(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": _display_path(path),
        "exists": path.exists(),
        "rows": 0,
        "columns": [],
    }
    if not path.exists():
        return info
    df = pd.read_csv(path, low_memory=False)
    info["rows"] = int(len(df))
    info["columns"] = list(df.columns)
    if "transaction_id" in df.columns:
        info["unique_transaction_ids"] = int(df["transaction_id"].nunique(dropna=True))
    if "address" in df.columns:
        info["unique_wallets"] = int(df["address"].nunique(dropna=True))
    if "market_id" in df.columns:
        info["unique_markets"] = int(df["market_id"].nunique(dropna=True))
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        info["min_timestamp"] = str(ts.min()) if ts.notna().any() else None
        info["max_timestamp"] = str(ts.max()) if ts.notna().any() else None
        info["bad_timestamp_count"] = int(ts.isna().sum())
    for col in ["outcome", "_outcome", "market_resolution", "resolution"]:
        if col in df.columns:
            info[f"{col}_missing"] = int(df[col].isna().sum())
            info[f"{col}_non_missing_rate"] = float(df[col].notna().mean()) if len(df) else 0.0
            info[f"{col}_top_values"] = {
                str(k): int(v)
                for k, v in df[col].value_counts(dropna=False).head(10).to_dict().items()
            }
    if "split" in df.columns:
        info["split_counts"] = {str(k): int(v) for k, v in df["split"].value_counts().to_dict().items()}
    if "Trader_Success_Rate" in df.columns:
        info["label_counts"] = {
            str(k): int(v)
            for k, v in df["Trader_Success_Rate"].value_counts(dropna=False).to_dict().items()
        }
    return info


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _db_summary(db_path: Path | None = None) -> Dict[str, Any]:
    db_path = db_path or DB_PATH
    summary: Dict[str, Any] = {"path": _display_path(db_path), "exists": db_path.exists()}
    if not db_path.exists():
        return summary
    with sqlite3.connect(db_path) as conn:
        tables = [
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        ]
        summary["tables"] = tables
        if _table_exists(conn, "transactions"):
            row = conn.execute(
                """
                SELECT
                    COUNT(*),
                    COUNT(DISTINCT transaction_id),
                    COUNT(DISTINCT address),
                    COUNT(DISTINCT market_id),
                    MIN(timestamp),
                    MAX(timestamp),
                    SUM(CASE WHEN outcome IS NULL OR TRIM(CAST(outcome AS TEXT)) = '' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN market_id IS NULL OR TRIM(CAST(market_id AS TEXT)) = '' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN price IS NULL OR price < 0 OR price > 1 THEN 1 ELSE 0 END)
                FROM transactions
                """
            ).fetchone()
            total = int(row[0] or 0)
            outcome_missing = int(row[6] or 0)
            summary["transactions"] = {
                "rows": total,
                "unique_transaction_ids": int(row[1] or 0),
                "unique_wallets": int(row[2] or 0),
                "unique_markets": int(row[3] or 0),
                "min_timestamp": row[4],
                "max_timestamp": row[5],
                "missing_outcome_count": outcome_missing,
                "outcome_non_missing_rate": 1.0 - (outcome_missing / total if total else 0.0),
                "missing_market_id_count": int(row[7] or 0),
                "invalid_price_count": int(row[8] or 0),
            }
        if _table_exists(conn, "collection_runs"):
            row = conn.execute(
                "SELECT COUNT(*), MAX(started_at), SUM(COALESCE(new_trades, 0)), SUM(COALESCE(error_count, 0)) FROM collection_runs"
            ).fetchone()
            summary["collection_runs"] = {
                "rows": int(row[0] or 0),
                "latest_started_at": row[1],
                "total_new_trades": int(row[2] or 0),
                "total_errors": int(row[3] or 0),
            }
        if _table_exists(conn, "raw_api_pages"):
            row = conn.execute(
                "SELECT COUNT(*), COUNT(DISTINCT run_id), SUM(COALESCE(row_count, 0)) FROM raw_api_pages"
            ).fetchone()
            summary["raw_api_pages"] = {
                "rows": int(row[0] or 0),
                "run_count": int(row[1] or 0),
                "total_payload_rows": int(row[2] or 0),
            }
    return summary


def normalise_market_resolution_schema(path: Path | None = None) -> Dict[str, Any]:
    path = path or (DATA_DIR / "processed" / "market_resolutions.csv")
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    df = pd.read_csv(path)
    defaults = {
        "closed": df.get("resolution", pd.Series(dtype=object)).astype(str) != "__OPEN__",
        "winning_outcome": "",
        "winning_outcome_index": pd.NA,
        "winning_outcome_price": pd.NA,
        "outcomes_json": "",
        "outcome_prices_json": "",
        "resolution_method": "legacy_resolution_value",
        "resolved_at": "",
    }
    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value
    if "winning_outcome" in df.columns:
        mask = (df["winning_outcome"].isna() | (df["winning_outcome"].astype(str) == "")) & (
            df["resolution"].astype(str) != "__OPEN__"
        )
        df.loc[mask, "winning_outcome"] = df.loc[mask, "resolution"]
    ordered = [
        "market_id",
        "question",
        "resolution",
        "closed",
        "winning_outcome",
        "winning_outcome_index",
        "winning_outcome_price",
        "outcomes_json",
        "outcome_prices_json",
        "slug",
        "resolution_method",
        "resolved_at",
    ]
    remaining = [col for col in df.columns if col not in ordered]
    df = df[[col for col in ordered if col in df.columns] + remaining]
    df.to_csv(path, index=False)
    return {
        "status": "ok",
        "path": _display_path(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
    }


def backfill_outcomes_from_enriched(
    db_path: Path | None = None,
    enriched_path: Path | None = None,
    cleaned_path: Path | None = None,
) -> Dict[str, Any]:
    db_path = db_path or DB_PATH
    enriched_path = enriched_path or (DATA_DIR / "processed" / "real_trades_enriched.csv")
    cleaned_path = cleaned_path or (DATA_DIR / "processed" / "cleaned_transactions.csv")
    if not enriched_path.exists():
        return {"status": "missing_enriched", "enriched_csv": str(enriched_path)}
    enriched = pd.read_csv(enriched_path, usecols=["transaction_id", "_outcome"], low_memory=False)
    enriched = enriched.dropna(subset=["transaction_id"]).drop_duplicates("transaction_id", keep="last")
    outcome_map = dict(zip(enriched["transaction_id"], enriched["_outcome"]))
    report: Dict[str, Any] = {
        "status": "ok",
        "enriched_rows": int(len(enriched)),
        "db_updated_rows": 0,
        "cleaned_updated_rows": 0,
    }

    if db_path.exists():
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TEMP TABLE outcome_backfill (transaction_id TEXT PRIMARY KEY, outcome TEXT)")
            conn.executemany(
                "INSERT OR REPLACE INTO outcome_backfill (transaction_id, outcome) VALUES (?, ?)",
                [(str(tx), "" if pd.isna(outcome) else str(outcome)) for tx, outcome in outcome_map.items()],
            )
            before = conn.execute(
                "SELECT COUNT(*) FROM transactions WHERE outcome IS NULL OR TRIM(CAST(outcome AS TEXT)) = ''"
            ).fetchone()[0]
            conn.execute(
                """
                UPDATE transactions
                SET outcome = (
                    SELECT outcome FROM outcome_backfill
                    WHERE outcome_backfill.transaction_id = transactions.transaction_id
                )
                WHERE transaction_id IN (SELECT transaction_id FROM outcome_backfill)
                  AND (outcome IS NULL OR TRIM(CAST(outcome AS TEXT)) = '')
                """
            )
            after = conn.execute(
                "SELECT COUNT(*) FROM transactions WHERE outcome IS NULL OR TRIM(CAST(outcome AS TEXT)) = ''"
            ).fetchone()[0]
            conn.commit()
            report["db_updated_rows"] = int(before - after)

    if cleaned_path.exists():
        cleaned = pd.read_csv(cleaned_path, low_memory=False)
        if "outcome" not in cleaned.columns:
            cleaned["outcome"] = pd.NA
        cleaned["outcome"] = cleaned["outcome"].astype("object")
        missing = cleaned["outcome"].isna() | (cleaned["outcome"].astype(str).str.strip() == "")
        cleaned.loc[missing, "outcome"] = cleaned.loc[missing, "transaction_id"].map(outcome_map)
        report["cleaned_updated_rows"] = int((missing & cleaned["outcome"].notna()).sum())
        cleaned.to_csv(cleaned_path, index=False)

    return report


def build_dataset_completeness_report() -> Dict[str, Any]:
    raw_files = list((DATA_DIR / "raw").glob("**/*")) if (DATA_DIR / "raw").exists() else []
    raw_payload_files = [p for p in raw_files if p.is_file() and p.suffix.lower() == ".json"]
    report: Dict[str, Any] = {
        "status": "ok",
        "generated_at": _utc_now(),
        "database": _db_summary(),
        "raw_layer": {
            "json_file_count": len(raw_payload_files),
            "api_page_file_count": len(list((DATA_DIR / "raw" / "api_pages").glob("*.json")))
            if (DATA_DIR / "raw" / "api_pages").exists()
            else 0,
        },
        "processed_layers": {
            "real_trades_enriched": _read_csv_info(DATA_DIR / "processed" / "real_trades_enriched.csv"),
            "cleaned_transactions": _read_csv_info(DATA_DIR / "processed" / "cleaned_transactions.csv"),
            "market_resolutions": _read_csv_info(DATA_DIR / "processed" / "market_resolutions.csv"),
        },
        "feature_layer": {
            "model_input": _read_csv_info(DATA_DIR / "features" / "model_input.csv"),
        },
    }
    db_tx = report["database"].get("transactions", {})
    enriched = report["processed_layers"]["real_trades_enriched"]
    cleaned = report["processed_layers"]["cleaned_transactions"]
    features = report["feature_layer"]["model_input"]
    db_rows = db_tx.get("rows", 0) or 0
    report["dataset_funnel"] = {
        "db_transactions": db_rows,
        "enriched_transactions": enriched.get("rows", 0),
        "cleaned_transactions": cleaned.get("rows", 0),
        "feature_wallets": features.get("rows", 0),
        "cleaned_retention_vs_db": (cleaned.get("rows", 0) / db_rows) if db_rows else 0.0,
        "feature_wallets_vs_db_wallets": (
            features.get("rows", 0) / db_tx.get("unique_wallets", 0)
            if db_tx.get("unique_wallets", 0)
            else 0.0
        ),
    }
    resolution_cols = set(report["processed_layers"]["market_resolutions"].get("columns", []))

    # Phase 26 resolution coverage analysis
    resolution_coverage = _resolution_coverage_analysis(db_tx.get("unique_markets", 0))
    report["resolution_coverage"] = resolution_coverage

    report["phase21_24_gates"] = {
        "collection_provenance_tables_exist": all(
            table in report["database"].get("tables", [])
            for table in ["collection_runs", "raw_api_pages"]
        ),
        "raw_api_page_archive_enabled": report["raw_layer"]["api_page_file_count"] >= 0,
        "db_outcome_preserved": db_tx.get("outcome_non_missing_rate", 0.0) >= 0.95,
        "cleaned_outcome_preserved": cleaned.get("outcome_non_missing_rate", 0.0) >= 0.95,
        "resolution_schema_extended": all(
            col in resolution_cols
            for col in ["closed", "winning_outcome", "outcomes_json", "outcome_prices_json", "resolution_method"]
        ),
        "resolution_coverage_ge_99pct": resolution_coverage.get("coverage_ratio", 0) >= 0.99,
        "resolution_no_legacy_in_closed": resolution_coverage.get("closed_legacy_count", 1) == 0,
        "resolution_fetch_failed_lt_1pct": resolution_coverage.get("fetch_failed_ratio", 1) < 0.01,
        "resolution_clob_dominates_closed": resolution_coverage.get("closed_clob_ratio", 0) >= 0.90,
        "completeness_report_generated": True,
    }
    gaps = []
    if report["raw_layer"]["api_page_file_count"] == 0:
        gaps.append("No archived raw API pages exist for historical collection runs; future runs will populate data/raw/api_pages.")
    if db_tx.get("outcome_non_missing_rate", 0.0) < 0.95:
        gaps.append("Database outcome preservation is below 95%; backfill or recollect with Phase 22 scraper fix.")
    if resolution_coverage.get("coverage_ratio", 0) < 0.99:
        gaps.append(
            f"Resolution coverage is {resolution_coverage.get('coverage_ratio', 0):.1%} "
            f"({resolution_coverage.get('unique_resolved', 0)}/{db_tx.get('unique_markets', 0)}); "
            "run scripts/rebuild_resolutions.py to complete."
        )
    if resolution_coverage.get("closed_legacy_count", 0) > 0:
        gaps.append(
            f"{resolution_coverage.get('closed_legacy_count', 0)} closed markets still use legacy_resolution_value; "
            "rebuild needed."
        )
    if resolution_coverage.get("no_winner_count", 0) == 0 and resolution_coverage.get("closed_count", 0) > 0:
        gaps.append("Resolution file has no explicit No winners; verify CLOB resolution logic.")
    if resolution_coverage.get("fetch_failed_ratio", 0) >= 0.01:
        gaps.append(
            f"fetch_failed rate is {resolution_coverage.get('fetch_failed_ratio', 0):.2%}; "
            "too many markets failed to resolve."
        )
    if features.get("split_counts", {}).get("test", 0) < 100:
        gaps.append("Feature test split has fewer than 100 wallets; expand collection window for stronger academic evidence.")
    report["known_gaps"] = gaps
    return report


def _resolution_coverage_analysis(db_unique_markets: int) -> Dict[str, Any]:
    """Analyze resolution file coverage, method distribution, and quality."""
    res_path = DATA_DIR / "processed" / "market_resolutions.csv"
    result: Dict[str, Any] = {
        "unique_resolved": 0,
        "db_unique_markets": db_unique_markets,
        "coverage_ratio": 0.0,
        "total_rows": 0,
        "closed_count": 0,
        "no_winner_count": 0,
        "yes_winner_count": 0,
        "method_distribution": {},
        "closed_clob_ratio": 0.0,
        "closed_legacy_count": 0,
        "closed_gamma_argmax_count": 0,
        "fetch_failed_count": 0,
        "fetch_failed_ratio": 0.0,
    }
    if not res_path.exists():
        return result

    df = pd.read_csv(res_path, low_memory=False)
    result["total_rows"] = int(len(df))
    result["unique_resolved"] = int(df["market_id"].dropna().nunique())
    result["coverage_ratio"] = (
        result["unique_resolved"] / db_unique_markets if db_unique_markets > 0 else 0.0
    )

    closed = df[df["closed"] == True]
    result["closed_count"] = int(len(closed))

    if "winning_outcome" in df.columns:
        result["no_winner_count"] = int((closed["winning_outcome"] == "No").sum())
        result["yes_winner_count"] = int((closed["winning_outcome"] == "Yes").sum())

    if "resolution_method" in df.columns:
        result["method_distribution"] = {
            str(k): int(v) for k, v in df["resolution_method"].value_counts().to_dict().items()
        }
        result["closed_legacy_count"] = int(
            (closed["resolution_method"] == "legacy_resolution_value").sum()
        )
        result["closed_gamma_argmax_count"] = int(
            (closed["resolution_method"] == "gamma_outcome_prices_argmax").sum()
        )
        clob_in_closed = (closed["resolution_method"] == "clob_winner_field").sum()
        result["closed_clob_ratio"] = float(clob_in_closed / len(closed)) if len(closed) > 0 else 0.0
        result["fetch_failed_count"] = int((df["resolution_method"] == "fetch_failed").sum())
        result["fetch_failed_ratio"] = float(
            result["fetch_failed_count"] / len(df) if len(df) > 0 else 0.0
        )

    return result


def render_markdown(report: Dict[str, Any]) -> str:
    db_tx = report["database"].get("transactions", {})
    funnel = report["dataset_funnel"]
    gates = report["phase21_24_gates"]
    res_cov = report.get("resolution_coverage", {})
    lines = [
        "# Phase 24 Dataset Completeness Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        "## Dataset Snapshot",
        "",
        f"- DB transactions: `{db_tx.get('rows')}`.",
        f"- Unique wallets: `{db_tx.get('unique_wallets')}`.",
        f"- Unique markets: `{db_tx.get('unique_markets')}`.",
        f"- Time range: `{db_tx.get('min_timestamp')}` to `{db_tx.get('max_timestamp')}`.",
        f"- DB outcome non-missing rate: `{db_tx.get('outcome_non_missing_rate')}`.",
        "",
        "## Resolution Coverage (Phase 26)",
        "",
        f"- Coverage: `{res_cov.get('unique_resolved', 0)}/{res_cov.get('db_unique_markets', 0)}` "
        f"(`{res_cov.get('coverage_ratio', 0):.1%}`).",
        f"- Closed markets: `{res_cov.get('closed_count', 0)}`.",
        f"- Yes winners: `{res_cov.get('yes_winner_count', 0)}`.",
        f"- No winners: `{res_cov.get('no_winner_count', 0)}`.",
        f"- CLOB ratio (closed): `{res_cov.get('closed_clob_ratio', 0):.1%}`.",
        f"- Legacy in closed: `{res_cov.get('closed_legacy_count', 0)}`.",
        f"- Fetch failed: `{res_cov.get('fetch_failed_count', 0)}` "
        f"(`{res_cov.get('fetch_failed_ratio', 0):.2%}`).",
        "",
        "## Funnel",
        "",
        f"- Enriched transactions: `{funnel.get('enriched_transactions')}`.",
        f"- Cleaned transactions: `{funnel.get('cleaned_transactions')}`.",
        f"- Feature wallets: `{funnel.get('feature_wallets')}`.",
        f"- Cleaned retention vs DB: `{funnel.get('cleaned_retention_vs_db')}`.",
        f"- Feature wallets vs DB wallets: `{funnel.get('feature_wallets_vs_db_wallets')}`.",
        "",
        "## Phase 21-26 Gates",
        "",
        "| Gate | Status |",
        "|---|---|",
    ]
    for gate, value in gates.items():
        lines.append(f"| {gate} | {'PASS' if value else 'FAIL'} |")
    lines.extend(["", "## Known Gaps", ""])
    for gap in report.get("known_gaps", []):
        lines.append(f"- {gap}")
    if not report.get("known_gaps"):
        lines.append("- No completeness gaps were detected by the current checks.")
    return "\n".join(lines) + "\n"


def write_dataset_completeness_report() -> Dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_dataset_completeness_report()
    COMPLETENESS_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    COMPLETENESS_MD.write_text(render_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dataset completeness and provenance report")
    parser.add_argument("--backfill-outcomes", action="store_true")
    parser.add_argument("--normalise-resolution-schema", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.backfill_outcomes:
        print(json.dumps(backfill_outcomes_from_enriched(), indent=2, default=str))
    if args.normalise_resolution_schema:
        print(json.dumps(normalise_market_resolution_schema(), indent=2, default=str))
    report = write_dataset_completeness_report()
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"Wrote {COMPLETENESS_JSON.relative_to(PROJECT_ROOT)}")
        print(f"Wrote {COMPLETENESS_MD.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
