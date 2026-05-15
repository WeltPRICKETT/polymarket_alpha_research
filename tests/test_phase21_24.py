import sqlite3

import pandas as pd

from src.data_quality.dataset_completeness import (
    backfill_outcomes_from_enriched,
    build_dataset_completeness_report,
    normalise_market_resolution_schema,
)


def test_backfill_outcomes_from_enriched_updates_db_and_cleaned_csv(tmp_path):
    db_path = tmp_path / "research.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE transactions (
                transaction_id TEXT,
                address TEXT,
                market_id TEXT,
                outcome TEXT,
                price REAL,
                timestamp TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO transactions VALUES ('tx1', 'a', 'm1', NULL, 0.5, '2026-01-01')"
        )

    enriched_path = tmp_path / "real_trades_enriched.csv"
    pd.DataFrame({"transaction_id": ["tx1"], "_outcome": ["Down"]}).to_csv(enriched_path, index=False)
    cleaned_path = tmp_path / "cleaned_transactions.csv"
    pd.DataFrame({"transaction_id": ["tx1"], "outcome": [None]}).to_csv(cleaned_path, index=False)

    report = backfill_outcomes_from_enriched(
        db_path=db_path,
        enriched_path=enriched_path,
        cleaned_path=cleaned_path,
    )

    with sqlite3.connect(db_path) as conn:
        outcome = conn.execute("SELECT outcome FROM transactions WHERE transaction_id='tx1'").fetchone()[0]

    assert report["status"] == "ok"
    assert outcome == "Down"
    assert pd.read_csv(cleaned_path).loc[0, "outcome"] == "Down"


def test_normalise_market_resolution_schema_adds_academic_metadata(tmp_path):
    path = tmp_path / "market_resolutions.csv"
    pd.DataFrame({
        "market_id": ["m1", "m2"],
        "question": ["q1", "q2"],
        "resolution": ["Yes", "__OPEN__"],
        "slug": ["s1", "s2"],
    }).to_csv(path, index=False)

    report = normalise_market_resolution_schema(path)
    df = pd.read_csv(path)

    assert report["status"] == "ok"
    assert {"closed", "winning_outcome", "outcomes_json", "resolution_method"}.issubset(df.columns)
    assert df.loc[df["market_id"] == "m1", "winning_outcome"].iloc[0] == "Yes"


def test_dataset_completeness_report_flags_core_gates(tmp_path, monkeypatch):
    import src.data_quality.dataset_completeness as dc

    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    audit_dir = tmp_path / "docs" / "audit"
    (data_dir / "processed").mkdir(parents=True)
    (data_dir / "features").mkdir(parents=True)
    results_dir.mkdir()
    audit_dir.mkdir(parents=True)
    db_path = data_dir / "research.db"
    monkeypatch.setattr(dc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(dc, "DATA_DIR", data_dir)
    monkeypatch.setattr(dc, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(dc, "AUDIT_DIR", audit_dir)
    monkeypatch.setattr(dc, "DB_PATH", db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE transactions (
                transaction_id TEXT,
                address TEXT,
                market_id TEXT,
                outcome TEXT,
                price REAL,
                timestamp TEXT
            )
            """
        )
        conn.execute("CREATE TABLE collection_runs (run_id TEXT, started_at TEXT, new_trades INTEGER, error_count INTEGER)")
        conn.execute("CREATE TABLE raw_api_pages (run_id TEXT, row_count INTEGER)")
        conn.execute("INSERT INTO transactions VALUES ('tx1', 'a', 'm1', 'Yes', 0.5, '2026-01-01')")

    pd.DataFrame({
        "transaction_id": ["tx1"],
        "address": ["a"],
        "market_id": ["m1"],
        "_outcome": ["Yes"],
        "timestamp": ["2026-01-01"],
    }).to_csv(data_dir / "processed" / "real_trades_enriched.csv", index=False)
    pd.DataFrame({
        "transaction_id": ["tx1"],
        "address": ["a"],
        "market_id": ["m1"],
        "outcome": ["Yes"],
        "timestamp": ["2026-01-01"],
    }).to_csv(data_dir / "processed" / "cleaned_transactions.csv", index=False)
    pd.DataFrame({
        "market_id": ["m1"],
        "resolution": ["Yes"],
        "closed": [True],
        "winning_outcome": ["Yes"],
        "outcomes_json": ["[]"],
        "outcome_prices_json": ["[]"],
        "resolution_method": ["test"],
    }).to_csv(data_dir / "processed" / "market_resolutions.csv", index=False)
    pd.DataFrame({
        "address": ["a"],
        "split": ["test"],
        "Trader_Success_Rate": [1],
    }).to_csv(data_dir / "features" / "model_input.csv", index=False)

    report = build_dataset_completeness_report()

    assert report["phase21_24_gates"]["collection_provenance_tables_exist"] is True
    assert report["phase21_24_gates"]["db_outcome_preserved"] is True
    assert report["phase21_24_gates"]["resolution_schema_extended"] is True
