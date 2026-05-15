import sqlite3
from pathlib import Path


def test_backtest_entrypoint_imports_current_engine():
    from src.backtesting.run_backtest import main

    assert callable(main)


def test_data_fetch_error_is_defined():
    from src.data_ingestion.polymarket_client import DataFetchError

    assert issubclass(DataFetchError, RuntimeError)


def test_api_exposes_backtest_status_route():
    from src.visualization.api import app

    paths = {route.path for route in app.routes}
    assert "/api/backtest/status" in paths
    assert "/api/backtest/results" in paths


def test_data_quality_report_from_sqlite(tmp_path: Path):
    from src.data_quality.audit import build_data_quality_report

    db_path = tmp_path / "research.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT,
                address TEXT,
                market_id TEXT,
                side TEXT,
                outcome TEXT,
                amount REAL,
                price REAL,
                timestamp DATETIME
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO transactions
            (transaction_id, address, market_id, side, outcome, amount, price, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("tx1", "0xabc", "m1", "BUY", "Yes", 10.0, 0.4, "2026-05-01 00:00:00"),
                ("tx1", "0xabc", "m1", "BUY", "Yes", 10.0, 1.2, "2026-05-01 00:01:00"),
                ("tx2", "", "m2", "SELL", "No", -1.0, 0.5, "2026-05-01 00:02:00"),
            ],
        )

    report = build_data_quality_report(db_path)

    assert report["database"]["exists"] is True
    assert report["transactions"]["row_count"] == 3
    assert report["transactions"]["duplicate_transaction_ids"] == 1
    assert report["transactions"]["missing_critical"]["address"] == 1
    assert report["transactions"]["invalid_price_count"] == 1
    assert report["transactions"]["invalid_amount_count"] == 1
    assert report["transactions"]["missing_outcome_count"] == 0


def test_storage_preserves_outcome_and_provenance_tables(tmp_path: Path):
    from src.data_ingestion.storage import Storage

    db_path = tmp_path / "research.db"
    storage = Storage(db_url=f"sqlite:///{db_path}")
    storage.save_transactions([
        {
            "transaction_id": "tx1",
            "address": "0xabc",
            "market_id": "m1",
            "side": "BUY",
            "_outcome": "Up",
            "amount": 10.0,
            "price": 0.4,
            "timestamp": "2026-05-01T00:00:00Z",
        }
    ])
    storage.start_collection_run("run1", mode="incremental", source="test", max_trades=100)
    storage.record_raw_api_page("run1", "/trades", '{"limit": 1}', 200, "abc", 1, "data/raw/api_pages/run1.json")
    storage.finish_collection_run("run1", status="ok", scanned_trades=1, new_trades=1, raw_pages=1)

    with sqlite3.connect(db_path) as conn:
        outcome = conn.execute("SELECT outcome FROM transactions WHERE transaction_id='tx1'").fetchone()[0]
        run_count = conn.execute("SELECT COUNT(*) FROM collection_runs").fetchone()[0]
        page_count = conn.execute("SELECT COUNT(*) FROM raw_api_pages").fetchone()[0]

    assert outcome == "Up"
    assert run_count == 1
    assert page_count == 1
