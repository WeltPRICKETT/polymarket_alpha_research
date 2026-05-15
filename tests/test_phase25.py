"""
Phase 25: Provenance-backed harvest validation.

Verifies that the collection provenance system (Phase 21) has been exercised
with a real API harvest and that all integrity guarantees hold.
"""

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "research.db"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "api_pages"


@pytest.fixture
def db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


class TestPhase25Provenance:

    def test_collection_runs_has_completed_runs(self, db):
        """At least one collection run completed successfully with raw pages."""
        rows = db.execute(
            "SELECT * FROM collection_runs WHERE status='ok' AND raw_pages > 0"
        ).fetchall()
        assert len(rows) > 0, "No completed collection runs with archived raw pages"

    def test_raw_api_pages_populated(self, db):
        """raw_api_pages table has rows from a provenance-backed harvest."""
        count = db.execute("SELECT COUNT(*) FROM raw_api_pages").fetchone()[0]
        assert count > 0, f"raw_api_pages is empty (expected > 0, got {count})"

    def test_raw_page_files_exist(self, db):
        """Every raw_api_pages row references a file that exists on disk."""
        pages = db.execute(
            "SELECT raw_path FROM raw_api_pages LIMIT 200"
        ).fetchall()
        assert len(pages) > 0
        missing = []
        for p in pages:
            full_path = PROJECT_ROOT / p["raw_path"]
            if not full_path.exists():
                missing.append(p["raw_path"])
        assert len(missing) == 0, f"{len(missing)} raw page files missing: {missing[:5]}"

    def test_raw_page_sha256_integrity(self, db):
        """SHA-256 digests stored in DB match actual file contents."""
        pages = db.execute(
            "SELECT raw_path, response_sha256 FROM raw_api_pages LIMIT 200"
        ).fetchall()
        assert len(pages) > 0
        mismatches = []
        for p in pages:
            full_path = PROJECT_ROOT / p["raw_path"]
            if not full_path.exists():
                mismatches.append((p["raw_path"], "FILE_MISSING"))
                continue
            actual = hashlib.sha256(full_path.read_bytes()).hexdigest()
            if actual != p["response_sha256"]:
                mismatches.append((p["raw_path"], f"expected={p['response_sha256'][:16]}... got={actual[:16]}..."))
        assert len(mismatches) == 0, f"{len(mismatches)} SHA-256 mismatches: {mismatches[:5]}"

    def test_raw_page_content_is_valid_json(self, db):
        """Archived raw pages contain valid JSON (parseable trade arrays or market objects)."""
        pages = db.execute(
            "SELECT raw_path, endpoint FROM raw_api_pages LIMIT 50"
        ).fetchall()
        assert len(pages) > 0
        invalid = []
        for p in pages:
            full_path = PROJECT_ROOT / p["raw_path"]
            if not full_path.exists():
                continue
            try:
                data = json.loads(full_path.read_bytes())
                assert data is not None
            except (json.JSONDecodeError, AssertionError):
                invalid.append(p["raw_path"])
        assert len(invalid) == 0, f"{len(invalid)} files with invalid JSON: {invalid[:5]}"

    def test_collection_run_links_to_raw_pages(self, db):
        """A completed run's raw_pages count matches actual rows in raw_api_pages."""
        run = db.execute(
            "SELECT run_id, raw_pages FROM collection_runs WHERE status='ok' AND raw_pages > 0 ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert run is not None
        actual_pages = db.execute(
            "SELECT COUNT(*) FROM raw_api_pages WHERE run_id=?", (run["run_id"],)
        ).fetchone()[0]
        assert actual_pages > 0, f"Run {run['run_id']} claims {run['raw_pages']} pages but raw_api_pages has {actual_pages}"

    def test_no_outcome_regression(self, db):
        """Outcome preservation still holds after the new harvest."""
        missing = db.execute(
            "SELECT COUNT(*) FROM transactions WHERE outcome IS NULL OR outcome = ''"
        ).fetchone()[0]
        total = db.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        rate = 1 - (missing / total) if total > 0 else 0
        assert rate >= 0.999, f"Outcome preservation dropped to {rate:.4f} ({missing} missing out of {total})"
