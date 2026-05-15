"""
Phase 26: Resolution verification and No-winner audit.

Verifies that market_resolutions.csv correctly identifies winners for all
outcome types (Yes, No, Up, Down, team names, etc.) using the CLOB API.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESOLUTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "market_resolutions.csv"


@pytest.fixture
def resolutions_df():
    assert RESOLUTIONS_PATH.exists(), f"market_resolutions.csv not found at {RESOLUTIONS_PATH}"
    return pd.read_csv(RESOLUTIONS_PATH, low_memory=False)


class TestPhase26ResolutionVerification:

    def test_no_winners_present(self, resolutions_df):
        """market_resolutions.csv contains 'No' winning outcomes (the main Phase 26 fix)."""
        closed = resolutions_df[resolutions_df["closed"] == True]
        no_winners = closed[closed["winning_outcome"] == "No"]
        assert len(no_winners) > 0, (
            f"No 'No' winners found in {len(closed)} closed markets. "
            "This was the known Phase 26 bug — resolution logic may still be broken."
        )

    def test_yes_winners_present(self, resolutions_df):
        """market_resolutions.csv still contains 'Yes' winners."""
        closed = resolutions_df[resolutions_df["closed"] == True]
        yes_winners = closed[closed["winning_outcome"] == "Yes"]
        assert len(yes_winners) > 0, "No 'Yes' winners found — regression"

    def test_diverse_outcomes(self, resolutions_df):
        """Multiple outcome types exist (not just Yes/No — also Up/Down, teams, etc.)."""
        closed = resolutions_df[resolutions_df["closed"] == True]
        unique_outcomes = closed["winning_outcome"].dropna().unique()
        assert len(unique_outcomes) >= 3, (
            f"Only {len(unique_outcomes)} unique outcomes: {unique_outcomes[:10]}. "
            "Expected diverse outcomes (Yes, No, Up, Down, team names, etc.)"
        )

    def test_no_all_yes_bias(self, resolutions_df):
        """Yes winners should not dominate 100% of closed markets (old bug indicator)."""
        closed = resolutions_df[resolutions_df["closed"] == True]
        if len(closed) == 0:
            pytest.skip("No closed markets")
        yes_ratio = (closed["winning_outcome"] == "Yes").sum() / len(closed)
        assert yes_ratio < 0.95, (
            f"Yes-winner ratio is {yes_ratio:.2%} — suspiciously high. "
            "The old bug always resolved to 'Yes'."
        )

    def test_resolution_method_uses_clob(self, resolutions_df):
        """Majority of resolutions should use clob_winner_field (the reliable method)."""
        closed = resolutions_df[resolutions_df["closed"] == True]
        if "resolution_method" not in closed.columns:
            pytest.skip("No resolution_method column")
        clob_count = (closed["resolution_method"] == "clob_winner_field").sum()
        total = len(closed)
        if total == 0:
            pytest.skip("No closed markets")
        clob_ratio = clob_count / total
        assert clob_ratio > 0.8, (
            f"Only {clob_ratio:.1%} of resolutions use clob_winner_field. "
            "Expected > 80% — CLOB is the reliable source."
        )

    def test_outcomes_json_is_valid(self, resolutions_df):
        """outcomes_json column contains valid JSON arrays."""
        sample = resolutions_df[resolutions_df["outcomes_json"].notna()].head(100)
        invalid = []
        for _, row in sample.iterrows():
            try:
                parsed = json.loads(row["outcomes_json"])
                assert isinstance(parsed, list)
            except (json.JSONDecodeError, AssertionError):
                invalid.append(row.get("market_id", "unknown"))
        assert len(invalid) == 0, f"{len(invalid)} rows with invalid outcomes_json"


class TestPhase26HardGates:
    """Hard gates that enforce full-rebuild completion. These MUST pass before Phase 26 is declared done."""

    @pytest.fixture
    def resolutions_df(self):
        assert RESOLUTIONS_PATH.exists()
        return pd.read_csv(RESOLUTIONS_PATH, low_memory=False)

    @pytest.fixture
    def db_market_count(self):
        import sqlite3
        db_path = PROJECT_ROOT / "data" / "research.db"
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(DISTINCT market_id) FROM transactions").fetchone()[0]
        conn.close()
        return count

    def test_resolution_coverage_ge_99_percent(self, resolutions_df, db_market_count):
        """market_resolutions.csv must cover >= 99% of DB markets."""
        unique_ids = resolutions_df["market_id"].dropna().nunique()
        coverage = unique_ids / db_market_count if db_market_count > 0 else 0
        assert coverage >= 0.99, (
            f"Resolution coverage is only {coverage:.1%} ({unique_ids}/{db_market_count}). "
            "Rebuild has not completed — run scripts/rebuild_resolutions.py to finish."
        )

    def test_no_legacy_resolution_value_in_closed(self, resolutions_df):
        """Closed markets must NOT use legacy_resolution_value method (known broken)."""
        closed = resolutions_df[resolutions_df["closed"] == True]
        if "resolution_method" not in closed.columns:
            pytest.fail("No resolution_method column")
        legacy = closed[closed["resolution_method"] == "legacy_resolution_value"]
        assert len(legacy) == 0, (
            f"{len(legacy)} closed markets still use legacy_resolution_value. "
            "These are invalid — rebuild did not cover them."
        )

    def test_no_gamma_argmax_in_closed(self, resolutions_df):
        """Closed markets should not rely on gamma_outcome_prices_argmax (unreliable source)."""
        closed = resolutions_df[resolutions_df["closed"] == True]
        if "resolution_method" not in closed.columns:
            pytest.skip("No resolution_method column")
        gamma = closed[closed["resolution_method"] == "gamma_outcome_prices_argmax"]
        gamma_ratio = len(gamma) / len(closed) if len(closed) > 0 else 0
        assert gamma_ratio < 0.01, (
            f"{gamma_ratio:.1%} of closed markets use gamma_outcome_prices_argmax. "
            "Expected < 1% — CLOB is the reliable source."
        )

    def test_fetch_failed_below_threshold(self, resolutions_df):
        """fetch_failed rate must be < 1% of total markets."""
        if "resolution_method" not in resolutions_df.columns:
            pytest.skip("No resolution_method column")
        failed = resolutions_df[resolutions_df["resolution_method"] == "fetch_failed"]
        total = len(resolutions_df)
        fail_rate = len(failed) / total if total > 0 else 0
        assert fail_rate < 0.01, (
            f"fetch_failed rate is {fail_rate:.2%} ({len(failed)}/{total}). "
            "Too many markets failed to resolve — check CLOB API access."
        )

    def test_clob_winner_dominates_closed(self, resolutions_df):
        """clob_winner_field must account for >= 90% of closed market resolutions."""
        closed = resolutions_df[resolutions_df["closed"] == True]
        if len(closed) == 0:
            pytest.skip("No closed markets")
        clob_count = (closed["resolution_method"] == "clob_winner_field").sum()
        ratio = clob_count / len(closed)
        assert ratio >= 0.90, (
            f"clob_winner_field only covers {ratio:.1%} of closed markets. "
            "Expected >= 90%."
        )


class TestResolutionLogicUnit:
    """Unit tests for the resolution parsing logic itself."""

    def test_clob_no_winner_correctly_identified(self):
        """CLOB API response with winner=True on 'No' token resolves correctly."""
        from src.data_ingestion.storage import Storage
        from src.data_ingestion.public_scraper import PublicScraper

        storage = Storage()
        scraper = PublicScraper(storage)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "question": "Will X happen?",
            "closed": True,
            "market_slug": "will-x-happen",
            "end_date_iso": "2026-01-01",
            "tokens": [
                {"outcome": "Yes", "winner": False, "price": 0},
                {"outcome": "No", "winner": True, "price": 1},
            ],
        }

        with patch.object(scraper.session, "get", return_value=mock_response):
            cid, result = scraper._fetch_single_resolution("0xtest123")

        assert result is not None
        assert result["winning_outcome"] == "No"
        assert result["resolution"] == "No"
        assert result["resolution_method"] == "clob_winner_field"
        assert result["closed"] is True

    def test_clob_yes_winner_correctly_identified(self):
        """CLOB API response with winner=True on 'Yes' token resolves correctly."""
        from src.data_ingestion.storage import Storage
        from src.data_ingestion.public_scraper import PublicScraper

        storage = Storage()
        scraper = PublicScraper(storage)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "question": "Will Y happen?",
            "closed": True,
            "market_slug": "will-y-happen",
            "end_date_iso": "2026-02-01",
            "tokens": [
                {"outcome": "Yes", "winner": True, "price": 1},
                {"outcome": "No", "winner": False, "price": 0},
            ],
        }

        with patch.object(scraper.session, "get", return_value=mock_response):
            cid, result = scraper._fetch_single_resolution("0xtest456")

        assert result["winning_outcome"] == "Yes"
        assert result["resolution"] == "Yes"

    def test_clob_open_market_no_winner(self):
        """CLOB API response for open market (no winner token) returns __OPEN__."""
        from src.data_ingestion.storage import Storage
        from src.data_ingestion.public_scraper import PublicScraper

        storage = Storage()
        scraper = PublicScraper(storage)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "question": "Will Z happen?",
            "closed": False,
            "market_slug": "will-z-happen",
            "tokens": [
                {"outcome": "Yes", "winner": False, "price": 0.6},
                {"outcome": "No", "winner": False, "price": 0.4},
            ],
        }

        with patch.object(scraper.session, "get", return_value=mock_response):
            cid, result = scraper._fetch_single_resolution("0xtest789")

        assert result["resolution"] == "__OPEN__"
        assert result["winning_outcome"] == ""

    def test_multi_outcome_winner(self):
        """Markets with non-binary outcomes (teams, etc.) resolve correctly."""
        from src.data_ingestion.storage import Storage
        from src.data_ingestion.public_scraper import PublicScraper

        storage = Storage()
        scraper = PublicScraper(storage)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "question": "Who will win?",
            "closed": True,
            "market_slug": "who-will-win",
            "end_date_iso": "2026-03-01",
            "tokens": [
                {"outcome": "Team A", "winner": False, "price": 0},
                {"outcome": "Team B", "winner": True, "price": 1},
                {"outcome": "Team C", "winner": False, "price": 0},
            ],
        }

        with patch.object(scraper.session, "get", return_value=mock_response):
            cid, result = scraper._fetch_single_resolution("0xtestmulti")

        assert result["winning_outcome"] == "Team B"
        assert result["resolution"] == "Team B"
