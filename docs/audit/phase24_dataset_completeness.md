# Phase 24 Dataset Completeness Report

Generated at: `2026-05-15T11:53:13.149289+00:00`

## Dataset Snapshot

- DB transactions: `1730663`.
- Unique wallets: `205289`.
- Unique markets: `98120`.
- Time range: `2026-03-18 00:57:21.000000` to `2026-05-15 11:34:40.000000`.
- DB outcome non-missing rate: `0.999998266560272`.

## Resolution Coverage (Phase 26)

- Coverage: `98022/98120` (`99.9%`).
- Closed markets: `89681`.
- Yes winners: `9219`.
- No winners: `32109`.
- CLOB ratio (closed): `99.5%`.
- Legacy in closed: `0`.
- Fetch failed: `50` (`0.05%`).

## Funnel

- Enriched transactions: `1730663`.
- Cleaned transactions: `1310539`.
- Feature wallets: `15799`.
- Cleaned retention vs DB: `0.7572467892362638`.
- Feature wallets vs DB wallets: `0.07695979813823439`.

## Phase 21-26 Gates

| Gate | Status |
|---|---|
| collection_provenance_tables_exist | PASS |
| raw_api_page_archive_enabled | PASS |
| db_outcome_preserved | PASS |
| cleaned_outcome_preserved | PASS |
| resolution_schema_extended | PASS |
| resolution_coverage_ge_99pct | PASS |
| resolution_no_legacy_in_closed | PASS |
| resolution_fetch_failed_lt_1pct | PASS |
| resolution_clob_dominates_closed | PASS |
| completeness_report_generated | PASS |

## Known Gaps

- No completeness gaps were detected by the current checks.
