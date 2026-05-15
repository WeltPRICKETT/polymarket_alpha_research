# Data Contract

This project keeps three data layers:

| Layer | Location | Purpose |
|---|---|---|
| Raw | `data/raw/` and public API responses | Source snapshots and API payloads |
| Processed | `data/processed/` and `data/research.db` | Cleaned transactions and market resolutions |
| Features | `data/features/` | Model-ready trader-level feature matrices |

## Collection Provenance

Every new public API harvest should create one row in `collection_runs` and one
`raw_api_pages` row per archived API response page. Raw page payloads are stored
under `data/raw/api_pages/` with a SHA-256 digest recorded in SQLite.

| Table | Required Fields | Notes |
|---|---|---|
| `collection_runs` | `run_id`, `mode`, `source`, `started_at`, `status` | Tracks harvest intent, scan counts, inserted trade counts, errors, and timestamp coverage |
| `raw_api_pages` | `run_id`, `endpoint`, `params_json`, `status_code`, `response_sha256`, `raw_path` | Links each stored API payload to a collection run |

Historical runs may not have archived API pages, but future harvests must populate
these tables so research artifacts can be traced back to source payloads.

## `transactions` Table

| Column | Type | Required | Notes |
|---|---:|---:|---|
| `transaction_id` | string | yes | Unique transaction hash or source id |
| `address` | string | yes | Trader wallet/proxy address |
| `market_id` | string | yes | Polymarket condition id |
| `side` | string | no | `BUY` or `SELL` when available |
| `outcome` | string | yes | Outcome token label such as `Yes`/`No`; required for post-Phase 22 research datasets |
| `amount` | float | no | Share size; must be non-negative when present |
| `price` | float | no | Share price; expected range `[0, 1]` |
| `timestamp` | datetime | yes | UTC execution time |

## `market_resolutions.csv`

Market resolution data must preserve the raw market shape rather than collapsing
resolved markets into a single composite label. The current academic schema is:

| Column | Notes |
|---|---|
| `market_id` | Polymarket condition id |
| `question` | Market question/title when available |
| `resolution` | Winning outcome label or `__OPEN__` |
| `closed` | Source closed flag |
| `winning_outcome` | Winning outcome string inferred from outcome prices |
| `winning_outcome_index` | Winning outcome position in `outcomes_json` |
| `winning_outcome_price` | Terminal price used for inference |
| `outcomes_json` | Raw outcomes array as JSON text |
| `outcome_prices_json` | Raw outcome prices array as JSON text |
| `slug` | Market slug when available |
| `resolution_method` | Inference method, such as `max_terminal_price` |
| `resolved_at` | Resolution timestamp when available |

## Quality Gates

The audit tool writes `results/data_quality_report.json` and `data/manifest.json`.

Minimum checks:

- `transaction_id`, `address`, `market_id`, and `timestamp` must not be blank.
- `transaction_id` duplicates are reported as data quality failures.
- `outcome` should be non-missing for cleaned research transactions.
- `price` must be in `[0, 1]` when present.
- `amount` must be non-negative when present.
- Feature matrices report split counts and label distribution by split.
- Phase 21-24 gates require collection provenance tables, outcome preservation,
  extended resolution schema, and a generated dataset completeness report.

Run manually:

```bash
python -m src.data_quality.audit
python -m src.data_quality.dataset_completeness
```

For large artifacts, the manifest stores a head/tail/size sample hash by default. Set
`FULL_MANIFEST_HASH=true` to compute full SHA-256 hashes for every artifact.
