# AI Handoff Log

Last updated: 2026-05-15T20:00+08:00

Purpose: this file is the first stop for any new AI agent or IDE session taking over the project. It summarizes what has already been done, what is currently trusted, what remains weak, and which commands should be run before making further changes.

---

## Quick Status Sentence

**PHASE 30 FROZEN** — release candidate ready for submission. All phases 25–30 complete. `bash scripts/reproduce.sh` exits 0. No further work required for submission.

---

## Current Project State

Academic research pipeline for identifying informed Polymarket traders. **Frozen release candidate** — all phases complete, results are honest, pipeline is reproducible.

| Area | Status |
|---|---|
| Phase 0 baseline | PASS |
| Phase 1-5 core pipeline gates | PASS |
| Phase 6 walk-forward formal validity | PASS |
| Phase 7 future-return label | PASS |
| Phase 8-10 sensitivity and rolling-origin validation | PASS |
| Phase 11-12 threshold policy and model walk-forward | PASS |
| Phase 13-19 execution realism, random baselines, concentration, bootstrap | PASS |
| Phase 20 academic evidence table and guardrails | PASS |
| Phase 21 collection provenance tables | PASS |
| Phase 22 outcome preservation | PASS |
| Phase 23 extended resolution schema | PASS |
| Phase 24 dataset completeness report | PASS (all gates green) |
| Phase 25 provenance-backed harvest | PASS |
| Phase 26 resolution verification (CLOB-based) | **PASS — all 15 tests including 5 hard gates** |
| Phase 27A label strategy review | **DONE — future_return primary, resolution robustness** |
| Phase 28 reproducibility package | **PASS — reproduce.sh exits 0, all artifacts verified** |
| Phase 29 commercial hardening | **DONE — model retrained, versioning added, health endpoint** |
| Phase 30 release candidate freeze | **FROZEN — honest metrics, interpretation docs, artifact manifest** |
| Downstream rebuild (future_return) | **DONE — train=13,654 / val=1,119 / test=1,026** |

---

## Work Completed This Session (2026-05-15)

### Session 1: Phase 25 + Phase 26 Core Fix

1. **Phase 25 — Provenance Harvest** (DONE)
   - Ran live 500-trade incremental harvest
   - Verified `collection_runs` + `raw_api_pages` + filesystem files + SHA-256 integrity
   - Created `tests/test_phase25.py` (7 tests, all PASS)
   - DB: 1,701,286 transactions, 2,472 raw API pages archived

2. **Phase 26 — Resolution Bug Discovery & Fix** (DONE)
   - Discovered Gamma API `/markets?conditionId=X` ignores conditionId entirely
   - All 35,173 "closed" markets were resolved as "Yes" due to argmax on unrelated open market
   - Switched to CLOB API (`clob.polymarket.com/markets/{cid}`) which has explicit `winner` boolean
   - 500-market validation: No=157, Down=67, Up=69, Yes=49 — diverse and correct
   - Created `tests/test_phase26.py` with unit tests (4 PASS)
   - Started full rebuild: `scripts/rebuild_resolutions.py`

### Session 2: Gate Hardening + Rebuild Management

3. **Hard Gate Tests Added** (`tests/test_phase26.py::TestPhase26HardGates`)
   - `test_resolution_coverage_ge_99_percent` — PASS (99.8%)
   - `test_no_legacy_resolution_value_in_closed` — PASS (0 legacy)
   - `test_no_gamma_argmax_in_closed` — PASS (0 gamma argmax)
   - `test_fetch_failed_below_threshold` — PASS (0.05%)
   - `test_clob_winner_dominates_closed` — PASS (99.5%)

4. **Completeness Report Enhanced** (`src/data_quality/dataset_completeness.py`)
   - All 10 gates PASS including 4 new resolution gates

5. **Full Rebuild Completed** (`scripts/rebuild_resolutions.py`)
   - 96,730 markets resolved
   - Methods: clob_winner_field=89,185, clob_no_winner=7,495, fetch_failed=50
   - Coverage: 99.8%

### Session 3: Downstream Pipeline Rebuild + Label Strategy

6. **Resolution label rebuild** (`--label-mode auto`)
   - 35,885 traders — train=33,648 / val=977 / test=1,260
   - Saved as `data/features/model_input_resolution.csv`

7. **Future-return label rebuild** (`--label-mode future_return`)
   - 15,799 eligible traders — train=13,654 / val=1,119 / test=1,026
   - Positive rate: ~20% (balanced)
   - Decision: future_return = primary label, resolution = robustness check

8. **Full Regression Verified** (both modes)
   - Phase 0 audit: all phases PASS, `phase7_future_return_label_active=true`
   - Test suite: 60 passed, 3 warnings (sklearn precision — expected)
   - Completeness report: all 10 gates PASS

### Session 4: Phase 28 — Reproducibility Package

9. **reproduce.sh** (`scripts/reproduce.sh`)
   - Single-command: audit → completeness → phase0 → pipeline → tests → artifact check
   - Exits 0 with ALL CHECKS PASSED

10. **README.md updated**
    - Dataset statistics updated (1.7M transactions, 203k wallets, 96k markets)
    - Collection period corrected (March–May 2026)
    - Resolution methodology section added
    - ML pipeline summary table added

11. **docs/reproducibility.md** created
    - Step-by-step verification guide
    - Evidence artifact table with paths
    - Phase-by-phase audit trail index
    - Label strategy explanation

---

## Current Data State

```
DB transactions:        1,706,505
DB unique wallets:      203,088+
DB unique markets:      96,884

market_resolutions.csv:
  rows:                 96,730
  clob_winner_field:    89,185
  clob_no_winner:       7,495
  fetch_failed:         50 (0.05%)
  coverage:             99.8%
  closed_markets:       89,597
  No winners:           32,109
  Yes winners:          9,219
  closed_legacy:        0
  closed_gamma_argmax:  0

model_input.csv (future_return mode — primary):
  eligible wallets:     15,799
  train:                13,654
  val:                  1,119
  test:                 1,026
  positive rate:        ~20% (balanced)
  label strategy:       future_return (independent)

model_input_resolution.csv (resolution mode — robustness check):
  total wallets:        35,885
  train:                33,648
  val:                  977
  test:                 1,260
```

---

## Frozen Artifact Manifest

```
model_input.csv SHA-256 prefix:  b1debd9bae99721e
best_model.pkl:                  XGBoost, AUC=0.593
Label strategy:                  future_return (independent)
Resolution coverage:             99.8% (CLOB-verified)
Test suite:                      60 passed, 3 warnings
Completeness gates:              10/10 PASS
reproduce.sh:                    exits 0 (verified 2026-05-15)
```

## What This Project Claims

This is an **empirical methodology contribution**, not an alpha signal. Specifically:

1. We built a fully reproducible ML pipeline for informed trader identification on Polymarket
2. We discovered and fixed a systemic resolution bug in the Gamma API (conditionId filter non-functional)
3. We demonstrate that future-return prediction is genuinely difficult (AUC=0.593) — models provide modest ranking signal but not reliable binary classification
4. Top-10% selection achieves 1.24x precision lift over base rate — real but small
5. The pipeline includes 60 automated tests, 10 data quality gates, temporal walk-forward validation, and honest reporting

Earlier iterations reported inflated metrics (AUC>0.99, 96.9% precision) that were artifacts of broken resolution labels and tiny test splits. These have been corrected.

## Optional Future Work (nothing blocking submission)

- Feature engineering improvements (more behavioral signals, market-specific features)
- Larger data collection window for more statistical power
- Alternative label definitions (Sharpe-based, drawdown-adjusted)
- Docker containerization for deployment
- Write final paper sections using the verified pipeline

---

## Key Files Modified This Session

| File | Change |
|---|---|
| `src/data_ingestion/public_scraper.py` | Rewrote `_fetch_single_resolution` → CLOB primary with 429 retry; added `CLOB_API` constant |
| `src/data_quality/dataset_completeness.py` | Added `_resolution_coverage_analysis()`, 4 new gates, markdown section |
| `tests/test_phase25.py` | NEW — 7 provenance integrity tests |
| `tests/test_phase26.py` | NEW — 4 unit + 6 integration + 5 hard gate tests |
| `scripts/rebuild_resolutions.py` | NEW — Batch CLOB rebuild with `--resume` support |
| `data/processed/market_resolutions.csv` | Rebuilt — 96,730 rows with CLOB-based resolution |
| `data/features/model_input.csv` | Rebuilt — 15,799 wallets (future_return mode) |
| `data/features/model_input_resolution.csv` | Snapshot — 35,885 wallets (resolution mode) |
| `scripts/reproduce.sh` | NEW — single-command full verification |
| `docs/reproducibility.md` | NEW — verification guide + artifact index |
| `README.md` | Updated dataset stats, resolution methodology, pipeline summary |
| `src/models/trainer.py` | Added data snapshot hash, label strategy, training date to metadata |
| `src/visualization/api.py` | Added `/api/collection-health` endpoint |
| `models/artifacts/best_model.pkl` | Retrained — XGBoost (AUC=0.593) on future-return labels |
| `models/artifacts/model_metadata.json` | Versioned with data_snapshot_hash + label_strategy |
| `AI_HANDOFF.md` | Updated with Phase 29 completion |

---

## Technical Discoveries (for future agents)

### Polymarket API Behavior

| Endpoint | Behavior |
|---|---|
| `gamma-api.polymarket.com/markets?conditionId=X` | **BROKEN** — ignores conditionId, returns paginated global list |
| `gamma-api.polymarket.com/markets?slug=X` | Works for Gamma-native slugs, NOT for data-api slugs |
| `gamma-api.polymarket.com/markets?closed=true` | Returns only closed markets (conditionId still ignored) |
| `clob.polymarket.com/markets/{conditionId}` | **CORRECT** — returns specific market, has `tokens[].winner` boolean |
| `data-api.polymarket.com/trades` | Trade feed; slugs here do NOT exist in Gamma API |

**Key insight**: conditionId from data-api trades maps correctly to CLOB API path parameter, but does NOT work as Gamma API query filter.

### CLOB API Rate Limiting

- Returns HTTP 429 when overloaded (~10+ concurrent requests)
- Safe concurrency: 5 workers with no delay
- Timeout: occasionally returns read timeout after extended use
- Retry strategy: exponential backoff (1.5s × attempt), max 3 retries
- If 429 persists: reduce workers to 3 and add 0.5s inter-request delay

### Resolution Method Reliability

| Method | Reliability | Action |
|---|---|---|
| `clob_winner_field` | HIGH | Trust — explicit winner boolean |
| `clob_no_winner` | HIGH | Market open or not yet resolved |
| `gamma_outcome_prices_argmax` | INVALID | Retry via CLOB; Gamma ignores conditionId |
| `gamma_ambiguous_prices` | INVALID | Cannot determine winner |
| `legacy_resolution_value` | INVALID | Was always "Yes" — must not appear in final data |
| `fetch_failed` | Transient | Retry with `--resume` — usually 429 rate limit |

---

## Guardrails For Future Agents

- Do NOT use Gamma API conditionId queries for resolution — always use CLOB
- Never reintroduce composite labels as a training target
- Keep active label as independent future-return based
- Preserve `outcome` through DB, cleaned transactions, and feature-building
- Do not use `git reset --hard`, `git checkout --`, or broad cleanup commands
- Phase 27 harvest quantity should be based on labeled wallet count, not raw trade count
- Update this file after every substantial phase completion

---

## Commands To Re-establish Trust

```bash
cd "/Users/mac/project 2/poly/polymarket_alpha_research"
source venv/bin/activate
python -m src.data_quality.audit
python -m src.data_quality.dataset_completeness
python -m src.audit.phase0
python -m pytest tests/ -v
```

Or simply:
```bash
bash scripts/reproduce.sh
```

Last full regression:

```
60 passed, 3 warnings (sklearn precision warnings — expected)
All 10 completeness gates PASS
Phase 0 audit: all phases PASS
reproduce.sh: ALL CHECKS PASSED
```
