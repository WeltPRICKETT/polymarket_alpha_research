# Data Description

## Data Source

| Field | Description |
|-------|-------------|
| **Source** | Polymarket Prediction Market Platform |
| **Trade API** | `https://data-api.polymarket.com/trades` (Global Trade Feed) |
| **Resolution API** | `https://clob.polymarket.com/markets/{conditionId}` (CLOB Market Outcomes) |
| **Blockchain** | Polygon (MATIC) |
| **Collection Method** | REST API polling with rate limiting |
| **Authentication** | None required (public endpoints) |
| **Collection Period** | March 18 – May 15, 2026 (approximately 8 weeks) |

## Dataset Overview

| Metric | Value |
|--------|-------|
| **Total Observations** | 1,706,505 transactions |
| **Unique Traders** | 203,300 wallet addresses |
| **Unique Markets** | 96,884 prediction markets |
| **Resolved (Closed) Markets** | 89,597 (99.8% resolution coverage) |
| **Core Variables** | 8 transaction fields |
| **File Format** | SQLite database (~834 MB, not in repo) + 1,000-row CSV sample |

## Variable Description

### Core Transaction Fields

| Variable Name | Data Type | Description | Example Value |
|--------------|-----------|-------------|---------------|
| `transaction_id` | String | Unique blockchain transaction hash | `0x7a8f9c2d3e4b5a6c...` |
| `address` | String | Trader wallet address (pseudonymous) | `0xabc123...def456` |
| `market_id` | String | Market condition ID (links to CLOB API) | `0x1234...abcd` |
| `side` | String | Trade direction | `BUY` or `SELL` |
| `amount` | Float | Trade size in USDC | `150.50` |
| `price` | Float | Price per share (probability, range 0–1) | `0.65` |
| `timestamp` | String (ISO 8601) | Trade execution time (UTC) | `2026-03-18T00:57:21Z` |
| `outcome` | String | Position outcome token name | `Yes`, `No`, `Up`, `Down` |

### Market Resolution Fields (in `market_resolutions.csv`)

| Variable Name | Data Type | Description | Example Value |
|--------------|-----------|-------------|---------------|
| `market_id` | String | Market condition ID | `0x1234...abcd` |
| `question` | String | Human-readable market question | `Will BTC exceed $100k?` |
| `resolution` | String | Final outcome or `__OPEN__` | `No` |
| `closed` | Boolean | Whether market has resolved | `True` |
| `winning_outcome` | String | Name of the winning token | `No` |
| `resolution_method` | String | How resolution was determined | `clob_winner_field` |

### Derived Features (in `model_input.csv`)

| Variable Name | Data Type | Description |
|--------------|-----------|-------------|
| `total_roi` | Float | Cumulative return on investment per trader |
| `win_rate` | Float | Ratio of profitable trades |
| `early_entry_score` | Float | How early the trader enters markets (0–1) |
| `contrarian_score` | Float | Tendency to trade against consensus |
| `information_ratio` | Float | Risk-adjusted return metric |
| `cross_market_diversification` | Float | Portfolio diversity across markets |
| `avg_holding_period` | Float | Mean duration of positions |
| `trading_frequency` | Float | Number of trades per unit time |

## Descriptive Statistics (Sample: 1,000 transactions)

| Statistic | `amount` (USDC) | `price` (probability) |
|-----------|-----------------|----------------------|
| Count | 1,000 | 1,000 |
| Mean | 100.54 | 0.585 |
| Std | 938.93 | 0.304 |
| Min | 0.02 | 0.001 |
| 25th percentile | 4.92 | 0.370 |
| Median | 9.82 | 0.580 |
| 75th percentile | 30.56 | 0.870 |
| Max | 20,210.33 | 0.999 |

### Categorical Distributions (Sample)

**Side**: BUY (approximately 50%) / SELL (approximately 50%)

**Outcome**: Up (31.8%), Down (31.1%), No (14.4%), Yes (10.2%), Under/Over/team names (12.5%)

## Collection Process

### Step 1: Transaction Data Collection
```
Endpoint: GET https://data-api.polymarket.com/trades
Parameters: limit=100, offset=N
Rate Limiting: 0.25s between requests
Pagination: Offset-based until empty response
Provenance: Each API response archived with SHA-256 hash
```

### Step 2: Market Resolution Fetching
```
Endpoint: GET https://clob.polymarket.com/markets/{conditionId}
Method: Concurrent fetching with ThreadPoolExecutor (5 workers)
Resolution: tokens[].winner boolean field determines outcome
Retry: Exponential backoff on HTTP 429 (1.5s x attempt, max 3)
```

**Important note**: The Gamma API endpoint (`gamma-api.polymarket.com/markets?conditionId=X`) was found to ignore the `conditionId` parameter entirely, returning an unrelated paginated global list. This caused all markets to resolve incorrectly as "Yes" in earlier iterations. The CLOB API is the only reliable resolution source.

### Step 3: Data Validation
- Duplicate `transaction_id` removal
- Timestamp parsing and timezone normalization (UTC)
- Wallet address normalization (lowercase)
- Resolution verification: 10 automated data quality gates

## Data Quality Issues

| Issue | Description | Mitigation |
|-------|-------------|------------|
| **Gamma API resolution bug** | `conditionId` filter non-functional; returned wrong market data | Switched to CLOB API with explicit `winner` boolean |
| **CLOB API rate limiting** | HTTP 429 at high concurrency (>10 workers) | Reduced to 5 workers + exponential backoff retry |
| **Open markets** | 7,495 markets (7.7%) still unresolved | Tracked as `clob_no_winner`; excluded from resolution labels |
| **Fetch failures** | 50 markets (0.05%) failed CLOB lookup | Negligible; below 1% threshold |
| **Missing outcome** | 3 transactions with null outcome | Excluded from analysis |
| **Wallet clustering** | Same user may control multiple addresses | Not addressed; analysis at wallet level |

## Research Alignment

### Data-to-Hypothesis Mapping

| Hypothesis | Data Required | Coverage |
|------------|---------------|----------|
| H1: Informed traders exist in prediction markets | Future ROI distribution across traders | 15,799 eligible traders labeled |
| H2: ML can identify informed traders from behavioral features | 7 behavioral features + future-return labels | Train=13,654, Val=1,119, Test=1,026 |
| H3: Identified traders provide ranking signal | Top-k precision vs base rate | 1.24x lift at top-10% |

## File Structure

```
data/
  raw/                          # Archived API responses (SHA-256 verified)
  processed/
    market_resolutions.csv      # 96,730 market outcomes (CLOB-verified)
    cleaned_transactions.csv    # 1,303,764 cleaned trades
  features/
    model_input.csv             # 15,799 labeled traders with features
  research.db                   # SQLite database (834 MB, not in repo)
  polymarket_sample.csv         # 1,000-row sample for GitHub
```

## Citation

```
Data collected from Polymarket (https://polymarket.com) via public APIs,
March-May 2026. Research project: "Identifying Informed Traders in
Decentralized Prediction Markets Using Machine Learning".
Author: Welt (Zicheng Peng), NYU Shanghai.
```

## License & Access

- **Data Source**: Public blockchain data (Polygon network)
- **API Terms**: Subject to Polymarket Terms of Service
- **Research Use**: Academic/non-commercial use only
- **Reproducibility**: Full collection code in `code/data_collection.ipynb`; verification via `bash scripts/reproduce.sh`
