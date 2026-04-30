# Data Description

## Data Source

| Field | Description |
|-------|-------------|
| **Source** | Polymarket Prediction Market Platform |
| **API Endpoint** | https://data-api.polymarket.com/trades (Global Trade Feed) |
| **Resolution API** | https://gamma-api.polymarket.com/markets (Market Outcomes) |
| **Blockchain** | Polygon (MATIC) |
| **Collection Method** | REST API polling with rate limiting |
| **Authentication** | None required (public endpoints) |
| **Collection Date** | March - April 2025 |
| **Time Period Covered** | Historical trades from platform inception through April 2025 |

## Dataset Overview

| Metric | Value |
|--------|-------|
| **Total Observations** | ~700,000 transactions (full dataset) |
| **Unique Traders** | ~19,000 wallet addresses |
| **Unique Markets** | 800+ prediction markets |
| **Variables** | 8 core transaction fields + derived features |
| **File Format** | SQLite database (.db) + CSV exports |

## Variable Description

### Core Transaction Fields

| Variable Name | Type | Description | Example Value |
|--------------|------|-------------|---------------|
| `transaction_id` | String | Blockchain transaction hash | `0x7a8f9c2d3e4b5a6c...` |
| `address` | String | Trader wallet address (pseudonymous) | `0xabc123...def456` |
| `market_id` | String | Market condition ID (unique identifier) | `0x1234...abcd` |
| `side` | String | Trade direction: BUY or SELL | `BUY` |
| `amount` | Float | Trade size in USDC | `150.50` |
| `price` | Float | Price per share (0.0 - 1.0) | `0.65` |
| `timestamp` | ISO 8601 | Trade execution time (UTC) | `2024-03-15T14:30:00Z` |
| `outcome` | String | Position taken (Yes/No or specific outcome) | `Yes` |

### Market Metadata Fields

| Variable Name | Type | Description | Example Value |
|--------------|------|-------------|---------------|
| `market_title` | String | Human-readable market question | `Will Bitcoin exceed $100k by Dec 2024?` |
| `market_resolution` | String | Final outcome: winning outcome or `__OPEN__` | `Yes` |

### Derived Features (Feature Engineering Layer)

| Feature Category | Variable Name | Type | Description |
|-----------------|--------------|------|-------------|
| **Temporal** | `hour_of_day` | Integer | Trade execution hour (0-23) |
| **Temporal** | `day_of_week` | Integer | Trade day (0-6) |
| **Behavioral** | `early_entry_pct` | Float | Percentile of entry timing in market lifecycle |
| **Behavioral** | `concentration_index` | Float | HHI concentration of trader's portfolio |
| **Profit** | `total_pnl` | Float | Cumulative profit/loss per trader |
| **Profit** | `win_rate` | Float | Ratio of profitable trades |
| **Profit** | `prediction_accuracy` | Float | Correct predictions / total resolved positions |

## Collection Process

### Step 1: Transaction Data Collection
```
Endpoint: GET https://data-api.polymarket.com/trades
Parameters: limit=100, offset=N
Rate Limiting: 0.25s between requests
Pagination: Offset-based until empty response
```

### Step 2: Market Resolution Fetching
```
Endpoint: GET https://gamma-api.polymarket.com/markets
Parameters: conditionId={market_id}, limit=1
Method: Concurrent fetching with ThreadPoolExecutor (5 workers)
Resolution Logic: max(outcomePrices) determines winner
```

### Step 3: Data Validation
- Duplicate transaction_id removal
- Timestamp parsing and timezone conversion
- Wallet address normalization (lowercase)
- Missing value flagging

## Data Quality Issues

| Issue | Description | Mitigation |
|-------|-------------|------------|
| **Missing timestamps** | Some legacy transactions lack precise timestamps | Excluded from temporal analysis |
| **Open markets** | ~40% of markets unresolved at collection time | Filtered to resolved markets only for labeling |
| **Wallet clustering** | Same user may control multiple addresses | Not addressed; analysis at wallet level |
| **API rate limits** | Occasional timeout errors | Retry logic with exponential backoff |
| **Data gaps** | Rare empty pages in pagination | Early termination after 3 consecutive empty pages |

## Research Alignment

### Data-to-Hypothesis Mapping

| Hypothesis | Data Required | Coverage |
|------------|---------------|----------|
| H1: Existence of informed traders | `prediction_accuracy` calculated from `market_resolution` | All resolved markets |
| H2: ML identifiability | Behavioral features + accuracy labels | 18,000+ labeled traders |
| H3: Information content | `price` time-series around identified trades | Subsample for event study |
| H4: Strategy effectiveness | Full transaction history for backtesting | All trades 2024-2025 |
| H5: Market condition heterogeneity | `amount` (volume) quintiles | Complete coverage |

## File Structure

```
data/
├── raw/                      # Original API responses
├── processed/
│   ├── trades_collected.csv  # Cleaned transaction data
│   ├── market_resolutions.csv  # Market outcomes
│   └── features/             # ML-ready feature matrices
├── research.db               # SQLite database (full dataset, ~700MB)
└── polymarket_sample.csv     # Sample for GitHub (1000 rows)
```

## Citation

If using this dataset, cite:
```
Data collected from Polymarket (https://polymarket.com) via public API, 
March-April 2025. Research project: "Information Processing and Alpha 
Mining Strategies in Blockchain Prediction Markets".
```

## License & Access

- **Data Source**: Public blockchain data (Polygon)
- **API Terms**: Subject to Polymarket Terms of Service
- **Research Use**: Academic/non-commercial use only
- **Reproducibility**: Full collection code available in `code/data_collection.ipynb`