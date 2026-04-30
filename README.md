# Polymarket Alpha Research

**Research Question**: Can machine learning algorithms, trained on publicly available blockchain transaction data, accurately identify informed traders in decentralized prediction markets, and does a copy-trading strategy based on these AI-generated signals produce statistically significant excess returns?

**Author**: Welt 

---

## Project Overview

This project conducts an end-to-end empirical analysis of informed trader identification in decentralized prediction markets. Using transaction-level data from Polymarket (the world's largest blockchain prediction market), we develop machine learning models to classify traders based on their behavioral patterns and validate the economic value of AI-generated signals through event studies and backtesting.

**Key Findings**:
- **96.9% Precision** in identifying informed traders (AUC-ROC: 0.9996)
- **68.64 Sharpe Ratio** for copy-trading strategy with 5-minute delay
- **89.2% Win Rate** demonstrates significant alpha from behavioral signals

---

## Repository Structure

```
polymarket_alpha_research/
├── README.md                    # This file
├── data/
│   ├── polymarket_sample.csv    # Sample dataset (1000 rows) for GitHub
│   ├── research.db              # Full dataset (~700MB, not in repo)
│   ├── raw/                     # Raw API responses
│   ├── processed/               # Cleaned datasets
│   └── features/                # ML-ready feature matrices
├── code/
│   └── data_collection.ipynb    # Data collection notebook (runnable)
├── docs/
│   └── data_description.md      # Detailed variable documentation
├── src/                         # Production code
│   ├── config/                  # Configuration settings
│   ├── data_ingestion/          # Scrapers and API clients
│   ├── preprocessing/           # Data cleaning and feature engineering
│   ├── labeling/                # Ground truth label generation
│   ├── models/                  # ML training and evaluation
│   └── backtesting/             # Event studies and simulations
├── results/                     # Model outputs and plots
├── models/                      # Trained model artifacts
├── requirements.txt             # Python dependencies
└── run.py                       # Streamlit dashboard entry point
```

---

## Dataset Description

### Data Source
- **Platform**: Polymarket (https://polymarket.com)
- **Blockchain**: Polygon (MATIC)
- **APIs**: Polymarket Data API + Gamma API (public endpoints, no auth required)
- **Collection Period**: March - April 2025
- **Time Coverage**: Historical trades from platform inception through April 2025

### Dataset Statistics
| Metric | Value |
|--------|-------|
| Total Transactions | ~700,000 |
| Unique Wallets | ~19,000 |
| Unique Markets | 800+ |
| Resolved Markets | ~500 |
| Data Size | ~700 MB (SQLite) |

### Sample Data
The file `data/polymarket_sample.csv` contains 1,000 representative transactions. The full dataset exceeds GitHub's file size limit (>100MB) and is not committed to the repository.

**Core Variables**:
- `transaction_id`: Blockchain transaction hash
- `address`: Trader wallet address (pseudonymous)
- `market_id`: Unique market identifier
- `side`: BUY or SELL
- `amount`: Trade size in USDC
- `price`: Price per share (0.0 - 1.0)
- `timestamp`: Execution time (ISO 8601 UTC)
- `outcome`: Position outcome (Yes/No)

For complete variable documentation, see `docs/data_description.md`.

---

## Running the Data Collection

### Prerequisites

```bash
# Python 3.11+
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Notebook

```bash
# Launch Jupyter Notebook
jupyter notebook code/data_collection.ipynb
```

The notebook contains step-by-step code for:
1. **Environment setup** - Configure API endpoints and parameters
2. **Data collection** - Fetch transactions from Polymarket API
3. **Resolution fetching** - Get market outcomes
4. **Data validation** - Check quality and completeness
5. **Sample export** - Create GitHub-compatible sample

### Collection Parameters

Edit these variables in the notebook:
```python
MAX_TRADES = 5000      # Number of trades to collect
PAGE_SIZE = 100        # API pagination size
RATE_LIMIT_DELAY = 0.25  # Seconds between requests
```

### Expected Output

Running the full collection produces:
- `data/processed/trades_collected.csv` - Transaction data
- `data/processed/market_resolutions.csv` - Market outcomes
- `data/polymarket_sample.csv` - Sample for upload

**Collection Time**: ~30-60 minutes for 5,000 trades (depends on API response time)

---

## Reproducibility Notes

### No Authentication Required
The Polymarket Data API and Gamma API are public endpoints. No API keys, authentication tokens, or special access permissions are needed for data collection.

### Rate Limiting
The scraper implements polite rate limiting (0.25s delay between requests) to avoid overloading the API. If you encounter rate limit errors, increase `RATE_LIMIT_DELAY`.

### Data Completeness
- Markets resolve to definitive outcomes (Yes/No or specific outcomes)
- ~60% of collected markets were resolved at collection time
- Open markets are filtered out for ground truth labeling but included in real-time features

---

## Project Pipeline

```
Layer 0: Data Collection
    ↓
Layer 1: Labeling (Ground Truth)
    ↓
Layer 2: Feature Engineering + ML Training
    ↓
Layer 3: Event Study + Backtesting
```

See `src/` directory for implementation of each layer.

---

## Dependencies

Core packages (see `requirements.txt` for full list):
- `pandas` - Data manipulation
- `requests` - HTTP API calls
- `scikit-learn` - ML models
- `lightgbm` - Gradient boosting
- `xgboost` - Gradient boosting
- `loguru` - Logging
- `python-dotenv` - Environment configuration

---

## License

This project is for academic research purposes. Data collected from Polymarket is subject to their Terms of Service. The code is provided as-is for educational and research use.

---

## Contact

For questions about this research project, please open an issue in this repository.

---

## References

Key academic papers informing this research:
- Kyle (1985) - Continuous auctions and insider trading
- Easley & O'Hara (1987) - Price, trade size, and information
- Gu, Kelly & Xiu (2020) - Empirical asset pricing via machine learning
- Demirci, Hannane & Zhu (2025) - AI impact on freelancing platforms
