# Polymarket Alpha Research

> **AI handoff note**: if you are taking over this project in a new agent or IDE
> session, read `AI_HANDOFF.md` first. It records the current Phase 0-24 status,
> validation commands, known dataset gaps, and recommended next workflow.

**Research Question**: Can machine learning algorithms, trained on publicly available blockchain transaction data, accurately identify informed traders in decentralized prediction markets, and does a copy-trading strategy based on these AI-generated signals produce statistically significant excess returns?

**Author**: Welt 

---

## Project Overview

This project conducts an end-to-end empirical analysis of informed trader identification in decentralized prediction markets. Using transaction-level data from Polymarket (the world's largest blockchain prediction market), we develop machine learning models to classify traders based on their behavioral patterns and validate the economic value of AI-generated signals through event studies and backtesting.

**Key Findings**:
- **XGBoost AUC-ROC: 0.593** — modest but statistically above random (0.500) for future-return prediction
- **Top-10% selection achieves 1.24x precision lift** over base rate, confirming a real ranking signal
- **60 automated tests, 10 data quality gates** ensure full reproducibility
- **Contribution is methodological**: rigorous pipeline design, CLOB-verified resolutions, temporal walk-forward validation, and honest reporting of results

See [`docs/model_interpretation.md`](docs/model_interpretation.md) for detailed analysis of model performance and threshold strategies.

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
- **Collection Period**: March – May 2026
- **Time Coverage**: March 18, 2026 through May 15, 2026 (approximately 8 weeks)

### Dataset Statistics
| Metric | Value |
|--------|-------|
| Total Transactions | 1,706,505 |
| Unique Wallets | 203,300 |
| Unique Markets | 96,884 |
| Resolved (Closed) Markets | 89,597 |
| Resolution Coverage | 99.8% (CLOB API verified) |
| Data Size | ~834 MB (SQLite) |
| Collection Period | March 18 – May 15, 2026 |

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

## Reproducibility

A single command verifies all audit gates, regenerates reports, and runs the full test suite:

```bash
bash scripts/reproduce.sh
```

For details on the verification pipeline, evidence artifacts, and phase-by-phase audit trail, see [`docs/reproducibility.md`](docs/reproducibility.md).

### No Authentication Required
The Polymarket Data API and CLOB API are public endpoints. No API keys, authentication tokens, or special access permissions are needed for data collection.

### Resolution Methodology
Market outcomes are determined exclusively via the CLOB API (`clob.polymarket.com/markets/{conditionId}`), which provides an explicit `winner` boolean per token. The Gamma API conditionId filter is known to be unreliable (see `AI_HANDOFF.md` for details).

### Data Completeness
- 99.8% of markets successfully resolved via CLOB API
- 89,597 closed markets with verified outcomes
- Diverse outcome types: Yes, No, Up, Down, team names, Over/Under, etc.
- Open markets (7,495) tracked but excluded from resolution-dependent labels

---

## Project Pipeline

```
Layer 0: Data Collection (Polymarket Data API + CLOB API)
    ↓
Layer 1: Resolution & Labeling (CLOB-verified outcomes → future-return labels)
    ↓
Layer 2: Feature Engineering + ML Training (temporal walk-forward split)
    ↓
Layer 3: Event Study + Backtesting (random baselines, bootstrap CI)
```

### ML Pipeline Summary
| Stage | Detail |
|-------|--------|
| Primary label | Future-return (independent, forward-looking ROI) |
| Robustness check | Resolution-based accuracy label |
| Split method | Temporal walk-forward (70/15/15) |
| Train wallets | 13,654 |
| Val wallets | 1,119 |
| Test wallets | 1,026 |
| Positive rate | ~20% (balanced) |

See `src/` directory for implementation and `docs/` for detailed documentation.

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
