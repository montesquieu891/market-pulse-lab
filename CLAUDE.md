# CLAUDE.md — Market Pulse Lab
> Context file for AI agents (GitHub Copilot, Claude, Cursor).
> Read this entire file before writing any code or suggesting any changes.

---

## Overview

**Market Pulse Lab** is a preprocessing-focused data science portfolio project.
The goal is NOT to build the best predictive model — the goal is to demonstrate
mastery of data preprocessing techniques across three data types:
- Tabular (stock fundamentals, market features)
- Time Series (daily price/volume data)
- NLP (financial news articles)

The final output is a unified feature matrix combining signals from all three
sources, production-ready and serializable, designed to predict short-term
stock volatility (next 5-day realized volatility).

**Target audience:** Potential employers in fintech, asset management, banking,
or any data-intensive finance role.

**Portfolio URL:** (add GitHub URL after first push)

---

## Business Context

### The Problem
Financial analysts and quant researchers combine price data with news sentiment
to assess short-term risk. Building such a pipeline requires:
1. Integrating multiple heterogeneous data sources
2. Handling missing data with financial domain knowledge
3. Engineering features that capture market microstructure and sentiment dynamics
4. Ensuring strict temporal integrity (no future leakage)

### The Question We're Answering
> "Can we predict whether a stock will be more or less volatile in the next
> 5 trading days, using price history and news sentiment as inputs?"

This is NOT a trading strategy. It is a volatility forecasting preprocessing
pipeline — a building block for risk management systems.

---

## Tech Stack

### Core
| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Main language |
| pandas | 2.x | Tabular manipulation |
| polars | latest | Large file processing (prices dataset) |
| numpy | latest | Numerical operations |
| scikit-learn | 1.4+ | Preprocessing pipelines, transformers |
| pyarrow | latest | Parquet I/O |
| duckdb | latest | SQL queries over raw files without loading |

### Time Series
| Tool | Purpose |
|------|---------|
| statsmodels | Decomposition, stationarity tests (ADF) |
| pandas (resample, rolling) | Resampling and rolling features |

### NLP
| Tool | Purpose |
|------|---------|
| spacy (en_core_web_sm) | Tokenization, NER, lemmatization |
| transformers (HuggingFace) | FinBERT sentiment embeddings |
| sentence-transformers | SBERT document embeddings |
| nltk | Stopwords, basic tokenization |
| scikit-learn TfidfVectorizer | Baseline TF-IDF vectors |

### Visualization & Reporting
| Tool | Purpose |
|------|---------|
| matplotlib / seaborn | EDA plots |
| plotly | Interactive time series charts |
| ydata-profiling | Automated EDA reports |

### Validation & Quality
| Tool | Purpose |
|------|---------|
| great_expectations | Data contract validation |
| pydantic | Schema validation for configs |

### Dev Environment
| Tool | Purpose |
|------|---------|
| VS Code + GitHub Copilot | Primary IDE |
| conda | Environment management |
| jupyterlab | Notebooks |
| black + isort | Code formatting |
| pytest | Unit tests |
| git + GitHub | Version control |

---

## Dataset Requirements

### Dataset A — Stock Prices
**Source:** Kaggle — `borismarjanovic/price-volume-data-for-all-us-stocks-etfs`
**Format:** One CSV per ticker (e.g., `aapl.us.txt`)
**Size:** ~500MB compressed
**Columns:** Date, Open, High, Low, Close, Volume, OpenInt
**Key constraint:** Date column is YYYY-MM-DD string, must be parsed on load
**Scope for this project:** S&P 500 tickers only, 2015–2023

**After ingestion, store as:**
`datasets/Data/Stocks/` — raw ticker files from Kaggle (current location)
`datasets/interim/prices_combined.parquet` — all tickers merged, typed, cleaned

### Dataset B — Financial News
**Source:** Kaggle — `miguelaenlle/massive-stock-news-analysis-db-for-nlp-backtests`
**Format:** CSV (~1.2GB)
**Columns:** date, ticker, title, article (full text)
**Key constraint:** ~30% of articles have null `article`, only `title` available
**Scope:** Same tickers and date range as Dataset A

**After ingestion, store as:**
`datasets/raw_analyst_ratings.csv` — current raw source file in this repo
`datasets/interim/news_cleaned.parquet` — after text cleaning pipeline

### Dataset C — Macro Indicators (external enrichment)
**Source:** FRED API (free, no auth required for basic endpoints)
**Indicators to fetch:**
- `VIX` — CBOE Volatility Index (market fear gauge)
- `DGS10` — 10-Year Treasury Rate
- `UNRATE` — US Unemployment Rate
- `CPIAUCSL` — CPI Inflation

**After ingestion, store as:**
`datasets/external/macro_indicators.parquet`

---

## Project Layout

```
preProcessing/
│
├── CLAUDE.md                  ← YOU ARE HERE. Read before touching anything.
├── README.md                  ← Public-facing documentation
├── environment.yml            ← Conda environment spec
├── pipeline_config.yaml       ← All parameters (dates, tickers, thresholds)
├── run_pipeline.py            ← Master script: raw data → feature matrix
├── IMPLEMENTATION_CHECKLIST.md
└── datasets/
  ├── Data/
  │   ├── Stocks/            ← Raw stock files (`*.us.txt`)
  │   └── ETFs/              ← Raw ETF files (`*.us.txt`)
  ├── raw_analyst_ratings.csv
  ├── raw_partner_headlines.csv
  └── analyst_ratings_processed.csv
```

Notes:
- This tree reflects the current repository state and is the authoritative source for paths.
- `src/`, `tests/`, `notebooks/`, and `data/` are target scaffold directories and may not exist yet.
- Until scaffolding is created, use paths under `datasets/` and `pipeline_config.yaml` at repo root.

---

## Domain Concepts

### Volatility
Volatility is the degree of price variation of a stock over time. We use
**realized volatility**: the standard deviation of log returns over a rolling
window. High volatility = high risk/opportunity. It is the TARGET variable.

```python
# Log return for day t
log_return = np.log(close_t / close_t_minus_1)

# 5-day realized volatility (annualized)
realized_vol_5d = log_returns.rolling(5).std() * np.sqrt(252)
```

### Sentiment Signal
News sentiment is a leading indicator of price movement and volatility.
We use **FinBERT** (a BERT model fine-tuned on financial text) to classify
each article/title as positive, negative, or neutral, then aggregate
sentiment scores per ticker per day.

### Feature Lag
All features derived from prices or news must be **lagged by at least 1 day**
before being used to predict future volatility. Using same-day features
creates look-ahead bias (data leakage), which is the most critical error
in financial ML.

### Trading Days
Stock markets are closed on weekends and holidays. The dataset only contains
trading days. When joining news (which includes weekends) to prices, weekend
news must be forward-filled to the next trading day.

### Ticker Universe
We work with a fixed universe of tickers (S&P 500 as of 2020) to avoid
survivorship bias issues. Tickers that were delisted during the period are
kept in the dataset.

---

## Key Business Rules

These rules are non-negotiable. Every pipeline function (under `src/` once scaffold exists) must respect them.

**RULE 1 — No Look-Ahead Bias (THE most important rule)**
Features used to predict volatility at time T must only use information
available at time T-1 or earlier. This means:
- All price features: minimum 1-day lag
- News features: same-day news is allowed only if published before market open
  (we cannot verify this, so we lag all news features by 1 day to be safe)
- Macro indicators: lag by 1 month (monthly data released with delay)

**RULE 2 — Train/Test Split is Temporal**
NEVER use random shuffling for train/test split.
Split date: 2021-01-01
- Train: 2015-01-01 to 2020-12-31
- Validation: 2021-01-01 to 2021-12-31
- Test: 2022-01-01 to 2023-12-31

**RULE 3 — Fit on Train, Transform on All**
Any scaler, imputer, or encoder must be fit ONLY on the training set.
Then applied to validation and test. This is enforced by always using
sklearn Pipeline objects.

**RULE 4 — Raw Data is Immutable**
Files in `datasets/Data/` and raw CSVs under `datasets/` are never modified, overwritten, or deleted by code.
All transformations produce new files in `datasets/interim/`, `datasets/features/`, or `datasets/external/`.

**RULE 5 — All Dates are UTC**
Convert all datetime columns to UTC on load. No naive datetimes anywhere.

**RULE 6 — Parquet is the Working Format**
After initial ingestion, all intermediate and final datasets are Parquet.
Never write intermediate CSVs.

---

## Entities

### Ticker
A stock symbol (e.g., "AAPL", "MSFT"). Primary key for joining all datasets.
Stored as uppercase string. Always validated against the ticker universe file.

### TradingDay
A calendar date on which markets were open. Stored as `datetime64[ns, UTC]`.
Reference file (when created): `datasets/reference/trading_calendar.csv`

### PriceRecord
One row in the prices dataset. Granularity: ticker × trading_day.
Fields: open, high, low, close, volume, log_return, realized_vol_5d, realized_vol_20d

### NewsRecord
One article or headline associated with a ticker and date.
Fields: ticker, date, title, article (nullable), source, sentiment_score,
sentiment_label, embedding_384d (SBERT vector)

### FeatureRecord
One row in the final feature matrix. Granularity: ticker × trading_day.
All features are lagged. Target variable: realized_vol_5d_forward
(5-day realized volatility starting from the NEXT trading day).

---

## Commands

### Environment Setup (run once)
```bash
conda env create -f environment.yml
conda activate market-pulse-lab
python -m spacy download en_core_web_sm
```

### Run Full Pipeline (raw → feature matrix)
```bash
python run_pipeline.py
```

### Run Individual Stages (once `src/` scaffold exists)
```bash
python -m src.ingestion.price_loader        # Load prices → parquet
python -m src.ingestion.news_loader         # Load news → parquet
python -m src.ingestion.fred_fetcher        # Fetch macro from FRED
python -m src.timeseries.features           # Build price features
python -m src.nlp.cleaner                   # Clean news text
python -m src.nlp.sentiment                 # Run FinBERT inference
python -m src.integration.joiner            # Join all sources
```

### Run Tests
```bash
pytest -v
pytest -v --cov=. --cov-report=html
```

### Format Code
```bash
black .
isort .
```

### Launch Notebooks
```bash
jupyter lab
```

### Validate Data
```bash
python run_pipeline.py --stage validate
python run_pipeline.py --stage validate --validation-level basic
python run_pipeline.py --stage validate --validation-level strict
```

---

## Gotchas

### Price Data
- Some ticker files use `,` as separator, some use `\t`. Use `sep=None, engine='python'` on read or DuckDB's auto-detect
- `OpenInt` column is always 0 for stocks (only relevant for futures/options). Drop it
- Prices are split-adjusted but NOT dividend-adjusted. Returns calculated from Close are total price returns, not total returns
- Some tickers have duplicate dates (data errors). Keep the last occurrence
- Volume of 0 on a trading day = data error or halt. Flag these rows

### News Data
- ~30% of rows have null `article`. Always fall back to `title` for NLP features. Never drop these rows
- Same article appears multiple times with different tickers (wire service stories). This is expected, not a bug
- Date column mixes formats: `YYYY-MM-DD` and `MM/DD/YYYY`. Normalize on load with `pd.to_datetime(..., infer_datetime_format=True)`
- Ticker column has noise: lowercase, with/without exchange suffix ("AAPL", "aapl", "AAPL.O"). Normalize to uppercase, strip suffixes

### FinBERT
- Model: `ProsusAI/finbert` from HuggingFace
- Input max length: 512 tokens. Financial articles often exceed this. Truncate to first 512 tokens (the lede contains the most information)
- Batch inference is 10x faster than row-by-row. Use `batch_size=32` minimum
- First run downloads ~440MB model. Cache in `models/finbert_cache/`
- GPU is NOT required. CPU inference on 4M articles takes ~4 hours. Run overnight or use Kaggle

### Joins
- Price data has no weekend rows. News data does. When joining, use `merge_asof` with direction='forward' to align weekend news to the next trading day
- After joining, ~15% of ticker×date combinations will have no news. This is expected. Fill sentiment features with 0 (neutral), NOT with mean imputation

### Memory
- Loading all ticker CSVs into one DataFrame at once will likely OOM on 8GB RAM machines
- Use DuckDB or chunked loading with polars for the initial merge
- The news dataset (~1.2GB CSV) fits in 8GB RAM but leaves little headroom. Convert to Parquet first, then work from Parquet

### Temporal Leakage (repeat because it's the most dangerous mistake)
- NEVER fit any transformer (scaler, imputer, encoder) on the full dataset
- NEVER compute rolling statistics using future data
- NEVER sort by a non-temporal column before splitting
- When in doubt, add a temporal assertion: `assert train_df['date'].max() < test_df['date'].min()`

---

## Constraints

- **Python 3.10+** required. No support for older versions (match statements used in utils)
- **No GPU required** but FinBERT inference is slow on CPU. Plan accordingly
- **Internet required** only for first run (Kaggle download, FRED API, HuggingFace model download). All subsequent runs are fully offline
- **Minimum 16GB RAM recommended** for full pipeline. 8GB works with chunked loading
- **Disk space:** ~5GB for raw data, ~2GB for processed data, ~500MB for models
- **No API keys required** except Kaggle credentials (free). FRED API works without key for basic indicators
- **All code must be type-hinted.** No functions without type annotations
- **All public functions must have docstrings** (Google style)
- **No Jupyter-only code.** All logic should live in `src/` once scaffold exists; notebooks should only call `src/` functions

---

## References

### Papers
- "FinBERT: A Pre-trained Financial Language Representation Model" — Yang et al. (2020)
- "Attention Is All You Need" — Vaswani et al. (2017) — BERT foundation
- "Volatility Forecasting with Machine Learning" — Bucci (2020)

### Books (see conversation context for full reading list)
- Designing Data-Intensive Applications — Kleppmann (Chapter 3, 4)
- Feature Engineering for Machine Learning — Zheng & Casari
- Python Feature Engineering Cookbook — Soledad Galli

### Datasets
- Prices: https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs
- News: https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlp-backtests
- Macro: https://fred.stlouisfed.org/

### Key Libraries Docs
- FinBERT: https://huggingface.co/ProsusAI/finbert
- SBERT: https://www.sbert.net/
- Great Expectations: https://docs.greatexpectations.io/
- pandas time series: https://pandas.pydata.org/docs/user_guide/timeseries.html

### Inspired By
- Advances in Financial Machine Learning — Marcos Lopez de Prado
- https://github.com/stefan-jansen/machine-learning-for-trading
