# Market Pulse Lab
### A Financial Data Preprocessing Portfolio

> Preprocessing mastery across tabular, time series, and NLP data —
> built on real financial market data.

---

## What This Project Demonstrates

This is not a modeling project. It is a **preprocessing engineering project**.
The goal is to show mastery of data preparation techniques across three
data types, in a domain (finance) where data quality mistakes have real consequences.

| Skill Area | Techniques Demonstrated |
|---|---|
| **Tabular** | Missing value imputation, categorical encoding, outlier detection, domain-aware feature ratios |
| **Time Series** | Lag features, rolling statistics, realized volatility, stationarity testing, temporal train/test splitting |
| **NLP** | Text cleaning pipeline, TF-IDF, FinBERT sentiment inference, SBERT embeddings, UMAP visualization |
| **Integration** | Multi-source joins, trading day alignment, feature versioning, anti-leakage enforcement |

---

## Data Sources

| Dataset | Source | Size |
|---|---|---|
| US Stock Prices (NYSE + NASDAQ) | [Kaggle — borismarjanovic](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs) | ~500MB |
| Financial News Articles (4.4M) | [Kaggle — miguelaenlle](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlp-backtests) | ~1.2GB |
| Macro Indicators (VIX, rates, CPI) | [FRED API](https://fred.stlouisfed.org/) | ~1MB |

---

## Setup

```bash
# 1. Clone
git clone <YOUR_REPO_URL>
cd market-pulse-lab

# 2. Create environment
conda env create -f environment.yml
conda activate market-pulse-lab

# 3. Download spaCy model
python -m spacy download en_core_web_sm

# 4. Add Kaggle credentials
# Place kaggle.json in ~/.kaggle/kaggle.json

# 5. Download datasets
kaggle datasets download -d borismarjanovic/price-volume-data-for-all-us-stocks-etfs -p datasets/ --unzip
kaggle datasets download -d miguelaenlle/massive-stock-news-analysis-db-for-nlp-backtests -p datasets/ --unzip

# 6. Run full pipeline
python run_pipeline.py --dev   # dev mode: 10 tickers, fast
python run_pipeline.py         # full run (takes ~4-6h for FinBERT)
```

Stage-by-stage execution:

```bash
python run_pipeline.py --stage ingest
python run_pipeline.py --stage eda
python run_pipeline.py --stage timeseries
python run_pipeline.py --stage nlp
python run_pipeline.py --stage integrate
python run_pipeline.py --stage validate --validation-level strict
```

---

## Notebooks

| Notebook | What You'll Learn |
|---|---|
| `01_diagnostic_eda.ipynb` | Profiling, missing patterns, distribution analysis |
| `02_price_timeseries_preprocessing.ipynb` | Volatility, lags, rolling features, stationarity |
| `03_news_nlp_preprocessing.ipynb` | Cleaning, TF-IDF, FinBERT, SBERT, UMAP |
| `05_integration_validation.ipynb` | Joining sources, Great Expectations validation |
| `06_modeling_baseline.ipynb` | Minimal model to validate feature quality |

---

## Key Design Decisions

**Why no random train/test split?**
Financial data has temporal dependencies. Random splitting leaks future
information into training. All splits are strictly temporal.

**Why lag all features by 1 day?**
Using same-day news or prices to predict same-day volatility creates
look-ahead bias. In production, you never have same-day data at decision time.

**Why FinBERT instead of generic BERT?**
Financial text has domain-specific language ("beat estimates", "missed guidance",
"raised outlook") that generic sentiment models misclassify. FinBERT was
fine-tuned specifically on financial communications.

---

## Baseline Modeling Results (Phase 6)

Models were trained with strict temporal splits and a train-only scaler.
To keep runtime practical in notebook execution, the train split was sampled to
250,000 rows (`modeling.max_train_rows` in `pipeline_config.yaml`).

| Model | Feature Set | Validation MAE | Test MAE |
|---|---|---:|---:|
| XGBoostRegressor | Price + Sentiment | 0.160300 | 0.159764 |
| LinearRegression | Price + Sentiment | 0.167450 | 0.175475 |
| LinearRegression | Price Only | 0.167452 | 0.175524 |

Interpretation:
- Sentiment features add a small but consistent improvement versus the linear price-only baseline.
- Non-linear modeling (XGBoost) captures substantially more signal than linear baselines.

Artifacts:
- Results table: `reports/modeling_baseline_results.csv`
- Top-20 feature importance plot: `reports/plots/modeling_top20_features.png`

---

## References

- [FinBERT paper](https://arxiv.org/abs/1908.10063) — Yang et al. (2020)
- [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) — Zheng & Casari
- [Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) — Lopez de Prado
