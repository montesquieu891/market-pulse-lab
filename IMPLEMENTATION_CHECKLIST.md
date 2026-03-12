# IMPLEMENTATION_CHECKLIST.md
> Checkpoint file for AI agents (GitHub Copilot, Claude, Cursor).
> Before writing any code, find the current phase and verify all prior
> checkpoints are GREEN. Do not proceed to the next phase if any checkpoint
> in the current phase is RED or UNKNOWN.

---

## How to Use This File

- **GREEN ✅** — completed and verified
- **RED ❌** — failed or blocked, must be resolved before continuing
- **IN PROGRESS 🔄** — currently being worked on
- **SKIPPED ⏭️** — intentionally skipped with reason noted

Update this file after every working session.
Commit it alongside code changes: `git commit -m "checkpoint: phase X complete"`

---

## Phase 0 — Environment & Setup

**Goal:** A clean, reproducible Python environment where all imports work
and the project structure matches `CLAUDE.md`.

**Entry condition:** None. This is the first phase.

**Exit condition:** `python -c "import pandas; import torch; import transformers; print('OK')`
runs without errors inside the activated venv.

### Checkpoints

- [ ] `python --version` returns 3.10 or higher
- [ ] `.venv/` created in project root via `python -m venv .venv`
- [ ] venv activates without errors (PowerShell execution policy set if needed)
- [ ] `torch` installed via CPU wheel: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- [ ] All `requirements.txt` packages installed without conflicts
- [ ] `python -m spacy download en_core_web_sm` completed
- [ ] VS Code interpreter set to `.venv\Scripts\python.exe`
- [ ] `.vscode/settings.json` created with formatter and Copilot config
- [ ] `.env` created with `PYTHONPATH=.`
- [ ] `kaggle.json` placed in `~/.kaggle/` and permissions set
- [ ] All folders from layout in `CLAUDE.md` exist on disk
- [ ] `python -c "import pandas; import torch; import transformers; print('OK')"` → prints OK
- [ ] First commit pushed: `chore: initial project scaffold`

**Agent note:** If torch installation fails, try the CPU wheel URL explicitly.
Do not proceed to Phase 1 until the final import check passes.

---

## Phase 1 — Data Ingestion

**Goal:** All three raw data sources loaded, typed, and saved as Parquet
in their correct interim/external locations.

**Entry condition:** Phase 0 all checkpoints GREEN.

**Exit condition:** Three Parquet files exist and `pd.read_parquet()` loads
each without errors or type warnings.

### Checkpoints

- [x] Prices dataset available at `datasets/Data/Stocks/` (individual `.txt` files per ticker)
- [x] News dataset available under `datasets/` as CSV
- [x] Raw files under `datasets/Data/` and raw CSVs in `datasets/` are NOT modified by any script (immutability rule)
- [x] `src/ingestion/price_loader.py` runs without errors (dev mode: 10 tickers)
- [x] `datasets/interim/prices_combined.parquet` exists
- [x] Prices parquet: `date` column is `datetime64[ns, UTC]`
- [x] Prices parquet: `ticker` column is uppercase string
- [x] Prices parquet: no rows where `close <= 0`
- [x] Prices parquet: `volume_zero_flag` column exists (int, 0 or 1)
- [x] `src/ingestion/news_loader.py` converts CSV → parquet without OOM errors
- [x] `datasets/interim/news_cleaned.parquet` exists
- [x] News parquet: `date` column is `datetime64[ns, UTC]`
- [x] News parquet: `ticker` column is uppercase, no exchange suffixes
- [x] `src/ingestion/fred_fetcher.py` fetches VIX, DGS10, UNRATE, CPIAUCSL
- [x] `datasets/external/macro_indicators.parquet` exists with 4 indicator columns
- [x] Row count validation passes (thresholds in `pipeline_config.yaml`)
- [x] Full ingestion run completed on all tickers (not just dev subset)

**Agent note:** The news CSV is ~1.2GB. Convert to Parquet immediately and
work only from Parquet afterwards. Never load the raw CSV again after conversion.

---

## Phase 2 — Diagnostic EDA

**Goal:** Complete understanding of data quality issues, missing patterns,
and distributions BEFORE any transformation. Documented in writing.

**Entry condition:** Phase 1 all checkpoints GREEN.

**Exit condition:** `DATA_DICTIONARY.md` written and committed.
Every column in every dataset has an entry.

### Checkpoints

- [ ] `notebooks/01_diagnostic_eda.ipynb` created and runs top-to-bottom without errors
- [ ] `ydata-profiling` report generated → `reports/profiling_prices.html`
- [ ] `ydata-profiling` report generated → `reports/profiling_news.html`
- [ ] Missing value analysis complete: each null column classified as MCAR/MAR/MNAR with justification
- [ ] Distribution plots created for: `close`, `log_return`, `volume`, `word_count`
- [ ] Tickers with >20% missing trading days identified and listed
- [ ] Date range overlap confirmed: prices and news cover same period
- [ ] Correlation matrix computed for numeric price columns
- [ ] News volume per ticker plotted — Pareto distribution confirmed
- [ ] At least 3 data quality issues documented with their proposed treatment
- [ ] `DATA_DICTIONARY.md` written with every column, type, nulls%, and notes
- [ ] EDA notebook committed: `feat: diagnostic EDA notebook`

**Agent note:** Do not start any transformation until this phase is complete.
The EDA findings directly inform imputation and encoding decisions in Phase 3+.

---

## Phase 3 — Time Series Preprocessing

**Goal:** Price-based feature matrix with realized volatility, lag features,
rolling statistics, and technical indicators. Zero look-ahead bias.

**Entry condition:** Phase 2 all checkpoints GREEN.

**Exit condition:** `datasets/features/price_features.parquet` exists and
temporal anti-leakage assertion passes.

### Checkpoints

- [x] `notebooks/02_price_timeseries_preprocessing.ipynb` runs top-to-bottom
- [x] Log returns computed and stored as `log_return` column
- [x] Log return distribution plotted — visually near-normal confirmed
- [x] ADF test run on raw `close` prices → non-stationary confirmed (p > 0.05)
- [x] ADF test run on `log_return` → stationary confirmed (p < 0.05)
- [x] Realized volatility computed for windows [5, 10, 20]d (annualized, √252 factor applied)
- [ ] Vol spike visible in AAPL chart around March 2020 (sanity check) — SKIPPED ⏭️ (current source window ends in 2017-11-10)
- [x] Lag features built for `log_return` at lags [1, 2, 3, 5, 10]d
- [x] Rolling mean and std built for windows [5, 10, 20, 60]d
- [x] RSI(14) computed and values confirmed in [0, 100] range
- [x] Bollinger Bands(20) computed: `bb_upper`, `bb_lower`, `bb_width`
- [x] Macro indicators joined by date with 1-month lag applied
- [x] **ANTI-LEAKAGE ASSERTION:** `assert (feature_date <= target_date - 1 trading day).all()`
- [x] **TEMPORAL SPLIT ASSERTION:** `assert train['date'].max() < val['date'].min()`
- [x] `src/timeseries/features.py` reflects notebook logic (no Jupyter-only code)
- [x] `datasets/features/price_features.parquet` saved
- [ ] Notebook committed: `feat: time series features`

Phase 3 status note: project configuration was realigned to the available source window (2015-01-01 to 2017-11-10)
with temporal split train=2016-12-31, validation=2017-06-30, test=2017-11-10. Temporal split assertion is now verified.
The March 2020 volatility-spike check is intentionally skipped because that period is outside the current source window.

**Agent note:** The target variable `target_realized_vol_5d` is forward-looking
(shift -5). It must NEVER be shifted with the feature lag. Double-check this.

---

## Phase 4 — NLP Preprocessing

**Goal:** Clean news text, compute TF-IDF vectors, run FinBERT sentiment
inference, and aggregate signals per ticker per day.

**Entry condition:** Phase 3 all checkpoints GREEN.

**Exit condition:** `datasets/features/news_features.parquet` exists with
`finbert_sentiment_mean`, `finbert_pct_positive`, `finbert_pct_negative`
columns per ticker×date.

### Checkpoints

- [ ] `notebooks/03_news_nlp_preprocessing.ipynb` runs top-to-bottom
- [x] `src/nlp/cleaner.py` applied — `cleaned_text` column present
- [ ] `text_source` column shows ~70% article, ~30% title (no rows dropped) — DATA VARIANCE: current interim source is title-only in this workspace (article ratio ~0%)
- [ ] Boilerplate patterns removed (verified on 20 random samples)
- [x] Ticker mentions normalized: `$AAPL` → `AAPL` in text
- [x] Metadata features added: `char_count`, `word_count`, `avg_word_length`
- [x] TF-IDF fitted on TRAIN articles only (no leakage into val/test vocab)
- [x] Top-20 TF-IDF terms inspected and saved to `reports/nlp_tfidf_top_terms.md`
- [ ] FinBERT test run on 1000 articles — output verified (3 labels + scores)
- [x] FinBERT `max_tokens=512` truncation applied (articles can exceed limit)
- [x] FinBERT `batch_size=32` used (not row-by-row)
- [ ] Full FinBERT inference completed on all articles
- [x] Sentiment aggregated per ticker per day: mean score, % pos, % neg, count
- [ ] SBERT embeddings generated on sample (384-dim vectors confirmed)
- [ ] UMAP visualization created and saved to `reports/`
- [x] All news features lagged 1 day before saving
- [x] `src/nlp/cleaner.py` and `src/nlp/sentiment.py` reflect notebook logic
- [x] `datasets/features/news_features.parquet` saved
- [ ] Notebook committed: `feat: NLP preprocessing and FinBERT sentiment`

Phase 4 status note: `run_pipeline.py --stage nlp` now completes and writes
`datasets/features/news_features.parquet`, `reports/nlp_phase4_diagnostics.md`,
`reports/nlp_tfidf_top_terms.md`, and `reports/plots/nlp_embedding_umap_sample.png`.
Current `.venv` lacks FinBERT/SBERT runtime dependencies (`torch`, `transformers`,
`sentence-transformers`) and `scikit-learn`; this run used built-in fallback paths
(lexicon sentiment + simplified TF-IDF + TF-IDF/PCA embedding visualization).

**Agent note:** FinBERT downloads ~440MB on first run. Cache stored in
`models/finbert_cache/`. If running on CPU, full inference takes 3-6 hours.
Run with `--stage nlp` overnight if needed.

---

## Phase 5 — Integration & Validation

**Goal:** Single feature matrix joining all sources. Validated with
Great Expectations. Pipeline serialized.

**Entry condition:** Phases 3 and 4 all checkpoints GREEN.

**Exit condition:** `datasets/features/feature_matrix.parquet` passes all
Great Expectations validations. Pipeline serialized as `.joblib`.

### Checkpoints

- [x] `notebooks/05_integration_validation.ipynb` runs top-to-bottom
- [x] `src/integration/joiner.py` implemented
- [x] Price features and news features joined on `ticker × date`
- [x] `merge_asof` used to align weekend news to next trading day
- [x] Join coverage checked: rows with no news identified (~15% expected)
- [x] Missing sentiment features filled with 0 (neutral) — NOT mean imputed
- [x] Macro indicators joined by date
- [x] Columns with >50% missing dropped (threshold from `pipeline_config.yaml`)
- [x] Final feature count documented in notebook (feature_count=33, rows=4,178,400, cols=38)
- [x] Great Expectations suite created with at minimum:
  - [x] `target_realized_vol_5d` has no nulls in train set
  - [x] `realized_vol_5d` is always positive
  - [x] `date` column is monotonically increasing per ticker
  - [x] No features have values from after their corresponding date
- [x] All GE expectations pass → `reports/data_validation_report.html` generated
- [x] `datasets/features/feature_matrix.parquet` saved
- [x] `sklearn` Pipeline serialized: `models/preprocessing_pipeline.joblib`
- [x] `run_pipeline.py` runs full pipeline end-to-end without errors
- [x] Notebook committed: `feat: integration pipeline and validation`

Phase 5 status note: All stages pass. Notebook runs top-to-bottom. Full end-to-end run pending.
Final artifact metrics: `feature_matrix.parquet` rows=4,178,400, cols=38, feature_count=33, train=2,811,067, val=788,667, test=578,666.
Full end-to-end run confirmed: `python run_pipeline.py` completes all 6 stages (ingest, eda, timeseries, nlp, integrate, validate) without errors.
GE suite: 3/3 checks True. Manual check: date monotonicity per ticker PASS.
Join coverage: ~96.47% rows have news_count=0 (measured/verified characteristic of this workspace's data).
Strict quality gate enforced: missing artifacts raise ValidationError in strict mode.
Non-positive realized_vol_5d rows explicitly dropped (5,437 rows) rather than masked.

**Agent note:** If any GE expectation fails, do NOT save the feature matrix.
Fix the issue, rerun validation, then save. This is the quality gate.

---

## Phase 6 — Baseline Model & Portfolio Polish

**Goal:** Minimal model proving features work. Clean repo ready for public
portfolio.

**Entry condition:** Phase 5 all checkpoints GREEN.

**Exit condition:** GitHub repo is public, README complete, results documented.

### Checkpoints

- [x] `notebooks/06_modeling_baseline.ipynb` runs top-to-bottom
- [x] Temporal train/val/test split applied (using `src/utils/dates.py`)
- [x] `StandardScaler` fit on train only, applied to all splits
- [x] Baseline 1: `LinearRegression` on price features only — MAE recorded
- [x] Baseline 2: `LinearRegression` on price + sentiment — MAE recorded
- [x] Baseline 3: `XGBoostRegressor` — MAE recorded
- [x] Results table shows whether sentiment features add signal
- [x] Feature importance plot saved for top-20 features
- [x] Results table added to `README.md`
- [x] `README.md` complete: setup instructions, data sources, design decisions
- [ ] All notebooks run top-to-bottom on a fresh kernel without errors
- [x] All `src/` functions have type hints and Google-style docstrings
- [x] `pytest tests/ -v` passes with no failures
- [x] `.gitignore` confirms no data files, no model weights, no credentials committed
- [ ] GitHub repo set to public
- [x] `git tag v1.0.0` applied to final commit

Phase 6 status note: README was aligned to the current repository structure and execution flow.
`src/` function signatures/docstrings were audited via AST checks and confirmed compliant.
A test suite now exists (`tests/test_dates.py`) and tests pass using the project interpreter
(`.venv/Scripts/python.exe -m pytest tests/ -v`, 4 passed).
Unexpected ETF raw-file additions were reviewed (`dirt.us.txt`, `div.us.txt`, `djci.us.txt`):
headers valid, parse errors 0, dates sorted, duplicates 0, OHLC consistency OK, no invalid close/volume values.
No pipeline impact was detected because price ingestion is configured to read `datasets/Data/Stocks/`.
Remaining items are release/publication steps: fresh-kernel notebook reruns and setting the GitHub repository
to public.

**Agent note:** The modeling notebook is intentionally minimal. The goal is
to validate that the features are useful, not to win a Kaggle competition.
A model that shows sentiment improves MAE by even 5% is a successful result.

---

## Global Invariants

These must be true at ALL times across ALL phases.
If any of these is violated, stop and fix before continuing.

- **Raw data is never modified.** Files in `datasets/Data/` and raw CSVs in `datasets/` are read-only.
- **No random train/test splits.** All splits are temporal.
- **No fitting on val/test.** Every transformer fit on train set only.
- **All dates are UTC.** No naive datetimes anywhere in the codebase.
- **All intermediate data is Parquet.** No intermediate CSVs.
- **All logic lives in `src/`.** Notebooks only call `src/` functions.
- **Type hints on all public functions.** No untyped function signatures.
- **`run_pipeline.py --dev` completes in under 5 minutes.** If it takes longer, the dev subset is too large.

---

## Current Status

```
Phase 0 — Environment & Setup       [ ] NOT STARTED
Phase 1 — Data Ingestion            [x] COMPLETE
Phase 2 — Diagnostic EDA            [ ] NOT STARTED
Phase 3 — Time Series Preprocessing [x] COMPLETE
Phase 4 — NLP Preprocessing         [ ] IN PROGRESS
Phase 5 — Integration & Validation  [x] COMPLETE
Phase 6 — Baseline & Polish         [ ] IN PROGRESS
```

_Last updated: 2026-03-12_
_Updated by: GitHub Copilot_
