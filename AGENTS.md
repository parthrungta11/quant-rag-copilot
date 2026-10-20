# Quant RAG Copilot — Architecture & Operational Guide

## Overview
Quant RAG Copilot is an end-to-end quantitative financial analytics and machine learning system that ingests market data and financial news sentiment (via FinBERT), prevents look-ahead bias, engineers stationary predictors, and performs chronological walk-forward return forecasting.

---

## Directory & Package Structure
```
quant-rag-copilot/
├── .env                                  # API keys (FINNHUB_API_KEY) - Excluded from Git
├── .gitignore                            # Excludes venv/, data/, .env
├── requirements.txt                      # Project dependencies
├── AGENTS.md                             # Agent operational guide and architecture documentation
├── README.md                             # Repository overview
├── data/                                 # Local datasets (git-ignored)
│   ├── raw/
│   │   ├── market_raw.parquet            # Raw OHLCV market data from yfinance (AAPL)
│   │   └── news_raw.parquet              # Deduplicated raw financial headlines (Finnhub + Yahoo fallback)
│   └── processed/
│       ├── market_features.parquet       # Market technicals + session-aligned FinBERT sentiment (451 rows)
│       ├── news_sentiment.parquet        # Processed headlines with FinBERT scores and assigned trading dates
│       └── forecast_predictions.parquet  # Walk-forward out-of-sample return predictions (251 steps)
├── reports/
│   └── baseline_validation_metrics.json  # Comprehensive validation metrics & ablation comparisons
└── src/
    ├── __init__.py                       # Package root marker
    ├── build_pipeline.py                 # End-to-end pipeline orchestrator & test runner
    ├── data_ingestion/
    │   ├── __init__.py                   # Data ingestion package marker
    │   ├── market_loader.py              # Historical OHLCV ingestion via yfinance
    │   └── news_sentiment.py             # Company news ingestion, FinBERT inference, and market-session alignment
    ├── features/
    │   ├── __init__.py                   # Feature package marker
    │   ├── feature_eng.py                # Technical indicator calculations (RSI, SMA, volatility)
    │   └── feature_transforms.py         # Stationary feature generation and forward return target construction
    └── models/
        ├── __init__.py                   # Models package marker
        └── baseline_forecasting.py       # Walk-forward validation engine, baselines, and ablation study
```

---

## Pipeline Stages

### Stage 1: Market Data Ingestion (`src/data_ingestion/market_loader.py`)
- Downloads historical daily adjusted prices and volumes for AAPL via `yfinance`.
- Persists raw data to `data/raw/market_raw.parquet`.

### Stage 2: Technical Feature Engineering (`src/features/feature_eng.py`)
- Calculates initial indicators: `Daily_Return`, `SMA_20`, `SMA_50`, `Volatility_20`, and `RSI_14`.
- Drops warm-up rows to establish complete indicator history, producing `data/processed/market_features.parquet` (451 rows spanning 2024-12-31 to 2026-10-19).

### Stage 3: News Sentiment Ingestion & FinBERT Inference (`src/data_ingestion/news_sentiment.py`)
- Primary source: Finnhub `company_news` with 1-day request chunking (`chunk_days=1`) to eliminate the ~250-article server-side response cap.
- Secondary fallback: Yahoo Finance news via `yfinance.Search`, strictly filtered to the requested date range.
- Date-boundary convention: Inclusive calendar dates in reference timezone (`America/New_York`). Timestamps are converted to `America/New_York` before range evaluation, safely discarding missing/invalid timestamps.
- Model: `ProsusAI/finbert` (Hugging Face Transformers / PyTorch), generating sentiment probabilities and continuous scores:
  $$\text{sentiment\_score} = P(\text{positive}) - P(\text{negative}) \in [-1.0, 1.0]$$
- **Look-Ahead Bias Elimination & Equity Calendar**:
  - Regular session cutoff: 16:00:00 ET.
  - Early-close cutoff: 13:00:00 ET on scheduled sessions (Black Friday, Christmas Eve weekday, day before July 4th if weekday).
  - Exchange holiday calendar: `USEquityMarketHolidayCalendar` (NYSE/NASDAQ rules: includes Good Friday and Juneteenth; excludes federal holidays Columbus Day and Veterans Day when exchanges are open).
  - Articles published at or after session close, over weekends, or on market holidays are strictly mapped to the **next open trading session**.
  - Invariant: An article is NEVER assigned to a trading session preceding its publication timestamp.
- **Safe Incremental Updates vs Full Rebuild**:
  - Incremental update (`full_rebuild=False`, default): Preserves historical sentiment for all market dates outside the specified refresh window `[refresh_start_date, refresh_end_date]`. Empty retrieval does NOT erase historical data.
  - Full rebuild (`full_rebuild=True`, `--full-rebuild`): Deliberately clears and replaces sentiment columns across the entire market dataset.
  - Missing Sentiment Invariant: Sessions lacking news coverage (417 of 451 sessions) have `sentiment_coverage = 0` and strictly retain `NaN` for all sentiment metrics (never imputed with `0.0`).

### Stage 4: Baseline Forecasting & Walk-Forward Validation (`src/models/baseline_forecasting.py`)
- **Stationary Feature Transformations** (`src/features/feature_transforms.py`):
  - Strictly backward-looking stationary predictors: `daily_return`, `norm_hl_range`, `spread_close_sma20`, `spread_close_sma50`, `volatility_20`, `rsi_14`, and `trailing_volume_ratio`.
  - Raw price levels (`Close`, `Open`, `High`, `Low`, raw `Volume`) are strictly excluded from model inputs.
- **Target Construction**:
  - Primary regression target: 1-session-ahead close-to-close forward return:
    $$R_{t+1} = \frac{\text{Close}_{t+1} - \text{Close}_t}{\text{Close}_t}$$
  - The final available market session ($t = T$, 2026-10-19) strictly retains $R_{T+1} = \text{NaN}$ (unrealized forward session).
- **Validation Scheme**:
  - Expanding-window walk-forward validation with an initial 200-session warm-up window ($W_{\text{min}} = 200$).
  - Evaluates 250 out-of-sample sessions (2025-10-20 to 2026-10-16).
  - All preprocessing (scalers, imputers) is fit **strictly on each training fold** $[0, t-1]$ and applied out-of-sample to session $t$.
- **Model Tracks & Missing Value Handling**:
  - `Naive Zero`: $\hat{R}_{t+1} = 0.0$ (driftless random walk).
  - `Naive Lag`: $\hat{R}_{t+1} = R_t$ (momentum persistence).
  - `Ridge Tech`: Regularized linear regression ($\alpha = 10.0$) on standardized technical features.
  - `HGB Tech`: `HistGradientBoostingRegressor` (`max_depth=3`, `min_samples_leaf=10`, `learning_rate=0.03`, `max_iter=50`, `l2_reg=1.0`) on technical features.
  - `Ridge Hybrid`: Ridge regression with training-fold-only `0.0` neutral imputation + explicit `sentiment_coverage` indicator.
  - `HGB Hybrid`: `HistGradientBoostingRegressor` on technical + sentiment features using native `NaN` routing.
- **Controlled Sub-Period Ablation**:
  - Technical-only and Hybrid models are evaluated on the exact same 33 news-covered sessions (2026-09-01 to 2026-10-16) to isolate the incremental value of sentiment.

---

## Validation Commands

```powershell
# Run the complete test suite across all stages
.\venv\Scripts\python.exe src/build_pipeline.py --test

# Run only the forecasting and walk-forward test suite
.\venv\Scripts\python.exe src/models/baseline_forecasting.py --test

# Run only the news sentiment pipeline test suite
.\venv\Scripts\python.exe src/data_ingestion/news_sentiment.py --test

# Run data integrity and trading session regression test suite
.\venv\Scripts\python.exe -m unittest tests/test_data_integrity.py

# Run forecasting validation and strategy integrity test suite
.\venv\Scripts\python.exe -m unittest tests/test_forecasting_integrity.py

# Run all regression test suites via unittest discovery
.\venv\Scripts\python.exe -m unittest discover tests

# Execute Stage 4 walk-forward validation on production dataset
.\venv\Scripts\python.exe src/models/baseline_forecasting.py

# Execute full pipeline end-to-end
.\venv\Scripts\python.exe src/build_pipeline.py --stage all
```

---

## Empirical Findings & Limitations
1. **Zero-Return Dominance**:
   - The naive driftless baseline ($\hat{R}_{t+1} = 0.0$) yields the lowest RMSE ($0.01520$) across the 250 walk-forward sessions, confirming that high-frequency daily equity returns have an extremely low signal-to-noise ratio.
2. **Technical Model Performance**:
   - `Ridge Tech` achieves the highest directional accuracy ($51.2\%$) across the 250-session test period, with $R^2_{\text{vs\_zero}} = -0.0121$ and $R^2_{\text{descriptive}} = -0.0185$, typical for unconstrained linear return forecasts without cross-asset features.
3. **Sentiment Sample Size Limitation & Tree Split Audit**:
   - Direct node-by-node inspection of all 251 fitted `HistGradientBoostingRegressor` models confirmed that **0 sentiment features were selected for splits across any training fold**.
   - An in-depth audit isolated two structural causes:
     a) For folds $t=200$ to $416$ (217 folds, 86.5% of the evaluation), the training window contained zero sentiment records, causing all sentiment features to be dropped prior to fitting (`nunique < 2`).
     b) For folds $t=417$ to $450$ (34 folds), non-null sentiment was present in only 0.7% to 7.3% of training samples (3 to 33 out of 420–450 samples). Under `min_samples_leaf=10` and L2 regularization, partitioning 400+ samples on a 93%+ NaN feature produces negligible loss reduction compared to continuous, 100% dense technical indicators.
   - **Controlled Experiment**: When HGB was trained exclusively on the 33 sentiment-covered sessions with dense observations, HGB actively split on `sentiment_score` and `sentiment_std`, confirming that 0 splits in walk-forward validation is a mathematical consequence of extreme sparsity diluted over dense technical history, not an inability of tree architectures to process sentiment.
4. **Exploratory Strategy Assumptions & Caveats**:
   - The sign-based trading metrics are strictly exploratory theoretical proxies and must **not** be presented as realizable trading performance.
   - Position mapping: $\hat{R} > 0 \rightarrow +1$ (long), $\hat{R} < 0 \rightarrow -1$ (short), $\hat{R} = 0 \rightarrow 0$ (cash/neutral). The zero-return baseline holds cash with 0.0 cumulative return and 0.0 Sharpe.
   - Unmodelled execution frictions: 0 bps transaction costs, 0 bps borrow fees, 0 slippage, and theoretical instantaneous execution at session Close $t$ (which is infeasible when incorporating news published up to 16:00:00 ET; real-world execution would occur at Open $t+1$).
