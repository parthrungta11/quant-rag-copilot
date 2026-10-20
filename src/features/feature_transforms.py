"""Stationary feature transformations for financial return forecasting.

Transforms raw price and technical series into stationary, backward-looking
features and constructs the 1-session-ahead forward return prediction target.
"""

from typing import List, Tuple
import pandas as pd
import numpy as np


TECHNICAL_FEATURE_COLUMNS: List[str] = [
    "daily_return",
    "norm_hl_range",
    "spread_close_sma20",
    "spread_close_sma50",
    "volatility_20",
    "rsi_14",
    "trailing_volume_ratio",
]

SENTIMENT_FEATURE_COLUMNS: List[str] = [
    "sentiment_score",
    "sentiment_std",
    "sentiment_count",
    "prob_positive",
    "prob_negative",
    "prob_neutral",
    "sentiment_coverage",
]

RAW_PRICE_COLUMNS: List[str] = ["Close", "High", "Low", "Open", "Volume"]
TARGET_COLUMN: str = "target_return_1d"
TARGET_DIRECTION_COLUMN: str = "target_direction_1d"


def build_stationary_features(
    market_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], List[str], str]:
    """Transform market and sentiment dataset into stationary forecasting features.

    Parameters
    ----------
    market_df : pd.DataFrame
        Market features dataframe with DatetimeIndex (from market_features.parquet),
        containing raw price columns, technical indicators, and merged sentiment.

    Returns
    -------
    Tuple[pd.DataFrame, List[str], List[str], str]
        - Transformed DataFrame containing only stationary features, sentiment features,
          and targets (raw price levels are strictly excluded from predictors).
        - List of technical feature column names.
        - List of sentiment feature column names.
        - Name of the forward return target column ('target_return_1d').

    Invariants Enforced
    -------------------
    1. Zero look-ahead bias: All features at index t use information available strictly
       at or before market close of session t.
    2. Forward target: target_return_1d at row t is (Close_{t+1} - Close_t) / Close_t.
       The final row t = T has target_return_1d == NaN.
    3. Missing sentiment preservation: Uncovered rows maintain NaN across sentiment metrics
       and sentiment_coverage == 0.
    4. Stationarity: Non-stationary price levels (Close, High, Low, Open, Volume) are excluded.
    """
    df = market_df.copy()

    # Validate monotonic chronological ordering to prevent forward leakage
    if not df.index.is_monotonic_increasing:
        raise ValueError("Input market_df index must be monotonically increasing to prevent look-ahead bias.")
    if df.index.has_duplicates:
        raise ValueError("Input market_df index contains duplicate timestamps.")

    # Validate required columns
    required_market_cols = ["Close", "High", "Low", "Open", "Volume", "Daily_Return", "SMA_20", "SMA_50", "Volatility_20", "RSI_14"]
    for col in required_market_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required market column: '{col}'")

    transformed = pd.DataFrame(index=df.index)

    # 1. Stationary technical indicators (all backward-looking)
    transformed["daily_return"] = df["Daily_Return"].astype(float)
    transformed["norm_hl_range"] = ((df["High"] - df["Low"]) / df["Close"]).astype(float)
    transformed["spread_close_sma20"] = ((df["Close"] - df["SMA_20"]) / df["SMA_20"]).astype(float)
    transformed["spread_close_sma50"] = ((df["Close"] - df["SMA_50"]) / df["SMA_50"]).astype(float)
    transformed["volatility_20"] = df["Volatility_20"].astype(float)
    transformed["rsi_14"] = (df["RSI_14"] / 100.0).astype(float)

    # Trailing 20-session volume ratio (strictly backward-looking, min_periods=1 prevents NaNs)
    volume_sma20 = df["Volume"].rolling(window=20, min_periods=1).mean()
    transformed["trailing_volume_ratio"] = (df["Volume"] / volume_sma20).astype(float)

    # 2. Sentiment features (preserving NaN in uncovered sessions)
    for col in SENTIMENT_FEATURE_COLUMNS:
        if col in df.columns:
            transformed[col] = df[col]
        else:
            # Fallback if sentiment columns are missing from market_df
            if col == "sentiment_coverage":
                transformed[col] = 0
            else:
                transformed[col] = np.nan

    # 3. Target construction: 1-session-ahead close-to-close return
    # Target at index t represents return from Close_t to Close_{t+1}
    target = df["Close"].shift(-1) / df["Close"] - 1.0
    transformed[TARGET_COLUMN] = target.astype(float)

    # Binary directional target (1 = positive return, 0 = non-positive return, NaN preserved)
    direction = pd.Series(np.nan, index=df.index, dtype=float)
    valid_mask = target.notna()
    direction.loc[valid_mask] = (target.loc[valid_mask] > 0.0).astype(float)
    transformed[TARGET_DIRECTION_COLUMN] = direction

    # Invariant checks: Confirm no raw price levels are present in feature lists
    for raw_col in RAW_PRICE_COLUMNS:
        if raw_col in TECHNICAL_FEATURE_COLUMNS or raw_col in SENTIMENT_FEATURE_COLUMNS:
            raise ValueError(f"Raw price column '{raw_col}' detected in feature definitions.")

    # Confirm final row target is NaN
    if not np.isnan(transformed[TARGET_COLUMN].iloc[-1]):
        raise ValueError("Final row target must be NaN (future close unavailable).")

    return transformed, TECHNICAL_FEATURE_COLUMNS, SENTIMENT_FEATURE_COLUMNS, TARGET_COLUMN
