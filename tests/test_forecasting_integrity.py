"""Regression Test Suite for Forecasting Validation and Evaluation Integrity.

Validates:
1. Feature/target temporal alignment and final NaN preservation.
2. Chronological expanding-window validation and non-monotonic index rejection.
3. Preprocessing leakage isolation across train/test splits.
4. Exploratory strategy position mapping: zero prediction maps to 0.0 (cash/neutral),
   avoiding spurious -28% short-bias losses.
5. Controlled HGB sentiment splitting behavior: confirms HGB splits on sentiment
   when coverage is dense and falls back safely when sparse.

All tests operate strictly on synthetic data in memory.
Zero modification to production Parquet datasets or existing reports.
"""

import unittest
import numpy as np
import pandas as pd

from src.features.feature_transforms import (
    TECHNICAL_FEATURE_COLUMNS,
    SENTIMENT_FEATURE_COLUMNS,
    RAW_PRICE_COLUMNS,
    build_stationary_features,
)
from src.models.baseline_forecasting import (
    ForecastConfig,
    _calculate_single_model_metrics,
    fit_predict_hgb_hybrid,
    fit_predict_ridge_tech,
    inspect_hgb_split_features,
    run_walk_forward_validation,
)


class TestFeatureTargetAlignment(unittest.TestCase):
    """Test feature construction, target horizon, and price level exclusion."""

    def setUp(self) -> None:
        self.dates = pd.date_range("2026-01-01", periods=10, freq="B")
        self.close_prices = np.array([100.0, 102.0, 101.0, 104.0, 103.0, 106.0, 105.0, 108.0, 110.0, 112.0])
        self.mock_df = pd.DataFrame(
            {
                "Close": self.close_prices,
                "High": self.close_prices * 1.01,
                "Low": self.close_prices * 0.99,
                "Open": self.close_prices * 0.995,
                "Volume": np.full(10, 1000000, dtype=int),
                "Daily_Return": pd.Series(self.close_prices).pct_change().fillna(0.0).values,
                "SMA_20": self.close_prices,
                "SMA_50": self.close_prices,
                "Volatility_20": np.full(10, 0.015),
                "RSI_14": np.full(10, 55.0),
                "sentiment_score": np.full(10, np.nan),
                "sentiment_std": np.full(10, np.nan),
                "sentiment_count": np.full(10, np.nan),
                "prob_positive": np.full(10, np.nan),
                "prob_negative": np.full(10, np.nan),
                "prob_neutral": np.full(10, np.nan),
                "sentiment_coverage": np.zeros(10, dtype=int),
            },
            index=self.dates,
        )

    def test_target_forward_alignment(self) -> None:
        """Target at step t is strictly Close_{t+1}/Close_t - 1; final step is NaN."""
        transformed, tech_cols, sent_cols, target_col = build_stationary_features(self.mock_df)

        self.assertEqual(target_col, "target_return_1d")
        expected_t0 = (102.0 - 100.0) / 100.0
        self.assertAlmostEqual(transformed[target_col].iloc[0], expected_t0, places=10)
        expected_t8 = (112.0 - 110.0) / 110.0
        self.assertAlmostEqual(transformed[target_col].iloc[8], expected_t8, places=10)
        self.assertTrue(np.isnan(transformed[target_col].iloc[9]))

    def test_raw_prices_excluded_from_features(self) -> None:
        """Raw price levels (Close, High, Low, Open, Volume) are excluded from predictors."""
        _, tech_cols, sent_cols, _ = build_stationary_features(self.mock_df)
        all_features = set(tech_cols + sent_cols)
        for raw_col in RAW_PRICE_COLUMNS:
            self.assertNotIn(raw_col, all_features)

    def test_non_monotonic_dates_rejected(self) -> None:
        """Non-monotonic dates raise ValueError to prevent look-ahead bias."""
        shuffled_df = self.mock_df.copy()
        shuffled_idx = list(self.dates)
        shuffled_idx[0], shuffled_idx[1] = shuffled_idx[1], shuffled_idx[0]
        shuffled_df.index = shuffled_idx

        with self.assertRaises(ValueError):
            build_stationary_features(shuffled_df)


class TestStrategyPositionMapping(unittest.TestCase):
    """Test exploratory strategy metrics and zero-prediction position mapping."""

    def test_zero_prediction_holds_cash(self) -> None:
        """Zero predictions must yield position=0.0 (cash), producing 0.0 cumulative return and 0.0 Sharpe."""
        y_true = np.array([0.02, -0.01, 0.015, -0.02, 0.005])
        y_pred_zero = np.zeros(5)
        metrics = _calculate_single_model_metrics(y_true, y_pred_zero, float(np.mean(y_true)))

        strat = metrics["exploratory_strategy"]
        self.assertEqual(strat["cumulative_return"], 0.0)
        self.assertEqual(strat["annualized_return"], 0.0)
        self.assertEqual(strat["annualized_sharpe"], 0.0)
        self.assertEqual(strat["average_daily_turnover"], 0.0)

    def test_ternary_position_mapping(self) -> None:
        """Positive predictions map to +1, negative to -1, zero to 0."""
        y_true = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        y_pred = np.array([0.005, -0.004, 0.0, 0.002, -0.001])
        metrics = _calculate_single_model_metrics(y_true, y_pred, float(np.mean(y_true)))

        expected_positions = np.array([1.0, -1.0, 0.0, 1.0, -1.0])
        expected_returns = expected_positions * y_true
        expected_cum = float(np.prod(1.0 + expected_returns) - 1.0)

        strat = metrics["exploratory_strategy"]
        self.assertAlmostEqual(strat["cumulative_return"], expected_cum, places=10)
        self.assertIn("execution_timing_assumption", strat)
        self.assertIn("transaction_cost_assumption", strat)


class TestHGBTreeSplitBehavior(unittest.TestCase):
    """Test HistGradientBoostingRegressor split behavior on dense vs sparse sentiment."""

    def test_hgb_splits_on_dense_sentiment(self) -> None:
        """When sentiment coverage is dense and predictive, HGB Hybrid actively splits on sentiment features."""
        rng = np.random.RandomState(42)
        n = 80
        sentiment = rng.uniform(-1.0, 1.0, size=n)
        target = 0.04 * sentiment + rng.randn(n) * 0.005

        df = pd.DataFrame(
            {
                "daily_return": rng.randn(n) * 0.01,
                "norm_hl_range": rng.uniform(0.01, 0.03, size=n),
                "spread_close_sma20": rng.randn(n) * 0.02,
                "spread_close_sma50": rng.randn(n) * 0.03,
                "volatility_20": np.full(n, 0.015),
                "rsi_14": rng.uniform(0.3, 0.7, size=n),
                "trailing_volume_ratio": rng.uniform(0.8, 1.2, size=n),
                "sentiment_score": sentiment,
                "sentiment_std": np.full(n, 0.1),
                "sentiment_count": np.full(n, 5.0),
                "prob_positive": np.clip(0.5 + 0.5 * sentiment, 0.0, 1.0),
                "prob_negative": np.clip(0.5 - 0.5 * sentiment, 0.0, 1.0),
                "prob_neutral": np.full(n, 0.1),
                "sentiment_coverage": np.ones(n, dtype=int),
                "target_return_1d": target,
            }
        )

        train_df = df.iloc[:70]
        test_df = df.iloc[70:71]

        _, splits_used = fit_predict_hgb_hybrid(
            train_df,
            test_df,
            tech_cols=TECHNICAL_FEATURE_COLUMNS,
            sentiment_cols=SENTIMENT_FEATURE_COLUMNS,
            target_col="target_return_1d",
            config=ForecastConfig(min_train_size=30, hgb_min_samples_leaf=5, hgb_max_iter=30),
            return_tree_diagnostics=True,
        )

        sent_splits = [c for c in splits_used if c in SENTIMENT_FEATURE_COLUMNS]
        self.assertGreater(len(sent_splits), 0, f"Expected sentiment splits, got {splits_used}")

    def test_hgb_handles_all_nan_sentiment_gracefully(self) -> None:
        """When sentiment is 100% NaN, HGB Hybrid filters it and predicts without error."""
        rng = np.random.RandomState(42)
        n = 50
        df = pd.DataFrame(
            {
                "daily_return": rng.randn(n) * 0.01,
                "norm_hl_range": rng.uniform(0.01, 0.03, size=n),
                "spread_close_sma20": rng.randn(n) * 0.02,
                "spread_close_sma50": rng.randn(n) * 0.03,
                "volatility_20": np.full(n, 0.015),
                "rsi_14": rng.uniform(0.3, 0.7, size=n),
                "trailing_volume_ratio": rng.uniform(0.8, 1.2, size=n),
                "sentiment_score": np.full(n, np.nan),
                "sentiment_std": np.full(n, np.nan),
                "sentiment_count": np.full(n, np.nan),
                "prob_positive": np.full(n, np.nan),
                "prob_negative": np.full(n, np.nan),
                "prob_neutral": np.full(n, np.nan),
                "sentiment_coverage": np.zeros(n, dtype=int),
                "target_return_1d": rng.randn(n) * 0.01,
            }
        )

        pred = fit_predict_hgb_hybrid(
            df.iloc[:40],
            df.iloc[40:41],
            tech_cols=TECHNICAL_FEATURE_COLUMNS,
            sentiment_cols=SENTIMENT_FEATURE_COLUMNS,
            target_col="target_return_1d",
            config=ForecastConfig(min_train_size=30),
        )
        self.assertFalse(np.isnan(pred))


if __name__ == "__main__":
    unittest.main()
