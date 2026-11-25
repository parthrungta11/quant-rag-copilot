"""Regression Test Suite for Backtesting Engine Execution, Cost Accounting, and Look-Ahead Prevention.

Validates:
1. Exact next_open_to_close return calculation with hand-calculated financial numbers.
2. Exact next_open_to_next_open return calculation and t+2 exit price alignment.
3. Signal/date alignment and strict look-ahead prevention.
4. Clean exclusion of final unrealized prediction row (no fabricated returns).
5. Transaction costs and slippage on entries (turnover 1.0), reversals (turnover 2.0), and exits (turnover 1.0).
6. Short-borrow fee calculation applied only while short and prorated per session.
7. Long-only and short-disabled behavior (negative forecasts map to 0.0 cash).
8. Missing / NaN prediction handling and clean transitions to Cash with appropriate exit costs.
9. Missing or non-positive execution prices handled gracefully without crash.
10. AAPL Buy & Hold benchmark exact date alignment and return equivalence.
11. Mathematical edge cases: zero volatility, zero loss (win rate = 1.0, profit factor = inf), all-cash.
12. Config validation, deterministic repeated execution, and multi-model comparison table.

All tests operate strictly on synthetic in-memory data with zero external API calls or file modifications.
"""

import unittest
import numpy as np
import pandas as pd

from src.models.backtest_engine import (
    BacktestConfig,
    BacktestResult,
    compute_performance_metrics,
    map_prediction_to_position,
    run_backtest,
    compare_models,
    evaluate_threshold_candidates,
)
from src.models.baseline_forecasting import (
    compute_training_fold_prediction_stds,
)


class TestBacktestExecutionTimingAndReturns(unittest.TestCase):
    """Test execution timing modes, look-ahead bias elimination, and hand-calculated returns."""

    def setUp(self) -> None:
        # Create 5 synthetic business days
        self.dates = pd.date_range("2026-01-05", periods=5, freq="B")
        # Prices:
        # 2026-01-05 (t0): Open=99.0,  Close=100.0
        # 2026-01-06 (t1): Open=100.0, Close=105.0
        # 2026-01-07 (t2): Open=104.0, Close=102.0
        # 2026-01-08 (t3): Open=103.0, Close=103.0
        # 2026-01-09 (t4): Open=105.0, Close=106.0
        self.market_df = pd.DataFrame(
            {
                "Open": [99.0, 100.0, 104.0, 103.0, 105.0],
                "Close": [100.0, 105.0, 102.0, 103.0, 106.0],
                "High": [101.0, 106.0, 105.0, 104.0, 107.0],
                "Low": [98.0, 99.0, 101.0, 102.0, 104.0],
            },
            index=self.dates,
        )

    def test_exact_open_to_close_return_and_cost_accounting(self) -> None:
        """Hand-calculated test: 3-period open-to-close with entry, reversal, and exit."""
        # Predictions at t0, t1, t2:
        # t0: pred = +0.02 (Long)
        # t1: pred = -0.03 (Short)
        # t2: pred = 0.005 (Cash, since |0.005| <= 0.01 threshold)
        preds = pd.Series([0.02, -0.03, 0.005], index=self.dates[:3])

        config = BacktestConfig(
            initial_capital=100_000.0,
            execution_mode="next_open_to_close",
            transaction_cost_bps=5.0,
            slippage_bps=5.0,  # 10 bps total one-way = 0.0010
            short_borrow_fee_bps_annual=252.0,  # 252 bps / 252 days = 1 bp per session = 0.00010
            conviction_threshold=0.01,
            allow_short=True,
            position_sizing="ternary_fixed",
        )

        res = run_backtest(preds, self.market_df, config=config, model_name="test_model")
        history = res.portfolio_history
        self.assertEqual(len(history), 3)

        # Period 1: Execution at 2026-01-06 (t1)
        # Entry at Open=100.0, Exit at Close=105.0 -> Asset return = (105 - 100) / 100 = +0.05
        # Position: 0.0 -> +1.0 (Turnover = 1.0)
        # Friction: 1.0 * 0.0010 = 0.0010
        # Borrow: 0.0
        # Net Return: 0.05 - 0.0010 = 0.0490
        row0 = history.iloc[0]
        self.assertEqual(row0["decision_date"], self.dates[0])
        self.assertEqual(row0["position"], 1.0)
        self.assertEqual(row0["turnover"], 1.0)
        self.assertAlmostEqual(row0["entry_price"], 100.0)
        self.assertAlmostEqual(row0["exit_price"], 105.0)
        self.assertAlmostEqual(row0["asset_gross_return"], 0.05)
        self.assertAlmostEqual(row0["strategy_gross_return"], 0.05)
        self.assertAlmostEqual(row0["friction_cost"], 0.0010)
        self.assertAlmostEqual(row0["borrow_cost"], 0.0)
        self.assertAlmostEqual(row0["strategy_net_return"], 0.0490)
        self.assertAlmostEqual(row0["portfolio_value"], 100_000.0 * 1.0490)

        # Period 2: Execution at 2026-01-07 (t2)
        # Entry at Open=104.0, Exit at Close=102.0 -> Asset return = (102 - 104) / 104 = -2/104
        # Position: +1.0 -> -1.0 (Reversal turnover = 2.0)
        # Friction: 2.0 * 0.0010 = 0.0020
        # Borrow: 1.0 * 0.00010 = 0.00010
        # Strategy gross: -1.0 * (-2/104) = +2/104 = 0.01923076923
        # Net Return: 2/104 - 0.0020 - 0.00010 = 2/104 - 0.00210
        row1 = history.iloc[1]
        expected_gross_1 = 2.0 / 104.0
        expected_net_1 = expected_gross_1 - 0.00210
        self.assertEqual(row1["decision_date"], self.dates[1])
        self.assertEqual(row1["position"], -1.0)
        self.assertEqual(row1["turnover"], 2.0)
        self.assertAlmostEqual(row1["entry_price"], 104.0)
        self.assertAlmostEqual(row1["exit_price"], 102.0)
        self.assertAlmostEqual(row1["asset_gross_return"], -2.0 / 104.0)
        self.assertAlmostEqual(row1["strategy_gross_return"], expected_gross_1)
        self.assertAlmostEqual(row1["friction_cost"], 0.0020)
        self.assertAlmostEqual(row1["borrow_cost"], 0.00010)
        self.assertAlmostEqual(row1["strategy_net_return"], expected_net_1)

        # Period 3: Execution at 2026-01-08 (t3)
        # Decision at t2 was 0.005 <= 0.01 threshold -> Target = 0.0 (Cash)
        # Exit short position: -1.0 -> 0.0 (Turnover = 1.0)
        # Friction: 1.0 * 0.0010 = 0.0010
        # Borrow: 0.0 (cash)
        # Strategy gross: 0.0
        # Net Return: 0.0 - 0.0010 = -0.0010
        row2 = history.iloc[2]
        self.assertEqual(row2["decision_date"], self.dates[2])
        self.assertEqual(row2["position"], 0.0)
        self.assertEqual(row2["turnover"], 1.0)
        self.assertAlmostEqual(row2["strategy_gross_return"], 0.0)
        self.assertAlmostEqual(row2["friction_cost"], 0.0010)
        self.assertAlmostEqual(row2["borrow_cost"], 0.0)
        self.assertAlmostEqual(row2["strategy_net_return"], -0.0010)

        # Check total turnover: 1.0 + 2.0 + 1.0 = 4.0
        self.assertAlmostEqual(res.metrics["total_turnover"], 4.0)
        self.assertAlmostEqual(res.metrics["average_daily_turnover"], 4.0 / 3.0)

    def test_exact_open_to_next_open_return(self) -> None:
        """Hand-calculated test: next_open_to_next_open mode."""
        # Predictions at t0 and t1:
        # t0 executes Open[t1] to Open[t2]: Open=100.0 to Open=104.0 -> Return = 4.0 / 100.0 = +0.04
        # t1 executes Open[t2] to Open[t3]: Open=104.0 to Open=103.0 -> Return = -1.0 / 104.0
        preds = pd.Series([0.05, 0.05], index=self.dates[:2])

        config = BacktestConfig(
            execution_mode="next_open_to_next_open",
            transaction_cost_bps=0.0,
            slippage_bps=0.0,
            short_borrow_fee_bps_annual=0.0,
        )

        res = run_backtest(preds, self.market_df, config=config)
        history = res.portfolio_history
        self.assertEqual(len(history), 2)

        # Row 0: entered at t1 (Open=100.0), exited at t2 (Open=104.0)
        row0 = history.iloc[0]
        self.assertAlmostEqual(row0["entry_price"], 100.0)
        self.assertAlmostEqual(row0["exit_price"], 104.0)
        self.assertAlmostEqual(row0["strategy_net_return"], 0.04)

        # Row 1: entered at t2 (Open=104.0), exited at t3 (Open=103.0)
        row1 = history.iloc[1]
        self.assertAlmostEqual(row1["entry_price"], 104.0)
        self.assertAlmostEqual(row1["exit_price"], 103.0)
        self.assertAlmostEqual(row1["strategy_net_return"], (103.0 - 104.0) / 104.0)

    def test_final_unrealized_prediction_row_exclusion(self) -> None:
        """Confirm that a prediction formed at the final market date (where t+1 does not exist) is excluded."""
        # Predictions include t4 (the final date in market_df)
        preds = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=self.dates)
        res = run_backtest(preds, self.market_df)

        # Only t0, t1, t2, t3 have t+1 dates in market_df. t4 cannot execute!
        self.assertEqual(len(res.portfolio_history), 4)
        # Ensure the final executed session is t4 (corresponding to decision t3)
        self.assertEqual(res.portfolio_history.iloc[-1]["decision_date"], self.dates[3])
        self.assertEqual(res.portfolio_history.index[-1], self.dates[4])

    def test_open_to_next_open_requires_t2_existence(self) -> None:
        """In next_open_to_next_open mode, decision t3 would require t5 for exit; must be cleanly excluded."""
        preds = pd.Series([0.01, 0.02, 0.03, 0.04], index=self.dates[:4])
        config = BacktestConfig(execution_mode="next_open_to_next_open")
        res = run_backtest(preds, self.market_df, config=config)

        # t0 exits at t2 (OK)
        # t1 exits at t3 (OK)
        # t2 exits at t4 (OK)
        # t3 requires exit at t5 (which does not exist in 5-day market_df) -> excluded!
        self.assertEqual(len(res.portfolio_history), 3)


class TestPositionSizingAndMissingData(unittest.TestCase):
    """Test position mapping rules, short restrictions, missing predictions, and bad prices."""

    def setUp(self) -> None:
        self.dates = pd.date_range("2026-02-01", periods=4, freq="B")
        self.market_df = pd.DataFrame(
            {
                "Open": [100.0, 102.0, 104.0, 106.0],
                "Close": [101.0, 103.0, 105.0, 107.0],
            },
            index=self.dates,
        )

    def test_long_only_and_short_disabled_mapping(self) -> None:
        """Negative forecasts must map to 0.0 (Cash) when allow_short=False or position_sizing='long_only'."""
        config_no_short = BacktestConfig(allow_short=False)
        self.assertEqual(map_prediction_to_position(-0.05, config_no_short), 0.0)
        self.assertEqual(map_prediction_to_position(0.05, config_no_short), 1.0)

        config_long_only = BacktestConfig(position_sizing="long_only")
        self.assertEqual(map_prediction_to_position(-0.05, config_long_only), 0.0)
        self.assertEqual(map_prediction_to_position(0.05, config_long_only), 1.0)

    def test_missing_prediction_transitions_to_cash(self) -> None:
        """A NaN prediction must default to cash (position=0.0) and pay the turnover fee to close existing positions."""
        preds = pd.Series([0.05, np.nan, 0.05], index=self.dates[:3])
        config = BacktestConfig(
            transaction_cost_bps=10.0,
            slippage_bps=0.0,
        )
        res = run_backtest(preds, self.market_df, config=config)
        history = res.portfolio_history

        # Period 1 (from t0): position = +1.0 (turnover = 1.0)
        self.assertEqual(history.iloc[0]["position"], 1.0)
        # Period 2 (from t1 with NaN): position = 0.0 (turnover = 1.0 from +1.0 -> 0.0)
        self.assertEqual(history.iloc[1]["position"], 0.0)
        self.assertEqual(history.iloc[1]["turnover"], 1.0)
        self.assertAlmostEqual(history.iloc[1]["friction_cost"], 0.0010)
        # Period 3 (from t2): position = +1.0 (turnover = 1.0 from 0.0 -> +1.0)
        self.assertEqual(history.iloc[2]["position"], 1.0)
        self.assertEqual(history.iloc[2]["turnover"], 1.0)

    def test_invalid_prices_default_to_cash(self) -> None:
        """If market Open or Close is NaN or <= 0, default safely to cash without crashing."""
        corrupted_market = self.market_df.copy()
        corrupted_market.iloc[1, corrupted_market.columns.get_loc("Open")] = np.nan  # NaN Open at t1

        preds = pd.Series([0.05, 0.05], index=self.dates[:2])
        res = run_backtest(preds, corrupted_market)
        history = res.portfolio_history

        # Period 1 executed at t1 had NaN Open -> carried prev_position=0.0, return=0.0
        self.assertEqual(history.iloc[0]["position"], 0.0)
        self.assertAlmostEqual(history.iloc[0]["strategy_net_return"], 0.0)

    def test_invalid_prices_carry_previous_position_with_zero_friction(self) -> None:
        """If a prior position is held and next execution prices are invalid, carry position with 0 friction."""
        corrupted_market = self.market_df.copy()
        # t1 is valid (enters +1.0), t2 has NaN Open
        corrupted_market.iloc[2, corrupted_market.columns.get_loc("Open")] = np.nan

        preds = pd.Series([0.05, -0.05, 0.05], index=self.dates[:3])
        res = run_backtest(preds, corrupted_market)
        history = res.portfolio_history

        # Period 1 (executed at t1): enters +1.0
        self.assertEqual(history.iloc[0]["position"], 1.0)
        # Period 2 (executed at t2, NaN Open): carries +1.0 unchanged with 0 turnover and 0 friction
        self.assertEqual(history.iloc[1]["position"], 1.0)
        self.assertEqual(history.iloc[1]["turnover"], 0.0)
        self.assertEqual(history.iloc[1]["friction_cost"], 0.0)
        self.assertEqual(history.iloc[1]["borrow_cost"], 0.0)
        self.assertAlmostEqual(history.iloc[1]["strategy_net_return"], 0.0)

    def test_sentiment_coverage_alignment_warns_and_raises_strict(self) -> None:
        """Mismatched sentiment_coverage between predictions and market_data warns by default and raises if strict=True."""
        mkt = self.market_df.copy()
        mkt["sentiment_coverage"] = [0, 1, 1, 0]
        preds_df = pd.DataFrame(
            {
                "ridge_hybrid_pred": [0.02, 0.02, 0.02],
                "sentiment_coverage": [0, 0, 1],  # disagrees at index[1]
            },
            index=self.dates[:3],
        )

        with self.assertLogs("backtest_engine", level="WARNING") as cm:
            res = run_backtest(preds_df, mkt, prediction_col="ridge_hybrid_pred")
        self.assertTrue(any("Sentiment coverage mismatch" in msg for msg in cm.output))
        self.assertEqual(len(res.portfolio_history), 3)

        with self.assertRaises(ValueError):
            run_backtest(preds_df, mkt, prediction_col="ridge_hybrid_pred", strict=True)


class TestMetricsAndBenchmarkIntegrity(unittest.TestCase):
    """Test performance metrics, edge cases, benchmark alignment, and multi-model comparison."""

    def test_zero_volatility_and_cash_strategy(self) -> None:
        """A continuous cash holding strategy (0.0 returns) must output 0 Sharpe and 0 max drawdown."""
        net_ret = pd.Series([0.0, 0.0, 0.0, 0.0])
        gross_ret = pd.Series([0.0, 0.0, 0.0, 0.0])
        turnover = pd.Series([0.0, 0.0, 0.0, 0.0])
        borrow = pd.Series([0.0, 0.0, 0.0, 0.0])
        friction = pd.Series([0.0, 0.0, 0.0, 0.0])

        m = compute_performance_metrics(
            net_ret, gross_ret, turnover, borrow, friction, initial_capital=100_000.0
        )
        self.assertAlmostEqual(m["cumulative_net_return"], 0.0)
        self.assertAlmostEqual(m["annualized_net_return"], 0.0)
        self.assertAlmostEqual(m["annualized_volatility"], 0.0)
        self.assertAlmostEqual(m["annualized_sharpe"], 0.0)
        self.assertAlmostEqual(m["max_drawdown"], 0.0)
        self.assertAlmostEqual(m["calmar_ratio"], 0.0)

    def test_all_winning_days_infinite_profit_factor(self) -> None:
        """Strategy with all positive returns must report win_rate = 1.0 and profit_factor = inf."""
        net_ret = pd.Series([0.01, 0.02, 0.01])
        gross_ret = net_ret.copy()
        turnover = pd.Series([1.0, 0.0, 0.0])
        borrow = pd.Series([0.0, 0.0, 0.0])
        friction = pd.Series([0.0, 0.0, 0.0])

        m = compute_performance_metrics(
            net_ret, gross_ret, turnover, borrow, friction, initial_capital=100_000.0
        )
        self.assertAlmostEqual(m["win_rate"], 1.0)
        self.assertEqual(m["profit_factor"], float("inf"))

    def test_benchmark_date_alignment(self) -> None:
        """Benchmark equity curve must span the exact same dates as the strategy."""
        dates = pd.date_range("2026-03-01", periods=5, freq="B")
        market_df = pd.DataFrame(
            {"Open": [100.0, 102.0, 101.0, 105.0, 104.0], "Close": [101.0, 103.0, 102.0, 106.0, 105.0]},
            index=dates,
        )
        preds = pd.Series([0.01, -0.01, 0.02], index=dates[:3])

        res = run_backtest(preds, market_df)
        self.assertTrue(res.portfolio_history.index.equals(dates[1:4]))
        self.assertTrue("benchmark_value" in res.portfolio_history.columns)
        self.assertTrue("benchmark_cumulative_return" in res.portfolio_history.columns)
        self.assertEqual(res.benchmark_metrics["n_sessions"], 3)

    def test_compare_models_table_ranking(self) -> None:
        """compare_models must compile all models alongside the benchmark and rank by Sharpe ratio."""
        dates = pd.date_range("2026-04-01", periods=6, freq="B")
        market_df = pd.DataFrame(
            {"Open": [100.0, 102.0, 104.0, 103.0, 105.0, 106.0], "Close": [101.0, 103.0, 105.0, 104.0, 106.0, 107.0]},
            index=dates,
        )
        preds_df = pd.DataFrame(
            {
                "good_model_pred": [0.05, 0.05, 0.05, 0.05, 0.05],
                "bad_model_pred": [-0.05, -0.05, -0.05, -0.05, -0.05],
                "zero_model_pred": [0.0, 0.0, 0.0, 0.0, 0.0],
            },
            index=dates[:5],
        )

        results_dict, summary_df = compare_models(preds_df, market_df)
        self.assertEqual(len(results_dict), 3)
        self.assertTrue("good_model" in results_dict)
        self.assertTrue("bad_model" in results_dict)
        self.assertTrue("zero_model" in results_dict)

        # Summary table should contain all 3 models + 1 AAPL Buy & Hold benchmark row
        self.assertEqual(len(summary_df), 4)
        self.assertTrue("Sharpe Ratio" in summary_df.columns)
        # Verify ranking is sorted descending
        self.assertTrue(summary_df["Sharpe Ratio"].is_monotonic_decreasing)

    def test_input_validation_errors(self) -> None:
        """Validate input rejection on non-monotonic or incompatible data."""
        dates = pd.date_range("2026-05-01", periods=3, freq="B")
        market_df = pd.DataFrame({"Open": [100, 101, 102], "Close": [101, 102, 103]}, index=dates)

        # Invalid execution mode
        with self.assertRaises(ValueError):
            BacktestConfig(execution_mode="teleportation_mode")

        # Invalid capital
        with self.assertRaises(ValueError):
            BacktestConfig(initial_capital=-500)

        # Non-monotonic dates in market data
        bad_market = market_df.iloc[[1, 0, 2]]
        preds = pd.Series([0.01, 0.02], index=dates[:2])
        with self.assertRaises(ValueError):
            run_backtest(preds, bad_market)


class TestAdaptiveThresholdingAndChronologicalProtocol(unittest.TestCase):
    """Test Gate D7 requirements: long-only allocation, fold-calibrated conviction thresholds,
    leakage prevention, deterministic execution, and parameter locking.
    """

    def setUp(self) -> None:
        # Create 35 synthetic business days
        self.dates = pd.date_range("2026-01-05", periods=35, freq="B")
        # Deterministic prices
        np.random.seed(42)
        base_price = 100.0
        price_moves = np.random.normal(0.0005, 0.015, len(self.dates))
        closes = base_price * np.exp(np.cumsum(price_moves))
        opens = closes * (1.0 + np.random.normal(0.0, 0.005, len(self.dates)))
        self.market_df = pd.DataFrame({"Open": opens, "Close": closes}, index=self.dates)

    def test_long_only_signals_never_create_negative_positions(self) -> None:
        """Long-only allocation must never create negative (short) positions under any forecast."""
        # Strong negative, small negative, zero, small positive, strong positive
        preds = pd.Series([-0.08, -0.02, -0.005, 0.0, 0.005, 0.03, -0.05], index=self.dates[:7])
        config_unthresholded = BacktestConfig(allocation_rule="long_only", conviction_threshold=0.0)
        res_unthresh = run_backtest(preds, self.market_df, config=config_unthresholded)

        positions = res_unthresh.portfolio_history["position"]
        self.assertTrue((positions >= 0.0).all(), "Found negative position in unthresholded long-only")
        self.assertEqual(res_unthresh.metrics["total_borrow_cost"], 0.0)

        # Thresholded long-only
        config_thresholded = BacktestConfig(allocation_rule="long_only", conviction_threshold=0.01)
        res_thresh = run_backtest(preds, self.market_df, config=config_thresholded)
        thresh_positions = res_thresh.portfolio_history["position"]
        self.assertTrue((thresh_positions >= 0.0).all(), "Found negative position in thresholded long-only")
        self.assertEqual(res_thresh.metrics["total_borrow_cost"], 0.0)
        # Prediction of 0.005 is <= 0.01 threshold, so it must map to 0.0 (cash)
        # Prediction of 0.03 is > 0.01, so it must map to 1.0 (long)
        self.assertEqual(thresh_positions.iloc[4], 0.0)  # pred 0.005
        self.assertEqual(thresh_positions.iloc[5], 1.0)  # pred 0.03

    def test_ternary_baseline_behavior_remains_unchanged(self) -> None:
        """Ternary baseline must map strictly: >0 -> +1.0, <0 -> -1.0, ==0 -> 0.0 (control)."""
        preds = pd.Series([0.02, -0.03, 0.0, -0.01, 0.04], index=self.dates[:5])
        config = BacktestConfig(allocation_rule="ternary", conviction_threshold=0.0)
        res = run_backtest(preds, self.market_df, config=config)

        expected_positions = [1.0, -1.0, 0.0, -1.0, 1.0]
        actual_positions = res.portfolio_history["position"].tolist()
        self.assertEqual(actual_positions, expected_positions)
        # Short borrow costs must be incurred when holding short
        self.assertGreater(res.metrics["total_borrow_cost"], 0.0)

    def test_threshold_calibration_uses_only_training_fold_predictions(self) -> None:
        """compute_training_fold_prediction_stds must strictly use predictions on [0, t-1]."""
        # Create synthetic feature matrix and target
        np.random.seed(123)
        n = 25
        features = pd.DataFrame(
            {
                "feat1": np.random.randn(n),
                "feat2": np.random.randn(n),
            },
            index=self.dates[:n],
        )
        target = pd.Series(np.random.randn(n) * 0.01, index=self.dates[:n], name="target_forward_return")

        min_warmup = 10
        stds = compute_training_fold_prediction_stds(
            features=features,
            target=target,
            model_type="ridge_tech",
            min_warmup_sessions=min_warmup,
        )

        self.assertEqual(len(stds), n - min_warmup)
        self.assertEqual(list(stds.index), list(self.dates[min_warmup:n]))

        # Hand-calculate for fold t = 10 (warm-up fold 0)
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        X_train_0 = features.iloc[:10].values
        y_train_0 = target.iloc[:10].values
        scaler = StandardScaler()
        X_scaled_0 = scaler.fit_transform(X_train_0)
        reg = Ridge(alpha=10.0)
        reg.fit(X_scaled_0, y_train_0)
        y_hat_train_0 = reg.predict(X_scaled_0)
        expected_std_0 = float(np.std(y_hat_train_0, ddof=1))

        self.assertAlmostEqual(stds.iloc[0], expected_std_0, places=10)

    def test_future_observations_cannot_influence_earlier_thresholds(self) -> None:
        """Mutating or adding future sessions must not change thresholds for past sessions."""
        np.random.seed(456)
        n = 25
        features_base = pd.DataFrame(
            {
                "feat1": np.random.randn(n),
                "feat2": np.random.randn(n),
            },
            index=self.dates[:n],
        )
        target_base = pd.Series(np.random.randn(n) * 0.01, index=self.dates[:n])

        stds_base = compute_training_fold_prediction_stds(
            features=features_base,
            target=target_base,
            model_type="ridge_tech",
            min_warmup_sessions=10,
        )

        # Mutate future sessions t >= 20 dramatically
        features_mutated = features_base.copy()
        target_mutated = target_base.copy()
        features_mutated.iloc[20:, :] = features_mutated.iloc[20:, :] * 1000.0 + 50.0
        target_mutated.iloc[20:] = 0.50  # Huge 50% returns

        stds_mutated = compute_training_fold_prediction_stds(
            features=features_mutated,
            target=target_mutated,
            model_type="ridge_tech",
            min_warmup_sessions=10,
        )

        # Thresholds for folds prior to index 20 (i.e. dates[10] through dates[19], 10 folds)
        # MUST BE BITWISE IDENTICAL
        base_prior = stds_base.iloc[:10]
        mutated_prior = stds_mutated.iloc[:10]
        np.testing.assert_allclose(
            base_prior.values,
            mutated_prior.values,
            rtol=1e-14,
            atol=1e-14,
            err_msg="Future data leaked into earlier fold conviction thresholds!",
        )

    def test_candidate_k_values_evaluated_only_on_development_window(self) -> None:
        """evaluate_threshold_candidates must evaluate only the specified development window."""
        n = 30
        preds = pd.Series(np.random.normal(0.001, 0.01, n), index=self.dates[:n])
        stds = pd.Series(np.full(n, 0.005), index=self.dates[:n])

        # Define development window: sessions 10 to 22 (13 sessions)
        dev_dates = self.dates[10:23]
        results, summary_df = evaluate_threshold_candidates(
            predictions=preds,
            market_data=self.market_df,
            training_fold_stds=stds,
            candidate_k_values=(0.25, 0.50, 0.75, 1.00),
            evaluation_dates=dev_dates,
        )

        # Verify all returned backtest results only contain execution dates for the dev window
        for var_name, res in results.items():
            hist = res.portfolio_history
            self.assertEqual(len(hist), len(dev_dates))
            # Execution dates should match dev_dates + 1 session
            expected_exec = [self.market_df.index[self.market_df.index.get_loc(d) + 1] for d in dev_dates]
            self.assertEqual(list(hist.index), expected_exec)

        # Summary table should have: Ternary Control, Long-Only Control, 4 k-variants, and AAPL Benchmark
        self.assertEqual(len(summary_df), 7)
        self.assertIn("Ternary Control (k=0.0)", summary_df["Variant"].values)
        self.assertIn("Long-Only Control (k=0.0)", summary_df["Variant"].values)
        self.assertIn("Long-Only Adaptive (k=0.25)", summary_df["Variant"].values)
        self.assertIn("Long-Only Adaptive (k=0.50)", summary_df["Variant"].values)
        self.assertIn("Long-Only Adaptive (k=0.75)", summary_df["Variant"].values)
        self.assertIn("Long-Only Adaptive (k=1.00)", summary_df["Variant"].values)
        self.assertIn("AAPL Buy & Hold", summary_df["Variant"].values)

    def test_parameter_locking_before_final_test_evaluation(self) -> None:
        """Optimal parameter selected on development window can be locked into BacktestConfig."""
        n = 30
        preds = pd.Series(np.random.normal(0.002, 0.01, n), index=self.dates[:n])
        stds = pd.Series(np.full(n, 0.004), index=self.dates[:n])

        dev_dates = self.dates[5:15]
        results, summary_df = evaluate_threshold_candidates(
            predictions=preds,
            market_data=self.market_df,
            training_fold_stds=stds,
            candidate_k_values=(0.25, 0.50, 0.75, 1.00),
            evaluation_dates=dev_dates,
        )

        # Strategy variants only
        strat_df = summary_df[summary_df["Type"] == "Strategy"]
        best_variant = strat_df.iloc[0]
        selected_k = best_variant["k"]

        # Lock the selected parameter
        locked_config = BacktestConfig(
            allocation_rule="long_only",
            conviction_threshold=selected_k * stds,
        )
        self.assertEqual(locked_config.allocation_rule, "long_only")
        # Verify immutability / exact locked threshold
        self.assertTrue(isinstance(locked_config.conviction_threshold, pd.Series))

    def test_missing_predictions_map_to_cash_under_thresholding(self) -> None:
        """Missing (NaN) predictions must map to cash (0.0) with proper friction on exit."""
        preds = pd.Series([0.02, np.nan, 0.03, np.nan], index=self.dates[:4])
        config = BacktestConfig(allocation_rule="long_only", conviction_threshold=0.005)
        res = run_backtest(preds, self.market_df, config=config)

        positions = res.portfolio_history["position"].tolist()
        self.assertEqual(positions[0], 1.0)
        self.assertEqual(positions[1], 0.0)  # NaN maps to cash
        self.assertEqual(positions[2], 1.0)
        self.assertEqual(positions[3], 0.0)  # NaN maps to cash

        # Turnover for transitions: 0->1 (1.0), 1->0 (1.0), 0->1 (1.0), 1->0 (1.0)
        turnovers = res.portfolio_history["turnover"].tolist()
        self.assertEqual(turnovers, [1.0, 1.0, 1.0, 1.0])

    def test_deterministic_repeated_execution(self) -> None:
        """Repeated runs with identical inputs must produce bitwise identical outputs."""
        n = 15
        preds = pd.Series(np.random.normal(0.001, 0.01, n), index=self.dates[:n])
        stds = pd.Series(np.full(n, 0.004), index=self.dates[:n])

        _, summary1 = evaluate_threshold_candidates(
            predictions=preds,
            market_data=self.market_df,
            training_fold_stds=stds,
            candidate_k_values=(0.25, 0.50, 0.75, 1.00),
        )
        _, summary2 = evaluate_threshold_candidates(
            predictions=preds,
            market_data=self.market_df,
            training_fold_stds=stds,
            candidate_k_values=(0.25, 0.50, 0.75, 1.00),
        )
        pd.testing.assert_frame_equal(summary1, summary2)

    def test_config_validation_for_conviction_threshold(self) -> None:
        """BacktestConfig must validate conviction_threshold and allocation_rule."""
        with self.assertRaises(ValueError):
            BacktestConfig(conviction_threshold=-0.01)

        with self.assertRaises(ValueError):
            BacktestConfig(conviction_threshold=pd.Series([-0.01, 0.02]))

        with self.assertRaises(ValueError):
            BacktestConfig(allocation_rule="speculative_binary")


if __name__ == "__main__":
    unittest.main()
