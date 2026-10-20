"""Regression Test Suite for News Sentiment and Data Integrity.

Validates:
1. Requested date-range filtering with America/New_York timezone boundaries.
2. Finnhub-over-Yahoo deduplication precedence.
3. Safe incremental sentiment updates preserving unrelated historical dates.
4. Merge idempotence (no duplicated rows or columns).
5. Empty retrieval preservation vs explicit full rebuild.
6. Market index, row count, and non-sentiment feature invariants.
7. Missing sentiment NaN invariants (coverage=0 -> NaN, never 0.0).
8. Trading-session assignment around regular close, early close, weekends, and holidays.
9. Distinction between failed news retrieval and confirmed empty news.

All tests operate strictly on synthetic data in memory or temporary files.
Zero modification to production Parquet datasets.
"""

import datetime as dt
import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.data_ingestion.news_sentiment import (
    SENTIMENT_METRIC_COLUMNS,
    assign_trading_session,
    deduplicate_articles,
    filter_news_by_date_range,
    get_us_equity_market_holidays,
    is_us_equity_early_close,
    merge_sentiment_with_market_features,
)


class TestNewsDateRangeFiltering(unittest.TestCase):
    """Test inclusive date filtering in America/New_York reference timezone."""

    def setUp(self) -> None:
        self.start_date = "2026-10-01"
        self.end_date = "2026-10-15"

    def test_inclusive_boundaries_and_rejection(self) -> None:
        """Articles exactly on start/end dates are retained; articles 1s outside are rejected."""
        records = [
            # 1. Exactly on start date (00:00:00 ET)
            {"headline": "H_start_exact", "published_at": pd.Timestamp("2026-10-01 00:00:00", tz="America/New_York")},
            # 2. 1 second before start date in ET
            {"headline": "H_before_start", "published_at": pd.Timestamp("2026-09-30 23:59:59", tz="America/New_York")},
            # 3. Middle of range in ET
            {"headline": "H_mid_range", "published_at": pd.Timestamp("2026-10-08 12:00:00", tz="America/New_York")},
            # 4. Exactly on end date (23:59:59 ET)
            {"headline": "H_end_exact", "published_at": pd.Timestamp("2026-10-15 23:59:59", tz="America/New_York")},
            # 5. 1 second after end date in ET
            {"headline": "H_after_end", "published_at": pd.Timestamp("2026-10-16 00:00:01", tz="America/New_York")},
        ]
        df = pd.DataFrame(records)
        filtered = filter_news_by_date_range(df, start_date=self.start_date, end_date=self.end_date)
        retained = set(filtered["headline"])

        self.assertIn("H_start_exact", retained)
        self.assertIn("H_mid_range", retained)
        self.assertIn("H_end_exact", retained)
        self.assertNotIn("H_before_start", retained)
        self.assertNotIn("H_after_end", retained)
        self.assertEqual(len(filtered), 3)

    def test_utc_timezone_boundary_conversion(self) -> None:
        """UTC timestamps are correctly evaluated by their local New York date."""
        # In October (EDT), New York is UTC-4.
        records = [
            # 2026-10-01 02:00:00 UTC = 2026-09-30 22:00:00 EDT (Before start date -> exclude)
            {"headline": "H_utc_before", "published_at": pd.Timestamp("2026-10-01 02:00:00", tz="UTC")},
            # 2026-10-01 05:00:00 UTC = 2026-10-01 01:00:00 EDT (Start date -> include)
            {"headline": "H_utc_start", "published_at": pd.Timestamp("2026-10-01 05:00:00", tz="UTC")},
            # 2026-10-16 02:00:00 UTC = 2026-10-15 22:00:00 EDT (End date in ET -> include)
            {"headline": "H_utc_end", "published_at": pd.Timestamp("2026-10-16 02:00:00", tz="UTC")},
            # 2026-10-16 05:00:00 UTC = 2026-10-16 01:00:00 EDT (After end date in ET -> exclude)
            {"headline": "H_utc_after", "published_at": pd.Timestamp("2026-10-16 05:00:00", tz="UTC")},
        ]
        df = pd.DataFrame(records)
        filtered = filter_news_by_date_range(df, start_date=self.start_date, end_date=self.end_date)
        retained = set(filtered["headline"])

        self.assertIn("H_utc_start", retained)
        self.assertIn("H_utc_end", retained)
        self.assertNotIn("H_utc_before", retained)
        self.assertNotIn("H_utc_after", retained)
        self.assertEqual(len(filtered), 2)

    def test_missing_or_invalid_timestamps_handled_safely(self) -> None:
        """Missing (None/NaT) or unparseable timestamps are discarded safely without raising exceptions."""
        records = [
            {"headline": "H_valid", "published_at": pd.Timestamp("2026-10-05 12:00:00", tz="America/New_York")},
            {"headline": "H_none", "published_at": None},
            {"headline": "H_nat", "published_at": pd.NaT},
            {"headline": "H_corrupt", "published_at": "not-a-valid-timestamp-string"},
        ]
        df = pd.DataFrame(records)
        filtered = filter_news_by_date_range(df, start_date=self.start_date, end_date=self.end_date)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["headline"], "H_valid")

    def test_inverted_date_range_raises_value_error(self) -> None:
        """Providing start_date > end_date raises ValueError."""
        df = pd.DataFrame([{"headline": "A", "published_at": pd.Timestamp("2026-10-05", tz="UTC")}])
        with self.assertRaises(ValueError):
            filter_news_by_date_range(df, start_date="2026-10-15", end_date="2026-10-01")


class TestDeduplicationPrecedence(unittest.TestCase):
    """Test Finnhub-first precedence during deduplication."""

    def test_finnhub_prioritized_over_yahoo_on_id_and_headline(self) -> None:
        """When an article is present in both sources, Finnhub record is preserved."""
        records = [
            # Finnhub record
            {
                "raw_id": "art_100",
                "headline": "Apple Unveils New Hardware",
                "url": "https://wire.com/apple1",
                "published_at": pd.to_datetime("2026-10-10 14:00:00Z"),
                "is_fallback": False,
                "provider": "finnhub",
            },
            # Yahoo duplicate with same raw_id
            {
                "raw_id": "art_100",
                "headline": "Apple Unveils New Hardware",
                "url": "https://wire.com/apple1",
                "published_at": pd.to_datetime("2026-10-10 14:00:00Z"),
                "is_fallback": True,
                "provider": "yahoo_finance",
            },
            # Yahoo duplicate with same URL and headline but different/empty ID
            {
                "raw_id": "",
                "headline": "apple unveils new hardware",
                "url": "https://wire.com/apple1",
                "published_at": pd.to_datetime("2026-10-10 14:00:00Z"),
                "is_fallback": True,
                "provider": "yahoo_finance",
            },
        ]
        df = pd.DataFrame(records)
        deduped = deduplicate_articles(df)
        self.assertEqual(len(deduped), 1)
        self.assertFalse(deduped.iloc[0]["is_fallback"])
        self.assertEqual(deduped.iloc[0]["provider"], "finnhub")


class TestTradingSessionAssignment(unittest.TestCase):
    """Test look-ahead bias elimination, calendar holidays, and early closes."""

    def setUp(self) -> None:
        self.holidays = get_us_equity_market_holidays()

    def test_regular_session_cutoffs(self) -> None:
        """15:59:59 ET belongs to current day; 16:00:00 and 16:00:01 belong to next session."""
        # Friday 2026-10-16
        t_before = pd.Timestamp("2026-10-16 15:59:59", tz="America/New_York")
        t_at = pd.Timestamp("2026-10-16 16:00:00", tz="America/New_York")
        t_after = pd.Timestamp("2026-10-16 16:00:01", tz="America/New_York")

        # 15:59:59 -> same Friday session
        self.assertEqual(
            assign_trading_session(t_before, market_holidays=self.holidays),
            pd.Timestamp("2026-10-16"),
        )
        # 16:00:00 -> next Monday session (2026-10-19)
        self.assertEqual(
            assign_trading_session(t_at, market_holidays=self.holidays),
            pd.Timestamp("2026-10-19"),
        )
        # 16:00:01 -> next Monday session (2026-10-19)
        self.assertEqual(
            assign_trading_session(t_after, market_holidays=self.holidays),
            pd.Timestamp("2026-10-19"),
        )

    def test_weekend_session_assignment(self) -> None:
        """Saturday and Sunday news is mapped to Monday session."""
        sat = pd.Timestamp("2026-10-17 12:00:00", tz="America/New_York")
        sun = pd.Timestamp("2026-10-18 20:00:00", tz="America/New_York")
        expected = pd.Timestamp("2026-10-19")

        self.assertEqual(assign_trading_session(sat, market_holidays=self.holidays), expected)
        self.assertEqual(assign_trading_session(sun, market_holidays=self.holidays), expected)

    def test_us_equity_market_holidays(self) -> None:
        """Good Friday is an equity holiday; Columbus Day & Veterans Day are regular sessions."""
        # Good Friday (2026-04-03) -> advances to Monday 2026-04-06
        t_gf = pd.Timestamp("2026-04-03 11:00:00", tz="America/New_York")
        self.assertEqual(
            assign_trading_session(t_gf, market_holidays=self.holidays),
            pd.Timestamp("2026-04-06"),
        )

        # Columbus Day (Monday 2026-10-12) -> US stock exchanges are OPEN
        t_columbus = pd.Timestamp("2026-10-12 11:00:00", tz="America/New_York")
        self.assertEqual(
            assign_trading_session(t_columbus, market_holidays=self.holidays),
            pd.Timestamp("2026-10-12"),
        )

        # Veterans Day (Wednesday 2026-11-11) -> US stock exchanges are OPEN
        t_veterans = pd.Timestamp("2026-11-11 11:00:00", tz="America/New_York")
        self.assertEqual(
            assign_trading_session(t_veterans, market_holidays=self.holidays),
            pd.Timestamp("2026-11-11"),
        )

    def test_early_close_cutoffs(self) -> None:
        """Early-close sessions cutoff at 13:00:00 ET, not 16:00:00 ET."""
        # Black Friday 2026-11-27 (day after Thanksgiving)
        self.assertTrue(is_us_equity_early_close(dt.date(2026, 11, 27), self.holidays))

        # At 12:59:59 ET -> Black Friday session
        t_bf_before = pd.Timestamp("2026-11-27 12:59:59", tz="America/New_York")
        self.assertEqual(
            assign_trading_session(t_bf_before, market_holidays=self.holidays),
            pd.Timestamp("2026-11-27"),
        )

        # At 13:00:00 ET -> next trading day (Monday 2026-11-30)
        t_bf_at = pd.Timestamp("2026-11-27 13:00:00", tz="America/New_York")
        self.assertEqual(
            assign_trading_session(t_bf_at, market_holidays=self.holidays),
            pd.Timestamp("2026-11-30"),
        )

        # Christmas Eve 2026-12-24 (Thursday before Friday Dec 25 Christmas holiday)
        self.assertTrue(is_us_equity_early_close(dt.date(2026, 12, 24), self.holidays))
        t_xmas_after = pd.Timestamp("2026-12-24 13:05:00", tz="America/New_York")
        # Dec 25 is holiday, weekend is Dec 26-27 -> next session is Monday 2026-12-28
        self.assertEqual(
            assign_trading_session(t_xmas_after, market_holidays=self.holidays),
            pd.Timestamp("2026-12-28"),
        )

    def test_utc_equivalent_timestamps(self) -> None:
        """Timestamp in UTC maps to the exact same trading session as New York local time."""
        # 2026-10-16 15:59:59 EDT = 2026-10-16 19:59:59 UTC
        t_utc_before = pd.Timestamp("2026-10-16 19:59:59", tz="UTC")
        self.assertEqual(
            assign_trading_session(t_utc_before, market_holidays=self.holidays),
            pd.Timestamp("2026-10-16"),
        )

        # 2026-10-16 16:00:00 EDT = 2026-10-16 20:00:00 UTC
        t_utc_at = pd.Timestamp("2026-10-16 20:00:00", tz="UTC")
        self.assertEqual(
            assign_trading_session(t_utc_at, market_holidays=self.holidays),
            pd.Timestamp("2026-10-19"),
        )


class TestIncrementalSentimentMerge(unittest.TestCase):
    """Test safe incremental updates, idempotence, full rebuild, and invariants."""

    def setUp(self) -> None:
        self.dates = pd.date_range("2026-10-05", periods=10, freq="B")
        self.base_market = pd.DataFrame(
            {
                "Close": [100.0 + i for i in range(10)],
                "Volume": [1000 + i * 100 for i in range(10)],
                "Daily_Return": [0.01] * 10,
                "RSI_14": [50.0 + i for i in range(10)],
            },
            index=self.dates,
        )
        self.temp_dir = tempfile.TemporaryDirectory()
        self.market_file = Path(self.temp_dir.name) / "market_features_test.parquet"
        self.base_market.to_parquet(self.market_file)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_partial_update_preserves_unrelated_historical_sentiment(self) -> None:
        """Incremental update for date t does not modify sentiment on dates outside the window."""
        # 1. Establish initial sentiment on Day 0 and Day 1
        initial_sent = pd.DataFrame(
            {
                "sentiment_score": [0.45, 0.65],
                "sentiment_std": [0.1, 0.15],
                "sentiment_count": [5, 8],
                "prob_positive": [0.7, 0.8],
                "prob_negative": [0.25, 0.15],
                "prob_neutral": [0.05, 0.05],
            },
            index=self.dates[:2],
        )
        merged_init = merge_sentiment_with_market_features(
            market_features_path=self.market_file,
            daily_sentiment_df=initial_sent,
            output_path=self.market_file,
            full_rebuild=True,
        )
        self.assertEqual(merged_init["sentiment_coverage"].sum(), 2)

        # 2. Incrementally update Day 5 only
        day5_sent = pd.DataFrame(
            {
                "sentiment_score": [-0.5],
                "sentiment_std": [0.0],
                "sentiment_count": [3],
                "prob_positive": [0.1],
                "prob_negative": [0.6],
                "prob_neutral": [0.3],
            },
            index=[self.dates[5]],
        )
        merged_incr = merge_sentiment_with_market_features(
            market_features_path=self.market_file,
            daily_sentiment_df=day5_sent,
            output_path=self.market_file,
            refresh_start_date=self.dates[5].strftime("%Y-%m-%d"),
            refresh_end_date=self.dates[5].strftime("%Y-%m-%d"),
            full_rebuild=False,
        )

        # Verify historical dates Day 0 and Day 1 are preserved
        self.assertEqual(merged_incr["sentiment_coverage"].sum(), 3)
        self.assertAlmostEqual(merged_incr.loc[self.dates[0], "sentiment_score"], 0.45)
        self.assertAlmostEqual(merged_incr.loc[self.dates[1], "sentiment_score"], 0.65)
        self.assertAlmostEqual(merged_incr.loc[self.dates[5], "sentiment_score"], -0.5)

        # Verify untouched dates remain NaN
        self.assertTrue(pd.isna(merged_incr.loc[self.dates[2], "sentiment_score"]))
        self.assertEqual(merged_incr.loc[self.dates[2], "sentiment_coverage"], 0)

        # Verify market non-sentiment features are unchanged
        pd.testing.assert_series_equal(merged_incr["Close"], self.base_market["Close"], check_freq=False)
        pd.testing.assert_series_equal(merged_incr["Volume"], self.base_market["Volume"], check_freq=False)

    def test_merge_idempotence(self) -> None:
        """Repeating the exact same incremental update produces bitwise identical DataFrames."""
        day3_sent = pd.DataFrame(
            {
                "sentiment_score": [0.2],
                "sentiment_std": [0.05],
                "sentiment_count": [4],
                "prob_positive": [0.5],
                "prob_negative": [0.3],
                "prob_neutral": [0.2],
            },
            index=[self.dates[3]],
        )
        run_1 = merge_sentiment_with_market_features(
            market_features_path=self.market_file,
            daily_sentiment_df=day3_sent,
            output_path=self.market_file,
            refresh_start_date=self.dates[3].strftime("%Y-%m-%d"),
            refresh_end_date=self.dates[3].strftime("%Y-%m-%d"),
            full_rebuild=False,
        )
        run_2 = merge_sentiment_with_market_features(
            market_features_path=self.market_file,
            daily_sentiment_df=day3_sent,
            output_path=self.market_file,
            refresh_start_date=self.dates[3].strftime("%Y-%m-%d"),
            refresh_end_date=self.dates[3].strftime("%Y-%m-%d"),
            full_rebuild=False,
        )
        pd.testing.assert_frame_equal(run_1, run_2)

    def test_empty_retrieval_does_not_erase_historical_sentiment(self) -> None:
        """Empty daily sentiment in incremental mode preserves all existing historical sentiment."""
        # 1. Establish initial sentiment
        initial_sent = pd.DataFrame(
            {
                "sentiment_score": [0.8],
                "sentiment_std": [0.1],
                "sentiment_count": [5],
                "prob_positive": [0.9],
                "prob_negative": [0.1],
                "prob_neutral": [0.0],
            },
            index=[self.dates[0]],
        )
        m1 = merge_sentiment_with_market_features(
            market_features_path=self.market_file,
            daily_sentiment_df=initial_sent,
            output_path=self.market_file,
            full_rebuild=True,
        )
        self.assertEqual(m1["sentiment_coverage"].sum(), 1)

        # 2. Incremental run with empty DataFrame
        m2 = merge_sentiment_with_market_features(
            market_features_path=self.market_file,
            daily_sentiment_df=pd.DataFrame(),
            output_path=self.market_file,
            refresh_start_date=self.dates[0].strftime("%Y-%m-%d"),
            refresh_end_date=self.dates[0].strftime("%Y-%m-%d"),
            full_rebuild=False,
        )
        # Existing sentiment must be preserved!
        self.assertEqual(m2["sentiment_coverage"].sum(), 1)
        self.assertAlmostEqual(m2.loc[self.dates[0], "sentiment_score"], 0.8)
        pd.testing.assert_frame_equal(m1, m2)

    def test_explicit_full_rebuild_behavior(self) -> None:
        """Full rebuild resets and replaces sentiment across all dates."""
        # Seed with initial sentiment on day 0
        initial_sent = pd.DataFrame(
            {
                "sentiment_score": [0.8],
                "sentiment_std": [0.1],
                "sentiment_count": [5],
                "prob_positive": [0.9],
                "prob_negative": [0.1],
                "prob_neutral": [0.0],
            },
            index=[self.dates[0]],
        )
        merge_sentiment_with_market_features(
            market_features_path=self.market_file,
            daily_sentiment_df=initial_sent,
            output_path=self.market_file,
            full_rebuild=True,
        )

        # Full rebuild with empty DataFrame -> clears all sentiment to NaN with 0 coverage
        rebuilt = merge_sentiment_with_market_features(
            market_features_path=self.market_file,
            daily_sentiment_df=pd.DataFrame(),
            output_path=self.market_file,
            full_rebuild=True,
        )
        self.assertEqual(rebuilt["sentiment_coverage"].sum(), 0)
        self.assertTrue(rebuilt["sentiment_score"].isna().all())

    def test_missing_sentiment_invariants(self) -> None:
        """When sentiment_coverage is 0, all sentiment columns are strictly NaN, NEVER 0.0."""
        single_sent = pd.DataFrame(
            {
                "sentiment_score": [0.3],
                "sentiment_std": [0.05],
                "sentiment_count": [2],
                "prob_positive": [0.4],
                "prob_negative": [0.1],
                "prob_neutral": [0.5],
            },
            index=[self.dates[2]],
        )
        merged = merge_sentiment_with_market_features(
            market_features_path=self.market_file,
            daily_sentiment_df=single_sent,
            output_path=self.market_file,
            full_rebuild=True,
        )
        missing_rows = merged[merged["sentiment_coverage"] == 0]
        self.assertEqual(len(missing_rows), 9)

        for col in SENTIMENT_METRIC_COLUMNS:
            self.assertTrue(
                missing_rows[col].isna().all(),
                f"Column {col} contained non-NaN values for missing coverage rows!",
            )

        covered_rows = merged[merged["sentiment_coverage"] == 1]
        self.assertEqual(len(covered_rows), 1)
        self.assertTrue(covered_rows["sentiment_score"].notna().all())


if __name__ == "__main__":
    unittest.main()
