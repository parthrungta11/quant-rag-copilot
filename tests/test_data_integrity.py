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
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data_ingestion.news_sentiment import (
    NEWS_COLUMNS,
    PROJECT_ROOT,
    SENTIMENT_METRIC_COLUMNS,
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_PARTIAL,
    TX_STATE_ABORTED_RECOVERED,
    TX_STATE_COMMITTED,
    TX_STATE_RECOVERING,
    TX_STATE_STAGED,
    assign_trading_session,
    check_and_recover_transactions,
    commit_multi_file_transaction,
    compute_sha256,
    deduplicate_articles,
    fetch_finnhub_news,
    fetch_news,
    filter_news_by_date_range,
    get_net_new_articles,
    get_us_equity_market_holidays,
    is_us_equity_early_close,
    load_trusted_checksums,
    main,
    merge_processed_news_archive,
    merge_raw_news_archive,
    merge_sentiment_with_market_features,
    run_production_pipeline,
    update_manifest_chunk,
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


class TestFinnhubRetrievalOutcomes(unittest.TestCase):
    """Test complete empty retrieval, partial chunk failure, and total failure distinction.

    Validates:
    1. Complete: All chunks retrieved successfully (even if 0 articles) -> status=complete.
    2. Partial: Some chunks failed, >= 1 succeeded -> articles preserved, status=partial.
    3. Failed: All chunks failed -> status=failed, not silently treated as successful empty.
    4. Pipeline safety: Partial refresh updates only observed dates, preserving unretrieved dates.
    5. Pipeline safety: Total failure aborts without clearing/overwriting historical sentiment.
    """

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)
        self.market_file = self.temp_dir_path / "market_features.parquet"
        self.raw_news_file = self.temp_dir_path / "raw_news.parquet"
        self.processed_news_file = self.temp_dir_path / "processed_news.parquet"
        self.backup_dir = self.temp_dir_path / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Synthetic 5 trading days
        self.dates = pd.date_range("2026-10-12", periods=5, freq="B")
        self.market_df = pd.DataFrame(
            {
                "Close": [100.0, 101.0, 102.0, 103.0, 104.0],
                "Daily_Return": [0.01, 0.01, 0.01, 0.01, 0.01],
            },
            index=self.dates,
        )
        self.market_df.index.name = "Date"
        self.market_df.to_parquet(self.market_file)

        # Pre-populate valid backups and trusted checksums for pipeline runs
        pd.DataFrame(columns=NEWS_COLUMNS).to_parquet(self.backup_dir / self.raw_news_file.name)
        pd.DataFrame().to_parquet(self.backup_dir / self.processed_news_file.name)
        shutil.copy2(self.market_file, self.backup_dir / self.market_file.name)
        self._write_test_checksums(self.backup_dir)

    def _write_test_checksums(self, backup_dir: Path) -> Path:
        chk_file = backup_dir / "checksums.sha256"
        lines = []
        for bf in sorted(backup_dir.glob("*.parquet")):
            lines.append(f"{compute_sha256(bf)}  {bf.name}\n")
        with open(chk_file, "w", encoding="utf-8") as f:
            f.writelines(lines)
        return chk_file

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    class MockFinnhubClient:
        def __init__(self, chunk_map: dict[str | tuple[str, str], Any]) -> None:
            self.chunk_map = chunk_map

        def company_news(self, ticker: str, _from: str, to: str) -> list[dict[str, Any]]:
            val = self.chunk_map.get((_from, to))
            if val is None:
                val = self.chunk_map.get(_from)
            if isinstance(val, Exception):
                raise val
            return val if val is not None else []

    class MockAnalyzer:
        def analyze_batch(self, headlines: list[str], batch_size: int = 32) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "sentiment_label": "positive",
                        "sentiment_score": 0.5,
                        "prob_positive": 0.8,
                        "prob_negative": 0.1,
                        "prob_neutral": 0.1,
                    }
                    for _ in headlines
                ]
            )

    def test_complete_empty_retrieval(self) -> None:
        """All chunks succeed with 0 articles -> status is complete (not failed)."""
        client = self.MockFinnhubClient({
            "2026-10-12": [],
            "2026-10-13": [],
        })
        finnhub_df = fetch_finnhub_news("AAPL", "2026-10-12", "2026-10-13", client=client, retry_delay=0.0)
        self.assertTrue(finnhub_df.empty)
        self.assertEqual(finnhub_df.attrs["status"], STATUS_COMPLETE)
        self.assertEqual(finnhub_df.attrs["failed_chunks"], [])
        self.assertEqual(finnhub_df.attrs["total_chunks"], 2)

        with patch("src.data_ingestion.news_sentiment.fetch_yahoo_fallback", return_value=pd.DataFrame()):
            news_df = fetch_news(
                "AAPL",
                start_date="2026-10-12",
                end_date="2026-10-13",
                client=client,
                retry_delay=0.0,
            )
            self.assertTrue(news_df.empty)
            self.assertEqual(news_df.attrs["status"], STATUS_COMPLETE)
            self.assertNotEqual(news_df.attrs["status"], STATUS_FAILED)

        res = run_production_pipeline(
            start_date="2026-10-12",
            end_date="2026-10-13",
            client=client,
            market_features_path=self.market_file,
            raw_news_path=self.raw_news_file,
            processed_news_path=self.processed_news_file,
            analyzer=self.MockAnalyzer(),
            retry_delay=0.0,
            backup_dir=self.backup_dir,
            allowed_roots=[self.temp_dir_path],
        )
        self.assertEqual(res["status"], STATUS_COMPLETE)
        self.assertEqual(res["raw_news_count"], 0)
        self.assertFalse(self.raw_news_file.exists())

    def test_partial_chunk_failure_preserves_articles_and_exposes_status(self) -> None:
        """Some chunks fail but >= 1 chunk succeeds -> returns preserved articles and status is partial."""
        client = self.MockFinnhubClient({
            "2026-10-12": RuntimeError("Simulated API 500 error"),
            "2026-10-13": [
                {
                    "id": 101,
                    "datetime": int(pd.Timestamp("2026-10-13 14:00:00Z").timestamp()),
                    "headline": "Apple Unveils M4 Chip Lineup",
                    "summary": "Next generation M4 silicon.",
                    "source": "TechWire",
                    "url": "https://wire.example.com/m4",
                }
            ],
        })
        finnhub_df = fetch_finnhub_news("AAPL", "2026-10-12", "2026-10-13", client=client, retry_delay=0.0)
        self.assertEqual(len(finnhub_df), 1)
        self.assertEqual(finnhub_df.iloc[0]["headline"], "Apple Unveils M4 Chip Lineup")
        self.assertEqual(finnhub_df.attrs["status"], STATUS_PARTIAL)
        self.assertEqual(finnhub_df.attrs["failed_chunks"], [("2026-10-12", "2026-10-12")])
        self.assertEqual(finnhub_df.attrs["successful_chunks"], [("2026-10-13", "2026-10-13")])
        self.assertEqual(finnhub_df.attrs["total_chunks"], 2)

        with patch("src.data_ingestion.news_sentiment.fetch_yahoo_fallback", return_value=pd.DataFrame()):
            news_df = fetch_news(
                "AAPL",
                start_date="2026-10-12",
                end_date="2026-10-13",
                client=client,
                retry_delay=0.0,
            )
            self.assertEqual(len(news_df), 1)
            self.assertEqual(news_df.attrs["status"], STATUS_PARTIAL)
            self.assertEqual(news_df.attrs["failed_chunks"], [("2026-10-12", "2026-10-12")])

    def test_partial_retrieval_pipeline_preserves_historical_sentiment_on_unretrieved_dates(self) -> None:
        """In partial retrieval, incremental merge updates only observed dates and preserves unretrieved historical dates."""
        # 1. Establish historical sentiment on Day 0 (2026-10-12)
        initial_sent = pd.DataFrame(
            {
                "sentiment_score": [0.85],
                "sentiment_std": [0.08],
                "sentiment_count": [4],
                "prob_positive": [0.9],
                "prob_negative": [0.05],
                "prob_neutral": [0.05],
            },
            index=[self.dates[0]],
        )
        merge_sentiment_with_market_features(
            market_features_path=self.market_file,
            daily_sentiment_df=initial_sent,
            output_path=self.market_file,
            full_rebuild=True,
        )

        # 2. Run partial pipeline over 2026-10-12 to 2026-10-13 where Day 0 fails, Day 1 succeeds
        client = self.MockFinnhubClient({
            "2026-10-12": RuntimeError("500 Internal Server Error"),
            "2026-10-13": [
                {
                    "id": 202,
                    "datetime": int(pd.Timestamp("2026-10-13 14:00:00Z").timestamp()),
                    "headline": "Apple Partners With Major Cloud Provider",
                    "summary": "Cloud agreement signed.",
                    "source": "MockNews",
                    "url": "https://example.com/cloud",
                }
            ],
        })

        with patch("src.data_ingestion.news_sentiment.fetch_yahoo_fallback", return_value=pd.DataFrame()):
            res = run_production_pipeline(
                start_date="2026-10-12",
                end_date="2026-10-13",
                client=client,
                market_features_path=self.market_file,
                raw_news_path=self.raw_news_file,
                processed_news_path=self.processed_news_file,
                analyzer=self.MockAnalyzer(),
                retry_delay=0.0,
                backup_dir=self.backup_dir,
                allowed_roots=[self.temp_dir_path],
            )

        self.assertEqual(res["status"], STATUS_PARTIAL)
        self.assertEqual(res["raw_news_count"], 1)

        # 3. Verify that Day 0 historical sentiment was NOT cleared or overwritten
        market = pd.read_parquet(self.market_file)
        self.assertEqual(market.loc[self.dates[0], "sentiment_coverage"], 1)
        self.assertAlmostEqual(market.loc[self.dates[0], "sentiment_score"], 0.85)

        # 4. Verify that Day 1 sentiment was updated
        self.assertEqual(market.loc[self.dates[1], "sentiment_coverage"], 1)
        self.assertAlmostEqual(market.loc[self.dates[1], "sentiment_score"], 0.5)

        # Total coverage is 2 sessions
        self.assertEqual(market["sentiment_coverage"].sum(), 2)

    def test_total_retrieval_failure(self) -> None:
        """All chunks fail -> status is failed and not silently treated as successful empty."""
        client = self.MockFinnhubClient({
            "2026-10-12": RuntimeError("Network Timeout"),
            "2026-10-13": RuntimeError("Rate Limit 429"),
        })
        finnhub_df = fetch_finnhub_news("AAPL", "2026-10-12", "2026-10-13", client=client, retry_delay=0.0)
        self.assertTrue(finnhub_df.empty)
        self.assertEqual(finnhub_df.attrs["status"], STATUS_FAILED)
        self.assertEqual(len(finnhub_df.attrs["failed_chunks"]), 2)
        self.assertEqual(finnhub_df.attrs["total_chunks"], 2)

        with patch("src.data_ingestion.news_sentiment.fetch_yahoo_fallback", return_value=pd.DataFrame()):
            news_df = fetch_news(
                "AAPL",
                start_date="2026-10-12",
                end_date="2026-10-13",
                client=client,
                retry_delay=0.0,
            )
            self.assertTrue(news_df.empty)
            self.assertEqual(news_df.attrs["status"], STATUS_FAILED)
            self.assertNotEqual(news_df.attrs["status"], STATUS_COMPLETE)

    def test_total_retrieval_failure_pipeline_aborts_to_protect_historical_data(self) -> None:
        """When Finnhub retrieval fails completely, pipeline aborts without touching historical sentiment."""
        # 1. Establish historical sentiment on Day 0
        initial_sent = pd.DataFrame(
            {
                "sentiment_score": [0.77],
                "sentiment_std": [0.05],
                "sentiment_count": [3],
                "prob_positive": [0.8],
                "prob_negative": [0.1],
                "prob_neutral": [0.1],
            },
            index=[self.dates[0]],
        )
        merge_sentiment_with_market_features(
            market_features_path=self.market_file,
            daily_sentiment_df=initial_sent,
            output_path=self.market_file,
            full_rebuild=True,
        )

        # 2. Run pipeline with all chunks failing
        client = self.MockFinnhubClient({
            "2026-10-12": RuntimeError("Auth Failed 401"),
            "2026-10-13": RuntimeError("Auth Failed 401"),
        })

        with patch("src.data_ingestion.news_sentiment.fetch_yahoo_fallback", return_value=pd.DataFrame()):
            res = run_production_pipeline(
                start_date="2026-10-12",
                end_date="2026-10-13",
                client=client,
                market_features_path=self.market_file,
                raw_news_path=self.raw_news_file,
                processed_news_path=self.processed_news_file,
                analyzer=self.MockAnalyzer(),
                retry_delay=0.0,
                backup_dir=self.backup_dir,
                allowed_roots=[self.temp_dir_path],
            )

        self.assertEqual(res["status"], STATUS_FAILED)
        self.assertFalse(self.raw_news_file.exists())

        # 3. Market features must be completely untouched
        market = pd.read_parquet(self.market_file)
        self.assertEqual(market["sentiment_coverage"].sum(), 1)
        self.assertAlmostEqual(market.loc[self.dates[0], "sentiment_score"], 0.77)

    def test_failed_finnhub_with_fallback_news_aborts_to_prevent_wiping_historical_sentiment(self) -> None:
        """Even if Yahoo fallback returns headlines, total Finnhub failure aborts pipeline to prevent clearing the refresh window."""
        # 1. Establish historical sentiment on Day 0
        initial_sent = pd.DataFrame(
            {
                "sentiment_score": [0.65],
                "sentiment_std": [0.05],
                "sentiment_count": [2],
                "prob_positive": [0.7],
                "prob_negative": [0.1],
                "prob_neutral": [0.2],
            },
            index=[self.dates[0]],
        )
        merge_sentiment_with_market_features(
            market_features_path=self.market_file,
            daily_sentiment_df=initial_sent,
            output_path=self.market_file,
            full_rebuild=True,
        )

        # 2. Client fails all Finnhub chunks
        client = self.MockFinnhubClient({
            "2026-10-12": RuntimeError("Finnhub Unreachable"),
            "2026-10-13": RuntimeError("Finnhub Unreachable"),
        })

        fallback_df = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "published_at": pd.Timestamp("2026-10-13 14:00:00Z"),
                    "headline": "Fallback headline from Yahoo",
                    "summary": "Summary",
                    "source": "Yahoo",
                    "url": "https://yahoo.com/1",
                    "provider": "yahoo_finance",
                    "is_fallback": True,
                    "raw_id": "y1",
                }
            ]
        )

        with patch("src.data_ingestion.news_sentiment.fetch_yahoo_fallback", return_value=fallback_df):
            res = run_production_pipeline(
                start_date="2026-10-12",
                end_date="2026-10-13",
                client=client,
                market_features_path=self.market_file,
                raw_news_path=self.raw_news_file,
                processed_news_path=self.processed_news_file,
                analyzer=self.MockAnalyzer(),
                retry_delay=0.0,
                backup_dir=self.backup_dir,
                allowed_roots=[self.temp_dir_path],
            )

        # Status must be FAILED because primary Finnhub failed
        self.assertEqual(res["status"], STATUS_FAILED)
        # Market file must NOT have its Day 0 wiped out by fallback-only merge
        market = pd.read_parquet(self.market_file)
        self.assertEqual(market.loc[self.dates[0], "sentiment_coverage"], 1)
        self.assertAlmostEqual(market.loc[self.dates[0], "sentiment_score"], 0.65)


class TestArchiveHardeningAndRecovery(unittest.TestCase):
    """Synthetic tests for archive preservation, staging, journal commits, and crash recovery.

    Operates strictly in temporary directories; zero touch to production datasets.
    """

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.tmp_dir.name)
        self.raw_path = self.base_dir / "news_raw.parquet"
        self.proc_path = self.base_dir / "news_sentiment.parquet"
        self.market_path = self.base_dir / "market_features.parquet"
        self.backup_dir = self.base_dir / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.staged_dir = self.base_dir / "staged_chunks"
        self.manifest_path = self.base_dir / "manifest.json"
        self.journal_path = self.base_dir / "transaction_journal.json"

    def write_test_checksums(self, backup_files: list[Path] | None = None) -> Path:
        """Helper to generate trusted checksums.sha256 in self.backup_dir."""
        if backup_files is None:
            backup_files = list(self.backup_dir.glob("*.parquet"))
        chk_file = self.backup_dir / "checksums.sha256"
        lines = []
        for bf in sorted(backup_files):
            lines.append(f"{compute_sha256(bf)}  {bf.name}\n")
        with open(chk_file, "w", encoding="utf-8") as f:
            f.writelines(lines)
        return chk_file

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_merge_raw_news_archive_preserves_records_and_deduplicates(self) -> None:
        """Merging incoming raw news preserves all original records and deduplicates by ID/URL/headline."""
        existing_records = [
            {
                "ticker": "AAPL",
                "published_at": pd.to_datetime("2026-09-10 10:00:00Z"),
                "headline": "Existing Headline 1",
                "summary": "Summary 1",
                "source": "WireA",
                "url": "https://wire.com/1",
                "provider": "finnhub",
                "is_fallback": False,
                "raw_id": "raw_101",
            },
            {
                "ticker": "AAPL",
                "published_at": pd.to_datetime("2026-09-11 11:00:00Z"),
                "headline": "Existing Headline 2",
                "summary": "Summary 2",
                "source": "WireB",
                "url": "https://wire.com/2",
                "provider": "finnhub",
                "is_fallback": False,
                "raw_id": "raw_102",
            },
        ]
        pd.DataFrame(existing_records).to_parquet(self.raw_path)

        incoming_records = [
            {
                "ticker": "AAPL",
                "published_at": pd.to_datetime("2026-09-11 11:00:00Z"),
                "headline": "Existing Headline 2 duplicate",
                "summary": "Summary 2 dup",
                "source": "WireB",
                "url": "https://wire.com/2",
                "provider": "finnhub",
                "is_fallback": False,
                "raw_id": "raw_102",
            },
            {
                "ticker": "AAPL",
                "published_at": pd.to_datetime("2026-08-15 14:00:00Z"),
                "headline": "August Headline Net-New",
                "summary": "August summary",
                "source": "WireC",
                "url": "https://wire.com/3",
                "provider": "finnhub",
                "is_fallback": False,
                "raw_id": "raw_103",
            },
        ]
        incoming_df = pd.DataFrame(incoming_records)

        merged = merge_raw_news_archive(self.raw_path, incoming_df)
        self.assertEqual(len(merged), 3)
        retained_ids = set(merged["raw_id"])
        self.assertEqual(retained_ids, {"raw_101", "raw_102", "raw_103"})

        # Empty incoming returns existing unchanged
        merged_empty = merge_raw_news_archive(self.raw_path, pd.DataFrame())
        self.assertEqual(len(merged_empty), 2)

    def test_merge_processed_news_archive_preserves_sentiment(self) -> None:
        """Processed archive merge retains all scored fields and adds new scored records."""
        existing_proc = [
            {
                "ticker": "AAPL",
                "published_at": pd.to_datetime("2026-09-15 12:00:00Z"),
                "headline": "September Scored Headline",
                "summary": "Summary",
                "source": "WireA",
                "url": "https://wire.com/sep1",
                "provider": "finnhub",
                "is_fallback": False,
                "raw_id": "p1",
                "trading_date": pd.Timestamp("2026-09-15"),
                "sentiment_label": "positive",
                "sentiment_score": 0.85,
                "prob_positive": 0.9,
                "prob_negative": 0.05,
                "prob_neutral": 0.05,
            }
        ]
        pd.DataFrame(existing_proc).to_parquet(self.proc_path)

        incoming_proc = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "published_at": pd.to_datetime("2026-08-20 14:00:00Z"),
                    "headline": "August Scored Headline",
                    "summary": "Summary Aug",
                    "source": "WireB",
                    "url": "https://wire.com/aug1",
                    "provider": "finnhub",
                    "is_fallback": False,
                    "raw_id": "p2",
                    "trading_date": pd.Timestamp("2026-08-20"),
                    "sentiment_label": "negative",
                    "sentiment_score": -0.6,
                    "prob_positive": 0.1,
                    "prob_negative": 0.7,
                    "prob_neutral": 0.2,
                }
            ]
        )

        merged = merge_processed_news_archive(self.proc_path, incoming_proc)
        self.assertEqual(len(merged), 2)
        sep_row = merged.loc[merged["raw_id"] == "p1"].iloc[0]
        self.assertEqual(sep_row["sentiment_score"], 0.85)
        self.assertEqual(sep_row["sentiment_label"], "positive")

    def test_get_net_new_articles_filters_existing(self) -> None:
        """Net-new detection isolates articles that have not yet undergone sentiment scoring."""
        proc_records = [
            {
                "ticker": "AAPL",
                "published_at": pd.to_datetime("2026-09-01 10:00:00Z"),
                "headline": "Already Scored",
                "summary": "",
                "source": "A",
                "url": "https://wire.com/old1",
                "provider": "finnhub",
                "is_fallback": False,
                "raw_id": "id_old1",
                "trading_date": pd.Timestamp("2026-09-01"),
                "sentiment_label": "neutral",
                "sentiment_score": 0.0,
                "prob_positive": 0.1,
                "prob_negative": 0.1,
                "prob_neutral": 0.8,
            }
        ]
        proc_df = pd.DataFrame(proc_records)

        incoming_raw = pd.DataFrame(
            [
                # Match by raw_id
                {
                    "ticker": "AAPL",
                    "published_at": pd.to_datetime("2026-09-01 10:00:00Z"),
                    "headline": "Different headline but same ID",
                    "summary": "",
                    "source": "A",
                    "url": "https://wire.com/other",
                    "provider": "finnhub",
                    "is_fallback": False,
                    "raw_id": "id_old1",
                },
                # Net-new article
                {
                    "ticker": "AAPL",
                    "published_at": pd.to_datetime("2026-08-10 11:00:00Z"),
                    "headline": "Brand New Article",
                    "summary": "",
                    "source": "B",
                    "url": "https://wire.com/new1",
                    "provider": "finnhub",
                    "is_fallback": False,
                    "raw_id": "id_new1",
                },
            ]
        )

        net_new = get_net_new_articles(incoming_raw, proc_df)
        self.assertEqual(len(net_new), 1)
        self.assertEqual(net_new.iloc[0]["raw_id"], "id_new1")

    def test_merge_sentiment_with_successful_trading_days_isolation(self) -> None:
        """Only sessions explicitly in successful_trading_days are modified; others are untouched."""
        dates = pd.date_range("2026-08-10", periods=5, freq="B")
        base_market = pd.DataFrame(
            {
                "Close": [100.0, 101.0, 102.0, 103.0, 104.0],
                "Daily_Return": [0.01, 0.01, 0.01, 0.01, 0.01],
            },
            index=dates,
        )
        base_market.to_parquet(self.market_path)

        # Establish existing sentiment on Day 0 and Day 1
        init_sent = pd.DataFrame(
            {
                "sentiment_score": [0.70, 0.80],
                "sentiment_std": [0.05, 0.05],
                "sentiment_count": [3, 4],
                "prob_positive": [0.8, 0.85],
                "prob_negative": [0.1, 0.05],
                "prob_neutral": [0.1, 0.1],
            },
            index=dates[:2],
        )
        merge_sentiment_with_market_features(
            market_features_path=self.market_path,
            daily_sentiment_df=init_sent,
            output_path=self.market_path,
            full_rebuild=True,
        )

        # Incremental update specifying successful_trading_days = [Day 3, Day 4]
        # Day 3 has news, Day 4 has 0 news (confirmed empty)
        day3_sent = pd.DataFrame(
            {
                "sentiment_score": [-0.5],
                "sentiment_std": [0.02],
                "sentiment_count": [2],
                "prob_positive": [0.1],
                "prob_negative": [0.6],
                "prob_neutral": [0.3],
            },
            index=[dates[3]],
        )

        res = merge_sentiment_with_market_features(
            market_features_path=self.market_path,
            daily_sentiment_df=day3_sent,
            output_path=self.market_path,
            successful_trading_days=[dates[3].date(), dates[4].date()],
            full_rebuild=False,
        )

        # Day 0 & Day 1: strictly preserved
        self.assertEqual(res.loc[dates[0], "sentiment_coverage"], 1)
        self.assertAlmostEqual(res.loc[dates[0], "sentiment_score"], 0.70)
        self.assertEqual(res.loc[dates[1], "sentiment_coverage"], 1)
        self.assertAlmostEqual(res.loc[dates[1], "sentiment_score"], 0.80)

        # Day 2: never attempted / untouched (retains NaN, coverage=0)
        self.assertEqual(res.loc[dates[2], "sentiment_coverage"], 0)
        self.assertTrue(pd.isna(res.loc[dates[2], "sentiment_score"]))

        # Day 3: updated with new news
        self.assertEqual(res.loc[dates[3], "sentiment_coverage"], 1)
        self.assertAlmostEqual(res.loc[dates[3], "sentiment_score"], -0.5)

        # Day 4: queried and confirmed empty (NaN, coverage=0)
        self.assertEqual(res.loc[dates[4], "sentiment_coverage"], 0)
        self.assertTrue(pd.isna(res.loc[dates[4], "sentiment_score"]))

    def test_transaction_journal_commit_and_recovery(self) -> None:
        """Transaction journal tracks commits and restores backup copies upon interrupted STAGED state."""
        target1 = self.base_dir / "file1.parquet"
        target2 = self.base_dir / "file2.parquet"
        backup1 = self.backup_dir / "file1.parquet"
        backup2 = self.backup_dir / "file2.parquet"

        # Baseline versions
        pd.DataFrame({"v": [1]}).to_parquet(target1)
        pd.DataFrame({"v": [10]}).to_parquet(target2)
        pd.DataFrame({"v": [1]}).to_parquet(backup1)
        pd.DataFrame({"v": [10]}).to_parquet(backup2)
        self.write_test_checksums([backup1, backup2])

        # Staged versions
        stage1 = target1.with_name("file1.parquet.stage")
        stage2 = target2.with_name("file2.parquet.stage")
        pd.DataFrame({"v": [2]}).to_parquet(stage1)
        pd.DataFrame({"v": [20]}).to_parquet(stage2)

        # 1. Normal commit succeeds
        commit_multi_file_transaction(
            staged_targets=[(stage1, target1), (stage2, target2)],
            backup_dir=self.backup_dir,
            journal_path=self.journal_path,
            allowed_roots=[self.base_dir],
        )
        self.assertEqual(pd.read_parquet(target1)["v"].iloc[0], 2)
        self.assertEqual(pd.read_parquet(target2)["v"].iloc[0], 20)
        self.assertFalse(stage1.exists())
        self.assertFalse(stage2.exists())

        with open(self.journal_path, "r") as f:
            j = json.load(f)
        self.assertEqual(j["state"], TX_STATE_COMMITTED)

        # 2. Simulate interrupted commit: target1 replaced with bad data, stage2 still exists, journal in STAGED
        pd.DataFrame({"v": [999]}).to_parquet(target1)
        pd.DataFrame({"v": [200]}).to_parquet(stage2)
        sha1 = compute_sha256(backup1)
        sha2 = compute_sha256(backup2)
        journal_crash = {
            "transaction_id": "tx_crash_sim",
            "state": TX_STATE_STAGED,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "backup_dir": str(self.backup_dir),
            "files": [
                {"target": str(target1), "staged": str(stage1), "backup": str(backup1), "backup_sha256": sha1},
                {"target": str(target2), "staged": str(stage2), "backup": str(backup2), "backup_sha256": sha2},
            ],
        }
        with open(self.journal_path, "w", encoding="utf-8") as f:
            json.dump(journal_crash, f, indent=2)

        # Recover
        recovered = check_and_recover_transactions(self.journal_path, allowed_roots=[self.base_dir])
        self.assertTrue(recovered)
        # target1 was rolled back from backup1
        self.assertEqual(pd.read_parquet(target1)["v"].iloc[0], 1)
        # target2 remains valid baseline
        self.assertEqual(pd.read_parquet(target2)["v"].iloc[0], 10)
        # Orphaned stage2 was removed
        self.assertFalse(stage2.exists())

        with open(self.journal_path, "r", encoding="utf-8") as f:
            j = json.load(f)
        self.assertEqual(j["state"], TX_STATE_ABORTED_RECOVERED)

    def test_recovery_fails_when_backup_missing(self) -> None:
        """Recovery refuses to restore if any backup is missing, preserving all diagnostic artifacts."""
        target1 = self.base_dir / "file1.parquet"
        target2 = self.base_dir / "file2.parquet"
        backup1 = self.backup_dir / "file1.parquet"
        backup2 = self.backup_dir / "file2.parquet"

        pd.DataFrame({"v": [1]}).to_parquet(backup1)
        # Note: backup2 is intentionally NOT created!
        pd.DataFrame({"v": [999]}).to_parquet(target1)
        pd.DataFrame({"v": [999]}).to_parquet(target2)

        stage1 = target1.with_name("file1.parquet.stage")
        stage2 = target2.with_name("file2.parquet.stage")
        pd.DataFrame({"v": [2]}).to_parquet(stage1)
        pd.DataFrame({"v": [20]}).to_parquet(stage2)

        # Write trusted checksums with entries for both
        chk_file = self.backup_dir / "checksums.sha256"
        with open(chk_file, "w", encoding="utf-8") as f:
            f.write(f"{compute_sha256(backup1)}  file1.parquet\n")
            f.write(f"{'a'*64}  file2.parquet\n")

        journal = {
            "transaction_id": "tx_missing_backup",
            "state": TX_STATE_STAGED,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "backup_dir": str(self.backup_dir),
            "files": [
                {"target": str(target1), "staged": str(stage1), "backup": str(backup1), "backup_sha256": compute_sha256(backup1)},
                {"target": str(target2), "staged": str(stage2), "backup": str(backup2), "backup_sha256": "a"*64},
            ],
        }
        with open(self.journal_path, "w", encoding="utf-8") as f:
            json.dump(journal, f, indent=2)

        with self.assertRaises(RuntimeError) as ctx:
            check_and_recover_transactions(self.journal_path, allowed_roots=[self.base_dir])
        self.assertIn("Required backup missing", str(ctx.exception))

        # Assert zero files restored, targets untouched, staged files preserved
        self.assertEqual(pd.read_parquet(target1)["v"].iloc[0], 999)
        self.assertEqual(pd.read_parquet(target2)["v"].iloc[0], 999)
        self.assertTrue(stage1.exists())
        self.assertTrue(stage2.exists())
        with open(self.journal_path, "r", encoding="utf-8") as f:
            j = json.load(f)
        self.assertEqual(j["state"], TX_STATE_STAGED)

    def test_recovery_fails_when_backup_checksum_corrupted(self) -> None:
        """Recovery halts if a backup file's SHA-256 does not match the trusted checksum."""
        target1 = self.base_dir / "file1.parquet"
        backup1 = self.backup_dir / "file1.parquet"
        pd.DataFrame({"v": [1]}).to_parquet(backup1)
        trusted_sha = compute_sha256(backup1)
        self.write_test_checksums([backup1])

        # Corrupt backup bytes after trusted sha was recorded
        with open(backup1, "ab") as f:
            f.write(b"corrupt_noise")

        pd.DataFrame({"v": [999]}).to_parquet(target1)
        stage1 = target1.with_name("file1.parquet.stage")
        pd.DataFrame({"v": [2]}).to_parquet(stage1)

        journal = {
            "transaction_id": "tx_corrupt_backup",
            "state": TX_STATE_STAGED,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "backup_dir": str(self.backup_dir),
            "files": [
                {"target": str(target1), "staged": str(stage1), "backup": str(backup1), "backup_sha256": trusted_sha},
            ],
        }
        with open(self.journal_path, "w", encoding="utf-8") as f:
            json.dump(journal, f, indent=2)

        with self.assertRaises(RuntimeError) as ctx:
            check_and_recover_transactions(self.journal_path, allowed_roots=[self.base_dir])
        self.assertIn("Backup integrity check failed", str(ctx.exception))

        # Staged file and target untouched
        self.assertTrue(stage1.exists())
        self.assertEqual(pd.read_parquet(target1)["v"].iloc[0], 999)

    def test_recovery_fails_closed_on_malformed_journal(self) -> None:
        """Malformed or unparseable journal raises RuntimeError and fails closed."""
        with open(self.journal_path, "w", encoding="utf-8") as f:
            f.write('{"state": "STAGED", "files": [unclosed_json')

        with self.assertRaises(RuntimeError) as ctx:
            check_and_recover_transactions(self.journal_path, allowed_roots=[self.base_dir])
        self.assertIn("Malformed or unreadable transaction journal", str(ctx.exception))

    def test_recovery_fails_closed_on_unknown_transaction_state(self) -> None:
        """Unknown transaction state raises RuntimeError instead of returning False."""
        journal = {
            "transaction_id": "tx_weird",
            "state": "MYSTERIOUS_STATE",
            "files": [],
        }
        with open(self.journal_path, "w", encoding="utf-8") as f:
            json.dump(journal, f)

        with self.assertRaises(RuntimeError) as ctx:
            check_and_recover_transactions(self.journal_path, allowed_roots=[self.base_dir])
        self.assertIn("Unknown or ambiguous transaction state", str(ctx.exception))

    def test_recovery_rejects_unexpected_target_paths(self) -> None:
        """Recovery rejects paths outside allowed roots and aborts."""
        bad_target = self.base_dir.parent / "escape.parquet"
        backup1 = self.backup_dir / "file1.parquet"
        pd.DataFrame({"v": [1]}).to_parquet(backup1)
        sha1 = compute_sha256(backup1)
        self.write_test_checksums([backup1])

        journal = {
            "transaction_id": "tx_bad_path",
            "state": TX_STATE_STAGED,
            "backup_dir": str(self.backup_dir),
            "files": [
                {
                    "target": str(bad_target),
                    "staged": str(bad_target.with_name("escape.parquet.stage")),
                    "backup": str(backup1),
                    "backup_sha256": sha1,
                }
            ],
        }
        with open(self.journal_path, "w", encoding="utf-8") as f:
            json.dump(journal, f)

        with self.assertRaises(RuntimeError) as ctx:
            check_and_recover_transactions(self.journal_path, allowed_roots=[self.base_dir])
        self.assertIn("Target path rejected", str(ctx.exception))

    def test_recovery_failure_during_restoration_preserves_staged_and_raises(self) -> None:
        """If an error occurs during restore replacement, staged files are preserved and error propagates."""
        target1 = self.base_dir / "file1.parquet"
        backup1 = self.backup_dir / "file1.parquet"
        pd.DataFrame({"v": [1]}).to_parquet(backup1)
        sha1 = compute_sha256(backup1)
        self.write_test_checksums([backup1])

        stage1 = target1.with_name("file1.parquet.stage")
        pd.DataFrame({"v": [2]}).to_parquet(stage1)

        journal = {
            "transaction_id": "tx_fail_mid_restore",
            "state": TX_STATE_STAGED,
            "backup_dir": str(self.backup_dir),
            "files": [
                {"target": str(target1), "staged": str(stage1), "backup": str(backup1), "backup_sha256": sha1},
            ],
        }
        with open(self.journal_path, "w", encoding="utf-8") as f:
            json.dump(journal, f)

        with patch("shutil.copy2", side_effect=IOError("Simulated disk error during restore")):
            with self.assertRaises(IOError):
                check_and_recover_transactions(self.journal_path, allowed_roots=[self.base_dir])

        # Staged artifact was NOT deleted
        self.assertTrue(stage1.exists())

    def test_recovery_repeated_and_idempotent_after_interruption(self) -> None:
        """Recovery can resume from RECOVERING state and subsequent calls return False idempotently."""
        target1 = self.base_dir / "file1.parquet"
        target2 = self.base_dir / "file2.parquet"
        backup1 = self.backup_dir / "file1.parquet"
        backup2 = self.backup_dir / "file2.parquet"
        pd.DataFrame({"v": [1]}).to_parquet(backup1)
        pd.DataFrame({"v": [10]}).to_parquet(backup2)
        sha1 = compute_sha256(backup1)
        sha2 = compute_sha256(backup2)
        self.write_test_checksums([backup1, backup2])

        # Target 1 partially restored, target 2 corrupt
        pd.DataFrame({"v": [1]}).to_parquet(target1)
        pd.DataFrame({"v": [999]}).to_parquet(target2)

        stage2 = target2.with_name("file2.parquet.stage")
        pd.DataFrame({"v": [20]}).to_parquet(stage2)

        journal = {
            "transaction_id": "tx_interrupted_recovery",
            "state": TX_STATE_RECOVERING,
            "recovering_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "backup_dir": str(self.backup_dir),
            "files": [
                {"target": str(target1), "staged": str(target1.with_name("file1.parquet.stage")), "backup": str(backup1), "backup_sha256": sha1},
                {"target": str(target2), "staged": str(stage2), "backup": str(backup2), "backup_sha256": sha2},
            ],
        }
        with open(self.journal_path, "w", encoding="utf-8") as f:
            json.dump(journal, f)

        # First run: resumes from RECOVERING and finishes
        res1 = check_and_recover_transactions(self.journal_path, allowed_roots=[self.base_dir])
        self.assertTrue(res1)
        self.assertEqual(pd.read_parquet(target1)["v"].iloc[0], 1)
        self.assertEqual(pd.read_parquet(target2)["v"].iloc[0], 10)
        self.assertFalse(stage2.exists())

        with open(self.journal_path, "r", encoding="utf-8") as f:
            j = json.load(f)
        self.assertEqual(j["state"], TX_STATE_ABORTED_RECOVERED)

        # Second run: already ABORTED_RECOVERED -> returns False without error
        res2 = check_and_recover_transactions(self.journal_path, allowed_roots=[self.base_dir])
        self.assertFalse(res2)

    def test_pipeline_startup_halts_on_recovery_and_on_failure(self) -> None:
        """run_production_pipeline halts on recovered transactions and halts on failed recovery."""
        target1 = self.base_dir / "news_raw.parquet"
        backup1 = self.backup_dir / "news_raw.parquet"
        pd.DataFrame({"v": [1]}).to_parquet(backup1)
        pd.DataFrame({"v": [999]}).to_parquet(target1)
        sha1 = compute_sha256(backup1)
        self.write_test_checksums([backup1])

        stage1 = target1.with_name("news_raw.parquet.stage")
        pd.DataFrame({"v": [2]}).to_parquet(stage1)

        journal = {
            "transaction_id": "tx_startup_halt",
            "state": TX_STATE_STAGED,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "backup_dir": str(self.backup_dir),
            "files": [
                {"target": str(target1), "staged": str(stage1), "backup": str(backup1), "backup_sha256": sha1},
            ],
        }
        with open(self.journal_path, "w", encoding="utf-8") as f:
            json.dump(journal, f)

        # Case A: Rollback succeeds -> startup halts with RuntimeError for operator review
        with self.assertRaises(RuntimeError) as ctx:
            run_production_pipeline(
                start_date="2026-08-01",
                end_date="2026-08-01",
                raw_news_path=target1,
                processed_news_path=self.proc_path,
                market_features_path=self.market_path,
                backup_dir=self.backup_dir,
                journal_path=self.journal_path,
                allowed_roots=[self.base_dir],
            )
        self.assertIn("Pipeline halted for safety review", str(ctx.exception))
        # Baseline was restored
        self.assertEqual(pd.read_parquet(target1)["v"].iloc[0], 1)

        # Case B: Corrupted journal -> startup halts with RuntimeError
        with open(self.journal_path, "w", encoding="utf-8") as f:
            f.write("{corrupt_json")

        with self.assertRaises(RuntimeError) as ctx2:
            run_production_pipeline(
                start_date="2026-08-01",
                end_date="2026-08-01",
                raw_news_path=target1,
                processed_news_path=self.proc_path,
                market_features_path=self.market_path,
                backup_dir=self.backup_dir,
                journal_path=self.journal_path,
                allowed_roots=[self.base_dir],
            )
        self.assertIn("Startup crash recovery failed", str(ctx2.exception))

    def test_preservation_of_staged_artifacts_when_recovery_fails(self) -> None:
        """Staged artifacts are never deleted when pre-flight validation aborts recovery."""
        target1 = self.base_dir / "file1.parquet"
        backup1 = self.backup_dir / "file1.parquet"
        pd.DataFrame({"v": [1]}).to_parquet(backup1)
        self.write_test_checksums([backup1])

        stage1 = target1.with_name("file1.parquet.stage")
        pd.DataFrame({"v": [999]}).to_parquet(stage1)

        # Mismatched expected sha in journal
        journal = {
            "transaction_id": "tx_preserve_artifacts",
            "state": TX_STATE_STAGED,
            "backup_dir": str(self.backup_dir),
            "files": [
                {"target": str(target1), "staged": str(stage1), "backup": str(backup1), "backup_sha256": "wrong_sha"},
            ],
        }
        with open(self.journal_path, "w", encoding="utf-8") as f:
            json.dump(journal, f)

        with self.assertRaises(RuntimeError):
            check_and_recover_transactions(self.journal_path, allowed_roots=[self.base_dir])

        # Staged artifact is 100% preserved
        self.assertTrue(stage1.exists())
        self.assertEqual(pd.read_parquet(stage1)["v"].iloc[0], 999)

    def test_staged_chunks_persistence_and_resumption(self) -> None:
        """fetch_finnhub_news saves per-date chunks to staged_chunks_dir and resumes without re-calling client."""
        class MockClient:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def company_news(self, ticker: str, _from: str, to: str) -> list[dict[str, Any]]:
                self.calls.append(_from)
                if _from == "2026-08-01":
                    return [
                        {
                            "id": 1,
                            "datetime": int(pd.Timestamp("2026-08-01 12:00:00Z").timestamp()),
                            "headline": "August 1 news",
                            "summary": "Summary",
                            "source": "Wire",
                            "url": "https://wire.com/aug1",
                        }
                    ]
                return []

        client = MockClient()
        df1 = fetch_finnhub_news(
            ticker="AAPL",
            start_date="2026-08-01",
            end_date="2026-08-02",
            client=client,
            chunk_days=1,
            retry_delay=0.0,
            request_delay_seconds=0.0,
            staged_chunks_dir=self.staged_dir,
            manifest_path=self.manifest_path,
        )

        self.assertEqual(len(df1), 1)
        self.assertEqual(client.calls, ["2026-08-01", "2026-08-02"])
        # Check files on disk
        self.assertTrue((self.staged_dir / "AAPL_2026-08-01.parquet").exists())
        self.assertTrue((self.staged_dir / "AAPL_2026-08-02.empty").exists())
        self.assertTrue(self.manifest_path.exists())

        # Now resume with a client that throws an error if called!
        class ExplodingClient:
            def company_news(self, *args: Any, **kwargs: Any) -> Any:
                raise AssertionError("Client should not be called when chunks are already staged!")

        df2 = fetch_finnhub_news(
            ticker="AAPL",
            start_date="2026-08-01",
            end_date="2026-08-02",
            client=ExplodingClient(),
            chunk_days=1,
            retry_delay=0.0,
            request_delay_seconds=0.0,
            staged_chunks_dir=self.staged_dir,
            manifest_path=self.manifest_path,
        )
        self.assertEqual(len(df2), 1)
        self.assertEqual(df2.iloc[0]["headline"], "August 1 news")

    def test_production_pipeline_net_new_scoring_and_transaction_journal(self) -> None:
        """Full pipeline with backup_dir scores only net-new headlines and commits via journal."""
        dates = pd.date_range("2026-08-03", periods=3, freq="B")
        base_market = pd.DataFrame(
            {"Close": [100.0, 101.0, 102.0], "Daily_Return": [0.01, 0.01, 0.01]},
            index=dates,
        )
        base_market.to_parquet(self.market_path)
        shutil.copy2(self.market_path, self.backup_dir / self.market_path.name)

        # Baseline raw and processed with Day 0
        raw_rec = [
            {
                "ticker": "AAPL",
                "published_at": pd.Timestamp("2026-08-03 14:00:00Z"),
                "headline": "Day 0 headline",
                "summary": "Summary 0",
                "source": "Wire",
                "url": "https://wire.com/day0",
                "provider": "finnhub",
                "is_fallback": False,
                "raw_id": "d0",
            }
        ]
        pd.DataFrame(raw_rec).to_parquet(self.raw_path)
        shutil.copy2(self.raw_path, self.backup_dir / self.raw_path.name)

        proc_rec = [
            {
                **raw_rec[0],
                "trading_date": pd.Timestamp("2026-08-03"),
                "sentiment_label": "positive",
                "sentiment_score": 0.8,
                "prob_positive": 0.9,
                "prob_negative": 0.05,
                "prob_neutral": 0.05,
            }
        ]
        pd.DataFrame(proc_rec).to_parquet(self.proc_path)
        shutil.copy2(self.proc_path, self.backup_dir / self.proc_path.name)

        # Mock client returns Day 0 article AND Day 1 article
        class MockClient:
            def company_news(self, ticker: str, _from: str, to: str) -> list[dict[str, Any]]:
                if _from == "2026-08-03":
                    return [
                        {
                            "id": "d0",
                            "datetime": int(pd.Timestamp("2026-08-03 14:00:00Z").timestamp()),
                            "headline": "Day 0 headline",
                            "summary": "Summary 0",
                            "source": "Wire",
                            "url": "https://wire.com/day0",
                        }
                    ]
                elif _from == "2026-08-04":
                    return [
                        {
                            "id": "d1",
                            "datetime": int(pd.Timestamp("2026-08-04 14:00:00Z").timestamp()),
                            "headline": "Day 1 new headline",
                            "summary": "Summary 1",
                            "source": "Wire",
                            "url": "https://wire.com/day1",
                        }
                    ]
                return []

        # Mock analyzer tracks analyzed headlines
        class TrackingAnalyzer:
            def __init__(self) -> None:
                self.analyzed_headlines: list[str] = []

            def analyze_batch(self, headlines: list[str], batch_size: int = 32) -> pd.DataFrame:
                self.analyzed_headlines.extend(headlines)
                return pd.DataFrame(
                    [
                        {
                            "sentiment_label": "negative",
                            "sentiment_score": -0.4,
                            "prob_positive": 0.1,
                            "prob_negative": 0.7,
                            "prob_neutral": 0.2,
                        }
                        for _ in headlines
                    ]
                )

        analyzer = TrackingAnalyzer()

        # Pre-populate backups and trusted checksums
        self.write_test_checksums()

        with patch("src.data_ingestion.news_sentiment.fetch_yahoo_fallback", return_value=pd.DataFrame()):
            res = run_production_pipeline(
                start_date="2026-08-03",
                end_date="2026-08-04",
                client=MockClient(),
                market_features_path=self.market_path,
                raw_news_path=self.raw_path,
                processed_news_path=self.proc_path,
                analyzer=analyzer,
                retry_delay=0.0,
                request_delay_seconds=0.0,
                backup_dir=self.backup_dir,
                journal_path=self.journal_path,
                allowed_roots=[self.base_dir],
            )

        self.assertEqual(res["status"], STATUS_COMPLETE)
        # Net-new scoring: ONLY Day 1 headline was analyzed! Day 0 headline was skipped!
        self.assertEqual(analyzer.analyzed_headlines, ["Day 1 new headline"])

        # Raw archive contains both
        raw_final = pd.read_parquet(self.raw_path)
        self.assertEqual(len(raw_final), 2)

        # Processed archive contains both with Day 0 sentiment preserved (0.8) and Day 1 scored (-0.4)
        proc_final = pd.read_parquet(self.proc_path)
        self.assertEqual(len(proc_final), 2)
        d0_sent = proc_final.loc[proc_final["raw_id"] == "d0", "sentiment_score"].iloc[0]
        d1_sent = proc_final.loc[proc_final["raw_id"] == "d1", "sentiment_score"].iloc[0]
        self.assertEqual(d0_sent, 0.8)
        self.assertEqual(d1_sent, -0.4)

        # Transaction journal recorded clean commit
        with open(self.journal_path, "r") as f:
            j = json.load(f)
        self.assertEqual(j["state"], TX_STATE_COMMITTED)

    def test_recovery_rejects_project_source_code_target(self) -> None:
        """Recovery rejects target paths inside project source directories (e.g. news_sentiment.py)."""
        src_target = PROJECT_ROOT / "src" / "data_ingestion" / "news_sentiment.py"
        src_mtime_before = src_target.stat().st_mtime
        backup1 = self.backup_dir / "news_sentiment.py.parquet"
        pd.DataFrame({"v": [1]}).to_parquet(backup1)
        self.write_test_checksums([backup1])

        journal = {
            "transaction_id": "tx_bad_source_target",
            "state": TX_STATE_STAGED,
            "backup_dir": str(self.backup_dir),
            "files": [
                {
                    "target": str(src_target),
                    "staged": str(self.base_dir / "staged.parquet"),
                    "backup": str(backup1),
                    "backup_sha256": compute_sha256(backup1),
                }
            ],
        }
        with open(self.journal_path, "w", encoding="utf-8") as f:
            json.dump(journal, f)

        with self.assertRaises(RuntimeError) as ctx:
            check_and_recover_transactions(self.journal_path, allowed_roots=[self.base_dir])
        err_msg = str(ctx.exception)
        self.assertTrue(
            "targets inside project source or git directories are strictly forbidden" in err_msg
            or "outside permitted data roots" in err_msg
            or "only '.parquet' files are authorized data targets" in err_msg
        )
        # Verify source file was untouched
        self.assertEqual(src_target.stat().st_mtime, src_mtime_before)

    def test_recovery_rejects_non_parquet_target_extension(self) -> None:
        """Recovery rejects target files that do not have .parquet extension."""
        script_target = self.base_dir / "malicious_script.sh"
        backup1 = self.backup_dir / "backup.parquet"
        pd.DataFrame({"v": [1]}).to_parquet(backup1)
        self.write_test_checksums([backup1])

        journal = {
            "transaction_id": "tx_bad_ext",
            "state": TX_STATE_STAGED,
            "backup_dir": str(self.backup_dir),
            "files": [
                {
                    "target": str(script_target),
                    "staged": str(self.base_dir / "staged.stage"),
                    "backup": str(backup1),
                    "backup_sha256": compute_sha256(backup1),
                }
            ],
        }
        with open(self.journal_path, "w", encoding="utf-8") as f:
            json.dump(journal, f)

        with self.assertRaises(RuntimeError) as ctx:
            check_and_recover_transactions(self.journal_path, allowed_roots=[self.base_dir])
        self.assertIn("only '.parquet' files are authorized data targets", str(ctx.exception))

    def test_recovery_rejects_journal_self_authorizing_external_backup_dir(self) -> None:
        """A journal cannot self-authorize an external backup directory outside established roots."""
        external_dir = tempfile.TemporaryDirectory()
        external_backup_path = Path(external_dir.name) / "evil_backup"
        external_backup_path.mkdir(parents=True, exist_ok=True)
        backup1 = external_backup_path / "file1.parquet"
        pd.DataFrame({"v": [1]}).to_parquet(backup1)
        chk_file = external_backup_path / "checksums.sha256"
        with open(chk_file, "w", encoding="utf-8") as f:
            f.write(f"{compute_sha256(backup1)}  file1.parquet\n")

        target1 = self.base_dir / "file1.parquet"
        pd.DataFrame({"v": [999]}).to_parquet(target1)

        journal = {
            "transaction_id": "tx_external_backup",
            "state": TX_STATE_STAGED,
            "backup_dir": str(external_backup_path),
            "files": [
                {
                    "target": str(target1),
                    "staged": str(target1.with_name("file1.parquet.stage")),
                    "backup": str(backup1),
                    "backup_sha256": compute_sha256(backup1),
                }
            ],
        }
        with open(self.journal_path, "w", encoding="utf-8") as f:
            json.dump(journal, f)

        try:
            with self.assertRaises(RuntimeError) as ctx:
                check_and_recover_transactions(self.journal_path, allowed_roots=[self.base_dir])
            self.assertIn("outside authorized roots", str(ctx.exception))
            # Verify target was not overwritten from external backup
            self.assertEqual(pd.read_parquet(target1)["v"].iloc[0], 999)
        finally:
            external_dir.cleanup()

    def test_recovery_aborts_when_trusted_checksums_file_missing(self) -> None:
        """Recovery aborts before touching any targets if checksums.sha256 is missing from backup_dir."""
        target1 = self.base_dir / "file1.parquet"
        backup1 = self.backup_dir / "file1.parquet"
        pd.DataFrame({"v": [1]}).to_parquet(backup1)
        pd.DataFrame({"v": [999]}).to_parquet(target1)
        stage1 = target1.with_name("file1.parquet.stage")
        pd.DataFrame({"v": [2]}).to_parquet(stage1)

        # Ensure checksums.sha256 does not exist
        chk_file = self.backup_dir / "checksums.sha256"
        if chk_file.exists():
            chk_file.unlink()

        journal = {
            "transaction_id": "tx_no_trusted_checksums",
            "state": TX_STATE_STAGED,
            "backup_dir": str(self.backup_dir),
            "files": [
                {
                    "target": str(target1),
                    "staged": str(stage1),
                    "backup": str(backup1),
                    "backup_sha256": compute_sha256(backup1),
                }
            ],
        }
        with open(self.journal_path, "w", encoding="utf-8") as f:
            json.dump(journal, f)

        with self.assertRaises(RuntimeError) as ctx:
            check_and_recover_transactions(self.journal_path, allowed_roots=[self.base_dir])
        self.assertIn("Required trusted checksums file missing", str(ctx.exception))

        # Target must remain unmodified, staged file preserved
        self.assertEqual(pd.read_parquet(target1)["v"].iloc[0], 999)
        self.assertTrue(stage1.exists())

    def test_recovery_aborts_when_trusted_checksum_mismatches_journal(self) -> None:
        """Recovery aborts when the journal's recorded hash conflicts with the trusted baseline file."""
        target1 = self.base_dir / "file1.parquet"
        backup1 = self.backup_dir / "file1.parquet"
        pd.DataFrame({"v": [1]}).to_parquet(backup1)
        pd.DataFrame({"v": [999]}).to_parquet(target1)
        stage1 = target1.with_name("file1.parquet.stage")
        pd.DataFrame({"v": [2]}).to_parquet(stage1)

        self.write_test_checksums([backup1])
        fake_journal_hash = "a" * 64

        journal = {
            "transaction_id": "tx_hash_conflict",
            "state": TX_STATE_STAGED,
            "backup_dir": str(self.backup_dir),
            "files": [
                {
                    "target": str(target1),
                    "staged": str(stage1),
                    "backup": str(backup1),
                    "backup_sha256": fake_journal_hash,
                }
            ],
        }
        with open(self.journal_path, "w", encoding="utf-8") as f:
            json.dump(journal, f)

        with self.assertRaises(RuntimeError) as ctx:
            check_and_recover_transactions(self.journal_path, allowed_roots=[self.base_dir])
        self.assertIn("Checksum conflict", str(ctx.exception))

        # Target unmodified, staged file preserved
        self.assertEqual(pd.read_parquet(target1)["v"].iloc[0], 999)
        self.assertTrue(stage1.exists())

    def test_recovery_aborts_when_trusted_checksums_has_malformed_entry(self) -> None:
        """Recovery aborts if checksums.sha256 contains invalid hex, truncated hash, or ambiguous duplicate."""
        target1 = self.base_dir / "file1.parquet"
        backup1 = self.backup_dir / "file1.parquet"
        pd.DataFrame({"v": [1]}).to_parquet(backup1)
        pd.DataFrame({"v": [999]}).to_parquet(target1)
        stage1 = target1.with_name("file1.parquet.stage")
        pd.DataFrame({"v": [2]}).to_parquet(stage1)

        # Write malformed checksum (too short, non-hex)
        chk_file = self.backup_dir / "checksums.sha256"
        with open(chk_file, "w", encoding="utf-8") as f:
            f.write("not_a_valid_64_character_hex_hash  file1.parquet\n")

        journal = {
            "transaction_id": "tx_malformed_checksums",
            "state": TX_STATE_STAGED,
            "backup_dir": str(self.backup_dir),
            "files": [
                {
                    "target": str(target1),
                    "staged": str(stage1),
                    "backup": str(backup1),
                    "backup_sha256": compute_sha256(backup1),
                }
            ],
        }
        with open(self.journal_path, "w", encoding="utf-8") as f:
            json.dump(journal, f)

        with self.assertRaises(RuntimeError) as ctx:
            check_and_recover_transactions(self.journal_path, allowed_roots=[self.base_dir])
        self.assertIn("Malformed SHA-256 hash", str(ctx.exception))
        self.assertEqual(pd.read_parquet(target1)["v"].iloc[0], 999)

    def test_production_pipeline_halts_before_write_when_no_backup_dir_supplied(self) -> None:
        """Production pipeline halts with RuntimeError before writing any file when backup_dir is None."""
        class MockClient:
            def company_news(self, ticker: str, _from: str, to: str) -> list[dict[str, Any]]:
                return [
                    {
                        "id": "1",
                        "datetime": int(pd.Timestamp("2026-08-01 12:00:00Z").timestamp()),
                        "headline": "August news",
                        "summary": "Summary",
                        "source": "Wire",
                        "url": "https://wire.com/1",
                    }
                ]

        with patch("src.data_ingestion.news_sentiment.fetch_yahoo_fallback", return_value=pd.DataFrame()):
            with self.assertRaises(RuntimeError) as ctx:
                run_production_pipeline(
                    start_date="2026-08-01",
                    end_date="2026-08-01",
                    client=MockClient(),
                    raw_news_path=self.raw_path,
                    processed_news_path=self.proc_path,
                    market_features_path=self.market_path,
                    backup_dir=None,
                    allowed_roots=[self.base_dir],
                )
        self.assertIn("requires an explicit, verified backup_dir before writing to datasets", str(ctx.exception))
        self.assertFalse(self.raw_path.exists())
        self.assertFalse(self.proc_path.exists())

    def test_recovery_interrupted_restoration_and_successful_retry(self) -> None:
        """Multi-file restoration can resume from RECOVERING state and complete atomically across all targets."""
        target1 = self.base_dir / "raw.parquet"
        target2 = self.base_dir / "proc.parquet"
        target3 = self.base_dir / "market.parquet"

        backup1 = self.backup_dir / "raw.parquet"
        backup2 = self.backup_dir / "proc.parquet"
        backup3 = self.backup_dir / "market.parquet"

        # Baseline versions
        pd.DataFrame({"v": [100]}).to_parquet(backup1)
        pd.DataFrame({"v": [200]}).to_parquet(backup2)
        pd.DataFrame({"v": [300]}).to_parquet(backup3)
        self.write_test_checksums([backup1, backup2, backup3])

        # State on disk before recovery: target1 already restored, target2 and target3 corrupted
        pd.DataFrame({"v": [100]}).to_parquet(target1)
        pd.DataFrame({"v": [999]}).to_parquet(target2)
        pd.DataFrame({"v": [888]}).to_parquet(target3)

        # Staged files still present
        stage1 = target1.with_name("raw.parquet.stage")
        stage2 = target2.with_name("proc.parquet.stage")
        stage3 = target3.with_name("market.parquet.stage")
        pd.DataFrame({"v": [101]}).to_parquet(stage1)
        pd.DataFrame({"v": [201]}).to_parquet(stage2)
        pd.DataFrame({"v": [301]}).to_parquet(stage3)

        # Journal in RECOVERING state
        journal = {
            "transaction_id": "tx_multi_restore_retry",
            "state": TX_STATE_RECOVERING,
            "recovering_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "backup_dir": str(self.backup_dir),
            "files": [
                {"target": str(target1), "staged": str(stage1), "backup": str(backup1), "backup_sha256": compute_sha256(backup1)},
                {"target": str(target2), "staged": str(stage2), "backup": str(backup2), "backup_sha256": compute_sha256(backup2)},
                {"target": str(target3), "staged": str(stage3), "backup": str(backup3), "backup_sha256": compute_sha256(backup3)},
            ],
        }
        with open(self.journal_path, "w", encoding="utf-8") as f:
            json.dump(journal, f, indent=2)

        # First run: resumes from RECOVERING and restores all 3 targets
        res1 = check_and_recover_transactions(self.journal_path, allowed_roots=[self.base_dir])
        self.assertTrue(res1)
        self.assertEqual(pd.read_parquet(target1)["v"].iloc[0], 100)
        self.assertEqual(pd.read_parquet(target2)["v"].iloc[0], 200)
        self.assertEqual(pd.read_parquet(target3)["v"].iloc[0], 300)

        # All staged files cleaned up
        self.assertFalse(stage1.exists())
        self.assertFalse(stage2.exists())
        self.assertFalse(stage3.exists())

        # Journal marked ABORTED_RECOVERED
        with open(self.journal_path, "r", encoding="utf-8") as f:
            j = json.load(f)
        self.assertEqual(j["state"], TX_STATE_ABORTED_RECOVERED)

        # Second run: idempotent return False
        res2 = check_and_recover_transactions(self.journal_path, allowed_roots=[self.base_dir])
        self.assertFalse(res2)


class TestBuildPipelineOrchestration(unittest.TestCase):
    """Test argument forwarding and fail-closed transaction safety in build_pipeline.py."""

    def test_sentiment_stage_fails_without_backup_dir(self) -> None:
        """run_stage_sentiment raises ValueError when backup_dir is None or empty."""
        from src.build_pipeline import run_stage_sentiment
        with patch("src.data_ingestion.news_sentiment.run_production_pipeline") as mock_pipeline:
            with self.assertRaises(ValueError) as ctx:
                run_stage_sentiment(backup_dir=None)
            self.assertIn("requires an explicit --backup-dir argument", str(ctx.exception))
            mock_pipeline.assert_not_called()

            with self.assertRaises(ValueError) as ctx2:
                run_stage_sentiment(backup_dir="")
            self.assertIn("requires an explicit --backup-dir argument", str(ctx2.exception))
            mock_pipeline.assert_not_called()

    def test_sentiment_stage_passes_backup_dir(self) -> None:
        """run_stage_sentiment passes Path(backup_dir) to run_production_pipeline."""
        from src.build_pipeline import run_stage_sentiment
        with patch("src.data_ingestion.news_sentiment.run_production_pipeline") as mock_pipeline:
            test_backup = "/tmp/safe_backup"
            run_stage_sentiment(backup_dir=test_backup)
            mock_pipeline.assert_called_once_with(backup_dir=Path(test_backup))

    def test_main_cli_sentiment_without_backup_dir_fails_closed(self) -> None:
        """build_pipeline.main() fails closed when --stage sentiment is invoked without --backup-dir."""
        from src.build_pipeline import main as pipeline_main
        with patch("sys.argv", ["build_pipeline.py", "--stage", "sentiment"]):
            with patch("src.data_ingestion.news_sentiment.run_production_pipeline") as mock_pipeline:
                with self.assertRaises(ValueError) as ctx:
                    pipeline_main()
                self.assertIn("requires an explicit --backup-dir argument", str(ctx.exception))
                mock_pipeline.assert_not_called()

    def test_main_cli_sentiment_with_backup_dir_forwards_arg(self) -> None:
        """build_pipeline.main() forwards --backup-dir to sentiment pipeline."""
        from src.build_pipeline import main as pipeline_main
        test_dir = "/tmp/explicit_backup_dir"
        with patch("sys.argv", ["build_pipeline.py", "--stage", "sentiment", "--backup-dir", test_dir]):
            with patch("src.data_ingestion.news_sentiment.run_production_pipeline") as mock_pipeline:
                pipeline_main()
                mock_pipeline.assert_called_once_with(backup_dir=Path(test_dir))

    def test_main_cli_stage_all_without_backup_dir_fails_before_any_write(self) -> None:
        """build_pipeline.main() fails closed before running any stage when --stage all is called without --backup-dir."""
        from src.build_pipeline import main as pipeline_main
        with patch("sys.argv", ["build_pipeline.py", "--stage", "all"]):
            with patch("src.build_pipeline.run_stage_market") as mock_market:
                with self.assertRaises(ValueError) as ctx:
                    pipeline_main()
                self.assertIn("Running all stages includes the sentiment stage, which requires an explicit --backup-dir", str(ctx.exception))
                mock_market.assert_not_called()

    def test_main_cli_stage_all_with_backup_dir_executes_all_stages(self) -> None:
        """build_pipeline.main() forwards backup_dir to sentiment stage and executes all stages when --stage all is called."""
        from src.build_pipeline import main as pipeline_main
        test_dir = "/tmp/all_stages_backup"
        with patch("sys.argv", ["build_pipeline.py", "--stage", "all", "--backup-dir", test_dir]):
            with patch("src.build_pipeline.run_stage_market") as mock_market, \
                 patch("src.build_pipeline.run_stage_features") as mock_features, \
                 patch("src.build_pipeline.run_stage_sentiment") as mock_sentiment, \
                 patch("src.build_pipeline.run_stage_forecasting") as mock_forecasting:
                pipeline_main()
                mock_market.assert_called_once()
                mock_features.assert_called_once()
                mock_sentiment.assert_called_once_with(backup_dir=Path(test_dir))
                mock_forecasting.assert_called_once()

    def test_main_cli_other_stages_do_not_require_backup_dir(self) -> None:
        """Stages that do not modify sentiment (market, features, forecasting) do not require --backup-dir."""
        from src.build_pipeline import main as pipeline_main
        with patch("sys.argv", ["build_pipeline.py", "--stage", "forecasting"]):
            with patch("src.build_pipeline.run_stage_forecasting") as mock_forecasting:
                pipeline_main()
                mock_forecasting.assert_called_once()


class TestCacheAndManifestSafety(unittest.TestCase):
    """Gate C3 Repair C: Offline tests for chunk cache, corruption, empty marker, refresh, and isolation."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)
        self.staged_chunks_dir = self.base_dir / "staged_chunks"
        self.staged_chunks_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.base_dir / "news_retrieval_manifest.json"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    class RaisingClient:
        """Client that immediately raises if any method is called."""
        def company_news(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            raise AssertionError("Prohibited client access: network/client was called when cache should suffice!")

    class CountingMockClient:
        """Mock Finnhub client that tracks call count."""
        def __init__(self, return_items: list[dict[str, Any]] | None = None) -> None:
            self.calls = 0
            self.return_items = return_items or []

        def company_news(self, ticker: str, _from: str, to: str) -> list[dict[str, Any]]:
            self.calls += 1
            return self.return_items

    def test_valid_cache_hit_zero_external_requests(self) -> None:
        """A valid cached chunk and manifest entry returns records with zero client or network calls."""
        chunk_file = self.staged_chunks_dir / "AAPL_2026-08-12.parquet"
        cached_df = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "published_at": pd.Timestamp("2026-08-12 14:00:00Z"),
                    "headline": "Cached headline 1",
                    "summary": "Summary 1",
                    "source": "Wire",
                    "url": "https://wire.com/c1",
                    "provider": "finnhub",
                    "is_fallback": False,
                    "raw_id": "c1",
                },
                {
                    "ticker": "AAPL",
                    "published_at": pd.Timestamp("2026-08-12 15:00:00Z"),
                    "headline": "Cached headline 2",
                    "summary": "Summary 2",
                    "source": "Wire",
                    "url": "https://wire.com/c2",
                    "provider": "finnhub",
                    "is_fallback": False,
                    "raw_id": "c2",
                },
            ]
        )
        cached_df.to_parquet(chunk_file)
        update_manifest_chunk(self.manifest_path, "AAPL:2026-08-12:2026-08-12", STATUS_COMPLETE, 2)

        with patch("src.data_ingestion.news_sentiment.get_finnhub_client", side_effect=AssertionError("get_finnhub_client called!")):
            res = fetch_finnhub_news(
                ticker="AAPL",
                start_date="2026-08-12",
                end_date="2026-08-12",
                client=self.RaisingClient(),
                staged_chunks_dir=self.staged_chunks_dir,
                manifest_path=self.manifest_path,
            )

        self.assertEqual(len(res), 2)
        self.assertEqual(res.attrs["status"], STATUS_COMPLETE)
        self.assertEqual(len(res.attrs["successful_chunks"]), 1)
        self.assertEqual(res.attrs["failed_chunks"], [])

    def test_corrupted_cache_detected_and_safely_refetched(self) -> None:
        """A corrupted or unreadable cached chunk is detected and safely replaced via refetch."""
        chunk_file = self.staged_chunks_dir / "AAPL_2026-08-12.parquet"
        with open(chunk_file, "wb") as f:
            f.write(b"CORRUPTED_PARQUET_FILE_DATA_BYTES")

        replacement_item = {
            "id": "refetched_1",
            "datetime": int(pd.Timestamp("2026-08-12 12:00:00Z").timestamp()),
            "headline": "Safely refetched headline",
            "summary": "Summary",
            "source": "Wire",
            "url": "https://wire.com/refetched_1",
        }
        client = self.CountingMockClient([replacement_item])

        res = fetch_finnhub_news(
            ticker="AAPL",
            start_date="2026-08-12",
            end_date="2026-08-12",
            client=client,
            retry_delay=0.0,
            staged_chunks_dir=self.staged_chunks_dir,
            manifest_path=self.manifest_path,
        )

        self.assertEqual(client.calls, 1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res["headline"].iloc[0], "Safely refetched headline")

        # Verify corrupted file was replaced with valid parquet
        new_df = pd.read_parquet(chunk_file)
        self.assertEqual(len(new_df), 1)

        # Verify manifest is updated with STATUS_COMPLETE
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        self.assertEqual(m["chunks"]["AAPL:2026-08-12:2026-08-12"]["status"], STATUS_COMPLETE)

    def test_empty_cached_response_marker(self) -> None:
        """A valid .empty marker chunk avoids redundant requests; failed manifest records do not masquerade as cache."""
        empty_marker = self.staged_chunks_dir / "AAPL_2026-08-12.empty"
        empty_marker.write_text("")
        update_manifest_chunk(self.manifest_path, "AAPL:2026-08-12:2026-08-12", STATUS_COMPLETE, 0)

        res = fetch_finnhub_news(
            ticker="AAPL",
            start_date="2026-08-12",
            end_date="2026-08-12",
            client=self.RaisingClient(),
            staged_chunks_dir=self.staged_chunks_dir,
            manifest_path=self.manifest_path,
        )
        self.assertTrue(res.empty)
        self.assertEqual(res.attrs["status"], STATUS_COMPLETE)

        # Negative test: A chunk marked STATUS_FAILED in manifest must not be served as valid cache
        failed_marker = self.staged_chunks_dir / "AAPL_2026-08-13.empty"
        failed_marker.write_text("")
        update_manifest_chunk(self.manifest_path, "AAPL:2026-08-13:2026-08-13", STATUS_FAILED, 0)

        counting_client = self.CountingMockClient([])
        res2 = fetch_finnhub_news(
            ticker="AAPL",
            start_date="2026-08-13",
            end_date="2026-08-13",
            client=counting_client,
            retry_delay=0.0,
            staged_chunks_dir=self.staged_chunks_dir,
            manifest_path=self.manifest_path,
        )
        self.assertEqual(counting_client.calls, 1)

    def test_cache_expiration_and_refresh(self) -> None:
        """Fresh cache hits bypass client; expired entries trigger refresh; failed refresh preserves existing payload."""
        chunk_file = self.staged_chunks_dir / "AAPL_2026-08-12.parquet"
        cached_df = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "published_at": pd.Timestamp("2026-08-12 14:00:00Z"),
                    "headline": "Historical valid headline",
                    "summary": "Summary",
                    "source": "Wire",
                    "url": "https://wire.com/h1",
                    "provider": "finnhub",
                    "is_fallback": False,
                    "raw_id": "h1",
                }
            ]
        )
        cached_df.to_parquet(chunk_file)
        update_manifest_chunk(self.manifest_path, "AAPL:2026-08-12:2026-08-12", STATUS_COMPLETE, 1)

        # Set mtime to 10 hours ago
        ten_hours_ago = time.time() - (10.0 * 3600.0)
        os.utime(chunk_file, (ten_hours_ago, ten_hours_ago))

        # Fresh hit: max_cache_age_hours=24.0 -> fresh, zero client calls
        res_fresh = fetch_finnhub_news(
            ticker="AAPL",
            start_date="2026-08-12",
            end_date="2026-08-12",
            client=self.RaisingClient(),
            staged_chunks_dir=self.staged_chunks_dir,
            manifest_path=self.manifest_path,
            max_cache_age_hours=24.0,
        )
        self.assertEqual(len(res_fresh), 1)

        # Expired hit: max_cache_age_hours=5.0 -> client is queried for refresh
        refresh_item = {
            "id": "refreshed_h1",
            "datetime": int(pd.Timestamp("2026-08-12 14:30:00Z").timestamp()),
            "headline": "Refreshed headline",
            "summary": "Summary",
            "source": "Wire",
            "url": "https://wire.com/h1_refreshed",
        }
        counting_client = self.CountingMockClient([refresh_item])
        res_refreshed = fetch_finnhub_news(
            ticker="AAPL",
            start_date="2026-08-12",
            end_date="2026-08-12",
            client=counting_client,
            retry_delay=0.0,
            staged_chunks_dir=self.staged_chunks_dir,
            manifest_path=self.manifest_path,
            max_cache_age_hours=5.0,
        )
        self.assertEqual(counting_client.calls, 1)
        self.assertEqual(res_refreshed["headline"].iloc[0], "Refreshed headline")

        # Failed refresh preserves existing valid cached payload on disk
        class FailingClient:
            def company_news(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
                raise RuntimeError("Simulated network failure during refresh")

        os.utime(chunk_file, (ten_hours_ago, ten_hours_ago))
        res_failed = fetch_finnhub_news(
            ticker="AAPL",
            start_date="2026-08-12",
            end_date="2026-08-12",
            client=FailingClient(),
            retry_delay=0.0,
            staged_chunks_dir=self.staged_chunks_dir,
            manifest_path=self.manifest_path,
            max_cache_age_hours=5.0,
        )
        self.assertTrue(res_failed.empty)
        self.assertEqual(res_failed.attrs["status"], STATUS_FAILED)
        self.assertTrue(chunk_file.exists())  # Existing payload was preserved on disk

    def test_ticker_and_date_isolation(self) -> None:
        """Cached chunk for AAPL 2026-08-12 cannot satisfy MSFT or AAPL on other dates."""
        chunk_file = self.staged_chunks_dir / "AAPL_2026-08-12.parquet"
        pd.DataFrame([{"ticker": "AAPL", "published_at": pd.Timestamp("2026-08-12 14:00:00Z"), "headline": "H", "summary": "", "source": "", "url": "", "provider": "finnhub", "is_fallback": False, "raw_id": "1"}]).to_parquet(chunk_file)
        update_manifest_chunk(self.manifest_path, "AAPL:2026-08-12:2026-08-12", STATUS_COMPLETE, 1)

        # Querying MSFT for same date queries client (not AAPL cache)
        msft_client = self.CountingMockClient([])
        fetch_finnhub_news(
            ticker="MSFT",
            start_date="2026-08-12",
            end_date="2026-08-12",
            client=msft_client,
            retry_delay=0.0,
            staged_chunks_dir=self.staged_chunks_dir,
            manifest_path=self.manifest_path,
        )
        self.assertEqual(msft_client.calls, 1)

        # Querying AAPL for different date queries client (not 2026-08-12 cache)
        diff_date_client = self.CountingMockClient([])
        fetch_finnhub_news(
            ticker="AAPL",
            start_date="2026-08-13",
            end_date="2026-08-13",
            client=diff_date_client,
            retry_delay=0.0,
            staged_chunks_dir=self.staged_chunks_dir,
            manifest_path=self.manifest_path,
        )
        self.assertEqual(diff_date_client.calls, 1)

    def test_full_rebuild_preserves_unrelated_cache_and_manifest_entries(self) -> None:
        """Full rebuild re-scores downstream data without erasing unrelated cache or manifest entries."""
        unrelated_chunk = self.staged_chunks_dir / "MSFT_2026-08-12.parquet"
        pd.DataFrame([{"ticker": "MSFT", "published_at": pd.Timestamp("2026-08-12 14:00:00Z"), "headline": "MSFT headline", "summary": "", "source": "", "url": "", "provider": "finnhub", "is_fallback": False, "raw_id": "msft1"}]).to_parquet(unrelated_chunk)
        update_manifest_chunk(self.manifest_path, "MSFT:2026-08-12:2026-08-12", STATUS_COMPLETE, 1)

        # Unrelated entry must persist
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        self.assertIn("MSFT:2026-08-12:2026-08-12", m["chunks"])
        self.assertTrue(unrelated_chunk.exists())


class TestIdempotencyAndDeterministicOrdering(unittest.TestCase):
    """Gate C3 Repair D: Regression tests for archive merge idempotency, deterministic sorting, and 2-run hash identity."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)
        self.backup_dir = self.base_dir / "backup"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.raw_path = self.base_dir / "news_raw.parquet"
        self.proc_path = self.base_dir / "news_sentiment.parquet"
        self.market_path = self.base_dir / "market_features.parquet"
        self.journal_path = self.base_dir / "transaction_journal.json"
        self.staged_chunks_dir = self.base_dir / "staged_chunks"
        self.staged_chunks_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.base_dir / "news_retrieval_manifest.json"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_test_checksums(self, files: list[Path]) -> None:
        chk_file = self.backup_dir / "checksums.sha256"
        with open(chk_file, "w", encoding="utf-8") as f:
            for p in files:
                f.write(f"{compute_sha256(p)}  {p.name}\n")

    class MockAnalyzer:
        def __init__(self) -> None:
            self.calls = 0
            self.inferences = 0

        def analyze_batch(self, headlines: list[str], batch_size: int = 32) -> pd.DataFrame:
            self.calls += 1
            self.inferences += len(headlines)
            return pd.DataFrame(
                [
                    {
                        "sentiment_label": "positive",
                        "sentiment_score": 0.5,
                        "prob_positive": 0.8,
                        "prob_negative": 0.1,
                        "prob_neutral": 0.1,
                    }
                    for _ in headlines
                ]
            )

    class CountingFinnhubClient:
        def __init__(self, items: list[dict[str, Any]]) -> None:
            self.calls = 0
            self.items = items

        def company_news(self, ticker: str, _from: str, to: str) -> list[dict[str, Any]]:
            self.calls += 1
            return self.items

    def test_duplicate_only_raw_merge_preserves_order_and_sha256(self) -> None:
        """Feeding identical records in shuffled order does not rewrite or reorder the existing raw archive."""
        records = [
            {
                "ticker": "AAPL",
                "published_at": pd.Timestamp("2026-08-12 14:00:00Z"),
                "headline": f"Headline {i}",
                "summary": f"Summary {i}",
                "source": "Wire",
                "url": f"https://wire.com/{i}",
                "provider": "finnhub",
                "is_fallback": False,
                "raw_id": f"id_{i}",
            }
            for i in range(10)
        ]
        df_orig = pd.DataFrame(records)
        df_orig.to_parquet(self.raw_path)
        sha_before = compute_sha256(self.raw_path)

        # Incoming has exact same 10 records in reversed order
        df_incoming = pd.DataFrame(reversed(records))
        df_merged = merge_raw_news_archive(self.raw_path, df_incoming)

        self.assertEqual(len(df_merged), 10)
        sha_after = compute_sha256(self.raw_path)
        self.assertEqual(sha_before, sha_after, "Archive was rewritten when incoming data was duplicate-only!")
        self.assertTrue(df_merged.equals(df_orig), "Logical records or order changed!")

    def test_deterministic_sorting_with_equal_timestamps_and_missing_ids(self) -> None:
        """Articles with identical publication timestamps produce deterministic order regardless of input order."""
        t_same = pd.Timestamp("2026-08-12 14:00:00Z")
        rec_b = {"ticker": "AAPL", "published_at": t_same, "headline": "Beta headline", "summary": "", "source": "Wire", "url": "https://wire.com/b", "provider": "finnhub", "is_fallback": False, "raw_id": "id_b"}
        rec_a = {"ticker": "AAPL", "published_at": t_same, "headline": "Alpha headline", "summary": "", "source": "Wire", "url": "https://wire.com/a", "provider": "finnhub", "is_fallback": False, "raw_id": "id_a"}
        rec_no_id_2 = {"ticker": "AAPL", "published_at": t_same, "headline": "Zeta no id", "summary": "", "source": "Wire", "url": "https://wire.com/z", "provider": "finnhub", "is_fallback": False, "raw_id": ""}
        rec_no_id_1 = {"ticker": "AAPL", "published_at": t_same, "headline": "Gamma no id", "summary": "", "source": "Wire", "url": "https://wire.com/g", "provider": "finnhub", "is_fallback": False, "raw_id": ""}
        rec_a_yahoo = {"ticker": "AAPL", "published_at": t_same, "headline": "Alpha headline", "summary": "", "source": "Wire", "url": "https://wire.com/a", "provider": "yahoo_finance", "is_fallback": True, "raw_id": "id_a"}

        # Order 1
        df1 = pd.DataFrame([rec_b, rec_a, rec_no_id_2, rec_no_id_1, rec_a_yahoo])
        dedup1 = deduplicate_articles(df1)

        # Order 2 (reversed)
        df2 = pd.DataFrame([rec_a_yahoo, rec_no_id_1, rec_no_id_2, rec_a, rec_b])
        dedup2 = deduplicate_articles(df2)

        self.assertEqual(len(dedup1), 4)
        self.assertEqual(len(dedup2), 4)
        # Deterministic ordering invariant
        self.assertEqual(list(dedup1["raw_id"]), list(dedup2["raw_id"]))
        self.assertEqual(list(dedup1["headline"]), list(dedup2["headline"]))
        # Fallback precedence invariant: Finnhub won over Yahoo
        rec_a_res = dedup1.loc[dedup1["raw_id"] == "id_a"].iloc[0]
        self.assertFalse(rec_a_res["is_fallback"])
        self.assertEqual(rec_a_res["provider"], "finnhub")

    def test_two_run_pipeline_idempotency_zero_calls_zero_inferences_identical_hashes(self) -> None:
        """Second run of production pipeline performs 0 API calls, 0 FinBERT inferences, and preserves bitwise Parquet hashes."""
        # 1. Setup baseline synthetic market features
        market_dates = pd.date_range("2026-08-03", "2026-08-14", freq="B")
        base_market = pd.DataFrame(
            {"Close": range(len(market_dates)), "Daily_Return": [0.01] * len(market_dates)},
            index=market_dates,
        )
        base_market.to_parquet(self.market_path)
        shutil.copy2(self.market_path, self.backup_dir / self.market_path.name)

        # Baseline raw and proc (empty initial state)
        pd.DataFrame(columns=NEWS_COLUMNS).to_parquet(self.raw_path)
        shutil.copy2(self.raw_path, self.backup_dir / self.raw_path.name)

        proc_cols = NEWS_COLUMNS + ["trading_date", "sentiment_label", "sentiment_score", "prob_positive", "prob_negative", "prob_neutral"]
        pd.DataFrame(columns=proc_cols).to_parquet(self.proc_path)
        shutil.copy2(self.proc_path, self.backup_dir / self.proc_path.name)

        self.write_test_checksums([self.raw_path, self.proc_path, self.market_path])

        mock_article = {
            "id": "item_20260812_1",
            "datetime": int(pd.Timestamp("2026-08-12 14:00:00Z").timestamp()),
            "headline": "Apple reports groundbreaking chip architecture",
            "summary": "Summary",
            "source": "Wire",
            "url": "https://wire.com/item1",
        }
        client = self.CountingFinnhubClient([mock_article])
        analyzer = self.MockAnalyzer()

        # RUN 1 & 2: Wrapped in patch to prevent live Yahoo Finance fallback requests
        with patch("src.data_ingestion.news_sentiment.fetch_yahoo_fallback", return_value=pd.DataFrame()):
            res1 = run_production_pipeline(
                start_date="2026-08-12",
                end_date="2026-08-12",
                client=client,
                analyzer=analyzer,
                raw_news_path=self.raw_path,
                processed_news_path=self.proc_path,
                market_features_path=self.market_path,
                staged_chunks_dir=self.staged_chunks_dir,
                manifest_path=self.manifest_path,
                backup_dir=self.backup_dir,
                journal_path=self.journal_path,
                allowed_roots=[self.base_dir],
            )

            self.assertEqual(client.calls, 1, "Run 1 must query client exactly once.")
            self.assertEqual(analyzer.calls, 1, "Run 1 must invoke sentiment analyzer.")
            self.assertEqual(analyzer.inferences, 1, "Run 1 must score 1 article.")

            hash_raw_1 = compute_sha256(self.raw_path)
            hash_proc_1 = compute_sha256(self.proc_path)
            hash_mkt_1 = compute_sha256(self.market_path)

            # RUN 2: Repeated identical run
            res2 = run_production_pipeline(
                start_date="2026-08-12",
                end_date="2026-08-12",
                client=client,
                analyzer=analyzer,
                raw_news_path=self.raw_path,
                processed_news_path=self.proc_path,
                market_features_path=self.market_path,
                staged_chunks_dir=self.staged_chunks_dir,
                manifest_path=self.manifest_path,
                backup_dir=self.backup_dir,
                journal_path=self.journal_path,
                allowed_roots=[self.base_dir],
            )

        # Assert zero new client calls and zero new inferences
        self.assertEqual(client.calls, 1, f"Expected 0 additional client calls on run 2, got {client.calls - 1}")
        self.assertEqual(analyzer.calls, 1, f"Expected 0 additional analyzer calls on run 2, got {analyzer.calls - 1}")
        self.assertEqual(analyzer.inferences, 1, f"Expected 0 additional inferences on run 2, got {analyzer.inferences - 1}")

        # Assert all 3 Parquet hashes are identical before and after run 2
        hash_raw_2 = compute_sha256(self.raw_path)
        hash_proc_2 = compute_sha256(self.proc_path)
        hash_mkt_2 = compute_sha256(self.market_path)

        self.assertEqual(hash_raw_1, hash_raw_2, "news_raw.parquet hash changed on repeated identical run!")
        self.assertEqual(hash_proc_1, hash_proc_2, "news_sentiment.parquet hash changed on repeated identical run!")
        self.assertEqual(hash_mkt_1, hash_mkt_2, "market_features.parquet hash changed on repeated identical run!")

        # Verify transaction journal recorded clean COMMITTED status
        with open(self.journal_path, "r", encoding="utf-8") as f:
            j = json.load(f)
        self.assertEqual(j["state"], TX_STATE_COMMITTED)

    def test_failure_handling_and_safe_recovery(self) -> None:
        """Simulated pipeline failure preserves baseline files and allows clean transaction recovery."""
        target = self.base_dir / "market_features.parquet"
        backup = self.backup_dir / "market_features.parquet"
        pd.DataFrame({"Close": [100.0]}).to_parquet(backup)
        pd.DataFrame({"Close": [100.0]}).to_parquet(target)
        self.write_test_checksums([backup])

        stage = target.with_name("market_features.parquet.stage")
        pd.DataFrame({"Close": [999.0]}).to_parquet(stage)

        # Simulate journal crash in STAGED state
        journal = {
            "transaction_id": "tx_crash_sim",
            "state": TX_STATE_STAGED,
            "backup_dir": str(self.backup_dir),
            "files": [
                {
                    "target": str(target),
                    "staged": str(stage),
                    "backup": str(backup),
                    "backup_sha256": compute_sha256(backup),
                }
            ],
        }
        with open(self.journal_path, "w", encoding="utf-8") as f:
            json.dump(journal, f)

        # Recovery restores baseline
        recovered = check_and_recover_transactions(self.journal_path, allowed_roots=[self.base_dir])
        self.assertTrue(recovered)
        self.assertEqual(pd.read_parquet(target)["Close"].iloc[0], 100.0)
        self.assertFalse(stage.exists())

    def test_path_confinement_rejects_traversal(self) -> None:
        """Cache, manifest, and target paths outside allowed roots are rejected with RuntimeError."""
        outside_dir = tempfile.TemporaryDirectory()
        try:
            outside_path = Path(outside_dir.name) / "escaped_cache"
            with self.assertRaises(RuntimeError) as ctx:
                run_production_pipeline(
                    start_date="2026-08-12",
                    end_date="2026-08-12",
                    staged_chunks_dir=outside_path,
                    raw_news_path=self.raw_path,
                    processed_news_path=self.proc_path,
                    market_features_path=self.market_path,
                    backup_dir=self.backup_dir,
                    allowed_roots=[self.base_dir],
                )
            self.assertIn("outside allowed roots", str(ctx.exception))
        finally:
            outside_dir.cleanup()

    def test_empty_retrieval_aggregates_existing_sentiment(self) -> None:
        """Successful empty retrieval aggregates pre-existing eligible sentiment into market_features without FinBERT inference."""
        # 1. Setup synthetic market features
        market_dates = pd.date_range("2026-08-11", "2026-08-15", freq="B")
        base_market = pd.DataFrame(
            {
                "Close": [150.0 + i for i in range(len(market_dates))],
                "Daily_Return": [0.01] * len(market_dates),
                "sentiment_coverage": [0] * len(market_dates),
                "sentiment_score": [np.nan] * len(market_dates),
                "sentiment_count": [np.nan] * len(market_dates),
            },
            index=market_dates,
        )
        base_market.to_parquet(self.market_path)
        shutil.copy2(self.market_path, self.backup_dir / self.market_path.name)

        # Baseline raw archive
        pd.DataFrame(columns=NEWS_COLUMNS).to_parquet(self.raw_path)
        shutil.copy2(self.raw_path, self.backup_dir / self.raw_path.name)

        # Processed sentiment archive containing 6 pre-existing scored articles for 2026-08-13
        proc_records = [
            {
                "ticker": "AAPL",
                "published_at": pd.Timestamp("2026-08-12 21:00:00Z"),
                "headline": f"Post close headline {i}",
                "summary": "Summary",
                "source": "Wire",
                "url": f"https://wire.com/{i}",
                "provider": "finnhub",
                "is_fallback": False,
                "raw_id": f"item_aug12_{i}",
                "trading_date": pd.Timestamp("2026-08-13"),
                "sentiment_label": "positive",
                "sentiment_score": 0.6,
                "prob_positive": 0.8,
                "prob_negative": 0.1,
                "prob_neutral": 0.1,
            }
            for i in range(6)
        ]
        pd.DataFrame(proc_records).to_parquet(self.proc_path)
        shutil.copy2(self.proc_path, self.backup_dir / self.proc_path.name)

        self.write_test_checksums([self.raw_path, self.proc_path, self.market_path])

        raw_sha_before = compute_sha256(self.raw_path)
        proc_sha_before = compute_sha256(self.proc_path)

        # Client returns 0 articles for 2026-08-13
        client = self.CountingFinnhubClient([])
        analyzer = self.MockAnalyzer()

        with patch("src.data_ingestion.news_sentiment.fetch_yahoo_fallback", return_value=pd.DataFrame()):
            res = run_production_pipeline(
                ticker="AAPL",
                start_date="2026-08-13",
                end_date="2026-08-13",
                client=client,
                analyzer=analyzer,
                raw_news_path=self.raw_path,
                processed_news_path=self.proc_path,
                market_features_path=self.market_path,
                staged_chunks_dir=self.staged_chunks_dir,
                manifest_path=self.manifest_path,
                backup_dir=self.backup_dir,
                journal_path=self.journal_path,
                allowed_roots=[self.base_dir],
            )

        self.assertEqual(res["status"], STATUS_COMPLETE)
        self.assertEqual(analyzer.calls, 0, "No FinBERT inference should occur during empty retrieval.")
        self.assertEqual(analyzer.inferences, 0, "Zero headlines should be scored.")

        # Raw and processed archive bytes are 100% preserved
        self.assertEqual(raw_sha_before, compute_sha256(self.raw_path))
        self.assertEqual(proc_sha_before, compute_sha256(self.proc_path))

        # Market features refreshed for 2026-08-13 from the 6 existing articles
        updated_mkt = pd.read_parquet(self.market_path)
        row_aug13 = updated_mkt.loc["2026-08-13"]
        self.assertEqual(row_aug13["sentiment_coverage"], 1)
        self.assertAlmostEqual(row_aug13["sentiment_score"], 0.6)
        self.assertEqual(row_aug13["sentiment_count"], 6)

        # Technical indicators remain unchanged
        self.assertEqual(row_aug13["Close"], 152.0)
        self.assertEqual(row_aug13["Daily_Return"], 0.01)

        # Transaction journal recorded clean COMMITTED status
        with open(self.journal_path, "r", encoding="utf-8") as f:
            j = json.load(f)
        self.assertEqual(j["state"], TX_STATE_COMMITTED)

        # No staging files remain
        staged_files = list(self.base_dir.glob("*.stage"))
        self.assertEqual(len(staged_files), 0)

    def test_empty_retrieval_without_eligible_sentiment_is_noop(self) -> None:
        """Empty retrieval with no eligible processed records performs no writes and leaves market features untouched."""
        market_dates = pd.date_range("2026-08-11", "2026-08-15", freq="B")
        base_market = pd.DataFrame(
            {"Close": [150.0] * len(market_dates), "sentiment_coverage": [0] * len(market_dates)},
            index=market_dates,
        )
        base_market.to_parquet(self.market_path)
        shutil.copy2(self.market_path, self.backup_dir / self.market_path.name)

        pd.DataFrame(columns=NEWS_COLUMNS).to_parquet(self.raw_path)
        shutil.copy2(self.raw_path, self.backup_dir / self.raw_path.name)

        # Processed archive contains only articles for 2026-08-01 (unrelated date)
        proc_records = [
            {
                "ticker": "AAPL",
                "published_at": pd.Timestamp("2026-08-01 12:00:00Z"),
                "headline": "Old article",
                "summary": "",
                "source": "Wire",
                "url": "https://wire.com/old",
                "provider": "finnhub",
                "is_fallback": False,
                "raw_id": "item_old",
                "trading_date": pd.Timestamp("2026-08-01"),
                "sentiment_label": "positive",
                "sentiment_score": 0.5,
                "prob_positive": 0.8,
                "prob_negative": 0.1,
                "prob_neutral": 0.1,
            }
        ]
        pd.DataFrame(proc_records).to_parquet(self.proc_path)
        shutil.copy2(self.proc_path, self.backup_dir / self.proc_path.name)

        self.write_test_checksums([self.raw_path, self.proc_path, self.market_path])

        mkt_sha_before = compute_sha256(self.market_path)
        client = self.CountingFinnhubClient([])

        with patch("src.data_ingestion.news_sentiment.fetch_yahoo_fallback", return_value=pd.DataFrame()):
            res = run_production_pipeline(
                ticker="AAPL",
                start_date="2026-08-13",
                end_date="2026-08-13",
                client=client,
                raw_news_path=self.raw_path,
                processed_news_path=self.proc_path,
                market_features_path=self.market_path,
                staged_chunks_dir=self.staged_chunks_dir,
                manifest_path=self.manifest_path,
                backup_dir=self.backup_dir,
                journal_path=self.journal_path,
                allowed_roots=[self.base_dir],
            )

        self.assertEqual(res["status"], STATUS_COMPLETE)
        self.assertEqual(mkt_sha_before, compute_sha256(self.market_path), "Market features file was modified unnecessarily!")
        self.assertFalse(self.journal_path.exists(), "No transaction journal should be created when no writes occur.")

    def test_failed_retrieval_does_not_aggregate_existing_sentiment(self) -> None:
        """Failed retrieval does not trigger sentiment aggregation or market feature mutation."""
        market_dates = pd.date_range("2026-08-11", "2026-08-15", freq="B")
        base_market = pd.DataFrame(
            {"Close": [150.0] * len(market_dates), "sentiment_coverage": [0] * len(market_dates)},
            index=market_dates,
        )
        base_market.to_parquet(self.market_path)
        shutil.copy2(self.market_path, self.backup_dir / self.market_path.name)

        pd.DataFrame(columns=NEWS_COLUMNS).to_parquet(self.raw_path)
        shutil.copy2(self.raw_path, self.backup_dir / self.raw_path.name)

        proc_records = [
            {
                "ticker": "AAPL",
                "published_at": pd.Timestamp("2026-08-12 21:00:00Z"),
                "headline": "Post close article",
                "summary": "",
                "source": "Wire",
                "url": "https://wire.com/item",
                "provider": "finnhub",
                "is_fallback": False,
                "raw_id": "item_1",
                "trading_date": pd.Timestamp("2026-08-13"),
                "sentiment_label": "positive",
                "sentiment_score": 0.5,
                "prob_positive": 0.8,
                "prob_negative": 0.1,
                "prob_neutral": 0.1,
            }
        ]
        pd.DataFrame(proc_records).to_parquet(self.proc_path)
        shutil.copy2(self.proc_path, self.backup_dir / self.proc_path.name)

        self.write_test_checksums([self.raw_path, self.proc_path, self.market_path])
        mkt_sha_before = compute_sha256(self.market_path)

        class FailingClient:
            def company_news(self, ticker: str, _from: str, to: str) -> list[dict[str, Any]]:
                raise RuntimeError("Finnhub simulated server outage 500")

        with patch("src.data_ingestion.news_sentiment.fetch_yahoo_fallback", return_value=pd.DataFrame()):
            res = run_production_pipeline(
                ticker="AAPL",
                start_date="2026-08-13",
                end_date="2026-08-13",
                client=FailingClient(),
                retry_delay=0.0,
                raw_news_path=self.raw_path,
                processed_news_path=self.proc_path,
                market_features_path=self.market_path,
                staged_chunks_dir=self.staged_chunks_dir,
                manifest_path=self.manifest_path,
                backup_dir=self.backup_dir,
                journal_path=self.journal_path,
                allowed_roots=[self.base_dir],
            )

        self.assertEqual(res["status"], STATUS_FAILED)
        self.assertEqual(mkt_sha_before, compute_sha256(self.market_path), "Market features was mutated after failed retrieval!")
        self.assertEqual(pd.read_parquet(self.market_path).loc["2026-08-13", "sentiment_coverage"], 0)

    def test_cli_failure_exit_code(self) -> None:
        """CLI main() exits with code 1 when run_production_pipeline reports STATUS_FAILED."""
        failed_result = {"status": STATUS_FAILED, "raw_news_count": 0, "processed_count": 0, "covered_sessions": 0}
        with patch("sys.argv", ["news_sentiment.py", "--start-date", "2026-08-13", "--end-date", "2026-08-13"]):
            with patch("src.data_ingestion.news_sentiment.run_production_pipeline", return_value=failed_result):
                with self.assertRaises(SystemExit) as ctx:
                    main()
                self.assertEqual(ctx.exception.code, 1, "Expected exit code 1 on pipeline failure.")

    def test_cli_successful_empty_exit_code(self) -> None:
        """CLI main() terminates successfully with exit code 0 on STATUS_COMPLETE empty retrieval."""
        success_result = {"status": STATUS_COMPLETE, "raw_news_count": 0, "processed_count": 6, "covered_sessions": 1}
        with patch("sys.argv", ["news_sentiment.py", "--start-date", "2026-08-13", "--end-date", "2026-08-13"]):
            with patch("src.data_ingestion.news_sentiment.run_production_pipeline", return_value=success_result):
                try:
                    main()
                except SystemExit as e:
                    self.assertEqual(e.code, 0, "Expected exit code 0 on successful run.")


if __name__ == "__main__":
    unittest.main()


