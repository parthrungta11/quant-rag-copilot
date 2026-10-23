"""News Sentiment Ingestion and Processing Pipeline for Quant RAG Copilot.

This module fetches financial news for a specified ticker (AAPL by default) using
Finnhub as primary source with Yahoo Finance as fallback. Headlines are scored
using FinBERT (ProsusAI/finbert). Look-ahead bias is prevented by mapping news
occurring after US equity market close (16:00 ET on standard days, 13:00 ET on early close days)
or on weekends/holidays to the next valid trading session. Daily sentiment aggregations are merged
into market_features.parquet incrementally or via full rebuild while preserving all market rows and
strictly tracking sentiment coverage without fabricating zeroes for missing news periods.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import finnhub
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from dotenv import load_dotenv
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("news_sentiment")

# Directory and file paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

RAW_NEWS_FILE = RAW_DATA_DIR / "news_raw.parquet"
PROCESSED_NEWS_FILE = PROCESSED_DATA_DIR / "news_sentiment.parquet"
MARKET_FEATURES_FILE = PROCESSED_DATA_DIR / "market_features.parquet"

FINBERT_MODEL_NAME = "ProsusAI/finbert"

STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"

NEWS_COLUMNS = [
    "ticker",
    "published_at",
    "headline",
    "summary",
    "source",
    "url",
    "provider",
    "is_fallback",
    "raw_id",
]


def load_api_key() -> str | None:
    """Load Finnhub API key from project root .env without logging secret values."""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file, override=True)
    else:
        load_dotenv(override=True)

    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        logger.warning("FINNHUB_API_KEY not found in environment or .env file.")
        return None
    return api_key.strip()


def get_finnhub_client() -> finnhub.Client | None:
    """Create an authenticated Finnhub client if an API key is available."""
    api_key = load_api_key()
    if not api_key:
        return None
    client = finnhub.Client(api_key=api_key)
    # Ensure reliable base URL in case api.finnhub.io DNS resolution fluctuates
    if hasattr(client, "API_URL") and "api.finnhub.io" in client.API_URL:
        client.API_URL = "https://finnhub.io/api/v1"
    return client


class USEquityMarketHolidayCalendar(AbstractHolidayCalendar):
    """US Equity Market (NYSE/NASDAQ) Holiday Calendar.

    Explicitly excludes Federal-only holidays when exchanges remain OPEN:
    - Columbus Day (Indigenous Peoples' Day)
    - Veterans Day
    Includes exchange-specific holidays:
    - Good Friday
    - Juneteenth (since 2021)
    - New Year's Day, MLK Jr. Day, Presidents' Day, Memorial Day,
      Independence Day, Labor Day, Thanksgiving Day, Christmas Day.
    """

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday(
            "Juneteenth",
            month=6,
            day=19,
            start_date="2021-06-18",
            observance=nearest_workday,
        ),
        Holiday("July 4th", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def get_us_equity_market_holidays(
    start_date: str = "2024-01-01",
    end_date: str = "2027-01-01",
) -> set[dt.date]:
    """Return set of US equity market holidays (NYSE/NASDAQ calendar)."""
    cal = USEquityMarketHolidayCalendar()
    return set(cal.holidays(start=start_date, end=end_date).date)


def get_us_market_holidays(
    start_date: str = "2024-01-01",
    end_date: str = "2027-01-01",
) -> set[dt.date]:
    """Backward-compatible alias returning US equity market holidays."""
    return get_us_equity_market_holidays(start_date=start_date, end_date=end_date)


def is_us_equity_early_close(
    session_date: dt.date,
    holidays: set[dt.date] | None = None,
) -> bool:
    """Determine whether a session is a scheduled 13:00 ET early close.

    US equity exchanges close at 13:00 ET on:
    1. Black Friday (day after Thanksgiving, 4th Friday in November).
    2. Christmas Eve (December 24) when on a weekday and not an observed holiday.
    3. Day before Independence Day (July 3) when July 4 is a weekday.
    """
    if session_date.weekday() >= 5:
        return False
    if holidays is not None and session_date in holidays:
        return False

    # 1. Day after Thanksgiving (Black Friday)
    if session_date.month == 11 and session_date.weekday() == 4:
        yesterday = session_date - dt.timedelta(days=1)
        if 22 <= yesterday.day <= 28:
            return True

    # 2. Christmas Eve (Dec 24) on a weekday
    if session_date.month == 12 and session_date.day == 24:
        return True

    # 3. Day before July 4th (July 3) if July 4 is a weekday
    if session_date.month == 7 and session_date.day == 3:
        july4 = session_date + dt.timedelta(days=1)
        if july4.weekday() < 5:
            return True

    return False


def get_valid_market_trading_days(
    market_features_path: Path = MARKET_FEATURES_FILE,
) -> tuple[set[dt.date], list[dt.date]]:
    """Extract valid historical trading days from market_features.parquet."""
    if not market_features_path.exists():
        logger.warning(
            f"Market features file not found at {market_features_path}. "
            "Will fall back to calendar-based trading day detection."
        )
        return set(), []

    try:
        df = pd.read_parquet(market_features_path)
        normalized_idx = pd.to_datetime(df.index).tz_localize(None).normalize()
        valid_dates = sorted(set(ts.date() for ts in normalized_idx))
        return set(valid_dates), list(valid_dates)
    except Exception as exc:
        logger.warning(
            f"Failed to load trading days from {market_features_path}: {exc}"
        )
        return set(), []


def assign_trading_session(
    published_at_utc: pd.Timestamp,
    valid_trading_days_set: set[dt.date] | None = None,
    sorted_trading_days: list[dt.date] | None = None,
    market_holidays: set[dt.date] | None = None,
) -> pd.Timestamp:
    """Map a news publication timestamp to the corresponding trading session date.

    Prevents look-ahead bias:
    - US Regular Trading Hours end at 16:00:00 (4:00 PM) Eastern Time (or 13:00:00 ET on early close days).
    - News published at or after regular session close is strictly assigned to the next valid trading day.
    - News published before close on a non-trading day (weekend or exchange holiday) is assigned to the next valid trading day.
    - News published before close on an active trading day belongs to that day.
    - Model forecast-time assumption: Return forecasts for t+1 are produced after the close of session t.
      Only information published before session close can be used. Any headline at or after close belongs to t+1.
    - Invariant: An article is NEVER assigned to a trading session that precedes its publication time.
    """
    if pd.isna(published_at_utc):
        raise ValueError("Cannot assign trading session to null timestamp.")

    if published_at_utc.tzinfo is None:
        published_at_utc = published_at_utc.tz_localize("UTC")

    dt_ny = published_at_utc.tz_convert("America/New_York")
    pub_date = dt_ny.date()

    if market_holidays is None:
        market_holidays = get_us_equity_market_holidays()

    # Determine session close cutoff time
    if pub_date.weekday() >= 5 or pub_date in market_holidays:
        cutoff_time = dt.time(0, 0, 0)
        candidate_date = pub_date + dt.timedelta(days=1)
    else:
        cutoff_time = (
            dt.time(13, 0, 0)
            if is_us_equity_early_close(pub_date, market_holidays)
            else dt.time(16, 0, 0)
        )
        if dt_ny.time() >= cutoff_time:
            candidate_date = pub_date + dt.timedelta(days=1)
        else:
            candidate_date = pub_date

    # 1. Prefer actual dates from the available market dataset if within range
    if sorted_trading_days and sorted_trading_days[0] <= candidate_date <= sorted_trading_days[-1]:
        future_trading_days = [d for d in sorted_trading_days if d >= candidate_date]
        if future_trading_days:
            assigned_date = future_trading_days[0]
        else:
            curr = candidate_date
            while curr.weekday() >= 5 or curr in market_holidays:
                curr += dt.timedelta(days=1)
            assigned_date = curr
    elif valid_trading_days_set and candidate_date in valid_trading_days_set:
        assigned_date = candidate_date
    else:
        # Fallback to equity calendar (for dates beyond or before historical dataset)
        curr = candidate_date
        while curr.weekday() >= 5 or curr in market_holidays:
            curr += dt.timedelta(days=1)
        assigned_date = curr

    # Invariant: Never assign news to a session that precedes its publication date
    assert assigned_date >= pub_date, (
        f"Assigned session {assigned_date} precedes publication date {pub_date}"
    )
    if assigned_date == pub_date:
        assert dt_ny.time() < cutoff_time, (
            f"News at {dt_ny.time()} on {pub_date} assigned to same session despite cutoff {cutoff_time}"
        )

    return pd.Timestamp(assigned_date)


def filter_news_by_date_range(
    df: pd.DataFrame,
    start_date: str | dt.date,
    end_date: str | dt.date,
    reference_tz: str = "America/New_York",
) -> pd.DataFrame:
    """Filter news articles to an inclusive calendar date range in the reference timezone.

    Date-boundary convention:
    - start_date and end_date are inclusive calendar dates in reference_tz.
    - Publication timestamps are converted to reference_tz (default 'America/New_York').
    - An article is included iff start_date <= dt_ref.date() <= end_date.
    - Missing (NaT/None) or unparseable timestamps are safely discarded with warning.
    """
    if df.empty:
        return df

    start_d = pd.to_datetime(start_date).date()
    end_d = pd.to_datetime(end_date).date()
    if start_d > end_d:
        raise ValueError(f"start_date ({start_d}) cannot be after end_date ({end_d})")

    if "published_at" not in df.columns:
        logger.warning("DataFrame has no 'published_at' column; cannot filter by date range.")
        return df

    pub_series = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    valid_mask = pub_series.notna()
    if not valid_mask.all():
        num_invalid = int((~valid_mask).sum())
        logger.warning(
            f"Discarding {num_invalid} article(s) with missing or unparseable publication timestamps."
        )

    valid_df = df[valid_mask].copy()
    valid_pub = pub_series[valid_mask]

    # Convert to reference timezone for local calendar date evaluation
    pub_ref = valid_pub.dt.tz_convert(reference_tz)
    pub_dates = pd.Series([ts.date() for ts in pub_ref], index=valid_df.index)

    in_range_mask = (pub_dates >= start_d) & (pub_dates <= end_d)
    filtered = valid_df[in_range_mask].reset_index(drop=True)

    logger.info(
        f"Filtered news by date range [{start_d} to {end_d}] in {reference_tz}: "
        f"{len(df)} input -> {len(filtered)} retained ({len(df) - len(filtered)} discarded)."
    )
    return filtered


def generate_date_chunks(
    start_date: dt.date,
    end_date: dt.date,
    chunk_days: int = 1,
) -> list[tuple[str, str]]:
    """Generate inclusive [from, to] date range strings covering start_date to end_date."""
    if start_date > end_date:
        return []
    chunks: list[tuple[str, str]] = []
    curr = start_date
    while curr <= end_date:
        chunk_end = min(curr + dt.timedelta(days=chunk_days - 1), end_date)
        chunks.append((curr.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        curr = chunk_end + dt.timedelta(days=1)
    return chunks


def deduplicate_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate news articles across chunks using stable IDs, URLs, and normalized headlines."""
    if df.empty:
        return df

    working = df.copy()
    working["published_at"] = pd.to_datetime(working["published_at"], utc=True)
    # Sort so primary Finnhub records (is_fallback=False) come before fallback records
    working = working.sort_values(
        by=["is_fallback", "published_at"],
        ascending=[True, False],
    )

    # 1. Deduplicate by stable non-empty raw_id (Finnhub article ID or Yahoo UUID)
    has_id = working["raw_id"].fillna("").astype(str).str.strip().ne("")
    with_id = working[has_id].drop_duplicates(subset=["raw_id"], keep="first")
    without_id = working[~has_id]
    combined = pd.concat([with_id, without_id], ignore_index=True)

    # 2. Deduplicate by non-empty URL
    combined["_url_clean"] = combined["url"].fillna("").astype(str).str.strip()
    has_url = combined["_url_clean"].ne("")
    with_url = combined[has_url].drop_duplicates(subset=["_url_clean"], keep="first")
    without_url = combined[~has_url]
    combined = pd.concat([with_url, without_url], ignore_index=True)

    # 3. Deduplicate by normalized lowercase headline and published timestamp
    combined["_headline_clean"] = combined["headline"].astype(str).str.lower().str.strip()
    deduped = combined.drop_duplicates(subset=["_headline_clean", "published_at"], keep="first")

    deduped = deduped.drop(columns=["_url_clean", "_headline_clean"], errors="ignore")
    return deduped.sort_values(by="published_at", ascending=False).reset_index(drop=True)


def fetch_finnhub_news(
    ticker: str,
    start_date: str,
    end_date: str,
    client: finnhub.Client | None = None,
    chunk_days: int = 1,
    retry_delay: float = 1.0,
) -> pd.DataFrame:
    """Fetch company news from Finnhub in chunked date intervals with rate-limit handling.

    Uses short date chunks (default 1 calendar day per request) to prevent truncation.
    Finnhub silently truncates responses at ~245-250 items sorted newest-first.

    Distinguishes three collection outcomes via df.attrs["status"]:
    - complete: All requested chunks were retrieved successfully (even if 0 articles).
    - partial: Some chunks failed, but >= 1 chunk succeeded; returns articles from successful chunks.
    - failed: All requested chunks failed after retries.
    """
    if client is None:
        client = get_finnhub_client()
        if client is None:
            raise RuntimeError("Finnhub client could not be initialized.")

    start_dt = pd.to_datetime(start_date).date()
    end_dt = pd.to_datetime(end_date).date()

    articles: list[dict[str, Any]] = []
    chunks = generate_date_chunks(start_dt, end_dt, chunk_days=chunk_days)
    failed_chunks: list[tuple[str, str]] = []
    successful_chunks: list[tuple[str, str]] = []

    logger.info(
        f"Querying Finnhub for {ticker} across {len(chunks)} chunk(s) "
        f"({start_dt} to {end_dt}, chunk_days={chunk_days})..."
    )

    for chunk_idx, (from_str, to_str) in enumerate(chunks, 1):
        retries = 3
        success = False
        batch_items: list[dict[str, Any]] = []

        for attempt in range(retries):
            try:
                batch = client.company_news(ticker, _from=from_str, to=to_str)
                batch_items = batch or []
                success = True
                break
            except Exception as exc:
                clean_err = str(exc).split("?")[0]
                logger.warning(
                    f"Finnhub request failed (attempt {attempt + 1}/{retries}) "
                    f"for [{from_str} to {to_str}]: {type(exc).__name__} ({clean_err})"
                )
                if retry_delay > 0:
                    time.sleep(retry_delay * (attempt + 1))

        if not success:
            failed_chunks.append((from_str, to_str))
            logger.error(
                f"Failed to fetch Finnhub news for window [{from_str} to {to_str}] after {retries} retries."
            )
            continue

        successful_chunks.append((from_str, to_str))

        chunk_timestamps: list[pd.Timestamp] = []
        for item in batch_items:
            headline = (item.get("headline") or "").strip()
            if not headline:
                continue

            raw_ts = item.get("datetime")
            if raw_ts is None:
                continue

            try:
                pub_utc = pd.to_datetime(raw_ts, unit="s", utc=True)
            except Exception:
                continue

            chunk_timestamps.append(pub_utc)

            articles.append(
                {
                    "ticker": ticker,
                    "published_at": pub_utc,
                    "headline": headline,
                    "summary": (item.get("summary") or "").strip(),
                    "source": (item.get("source") or "Unknown").strip(),
                    "url": (item.get("url") or "").strip(),
                    "provider": "finnhub",
                    "is_fallback": False,
                    "raw_id": str(item.get("id") or ""),
                }
            )

        earliest_ts = min(chunk_timestamps).strftime("%Y-%m-%d %H:%M:%SZ") if chunk_timestamps else "None"
        latest_ts = max(chunk_timestamps).strftime("%Y-%m-%d %H:%M:%SZ") if chunk_timestamps else "None"
        logger.info(
            f"Chunk {chunk_idx}/{len(chunks)} [{from_str} to {to_str}]: "
            f"received {len(batch_items)} items, parsed {len(chunk_timestamps)} valid articles "
            f"(earliest={earliest_ts}, latest={latest_ts})"
        )

        if len(batch_items) >= 240:
            logger.warning(
                f"Chunk [{from_str} to {to_str}] returned {len(batch_items)} items, "
                "which is near Finnhub's response cap (~250). Earlier headlines in this window may be truncated."
            )

        if retry_delay > 0:
            time.sleep(0.15)

    total_chunks = len(chunks)
    failed_count = len(failed_chunks)
    success_count = len(successful_chunks)

    if total_chunks > 0 and success_count == 0:
        status = STATUS_FAILED
    elif failed_count > 0:
        status = STATUS_PARTIAL
    else:
        status = STATUS_COMPLETE

    if status == STATUS_FAILED:
        logger.error(
            f"Finnhub news collection failed: all {total_chunks} chunk(s) failed."
        )
    elif status == STATUS_PARTIAL:
        logger.warning(
            f"Finnhub news collection completed with {failed_count}/{total_chunks} failed chunk(s): {failed_chunks}. "
            f"Retained {len(articles)} articles from {success_count} successful chunk(s)."
        )
    else:
        logger.info(
            f"Finnhub news collection complete across all {total_chunks} chunk(s)."
        )

    if not articles:
        df = pd.DataFrame(columns=NEWS_COLUMNS)
    else:
        df = pd.DataFrame(articles)

    df.attrs["status"] = status
    df.attrs["failed_chunks"] = failed_chunks
    df.attrs["successful_chunks"] = successful_chunks
    df.attrs["total_chunks"] = total_chunks

    return df


def fetch_yahoo_fallback(
    ticker: str = "AAPL",
    limit: int = 50,
    start_date: str | dt.date | None = None,
    end_date: str | dt.date | None = None,
) -> pd.DataFrame:
    """Fetch recent headlines using Yahoo Finance as a fallback source.

    Safely filters returned articles to [start_date, end_date] if provided,
    preventing Yahoo Finance from silently introducing headlines outside the requested range.
    """
    logger.info(f"Fetching recent headlines from Yahoo Finance fallback for {ticker}...")
    articles: list[dict[str, Any]] = []

    try:
        search = yf.Search(ticker, news_count=limit)
        items = getattr(search, "news", []) or []
    except Exception as exc:
        logger.warning(f"yf.Search failed: {exc}. Trying Ticker.news...")
        try:
            t = yf.Ticker(ticker)
            items = getattr(t, "news", []) or []
        except Exception as inner_exc:
            logger.error(f"Yahoo Finance news fetch failed: {inner_exc}")
            items = []

    for item in items:
        headline = (item.get("title") or "").strip()
        if not headline:
            continue

        raw_ts = item.get("providerPublishTime")
        if raw_ts is None:
            continue

        try:
            pub_utc = pd.to_datetime(raw_ts, unit="s", utc=True)
        except Exception:
            continue

        articles.append(
            {
                "ticker": ticker,
                "published_at": pub_utc,
                "headline": headline,
                "summary": (item.get("summary") or "").strip(),
                "source": (item.get("publisher") or "Yahoo").strip(),
                "url": (item.get("link") or "").strip(),
                "provider": "yahoo_finance",
                "is_fallback": True,
                "raw_id": str(item.get("uuid") or ""),
            }
        )

    if not articles:
        logger.warning("No fallback news returned from Yahoo Finance.")
        return pd.DataFrame()

    df = pd.DataFrame(articles)
    if start_date is not None and end_date is not None:
        df = filter_news_by_date_range(df, start_date=start_date, end_date=end_date)

    return df


def fetch_news(
    ticker: str = "AAPL",
    start_date: str = "2026-09-01",
    end_date: str = "2026-10-20",
    use_fallback_only: bool = False,
    include_fallback_supplement: bool = True,
    client: finnhub.Client | None = None,
    chunk_days: int = 1,
    retry_delay: float = 1.0,
) -> pd.DataFrame:
    """Fetch news from Finnhub (primary) and Yahoo Finance (fallback).

    Ensures duplicate articles are dropped while prioritizing primary Finnhub records,
    and strictly enforces date range filtering on the combined, deduplicated dataset.

    Exposes retrieval metadata via df.attrs["status"]:
    - complete: Finnhub retrieved all requested chunks successfully (even if 0 articles).
    - partial: Finnhub had some failed chunks but preserved articles from successful chunks.
    - failed: Finnhub failed across all chunks.
    """
    frames: list[pd.DataFrame] = []
    finnhub_status = "not_attempted" if use_fallback_only else STATUS_FAILED
    failed_chunks: list[tuple[str, str]] = []
    successful_chunks: list[tuple[str, str]] = []
    total_chunks: int = 0

    if not use_fallback_only:
        try:
            finnhub_df = fetch_finnhub_news(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                client=client,
                chunk_days=chunk_days,
                retry_delay=retry_delay,
            )
            finnhub_status = finnhub_df.attrs.get("status", STATUS_COMPLETE)
            failed_chunks = finnhub_df.attrs.get("failed_chunks", [])
            successful_chunks = finnhub_df.attrs.get("successful_chunks", [])
            total_chunks = finnhub_df.attrs.get("total_chunks", 0)

            if finnhub_status == STATUS_COMPLETE:
                if not finnhub_df.empty:
                    frames.append(finnhub_df)
                    logger.info(f"Retrieved {len(finnhub_df)} articles from Finnhub (complete).")
                else:
                    logger.info("Finnhub returned 0 articles (collection complete across all chunks).")
            elif finnhub_status == STATUS_PARTIAL:
                if not finnhub_df.empty:
                    frames.append(finnhub_df)
                logger.warning(
                    f"Finnhub retrieval partial: {len(failed_chunks)}/{total_chunks} chunks failed. "
                    f"Preserved {len(finnhub_df)} articles from successful chunks."
                )
            elif finnhub_status == STATUS_FAILED:
                logger.error(
                    f"Finnhub retrieval failed across all {total_chunks} chunk(s). "
                    "Triggering Yahoo fallback."
                )
        except Exception as exc:
            finnhub_status = STATUS_FAILED
            logger.error(f"Error fetching primary Finnhub news: {exc}. Falling back to Yahoo Finance.")

    # Fetch Yahoo Finance fallback if requested or if primary failed
    if use_fallback_only or not frames or include_fallback_supplement:
        try:
            yahoo_df = fetch_yahoo_fallback(
                ticker=ticker,
                limit=50,
                start_date=start_date,
                end_date=end_date,
            )
            if not yahoo_df.empty:
                frames.append(yahoo_df)
                logger.info(f"Retrieved {len(yahoo_df)} articles from Yahoo Finance fallback.")
        except Exception as exc:
            logger.error(f"Error fetching Yahoo fallback: {exc}")

    if not frames:
        logger.warning("No news collected from either Finnhub or Yahoo Finance.")
        result_df = pd.DataFrame(columns=NEWS_COLUMNS)
    else:
        combined = pd.concat(frames, ignore_index=True)
        deduped = deduplicate_articles(combined)
        result_df = filter_news_by_date_range(deduped, start_date=start_date, end_date=end_date)
        logger.info(f"Total deduplicated and date-filtered news articles: {len(result_df)}")

    result_df.attrs["status"] = finnhub_status
    result_df.attrs["finnhub_status"] = finnhub_status
    result_df.attrs["failed_chunks"] = failed_chunks
    result_df.attrs["successful_chunks"] = successful_chunks
    result_df.attrs["total_chunks"] = total_chunks

    return result_df


class FinBERTSentimentAnalyzer:
    """Wrapper for FinBERT model inference with batched tokenization."""

    def __init__(self, model_name: str = FINBERT_MODEL_NAME, device: str | None = None) -> None:
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading FinBERT model '{model_name}' on device '{self.device}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Map model class indices: {0: 'positive', 1: 'negative', 2: 'neutral'}
        self.id2label = self.model.config.id2label

    def analyze_batch(
        self,
        headlines: list[str],
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """Classify a list of headlines and generate sentiment labels and scores.

        Sentiment score is computed as: Prob(positive) - Prob(negative),
        yielding a continuous numeric score in [-1.0, 1.0].
        """
        if not headlines:
            return pd.DataFrame(
                columns=[
                    "sentiment_label",
                    "sentiment_score",
                    "prob_positive",
                    "prob_negative",
                    "prob_neutral",
                ]
            )

        results: list[dict[str, Any]] = []

        for i in range(0, len(headlines), batch_size):
            batch_texts = headlines[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**encoded).logits
                probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

            for p in probs:
                label_probs = {self.id2label[idx].lower(): float(p[idx]) for idx in range(len(p))}

                pos = label_probs.get("positive", 0.0)
                neg = label_probs.get("negative", 0.0)
                neu = label_probs.get("neutral", 0.0)

                pred_idx = int(np.argmax(p))
                best_label = self.id2label[pred_idx].lower()
                composite_score = float(pos - neg)

                results.append(
                    {
                        "sentiment_label": best_label,
                        "sentiment_score": composite_score,
                        "prob_positive": pos,
                        "prob_negative": neg,
                        "prob_neutral": neu,
                    }
                )

        return pd.DataFrame(results)


def process_news_sentiment(
    raw_news_df: pd.DataFrame,
    analyzer: Any | None = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Enrich raw news articles with look-ahead-bias-free trading dates and FinBERT sentiment."""
    if raw_news_df.empty:
        logger.warning("Empty raw news DataFrame provided to process_news_sentiment.")
        return pd.DataFrame()

    df = raw_news_df.copy()

    # Pre-load valid market trading days and holidays
    valid_set, sorted_days = get_valid_market_trading_days()
    market_holidays = get_us_equity_market_holidays()

    logger.info("Assigning trading sessions to prevent look-ahead bias...")
    trading_dates = []
    for ts in df["published_at"]:
        session_dt = assign_trading_session(
            published_at_utc=ts,
            valid_trading_days_set=valid_set,
            sorted_trading_days=sorted_days,
            market_holidays=market_holidays,
        )
        trading_dates.append(session_dt)

    df["trading_date"] = pd.to_datetime(trading_dates)

    # Sentiment inference
    if analyzer is None:
        analyzer = FinBERTSentimentAnalyzer()

    logger.info(f"Running FinBERT sentiment inference on {len(df)} headlines...")
    sentiment_df = analyzer.analyze_batch(df["headline"].tolist(), batch_size=batch_size)

    for col in sentiment_df.columns:
        df[col] = sentiment_df[col].values

    return df


def aggregate_daily_sentiment(processed_news_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate article-level sentiment by trading session date.

    Computes mean sentiment score, standard deviation, headline count,
    and average sentiment class probabilities for each trading session.
    """
    if processed_news_df.empty:
        return pd.DataFrame()

    grouped = processed_news_df.groupby("trading_date").agg(
        sentiment_score=("sentiment_score", "mean"),
        sentiment_std=("sentiment_score", "std"),
        sentiment_count=("sentiment_score", "count"),
        prob_positive=("prob_positive", "mean"),
        prob_negative=("prob_negative", "mean"),
        prob_neutral=("prob_neutral", "mean"),
    )

    grouped.index.name = "Date"
    return grouped


SENTIMENT_METRIC_COLUMNS = [
    "sentiment_score",
    "sentiment_std",
    "sentiment_count",
    "prob_positive",
    "prob_negative",
    "prob_neutral",
]

SENTIMENT_FEATURE_COLUMNS = SENTIMENT_METRIC_COLUMNS + ["sentiment_coverage"]


def merge_sentiment_with_market_features(
    market_features_path: Path = MARKET_FEATURES_FILE,
    daily_sentiment_df: pd.DataFrame | None = None,
    output_path: Path | None = MARKET_FEATURES_FILE,
    refresh_start_date: str | dt.date | None = None,
    refresh_end_date: str | dt.date | None = None,
    full_rebuild: bool = False,
) -> pd.DataFrame:
    """Merge daily sentiment into market_features.parquet safely and incrementally.

    Behavior:
    1. full_rebuild=False (Incremental Mode, Default):
       - Preserves sentiment values for market dates not being refreshed.
       - If refresh_start_date and refresh_end_date are provided, only trading dates in
         that inclusive window are refreshed.
       - If refresh dates are omitted, only dates present in daily_sentiment_df are updated.
       - If daily_sentiment_df is empty or None, all existing historical sentiment is strictly preserved.
       - A failed news retrieval does not erase existing sentiment.
    2. full_rebuild=True (Explicit Full Rebuild):
       - Resets and replaces sentiment columns across the entire market dataset.
       - Dates without news sentiment receive strictly NaN and sentiment_coverage = 0.
       - If daily_sentiment_df is empty or None, all market rows receive NaN with 0 coverage.
    3. Invariants:
       - Preserves original market index, row count, and non-sentiment features.
       - Missing sentiment is strictly NaN (never imputed with 0.0).
       - sentiment_coverage is strictly 1 when sentiment_score is notna, else 0.
       - Fully idempotent: repeating the same merge does not duplicate columns or change rows.
    """
    if not market_features_path.exists():
        raise FileNotFoundError(f"Market features file does not exist at {market_features_path}")

    market_df = pd.read_parquet(market_features_path)
    original_len = len(market_df)
    original_index = market_df.index.copy()
    original_index_name = market_df.index.name

    # Normalize market DatetimeIndex for exact day alignment
    normalized_market_idx = pd.to_datetime(market_df.index).tz_localize(None).normalize()
    market_df.index = normalized_market_idx
    market_df.index.name = "Date"

    # Ensure sentiment columns exist in market_df
    for col in SENTIMENT_METRIC_COLUMNS:
        if col not in market_df.columns:
            market_df[col] = np.nan
    if "sentiment_coverage" not in market_df.columns:
        market_df["sentiment_coverage"] = 0

    # Handle empty daily sentiment
    if daily_sentiment_df is None or daily_sentiment_df.empty:
        if full_rebuild:
            logger.info(
                "Full rebuild requested with empty daily sentiment. "
                "Resetting all historical sentiment to NaN with 0 coverage."
            )
            for col in SENTIMENT_METRIC_COLUMNS:
                market_df[col] = np.nan
            market_df["sentiment_coverage"] = 0
        else:
            logger.warning(
                "Empty or None daily sentiment provided for incremental update. "
                "Preserving all existing historical sentiment without modification."
            )
    else:
        daily_sent = daily_sentiment_df.copy()
        daily_sent.index = pd.to_datetime(daily_sent.index).tz_localize(None).normalize()
        daily_sent.index.name = "Date"

        if daily_sent.index.has_duplicates:
            daily_sent = daily_sent.loc[~daily_sent.index.duplicated(keep="first")]

        if full_rebuild:
            logger.info("Executing full sentiment rebuild across all market sessions.")
            for col in SENTIMENT_METRIC_COLUMNS:
                market_df[col] = np.nan
            market_df["sentiment_coverage"] = 0

            common_idx = market_df.index.intersection(daily_sent.index)
            for col in SENTIMENT_METRIC_COLUMNS:
                if col in daily_sent.columns:
                    market_df.loc[common_idx, col] = daily_sent.loc[common_idx, col]
            market_df.loc[common_idx, "sentiment_coverage"] = 1
        else:
            # Incremental update: determine refresh window
            if refresh_start_date is not None and refresh_end_date is not None:
                ref_start = pd.to_datetime(refresh_start_date).date()
                ref_end = pd.to_datetime(refresh_end_date).date()
                market_dates = pd.Series([d.date() for d in market_df.index], index=market_df.index)
                refresh_mask = (market_dates >= ref_start) & (market_dates <= ref_end)
                logger.info(
                    f"Executing incremental sentiment update for window [{ref_start} to {ref_end}]. "
                    f"Market sessions in window: {int(refresh_mask.sum())}."
                )
            else:
                refresh_mask = market_df.index.isin(daily_sent.index)
                logger.info(
                    f"Executing incremental sentiment update for {int(refresh_mask.sum())} specific session(s)."
                )

            refreshed_indices = market_df.index[refresh_mask]
            for idx in refreshed_indices:
                if idx in daily_sent.index:
                    for col in SENTIMENT_METRIC_COLUMNS:
                        if col in daily_sent.columns:
                            market_df.loc[idx, col] = daily_sent.loc[idx, col]
                    market_df.loc[idx, "sentiment_coverage"] = 1
                else:
                    for col in SENTIMENT_METRIC_COLUMNS:
                        market_df.loc[idx, col] = np.nan
                    market_df.loc[idx, "sentiment_coverage"] = 0

    # Invariant enforcement
    missing_mask = market_df["sentiment_coverage"] == 0
    for col in SENTIMENT_METRIC_COLUMNS:
        market_df.loc[missing_mask, col] = np.nan

    # Restore original index and index name
    market_df.index = original_index
    market_df.index.name = original_index_name

    # Validate row count and index preservation
    assert len(market_df) == original_len, (
        f"Row count mismatch after merge! Expected {original_len}, got {len(market_df)}"
    )
    assert (market_df.index == original_index).all(), (
        "Market features index was modified during merge!"
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        market_df.to_parquet(output_path)
        logger.info(
            f"Successfully merged sentiment into {output_path}. "
            f"Total rows: {len(market_df)}, Covered trading days: {market_df['sentiment_coverage'].sum()}"
        )

    return market_df


def run_test_mode(ticker: str = "AAPL") -> None:
    """Run an automated, offline test mode validating schemas, logic, and invariants."""
    print("=" * 70)
    print("RUNNING OFFLINE INTEGRITY TEST MODE FOR NEWS SENTIMENT PIPELINE")
    print("=" * 70)

    # 1. Test chunk boundaries and single-day chunks
    print("[1/7] Testing chunk boundaries and single-day chunk generation...")
    oct_chunks = generate_date_chunks(dt.date(2026, 10, 2), dt.date(2026, 10, 13), chunk_days=1)
    assert len(oct_chunks) == 12, f"Expected 12 single-day chunks, got {len(oct_chunks)}"
    assert oct_chunks[0] == ("2026-10-02", "2026-10-02")
    assert oct_chunks[-1] == ("2026-10-13", "2026-10-13")
    print(f"  [OK] Single-day chunking verified: {len(oct_chunks)} chunks without gaps.")

    # 2. Test multi-tier article deduplication (Finnhub over Yahoo precedence)
    print("[2/7] Testing multi-tier article deduplication across chunks...")
    test_dup_df = pd.DataFrame(
        [
            {
                "raw_id": "id_1",
                "headline": "Apple announces quarterly earnings",
                "url": "https://example.com/1",
                "published_at": pd.to_datetime("2026-10-10 10:00:00Z"),
                "is_fallback": False,
                "source": "WireA",
                "ticker": "AAPL",
                "summary": "",
            },
            {
                "raw_id": "id_1",
                "headline": "Apple announces quarterly earnings",
                "url": "https://example.com/1",
                "published_at": pd.to_datetime("2026-10-10 10:00:00Z"),
                "is_fallback": True,
                "source": "WireA",
                "ticker": "AAPL",
                "summary": "",
            },
            {
                "raw_id": "",
                "headline": "Syndicated headline",
                "url": "https://example.com/syndicated",
                "published_at": pd.to_datetime("2026-10-10 11:00:00Z"),
                "is_fallback": False,
                "source": "WireB",
                "ticker": "AAPL",
                "summary": "",
            },
            {
                "raw_id": "id_2",
                "headline": "Syndicated headline variation",
                "url": "https://example.com/syndicated",
                "published_at": pd.to_datetime("2026-10-10 11:00:00Z"),
                "is_fallback": True,
                "source": "WireB",
                "ticker": "AAPL",
                "summary": "",
            },
            {
                "raw_id": "",
                "headline": "Headline without ID or URL",
                "url": "",
                "published_at": pd.to_datetime("2026-10-10 12:00:00Z"),
                "is_fallback": False,
                "source": "WireC",
                "ticker": "AAPL",
                "summary": "",
            },
            {
                "raw_id": "",
                "headline": "  HEADLINE WITHOUT ID OR URL  ",
                "url": "",
                "published_at": pd.to_datetime("2026-10-10 12:00:00Z"),
                "is_fallback": False,
                "source": "WireC",
                "ticker": "AAPL",
                "summary": "",
            },
            {
                "raw_id": "",
                "headline": "Unique separate article",
                "url": "https://example.com/unique",
                "published_at": pd.to_datetime("2026-10-10 13:00:00Z"),
                "is_fallback": False,
                "source": "WireD",
                "ticker": "AAPL",
                "summary": "",
            },
        ]
    )
    deduped_test = deduplicate_articles(test_dup_df)
    assert len(deduped_test) == 4, f"Deduplication expected 4 unique articles, got {len(deduped_test)}"
    # Verify Finnhub article was kept when colliding with Yahoo
    assert not deduped_test.loc[deduped_test["raw_id"] == "id_1", "is_fallback"].iloc[0]
    print(f"  [OK] Multi-tier deduplication verified: 7 raw articles -> {len(deduped_test)} unique articles (Finnhub prioritized).")

    # 3. Test date range filtering in America/New_York
    print("[3/7] Testing date range filtering with timezone boundaries...")
    sample_dates_df = pd.DataFrame(
        [
            # Exactly on start date in ET
            {"headline": "Start day", "published_at": pd.Timestamp("2026-10-01 00:00:00", tz="America/New_York")},
            # 1 second before start date in ET
            {"headline": "Before start", "published_at": pd.Timestamp("2026-09-30 23:59:59", tz="America/New_York")},
            # Exactly on end date in ET
            {"headline": "End day", "published_at": pd.Timestamp("2026-10-15 23:59:59", tz="America/New_York")},
            # 1 second after end date in ET
            {"headline": "After end", "published_at": pd.Timestamp("2026-10-16 00:00:01", tz="America/New_York")},
            # Expressed in UTC: 2026-10-16 02:00:00 UTC = 2026-10-15 22:00:00 EDT (inside range!)
            {"headline": "UTC boundary inside", "published_at": pd.Timestamp("2026-10-16 02:00:00", tz="UTC")},
            # Expressed in UTC: 2026-10-01 02:00:00 UTC = 2026-09-30 22:00:00 EDT (outside range!)
            {"headline": "UTC boundary outside", "published_at": pd.Timestamp("2026-10-01 02:00:00", tz="UTC")},
            # Corrupted / missing timestamp
            {"headline": "Null timestamp", "published_at": None},
        ]
    )
    filtered = filter_news_by_date_range(sample_dates_df, start_date="2026-10-01", end_date="2026-10-15")
    retained_headlines = set(filtered["headline"])
    assert "Start day" in retained_headlines, "Start day was incorrectly excluded"
    assert "End day" in retained_headlines, "End day was incorrectly excluded"
    assert "UTC boundary inside" in retained_headlines, "UTC inside boundary was incorrectly excluded"
    assert "Before start" not in retained_headlines, "Before start was incorrectly included"
    assert "After end" not in retained_headlines, "After end was incorrectly included"
    assert "UTC boundary outside" not in retained_headlines, "UTC outside boundary was incorrectly included"
    assert "Null timestamp" not in retained_headlines, "Null timestamp was incorrectly included"
    assert len(filtered) == 3, f"Expected 3 retained articles, got {len(filtered)}"
    print(f"  [OK] Date range filtering verified ({len(sample_dates_df)} -> {len(filtered)} retained).")

    # 4. Trading session assignment and look-ahead bias prevention
    print("[4/7] Testing trading session assignment, early closes, and holidays...")
    equity_holidays = get_us_equity_market_holidays()

    edge_cases = [
        ("Regular session before close (15:59:59 ET)", pd.Timestamp("2026-10-16 15:59:59", tz="America/New_York"), pd.Timestamp("2026-10-16")),
        ("Regular session at close (16:00:00 ET)", pd.Timestamp("2026-10-16 16:00:00", tz="America/New_York"), pd.Timestamp("2026-10-19")),
        ("Regular session after close (16:00:01 ET)", pd.Timestamp("2026-10-16 16:00:01", tz="America/New_York"), pd.Timestamp("2026-10-19")),
        ("Weekend Saturday (12:00:00 ET)", pd.Timestamp("2026-10-17 12:00:00", tz="America/New_York"), pd.Timestamp("2026-10-19")),
        ("Weekend Sunday (18:00:00 ET)", pd.Timestamp("2026-10-18 18:00:00", tz="America/New_York"), pd.Timestamp("2026-10-19")),
        ("Good Friday (2026-04-03 10:00:00 ET)", pd.Timestamp("2026-04-03 10:00:00", tz="America/New_York"), pd.Timestamp("2026-04-06")),
        ("Columbus Day (2026-10-12 11:00:00 ET - open for equities)", pd.Timestamp("2026-10-12 11:00:00", tz="America/New_York"), pd.Timestamp("2026-10-12")),
        ("Black Friday early close before 13:00 (12:59:59 ET)", pd.Timestamp("2026-11-27 12:59:59", tz="America/New_York"), pd.Timestamp("2026-11-27")),
        ("Black Friday early close at 13:00 (13:00:00 ET)", pd.Timestamp("2026-11-27 13:00:00", tz="America/New_York"), pd.Timestamp("2026-11-30")),
        ("UTC equivalent before close (19:59:59 UTC = 15:59:59 EDT)", pd.Timestamp("2026-10-16 19:59:59", tz="UTC"), pd.Timestamp("2026-10-16")),
        ("UTC equivalent at close (20:00:00 UTC = 16:00:00 EDT)", pd.Timestamp("2026-10-16 20:00:00", tz="UTC"), pd.Timestamp("2026-10-19")),
    ]

    for label, ts, expected in edge_cases:
        res = assign_trading_session(ts, market_holidays=equity_holidays)
        assert res == expected, f"Edge case failed for '{label}': got {res}, expected {expected}"
        print(f"  [OK] {label} -> {res.date()}")

    # 5. Offline sentiment scoring & daily aggregation with mock analyzer
    print("[5/7] Testing sentiment scoring pipeline & aggregation (offline mock)...")
    class MockSentimentAnalyzer:
        def analyze_batch(self, headlines: list[str], batch_size: int = 32) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "sentiment_label": "positive" if i % 2 == 0 else "negative",
                        "sentiment_score": 0.6 if i % 2 == 0 else -0.4,
                        "prob_positive": 0.8 if i % 2 == 0 else 0.1,
                        "prob_negative": 0.1 if i % 2 == 0 else 0.7,
                        "prob_neutral": 0.1 if i % 2 == 0 else 0.2,
                    }
                    for i in range(len(headlines))
                ]
            )

    mock_news = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "published_at": pd.Timestamp("2026-10-16 14:00:00", tz="America/New_York"),
                "headline": "Apple reports strong revenue growth across services.",
                "summary": "Services growth accelerates.",
                "source": "MockWire",
                "url": "https://example.com/mock1",
                "provider": "finnhub",
                "is_fallback": False,
                "raw_id": "m1",
            },
            {
                "ticker": ticker,
                "published_at": pd.Timestamp("2026-10-16 15:00:00", tz="America/New_York"),
                "headline": "Tech supply constraints impact hardware delivery times.",
                "summary": "Delivery constraints noted.",
                "source": "MockWire",
                "url": "https://example.com/mock2",
                "provider": "finnhub",
                "is_fallback": False,
                "raw_id": "m2",
            },
        ]
    )

    processed_df = process_news_sentiment(mock_news, analyzer=MockSentimentAnalyzer())
    assert len(processed_df) == 2
    assert "trading_date" in processed_df.columns
    assert (processed_df["trading_date"] == pd.Timestamp("2026-10-16")).all()

    daily_sent = aggregate_daily_sentiment(processed_df)
    assert len(daily_sent) == 1
    assert daily_sent.index[0] == pd.Timestamp("2026-10-16")
    assert np.isclose(daily_sent.loc[pd.Timestamp("2026-10-16"), "sentiment_score"], 0.1)
    print("  [OK] Daily sentiment aggregated successfully.")

    # 6. Incremental vs Full Rebuild merge & invariants
    print("[6/7] Testing incremental merge, full rebuild, idempotence, and invariants...")
    # Synthetic market DataFrame (5 trading days)
    dates = pd.date_range("2026-10-12", periods=5, freq="B")
    base_market = pd.DataFrame(
        {"Close": [100.0, 101.0, 102.0, 103.0, 104.0], "Daily_Return": [0.01, 0.01, 0.01, 0.01, 0.01]},
        index=dates,
    )
    temp_market_path = PROCESSED_DATA_DIR / "temp_test_market_features.parquet"
    base_market.to_parquet(temp_market_path)

    try:
        # 6a. Initial merge with day 0 & day 1
        initial_sent = pd.DataFrame(
            {
                "sentiment_score": [0.5, 0.7],
                "sentiment_std": [0.1, 0.2],
                "sentiment_count": [4, 6],
                "prob_positive": [0.7, 0.8],
                "prob_negative": [0.2, 0.1],
                "prob_neutral": [0.1, 0.1],
            },
            index=dates[:2],
        )
        merged_init = merge_sentiment_with_market_features(
            market_features_path=temp_market_path,
            daily_sentiment_df=initial_sent,
            output_path=temp_market_path,
            full_rebuild=True,
        )
        assert merged_init["sentiment_coverage"].sum() == 2
        assert merged_init.loc[dates[0], "sentiment_score"] == 0.5
        assert pd.isna(merged_init.loc[dates[2], "sentiment_score"])
        print("  [OK] Initial merge established sentiment on 2 sessions.")

        # 6b. Incremental update: refresh day 3 only (must preserve days 0 & 1)
        day3_sent = pd.DataFrame(
            {
                "sentiment_score": [-0.35],
                "sentiment_std": [0.05],
                "sentiment_count": [2],
                "prob_positive": [0.2],
                "prob_negative": [0.6],
                "prob_neutral": [0.2],
            },
            index=[dates[3]],
        )
        merged_incr = merge_sentiment_with_market_features(
            market_features_path=temp_market_path,
            daily_sentiment_df=day3_sent,
            output_path=temp_market_path,
            refresh_start_date=dates[3].strftime("%Y-%m-%d"),
            refresh_end_date=dates[3].strftime("%Y-%m-%d"),
            full_rebuild=False,
        )
        assert merged_incr["sentiment_coverage"].sum() == 3
        assert merged_incr.loc[dates[0], "sentiment_score"] == 0.5, "Day 0 historical sentiment was altered!"
        assert merged_incr.loc[dates[1], "sentiment_score"] == 0.7, "Day 1 historical sentiment was altered!"
        assert merged_incr.loc[dates[3], "sentiment_score"] == -0.35, "Day 3 sentiment was not updated!"
        print("  [OK] Incremental update strictly preserved unrelated historical sentiment.")

        # 6c. Idempotence: repeated incremental merge produces identical results
        merged_repeat = merge_sentiment_with_market_features(
            market_features_path=temp_market_path,
            daily_sentiment_df=day3_sent,
            output_path=temp_market_path,
            refresh_start_date=dates[3].strftime("%Y-%m-%d"),
            refresh_end_date=dates[3].strftime("%Y-%m-%d"),
            full_rebuild=False,
        )
        pd.testing.assert_frame_equal(merged_incr, merged_repeat)
        print("  [OK] Repeated update is fully idempotent.")

        # 6d. Empty retrieval preservation: incremental update with empty sentiment does NOT erase data
        merged_empty = merge_sentiment_with_market_features(
            market_features_path=temp_market_path,
            daily_sentiment_df=pd.DataFrame(),
            output_path=temp_market_path,
            refresh_start_date=dates[3].strftime("%Y-%m-%d"),
            refresh_end_date=dates[3].strftime("%Y-%m-%d"),
            full_rebuild=False,
        )
        pd.testing.assert_frame_equal(merged_incr, merged_empty)
        print("  [OK] Empty retrieval in incremental mode preserved all existing historical sentiment.")

        # 6e. Explicit full rebuild with empty sentiment resets to NaN and 0 coverage
        merged_reset = merge_sentiment_with_market_features(
            market_features_path=temp_market_path,
            daily_sentiment_df=pd.DataFrame(),
            output_path=temp_market_path,
            full_rebuild=True,
        )
        assert merged_reset["sentiment_coverage"].sum() == 0
        assert merged_reset["sentiment_score"].isna().all()
        print("  [OK] Explicit full rebuild cleanly reset all sentiment columns to NaN.")

        # 6f. Missing sentiment invariant: missing coverage rows are strictly NaN, NEVER 0.0
        missing_rows = merged_init[merged_init["sentiment_coverage"] == 0]
        assert missing_rows["sentiment_score"].isna().all(), "Invariant violation: NaN replaced by 0.0"
        assert missing_rows["prob_positive"].isna().all()
        assert missing_rows["prob_negative"].isna().all()
        print("  [OK] Invariant verified: Missing sentiment is strictly NaN, never imputed with 0.0.")

        # 7. Complete empty retrieval, partial chunk failure, and total failure distinction
        print("[7/7] Testing Finnhub collection outcomes (complete empty, partial, failed)...")
        class MockFinnhubClient:
            def __init__(self, behaviors: dict[str, Any]) -> None:
                self.behaviors = behaviors

            def company_news(self, ticker: str, _from: str, to: str) -> list[dict[str, Any]]:
                val = self.behaviors.get((_from, to)) or self.behaviors.get(_from)
                if isinstance(val, Exception):
                    raise val
                return val or []

        # 7a. Complete empty retrieval
        complete_client = MockFinnhubClient({
            "2026-10-10": [],
            "2026-10-11": [],
        })
        res_complete = fetch_finnhub_news("AAPL", "2026-10-10", "2026-10-11", client=complete_client, retry_delay=0.0)
        assert res_complete.empty, "Expected empty DataFrame for complete 0-article retrieval"
        assert res_complete.attrs["status"] == STATUS_COMPLETE
        assert res_complete.attrs["failed_chunks"] == []
        assert res_complete.attrs["total_chunks"] == 2
        print("  [OK] Complete empty retrieval correctly identified.")

        # 7b. Partial chunk failure
        partial_client = MockFinnhubClient({
            "2026-10-10": RuntimeError("Simulated API 500"),
            "2026-10-11": [
                {
                    "id": 999,
                    "datetime": int(pd.Timestamp("2026-10-11 12:00:00Z").timestamp()),
                    "headline": "Apple Launches Service Update",
                    "summary": "Service update details.",
                    "source": "WireA",
                    "url": "https://example.com/art999",
                }
            ],
        })
        res_partial = fetch_finnhub_news("AAPL", "2026-10-10", "2026-10-11", client=partial_client, retry_delay=0.0)
        assert len(res_partial) == 1, "Expected successful chunk article to be preserved"
        assert res_partial.attrs["status"] == STATUS_PARTIAL
        assert len(res_partial.attrs["failed_chunks"]) == 1
        assert res_partial.attrs["failed_chunks"][0] == ("2026-10-10", "2026-10-10")
        assert len(res_partial.attrs["successful_chunks"]) == 1
        print("  [OK] Partial chunk failure correctly identified and successful articles preserved.")

        # 7c. Total failure
        failed_client = MockFinnhubClient({
            "2026-10-10": RuntimeError("Simulated API 500"),
            "2026-10-11": RuntimeError("Simulated Rate Limit"),
        })
        res_failed = fetch_finnhub_news("AAPL", "2026-10-10", "2026-10-11", client=failed_client, retry_delay=0.0)
        assert res_failed.empty
        assert res_failed.attrs["status"] == STATUS_FAILED
        assert len(res_failed.attrs["failed_chunks"]) == 2
        print("  [OK] Total retrieval failure correctly identified (not treated as successful empty).")

    finally:
        if temp_market_path.exists():
            temp_market_path.unlink()

    print("=" * 70)
    print("ALL NEWS SENTIMENT INTEGRITY TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


def run_production_pipeline(
    ticker: str = "AAPL",
    start_date: str = "2026-09-01",
    end_date: str | None = None,
    chunk_days: int = 1,
    full_rebuild: bool = False,
    batch_size: int = 32,
    client: finnhub.Client | None = None,
    market_features_path: Path | None = None,
    raw_news_path: Path | None = None,
    processed_news_path: Path | None = None,
    analyzer: Any = None,
    retry_delay: float = 1.0,
) -> dict[str, Any]:
    """Execute news ingestion, FinBERT scoring, and market features merge.

    Safely distinguishes:
    1. Complete: All chunks succeeded. If 0 articles, confirmed empty without modifying features.
    2. Partial: Some chunks failed; preserves articles from successful chunks, and restricts
       incremental update strictly to observed dates to prevent wiping unretrieved dates.
    3. Failed: All chunks failed; aborts pipeline to prevent clearing/overwriting historical data.
    """
    if end_date is None:
        end_date = dt.date.today().strftime("%Y-%m-%d")

    raw_file = raw_news_path or RAW_NEWS_FILE
    proc_file = processed_news_path or PROCESSED_NEWS_FILE
    market_file = market_features_path or MARKET_FEATURES_FILE

    raw_file.parent.mkdir(parents=True, exist_ok=True)
    proc_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Ingest raw news
    raw_news = fetch_news(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        include_fallback_supplement=True,
        chunk_days=chunk_days,
        client=client,
        retry_delay=retry_delay,
    )

    fetch_status = raw_news.attrs.get("status", STATUS_COMPLETE)

    if fetch_status == STATUS_FAILED:
        logger.error(
            f"Finnhub news retrieval failed completely for {ticker} [{start_date} to {end_date}]. "
            "Aborting pipeline to avoid overwriting or clearing historical sentiment."
        )
        return {"status": STATUS_FAILED, "raw_news_count": len(raw_news), "processed_count": 0, "covered_sessions": 0}

    if raw_news.empty:
        if fetch_status == STATUS_COMPLETE:
            logger.info(
                f"News retrieval complete for {ticker} [{start_date} to {end_date}]: "
                "confirmed zero articles published. Exiting pipeline."
            )
        elif fetch_status == STATUS_PARTIAL:
            logger.warning(
                f"News retrieval was partial for {ticker} [{start_date} to {end_date}] "
                "and returned zero articles. Exiting without modifying market features."
            )
        else:
            logger.error("No news articles collected. Exiting pipeline.")
        return {"status": fetch_status, "raw_news_count": 0, "processed_count": 0, "covered_sessions": 0}

    raw_news.to_parquet(raw_file)
    logger.info(f"Saved {len(raw_news)} raw news articles to {raw_file}")

    # 2. Process sentiment with FinBERT and assign look-ahead-bias-free trading dates
    processed_news = process_news_sentiment(
        raw_news_df=raw_news,
        batch_size=batch_size,
        analyzer=analyzer,
    )
    processed_news.to_parquet(proc_file)
    logger.info(f"Saved {len(processed_news)} processed news sentiment records to {proc_file}")

    # 3. Aggregate daily sentiment by trading session
    daily_sentiment = aggregate_daily_sentiment(processed_news)

    # 4. Merge daily sentiment into market_features.parquet
    covered_sessions = 0
    if market_file.exists():
        # When collection is partial, restrict incremental refresh strictly to observed trading dates
        # so historical sentiment on unretrieved/failed dates is preserved.
        refresh_start = None if fetch_status == STATUS_PARTIAL else start_date
        refresh_end = None if fetch_status == STATUS_PARTIAL else end_date

        if fetch_status == STATUS_PARTIAL:
            logger.warning(
                "Finnhub collection was partial. Updating only trading sessions with retrieved news "
                "to preserve historical sentiment for failed chunk windows."
            )

        merged = merge_sentiment_with_market_features(
            market_features_path=market_file,
            daily_sentiment_df=daily_sentiment,
            output_path=market_file,
            refresh_start_date=refresh_start,
            refresh_end_date=refresh_end,
            full_rebuild=full_rebuild,
        )
        covered_sessions = int(merged["sentiment_coverage"].sum())
        print("\nPipeline execution complete.")
        print(f"Market features rows: {len(merged)}")
        print(f"Trading days with sentiment coverage: {covered_sessions}")
    else:
        logger.warning(
            f"Market features file {market_file} not found. Skipping merge."
        )

    return {
        "status": fetch_status,
        "raw_news_count": len(raw_news),
        "processed_count": len(processed_news),
        "covered_sessions": covered_sessions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quant RAG Copilot: Financial News Sentiment Pipeline"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker symbol (default: AAPL)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2026-09-01",
        help="Start date YYYY-MM-DD for news retrieval (default: 2026-09-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=dt.date.today().strftime("%Y-%m-%d"),
        help="End date YYYY-MM-DD for news retrieval",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=1,
        help="Number of calendar days per Finnhub request chunk (default: 1)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run short test mode validating schemas and pipeline logic",
    )
    parser.add_argument(
        "--fallback-only",
        action="store_true",
        help="Force Yahoo Finance fallback news without querying Finnhub",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for FinBERT inference (default: 32)",
    )
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help="Perform full sentiment rebuild replacing all historical sentiment across entire market dataset",
    )
    args = parser.parse_args()

    if args.test:
        run_test_mode(ticker=args.ticker)
        return

    run_production_pipeline(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        chunk_days=args.chunk_days,
        full_rebuild=args.full_rebuild,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
