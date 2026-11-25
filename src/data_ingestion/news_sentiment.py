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
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
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

DEFAULT_STAGED_CHUNKS_DIR = PROJECT_ROOT / "data" / "raw" / "staged_chunks"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "raw" / "news_retrieval_manifest.json"
DEFAULT_TRANSACTION_JOURNAL = PROJECT_ROOT / "data" / "backups" / "transaction_journal.json"

TX_STATE_STAGED = "STAGED"
TX_STATE_RECOVERING = "RECOVERING"
TX_STATE_COMMITTED = "COMMITTED"
TX_STATE_ABORTED_RECOVERED = "ABORTED_RECOVERED"

# Authorized directory roots for production safety (rejecting whole-PROJECT_ROOT traversal)
DEFAULT_ALLOWED_DATA_ROOTS = [
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "data" / "processed",
]
DEFAULT_ALLOWED_BACKUP_ROOTS = [
    PROJECT_ROOT / "data" / "backups",
]
DEFAULT_ALLOWED_JOURNAL_ROOTS = [
    PROJECT_ROOT / "data" / "backups",
]

HEX_64_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")

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
    """Deduplicate news articles across chunks using stable IDs, URLs, and normalized headlines.

    Deterministic tie-breaking:
    Equal publication timestamps are deterministically sorted by raw_id, then URL, then headline,
    ensuring identical row ordering regardless of input concatenation permutations while strictly
    preserving primary Finnhub precedence (is_fallback=False precedes is_fallback=True).
    """
    if df.empty:
        return df

    working = df.copy()
    working["published_at"] = pd.to_datetime(working["published_at"], utc=True)

    # Deterministic helper keys for stable tie-breaking
    working["_raw_id_clean"] = working["raw_id"].fillna("").astype(str).str.strip()
    working["_url_clean"] = working["url"].fillna("").astype(str).str.strip()
    working["_headline_clean"] = working["headline"].fillna("").astype(str).str.lower().str.strip()

    # Sort so primary Finnhub records (is_fallback=False) come before fallback records,
    # with deterministic tie-breaking for equal publication timestamps.
    working = working.sort_values(
        by=["is_fallback", "published_at", "_raw_id_clean", "_url_clean", "_headline_clean"],
        ascending=[True, False, True, True, True],
    )

    # 1. Deduplicate by stable non-empty raw_id (Finnhub article ID or Yahoo UUID)
    has_id = working["_raw_id_clean"].ne("")
    with_id = working[has_id].drop_duplicates(subset=["_raw_id_clean"], keep="first")
    without_id = working[~has_id]
    combined = pd.concat([with_id, without_id], ignore_index=True)

    # 2. Deduplicate by non-empty URL
    has_url = combined["_url_clean"].ne("")
    with_url = combined[has_url].drop_duplicates(subset=["_url_clean"], keep="first")
    without_url = combined[~has_url]
    combined = pd.concat([with_url, without_url], ignore_index=True)

    # 3. Deduplicate by normalized lowercase headline and published timestamp
    deduped = combined.drop_duplicates(subset=["_headline_clean", "published_at"], keep="first")

    # Final deterministic sort: newest first, with deterministic tie-breaker on ID/URL/headline
    deduped = deduped.sort_values(
        by=["published_at", "_raw_id_clean", "_url_clean", "_headline_clean"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    deduped = deduped.drop(columns=["_raw_id_clean", "_url_clean", "_headline_clean"], errors="ignore")
    return deduped


def merge_raw_news_archive(
    existing_raw_path: Path,
    incoming_raw_df: pd.DataFrame,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Safely combine incoming raw news with existing raw archive, preserving all original records.

    Idempotent no-op:
    If incoming_raw_df contains zero net-new articles, returns existing_df directly without
    re-concatenating or re-sorting, preventing unnecessary Parquet byte mutations.
    """
    if not existing_raw_path.exists():
        merged = deduplicate_articles(incoming_raw_df) if not incoming_raw_df.empty else incoming_raw_df
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(output_path)
        return merged

    if incoming_raw_df.empty:
        existing_df = pd.read_parquet(existing_raw_path)
        if output_path is not None and output_path.resolve() != existing_raw_path.resolve():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(existing_raw_path, output_path)
        return existing_df

    existing_df = pd.read_parquet(existing_raw_path)
    net_new = get_net_new_articles(incoming_raw_df, existing_df)
    if net_new.empty:
        logger.info("Zero net-new articles detected for raw news archive. Preserving existing archive without modification.")
        if output_path is not None and output_path.resolve() != existing_raw_path.resolve():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(existing_raw_path, output_path)
        return existing_df

    combined = pd.concat([existing_df, incoming_raw_df], ignore_index=True)
    merged = deduplicate_articles(combined)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(output_path)
    return merged


def merge_processed_news_archive(
    existing_processed_path: Path,
    incoming_processed_df: pd.DataFrame,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Safely combine incoming processed news with existing processed archive, preserving all original records.

    Idempotent no-op:
    If incoming_processed_df contains zero net-new articles, returns existing_df directly without
    re-concatenating or re-sorting, preventing unnecessary Parquet byte mutations.
    """
    if not existing_processed_path.exists():
        merged = deduplicate_articles(incoming_processed_df) if not incoming_processed_df.empty else incoming_processed_df
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(output_path)
        return merged

    if incoming_processed_df.empty:
        existing_df = pd.read_parquet(existing_processed_path)
        if output_path is not None and output_path.resolve() != existing_processed_path.resolve():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(existing_processed_path, output_path)
        return existing_df

    existing_df = pd.read_parquet(existing_processed_path)
    net_new = get_net_new_articles(incoming_processed_df, existing_df)
    if net_new.empty:
        logger.info("Zero net-new articles detected for processed news archive. Preserving existing archive without modification.")
        if output_path is not None and output_path.resolve() != existing_processed_path.resolve():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(existing_processed_path, output_path)
        return existing_df

    combined = pd.concat([existing_df, incoming_processed_df], ignore_index=True)
    merged = deduplicate_articles(combined)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(output_path)
    return merged


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def durable_json_write(file_path: Path, data: dict[str, Any]) -> None:
    """Atomically write JSON data to file with fsync to guarantee durability."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(file_path)


def is_safe_contained_path(path: Path, allowed_roots: list[Path]) -> bool:
    """Check if resolved path is strictly contained within any of allowed_roots without symlink escapes."""
    try:
        resolved = path.resolve(strict=False)
    except Exception:
        return False

    for root in allowed_roots:
        resolved_root = root.resolve()
        try:
            if hasattr(resolved, "is_relative_to"):
                if resolved.is_relative_to(resolved_root):
                    return True
            else:
                resolved.relative_to(resolved_root)
                return True
        except (ValueError, TypeError):
            continue
    return False


def check_no_symlinks(path: Path) -> None:
    """Reject symlinks in the path or any of its parent directory components."""
    curr = path
    while curr != curr.parent:
        if curr.is_symlink():
            raise RuntimeError(f"FATAL: Symlinks prohibited: '{curr}' is a symlink.")
        curr = curr.parent


def validate_path_safety(
    path: Path,
    allowed_roots: list[Path],
) -> Path:
    """Validate that path is contained within one of the allowed roots and has no symlink/path traversal."""
    check_no_symlinks(path)
    resolved = path.resolve()
    if not is_safe_contained_path(resolved, allowed_roots):
        raise RuntimeError(
            f"FATAL: Unauthorized path rejected: '{path}' (resolved: '{resolved}') is outside allowed roots."
        )
    return resolved


def validate_target_path(
    target: Path,
    allowed_roots: list[Path] | None = None,
) -> Path:
    """Validate that target path is strictly an authorized production dataset."""
    effective_roots = allowed_roots if allowed_roots is not None else DEFAULT_ALLOWED_DATA_ROOTS
    resolved = target.resolve()
    check_no_symlinks(target)

    # Disallow source code, git, env files explicitly
    prohibited_roots = [
        PROJECT_ROOT / "src",
        PROJECT_ROOT / ".git",
    ]
    for p_root in prohibited_roots:
        if is_safe_contained_path(resolved, [p_root]):
            raise RuntimeError(
                f"FATAL: Target path '{target}' rejected: targets inside project source or git directories are strictly forbidden."
            )

    if resolved == (PROJECT_ROOT / ".env").resolve():
        raise RuntimeError(f"FATAL: Target path '{target}' rejected: .env cannot be a target.")

    # Must have .parquet extension
    if target.suffix.lower() != ".parquet":
        raise RuntimeError(
            f"FATAL: Target path '{target}' rejected: only '.parquet' files are authorized data targets, got '{target.suffix}'."
        )

    if not is_safe_contained_path(resolved, effective_roots):
        raise RuntimeError(
            f"FATAL: Target path rejected: '{target}' (resolved: '{resolved}') is outside permitted data roots."
        )
    return resolved


def validate_staged_path(
    staged: Path,
    target: Path,
    allowed_roots: list[Path] | None = None,
) -> Path:
    """Validate that staged path is safe, within allowed roots, and matches target naming."""
    effective_roots = allowed_roots if allowed_roots is not None else DEFAULT_ALLOWED_DATA_ROOTS
    resolved = staged.resolve()
    check_no_symlinks(staged)

    if not is_safe_contained_path(resolved, effective_roots):
        raise RuntimeError(
            f"FATAL: Staged path rejected: '{staged}' is outside permitted data roots."
        )

    if not (staged.name.endswith(".stage") or staged.suffix.lower() == ".parquet"):
        raise RuntimeError(
            f"FATAL: Staged file '{staged}' rejected: unexpected extension."
        )
    return resolved


def validate_backup_path(
    backup: Path,
    allowed_backup_roots: list[Path] | None = None,
) -> Path:
    """Validate that backup path is strictly within authorized backup roots."""
    effective_roots = allowed_backup_roots if allowed_backup_roots is not None else DEFAULT_ALLOWED_BACKUP_ROOTS
    resolved = backup.resolve()
    check_no_symlinks(backup)

    if backup.suffix.lower() != ".parquet":
        raise RuntimeError(
            f"FATAL: Backup file '{backup}' rejected: must have '.parquet' extension."
        )

    if not is_safe_contained_path(resolved, effective_roots):
        raise RuntimeError(
            f"FATAL: Backup path rejected: '{backup}' is outside authorized backup roots."
        )
    return resolved


def validate_directory_safety(
    dir_path: Path,
    allowed_roots: list[Path] | None = None,
) -> Path:
    """Validate that a directory is strictly within authorized roots and has no symlink traversal."""
    resolved = dir_path.resolve()
    check_no_symlinks(dir_path)

    effective_roots = allowed_roots if allowed_roots is not None else DEFAULT_ALLOWED_BACKUP_ROOTS
    if not is_safe_contained_path(resolved, effective_roots):
        raise RuntimeError(
            f"FATAL: Directory rejected: '{dir_path}' is outside authorized roots."
        )
    return resolved


def validate_journal_path(
    journal_path: Path,
    allowed_journal_roots: list[Path] | None = None,
) -> Path:
    """Validate that journal file is strictly within authorized journal roots and has .json extension."""
    resolved = journal_path.resolve()
    check_no_symlinks(journal_path)

    if journal_path.suffix.lower() != ".json":
        raise RuntimeError(
            f"FATAL: Transaction journal rejected: '{journal_path}' must have '.json' extension."
        )

    effective_roots = allowed_journal_roots if allowed_journal_roots is not None else DEFAULT_ALLOWED_JOURNAL_ROOTS
    if not is_safe_contained_path(resolved, effective_roots):
        raise RuntimeError(
            f"FATAL: Transaction journal rejected: '{journal_path}' is outside authorized journal roots."
        )
    return resolved


def load_trusted_checksums(backup_dir: Path | None, require_file: bool = True) -> dict[str, str]:
    """Load trusted SHA-256 checksums from checksums.sha256 in backup_dir.

    Strictly validates:
    - backup_dir must be provided and exist.
    - checksums.sha256 must exist if require_file=True.
    - Each line must contain: <64-hex SHA-256> [*]<filename>.
    - Reject malformed, non-hex, duplicate, or ambiguous entries.
    """
    if not backup_dir:
        raise RuntimeError("FATAL: backup_dir not provided when loading trusted checksums.")
    backup_dir = Path(backup_dir)
    if not backup_dir.exists() or not backup_dir.is_dir():
        raise RuntimeError(f"FATAL: backup_dir does not exist or is not a directory: {backup_dir}")

    checksum_file = backup_dir / "checksums.sha256"
    if not checksum_file.exists():
        if require_file:
            raise RuntimeError(
                f"FATAL: Required trusted checksums file missing at {checksum_file}. "
                "Cannot establish trusted baseline independently of transaction journal."
            )
        return {}

    checksums: dict[str, str] = {}
    with open(checksum_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines and require_file:
        raise RuntimeError(f"FATAL: Trusted checksums file {checksum_file} is empty.")

    valid_entry_found = False
    for line_idx, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise RuntimeError(
                f"FATAL: Malformed entry in {checksum_file} line {line_idx}: expected '<hash> <filename>', got '{line}'"
            )
        sha, name = parts
        sha = sha.strip().lower()
        if not HEX_64_PATTERN.match(sha):
            raise RuntimeError(
                f"FATAL: Malformed SHA-256 hash in {checksum_file} line {line_idx}: '{sha}' (must be 64 hex characters)"
            )
        cleaned_name = name.strip()
        if cleaned_name.startswith("*"):
            cleaned_name = cleaned_name[1:].strip()
        if "/" in cleaned_name or "\\" in cleaned_name or ".." in cleaned_name:
            raise RuntimeError(
                f"FATAL: Path traversal or directory separator rejected in {checksum_file} line {line_idx}: '{cleaned_name}'"
            )
        filename = Path(cleaned_name).name
        if not filename:
            raise RuntimeError(
                f"FATAL: Missing filename in {checksum_file} line {line_idx}: '{line}'"
            )

        if filename in checksums:
            if checksums[filename] != sha:
                raise RuntimeError(
                    f"FATAL: Ambiguous conflicting checksums for '{filename}' in {checksum_file} line {line_idx}"
                )
            raise RuntimeError(
                f"FATAL: Duplicate checksum entry for '{filename}' in {checksum_file} line {line_idx}"
            )

        checksums[filename] = sha
        valid_entry_found = True

    if require_file and not valid_entry_found:
        raise RuntimeError(f"FATAL: No valid checksum entries found in {checksum_file}.")

    return checksums


def check_and_recover_transactions(
    journal_path: Path | None = None,
    allowed_roots: list[Path] | None = None,
    backup_dir: Path | None = None,
) -> bool:
    """Check for interrupted multi-file transactions and restore consistent state before execution."""
    effective_journal_roots = allowed_roots if allowed_roots is not None else DEFAULT_ALLOWED_JOURNAL_ROOTS
    effective_backup_roots = allowed_roots if allowed_roots is not None else DEFAULT_ALLOWED_BACKUP_ROOTS
    effective_data_roots = allowed_roots if allowed_roots is not None else DEFAULT_ALLOWED_DATA_ROOTS

    if journal_path is None:
        if backup_dir is not None:
            candidate = Path(backup_dir) / "transaction_journal.json"
            if is_safe_contained_path(candidate.resolve(), effective_journal_roots):
                journal_path = candidate
            else:
                journal_path = DEFAULT_TRANSACTION_JOURNAL
        elif allowed_roots:
            journal_path = allowed_roots[0] / "transaction_journal.json"
        else:
            journal_path = DEFAULT_TRANSACTION_JOURNAL
    else:
        journal_path = Path(journal_path)

    validate_journal_path(journal_path, effective_journal_roots)

    if not journal_path.exists():
        return False

    try:
        with open(journal_path, "r", encoding="utf-8") as f:
            journal = json.load(f)
    except Exception as exc:
        raise RuntimeError(
            f"FATAL: Malformed or unreadable transaction journal at {journal_path}: {exc}. Fail-closed."
        ) from exc

    if not isinstance(journal, dict):
        raise RuntimeError(
            f"FATAL: Malformed transaction journal at {journal_path}: root must be a JSON object, got {type(journal).__name__}."
        )

    state = journal.get("state")
    if state in (TX_STATE_COMMITTED, TX_STATE_ABORTED_RECOVERED):
        return False

    if state not in (TX_STATE_STAGED, TX_STATE_RECOVERING):
        raise RuntimeError(
            f"FATAL: Unknown or ambiguous transaction state '{state}' in {journal_path}. Halting pipeline."
        )

    files = journal.get("files")
    if not isinstance(files, list) or len(files) == 0:
        raise RuntimeError(
            f"FATAL: Transaction journal in '{state}' state has no valid files list: {journal_path}"
        )

    backup_dir_str = journal.get("backup_dir", "")
    if not backup_dir_str:
        raise RuntimeError(
            f"FATAL: Transaction journal has missing or empty 'backup_dir' field: {journal_path}"
        )
    backup_dir = Path(backup_dir_str)
    validate_directory_safety(backup_dir, effective_backup_roots)

    if not backup_dir.exists() or not backup_dir.is_dir():
        raise RuntimeError(
            f"FATAL: Backup directory specified in journal does not exist: {backup_dir}"
        )

    trusted_hashes = load_trusted_checksums(backup_dir, require_file=True)

    # -------------------------------------------------------------
    # PHASE 1: PRE-FLIGHT VALIDATION (All or Nothing - Zero writes)
    # -------------------------------------------------------------
    validated_plan: list[dict[str, Any]] = []
    for item in files:
        if not isinstance(item, dict) or "target" not in item:
            raise RuntimeError(f"FATAL: Malformed file entry in transaction journal: {item}")

        target = Path(item["target"])
        validate_target_path(target, effective_data_roots)

        staged_path_str = item.get("staged")
        staged = Path(staged_path_str) if staged_path_str else target.with_name(target.name + ".stage")
        validate_staged_path(staged, target, effective_data_roots)

        if "backup" in item and item["backup"]:
            backup = Path(item["backup"])
        else:
            backup = backup_dir / target.name

        validate_backup_path(backup, effective_backup_roots)

        if not backup.exists():
            raise RuntimeError(
                f"FATAL: Required backup missing: {backup}. Cannot safely restore baseline. Preserving all artifacts."
            )

        if target.name not in trusted_hashes:
            raise RuntimeError(
                f"FATAL: No trusted baseline checksum found for '{target.name}' in {backup_dir / 'checksums.sha256'}."
            )
        trusted_sha = trusted_hashes[target.name]

        journal_sha = item.get("backup_sha256")
        if journal_sha:
            clean_journal_sha = journal_sha.strip().lower()
            if not HEX_64_PATTERN.match(clean_journal_sha):
                raise RuntimeError(
                    f"FATAL: Malformed SHA-256 hash in journal for {target.name}: '{journal_sha}'"
                )
            if clean_journal_sha != trusted_sha:
                raise RuntimeError(
                    f"FATAL: Checksum conflict for {target.name}: journal records {journal_sha}, "
                    f"but trusted baseline records {trusted_sha}."
                )

        actual_sha = compute_sha256(backup)
        if actual_sha != trusted_sha:
            raise RuntimeError(
                f"FATAL: Backup integrity check failed for {backup}: expected {trusted_sha}, found {actual_sha}."
            )

        validated_plan.append({
            "target": target,
            "staged": staged,
            "backup": backup,
            "expected_sha": trusted_sha,
        })

    logger.critical(
        f"Detected incomplete transaction ({journal.get('transaction_id')}) in state '{state}'. "
        f"All {len(validated_plan)} backups verified against trusted baseline. Initiating durable rollback..."
    )

    # -------------------------------------------------------------
    # PHASE 2: ATOMIC RESTORATION WITH PROGRESS TRACKING
    # -------------------------------------------------------------
    journal["state"] = TX_STATE_RECOVERING
    journal["recovering_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    durable_json_write(journal_path, journal)

    try:
        for item_meta in validated_plan:
            backup = item_meta["backup"]
            target = item_meta["target"]
            expected_sha = item_meta["expected_sha"]

            tmp_restore = target.with_suffix(target.suffix + ".restore_tmp")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, tmp_restore)

            restored_tmp_sha = compute_sha256(tmp_restore)
            if restored_tmp_sha != expected_sha:
                if tmp_restore.exists():
                    tmp_restore.unlink()
                raise RuntimeError(
                    f"FATAL: Checksum verification failed on temporary restore file {tmp_restore}: "
                    f"expected {expected_sha}, got {restored_tmp_sha}"
                )

            tmp_restore.replace(target)
            logger.info(f"Atomically restored {target} from verified backup {backup}")

        # -------------------------------------------------------------
        # PHASE 3: POST-RESTORATION VERIFICATION & STAGED CLEANUP
        # -------------------------------------------------------------
        for item_meta in validated_plan:
            target = item_meta["target"]
            expected_sha = item_meta["expected_sha"]
            if not target.exists() or compute_sha256(target) != expected_sha:
                raise RuntimeError(
                    f"FATAL: Post-restoration verification failed for {target}. Baseline not restored."
                )

        for item_meta in validated_plan:
            staged = item_meta["staged"]
            if staged.exists():
                try:
                    staged.unlink()
                    logger.info(f"Cleaned up staged file after verified recovery: {staged}")
                except Exception as e:
                    logger.warning(f"Failed to remove staged file {staged}: {e}")

        journal["state"] = TX_STATE_ABORTED_RECOVERED
        journal["recovered_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        durable_json_write(journal_path, journal)

        logger.info("Transaction recovery complete. All files restored and verified against baseline checksums.")
        return True

    except Exception as exc:
        logger.critical(
            f"FATAL: Recovery failed during restoration: {exc}. "
            "Journal preserved in RECOVERING state, staged artifacts preserved."
        )
        raise


def commit_multi_file_transaction(
    staged_targets: list[tuple[Path, Path]],
    backup_dir: Path | None = None,
    journal_path: Path | None = None,
    allowed_roots: list[Path] | None = None,
) -> None:
    """Commit multiple staged files atomically with write-ahead journal tracking."""
    if backup_dir is None:
        raise ValueError("commit_multi_file_transaction requires a valid backup_dir containing pre-transaction backups.")
    backup_dir = Path(backup_dir)

    effective_journal_roots = allowed_roots if allowed_roots is not None else DEFAULT_ALLOWED_JOURNAL_ROOTS
    effective_backup_roots = allowed_roots if allowed_roots is not None else DEFAULT_ALLOWED_BACKUP_ROOTS
    effective_data_roots = allowed_roots if allowed_roots is not None else DEFAULT_ALLOWED_DATA_ROOTS

    if journal_path is None:
        candidate = backup_dir / "transaction_journal.json"
        if is_safe_contained_path(candidate.resolve(), effective_journal_roots):
            journal_path = candidate
        else:
            journal_path = DEFAULT_TRANSACTION_JOURNAL
    else:
        journal_path = Path(journal_path)

    validate_journal_path(journal_path, effective_journal_roots)
    validate_directory_safety(backup_dir, effective_backup_roots)

    if not backup_dir.exists() or not backup_dir.is_dir():
        raise RuntimeError(f"FATAL: Specified backup_dir does not exist or is not a directory: {backup_dir}")

    journal_path.parent.mkdir(parents=True, exist_ok=True)
    tx_id = f"tx_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    trusted_hashes = load_trusted_checksums(backup_dir, require_file=True)
    files_meta: list[dict[str, str]] = []

    for staged, target in staged_targets:
        validate_target_path(target, effective_data_roots)
        validate_staged_path(staged, target, effective_data_roots)

        if not staged.exists():
            raise RuntimeError(f"FATAL: Staged file does not exist before commit: {staged}")

        backup = backup_dir / target.name
        validate_backup_path(backup, effective_backup_roots)
        if not backup.exists():
            raise RuntimeError(f"FATAL: Missing required backup for target {target}: {backup} does not exist.")

        if target.name not in trusted_hashes:
            raise RuntimeError(
                f"FATAL: Target '{target.name}' missing from trusted checksums file {backup_dir / 'checksums.sha256'}."
            )
        trusted_sha = trusted_hashes[target.name]

        actual_backup_sha = compute_sha256(backup)
        if actual_backup_sha != trusted_sha:
            raise RuntimeError(
                f"FATAL: Backup {backup} failed checksum verification against checksums.sha256: "
                f"expected {trusted_sha}, got {actual_backup_sha}"
            )

        files_meta.append({
            "target": str(target.resolve()),
            "staged": str(staged.resolve()),
            "backup": str(backup.resolve()),
            "backup_sha256": actual_backup_sha,
        })

    journal_data = {
        "transaction_id": tx_id,
        "state": TX_STATE_STAGED,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "backup_dir": str(backup_dir.resolve()),
        "files": files_meta,
    }

    durable_json_write(journal_path, journal_data)

    try:
        for staged, target in staged_targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            staged.replace(target)

        journal_data["state"] = TX_STATE_COMMITTED
        journal_data["committed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        durable_json_write(journal_path, journal_data)
    except Exception as exc:
        logger.critical(f"Error during atomic commit phase: {exc}. Invoking immediate recovery.")
        check_and_recover_transactions(journal_path, allowed_roots=allowed_roots)
        raise


def update_manifest_chunk(
    manifest_path: Path,
    chunk_key: str,
    status: str,
    article_count: int,
) -> None:
    """Atomically record chunk retrieval status and article count to manifest."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"chunks": {}, "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
        except Exception:
            pass

    if "chunks" not in manifest:
        manifest["chunks"] = {}

    manifest["chunks"][chunk_key] = {
        "status": status,
        "article_count": article_count,
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    manifest["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()

    tmp_manifest = manifest_path.with_suffix(".tmp")
    with open(tmp_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    tmp_manifest.replace(manifest_path)


def get_net_new_articles(
    incoming_raw_df: pd.DataFrame,
    existing_processed_df: pd.DataFrame,
) -> pd.DataFrame:
    """Identify articles in incoming_raw_df that are not already present in existing_processed_df."""
    if incoming_raw_df.empty or existing_processed_df.empty:
        return incoming_raw_df

    proc = existing_processed_df.copy()
    proc["published_at"] = pd.to_datetime(proc["published_at"], utc=True)

    proc_ids = set(proc["raw_id"].fillna("").astype(str).str.strip().loc[lambda s: s != ""])
    proc_urls = set(proc["url"].fillna("").astype(str).str.strip().loc[lambda s: s != ""])
    proc_head_ts = set(
        zip(
            proc["headline"].astype(str).str.lower().str.strip(),
            proc["published_at"],
        )
    )

    incoming = incoming_raw_df.copy()
    incoming["published_at"] = pd.to_datetime(incoming["published_at"], utc=True)

    is_new = []
    for _, row in incoming.iterrows():
        raw_id = str(row.get("raw_id", "") or "").strip()
        url = str(row.get("url", "") or "").strip()
        head_ts = (str(row.get("headline", "") or "").lower().strip(), row["published_at"])

        if raw_id and raw_id in proc_ids:
            is_new.append(False)
        elif url and url in proc_urls:
            is_new.append(False)
        elif head_ts in proc_head_ts:
            is_new.append(False)
        else:
            is_new.append(True)

    return incoming_raw_df.loc[is_new].reset_index(drop=True)


def fetch_finnhub_news(
    ticker: str,
    start_date: str,
    end_date: str,
    client: finnhub.Client | None = None,
    chunk_days: int = 1,
    retry_delay: float = 1.0,
    request_delay_seconds: float = 0.15,
    staged_chunks_dir: Path | None = None,
    manifest_path: Path | None = None,
    max_cache_age_hours: float | None = None,
) -> pd.DataFrame:
    """Fetch company news from Finnhub in chunked date intervals with rate-limit handling and staging.

    Uses short date chunks (default 1 calendar day per request) to prevent truncation.
    Finnhub silently truncates responses at ~245-250 items sorted newest-first.

    Durable staging:
    If staged_chunks_dir is provided, successfully retrieved payloads are persisted per-date
    (or marked as empty via .empty marker), allowing offline resumption without re-fetching.

    Distinguishes three collection outcomes via df.attrs["status"]:
    - complete: All requested chunks were retrieved successfully (even if 0 articles).
    - partial: Some chunks failed, but >= 1 chunk succeeded; returns articles from successful chunks.
    - failed: All requested chunks failed after retries.
    """
    start_dt = pd.to_datetime(start_date).date()
    end_dt = pd.to_datetime(end_date).date()

    articles: list[dict[str, Any]] = []
    chunks = generate_date_chunks(start_dt, end_dt, chunk_days=chunk_days)
    failed_chunks: list[tuple[str, str]] = []
    successful_chunks: list[tuple[str, str]] = []

    if staged_chunks_dir is not None:
        staged_chunks_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Querying Finnhub for {ticker} across {len(chunks)} chunk(s) "
        f"({start_dt} to {end_dt}, chunk_days={chunk_days})..."
    )

    for chunk_idx, (from_str, to_str) in enumerate(chunks, 1):
        chunk_stem = f"{ticker}_{from_str}" if from_str == to_str else f"{ticker}_{from_str}_{to_str}"
        chunk_key = f"{ticker}:{from_str}:{to_str}"
        parquet_file = (staged_chunks_dir / f"{chunk_stem}.parquet") if staged_chunks_dir else None
        empty_file = (staged_chunks_dir / f"{chunk_stem}.empty") if staged_chunks_dir else None

        # Check manifest status if manifest exists: do not treat a failed retrieval as cached
        chunk_failed_in_manifest = False
        if manifest_path and manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as mf:
                    m_data = json.load(mf)
                c_info = m_data.get("chunks", {}).get(chunk_key) or m_data.get("chunks", {}).get(f"{from_str}:{to_str}")
                if c_info and c_info.get("status") == STATUS_FAILED:
                    chunk_failed_in_manifest = True
            except Exception:
                pass

        # Check expiration if configured
        is_expired = False
        if max_cache_age_hours is not None:
            cache_target = empty_file if (empty_file and empty_file.exists()) else parquet_file
            if cache_target and cache_target.exists():
                age_seconds = time.time() - cache_target.stat().st_mtime
                if age_seconds > max_cache_age_hours * 3600.0:
                    is_expired = True
                    logger.info(
                        f"Cached chunk {chunk_stem} has expired (age: {age_seconds / 3600.0:.2f}h > {max_cache_age_hours}h). "
                        "Attempting refresh from client..."
                    )

        # Check for pre-existing staged payload (if not expired and not marked failed in manifest)
        if not is_expired and not chunk_failed_in_manifest:
            if empty_file and empty_file.exists():
                logger.info(f"Loaded staged empty chunk for [{from_str} to {to_str}].")
                successful_chunks.append((from_str, to_str))
                if manifest_path:
                    update_manifest_chunk(manifest_path, chunk_key, STATUS_COMPLETE, 0)
                continue

            if parquet_file and parquet_file.exists():
                try:
                    staged_df = pd.read_parquet(parquet_file)
                    for col in NEWS_COLUMNS:
                        if col not in staged_df.columns:
                            raise ValueError(f"Corrupted chunk cache: missing required column '{col}'")
                    staged_records = staged_df.to_dict(orient="records")
                    for rec in staged_records:
                        rec["published_at"] = pd.to_datetime(rec["published_at"], utc=True)
                        articles.append(rec)
                    logger.info(
                        f"Loaded {len(staged_records)} staged article(s) for [{from_str} to {to_str}] from {parquet_file.name}."
                    )
                    successful_chunks.append((from_str, to_str))
                    if manifest_path:
                        update_manifest_chunk(manifest_path, chunk_key, STATUS_COMPLETE, len(staged_records))
                    continue
                except Exception as exc:
                    logger.warning(f"Failed to read staged chunk file {parquet_file}: {exc}. Refetching...")

        # If not already staged, fetch from client
        if client is None:
            client = get_finnhub_client()
            if client is None:
                raise RuntimeError("Finnhub client could not be initialized.")

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
                err_msg = str(exc).lower()
                is_rate_limit = "429" in err_msg or "rate limit" in err_msg
                logger.warning(
                    f"Finnhub request failed (attempt {attempt + 1}/{retries}) "
                    f"for [{from_str} to {to_str}]: {type(exc).__name__} ({clean_err})"
                )
                if retry_delay > 0:
                    sleep_secs = max(retry_delay * (attempt + 1) * 2.0, 2.0) if is_rate_limit else retry_delay * (attempt + 1)
                    time.sleep(sleep_secs)

        if not success:
            failed_chunks.append((from_str, to_str))
            logger.error(
                f"Failed to fetch Finnhub news for window [{from_str} to {to_str}] after {retries} retries."
            )
            # If this was an expired cache refresh attempt, preserve existing valid cache payload on disk
            if is_expired and parquet_file and parquet_file.exists():
                logger.warning(
                    f"Refresh failed for expired chunk [{from_str} to {to_str}]. "
                    f"Preserving existing valid cache payload on disk."
                )
            elif manifest_path:
                update_manifest_chunk(manifest_path, chunk_key, STATUS_FAILED, 0)
            continue

        successful_chunks.append((from_str, to_str))

        chunk_articles: list[dict[str, Any]] = []
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
            art = {
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
            chunk_articles.append(art)
            articles.append(art)

        # Stage payload durably using atomic write
        if staged_chunks_dir is not None:
            if not chunk_articles:
                empty_file.touch()
                if parquet_file and parquet_file.exists():
                    try:
                        parquet_file.unlink()
                    except Exception:
                        pass
            else:
                tmp_chunk = parquet_file.with_suffix(".parquet.tmp")
                pd.DataFrame(chunk_articles).to_parquet(tmp_chunk)
                tmp_chunk.replace(parquet_file)
                if empty_file and empty_file.exists():
                    try:
                        empty_file.unlink()
                    except Exception:
                        pass

        if manifest_path is not None:
            update_manifest_chunk(manifest_path, chunk_key, STATUS_COMPLETE, len(chunk_articles))

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

        if request_delay_seconds > 0 and chunk_idx < len(chunks):
            time.sleep(request_delay_seconds)

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
    request_delay_seconds: float = 0.15,
    staged_chunks_dir: Path | None = None,
    manifest_path: Path | None = None,
    max_cache_age_hours: float | None = None,
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
                request_delay_seconds=request_delay_seconds,
                staged_chunks_dir=staged_chunks_dir,
                manifest_path=manifest_path,
                max_cache_age_hours=max_cache_age_hours,
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
    successful_trading_days: set[dt.date] | list[dt.date] | None = None,
    full_rebuild: bool = False,
) -> pd.DataFrame:
    """Merge daily sentiment into market_features.parquet safely and incrementally.

    Behavior:
    1. full_rebuild=False (Incremental Mode, Default):
       - Preserves sentiment values for market dates not being refreshed.
       - If successful_trading_days is provided, only market sessions in that set are
         refreshed; all other dates (failed chunks, unattempted dates, or historical dates)
         are strictly untouched.
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
        elif successful_trading_days is not None:
            succ_set = {pd.to_datetime(d).date() for d in successful_trading_days}
            market_dates = pd.Series([d.date() for d in market_df.index], index=market_df.index)
            refresh_mask = market_dates.isin(succ_set)
            logger.info(
                f"Empty sentiment provided with explicit successful trading days. "
                f"Marking {int(refresh_mask.sum())} session(s) as NaN with 0 coverage."
            )
            for col in SENTIMENT_METRIC_COLUMNS:
                market_df.loc[refresh_mask, col] = np.nan
            market_df.loc[refresh_mask, "sentiment_coverage"] = 0
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
            if successful_trading_days is not None:
                succ_set = {pd.to_datetime(d).date() for d in successful_trading_days}
                market_dates = pd.Series([d.date() for d in market_df.index], index=market_df.index)
                refresh_mask = market_dates.isin(succ_set)
                logger.info(
                    f"Executing incremental sentiment update for {int(refresh_mask.sum())} explicit successful session(s)."
                )
            elif refresh_start_date is not None and refresh_end_date is not None:
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
    test_tmp_dir = tempfile.TemporaryDirectory()
    temp_market_path = Path(test_tmp_dir.name) / "temp_test_market_features.parquet"
    dates = pd.date_range("2026-10-12", periods=5, freq="B")
    base_market = pd.DataFrame(
        {"Close": [100.0, 101.0, 102.0, 103.0, 104.0], "Daily_Return": [0.01, 0.01, 0.01, 0.01, 0.01]},
        index=dates,
    )
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
        test_tmp_dir.cleanup()

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
    request_delay_seconds: float = 0.15,
    staged_chunks_dir: Path | None = None,
    manifest_path: Path | None = None,
    backup_dir: Path | None = None,
    journal_path: Path | None = None,
    allowed_roots: list[Path] | None = None,
    max_cache_age_hours: float | None = None,
) -> dict[str, Any]:
    """Execute news ingestion, FinBERT scoring, and market features merge with transaction safety.

    Safely distinguishes:
    1. Complete: All chunks succeeded. If 0 articles, confirmed empty without modifying features.
    2. Partial: Some chunks failed; preserves articles from successful chunks, and restricts
       incremental update strictly to observed dates to prevent wiping unretrieved dates.
    3. Failed: All chunks failed; aborts pipeline to prevent clearing/overwriting historical data.
    4. Crash-Resilience: Supports Write-Ahead Transaction Journal and per-chunk staging.
    """
    if end_date is None:
        end_date = dt.date.today().strftime("%Y-%m-%d")

    raw_file = raw_news_path or RAW_NEWS_FILE
    proc_file = processed_news_path or PROCESSED_NEWS_FILE
    market_file = market_features_path or MARKET_FEATURES_FILE

    raw_file.parent.mkdir(parents=True, exist_ok=True)
    proc_file.parent.mkdir(parents=True, exist_ok=True)

    # Resolve and validate cache and manifest paths
    effective_data_roots = allowed_roots if allowed_roots is not None else (DEFAULT_ALLOWED_DATA_ROOTS + [raw_file.parent])
    effective_staged_chunks_dir = Path(staged_chunks_dir) if staged_chunks_dir is not None else (raw_file.parent / "staged_chunks")
    effective_manifest_path = Path(manifest_path) if manifest_path is not None else (raw_file.parent / "news_retrieval_manifest.json")

    validate_path_safety(effective_staged_chunks_dir, effective_data_roots)
    validate_path_safety(effective_manifest_path, effective_data_roots)
    if effective_manifest_path.suffix.lower() != ".json":
        raise RuntimeError(f"FATAL: Manifest path '{effective_manifest_path}' rejected: must have '.json' extension.")

    # 0. Startup transaction recovery
    effective_journal_roots = allowed_roots if allowed_roots is not None else DEFAULT_ALLOWED_JOURNAL_ROOTS
    if journal_path is not None:
        journal_file = Path(journal_path)
    elif backup_dir is not None:
        candidate = Path(backup_dir) / "transaction_journal.json"
        if is_safe_contained_path(candidate.resolve(), effective_journal_roots):
            journal_file = candidate
        else:
            journal_file = DEFAULT_TRANSACTION_JOURNAL
    elif allowed_roots:
        journal_file = allowed_roots[0] / "transaction_journal.json"
    else:
        journal_file = DEFAULT_TRANSACTION_JOURNAL

    try:
        recovered = check_and_recover_transactions(
            journal_path=journal_file,
            allowed_roots=allowed_roots,
            backup_dir=Path(backup_dir) if backup_dir is not None else None,
        )
    except Exception as exc:
        logger.critical(f"Startup crash recovery failed: {exc}. Refusing to start pipeline.")
        raise RuntimeError(f"Startup crash recovery failed: {exc}") from exc

    if recovered:
        logger.critical(
            "Transaction rollback completed. Repository has been restored to pre-trial baseline. "
            "Halting pipeline execution to prevent overwriting without operator confirmation."
        )
        raise RuntimeError("Transaction rollback completed. Pipeline halted for safety review.")

    # 1. Ingest raw news
    raw_news = fetch_news(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        include_fallback_supplement=True,
        chunk_days=chunk_days,
        client=client,
        retry_delay=retry_delay,
        request_delay_seconds=request_delay_seconds,
        staged_chunks_dir=effective_staged_chunks_dir,
        manifest_path=effective_manifest_path,
        max_cache_age_hours=max_cache_age_hours,
    )

    fetch_status = raw_news.attrs.get("status", STATUS_COMPLETE)

    if fetch_status == STATUS_FAILED:
        logger.error(
            f"Finnhub news retrieval failed completely for {ticker} [{start_date} to {end_date}]. "
            "Aborting pipeline to avoid overwriting or clearing historical sentiment."
        )
        return {"status": STATUS_FAILED, "raw_news_count": len(raw_news), "processed_count": 0, "covered_sessions": 0}

    # Map successful chunks to trading sessions
    successful_chunks = raw_news.attrs.get("successful_chunks", [])
    successful_calendar_days = set()
    for from_str, to_str in successful_chunks:
        cur = pd.to_datetime(from_str).date()
        end_cur = pd.to_datetime(to_str).date()
        while cur <= end_cur:
            successful_calendar_days.add(cur)
            cur += dt.timedelta(days=1)

    successful_trading_days = None
    if market_file.exists():
        market_df_check = pd.read_parquet(market_file)
        successful_trading_days = [
            d.date() for d in pd.to_datetime(market_df_check.index)
            if d.date() in successful_calendar_days
        ]

    if raw_news.empty:
        if fetch_status == STATUS_COMPLETE:
            logger.info(
                f"News retrieval complete for {ticker} [{start_date} to {end_date}]: "
                "confirmed zero articles published."
            )
        elif fetch_status == STATUS_PARTIAL:
            logger.warning(
                f"News retrieval was partial for {ticker} [{start_date} to {end_date}] "
                "and returned zero articles. Exiting without modifying market features."
            )
        else:
            logger.error("No news articles collected. Exiting pipeline.")

        if full_rebuild and market_file.exists():
            if backup_dir is None:
                raise RuntimeError(
                    "FATAL: Production pipeline execution requires an explicit, verified backup_dir before writing to datasets. "
                    "Direct overwrite is strictly prohibited."
                )
            staged_market = market_file.with_name(market_file.name + ".stage")
            merged = merge_sentiment_with_market_features(
                market_features_path=market_file,
                daily_sentiment_df=pd.DataFrame(),
                output_path=staged_market,
                full_rebuild=True,
            )
            commit_multi_file_transaction(
                staged_targets=[(staged_market, market_file)],
                backup_dir=backup_dir,
                journal_path=journal_file,
                allowed_roots=allowed_roots,
            )
            return {"status": fetch_status, "raw_news_count": 0, "processed_count": 0, "covered_sessions": int(merged["sentiment_coverage"].sum())}

        # Check if existing processed sentiment has eligible records for requested successful trading sessions
        if not full_rebuild and fetch_status == STATUS_COMPLETE and market_file.exists() and proc_file.exists() and successful_trading_days:
            existing_proc_df = pd.read_parquet(proc_file)
            if not existing_proc_df.empty and "trading_date" in existing_proc_df.columns:
                existing_trading_dates = pd.to_datetime(existing_proc_df["trading_date"]).dt.date
                eligible_mask = existing_trading_dates.isin(set(successful_trading_days))
                if "ticker" in existing_proc_df.columns:
                    eligible_mask = eligible_mask & (existing_proc_df["ticker"] == ticker)

                if eligible_mask.any():
                    logger.info(
                        f"Found {int(eligible_mask.sum())} existing processed article(s) eligible for "
                        f"{len(successful_trading_days)} requested trading session(s). Aggregating sentiment..."
                    )
                    if backup_dir is None:
                        raise RuntimeError(
                            "FATAL: Production pipeline execution requires an explicit, verified backup_dir before writing to datasets. "
                            "Direct overwrite is strictly prohibited."
                        )
                    daily_sentiment = aggregate_daily_sentiment(existing_proc_df)
                    staged_market = market_file.with_name(market_file.name + ".stage")
                    merged_market = merge_sentiment_with_market_features(
                        market_features_path=market_file,
                        daily_sentiment_df=daily_sentiment,
                        output_path=staged_market,
                        successful_trading_days=successful_trading_days,
                        full_rebuild=False,
                    )
                    covered_sessions = int(merged_market["sentiment_coverage"].sum())
                    commit_multi_file_transaction(
                        staged_targets=[(staged_market, market_file)],
                        backup_dir=backup_dir,
                        journal_path=journal_file,
                        allowed_roots=allowed_roots,
                    )
                    raw_count = len(pd.read_parquet(raw_file)) if raw_file.exists() else 0
                    return {
                        "status": fetch_status,
                        "raw_news_count": raw_count,
                        "processed_count": len(existing_proc_df),
                        "covered_sessions": covered_sessions,
                    }

        return {"status": fetch_status, "raw_news_count": 0, "processed_count": 0, "covered_sessions": 0}

    # Mandatory transaction safety check: before any writes, require verified backup_dir
    if backup_dir is None:
        raise RuntimeError(
            "FATAL: Production pipeline execution requires an explicit, verified backup_dir before writing to datasets. "
            "Direct overwrite is strictly prohibited."
        )

    staged_raw = raw_file.with_name(raw_file.name + ".stage")
    staged_proc = proc_file.with_name(proc_file.name + ".stage")

    # 2. Merge raw news archive
    if full_rebuild:
        merged_raw_df = deduplicate_articles(raw_news)
        merged_raw_df.to_parquet(staged_raw)
    else:
        merged_raw_df = merge_raw_news_archive(raw_file, raw_news, output_path=staged_raw)

    # 3. Process sentiment with FinBERT (net-new headlines only in incremental mode)
    if full_rebuild:
        final_processed_df = process_news_sentiment(
            raw_news_df=merged_raw_df,
            batch_size=batch_size,
            analyzer=analyzer,
        )
        final_processed_df.to_parquet(staged_proc)
    else:
        if proc_file.exists():
            existing_proc_df = pd.read_parquet(proc_file)
            net_new_raw = get_net_new_articles(raw_news, existing_proc_df)
            if not net_new_raw.empty:
                logger.info(f"Identified {len(net_new_raw)} net-new headlines for FinBERT scoring.")
                newly_processed = process_news_sentiment(
                    raw_news_df=net_new_raw,
                    batch_size=batch_size,
                    analyzer=analyzer,
                )
                combined_proc = pd.concat([existing_proc_df, newly_processed], ignore_index=True)
                final_processed_df = deduplicate_articles(combined_proc)
                final_processed_df.to_parquet(staged_proc)
            else:
                logger.info("Zero net-new headlines identified. Skipping FinBERT inference.")
                final_processed_df = existing_proc_df
                shutil.copy2(proc_file, staged_proc)
        else:
            final_processed_df = process_news_sentiment(
                raw_news_df=merged_raw_df,
                batch_size=batch_size,
                analyzer=analyzer,
            )
            final_processed_df.to_parquet(staged_proc)

    # 4. Aggregate daily sentiment by trading session
    daily_sentiment = aggregate_daily_sentiment(final_processed_df)

    # 5. Commit with transaction journal
    staged_items = [(staged_raw, raw_file), (staged_proc, proc_file)]

    covered_sessions = 0
    if market_file.exists():
        staged_market = market_file.with_name(market_file.name + ".stage")
        merged_market = merge_sentiment_with_market_features(
            market_features_path=market_file,
            daily_sentiment_df=daily_sentiment,
            output_path=staged_market,
            successful_trading_days=successful_trading_days,
            full_rebuild=full_rebuild,
        )
        covered_sessions = int(merged_market["sentiment_coverage"].sum())
        staged_items.append((staged_market, market_file))

    commit_multi_file_transaction(
        staged_targets=staged_items,
        backup_dir=backup_dir,
        journal_path=journal_file,
        allowed_roots=allowed_roots,
    )

    logger.info(f"Saved {len(merged_raw_df)} total raw news articles to {raw_file}")
    logger.info(f"Saved {len(final_processed_df)} total processed news sentiment records to {proc_file}")
    if market_file.exists():
        print("\nPipeline execution complete.")
        print(f"Trading days with sentiment coverage: {covered_sessions}")

    return {
        "status": fetch_status,
        "raw_news_count": len(merged_raw_df),
        "processed_count": len(final_processed_df),
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
    parser.add_argument(
        "--backup-dir",
        type=str,
        default=None,
        help="Path to verified pre-transaction backup directory containing checksums.sha256",
    )
    parser.add_argument(
        "--journal-path",
        type=str,
        default=None,
        help="Path to write-ahead transaction journal JSON file",
    )
    args = parser.parse_args()

    if args.test:
        run_test_mode(ticker=args.ticker)
        return

    backup_dir = Path(args.backup_dir) if args.backup_dir else None
    journal_path = Path(args.journal_path) if args.journal_path else None

    result = run_production_pipeline(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        chunk_days=args.chunk_days,
        full_rebuild=args.full_rebuild,
        batch_size=args.batch_size,
        backup_dir=backup_dir,
        journal_path=journal_path,
    )

    if isinstance(result, dict) and result.get("status") == STATUS_FAILED:
        logger.error(f"Pipeline execution reported failure status: {result}")
        sys.exit(1)



if __name__ == "__main__":
    main()
