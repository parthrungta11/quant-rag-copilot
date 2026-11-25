"""Read-only diagnostic script to inspect canonical dataset counts and forecast staleness.

Never modifies or writes any files. Compares live parquet counts against
tests/fixtures/canonical_dataset_state.json and checks whether
data/processed/forecast_predictions.parquet is stale relative to
data/processed/market_features.parquet.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "tests" / "fixtures" / "canonical_dataset_state.json"
MARKET_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "market_features.parquet"
NEWS_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "news_raw.parquet"
NEWS_SENTIMENT_PATH = PROJECT_ROOT / "data" / "processed" / "news_sentiment.parquet"
FORECAST_PREDS_PATH = PROJECT_ROOT / "data" / "processed" / "forecast_predictions.parquet"


def check_data_state(
    manifest_path: Path = MANIFEST_PATH,
    market_features_path: Path = MARKET_FEATURES_PATH,
    news_raw_path: Path = NEWS_RAW_PATH,
    news_sentiment_path: Path = NEWS_SENTIMENT_PATH,
    forecast_preds_path: Path = FORECAST_PREDS_PATH,
) -> Dict[str, Any]:
    """Inspect dataset counts and staleness strictly read-only."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest missing: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    mkt = pd.read_parquet(market_features_path)
    raw = pd.read_parquet(news_raw_path)
    proc = pd.read_parquet(news_sentiment_path)

    actual = {
        "total_sessions": int(len(mkt)),
        "covered_sessions": int((mkt["sentiment_coverage"] == 1).sum()),
        "missing_sessions": int(mkt["sentiment_score"].isna().sum()),
        "news_raw_rows": int(len(raw)),
        "news_sentiment_rows": int(len(proc)),
        "unique_trading_dates": int(proc["trading_date"].nunique()),
    }

    mismatches = [k for k, v in actual.items() if v != manifest.get(k)]

    is_stale = False
    preds_cov = None
    preds_len = None
    if forecast_preds_path.exists():
        preds = pd.read_parquet(forecast_preds_path)
        preds_len = int(len(preds))
        preds_cov = int((preds["sentiment_coverage"] == 1).sum())
        mkt_mtime = market_features_path.stat().st_mtime
        preds_mtime = forecast_preds_path.stat().st_mtime
        if preds_cov < actual["covered_sessions"] or preds_mtime < mkt_mtime:
            is_stale = True

    return {
        "verified_on": manifest.get("verified_on"),
        "manifest": manifest,
        "actual": actual,
        "mismatches": mismatches,
        "forecast_rows": preds_len,
        "forecast_covered_sessions": preds_cov,
        "is_stale": is_stale,
    }


def main() -> int:
    try:
        report = check_data_state()
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print("=== Canonical Dataset State Check (Read-Only) ===")
    print(f"Manifest verified_on: {report['verified_on']}")
    for key, val in report["actual"].items():
        expected = report["manifest"].get(key)
        status = "OK" if val == expected else f"MISMATCH (expected {expected})"
        print(f"  {key:<24}: {val:<6} [{status}]")

    if report["forecast_rows"] is not None:
        print("\n=== Forecast Predictions Staleness Check ===")
        print(f"  forecast_predictions rows            : {report['forecast_rows']}")
        print(f"  forecast_predictions covered_sessions: {report['forecast_covered_sessions']}")
        if report["is_stale"]:
            print(
                f"  [STALE] forecast_predictions.parquet ({report['forecast_covered_sessions']} covered sessions) "
                f"is stale relative to market_features.parquet ({report['actual']['covered_sessions']} covered sessions)."
            )
        else:
            print("  [OK] forecast_predictions.parquet is up to date.")

    return 1 if report["mismatches"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
