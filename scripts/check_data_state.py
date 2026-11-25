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

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "tests" / "fixtures" / "canonical_dataset_state.json"
MARKET_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "market_features.parquet"
NEWS_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "news_raw.parquet"
NEWS_SENTIMENT_PATH = PROJECT_ROOT / "data" / "processed" / "news_sentiment.parquet"
FORECAST_PREDS_PATH = PROJECT_ROOT / "data" / "processed" / "forecast_predictions.parquet"


def main() -> int:
    if not MANIFEST_PATH.exists():
        print(f"[ERROR] Manifest missing: {MANIFEST_PATH}", file=sys.stderr)
        return 1

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    mkt = pd.read_parquet(MARKET_FEATURES_PATH)
    raw = pd.read_parquet(NEWS_RAW_PATH)
    proc = pd.read_parquet(NEWS_SENTIMENT_PATH)

    actual = {
        "total_sessions": int(len(mkt)),
        "covered_sessions": int((mkt["sentiment_coverage"] == 1).sum()),
        "missing_sessions": int(mkt["sentiment_score"].isna().sum()),
        "news_raw_rows": int(len(raw)),
        "news_sentiment_rows": int(len(proc)),
        "unique_trading_dates": int(proc["trading_date"].nunique()),
    }

    print("=== Canonical Dataset State Check (Read-Only) ===")
    print(f"Manifest verified_on: {manifest.get('verified_on')}")
    mismatches = []
    for key, val in actual.items():
        expected = manifest.get(key)
        status = "OK" if val == expected else f"MISMATCH (expected {expected})"
        print(f"  {key:<24}: {val:<6} [{status}]")
        if val != expected:
            mismatches.append(key)

    if FORECAST_PREDS_PATH.exists():
        preds = pd.read_parquet(FORECAST_PREDS_PATH)
        preds_cov = int((preds["sentiment_coverage"] == 1).sum())
        mkt_mtime = MARKET_FEATURES_PATH.stat().st_mtime
        preds_mtime = FORECAST_PREDS_PATH.stat().st_mtime
        print("\n=== Forecast Predictions Staleness Check ===")
        print(f"  forecast_predictions rows            : {len(preds)}")
        print(f"  forecast_predictions covered_sessions: {preds_cov}")
        if preds_cov < actual["covered_sessions"] or preds_mtime < mkt_mtime:
            print(
                f"  [STALE] forecast_predictions.parquet ({preds_cov} covered sessions, "
                f"mtime={preds_mtime:.0f}) is stale relative to market_features.parquet "
                f"({actual['covered_sessions']} covered sessions, mtime={mkt_mtime:.0f})."
            )
        else:
            print("  [OK] forecast_predictions.parquet is up to date.")

    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
