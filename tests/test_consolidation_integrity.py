"""Offline regression tests for Phase 3 consolidation invariants.

Validates strictly on synthetic data and manifest fixtures:
1. partition_masks structural integrity and load_dev_partition Tier 3 guardrails.
2. Staleness detection between forecast predictions and market features.
3. Canonical dataset manifest schema and count invariants.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.check_data_state import check_data_state
from src.utils.partition_loader import load_dev_partition, validate_partition_masks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "tests" / "fixtures" / "canonical_dataset_state.json"


def _build_synthetic_partition_masks(n_total: int = 1455) -> pd.DataFrame:
    dates = pd.bdate_range("2021-01-04", periods=n_total)
    df = pd.DataFrame(index=dates)
    df["tier1_warmup"] = False
    df["tier2_dev"] = False
    df["embargo_boundary"] = False
    df["tier3_holdout"] = False
    df.iloc[0:400, df.columns.get_loc("tier1_warmup")] = True
    df.iloc[400:1150, df.columns.get_loc("tier2_dev")] = True
    df.iloc[1150:1155, df.columns.get_loc("embargo_boundary")] = True
    df.iloc[1155:1455, df.columns.get_loc("tier3_holdout")] = True
    return df


class TestPartitionMasksAndLoader(unittest.TestCase):
    """Test partition_masks boundaries, disjointness, embargo, and Tier 3 access guard."""

    def test_partition_boundaries_disjoint_and_embargo_length(self) -> None:
        masks = _build_synthetic_partition_masks()
        validate_partition_masks(masks)

        t1_idx = np.where(masks["tier1_warmup"].values)[0]
        t2_idx = np.where(masks["tier2_dev"].values)[0]
        emb_idx = np.where(masks["embargo_boundary"].values)[0]
        t3_idx = np.where(masks["tier3_holdout"].values)[0]

        self.assertEqual((int(t1_idx[0]), int(t1_idx[-1]), len(t1_idx)), (0, 399, 400))
        self.assertEqual((int(t2_idx[0]), int(t2_idx[-1]), len(t2_idx)), (400, 1149, 750))
        self.assertEqual((int(emb_idx[0]), int(emb_idx[-1]), len(emb_idx)), (1150, 1154, 5))
        self.assertEqual((int(t3_idx[0]), int(t3_idx[-1]), len(t3_idx)), (1155, 1454, 300))

        # Embargo must be >= maximum forward horizon (5 sessions)
        max_horizon = 5
        self.assertGreaterEqual(len(emb_idx), max_horizon)

        # Every row covered by exactly one partition
        row_sums = (
            masks["tier1_warmup"].astype(int)
            + masks["tier2_dev"].astype(int)
            + masks["embargo_boundary"].astype(int)
            + masks["tier3_holdout"].astype(int)
        )
        self.assertTrue((row_sums == 1).all())

    def test_load_dev_partition_excludes_tier3_and_blocks_without_override(self) -> None:
        masks = _build_synthetic_partition_masks()
        features = pd.DataFrame({"feat": np.arange(len(masks), dtype=float)}, index=masks.index)

        feat_dev, masks_dev = load_dev_partition(masks, features)
        self.assertEqual(len(feat_dev), 1150)
        self.assertEqual(len(masks_dev), 1150)
        self.assertFalse(masks_dev["tier3_holdout"].any())
        self.assertFalse(masks_dev["embargo_boundary"].any())

        with self.assertRaises(PermissionError):
            load_dev_partition(masks, features, include_tier3=True)

        feat_with_t3, masks_with_t3 = load_dev_partition(
            masks, features, include_tier3=True, allow_holdout=True
        )
        self.assertEqual(len(feat_with_t3), 1450)
        self.assertTrue(masks_with_t3["tier3_holdout"].any())


class TestStalenessDetection(unittest.TestCase):
    """Test check_data_state staleness warning on synthetic temp fixtures."""

    def test_flags_stale_when_predictions_have_fewer_covered_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dates = pd.bdate_range("2026-08-01", periods=10)

            manifest = {
                "total_sessions": 10,
                "covered_sessions": 4,
                "missing_sessions": 6,
                "news_raw_rows": 5,
                "news_sentiment_rows": 5,
                "unique_trading_dates": 4,
                "verified_on": "2026-11-25",
            }
            manifest_p = tmp / "manifest.json"
            manifest_p.write_text(json.dumps(manifest), encoding="utf-8")

            mkt = pd.DataFrame(
                {
                    "sentiment_coverage": [0] * 6 + [1] * 4,
                    "sentiment_score": [np.nan] * 6 + [0.1, 0.2, 0.3, 0.4],
                },
                index=dates,
            )
            mkt_p = tmp / "market_features.parquet"
            mkt.to_parquet(mkt_p)

            news_dates = dates[6:10]
            news_df = pd.DataFrame(
                {"headline": [f"h{i}" for i in range(5)], "trading_date": list(news_dates) + [news_dates[-1]]}
            )
            raw_p = tmp / "news_raw.parquet"
            proc_p = tmp / "news_sentiment.parquet"
            news_df.to_parquet(raw_p)
            news_df.to_parquet(proc_p)

            # Stale predictions: only 2 covered sessions instead of 4
            preds = pd.DataFrame(
                {"sentiment_coverage": [0] * 8 + [1] * 2},
                index=dates,
            )
            preds_p = tmp / "forecast_predictions.parquet"
            preds.to_parquet(preds_p)
            # Force newer mtime on preds to prove covered_sessions check triggers STALE alone
            now = mkt_p.stat().st_mtime + 10
            os.utime(preds_p, (now, now))

            report = check_data_state(
                manifest_path=manifest_p,
                market_features_path=mkt_p,
                news_raw_path=raw_p,
                news_sentiment_path=proc_p,
                forecast_preds_path=preds_p,
            )
            self.assertEqual(report["mismatches"], [])
            self.assertTrue(report["is_stale"])


class TestCanonicalManifestSchema(unittest.TestCase):
    """Test canonical_dataset_state.json schema and mathematical invariants."""

    def test_manifest_counts_are_consistent(self) -> None:
        self.assertTrue(MANIFEST_PATH.exists(), f"Missing manifest at {MANIFEST_PATH}")
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

        required_keys = {
            "total_sessions",
            "covered_sessions",
            "missing_sessions",
            "news_raw_rows",
            "news_sentiment_rows",
            "unique_trading_dates",
            "baseline_covered_sessions",
            "baseline_missing_sessions",
            "verified_on",
        }
        self.assertTrue(required_keys.issubset(data.keys()))
        self.assertEqual(data["covered_sessions"] + data["missing_sessions"], data["total_sessions"])
        self.assertEqual(
            data["baseline_covered_sessions"] + data["baseline_missing_sessions"],
            data["total_sessions"],
        )
        self.assertEqual(data["news_raw_rows"], data["news_sentiment_rows"])
