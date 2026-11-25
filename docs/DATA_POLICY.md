# Quant RAG Copilot — Canonical Dataset Policy & Provenance

**Policy Effective Date:** 2026-11-25  
**Verified Against Active Data:** 2026-11-25  
**Status:** Active Governance Policy

---

## 1. Canonical Active Dataset State (40-Session State)

The active production dataset under `data/raw/` and `data/processed/` is defined by the following verified invariants (measured 2026-11-25):

- **Market Features (`data/processed/market_features.parquet`):**
  - Total trading sessions: **451** (`2024-12-31` to `2026-10-19`)
  - Sentiment-covered sessions (`sentiment_coverage == 1`): **40** (`2026-08-12` to `2026-08-19` [6 sessions] and `2026-09-01` to `2026-10-19` [34 sessions])
  - Missing sentiment sessions (`sentiment_coverage == 0`, strictly `NaN` in all sentiment columns): **411**
- **News Archives (`data/raw/news_raw.parquet` and `data/processed/news_sentiment.parquet`):**
  - Total deduplicated news rows: **2,556** (2,417 baseline rows + 139 verified Finnhub articles from `2026-08-12` to `2026-08-19`)
  - Unique `trading_date` values in `news_sentiment.parquet`: **42**

All count invariants are stored in `tests/fixtures/canonical_dataset_state.json` and enforced by automated regression checks.

---

## 2. Historical 34-Session Baseline (Frozen)

The earlier 34-session state is preserved permanently as a historical baseline:

- **Location:** `data/backups/pre_august_trial/`
  - `market_features.parquet`: 451 rows, **34** covered sessions (`2026-09-01` to `2026-10-19`), **417** missing sessions (SHA-256: `1234DE37317FCDA199398E4091A8338877F4DF92D9F9334690BD2426FB7714BE`)
  - `news_raw.parquet`: 2,417 rows (SHA-256: `179FCB16B39ACE4CE646B4249D2B2804D9E9656AC3BEA7110163B77631F5BDB7`)
  - `news_sentiment.parquet`: 2,417 rows, 35 unique `trading_date` values (SHA-256: `7F2CC0146BFB90371C293888EDC02D8528C2175E86EEBA23B2FF3223380AD0F8`)
  - Full inventory and checksums are recorded in `data/backups/pre_august_trial/baseline_inventory.json` and `checksums.sha256`.
- Five intermediate single-day trial snapshots (`pre_august13_trial` through `pre_august19_trial`) are likewise frozen under `data/backups/`.
- **Immutability Rule:** Historical baseline snapshots and `reports/baseline_validation_metrics.json` (SHA-256: `22239122762A8DEE521EDEC62005D71183850F017989BBE081C348E319F2156D`) are permanently frozen and must never be modified or overwritten.

---

## 3. Disambiguation of Commonly Conflated Counts

| Metric | Value (Active) | Exact Definition |
| :--- | :---: | :--- |
| **Unique news dates** | **42** | Distinct `trading_date` values in `news_sentiment.parquet`. Includes `2026-08-20` (1 post-close Aug 19 article not refreshed in `market_features.parquet`) and `2026-10-20` (beyond the `2026-10-19` market dataset boundary). |
| **Covered sessions** | **40** | Rows in `market_features.parquet` where `sentiment_coverage == 1` and sentiment metrics are non-null. |
| **Realized covered sessions** | **39** | Covered sessions in `market_features.parquet` with a realized 1-day forward return target ($t < T$, excluding final session `2026-10-19`). |
| **Evaluated-in-frozen-predictions** | **33** | Covered sessions with realized forward returns recorded in the frozen `data/processed/forecast_predictions.parquet` (`2026-09-01` to `2026-10-16`). |

---

## 4. Staleness Statement & Report Versioning Rule

> [!WARNING]
> **`data/processed/forecast_predictions.parquet` and `reports/baseline_validation_metrics.json` are STALE relative to the active 40-session `market_features.parquet`.**  
> Both artifacts were generated on `2026-10-23` under the 34-session baseline (`33` evaluated covered sessions). They must **not** be presented or interpreted as evaluation results for the 40-session canonical dataset.

**Mandatory Versioning Rule for Future Reports:**
- Any future forecasting run or evaluation against the 40-session canonical dataset **MUST** write to a new, separately named, versioned report file (e.g., `reports/validation_metrics_40session_v1.json`) and a separate prediction file if regenerated.
- Every new report **MUST** record full data provenance in its metadata: SHA-256 hashes of all input parquet files, exact row and coverage counts, and UTC generation timestamp.
- `reports/baseline_validation_metrics.json` must **never** be overwritten.

---

## 5. Procedure for Updating the Canonical State

After any future authorized news or market ingestion run:
1. Verify the new dataset counts (total sessions, covered sessions, missing sessions, raw/processed news rows, unique trading dates) using `scripts/check_data_state.py`.
2. Update `tests/fixtures/canonical_dataset_state.json` with the new exact counts and `verified_on` date.
3. Confirm that `run_test_mode` Test 7 in `src/models/baseline_forecasting.py` passes against the updated manifest without relaxing assertions.
4. Update `AGENTS.md` to document the new canonical counts alongside any preserved historical baseline notes.
5. Commit the updated manifest, tests, and documentation together.

---

## 6. Holdout & Leakage Note (5-Year Validation Protocol)

All sentiment-covered sessions in the canonical dataset (`2026-08-12` to `2026-10-19`) fall strictly inside **Tier 3 (Sealed Holdout)** of the 5-year multi-asset validation protocol (`2025-08-11` to `2026-10-19`, indices `1155–1454` in `data/staging/features_5y/partition_masks.parquet`).

Accordingly:
- Sentiment-based models **must not** be tuned, selected, or hyper-parameterized against this August–October 2026 window.
- Any evaluation performed on this window is strictly a **one-shot diagnostic** rather than a development benchmark.
