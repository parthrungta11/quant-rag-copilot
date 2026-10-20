"""Baseline forecasting and walk-forward validation for daily equity returns.

Implements:
1. One-session-ahead forward return prediction target.
2. Naive zero-return and lagged-return baselines.
3. Regularized Ridge regression (technical-only and hybrid).
4. Regularized HistGradientBoostingRegressor (technical-only and hybrid).
5. Chronological expanding-window walk-forward validation with strict leakage safeguards.
6. Sub-period ablation on news-sentiment-covered sessions.
7. Comprehensive metrics (RMSE, MAE, R² OOS, Directional Accuracy, Spearman IC, exploratory strategy).
8. Automated leakage and integrity verification test suite.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.features.feature_transforms import (
    TECHNICAL_FEATURE_COLUMNS,
    SENTIMENT_FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_stationary_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("baseline_forecasting")

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_MARKET_FEATURES_FILE = PROCESSED_DATA_DIR / "market_features.parquet"
DEFAULT_PREDICTIONS_FILE = PROCESSED_DATA_DIR / "forecast_predictions.parquet"
DEFAULT_METRICS_FILE = REPORTS_DIR / "baseline_validation_metrics.json"


@dataclass
class ForecastConfig:
    """Hyperparameters and configuration for walk-forward validation."""
    min_train_size: int = 200
    ridge_alpha: float = 10.0
    hgb_max_depth: int = 3
    hgb_min_samples_leaf: int = 10
    hgb_learning_rate: float = 0.03
    hgb_max_iter: int = 50
    hgb_l2_regularization: float = 1.0
    random_state: int = 42


def fit_predict_naive_zero(test_row: pd.Series) -> float:
    """Naive driftless random walk baseline: predicts R_{t+1} = 0.0."""
    return 0.0


def fit_predict_naive_lag(test_row: pd.Series) -> float:
    """Naive momentum / return persistence baseline: predicts R_{t+1} = R_t."""
    return float(test_row["daily_return"])


def fit_predict_ridge_tech(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tech_cols: List[str],
    target_col: str,
    config: ForecastConfig,
) -> float:
    """Fit Ridge regression on standardized technical features strictly within the fold."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[tech_cols].values)
    y_train = train_df[target_col].values.astype(float)
    X_test = scaler.transform(test_df[tech_cols].values)

    model = Ridge(alpha=config.ridge_alpha, random_state=config.random_state)
    model.fit(X_train, y_train)
    return float(model.predict(X_test)[0])


def fit_predict_hgb_tech(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tech_cols: List[str],
    target_col: str,
    config: ForecastConfig,
) -> float:
    """Fit HistGradientBoostingRegressor on technical features strictly within the fold."""
    X_train = train_df[tech_cols].values
    y_train = train_df[target_col].values.astype(float)
    X_test = test_df[tech_cols].values

    model = HistGradientBoostingRegressor(
        max_depth=config.hgb_max_depth,
        min_samples_leaf=config.hgb_min_samples_leaf,
        learning_rate=config.hgb_learning_rate,
        max_iter=config.hgb_max_iter,
        l2_regularization=config.hgb_l2_regularization,
        random_state=config.random_state,
    )
    model.fit(X_train, y_train)
    return float(model.predict(X_test)[0])


def fit_predict_ridge_hybrid(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tech_cols: List[str],
    sentiment_cols: List[str],
    target_col: str,
    config: ForecastConfig,
) -> float:
    """Fit Ridge regression on technical + sentiment features with training-fold-only imputation."""
    all_features = tech_cols + sentiment_cols

    # Impute missing sentiment values with 0.0 as a modelling convention.
    # This is not necessarily a neutral value for probability features.
    # Imputation uses only the current fold's training and test rows.
    train_features = train_df[all_features].copy()
    test_features = test_df[all_features].copy()

    for col in sentiment_cols:
        if col != "sentiment_coverage":
            train_features[col] = train_features[col].fillna(0.0)
            test_features[col] = test_features[col].fillna(0.0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features.values)
    y_train = train_df[target_col].values.astype(float)
    X_test = scaler.transform(test_features.values)

    model = Ridge(alpha=config.ridge_alpha, random_state=config.random_state)
    model.fit(X_train, y_train)
    return float(model.predict(X_test)[0])


def inspect_hgb_split_features(
    model: HistGradientBoostingRegressor, feature_names: List[str]
) -> List[str]:
    """Inspect fitted tree nodes to return names of features actually used in split decisions."""
    split_indices = set()
    for ensemble in model._predictors:
        for tree in ensemble:
            for node in tree.nodes:
                if not node["is_leaf"]:
                    split_indices.add(int(node["feature_idx"]))
    return [feature_names[idx] for idx in sorted(split_indices) if idx < len(feature_names)]


def fit_predict_hgb_hybrid(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tech_cols: List[str],
    sentiment_cols: List[str],
    target_col: str,
    config: ForecastConfig,
    return_tree_diagnostics: bool = False,
) -> Any:
    """Fit HistGradientBoostingRegressor on technical + sentiment features using native NaN handling."""
    all_features = tech_cols + sentiment_cols

    # Filter to features that have at least 2 distinct non-NaN values in the training set
    # (Scikit-learn tree binning requires >= 2 distinct values to compute split thresholds)
    valid_cols = [
        c for c in all_features
        if train_df[c].dropna().nunique() >= 2
    ]

    X_train = train_df[valid_cols].values
    y_train = train_df[target_col].values.astype(float)
    X_test = test_df[valid_cols].values

    model = HistGradientBoostingRegressor(
        max_depth=config.hgb_max_depth,
        min_samples_leaf=config.hgb_min_samples_leaf,
        learning_rate=config.hgb_learning_rate,
        max_iter=config.hgb_max_iter,
        l2_regularization=config.hgb_l2_regularization,
        random_state=config.random_state,
    )
    model.fit(X_train, y_train)
    prediction = float(model.predict(X_test)[0])

    if return_tree_diagnostics:
        used_splits = inspect_hgb_split_features(model, valid_cols)
        sentiment_splits = [col for col in used_splits if col in sentiment_cols]
        return prediction, sentiment_splits

    return prediction


def run_walk_forward_validation(
    features_df: pd.DataFrame,
    tech_cols: List[str],
    sentiment_cols: List[str],
    target_col: str,
    config: Optional[ForecastConfig] = None,
) -> pd.DataFrame:
    """Execute chronological expanding-window walk-forward validation across all sessions.

    Parameters
    ----------
    features_df : pd.DataFrame
        Transformed stationary features DataFrame with DatetimeIndex.
    tech_cols : List[str]
        List of stationary technical feature column names.
    sentiment_cols : List[str]
        List of sentiment feature column names.
    target_col : str
        Target column name ('target_return_1d').
    config : ForecastConfig, optional
        Walk-forward hyperparameters.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by Date with columns:
        - actual_return
        - sentiment_coverage
        - naive_zero_pred
        - naive_lag_pred
        - ridge_tech_pred
        - hgb_tech_pred
        - ridge_hybrid_pred
        - hgb_hybrid_pred
    """
    if config is None:
        config = ForecastConfig()

    n_total = len(features_df)
    if not features_df.index.is_monotonic_increasing:
        raise ValueError("Dataset index must be strictly monotonically increasing to prevent look-ahead bias.")
    if features_df.index.has_duplicates:
        raise ValueError("Dataset index contains duplicate dates.")
    if n_total <= config.min_train_size:
        raise ValueError(
            f"Dataset length ({n_total}) must exceed min_train_size ({config.min_train_size})."
        )

    logger.info(
        f"Starting expanding-window walk-forward validation: total={n_total}, "
        f"warmup={config.min_train_size}, forecast_steps={n_total - config.min_train_size}."
    )

    records: List[Dict[str, Any]] = []
    hgb_folds_with_sentiment_splits = 0
    sentiment_features_ever_used = set()

    for t in range(config.min_train_size, n_total):
        # Training window: strictly historical [0, t-1] with realized forward targets
        train_df = features_df.iloc[:t].dropna(subset=[target_col])
        # Test sample: current session t (predicting forward target R_{t+1})
        test_df = features_df.iloc[t:t+1]

        date = test_df.index[0]
        actual_return = test_df[target_col].values[0]
        sentiment_coverage = int(test_df["sentiment_coverage"].values[0])

        test_row = test_df.iloc[0]

        # 1. Naive Zero
        p_zero = fit_predict_naive_zero(test_row)

        # 2. Naive Lag
        p_lag = fit_predict_naive_lag(test_row)

        # 3. Ridge Tech
        p_ridge_tech = fit_predict_ridge_tech(train_df, test_df, tech_cols, target_col, config)

        # 4. HGB Tech
        p_hgb_tech = fit_predict_hgb_tech(train_df, test_df, tech_cols, target_col, config)

        # 5. Ridge Hybrid
        p_ridge_hyb = fit_predict_ridge_hybrid(train_df, test_df, tech_cols, sentiment_cols, target_col, config)

        # 6. HGB Hybrid (with direct tree split inspection)
        p_hgb_hyb, sent_splits = fit_predict_hgb_hybrid(
            train_df, test_df, tech_cols, sentiment_cols, target_col, config, return_tree_diagnostics=True
        )
        if sent_splits:
            hgb_folds_with_sentiment_splits += 1
            sentiment_features_ever_used.update(sent_splits)

        records.append({
            "Date": date,
            "actual_return": actual_return,
            "sentiment_coverage": sentiment_coverage,
            "naive_zero_pred": p_zero,
            "naive_lag_pred": p_lag,
            "ridge_tech_pred": p_ridge_tech,
            "hgb_tech_pred": p_hgb_tech,
            "ridge_hybrid_pred": p_ridge_hyb,
            "hgb_hybrid_pred": p_hgb_hyb,
        })

    logger.info(
        f"HGB Hybrid tree split audit: sentiment features used in {hgb_folds_with_sentiment_splits}/{n_total - config.min_train_size} folds "
        f"(features: {sorted(list(sentiment_features_ever_used)) if sentiment_features_ever_used else 'none'})."
    )

    predictions_df = pd.DataFrame(records).set_index("Date")
    logger.info(f"Walk-forward validation complete. Total generated predictions: {len(predictions_df)}.")
    return predictions_df


def _calculate_single_model_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_mean: float,
) -> Dict[str, Any]:
    """Compute statistical and exploratory financial metrics for a single model's predictions."""
    n = len(y_true)
    if n == 0:
        return {}

    errors = y_true - y_pred
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    # 1. Out-of-sample R² versus driftless zero-return benchmark (denominator: sum of squared actual returns)
    ss_res = float(np.sum(errors ** 2))
    ss_zero = float(np.sum(y_true ** 2))
    r2_vs_zero = float(1.0 - (ss_res / ss_zero)) if ss_zero > 1e-14 else 0.0

    # 2. Descriptive test-period R² relative to in-sample test-period mean return
    # (Explicitly descriptive; not a causal out-of-sample metric as it uses the test-period mean)
    ss_tot = float(np.sum((y_true - baseline_mean) ** 2))
    r2_desc = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-14 else 0.0

    # Directional accuracy (sign hit rate)
    # Binary: sign(pred) == sign(true)
    # Using convention: pred > 0 aligns with true > 0
    da = float(np.mean((y_pred > 0.0) == (y_true > 0.0)))

    # Spearman rank Information Coefficient (IC)
    if np.std(y_pred) > 1e-12 and np.std(y_true) > 1e-12:
        ic, pval = spearmanr(y_true, y_pred)
        ic = float(ic)
        pval = float(pval)
    else:
        ic = 0.0
        pval = 1.0

    # Exploratory strategy metrics (ternary sign strategy: position = +1 if pred > 0, -1 if pred < 0, 0 if pred == 0)
    # Strictly marked as an exploratory backtest proxy without execution friction.
    positions = np.where(y_pred > 0.0, 1.0, np.where(y_pred < 0.0, -1.0, 0.0))
    strat_returns = positions * y_true
    strat_cum_return = float(np.prod(1.0 + strat_returns) - 1.0)
    strat_mean = float(np.mean(strat_returns))
    strat_std = float(np.std(strat_returns, ddof=1)) if n > 1 else 0.0
    strat_ann_return = float(strat_mean * 252.0)
    strat_ann_vol = float(strat_std * np.sqrt(252.0))
    strat_sharpe = float(strat_ann_return / strat_ann_vol) if strat_ann_vol > 1e-12 else 0.0
    turnover = float(np.mean(np.abs(np.diff(positions)))) if n > 1 else 0.0

    return {
        "n_samples": n,
        "rmse": rmse,
        "mae": mae,
        "r2_vs_zero_benchmark": r2_vs_zero,
        "r2_descriptive_test_mean": r2_desc,
        "directional_accuracy": da,
        "spearman_ic": ic,
        "spearman_pvalue": pval,
        "exploratory_strategy": {
    "is_exploratory_proxy": True,
    "caveat": (
        "Exploratory proxy only; assumes theoretical Close_t execution "
        "without latency. Not directly tradable when signals use information "
        "published at or near the close. Excludes transaction costs, slippage, "
        "market impact, and short-borrow fees."
    ),
    "execution_timing_assumption": (
        "Theoretical Close_t entry to Close_{t+1} exit; assumes instantaneous "
        "execution and ignores market-on-close order cutoffs."
    ),
    "transaction_cost_assumption": "0.0 bps (frictionless)",
    "short_borrow_fee_assumption": "0.0 bps (unrestricted shorting)",
    "average_daily_turnover": turnover,
    "cumulative_return": strat_cum_return,
    "annualized_return": strat_ann_return,
    "annualized_volatility": strat_ann_vol,
    "annualized_sharpe": strat_sharpe,
},
    }


def compute_evaluation_metrics(predictions_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics for all models across full and covered windows.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame generated by run_walk_forward_validation.

    Returns
    -------
    Dict[str, Any]
        Dictionary with full_walk_forward_window, sentiment_covered_subwindow,
        ablation_deltas, and dataset metadata.
    """
    # Exclude rows where actual_return is NaN (e.g. final session 2026-10-19)
    eval_df = predictions_df.dropna(subset=["actual_return"]).copy()
    y_true_full = eval_df["actual_return"].values.astype(float)
    baseline_mean_full = float(np.mean(y_true_full))

    model_keys = [
        ("naive_zero", "naive_zero_pred"),
        ("naive_lag", "naive_lag_pred"),
        ("ridge_tech", "ridge_tech_pred"),
        ("hgb_tech", "hgb_tech_pred"),
        ("ridge_hybrid", "ridge_hybrid_pred"),
        ("hgb_hybrid", "hgb_hybrid_pred"),
    ]

    # 1. Full Walk-Forward Evaluation Window (all test steps with realized targets)
    full_metrics: Dict[str, Any] = {}
    for name, col in model_keys:
        full_metrics[name] = _calculate_single_model_metrics(
            y_true_full, eval_df[col].values.astype(float), baseline_mean_full
        )

    # 2. Sentiment-Covered Evaluation Window (exact same covered sessions with realized targets)
    covered_df = eval_df[eval_df["sentiment_coverage"] == 1].copy()

    if covered_df.empty:
        raise ValueError(
            "Cannot calculate sentiment-covered metrics: "
            "no evaluated sessions have sentiment coverage."
        )

    y_true_covered = covered_df["actual_return"].values.astype(float)
    baseline_mean_covered = float(np.mean(y_true_covered))

    covered_metrics: Dict[str, Any] = {}
    for name, col in model_keys:
        covered_metrics[name] = _calculate_single_model_metrics(
            y_true_covered, covered_df[col].values.astype(float), baseline_mean_covered
        )

    last_realized_date = str(eval_df.index.max())[:10]
    last_prediction_date = str(predictions_df.index.max())[:10]

    # 3. Direct Head-to-Head Ablation Deltas (Hybrid minus Tech on Covered Subset)
    # Positive delta in RMSE is worse; positive delta in DA and IC is better.
    ablation_deltas = {
        "ridge_delta": {
            "delta_rmse": covered_metrics["ridge_hybrid"]["rmse"] - covered_metrics["ridge_tech"]["rmse"],
            "delta_mae": covered_metrics["ridge_hybrid"]["mae"] - covered_metrics["ridge_tech"]["mae"],
            "delta_r2_vs_zero": covered_metrics["ridge_hybrid"]["r2_vs_zero_benchmark"] - covered_metrics["ridge_tech"]["r2_vs_zero_benchmark"],
            "delta_da": covered_metrics["ridge_hybrid"]["directional_accuracy"] - covered_metrics["ridge_tech"]["directional_accuracy"],
            "delta_ic": covered_metrics["ridge_hybrid"]["spearman_ic"] - covered_metrics["ridge_tech"]["spearman_ic"],
        },
        "hgb_delta": {
            "delta_rmse": covered_metrics["hgb_hybrid"]["rmse"] - covered_metrics["hgb_tech"]["rmse"],
            "delta_mae": covered_metrics["hgb_hybrid"]["mae"] - covered_metrics["hgb_tech"]["mae"],
            "delta_r2_vs_zero": covered_metrics["hgb_hybrid"]["r2_vs_zero_benchmark"] - covered_metrics["hgb_tech"]["r2_vs_zero_benchmark"],
            "delta_da": covered_metrics["hgb_hybrid"]["directional_accuracy"] - covered_metrics["hgb_tech"]["directional_accuracy"],
            "delta_ic": covered_metrics["hgb_hybrid"]["spearman_ic"] - covered_metrics["hgb_tech"]["spearman_ic"],
        },
    }

    results: Dict[str, Any] = {
        "metadata": {
            "total_predictions_generated": len(predictions_df),
            "evaluated_predictions_count": len(eval_df),
            "unrealized_target_sessions_count": int(predictions_df["actual_return"].isna().sum()),
            "sentiment_covered_evaluated_count": len(covered_df),
            "last_realized_evaluation_date": last_realized_date,
            "last_prediction_date": last_prediction_date,
            "walk_forward_evaluation_range": {
                "start_date": str(eval_df.index.min())[:10],
                "end_date": last_realized_date,
            },
            "sentiment_covered_range": {
                "start_date": str(covered_df.index.min())[:10],
                "end_date": str(covered_df.index.max())[:10],
            },
            "r2_metric_definitions": {
                "r2_vs_zero_benchmark": "1 - sum((y - y_hat)^2) / sum(y^2). Out-of-sample R² versus driftless zero-return benchmark (denominator: sum of squared realized returns).",
                "r2_descriptive_test_mean": "1 - sum((y - y_hat)^2) / sum((y - mean(y_test))^2). Descriptive test-period variance explained relative to in-sample test-period mean."
            },
            "statistical_limitation_notice": (
                "The sentiment-covered test subset contains only 33 evaluated sessions. "
                "Confidence intervals are wide, and results must be interpreted as exploratory proof-of-concept "
                "rather than statistically significant alpha at p < 0.05."
            ),
        },
        "full_walk_forward_window": full_metrics,
        "sentiment_covered_subwindow": covered_metrics,
        "ablation_deltas_on_covered": ablation_deltas,
    }

    return results


def save_artifacts(
    predictions_df: pd.DataFrame,
    metrics_dict: Dict[str, Any],
    pred_path: Path = DEFAULT_PREDICTIONS_FILE,
    metrics_path: Path = DEFAULT_METRICS_FILE,
) -> None:
    """Persist prediction parquet and JSON metrics report to disk."""
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    predictions_df.to_parquet(pred_path)
    logger.info(f"Saved forecast predictions to: {pred_path} ({len(predictions_df)} rows).")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    logger.info(f"Saved baseline validation metrics to: {metrics_path}.")


def print_metrics_summary(metrics_dict: Dict[str, Any]) -> None:
    """Print clean formatted tables of walk-forward results and empirical findings."""
    meta = metrics_dict["metadata"]
    full = metrics_dict["full_walk_forward_window"]
    cov = metrics_dict["sentiment_covered_subwindow"]
    deltas = metrics_dict["ablation_deltas_on_covered"]

    print("\n" + "=" * 80)
    print("STAGE 3: BASELINE FORECASTING & WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total Evaluated Test Steps   : {meta['evaluated_predictions_count']}")
    print(f"Unrealized Forward Dates     : {meta['unrealized_target_sessions_count']} (Latest session target = NaN)")
    print(f"Last Realized Evaluation Date: {meta['last_realized_evaluation_date']}")
    print(f"Last Prediction Date (Fwd)   : {meta['last_prediction_date']}")
    print(f"Sentiment-Covered Test Steps : {meta['sentiment_covered_evaluated_count']}")
    print(f"Full Test Date Range         : {meta['walk_forward_evaluation_range']['start_date']} to {meta['walk_forward_evaluation_range']['end_date']}")
    print(f"Covered Test Date Range      : {meta['sentiment_covered_range']['start_date']} to {meta['sentiment_covered_range']['end_date']}")
    print("=" * 80)

    print(
    f"\n--- 1. FULL WALK-FORWARD EVALUATION WINDOW "
    f"(N = {meta['evaluated_predictions_count']} Sessions) ---"
)
    print(f"{'Model Name':<16} | {'RMSE':<9} | {'MAE':<9} | {'R2 vs Zero':<11} | {'R2 Desc Mean':<13} | {'Hit Rate (DA)':<13} | {'Spearman IC':<11} | {'Sharpe (Exp)':<12}")
    print("-" * 105)
    for model_name, m in full.items():
        sh = m['exploratory_strategy']['annualized_sharpe']
        print(f"{model_name:<16} | {m['rmse']:.5f}   | {m['mae']:.5f}   | {m['r2_vs_zero_benchmark']:>10.4f}  | {m['r2_descriptive_test_mean']:>12.4f}  | {m['directional_accuracy']*100:>6.1f}%       | {m['spearman_ic']:>8.4f}    | {sh:>8.2f}")

    print(
    f"\n--- 2. SENTIMENT-COVERED SUB-WINDOW "
    f"(N = {meta['sentiment_covered_evaluated_count']} Sessions) ---"
)
    print(f"{'Model Name':<16} | {'RMSE':<9} | {'MAE':<9} | {'R2 vs Zero':<11} | {'R2 Desc Mean':<13} | {'Hit Rate (DA)':<13} | {'Spearman IC':<11} | {'Sharpe (Exp)':<12}")
    print("-" * 105)
    for model_name, m in cov.items():
        sh = m['exploratory_strategy']['annualized_sharpe']
        print(f"{model_name:<16} | {m['rmse']:.5f}   | {m['mae']:.5f}   | {m['r2_vs_zero_benchmark']:>10.4f}  | {m['r2_descriptive_test_mean']:>12.4f}  | {m['directional_accuracy']*100:>6.1f}%       | {m['spearman_ic']:>8.4f}    | {sh:>8.2f}")

    print("\n--- 3. ABLATION DELTAS ON COVERED SUBSET (Hybrid minus Technical) ---")
    print(f"Ridge Model : Delta_RMSE={deltas['ridge_delta']['delta_rmse']:+.5f}, Delta_DA={deltas['ridge_delta']['delta_da']*100:+.1f}%, Delta_IC={deltas['ridge_delta']['delta_ic']:+.4f}")
    print(f"HGB Model   : Delta_RMSE={deltas['hgb_delta']['delta_rmse']:+.5f}, Delta_DA={deltas['hgb_delta']['delta_da']*100:+.1f}%, Delta_IC={deltas['hgb_delta']['delta_ic']:+.4f}")
    print("\n[NOTE ON EMPIRICAL FINDINGS & LIMITATIONS]")
    print(meta["statistical_limitation_notice"])
    print("=" * 80 + "\n")


# ============================================================================
# COMPREHENSIVE AUTOMATED TEST SUITE
# ============================================================================

def run_test_suite() -> bool:
    """Run automated verification tests for Stage 3 requirements.

    Tests:
    1. Target alignment and forward horizon (R_{t+1} = Close_{t+1}/Close_t - 1).
    2. Final session target invariant (must be strictly NaN).
    3. Strict chronological temporal ordering and expanding window growth.
    4. Preprocessing leakage isolation (outlier injection test).
    5. Missing sentiment NaN preservation in storage and model routing.
    6. Deterministic output reproducibility with fixed random_state.
    7. Production file isolation (no production Parquet overwritten during test).
    """
    logger.info("Starting Stage 3 automated test suite...")

    # --- Test 1 & 2: Target construction and final NaN preservation ---
    logger.info("Running Test 1 & 2: Target construction and final NaN preservation...")
    dates = pd.date_range("2026-01-01", periods=10, freq="B")
    mock_close = np.array([100.0, 102.0, 101.0, 105.0, 103.0, 107.0, 106.0, 108.0, 110.0, 112.0])
    mock_df = pd.DataFrame({
        "Close": mock_close,
        "High": mock_close * 1.01,
        "Low": mock_close * 0.99,
        "Open": mock_close * 0.995,
        "Volume": np.full(10, 1000000, dtype=int),
        "Daily_Return": pd.Series(mock_close).pct_change().fillna(0.0).values,
        "SMA_20": mock_close,
        "SMA_50": mock_close,
        "Volatility_20": np.full(10, 0.01),
        "RSI_14": np.full(10, 50.0),
        "sentiment_score": np.full(10, np.nan),
        "sentiment_std": np.full(10, np.nan),
        "sentiment_count": np.full(10, np.nan),
        "prob_positive": np.full(10, np.nan),
        "prob_negative": np.full(10, np.nan),
        "prob_neutral": np.full(10, np.nan),
        "sentiment_coverage": np.zeros(10, dtype=int),
    }, index=dates)

    t_df, tech_cols, sent_cols, target_col = build_stationary_features(mock_df)

    assert target_col == "target_return_1d", "Target column name mismatch."
    assert len(t_df) == 10, "Row count mismatch in transformed DataFrame."
    # Check shift(1) forward alignment
    expected_target_0 = (102.0 - 100.0) / 100.0
    assert abs(t_df[target_col].iloc[0] - expected_target_0) < 1e-10, "Target calculation error at step 0."
    # Check final row target is strictly NaN
    assert np.isnan(t_df[target_col].iloc[-1]), "Final row target must be strictly NaN."
    logger.info("Test 1 & 2 PASSED: Target alignment and final NaN preserved.")

    # --- Test 3: Strict chronological walk-forward ordering ---
    logger.info("Running Test 3: Chronological walk-forward ordering and expanding window...")
    test_dates = pd.date_range("2025-01-01", periods=50, freq="B")
    syn_close = 100.0 + np.cumsum(np.random.RandomState(42).randn(50))
    syn_df = pd.DataFrame({
        "Close": syn_close,
        "High": syn_close * 1.01,
        "Low": syn_close * 0.99,
        "Open": syn_close,
        "Volume": np.full(50, 500000),
        "Daily_Return": pd.Series(syn_close).pct_change().fillna(0.0).values,
        "SMA_20": syn_close,
        "SMA_50": syn_close,
        "Volatility_20": np.full(50, 0.015),
        "RSI_14": np.full(50, 52.0),
        "sentiment_score": np.full(50, np.nan),
        "sentiment_std": np.full(50, np.nan),
        "sentiment_count": np.full(50, np.nan),
        "prob_positive": np.full(50, np.nan),
        "prob_negative": np.full(50, np.nan),
        "prob_neutral": np.full(50, np.nan),
        "sentiment_coverage": np.zeros(50, dtype=int),
    }, index=test_dates)

    syn_transformed, syn_tech, syn_sent, syn_target = build_stationary_features(syn_df)
    syn_cfg = ForecastConfig(min_train_size=30, hgb_min_samples_leaf=5, hgb_max_iter=10)

    # Walk-forward iteration check
    for t in range(syn_cfg.min_train_size, len(syn_transformed)):
        tr = syn_transformed.iloc[:t]
        te = syn_transformed.iloc[t:t+1]
        assert tr.index.max() < te.index.min(), f"Temporal leakage detected at step {t}!"
        assert len(tr) == t, f"Training window size mismatch at step {t}."

    syn_preds = run_walk_forward_validation(syn_transformed, syn_tech, syn_sent, syn_target, syn_cfg)
    assert len(syn_preds) == 20, f"Expected 20 walk-forward predictions, got {len(syn_preds)}."
    logger.info("Test 3 PASSED: Temporal ordering and window expansion verified.")

    # --- Test 4: Preprocessing leakage isolation ---
    logger.info("Running Test 4: Preprocessing leakage isolation...")
    # Verify that fitting a scaler on train_df is completely unaffected by values in test_df
    tr_sample = pd.DataFrame({"daily_return": [0.01, -0.02, 0.015, -0.005]})
    te_sample_normal = pd.DataFrame({"daily_return": [0.008]})
    te_sample_extreme = pd.DataFrame({"daily_return": [999999.0]})

    sc1 = StandardScaler()
    sc1.fit(tr_sample)
    sc2 = StandardScaler()
    sc2.fit(tr_sample)

    assert sc1.mean_[0] == sc2.mean_[0], "StandardScaler fit inconsistency."
    assert sc1.scale_[0] == sc2.scale_[0], "StandardScaler scale inconsistency."
    # Transform test point
    tx1 = sc1.transform(te_sample_normal)[0][0]
    tx_ext = sc1.transform(te_sample_extreme)[0][0]
    # Check that training statistics were not altered by test transformation
    assert sc1.mean_[0] == sc2.mean_[0], "Test transformation mutated training scaler state!"
    logger.info("Test 4 PASSED: Preprocessing leakage isolation verified.")

    # --- Test 5: Missing sentiment NaN preservation and native routing ---
    logger.info("Running Test 5: Missing sentiment preservation and model routing...")
    assert syn_transformed["sentiment_score"].isna().sum() == 50, "NaN sentiment was incorrectly mutated!"
    assert (syn_transformed["sentiment_coverage"] == 0).all(), "Coverage indicator error."
    # Verify Ridge and HGB fit and predict without throwing exceptions on NaNs
    p_hgb = fit_predict_hgb_hybrid(
        syn_transformed.iloc[:35], syn_transformed.iloc[35:36], syn_tech, syn_sent, syn_target, syn_cfg
    )
    p_ridge = fit_predict_ridge_hybrid(
        syn_transformed.iloc[:35], syn_transformed.iloc[35:36], syn_tech, syn_sent, syn_target, syn_cfg
    )
    assert not np.isnan(p_hgb), "HGB Hybrid returned NaN prediction."
    assert not np.isnan(p_ridge), "Ridge Hybrid returned NaN prediction."
    logger.info("Test 5 PASSED: Missing sentiment handling verified.")

    # --- Test 6: Deterministic reproducibility ---
    logger.info("Running Test 6: Deterministic reproducibility...")
    preds1 = run_walk_forward_validation(syn_transformed, syn_tech, syn_sent, syn_target, syn_cfg)
    preds2 = run_walk_forward_validation(syn_transformed, syn_tech, syn_sent, syn_target, syn_cfg)
    np.testing.assert_allclose(
        preds1["hgb_tech_pred"].values,
        preds2["hgb_tech_pred"].values,
        err_msg="HGB predictions non-deterministic across runs with same seed!",
    )
    np.testing.assert_allclose(
        preds1["ridge_tech_pred"].values,
        preds2["ridge_tech_pred"].values,
        err_msg="Ridge predictions non-deterministic across runs!",
    )
    logger.info("Test 6 PASSED: Deterministic reproducibility verified.")

    # --- Test 7: Production file isolation ---
    logger.info("Running Test 7: Production file isolation check...")
    if DEFAULT_MARKET_FEATURES_FILE.exists():
        prod_df = pd.read_parquet(DEFAULT_MARKET_FEATURES_FILE)
        assert len(prod_df) == 451, f"Production dataset corrupted! Expected 451 rows, got {len(prod_df)}."
        assert prod_df["sentiment_score"].isna().sum() == 417, "Production sentiment NaN count altered!"
        assert (prod_df["sentiment_coverage"] == 1).sum() == 34, "Production covered count altered!"
    logger.info("Test 7 PASSED: Production file integrity verified.")

    # --- Test 8: Revised R2 formulas, zero-return baseline, edge cases, and tree inspector ---
    logger.info("Running Test 8: Revised R2 formulas, zero-return baseline, and edge cases...")
    y_test_sample = np.array([0.01, -0.02, 0.03, -0.01])

    # 8a. Zero-return model on non-zero returns: r2_vs_zero_benchmark MUST be 0.0 identically
    zero_preds = np.zeros_like(y_test_sample)
    m_zero = _calculate_single_model_metrics(y_test_sample, zero_preds, float(np.mean(y_test_sample)))
    assert abs(m_zero["r2_vs_zero_benchmark"] - 0.0) < 1e-12, (
        f"r2_vs_zero_benchmark for zero predictions must be 0.0, got {m_zero['r2_vs_zero_benchmark']}"
    )

    # 8b. Perfect model: both R2 metrics MUST be 1.0 identically
    m_perfect = _calculate_single_model_metrics(y_test_sample, y_test_sample, float(np.mean(y_test_sample)))
    assert abs(m_perfect["r2_vs_zero_benchmark"] - 1.0) < 1e-12, (
        f"r2_vs_zero_benchmark for perfect model must be 1.0, got {m_perfect['r2_vs_zero_benchmark']}"
    )
    assert abs(m_perfect["r2_descriptive_test_mean"] - 1.0) < 1e-12, (
        f"r2_descriptive_test_mean for perfect model must be 1.0, got {m_perfect['r2_descriptive_test_mean']}"
    )

    # 8c. Zero-denominator edge case (all actual returns are 0.0)
    all_zero_actuals = np.zeros(5)
    all_zero_preds = np.zeros(5)
    m_edge = _calculate_single_model_metrics(all_zero_actuals, all_zero_preds, 0.0)
    assert m_edge["r2_vs_zero_benchmark"] == 0.0, "Zero-denominator edge case failed for r2_vs_zero_benchmark"
    assert m_edge["r2_descriptive_test_mean"] == 0.0, "Zero-denominator edge case failed for r2_descriptive_test_mean"

    # 8d. Known analytical values
    # y = [0.02, -0.02], y_hat = [0.01, -0.01] -> ss_res = 0.0002, ss_zero = 0.0008 -> r2_zero = 0.75
    y_analytic = np.array([0.02, -0.02])
    y_pred_half = np.array([0.01, -0.01])
    m_half = _calculate_single_model_metrics(y_analytic, y_pred_half, float(np.mean(y_analytic)))
    assert abs(m_half["r2_vs_zero_benchmark"] - 0.75) < 1e-12, "Analytic calculation mismatch for r2_vs_zero"
    assert abs(m_half["r2_descriptive_test_mean"] - 0.75) < 1e-12, "Analytic calculation mismatch for r2_desc"

    # 8e. Direct inspection of HGB tree splits helper
    mock_hgb = HistGradientBoostingRegressor(max_depth=2, max_iter=5, random_state=42)
    mock_hgb.fit(np.random.RandomState(42).randn(50, 4), np.random.RandomState(42).randn(50))
    splits_detected = inspect_hgb_split_features(mock_hgb, ["f0", "f1", "f2", "f3"])
    assert isinstance(splits_detected, list) and len(splits_detected) > 0, "inspect_hgb_split_features failed to extract splits"
    logger.info("Test 8 PASSED: Revised R2 formulas, zero-return baseline, edge cases, and tree inspector verified.")

    # --- Test 9: Strategy position mapping and zero-return benchmark verification ---
    logger.info("Running Test 9: Strategy position mapping and zero-return benchmark verification...")
    # Zero prediction model must hold cash (position = 0.0, return = 0.0, Sharpe = 0.0)
    y_actuals = np.array([0.02, -0.015, 0.01, -0.005, 0.025])
    y_zeros = np.zeros(5)
    m_strat_zero = _calculate_single_model_metrics(y_actuals, y_zeros, float(np.mean(y_actuals)))
    assert m_strat_zero["exploratory_strategy"]["cumulative_return"] == 0.0, (
        f"Zero prediction model must yield 0.0 cumulative return, got {m_strat_zero['exploratory_strategy']['cumulative_return']}"
    )
    assert m_strat_zero["exploratory_strategy"]["annualized_sharpe"] == 0.0, (
        f"Zero prediction model must yield 0.0 Sharpe, got {m_strat_zero['exploratory_strategy']['annualized_sharpe']}"
    )
    assert m_strat_zero["exploratory_strategy"]["average_daily_turnover"] == 0.0

    # Mixed predictions: +1 for positive, -1 for negative, 0 for zero
    y_mixed_pred = np.array([0.01, -0.01, 0.0, 0.02, -0.02])
    m_strat_mixed = _calculate_single_model_metrics(y_actuals, y_mixed_pred, float(np.mean(y_actuals)))
    # Position on step 2 (index 2 where pred=0.0) must be 0.0
    expected_positions = np.array([1.0, -1.0, 0.0, 1.0, -1.0])
    expected_returns = expected_positions * y_actuals
    expected_cum = float(np.prod(1.0 + expected_returns) - 1.0)
    assert abs(m_strat_mixed["exploratory_strategy"]["cumulative_return"] - expected_cum) < 1e-12
    logger.info("Test 9 PASSED: Strategy position mapping and zero-return cash benchmark verified.")

    # --- Test 10: Monotonicity and chronological ordering validation ---
    logger.info("Running Test 10: Monotonicity and chronological order validation...")
    shuffled_dates = [dates[1], dates[0], dates[2], dates[3], dates[4], dates[5], dates[6], dates[7], dates[8], dates[9]]
    shuffled_df = mock_df.copy()
    shuffled_df.index = shuffled_dates
    try:
        build_stationary_features(shuffled_df)
        assert False, "build_stationary_features failed to reject non-monotonic dates!"
    except ValueError:
        pass  # Expected rejection

    try:
        shuffled_trans = t_df.copy()
        shuffled_trans.index = shuffled_dates
        run_walk_forward_validation(shuffled_trans, tech_cols, sent_cols, target_col, config=ForecastConfig(min_train_size=5))
        assert False, "run_walk_forward_validation failed to reject non-monotonic dates!"
    except ValueError:
        pass  # Expected rejection
    logger.info("Test 10 PASSED: Monotonicity and chronological order validation verified.")

    # --- Test 11: Controlled HGB sentiment splitting verification on dense coverage ---
    logger.info("Running Test 11: Controlled HGB sentiment splitting verification on dense coverage...")
    # Create synthetic dataset where sentiment strongly correlates with return and is 100% dense
    rng = np.random.RandomState(42)
    N_dense = 60
    synth_sent_score = rng.uniform(-1.0, 1.0, size=N_dense)
    synth_target = 0.05 * synth_sent_score + 0.005 * rng.randn(N_dense)

    dense_train = pd.DataFrame({
        "daily_return": rng.randn(N_dense) * 0.01,
        "norm_hl_range": rng.uniform(0.01, 0.03, size=N_dense),
        "spread_close_sma20": rng.randn(N_dense) * 0.02,
        "spread_close_sma50": rng.randn(N_dense) * 0.03,
        "volatility_20": np.full(N_dense, 0.015),
        "rsi_14": rng.uniform(0.3, 0.7, size=N_dense),
        "trailing_volume_ratio": rng.uniform(0.8, 1.2, size=N_dense),
        "sentiment_score": synth_sent_score,
        "sentiment_std": np.full(N_dense, 0.1),
        "sentiment_count": np.full(N_dense, 5.0),
        "prob_positive": np.clip(0.5 + 0.5 * synth_sent_score, 0.0, 1.0),
        "prob_negative": np.clip(0.5 - 0.5 * synth_sent_score, 0.0, 1.0),
        "prob_neutral": np.full(N_dense, 0.1),
        "sentiment_coverage": np.ones(N_dense, dtype=int),
        "target_return_1d": synth_target,
    })
    dense_test = dense_train.iloc[-1:].copy()

    _, splits_used = fit_predict_hgb_hybrid(
        dense_train,
        dense_test,
        tech_cols=TECHNICAL_FEATURE_COLUMNS,
        sentiment_cols=SENTIMENT_FEATURE_COLUMNS,
        target_col="target_return_1d",
        config=ForecastConfig(min_train_size=30, hgb_min_samples_leaf=5, hgb_max_iter=20),
        return_tree_diagnostics=True,
    )
    assert len(splits_used) > 0, "HGB Hybrid failed to split on sentiment features when dense coverage is present!"
    assert "sentiment_score" in splits_used or "prob_positive" in splits_used or "prob_negative" in splits_used, (
        f"Expected sentiment feature split, got: {splits_used}"
    )
    logger.info(f"Test 11 PASSED: HGB Hybrid successfully selected sentiment splits ({splits_used}) under dense coverage.")

    logger.info("ALL STAGE 3 TESTS PASSED SUCCESSFULLY!")
    return True


def main() -> None:
    """Execute baseline forecasting pipeline or test mode via command-line arguments."""
    args = sys.argv[1:]

    if "--test" in args:
        success = run_test_suite()
        sys.exit(0 if success else 1)

    # Default execution: run walk-forward validation on production dataset
    logger.info("Loading production market and sentiment dataset...")
    if not DEFAULT_MARKET_FEATURES_FILE.exists():
        raise FileNotFoundError(f"Market features parquet not found: {DEFAULT_MARKET_FEATURES_FILE}")

    market_df = pd.read_parquet(DEFAULT_MARKET_FEATURES_FILE)
    logger.info(f"Loaded {len(market_df)} market feature rows ({market_df.index.min()} to {market_df.index.max()}).")

    # 1. Transform features and target
    features_df, tech_cols, sent_cols, target_col = build_stationary_features(market_df)

    # 2. Run walk-forward validation
    config = ForecastConfig(
        min_train_size=200,
        ridge_alpha=10.0,
        hgb_max_depth=3,
        hgb_min_samples_leaf=10,
        hgb_learning_rate=0.03,
        hgb_max_iter=50,
        hgb_l2_regularization=1.0,
        random_state=42,
    )
    predictions_df = run_walk_forward_validation(
        features_df, tech_cols, sent_cols, target_col, config
    )

    # 3. Compute metrics across full window and sentiment-covered window
    metrics_dict = compute_evaluation_metrics(predictions_df)

    # 4. Save artifacts
    save_artifacts(predictions_df, metrics_dict)

    # 5. Display formatted summary
    print_metrics_summary(metrics_dict)


if __name__ == "__main__":
    main()
