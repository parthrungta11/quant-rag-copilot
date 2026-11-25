"""Realistic execution and portfolio backtesting engine for quantitative return forecasts.

Implements realistic forward execution timing, transaction friction (broker fees + slippage),
short borrowing costs, position sizing, benchmark comparisons, and comprehensive portfolio metrics.

Core Timing Invariant
---------------------
A forecast generated at session t (incorporating technical indicators up to Close[t] and
news sentiment published up to 16:00:00 EDT) cannot execute instantaneously at Close[t].
Market-on-Close (MOC) cutoffs and compute latency require order routing overnight/pre-market.
Execution is simulated at subsequent available prices:
1. 'next_open_to_close' (Primary):
   - Entry: Open[t+1]
   - Exit: Close[t+1]
   - Asset Period Return: (Close[t+1] - Open[t+1]) / Open[t+1]
   - Overnight gap risk is eliminated; positions are held strictly intraday.
2. 'next_open_to_next_open' (Secondary):
   - Entry: Open[t+1]
   - Exit: Open[t+2]
   - Asset Period Return: (Open[t+2] - Open[t+1]) / Open[t+1]
   - Requires valid Open[t+2]; final row without t+2 is cleanly excluded.

Strict Separation
-----------------
Forecasting accuracy metrics (RMSE, MAE, R², Directional Accuracy, IC) are strictly distinct
from portfolio strategy metrics (Cumulative Return, Sharpe, Sortino, Max Drawdown, Turnover).
Strategy returns are computed directly from execution prices and position turnover, never
substituted with the forecasting target 'actual_return'.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


logger = logging.getLogger("backtest_engine")


@dataclass
class BacktestConfig:
    """Configuration parameters for execution simulation and portfolio backtesting.

    Parameters
    ----------
    initial_capital : float
        Initial portfolio equity in currency units (default: 100,000.0).
    execution_mode : str
        Execution price timing convention:
        - 'next_open_to_close': Enter at Open[t+1], exit at Close[t+1] (default).
        - 'next_open_to_next_open': Enter at Open[t+1], exit at Open[t+2].
    transaction_cost_bps : float
        One-way broker commission and exchange fee in basis points per unit of turnover (default: 5.0 bps).
    slippage_bps : float
        One-way market impact and execution slippage in basis points per unit of turnover (default: 5.0 bps).
    short_borrow_fee_bps_annual : float
        Annualized cost to borrow stock for short exposure in basis points (default: 100.0 bps = 1.0%/year).
    allow_short : bool
        If True, negative signals take short positions (-1.0). If False, negative signals hold cash (0.0).
    position_sizing : str
        Position allocation methodology:
        - 'ternary_fixed': Fixed unit exposure in {-1.0, 0.0, +1.0} based on threshold (default).
        - 'long_only': Fixed unit exposure in {0.0, +1.0} based on threshold.
    conviction_threshold : float
        Minimum absolute forecast |y_hat| required to enter a position. Signals with |y_hat| <= threshold
        default to Cash (0.0).
    risk_free_rate_annual : float
        Annualized risk-free benchmark rate for Sharpe and Sortino ratio calculations (default: 0.0).
    periods_per_year : float
        Number of trading sessions per calendar year for metric annualization (default: 252.0).
    """

    initial_capital: float = 100_000.0
    execution_mode: str = "next_open_to_close"
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 5.0
    short_borrow_fee_bps_annual: float = 100.0
    allow_short: bool = True
    position_sizing: str = "ternary_fixed"
    allocation_rule: Optional[str] = None
    conviction_threshold: Union[float, pd.Series] = 0.0
    risk_free_rate_annual: float = 0.0
    periods_per_year: float = 252.0

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {self.initial_capital}")
        valid_modes = {"next_open_to_close", "next_open_to_next_open"}
        if self.execution_mode not in valid_modes:
            raise ValueError(f"Invalid execution_mode '{self.execution_mode}'. Allowed: {valid_modes}")

        if self.allocation_rule is not None:
            if self.allocation_rule in {"ternary", "ternary_fixed"}:
                self.position_sizing = "ternary_fixed"
                self.allocation_rule = "ternary"
            elif self.allocation_rule in {"long_only", "long"}:
                self.position_sizing = "long_only"
                self.allow_short = False
                self.allocation_rule = "long_only"
            else:
                raise ValueError(f"Invalid allocation_rule '{self.allocation_rule}'. Allowed: {{'ternary', 'long_only'}}")
        elif self.position_sizing == "long_only":
            self.allow_short = False
            self.allocation_rule = "long_only"
        else:
            self.allocation_rule = "ternary"

        valid_sizings = {"ternary_fixed", "long_only"}
        if self.position_sizing not in valid_sizings:
            raise ValueError(f"Invalid position_sizing '{self.position_sizing}'. Allowed: {valid_sizings}")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps cannot be negative")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps cannot be negative")
        if self.short_borrow_fee_bps_annual < 0:
            raise ValueError("short_borrow_fee_bps_annual cannot be negative")
        if isinstance(self.conviction_threshold, (int, float)):
            if self.conviction_threshold < 0:
                raise ValueError("conviction_threshold cannot be negative")
        elif isinstance(self.conviction_threshold, pd.Series):
            if (self.conviction_threshold < 0).any():
                raise ValueError("conviction_threshold values cannot be negative")
        else:
            raise TypeError(f"conviction_threshold must be float or pd.Series, got {type(self.conviction_threshold)}")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")


@dataclass
class BacktestResult:
    """Comprehensive portfolio performance and ledger artifacts returned by a backtest."""

    model_name: str
    config: BacktestConfig
    portfolio_history: pd.DataFrame
    metrics: Dict[str, Any]
    benchmark_metrics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


def map_prediction_to_position(
    prediction: float,
    config: BacktestConfig,
    threshold: Optional[float] = None,
) -> float:
    """Map a scalar return prediction to a discrete portfolio position under configured rules.

    Parameters
    ----------
    prediction : float
        Point forecast of forward return.
    config : BacktestConfig
        Configuration defining position_sizing, allow_short, and conviction_threshold.
    threshold : Optional[float]
        Explicit conviction threshold to evaluate against. If None, falls back to
        config.conviction_threshold (if scalar float) or 0.0.

    Returns
    -------
    float
        Target position exposure: +1.0 (Long), -1.0 (Short), or 0.0 (Cash/Neutral).
    """
    if pd.isna(prediction):
        return 0.0

    if threshold is not None:
        thresh = threshold
    elif isinstance(config.conviction_threshold, (int, float)):
        thresh = float(config.conviction_threshold)
    else:
        thresh = 0.0

    if config.position_sizing == "long_only":
        return 1.0 if prediction > thresh else 0.0

    # ternary_fixed mode
    if prediction > thresh:
        return 1.0
    elif prediction < -thresh:
        return -1.0 if config.allow_short else 0.0
    else:
        return 0.0


def _calculate_drawdown_series(equity_curve: pd.Series) -> Tuple[pd.Series, float, int]:
    """Calculate drawdown percentage series, maximum drawdown, and maximum drawdown duration.

    Parameters
    ----------
    equity_curve : pd.Series
        Compounded portfolio equity series.

    Returns
    -------
    Tuple[pd.Series, float, int]
        - Drawdown series (values in [0.0, 1.0] representing drawdown from running peak).
        - Maximum drawdown scalar (float >= 0.0).
        - Maximum drawdown duration in trading sessions (integer >= 0).
    """
    if len(equity_curve) == 0:
        return pd.Series(dtype=float), 0.0, 0

    running_peak = equity_curve.cummax()
    # Avoid zero division if peak drops to 0 or below
    safe_peak = running_peak.where(running_peak > 1e-12, 1e-12)
    drawdown = (running_peak - equity_curve) / safe_peak
    drawdown = drawdown.clip(lower=0.0)

    max_dd = float(drawdown.max()) if len(drawdown) > 0 else 0.0

    # Calculate maximum drawdown duration (consecutive periods where equity < peak)
    in_drawdown = drawdown > 1e-12
    max_duration = 0
    current_duration = 0
    for val in in_drawdown:
        if val:
            current_duration += 1
            if current_duration > max_duration:
                max_duration = current_duration
        else:
            current_duration = 0

    return drawdown, max_dd, max_duration


def compute_performance_metrics(
    net_returns: pd.Series,
    gross_returns: pd.Series,
    turnover_series: pd.Series,
    borrow_costs: pd.Series,
    friction_costs: pd.Series,
    initial_capital: float,
    periods_per_year: float = 252.0,
    risk_free_rate_annual: float = 0.0,
) -> Dict[str, Any]:
    """Compute financial performance, risk, and friction metrics from ledger series.

    Parameters
    ----------
    net_returns : pd.Series
        Periodic net return series (after friction and borrow fees).
    gross_returns : pd.Series
        Periodic gross return series (before friction and borrow fees).
    turnover_series : pd.Series
        Periodic absolute turnover series.
    borrow_costs : pd.Series
        Periodic short borrow fee deductions.
    friction_costs : pd.Series
        Periodic transaction fee and slippage deductions.
    initial_capital : float
        Starting equity.
    periods_per_year : float
        Annualization factor (default 252.0).
    risk_free_rate_annual : float
        Annualized risk-free rate for Sharpe/Sortino.

    Returns
    -------
    Dict[str, Any]
        Dictionary of standardized financial metrics.
    """
    n_sessions = len(net_returns)
    if n_sessions == 0:
        return {
            "n_sessions": 0,
            "cumulative_net_return": 0.0,
            "cumulative_gross_return": 0.0,
            "annualized_net_return": 0.0,
            "annualized_gross_return": 0.0,
            "annualized_volatility": 0.0,
            "annualized_sharpe": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_duration_sessions": 0,
            "calmar_ratio": 0.0,
            "average_daily_turnover": 0.0,
            "total_turnover": 0.0,
            "total_friction_cost": 0.0,
            "total_borrow_cost": 0.0,
            "friction_drag": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
            "ending_equity": initial_capital,
        }

    # Equity curves
    net_equity = initial_capital * (1.0 + net_returns).cumprod()
    gross_equity = initial_capital * (1.0 + gross_returns).cumprod()

    cum_net = float(net_equity.iloc[-1] / initial_capital - 1.0)
    cum_gross = float(gross_equity.iloc[-1] / initial_capital - 1.0)
    friction_drag = float(cum_gross - cum_net)

    # Annualized Returns (Compound Annual Growth Rate)
    years = n_sessions / periods_per_year
    if years > 0 and net_equity.iloc[-1] > 0:
        ann_net = float((net_equity.iloc[-1] / initial_capital) ** (1.0 / years) - 1.0)
    elif net_equity.iloc[-1] <= 0:
        ann_net = -1.0
    else:
        ann_net = 0.0

    if years > 0 and gross_equity.iloc[-1] > 0:
        ann_gross = float((gross_equity.iloc[-1] / initial_capital) ** (1.0 / years) - 1.0)
    elif gross_equity.iloc[-1] <= 0:
        ann_gross = -1.0
    else:
        ann_gross = 0.0

    # Volatility
    ann_vol = float(net_returns.std(ddof=1) * np.sqrt(periods_per_year)) if n_sessions > 1 else 0.0

    # Sharpe Ratio
    excess_return = ann_net - risk_free_rate_annual
    sharpe = float(excess_return / ann_vol) if ann_vol > 1e-12 else 0.0

    # Sortino Ratio (downside volatility below risk-free periodic return)
    rf_periodic = risk_free_rate_annual / periods_per_year
    downside_diff = np.minimum(0.0, net_returns - rf_periodic)
    downside_dev = float(np.sqrt(np.mean(downside_diff ** 2)) * np.sqrt(periods_per_year))
    sortino = float(excess_return / downside_dev) if downside_dev > 1e-12 else 0.0

    # Drawdown metrics
    _, max_dd, max_dd_dur = _calculate_drawdown_series(net_equity)
    calmar = float(ann_net / max_dd) if max_dd > 1e-12 else 0.0

    # Turnover & Friction
    avg_turnover = float(turnover_series.mean())
    tot_turnover = float(turnover_series.sum())
    tot_friction = float(friction_costs.sum())
    tot_borrow = float(borrow_costs.sum())

    # Trade Statistics (Win rate & Profit factor on active days)
    active_mask = (turnover_series > 1e-6) | (net_returns != 0.0)
    active_returns = net_returns[active_mask]
    pos_returns = active_returns[active_returns > 0.0]
    neg_returns = active_returns[active_returns < 0.0]

    n_active = len(active_returns)
    n_wins = len(pos_returns)
    win_rate = float(n_wins / n_active) if n_active > 0 else 0.0

    gross_profit = float(pos_returns.sum())
    gross_loss = float(neg_returns.abs().sum())
    if gross_loss > 1e-12:
        profit_factor = float(gross_profit / gross_loss)
    elif gross_profit > 1e-12:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    # Count of non-zero position initiations/rebalancings
    total_trades = int((turnover_series > 1e-6).sum())

    return {
        "n_sessions": n_sessions,
        "cumulative_net_return": cum_net,
        "cumulative_gross_return": cum_gross,
        "annualized_net_return": ann_net,
        "annualized_gross_return": ann_gross,
        "annualized_volatility": ann_vol,
        "annualized_sharpe": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "max_drawdown_duration_sessions": max_dd_dur,
        "calmar_ratio": calmar,
        "average_daily_turnover": avg_turnover,
        "total_turnover": tot_turnover,
        "total_friction_cost": tot_friction,
        "total_borrow_cost": tot_borrow,
        "friction_drag": friction_drag,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_trades": total_trades,
        "ending_equity": float(net_equity.iloc[-1]),
    }


def run_backtest(
    predictions: Union[pd.Series, pd.DataFrame],
    market_data: pd.DataFrame,
    config: Optional[BacktestConfig] = None,
    model_name: str = "strategy",
    prediction_col: Optional[str] = None,
    threshold_series: Optional[pd.Series] = None,
    strict: bool = False,
) -> BacktestResult:
    """Execute realistic out-of-sample portfolio simulation under verified price timing.

    Parameters
    ----------
    predictions : Union[pd.Series, pd.DataFrame]
        Series of forward return predictions indexed by decision session t, or DataFrame
        containing predictions. If DataFrame, prediction_col must be specified or exactly
        one prediction column must be present.
    market_data : pd.DataFrame
        Market data DataFrame with DatetimeIndex containing required OHLC price columns
        ('Open', 'Close'). Must cover decision dates and subsequent execution dates.
    config : Optional[BacktestConfig]
        Simulation configuration. If None, default BacktestConfig is used.
    model_name : str
        Human-readable label for the model / strategy.
    prediction_col : Optional[str]
        Column name to extract if predictions is a DataFrame.
    threshold_series : Optional[pd.Series]
        Optional Series of decision-date indexed conviction thresholds tau_t. If provided,
        overrides config.conviction_threshold for each decision session t.

    Returns
    -------
    BacktestResult
        Result containing portfolio history DataFrame, performance metrics, and benchmark metrics.

    Execution Flow & Look-Ahead Prevention
    --------------------------------------
    1. For each decision session t in predictions index:
       - The signal pos_{t+1} is derived strictly from prediction[t].
       - If prediction[t] is NaN or missing, pos_{t+1} = 0.0 (cash).
    2. Session t+1 is identified as the immediately subsequent chronological session in market_data.
       - If session t+1 does not exist (e.g. final unrealized prediction row), that row is cleanly excluded.
    3. In 'next_open_to_close' mode:
       - Entry occurs at Open[t+1]
       - Exit occurs at Close[t+1]
       - Gross asset return = (Close[t+1] - Open[t+1]) / Open[t+1]
    4. In 'next_open_to_next_open' mode:
       - Entry occurs at Open[t+1]
       - Exit occurs at Open[t+2]
       - Gross asset return = (Open[t+2] - Open[t+1]) / Open[t+1]
       - Requires session t+2 in market_data; rows without t+2 are cleanly excluded.
    5. Portfolio turnover, transaction costs, slippage, and short borrow fees are deducted
       from gross returns to generate exact net portfolio returns and compounded equity.
    """
    if config is None:
        config = BacktestConfig()

    # 1. Parse prediction series
    if isinstance(predictions, pd.DataFrame):
        if prediction_col is not None:
            if prediction_col not in predictions.columns:
                raise KeyError(f"Specified prediction_col '{prediction_col}' not found in predictions DataFrame.")
            pred_series = predictions[prediction_col].copy()
            if model_name == "strategy":
                model_name = prediction_col
        else:
            # Check for candidate prediction columns
            pred_cols = [c for c in predictions.columns if c.endswith("_pred") or "pred" in c.lower()]
            if len(pred_cols) == 1:
                pred_series = predictions[pred_cols[0]].copy()
                if model_name == "strategy":
                    model_name = pred_cols[0]
            elif len(predictions.columns) == 1:
                pred_series = predictions.iloc[:, 0].copy()
                if model_name == "strategy":
                    model_name = predictions.columns[0]
            else:
                raise ValueError(
                    f"predictions DataFrame has multiple columns {list(predictions.columns)}. "
                    "Must specify prediction_col explicitly."
                )
    elif isinstance(predictions, pd.Series):
        pred_series = predictions.copy()
    else:
        raise TypeError(f"predictions must be pd.Series or pd.DataFrame, got {type(predictions)}")

    # 2. Validate indices and required columns
    if not isinstance(market_data.index, pd.DatetimeIndex):
        raise TypeError("market_data index must be a pd.DatetimeIndex.")
    if not isinstance(pred_series.index, pd.DatetimeIndex):
        raise TypeError("predictions index must be a pd.DatetimeIndex.")

    if not market_data.index.is_monotonic_increasing:
        raise ValueError("market_data index must be monotonically increasing.")
    if not pred_series.index.is_monotonic_increasing:
        raise ValueError("predictions index must be monotonically increasing.")

    required_market_cols = ["Open", "Close"]
    for col in required_market_cols:
        if col not in market_data.columns:
            raise KeyError(f"market_data missing required column: '{col}'")

    # Optional sentiment_coverage alignment check (backward-compatible warning; raises if strict=True)
    if (
        isinstance(predictions, pd.DataFrame)
        and "sentiment_coverage" in predictions.columns
        and "sentiment_coverage" in market_data.columns
    ):
        common_dates = predictions.index.intersection(market_data.index)
        if len(common_dates) > 0:
            pred_cov = predictions.loc[common_dates, "sentiment_coverage"].fillna(-1)
            mkt_cov = market_data.loc[common_dates, "sentiment_coverage"].fillna(-1)
            mismatches = int((pred_cov != mkt_cov).sum())
            if mismatches > 0:
                msg = (
                    f"Sentiment coverage mismatch between predictions and market_data on "
                    f"{mismatches}/{len(common_dates)} overlapping session(s). "
                    "Predictions may be stale relative to market_data."
                )
                logger.warning(msg)
                if strict:
                    raise ValueError(msg)

    # 3. Align chronological execution sessions
    market_index = market_data.index
    market_loc_map = {ts: idx for idx, ts in enumerate(market_index)}

    records = []
    prev_position = 0.0

    one_way_cost_rate = (config.transaction_cost_bps + config.slippage_bps) / 10_000.0
    borrow_rate_per_session = (config.short_borrow_fee_bps_annual / 10_000.0) / config.periods_per_year

    for decision_date, pred_val in pred_series.items():
        if decision_date not in market_loc_map:
            logger.warning(f"Decision date {decision_date} not present in market_data; skipping.")
            continue

        loc_t = market_loc_map[decision_date]
        loc_t1 = loc_t + 1

        # Check if execution session t+1 exists
        if loc_t1 >= len(market_index):
            # Final unrealized prediction row (future market prices unavailable)
            continue

        exec_date = market_index[loc_t1]
        open_t1 = market_data.iloc[loc_t1]["Open"]
        close_t1 = market_data.iloc[loc_t1]["Close"]

        # Resolve current conviction threshold for this decision session
        if threshold_series is not None:
            if decision_date in threshold_series.index:
                val = threshold_series.loc[decision_date]
                current_thresh = float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)
            else:
                current_thresh = 0.0
        elif isinstance(config.conviction_threshold, pd.Series):
            if decision_date in config.conviction_threshold.index:
                val = config.conviction_threshold.loc[decision_date]
                current_thresh = float(val.iloc[0]) if isinstance(val, pd.Series) else float(val)
            else:
                current_thresh = 0.0
        else:
            current_thresh = float(config.conviction_threshold)

        # Validate prices
        invalid_prices = False
        if pd.isna(open_t1) or pd.isna(close_t1) or open_t1 <= 0 or close_t1 <= 0:
            logger.warning(
                f"Invalid execution prices at session {exec_date} (Open={open_t1}, Close={close_t1}); "
                "carrying previous position unchanged with zero friction."
            )
            target_pos = prev_position
            asset_ret = 0.0
            entry_price = float(open_t1) if pd.notna(open_t1) else np.nan
            exit_price = float(close_t1) if pd.notna(close_t1) else np.nan
            exit_date = exec_date
            invalid_prices = True
        elif config.execution_mode == "next_open_to_close":
            target_pos = map_prediction_to_position(pred_val, config, threshold=current_thresh)
            entry_price = float(open_t1)
            exit_price = float(close_t1)
            exit_date = exec_date
            asset_ret = (exit_price - entry_price) / entry_price
        elif config.execution_mode == "next_open_to_next_open":
            loc_t2 = loc_t1 + 1
            if loc_t2 >= len(market_index):
                # Cannot realize open-to-next-open return without t+2
                continue
            open_t2 = market_data.iloc[loc_t2]["Open"]
            if pd.isna(open_t2) or open_t2 <= 0:
                logger.warning(f"Invalid exit price at session t+2 {market_index[loc_t2]}; skipping.")
                continue
            target_pos = map_prediction_to_position(pred_val, config, threshold=current_thresh)
            entry_price = float(open_t1)
            exit_price = float(open_t2)
            exit_date = market_index[loc_t2]
            asset_ret = (exit_price - entry_price) / entry_price
        else:
            raise ValueError(f"Unsupported execution_mode: {config.execution_mode}")

        if invalid_prices:
            turnover = 0.0
            friction_cost = 0.0
            borrow_cost = 0.0
            strat_gross = 0.0
            strat_net = 0.0
        else:
            # Position Turnover & Transaction Friction
            turnover = abs(target_pos - prev_position)
            friction_cost = turnover * one_way_cost_rate

            # Short Borrow Cost (applied only while short)
            if target_pos < 0.0:
                borrow_cost = abs(target_pos) * borrow_rate_per_session
            else:
                borrow_cost = 0.0

            # Strategy Returns
            strat_gross = target_pos * asset_ret
            strat_net = strat_gross - friction_cost - borrow_cost

        # Buy & Hold AAPL Benchmark Return for the identical session
        # (Evaluated on the exact same price path: Open[t+1] to Close[t+1] or Open[t+1] to Open[t+2])
        benchmark_asset_ret = asset_ret

        records.append({
            "decision_date": decision_date,
            "execution_date": exec_date,
            "exit_date": exit_date,
            "prediction": float(pred_val) if pd.notna(pred_val) else np.nan,
            "conviction_threshold": current_thresh,
            "position": target_pos,
            "prev_position": prev_position,
            "turnover": turnover,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "asset_gross_return": asset_ret,
            "strategy_gross_return": strat_gross,
            "friction_cost": friction_cost,
            "borrow_cost": borrow_cost,
            "strategy_net_return": strat_net,
            "benchmark_gross_return": benchmark_asset_ret,
        })

        prev_position = target_pos

    if not records:
        empty_history = pd.DataFrame(columns=[
            "decision_date", "execution_date", "exit_date", "prediction", "conviction_threshold", "position",
            "prev_position", "turnover", "entry_price", "exit_price", "asset_gross_return",
            "strategy_gross_return", "friction_cost", "borrow_cost", "strategy_net_return",
            "portfolio_value", "cumulative_net_return", "drawdown", "benchmark_gross_return",
            "benchmark_value", "benchmark_cumulative_return",
        ])
        empty_metrics = compute_performance_metrics(
            pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float),
            pd.Series(dtype=float), pd.Series(dtype=float), config.initial_capital,
            config.periods_per_year, config.risk_free_rate_annual
        )
        return BacktestResult(
            model_name=model_name,
            config=config,
            portfolio_history=empty_history,
            metrics=empty_metrics,
            benchmark_metrics=empty_metrics,
            metadata={"n_records": 0, "status": "empty"},
        )

    history_df = pd.DataFrame(records)
    history_df.set_index("execution_date", inplace=True)

    # Calculate compounded portfolio equity and drawdown
    net_returns = history_df["strategy_net_return"]
    gross_returns = history_df["strategy_gross_return"]
    turnovers = history_df["turnover"]
    borrow_costs = history_df["borrow_cost"]
    friction_costs = history_df["friction_cost"]

    portfolio_equity = config.initial_capital * (1.0 + net_returns).cumprod()
    history_df["portfolio_value"] = portfolio_equity
    history_df["cumulative_net_return"] = portfolio_equity / config.initial_capital - 1.0

    drawdown_series, _, _ = _calculate_drawdown_series(portfolio_equity)
    history_df["drawdown"] = drawdown_series

    # Benchmark equity (AAPL Buy-and-Hold over the identical execution window)
    benchmark_net_returns = history_df["benchmark_gross_return"]
    # AAPL buy and hold incurs initial 1-way entry fee and final exit fee (frictionless comparison also provided)
    benchmark_equity = config.initial_capital * (1.0 + benchmark_net_returns).cumprod()
    history_df["benchmark_value"] = benchmark_equity
    history_df["benchmark_cumulative_return"] = benchmark_equity / config.initial_capital - 1.0

    # Strategy Metrics
    metrics = compute_performance_metrics(
        net_returns=net_returns,
        gross_returns=gross_returns,
        turnover_series=turnovers,
        borrow_costs=borrow_costs,
        friction_costs=friction_costs,
        initial_capital=config.initial_capital,
        periods_per_year=config.periods_per_year,
        risk_free_rate_annual=config.risk_free_rate_annual,
    )

    # Benchmark Metrics
    benchmark_turnover = pd.Series(0.0, index=history_df.index)
    if len(benchmark_turnover) > 0:
        benchmark_turnover.iloc[0] = 1.0  # initial entry turnover
    benchmark_friction = benchmark_turnover * one_way_cost_rate
    benchmark_borrow = pd.Series(0.0, index=history_df.index)

    benchmark_metrics = compute_performance_metrics(
        net_returns=benchmark_net_returns - benchmark_friction,
        gross_returns=benchmark_net_returns,
        turnover_series=benchmark_turnover,
        borrow_costs=benchmark_borrow,
        friction_costs=benchmark_friction,
        initial_capital=config.initial_capital,
        periods_per_year=config.periods_per_year,
        risk_free_rate_annual=config.risk_free_rate_annual,
    )

    metadata = {
        "n_evaluated_sessions": len(history_df),
        "first_execution_date": str(history_df.index[0].date()) if len(history_df) > 0 else None,
        "last_execution_date": str(history_df.index[-1].date()) if len(history_df) > 0 else None,
        "execution_mode": config.execution_mode,
        "position_sizing": config.position_sizing,
        "allow_short": config.allow_short,
        "transaction_cost_bps": config.transaction_cost_bps,
        "slippage_bps": config.slippage_bps,
        "short_borrow_fee_bps_annual": config.short_borrow_fee_bps_annual,
    }

    return BacktestResult(
        model_name=model_name,
        config=config,
        portfolio_history=history_df,
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
        metadata=metadata,
    )


def compare_models(
    predictions_df: pd.DataFrame,
    market_data: pd.DataFrame,
    prediction_cols: Optional[List[str]] = None,
    config: Optional[BacktestConfig] = None,
) -> Tuple[Dict[str, BacktestResult], pd.DataFrame]:
    """Execute backtests for multiple model prediction columns and compile comparative table.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame of out-of-sample walk-forward predictions.
    market_data : pd.DataFrame
        Market data DataFrame containing 'Open' and 'Close'.
    prediction_cols : Optional[List[str]]
        List of column names to evaluate. If None, automatically detects columns ending with '_pred'.
    config : Optional[BacktestConfig]
        Common backtest configuration applied to all models.

    Returns
    -------
    Tuple[Dict[str, BacktestResult], pd.DataFrame]
        - Dictionary mapping model names to their respective BacktestResult objects.
        - Comparison summary DataFrame ranking all models and AAPL Buy-and-Hold by Sharpe ratio.
    """
    if config is None:
        config = BacktestConfig()

    if prediction_cols is None:
        prediction_cols = [c for c in predictions_df.columns if c.endswith("_pred") or "pred" in c.lower()]
        if not prediction_cols:
            raise ValueError(f"No prediction columns found in predictions_df. Available: {list(predictions_df.columns)}")

    results = {}
    summary_rows = []

    benchmark_recorded = False

    for col in prediction_cols:
        model_label = col.replace("_pred", "")
        res = run_backtest(
            predictions=predictions_df[col],
            market_data=market_data,
            config=config,
            model_name=model_label,
        )
        results[model_label] = res

        m = res.metrics
        summary_rows.append({
            "Model": model_label,
            "Type": "Strategy",
            "Cum Net Return": m["cumulative_net_return"],
            "Ann Net Return": m["annualized_net_return"],
            "Ann Volatility": m["annualized_volatility"],
            "Sharpe Ratio": m["annualized_sharpe"],
            "Sortino Ratio": m["sortino_ratio"],
            "Max Drawdown": m["max_drawdown"],
            "Calmar Ratio": m["calmar_ratio"],
            "Daily Turnover": m["average_daily_turnover"],
            "Friction Drag": m["friction_drag"],
            "Win Rate": m["win_rate"],
            "Profit Factor": m["profit_factor"],
            "Trades": m["total_trades"],
        })

        if not benchmark_recorded and len(res.benchmark_metrics) > 0:
            bm = res.benchmark_metrics
            summary_rows.append({
                "Model": "AAPL Buy & Hold",
                "Type": "Benchmark",
                "Cum Net Return": bm["cumulative_net_return"],
                "Ann Net Return": bm["annualized_net_return"],
                "Ann Volatility": bm["annualized_volatility"],
                "Sharpe Ratio": bm["annualized_sharpe"],
                "Sortino Ratio": bm["sortino_ratio"],
                "Max Drawdown": bm["max_drawdown"],
                "Calmar Ratio": bm["calmar_ratio"],
                "Daily Turnover": bm["average_daily_turnover"],
                "Friction Drag": bm["friction_drag"],
                "Win Rate": bm["win_rate"],
                "Profit Factor": bm["profit_factor"],
                "Trades": bm["total_trades"],
            })
            benchmark_recorded = True

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df.sort_values("Sharpe Ratio", ascending=False, inplace=True)
        summary_df.reset_index(drop=True, inplace=True)

    return results, summary_df


def evaluate_threshold_candidates(
    predictions: Union[pd.DataFrame, pd.Series],
    market_data: pd.DataFrame,
    training_fold_stds: pd.Series,
    candidate_k_values: Sequence[float] = (0.25, 0.50, 0.75, 1.00),
    base_config: Optional[BacktestConfig] = None,
    evaluation_dates: Optional[Sequence[pd.Timestamp]] = None,
    model_name: str = "ridge_tech",
    prediction_col: Optional[str] = None,
) -> Tuple[Dict[str, BacktestResult], pd.DataFrame]:
    """Evaluate controlled strategy variants across baseline and candidate conviction thresholds.

    Variant Architecture (Gate D7):
    --------------------------------
    A. Existing Ternary Baseline (Control 1):
       - Long, short, or cash.
       - Threshold = 0.0.
       - Preserves existing strategy behavior as baseline control.
    B. Long-Only Baseline (Control 2):
       - Positive signal maps to long (+1.0).
       - Zero or negative signal maps to cash (0.0).
       - Zero short positions.
       - Threshold = 0.0.
    C. Long-Only with Adaptive Dynamic Conviction Threshold:
       - Calibrated strictly using past training-fold prediction dispersion.
       - tau_t = k * std(training_fold_predictions).
       - Evaluated across candidate k values (e.g. 0.25, 0.50, 0.75, 1.00).
       - Positive signal > tau_t maps to long (+1.0).
       - Signal <= tau_t maps to cash (0.0).
       - Zero short positions.

    Parameters
    ----------
    predictions : Union[pd.DataFrame, pd.Series]
        Predictions series or DataFrame containing predictions.
    market_data : pd.DataFrame
        OHLC market data covering the evaluation sessions and execution sessions.
    training_fold_stds : pd.Series
        Decision-date indexed series of standard deviations of in-sample predictions
        calibrated strictly within each corresponding training fold [0, t-1].
    candidate_k_values : Sequence[float]
        Multiplier values for the dynamic threshold tau_t = k * sigma_t (default: 0.25, 0.50, 0.75, 1.00).
    base_config : Optional[BacktestConfig]
        Base configuration settings (fees, slippage, execution mode, capital). If None, default is used.
    evaluation_dates : Optional[Sequence[pd.Timestamp]]
        Optional sequence of decision dates restricting the evaluation window (e.g., development window).
    model_name : str
        Label prefix for the model (default: 'ridge_tech').
    prediction_col : Optional[str]
        Prediction column name if predictions is a DataFrame.

    Returns
    -------
    Tuple[Dict[str, BacktestResult], pd.DataFrame]
        A dictionary mapping variant labels to BacktestResult objects, and a comparative
        summary DataFrame sorted by Sharpe Ratio descending.
    """
    if base_config is None:
        base_config = BacktestConfig()

    # 1. Parse prediction series
    if isinstance(predictions, pd.DataFrame):
        if prediction_col is not None:
            if prediction_col not in predictions.columns:
                raise KeyError(f"Specified prediction_col '{prediction_col}' not found in predictions DataFrame.")
            pred_series = predictions[prediction_col].copy()
        else:
            pred_cols = [c for c in predictions.columns if c.endswith("_pred") or "pred" in c.lower()]
            if len(pred_cols) == 1:
                pred_series = predictions[pred_cols[0]].copy()
            elif len(predictions.columns) == 1:
                pred_series = predictions.iloc[:, 0].copy()
            else:
                raise ValueError("Multiple columns in predictions DataFrame; specify 'prediction_col'.")
    elif isinstance(predictions, pd.Series):
        pred_series = predictions.copy()
    else:
        raise TypeError(f"predictions must be pd.Series or pd.DataFrame, got {type(predictions)}")

    # 2. Restrict to evaluation window if specified
    stds_series = training_fold_stds.copy()
    if evaluation_dates is not None:
        eval_dt_index = pd.DatetimeIndex(evaluation_dates)
        pred_series = pred_series.loc[pred_series.index.isin(eval_dt_index)]
        stds_series = stds_series.loc[stds_series.index.isin(eval_dt_index)]
        if pred_series.empty:
            raise ValueError("No predictions match the specified evaluation_dates.")

    def _spawn_config(allocation_rule: str, threshold: Union[float, pd.Series]) -> BacktestConfig:
        return BacktestConfig(
            initial_capital=base_config.initial_capital,
            transaction_cost_bps=base_config.transaction_cost_bps,
            slippage_bps=base_config.slippage_bps,
            short_borrow_fee_bps_annual=base_config.short_borrow_fee_bps_annual,
            periods_per_year=base_config.periods_per_year,
            risk_free_rate_annual=base_config.risk_free_rate_annual,
            execution_mode=base_config.execution_mode,
            allocation_rule=allocation_rule,
            conviction_threshold=threshold,
        )

    # 3. Define variants
    # Variant A: Ternary Control (k=0.0)
    # Variant B: Long-Only Control (k=0.0)
    # Variant C: Long-Only Adaptive (k in candidate_k_values)
    variants = [
        ("Ternary Control (k=0.0)", "ternary", 0.0, None),
        ("Long-Only Control (k=0.0)", "long_only", 0.0, None),
    ]
    for k in candidate_k_values:
        thresh_series = k * stds_series
        variants.append((f"Long-Only Adaptive (k={k:.2f})", "long_only", k, thresh_series))

    results: Dict[str, BacktestResult] = {}
    summary_rows = []
    benchmark_recorded = False

    for var_label, alloc_rule, k_val, dynamic_thresh in variants:
        cfg = _spawn_config(
            allocation_rule=alloc_rule,
            threshold=dynamic_thresh if dynamic_thresh is not None else 0.0,
        )
        res = run_backtest(
            predictions=pred_series,
            market_data=market_data,
            config=cfg,
            threshold_series=dynamic_thresh,
            model_name=f"{model_name}_{var_label}",
        )
        results[var_label] = res

        m = res.metrics
        mean_thresh_bps = (
            float(dynamic_thresh.mean() * 10_000.0) if dynamic_thresh is not None and len(dynamic_thresh) > 0 else 0.0
        )

        summary_rows.append({
            "Variant": var_label,
            "Allocation": alloc_rule,
            "k": k_val,
            "Mean Threshold (bps)": mean_thresh_bps,
            "Type": "Strategy",
            "Cum Net Return": m["cumulative_net_return"],
            "Ann Net Return": m["annualized_net_return"],
            "Ann Volatility": m["annualized_volatility"],
            "Sharpe Ratio": m["annualized_sharpe"],
            "Sortino Ratio": m["sortino_ratio"],
            "Max Drawdown": m["max_drawdown"],
            "Calmar Ratio": m["calmar_ratio"],
            "Daily Turnover": m["average_daily_turnover"],
            "Friction Drag": m["friction_drag"],
            "Win Rate": m["win_rate"],
            "Profit Factor": m["profit_factor"],
            "Trades": m["total_trades"],
        })

        if not benchmark_recorded and len(res.benchmark_metrics) > 0:
            bm = res.benchmark_metrics
            summary_rows.append({
                "Variant": "AAPL Buy & Hold",
                "Allocation": "long_only",
                "k": 0.0,
                "Mean Threshold (bps)": 0.0,
                "Type": "Benchmark",
                "Cum Net Return": bm["cumulative_net_return"],
                "Ann Net Return": bm["annualized_net_return"],
                "Ann Volatility": bm["annualized_volatility"],
                "Sharpe Ratio": bm["annualized_sharpe"],
                "Sortino Ratio": bm["sortino_ratio"],
                "Max Drawdown": bm["max_drawdown"],
                "Calmar Ratio": bm["calmar_ratio"],
                "Daily Turnover": bm["average_daily_turnover"],
                "Friction Drag": bm["friction_drag"],
                "Win Rate": bm["win_rate"],
                "Profit Factor": bm["profit_factor"],
                "Trades": bm["total_trades"],
            })
            benchmark_recorded = True

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df.sort_values("Sharpe Ratio", ascending=False, inplace=True)
        summary_df.reset_index(drop=True, inplace=True)

    return results, summary_df

