"""Walk-forward out-of-sample validation of the sizing pipeline.

Tests whether the 7-layer sizing approach (volatility, Hurst, ATR scaling,
correlation adjustment, Kelly criterion, regime filter, position cap)
produces better risk-adjusted returns than naive equal-weight (1/N) sizing.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import yfinance as yf

from .sizing import compute_position_size

logger = logging.getLogger(__name__)


def _fetch_history(tickers: list[str], days: int) -> dict[str, np.ndarray]:
    """Fetch daily close prices for *tickers* over *days* calendar days.

    Returns {ticker: np.ndarray of close prices} for tickers with enough data.
    """
    end = datetime.now()
    start = end - timedelta(days=int(days * 1.5))  # extra buffer for weekends

    result: dict[str, np.ndarray] = {}
    try:
        data = yf.download(
            " ".join(tickers),
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
        )
        if data.empty:
            return result

        close = data["Close"] if "Close" in data.columns else data
        for t in tickers:
            if t in close.columns:
                series = close[t].dropna().values
                if len(series) >= days:
                    result[t] = series[-days:]
            elif len(tickers) == 1:
                series = close.dropna().values
                if len(series) >= days:
                    result[t] = series[-days:]
    except Exception:
        logger.warning("yfinance download failed for walk-forward", exc_info=True)

    return result


def _compute_hurst(prices: np.ndarray) -> float:
    """Compute Hurst exponent via rescaled range (R/S) method."""
    if len(prices) < 20:
        return 0.5

    returns = np.diff(np.log(prices))
    n = len(returns)
    if n < 20:
        return 0.5

    max_k = min(n // 2, 100)
    sizes = []
    rs_values = []

    for size in [int(n / k) for k in range(2, min(max_k, n))]:
        if size < 10:
            continue
        n_chunks = n // size
        if n_chunks < 1:
            continue

        rs_list = []
        for i in range(n_chunks):
            chunk = returns[i * size: (i + 1) * size]
            mean = np.mean(chunk)
            deviations = np.cumsum(chunk - mean)
            r = np.max(deviations) - np.min(deviations)
            s = np.std(chunk, ddof=1)
            if s > 0:
                rs_list.append(r / s)

        if rs_list:
            sizes.append(size)
            rs_values.append(np.mean(rs_list))

    if len(sizes) < 3:
        return 0.5

    log_sizes = np.log(sizes)
    log_rs = np.log(rs_values)
    coeffs = np.polyfit(log_sizes, log_rs, 1)
    return float(np.clip(coeffs[0], 0.01, 0.99))


def _compute_atr(prices: np.ndarray, period: int = 14) -> float:
    """Compute ATR from close prices (approximation using close-to-close)."""
    if len(prices) < period + 1:
        return 0.0
    true_ranges = np.abs(np.diff(prices))
    atr = np.mean(true_ranges[-period:])
    return float(atr)


def _compute_avg_correlation(returns_matrix: np.ndarray) -> float:
    """Average absolute pairwise correlation across columns."""
    n_assets = returns_matrix.shape[1]
    if n_assets < 2:
        return 0.0
    corr = np.corrcoef(returns_matrix, rowvar=False)
    mask = ~np.eye(n_assets, dtype=bool)
    return float(np.mean(np.abs(corr[mask])))


def walk_forward_test(
    tickers: list[str],
    train_days: int = 252,
    test_days: int = 63,
    n_windows: int = 4,
    portfolio_value: float = 100_000.0,
    risk_per_trade_pct: float = 1.0,
    fetch_fn=None,
) -> dict:
    """Walk-forward out-of-sample validation of the sizing pipeline.

    For each window:
    1. Train: compute Hurst, ATR, correlations on train_days of data
    2. Size: use compute_position_size with trained parameters
    3. Test: measure actual return over test_days with the computed size
    4. Baseline: compare against equal-weight sizing (1/N)

    Parameters
    ----------
    tickers : list of ticker symbols
    train_days : training window in trading days (default 252 = 1 year)
    test_days : test window in trading days (default 63 = 1 quarter)
    n_windows : number of walk-forward windows
    portfolio_value : starting portfolio value for sizing
    risk_per_trade_pct : risk budget per trade as % of portfolio
    fetch_fn : optional override for data fetching (for testing)

    Returns
    -------
    dict with 'windows' list and 'aggregate' summary.
    """
    total_days = train_days + test_days * n_windows
    fetcher = fetch_fn or _fetch_history

    # Fetch all data at once
    all_prices = fetcher(tickers, total_days)

    valid_tickers = [t for t in tickers if t in all_prices and len(all_prices[t]) >= total_days]
    if not valid_tickers:
        return {
            "windows": [],
            "aggregate": {
                "avg_strategy_return": 0.0,
                "avg_baseline_return": 0.0,
                "strategy_wins": 0,
                "information_ratio": 0.0,
                "n_windows": 0,
                "error": "Insufficient data for walk-forward test",
            },
        }

    n_assets = len(valid_tickers)
    windows = []

    for w in range(n_windows):
        train_start = w * test_days
        train_end = train_start + train_days
        test_start = train_end
        test_end = test_start + test_days

        # ---- Train phase: compute parameters ----
        train_prices = {t: all_prices[t][train_start:train_end] for t in valid_tickers}
        train_returns = {
            t: np.diff(np.log(train_prices[t])) for t in valid_tickers
        }

        # Compute Hurst and ATR per ticker
        hursts = {t: _compute_hurst(train_prices[t]) for t in valid_tickers}
        atrs = {t: _compute_atr(train_prices[t]) for t in valid_tickers}

        # Correlation across assets
        returns_matrix = np.column_stack([train_returns[t] for t in valid_tickers])
        avg_corr = _compute_avg_correlation(returns_matrix)

        # ---- Size phase: use the sizing pipeline ----
        strategy_sizes = {}
        baseline_size_per_asset = portfolio_value / n_assets  # equal-weight

        for t in valid_tickers:
            price = float(train_prices[t][-1])
            atr = atrs[t] if atrs[t] > 0 else price * 0.02
            result = compute_position_size(
                price=price,
                atr_14=atr,
                portfolio_value=portfolio_value,
                max_position_pct=100.0 / n_assets * 2,  # 2x equal weight cap
                existing_correlation=avg_corr,
                regime=None,
                hurst=hursts[t],
            )
            strategy_sizes[t] = result["shares"] * price  # dollar allocation

        # Normalize strategy sizes to sum to portfolio_value
        total_allocated = sum(strategy_sizes.values())
        if total_allocated > 0:
            scale = portfolio_value / total_allocated
            strategy_sizes = {t: v * scale for t, v in strategy_sizes.items()}

        # ---- Test phase: compute OOS returns ----
        test_prices = {t: all_prices[t][test_start:test_end] for t in valid_tickers}

        strategy_return = 0.0
        baseline_return = 0.0

        for t in valid_tickers:
            if len(test_prices[t]) < 2:
                continue
            ticker_return = (test_prices[t][-1] / test_prices[t][0]) - 1.0

            # Strategy: weighted by sizing pipeline
            strategy_weight = strategy_sizes.get(t, 0) / portfolio_value if portfolio_value > 0 else 0
            strategy_return += strategy_weight * ticker_return

            # Baseline: equal weight
            baseline_weight = 1.0 / n_assets
            baseline_return += baseline_weight * ticker_return

        # Compute Sharpe-like metric (annualized, using daily returns in test period)
        test_returns_matrix = np.column_stack([
            np.diff(np.log(test_prices[t])) for t in valid_tickers
            if len(test_prices[t]) > 1
        ])

        if test_returns_matrix.size > 0:
            # Strategy weighted daily returns
            strat_weights = np.array([
                strategy_sizes.get(t, 0) / portfolio_value for t in valid_tickers
            ])
            base_weights = np.full(n_assets, 1.0 / n_assets)

            strat_daily = test_returns_matrix @ strat_weights
            base_daily = test_returns_matrix @ base_weights

            strat_sharpe = _annualized_sharpe(strat_daily)
            base_sharpe = _annualized_sharpe(base_daily)
        else:
            strat_sharpe = 0.0
            base_sharpe = 0.0

        windows.append({
            "window": w + 1,
            "train_period": f"day {train_start}-{train_end}",
            "test_period": f"day {test_start}-{test_end}",
            "strategy_return": round(strategy_return * 100, 4),
            "baseline_return": round(baseline_return * 100, 4),
            "strategy_sharpe": round(strat_sharpe, 4),
            "baseline_sharpe": round(base_sharpe, 4),
            "excess_return": round((strategy_return - baseline_return) * 100, 4),
            "hursts": {t: round(hursts[t], 3) for t in valid_tickers},
            "avg_correlation": round(avg_corr, 3),
        })

    # ---- Aggregate ----
    if windows:
        strat_returns = [w["strategy_return"] for w in windows]
        base_returns = [w["baseline_return"] for w in windows]
        excess = [w["excess_return"] for w in windows]

        strategy_wins = sum(1 for e in excess if e > 0)

        ir_std = float(np.std(excess, ddof=1)) if len(excess) > 1 else 1.0
        information_ratio = float(np.mean(excess)) / ir_std if ir_std > 0 else 0.0

        aggregate = {
            "avg_strategy_return": round(float(np.mean(strat_returns)), 4),
            "avg_baseline_return": round(float(np.mean(base_returns)), 4),
            "strategy_wins": strategy_wins,
            "n_windows": n_windows,
            "information_ratio": round(information_ratio, 4),
        }
    else:
        aggregate = {
            "avg_strategy_return": 0.0,
            "avg_baseline_return": 0.0,
            "strategy_wins": 0,
            "n_windows": 0,
            "information_ratio": 0.0,
        }

    return {"windows": windows, "aggregate": aggregate}


def _annualized_sharpe(daily_returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Annualized Sharpe ratio from daily returns."""
    if len(daily_returns) < 2:
        return 0.0
    excess = daily_returns - risk_free / 252
    std = float(np.std(excess, ddof=1))
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(252))
