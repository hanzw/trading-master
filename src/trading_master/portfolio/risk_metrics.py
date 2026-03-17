"""Portfolio risk metrics: VaR, drawdown, risk-adjusted return ratios."""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# ── Value at Risk ──────────────────────────────────────────────────────

def parametric_var(
    weights: np.ndarray,
    returns: np.ndarray,
    confidence: float = 0.95,
    portfolio_value: float = 1.0,
) -> float:
    """Parametric (variance-covariance) VaR.

    Parameters
    ----------
    weights : (N,) array of portfolio weights.
    returns : (T, N) daily returns matrix.
    confidence : confidence level (e.g. 0.95 or 0.99).
    portfolio_value : dollar value of the portfolio.

    Returns
    -------
    Positive dollar loss at the given confidence level.
    """
    weights = np.asarray(weights, dtype=float)
    returns = np.asarray(returns, dtype=float)

    if weights.size == 0 or returns.size == 0:
        return 0.0

    # Handle single-asset case
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)
    if weights.ndim == 0:
        weights = weights.reshape(1)

    mu = returns.mean(axis=0)
    mu_p = float(weights @ mu)

    cov = np.cov(returns, rowvar=False)
    # np.cov returns scalar for single-asset
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])

    sigma_p = float(np.sqrt(weights @ cov @ weights))

    if sigma_p == 0.0:
        return 0.0

    z = stats.norm.ppf(confidence)
    var = -(mu_p - z * sigma_p) * portfolio_value
    return max(var, 0.0)


def historical_var(
    weights: np.ndarray,
    returns: np.ndarray,
    confidence: float = 0.95,
    portfolio_value: float = 1.0,
) -> float:
    """Historical simulation VaR.

    Uses the empirical alpha-percentile of the portfolio return distribution.
    """
    weights = np.asarray(weights, dtype=float)
    returns = np.asarray(returns, dtype=float)

    if weights.size == 0 or returns.size == 0:
        return 0.0

    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)
    if weights.ndim == 0:
        weights = weights.reshape(1)

    port_returns = returns @ weights
    alpha = 1.0 - confidence
    var_return = np.percentile(port_returns, alpha * 100)
    var = -var_return * portfolio_value
    return max(var, 0.0)


def cvar(
    weights: np.ndarray,
    returns: np.ndarray,
    confidence: float = 0.95,
    portfolio_value: float = 1.0,
) -> float:
    """Conditional VaR (Expected Shortfall).

    Mean loss in the tail beyond the VaR threshold.
    """
    weights = np.asarray(weights, dtype=float)
    returns = np.asarray(returns, dtype=float)

    if weights.size == 0 or returns.size == 0:
        return 0.0

    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)
    if weights.ndim == 0:
        weights = weights.reshape(1)

    port_returns = returns @ weights
    alpha = 1.0 - confidence
    threshold = np.percentile(port_returns, alpha * 100)
    tail = port_returns[port_returns <= threshold]

    if tail.size == 0:
        return 0.0

    cvar_val = -float(np.mean(tail)) * portfolio_value
    return max(cvar_val, 0.0)


# ── Drawdown ───────────────────────────────────────────────────────────

def max_drawdown(equity_curve: np.ndarray) -> dict:
    """Compute maximum drawdown from an equity curve.

    Parameters
    ----------
    equity_curve : 1-D array of portfolio values over time.

    Returns
    -------
    dict with max_dd (0-1), peak_idx, trough_idx, recovery_idx, dd_duration.
    """
    equity = np.asarray(equity_curve, dtype=float)

    result = {
        "max_dd": 0.0,
        "peak_idx": 0,
        "trough_idx": 0,
        "recovery_idx": None,
        "dd_duration": 0,
    }

    if equity.size < 2:
        return result

    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / np.where(running_max == 0, 1.0, running_max)

    trough_idx = int(np.argmin(drawdowns))
    max_dd = -float(drawdowns[trough_idx])

    if max_dd == 0.0:
        return result

    peak_idx = int(np.argmax(equity[:trough_idx + 1]))

    # Find recovery: first time equity >= peak value after trough
    peak_value = equity[peak_idx]
    recovery_idx = None
    for i in range(trough_idx + 1, len(equity)):
        if equity[i] >= peak_value:
            recovery_idx = int(i)
            break

    dd_duration = (recovery_idx - peak_idx) if recovery_idx is not None else (len(equity) - 1 - peak_idx)

    return {
        "max_dd": max_dd,
        "peak_idx": peak_idx,
        "trough_idx": trough_idx,
        "recovery_idx": recovery_idx,
        "dd_duration": dd_duration,
    }


# ── Risk-adjusted return ratios ───────────────────────────────────────

def sharpe_ratio(returns: np.ndarray, rf: float = 0.0) -> float:
    """Annualized Sharpe ratio.

    ``Sharpe = mean(excess) / std(excess) * sqrt(252)``
    """
    returns = np.asarray(returns, dtype=float).flatten()

    if returns.size < 2:
        return 0.0

    excess = returns - rf / TRADING_DAYS
    std = float(np.std(excess, ddof=1))

    if std < 1e-14:
        return 0.0

    return float(np.mean(excess) / std * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: np.ndarray, rf: float = 0.0) -> float:
    """Annualized Sortino ratio using downside deviation."""
    returns = np.asarray(returns, dtype=float).flatten()

    if returns.size < 2:
        return 0.0

    excess = returns - rf / TRADING_DAYS

    # Downside deviation uses ALL observations: negative excess returns
    # contribute their squared value, non-negative returns contribute zero.
    downside_sq = np.minimum(excess, 0.0) ** 2

    if not np.any(excess < 0):
        return 0.0  # no downside => undefined, return 0

    downside_std = float(np.sqrt(np.mean(downside_sq)))

    if downside_std == 0.0:
        return 0.0

    return float(np.mean(excess) / downside_std * np.sqrt(TRADING_DAYS))


def calmar_ratio(equity_curve: np.ndarray) -> float:
    """Calmar ratio = annualized return / max drawdown."""
    equity = np.asarray(equity_curve, dtype=float)

    if equity.size < 2 or equity[0] <= 0:
        return 0.0

    dd_info = max_drawdown(equity)
    mdd = dd_info["max_dd"]

    if mdd == 0.0:
        return 0.0

    total_return = equity[-1] / equity[0] - 1.0
    n_days = len(equity)
    annualized = (1.0 + total_return) ** (TRADING_DAYS / n_days) - 1.0

    return annualized / mdd


# ── Beta ──────────────────────────────────────────────────────────────

def compute_beta(asset_returns: np.ndarray, market_returns: np.ndarray) -> float:
    """OLS beta: cov(asset, market) / var(market)."""
    asset = np.asarray(asset_returns, dtype=float).flatten()
    market = np.asarray(market_returns, dtype=float).flatten()

    min_len = min(len(asset), len(market))
    if min_len < 2:
        return 0.0

    asset = asset[:min_len]
    market = market[:min_len]

    var_m = float(np.var(market, ddof=1))
    if var_m == 0.0:
        return 0.0

    cov_am = float(np.cov(asset, market, ddof=1)[0, 1])
    return cov_am / var_m


# ── Dashboard ─────────────────────────────────────────────────────────

def portfolio_risk_dashboard(
    positions_returns: np.ndarray,
    weights: np.ndarray,
    benchmark_returns: np.ndarray | None = None,
    portfolio_value: float = 1.0,
) -> dict:
    """Master function returning a dict with all risk metrics.

    Parameters
    ----------
    positions_returns : (T, N) daily returns for N positions.
    weights : (N,) portfolio weights.
    benchmark_returns : (T,) optional benchmark/market returns.
    portfolio_value : total dollar value of the portfolio.

    Returns
    -------
    Dict with sharpe, sortino, var_95, var_99, cvar_95, max_dd, beta, calmar.
    """
    weights = np.asarray(weights, dtype=float)
    positions_returns = np.asarray(positions_returns, dtype=float)

    if weights.size == 0 or positions_returns.size == 0:
        return {
            "sharpe": 0.0,
            "sortino": 0.0,
            "var_95": 0.0,
            "var_99": 0.0,
            "cvar_95": 0.0,
            "max_dd": 0.0,
            "beta": 0.0,
            "calmar": 0.0,
        }

    if positions_returns.ndim == 1:
        positions_returns = positions_returns.reshape(-1, 1)

    port_returns = positions_returns @ weights

    # Build equity curve from returns (starting at portfolio_value)
    equity_curve = portfolio_value * np.concatenate([[1.0], np.cumprod(1.0 + port_returns)])

    beta = 0.0
    if benchmark_returns is not None:
        bench = np.asarray(benchmark_returns, dtype=float).flatten()
        if bench.size >= 2:
            beta = compute_beta(port_returns, bench)

    return {
        "sharpe": sharpe_ratio(port_returns),
        "sortino": sortino_ratio(port_returns),
        "var_95": parametric_var(weights, positions_returns, 0.95, portfolio_value),
        "var_99": parametric_var(weights, positions_returns, 0.99, portfolio_value),
        "cvar_95": cvar(weights, positions_returns, 0.95, portfolio_value),
        "max_dd": max_drawdown(equity_curve)["max_dd"],
        "beta": beta,
        "calmar": calmar_ratio(equity_curve),
    }
