"""Markowitz Mean-Variance Optimization — efficient frontier computation.

Given expected returns and a covariance matrix, finds optimal portfolios
that maximize return for a given risk level (or minimize risk for a given return).

Key outputs:
  - Minimum variance portfolio
  - Maximum Sharpe ratio (tangency) portfolio
  - Efficient frontier curve
  - Optimal weights for a target return or risk level
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class PortfolioPoint:
    """A single point on the efficient frontier."""

    weights: np.ndarray
    expected_return: float   # annualized
    volatility: float        # annualized
    sharpe_ratio: float


def _portfolio_stats(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.04,
) -> tuple[float, float, float]:
    """Compute annualized return, volatility, and Sharpe ratio."""
    w = np.asarray(weights)
    ret = float(w @ expected_returns)
    vol = float(np.sqrt(w @ cov_matrix @ w))
    sharpe = (ret - risk_free_rate) / vol if vol > 1e-12 else 0.0
    return ret, vol, sharpe


def minimum_variance_portfolio(
    cov_matrix: np.ndarray,
    expected_returns: np.ndarray | None = None,
    long_only: bool = True,
    risk_free_rate: float = 0.04,
) -> PortfolioPoint:
    """Find the minimum variance portfolio.

    Parameters
    ----------
    cov_matrix : (n, n) annualized covariance matrix
    expected_returns : (n,) annualized expected returns (for Sharpe computation)
    long_only : if True, constrain weights >= 0
    risk_free_rate : for Sharpe ratio calculation
    """
    cov = np.asarray(cov_matrix, dtype=float)
    n = cov.shape[0]

    if expected_returns is None:
        expected_returns = np.zeros(n)
    mu = np.asarray(expected_returns, dtype=float)

    def objective(w):
        return w @ cov @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n if long_only else [(None, None)] * n
    x0 = np.full(n, 1.0 / n)

    result = minimize(
        objective, x0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"ftol": 1e-12},
    )

    w = result.x
    ret, vol, sharpe = _portfolio_stats(w, mu, cov, risk_free_rate)

    return PortfolioPoint(
        weights=w, expected_return=ret, volatility=vol, sharpe_ratio=sharpe,
    )


def max_sharpe_portfolio(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.04,
    long_only: bool = True,
) -> PortfolioPoint:
    """Find the tangency portfolio (maximum Sharpe ratio).

    Parameters
    ----------
    expected_returns : (n,) annualized expected returns
    cov_matrix : (n, n) annualized covariance matrix
    risk_free_rate : annualized risk-free rate
    long_only : if True, constrain weights >= 0
    """
    cov = np.asarray(cov_matrix, dtype=float)
    mu = np.asarray(expected_returns, dtype=float)
    n = len(mu)

    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        if vol < 1e-12:
            return 0.0
        return -(ret - risk_free_rate) / vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n if long_only else [(None, None)] * n
    x0 = np.full(n, 1.0 / n)

    result = minimize(
        neg_sharpe, x0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"ftol": 1e-12},
    )

    w = result.x
    ret, vol, sharpe = _portfolio_stats(w, mu, cov, risk_free_rate)

    return PortfolioPoint(
        weights=w, expected_return=ret, volatility=vol, sharpe_ratio=sharpe,
    )


def target_return_portfolio(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    target_return: float,
    risk_free_rate: float = 0.04,
    long_only: bool = True,
) -> PortfolioPoint:
    """Find the minimum-variance portfolio achieving a target return.

    Parameters
    ----------
    expected_returns : (n,) annualized expected returns
    cov_matrix : (n, n) annualized covariance matrix
    target_return : desired annualized return
    risk_free_rate : for Sharpe ratio
    long_only : if True, constrain weights >= 0
    """
    cov = np.asarray(cov_matrix, dtype=float)
    mu = np.asarray(expected_returns, dtype=float)
    n = len(mu)

    def objective(w):
        return w @ cov @ w

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: w @ mu - target_return},
    ]
    bounds = [(0.0, 1.0)] * n if long_only else [(None, None)] * n
    x0 = np.full(n, 1.0 / n)

    result = minimize(
        objective, x0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"ftol": 1e-12},
    )

    w = result.x
    ret, vol, sharpe = _portfolio_stats(w, mu, cov, risk_free_rate)

    return PortfolioPoint(
        weights=w, expected_return=ret, volatility=vol, sharpe_ratio=sharpe,
    )


def efficient_frontier(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    n_points: int = 50,
    risk_free_rate: float = 0.04,
    long_only: bool = True,
) -> list[PortfolioPoint]:
    """Compute the efficient frontier.

    Generates n_points portfolios ranging from minimum variance
    to maximum return.

    Parameters
    ----------
    expected_returns : (n,) annualized expected returns
    cov_matrix : (n, n) annualized covariance matrix
    n_points : number of points on the frontier
    risk_free_rate : for Sharpe ratio
    long_only : if True, constrain weights >= 0

    Returns
    -------
    List of PortfolioPoint sorted by volatility.
    """
    mu = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)

    # Find min and max achievable returns
    min_var = minimum_variance_portfolio(cov, mu, long_only, risk_free_rate)
    min_ret = min_var.expected_return

    if long_only:
        max_ret = float(np.max(mu))
    else:
        max_ret = float(np.max(mu)) * 1.5  # Allow some overshoot for short

    if max_ret <= min_ret:
        return [min_var]

    target_returns = np.linspace(min_ret, max_ret, n_points)
    frontier = []

    for target in target_returns:
        try:
            point = target_return_portfolio(
                mu, cov, target, risk_free_rate, long_only,
            )
            frontier.append(point)
        except Exception:
            continue

    # Sort by volatility
    frontier.sort(key=lambda p: p.volatility)
    return frontier
