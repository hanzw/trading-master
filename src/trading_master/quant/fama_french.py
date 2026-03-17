"""Fama-French 5-Factor model: decompose returns into market, size, value, profitability, and investment factors.

Factors:
  - Mkt-RF: Market excess return (market return minus risk-free rate)
  - SMB:    Small Minus Big (size factor)
  - HML:    High Minus Low (value factor — book-to-market)
  - RMW:    Robust Minus Weak (profitability factor)
  - CMA:    Conservative Minus Aggressive (investment factor)

The model: R_i - R_f = alpha_i + beta_mkt*(Mkt-RF) + beta_smb*SMB + beta_hml*HML
                        + beta_rmw*RMW + beta_cma*CMA + epsilon_i
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

FACTOR_NAMES = ("Mkt-RF", "SMB", "HML", "RMW", "CMA")


@dataclass
class FF5Result:
    """Result of a Fama-French 5-factor regression."""

    ticker: str
    alpha: float
    betas: dict[str, float]  # factor name → beta
    r_squared: float
    residual_std: float
    t_stats: dict[str, float]  # factor name → t-statistic (incl. alpha)
    n_obs: int

    @property
    def alpha_annualized(self) -> float:
        """Annualized Jensen's alpha (assuming daily data, 252 trading days)."""
        return self.alpha * 252

    @property
    def significant_factors(self) -> list[str]:
        """Factors with |t-stat| > 2.0 (approx. 95% significance)."""
        return [
            name for name, t in self.t_stats.items()
            if abs(t) > 2.0 and name != "alpha"
        ]


def ols_regression(
    y: np.ndarray,
    X: np.ndarray,
    add_intercept: bool = True,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Ordinary least squares regression.

    Parameters
    ----------
    y : (n,) dependent variable
    X : (n, k) independent variables
    add_intercept : if True, prepend a column of ones

    Returns
    -------
    (coefficients, r_squared, t_statistics)
      coefficients: (k+1,) if intercept else (k,) — intercept first if added
      r_squared: coefficient of determination
      t_statistics: t-stats for each coefficient
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)

    if add_intercept:
        ones = np.ones((X.shape[0], 1))
        X = np.hstack([ones, X])

    n, k = X.shape

    if n <= k:
        raise ValueError(f"Not enough observations ({n}) for {k} parameters.")

    # OLS: beta = (X'X)^{-1} X'y
    XtX = X.T @ X
    Xty = X.T @ y
    betas = np.linalg.solve(XtX, Xty)

    # Residuals
    y_hat = X @ betas
    residuals = y - y_hat

    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Standard errors and t-statistics
    dof = n - k
    sigma2 = ss_res / dof
    cov_beta = sigma2 * np.linalg.inv(XtX)
    se = np.sqrt(np.maximum(np.diag(cov_beta), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(se > 0, betas / se, 0.0)

    return betas, r_squared, t_stats


def ff5_decompose(
    excess_returns: np.ndarray,
    factor_returns: np.ndarray,
    ticker: str = "UNKNOWN",
) -> FF5Result:
    """Run Fama-French 5-factor regression on a single asset.

    Parameters
    ----------
    excess_returns : (n,) asset excess returns (R_i - R_f)
    factor_returns : (n, 5) factor returns [Mkt-RF, SMB, HML, RMW, CMA]
    ticker : asset ticker for labeling

    Returns
    -------
    FF5Result with alpha, betas, R², residual std, and t-statistics.
    """
    excess_returns = np.asarray(excess_returns, dtype=float)
    factor_returns = np.asarray(factor_returns, dtype=float)

    if factor_returns.ndim == 1:
        raise ValueError("factor_returns must be 2-D with shape (n, 5).")
    if factor_returns.shape[1] != 5:
        raise ValueError(
            f"Expected 5 factors, got {factor_returns.shape[1]}."
        )
    if len(excess_returns) != factor_returns.shape[0]:
        raise ValueError("Mismatched lengths between returns and factors.")

    betas, r_sq, t_stats = ols_regression(excess_returns, factor_returns)

    # betas[0] = alpha (intercept), betas[1:6] = factor betas
    alpha = betas[0]
    factor_betas = dict(zip(FACTOR_NAMES, betas[1:]))
    t_stat_dict = {"alpha": t_stats[0]}
    t_stat_dict.update(dict(zip(FACTOR_NAMES, t_stats[1:])))

    residuals = excess_returns - (
        np.column_stack([np.ones(len(excess_returns)), factor_returns]) @ betas
    )
    residual_std = float(np.std(residuals, ddof=1))

    return FF5Result(
        ticker=ticker,
        alpha=float(alpha),
        betas=factor_betas,
        r_squared=float(r_sq),
        residual_std=residual_std,
        t_stats=t_stat_dict,
        n_obs=len(excess_returns),
    )


def ff5_decompose_portfolio(
    excess_returns_matrix: np.ndarray,
    factor_returns: np.ndarray,
    tickers: list[str],
) -> list[FF5Result]:
    """Run FF5 regression on multiple assets.

    Parameters
    ----------
    excess_returns_matrix : (n, m) matrix of excess returns for m assets
    factor_returns : (n, 5) factor returns
    tickers : list of m ticker names

    Returns
    -------
    List of FF5Result, one per asset.
    """
    excess_returns_matrix = np.asarray(excess_returns_matrix, dtype=float)
    if excess_returns_matrix.ndim == 1:
        excess_returns_matrix = excess_returns_matrix.reshape(-1, 1)

    m = excess_returns_matrix.shape[1]
    if len(tickers) != m:
        raise ValueError(f"Expected {m} tickers, got {len(tickers)}.")

    results = []
    for i in range(m):
        result = ff5_decompose(
            excess_returns_matrix[:, i],
            factor_returns,
            ticker=tickers[i],
        )
        results.append(result)

    return results


def generate_synthetic_factors(
    n_days: int = 252,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic FF5 factor data for testing/demo purposes.

    Returns
    -------
    (factor_returns, risk_free_rates)
      factor_returns: (n_days, 5) daily factor returns
      risk_free_rates: (n_days,) daily risk-free rate
    """
    rng = np.random.default_rng(seed)

    # Realistic daily factor statistics (annualized mean, annualized vol)
    factor_stats = {
        "Mkt-RF": (0.08, 0.16),   # Market premium ~8%, vol ~16%
        "SMB":    (0.02, 0.10),    # Size premium ~2%, vol ~10%
        "HML":    (0.03, 0.10),    # Value premium ~3%, vol ~10%
        "RMW":    (0.03, 0.08),    # Profitability premium ~3%, vol ~8%
        "CMA":    (0.02, 0.07),    # Investment premium ~2%, vol ~7%
    }

    factors = np.zeros((n_days, 5))
    for i, (_, (ann_mean, ann_vol)) in enumerate(factor_stats.items()):
        daily_mean = ann_mean / 252
        daily_vol = ann_vol / np.sqrt(252)
        factors[:, i] = rng.normal(daily_mean, daily_vol, n_days)

    # Risk-free rate (~4% annually)
    rf_daily = 0.04 / 252
    risk_free = np.full(n_days, rf_daily)

    return factors, risk_free


def attribute_returns(result: FF5Result) -> dict[str, float]:
    """Break down the expected return contribution from each factor.

    Uses the factor betas and long-run average factor premia
    to estimate how much each factor contributes to expected returns.

    Returns dict mapping factor name → annualized return contribution.
    """
    # Long-run annual factor premia (approximate historical averages)
    factor_premia = {
        "Mkt-RF": 0.08,
        "SMB":    0.02,
        "HML":    0.03,
        "RMW":    0.03,
        "CMA":    0.02,
    }

    attribution = {"alpha": result.alpha_annualized}
    for factor_name, beta in result.betas.items():
        premium = factor_premia.get(factor_name, 0.0)
        attribution[factor_name] = beta * premium

    attribution["total"] = sum(attribution.values())
    return attribution
