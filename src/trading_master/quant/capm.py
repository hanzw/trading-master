"""CAPM (Capital Asset Pricing Model) + Jensen's Alpha.

The CAPM: E[R_i] = R_f + beta_i * (E[R_m] - R_f)

Jensen's Alpha: alpha_i = R_i - [R_f + beta_i * (R_m - R_f)]
  - Positive alpha → outperformance vs. CAPM prediction
  - Negative alpha → underperformance

Additional metrics:
  - Treynor Ratio: (R_p - R_f) / beta_p
  - Sharpe Ratio: (R_p - R_f) / sigma_p
  - Information Ratio: alpha / tracking_error
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CAPMResult:
    """Result of a CAPM regression for a single asset."""

    ticker: str
    alpha: float           # daily Jensen's alpha (intercept)
    beta: float            # market beta
    r_squared: float
    alpha_t_stat: float
    beta_t_stat: float
    n_obs: int
    residual_std: float    # daily tracking error

    @property
    def alpha_annualized(self) -> float:
        """Annualized alpha (assuming 252 trading days)."""
        return self.alpha * 252

    @property
    def treynor_ratio(self) -> float | None:
        """Treynor ratio requires knowing average excess return; not available from regression alone."""
        return None

    @property
    def information_ratio(self) -> float:
        """Alpha / tracking error (annualized)."""
        ann_te = self.residual_std * np.sqrt(252)
        if ann_te < 1e-12:
            return 0.0
        return self.alpha_annualized / ann_te

    @property
    def is_alpha_significant(self) -> bool:
        """Whether alpha is statistically significant at 95% level."""
        return abs(self.alpha_t_stat) > 1.96


def capm_regression(
    asset_excess_returns: np.ndarray,
    market_excess_returns: np.ndarray,
    ticker: str = "UNKNOWN",
) -> CAPMResult:
    """Run CAPM regression: R_i - R_f = alpha + beta * (R_m - R_f) + epsilon.

    Parameters
    ----------
    asset_excess_returns : (n,) daily excess returns of the asset
    market_excess_returns : (n,) daily excess returns of the market
    ticker : asset identifier

    Returns
    -------
    CAPMResult with alpha, beta, R², and t-statistics.
    """
    y = np.asarray(asset_excess_returns, dtype=float)
    x = np.asarray(market_excess_returns, dtype=float)

    if len(y) != len(x):
        raise ValueError("Asset and market returns must have the same length.")
    n = len(y)
    if n < 3:
        raise ValueError(f"Need at least 3 observations, got {n}.")

    # OLS: y = alpha + beta * x
    X = np.column_stack([np.ones(n), x])
    XtX = X.T @ X
    Xty = X.T @ y
    betas = np.linalg.solve(XtX, Xty)

    alpha, beta = betas[0], betas[1]

    # Residuals
    y_hat = X @ betas
    residuals = y - y_hat

    # R²
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Standard errors
    dof = n - 2
    sigma2 = ss_res / dof
    cov_beta = sigma2 * np.linalg.inv(XtX)
    se = np.sqrt(np.diag(cov_beta))

    alpha_t = alpha / se[0] if se[0] > 0 else 0.0
    beta_t = beta / se[1] if se[1] > 0 else 0.0

    return CAPMResult(
        ticker=ticker,
        alpha=float(alpha),
        beta=float(beta),
        r_squared=float(r_sq),
        alpha_t_stat=float(alpha_t),
        beta_t_stat=float(beta_t),
        n_obs=n,
        residual_std=float(np.std(residuals, ddof=1)),
    )


def capm_expected_return(
    beta: float,
    risk_free_rate: float = 0.04,
    market_premium: float = 0.08,
) -> float:
    """Compute CAPM expected annual return: R_f + beta * (E[R_m] - R_f).

    Parameters
    ----------
    beta : asset's market beta
    risk_free_rate : annualized risk-free rate (default 4%)
    market_premium : annualized equity risk premium (default 8%)
    """
    return risk_free_rate + beta * market_premium


def capm_portfolio(
    asset_excess_returns: np.ndarray,
    market_excess_returns: np.ndarray,
    tickers: list[str],
) -> list[CAPMResult]:
    """Run CAPM regressions for multiple assets.

    Parameters
    ----------
    asset_excess_returns : (n, m) excess returns for m assets
    market_excess_returns : (n,) market excess returns
    tickers : list of m ticker names

    Returns
    -------
    List of CAPMResult, one per asset.
    """
    returns = np.asarray(asset_excess_returns, dtype=float)
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)

    m = returns.shape[1]
    if len(tickers) != m:
        raise ValueError(f"Expected {m} tickers, got {len(tickers)}.")

    return [
        capm_regression(returns[:, i], market_excess_returns, ticker=tickers[i])
        for i in range(m)
    ]


def security_market_line(
    results: list[CAPMResult],
    risk_free_rate: float = 0.04,
    market_premium: float = 0.08,
) -> list[dict]:
    """Compute Security Market Line (SML) analysis.

    For each asset, compare actual return vs. CAPM-predicted return.
    Assets above the SML have positive alpha (undervalued).
    Assets below the SML have negative alpha (overvalued).

    Returns list of dicts with ticker, beta, expected_return, alpha_annualized,
    and valuation assessment.
    """
    sml = []
    for r in results:
        expected = capm_expected_return(r.beta, risk_free_rate, market_premium)
        actual = expected + r.alpha_annualized  # CAPM expected + alpha
        if r.alpha_annualized > 0.01:
            assessment = "undervalued"
        elif r.alpha_annualized < -0.01:
            assessment = "overvalued"
        else:
            assessment = "fairly_valued"

        sml.append({
            "ticker": r.ticker,
            "beta": round(r.beta, 4),
            "expected_return": round(expected, 4),
            "actual_return": round(actual, 4),
            "alpha_annualized": round(r.alpha_annualized, 4),
            "assessment": assessment,
            "alpha_significant": r.is_alpha_significant,
        })

    return sml
