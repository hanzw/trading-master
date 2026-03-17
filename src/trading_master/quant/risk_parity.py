"""Risk Parity (Equal Risk Contribution) portfolio allocation.

Risk Parity allocates weights so that each asset contributes equally
to total portfolio risk. Unlike equal-weight or min-variance, this
approach ensures no single asset dominates the portfolio's risk profile.

The optimization minimizes the sum of squared differences between each
asset's risk contribution and the target (1/N of total risk). Supports
custom risk budgets for tilted allocations.

References:
  - Maillard, Roncalli, Teiletche (2010) — "The Properties of
    Equally Weighted Risk Contribution Portfolios"
  - Qian (2005) — "Risk Parity Portfolios"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class RiskParityResult:
    """Result of Risk Parity allocation."""

    weights: np.ndarray
    tickers: list[str]
    risk_contributions: np.ndarray       # each asset's absolute risk contribution
    risk_contribution_pct: np.ndarray    # each asset's % of total risk
    portfolio_volatility: float
    target_budget: np.ndarray            # the target risk budget (sums to 1)
    converged: bool

    @property
    def weight_dict(self) -> dict[str, float]:
        """Mapping from ticker to weight."""
        return dict(zip(self.tickers, self.weights.tolist()))

    @property
    def risk_dict(self) -> dict[str, float]:
        """Mapping from ticker to risk contribution percentage."""
        return dict(zip(self.tickers, self.risk_contribution_pct.tolist()))


def _risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Compute each asset's marginal risk contribution.

    RC_i = w_i * (Sigma @ w)_i / sigma_p
    """
    portfolio_var = float(weights @ cov @ weights)
    portfolio_vol = np.sqrt(max(portfolio_var, 1e-16))
    marginal = cov @ weights
    rc = weights * marginal / portfolio_vol
    return rc


def _risk_parity_objective(weights: np.ndarray, cov: np.ndarray, budget: np.ndarray) -> float:
    """Objective: minimize sum of squared deviations from target risk budget.

    We want RC_i / sigma_p = budget_i for all i.
    Minimize sum_i (RC_i / sigma_p - budget_i)^2
    """
    portfolio_var = float(weights @ cov @ weights)
    if portfolio_var < 1e-16:
        return 1e6
    portfolio_vol = np.sqrt(portfolio_var)
    marginal = cov @ weights
    rc = weights * marginal / portfolio_vol
    rc_pct = rc / portfolio_vol
    return float(np.sum((rc_pct - budget) ** 2))


def _risk_budget_objective(weights: np.ndarray, cov: np.ndarray, budget: np.ndarray) -> float:
    """Alternative objective using log-barrier for better convergence.

    Minimize sum_{i,j} (w_i*(Sigma@w)_i - w_j*(Sigma@w)_j)^2
    when budget is equal, or the budgeted variant.
    """
    marginal = cov @ weights
    rc = weights * marginal  # un-normalized risk contributions
    target_rc = budget * float(weights @ cov @ weights)
    return float(np.sum((rc - target_rc) ** 2))


def risk_parity(
    cov_matrix: np.ndarray,
    risk_budget: np.ndarray | None = None,
    tickers: list[str] | None = None,
    max_iter: int = 1000,
) -> RiskParityResult:
    """Compute Risk Parity portfolio weights.

    Parameters
    ----------
    cov_matrix : (n, n) covariance matrix
    risk_budget : target risk budget per asset (sums to 1).
                  If None, equal risk (1/n each).
    tickers : asset names (optional, defaults to indices)
    max_iter : max optimization iterations

    Returns
    -------
    RiskParityResult with weights, risk contributions, and convergence info.
    """
    cov = np.asarray(cov_matrix, dtype=float)
    n = cov.shape[0]

    if tickers is None:
        tickers = [str(i) for i in range(n)]

    if len(tickers) != n:
        raise ValueError(f"Expected {n} tickers, got {len(tickers)}.")

    # Default: equal risk budget
    if risk_budget is None:
        budget = np.ones(n) / n
    else:
        budget = np.asarray(risk_budget, dtype=float)
        if len(budget) != n:
            raise ValueError(f"Expected {n} risk budget entries, got {len(budget)}.")
        budget_sum = budget.sum()
        if abs(budget_sum - 1.0) > 1e-6:
            budget = budget / budget_sum  # normalize

    # Single asset edge case
    if n == 1:
        return RiskParityResult(
            weights=np.array([1.0]),
            tickers=tickers,
            risk_contributions=np.array([np.sqrt(float(cov[0, 0]))]),
            risk_contribution_pct=np.array([1.0]),
            portfolio_volatility=np.sqrt(float(cov[0, 0])),
            target_budget=budget,
            converged=True,
        )

    # Initial guess: inverse-volatility weights (good starting point)
    vols = np.sqrt(np.diag(cov))
    vols = np.maximum(vols, 1e-12)
    w0 = (1.0 / vols) / (1.0 / vols).sum()

    # Constraints: weights sum to 1
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Bounds: long-only, each weight in [1e-6, 1] to avoid zeros
    bounds = [(1e-6, 1.0)] * n

    # Optimize using the budgeted objective
    result = minimize(
        _risk_budget_objective,
        w0,
        args=(cov, budget),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": max_iter, "ftol": 1e-15},
    )

    weights = result.x
    weights = weights / weights.sum()  # ensure exact normalization

    # Compute final risk contributions
    portfolio_var = float(weights @ cov @ weights)
    portfolio_vol = np.sqrt(max(portfolio_var, 1e-16))
    rc = _risk_contributions(weights, cov)
    rc_total = rc.sum()
    rc_pct = rc / rc_total if rc_total > 1e-16 else np.ones(n) / n

    return RiskParityResult(
        weights=weights,
        tickers=tickers,
        risk_contributions=rc,
        risk_contribution_pct=rc_pct,
        portfolio_volatility=portfolio_vol,
        target_budget=budget,
        converged=result.success,
    )


def inverse_volatility(
    cov_matrix: np.ndarray,
    tickers: list[str] | None = None,
) -> RiskParityResult:
    """Simple inverse-volatility weighting (analytical Risk Parity proxy).

    This is a fast, closed-form approximation that ignores correlations.
    Useful as a baseline or when the covariance matrix is unreliable.

    Parameters
    ----------
    cov_matrix : (n, n) covariance matrix
    tickers : asset names

    Returns
    -------
    RiskParityResult with inverse-vol weights.
    """
    cov = np.asarray(cov_matrix, dtype=float)
    n = cov.shape[0]

    if tickers is None:
        tickers = [str(i) for i in range(n)]

    if len(tickers) != n:
        raise ValueError(f"Expected {n} tickers, got {len(tickers)}.")

    vols = np.sqrt(np.diag(cov))
    vols = np.maximum(vols, 1e-12)
    weights = (1.0 / vols) / (1.0 / vols).sum()

    # Compute risk contributions for the result
    portfolio_var = float(weights @ cov @ weights)
    portfolio_vol = np.sqrt(max(portfolio_var, 1e-16))
    rc = _risk_contributions(weights, cov)
    rc_total = rc.sum()
    rc_pct = rc / rc_total if rc_total > 1e-16 else np.ones(n) / n

    return RiskParityResult(
        weights=weights,
        tickers=tickers,
        risk_contributions=rc,
        risk_contribution_pct=rc_pct,
        portfolio_volatility=portfolio_vol,
        target_budget=np.ones(n) / n,
        converged=True,
    )
