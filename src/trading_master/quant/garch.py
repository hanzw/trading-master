"""GARCH(1,1) volatility model: time-varying conditional variance estimation.

The GARCH(1,1) model:
  sigma²_t = omega + alpha * epsilon²_{t-1} + beta * sigma²_{t-1}

Where:
  - omega > 0: long-run variance weight
  - alpha >= 0: shock coefficient (ARCH effect)
  - beta >= 0: persistence coefficient (GARCH effect)
  - alpha + beta < 1: stationarity condition
  - Long-run variance = omega / (1 - alpha - beta)

This implementation uses maximum likelihood estimation (MLE) with
normal innovations, and provides volatility forecasting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class GARCHResult:
    """Result of GARCH(1,1) estimation."""

    omega: float
    alpha: float
    beta: float
    log_likelihood: float
    conditional_volatility: np.ndarray  # (n,) annualized vol series
    n_obs: int
    converged: bool

    @property
    def persistence(self) -> float:
        """alpha + beta — persistence of volatility shocks."""
        return self.alpha + self.beta

    @property
    def long_run_variance(self) -> float:
        """Unconditional (long-run) daily variance: omega / (1 - alpha - beta)."""
        denom = 1.0 - self.alpha - self.beta
        if denom <= 0:
            return float("inf")
        return self.omega / denom

    @property
    def long_run_volatility(self) -> float:
        """Annualized long-run volatility."""
        return np.sqrt(self.long_run_variance * 252)

    @property
    def half_life(self) -> float:
        """Half-life of volatility shocks in days: ln(2) / -ln(alpha+beta)."""
        p = self.persistence
        if p <= 0 or p >= 1:
            return float("inf")
        return np.log(2) / (-np.log(p))


def _garch_log_likelihood(
    params: np.ndarray,
    returns: np.ndarray,
) -> float:
    """Negative log-likelihood for GARCH(1,1) with normal innovations.

    Parameters
    ----------
    params : [omega, alpha, beta]
    returns : (n,) return series (demeaned)

    Returns
    -------
    Negative log-likelihood (for minimization).
    """
    omega, alpha, beta = params
    n = len(returns)

    # Initialize conditional variance with sample variance
    sigma2 = np.empty(n)
    sigma2[0] = np.var(returns)

    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        # Floor to avoid numerical issues
        if sigma2[t] < 1e-12:
            sigma2[t] = 1e-12

    # Log-likelihood: sum of -0.5 * (log(2*pi) + log(sigma2_t) + r²_t/sigma2_t)
    ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + returns ** 2 / sigma2)

    return -ll  # Minimize negative LL


def fit_garch(
    returns: np.ndarray,
    initial_params: tuple[float, float, float] | None = None,
) -> GARCHResult:
    """Fit a GARCH(1,1) model via MLE.

    Parameters
    ----------
    returns : (n,) daily return series
    initial_params : optional (omega, alpha, beta) starting values

    Returns
    -------
    GARCHResult with estimated parameters and conditional volatility.
    """
    returns = np.asarray(returns, dtype=float)
    # Demean returns
    returns = returns - returns.mean()
    n = len(returns)

    if n < 10:
        raise ValueError(f"Need at least 10 observations, got {n}.")

    sample_var = np.var(returns)

    if initial_params is None:
        # Sensible starting values
        omega_0 = sample_var * 0.05
        alpha_0 = 0.08
        beta_0 = 0.85
    else:
        omega_0, alpha_0, beta_0 = initial_params

    # Try multiple starting points and pick the best
    starts = [
        np.array([omega_0, alpha_0, beta_0]),
        np.array([sample_var * 0.02, 0.05, 0.90]),
        np.array([sample_var * 0.10, 0.10, 0.80]),
        np.array([sample_var * 0.01, 0.15, 0.75]),
    ]

    # Bounds: omega > 0, alpha >= 0, beta >= 0
    bounds = [
        (1e-10, 10 * sample_var),  # omega
        (1e-6, 0.50),              # alpha
        (0.50, 0.9999),            # beta
    ]

    best_result = None
    best_nll = float("inf")

    for x0 in starts:
        # Clip starting values to bounds
        x0_clipped = np.array([
            np.clip(x0[0], bounds[0][0], bounds[0][1]),
            np.clip(x0[1], bounds[1][0], bounds[1][1]),
            np.clip(x0[2], bounds[2][0], bounds[2][1]),
        ])

        try:
            res = minimize(
                _garch_log_likelihood,
                x0_clipped,
                args=(returns,),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-12},
            )
            if res.fun < best_nll:
                best_nll = res.fun
                best_result = res
        except Exception:
            continue

    if best_result is None:
        # Fallback: use starting values
        best_result = type("R", (), {
            "x": starts[0], "fun": float("inf"), "success": False,
        })()

    result = best_result
    omega, alpha, beta = result.x

    # Enforce stationarity
    if alpha + beta >= 1.0:
        scale = 0.999 / (alpha + beta)
        alpha *= scale
        beta *= scale

    # Reconstruct conditional variance series
    sigma2 = np.empty(n)
    sigma2[0] = sample_var

    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        if sigma2[t] < 1e-12:
            sigma2[t] = 1e-12

    # Annualized conditional volatility
    cond_vol = np.sqrt(sigma2 * 252)

    return GARCHResult(
        omega=float(omega),
        alpha=float(alpha),
        beta=float(beta),
        log_likelihood=float(-result.fun),
        conditional_volatility=cond_vol,
        n_obs=n,
        converged=result.success,
    )


def forecast_volatility(
    result: GARCHResult,
    returns: np.ndarray,
    horizon: int = 10,
) -> np.ndarray:
    """Forecast conditional volatility h steps ahead.

    Uses the GARCH(1,1) multi-step forecast formula:
      sigma²_{t+h} = V_L + (alpha+beta)^h * (sigma²_t - V_L)

    where V_L = omega / (1 - alpha - beta) is the long-run variance.

    Parameters
    ----------
    result : fitted GARCHResult
    returns : the return series used for fitting (to get last sigma²)
    horizon : number of days to forecast

    Returns
    -------
    (horizon,) array of annualized volatility forecasts.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns - returns.mean()

    # Last conditional variance
    sigma2_last = np.var(returns)  # initial
    for t in range(1, len(returns)):
        sigma2_last = (
            result.omega
            + result.alpha * returns[t - 1] ** 2
            + result.beta * sigma2_last
        )

    long_run_var = result.long_run_variance
    persistence = result.persistence

    forecasts = np.empty(horizon)
    for h in range(1, horizon + 1):
        forecasts[h - 1] = long_run_var + (persistence ** h) * (sigma2_last - long_run_var)

    # Annualize
    return np.sqrt(np.maximum(forecasts, 0.0) * 252)


def volatility_regime(
    conditional_vol: np.ndarray,
    low_quantile: float = 0.25,
    high_quantile: float = 0.75,
) -> np.ndarray:
    """Classify volatility regime: 'low', 'normal', or 'high'.

    Parameters
    ----------
    conditional_vol : (n,) annualized conditional volatility series
    low_quantile : below this quantile → 'low'
    high_quantile : above this quantile → 'high'

    Returns
    -------
    (n,) array of regime labels.
    """
    low_thresh = np.quantile(conditional_vol, low_quantile)
    high_thresh = np.quantile(conditional_vol, high_quantile)

    regimes = np.full(len(conditional_vol), "normal", dtype=object)
    regimes[conditional_vol < low_thresh] = "low"
    regimes[conditional_vol > high_thresh] = "high"

    return regimes
