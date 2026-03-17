"""Extreme Value Theory (EVT) — tail risk estimation via GPD.

EVT models the distribution of extreme losses using the Peaks-Over-Threshold
(POT) method with a Generalized Pareto Distribution (GPD). Unlike normal
or Student-t assumptions, EVT captures the true shape of the loss tail.

Key outputs:
  - Tail VaR and tail CVaR at arbitrary confidence levels
  - Shape parameter (xi): xi > 0 = heavy tail (Frechet), xi = 0 = exponential,
    xi < 0 = bounded tail (Weibull)
  - Expected Shortfall beyond the threshold

References:
  - McNeil & Frey (2000) — "Estimation of tail-related risk measures
    for heteroscedastic financial time series"
  - Pickands (1975) — "Statistical inference using extreme order statistics"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import genpareto
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


@dataclass
class EVTResult:
    """Result of EVT tail risk analysis."""

    # GPD parameters
    shape: float           # xi (shape param): >0 heavy tail, <0 bounded, ~0 exponential
    scale: float           # sigma (scale param)
    threshold: float       # u (the POT threshold)
    n_exceedances: int     # number of observations beyond threshold
    n_total: int           # total observations

    # Risk measures
    var_95: float          # Value-at-Risk at 95%
    var_99: float          # Value-at-Risk at 99%
    cvar_95: float         # Conditional VaR (Expected Shortfall) at 95%
    cvar_99: float         # Conditional VaR (Expected Shortfall) at 99%

    # Tail descriptors
    tail_type: str         # "heavy", "exponential", "bounded"
    tail_index: float      # 1/xi for heavy tails (higher = heavier)

    # Diagnostics
    exceedance_rate: float  # fraction of obs exceeding threshold
    mean_excess: float      # mean of exceedances over threshold
    ks_pvalue: float | None = None  # KS test p-value for GPD fit

    @property
    def is_heavy_tailed(self) -> bool:
        return self.shape > 0.05

    @property
    def dollar_var_99(self) -> float:
        """Placeholder — multiply by portfolio value externally."""
        return self.var_99


def _select_threshold(losses: np.ndarray, quantile: float = 0.90) -> float:
    """Select the POT threshold as a quantile of the loss distribution.

    The 90th percentile is a common default that balances having enough
    exceedances for fitting while focusing on the tail.
    """
    return float(np.quantile(losses, quantile))


def _fit_gpd(exceedances: np.ndarray) -> tuple[float, float, float]:
    """Fit GPD to exceedances using MLE.

    Returns (shape, loc, scale). loc is fixed at 0 for POT exceedances.
    """
    # genpareto.fit returns (shape, loc, scale)
    shape, loc, scale = genpareto.fit(exceedances, floc=0)
    return shape, loc, scale


def _gpd_var(
    shape: float,
    scale: float,
    threshold: float,
    n_total: int,
    n_exceed: int,
    confidence: float,
) -> float:
    """Compute VaR using the GPD tail estimate.

    VaR_p = u + (sigma/xi) * [((n/N_u) * (1-p))^(-xi) - 1]
    """
    if n_exceed == 0:
        return threshold

    p = 1.0 - confidence
    rate = n_exceed / n_total

    if abs(shape) < 1e-8:
        # Exponential case
        return threshold + scale * np.log(rate / p)
    else:
        return threshold + (scale / shape) * ((rate / p) ** shape - 1.0)


def _gpd_cvar(
    shape: float,
    scale: float,
    threshold: float,
    n_total: int,
    n_exceed: int,
    confidence: float,
) -> float:
    """Compute CVaR (Expected Shortfall) using GPD.

    ES_p = VaR_p / (1 - xi) + (scale - xi * u) / (1 - xi)
    """
    var = _gpd_var(shape, scale, threshold, n_total, n_exceed, confidence)

    if shape >= 1.0:
        # Mean doesn't exist for xi >= 1
        return var * 1.5  # conservative estimate

    cvar = var / (1.0 - shape) + (scale - shape * threshold) / (1.0 - shape)
    return max(cvar, var)  # CVaR >= VaR always


def evt_tail_risk(
    returns: np.ndarray,
    threshold_quantile: float = 0.90,
    confidence_levels: tuple[float, ...] = (0.95, 0.99),
) -> EVTResult:
    """Estimate tail risk using Extreme Value Theory (POT + GPD).

    Parameters
    ----------
    returns : array of portfolio or asset returns (can be positive and negative)
    threshold_quantile : quantile for POT threshold selection (default 90th pctile of losses)
    confidence_levels : confidence levels for VaR/CVaR (must include 0.95 and 0.99)

    Returns
    -------
    EVTResult with GPD parameters, VaR, CVaR, and tail diagnostics.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]

    if len(returns) < 30:
        raise ValueError(f"Need at least 30 observations, got {len(returns)}.")

    # Convert to losses (positive values = losses)
    losses = -returns
    n_total = len(losses)

    # Select threshold
    threshold = _select_threshold(losses, threshold_quantile)

    # Extract exceedances
    exceedances = losses[losses > threshold] - threshold

    if len(exceedances) < 10:
        raise ValueError(
            f"Only {len(exceedances)} exceedances above threshold "
            f"{threshold:.4f}. Need at least 10. Try lowering threshold_quantile."
        )

    n_exceed = len(exceedances)

    # Fit GPD
    shape, _, scale = _fit_gpd(exceedances)

    # Compute risk measures
    var_95 = _gpd_var(shape, scale, threshold, n_total, n_exceed, 0.95)
    var_99 = _gpd_var(shape, scale, threshold, n_total, n_exceed, 0.99)
    cvar_95 = _gpd_cvar(shape, scale, threshold, n_total, n_exceed, 0.95)
    cvar_99 = _gpd_cvar(shape, scale, threshold, n_total, n_exceed, 0.99)

    # Tail classification
    if shape > 0.05:
        tail_type = "heavy"
    elif shape < -0.05:
        tail_type = "bounded"
    else:
        tail_type = "exponential"

    tail_index = 1.0 / shape if abs(shape) > 1e-6 else float("inf")

    # KS test for goodness of fit
    ks_pvalue = None
    try:
        from scipy.stats import kstest
        stat, pval = kstest(exceedances, "genpareto", args=(shape, 0, scale))
        ks_pvalue = float(pval)
    except Exception:
        pass

    return EVTResult(
        shape=float(shape),
        scale=float(scale),
        threshold=float(threshold),
        n_exceedances=n_exceed,
        n_total=n_total,
        var_95=float(var_95),
        var_99=float(var_99),
        cvar_95=float(cvar_95),
        cvar_99=float(cvar_99),
        tail_type=tail_type,
        tail_index=float(tail_index),
        exceedance_rate=n_exceed / n_total,
        mean_excess=float(exceedances.mean()),
        ks_pvalue=ks_pvalue,
    )


def mean_excess_plot_data(
    returns: np.ndarray,
    n_thresholds: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute data for the Mean Excess (ME) plot.

    The ME plot helps validate the GPD assumption: for GPD data,
    the mean excess function should be approximately linear.

    Parameters
    ----------
    returns : array of returns
    n_thresholds : number of threshold points

    Returns
    -------
    (thresholds, mean_excesses) arrays for plotting.
    """
    losses = -np.asarray(returns, dtype=float)
    losses = losses[np.isfinite(losses)]
    losses.sort()

    # Thresholds from 50th to 95th percentile
    lo = np.quantile(losses, 0.50)
    hi = np.quantile(losses, 0.95)
    thresholds = np.linspace(lo, hi, n_thresholds)

    mean_excesses = np.zeros(n_thresholds)
    for i, u in enumerate(thresholds):
        exceed = losses[losses > u] - u
        mean_excesses[i] = exceed.mean() if len(exceed) > 0 else 0.0

    return thresholds, mean_excesses
