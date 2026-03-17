"""Pairs Trading / Cointegration — statistical arbitrage via mean-reversion.

Tests whether two assets share a long-run equilibrium relationship
(cointegration) and computes trading signals from the spread.

Unlike correlation (which measures co-movement), cointegration detects
pairs whose price *ratio* or *spread* is mean-reverting — even if the
individual prices are non-stationary (random walks).

Key outputs:
  - Engle-Granger cointegration test (ADF on residuals)
  - Spread z-score for signal generation
  - Half-life of mean reversion (Ornstein-Uhlenbeck)
  - Hedge ratio (OLS beta)

References:
  - Engle & Granger (1987) — "Co-Integration and Error Correction"
  - Vidyamurthy (2004) — "Pairs Trading: Quantitative Methods and Analysis"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.stats import linregress

logger = logging.getLogger(__name__)


@dataclass
class CointegrationResult:
    """Result of cointegration analysis between two assets."""

    ticker_a: str
    ticker_b: str

    # Cointegration test
    hedge_ratio: float         # OLS beta: how many shares of B per share of A
    intercept: float           # OLS intercept
    adf_statistic: float       # ADF test statistic on spread
    adf_pvalue: float          # p-value (< 0.05 = cointegrated)
    is_cointegrated: bool      # convenience flag

    # Spread analysis
    spread_mean: float
    spread_std: float
    current_spread: float
    current_zscore: float

    # Mean reversion
    half_life: float           # days for spread to revert halfway to mean
    mean_reversion_speed: float  # lambda in OU process

    # Signal
    signal: str                # "BUY_A_SELL_B", "SELL_A_BUY_B", "NEUTRAL"
    signal_strength: float     # absolute z-score

    n_observations: int

    @property
    def is_mean_reverting(self) -> bool:
        """True if half-life is positive and reasonable (< 60 trading days)."""
        return 0 < self.half_life < 60


def _adf_test(series: np.ndarray, max_lags: int = 1) -> tuple[float, float]:
    """Simplified Augmented Dickey-Fuller test.

    Tests H0: unit root (non-stationary) vs H1: stationary.
    Returns (adf_statistic, p_value).
    """
    y = np.asarray(series, dtype=float)
    n = len(y)

    if n < 20:
        return 0.0, 1.0

    # Delta y_t = alpha + rho * y_{t-1} + epsilon_t
    dy = np.diff(y)
    y_lag = y[:-1]

    # OLS: dy = a + rho * y_lag (+ lagged diffs if max_lags > 0)
    T = len(dy)

    if max_lags > 0 and T > max_lags + 2:
        # Include lagged differences
        X_parts = [np.ones(T - max_lags), y_lag[max_lags:]]
        for lag in range(1, max_lags + 1):
            X_parts.append(dy[max_lags - lag:T - lag])
        X = np.column_stack(X_parts)
        y_dep = dy[max_lags:]
    else:
        X = np.column_stack([np.ones(T), y_lag])
        y_dep = dy

    # OLS
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        coeffs = XtX_inv @ (X.T @ y_dep)
        residuals = y_dep - X @ coeffs
        s2 = float(np.sum(residuals ** 2) / (len(y_dep) - X.shape[1]))
        se = np.sqrt(np.diag(XtX_inv) * s2)
        rho = coeffs[1]
        adf_stat = rho / se[1] if se[1] > 1e-12 else 0.0
    except np.linalg.LinAlgError:
        return 0.0, 1.0

    # Approximate p-value using MacKinnon critical values (n > 250 case)
    # Critical values: 1%=-3.43, 5%=-2.86, 10%=-2.57
    if adf_stat < -3.43:
        p_value = 0.005
    elif adf_stat < -2.86:
        p_value = 0.03
    elif adf_stat < -2.57:
        p_value = 0.07
    elif adf_stat < -1.94:
        p_value = 0.15
    elif adf_stat < -1.62:
        p_value = 0.25
    else:
        # Linear interpolation toward 1.0
        p_value = min(0.5 + 0.1 * (adf_stat + 1.62), 1.0)

    return float(adf_stat), float(p_value)


def _half_life(spread: np.ndarray) -> tuple[float, float]:
    """Estimate half-life of mean reversion via OLS on spread changes.

    Fits: delta_spread_t = lambda * (spread_{t-1} - mean) + epsilon
    Half-life = -ln(2) / lambda

    Returns (half_life_days, lambda).
    """
    spread = np.asarray(spread, dtype=float)
    spread_centered = spread - spread.mean()
    y = np.diff(spread_centered)
    x = spread_centered[:-1]

    if len(x) < 10 or np.std(x) < 1e-12:
        return float("inf"), 0.0

    slope = linregress(x, y)
    lam = slope.slope

    if lam >= 0:
        # Not mean-reverting
        return float("inf"), float(lam)

    hl = -np.log(2) / lam
    return float(hl), float(lam)


def cointegration_test(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    ticker_a: str = "A",
    ticker_b: str = "B",
    zscore_entry: float = 2.0,
    zscore_exit: float = 0.5,
) -> CointegrationResult:
    """Test cointegration between two price series and compute trading signals.

    Parameters
    ----------
    prices_a : (n,) price series for asset A
    prices_b : (n,) price series for asset B
    ticker_a, ticker_b : asset identifiers
    zscore_entry : z-score threshold for entry signals
    zscore_exit : z-score threshold for exit (neutral)

    Returns
    -------
    CointegrationResult with test statistics, spread, and signals.
    """
    a = np.asarray(prices_a, dtype=float)
    b = np.asarray(prices_b, dtype=float)

    if len(a) != len(b):
        raise ValueError(f"Price series must have same length: {len(a)} vs {len(b)}.")
    n = len(a)
    if n < 30:
        raise ValueError(f"Need at least 30 observations, got {n}.")

    # Step 1: OLS regression — A = intercept + hedge_ratio * B + residual
    slope_result = linregress(b, a)
    hedge_ratio = slope_result.slope
    intercept = slope_result.intercept

    # Step 2: Compute spread (residuals)
    spread = a - (hedge_ratio * b + intercept)

    # Step 3: ADF test on spread
    adf_stat, adf_pvalue = _adf_test(spread)
    is_cointegrated = adf_pvalue < 0.05

    # Step 4: Spread statistics
    spread_mean = float(spread.mean())
    spread_std = float(spread.std())
    current_spread = float(spread[-1])

    if spread_std > 1e-12:
        current_zscore = (current_spread - spread_mean) / spread_std
    else:
        current_zscore = 0.0

    # Step 5: Half-life
    hl, lam = _half_life(spread)

    # Step 6: Trading signal
    abs_z = abs(current_zscore)
    if current_zscore > zscore_entry:
        signal = "SELL_A_BUY_B"  # spread too high, short A / long B
    elif current_zscore < -zscore_entry:
        signal = "BUY_A_SELL_B"  # spread too low, long A / short B
    elif abs_z < zscore_exit:
        signal = "NEUTRAL"       # close to mean
    else:
        signal = "NEUTRAL"

    return CointegrationResult(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        hedge_ratio=float(hedge_ratio),
        intercept=float(intercept),
        adf_statistic=adf_stat,
        adf_pvalue=adf_pvalue,
        is_cointegrated=is_cointegrated,
        spread_mean=spread_mean,
        spread_std=spread_std,
        current_spread=current_spread,
        current_zscore=current_zscore,
        half_life=hl,
        mean_reversion_speed=lam,
        signal=signal,
        signal_strength=abs_z,
        n_observations=n,
    )


def spread_zscore_series(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    lookback: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute rolling spread and z-score series for visualization.

    Parameters
    ----------
    prices_a, prices_b : price series
    lookback : rolling window for z-score normalization

    Returns
    -------
    (spread, zscore, hedge_ratios) — all (n,) arrays.
    """
    a = np.asarray(prices_a, dtype=float)
    b = np.asarray(prices_b, dtype=float)
    n = len(a)

    if n < lookback + 10:
        raise ValueError(f"Need at least {lookback + 10} observations.")

    # Rolling OLS for hedge ratio (expanding window, min lookback)
    hedge_ratios = np.full(n, np.nan)
    spread = np.full(n, np.nan)
    zscore = np.full(n, np.nan)

    for t in range(lookback, n):
        window_a = a[t - lookback:t + 1]
        window_b = b[t - lookback:t + 1]
        sl = linregress(window_b, window_a)
        hr = sl.slope
        hedge_ratios[t] = hr
        sp = a[t] - hr * b[t] - sl.intercept
        spread[t] = sp

    # Rolling z-score on spread
    valid = ~np.isnan(spread)
    for t in range(lookback, n):
        window = spread[max(lookback, t - lookback):t + 1]
        window = window[~np.isnan(window)]
        if len(window) >= 10:
            mu = window.mean()
            sigma = window.std()
            if sigma > 1e-12:
                zscore[t] = (spread[t] - mu) / sigma

    return spread, zscore, hedge_ratios
