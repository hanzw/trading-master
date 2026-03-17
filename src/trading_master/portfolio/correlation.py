"""Correlation analysis, concentration risk, and minimum-variance weights."""

from __future__ import annotations

import logging

import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)


# ── Data fetching ──────────────────────────────────────────────────────

def fetch_returns(
    tickers: list[str],
    lookback_days: int = 200,
) -> tuple[np.ndarray | None, list[str]]:
    """Fetch daily returns for *tickers* from yfinance.

    Returns
    -------
    (returns_array, valid_tickers) where returns_array is (T, N) and
    valid_tickers lists tickers that had sufficient data.  Returns
    (None, []) on complete failure.
    """
    if not tickers:
        return None, []

    try:
        import pandas as pd

        data = yf.download(
            tickers,
            period=f"{lookback_days}d",
            auto_adjust=True,
            progress=False,
        )

        if data.empty:
            return None, []

        # yf.download returns MultiIndex columns for multiple tickers
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        else:
            # Single ticker: wrap in DataFrame
            close = data[["Close"]]
            close.columns = [tickers[0]]

        # Drop tickers with insufficient data (>50 % NaN)
        threshold = len(close) * 0.5
        close = close.dropna(axis=1, thresh=int(threshold))
        valid_tickers = list(close.columns)

        if not valid_tickers:
            return None, []

        close = close.ffill().dropna()
        returns = close.pct_change().dropna().values

        if returns.shape[0] < 2:
            return None, []

        return returns, valid_tickers

    except Exception:
        logger.exception("fetch_returns failed")
        return None, []


# ── Rolling covariance / correlation ──────────────────────────────────

def rolling_covariance(
    tickers: list[str],
    window: int = 60,
    lookback_days: int = 200,
) -> tuple[np.ndarray | None, np.ndarray | None, list[str]]:
    """Annualized covariance and correlation from the last *window* days.

    Returns (cov_matrix, corr_matrix, valid_tickers).
    """
    returns, valid = fetch_returns(tickers, lookback_days)

    if returns is None or returns.shape[0] < window:
        return None, None, []

    recent = returns[-window:]
    cov = np.cov(recent, rowvar=False) * 252  # annualise
    # Ensure 2-D for single-asset
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])

    std = np.sqrt(np.diag(cov))
    outer = np.outer(std, std)
    corr = np.where(outer == 0, 0.0, cov / outer)

    return cov, corr, valid


# ── Concentration risk ────────────────────────────────────────────────

def concentration_risk(cov_matrix: np.ndarray) -> dict:
    """Eigendecomposition-based concentration analysis.

    Returns
    -------
    dict with eigenvalues, explained_variance_ratios, effective_num_bets,
    top1_dominance, concentrated (bool: top eigenvalue explains >60 %).
    """
    cov = np.asarray(cov_matrix, dtype=float)

    if cov.size == 0:
        return {
            "eigenvalues": np.array([]),
            "explained_variance_ratios": np.array([]),
            "effective_num_bets": 0.0,
            "top1_dominance": 0.0,
            "concentrated": False,
        }

    eigenvalues = np.linalg.eigvalsh(cov)
    # Sort descending
    eigenvalues = eigenvalues[::-1]
    # Clip negatives from numerical noise
    eigenvalues = np.maximum(eigenvalues, 0.0)

    total = eigenvalues.sum()
    if total == 0:
        ratios = np.zeros_like(eigenvalues)
        top1 = 0.0
        enb = 0.0
    else:
        ratios = eigenvalues / total
        top1 = float(ratios[0])
        # Effective number of bets (entropy-based)
        nonzero = ratios[ratios > 0]
        enb = float(np.exp(-np.sum(nonzero * np.log(nonzero))))

    return {
        "eigenvalues": eigenvalues,
        "explained_variance_ratios": ratios,
        "effective_num_bets": enb,
        "top1_dominance": top1,
        "concentrated": top1 > 0.6,
    }


# ── Correlation check for new positions ──────────────────────────────

def check_correlation_ok(
    new_ticker: str,
    existing_tickers: list[str],
    max_avg_correlation: float = 0.7,
    lookback_days: int = 200,
) -> tuple[bool, float, dict]:
    """Check if *new_ticker* is sufficiently uncorrelated with existing holdings.

    Returns (ok, avg_correlation, details_dict).
    """
    if not existing_tickers:
        return True, 0.0, {"message": "no existing tickers to compare"}

    all_tickers = [new_ticker] + list(existing_tickers)
    returns, valid = fetch_returns(all_tickers, lookback_days)

    if returns is None or new_ticker not in valid:
        return True, 0.0, {"message": "insufficient data, allowing by default"}

    new_idx = valid.index(new_ticker)
    new_col = returns[:, new_idx]

    correlations = {}
    raw_corrs = []
    for i, t in enumerate(valid):
        if t == new_ticker:
            continue
        corr_val = float(np.corrcoef(new_col, returns[:, i])[0, 1])
        if np.isnan(corr_val):
            corr_val = 0.0
        correlations[t] = corr_val
        raw_corrs.append(corr_val)

    if not raw_corrs:
        return True, 0.0, {"message": "no overlapping tickers with data"}

    # Only positive correlations contribute to concentration risk;
    # negative correlations are hedges and should not be penalised.
    pos_corrs = [c for c in raw_corrs if c > 0]
    avg = float(np.mean(pos_corrs)) if pos_corrs else 0.0
    ok = avg <= max_avg_correlation

    return ok, avg, {
        "pairwise_correlations": correlations,
        "avg_positive_correlation": avg,
        "threshold": max_avg_correlation,
    }


# ── Minimum variance portfolio ───────────────────────────────────────

def minimum_variance_weights(
    cov_matrix: np.ndarray,
    long_only: bool = True,
) -> np.ndarray:
    """Analytical minimum-variance portfolio weights.

    ``w = Sigma^{-1} @ 1 / (1' @ Sigma^{-1} @ 1)``

    Parameters
    ----------
    cov_matrix : (N, N) covariance matrix.
    long_only : if True, clip negative weights to 0 and renormalize.
    """
    cov = np.asarray(cov_matrix, dtype=float)

    if cov.size == 0:
        return np.array([])

    n = cov.shape[0]

    if n == 1:
        return np.array([1.0])

    # Regularize: shrink toward diagonal
    alpha = 0.1  # shrinkage factor
    cov_reg = (1 - alpha) * cov + alpha * np.eye(n) * np.trace(cov) / n

    try:
        inv_cov = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        # Singular matrix: use pseudo-inverse
        inv_cov = np.linalg.pinv(cov_reg)

    ones = np.ones(n)
    w = inv_cov @ ones
    denom = ones @ w

    if denom == 0:
        return np.full(n, 1.0 / n)

    w = w / denom

    if long_only:
        w = np.clip(w, 0.0, None)
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        else:
            w = np.full(n, 1.0 / n)

    return w
