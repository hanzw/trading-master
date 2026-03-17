"""Black-Litterman model: combine market equilibrium with analyst views."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ── Signal-to-return mapping ─────────────────────────────────────────

_SIGNAL_RETURNS: dict[str, float] = {
    "STRONG_BUY": 0.10,
    "BUY": 0.05,
    "HOLD": 0.00,
    "SELL": -0.05,
    "STRONG_SELL": -0.10,
}


def _confidence_to_uncertainty(confidence: float) -> float:
    """Map analyst confidence (0-100) to BL uncertainty (Omega diagonal).

    High confidence → low uncertainty, low confidence → high uncertainty.
    Linear interpolation: confidence=100 → 0.001, confidence=0 → 0.25.
    """
    confidence = max(0.0, min(100.0, confidence))
    # Linear: uncertainty = 0.25 - (0.249 * confidence / 100)
    return 0.25 - 0.249 * (confidence / 100.0)


# ── Core functions ───────────────────────────────────────────────────

def implied_equilibrium_returns(
    cov_matrix: np.ndarray,
    market_weights: np.ndarray,
    risk_aversion: float = 2.5,
) -> np.ndarray:
    """Compute implied equilibrium excess returns: Pi = delta * Sigma @ w_mkt."""
    cov = np.asarray(cov_matrix, dtype=float)
    w = np.asarray(market_weights, dtype=float)
    return risk_aversion * cov @ w


def signal_to_views(
    analyst_reports: list[dict],
    tickers: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert LLM analyst signals to Black-Litterman views.

    Maps: STRONG_BUY=+10%, BUY=+5%, HOLD=0%, SELL=-5%, STRONG_SELL=-10%
    Confidence -> Omega (diagonal uncertainty matrix).

    Parameters
    ----------
    analyst_reports : list of dicts with keys 'ticker', 'signal', 'confidence'.
    tickers : ordered list of asset tickers (defines column indices).

    Returns
    -------
    (P, Q, Omega)
      P: (K, N) picking matrix
      Q: (K,) view returns
      Omega: (K, K) diagonal uncertainty matrix
    """
    n = len(tickers)
    ticker_idx = {t: i for i, t in enumerate(tickers)}

    # Aggregate views per ticker (average if multiple reports for same ticker)
    views_by_ticker: dict[str, list[tuple[float, float]]] = {}
    for report in analyst_reports:
        t = report.get("ticker", "").upper()
        sig = report.get("signal", "HOLD")
        # Handle Signal enum objects
        if hasattr(sig, "value"):
            sig = sig.value
        conf = float(report.get("confidence", 50))

        if t not in ticker_idx:
            continue

        ret = _SIGNAL_RETURNS.get(sig, 0.0)
        views_by_ticker.setdefault(t, []).append((ret, conf))

    if not views_by_ticker:
        # No views: return empty arrays
        return np.zeros((0, n)), np.zeros(0), np.zeros((0, 0))

    k = len(views_by_ticker)
    P = np.zeros((k, n))
    Q = np.zeros(k)
    omega_diag = np.zeros(k)

    for row, (ticker, entries) in enumerate(views_by_ticker.items()):
        col = ticker_idx[ticker]
        P[row, col] = 1.0

        # Average return and confidence across analysts for same ticker
        avg_ret = np.mean([e[0] for e in entries])
        avg_conf = np.mean([e[1] for e in entries])

        Q[row] = avg_ret
        omega_diag[row] = _confidence_to_uncertainty(avg_conf)

    Omega = np.diag(omega_diag)
    return P, Q, Omega


def black_litterman_returns(
    cov_matrix: np.ndarray,
    equilibrium_returns: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    tau: float = 0.05,
) -> np.ndarray:
    """Compute Black-Litterman posterior expected returns.

    E[R] = [(tau*Sigma)^{-1} + P'*Omega^{-1}*P]^{-1}
           * [(tau*Sigma)^{-1}*Pi + P'*Omega^{-1}*Q]

    If no views are provided (P is empty), returns equilibrium_returns.
    """
    cov = np.asarray(cov_matrix, dtype=float)
    pi = np.asarray(equilibrium_returns, dtype=float)
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    Omega = np.asarray(Omega, dtype=float)

    tau_sigma = tau * cov
    tau_sigma_inv = np.linalg.inv(tau_sigma)

    # No views case
    if P.size == 0 or P.shape[0] == 0:
        return pi

    omega_inv = np.linalg.inv(Omega)

    # Posterior precision
    precision = tau_sigma_inv + P.T @ omega_inv @ P
    # Posterior mean
    mean_part = tau_sigma_inv @ pi + P.T @ omega_inv @ Q

    return np.linalg.solve(precision, mean_part)


def bl_optimal_weights(
    cov_matrix: np.ndarray,
    bl_returns: np.ndarray,
    risk_aversion: float = 2.5,
    long_only: bool = True,
) -> np.ndarray:
    """Compute optimal weights from BL returns: w* = (delta*Sigma)^{-1} * E[R].

    If long_only, clip negatives and renormalize.
    """
    cov = np.asarray(cov_matrix, dtype=float)
    mu = np.asarray(bl_returns, dtype=float)

    inv_cov = np.linalg.inv(risk_aversion * cov)
    w = inv_cov @ mu

    if long_only:
        w = np.clip(w, 0.0, None)
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        else:
            w = np.full(len(mu), 1.0 / len(mu))
    else:
        # Normalize to sum to 1
        w_sum = w.sum()
        if abs(w_sum) > 1e-12:
            w = w / w_sum

    return w


def run_black_litterman(
    tickers: list[str],
    analyst_reports: list[dict],
    current_weights: np.ndarray | None = None,
    lookback_days: int = 252,
) -> dict:
    """Full BL pipeline.

    1. Fetch returns & compute covariance
    2. Compute equilibrium returns from current weights (or equal weights)
    3. Convert analyst signals to views (P, Q, Omega)
    4. Compute BL posterior returns
    5. Compute optimal weights

    Returns dict with bl_returns, optimal_weights, current_weights,
    equilibrium_returns, tickers, and suggested_trades.
    """
    from ..portfolio.correlation import rolling_covariance

    cov, _corr, valid_tickers = rolling_covariance(
        tickers, window=min(60, lookback_days), lookback_days=lookback_days,
    )

    if cov is None or len(valid_tickers) == 0:
        return {
            "error": "Could not compute covariance matrix — insufficient data.",
            "tickers": tickers,
        }

    n = len(valid_tickers)

    # Build current weights vector aligned to valid tickers
    if current_weights is not None:
        # Map original tickers to valid tickers
        ticker_to_weight = dict(zip(tickers, current_weights))
        w_current = np.array([
            ticker_to_weight.get(t, 0.0) for t in valid_tickers
        ])
        w_sum = w_current.sum()
        if w_sum > 0:
            w_current = w_current / w_sum
        else:
            w_current = np.full(n, 1.0 / n)
    else:
        w_current = np.full(n, 1.0 / n)

    # 1. Equilibrium returns
    pi = implied_equilibrium_returns(cov, w_current)

    # 2. Views from analyst reports
    P, Q, Omega = signal_to_views(analyst_reports, valid_tickers)

    # 3. BL posterior returns
    bl_ret = black_litterman_returns(cov, pi, P, Q, Omega)

    # 4. Optimal weights
    w_optimal = bl_optimal_weights(cov, bl_ret)

    # 5. Suggested trades
    trades = []
    for i, t in enumerate(valid_tickers):
        cur_pct = w_current[i] * 100
        tgt_pct = w_optimal[i] * 100
        diff = tgt_pct - cur_pct
        direction = "BUY" if diff > 0.5 else ("SELL" if diff < -0.5 else "HOLD")
        trades.append({
            "ticker": t,
            "current_pct": round(cur_pct, 2),
            "target_pct": round(tgt_pct, 2),
            "direction": direction,
        })

    return {
        "tickers": valid_tickers,
        "equilibrium_returns": pi,
        "bl_returns": bl_ret,
        "optimal_weights": w_optimal,
        "current_weights": w_current,
        "suggested_trades": trades,
    }
