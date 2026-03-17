"""Quantitative position sizing utilities."""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25,
) -> float:
    """Fractional Kelly criterion.

    Returns the optimal fraction of capital to risk (0 to 1).
    *fraction* scales the full Kelly (default 0.25 = quarter-Kelly).
    """
    if win_rate <= 0 or win_rate >= 1 or avg_win <= 0 or avg_loss <= 0:
        return 0.0

    # Kelly formula: f* = (p * b - q) / b
    # where p = win_rate, q = 1-p, b = avg_win / avg_loss
    b = avg_win / avg_loss
    q = 1.0 - win_rate
    full_kelly = (win_rate * b - q) / b

    if full_kelly <= 0:
        return 0.0

    return min(full_kelly * fraction, 1.0)


def volatility_adjusted_shares(
    price: float,
    atr_14: float,
    portfolio_value: float,
    risk_per_trade_pct: float = 1.0,
    holding_days: int = 20,
) -> int:
    """Size position so expected loss over holding period = risk_per_trade_pct of portfolio.

    ATR is scaled by sqrt(holding_days) for multi-day risk estimation.
    Returns the number of whole shares (int).
    """
    if price <= 0 or atr_14 <= 0 or portfolio_value <= 0 or risk_per_trade_pct <= 0:
        return 0

    dollar_risk = portfolio_value * (risk_per_trade_pct / 100.0)
    scaled_atr = atr_14 * math.sqrt(holding_days)
    shares = dollar_risk / scaled_atr

    return max(int(shares), 0)


def correlation_adjusted_size(
    base_shares: int,
    new_ticker_returns: np.ndarray,
    existing_returns: np.ndarray,
) -> int:
    """Reduce *base_shares* based on average correlation with existing holdings.

    *existing_returns* may be 1-D (single holding) or 2-D (rows = time, cols = holdings).
    The adjustment multiplier is ``1 - avg_abs_correlation``, floored at 20 % of *base_shares*.
    """
    if base_shares <= 0:
        return 0

    if new_ticker_returns.size == 0 or existing_returns.size == 0:
        return base_shares

    try:
        new = np.asarray(new_ticker_returns).flatten()
        existing = np.asarray(existing_returns)

        if existing.ndim == 1:
            existing = existing.reshape(-1, 1)

        # Align lengths to shorter series
        min_len = min(len(new), existing.shape[0])
        if min_len < 2:
            return base_shares

        new = new[:min_len]
        existing = existing[:min_len, :]

        correlations = []
        for col in range(existing.shape[1]):
            std_new = np.std(new, ddof=1)
            std_col = np.std(existing[:, col], ddof=1)
            if std_new == 0 or std_col == 0:
                correlations.append(0.0)
            else:
                corr = np.corrcoef(new, existing[:, col])[0, 1]
                if np.isnan(corr):
                    correlations.append(0.0)
                else:
                    correlations.append(abs(corr))

        avg_corr = float(np.mean(correlations)) if correlations else 0.0
        multiplier = max(1.0 - avg_corr, 0.2)  # floor at 20 %
        adjusted = int(base_shares * multiplier)
        floor = max(int(base_shares * 0.2), 1) if base_shares > 0 else 0
        return max(adjusted, floor)

    except Exception:
        logger.warning("Correlation adjustment failed, returning base_shares")
        return base_shares


_REGIME_MULTIPLIERS: dict[str, float] = {
    "bull": 1.0,
    "sideways": 0.75,
    "bear": 0.5,
    "crisis": 0.25,
}


def regime_adjusted_size(base_shares: int, regime: str, price: float) -> int:
    """Apply regime-based multiplier to position size.

    BULL: 1.0x (no change)
    SIDEWAYS: 0.75x
    BEAR: 0.5x
    CRISIS: 0.25x
    """
    if base_shares <= 0:
        return 0
    multiplier = _REGIME_MULTIPLIERS.get(regime.lower(), 1.0)
    adjusted = int(base_shares * multiplier)
    # Ensure at least 1 share if base was positive (unless crisis rounds to 0)
    return max(adjusted, 0)


def compute_position_size(
    price: float,
    atr_14: float,
    portfolio_value: float,
    max_position_pct: float = 8.0,
    existing_correlation: float = 0.0,
    regime: str | None = None,
    holding_days: int = 20,
) -> dict:
    """Master sizing function.

    Combines volatility-adjusted sizing with a hard cap and correlation haircut.

    Returns::

        {
            "shares": int,
            "method": str,
            "dollar_amount": float,
            "pct_of_portfolio": float,
        }
    """
    if price <= 0 or portfolio_value <= 0:
        return {
            "shares": 0,
            "method": "invalid_input",
            "dollar_amount": 0.0,
            "pct_of_portfolio": 0.0,
        }

    # 1. Volatility-based size (scaled by holding period)
    vol_shares = volatility_adjusted_shares(
        price, atr_14, portfolio_value, holding_days=holding_days,
    )
    method = "volatility_adjusted"

    # 2. Hard cap based on max_position_pct
    max_dollar = portfolio_value * (max_position_pct / 100.0)
    cap_shares = int(max_dollar / price) if price > 0 else 0

    shares = min(vol_shares, cap_shares)
    if shares == cap_shares and cap_shares < vol_shares:
        method = "max_position_cap"

    # 3. Correlation adjustment
    if existing_correlation > 0:
        multiplier = max(1.0 - existing_correlation, 0.2)
        adjusted = int(shares * multiplier)
        floor = max(int(shares * 0.2), 1) if shares > 0 else 0
        shares = max(adjusted, floor)
        method += "+correlation_adj"

    # 4. Regime adjustment
    if regime is not None:
        shares = regime_adjusted_size(shares, regime, price)
        method += "+regime_adj"

    shares = max(shares, 0)
    dollar_amount = shares * price
    pct = (dollar_amount / portfolio_value * 100.0) if portfolio_value > 0 else 0.0

    result = {
        "shares": shares,
        "method": method,
        "dollar_amount": dollar_amount,
        "pct_of_portfolio": round(pct, 4),
    }
    if regime is not None:
        result["regime"] = regime.lower()
        result["regime_multiplier"] = _REGIME_MULTIPLIERS.get(regime.lower(), 1.0)

    return result
