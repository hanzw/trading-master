"""Portfolio Risk Dashboard — unified view of all quant signals.

Aggregates multiple quant modules into a single assessment:
  - HMM market regime (current state + confidence)
  - Portfolio-level risk metrics (Sharpe, Sortino, max drawdown)
  - EVT tail risk summary for top holdings
  - Sector rotation leaders/laggards
  - Overall risk score (composite)

This is the "CEO view" — one glance shows whether the portfolio
needs attention or is healthy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DashboardMetrics:
    """Aggregated risk/return metrics for dashboard display."""

    # Regime
    regime: str = "unknown"
    regime_confidence: float = 0.0

    # Portfolio risk
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    portfolio_volatility: float = 0.0

    # Tail risk
    tail_type: str = "unknown"
    var_99: float = 0.0
    cvar_99: float = 0.0

    # Sector
    top_sector: str = ""
    bottom_sector: str = ""

    # Composite
    risk_score: float = 0.0         # 0 (safe) to 100 (dangerous)
    risk_level: str = "unknown"     # "low", "moderate", "elevated", "high", "extreme"
    health_summary: str = ""

    @property
    def is_healthy(self) -> bool:
        return self.risk_score < 40


def compute_risk_score(
    regime: str = "neutral",
    sharpe: float = 0.0,
    max_dd: float = 0.0,
    tail_type: str = "exponential",
    portfolio_vol: float = 0.0,
) -> tuple[float, str]:
    """Compute a composite risk score (0-100) from multiple inputs.

    Returns (score, level).
    """
    score = 0.0

    # Regime component (0-30)
    regime_scores = {
        "crisis": 30, "bear": 20, "neutral": 10, "sideways": 10, "bull": 5,
    }
    score += regime_scores.get(regime.lower(), 15)

    # Sharpe component (0-20): negative Sharpe is risky
    if sharpe < 0:
        score += min(20, abs(sharpe) * 10)
    elif sharpe < 0.5:
        score += 10
    elif sharpe < 1.0:
        score += 5

    # Drawdown component (0-25)
    dd_pct = abs(max_dd) * 100
    if dd_pct > 20:
        score += 25
    elif dd_pct > 10:
        score += 15
    elif dd_pct > 5:
        score += 10
    else:
        score += 5

    # Tail component (0-15)
    tail_scores = {"heavy": 15, "exponential": 8, "bounded": 3}
    score += tail_scores.get(tail_type, 8)

    # Volatility component (0-10)
    ann_vol = portfolio_vol * np.sqrt(252) if portfolio_vol > 0 else 0
    if ann_vol > 0.30:
        score += 10
    elif ann_vol > 0.20:
        score += 7
    elif ann_vol > 0.15:
        score += 5
    else:
        score += 2

    score = min(score, 100)

    # Classify
    if score < 20:
        level = "low"
    elif score < 40:
        level = "moderate"
    elif score < 60:
        level = "elevated"
    elif score < 80:
        level = "high"
    else:
        level = "extreme"

    return float(score), level


def build_dashboard(
    portfolio_returns: np.ndarray | None = None,
    regime: str = "unknown",
    regime_confidence: float = 0.0,
    tail_type: str = "unknown",
    var_99: float = 0.0,
    cvar_99: float = 0.0,
    top_sector: str = "",
    bottom_sector: str = "",
) -> DashboardMetrics:
    """Build a unified dashboard from available data.

    Parameters
    ----------
    portfolio_returns : (T,) daily portfolio returns (optional)
    regime : current market regime label
    regime_confidence : HMM confidence for current regime
    tail_type : EVT tail classification
    var_99 : EVT VaR at 99%
    cvar_99 : EVT CVaR at 99%
    top_sector : leading sector ticker
    bottom_sector : lagging sector ticker

    Returns
    -------
    DashboardMetrics with all fields populated.
    """
    sharpe = 0.0
    sortino = 0.0
    max_dd = 0.0
    port_vol = 0.0

    if portfolio_returns is not None and len(portfolio_returns) >= 20:
        from ..portfolio.risk_metrics import sharpe_ratio, sortino_ratio, max_drawdown
        ret = np.asarray(portfolio_returns, dtype=float)

        sharpe = sharpe_ratio(ret)
        sortino = sortino_ratio(ret)
        port_vol = float(np.std(ret))

        equity = 10000 * np.concatenate([[1.0], np.cumprod(1 + ret)])
        dd_info = max_drawdown(equity)
        max_dd = dd_info["max_dd"]

    score, level = compute_risk_score(
        regime=regime,
        sharpe=sharpe,
        max_dd=max_dd,
        tail_type=tail_type,
        portfolio_vol=port_vol,
    )

    # Build health summary
    parts = []
    if level in ("high", "extreme"):
        parts.append(f"RISK {level.upper()}")
    if regime.lower() in ("crisis", "bear"):
        parts.append(f"{regime.upper()} regime")
    if tail_type == "heavy":
        parts.append("heavy-tailed returns")
    if max_dd > 0.10:
        parts.append(f"drawdown {max_dd:.0%}")
    if not parts:
        parts.append("Portfolio healthy — no elevated risks detected")

    health_summary = "; ".join(parts)

    return DashboardMetrics(
        regime=regime,
        regime_confidence=regime_confidence,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        portfolio_volatility=port_vol,
        tail_type=tail_type,
        var_99=var_99,
        cvar_99=cvar_99,
        top_sector=top_sector,
        bottom_sector=bottom_sector,
        risk_score=score,
        risk_level=level,
        health_summary=health_summary,
    )
