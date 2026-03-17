"""Sector Rotation — relative strength and momentum across market sectors.

Analyzes sector ETFs to identify which sectors are leading or lagging
the broad market. Combines:
  - Relative strength (sector return vs SPY)
  - Momentum (trailing returns over multiple windows)
  - Trend score (price vs moving averages)

Sector ETFs tracked:
  XLK (Technology), XLF (Financials), XLV (Healthcare), XLY (Consumer Disc.),
  XLP (Consumer Staples), XLE (Energy), XLI (Industrials), XLB (Materials),
  XLRE (Real Estate), XLU (Utilities), XLC (Communication Services)

References:
  - Faber (2007) — "A Quantitative Approach to Tactical Asset Allocation"
  - Stangl, Jacobsen, & Visaltanachoti (2009) — "Sector rotation over business cycles"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Standard sector ETF universe
SECTOR_ETFS: dict[str, str] = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLC": "Communication Services",
}


@dataclass
class SectorScore:
    """Score for a single sector."""
    ticker: str
    name: str
    momentum_1m: float       # 1-month return
    momentum_3m: float       # 3-month return
    momentum_6m: float       # 6-month return
    relative_strength: float  # return vs benchmark over 3m
    trend_score: float        # composite: above SMA20/50/200 → +1 each
    composite_score: float    # weighted combination
    rank: int = 0


@dataclass
class SectorRotationResult:
    """Result of sector rotation analysis."""
    sectors: list[SectorScore]
    benchmark_return_3m: float
    analysis_date: str
    n_sectors: int

    @property
    def leaders(self) -> list[SectorScore]:
        """Top 3 sectors by composite score."""
        return self.sectors[:3]

    @property
    def laggards(self) -> list[SectorScore]:
        """Bottom 3 sectors by composite score."""
        return self.sectors[-3:]

    @property
    def score_dict(self) -> dict[str, float]:
        """Ticker → composite score mapping."""
        return {s.ticker: s.composite_score for s in self.sectors}


def _compute_returns(prices: np.ndarray, windows: list[int]) -> dict[int, float]:
    """Compute trailing returns for given windows (in trading days)."""
    result = {}
    n = len(prices)
    for w in windows:
        if n > w and prices[-(w + 1)] > 0:
            result[w] = (prices[-1] / prices[-(w + 1)]) - 1.0
        else:
            result[w] = 0.0
    return result


def _trend_score(prices: np.ndarray) -> float:
    """Score 0-3: +1 for price above each of SMA20, SMA50, SMA200."""
    n = len(prices)
    score = 0.0
    current = prices[-1]

    for window in [20, 50, 200]:
        if n >= window:
            sma = prices[-window:].mean()
            if current > sma:
                score += 1.0

    return score


def analyze_sectors(
    price_data: dict[str, np.ndarray],
    benchmark_prices: np.ndarray | None = None,
    momentum_weights: tuple[float, float, float] = (0.2, 0.4, 0.4),
    relative_strength_weight: float = 0.3,
    trend_weight: float = 0.2,
    momentum_weight: float = 0.5,
) -> SectorRotationResult:
    """Analyze sector rotation from price data.

    Parameters
    ----------
    price_data : {ticker: prices_array} for each sector ETF
    benchmark_prices : SPY or similar benchmark prices (optional)
    momentum_weights : weights for (1m, 3m, 6m) momentum components
    relative_strength_weight : weight of relative strength in composite
    trend_weight : weight of trend score in composite
    momentum_weight : weight of momentum in composite

    Returns
    -------
    SectorRotationResult with sectors sorted by composite score (best first).
    """
    if not price_data:
        raise ValueError("No price data provided.")

    # Compute benchmark 3m return
    benchmark_3m = 0.0
    if benchmark_prices is not None and len(benchmark_prices) > 63:
        benchmark_3m = (benchmark_prices[-1] / benchmark_prices[-64]) - 1.0

    sectors: list[SectorScore] = []

    for ticker, prices in price_data.items():
        if len(prices) < 30:
            continue

        name = SECTOR_ETFS.get(ticker, ticker)
        returns = _compute_returns(prices, [21, 63, 126])  # ~1m, 3m, 6m

        mom_1m = returns.get(21, 0.0)
        mom_3m = returns.get(63, 0.0)
        mom_6m = returns.get(126, 0.0)

        # Weighted momentum
        w1, w3, w6 = momentum_weights
        momentum_composite = w1 * mom_1m + w3 * mom_3m + w6 * mom_6m

        # Relative strength vs benchmark
        rel_strength = mom_3m - benchmark_3m

        # Trend (0-3)
        trend = _trend_score(prices)

        # Normalize trend to ~same scale as returns (0-3 → 0-0.15)
        trend_normalized = trend / 20.0

        # Composite score
        composite = (
            momentum_weight * momentum_composite
            + relative_strength_weight * rel_strength
            + trend_weight * trend_normalized
        )

        sectors.append(SectorScore(
            ticker=ticker,
            name=name,
            momentum_1m=mom_1m,
            momentum_3m=mom_3m,
            momentum_6m=mom_6m,
            relative_strength=rel_strength,
            trend_score=trend,
            composite_score=composite,
        ))

    # Sort by composite score (best first)
    sectors.sort(key=lambda s: s.composite_score, reverse=True)

    # Assign ranks
    for i, s in enumerate(sectors):
        s.rank = i + 1

    from datetime import datetime
    analysis_date = datetime.now().strftime("%Y-%m-%d")

    return SectorRotationResult(
        sectors=sectors,
        benchmark_return_3m=benchmark_3m,
        analysis_date=analysis_date,
        n_sectors=len(sectors),
    )
