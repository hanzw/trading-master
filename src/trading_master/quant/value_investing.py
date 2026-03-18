"""Value Investing Strategy Engine — Buffett/Duan Yongping style.

Core principles encoded as quantitative factors:
1. MOAT (competitive advantage) — sustained high ROE, stable margins
2. QUALITY (financial strength) — low debt, high FCF, earnings quality
3. VALUATION (margin of safety) — PE/PB/FCF yield vs history & peers
4. MOMENTUM FILTER — avoid value traps by requiring price stabilization

Strategy:
  Score each stock on Quality (40%) + Valuation (40%) + Momentum (20%)
  Buy top N stocks, rebalance quarterly
  Hold unless thesis breaks (quality deteriorates or valuation extreme)

References:
  - Buffett: "wonderful company at a fair price"
  - Duan Yongping: "buy what you understand, hold forever"
  - Greenblatt: Magic Formula (high ROIC + high earnings yield)
  - Piotroski F-Score: financial strength scoring
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StockScore:
    """Score for a single stock."""
    ticker: str
    name: str = ""
    sector: str = ""
    price: float = 0.0
    market_cap: float = 0.0

    # Quality metrics (raw)
    roe: float | None = None
    profit_margin: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    fcf_yield: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None

    # Valuation metrics (raw)
    trailing_pe: float | None = None
    forward_pe: float | None = None
    pb: float | None = None
    ps: float | None = None
    ev_ebitda: float | None = None
    earnings_yield: float | None = None  # 1/PE

    # Momentum metrics (raw)
    from_52w_high: float | None = None
    pct_52w_range: float | None = None
    sma200_pct: float | None = None  # price vs SMA200

    # Analyst
    analyst_upside: float | None = None
    beta: float | None = None

    # Scores (0-100)
    quality_score: float = 0.0
    valuation_score: float = 0.0
    momentum_score: float = 0.0
    composite_score: float = 0.0

    # Sub-scores
    moat_score: float = 0.0      # ROE + margin stability
    financial_score: float = 0.0  # debt + FCF + current ratio
    piotroski: int = 0           # F-Score 0-9

    reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)  # warnings


def compute_piotroski(
    roe: float | None,
    profit_margin: float | None,
    fcf_yield: float | None,
    debt_to_equity: float | None,
    current_ratio: float | None,
    revenue_growth: float | None,
    earnings_growth: float | None,
) -> int:
    """Piotroski F-Score (simplified, 0-9).

    Each criterion scores 1 if positive:
    1. Positive ROE
    2. Positive FCF yield
    3. ROE > 15% (improving profitability proxy)
    4. FCF > earnings (earnings quality)
    5. Low debt (D/E < 100%)
    6. Current ratio > 1 (liquidity)
    7. No share dilution (proxied by positive earnings growth)
    8. Positive revenue growth
    9. Margin > 15% (gross margin proxy)
    """
    score = 0
    if roe and roe > 0:
        score += 1
    if fcf_yield and fcf_yield > 0:
        score += 1
    if roe and roe > 15:
        score += 1
    if fcf_yield and fcf_yield > 3:  # FCF yield > 3% as quality proxy
        score += 1
    if debt_to_equity is not None and debt_to_equity < 100:
        score += 1
    if current_ratio and current_ratio > 1:
        score += 1
    if earnings_growth and earnings_growth > 0:
        score += 1
    if revenue_growth and revenue_growth > 0:
        score += 1
    if profit_margin and profit_margin > 15:
        score += 1
    return score


def score_quality(stock: StockScore) -> float:
    """Score quality 0-100: moat strength + financial health.

    Buffett: "wonderful company" = high ROE + wide margins + low debt
    """
    score = 0.0
    reasons = []

    # ROE (0-30 points)
    if stock.roe is not None:
        if stock.roe >= 30:
            score += 30; reasons.append(f"ROE={stock.roe:.0f}% exceptional")
        elif stock.roe >= 20:
            score += 25
        elif stock.roe >= 15:
            score += 18
        elif stock.roe >= 10:
            score += 10
        else:
            score += 3

    # Profit margin (0-25 points)
    if stock.profit_margin is not None:
        if stock.profit_margin >= 25:
            score += 25; reasons.append(f"margin={stock.profit_margin:.0f}% wide moat")
        elif stock.profit_margin >= 18:
            score += 20
        elif stock.profit_margin >= 12:
            score += 14
        elif stock.profit_margin >= 8:
            score += 8
        else:
            score += 3

    # Debt (0-20 points) — lower is better
    if stock.debt_to_equity is not None:
        if stock.debt_to_equity < 30:
            score += 20; reasons.append("very low debt")
        elif stock.debt_to_equity < 60:
            score += 16
        elif stock.debt_to_equity < 100:
            score += 12
        elif stock.debt_to_equity < 200:
            score += 6
        else:
            score += 0; stock.flags.append("high debt")

    # FCF yield (0-15 points)
    if stock.fcf_yield is not None:
        if stock.fcf_yield >= 8:
            score += 15; reasons.append(f"FCF yield={stock.fcf_yield:.1f}%")
        elif stock.fcf_yield >= 5:
            score += 12
        elif stock.fcf_yield >= 3:
            score += 8
        elif stock.fcf_yield > 0:
            score += 4

    # Revenue growth (0-10 points)
    if stock.revenue_growth is not None:
        if stock.revenue_growth >= 15:
            score += 10; reasons.append(f"revenue +{stock.revenue_growth:.0f}%")
        elif stock.revenue_growth >= 8:
            score += 7
        elif stock.revenue_growth >= 3:
            score += 4
        elif stock.revenue_growth > 0:
            score += 2

    stock.reasons.extend(reasons)
    return min(score, 100.0)


def score_valuation(stock: StockScore) -> float:
    """Score valuation 0-100: margin of safety.

    Buffett: "fair price" = PE reasonable, PB reasonable, FCF yield high
    Greenblatt: earnings yield = high is good
    """
    score = 0.0
    reasons = []

    # Trailing PE (0-30 points) — lower is cheaper
    pe = stock.trailing_pe
    if pe is not None and pe > 0:
        if pe < 10:
            score += 30; reasons.append(f"PE={pe:.0f} very cheap")
        elif pe < 15:
            score += 25; reasons.append(f"PE={pe:.0f} cheap")
        elif pe < 20:
            score += 18
        elif pe < 25:
            score += 12
        elif pe < 35:
            score += 5
        else:
            score += 0; stock.flags.append(f"PE={pe:.0f} expensive")

    # Forward PE (0-20 points)
    fpe = stock.forward_pe
    if fpe is not None and fpe > 0:
        if fpe < 10:
            score += 20
        elif fpe < 14:
            score += 16
        elif fpe < 18:
            score += 11
        elif fpe < 22:
            score += 6
        else:
            score += 2

    # Price to Book (0-15 points)
    if stock.pb is not None and stock.pb > 0:
        if stock.pb < 2:
            score += 15; reasons.append(f"PB={stock.pb:.1f} deep value")
        elif stock.pb < 4:
            score += 10
        elif stock.pb < 8:
            score += 5
        else:
            score += 1

    # FCF yield as valuation (0-15 points)
    if stock.fcf_yield is not None:
        if stock.fcf_yield >= 8:
            score += 15
        elif stock.fcf_yield >= 5:
            score += 11
        elif stock.fcf_yield >= 3:
            score += 7
        elif stock.fcf_yield > 0:
            score += 3

    # Analyst upside (0-10 points)
    if stock.analyst_upside is not None:
        if stock.analyst_upside >= 30:
            score += 10; reasons.append(f"target +{stock.analyst_upside:.0f}%")
        elif stock.analyst_upside >= 15:
            score += 7
        elif stock.analyst_upside >= 5:
            score += 3

    # EV/EBITDA (0-10 points)
    if stock.ev_ebitda is not None and stock.ev_ebitda > 0:
        if stock.ev_ebitda < 8:
            score += 10
        elif stock.ev_ebitda < 12:
            score += 7
        elif stock.ev_ebitda < 16:
            score += 4

    stock.reasons.extend(reasons)
    return min(score, 100.0)


def score_momentum(stock: StockScore) -> float:
    """Score momentum 0-100: avoid value traps.

    Duan Yongping: buy when others are fearful, but not into a falling knife.
    Look for stocks that are cheap AND starting to recover.
    """
    score = 50.0  # neutral baseline
    reasons = []

    # Distance from 52W high (beaten down = potential, but not too much)
    fh = stock.from_52w_high
    if fh is not None:
        if -5 <= fh <= 0:
            score += 10  # near high = strong
        elif -15 <= fh < -5:
            score += 15  # mild pullback = best entry
            reasons.append(f"{fh:+.0f}% from high: good entry")
        elif -30 <= fh < -15:
            score += 5   # significant drop, could be value or trap
        elif fh < -30:
            score -= 10  # severe, may be broken
            stock.flags.append(f"{fh:.0f}% from high: check thesis")

    # Position in 52W range
    rng = stock.pct_52w_range
    if rng is not None:
        if rng >= 70:
            score += 10  # trending up
        elif rng >= 40:
            score += 5   # middle
        elif rng < 20:
            score -= 5   # near lows

    # SMA200 position
    sma = stock.sma200_pct
    if sma is not None:
        if sma > 5:
            score += 10; reasons.append("above SMA200")
        elif sma > -5:
            score += 5  # near SMA200
        elif sma < -15:
            score -= 10; stock.flags.append("well below SMA200")

    # Beta preference (lower = more stable for long-term hold)
    if stock.beta is not None:
        if stock.beta < 0.8:
            score += 10; reasons.append(f"low beta={stock.beta:.1f}")
        elif stock.beta < 1.2:
            score += 5
        elif stock.beta > 1.5:
            score -= 5

    stock.reasons.extend(reasons)
    return max(0, min(score, 100.0))


def score_stock(stock: StockScore, weights: tuple[float, float, float] = (0.40, 0.40, 0.20)) -> StockScore:
    """Compute all scores for a stock.

    weights: (quality_weight, valuation_weight, momentum_weight)
    """
    stock.quality_score = score_quality(stock)
    stock.valuation_score = score_valuation(stock)
    stock.momentum_score = score_momentum(stock)
    stock.piotroski = compute_piotroski(
        stock.roe, stock.profit_margin, stock.fcf_yield,
        stock.debt_to_equity, stock.current_ratio,
        stock.revenue_growth, stock.earnings_growth,
    )
    stock.moat_score = stock.quality_score
    stock.financial_score = stock.piotroski / 9.0 * 100

    w_q, w_v, w_m = weights
    stock.composite_score = (
        w_q * stock.quality_score
        + w_v * stock.valuation_score
        + w_m * stock.momentum_score
    )
    return stock


@dataclass
class ValueScreenResult:
    """Result of value investing screen."""
    stocks: list[StockScore]
    n_screened: int
    n_passed_quality: int
    strategy_name: str = "Buffett-Duan Value"
    weights: tuple[float, float, float] = (0.40, 0.40, 0.20)

    @property
    def top_picks(self) -> list[StockScore]:
        return self.stocks[:10]

    @property
    def buy_candidates(self) -> list[StockScore]:
        """Stocks with composite > 60 and Piotroski >= 6."""
        return [s for s in self.stocks if s.composite_score >= 60 and s.piotroski >= 6]


def run_value_screen(
    stock_data: list[dict],
    quality_weight: float = 0.40,
    valuation_weight: float = 0.40,
    momentum_weight: float = 0.20,
    min_market_cap_b: float = 10.0,
    min_roe: float = 10.0,
    min_margin: float = 8.0,
    max_debt_equity: float = 300.0,
) -> ValueScreenResult:
    """Run the full value investing screen.

    Parameters
    ----------
    stock_data : list of dicts with fundamental data per stock
    quality/valuation/momentum_weight : scoring weights (must sum to 1)
    min_market_cap_b : minimum market cap in billions
    min_roe : minimum ROE % for quality gate
    min_margin : minimum profit margin % for quality gate
    max_debt_equity : maximum D/E ratio

    Returns
    -------
    ValueScreenResult with ranked stocks.
    """
    weights = (quality_weight, valuation_weight, momentum_weight)
    n_total = len(stock_data)
    scored: list[StockScore] = []

    for d in stock_data:
        # Build StockScore from dict
        s = StockScore(
            ticker=d.get("ticker", ""),
            name=d.get("name", ""),
            sector=d.get("sector", ""),
            price=d.get("price", 0),
            market_cap=d.get("market_cap_B", 0),
            roe=d.get("roe"),
            profit_margin=d.get("profit_margin"),
            debt_to_equity=d.get("debt_equity"),
            fcf_yield=d.get("fcf_yield"),
            revenue_growth=d.get("revenue_growth"),
            earnings_growth=d.get("earnings_growth"),
            trailing_pe=d.get("pe"),
            forward_pe=d.get("fwd_pe"),
            pb=d.get("pb"),
            ps=d.get("ps"),
            ev_ebitda=d.get("ev_ebitda"),
            from_52w_high=d.get("from_high"),
            pct_52w_range=d.get("pct_52w_range"),
            analyst_upside=d.get("upside"),
            beta=d.get("beta"),
        )

        # Quality gate
        if s.market_cap < min_market_cap_b:
            continue
        if s.roe is not None and s.roe < min_roe:
            continue
        if s.profit_margin is not None and s.profit_margin < min_margin:
            continue
        if s.debt_to_equity is not None and s.debt_to_equity > max_debt_equity:
            continue

        score_stock(s, weights)
        scored.append(s)

    # Sort by composite score
    scored.sort(key=lambda x: x.composite_score, reverse=True)

    # Assign ranks
    for i, s in enumerate(scored):
        pass  # rank is implicit in list position

    return ValueScreenResult(
        stocks=scored,
        n_screened=n_total,
        n_passed_quality=len(scored),
        strategy_name="Buffett-Duan Value",
        weights=weights,
    )
