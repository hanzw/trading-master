"""Pydantic models for the entire system."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ── Enums ──────────────────────────────────────────────────────────────

class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class ActionSource(str, Enum):
    MANUAL = "manual"
    ROBINHOOD = "robinhood"
    CSV_IMPORT = "csv_import"
    EXTERNAL = "external"


class AssetClass(str, Enum):
    US_EQUITY = "us_equity"
    INTL_EQUITY = "intl_equity"
    FIXED_INCOME = "fixed_income"
    SHORT_TERM_TREASURY = "short_term_treasury"
    REITS = "reits"
    COMMODITIES = "commodities"
    ALTERNATIVES = "alternatives"
    CASH = "cash"


class MarketRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"


class Signal(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


# ── Allocation ────────────────────────────────────────────────────────

class AllocationTarget(BaseModel):
    asset_class: AssetClass
    target_pct: float
    min_pct: float
    max_pct: float
    current_pct: float = 0.0
    drift_pct: float = 0.0  # current - target


class AllocationModel(BaseModel):
    name: str  # "balanced", "growth", "conservative"
    targets: list[AllocationTarget]
    rebalance_threshold_pct: float = 5.0  # rebalance if any class drifts > this


# ── Portfolio ──────────────────────────────────────────────────────────

class Position(BaseModel):
    ticker: str
    quantity: float
    avg_cost: float
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    pnl_pct: float = 0.0
    sector: str = ""

    def update_market(self, price: float) -> None:
        self.current_price = price
        self.market_value = price * self.quantity
        cost_basis = self.avg_cost * self.quantity
        self.unrealized_pnl = self.market_value - cost_basis
        self.pnl_pct = (self.unrealized_pnl / cost_basis * 100) if cost_basis else 0.0


class PortfolioState(BaseModel):
    positions: dict[str, Position] = Field(default_factory=dict)
    cash: float = 10000.0
    total_value: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)

    def recalculate(self) -> None:
        positions_value = sum(p.market_value for p in self.positions.values())
        self.total_value = positions_value + self.cash


class ActionRecord(BaseModel):
    id: int | None = None
    ticker: str
    action: Action
    quantity: float
    price: float
    timestamp: datetime = Field(default_factory=datetime.now)
    source: ActionSource = ActionSource.MANUAL
    reasoning: str = ""
    portfolio_before: dict[str, Any] = Field(default_factory=dict)
    portfolio_after: dict[str, Any] = Field(default_factory=dict)


# ── Data ───────────────────────────────────────────────────────────────

class MarketData(BaseModel):
    ticker: str
    current_price: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: int = 0
    market_cap: float = 0.0
    pe_ratio: float | None = None
    forward_pe: float | None = None
    dividend_yield: float | None = None
    beta: float | None = None
    fifty_two_week_high: float = 0.0
    fifty_two_week_low: float = 0.0
    avg_volume: int = 0
    sector: str = ""
    industry: str = ""


class FundamentalData(BaseModel):
    ticker: str
    revenue: float | None = None
    revenue_growth: float | None = None
    net_income: float | None = None
    eps: float | None = None
    pe_ratio: float | None = None
    forward_pe: float | None = None
    peg_ratio: float | None = None
    price_to_book: float | None = None
    debt_to_equity: float | None = None
    free_cash_flow: float | None = None
    profit_margin: float | None = None
    roe: float | None = None
    current_ratio: float | None = None
    summary: str = ""


class TechnicalData(BaseModel):
    ticker: str
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    sma_20: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    bollinger_upper: float | None = None
    bollinger_lower: float | None = None
    atr_14: float | None = None
    volume_sma_20: float | None = None
    trend: str = ""  # "bullish", "bearish", "neutral"
    signals: list[str] = Field(default_factory=list)


class SentimentData(BaseModel):
    ticker: str
    overall_score: float = 0.0  # -1 to +1
    news_score: float = 0.0
    reddit_score: float = 0.0
    news_headlines: list[str] = Field(default_factory=list)
    reddit_posts: list[str] = Field(default_factory=list)
    key_themes: list[str] = Field(default_factory=list)


class MacroData(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    # Rates (from yfinance tickers)
    us_10yr_yield: float | None = None       # ^TNX
    us_2yr_yield: float | None = None        # ^IRX proxy or 2yr
    yield_curve_spread: float | None = None  # 10yr - 2yr
    yield_curve_inverted: bool = False
    # Volatility
    vix: float | None = None                 # ^VIX
    vix_regime: str = "normal"               # "low" (<15), "normal" (15-25), "high" (25-35), "extreme" (>35)
    # Market breadth (from SPY)
    sp500_price: float | None = None
    sp500_sma200: float | None = None
    sp500_above_sma200: bool = True
    # Regime
    regime: MarketRegime = MarketRegime.SIDEWAYS
    regime_signals: list[str] = Field(default_factory=list)
    summary: str = ""


# ── Dividends ─────────────────────────────────────────────────────────

class DividendInfo(BaseModel):
    ticker: str
    annual_dividend: float = 0.0
    dividend_yield: float = 0.0
    payout_ratio: float | None = None
    dividend_growth_rate_5yr: float | None = None
    consecutive_increase_years: int = 0
    ex_dividend_date: str | None = None
    sustainability_score: float = 0.0  # 0-100


# ── Backtest ──────────────────────────────────────────────────────────

class BacktestResult(BaseModel):
    total_recommendations: int = 0
    hit_rate_30d: float | None = None
    hit_rate_90d: float | None = None
    hit_rate_180d: float | None = None
    avg_return_90d: float | None = None
    agent_accuracy: dict[str, dict] = Field(default_factory=dict)
    calibration: list[dict] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.now)


# ── Agent ──────────────────────────────────────────────────────────────

class AnalystReport(BaseModel):
    analyst: str  # "fundamental", "technical", "sentiment"
    signal: Signal
    confidence: float  # 0-100
    summary: str
    bull_case: str = ""
    bear_case: str = ""
    price_target: float | None = None
    key_factors: list[str] = Field(default_factory=list)
    revised: bool = False  # True after debate round
    revision_notes: str = ""


class RiskAssessment(BaseModel):
    risk_score: float  # 0-100 (higher = riskier)
    max_position_size: float = 0.0  # shares
    suggested_stop_loss: float | None = None
    portfolio_impact: str = ""
    warnings: list[str] = Field(default_factory=list)
    approved: bool = True


class Recommendation(BaseModel):
    id: int | None = None
    ticker: str
    action: Action
    confidence: float  # 0-100
    summary: str
    analyst_reports: list[AnalystReport] = Field(default_factory=list)
    risk_assessment: RiskAssessment | None = None
    debate_notes: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    llm_tokens_used: int = 0
    llm_cost_usd: float = 0.0


# ── LangGraph State ───────────────────────────────────────────────────

class AnalysisState(BaseModel):
    """State object flowing through the LangGraph."""
    ticker: str
    market_data: MarketData | None = None
    fundamental_data: FundamentalData | None = None
    technical_data: TechnicalData | None = None
    sentiment_data: SentimentData | None = None
    macro_data: MacroData | None = None
    analyst_reports: list[AnalystReport] = Field(default_factory=list)
    debate_reports: list[AnalystReport] = Field(default_factory=list)
    risk_assessment: RiskAssessment | None = None
    recommendation: Recommendation | None = None
    portfolio_state: PortfolioState | None = None
    errors: list[str] = Field(default_factory=list)
    total_tokens: int = 0
    total_cost: float = 0.0

    model_config = ConfigDict(arbitrary_types_allowed=True)
