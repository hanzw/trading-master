"""Tests for Pydantic models."""

from trading_master.models import (
    Action,
    ActionRecord,
    ActionSource,
    AnalysisState,
    AnalystReport,
    FundamentalData,
    MarketData,
    PortfolioState,
    Position,
    Recommendation,
    RiskAssessment,
    SentimentData,
    Signal,
    TechnicalData,
)


def test_position_update_market():
    pos = Position(ticker="AAPL", quantity=10, avg_cost=150.0)
    pos.update_market(175.0)
    assert pos.current_price == 175.0
    assert pos.market_value == 1750.0
    assert pos.unrealized_pnl == 250.0
    assert round(pos.pnl_pct, 2) == 16.67


def test_position_update_market_loss():
    pos = Position(ticker="TSLA", quantity=5, avg_cost=200.0)
    pos.update_market(180.0)
    assert pos.unrealized_pnl == -100.0
    assert pos.pnl_pct < 0


def test_portfolio_state_recalculate():
    p1 = Position(ticker="AAPL", quantity=10, avg_cost=150.0, market_value=1750.0)
    p2 = Position(ticker="MSFT", quantity=5, avg_cost=400.0, market_value=2100.0)
    state = PortfolioState(positions={"AAPL": p1, "MSFT": p2}, cash=5000.0)
    state.recalculate()
    assert state.total_value == 1750.0 + 2100.0 + 5000.0


def test_action_record_defaults():
    record = ActionRecord(ticker="AAPL", action=Action.BUY, quantity=10, price=150.0)
    assert record.source == ActionSource.MANUAL
    assert record.reasoning == ""
    assert record.id is None


def test_analyst_report():
    report = AnalystReport(
        analyst="fundamental",
        signal=Signal.BUY,
        confidence=75.0,
        summary="Strong fundamentals",
        bull_case="Revenue growth",
        bear_case="High valuation",
        key_factors=["EPS beat", "Revenue growth"],
    )
    assert report.analyst == "fundamental"
    assert report.signal == Signal.BUY
    assert not report.revised


def test_recommendation():
    rec = Recommendation(
        ticker="AAPL",
        action=Action.BUY,
        confidence=80.0,
        summary="Consensus buy",
    )
    assert rec.action == Action.BUY
    assert rec.llm_tokens_used == 0
    assert rec.analyst_reports == []


def test_analysis_state():
    state = AnalysisState(ticker="AAPL")
    assert state.market_data is None
    assert state.analyst_reports == []
    assert state.errors == []
    assert state.total_tokens == 0


def test_market_data_defaults():
    md = MarketData(ticker="AAPL")
    assert md.current_price == 0.0
    assert md.sector == ""


def test_fundamental_data_defaults():
    fd = FundamentalData(ticker="AAPL")
    assert fd.revenue is None
    assert fd.summary == ""


def test_technical_data_defaults():
    td = TechnicalData(ticker="AAPL")
    assert td.rsi_14 is None
    assert td.trend == ""
    assert td.signals == []


def test_sentiment_data_defaults():
    sd = SentimentData(ticker="AAPL")
    assert sd.overall_score == 0.0
    assert sd.key_themes == []


def test_risk_assessment_defaults():
    ra = RiskAssessment(risk_score=50.0)
    assert ra.approved is True
    assert ra.warnings == []


def test_signal_enum():
    assert Signal.STRONG_BUY.value == "STRONG_BUY"
    assert Signal("SELL") == Signal.SELL


def test_action_enum():
    assert Action.BUY.value == "BUY"
    assert Action("HOLD") == Action.HOLD
