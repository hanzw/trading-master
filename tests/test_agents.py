"""Tests for agent modules (mocking LLM calls)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from trading_master.models import (
    Action,
    AnalysisState,
    AnalystReport,
    FundamentalData,
    MarketData,
    PortfolioState,
    Recommendation,
    RiskAssessment,
    SentimentData,
    Signal,
    TechnicalData,
)


def _mock_llm_response(data: dict) -> tuple[str, int, float]:
    return json.dumps(data), 100, 0.001


def _async_mock_return(value):
    """Return an async function that returns the given value (for mocking achat)."""
    async def _mock(*args, **kwargs):
        return value
    return _mock


def _async_mock_raise(exc):
    """Return an async function that raises the given exception."""
    async def _mock(*args, **kwargs):
        raise exc
    return _mock


@pytest.fixture
def analysis_state():
    return AnalysisState(
        ticker="AAPL",
        market_data=MarketData(
            ticker="AAPL", current_price=175.0, market_cap=2.8e12,
            pe_ratio=28.0, sector="Technology",
        ),
        fundamental_data=FundamentalData(
            ticker="AAPL", revenue=394e9, eps=6.5, pe_ratio=28.0,
        ),
        technical_data=TechnicalData(
            ticker="AAPL", rsi_14=55.0, trend="bullish",
            sma_20=172.0, sma_50=168.0, sma_200=160.0,
        ),
        sentiment_data=SentimentData(
            ticker="AAPL", overall_score=0.3, news_score=0.4,
            key_themes=["AI", "iPhone"],
        ),
    )


@pytest.mark.asyncio
async def test_analyze_fundamental(analysis_state):
    mock_response = _mock_llm_response({
        "signal": "BUY",
        "confidence": 75,
        "summary": "Strong fundamentals",
        "bull_case": "Revenue growth",
        "bear_case": "High valuation",
        "price_target": 200.0,
        "key_factors": ["EPS beat", "Margin expansion"],
    })

    with patch("trading_master.agents.fundamental.get_llm") as mock:
        mock.return_value.achat = _async_mock_return(mock_response)
        from trading_master.agents.fundamental import analyze_fundamental
        report = await analyze_fundamental(analysis_state)

    assert report.analyst == "fundamental"
    assert report.signal == Signal.BUY
    assert report.confidence == 75
    assert report.price_target == 200.0


@pytest.mark.asyncio
async def test_analyze_technical(analysis_state):
    mock_response = _mock_llm_response({
        "signal": "BUY",
        "confidence": 65,
        "summary": "Bullish trend",
        "bull_case": "Above all SMAs",
        "bear_case": "RSI approaching overbought",
        "price_target": 190.0,
        "key_factors": ["Golden cross", "Volume increasing"],
    })

    with patch("trading_master.agents.technical.get_llm") as mock:
        mock.return_value.achat = _async_mock_return(mock_response)
        from trading_master.agents.technical import analyze_technical
        report = await analyze_technical(analysis_state)

    assert report.analyst == "technical"
    assert report.signal == Signal.BUY


@pytest.mark.asyncio
async def test_analyze_sentiment(analysis_state):
    mock_response = _mock_llm_response({
        "signal": "HOLD",
        "confidence": 55,
        "summary": "Mixed sentiment",
        "bull_case": "AI hype",
        "bear_case": "Market uncertainty",
        "price_target": None,
        "key_factors": ["AI narrative", "Mixed news"],
    })

    with patch("trading_master.agents.sentiment.get_llm") as mock:
        mock.return_value.achat = _async_mock_return(mock_response)
        from trading_master.agents.sentiment import analyze_sentiment
        report = await analyze_sentiment(analysis_state)

    assert report.analyst == "sentiment"
    assert report.signal == Signal.HOLD


@pytest.mark.asyncio
async def test_assess_risk(analysis_state):
    analysis_state.analyst_reports = [
        AnalystReport(analyst="fundamental", signal=Signal.BUY, confidence=75, summary="Strong"),
        AnalystReport(analyst="technical", signal=Signal.BUY, confidence=65, summary="Bullish"),
    ]
    analysis_state.portfolio_state = PortfolioState(cash=10000.0, total_value=10000.0)

    mock_response = _mock_llm_response({
        "risk_score": 35,
        "max_position_size": 20,
        "suggested_stop_loss": 157.5,
        "portfolio_impact": "Moderate increase in tech exposure",
        "warnings": [],
        "approved": True,
    })

    with patch("trading_master.agents.risk.get_llm") as mock:
        mock.return_value.achat = _async_mock_return(mock_response)
        from trading_master.agents.risk import assess_risk
        assessment = await assess_risk(analysis_state)

    assert assessment.risk_score == 35
    assert assessment.approved is True
    assert assessment.suggested_stop_loss == 157.5


@pytest.mark.asyncio
async def test_run_debate(analysis_state):
    analysis_state.analyst_reports = [
        AnalystReport(analyst="fundamental", signal=Signal.BUY, confidence=75, summary="Strong"),
        AnalystReport(analyst="technical", signal=Signal.HOLD, confidence=55, summary="Neutral"),
    ]

    mock_response = _mock_llm_response({
        "signal": "BUY",
        "confidence": 70,
        "summary": "Revised after debate",
        "bull_case": "Fundamentals support",
        "bear_case": "Technical caution",
        "price_target": 190.0,
        "key_factors": ["Combined analysis"],
        "revision_notes": "Adjusted confidence based on technicals",
    })

    with patch("trading_master.agents.moderator.get_llm") as mock:
        mock.return_value.achat = _async_mock_return(mock_response)
        from trading_master.agents.moderator import run_debate
        revised = await run_debate(analysis_state)

    assert len(revised) == 2
    assert all(r.revised for r in revised)


@pytest.mark.asyncio
async def test_synthesize(analysis_state):
    analysis_state.debate_reports = [
        AnalystReport(analyst="fundamental", signal=Signal.BUY, confidence=75, summary="Strong", revised=True),
        AnalystReport(analyst="technical", signal=Signal.BUY, confidence=70, summary="Bullish", revised=True),
    ]
    analysis_state.risk_assessment = RiskAssessment(
        risk_score=35, max_position_size=20, approved=True,
    )

    mock_response = _mock_llm_response({
        "action": "BUY",
        "confidence": 75,
        "summary": "Consensus buy recommendation",
        "debate_notes": "Both analysts agree after debate",
    })

    with patch("trading_master.agents.moderator.get_llm") as mock:
        mock.return_value.achat = _async_mock_return(mock_response)
        from trading_master.agents.moderator import synthesize
        rec = await synthesize(analysis_state)

    assert rec.ticker == "AAPL"
    assert rec.action == Action.BUY
    assert rec.confidence == 75


@pytest.mark.asyncio
async def test_analyst_handles_llm_failure(analysis_state):
    with patch("trading_master.agents.fundamental.get_llm") as mock:
        mock.return_value.achat = _async_mock_raise(Exception("API error"))
        from trading_master.agents.fundamental import analyze_fundamental
        report = await analyze_fundamental(analysis_state)

    assert report.signal == Signal.HOLD
    assert report.confidence == 0.0
    assert "failed" in report.summary.lower()


@pytest.mark.asyncio
async def test_analyst_handles_bad_json(analysis_state):
    with patch("trading_master.agents.fundamental.get_llm") as mock:
        mock.return_value.achat = _async_mock_return(("not valid json at all", 50, 0.001))
        from trading_master.agents.fundamental import analyze_fundamental
        report = await analyze_fundamental(analysis_state)

    assert report.signal == Signal.HOLD
    assert report.confidence == 0.0
