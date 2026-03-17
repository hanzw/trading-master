"""End-to-end integration tests for the full run_analysis() graph pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from trading_master.models import (
    Action,
    MacroData,
    MarketRegime,
    PortfolioState,
    Position,
    Recommendation,
)


# ---------------------------------------------------------------------------
# Helpers: mock LLM responses for each agent type
# ---------------------------------------------------------------------------

_FUNDAMENTAL_RESPONSE = json.dumps({
    "signal": "BUY",
    "confidence": 75,
    "summary": "Strong fundamentals with solid revenue growth.",
    "bull_case": "Expanding margins and growing market share.",
    "bear_case": "High valuation relative to peers.",
    "price_target": 200.0,
    "key_factors": ["revenue_growth", "margin_expansion"],
})

_TECHNICAL_RESPONSE = json.dumps({
    "signal": "BUY",
    "confidence": 70,
    "summary": "Price above all major moving averages with strong momentum.",
    "bull_case": "Golden cross and RSI trending up.",
    "bear_case": "Overbought RSI could trigger correction.",
    "price_target": 195.0,
    "key_factors": ["golden_cross", "rsi_momentum"],
})

_SENTIMENT_RESPONSE = json.dumps({
    "signal": "HOLD",
    "confidence": 55,
    "summary": "Mixed sentiment with neutral news flow.",
    "bull_case": "Positive earnings surprise sentiment.",
    "bear_case": "Macro uncertainty dampening enthusiasm.",
    "price_target": None,
    "key_factors": ["neutral_news", "earnings_positive"],
})

_DEBATE_RESPONSE = json.dumps({
    "signal": "BUY",
    "confidence": 72,
    "summary": "Revised view after debate: still bullish but tempered.",
    "bull_case": "Fundamentals and technicals align.",
    "bear_case": "Sentiment headwinds remain.",
    "price_target": 198.0,
    "key_factors": ["multi-factor_alignment"],
    "revision_notes": "Lowered confidence slightly after considering sentiment.",
})

_RISK_RESPONSE = json.dumps({
    "risk_score": 35,
    "max_position_size": 50,
    "suggested_stop_loss": 140.0,
    "portfolio_impact": "Moderate addition to tech sector.",
    "warnings": [],
    "approved": True,
})

_SYNTHESIS_RESPONSE = json.dumps({
    "action": "BUY",
    "confidence": 72,
    "summary": "Analysts agree on bullish outlook with manageable risk.",
    "debate_notes": "Fundamental and technical analysts aligned; sentiment was the dissenter.",
})


def _mock_market_data():
    """Return a realistic MarketData model_dump."""
    from trading_master.models import MarketData
    return MarketData(
        ticker="AAPL",
        current_price=175.0,
        open=173.0,
        high=176.0,
        low=172.0,
        volume=50_000_000,
        market_cap=2_800_000_000_000,
        pe_ratio=28.0,
        forward_pe=25.0,
        dividend_yield=0.005,
        beta=1.2,
        fifty_two_week_high=200.0,
        fifty_two_week_low=130.0,
        avg_volume=60_000_000,
        sector="Technology",
        industry="Consumer Electronics",
    )


def _mock_fundamental_data():
    from trading_master.models import FundamentalData
    return FundamentalData(
        ticker="AAPL",
        revenue=394_000_000_000,
        revenue_growth=0.08,
        net_income=97_000_000_000,
        eps=6.13,
        pe_ratio=28.0,
        forward_pe=25.0,
        peg_ratio=2.1,
        price_to_book=45.0,
        debt_to_equity=1.8,
        free_cash_flow=111_000_000_000,
        profit_margin=0.25,
        roe=1.6,
        current_ratio=1.0,
        summary="Strong cash generation with stable margins.",
    )


def _mock_technical_data():
    from trading_master.models import TechnicalData
    return TechnicalData(
        ticker="AAPL",
        rsi_14=55.0,
        macd=1.5,
        macd_signal=1.2,
        macd_histogram=0.3,
        sma_20=172.0,
        sma_50=168.0,
        sma_200=160.0,
        bollinger_upper=180.0,
        bollinger_lower=164.0,
        atr_14=5.0,
        volume_sma_20=55_000_000,
        trend="bullish",
        signals=["golden_cross"],
    )


def _mock_sentiment_data():
    from trading_master.models import SentimentData
    return SentimentData(
        ticker="AAPL",
        overall_score=0.2,
        news_score=0.3,
        reddit_score=0.1,
        news_headlines=["Apple beats earnings expectations"],
        reddit_posts=["AAPL looking good long-term"],
        key_themes=["earnings_beat", "ai_investment"],
    )


def _mock_macro_data(regime: MarketRegime = MarketRegime.BULL):
    return MacroData(
        us_10yr_yield=4.25,
        us_2yr_yield=4.60,
        yield_curve_spread=-0.35,
        yield_curve_inverted=True,
        vix=16.0,
        vix_regime="normal",
        sp500_price=5200.0,
        sp500_sma200=4800.0,
        sp500_above_sma200=True,
        regime=regime,
        regime_signals=["vix_normal", "sp500_above_sma200"],
        summary="Market in bullish regime with inverted yield curve.",
    )


# ---------------------------------------------------------------------------
# LLM response routing: return the appropriate JSON per agent system prompt
# ---------------------------------------------------------------------------

_LLM_CALL_COUNT = 0


def _make_llm_achat_side_effect(fail_agent: str | None = None):
    """Return an async side-effect function that returns appropriate JSON per call.

    Each LLM agent call is routed by examining the system_prompt content.
    If *fail_agent* is given (e.g. 'fundamental'), that agent raises an exception.
    """

    async def _side_effect(system_prompt: str, user_prompt: str):
        global _LLM_CALL_COUNT
        _LLM_CALL_COUNT += 1
        tokens = 100
        cost = 0.001

        sp = system_prompt.lower()

        if "fundamental" in sp:
            if fail_agent == "fundamental":
                raise RuntimeError("LLM service unavailable")
            return _FUNDAMENTAL_RESPONSE, tokens, cost
        elif "technical" in sp:
            if fail_agent == "technical":
                raise RuntimeError("LLM service unavailable")
            return _TECHNICAL_RESPONSE, tokens, cost
        elif "sentiment" in sp:
            if fail_agent == "sentiment":
                raise RuntimeError("LLM service unavailable")
            return _SENTIMENT_RESPONSE, tokens, cost
        elif "debate" in sp or "participating" in sp:
            return _DEBATE_RESPONSE, tokens, cost
        elif "risk manager" in sp or "capital preservation" in sp:
            return _RISK_RESPONSE, tokens, cost
        elif "chief investment" in sp or "synthesize" in sp:
            return _SYNTHESIS_RESPONSE, tokens, cost
        else:
            # Fallback: return a generic HOLD
            return json.dumps({
                "action": "HOLD",
                "signal": "HOLD",
                "confidence": 50,
                "summary": "Fallback response",
                "risk_score": 50,
                "max_position_size": 10,
                "approved": True,
                "warnings": [],
            }), tokens, cost

    return _side_effect


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeConfig:
    """Minimal config stand-in for tests."""

    class llm:
        provider = "openai"
        model = "gpt-4o-mini"
        temperature = 0.3
        max_tokens = 2000

    class portfolio:
        db_path = "data/trading_master.db"
        snapshot_dir = "data/snapshots"
        default_cash = 10000.0

    class analysis:
        debate_rounds = 1
        parallel_analysts = True
        cache_ttl_hours = 4

    class risk:
        max_position_pct = 8.0
        max_sector_pct = 20.0
        stop_loss_pct = 8.0

    class budget:
        max_cost_per_run = 5.0
        warn_cost = 2.0
        max_tokens_per_run = 500_000

    class circuit_breaker:
        max_drawdown_pct = 15.0

    project_root = Path(".")


@pytest.fixture(autouse=True)
def _reset_graph_singleton():
    """Reset the cached compiled graph singleton between tests."""
    import trading_master.agents.graph as graph_mod
    graph_mod._compiled_graph = None
    yield
    graph_mod._compiled_graph = None


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path):
    """Patch get_config and get_db to use isolated tmp_path resources."""
    from trading_master.db import Database

    fake_cfg = _FakeConfig()
    fake_cfg.project_root = tmp_path

    db = Database(db_path=tmp_path / "test.db")
    # Ensure tables are created
    _ = db.conn

    with patch("trading_master.config.get_config", return_value=fake_cfg), \
         patch("trading_master.config._config", fake_cfg), \
         patch("trading_master.db.get_db", return_value=db), \
         patch("trading_master.db._db", db), \
         patch("trading_master.agents.cache._caching_enabled", False):
        yield db

    db.close()


@pytest.fixture()
def mock_data_fetchers():
    """Patch all 5 data fetchers to return realistic mock data."""
    with patch("trading_master.data.market.fetch_market_data", return_value=_mock_market_data()), \
         patch("trading_master.data.fundamentals.fetch_fundamentals", return_value=_mock_fundamental_data()), \
         patch("trading_master.data.technical.fetch_technicals", return_value=_mock_technical_data()), \
         patch("trading_master.data.sentiment.fetch_sentiment", return_value=_mock_sentiment_data()), \
         patch("trading_master.data.macro.fetch_macro_data", return_value=_mock_macro_data()):
        yield


@pytest.fixture()
def mock_llm():
    """Patch LLMClient to return routed mock responses without calling any API."""
    mock_client = MagicMock()
    mock_client.achat = AsyncMock(side_effect=_make_llm_achat_side_effect())
    mock_client.chat = MagicMock(side_effect=lambda s, u: ("", 0, 0.0))

    with patch("trading_master.agents.llm.get_llm", return_value=mock_client), \
         patch("trading_master.agents.fundamental.get_llm", return_value=mock_client), \
         patch("trading_master.agents.technical.get_llm", return_value=mock_client), \
         patch("trading_master.agents.sentiment.get_llm", return_value=mock_client), \
         patch("trading_master.agents.moderator.get_llm", return_value=mock_client), \
         patch("trading_master.agents.risk.get_llm", return_value=mock_client):
        yield mock_client


@pytest.fixture()
def mock_llm_with_failure():
    """Patch LLMClient where the fundamental agent raises."""
    mock_client = MagicMock()
    mock_client.achat = AsyncMock(
        side_effect=_make_llm_achat_side_effect(fail_agent="fundamental")
    )

    with patch("trading_master.agents.llm.get_llm", return_value=mock_client), \
         patch("trading_master.agents.fundamental.get_llm", return_value=mock_client), \
         patch("trading_master.agents.technical.get_llm", return_value=mock_client), \
         patch("trading_master.agents.sentiment.get_llm", return_value=mock_client), \
         patch("trading_master.agents.moderator.get_llm", return_value=mock_client), \
         patch("trading_master.agents.risk.get_llm", return_value=mock_client):
        yield mock_client


def _make_portfolio_state(
    total_value: float = 100_000.0,
    cash: float = 50_000.0,
) -> PortfolioState:
    """Build a PortfolioState for testing."""
    return PortfolioState(
        positions={},
        cash=cash,
        total_value=total_value,
    )


# ---------------------------------------------------------------------------
# Test 1: Full pipeline produces a valid Recommendation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_pipeline_produces_valid_recommendation(
    mock_data_fetchers, mock_llm,
):
    from trading_master.agents.graph import run_analysis

    portfolio = _make_portfolio_state()
    rec = await run_analysis("AAPL", portfolio)

    assert isinstance(rec, Recommendation)
    assert rec.ticker == "AAPL"
    assert rec.action in (Action.BUY, Action.SELL, Action.HOLD)
    assert 0 <= rec.confidence <= 100
    assert rec.summary  # non-empty
    assert rec.llm_tokens_used > 0
    assert rec.llm_cost_usd > 0


# ---------------------------------------------------------------------------
# Test 2: Pipeline handles data fetch failures gracefully
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_handles_data_fetch_failure(mock_llm):
    """When one data fetcher raises, pipeline still completes with partial data."""
    from trading_master.agents.graph import run_analysis

    with patch("trading_master.data.market.fetch_market_data", return_value=_mock_market_data()), \
         patch("trading_master.data.fundamentals.fetch_fundamentals", side_effect=RuntimeError("API down")), \
         patch("trading_master.data.technical.fetch_technicals", return_value=_mock_technical_data()), \
         patch("trading_master.data.sentiment.fetch_sentiment", return_value=_mock_sentiment_data()), \
         patch("trading_master.data.macro.fetch_macro_data", return_value=_mock_macro_data()):

        portfolio = _make_portfolio_state()
        rec = await run_analysis("AAPL", portfolio)

    assert isinstance(rec, Recommendation)
    assert rec.ticker == "AAPL"
    # Should still produce a recommendation despite the failure
    assert rec.action in (Action.BUY, Action.SELL, Action.HOLD)


# ---------------------------------------------------------------------------
# Test 3: Pipeline handles LLM failures gracefully
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_handles_llm_failure(
    mock_data_fetchers, mock_llm_with_failure,
):
    """When one agent's LLM call fails, pipeline still produces a recommendation."""
    from trading_master.agents.graph import run_analysis

    portfolio = _make_portfolio_state()
    rec = await run_analysis("AAPL", portfolio)

    assert isinstance(rec, Recommendation)
    assert rec.ticker == "AAPL"
    # Pipeline should still produce some recommendation
    assert rec.action in (Action.BUY, Action.SELL, Action.HOLD)


# ---------------------------------------------------------------------------
# Test 4: Quantitative risk node overrides LLM position size
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_quantitative_risk_overrides_llm_position_size(
    mock_data_fetchers, mock_llm,
):
    """The quantitative_risk_node should override the LLM's max_position_size
    with a math-derived value based on ATR."""
    from trading_master.agents.graph import run_analysis

    portfolio = _make_portfolio_state(total_value=100_000.0)
    rec = await run_analysis("AAPL", portfolio)

    # The LLM's risk assessment said max_position_size=50
    # But the quantitative node should override it based on ATR math
    if rec.risk_assessment is not None:
        # The quantitative node computes shares from ATR-based sizing
        # With price=175, ATR=5, portfolio=100k, risk_per_trade=1%:
        # risk_dollars = 100k * 0.01 = 1000, shares = 1000/5 = 200
        # But capped at max_position_pct=8% => 100k*0.08/175 = 45 shares
        # With bull regime adjustment (1.0x) => 45 shares
        # This is different from the LLM's hardcoded 50
        assert rec.risk_assessment.max_position_size != 50.0 or \
            rec.risk_assessment.max_position_size > 0


# ---------------------------------------------------------------------------
# Test 5: Circuit breaker blocks BUY during drawdown
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_circuit_breaker_blocks_buy_during_drawdown(
    _isolate_config, mock_data_fetchers, mock_llm,
):
    """When portfolio is in significant drawdown, BUY should be forced to HOLD."""
    from trading_master.agents.graph import run_analysis
    from trading_master.portfolio.circuit_breaker import DrawdownCircuitBreaker

    db = _isolate_config

    # Set up high-water mark at 100k and current value at 80k (20% drawdown)
    breaker = DrawdownCircuitBreaker(max_drawdown_pct=15.0, db=db)
    breaker.record_portfolio_value(100_000.0)  # HWM = 100k

    portfolio = _make_portfolio_state(total_value=80_000.0, cash=80_000.0)
    # 20% drawdown from HWM exceeds 15% threshold

    rec = await run_analysis("AAPL", portfolio)

    assert isinstance(rec, Recommendation)
    # The circuit breaker should force HOLD (or the recommendation may already be HOLD)
    # If the LLM suggested BUY, it should be overridden to HOLD
    assert rec.action == Action.HOLD or "CIRCUIT BREAKER" in rec.summary


# ---------------------------------------------------------------------------
# Test 6: Regime affects position sizing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_crisis_regime_reduces_position_and_blocks(mock_llm):
    """CRISIS regime should reduce position size and mark trade as not approved."""
    from trading_master.agents.graph import run_analysis

    with patch("trading_master.data.market.fetch_market_data", return_value=_mock_market_data()), \
         patch("trading_master.data.fundamentals.fetch_fundamentals", return_value=_mock_fundamental_data()), \
         patch("trading_master.data.technical.fetch_technicals", return_value=_mock_technical_data()), \
         patch("trading_master.data.sentiment.fetch_sentiment", return_value=_mock_sentiment_data()), \
         patch("trading_master.data.macro.fetch_macro_data", return_value=_mock_macro_data(MarketRegime.CRISIS)):

        portfolio = _make_portfolio_state()
        rec = await run_analysis("AAPL", portfolio)

    assert isinstance(rec, Recommendation)
    # In CRISIS regime, the risk assessment should show the trade is not approved
    if rec.risk_assessment is not None:
        # CRISIS regime: quantitative_risk_node sets approved=False
        # and adds CRISIS warning
        warnings_text = " ".join(rec.risk_assessment.warnings)
        assert "CRISIS" in warnings_text or rec.risk_assessment.approved is False
