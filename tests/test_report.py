"""Tests for Rich console report output (output/report.py)."""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from trading_master.models import (
    Action,
    AnalystReport,
    Position,
    PortfolioState,
    Recommendation,
    RiskAssessment,
    Signal,
)
from trading_master.output import report


# ── Helpers ─────────────────────────────────────────────────────────


def _capture_output(func, *args, **kwargs) -> str:
    """Capture Rich console output as plain text."""
    buf = StringIO()
    test_console = Console(file=buf, force_terminal=True, width=120)
    original = report.console
    report.console = test_console
    try:
        func(*args, **kwargs)
    finally:
        report.console = original
    return buf.getvalue()


def _make_recommendation(**overrides) -> Recommendation:
    defaults = dict(
        ticker="AAPL",
        action=Action.BUY,
        confidence=75.0,
        summary="Strong fundamentals and bullish technicals.",
        llm_tokens_used=5000,
        llm_cost_usd=0.0150,
    )
    defaults.update(overrides)
    return Recommendation(**defaults)


def _make_portfolio() -> PortfolioState:
    return PortfolioState(
        positions={
            "AAPL": Position(
                ticker="AAPL", quantity=50, avg_cost=150.0, current_price=170.0,
                market_value=8500.0, unrealized_pnl=1000.0, pnl_pct=13.3, sector="Tech",
            ),
            "MSFT": Position(
                ticker="MSFT", quantity=30, avg_cost=300.0, current_price=280.0,
                market_value=8400.0, unrealized_pnl=-600.0, pnl_pct=-6.7, sector="Tech",
            ),
        },
        cash=3100.0,
        total_value=20000.0,
    )


# ── _confidence_bar ────────────────────────────────────────────────


class TestConfidenceBar:
    def test_high_confidence(self):
        bar = report._confidence_bar(90.0)
        text = bar.plain
        assert "90%" in text

    def test_medium_confidence(self):
        bar = report._confidence_bar(50.0)
        text = bar.plain
        assert "50%" in text

    def test_low_confidence(self):
        bar = report._confidence_bar(20.0)
        text = bar.plain
        assert "20%" in text

    def test_zero_confidence(self):
        bar = report._confidence_bar(0.0)
        text = bar.plain
        assert "0%" in text

    def test_full_confidence(self):
        bar = report._confidence_bar(100.0)
        text = bar.plain
        assert "100%" in text


# ── _pnl_style ────────────────────────────────────────────────────


class TestPnlStyle:
    def test_positive(self):
        assert report._pnl_style(100.0) == "green"

    def test_negative(self):
        assert report._pnl_style(-50.0) == "red"

    def test_zero(self):
        assert report._pnl_style(0.0) == "dim"


# ── print_recommendation ──────────────────────────────────────────


class TestPrintRecommendation:
    def test_shows_ticker(self):
        rec = _make_recommendation()
        output = _capture_output(report.print_recommendation, rec)
        assert "AAPL" in output

    def test_shows_action(self):
        rec = _make_recommendation(action=Action.BUY)
        output = _capture_output(report.print_recommendation, rec)
        assert "BUY" in output

    def test_shows_sell_action(self):
        rec = _make_recommendation(action=Action.SELL)
        output = _capture_output(report.print_recommendation, rec)
        assert "SELL" in output

    def test_shows_hold_action(self):
        rec = _make_recommendation(action=Action.HOLD)
        output = _capture_output(report.print_recommendation, rec)
        assert "HOLD" in output

    def test_shows_confidence(self):
        rec = _make_recommendation(confidence=85.0)
        output = _capture_output(report.print_recommendation, rec)
        assert "85%" in output

    def test_shows_summary(self):
        rec = _make_recommendation(summary="Test summary text.")
        output = _capture_output(report.print_recommendation, rec)
        assert "Test summary text" in output

    def test_shows_analyst_reports(self):
        rec = _make_recommendation(
            analyst_reports=[
                AnalystReport(
                    analyst="fundamental",
                    signal=Signal.BUY,
                    confidence=80.0,
                    summary="Revenue growth strong.",
                    key_factors=["revenue", "margins"],
                ),
            ]
        )
        output = _capture_output(report.print_recommendation, rec)
        assert "fundamental" in output
        assert "BUY" in output

    def test_shows_risk_assessment(self):
        rec = _make_recommendation(
            risk_assessment=RiskAssessment(
                risk_score=65.0,
                max_position_size=100,
                suggested_stop_loss=155.0,
                portfolio_impact="Low impact",
                approved=True,
                warnings=["High volatility"],
            ),
        )
        output = _capture_output(report.print_recommendation, rec)
        assert "65" in output
        assert "100" in output
        assert "$155.00" in output
        assert "APPROVED" in output
        assert "High volatility" in output

    def test_shows_rejected_risk(self):
        rec = _make_recommendation(
            risk_assessment=RiskAssessment(
                risk_score=90.0,
                approved=False,
                warnings=["CVaR exceeded"],
            ),
        )
        output = _capture_output(report.print_recommendation, rec)
        assert "REJECTED" in output

    def test_shows_debate_notes(self):
        rec = _make_recommendation(debate_notes="Analysts debated valuations.")
        output = _capture_output(report.print_recommendation, rec)
        assert "debated" in output

    def test_shows_cost_info(self):
        rec = _make_recommendation(llm_tokens_used=10000, llm_cost_usd=0.05)
        output = _capture_output(report.print_recommendation, rec)
        assert "10,000" in output
        assert "$0.0500" in output

    def test_no_crash_minimal_recommendation(self):
        rec = _make_recommendation(
            analyst_reports=[],
            risk_assessment=None,
            debate_notes="",
            llm_tokens_used=0,
            llm_cost_usd=0.0,
        )
        output = _capture_output(report.print_recommendation, rec)
        assert "AAPL" in output


# ── print_portfolio ────────────────────────────────────────────────


class TestPrintPortfolio:
    def test_shows_tickers(self):
        state = _make_portfolio()
        output = _capture_output(report.print_portfolio, state)
        assert "AAPL" in output
        assert "MSFT" in output

    def test_shows_cash(self):
        state = _make_portfolio()
        output = _capture_output(report.print_portfolio, state)
        assert "CASH" in output
        assert "$3,100.00" in output

    def test_shows_total(self):
        state = _make_portfolio()
        output = _capture_output(report.print_portfolio, state)
        assert "$20,000.00" in output
        assert "TOTAL" in output

    def test_shows_pnl(self):
        state = _make_portfolio()
        output = _capture_output(report.print_portfolio, state)
        assert "1,000.00" in output
        assert "13.3%" in output

    def test_empty_portfolio(self):
        state = PortfolioState(positions={}, cash=10000, total_value=10000)
        output = _capture_output(report.print_portfolio, state)
        assert "CASH" in output
        assert "TOTAL" in output


# ── print_actions ──────────────────────────────────────────────────


class TestPrintActions:
    def test_empty_actions(self):
        output = _capture_output(report.print_actions, [])
        assert "No actions found" in output

    def test_shows_buy_action(self):
        actions = [{
            "timestamp": "2024-01-15 10:30:00",
            "ticker": "AAPL",
            "action": "BUY",
            "quantity": 50,
            "price": 150.0,
            "source": "manual",
            "reasoning": "Strong fundamentals",
        }]
        output = _capture_output(report.print_actions, actions)
        assert "AAPL" in output
        assert "BUY" in output
        assert "$150.00" in output

    def test_shows_sell_action(self):
        actions = [{
            "timestamp": "2024-01-20 14:00:00",
            "ticker": "MSFT",
            "action": "SELL",
            "quantity": 30,
            "price": 280.0,
            "source": "robinhood",
            "reasoning": "Taking profit",
        }]
        output = _capture_output(report.print_actions, actions)
        assert "SELL" in output
        assert "MSFT" in output


# ── print_recommendations_list ─────────────────────────────────────


class TestPrintRecommendationsList:
    def test_empty_list(self):
        output = _capture_output(report.print_recommendations_list, [])
        assert "No recommendations found" in output

    def test_shows_recommendations(self):
        recs = [{
            "timestamp": "2024-01-15 10:30:00",
            "ticker": "AAPL",
            "action": "BUY",
            "confidence": 80,
            "summary": "Strong buy signal from all analysts.",
        }]
        output = _capture_output(report.print_recommendations_list, recs)
        assert "AAPL" in output
        assert "BUY" in output
        assert "80%" in output

    def test_truncates_long_summary(self):
        recs = [{
            "timestamp": "2024-01-15",
            "ticker": "AAPL",
            "action": "HOLD",
            "confidence": 50,
            "summary": "A" * 200,
        }]
        output = _capture_output(report.print_recommendations_list, recs)
        # Should not crash, summary gets truncated to 80 chars
        assert "AAPL" in output
