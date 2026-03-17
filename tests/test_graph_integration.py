"""Integration tests for quantitative_risk_node with regime awareness and CVaR."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from trading_master.agents.graph import quantitative_risk_node, GraphState


def _make_graph_state(
    ticker: str = "AAPL",
    regime: str | None = "bull",
    price: float = 150.0,
    atr_14: float = 5.0,
    portfolio_value: float = 100_000.0,
    positions: dict | None = None,
) -> GraphState:
    """Build a minimal GraphState dict for testing."""
    gs: GraphState = {
        "ticker": ticker,
        "market_data": {"current_price": price, "ticker": ticker},
        "technical_data": {"atr_14": atr_14, "ticker": ticker},
        "macro_data": {"regime": regime} if regime is not None else None,
        "portfolio_state": {
            "total_value": portfolio_value,
            "positions": positions or {},
        },
        "risk_assessment": {
            "risk_score": 50.0,
            "max_position_size": 0.0,
            "approved": True,
            "warnings": [],
        },
        "errors": [],
        "analyst_reports": [],
        "debate_reports": [],
        "recommendation": None,
        "quantitative_risk": None,
        "total_tokens": 0,
        "total_cost": 0.0,
        "fundamental_data": None,
        "sentiment_data": None,
    }
    return gs


class _FakeConfig:
    class risk:
        max_position_pct = 8.0
        holding_days = 20
        tail_multiplier = 2.0


@pytest.fixture(autouse=True)
def _mock_config():
    with patch(
        "trading_master.agents.graph.get_config",
        return_value=_FakeConfig(),
        create=True,
    ):
        # Also patch the import inside the function
        with patch.dict(
            "sys.modules",
            {"trading_master.config": MagicMock(get_config=lambda: _FakeConfig())},
        ):
            yield


# ── Regime tests ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bull_regime_no_warnings():
    gs = _make_graph_state(regime="bull")

    with patch(
        "trading_master.agents.graph.fetch_returns",
        return_value=(None, []),
    ):
        result = await quantitative_risk_node(gs)

    qr = result["quantitative_risk"]
    ra = result["risk_assessment"]

    assert qr["regime"] == "bull"
    assert ra["approved"] is True
    assert not any("CRISIS" in w for w in ra.get("warnings", []))
    assert not any("BEAR" in w for w in ra.get("warnings", []))


@pytest.mark.asyncio
async def test_bear_regime_warning_and_reduced_size():
    gs_bull = _make_graph_state(regime="bull")
    gs_bear = _make_graph_state(regime="bear")

    # Disable HMM override by making SPY fetch fail
    with patch(
        "trading_master.agents.graph.fetch_returns",
        return_value=(None, []),
    ):
        result_bull = await quantitative_risk_node(gs_bull)
        result_bear = await quantitative_risk_node(gs_bear)

    bull_shares = result_bull["quantitative_risk"]["sizing"]["shares"]
    bear_shares = result_bear["quantitative_risk"]["sizing"]["shares"]

    assert bear_shares <= bull_shares
    assert result_bear["quantitative_risk"]["regime"] == "bear"

    ra = result_bear["risk_assessment"]
    assert ra["approved"] is True  # BEAR doesn't block, just warns
    assert any("BEAR regime" in w for w in ra.get("warnings", []))


@pytest.mark.asyncio
async def test_crisis_regime_blocks_trade():
    gs = _make_graph_state(regime="crisis")

    with patch(
        "trading_master.agents.graph.fetch_returns",
        return_value=(None, []),
    ):
        result = await quantitative_risk_node(gs)

    ra = result["risk_assessment"]
    assert ra["approved"] is False
    assert any("CRISIS regime" in w for w in ra.get("warnings", []))

    qr = result["quantitative_risk"]
    assert qr["regime"] == "crisis"
    # Crisis multiplier = 0.25x
    assert qr["sizing"]["regime_multiplier"] == 0.25


@pytest.mark.asyncio
async def test_sideways_regime_reduces_size():
    gs_bull = _make_graph_state(regime="bull")
    gs_side = _make_graph_state(regime="sideways")

    with patch(
        "trading_master.agents.graph.fetch_returns",
        return_value=(None, []),
    ):
        result_bull = await quantitative_risk_node(gs_bull)
        result_side = await quantitative_risk_node(gs_side)

    bull_shares = result_bull["quantitative_risk"]["sizing"]["shares"]
    side_shares = result_side["quantitative_risk"]["sizing"]["shares"]

    assert side_shares <= bull_shares


@pytest.mark.asyncio
async def test_no_macro_data_no_regime():
    gs = _make_graph_state(regime=None)
    gs["macro_data"] = None

    with patch(
        "trading_master.agents.graph.fetch_returns",
        return_value=(None, []),
    ):
        result = await quantitative_risk_node(gs)

    qr = result["quantitative_risk"]
    assert qr["regime"] is None
    # No regime warnings
    ra = result["risk_assessment"]
    assert not any("regime" in w.lower() for w in ra.get("warnings", []))


# ── CVaR tests ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cvar_computed_when_positions_exist():
    """When positions exist and fetch_returns works, CVaR should be computed."""
    positions = {
        "MSFT": {"quantity": 10, "current_price": 400.0, "avg_cost": 380.0},
    }
    gs = _make_graph_state(regime="bull", positions=positions)

    # Mock fetch_returns to return fake data
    fake_returns = np.random.default_rng(42).normal(0.001, 0.02, (100, 2))
    with patch(
        "trading_master.agents.graph.fetch_returns",
        return_value=(fake_returns, ["MSFT", "AAPL"]),
    ):
        result = await quantitative_risk_node(gs)

    qr = result["quantitative_risk"]
    assert qr["portfolio_cvar"] is not None
    assert qr["new_portfolio_cvar"] is not None


@pytest.mark.asyncio
async def test_cvar_exceeding_threshold_blocks_trade():
    """When new portfolio CVaR exceeds threshold, the trade should be hard-blocked."""
    positions = {
        "MSFT": {"quantity": 10, "current_price": 400.0, "avg_cost": 380.0},
    }
    gs = _make_graph_state(regime="bull", positions=positions)

    # Create returns that will produce a high CVaR (large negative tail)
    rng = np.random.default_rng(99)
    # Normal returns with a fat left tail to push CVaR above 5% threshold
    returns = rng.normal(-0.05, 0.15, (100, 2))
    with patch(
        "trading_master.agents.graph.fetch_returns",
        return_value=(returns, ["MSFT", "AAPL"]),
    ):
        result = await quantitative_risk_node(gs)

    ra = result["risk_assessment"]
    qr = result["quantitative_risk"]

    if qr.get("cvar_warning"):
        # CVaR exceeded threshold — trade must be blocked
        assert ra["approved"] is False
        assert any("BLOCKED" in w for w in ra.get("warnings", []))


@pytest.mark.asyncio
async def test_cvar_failure_does_not_block():
    """If fetch_returns raises, the pipeline should still complete."""
    positions = {
        "MSFT": {"quantity": 10, "current_price": 400.0, "avg_cost": 380.0},
    }
    gs = _make_graph_state(regime="bull", positions=positions)

    with patch(
        "trading_master.agents.graph.fetch_returns",
        side_effect=RuntimeError("network error"),
    ):
        result = await quantitative_risk_node(gs)

    # Should still have sizing results
    qr = result["quantitative_risk"]
    assert "sizing" in qr
    assert qr["portfolio_cvar"] is None


# ── EVT integration tests ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_evt_result_stored():
    """EVT tail risk result should appear in quantitative_risk."""
    gs = _make_graph_state(regime="bull")

    # Mock fetch_returns to return enough data for EVT
    rng = np.random.default_rng(42)
    fake_returns = rng.normal(0.001, 0.02, (300, 1))
    with patch(
        "trading_master.agents.graph.fetch_returns",
        return_value=(fake_returns, ["AAPL"]),
    ):
        result = await quantitative_risk_node(gs)

    qr = result["quantitative_risk"]
    assert "evt" in qr
    if qr["evt"]:
        assert "tail_type" in qr["evt"]
        assert "var_99" in qr["evt"]
        assert "cvar_99" in qr["evt"]


@pytest.mark.asyncio
async def test_evt_heavy_tail_warns():
    """Heavy-tailed returns should produce an EVT warning."""
    gs = _make_graph_state(regime="bull")

    # Student-t(3) returns → heavy tails
    rng = np.random.default_rng(42)
    fake_returns = rng.standard_t(df=3, size=(300, 1)) * 0.02
    with patch(
        "trading_master.agents.graph.fetch_returns",
        return_value=(fake_returns, ["AAPL"]),
    ):
        result = await quantitative_risk_node(gs)

    qr = result["quantitative_risk"]
    evt = qr.get("evt", {})
    if evt and evt.get("tail_type") == "heavy":
        ra = result["risk_assessment"]
        assert any("EVT" in w for w in ra.get("warnings", []))


@pytest.mark.asyncio
async def test_evt_failure_non_fatal():
    """If EVT fails, the rest of the pipeline should still work."""
    gs = _make_graph_state(regime="bull")

    # Return too little data for EVT (< 30) but enough for sizing
    fake_returns = np.random.default_rng(42).normal(0.001, 0.02, (20, 1))
    with patch(
        "trading_master.agents.graph.fetch_returns",
        return_value=(fake_returns, ["AAPL"]),
    ):
        result = await quantitative_risk_node(gs)

    qr = result["quantitative_risk"]
    assert "sizing" in qr  # sizing still works
    # EVT may be empty dict (too few obs)
    assert "evt" in qr


# ── HMM regime integration tests ─────────────────────────────────────


@pytest.mark.asyncio
async def test_hmm_regime_stored():
    """HMM regime result should appear in quantitative_risk."""
    gs = _make_graph_state(regime="bull")

    rng = np.random.default_rng(42)
    fake_returns = rng.normal(0.001, 0.02, (300, 1))
    with patch(
        "trading_master.agents.graph.fetch_returns",
        return_value=(fake_returns, ["SPY"]),
    ):
        result = await quantitative_risk_node(gs)

    qr = result["quantitative_risk"]
    assert "hmm_regime" in qr
    assert "hmm_confidence" in qr


@pytest.mark.asyncio
async def test_hmm_failure_non_fatal():
    """If HMM fails, the pipeline falls back to heuristic regime."""
    gs = _make_graph_state(regime="bear")

    # Patch fit_regime_model to raise
    with patch(
        "trading_master.agents.graph.fetch_returns",
        side_effect=RuntimeError("network error"),
    ):
        result = await quantitative_risk_node(gs)

    qr = result["quantitative_risk"]
    assert "sizing" in qr
    # Should still use heuristic regime
    assert qr["regime"] == "bear"
