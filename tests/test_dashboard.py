"""Tests for Portfolio Risk Dashboard."""

from __future__ import annotations

import numpy as np
import pytest

from trading_master.quant.dashboard import (
    DashboardMetrics,
    build_dashboard,
    compute_risk_score,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def healthy_returns():
    """Steady positive returns — healthy portfolio."""
    rng = np.random.default_rng(42)
    return rng.normal(0.001, 0.01, 252)


@pytest.fixture
def stressed_returns():
    """Volatile negative returns — stressed portfolio."""
    rng = np.random.default_rng(42)
    ret = rng.normal(-0.002, 0.03, 252)
    ret[100:110] = -0.05  # crash period
    return ret


@pytest.fixture
def flat_returns():
    """Near-zero returns."""
    rng = np.random.default_rng(42)
    return rng.normal(0.0, 0.005, 100)


# ── compute_risk_score ─────────────────────────────────────────────


class TestComputeRiskScore:
    def test_bull_regime_low_score(self):
        score, level = compute_risk_score(regime="bull", sharpe=1.5, max_dd=0.03, tail_type="bounded")
        assert score < 30
        assert level in ("low", "moderate")

    def test_crisis_regime_high_score(self):
        score, level = compute_risk_score(regime="crisis", sharpe=-1.0, max_dd=0.25, tail_type="heavy")
        assert score > 60
        assert level in ("high", "extreme")

    def test_score_range(self):
        score, _ = compute_risk_score()
        assert 0 <= score <= 100

    def test_level_values(self):
        for regime in ("bull", "bear", "crisis", "neutral"):
            _, level = compute_risk_score(regime=regime)
            assert level in ("low", "moderate", "elevated", "high", "extreme")

    def test_negative_sharpe_increases_score(self):
        s1, _ = compute_risk_score(sharpe=1.0)
        s2, _ = compute_risk_score(sharpe=-1.0)
        assert s2 > s1

    def test_heavy_tail_increases_score(self):
        s1, _ = compute_risk_score(tail_type="bounded")
        s2, _ = compute_risk_score(tail_type="heavy")
        assert s2 > s1

    def test_high_drawdown_increases_score(self):
        s1, _ = compute_risk_score(max_dd=0.02)
        s2, _ = compute_risk_score(max_dd=0.25)
        assert s2 > s1

    def test_high_vol_increases_score(self):
        s1, _ = compute_risk_score(portfolio_vol=0.005)
        s2, _ = compute_risk_score(portfolio_vol=0.03)
        assert s2 > s1


# ── build_dashboard ────────────────────────────────────────────────


class TestBuildDashboard:
    def test_returns_dashboard_metrics(self, healthy_returns):
        result = build_dashboard(portfolio_returns=healthy_returns, regime="bull")
        assert isinstance(result, DashboardMetrics)

    def test_regime_stored(self, healthy_returns):
        result = build_dashboard(portfolio_returns=healthy_returns, regime="bear", regime_confidence=0.85)
        assert result.regime == "bear"
        assert result.regime_confidence == 0.85

    def test_sharpe_computed(self, healthy_returns):
        result = build_dashboard(portfolio_returns=healthy_returns)
        assert result.sharpe_ratio != 0.0

    def test_sortino_computed(self, healthy_returns):
        result = build_dashboard(portfolio_returns=healthy_returns)
        # Healthy returns should have positive Sortino
        assert result.sortino_ratio > 0

    def test_max_drawdown_computed(self, stressed_returns):
        result = build_dashboard(portfolio_returns=stressed_returns)
        assert result.max_drawdown > 0

    def test_risk_score_computed(self, healthy_returns):
        result = build_dashboard(portfolio_returns=healthy_returns, regime="bull")
        assert 0 <= result.risk_score <= 100
        assert result.risk_level in ("low", "moderate", "elevated", "high", "extreme")

    def test_tail_type_stored(self):
        result = build_dashboard(tail_type="heavy", var_99=0.05, cvar_99=0.08)
        assert result.tail_type == "heavy"
        assert result.var_99 == 0.05
        assert result.cvar_99 == 0.08

    def test_sector_stored(self):
        result = build_dashboard(top_sector="XLK", bottom_sector="XLU")
        assert result.top_sector == "XLK"
        assert result.bottom_sector == "XLU"

    def test_no_returns_still_works(self):
        result = build_dashboard(regime="bull")
        assert result.sharpe_ratio == 0.0
        assert result.risk_score >= 0

    def test_short_returns_still_works(self):
        result = build_dashboard(portfolio_returns=np.array([0.01, -0.01, 0.005]))
        assert result.sharpe_ratio == 0.0  # too short

    def test_health_summary_healthy(self, healthy_returns):
        result = build_dashboard(portfolio_returns=healthy_returns, regime="bull", tail_type="bounded")
        assert "healthy" in result.health_summary.lower() or result.risk_level in ("low", "moderate")

    def test_health_summary_stressed(self, stressed_returns):
        result = build_dashboard(
            portfolio_returns=stressed_returns,
            regime="crisis",
            tail_type="heavy",
        )
        assert len(result.health_summary) > 0
        # Should mention at least one concern
        assert any(word in result.health_summary.lower() for word in
                    ["risk", "crisis", "heavy", "drawdown"])


# ── is_healthy property ────────────────────────────────────────────


class TestIsHealthy:
    def test_low_score_healthy(self):
        result = build_dashboard(regime="bull", tail_type="bounded")
        if result.risk_score < 40:
            assert result.is_healthy

    def test_crisis_not_healthy(self, stressed_returns):
        result = build_dashboard(
            portfolio_returns=stressed_returns,
            regime="crisis",
            tail_type="heavy",
        )
        # Crisis with stressed returns should have high score
        assert result.risk_score > 40 or not result.is_healthy


# ── Edge cases ─────────────────────────────────────────────────────


class TestEdgeCases:
    def test_all_defaults(self):
        result = build_dashboard()
        assert isinstance(result, DashboardMetrics)
        assert result.regime == "unknown"

    def test_extreme_inputs(self):
        rng = np.random.default_rng(42)
        extreme_ret = rng.normal(-0.1, 0.1, 252)
        result = build_dashboard(
            portfolio_returns=extreme_ret,
            regime="crisis",
            tail_type="heavy",
            var_99=0.20,
            cvar_99=0.30,
        )
        assert result.risk_score <= 100
        assert result.risk_level == "extreme" or result.risk_score > 60
