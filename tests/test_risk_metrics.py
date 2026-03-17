"""Tests for portfolio.risk_metrics module."""

import numpy as np
import pytest

from trading_master.portfolio.risk_metrics import (
    calmar_ratio,
    compute_beta,
    cvar,
    historical_var,
    max_drawdown,
    parametric_var,
    portfolio_risk_dashboard,
    sharpe_ratio,
    sortino_ratio,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _make_returns(seed=42, T=500, N=3):
    """Create reproducible synthetic daily returns."""
    rng = np.random.RandomState(seed)
    return rng.normal(0.0005, 0.02, size=(T, N))


def _equal_weights(N=3):
    return np.ones(N) / N


# ── Parametric VaR ─────────────────────────────────────────────────────

class TestParametricVar:
    def test_positive_result(self):
        ret = _make_returns()
        w = _equal_weights()
        v = parametric_var(w, ret, 0.95, 100_000)
        assert v > 0

    def test_higher_confidence_higher_var(self):
        ret = _make_returns()
        w = _equal_weights()
        v95 = parametric_var(w, ret, 0.95, 100_000)
        v99 = parametric_var(w, ret, 0.99, 100_000)
        assert v99 > v95

    def test_empty(self):
        assert parametric_var(np.array([]), np.array([]), 0.95) == 0.0

    def test_single_asset(self):
        ret = _make_returns(N=1)
        v = parametric_var(np.array([1.0]), ret, 0.95, 10_000)
        assert v > 0

    def test_zero_std(self):
        ret = np.zeros((100, 2))
        w = np.array([0.5, 0.5])
        assert parametric_var(w, ret, 0.95) == 0.0


# ── Historical VaR ─────────────────────────────────────────────────────

class TestHistoricalVar:
    def test_positive_result(self):
        ret = _make_returns()
        v = historical_var(_equal_weights(), ret, 0.95, 100_000)
        assert v > 0

    def test_empty(self):
        assert historical_var(np.array([]), np.array([]), 0.95) == 0.0

    def test_all_positive_returns(self):
        ret = np.abs(_make_returns()) + 0.001
        v = historical_var(_equal_weights(), ret, 0.95, 100_000)
        # Even with all-positive returns, VaR should be >= 0
        assert v >= 0.0


# ── CVaR ───────────────────────────────────────────────────────────────

class TestCVar:
    def test_cvar_gte_var(self):
        ret = _make_returns()
        w = _equal_weights()
        v = historical_var(w, ret, 0.95, 100_000)
        cv = cvar(w, ret, 0.95, 100_000)
        assert cv >= v - 1e-6  # CVaR >= VaR (with floating tolerance)

    def test_empty(self):
        assert cvar(np.array([]), np.array([]), 0.95) == 0.0


# ── Max Drawdown ───────────────────────────────────────────────────────

class TestMaxDrawdown:
    def test_monotonically_increasing(self):
        equity = np.arange(1, 101, dtype=float)
        result = max_drawdown(equity)
        assert result["max_dd"] == 0.0

    def test_known_drawdown(self):
        equity = np.array([100, 110, 90, 80, 95, 120], dtype=float)
        result = max_drawdown(equity)
        # Peak = 110, trough = 80 => dd = 30/110 ~ 0.2727
        assert abs(result["max_dd"] - 30 / 110) < 1e-6
        assert result["peak_idx"] == 1
        assert result["trough_idx"] == 3

    def test_empty(self):
        result = max_drawdown(np.array([]))
        assert result["max_dd"] == 0.0

    def test_single_element(self):
        result = max_drawdown(np.array([100.0]))
        assert result["max_dd"] == 0.0

    def test_recovery_found(self):
        equity = np.array([100, 120, 90, 130], dtype=float)
        result = max_drawdown(equity)
        assert result["recovery_idx"] == 3


# ── Sharpe Ratio ───────────────────────────────────────────────────────

class TestSharpeRatio:
    def test_positive_returns(self):
        rng = np.random.RandomState(123)
        ret = rng.normal(0.001, 0.01, 252)
        s = sharpe_ratio(ret)
        assert s > 0

    def test_zero_std(self):
        ret = np.ones(100) * 0.001
        assert sharpe_ratio(ret) == 0.0

    def test_empty(self):
        assert sharpe_ratio(np.array([])) == 0.0

    def test_single_element(self):
        assert sharpe_ratio(np.array([0.01])) == 0.0


# ── Sortino Ratio ──────────────────────────────────────────────────────

class TestSortinoRatio:
    def test_positive(self):
        rng = np.random.RandomState(123)
        ret = rng.normal(0.001, 0.01, 252)
        s = sortino_ratio(ret)
        assert s > 0

    def test_all_positive_returns(self):
        ret = np.abs(np.random.RandomState(1).normal(0, 0.01, 100)) + 0.001
        # No negative returns => sortino = 0 by convention
        assert sortino_ratio(ret) == 0.0

    def test_empty(self):
        assert sortino_ratio(np.array([])) == 0.0


# ── Calmar Ratio ───────────────────────────────────────────────────────

class TestCalmarRatio:
    def test_positive(self):
        equity = np.array([100, 110, 95, 105, 115, 130], dtype=float)
        c = calmar_ratio(equity)
        assert c > 0

    def test_no_drawdown(self):
        equity = np.arange(100, 200, dtype=float)
        assert calmar_ratio(equity) == 0.0

    def test_empty(self):
        assert calmar_ratio(np.array([])) == 0.0


# ── Beta ───────────────────────────────────────────────────────────────

class TestComputeBeta:
    def test_same_returns(self):
        ret = np.random.RandomState(7).normal(0, 0.01, 200)
        b = compute_beta(ret, ret)
        assert abs(b - 1.0) < 1e-6

    def test_zero_market_variance(self):
        asset = np.random.RandomState(8).normal(0, 0.01, 100)
        market = np.zeros(100)
        assert compute_beta(asset, market) == 0.0

    def test_empty(self):
        assert compute_beta(np.array([]), np.array([])) == 0.0


# ── Dashboard ──────────────────────────────────────────────────────────

class TestDashboard:
    def test_returns_all_keys(self):
        ret = _make_returns()
        w = _equal_weights()
        bench = np.random.RandomState(99).normal(0, 0.01, 500)
        d = portfolio_risk_dashboard(ret, w, bench, 100_000)
        for key in ["sharpe", "sortino", "var_95", "var_99", "cvar_95", "max_dd", "beta", "calmar"]:
            assert key in d

    def test_empty(self):
        d = portfolio_risk_dashboard(np.array([]), np.array([]))
        assert d["sharpe"] == 0.0
        assert d["var_95"] == 0.0

    def test_no_benchmark(self):
        ret = _make_returns()
        w = _equal_weights()
        d = portfolio_risk_dashboard(ret, w)
        assert d["beta"] == 0.0
