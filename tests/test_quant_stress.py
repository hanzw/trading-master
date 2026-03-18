"""Stress tests for quant modules — edge cases, extreme inputs, cross-module consistency."""

from __future__ import annotations

import numpy as np
import pytest

from trading_master.quant import (
    risk_parity,
    hrp_allocation,
    compare_allocations,
    minimum_variance_portfolio,
    max_sharpe_portfolio,
    evt_tail_risk,
    fit_regime_model,
    multi_timeframe_analysis,
    cointegration_test,
    compute_risk_score,
    fit_garch,
    capm_regression,
)


# ── Large universe ─────────────────────────────────────────────────


class TestLargeUniverse:
    def test_20_asset_risk_parity(self):
        rng = np.random.default_rng(42)
        n = 20
        A = rng.normal(0, 0.01, (n, n))
        cov = A @ A.T + np.eye(n) * 0.01
        result = risk_parity(cov)
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(result.weights > 0)

    def test_20_asset_hrp(self):
        rng = np.random.default_rng(42)
        n = 20
        A = rng.normal(0, 0.01, (n, n))
        cov = A @ A.T + np.eye(n) * 0.01
        result = hrp_allocation(cov)
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_20_asset_compare(self):
        rng = np.random.default_rng(42)
        n = 20
        mu = rng.normal(0.08, 0.03, n)
        A = rng.normal(0, 0.01, (n, n))
        cov = A @ A.T + np.eye(n) * 0.01
        result = compare_allocations(mu, cov)
        assert len(result.consensus_weights) == n
        assert result.consensus_weights.sum() == pytest.approx(1.0, abs=1e-6)


# ── Extreme returns ────────────────────────────────────────────────


class TestExtremeReturns:
    def test_evt_very_volatile(self):
        rng = np.random.default_rng(42)
        returns = rng.standard_t(df=2, size=500) * 0.05
        result = evt_tail_risk(returns)
        assert result.var_99 > 0
        assert result.cvar_99 >= result.var_99

    def test_evt_all_positive(self):
        rng = np.random.default_rng(42)
        returns = np.abs(rng.normal(0.01, 0.02, 200))
        result = evt_tail_risk(returns)
        assert isinstance(result.tail_type, str)

    def test_regime_very_short(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, 30)  # minimum allowed
        result = fit_regime_model(returns, n_regimes=2)
        assert result.n_regimes == 2

    def test_regime_very_long(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, 2000)
        result = fit_regime_model(returns, n_regimes=2)
        assert len(result.regime_sequence) == 2000

    def test_garch_near_zero_volatility(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.0001, 100)
        result = fit_garch(returns)
        assert result.omega >= 0

    def test_capm_perfect_correlation(self):
        rng = np.random.default_rng(42)
        market = rng.normal(0.001, 0.02, 100)
        asset = 1.5 * market + rng.normal(0, 0.001, 100)
        result = capm_regression(asset, market)
        assert result.beta == pytest.approx(1.5, abs=0.1)
        assert result.r_squared > 0.9


# ── Cross-module consistency ───────────────────────────────────────


class TestCrossModuleConsistency:
    def test_hrp_vs_risk_parity_both_valid(self):
        """Both should produce valid weights on same inputs."""
        cov = np.array([[0.04, 0.01], [0.01, 0.02]])
        hrp = hrp_allocation(cov)
        rp = risk_parity(cov)
        assert hrp.weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert rp.weights.sum() == pytest.approx(1.0, abs=1e-6)
        # Both should favor the lower-vol asset
        assert hrp.weights[1] > hrp.weights[0]
        assert rp.weights[1] > rp.weights[0]

    def test_compare_includes_all_three(self):
        mu = np.array([0.10, 0.06])
        cov = np.array([[0.04, 0.005], [0.005, 0.01]])
        result = compare_allocations(mu, cov)
        assert len(result.markowitz_weights) == 2
        assert len(result.hrp_weights) == 2
        assert len(result.risk_parity_weights) == 2

    def test_risk_score_monotonic_with_regime(self):
        scores = {}
        for regime in ["bull", "neutral", "bear", "crisis"]:
            s, _ = compute_risk_score(regime=regime)
            scores[regime] = s
        assert scores["bull"] <= scores["bear"]
        assert scores["bear"] <= scores["crisis"]

    def test_multi_timeframe_consistent_with_trend(self):
        # Strong uptrend should produce bullish on all timeframes
        prices = np.linspace(100, 200, 300)
        result = multi_timeframe_analysis(prices)
        assert result.consensus_score > 0
        assert result.n_bullish >= 2


# ── Numerical edge cases ──────────────────────────────────────────


class TestNumericalEdgeCases:
    def test_identical_assets_risk_parity(self):
        """All identical assets → equal weights."""
        cov = np.eye(3) * 0.04
        result = risk_parity(cov)
        np.testing.assert_allclose(result.weights, [1/3, 1/3, 1/3], atol=0.01)

    def test_highly_correlated_pair(self):
        """Nearly 1.0 correlation should still produce valid weights."""
        cov = np.array([[0.04, 0.0399], [0.0399, 0.04]])
        result = risk_parity(cov)
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-4)

    def test_very_small_covariance(self):
        cov = np.array([[1e-8, 0], [0, 1e-8]])
        result = hrp_allocation(cov)
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-4)

    def test_cointegration_trending_pair(self):
        rng = np.random.default_rng(42)
        trend = np.cumsum(rng.normal(0, 1, 200))
        a = 100 + 2 * trend + rng.normal(0, 0.5, 200)
        b = 50 + trend
        result = cointegration_test(a, b)
        assert result.hedge_ratio > 1.0  # should be ~2.0
        assert result.n_observations == 200
