"""Tests for Markowitz Mean-Variance Optimization."""

from __future__ import annotations

import numpy as np
import pytest

from trading_master.quant.markowitz import (
    PortfolioPoint,
    efficient_frontier,
    max_sharpe_portfolio,
    minimum_variance_portfolio,
    target_return_portfolio,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def two_asset_data():
    """Two assets: one high-return/high-risk, one low-return/low-risk."""
    mu = np.array([0.12, 0.06])       # 12% and 6% annualized
    cov = np.array([
        [0.04, 0.005],                 # 20% vol, low correlation
        [0.005, 0.01],                 # 10% vol
    ])
    return mu, cov


@pytest.fixture
def three_asset_data():
    """Three assets with different risk/return profiles."""
    mu = np.array([0.15, 0.10, 0.05])
    cov = np.array([
        [0.0625, 0.015, 0.003],        # 25% vol
        [0.015, 0.0225, 0.005],         # 15% vol
        [0.003, 0.005, 0.0064],         # 8% vol
    ])
    return mu, cov


# ── minimum_variance_portfolio ──────────────────────────────────────


class TestMinimumVariancePortfolio:
    def test_weights_sum_to_one(self, two_asset_data):
        mu, cov = two_asset_data
        result = minimum_variance_portfolio(cov, mu)
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-8)

    def test_long_only_non_negative(self, two_asset_data):
        mu, cov = two_asset_data
        result = minimum_variance_portfolio(cov, mu, long_only=True)
        assert np.all(result.weights >= -1e-10)

    def test_favors_low_vol_asset(self, two_asset_data):
        mu, cov = two_asset_data
        result = minimum_variance_portfolio(cov, mu)
        # Asset 1 (10% vol) should get more weight than asset 0 (20% vol)
        assert result.weights[1] > result.weights[0]

    def test_volatility_positive(self, two_asset_data):
        mu, cov = two_asset_data
        result = minimum_variance_portfolio(cov, mu)
        assert result.volatility > 0

    def test_no_expected_returns(self, two_asset_data):
        _, cov = two_asset_data
        result = minimum_variance_portfolio(cov)
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-8)
        # With zero expected returns, Sharpe should be negative (below Rf)
        assert result.sharpe_ratio < 0

    def test_three_assets(self, three_asset_data):
        mu, cov = three_asset_data
        result = minimum_variance_portfolio(cov, mu)
        assert len(result.weights) == 3
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-8)


# ── max_sharpe_portfolio ────────────────────────────────────────────


class TestMaxSharpePortfolio:
    def test_weights_sum_to_one(self, two_asset_data):
        mu, cov = two_asset_data
        result = max_sharpe_portfolio(mu, cov)
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-8)

    def test_positive_sharpe(self, two_asset_data):
        mu, cov = two_asset_data
        result = max_sharpe_portfolio(mu, cov, risk_free_rate=0.02)
        # Both assets have returns > Rf, so Sharpe should be positive
        assert result.sharpe_ratio > 0

    def test_higher_sharpe_than_equal_weight(self, two_asset_data):
        mu, cov = two_asset_data
        tangency = max_sharpe_portfolio(mu, cov)

        # Equal-weight portfolio
        w_eq = np.array([0.5, 0.5])
        ret_eq = w_eq @ mu
        vol_eq = np.sqrt(w_eq @ cov @ w_eq)
        sharpe_eq = (ret_eq - 0.04) / vol_eq

        assert tangency.sharpe_ratio >= sharpe_eq - 1e-6

    def test_long_only(self, two_asset_data):
        mu, cov = two_asset_data
        result = max_sharpe_portfolio(mu, cov, long_only=True)
        assert np.all(result.weights >= -1e-10)

    def test_three_assets(self, three_asset_data):
        mu, cov = three_asset_data
        result = max_sharpe_portfolio(mu, cov)
        assert len(result.weights) == 3
        assert result.sharpe_ratio > 0


# ── target_return_portfolio ─────────────────────────────────────────


class TestTargetReturnPortfolio:
    def test_achieves_target(self, two_asset_data):
        mu, cov = two_asset_data
        target = 0.09
        result = target_return_portfolio(mu, cov, target)
        assert result.expected_return == pytest.approx(target, abs=0.001)

    def test_weights_sum_to_one(self, two_asset_data):
        mu, cov = two_asset_data
        result = target_return_portfolio(mu, cov, 0.08)
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-8)

    def test_higher_target_more_risky(self, two_asset_data):
        mu, cov = two_asset_data
        low = target_return_portfolio(mu, cov, 0.07)
        high = target_return_portfolio(mu, cov, 0.11)
        assert high.volatility > low.volatility

    def test_three_assets(self, three_asset_data):
        mu, cov = three_asset_data
        result = target_return_portfolio(mu, cov, 0.10)
        assert result.expected_return == pytest.approx(0.10, abs=0.001)


# ── efficient_frontier ──────────────────────────────────────────────


class TestEfficientFrontier:
    def test_returns_list(self, two_asset_data):
        mu, cov = two_asset_data
        frontier = efficient_frontier(mu, cov, n_points=20)
        assert len(frontier) > 0
        assert all(isinstance(p, PortfolioPoint) for p in frontier)

    def test_sorted_by_volatility(self, two_asset_data):
        mu, cov = two_asset_data
        frontier = efficient_frontier(mu, cov, n_points=20)
        vols = [p.volatility for p in frontier]
        assert vols == sorted(vols)

    def test_return_increases_with_risk(self, two_asset_data):
        mu, cov = two_asset_data
        frontier = efficient_frontier(mu, cov, n_points=20)
        # On the efficient frontier, return should generally increase with vol
        if len(frontier) >= 2:
            assert frontier[-1].expected_return >= frontier[0].expected_return - 0.01

    def test_all_weights_valid(self, two_asset_data):
        mu, cov = two_asset_data
        frontier = efficient_frontier(mu, cov, n_points=10, long_only=True)
        for point in frontier:
            assert point.weights.sum() == pytest.approx(1.0, abs=1e-6)
            assert np.all(point.weights >= -1e-8)

    def test_three_assets(self, three_asset_data):
        mu, cov = three_asset_data
        frontier = efficient_frontier(mu, cov, n_points=15)
        assert len(frontier) > 5

    def test_single_asset(self):
        mu = np.array([0.10])
        cov = np.array([[0.04]])
        frontier = efficient_frontier(mu, cov, n_points=5)
        assert len(frontier) >= 1
        # Single asset: weight should be 1.0
        assert frontier[0].weights[0] == pytest.approx(1.0, abs=1e-6)

    def test_custom_risk_free_rate(self, two_asset_data):
        mu, cov = two_asset_data
        frontier = efficient_frontier(mu, cov, n_points=10, risk_free_rate=0.02)
        # All Sharpe ratios should be computed with Rf=0.02
        for point in frontier:
            expected_sharpe = (point.expected_return - 0.02) / point.volatility
            assert point.sharpe_ratio == pytest.approx(expected_sharpe, abs=1e-4)
