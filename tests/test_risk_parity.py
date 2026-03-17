"""Tests for Risk Parity (Equal Risk Contribution) allocation."""

from __future__ import annotations

import numpy as np
import pytest

from trading_master.quant.risk_parity import (
    RiskParityResult,
    inverse_volatility,
    risk_parity,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def two_asset_cov():
    """Two assets: one high-vol, one low-vol, low correlation."""
    cov = np.array([
        [0.04, 0.005],     # 20% vol
        [0.005, 0.01],     # 10% vol
    ])
    return cov


@pytest.fixture
def three_asset_cov():
    """Three assets with different risk profiles."""
    cov = np.array([
        [0.0625, 0.015, 0.003],    # 25% vol
        [0.015, 0.0225, 0.005],     # 15% vol
        [0.003, 0.005, 0.0064],     # 8% vol
    ])
    return cov


@pytest.fixture
def uncorrelated_cov():
    """Three uncorrelated assets with different volatilities."""
    cov = np.array([
        [0.04, 0.0, 0.0],      # 20% vol
        [0.0,  0.01, 0.0],     # 10% vol
        [0.0,  0.0,  0.0025],  # 5% vol
    ])
    return cov


@pytest.fixture
def five_asset_cov():
    """Five assets: two equities, two bonds, one commodity."""
    cov = np.array([
        [0.04,   0.025,  0.002,  0.001,  0.005],
        [0.025,  0.0625, 0.003,  0.001,  0.008],
        [0.002,  0.003,  0.0025, 0.001,  0.000],
        [0.001,  0.001,  0.001,  0.0016, 0.000],
        [0.005,  0.008,  0.000,  0.000,  0.0225],
    ])
    return cov


# ── Basic properties ───────────────────────────────────────────────


class TestRiskParityBasicProperties:
    def test_weights_sum_to_one(self, two_asset_cov):
        result = risk_parity(two_asset_cov)
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_all_weights_positive(self, three_asset_cov):
        result = risk_parity(three_asset_cov)
        assert np.all(result.weights > 0)

    def test_returns_correct_type(self, two_asset_cov):
        result = risk_parity(two_asset_cov)
        assert isinstance(result, RiskParityResult)

    def test_correct_number_of_weights(self, three_asset_cov):
        result = risk_parity(three_asset_cov, tickers=["A", "B", "C"])
        assert len(result.weights) == 3
        assert len(result.tickers) == 3
        assert len(result.risk_contributions) == 3

    def test_risk_contributions_sum_to_vol(self, two_asset_cov):
        result = risk_parity(two_asset_cov)
        assert result.risk_contributions.sum() == pytest.approx(
            result.portfolio_volatility, abs=1e-4
        )

    def test_risk_contribution_pct_sums_to_one(self, three_asset_cov):
        result = risk_parity(three_asset_cov)
        assert result.risk_contribution_pct.sum() == pytest.approx(1.0, abs=1e-4)

    def test_portfolio_volatility_positive(self, two_asset_cov):
        result = risk_parity(two_asset_cov)
        assert result.portfolio_volatility > 0

    def test_converged(self, three_asset_cov):
        result = risk_parity(three_asset_cov)
        assert result.converged


# ── Equal risk contribution ────────────────────────────────────────


class TestEqualRiskContribution:
    def test_two_asset_equal_risk(self, two_asset_cov):
        result = risk_parity(two_asset_cov)
        # Each asset should contribute ~50% of risk
        assert result.risk_contribution_pct[0] == pytest.approx(0.5, abs=0.02)
        assert result.risk_contribution_pct[1] == pytest.approx(0.5, abs=0.02)

    def test_three_asset_equal_risk(self, three_asset_cov):
        result = risk_parity(three_asset_cov)
        target = 1.0 / 3
        for rc in result.risk_contribution_pct:
            assert rc == pytest.approx(target, abs=0.03)

    def test_uncorrelated_equal_risk(self, uncorrelated_cov):
        """With zero correlations, risk parity should achieve near-perfect ERC."""
        result = risk_parity(uncorrelated_cov)
        target = 1.0 / 3
        for rc in result.risk_contribution_pct:
            assert rc == pytest.approx(target, abs=0.01)

    def test_low_vol_gets_more_weight(self, two_asset_cov):
        """Lower-vol asset needs more weight to contribute equal risk."""
        result = risk_parity(two_asset_cov)
        # Asset 1 (10% vol) should have more weight than asset 0 (20% vol)
        assert result.weights[1] > result.weights[0]

    def test_five_asset_equal_risk(self, five_asset_cov):
        result = risk_parity(five_asset_cov)
        target = 0.2
        for rc in result.risk_contribution_pct:
            assert rc == pytest.approx(target, abs=0.05)


# ── Custom risk budgets ───────────────────────────────────────────


class TestCustomRiskBudget:
    def test_custom_budget_respected(self, two_asset_cov):
        budget = np.array([0.7, 0.3])
        result = risk_parity(two_asset_cov, risk_budget=budget)
        # Asset 0 should contribute ~70% of risk
        assert result.risk_contribution_pct[0] > result.risk_contribution_pct[1]

    def test_budget_normalization(self, two_asset_cov):
        """Budget that doesn't sum to 1 should be auto-normalized."""
        budget = np.array([2.0, 1.0])  # 2:1 ratio = 0.667:0.333
        result = risk_parity(two_asset_cov, risk_budget=budget)
        assert result.target_budget.sum() == pytest.approx(1.0, abs=1e-6)

    def test_target_budget_stored(self, three_asset_cov):
        budget = np.array([0.5, 0.3, 0.2])
        result = risk_parity(three_asset_cov, risk_budget=budget)
        np.testing.assert_allclose(result.target_budget, budget, atol=1e-6)


# ── Single asset edge case ─────────────────────────────────────────


class TestSingleAsset:
    def test_single_asset_weight_one(self):
        cov = np.array([[0.04]])
        result = risk_parity(cov, tickers=["ONLY"])
        assert result.weights[0] == pytest.approx(1.0)
        assert result.converged is True

    def test_single_asset_risk(self):
        cov = np.array([[0.04]])
        result = risk_parity(cov)
        assert result.portfolio_volatility == pytest.approx(0.2, abs=1e-6)
        assert result.risk_contribution_pct[0] == pytest.approx(1.0)


# ── Property accessors ─────────────────────────────────────────────


class TestPropertyAccessors:
    def test_weight_dict(self, two_asset_cov):
        result = risk_parity(two_asset_cov, tickers=["SPY", "TLT"])
        wd = result.weight_dict
        assert "SPY" in wd
        assert "TLT" in wd
        assert sum(wd.values()) == pytest.approx(1.0, abs=1e-6)

    def test_risk_dict(self, two_asset_cov):
        result = risk_parity(two_asset_cov, tickers=["SPY", "TLT"])
        rd = result.risk_dict
        assert "SPY" in rd
        assert "TLT" in rd
        assert sum(rd.values()) == pytest.approx(1.0, abs=1e-4)


# ── Inverse volatility ────────────────────────────────────────────


class TestInverseVolatility:
    def test_weights_sum_to_one(self, three_asset_cov):
        result = inverse_volatility(three_asset_cov)
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-8)

    def test_all_weights_positive(self, three_asset_cov):
        result = inverse_volatility(three_asset_cov)
        assert np.all(result.weights > 0)

    def test_low_vol_gets_more_weight(self, uncorrelated_cov):
        result = inverse_volatility(uncorrelated_cov, tickers=["HI", "MED", "LO"])
        # 5% vol asset should get most weight
        assert result.weights[2] > result.weights[1] > result.weights[0]

    def test_converged_always_true(self, two_asset_cov):
        result = inverse_volatility(two_asset_cov)
        assert result.converged is True

    def test_returns_risk_parity_result(self, two_asset_cov):
        result = inverse_volatility(two_asset_cov)
        assert isinstance(result, RiskParityResult)

    def test_uncorrelated_matches_risk_parity(self, uncorrelated_cov):
        """For uncorrelated assets, inverse-vol ≈ risk parity."""
        rp = risk_parity(uncorrelated_cov)
        iv = inverse_volatility(uncorrelated_cov)
        np.testing.assert_allclose(rp.weights, iv.weights, atol=0.02)


# ── Validation ─────────────────────────────────────────────────────


class TestValidation:
    def test_ticker_count_mismatch_raises(self, two_asset_cov):
        with pytest.raises(ValueError, match="Expected 2 tickers"):
            risk_parity(two_asset_cov, tickers=["A", "B", "C"])

    def test_budget_count_mismatch_raises(self, two_asset_cov):
        with pytest.raises(ValueError, match="Expected 2 risk budget"):
            risk_parity(two_asset_cov, risk_budget=np.array([0.5, 0.3, 0.2]))

    def test_default_tickers_are_indices(self, two_asset_cov):
        result = risk_parity(two_asset_cov)
        assert result.tickers == ["0", "1"]

    def test_inverse_vol_ticker_mismatch_raises(self, two_asset_cov):
        with pytest.raises(ValueError, match="Expected 2 tickers"):
            inverse_volatility(two_asset_cov, tickers=["A"])


# ── Comparison with other methods ──────────────────────────────────


class TestComparisons:
    def test_risk_parity_more_balanced_than_equal_weight(self, five_asset_cov):
        """Risk parity should have more balanced risk contributions than 1/N."""
        rp = risk_parity(five_asset_cov)

        # Equal weight risk contributions
        ew = np.ones(5) / 5
        marginal = five_asset_cov @ ew
        ew_vol = np.sqrt(float(ew @ five_asset_cov @ ew))
        ew_rc = ew * marginal / ew_vol
        ew_rc_pct = ew_rc / ew_rc.sum()

        # Risk parity should have lower std of risk contributions
        rp_std = np.std(rp.risk_contribution_pct)
        ew_std = np.std(ew_rc_pct)
        assert rp_std < ew_std

    def test_different_from_min_variance(self, three_asset_cov):
        """Risk parity should differ from min-variance (which concentrates in low-vol)."""
        rp = risk_parity(three_asset_cov)
        # Min-variance would put even more weight on the lowest-vol asset
        # Risk parity balances risk, not variance
        assert rp.weights[0] > 0.05  # high-vol asset still gets meaningful weight
