"""Tests for the CAPM + Jensen's Alpha model."""

from __future__ import annotations

import numpy as np
import pytest

from trading_master.quant.capm import (
    CAPMResult,
    capm_expected_return,
    capm_portfolio,
    capm_regression,
    security_market_line,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def market_returns():
    """Simulated market excess returns."""
    rng = np.random.default_rng(42)
    return rng.normal(0.0003, 0.01, 500)


@pytest.fixture
def asset_with_known_beta(market_returns):
    """Asset with beta=1.5, alpha=0.0002 daily."""
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 0.005, len(market_returns))
    return 0.0002 + 1.5 * market_returns + noise


# ── capm_regression ─────────────────────────────────────────────────


class TestCAPMRegression:
    def test_recovers_beta(self, market_returns, asset_with_known_beta):
        result = capm_regression(asset_with_known_beta, market_returns, "TEST")
        assert result.beta == pytest.approx(1.5, abs=0.1)

    def test_alpha_reasonable_magnitude(self, market_returns, asset_with_known_beta):
        result = capm_regression(asset_with_known_beta, market_returns, "TEST")
        # Daily alpha should be small in magnitude (noise can shift the sign)
        assert abs(result.alpha) < 0.005

    def test_r_squared(self, market_returns, asset_with_known_beta):
        result = capm_regression(asset_with_known_beta, market_returns, "TEST")
        # Good signal-to-noise, R² should be high
        assert result.r_squared > 0.5

    def test_ticker_stored(self, market_returns, asset_with_known_beta):
        result = capm_regression(asset_with_known_beta, market_returns, "AAPL")
        assert result.ticker == "AAPL"

    def test_n_obs(self, market_returns, asset_with_known_beta):
        result = capm_regression(asset_with_known_beta, market_returns)
        assert result.n_obs == 500

    def test_perfect_market_tracking(self, market_returns):
        """Asset that perfectly tracks market: beta=1, alpha=0, R²=1."""
        result = capm_regression(market_returns, market_returns, "SPY")
        assert result.beta == pytest.approx(1.0, abs=1e-10)
        assert result.alpha == pytest.approx(0.0, abs=1e-10)
        assert result.r_squared == pytest.approx(1.0, abs=1e-10)

    def test_zero_beta(self, market_returns):
        """Asset uncorrelated with market."""
        rng = np.random.default_rng(123)
        asset = rng.normal(0, 0.01, len(market_returns))
        result = capm_regression(asset, market_returns)
        assert result.beta == pytest.approx(0.0, abs=0.15)

    def test_negative_beta(self, market_returns):
        """Asset inversely correlated with market."""
        asset = -0.8 * market_returns
        result = capm_regression(asset, market_returns)
        assert result.beta < 0
        assert result.beta == pytest.approx(-0.8, abs=0.01)

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            capm_regression(np.ones(100), np.ones(50))

    def test_too_few_observations(self):
        with pytest.raises(ValueError, match="at least 3"):
            capm_regression(np.ones(2), np.ones(2))

    def test_t_stats_significant_for_strong_signal(
        self, market_returns, asset_with_known_beta,
    ):
        result = capm_regression(asset_with_known_beta, market_returns)
        # Beta t-stat should be very significant
        assert abs(result.beta_t_stat) > 2.0


# ── CAPMResult properties ──────────────────────────────────────────


class TestCAPMResultProperties:
    def test_alpha_annualized(self):
        r = CAPMResult(
            ticker="X", alpha=0.0002, beta=1.0,
            r_squared=0.8, alpha_t_stat=2.5, beta_t_stat=30.0,
            n_obs=252, residual_std=0.01,
        )
        assert r.alpha_annualized == pytest.approx(0.0504, abs=1e-6)

    def test_information_ratio(self):
        r = CAPMResult(
            ticker="X", alpha=0.0002, beta=1.0,
            r_squared=0.8, alpha_t_stat=2.5, beta_t_stat=30.0,
            n_obs=252, residual_std=0.01,
        )
        # IR = annualized alpha / annualized tracking error
        ann_alpha = 0.0002 * 252
        ann_te = 0.01 * np.sqrt(252)
        assert r.information_ratio == pytest.approx(ann_alpha / ann_te, abs=1e-6)

    def test_is_alpha_significant_true(self):
        r = CAPMResult(
            ticker="X", alpha=0.001, beta=1.0,
            r_squared=0.8, alpha_t_stat=2.5, beta_t_stat=30.0,
            n_obs=252, residual_std=0.01,
        )
        assert r.is_alpha_significant is True

    def test_is_alpha_significant_false(self):
        r = CAPMResult(
            ticker="X", alpha=0.00001, beta=1.0,
            r_squared=0.8, alpha_t_stat=0.5, beta_t_stat=30.0,
            n_obs=252, residual_std=0.01,
        )
        assert r.is_alpha_significant is False


# ── capm_expected_return ────────────────────────────────────────────


class TestCAPMExpectedReturn:
    def test_market_beta(self):
        # beta=1: expected = Rf + 1*MRP = 0.04 + 0.08 = 0.12
        assert capm_expected_return(1.0) == pytest.approx(0.12)

    def test_zero_beta(self):
        # beta=0: expected = Rf = 0.04
        assert capm_expected_return(0.0) == pytest.approx(0.04)

    def test_high_beta(self):
        # beta=2: expected = 0.04 + 2*0.08 = 0.20
        assert capm_expected_return(2.0) == pytest.approx(0.20)

    def test_custom_rates(self):
        # beta=1.5, Rf=0.05, MRP=0.06
        expected = capm_expected_return(1.5, risk_free_rate=0.05, market_premium=0.06)
        assert expected == pytest.approx(0.05 + 1.5 * 0.06)

    def test_negative_beta(self):
        # beta=-0.5: expected = 0.04 + (-0.5)*0.08 = 0.0
        assert capm_expected_return(-0.5) == pytest.approx(0.0)


# ── capm_portfolio ──────────────────────────────────────────────────


class TestCAPMPortfolio:
    def test_multiple_assets(self, market_returns):
        rng = np.random.default_rng(5)
        n = len(market_returns)
        # 3 assets with different betas
        assets = np.column_stack([
            1.2 * market_returns + rng.normal(0, 0.003, n),
            0.8 * market_returns + rng.normal(0, 0.005, n),
            1.0 * market_returns + rng.normal(0, 0.002, n),
        ])

        results = capm_portfolio(assets, market_returns, ["A", "B", "C"])
        assert len(results) == 3
        assert results[0].ticker == "A"
        # Asset A has beta ~1.2
        assert results[0].beta == pytest.approx(1.2, abs=0.1)

    def test_1d_input(self, market_returns):
        asset = 1.0 * market_returns
        results = capm_portfolio(asset, market_returns, ["SPY"])
        assert len(results) == 1

    def test_mismatched_tickers(self, market_returns):
        assets = np.column_stack([market_returns, market_returns])
        with pytest.raises(ValueError, match="Expected 2 tickers"):
            capm_portfolio(assets, market_returns, ["A"])


# ── security_market_line ────────────────────────────────────────────


class TestSecurityMarketLine:
    def test_sml_output(self):
        results = [
            CAPMResult("AAPL", 0.0003, 1.2, 0.85, 3.0, 40.0, 252, 0.01),
            CAPMResult("T", -0.0002, 0.6, 0.70, -1.5, 20.0, 252, 0.008),
            CAPMResult("SPY", 0.0, 1.0, 0.99, 0.0, 100.0, 252, 0.001),
        ]

        sml = security_market_line(results)
        assert len(sml) == 3

        # AAPL: positive alpha → undervalued
        assert sml[0]["ticker"] == "AAPL"
        assert sml[0]["assessment"] == "undervalued"
        assert sml[0]["alpha_annualized"] > 0

        # T: negative alpha → overvalued
        assert sml[1]["ticker"] == "T"
        assert sml[1]["assessment"] == "overvalued"

        # SPY: zero alpha → fairly valued
        assert sml[2]["ticker"] == "SPY"
        assert sml[2]["assessment"] == "fairly_valued"

    def test_sml_expected_return(self):
        results = [
            CAPMResult("X", 0.0, 1.5, 0.9, 0.0, 50.0, 252, 0.01),
        ]
        sml = security_market_line(results, risk_free_rate=0.04, market_premium=0.08)
        # Expected = 0.04 + 1.5*0.08 = 0.16
        assert sml[0]["expected_return"] == pytest.approx(0.16, abs=0.001)

    def test_sml_keys(self):
        results = [CAPMResult("A", 0.001, 1.0, 0.8, 2.5, 30.0, 100, 0.01)]
        sml = security_market_line(results)
        entry = sml[0]
        assert "ticker" in entry
        assert "beta" in entry
        assert "expected_return" in entry
        assert "actual_return" in entry
        assert "alpha_annualized" in entry
        assert "assessment" in entry
        assert "alpha_significant" in entry
