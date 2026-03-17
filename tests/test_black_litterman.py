"""Tests for the Black-Litterman model."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from trading_master.quant.black_litterman import (
    _confidence_to_uncertainty,
    black_litterman_returns,
    bl_optimal_weights,
    implied_equilibrium_returns,
    run_black_litterman,
    signal_to_views,
)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def two_asset_cov():
    """Simple 2-asset covariance matrix (annualized)."""
    return np.array([
        [0.04, 0.006],
        [0.006, 0.01],
    ])


@pytest.fixture
def two_asset_weights():
    """Equal weights for 2 assets."""
    return np.array([0.5, 0.5])


# ── implied_equilibrium_returns ──────────────────────────────────────

class TestImpliedEquilibriumReturns:
    def test_basic(self, two_asset_cov, two_asset_weights):
        pi = implied_equilibrium_returns(two_asset_cov, two_asset_weights, risk_aversion=2.5)
        # Pi = 2.5 * [[0.04, 0.006], [0.006, 0.01]] @ [0.5, 0.5]
        # = 2.5 * [0.023, 0.008] = [0.0575, 0.02]
        assert pi.shape == (2,)
        np.testing.assert_allclose(pi, [0.0575, 0.02], atol=1e-10)

    def test_single_asset(self):
        cov = np.array([[0.04]])
        w = np.array([1.0])
        pi = implied_equilibrium_returns(cov, w, risk_aversion=3.0)
        np.testing.assert_allclose(pi, [0.12], atol=1e-10)

    def test_different_risk_aversion(self, two_asset_cov, two_asset_weights):
        pi_low = implied_equilibrium_returns(two_asset_cov, two_asset_weights, risk_aversion=1.0)
        pi_high = implied_equilibrium_returns(two_asset_cov, two_asset_weights, risk_aversion=5.0)
        # Higher risk aversion → higher implied returns
        assert np.all(pi_high > pi_low)


# ── signal_to_views ──────────────────────────────────────────────────

class TestSignalToViews:
    def test_strong_buy_mapping(self):
        reports = [{"ticker": "AAPL", "signal": "STRONG_BUY", "confidence": 90}]
        tickers = ["AAPL", "MSFT"]
        P, Q, Omega = signal_to_views(reports, tickers)

        assert P.shape == (1, 2)
        assert Q.shape == (1,)
        assert Omega.shape == (1, 1)

        # P should pick AAPL (index 0)
        np.testing.assert_array_equal(P[0], [1.0, 0.0])
        # STRONG_BUY → +10%
        assert Q[0] == pytest.approx(0.10)

    def test_sell_mapping(self):
        reports = [{"ticker": "MSFT", "signal": "SELL", "confidence": 70}]
        tickers = ["AAPL", "MSFT"]
        P, Q, Omega = signal_to_views(reports, tickers)

        np.testing.assert_array_equal(P[0], [0.0, 1.0])
        assert Q[0] == pytest.approx(-0.05)

    def test_hold_mapping(self):
        reports = [{"ticker": "AAPL", "signal": "HOLD", "confidence": 50}]
        tickers = ["AAPL"]
        _, Q, _ = signal_to_views(reports, tickers)
        assert Q[0] == pytest.approx(0.0)

    def test_confidence_to_uncertainty(self):
        # High confidence → low uncertainty
        u_high = _confidence_to_uncertainty(100)
        u_low = _confidence_to_uncertainty(0)
        assert u_high < u_low
        assert u_high == pytest.approx(0.001)
        assert u_low == pytest.approx(0.25)

    def test_confidence_omega_diagonal(self):
        reports = [
            {"ticker": "AAPL", "signal": "BUY", "confidence": 90},
            {"ticker": "MSFT", "signal": "SELL", "confidence": 30},
        ]
        tickers = ["AAPL", "MSFT"]
        _, _, Omega = signal_to_views(reports, tickers)

        # Omega is diagonal
        assert Omega[0, 1] == 0.0
        assert Omega[1, 0] == 0.0
        # High confidence → smaller diagonal entry
        assert Omega[0, 0] < Omega[1, 1]

    def test_multiple_reports_same_ticker_averaged(self):
        reports = [
            {"ticker": "AAPL", "signal": "BUY", "confidence": 80},
            {"ticker": "AAPL", "signal": "STRONG_BUY", "confidence": 90},
        ]
        tickers = ["AAPL"]
        P, Q, Omega = signal_to_views(reports, tickers)

        # Should produce 1 view (averaged)
        assert P.shape == (1, 1)
        assert Q[0] == pytest.approx((0.05 + 0.10) / 2)

    def test_no_matching_tickers(self):
        reports = [{"ticker": "TSLA", "signal": "BUY", "confidence": 80}]
        tickers = ["AAPL", "MSFT"]
        P, Q, Omega = signal_to_views(reports, tickers)
        assert P.shape[0] == 0

    def test_empty_reports(self):
        P, Q, Omega = signal_to_views([], ["AAPL"])
        assert P.shape[0] == 0

    def test_signal_enum_value(self):
        """Handle Signal enum objects (not just strings)."""
        from trading_master.models import Signal

        reports = [{"ticker": "AAPL", "signal": Signal.STRONG_SELL, "confidence": 70}]
        tickers = ["AAPL"]
        _, Q, _ = signal_to_views(reports, tickers)
        assert Q[0] == pytest.approx(-0.10)


# ── black_litterman_returns ──────────────────────────────────────────

class TestBlackLittermanReturns:
    def test_no_views_returns_equilibrium(self, two_asset_cov):
        pi = np.array([0.06, 0.02])
        P = np.zeros((0, 2))
        Q = np.zeros(0)
        Omega = np.zeros((0, 0))

        bl = black_litterman_returns(two_asset_cov, pi, P, Q, Omega)
        np.testing.assert_allclose(bl, pi)

    def test_strong_view_shifts_returns(self, two_asset_cov):
        """A strong bullish view on asset 0 should increase its BL return."""
        pi = np.array([0.06, 0.02])
        P = np.array([[1.0, 0.0]])
        Q = np.array([0.15])  # Very bullish on asset 0
        Omega = np.array([[0.01]])  # High confidence

        bl = black_litterman_returns(two_asset_cov, pi, P, Q, Omega)
        # BL return for asset 0 should be pulled toward 0.15
        assert bl[0] > pi[0]

    def test_formula_2asset(self, two_asset_cov):
        """Verify BL formula with a manual computation for 2 assets."""
        tau = 0.05
        pi = np.array([0.06, 0.02])
        P = np.array([[1.0, 0.0]])
        Q = np.array([0.10])
        Omega = np.array([[0.025]])

        cov = two_asset_cov
        tau_sigma = tau * cov
        tau_sigma_inv = np.linalg.inv(tau_sigma)
        omega_inv = np.linalg.inv(Omega)

        precision = tau_sigma_inv + P.T @ omega_inv @ P
        mean_part = tau_sigma_inv @ pi + P.T @ omega_inv @ Q
        expected = np.linalg.solve(precision, mean_part)

        bl = black_litterman_returns(cov, pi, P, Q, Omega, tau=tau)
        np.testing.assert_allclose(bl, expected, atol=1e-10)


# ── bl_optimal_weights ───────────────────────────────────────────────

class TestBLOptimalWeights:
    def test_long_only_non_negative(self, two_asset_cov):
        bl_ret = np.array([0.08, -0.02])  # Bearish on asset 2
        w = bl_optimal_weights(two_asset_cov, bl_ret, long_only=True)
        assert np.all(w >= 0)
        assert w.sum() == pytest.approx(1.0)

    def test_long_only_sums_to_one(self, two_asset_cov):
        bl_ret = np.array([0.06, 0.04])
        w = bl_optimal_weights(two_asset_cov, bl_ret, long_only=True)
        assert w.sum() == pytest.approx(1.0)

    def test_allow_short(self, two_asset_cov):
        bl_ret = np.array([0.10, -0.05])
        w = bl_optimal_weights(two_asset_cov, bl_ret, long_only=False)
        assert w.sum() == pytest.approx(1.0)
        # With a negative return for asset 2, it might get shorted

    def test_higher_return_gets_more_weight(self, two_asset_cov):
        bl_ret = np.array([0.10, 0.02])
        w = bl_optimal_weights(two_asset_cov, bl_ret, long_only=True)
        assert w[0] > w[1]


# ── run_black_litterman (end-to-end with mocked covariance) ─────────

class TestRunBlackLitterman:
    def test_end_to_end(self):
        """Full pipeline with mocked market data."""
        mock_cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.03, 0.008],
            [0.005, 0.008, 0.02],
        ])
        mock_corr = np.eye(3)  # Not used by BL
        valid_tickers = ["AAPL", "MSFT", "GOOGL"]

        reports = [
            {"ticker": "AAPL", "signal": "STRONG_BUY", "confidence": 85},
            {"ticker": "MSFT", "signal": "BUY", "confidence": 70},
            {"ticker": "GOOGL", "signal": "SELL", "confidence": 60},
        ]

        with patch(
            "trading_master.portfolio.correlation.rolling_covariance",
            return_value=(mock_cov, mock_corr, valid_tickers),
        ):
            result = run_black_litterman(
                tickers=["AAPL", "MSFT", "GOOGL"],
                analyst_reports=reports,
            )

        assert "error" not in result
        assert result["tickers"] == valid_tickers
        assert len(result["bl_returns"]) == 3
        assert len(result["optimal_weights"]) == 3
        assert len(result["suggested_trades"]) == 3

        # Weights should sum to 1
        np.testing.assert_allclose(result["optimal_weights"].sum(), 1.0, atol=1e-10)

        # AAPL (STRONG_BUY) should have highest BL return
        assert result["bl_returns"][0] > result["bl_returns"][2]

        # Each trade entry has the right keys
        for t in result["suggested_trades"]:
            assert "ticker" in t
            assert "current_pct" in t
            assert "target_pct" in t
            assert "direction" in t

    def test_end_to_end_no_views(self):
        """Pipeline with no analyst reports uses equilibrium only."""
        mock_cov = np.array([[0.04, 0.01], [0.01, 0.02]])
        valid_tickers = ["AAPL", "MSFT"]

        with patch(
            "trading_master.portfolio.correlation.rolling_covariance",
            return_value=(mock_cov, np.eye(2), valid_tickers),
        ):
            result = run_black_litterman(
                tickers=["AAPL", "MSFT"],
                analyst_reports=[],
            )

        assert "error" not in result
        # Without views, BL returns should equal equilibrium returns
        np.testing.assert_allclose(
            result["bl_returns"],
            result["equilibrium_returns"],
            atol=1e-10,
        )

    def test_covariance_failure(self):
        """When covariance computation fails, return error dict."""
        with patch(
            "trading_master.portfolio.correlation.rolling_covariance",
            return_value=(None, None, []),
        ):
            result = run_black_litterman(
                tickers=["AAPL"],
                analyst_reports=[],
            )

        assert "error" in result

    def test_with_current_weights(self):
        """Pipeline respects provided current_weights."""
        mock_cov = np.array([[0.04, 0.01], [0.01, 0.02]])
        valid_tickers = ["AAPL", "MSFT"]

        with patch(
            "trading_master.portfolio.correlation.rolling_covariance",
            return_value=(mock_cov, np.eye(2), valid_tickers),
        ):
            result = run_black_litterman(
                tickers=["AAPL", "MSFT"],
                analyst_reports=[],
                current_weights=np.array([0.7, 0.3]),
            )

        assert "error" not in result
        np.testing.assert_allclose(result["current_weights"], [0.7, 0.3], atol=1e-10)
