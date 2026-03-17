"""Tests for portfolio.correlation module."""

import numpy as np
import pytest

from trading_master.portfolio.correlation import (
    concentration_risk,
    minimum_variance_weights,
)


# ── Concentration risk ────────────────────────────────────────────────

class TestConcentrationRisk:
    def test_identity_matrix(self):
        """Identity matrix => all eigenvalues equal, not concentrated."""
        n = 5
        cov = np.eye(n)
        result = concentration_risk(cov)
        evr = result["explained_variance_ratios"]

        # All ratios should be equal (1/n)
        np.testing.assert_allclose(evr, np.ones(n) / n, atol=1e-10)
        assert result["concentrated"] is False
        # Effective number of bets should be close to n
        assert abs(result["effective_num_bets"] - n) < 0.01

    def test_rank1_matrix(self):
        """Rank-1 matrix => one dominant eigenvalue => concentrated."""
        v = np.array([1, 2, 3, 4, 5], dtype=float)
        cov = np.outer(v, v)  # rank 1
        result = concentration_risk(cov)

        assert result["top1_dominance"] > 0.99
        assert result["concentrated"] is True
        assert result["effective_num_bets"] < 1.1

    def test_empty(self):
        result = concentration_risk(np.array([]))
        assert result["effective_num_bets"] == 0.0
        assert result["concentrated"] is False

    def test_single_asset(self):
        cov = np.array([[0.04]])
        result = concentration_risk(cov)
        assert result["top1_dominance"] == 1.0


# ── Minimum variance weights ──────────────────────────────────────────

class TestMinimumVarianceWeights:
    def test_sums_to_one(self):
        rng = np.random.RandomState(42)
        n = 5
        A = rng.randn(100, n)
        cov = np.cov(A, rowvar=False)
        w = minimum_variance_weights(cov)
        assert abs(w.sum() - 1.0) < 1e-10

    def test_identity_equal_weights(self):
        """With identity covariance, min-var weights are equal."""
        n = 4
        cov = np.eye(n)
        w = minimum_variance_weights(cov)
        np.testing.assert_allclose(w, np.ones(n) / n, atol=1e-10)

    def test_single_asset(self):
        w = minimum_variance_weights(np.array([[0.04]]))
        assert len(w) == 1
        assert abs(w[0] - 1.0) < 1e-10

    def test_empty(self):
        w = minimum_variance_weights(np.array([]))
        assert len(w) == 0

    def test_low_variance_asset_gets_higher_weight(self):
        """Asset with lower variance should get more weight in min-var portfolio."""
        # Diagonal cov: asset 0 has low var, asset 1 has high var
        cov = np.diag([0.01, 1.0])
        w = minimum_variance_weights(cov)
        assert abs(w.sum() - 1.0) < 1e-10
        assert w[0] > w[1]


# ── check_correlation_ok with synthetic data ──────────────────────────
# These tests mock fetch_returns to avoid network calls.

class TestCheckCorrelationOk:
    def test_perfectly_correlated(self, monkeypatch):
        """Perfectly correlated assets should fail the check."""
        from trading_master.portfolio import correlation as mod

        rng = np.random.RandomState(10)
        base = rng.normal(0, 0.01, 200)
        # All tickers return same series
        def mock_fetch(tickers, lookback_days=200):
            n = len(tickers)
            returns = np.column_stack([base[:200]] * n)
            return returns, tickers

        monkeypatch.setattr(mod, "fetch_returns", mock_fetch)

        ok, avg, details = mod.check_correlation_ok("NEW", ["A", "B"], max_avg_correlation=0.7)
        assert ok is False
        assert avg > 0.99

    def test_uncorrelated(self, monkeypatch):
        """Uncorrelated assets should pass the check."""
        from trading_master.portfolio import correlation as mod

        rng = np.random.RandomState(20)

        def mock_fetch(tickers, lookback_days=200):
            n = len(tickers)
            returns = rng.normal(0, 0.01, (200, n))
            return returns, tickers

        monkeypatch.setattr(mod, "fetch_returns", mock_fetch)

        ok, avg, details = mod.check_correlation_ok("NEW", ["A", "B"], max_avg_correlation=0.7)
        assert ok is True
        assert avg < 0.3

    def test_no_existing_tickers(self):
        """No existing tickers => always ok."""
        from trading_master.portfolio.correlation import check_correlation_ok
        ok, avg, details = check_correlation_ok("NEW", [])
        assert ok is True
        assert avg == 0.0
