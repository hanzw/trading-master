"""Tests for Monte Carlo portfolio simulation."""

from __future__ import annotations

import numpy as np
import pytest

from trading_master.quant.monte_carlo import (
    simulate_portfolio_paths,
    stress_test_scenarios,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_portfolio():
    """60/40 equities/bonds portfolio."""
    weights = np.array([0.6, 0.4])
    expected_returns = np.array([0.10, 0.04])
    cov_matrix = np.array([[0.04, 0.005], [0.005, 0.0025]])
    return weights, expected_returns, cov_matrix


# ---------------------------------------------------------------------------
# simulate_portfolio_paths
# ---------------------------------------------------------------------------

class TestSimulatePortfolioPaths:
    def test_path_shapes(self, simple_portfolio):
        w, mu, cov = simple_portfolio
        res = simulate_portfolio_paths(w, mu, cov, n_simulations=100, horizon_days=50)
        assert res["paths"].shape == (100, 50)
        assert res["final_values"].shape == (100,)

    def test_percentile_ordering(self, simple_portfolio):
        w, mu, cov = simple_portfolio
        res = simulate_portfolio_paths(w, mu, cov, n_simulations=5000)
        pcts = res["percentiles"]
        assert pcts[5] <= pcts[25] <= pcts[50] <= pcts[75] <= pcts[95]

    def test_prob_loss_in_range(self, simple_portfolio):
        w, mu, cov = simple_portfolio
        res = simulate_portfolio_paths(w, mu, cov, n_simulations=1000)
        assert 0.0 <= res["prob_loss"] <= 1.0
        assert 0.0 <= res["prob_gain_20pct"] <= 1.0

    def test_deterministic_with_seed(self, simple_portfolio):
        w, mu, cov = simple_portfolio
        r1 = simulate_portfolio_paths(w, mu, cov, n_simulations=200, seed=123)
        r2 = simulate_portfolio_paths(w, mu, cov, n_simulations=200, seed=123)
        np.testing.assert_array_equal(r1["paths"], r2["paths"])

    def test_different_seeds_differ(self, simple_portfolio):
        w, mu, cov = simple_portfolio
        r1 = simulate_portfolio_paths(w, mu, cov, n_simulations=200, seed=1)
        r2 = simulate_portfolio_paths(w, mu, cov, n_simulations=200, seed=2)
        assert not np.allclose(r1["final_values"], r2["final_values"])

    def test_max_drawdown_non_negative(self, simple_portfolio):
        w, mu, cov = simple_portfolio
        res = simulate_portfolio_paths(w, mu, cov, n_simulations=500)
        assert res["max_drawdown_median"] >= 0.0

    def test_single_asset(self):
        """Single-asset portfolio should work."""
        w = np.array([1.0])
        mu = np.array([0.08])
        cov = np.array([[0.03]])
        res = simulate_portfolio_paths(w, mu, cov, n_simulations=100, horizon_days=20)
        assert res["paths"].shape == (100, 20)

    def test_expected_value_positive_drift(self, simple_portfolio):
        """With positive expected returns, mean terminal > initial."""
        w, mu, cov = simple_portfolio
        res = simulate_portfolio_paths(w, mu, cov, n_simulations=10000, seed=42)
        assert res["expected_value"] > 1_000_000.0


# ---------------------------------------------------------------------------
# stress_test_scenarios
# ---------------------------------------------------------------------------

class TestStressTestScenarios:
    def test_all_scenarios_present(self):
        w = np.array([0.6, 0.4])
        cov = np.array([[0.04, 0.005], [0.005, 0.0025]])
        results = stress_test_scenarios(w, cov, 1_000_000.0, ["equities", "bonds"])
        assert len(results) == 5
        names = {r["scenario"] for r in results}
        assert "2008 Financial Crisis" in names
        assert "Flash Crash" in names

    def test_dollar_impact_matches(self):
        w = np.array([1.0])
        cov = np.array([[0.04]])
        val = 500_000.0
        results = stress_test_scenarios(w, cov, val, ["equities"])
        for r in results:
            assert pytest.approx(r["dollar_impact"]) == r["return_pct"] * val

    def test_default_asset_classes(self):
        """When asset_classes is None, all treated as equities."""
        w = np.array([0.5, 0.5])
        cov = np.eye(2) * 0.04
        results = stress_test_scenarios(w, cov, 1_000_000.0)
        # 2008 equities shock is -38%, so portfolio return = -38%
        crisis = [r for r in results if "2008" in r["scenario"]][0]
        assert pytest.approx(crisis["return_pct"]) == -0.38

    def test_mixed_portfolio(self):
        w = np.array([0.5, 0.3, 0.2])
        cov = np.eye(3) * 0.01
        val = 1_000_000.0
        results = stress_test_scenarios(
            w, cov, val, ["equities", "bonds", "commodities"]
        )
        # 2008: 0.5*(-0.38) + 0.3*(0.20) + 0.2*(-0.25) = -0.19 + 0.06 - 0.05 = -0.18
        crisis = [r for r in results if "2008" in r["scenario"]][0]
        assert pytest.approx(crisis["return_pct"], abs=1e-10) == -0.18
