"""Tests for Regime Switching (HMM) model."""

from __future__ import annotations

import numpy as np
import pytest

from trading_master.quant.regime import RegimeResult, fit_regime_model


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def clear_two_regime():
    """Synthetic returns with two very distinct regimes."""
    rng = np.random.default_rng(42)
    bull = rng.normal(0.001, 0.01, size=200)   # low-vol positive
    bear = rng.normal(-0.002, 0.03, size=200)  # high-vol negative
    return np.concatenate([bull, bear, bull])


@pytest.fixture
def three_regime_data():
    """Synthetic returns with bull, neutral, and bear phases."""
    rng = np.random.default_rng(42)
    bull = rng.normal(0.002, 0.008, size=150)
    neutral = rng.normal(0.0, 0.015, size=150)
    bear = rng.normal(-0.003, 0.035, size=100)
    return np.concatenate([bull, neutral, bear, bull])


@pytest.fixture
def random_returns():
    """Generic random returns for basic property tests."""
    rng = np.random.default_rng(42)
    return rng.normal(0.0005, 0.02, size=500)


# ── Basic properties ───────────────────────────────────────────────


class TestRegimeBasicProperties:
    def test_returns_regime_result(self, random_returns):
        result = fit_regime_model(random_returns, n_regimes=2)
        assert isinstance(result, RegimeResult)

    def test_correct_n_regimes(self, random_returns):
        result = fit_regime_model(random_returns, n_regimes=2)
        assert result.n_regimes == 2
        assert len(result.means) == 2
        assert len(result.volatilities) == 2

    def test_three_regimes(self, random_returns):
        result = fit_regime_model(random_returns, n_regimes=3)
        assert result.n_regimes == 3
        assert len(result.regime_labels) == 3

    def test_transition_matrix_rows_sum_to_one(self, random_returns):
        result = fit_regime_model(random_returns, n_regimes=2)
        row_sums = result.transition_matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_transition_matrix_non_negative(self, random_returns):
        result = fit_regime_model(random_returns, n_regimes=2)
        assert np.all(result.transition_matrix >= 0)

    def test_stationary_probs_sum_to_one(self, random_returns):
        result = fit_regime_model(random_returns, n_regimes=2)
        assert result.stationary_probs.sum() == pytest.approx(1.0, abs=1e-4)

    def test_regime_sequence_length(self, random_returns):
        result = fit_regime_model(random_returns, n_regimes=2)
        assert len(result.regime_sequence) == len(random_returns)

    def test_regime_probs_shape(self, random_returns):
        result = fit_regime_model(random_returns, n_regimes=3)
        assert result.regime_probs.shape == (len(random_returns), 3)

    def test_regime_probs_sum_to_one(self, random_returns):
        result = fit_regime_model(random_returns, n_regimes=2)
        row_sums = result.regime_probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.01)

    def test_volatilities_positive(self, random_returns):
        result = fit_regime_model(random_returns, n_regimes=2)
        assert np.all(result.volatilities > 0)

    def test_variances_match_volatilities(self, random_returns):
        result = fit_regime_model(random_returns, n_regimes=2)
        np.testing.assert_allclose(result.variances, result.volatilities**2, atol=1e-10)


# ── Regime sorting ─────────────────────────────────────────────────


class TestRegimeSorting:
    def test_means_sorted_ascending(self, clear_two_regime):
        result = fit_regime_model(clear_two_regime, n_regimes=2)
        assert result.means[0] <= result.means[1]

    def test_three_regime_means_sorted(self, three_regime_data):
        result = fit_regime_model(three_regime_data, n_regimes=3)
        assert result.means[0] <= result.means[1] <= result.means[2]

    def test_bear_has_higher_vol_than_bull(self, clear_two_regime):
        result = fit_regime_model(clear_two_regime, n_regimes=2)
        # Bear (index 0, lowest mean) should have higher vol
        assert result.volatilities[0] > result.volatilities[1]


# ── Regime detection accuracy ──────────────────────────────────────


class TestRegimeDetection:
    def test_detects_regime_switch(self, clear_two_regime):
        result = fit_regime_model(clear_two_regime, n_regimes=2)
        # First 200 points are bull, next 200 are bear
        # Check that regimes differ between the two blocks
        first_block = result.regime_sequence[:150]
        second_block = result.regime_sequence[250:350]
        # Majority of first block should be one regime, second block the other
        first_mode = np.bincount(first_block).argmax()
        second_mode = np.bincount(second_block).argmax()
        assert first_mode != second_mode

    def test_current_regime_valid(self, random_returns):
        result = fit_regime_model(random_returns, n_regimes=2)
        assert result.current_regime in (0, 1)

    def test_current_probs_sum_to_one(self, random_returns):
        result = fit_regime_model(random_returns, n_regimes=2)
        assert result.current_probs.sum() == pytest.approx(1.0, abs=0.01)


# ── Labels ─────────────────────────────────────────────────────────


class TestRegimeLabels:
    def test_two_regime_labels(self, random_returns):
        result = fit_regime_model(random_returns, n_regimes=2)
        assert result.regime_labels == ["bear", "bull"]

    def test_three_regime_labels(self, random_returns):
        result = fit_regime_model(random_returns, n_regimes=3)
        assert result.regime_labels == ["bear", "neutral", "bull"]

    def test_current_label(self, random_returns):
        result = fit_regime_model(random_returns, n_regimes=2)
        assert result.current_label in ("bear", "bull")

    def test_regime_summary(self, random_returns):
        result = fit_regime_model(random_returns, n_regimes=2)
        summary = result.regime_summary
        assert "bear" in summary
        assert "bull" in summary
        assert "volatility" in summary["bear"]
        assert "persistence" in summary["bull"]


# ── Self-persistence ───────────────────────────────────────────────


class TestPersistence:
    def test_high_self_persistence(self, clear_two_regime):
        """Regimes should be 'sticky' — high diagonal in transition matrix."""
        result = fit_regime_model(clear_two_regime, n_regimes=2)
        for i in range(2):
            assert result.transition_matrix[i, i] > 0.5


# ── Validation ─────────────────────────────────────────────────────


class TestValidation:
    def test_too_few_observations(self):
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="at least 30"):
            fit_regime_model(rng.normal(size=20), n_regimes=2)

    def test_too_few_regimes(self, random_returns):
        with pytest.raises(ValueError, match="at least 2"):
            fit_regime_model(random_returns, n_regimes=1)

    def test_handles_nan(self, random_returns):
        data = random_returns.copy()
        data[0] = np.nan
        data[10] = np.inf
        result = fit_regime_model(data, n_regimes=2)
        assert result.n_regimes == 2

    def test_reproducible_with_seed(self, random_returns):
        r1 = fit_regime_model(random_returns, n_regimes=2, seed=99)
        r2 = fit_regime_model(random_returns, n_regimes=2, seed=99)
        np.testing.assert_allclose(r1.means, r2.means, atol=1e-8)


# ── Convergence ────────────────────────────────────────────────────


class TestConvergence:
    def test_converges_on_clear_data(self, clear_two_regime):
        result = fit_regime_model(clear_two_regime, n_regimes=2, max_iter=200)
        assert result.converged

    def test_log_likelihood_finite(self, random_returns):
        result = fit_regime_model(random_returns, n_regimes=2)
        assert np.isfinite(result.log_likelihood)

    def test_n_iterations_reasonable(self, random_returns):
        result = fit_regime_model(random_returns, n_regimes=2, max_iter=200)
        assert 1 <= result.n_iterations <= 200
