"""Tests for the GARCH(1,1) volatility model."""

from __future__ import annotations

import numpy as np
import pytest

from trading_master.quant.garch import (
    GARCHResult,
    fit_garch,
    forecast_volatility,
    volatility_regime,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def simulated_garch_returns():
    """Simulate returns from a known GARCH(1,1) process."""
    rng = np.random.default_rng(42)
    n = 1000
    omega, alpha, beta = 0.00001, 0.08, 0.88

    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)  # unconditional variance

    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        returns[t] = rng.normal(0, np.sqrt(sigma2[t]))

    return returns, omega, alpha, beta


@pytest.fixture
def simple_returns():
    """Simple noisy return series."""
    rng = np.random.default_rng(99)
    return rng.normal(0.0005, 0.015, 500)


# ── fit_garch ───────────────────────────────────────────────────────


class TestFitGarch:
    def test_converges(self, simulated_garch_returns):
        returns, _, _, _ = simulated_garch_returns
        result = fit_garch(returns)
        assert result.converged

    def test_recovers_parameters(self, simulated_garch_returns):
        returns, omega, alpha, beta = simulated_garch_returns
        result = fit_garch(returns)

        # Parameters should be in the right ballpark
        # MLE can have estimation error, especially with 1000 obs
        assert result.alpha == pytest.approx(alpha, abs=0.08)
        assert result.beta == pytest.approx(beta, abs=0.15)
        assert result.persistence == pytest.approx(alpha + beta, abs=0.15)

    def test_stationarity(self, simulated_garch_returns):
        returns, _, _, _ = simulated_garch_returns
        result = fit_garch(returns)
        # alpha + beta < 1
        assert result.persistence < 1.0

    def test_conditional_volatility_shape(self, simulated_garch_returns):
        returns, _, _, _ = simulated_garch_returns
        result = fit_garch(returns)
        assert result.conditional_volatility.shape == (len(returns),)
        assert np.all(result.conditional_volatility > 0)

    def test_conditional_volatility_varies(self, simulated_garch_returns):
        returns, _, _, _ = simulated_garch_returns
        result = fit_garch(returns)
        vol = result.conditional_volatility
        # Volatility should not be constant (GARCH captures clustering)
        assert vol.std() > 0

    def test_n_obs(self, simulated_garch_returns):
        returns, _, _, _ = simulated_garch_returns
        result = fit_garch(returns)
        assert result.n_obs == len(returns)

    def test_log_likelihood_finite(self, simulated_garch_returns):
        returns, _, _, _ = simulated_garch_returns
        result = fit_garch(returns)
        assert np.isfinite(result.log_likelihood)

    def test_too_few_observations(self):
        with pytest.raises(ValueError, match="at least 10"):
            fit_garch(np.array([0.01, -0.02, 0.005]))

    def test_custom_initial_params(self, simple_returns):
        result = fit_garch(simple_returns, initial_params=(1e-5, 0.1, 0.8))
        assert result.converged
        assert result.omega > 0

    def test_long_run_volatility_reasonable(self, simple_returns):
        result = fit_garch(simple_returns)
        # Annualized long-run vol should be in a sensible range (1%-200%)
        assert 0.01 < result.long_run_volatility < 2.0

    def test_half_life_positive(self, simulated_garch_returns):
        returns, _, _, _ = simulated_garch_returns
        result = fit_garch(returns)
        assert result.half_life > 0
        assert np.isfinite(result.half_life)


# ── GARCHResult properties ──────────────────────────────────────────


class TestGARCHResultProperties:
    def test_persistence(self):
        r = GARCHResult(
            omega=1e-5, alpha=0.08, beta=0.88,
            log_likelihood=-1000.0,
            conditional_volatility=np.ones(100),
            n_obs=100, converged=True,
        )
        assert r.persistence == pytest.approx(0.96)

    def test_long_run_variance(self):
        r = GARCHResult(
            omega=0.00004, alpha=0.08, beta=0.88,
            log_likelihood=-1000.0,
            conditional_volatility=np.ones(100),
            n_obs=100, converged=True,
        )
        # omega / (1 - 0.96) = 0.00004 / 0.04 = 0.001
        assert r.long_run_variance == pytest.approx(0.001, abs=1e-8)

    def test_long_run_variance_unstable(self):
        r = GARCHResult(
            omega=1e-5, alpha=0.5, beta=0.6,
            log_likelihood=-1000.0,
            conditional_volatility=np.ones(100),
            n_obs=100, converged=True,
        )
        # alpha + beta >= 1 → infinite long-run variance
        assert r.long_run_variance == float("inf")

    def test_half_life(self):
        r = GARCHResult(
            omega=1e-5, alpha=0.05, beta=0.90,
            log_likelihood=-1000.0,
            conditional_volatility=np.ones(100),
            n_obs=100, converged=True,
        )
        # persistence = 0.95, half-life = ln(2) / -ln(0.95) ≈ 13.5 days
        assert r.half_life == pytest.approx(13.51, abs=0.1)


# ── forecast_volatility ────────────────────────────────────────────


class TestForecastVolatility:
    def test_forecast_shape(self, simulated_garch_returns):
        returns, _, _, _ = simulated_garch_returns
        result = fit_garch(returns)
        fc = forecast_volatility(result, returns, horizon=20)
        assert fc.shape == (20,)

    def test_forecast_positive(self, simulated_garch_returns):
        returns, _, _, _ = simulated_garch_returns
        result = fit_garch(returns)
        fc = forecast_volatility(result, returns, horizon=10)
        assert np.all(fc > 0)

    def test_forecast_converges_to_long_run(self, simulated_garch_returns):
        returns, _, _, _ = simulated_garch_returns
        result = fit_garch(returns)
        fc = forecast_volatility(result, returns, horizon=500)
        # Far-horizon forecast should approach long-run vol
        assert fc[-1] == pytest.approx(result.long_run_volatility, abs=0.01)

    def test_single_step_forecast(self, simple_returns):
        result = fit_garch(simple_returns)
        fc = forecast_volatility(result, simple_returns, horizon=1)
        assert len(fc) == 1
        assert fc[0] > 0


# ── volatility_regime ──────────────────────────────────────────────


class TestVolatilityRegime:
    def test_three_regimes(self):
        vol = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55])
        regimes = volatility_regime(vol)
        assert "low" in regimes
        assert "normal" in regimes
        assert "high" in regimes

    def test_all_same_vol(self):
        vol = np.full(100, 0.2)
        regimes = volatility_regime(vol)
        # With constant vol, quantile thresholds are all 0.2
        # Everything at or below low_thresh → "low", nothing above high_thresh
        # Actually with constant values, all quantiles = 0.2
        # so nothing < 0.2 (low) and nothing > 0.2 (high) → all "normal"
        assert all(r == "normal" for r in regimes)

    def test_custom_quantiles(self):
        vol = np.linspace(0.1, 0.5, 100)
        regimes = volatility_regime(vol, low_quantile=0.10, high_quantile=0.90)
        low_count = np.sum(regimes == "low")
        high_count = np.sum(regimes == "high")
        # ~10% should be low, ~10% high
        assert low_count == pytest.approx(10, abs=2)
        assert high_count == pytest.approx(10, abs=2)

    def test_output_length(self):
        vol = np.random.default_rng(0).uniform(0.1, 0.5, 200)
        regimes = volatility_regime(vol)
        assert len(regimes) == 200
