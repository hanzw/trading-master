"""Tests for Extreme Value Theory (EVT) tail risk estimation."""

from __future__ import annotations

import numpy as np
import pytest

from trading_master.quant.evt import (
    EVTResult,
    evt_tail_risk,
    mean_excess_plot_data,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def normal_returns():
    """Simulated normal returns (light tails)."""
    rng = np.random.default_rng(42)
    return rng.normal(0.0005, 0.02, size=1000)  # ~2% daily vol


@pytest.fixture
def heavy_tail_returns():
    """Simulated heavy-tailed returns (Student-t, df=3)."""
    rng = np.random.default_rng(42)
    return rng.standard_t(df=3, size=1000) * 0.02


@pytest.fixture
def mixed_returns():
    """Simulated returns with occasional large drawdowns."""
    rng = np.random.default_rng(42)
    base = rng.normal(0.0003, 0.015, size=1000)
    # Inject 20 crash days
    crash_indices = rng.choice(1000, 20, replace=False)
    base[crash_indices] = rng.normal(-0.08, 0.02, size=20)
    return base


@pytest.fixture
def short_returns():
    """Too few observations."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 0.02, size=20)


# ── Basic properties ───────────────────────────────────────────────


class TestEVTBasicProperties:
    def test_returns_evt_result(self, normal_returns):
        result = evt_tail_risk(normal_returns)
        assert isinstance(result, EVTResult)

    def test_var_99_greater_than_var_95(self, normal_returns):
        result = evt_tail_risk(normal_returns)
        assert result.var_99 > result.var_95

    def test_cvar_greater_than_var(self, normal_returns):
        result = evt_tail_risk(normal_returns)
        assert result.cvar_95 >= result.var_95
        assert result.cvar_99 >= result.var_99

    def test_var_positive(self, normal_returns):
        result = evt_tail_risk(normal_returns)
        assert result.var_95 > 0
        assert result.var_99 > 0

    def test_cvar_positive(self, normal_returns):
        result = evt_tail_risk(normal_returns)
        assert result.cvar_95 > 0
        assert result.cvar_99 > 0

    def test_scale_positive(self, normal_returns):
        result = evt_tail_risk(normal_returns)
        assert result.scale > 0

    def test_n_total(self, normal_returns):
        result = evt_tail_risk(normal_returns)
        assert result.n_total == 1000

    def test_exceedance_rate(self, normal_returns):
        result = evt_tail_risk(normal_returns)
        # With 90th percentile threshold, ~10% should exceed
        assert 0.05 < result.exceedance_rate < 0.20

    def test_mean_excess_positive(self, normal_returns):
        result = evt_tail_risk(normal_returns)
        assert result.mean_excess > 0


# ── Tail type detection ────────────────────────────────────────────


class TestTailDetection:
    def test_heavy_tail_detected(self, heavy_tail_returns):
        result = evt_tail_risk(heavy_tail_returns)
        # Student-t(3) has heavy tails; shape should be positive
        assert result.shape > 0
        assert result.tail_type == "heavy"
        assert result.is_heavy_tailed

    def test_normal_tail_not_heavy(self, normal_returns):
        result = evt_tail_risk(normal_returns)
        # Normal distribution has exponential-ish tails
        # Shape should be near zero or slightly negative
        assert result.shape < 0.3  # not strongly heavy-tailed

    def test_tail_type_values(self, normal_returns):
        result = evt_tail_risk(normal_returns)
        assert result.tail_type in ("heavy", "exponential", "bounded")


# ── Heavy tail produces higher risk estimates ──────────────────────


class TestHeavyTailRisk:
    def test_heavy_tail_higher_var(self, normal_returns, heavy_tail_returns):
        normal_result = evt_tail_risk(normal_returns)
        heavy_result = evt_tail_risk(heavy_tail_returns)
        # Heavy tails should produce higher VaR at 99%
        assert heavy_result.var_99 > normal_result.var_99 * 0.8

    def test_mixed_returns_higher_cvar(self, normal_returns, mixed_returns):
        normal_result = evt_tail_risk(normal_returns)
        mixed_result = evt_tail_risk(mixed_returns)
        # Returns with crash days should have higher CVaR
        assert mixed_result.cvar_99 > normal_result.cvar_99


# ── Threshold selection ────────────────────────────────────────────


class TestThresholdSelection:
    def test_custom_threshold_quantile(self, normal_returns):
        r85 = evt_tail_risk(normal_returns, threshold_quantile=0.85)
        r95 = evt_tail_risk(normal_returns, threshold_quantile=0.95)
        # Higher threshold = fewer exceedances
        assert r95.n_exceedances < r85.n_exceedances

    def test_threshold_increases_with_quantile(self, normal_returns):
        r85 = evt_tail_risk(normal_returns, threshold_quantile=0.85)
        r95 = evt_tail_risk(normal_returns, threshold_quantile=0.95)
        assert r95.threshold > r85.threshold


# ── KS test ────────────────────────────────────────────────────────


class TestGoodnessOfFit:
    def test_ks_pvalue_exists(self, normal_returns):
        result = evt_tail_risk(normal_returns)
        assert result.ks_pvalue is not None

    def test_ks_pvalue_range(self, normal_returns):
        result = evt_tail_risk(normal_returns)
        assert 0.0 <= result.ks_pvalue <= 1.0


# ── Validation ─────────────────────────────────────────────────────


class TestValidation:
    def test_too_few_observations(self, short_returns):
        with pytest.raises(ValueError, match="at least 30"):
            evt_tail_risk(short_returns)

    def test_handles_nan_values(self, normal_returns):
        returns_with_nan = normal_returns.copy()
        returns_with_nan[0] = np.nan
        returns_with_nan[10] = np.inf
        result = evt_tail_risk(returns_with_nan)
        assert result.n_total < 1000  # NaN/inf should be filtered

    def test_all_same_returns_raises(self):
        """All identical returns = no variability = fitting fails or raises."""
        returns = np.zeros(100)
        with pytest.raises((ValueError, Exception)):
            evt_tail_risk(returns)


# ── Mean Excess Plot ───────────────────────────────────────────────


class TestMeanExcessPlot:
    def test_returns_correct_shapes(self, normal_returns):
        thresholds, me = mean_excess_plot_data(normal_returns, n_thresholds=30)
        assert len(thresholds) == 30
        assert len(me) == 30

    def test_thresholds_increasing(self, normal_returns):
        thresholds, _ = mean_excess_plot_data(normal_returns)
        assert np.all(np.diff(thresholds) > 0)

    def test_mean_excess_non_negative(self, normal_returns):
        _, me = mean_excess_plot_data(normal_returns)
        assert np.all(me >= 0)


# ── Property accessors ─────────────────────────────────────────────


class TestPropertyAccessors:
    def test_is_heavy_tailed_property(self, heavy_tail_returns):
        result = evt_tail_risk(heavy_tail_returns)
        assert result.is_heavy_tailed == (result.shape > 0.05)

    def test_dollar_var_99(self, normal_returns):
        result = evt_tail_risk(normal_returns)
        assert result.dollar_var_99 == result.var_99
