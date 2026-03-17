"""Tests for Pairs Trading / Cointegration analysis."""

from __future__ import annotations

import numpy as np
import pytest

from trading_master.quant.cointegration import (
    CointegrationResult,
    cointegration_test,
    spread_zscore_series,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def cointegrated_pair():
    """Two price series that are cointegrated (shared random walk + mean-reverting spread)."""
    rng = np.random.default_rng(42)
    n = 500
    # Common stochastic trend
    trend = np.cumsum(rng.normal(0, 1, n))
    # Asset A = 2 * trend + mean-reverting noise
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.8 * noise[t - 1] + rng.normal(0, 0.5)
    prices_a = 100 + 2 * trend + noise
    prices_b = 50 + trend
    return prices_a, prices_b


@pytest.fixture
def non_cointegrated_pair():
    """Two independent random walks (not cointegrated)."""
    rng = np.random.default_rng(42)
    n = 500
    prices_a = 100 + np.cumsum(rng.normal(0.01, 1, n))
    prices_b = 100 + np.cumsum(rng.normal(-0.01, 1.2, n))
    return prices_a, prices_b


@pytest.fixture
def extreme_spread_pair():
    """Cointegrated pair with current spread at extreme z-score."""
    rng = np.random.default_rng(42)
    n = 300
    trend = np.cumsum(rng.normal(0, 1, n))
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.7 * noise[t - 1] + rng.normal(0, 0.3)
    # Push the last values to create extreme spread
    noise[-20:] += 5.0  # large positive deviation
    prices_a = 100 + 1.5 * trend + noise
    prices_b = 50 + trend
    return prices_a, prices_b


# ── Basic properties ───────────────────────────────────────────────


class TestCointegrationBasicProperties:
    def test_returns_result_type(self, cointegrated_pair):
        a, b = cointegrated_pair
        result = cointegration_test(a, b)
        assert isinstance(result, CointegrationResult)

    def test_hedge_ratio_reasonable(self, cointegrated_pair):
        a, b = cointegrated_pair
        result = cointegration_test(a, b)
        # True hedge ratio is ~2.0
        assert 1.0 < result.hedge_ratio < 3.0

    def test_n_observations(self, cointegrated_pair):
        a, b = cointegrated_pair
        result = cointegration_test(a, b)
        assert result.n_observations == 500

    def test_spread_std_positive(self, cointegrated_pair):
        a, b = cointegrated_pair
        result = cointegration_test(a, b)
        assert result.spread_std > 0

    def test_ticker_labels(self, cointegrated_pair):
        a, b = cointegrated_pair
        result = cointegration_test(a, b, ticker_a="AAPL", ticker_b="MSFT")
        assert result.ticker_a == "AAPL"
        assert result.ticker_b == "MSFT"


# ── Cointegration detection ────────────────────────────────────────


class TestCointegrationDetection:
    def test_detects_cointegrated_pair(self, cointegrated_pair):
        a, b = cointegrated_pair
        result = cointegration_test(a, b)
        assert result.is_cointegrated
        assert result.adf_pvalue < 0.05

    def test_rejects_non_cointegrated(self, non_cointegrated_pair):
        a, b = non_cointegrated_pair
        result = cointegration_test(a, b)
        assert not result.is_cointegrated
        assert result.adf_pvalue > 0.05

    def test_adf_statistic_negative_for_cointegrated(self, cointegrated_pair):
        a, b = cointegrated_pair
        result = cointegration_test(a, b)
        assert result.adf_statistic < -2.0


# ── Mean reversion ─────────────────────────────────────────────────


class TestMeanReversion:
    def test_cointegrated_has_finite_halflife(self, cointegrated_pair):
        a, b = cointegrated_pair
        result = cointegration_test(a, b)
        assert result.half_life > 0
        assert result.half_life < 100

    def test_cointegrated_is_mean_reverting(self, cointegrated_pair):
        a, b = cointegrated_pair
        result = cointegration_test(a, b)
        assert result.is_mean_reverting

    def test_non_cointegrated_slower_than_cointegrated(self, cointegrated_pair, non_cointegrated_pair):
        a_c, b_c = cointegrated_pair
        a_n, b_n = non_cointegrated_pair
        r_coint = cointegration_test(a_c, b_c)
        r_non = cointegration_test(a_n, b_n)
        # Non-cointegrated should have longer half-life
        assert r_non.half_life > r_coint.half_life

    def test_mean_reversion_speed_negative(self, cointegrated_pair):
        a, b = cointegrated_pair
        result = cointegration_test(a, b)
        assert result.mean_reversion_speed < 0


# ── Trading signals ────────────────────────────────────────────────


class TestTradingSignals:
    def test_signal_values(self, cointegrated_pair):
        a, b = cointegrated_pair
        result = cointegration_test(a, b)
        assert result.signal in ("BUY_A_SELL_B", "SELL_A_BUY_B", "NEUTRAL")

    def test_extreme_spread_generates_signal(self, extreme_spread_pair):
        a, b = extreme_spread_pair
        result = cointegration_test(a, b)
        # Large positive deviation → SELL_A_BUY_B
        if result.current_zscore > 2.0:
            assert result.signal == "SELL_A_BUY_B"
        elif result.current_zscore < -2.0:
            assert result.signal == "BUY_A_SELL_B"

    def test_signal_strength_is_abs_zscore(self, cointegrated_pair):
        a, b = cointegrated_pair
        result = cointegration_test(a, b)
        assert result.signal_strength == pytest.approx(abs(result.current_zscore), abs=1e-10)

    def test_custom_zscore_thresholds(self, extreme_spread_pair):
        a, b = extreme_spread_pair
        r_tight = cointegration_test(a, b, zscore_entry=1.0)
        r_loose = cointegration_test(a, b, zscore_entry=5.0)
        # Tighter threshold more likely to trigger signal
        if abs(r_tight.current_zscore) > 1.0:
            assert r_tight.signal != "NEUTRAL"
        if abs(r_loose.current_zscore) < 5.0:
            assert r_loose.signal == "NEUTRAL"


# ── Spread z-score series ──────────────────────────────────────────


class TestSpreadZscoreSeries:
    def test_returns_correct_shapes(self, cointegrated_pair):
        a, b = cointegrated_pair
        spread, zscore, hr = spread_zscore_series(a, b, lookback=60)
        assert len(spread) == len(a)
        assert len(zscore) == len(a)
        assert len(hr) == len(a)

    def test_first_lookback_values_nan(self, cointegrated_pair):
        a, b = cointegrated_pair
        spread, zscore, hr = spread_zscore_series(a, b, lookback=60)
        assert np.isnan(spread[0])
        assert np.isnan(zscore[0])

    def test_later_values_not_nan(self, cointegrated_pair):
        a, b = cointegrated_pair
        spread, zscore, hr = spread_zscore_series(a, b, lookback=60)
        # After lookback, most should be valid
        valid_spread = ~np.isnan(spread[100:])
        assert valid_spread.sum() > len(valid_spread) * 0.9

    def test_custom_lookback(self, cointegrated_pair):
        a, b = cointegrated_pair
        s30, _, _ = spread_zscore_series(a, b, lookback=30)
        s90, _, _ = spread_zscore_series(a, b, lookback=90)
        # 30-day lookback should have fewer NaN values
        assert np.isnan(s30).sum() < np.isnan(s90).sum()


# ── Validation ─────────────────────────────────────────────────────


class TestValidation:
    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            cointegration_test(np.ones(100), np.ones(50))

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError, match="at least 30"):
            cointegration_test(np.ones(20), np.ones(20))

    def test_spread_series_too_short_raises(self):
        with pytest.raises(ValueError):
            spread_zscore_series(np.ones(30), np.ones(30), lookback=60)


# ── Edge cases ─────────────────────────────────────────────────────


class TestEdgeCases:
    def test_identical_series(self):
        """Identical prices → perfect cointegration, zero spread."""
        prices = np.cumsum(np.random.default_rng(42).normal(0, 1, 200)) + 100
        result = cointegration_test(prices, prices, ticker_a="X", ticker_b="X")
        assert result.hedge_ratio == pytest.approx(1.0, abs=0.01)
        assert abs(result.current_zscore) < 0.5

    def test_scaled_series(self):
        """Perfectly scaled prices (A = 3*B) → cointegrated."""
        rng = np.random.default_rng(42)
        b = np.cumsum(rng.normal(0, 1, 300)) + 50
        a = 3 * b + rng.normal(0, 0.5, 300)  # small noise
        result = cointegration_test(a, b)
        assert result.hedge_ratio == pytest.approx(3.0, abs=0.2)
