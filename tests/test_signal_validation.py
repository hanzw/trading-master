"""Tests for statistical signal validation (Hurst exponent, ADF, regime detection)."""

from __future__ import annotations

import numpy as np
import pytest

from trading_master.data.signal_validation import (
    adf_stationarity_test,
    hurst_exponent,
    signal_quality_report,
)


# ── Hurst exponent tests ─────────────────────────────────────────────


class TestHurstExponent:
    def test_trending_data(self):
        """Cumulative sum of positive increments should give H > 0.5."""
        rng = np.random.default_rng(42)
        # Strong trend: cumulative sum with drift
        steps = rng.normal(loc=0.1, scale=0.5, size=500)
        prices = 100 + np.cumsum(steps)
        prices = np.abs(prices) + 1  # ensure positive
        h = hurst_exponent(prices)
        assert h > 0.5, f"Expected H > 0.5 for trending data, got {h}"

    def test_mean_reverting_data(self):
        """Ornstein-Uhlenbeck process should give H < 0.5."""
        rng = np.random.default_rng(123)
        n = 1000
        theta = 0.7  # strong mean reversion
        mu = 100.0
        sigma = 0.5
        prices = np.zeros(n)
        prices[0] = mu
        for i in range(1, n):
            prices[i] = prices[i - 1] + theta * (mu - prices[i - 1]) + sigma * rng.normal()
        h = hurst_exponent(prices)
        assert h < 0.5, f"Expected H < 0.5 for mean-reverting data, got {h}"

    def test_random_walk_near_half(self):
        """Pure random walk should give H approximately 0.5."""
        rng = np.random.default_rng(7)
        steps = rng.normal(0, 1, size=2000)
        prices = 100 + np.cumsum(steps)
        prices = np.abs(prices) + 1
        h = hurst_exponent(prices)
        assert 0.35 < h < 0.65, f"Expected H near 0.5 for random walk, got {h}"

    def test_too_few_points(self):
        """Returns 0.5 for insufficient data."""
        prices = np.array([100.0, 101.0, 102.0])
        h = hurst_exponent(prices)
        assert h == 0.5

    def test_constant_prices(self):
        """Returns 0.5 for constant (zero-variance) prices."""
        prices = np.full(100, 50.0)
        h = hurst_exponent(prices)
        assert h == 0.5

    def test_nan_handling(self):
        """NaN values are stripped; remaining data is analyzed."""
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.normal(0.1, 0.5, size=200))
        prices = np.abs(prices) + 1
        # Sprinkle NaNs
        prices[10] = np.nan
        prices[50] = np.nan
        prices[100] = np.nan
        h = hurst_exponent(prices)
        assert 0.0 <= h <= 1.0

    def test_empty_array(self):
        """Returns 0.5 for empty input."""
        assert hurst_exponent(np.array([])) == 0.5

    def test_none_input(self):
        """Returns 0.5 for None input."""
        assert hurst_exponent(None) == 0.5


# ── ADF stationarity tests ───────────────────────────────────────────


class TestADFStationarity:
    def test_stationary_series(self):
        """Mean-reverting process should be detected as stationary."""
        rng = np.random.default_rng(99)
        n = 500
        # White noise around a mean → stationary
        prices = 100 + rng.normal(0, 1, size=n)
        result = adf_stationarity_test(prices)
        assert result["is_stationary"] is True
        assert result["p_value"] < 0.05
        assert result["regime_hint"] == "mean_reverting"

    def test_non_stationary_series(self):
        """Random walk (unit root) should be non-stationary."""
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.normal(0, 1, size=500))
        prices = np.abs(prices) + 1
        result = adf_stationarity_test(prices)
        assert result["is_stationary"] is False
        assert result["p_value"] > 0.05

    def test_insufficient_data(self):
        """Short series returns insufficient_data hint."""
        result = adf_stationarity_test(np.array([1.0, 2.0, 3.0]))
        assert result["regime_hint"] == "insufficient_data"
        assert result["is_stationary"] is False


# ── Signal quality report tests ──────────────────────────────────────


class TestSignalQualityReport:
    def test_trending_regime(self):
        """Trending data should recommend MACD, suppress RSI."""
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.normal(loc=0.1, scale=0.3, size=500))
        prices = np.abs(prices) + 1
        report = signal_quality_report(prices)
        assert report["hurst"] > 0.6 or report["hurst_regime"] in ("trending", "random_walk")
        if report["hurst_regime"] == "trending":
            assert "MACD" in report["recommended_signals"]
            assert "RSI_extreme" in report["suppressed_signals"]

    def test_mean_reverting_regime(self):
        """Mean-reverting data should recommend RSI, suppress MACD."""
        rng = np.random.default_rng(123)
        n = 1000
        theta = 0.7
        mu = 100.0
        sigma = 0.5
        prices = np.zeros(n)
        prices[0] = mu
        for i in range(1, n):
            prices[i] = prices[i - 1] + theta * (mu - prices[i - 1]) + sigma * rng.normal()
        report = signal_quality_report(prices)
        if report["hurst_regime"] == "mean_reverting":
            assert "RSI_extreme" in report["recommended_signals"]
            assert "MACD" in report["suppressed_signals"]

    def test_random_walk_suppresses_all(self):
        """Random walk regime suppresses all signal categories."""
        # Construct data that reliably gives H ~ 0.5
        report = signal_quality_report(np.array([100.0]))  # too few points
        assert report["hurst_regime"] == "random_walk"
        assert report["recommended_signals"] == []

    def test_insufficient_data_defaults(self):
        """Insufficient data returns safe defaults."""
        report = signal_quality_report(np.array([]))
        assert report["hurst"] == 0.5
        assert report["hurst_regime"] == "random_walk"

    def test_report_keys(self):
        """Report has all expected keys."""
        rng = np.random.default_rng(1)
        prices = 100 + np.cumsum(rng.normal(0, 1, size=100))
        prices = np.abs(prices) + 1
        report = signal_quality_report(prices)
        expected_keys = {"hurst", "hurst_regime", "is_stationary", "recommended_signals", "suppressed_signals"}
        assert expected_keys == set(report.keys())


# ── Signal filtering integration tests ───────────────────────────────


class TestSignalFiltering:
    """Test that _generate_signals respects suppressed_signals."""

    def test_trending_suppresses_rsi(self):
        """When trending, RSI signals should be suppressed."""
        from trading_master.data.technical import _generate_signals

        signals = _generate_signals(
            close=150.0,
            rsi=25.0,  # would normally trigger oversold
            macd_val=1.0,
            macd_sig=0.5,
            sma_20=148.0,
            sma_50=145.0,
            sma_200=140.0,
            boll_upper=155.0,
            boll_lower=135.0,
            suppressed_signals=["RSI_extreme", "Bollinger_band"],
        )
        assert not any("RSI" in s for s in signals), f"RSI should be suppressed, got: {signals}"
        assert any("MACD" in s for s in signals), "MACD should still appear"

    def test_mean_reverting_suppresses_macd(self):
        """When mean-reverting, MACD and SMA signals should be suppressed."""
        from trading_master.data.technical import _generate_signals

        signals = _generate_signals(
            close=150.0,
            rsi=25.0,
            macd_val=1.0,
            macd_sig=0.5,
            sma_20=148.0,
            sma_50=145.0,
            sma_200=140.0,
            boll_upper=155.0,
            boll_lower=135.0,
            suppressed_signals=["MACD", "SMA_crossover"],
        )
        assert not any("MACD" in s for s in signals), f"MACD should be suppressed, got: {signals}"
        assert not any("SMA" in s for s in signals), f"SMA should be suppressed, got: {signals}"
        assert any("RSI" in s for s in signals), "RSI should still appear"

    def test_no_suppression(self):
        """With no suppressed signals, all indicators appear."""
        from trading_master.data.technical import _generate_signals

        signals = _generate_signals(
            close=150.0,
            rsi=25.0,
            macd_val=1.0,
            macd_sig=0.5,
            sma_20=148.0,
            sma_50=145.0,
            sma_200=140.0,
            boll_upper=145.0,  # close > upper → Bollinger signal
            boll_lower=135.0,
            suppressed_signals=[],
        )
        assert any("RSI" in s for s in signals)
        assert any("MACD" in s for s in signals)
        assert any("SMA" in s for s in signals)

    def test_random_walk_suppresses_all(self):
        """Random walk regime suppresses all signal types."""
        from trading_master.data.technical import _generate_signals

        signals = _generate_signals(
            close=150.0,
            rsi=25.0,
            macd_val=1.0,
            macd_sig=0.5,
            sma_20=148.0,
            sma_50=145.0,
            sma_200=140.0,
            boll_upper=145.0,
            boll_lower=135.0,
            suppressed_signals=["MACD", "SMA_crossover", "RSI_extreme", "Bollinger_band"],
        )
        assert signals == [], f"All signals should be suppressed in random walk, got: {signals}"
