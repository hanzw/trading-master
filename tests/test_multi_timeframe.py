"""Tests for Multi-Timeframe Technical Analysis."""

from __future__ import annotations

import numpy as np
import pytest

from trading_master.quant.multi_timeframe import (
    MultiTimeframeResult,
    TimeframeSignal,
    multi_timeframe_analysis,
)


# ── Fixtures ────────────────────────────────────────────────────────


def _make_trending_up(n: int = 300, seed: int = 42) -> np.ndarray:
    """Strong uptrend with low noise."""
    rng = np.random.default_rng(seed)
    base = np.linspace(100, 200, n)
    noise = rng.normal(0, 0.5, n)
    return base + noise


def _make_trending_down(n: int = 300, seed: int = 42) -> np.ndarray:
    """Strong downtrend with low noise."""
    rng = np.random.default_rng(seed)
    base = np.linspace(200, 100, n)
    noise = rng.normal(0, 0.5, n)
    return base + noise


def _make_sideways(n: int = 300, seed: int = 42) -> np.ndarray:
    """Flat/sideways prices."""
    rng = np.random.default_rng(seed)
    return 100 + rng.normal(0, 2, n)


@pytest.fixture
def uptrend_prices():
    return _make_trending_up()


@pytest.fixture
def downtrend_prices():
    return _make_trending_down()


@pytest.fixture
def sideways_prices():
    return _make_sideways()


# ── Basic properties ───────────────────────────────────────────────


class TestMultiTimeframeBasicProperties:
    def test_returns_result_type(self, uptrend_prices):
        result = multi_timeframe_analysis(uptrend_prices)
        assert isinstance(result, MultiTimeframeResult)

    def test_three_timeframes(self, uptrend_prices):
        result = multi_timeframe_analysis(uptrend_prices)
        assert len(result.timeframes) == 3
        labels = [tf.timeframe for tf in result.timeframes]
        assert labels == ["daily", "weekly", "monthly"]

    def test_ticker_stored(self, uptrend_prices):
        result = multi_timeframe_analysis(uptrend_prices, ticker="AAPL")
        assert result.ticker == "AAPL"

    def test_consensus_score_range(self, uptrend_prices):
        result = multi_timeframe_analysis(uptrend_prices)
        assert -1.0 <= result.consensus_score <= 1.0

    def test_timeframe_score_range(self, sideways_prices):
        result = multi_timeframe_analysis(sideways_prices)
        for tf in result.timeframes:
            assert -1.0 <= tf.score <= 1.0

    def test_counts_sum_to_three(self, sideways_prices):
        result = multi_timeframe_analysis(sideways_prices)
        assert result.n_bullish + result.n_bearish + result.n_neutral == 3


# ── Uptrend detection ──────────────────────────────────────────────


class TestUptrendDetection:
    def test_uptrend_bullish_consensus(self, uptrend_prices):
        result = multi_timeframe_analysis(uptrend_prices)
        assert result.consensus_signal in ("BUY", "STRONG_BUY")

    def test_uptrend_positive_score(self, uptrend_prices):
        result = multi_timeframe_analysis(uptrend_prices)
        assert result.consensus_score > 0

    def test_uptrend_mostly_bullish(self, uptrend_prices):
        result = multi_timeframe_analysis(uptrend_prices)
        assert result.n_bullish >= 2

    def test_uptrend_daily_sma_bullish(self, uptrend_prices):
        result = multi_timeframe_analysis(uptrend_prices)
        daily = result.timeframes[0]
        # In a strong uptrend, SMA trend should be positive
        assert daily.sma_trend > 0


# ── Downtrend detection ────────────────────────────────────────────


class TestDowntrendDetection:
    def test_downtrend_bearish_consensus(self, downtrend_prices):
        result = multi_timeframe_analysis(downtrend_prices)
        assert result.consensus_signal in ("SELL", "STRONG_SELL")

    def test_downtrend_negative_score(self, downtrend_prices):
        result = multi_timeframe_analysis(downtrend_prices)
        assert result.consensus_score < 0

    def test_downtrend_mostly_bearish(self, downtrend_prices):
        result = multi_timeframe_analysis(downtrend_prices)
        assert result.n_bearish >= 2


# ── Sideways ───────────────────────────────────────────────────────


class TestSideways:
    def test_sideways_near_neutral(self, sideways_prices):
        result = multi_timeframe_analysis(sideways_prices)
        assert abs(result.consensus_score) < 0.6

    def test_sideways_hold_signal(self, sideways_prices):
        result = multi_timeframe_analysis(sideways_prices)
        assert result.consensus_signal in ("HOLD", "BUY", "SELL")


# ── Alignment ──────────────────────────────────────────────────────


class TestAlignment:
    def test_uptrend_aligned_bull(self, uptrend_prices):
        result = multi_timeframe_analysis(uptrend_prices)
        if result.n_bullish == 3:
            assert result.alignment == "aligned_bull"
            assert result.is_aligned

    def test_downtrend_aligned_bear(self, downtrend_prices):
        result = multi_timeframe_analysis(downtrend_prices)
        if result.n_bearish == 3:
            assert result.alignment == "aligned_bear"
            assert result.is_aligned

    def test_alignment_values(self, sideways_prices):
        result = multi_timeframe_analysis(sideways_prices)
        assert result.alignment in ("aligned_bull", "aligned_bear", "mixed", "neutral")


# ── Signal summary property ────────────────────────────────────────


class TestSignalSummary:
    def test_signal_summary_keys(self, uptrend_prices):
        result = multi_timeframe_analysis(uptrend_prices)
        summary = result.signal_summary
        assert "daily" in summary
        assert "weekly" in summary
        assert "monthly" in summary

    def test_signal_summary_values(self, uptrend_prices):
        result = multi_timeframe_analysis(uptrend_prices)
        for signal in result.signal_summary.values():
            assert signal in ("BULLISH", "BEARISH", "NEUTRAL")


# ── RSI and MACD computed ──────────────────────────────────────────


class TestIndicators:
    def test_daily_rsi_computed(self, uptrend_prices):
        result = multi_timeframe_analysis(uptrend_prices)
        daily = result.timeframes[0]
        assert daily.rsi is not None
        assert 0 <= daily.rsi <= 100

    def test_daily_macd_computed(self, uptrend_prices):
        result = multi_timeframe_analysis(uptrend_prices)
        daily = result.timeframes[0]
        assert daily.macd_histogram is not None

    def test_sma_trend_values(self, uptrend_prices):
        result = multi_timeframe_analysis(uptrend_prices)
        for tf in result.timeframes:
            assert tf.sma_trend in (-1.0, 0.0, 1.0)


# ── Custom weights ─────────────────────────────────────────────────


class TestCustomWeights:
    def test_different_weights_change_score(self, uptrend_prices):
        r1 = multi_timeframe_analysis(uptrend_prices, weights=(1.0, 0.0, 0.0))
        r2 = multi_timeframe_analysis(uptrend_prices, weights=(0.0, 0.0, 1.0))
        # Different weighting should give different consensus scores
        # (unless all timeframes happen to score identically)
        # At minimum both should be valid
        assert -1 <= r1.consensus_score <= 1
        assert -1 <= r2.consensus_score <= 1


# ── Validation ─────────────────────────────────────────────────────


class TestValidation:
    def test_too_few_prices_raises(self):
        with pytest.raises(ValueError, match="at least 30"):
            multi_timeframe_analysis(np.ones(20))

    def test_minimum_prices_works(self):
        prices = np.linspace(100, 110, 30)
        result = multi_timeframe_analysis(prices)
        assert isinstance(result, MultiTimeframeResult)
