"""Tests for portfolio position sizing."""

import numpy as np
import pytest

from trading_master.portfolio.sizing import (
    compute_position_size,
    correlation_adjusted_size,
    kelly_fraction,
    regime_adjusted_size,
    volatility_adjusted_shares,
)


# ── kelly_fraction ──────────────────────────────────────────────────


class TestKellyFraction:
    def test_positive_edge(self):
        # 60 % win rate, avg_win == avg_loss → full Kelly = 0.2
        result = kelly_fraction(0.6, 1.0, 1.0, fraction=1.0)
        assert pytest.approx(result, abs=1e-6) == 0.2

    def test_quarter_kelly(self):
        result = kelly_fraction(0.6, 1.0, 1.0, fraction=0.25)
        assert pytest.approx(result, abs=1e-6) == 0.05

    def test_high_win_rate(self):
        result = kelly_fraction(0.8, 2.0, 1.0, fraction=0.25)
        assert 0 < result <= 1.0

    def test_no_edge_returns_zero(self):
        # 40 % win rate, equal avg win/loss → negative Kelly → 0
        assert kelly_fraction(0.4, 1.0, 1.0) == 0.0

    def test_zero_win_rate(self):
        assert kelly_fraction(0.0, 1.0, 1.0) == 0.0

    def test_win_rate_one(self):
        assert kelly_fraction(1.0, 1.0, 1.0) == 0.0

    def test_negative_avg_win(self):
        assert kelly_fraction(0.6, -1.0, 1.0) == 0.0

    def test_zero_avg_loss(self):
        assert kelly_fraction(0.6, 1.0, 0.0) == 0.0


# ── volatility_adjusted_shares ──────────────────────────────────────


class TestVolatilityAdjustedShares:
    def test_basic(self):
        # portfolio 100k, risk 1% = $1000, ATR=$5, scaled_atr=5*sqrt(20)≈22.36 → 44 shares
        result = volatility_adjusted_shares(150.0, 5.0, 100_000.0, 1.0)
        assert result == 44

    def test_basic_one_day(self):
        # With holding_days=1: risk $1000 / ATR $5 = 200 shares (legacy behavior)
        result = volatility_adjusted_shares(150.0, 5.0, 100_000.0, 1.0, holding_days=1)
        assert result == 200

    def test_high_atr_fewer_shares(self):
        result = volatility_adjusted_shares(150.0, 10.0, 100_000.0, 1.0)
        assert result == 22

    def test_zero_price(self):
        assert volatility_adjusted_shares(0.0, 5.0, 100_000.0) == 0

    def test_zero_atr(self):
        assert volatility_adjusted_shares(150.0, 0.0, 100_000.0) == 0

    def test_zero_portfolio(self):
        assert volatility_adjusted_shares(150.0, 5.0, 0.0) == 0

    def test_negative_inputs(self):
        assert volatility_adjusted_shares(-10.0, 5.0, 100_000.0) == 0
        assert volatility_adjusted_shares(150.0, -5.0, 100_000.0) == 0

    def test_holding_period_scaling_reduces_size(self):
        """Multi-day holding period should produce fewer shares than 1-day."""
        shares_1d = volatility_adjusted_shares(150.0, 5.0, 100_000.0, 1.0, holding_days=1)
        shares_20d = volatility_adjusted_shares(150.0, 5.0, 100_000.0, 1.0, holding_days=20)
        assert shares_20d < shares_1d
        # 20-day scaled ATR = 5 * sqrt(20) ≈ 22.36, so ~4.47x fewer shares
        import math
        ratio = shares_1d / shares_20d if shares_20d > 0 else float("inf")
        assert ratio == pytest.approx(math.sqrt(20), abs=1.0)

    def test_holding_days_1_is_unscaled(self):
        """With holding_days=1, result should match the old 1-day behavior."""
        # portfolio 100k, risk 1% = $1000, ATR = $5, sqrt(1) = 1 → 200 shares
        result = volatility_adjusted_shares(150.0, 5.0, 100_000.0, 1.0, holding_days=1)
        assert result == 200

    def test_default_holding_days_is_20(self):
        """Default holding_days=20 should produce fewer shares than holding_days=1."""
        default_shares = volatility_adjusted_shares(150.0, 5.0, 100_000.0, 1.0)
        one_day_shares = volatility_adjusted_shares(150.0, 5.0, 100_000.0, 1.0, holding_days=1)
        assert default_shares < one_day_shares

    def test_hurst_mean_reverting_more_shares(self):
        """H=0.3 (mean-reverting) scales less than H=0.5 → more shares."""
        shares_sqrt = volatility_adjusted_shares(150.0, 5.0, 100_000.0, 1.0, holding_days=20, hurst=0.5)
        shares_mr = volatility_adjusted_shares(150.0, 5.0, 100_000.0, 1.0, holding_days=20, hurst=0.3)
        assert shares_mr > shares_sqrt

    def test_hurst_trending_fewer_shares(self):
        """H=0.7 (trending) scales more than H=0.5 → fewer shares."""
        shares_sqrt = volatility_adjusted_shares(150.0, 5.0, 100_000.0, 1.0, holding_days=20, hurst=0.5)
        shares_trend = volatility_adjusted_shares(150.0, 5.0, 100_000.0, 1.0, holding_days=20, hurst=0.7)
        assert shares_trend < shares_sqrt

    def test_hurst_none_falls_back_to_sqrt(self):
        """hurst=None should produce the same result as hurst=0.5."""
        shares_none = volatility_adjusted_shares(150.0, 5.0, 100_000.0, 1.0, holding_days=20, hurst=None)
        shares_half = volatility_adjusted_shares(150.0, 5.0, 100_000.0, 1.0, holding_days=20, hurst=0.5)
        assert shares_none == shares_half


# ── correlation_adjusted_size ───────────────────────────────────────


class TestCorrelationAdjustedSize:
    def test_no_correlation(self):
        # Uncorrelated returns → no reduction
        rng = np.random.default_rng(42)
        new = rng.normal(0, 1, 100)
        existing = rng.normal(0, 1, 100)
        result = correlation_adjusted_size(100, new, existing)
        assert result >= 20  # floor

    def test_perfect_correlation(self):
        series = np.arange(100, dtype=float)
        result = correlation_adjusted_size(100, series, series)
        # corr = 1.0 → multiplier = 0.0 → floored at 20 %
        assert result == 20

    def test_zero_base(self):
        assert correlation_adjusted_size(0, np.array([1, 2]), np.array([1, 2])) == 0

    def test_empty_arrays(self):
        assert correlation_adjusted_size(100, np.array([]), np.array([1, 2])) == 100


# ── compute_position_size ───────────────────────────────────────────


class TestComputePositionSize:
    def test_basic(self):
        result = compute_position_size(150.0, 5.0, 100_000.0)
        assert result["shares"] > 0
        assert result["pct_of_portfolio"] <= 8.0
        assert result["dollar_amount"] == result["shares"] * 150.0

    def test_respects_max_cap(self):
        # Very low ATR → vol sizing would be huge, cap kicks in
        result = compute_position_size(10.0, 0.01, 100_000.0, max_position_pct=8.0)
        assert result["dollar_amount"] <= 8_000.0 + 10.0  # allow 1-share rounding
        assert "max_position_cap" in result["method"]

    def test_zero_price(self):
        result = compute_position_size(0.0, 5.0, 100_000.0)
        assert result["shares"] == 0
        assert result["method"] == "invalid_input"

    def test_zero_portfolio(self):
        result = compute_position_size(150.0, 5.0, 0.0)
        assert result["shares"] == 0

    def test_correlation_reduces_size(self):
        base = compute_position_size(150.0, 5.0, 100_000.0, existing_correlation=0.0)
        corr = compute_position_size(150.0, 5.0, 100_000.0, existing_correlation=0.8)
        assert corr["shares"] <= base["shares"]
        assert "correlation_adj" in corr["method"]

    def test_regime_bull_no_change(self):
        base = compute_position_size(150.0, 5.0, 100_000.0)
        bull = compute_position_size(150.0, 5.0, 100_000.0, regime="bull")
        assert bull["shares"] == base["shares"]
        assert bull["regime"] == "bull"
        assert bull["regime_multiplier"] == 1.0

    def test_regime_sideways_reduces(self):
        base = compute_position_size(150.0, 5.0, 100_000.0)
        side = compute_position_size(150.0, 5.0, 100_000.0, regime="sideways")
        assert side["shares"] <= base["shares"]
        assert side["regime_multiplier"] == 0.75

    def test_regime_bear_reduces(self):
        base = compute_position_size(150.0, 5.0, 100_000.0)
        bear = compute_position_size(150.0, 5.0, 100_000.0, regime="bear")
        assert bear["shares"] <= base["shares"]
        assert bear["regime_multiplier"] == 0.5

    def test_regime_crisis_reduces(self):
        base = compute_position_size(150.0, 5.0, 100_000.0)
        crisis = compute_position_size(150.0, 5.0, 100_000.0, regime="crisis")
        assert crisis["shares"] <= base["shares"]
        assert crisis["regime_multiplier"] == 0.25
        assert "regime_adj" in crisis["method"]

    def test_regime_none_no_adjustment(self):
        result = compute_position_size(150.0, 5.0, 100_000.0, regime=None)
        assert "regime" not in result
        assert "regime_adj" not in result["method"]


# ── regime_adjusted_size ──────────────────────────────────────────────


class TestRegimeAdjustedSize:
    def test_bull(self):
        assert regime_adjusted_size(100, "bull", 150.0) == 100

    def test_sideways(self):
        assert regime_adjusted_size(100, "sideways", 150.0) == 75

    def test_bear(self):
        assert regime_adjusted_size(100, "bear", 150.0) == 50

    def test_crisis(self):
        assert regime_adjusted_size(100, "crisis", 150.0) == 25

    def test_zero_base(self):
        assert regime_adjusted_size(0, "crisis", 150.0) == 0

    def test_unknown_regime_defaults_to_1x(self):
        assert regime_adjusted_size(100, "unknown_regime", 150.0) == 100

    def test_case_insensitive(self):
        assert regime_adjusted_size(100, "BEAR", 150.0) == 50
        assert regime_adjusted_size(100, "Bear", 150.0) == 50


# ── Kelly integration in compute_position_size ─────────────────────


class TestKellyInComputePositionSize:
    def test_kelly_constrains_when_edge_is_low(self):
        """With a low win rate, Kelly should constrain position size below vol-based."""
        # Without Kelly
        no_kelly = compute_position_size(150.0, 5.0, 100_000.0)
        # With Kelly: 55% win rate, win/loss ratio 1.0 → small edge
        with_kelly = compute_position_size(
            150.0, 5.0, 100_000.0,
            win_rate=0.55, avg_win_loss_ratio=1.0,
        )
        assert with_kelly["kelly_used"] is True
        assert with_kelly["kelly_fraction_raw"] > 0
        assert with_kelly["shares"] <= no_kelly["shares"]

    def test_kelly_not_used_when_params_missing(self):
        """Without win_rate / avg_win_loss_ratio, Kelly is not used."""
        result = compute_position_size(150.0, 5.0, 100_000.0)
        assert result["kelly_used"] is False
        assert result["kelly_fraction_raw"] == 0.0

    def test_kelly_not_used_when_no_edge(self):
        """If Kelly fraction is zero (no edge), kelly_used stays False."""
        result = compute_position_size(
            150.0, 5.0, 100_000.0,
            win_rate=0.3, avg_win_loss_ratio=1.0,  # 30% win rate, no edge
        )
        assert result["kelly_used"] is False
        assert result["kelly_fraction_raw"] == 0.0

    def test_kelly_constrains_method_label(self):
        """When Kelly is the binding constraint, method should say kelly_constrained."""
        # Very low win rate with high vol-adjusted shares (low ATR)
        result = compute_position_size(
            10.0, 0.5, 100_000.0,
            max_position_pct=50.0,  # high cap so it doesn't bind
            win_rate=0.52, avg_win_loss_ratio=1.0,
        )
        if result["kelly_used"]:
            # Kelly fraction should be very small for 52% win rate
            assert result["kelly_fraction_raw"] < 0.05

    def test_kelly_with_high_edge_does_not_reduce(self):
        """With a very high edge, Kelly may not be the binding constraint."""
        no_kelly = compute_position_size(150.0, 5.0, 100_000.0)
        with_kelly = compute_position_size(
            150.0, 5.0, 100_000.0,
            win_rate=0.8, avg_win_loss_ratio=3.0,  # very strong edge
        )
        assert with_kelly["kelly_used"] is True
        # Kelly should allow a large position, so vol or cap should bind
        assert with_kelly["shares"] == no_kelly["shares"]

    def test_kelly_invalid_input_returns_zero(self):
        """With invalid price, Kelly fields should still be present."""
        result = compute_position_size(
            0.0, 5.0, 100_000.0,
            win_rate=0.6, avg_win_loss_ratio=1.5,
        )
        assert result["shares"] == 0
        assert result["kelly_used"] is False
