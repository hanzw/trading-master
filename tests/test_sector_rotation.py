"""Tests for Sector Rotation analysis."""

from __future__ import annotations

import numpy as np
import pytest

from trading_master.quant.sector_rotation import (
    SectorRotationResult,
    SectorScore,
    analyze_sectors,
)


# ── Fixtures ────────────────────────────────────────────────────────


def _make_prices(base: float, daily_return: float, n: int = 252, seed: int = 42) -> np.ndarray:
    """Generate synthetic price series with given drift and low noise."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(daily_return, 0.005, n)  # low noise so drift dominates
    return base * np.cumprod(1 + returns)


@pytest.fixture
def sector_prices():
    """Diverse sector price data with clear leaders and laggards."""
    return {
        "XLK": _make_prices(100, 0.001, seed=1),     # strong uptrend
        "XLF": _make_prices(100, 0.0005, seed=2),     # moderate
        "XLV": _make_prices(100, 0.0003, seed=3),     # mild
        "XLE": _make_prices(100, -0.0005, seed=4),     # declining
        "XLU": _make_prices(100, -0.001, seed=5),      # weak
    }


@pytest.fixture
def benchmark_prices():
    """SPY-like benchmark."""
    return _make_prices(100, 0.0004, seed=10)


@pytest.fixture
def equal_sectors():
    """All sectors with identical performance."""
    return {
        "XLK": _make_prices(100, 0.0003, seed=42),
        "XLF": _make_prices(100, 0.0003, seed=42),
        "XLV": _make_prices(100, 0.0003, seed=42),
    }


# ── Basic properties ───────────────────────────────────────────────


class TestSectorRotationBasicProperties:
    def test_returns_result_type(self, sector_prices):
        result = analyze_sectors(sector_prices)
        assert isinstance(result, SectorRotationResult)

    def test_correct_n_sectors(self, sector_prices):
        result = analyze_sectors(sector_prices)
        assert result.n_sectors == 5
        assert len(result.sectors) == 5

    def test_sectors_have_scores(self, sector_prices):
        result = analyze_sectors(sector_prices)
        for s in result.sectors:
            assert isinstance(s, SectorScore)
            assert s.ticker in sector_prices

    def test_ranks_assigned(self, sector_prices):
        result = analyze_sectors(sector_prices)
        ranks = [s.rank for s in result.sectors]
        assert sorted(ranks) == [1, 2, 3, 4, 5]

    def test_sorted_by_composite_score(self, sector_prices):
        result = analyze_sectors(sector_prices)
        scores = [s.composite_score for s in result.sectors]
        assert scores == sorted(scores, reverse=True)

    def test_analysis_date_set(self, sector_prices):
        result = analyze_sectors(sector_prices)
        assert len(result.analysis_date) == 10  # YYYY-MM-DD


# ── Leaders and laggards ───────────────────────────────────────────


class TestLeadersLaggards:
    def test_leaders_returns_top_3(self, sector_prices):
        result = analyze_sectors(sector_prices)
        assert len(result.leaders) == 3
        assert result.leaders[0].rank == 1

    def test_laggards_returns_bottom_3(self, sector_prices):
        result = analyze_sectors(sector_prices)
        assert len(result.laggards) == 3
        assert result.laggards[-1].rank == 5

    def test_strong_sector_is_leader(self, sector_prices):
        result = analyze_sectors(sector_prices)
        leader_tickers = [s.ticker for s in result.leaders]
        # XLK has strongest drift (+0.1% daily) — should be a leader
        assert "XLK" in leader_tickers

    def test_weak_sector_is_laggard(self, sector_prices):
        result = analyze_sectors(sector_prices)
        laggard_tickers = [s.ticker for s in result.laggards]
        # XLU has weakest drift (-0.1% daily) — should be a laggard
        assert "XLU" in laggard_tickers


# ── Momentum ───────────────────────────────────────────────────────


class TestMomentum:
    def test_positive_drift_positive_momentum(self, sector_prices):
        result = analyze_sectors(sector_prices)
        xlk = next(s for s in result.sectors if s.ticker == "XLK")
        assert xlk.momentum_3m > 0

    def test_negative_drift_negative_momentum(self, sector_prices):
        result = analyze_sectors(sector_prices)
        xlu = next(s for s in result.sectors if s.ticker == "XLU")
        assert xlu.momentum_3m < 0

    def test_momentum_ordering(self, sector_prices):
        result = analyze_sectors(sector_prices)
        xlk = next(s for s in result.sectors if s.ticker == "XLK")
        xlu = next(s for s in result.sectors if s.ticker == "XLU")
        assert xlk.momentum_3m > xlu.momentum_3m


# ── Relative strength ──────────────────────────────────────────────


class TestRelativeStrength:
    def test_with_benchmark(self, sector_prices, benchmark_prices):
        result = analyze_sectors(sector_prices, benchmark_prices=benchmark_prices)
        assert result.benchmark_return_3m != 0.0
        # Leader should have positive relative strength
        assert result.leaders[0].relative_strength > result.laggards[-1].relative_strength

    def test_without_benchmark_zero(self, sector_prices):
        result = analyze_sectors(sector_prices)
        # Without benchmark, benchmark_3m = 0, so relative strength = momentum_3m
        assert result.benchmark_return_3m == 0.0


# ── Trend score ────────────────────────────────────────────────────


class TestTrendScore:
    def test_trend_score_range(self, sector_prices):
        result = analyze_sectors(sector_prices)
        for s in result.sectors:
            assert 0 <= s.trend_score <= 3

    def test_strong_uptrend_high_score(self, sector_prices):
        result = analyze_sectors(sector_prices)
        xlk = next(s for s in result.sectors if s.ticker == "XLK")
        # Strong uptrend should be above most SMAs
        assert xlk.trend_score >= 2


# ── Score dict property ────────────────────────────────────────────


class TestScoreDict:
    def test_score_dict_keys(self, sector_prices):
        result = analyze_sectors(sector_prices)
        sd = result.score_dict
        assert set(sd.keys()) == set(sector_prices.keys())

    def test_score_dict_values_match(self, sector_prices):
        result = analyze_sectors(sector_prices)
        sd = result.score_dict
        for s in result.sectors:
            assert sd[s.ticker] == s.composite_score


# ── Custom weights ─────────────────────────────────────────────────


class TestCustomWeights:
    def test_different_weights_different_scores(self, sector_prices):
        r1 = analyze_sectors(sector_prices, momentum_weight=1.0, trend_weight=0.0, relative_strength_weight=0.0)
        r2 = analyze_sectors(sector_prices, momentum_weight=0.0, trend_weight=1.0, relative_strength_weight=0.0)
        # Different weighting should produce different composite scores
        s1 = [s.composite_score for s in r1.sectors]
        s2 = [s.composite_score for s in r2.sectors]
        assert s1 != s2


# ── Validation ─────────────────────────────────────────────────────


class TestValidation:
    def test_empty_data_raises(self):
        with pytest.raises(ValueError, match="No price data"):
            analyze_sectors({})

    def test_short_series_skipped(self):
        result = analyze_sectors({"XLK": np.ones(10)})  # too short
        assert result.n_sectors == 0

    def test_mixed_lengths(self):
        """Short series are skipped, long ones are kept."""
        result = analyze_sectors({
            "XLK": _make_prices(100, 0.001, n=252, seed=1),
            "XLF": np.ones(10),  # too short
        })
        assert result.n_sectors == 1
        assert result.sectors[0].ticker == "XLK"


# ── Equal performance ──────────────────────────────────────────────


class TestEqualPerformance:
    def test_equal_sectors_similar_scores(self, equal_sectors):
        result = analyze_sectors(equal_sectors)
        scores = [s.composite_score for s in result.sectors]
        # All identical data → identical scores
        assert max(scores) - min(scores) < 1e-10
