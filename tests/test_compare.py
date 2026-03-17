"""Tests for Portfolio Optimizer Comparison."""

from __future__ import annotations

import numpy as np
import pytest

from trading_master.quant.compare import ComparisonResult, compare_allocations


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def two_asset():
    mu = np.array([0.12, 0.06])
    cov = np.array([[0.04, 0.005], [0.005, 0.01]])
    return mu, cov


@pytest.fixture
def three_asset():
    mu = np.array([0.15, 0.10, 0.05])
    cov = np.array([
        [0.0625, 0.015, 0.003],
        [0.015, 0.0225, 0.005],
        [0.003, 0.005, 0.0064],
    ])
    return mu, cov


@pytest.fixture
def five_asset():
    mu = np.array([0.12, 0.10, 0.06, 0.04, 0.08])
    cov = np.array([
        [0.04,   0.025,  0.002,  0.001,  0.005],
        [0.025,  0.0625, 0.003,  0.001,  0.008],
        [0.002,  0.003,  0.0025, 0.001,  0.000],
        [0.001,  0.001,  0.001,  0.0016, 0.000],
        [0.005,  0.008,  0.000,  0.000,  0.0225],
    ])
    return mu, cov


# ── Basic properties ───────────────────────────────────────────────


class TestCompareBasicProperties:
    def test_returns_comparison_result(self, two_asset):
        mu, cov = two_asset
        result = compare_allocations(mu, cov, tickers=["A", "B"])
        assert isinstance(result, ComparisonResult)

    def test_correct_tickers(self, two_asset):
        mu, cov = two_asset
        result = compare_allocations(mu, cov, tickers=["SPY", "TLT"])
        assert result.tickers == ["SPY", "TLT"]

    def test_default_tickers(self, two_asset):
        mu, cov = two_asset
        result = compare_allocations(mu, cov)
        assert result.tickers == ["0", "1"]

    def test_all_weights_sum_to_one(self, three_asset):
        mu, cov = three_asset
        result = compare_allocations(mu, cov, tickers=["A", "B", "C"])
        assert result.markowitz_weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert result.hrp_weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert result.risk_parity_weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert result.consensus_weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_all_weights_non_negative(self, three_asset):
        mu, cov = three_asset
        result = compare_allocations(mu, cov)
        assert np.all(result.markowitz_weights >= -1e-8)
        assert np.all(result.hrp_weights >= 0)
        assert np.all(result.risk_parity_weights >= 0)
        assert np.all(result.consensus_weights >= 0)

    def test_correct_number_of_weights(self, five_asset):
        mu, cov = five_asset
        result = compare_allocations(mu, cov)
        assert len(result.markowitz_weights) == 5
        assert len(result.hrp_weights) == 5
        assert len(result.risk_parity_weights) == 5
        assert len(result.consensus_weights) == 5
        assert len(result.dispersion) == 5


# ── Consensus ──────────────────────────────────────────────────────


class TestConsensus:
    def test_consensus_is_average(self, two_asset):
        mu, cov = two_asset
        result = compare_allocations(mu, cov)
        avg = (result.markowitz_weights + result.hrp_weights + result.risk_parity_weights) / 3
        avg = avg / avg.sum()
        np.testing.assert_allclose(result.consensus_weights, avg, atol=1e-6)

    def test_consensus_between_extremes(self, three_asset):
        mu, cov = three_asset
        result = compare_allocations(mu, cov)
        for i in range(3):
            methods = [result.markowitz_weights[i], result.hrp_weights[i], result.risk_parity_weights[i]]
            assert result.consensus_weights[i] >= min(methods) - 1e-6
            assert result.consensus_weights[i] <= max(methods) + 1e-6


# ── Dispersion ─────────────────────────────────────────────────────


class TestDispersion:
    def test_dispersion_non_negative(self, three_asset):
        mu, cov = three_asset
        result = compare_allocations(mu, cov)
        assert np.all(result.dispersion >= 0)

    def test_max_dispersion_ticker_exists(self, five_asset):
        mu, cov = five_asset
        result = compare_allocations(mu, cov, tickers=["A", "B", "C", "D", "E"])
        assert result.max_dispersion_ticker in result.tickers

    def test_min_dispersion_ticker_exists(self, five_asset):
        mu, cov = five_asset
        result = compare_allocations(mu, cov, tickers=["A", "B", "C", "D", "E"])
        assert result.min_dispersion_ticker in result.tickers

    def test_max_dispersion_correct(self, three_asset):
        mu, cov = three_asset
        result = compare_allocations(mu, cov, tickers=["A", "B", "C"])
        max_idx = np.argmax(result.dispersion)
        assert result.max_dispersion_ticker == result.tickers[max_idx]


# ── Weight table property ──────────────────────────────────────────


class TestWeightTable:
    def test_weight_table_length(self, three_asset):
        mu, cov = three_asset
        result = compare_allocations(mu, cov, tickers=["A", "B", "C"])
        table = result.weight_table
        assert len(table) == 3

    def test_weight_table_keys(self, two_asset):
        mu, cov = two_asset
        result = compare_allocations(mu, cov, tickers=["X", "Y"])
        for row in result.weight_table:
            assert "ticker" in row
            assert "markowitz" in row
            assert "hrp" in row
            assert "risk_parity" in row
            assert "consensus" in row
            assert "dispersion" in row

    def test_weight_table_tickers(self, two_asset):
        mu, cov = two_asset
        result = compare_allocations(mu, cov, tickers=["SPY", "TLT"])
        tickers = [r["ticker"] for r in result.weight_table]
        assert tickers == ["SPY", "TLT"]


# ── Single asset edge case ─────────────────────────────────────────


class TestSingleAsset:
    def test_single_asset(self):
        mu = np.array([0.10])
        cov = np.array([[0.04]])
        result = compare_allocations(mu, cov, tickers=["ONLY"])
        assert result.markowitz_weights[0] == pytest.approx(1.0)
        assert result.hrp_weights[0] == pytest.approx(1.0)
        assert result.risk_parity_weights[0] == pytest.approx(1.0)
        assert result.dispersion[0] == pytest.approx(0.0)


# ── Validation ─────────────────────────────────────────────────────


class TestValidation:
    def test_ticker_mismatch_raises(self, two_asset):
        mu, cov = two_asset
        with pytest.raises(ValueError, match="Expected 2 tickers"):
            compare_allocations(mu, cov, tickers=["A", "B", "C"])


# ── Methods differ ─────────────────────────────────────────────────


class TestMethodsDiffer:
    def test_methods_produce_different_weights(self, five_asset):
        """The three methods should generally give different weights."""
        mu, cov = five_asset
        result = compare_allocations(mu, cov)
        # At least some dispersion should exist
        assert np.max(result.dispersion) > 0.01
