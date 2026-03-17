"""Tests for Hierarchical Risk Parity (HRP) allocation."""

from __future__ import annotations

import numpy as np
import pytest

from trading_master.quant.hrp import HRPResult, hrp_allocation


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def two_asset_cov():
    """Two assets: one high-vol, one low-vol, low correlation."""
    cov = np.array([
        [0.04, 0.005],     # 20% vol
        [0.005, 0.01],     # 10% vol
    ])
    return cov


@pytest.fixture
def three_asset_cov():
    """Three assets with different risk profiles."""
    cov = np.array([
        [0.0625, 0.015, 0.003],    # 25% vol
        [0.015, 0.0225, 0.005],     # 15% vol
        [0.003, 0.005, 0.0064],     # 8% vol
    ])
    return cov


@pytest.fixture
def five_asset_cov():
    """Five assets with a realistic block-correlation structure."""
    # Two correlated equity-like + two bond-like + one commodity
    cov = np.array([
        [0.04,   0.025,  0.002,  0.001,  0.005],
        [0.025,  0.0625, 0.003,  0.001,  0.008],
        [0.002,  0.003,  0.0025, 0.001,  0.000],
        [0.001,  0.001,  0.001,  0.0016, 0.000],
        [0.005,  0.008,  0.000,  0.000,  0.0225],
    ])
    return cov


# ── Basic properties ───────────────────────────────────────────────


class TestHRPBasicProperties:
    def test_weights_sum_to_one(self, two_asset_cov):
        result = hrp_allocation(two_asset_cov)
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-8)

    def test_all_weights_non_negative(self, three_asset_cov):
        result = hrp_allocation(three_asset_cov)
        assert np.all(result.weights >= 0)

    def test_returns_hrp_result(self, two_asset_cov):
        result = hrp_allocation(two_asset_cov)
        assert isinstance(result, HRPResult)

    def test_correct_number_of_weights(self, three_asset_cov):
        result = hrp_allocation(three_asset_cov, tickers=["A", "B", "C"])
        assert len(result.weights) == 3
        assert len(result.tickers) == 3

    def test_weight_dict(self, two_asset_cov):
        result = hrp_allocation(two_asset_cov, tickers=["SPY", "TLT"])
        wd = result.weight_dict
        assert "SPY" in wd
        assert "TLT" in wd
        assert sum(wd.values()) == pytest.approx(1.0, abs=1e-8)


# ── Risk-based allocation logic ────────────────────────────────────


class TestHRPRiskAllocation:
    def test_low_vol_asset_gets_more_weight(self, two_asset_cov):
        result = hrp_allocation(two_asset_cov, tickers=["HIGH_VOL", "LOW_VOL"])
        # Asset 1 (10% vol) should get more weight than asset 0 (20% vol)
        assert result.weights[1] > result.weights[0]

    def test_diversification_across_clusters(self, five_asset_cov):
        tickers = ["EQ1", "EQ2", "BD1", "BD2", "COMM"]
        result = hrp_allocation(five_asset_cov, tickers=tickers)
        # No single asset should dominate
        assert np.max(result.weights) < 0.60
        # Low-vol bonds should collectively get meaningful weight
        bond_weight = result.weights[2] + result.weights[3]
        assert bond_weight > 0.15

    def test_three_asset_lowest_vol_gets_most(self, three_asset_cov):
        result = hrp_allocation(three_asset_cov, tickers=["HIGH", "MED", "LOW"])
        # Asset 2 (8% vol) should get the most weight
        assert result.weights[2] > result.weights[0]


# ── Single asset edge case ─────────────────────────────────────────


class TestHRPSingleAsset:
    def test_single_asset_weight_is_one(self):
        cov = np.array([[0.04]])
        result = hrp_allocation(cov, tickers=["ONLY"])
        assert result.weights[0] == pytest.approx(1.0)
        assert result.n_clusters == 1

    def test_single_asset_cluster_order(self):
        cov = np.array([[0.04]])
        result = hrp_allocation(cov)
        assert result.cluster_order == [0]


# ── Linkage methods ────────────────────────────────────────────────


class TestHRPLinkageMethods:
    @pytest.mark.parametrize("method", ["single", "complete", "average", "ward"])
    def test_all_linkage_methods_valid(self, three_asset_cov, method):
        result = hrp_allocation(three_asset_cov, linkage_method=method)
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-8)
        assert np.all(result.weights >= 0)

    def test_different_linkage_different_weights(self, five_asset_cov):
        r_single = hrp_allocation(five_asset_cov, linkage_method="single")
        r_complete = hrp_allocation(five_asset_cov, linkage_method="complete")
        # Weights should differ (at least slightly) between methods
        assert not np.allclose(r_single.weights, r_complete.weights, atol=1e-6)


# ── Custom correlation matrix ──────────────────────────────────────


class TestHRPCustomCorrelation:
    def test_explicit_corr_matrix(self, two_asset_cov):
        corr = np.array([
            [1.0, 0.25],
            [0.25, 1.0],
        ])
        result = hrp_allocation(two_asset_cov, corr_matrix=corr)
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-8)

    def test_identity_corr_gives_inverse_var(self):
        """With zero correlation, HRP should approximate inverse-variance weights."""
        cov = np.array([
            [0.04, 0.0],
            [0.0, 0.01],
        ])
        corr = np.eye(2)
        result = hrp_allocation(cov, corr_matrix=corr)
        # Inverse variance: 1/0.04=25, 1/0.01=100 → 20%, 80%
        assert result.weights[1] > result.weights[0]


# ── Validation ─────────────────────────────────────────────────────


class TestHRPValidation:
    def test_ticker_count_mismatch_raises(self, two_asset_cov):
        with pytest.raises(ValueError, match="Expected 2 tickers"):
            hrp_allocation(two_asset_cov, tickers=["A", "B", "C"])

    def test_default_tickers_are_indices(self, two_asset_cov):
        result = hrp_allocation(two_asset_cov)
        assert result.tickers == ["0", "1"]


# ── Cluster info ───────────────────────────────────────────────────


class TestHRPClusterInfo:
    def test_cluster_order_covers_all_assets(self, three_asset_cov):
        result = hrp_allocation(three_asset_cov)
        assert sorted(result.cluster_order) == [0, 1, 2]

    def test_n_clusters_positive(self, five_asset_cov):
        result = hrp_allocation(five_asset_cov)
        assert result.n_clusters >= 1
        assert result.n_clusters <= 5
