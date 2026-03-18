"""Tests for the Fama-French 5-Factor model."""

from __future__ import annotations

import numpy as np
import pytest

from trading_master.quant.fama_french import (
    FF5Result,
    FACTOR_NAMES,
    attribute_returns,
    fetch_french_factors,
    ff5_decompose,
    ff5_decompose_portfolio,
    generate_synthetic_factors,
    ols_regression,
)


# ── OLS regression ──────────────────────────────────────────────────


class TestOLSRegression:
    def test_perfect_fit(self):
        """y = 2 + 3*x should give alpha=2, beta=3, R²=1."""
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = 2 + 3 * x
        X = x.reshape(-1, 1)
        betas, r_sq, t_stats = ols_regression(y, X)

        assert betas[0] == pytest.approx(2.0, abs=1e-10)
        assert betas[1] == pytest.approx(3.0, abs=1e-10)
        assert r_sq == pytest.approx(1.0, abs=1e-10)

    def test_multivariate(self):
        """y = 1 + 2*x1 + 3*x2."""
        rng = np.random.default_rng(42)
        n = 100
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        y = 1.0 + 2.0 * x1 + 3.0 * x2
        X = np.column_stack([x1, x2])

        betas, r_sq, _ = ols_regression(y, X)
        assert betas[0] == pytest.approx(1.0, abs=1e-10)
        assert betas[1] == pytest.approx(2.0, abs=1e-10)
        assert betas[2] == pytest.approx(3.0, abs=1e-10)
        assert r_sq == pytest.approx(1.0, abs=1e-8)

    def test_noisy_data(self):
        """With noise, R² should be between 0 and 1."""
        rng = np.random.default_rng(99)
        n = 200
        x = rng.normal(0, 1, n)
        y = 1.0 + 2.0 * x + rng.normal(0, 0.5, n)
        X = x.reshape(-1, 1)

        betas, r_sq, t_stats = ols_regression(y, X)
        assert 0 < r_sq < 1
        assert betas[0] == pytest.approx(1.0, abs=0.2)
        assert betas[1] == pytest.approx(2.0, abs=0.2)
        # With strong signal, t-stat for beta should be significant
        assert abs(t_stats[1]) > 2.0

    def test_insufficient_observations(self):
        """Should raise ValueError if n <= k."""
        X = np.array([[1, 2, 3]])  # 1 observation, 3 features
        y = np.array([1.0])
        with pytest.raises(ValueError, match="Not enough observations"):
            ols_regression(y, X)

    def test_no_intercept(self):
        """y = 3*x (no intercept)."""
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = 3 * x
        X = x.reshape(-1, 1)
        betas, r_sq, _ = ols_regression(y, X, add_intercept=False)
        assert len(betas) == 1
        assert betas[0] == pytest.approx(3.0, abs=1e-10)


# ── FF5 single-asset decomposition ─────────────────────────────────


class TestFF5Decompose:
    @pytest.fixture
    def synthetic_data(self):
        """Generate a synthetic asset with known factor exposures."""
        rng = np.random.default_rng(42)
        n = 500
        factors = rng.normal(0, 0.01, (n, 5))

        # True betas: Mkt=1.2, SMB=0.3, HML=-0.2, RMW=0.4, CMA=0.1
        true_betas = np.array([1.2, 0.3, -0.2, 0.4, 0.1])
        true_alpha = 0.0001  # small daily alpha

        excess_returns = true_alpha + factors @ true_betas + rng.normal(0, 0.005, n)
        return excess_returns, factors, true_betas, true_alpha

    def test_recovers_betas(self, synthetic_data):
        excess_returns, factors, true_betas, _ = synthetic_data
        result = ff5_decompose(excess_returns, factors, ticker="TEST")

        assert result.ticker == "TEST"
        assert result.n_obs == 500

        for i, name in enumerate(FACTOR_NAMES):
            assert result.betas[name] == pytest.approx(
                true_betas[i], abs=0.15
            ), f"Beta for {name} too far from true value"

    def test_r_squared_reasonable(self, synthetic_data):
        excess_returns, factors, _, _ = synthetic_data
        result = ff5_decompose(excess_returns, factors)
        # With known factor structure + noise, R² should be moderate to high
        assert 0.3 < result.r_squared < 1.0

    def test_t_stats_present(self, synthetic_data):
        excess_returns, factors, _, _ = synthetic_data
        result = ff5_decompose(excess_returns, factors)

        assert "alpha" in result.t_stats
        for name in FACTOR_NAMES:
            assert name in result.t_stats

    def test_significant_factors(self, synthetic_data):
        excess_returns, factors, _, _ = synthetic_data
        result = ff5_decompose(excess_returns, factors)

        # Mkt-RF beta is large (1.2), so it should be significant
        assert "Mkt-RF" in result.significant_factors

    def test_alpha_annualized(self, synthetic_data):
        excess_returns, factors, _, _true_alpha = synthetic_data
        result = ff5_decompose(excess_returns, factors)
        # Annualized alpha = daily alpha * 252; with estimation noise
        # the regression alpha absorbs some noise, so just check it's
        # a reasonable magnitude (< 50% annualized)
        assert abs(result.alpha_annualized) < 0.50

    def test_residual_std_positive(self, synthetic_data):
        excess_returns, factors, _, _ = synthetic_data
        result = ff5_decompose(excess_returns, factors)
        assert result.residual_std > 0

    def test_wrong_factor_dimensions(self):
        with pytest.raises(ValueError, match="Expected 5 factors"):
            ff5_decompose(
                np.ones(100), np.ones((100, 3)), ticker="BAD"
            )

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="Mismatched lengths"):
            ff5_decompose(
                np.ones(100), np.ones((50, 5))
            )

    def test_1d_factors_rejected(self):
        with pytest.raises(ValueError, match="2-D"):
            ff5_decompose(np.ones(10), np.ones(10))


# ── Portfolio decomposition ─────────────────────────────────────────


class TestFF5DecomposePortfolio:
    def test_multiple_assets(self):
        rng = np.random.default_rng(7)
        n = 200
        factors = rng.normal(0, 0.01, (n, 5))

        # 3 assets with different exposures
        betas_matrix = np.array([
            [1.0, 0.5, 0.0, 0.3, 0.1],
            [0.8, -0.3, 0.4, 0.0, 0.2],
            [1.5, 0.0, -0.5, 0.6, -0.1],
        ])
        excess_returns = (factors @ betas_matrix.T) + rng.normal(0, 0.003, (n, 3))

        results = ff5_decompose_portfolio(
            excess_returns, factors, tickers=["A", "B", "C"]
        )

        assert len(results) == 3
        assert results[0].ticker == "A"
        assert results[1].ticker == "B"
        assert results[2].ticker == "C"

        # Each should have reasonable R²
        for r in results:
            assert r.r_squared > 0.2

    def test_mismatched_tickers(self):
        factors = np.ones((50, 5))
        returns = np.ones((50, 2))
        with pytest.raises(ValueError, match="Expected 2 tickers"):
            ff5_decompose_portfolio(returns, factors, tickers=["A"])

    def test_single_asset_1d_input(self):
        """1-D excess returns should be reshaped automatically."""
        rng = np.random.default_rng(3)
        n = 100
        factors = rng.normal(0, 0.01, (n, 5))
        excess = rng.normal(0, 0.01, n)

        results = ff5_decompose_portfolio(excess, factors, tickers=["SOLO"])
        assert len(results) == 1
        assert results[0].ticker == "SOLO"


# ── Synthetic factor generation ─────────────────────────────────────


class TestGenerateSyntheticFactors:
    def test_shapes(self):
        factors, rf = generate_synthetic_factors(n_days=100, seed=0)
        assert factors.shape == (100, 5)
        assert rf.shape == (100,)

    def test_deterministic_with_seed(self):
        f1, r1 = generate_synthetic_factors(n_days=50, seed=42)
        f2, r2 = generate_synthetic_factors(n_days=50, seed=42)
        np.testing.assert_array_equal(f1, f2)
        np.testing.assert_array_equal(r1, r2)

    def test_rf_rate_reasonable(self):
        _, rf = generate_synthetic_factors(n_days=252, seed=0)
        # Daily risk-free should be ~0.04/252 ≈ 0.000159
        assert rf[0] == pytest.approx(0.04 / 252, abs=1e-8)

    def test_market_factor_has_positive_mean(self):
        """Over enough days, market factor should average positive."""
        factors, _ = generate_synthetic_factors(n_days=10000, seed=1)
        mkt_mean = factors[:, 0].mean()
        assert mkt_mean > 0  # Market premium is positive


# ── Return attribution ──────────────────────────────────────────────


class TestAttributeReturns:
    def test_basic_attribution(self):
        result = FF5Result(
            ticker="AAPL",
            alpha=0.0002,   # daily
            betas={"Mkt-RF": 1.2, "SMB": 0.3, "HML": -0.1, "RMW": 0.4, "CMA": 0.2},
            r_squared=0.85,
            residual_std=0.01,
            t_stats={"alpha": 1.5, "Mkt-RF": 10.0, "SMB": 2.5, "HML": -0.8, "RMW": 3.2, "CMA": 1.6},
            n_obs=252,
        )

        attr = attribute_returns(result)

        # Alpha contribution: 0.0002 * 252 = 0.0504
        assert attr["alpha"] == pytest.approx(0.0504, abs=1e-6)

        # Mkt-RF contribution: 1.2 * 0.08 = 0.096
        assert attr["Mkt-RF"] == pytest.approx(0.096, abs=1e-6)

        # SMB contribution: 0.3 * 0.02 = 0.006
        assert attr["SMB"] == pytest.approx(0.006, abs=1e-6)

        # HML contribution: -0.1 * 0.03 = -0.003
        assert attr["HML"] == pytest.approx(-0.003, abs=1e-6)

        # Total should sum correctly
        expected_total = (
            attr["alpha"] + attr["Mkt-RF"] + attr["SMB"]
            + attr["HML"] + attr["RMW"] + attr["CMA"]
        )
        assert attr["total"] == pytest.approx(expected_total, abs=1e-10)

    def test_zero_alpha(self):
        result = FF5Result(
            ticker="SPY",
            alpha=0.0,
            betas={"Mkt-RF": 1.0, "SMB": 0.0, "HML": 0.0, "RMW": 0.0, "CMA": 0.0},
            r_squared=0.99,
            residual_std=0.001,
            t_stats={"alpha": 0.0, "Mkt-RF": 50.0, "SMB": 0.0, "HML": 0.0, "RMW": 0.0, "CMA": 0.0},
            n_obs=252,
        )

        attr = attribute_returns(result)
        assert attr["alpha"] == pytest.approx(0.0)
        # Pure market exposure: 1.0 * 0.08 = 0.08
        assert attr["total"] == pytest.approx(0.08, abs=1e-6)

    def test_all_factors_present(self):
        result = FF5Result(
            ticker="X",
            alpha=0.0,
            betas={"Mkt-RF": 1.0, "SMB": 1.0, "HML": 1.0, "RMW": 1.0, "CMA": 1.0},
            r_squared=0.5,
            residual_std=0.01,
            t_stats={"alpha": 0.0, "Mkt-RF": 5.0, "SMB": 3.0, "HML": 2.5, "RMW": 2.0, "CMA": 1.5},
            n_obs=100,
        )

        attr = attribute_returns(result)
        assert "alpha" in attr
        assert "total" in attr
        for name in FACTOR_NAMES:
            assert name in attr


# ── fetch_french_factors ───────────────────────────────────────────


class TestFetchFrenchFactors:
    def test_returns_tuple(self):
        """Mock the download and verify parsing logic."""
        from unittest.mock import patch, MagicMock
        import io
        import zipfile

        # Create a minimal CSV zip in memory
        csv_content = (
            "header line\n"
            ",Mkt-RF,SMB,HML,RMW,CMA,RF\n"
            "20240101,   0.50,   0.10,  -0.20,   0.15,   0.05,   0.02\n"
            "20240102,  -0.30,   0.05,   0.10,  -0.05,   0.08,   0.02\n"
            "20240103,   0.80,  -0.15,   0.30,   0.10,  -0.10,   0.02\n"
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("test.CSV", csv_content)
        buf.seek(0)

        mock_resp = MagicMock()
        mock_resp.read.return_value = buf.read()

        with patch("urllib.request.urlopen", return_value=mock_resp):
            factors, rf = fetch_french_factors()

        assert factors.shape == (3, 5)
        assert rf.shape == (3,)
        # Values should be in decimal (divided by 100)
        assert factors[0, 0] == pytest.approx(0.005)   # 0.50% → 0.005
        assert rf[0] == pytest.approx(0.0002)           # 0.02% → 0.0002

    def test_n_days_truncation(self):
        from unittest.mock import patch, MagicMock
        import io, zipfile

        lines = [f"2024{i:04d},  0.50,  0.10, -0.20,  0.15,  0.05,  0.02" for i in range(100, 200)]
        csv_content = "header\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n" + "\n".join(lines)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("test.CSV", csv_content)
        buf.seek(0)

        mock_resp = MagicMock()
        mock_resp.read.return_value = buf.read()

        with patch("urllib.request.urlopen", return_value=mock_resp):
            factors, rf = fetch_french_factors(n_days=50)

        assert len(factors) == 50
        assert len(rf) == 50

    def test_download_failure_falls_back(self):
        from unittest.mock import patch
        with patch("urllib.request.urlopen", side_effect=Exception("network error")):
            factors, rf = fetch_french_factors(n_days=100)
        # Should fall back to synthetic
        assert factors.shape == (100, 5)
        assert rf.shape == (100,)

    def test_empty_csv_falls_back(self):
        from unittest.mock import patch, MagicMock
        import io, zipfile

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("test.CSV", "just a header\nno data\n")
        buf.seek(0)

        mock_resp = MagicMock()
        mock_resp.read.return_value = buf.read()

        with patch("urllib.request.urlopen", return_value=mock_resp):
            factors, rf = fetch_french_factors(n_days=50)
        # Falls back to synthetic
        assert factors.shape == (50, 5)
