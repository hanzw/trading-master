"""Tests for quant package public API — verifies all exports are importable."""

from __future__ import annotations

import pytest


class TestQuantPackageImports:
    """Verify every function in quant.__all__ is importable from the package."""

    def test_import_package(self):
        import trading_master.quant as q
        assert hasattr(q, "__all__")
        assert len(q.__all__) > 20

    def test_all_exports_importable(self):
        from trading_master import quant
        for name in quant.__all__:
            assert hasattr(quant, name), f"quant.{name} not importable"

    # Individual imports — one per module to catch import errors precisely

    def test_import_black_litterman(self):
        from trading_master.quant import run_black_litterman
        assert callable(run_black_litterman)

    def test_import_capm(self):
        from trading_master.quant import capm_regression, capm_expected_return
        assert callable(capm_regression)

    def test_import_cointegration(self):
        from trading_master.quant import cointegration_test, spread_zscore_series
        assert callable(cointegration_test)

    def test_import_compare(self):
        from trading_master.quant import compare_allocations
        assert callable(compare_allocations)

    def test_import_dashboard(self):
        from trading_master.quant import build_dashboard, compute_risk_score
        assert callable(build_dashboard)

    def test_import_dcf(self):
        from trading_master.quant import auto_dcf
        assert callable(auto_dcf)

    def test_import_evt(self):
        from trading_master.quant import evt_tail_risk, mean_excess_plot_data
        assert callable(evt_tail_risk)

    def test_import_fama_french(self):
        from trading_master.quant import ff5_decompose, attribute_returns
        assert callable(ff5_decompose)

    def test_import_garch(self):
        from trading_master.quant import fit_garch, forecast_volatility
        assert callable(fit_garch)

    def test_import_hrp(self):
        from trading_master.quant import hrp_allocation
        assert callable(hrp_allocation)

    def test_import_markowitz(self):
        from trading_master.quant import (
            minimum_variance_portfolio,
            max_sharpe_portfolio,
            efficient_frontier,
        )
        assert callable(minimum_variance_portfolio)
        assert callable(max_sharpe_portfolio)

    def test_import_monte_carlo(self):
        from trading_master.quant import simulate_portfolio_paths, stress_test_scenarios
        assert callable(simulate_portfolio_paths)

    def test_import_multi_timeframe(self):
        from trading_master.quant import multi_timeframe_analysis
        assert callable(multi_timeframe_analysis)

    def test_import_regime(self):
        from trading_master.quant import fit_regime_model
        assert callable(fit_regime_model)

    def test_import_risk_parity(self):
        from trading_master.quant import risk_parity, inverse_volatility
        assert callable(risk_parity)

    def test_import_sector_rotation(self):
        from trading_master.quant import analyze_sectors
        assert callable(analyze_sectors)
