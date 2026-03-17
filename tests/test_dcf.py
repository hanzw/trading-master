"""Tests for DCF valuation model."""

from __future__ import annotations

import pytest

from trading_master.quant.dcf import dcf_valuation, gordon_growth_model


# ---------------------------------------------------------------------------
# dcf_valuation
# ---------------------------------------------------------------------------

class TestDCFValuation:
    def test_basic_valuation(self):
        """Known-input smoke test."""
        result = dcf_valuation(
            fcf_current=100.0,
            growth_rate_5yr=0.10,
            terminal_growth=0.03,
            discount_rate=0.10,
            shares_outstanding=1.0,
        )
        # 5 projected FCFs
        assert len(result["fcf_projections"]) == 5
        assert result["fcf_projections"][0] == pytest.approx(110.0)
        assert result["fcf_projections"][4] == pytest.approx(100 * 1.1**5)
        assert result["intrinsic_value"] > 0
        assert result["pv_fcf"] > 0
        assert result["pv_terminal"] > 0
        assert result["intrinsic_value"] == pytest.approx(
            (result["pv_fcf"] + result["pv_terminal"]) / 1.0
        )

    def test_margin_of_safety(self):
        result = dcf_valuation(fcf_current=100, growth_rate_5yr=0.10)
        assert result["with_margin_of_safety"] == pytest.approx(
            result["intrinsic_value"] * 0.75
        )

    def test_upside_with_price(self):
        result = dcf_valuation(
            fcf_current=100,
            growth_rate_5yr=0.10,
            current_price=50.0,
            shares_outstanding=1.0,
        )
        expected_upside = (result["intrinsic_value"] - 50.0) / 50.0
        assert result["upside_pct"] == pytest.approx(expected_upside)

    def test_upside_none_without_price(self):
        result = dcf_valuation(fcf_current=100, growth_rate_5yr=0.10)
        assert result["upside_pct"] is None

    def test_zero_growth(self):
        result = dcf_valuation(
            fcf_current=100,
            growth_rate_5yr=0.0,
            terminal_growth=0.02,
            discount_rate=0.10,
        )
        # All 5 FCFs should be 100
        for fcf in result["fcf_projections"]:
            assert fcf == pytest.approx(100.0)
        assert result["intrinsic_value"] > 0

    def test_negative_growth(self):
        result = dcf_valuation(
            fcf_current=100,
            growth_rate_5yr=-0.05,
            terminal_growth=0.02,
            discount_rate=0.10,
        )
        # FCFs should be decreasing
        for i in range(1, 5):
            assert result["fcf_projections"][i] < result["fcf_projections"][i - 1]

    def test_discount_rate_lte_terminal_raises(self):
        with pytest.raises(ValueError, match="discount_rate.*must exceed"):
            dcf_valuation(fcf_current=100, growth_rate_5yr=0.10, discount_rate=0.03, terminal_growth=0.03)

        with pytest.raises(ValueError, match="discount_rate.*must exceed"):
            dcf_valuation(fcf_current=100, growth_rate_5yr=0.10, discount_rate=0.02, terminal_growth=0.03)

    def test_terminal_value_formula(self):
        result = dcf_valuation(
            fcf_current=100,
            growth_rate_5yr=0.10,
            terminal_growth=0.03,
            discount_rate=0.10,
        )
        fcf5 = result["fcf_projections"][-1]
        expected_tv = fcf5 * (1.03) / (0.10 - 0.03)
        assert result["terminal_value"] == pytest.approx(expected_tv)


# ---------------------------------------------------------------------------
# gordon_growth_model
# ---------------------------------------------------------------------------

class TestGordonGrowthModel:
    def test_basic_formula(self):
        # D=2, g=5%, r=10% => P = 2*1.05/0.05 = 42
        assert gordon_growth_model(2.0, 0.05, 0.10) == pytest.approx(42.0)

    def test_zero_growth(self):
        # D=3, g=0, r=8% => P = 3*1.0/0.08 = 37.5
        assert gordon_growth_model(3.0, 0.0, 0.08) == pytest.approx(37.5)

    def test_negative_growth(self):
        # D=4, g=-2%, r=10% => P = 4*0.98/0.12 = 32.6667
        assert gordon_growth_model(4.0, -0.02, 0.10) == pytest.approx(4 * 0.98 / 0.12)

    def test_discount_lte_growth_raises(self):
        with pytest.raises(ValueError):
            gordon_growth_model(2.0, 0.10, 0.10)
        with pytest.raises(ValueError):
            gordon_growth_model(2.0, 0.10, 0.05)
