"""Tests for multi-year fundamental data fetching and computation."""

from __future__ import annotations

import pandas as pd
import pytest

from trading_master.models import FundamentalData
from trading_master.data.fundamentals import (
    compute_cagr,
    classify_margin_trend,
    compute_earnings_quality,
    compute_accruals_ratio,
    _series_from_statement,
    _fetch_multi_year,
)


# ── FundamentalData model tests ─────────────────────────────────────


class TestFundamentalDataModel:
    def test_default_multi_year_fields(self):
        fd = FundamentalData(ticker="AAPL")
        assert fd.revenue_history == []
        assert fd.net_income_history == []
        assert fd.fcf_history == []
        assert fd.margin_history == []
        assert fd.revenue_cagr_3yr is None
        assert fd.margin_trend == ""
        assert fd.earnings_quality is None
        assert fd.accruals_ratio is None

    def test_with_multi_year_fields(self):
        fd = FundamentalData(
            ticker="AAPL",
            revenue_history=[394e9, 383e9, 365e9],
            net_income_history=[100e9, 95e9, 90e9],
            fcf_history=[110e9, 100e9, 92e9],
            margin_history=[0.31, 0.30, 0.28],
            revenue_cagr_3yr=0.038,
            margin_trend="expanding",
            earnings_quality=92.0,
            accruals_ratio=0.02,
        )
        assert len(fd.revenue_history) == 3
        assert fd.revenue_cagr_3yr == 0.038
        assert fd.margin_trend == "expanding"
        assert fd.earnings_quality == 92.0
        assert fd.accruals_ratio == 0.02

    def test_backward_compat_no_multi_year(self):
        """Old-style construction without multi-year fields still works."""
        fd = FundamentalData(
            ticker="MSFT",
            revenue=200e9,
            pe_ratio=30.0,
            summary="test",
        )
        assert fd.ticker == "MSFT"
        assert fd.revenue_history == []


# ── CAGR computation ────────────────────────────────────────────────


class TestComputeCagr:
    def test_positive_growth(self):
        # 100 -> 110 -> 121 over 2 years => 10% CAGR
        values = [121.0, 110.0, 100.0]  # newest first
        cagr = compute_cagr(values)
        assert cagr is not None
        assert abs(cagr - 0.10) < 0.001

    def test_negative_growth(self):
        # 100 -> 90 -> 81 over 2 years
        values = [81.0, 90.0, 100.0]  # newest first
        cagr = compute_cagr(values)
        assert cagr is not None
        assert cagr < 0

    def test_zero_oldest_returns_none(self):
        values = [100.0, 50.0, 0.0]
        assert compute_cagr(values) is None

    def test_negative_oldest_returns_none(self):
        values = [100.0, 50.0, -10.0]
        assert compute_cagr(values) is None

    def test_single_value_returns_none(self):
        assert compute_cagr([100.0]) is None

    def test_empty_returns_none(self):
        assert compute_cagr([]) is None

    def test_two_values(self):
        # 100 -> 200 = 100% growth in 1 year
        cagr = compute_cagr([200.0, 100.0])
        assert cagr is not None
        assert abs(cagr - 1.0) < 0.001


# ── Margin trend classification ─────────────────────────────────────


class TestClassifyMarginTrend:
    def test_expanding(self):
        # newest margin much higher than oldest
        assert classify_margin_trend([0.35, 0.30, 0.25]) == "expanding"

    def test_compressing(self):
        # newest margin much lower than oldest
        assert classify_margin_trend([0.20, 0.25, 0.30]) == "compressing"

    def test_stable(self):
        # diff within 2pp
        assert classify_margin_trend([0.30, 0.29, 0.29]) == "stable"

    def test_exactly_at_threshold_expanding(self):
        # diff = 0.021 > 0.02
        assert classify_margin_trend([0.321, 0.31, 0.30]) == "expanding"

    def test_exactly_at_threshold_stable(self):
        # diff = 0.01 -> within 0.02 threshold -> stable
        assert classify_margin_trend([0.31, 0.305, 0.30]) == "stable"

    def test_single_value(self):
        assert classify_margin_trend([0.30]) == ""

    def test_empty(self):
        assert classify_margin_trend([]) == ""


# ── Earnings quality ─────────────────────────────────────────────────


class TestComputeEarningsQuality:
    def test_strong_quality(self):
        # OCF = NI for each year -> quality = 100
        ni = [100.0, 90.0, 80.0]
        ocf = [100.0, 90.0, 80.0]
        eq = compute_earnings_quality(ni, ocf)
        assert eq is not None
        assert eq == 100.0

    def test_moderate_quality(self):
        ni = [100.0, 100.0]
        ocf = [70.0, 80.0]  # avg ratio = 0.75
        eq = compute_earnings_quality(ni, ocf)
        assert eq is not None
        assert abs(eq - 75.0) < 0.1

    def test_capped_at_100(self):
        # OCF > NI -> ratio > 1 but capped at 100
        ni = [80.0]
        ocf = [120.0]
        eq = compute_earnings_quality(ni, ocf)
        assert eq is not None
        assert eq == 100.0

    def test_skips_negative_net_income(self):
        ni = [-50.0, 100.0]
        ocf = [30.0, 90.0]  # only second pair valid, ratio = 0.9
        eq = compute_earnings_quality(ni, ocf)
        assert eq is not None
        assert abs(eq - 90.0) < 0.1

    def test_all_negative_ni_returns_none(self):
        ni = [-50.0, -30.0]
        ocf = [20.0, 10.0]
        assert compute_earnings_quality(ni, ocf) is None

    def test_empty_inputs(self):
        assert compute_earnings_quality([], [100.0]) is None
        assert compute_earnings_quality([100.0], []) is None
        assert compute_earnings_quality([], []) is None


# ── Accruals ratio ───────────────────────────────────────────────────


class TestComputeAccrualsRatio:
    def test_normal(self):
        # NI=100, OCF=90, Assets=1000 -> (100-90)/1000 = 0.01
        ar = compute_accruals_ratio(100.0, 90.0, 1000.0)
        assert ar is not None
        assert abs(ar - 0.01) < 0.0001

    def test_negative_accruals(self):
        # OCF > NI -> negative ratio (good)
        ar = compute_accruals_ratio(80.0, 100.0, 1000.0)
        assert ar is not None
        assert ar < 0

    def test_missing_ni(self):
        assert compute_accruals_ratio(None, 90.0, 1000.0) is None

    def test_missing_ocf(self):
        assert compute_accruals_ratio(100.0, None, 1000.0) is None

    def test_missing_assets(self):
        assert compute_accruals_ratio(100.0, 90.0, None) is None

    def test_zero_assets(self):
        assert compute_accruals_ratio(100.0, 90.0, 0.0) is None


# ── _series_from_statement ───────────────────────────────────────────


class TestSeriesFromStatement:
    def test_normal_dataframe(self):
        df = pd.DataFrame(
            {"2023": [100.0, 50.0], "2022": [90.0, 45.0]},
            index=["Total Revenue", "Net Income"],
        )
        result = _series_from_statement(df, "Total Revenue")
        assert result == [100.0, 90.0]

    def test_missing_row(self):
        df = pd.DataFrame(
            {"2023": [100.0]},
            index=["Total Revenue"],
        )
        result = _series_from_statement(df, "Net Income")
        assert result == []

    def test_none_df(self):
        assert _series_from_statement(None, "Total Revenue") == []

    def test_empty_df(self):
        df = pd.DataFrame()
        assert _series_from_statement(df, "Total Revenue") == []


# ── _fetch_multi_year with mock stock ────────────────────────────────


class _MockStock:
    """Minimal mock of yfinance Ticker for testing _fetch_multi_year."""

    def __init__(self, financials=None, cashflow=None, balance_sheet=None):
        self.financials = financials
        self.cashflow = cashflow
        self.balance_sheet = balance_sheet


class TestFetchMultiYear:
    def test_with_data(self):
        fin = pd.DataFrame(
            {"2023": [400e9, 100e9], "2022": [380e9, 95e9], "2021": [360e9, 90e9]},
            index=["Total Revenue", "Net Income"],
        )
        cf = pd.DataFrame(
            {"2023": [120e9, -10e9], "2022": [110e9, -9e9], "2021": [100e9, -8e9]},
            index=["Operating Cash Flow", "Capital Expenditure"],
        )
        bs = pd.DataFrame(
            {"2023": [350e9], "2022": [340e9], "2021": [330e9]},
            index=["Total Assets"],
        )
        stock = _MockStock(financials=fin, cashflow=cf, balance_sheet=bs)
        result = _fetch_multi_year(stock)

        assert len(result["revenue_history"]) == 3
        assert len(result["net_income_history"]) == 3
        assert len(result["fcf_history"]) == 3
        assert len(result["margin_history"]) == 3
        assert result["revenue_cagr_3yr"] is not None
        assert result["margin_trend"] in ("expanding", "compressing", "stable")
        assert result["earnings_quality"] is not None
        assert result["accruals_ratio"] is not None

    def test_empty_dataframes(self):
        stock = _MockStock(
            financials=pd.DataFrame(),
            cashflow=pd.DataFrame(),
            balance_sheet=pd.DataFrame(),
        )
        result = _fetch_multi_year(stock)
        assert result["revenue_history"] == []
        assert result["net_income_history"] == []
        assert result["fcf_history"] == []
        assert result["margin_history"] == []
        assert result["revenue_cagr_3yr"] is None
        assert result["margin_trend"] == ""
        assert result["earnings_quality"] is None
        assert result["accruals_ratio"] is None

    def test_none_dataframes(self):
        stock = _MockStock(financials=None, cashflow=None, balance_sheet=None)
        result = _fetch_multi_year(stock)
        assert result["revenue_history"] == []
        assert result["revenue_cagr_3yr"] is None
