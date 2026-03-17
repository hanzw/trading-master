"""Tests for dividend models and portfolio income computation."""

from unittest.mock import MagicMock, patch

import pytest

from trading_master.db import Database
from trading_master.models import DividendInfo


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "test.db")


# ── DividendInfo model defaults ──────────────────────────────────────


def test_dividend_info_defaults():
    info = DividendInfo(ticker="AAPL")
    assert info.ticker == "AAPL"
    assert info.annual_dividend == 0.0
    assert info.dividend_yield == 0.0
    assert info.payout_ratio is None
    assert info.dividend_growth_rate_5yr is None
    assert info.consecutive_increase_years == 0
    assert info.ex_dividend_date is None
    assert info.sustainability_score == 0.0


def test_dividend_info_with_values():
    info = DividendInfo(
        ticker="MSFT",
        annual_dividend=3.0,
        dividend_yield=0.008,
        payout_ratio=0.35,
        dividend_growth_rate_5yr=10.5,
        consecutive_increase_years=20,
        ex_dividend_date="2024-11-14",
        sustainability_score=85.0,
    )
    assert info.annual_dividend == 3.0
    assert info.consecutive_increase_years == 20
    assert info.sustainability_score == 85.0


def test_dividend_info_serialization():
    info = DividendInfo(ticker="AAPL", annual_dividend=1.0)
    d = info.model_dump()
    assert d["ticker"] == "AAPL"
    assert d["annual_dividend"] == 1.0
    restored = DividendInfo(**d)
    assert restored == info


# ── compute_portfolio_income with mock data ──────────────────────────


def _make_mock_dividend_info(ticker, annual_div, div_yield, growth_rate=5.0):
    return DividendInfo(
        ticker=ticker,
        annual_dividend=annual_div,
        dividend_yield=div_yield,
        payout_ratio=0.4,
        dividend_growth_rate_5yr=growth_rate,
        consecutive_increase_years=5,
        sustainability_score=70.0,
    )


@patch("trading_master.data.dividends.fetch_dividend_info")
def test_compute_portfolio_income_basic(mock_fetch, db):
    from trading_master.data.dividends import compute_portfolio_income

    def side_effect(ticker, db=None):
        data = {
            "AAPL": _make_mock_dividend_info("AAPL", 1.0, 0.005, 8.0),
            "MSFT": _make_mock_dividend_info("MSFT", 3.0, 0.008, 10.0),
        }
        return data[ticker]

    mock_fetch.side_effect = side_effect

    positions = {
        "AAPL": {"quantity": 100, "avg_cost": 150.0},
        "MSFT": {"quantity": 50, "avg_cost": 400.0},
    }
    result = compute_portfolio_income(positions, db=db)

    assert result["total_annual_income"] == 250.0  # 100*1.0 + 50*3.0
    assert abs(result["monthly_average"] - 250.0 / 12) < 0.01
    assert result["yield_on_cost"] > 0
    assert "AAPL" in result["breakdown"]
    assert "MSFT" in result["breakdown"]
    assert result["breakdown"]["AAPL"]["annual_income"] == 100.0
    assert result["breakdown"]["MSFT"]["annual_income"] == 150.0


@patch("trading_master.data.dividends.fetch_dividend_info")
def test_compute_portfolio_income_empty(mock_fetch, db):
    from trading_master.data.dividends import compute_portfolio_income

    result = compute_portfolio_income({}, db=db)
    assert result["total_annual_income"] == 0.0
    assert result["monthly_average"] == 0.0
    assert result["breakdown"] == {}
    mock_fetch.assert_not_called()


@patch("trading_master.data.dividends.fetch_dividend_info")
def test_compute_portfolio_income_zero_quantity(mock_fetch, db):
    from trading_master.data.dividends import compute_portfolio_income

    positions = {"AAPL": {"quantity": 0, "avg_cost": 150.0}}
    result = compute_portfolio_income(positions, db=db)
    assert result["total_annual_income"] == 0.0
    mock_fetch.assert_not_called()


@patch("trading_master.data.dividends.fetch_dividend_info")
def test_compute_portfolio_income_projected_growth(mock_fetch, db):
    from trading_master.data.dividends import compute_portfolio_income

    mock_fetch.return_value = _make_mock_dividend_info("AAPL", 1.0, 0.005, 10.0)

    positions = {"AAPL": {"quantity": 100, "avg_cost": 150.0}}
    result = compute_portfolio_income(positions, db=db)

    # With 10% growth, 5yr projection should be > current
    assert result["projected_5yr_income"] > result["total_annual_income"]


@patch("trading_master.data.dividends.fetch_dividend_info")
def test_compute_portfolio_income_with_position_objects(mock_fetch, db):
    """Test that Position-like objects (with .quantity, .avg_cost) work."""
    from trading_master.data.dividends import compute_portfolio_income

    mock_fetch.return_value = _make_mock_dividend_info("AAPL", 1.0, 0.005)

    class FakePosition:
        quantity = 50
        avg_cost = 200.0

    positions = {"AAPL": FakePosition()}
    result = compute_portfolio_income(positions, db=db)
    assert result["total_annual_income"] == 50.0
