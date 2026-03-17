"""Tests for the drawdown circuit breaker."""

import pytest

from trading_master.db import Database
from trading_master.models import Action
from trading_master.portfolio.circuit_breaker import DrawdownCircuitBreaker


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "test.db")


@pytest.fixture
def breaker(db):
    return DrawdownCircuitBreaker(max_drawdown_pct=15.0, db=db)


# ── High-water mark ──────────────────────────────────────────────────


def test_initial_hwm_is_zero(breaker):
    assert breaker.get_high_water_mark() == 0.0


def test_record_and_get_hwm(breaker):
    breaker.record_portfolio_value(10000.0)
    assert breaker.get_high_water_mark() == 10000.0


def test_hwm_updates_on_new_high(breaker):
    breaker.record_portfolio_value(10000.0)
    breaker.record_portfolio_value(12000.0)
    assert breaker.get_high_water_mark() == 12000.0


def test_hwm_does_not_decrease(breaker):
    breaker.record_portfolio_value(12000.0)
    breaker.record_portfolio_value(9000.0)
    assert breaker.get_high_water_mark() == 12000.0


# ── Drawdown calculation ─────────────────────────────────────────────


def test_drawdown_zero_when_no_hwm(breaker):
    assert breaker.get_current_drawdown(10000.0) == 0.0


def test_drawdown_zero_when_at_hwm(breaker):
    breaker.record_portfolio_value(10000.0)
    assert breaker.get_current_drawdown(10000.0) == 0.0


def test_drawdown_zero_when_above_hwm(breaker):
    breaker.record_portfolio_value(10000.0)
    assert breaker.get_current_drawdown(11000.0) == 0.0


def test_drawdown_calculation(breaker):
    breaker.record_portfolio_value(10000.0)
    # 10% drawdown
    dd = breaker.get_current_drawdown(9000.0)
    assert abs(dd - 10.0) < 0.01


def test_drawdown_at_threshold(breaker):
    breaker.record_portfolio_value(10000.0)
    dd = breaker.get_current_drawdown(8500.0)
    assert abs(dd - 15.0) < 0.01


# ── is_triggered ─────────────────────────────────────────────────────


def test_not_triggered_below_threshold(breaker):
    breaker.record_portfolio_value(10000.0)
    assert not breaker.is_triggered(9000.0)  # 10% < 15%


def test_triggered_at_threshold(breaker):
    breaker.record_portfolio_value(10000.0)
    assert breaker.is_triggered(8500.0)  # 15% == 15%


def test_triggered_above_threshold(breaker):
    breaker.record_portfolio_value(10000.0)
    assert breaker.is_triggered(8000.0)  # 20% > 15%


def test_not_triggered_no_hwm(breaker):
    assert not breaker.is_triggered(10000.0)


# ── filter_recommendation ────────────────────────────────────────────


def test_filter_blocks_buy_when_triggered(breaker):
    breaker.record_portfolio_value(10000.0)
    result = breaker.filter_recommendation(Action.BUY, 8000.0)
    assert result == Action.HOLD


def test_filter_allows_sell_when_triggered(breaker):
    breaker.record_portfolio_value(10000.0)
    result = breaker.filter_recommendation(Action.SELL, 8000.0)
    assert result == Action.SELL


def test_filter_allows_hold_when_triggered(breaker):
    breaker.record_portfolio_value(10000.0)
    result = breaker.filter_recommendation(Action.HOLD, 8000.0)
    assert result == Action.HOLD


def test_filter_passes_buy_when_not_triggered(breaker):
    breaker.record_portfolio_value(10000.0)
    result = breaker.filter_recommendation(Action.BUY, 9500.0)
    assert result == Action.BUY


def test_filter_passes_sell_when_not_triggered(breaker):
    breaker.record_portfolio_value(10000.0)
    result = breaker.filter_recommendation(Action.SELL, 9500.0)
    assert result == Action.SELL


# ── status ────────────────────────────────────────────────────────────


def test_status_with_value(breaker):
    breaker.record_portfolio_value(10000.0)
    s = breaker.status_with_value(8500.0)
    assert s["hwm"] == 10000.0
    assert abs(s["current_dd_pct"] - 15.0) < 0.01
    assert s["triggered"] is True
    assert s["threshold"] == 15.0


def test_status_basic(breaker):
    s = breaker.status()
    assert s["threshold"] == 15.0
    assert s["hwm"] == 0.0


# ── Custom threshold ─────────────────────────────────────────────────


def test_custom_threshold(db):
    breaker = DrawdownCircuitBreaker(max_drawdown_pct=5.0, db=db)
    breaker.record_portfolio_value(10000.0)
    assert breaker.is_triggered(9400.0)  # 6% > 5%
    assert not breaker.is_triggered(9600.0)  # 4% < 5%
