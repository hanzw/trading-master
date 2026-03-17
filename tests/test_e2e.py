"""End-to-end integration tests using real SQLite DB and CLI runner.

These tests use a real SQLite database seeded with test positions.
Only network calls (yfinance, market data) are mocked.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from trading_master.cli import app
from trading_master.db import Database
from trading_master.models import MacroData, MarketRegime

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeConfig:
    class llm:
        provider = "openai"
        model = "gpt-4o-mini"
        temperature = 0.3
        max_tokens = 2000

    class portfolio:
        db_path = "data/trading_master.db"
        snapshot_dir = "data/snapshots"
        default_cash = 10000.0

    class analysis:
        debate_rounds = 1
        parallel_analysts = True
        cache_ttl_hours = 4

    class risk:
        max_position_pct = 8.0
        max_sector_pct = 20.0
        stop_loss_pct = 8.0
        holding_days = 20
        tail_multiplier = 2.0

    class budget:
        max_cost_per_run = 5.0
        warn_cost = 2.0
        max_tokens_per_run = 500_000

    class circuit_breaker:
        max_drawdown_pct = 15.0

    project_root = Path(".")


MOCK_PRICES = {"AAPL": 180.0, "MSFT": 410.0, "VIG": 185.0}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def setup_portfolio(tmp_path):
    """Set up a real DB with test positions and cash, returning the DB."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    _ = db.conn  # trigger table creation

    db.set_cash(100_000.0)
    db.upsert_position("AAPL", 50, 175.0, "Technology")
    db.upsert_position("MSFT", 30, 400.0, "Technology")
    db.upsert_position("VIG", 100, 180.0, "")

    fake_cfg = _FakeConfig()
    fake_cfg.project_root = tmp_path

    with patch("trading_master.config.get_config", return_value=fake_cfg), \
         patch("trading_master.config._config", fake_cfg), \
         patch("trading_master.config.get_db_path", return_value=db_path), \
         patch("trading_master.db.get_db", return_value=db), \
         patch("trading_master.db._db", db):
        yield db

    db.close()


def _patch_prices():
    """Return a context manager that mocks yfinance price fetching."""
    return patch(
        "trading_master.portfolio.tracker.PortfolioTracker._fetch_prices",
        return_value=MOCK_PRICES,
    )


# ---------------------------------------------------------------------------
# Test 1: tm portfolio show works with seeded data
# ---------------------------------------------------------------------------

def test_portfolio_show_with_data(setup_portfolio):
    """Portfolio show should display all seeded positions with mocked prices."""
    with _patch_prices():
        result = runner.invoke(app, ["portfolio", "show"])
    assert result.exit_code == 0, result.output
    assert "AAPL" in result.output
    assert "MSFT" in result.output


# ---------------------------------------------------------------------------
# Test 2: tm allocation show produces output
# ---------------------------------------------------------------------------

def test_allocation_show(setup_portfolio):
    """Allocation show should display the regime-adjusted allocation table."""
    with _patch_prices(), \
         patch("trading_master.data.macro.fetch_macro_data") as mock_macro:
        mock_macro.return_value = MacroData(vix=20.0, regime=MarketRegime.SIDEWAYS)
        result = runner.invoke(app, ["allocation", "show", "--model", "balanced", "--regime", "sideways"])
    assert result.exit_code == 0, result.output
    assert "Asset Allocation" in result.output or "allocation" in result.output.lower()


# ---------------------------------------------------------------------------
# Test 3: tm stop-loss auto + tm stop-loss show round-trip
# ---------------------------------------------------------------------------

def test_stop_loss_roundtrip(setup_portfolio):
    """Auto-setting stop-losses then showing them should display all positions."""
    with _patch_prices():
        result = runner.invoke(app, ["stop-loss", "auto"])
    assert result.exit_code == 0, result.output

    with _patch_prices():
        result2 = runner.invoke(app, ["stop-loss", "show"])
    assert result2.exit_code == 0, result2.output
    assert "AAPL" in result2.output


# ---------------------------------------------------------------------------
# Test 4: tm watchlist add + tm watchlist show
# ---------------------------------------------------------------------------

def test_watchlist_roundtrip(setup_portfolio):
    """Adding a watchlist item then showing should display it."""
    result = runner.invoke(app, ["watchlist", "add", "TSLA", "--target-price", "200"])
    assert result.exit_code == 0, result.output
    assert "TSLA" in result.output

    # Mock fetch_market_data for the show command (imported inside the function)
    mock_market = MagicMock()
    mock_market.current_price = 250.0
    mock_market.pe_ratio = 80.0
    mock_market.dividend_yield = 0.0
    with patch("trading_master.data.market.fetch_market_data", return_value=mock_market):
        result2 = runner.invoke(app, ["watchlist", "show"])
    assert result2.exit_code == 0, result2.output
    assert "TSLA" in result2.output


# ---------------------------------------------------------------------------
# Test 5: tm action buy + tm portfolio history
# ---------------------------------------------------------------------------

def test_action_buy_roundtrip(setup_portfolio):
    """Buying shares then viewing history should show the trade."""
    result = runner.invoke(app, ["action", "buy", "GOOGL", "5", "--price", "150"])
    assert result.exit_code == 0, result.output
    assert "GOOGL" in result.output

    result2 = runner.invoke(app, ["portfolio", "history"])
    assert result2.exit_code == 0, result2.output
    assert "GOOGL" in result2.output


# ---------------------------------------------------------------------------
# Test 6: tm report --cron produces plain text
# ---------------------------------------------------------------------------

def test_report_cron(setup_portfolio):
    """Cron report should produce plain text output without crashing."""
    mock_alerts_result = {
        "watchlist_alerts": [],
        "stop_loss_alerts": [],
        "circuit_breaker": {"triggered": False, "current_dd_pct": 0.0, "hwm": 100000.0, "threshold": 15.0},
        "macro_regime": "sideways",
        "regime_changed": False,
        "summary": "No alerts",
        "alert_count": 0,
    }
    with _patch_prices(), \
         patch("trading_master.data.macro.fetch_macro_data") as mock_macro, \
         patch("trading_master.alerts.fetch_macro_data") as mock_macro2, \
         patch("trading_master.portfolio.stop_loss.yf"), \
         patch("trading_master.alerts.StopLossMonitor.check_all", return_value=[]), \
         patch("trading_master.alerts.WatchlistManager.check_alerts", return_value=[]):
        macro_data = MacroData(vix=20.0, regime=MarketRegime.SIDEWAYS)
        mock_macro.return_value = macro_data
        mock_macro2.return_value = macro_data
        result = runner.invoke(app, ["report", "--cron"])
    assert result.exit_code == 0, result.output
    # Report should contain some recognizable text
    out_lower = result.output.lower()
    assert "portfolio" in out_lower or "daily" in out_lower or "report" in out_lower


# ---------------------------------------------------------------------------
# Test 7: tm action sell + cash balance update
# ---------------------------------------------------------------------------

def test_action_sell_updates_cash(setup_portfolio):
    """Selling shares should increase cash balance."""
    db = setup_portfolio
    cash_before = db.get_cash()

    result = runner.invoke(app, ["action", "sell", "AAPL", "10", "--price", "180"])
    assert result.exit_code == 0, result.output
    assert "AAPL" in result.output

    cash_after = db.get_cash()
    assert cash_after == cash_before + (10 * 180.0)


# ---------------------------------------------------------------------------
# Test 8: tm portfolio health round-trip
# ---------------------------------------------------------------------------

def test_portfolio_health(setup_portfolio):
    """Portfolio health should run all checks without crashing."""
    with _patch_prices(), \
         patch("trading_master.data.macro.fetch_macro_data") as mock_macro, \
         patch("trading_master.portfolio.stop_loss.yf"), \
         patch("trading_master.data.dividends.yf"):
        mock_macro.return_value = MacroData(vix=20.0, regime=MarketRegime.SIDEWAYS)
        result = runner.invoke(app, ["portfolio", "health", "--regime", "sideways"])
    assert result.exit_code == 0, result.output
    assert "Health" in result.output or "health" in result.output.lower()
