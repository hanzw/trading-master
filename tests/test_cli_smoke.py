"""CLI smoke tests — ensure commands don't crash on basic invocations."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from trading_master.cli import app
from trading_master.models import (
    MacroData,
    MarketRegime,
    PortfolioState,
    TechnicalData,
)

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

    class budget:
        max_cost_per_run = 5.0
        warn_cost = 2.0
        max_tokens_per_run = 500_000

    class circuit_breaker:
        max_drawdown_pct = 15.0

    project_root = Path(".")


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path):
    """Redirect all DB and config access to a temp directory."""
    from trading_master.db import Database

    fake_cfg = _FakeConfig()
    fake_cfg.project_root = tmp_path

    db = Database(db_path=tmp_path / "test.db")
    _ = db.conn  # trigger table creation

    with patch("trading_master.config.get_config", return_value=fake_cfg), \
         patch("trading_master.config._config", fake_cfg), \
         patch("trading_master.config.get_db_path", return_value=tmp_path / "test.db"), \
         patch("trading_master.db.get_db", return_value=db), \
         patch("trading_master.db._db", db):
        yield db

    db.close()


# ---------------------------------------------------------------------------
# Test 1: --help
# ---------------------------------------------------------------------------

def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Trading Master" in result.output


# ---------------------------------------------------------------------------
# Test 2: portfolio show with empty DB
# ---------------------------------------------------------------------------

def test_portfolio_show_empty():
    """With an empty database, 'portfolio show' should display without crashing."""
    # PortfolioTracker.get_state calls yfinance to fetch prices; mock it
    with patch("trading_master.portfolio.tracker.yf") as mock_yf:
        result = runner.invoke(app, ["portfolio", "show"])
    # Should not crash — exit_code 0
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Test 3: macro command
# ---------------------------------------------------------------------------

def test_macro_command():
    """Macro command should display macro data without error when yfinance is mocked."""
    mock_macro = MacroData(
        us_10yr_yield=4.25,
        us_2yr_yield=4.60,
        yield_curve_spread=-0.35,
        yield_curve_inverted=True,
        vix=18.5,
        vix_regime="normal",
        sp500_price=5200.0,
        sp500_sma200=4800.0,
        sp500_above_sma200=True,
        regime=MarketRegime.BULL,
        regime_signals=["vix_normal"],
        summary="Bull regime.",
    )

    with patch("trading_master.data.macro.fetch_macro_data", return_value=mock_macro), \
         patch("trading_master.cli.fetch_macro_data", mock_macro, create=True):
        # The CLI imports fetch_macro_data inside the function, so patch the module
        with patch.dict("sys.modules", {}):
            # Simpler approach: just patch the data module
            with patch("trading_master.data.macro.fetch_macro_data", return_value=mock_macro):
                result = runner.invoke(app, ["macro"])

    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Test 4: risk sizing with no portfolio positions
# ---------------------------------------------------------------------------

def test_risk_sizing_no_positions():
    """Risk sizing should handle gracefully when the portfolio exists but is empty."""
    mock_tech = TechnicalData(
        ticker="AAPL",
        rsi_14=55.0,
        macd=1.5,
        macd_signal=1.2,
        sma_20=172.0,
        sma_50=168.0,
        sma_200=160.0,
        atr_14=5.0,
        trend="bullish",
    )
    mock_state = PortfolioState(
        positions={},
        cash=10_000.0,
        total_value=10_000.0,
    )

    with patch("trading_master.portfolio.tracker.PortfolioTracker.get_state", return_value=mock_state), \
         patch("trading_master.data.technical.fetch_technicals", return_value=mock_tech):
        result = runner.invoke(app, ["risk", "sizing", "AAPL"])

    # Should succeed without crashing
    assert result.exit_code == 0
    assert "AAPL" in result.output


# ---------------------------------------------------------------------------
# Test 5: subcommand --help variants
# ---------------------------------------------------------------------------

def test_portfolio_help():
    result = runner.invoke(app, ["portfolio", "--help"])
    assert result.exit_code == 0
    assert "Portfolio" in result.output or "portfolio" in result.output.lower()


def test_risk_help():
    result = runner.invoke(app, ["risk", "--help"])
    assert result.exit_code == 0
