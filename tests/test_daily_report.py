"""Tests for daily portfolio health report generator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from trading_master.models import (
    MacroData,
    MarketRegime,
    PortfolioState,
    Position,
)
from trading_master.output.daily_report import (
    generate_daily_report,
    generate_cron_report,
    _build_report_text,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _mock_state(
    total_value: float = 100_000,
    cash: float = 20_000,
    positions: dict | None = None,
) -> PortfolioState:
    if positions is None:
        positions = {
            "AAPL": Position(
                ticker="AAPL", quantity=50, avg_cost=150, current_price=170,
                market_value=8500, unrealized_pnl=1000, pnl_pct=13.3, sector="Technology",
            ),
            "MSFT": Position(
                ticker="MSFT", quantity=30, avg_cost=300, current_price=350,
                market_value=10500, unrealized_pnl=1500, pnl_pct=16.7, sector="Technology",
            ),
        }
    return PortfolioState(
        positions=positions,
        cash=cash,
        total_value=total_value,
    )


def _mock_macro(**overrides) -> MacroData:
    defaults = dict(
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
    defaults.update(overrides)
    return MacroData(**defaults)


def _mock_alerts(**overrides) -> dict:
    defaults = {
        "alert_count": 0,
        "summary": "No alerts",
        "watchlist_alerts": [],
        "stop_loss_alerts": [],
        "circuit_breaker": {"triggered": False},
        "regime_changed": False,
    }
    defaults.update(overrides)
    return defaults


@pytest.fixture
def mock_env(tmp_path):
    """Mock all external dependencies for report generation."""
    state = _mock_state()
    macro = _mock_macro()
    alerts = _mock_alerts()

    mock_db = MagicMock()
    mock_db.get_all_positions.return_value = []

    mock_tracker = MagicMock()
    mock_tracker.get_state.return_value = state

    with patch("trading_master.output.daily_report.get_db", return_value=mock_db), \
         patch("trading_master.output.daily_report.PortfolioTracker", return_value=mock_tracker), \
         patch("trading_master.output.daily_report.fetch_macro_data", return_value=macro), \
         patch("trading_master.output.daily_report.run_all_alerts", return_value=alerts), \
         patch("trading_master.output.daily_report.format_alert_report", return_value="All clear."):
        yield {
            "state": state,
            "macro": macro,
            "alerts": alerts,
            "db": mock_db,
            "tracker": mock_tracker,
            "tmp_path": tmp_path,
        }


# ── Basic structure ────────────────────────────────────────────────


class TestReportStructure:
    def test_contains_header(self, mock_env):
        report = _build_report_text()
        assert "DAILY PORTFOLIO HEALTH REPORT" in report

    def test_contains_footer(self, mock_env):
        report = _build_report_text()
        assert "END OF REPORT" in report

    def test_contains_all_sections(self, mock_env):
        report = _build_report_text()
        assert "MACRO ENVIRONMENT" in report
        assert "ALERT SUMMARY" in report
        assert "ALLOCATION DRIFT" in report
        assert "TOP CONCERNS" in report
        assert "ACTION ITEMS" in report

    def test_returns_string(self, mock_env):
        report = _build_report_text()
        assert isinstance(report, str)
        assert len(report) > 100


# ── Portfolio section ──────────────────────────────────────────────


class TestPortfolioSection:
    def test_shows_portfolio_value(self, mock_env):
        report = _build_report_text()
        assert "$100,000.00" in report

    def test_shows_cash(self, mock_env):
        report = _build_report_text()
        assert "$20,000.00" in report

    def test_shows_position_count(self, mock_env):
        report = _build_report_text()
        assert "Positions:" in report
        assert "2" in report

    def test_shows_cash_percentage(self, mock_env):
        report = _build_report_text()
        assert "20.0%" in report


# ── Macro section ──────────────────────────────────────────────────


class TestMacroSection:
    def test_shows_regime(self, mock_env):
        report = _build_report_text()
        assert "BULL" in report

    def test_shows_vix(self, mock_env):
        report = _build_report_text()
        assert "18.5" in report

    def test_shows_yields(self, mock_env):
        report = _build_report_text()
        assert "4.25" in report

    def test_inverted_yield_curve_flagged(self, mock_env):
        report = _build_report_text()
        assert "INVERTED" in report

    def test_sp500_vs_sma(self, mock_env):
        report = _build_report_text()
        assert "ABOVE" in report

    def test_macro_failure_handled(self, mock_env):
        with patch("trading_master.output.daily_report.fetch_macro_data",
                    side_effect=RuntimeError("network error")):
            report = _build_report_text()
        assert "Macro data unavailable" in report
        # Report should still complete
        assert "END OF REPORT" in report


# ── Alert section ──────────────────────────────────────────────────


class TestAlertSection:
    def test_shows_alert_summary(self, mock_env):
        report = _build_report_text()
        assert "All clear" in report

    def test_alert_failure_handled(self, mock_env):
        with patch("trading_master.output.daily_report.run_all_alerts",
                    side_effect=RuntimeError("alert error")):
            report = _build_report_text()
        assert "Alert check failed" in report
        assert "END OF REPORT" in report


# ── Concerns section ───────────────────────────────────────────────


class TestConcernsSection:
    def test_healthy_portfolio(self, mock_env):
        report = _build_report_text()
        assert "No concerns" in report or "healthy" in report.lower()


# ── Action items section ───────────────────────────────────────────


class TestActionItems:
    def test_no_actions_when_clean(self, mock_env):
        report = _build_report_text()
        assert "No action required" in report

    def test_stop_loss_action(self, mock_env):
        alerts = _mock_alerts(
            alert_count=1,
            stop_loss_alerts=[{
                "ticker": "AAPL",
                "current_price": 140.0,
                "stop_price": 145.0,
            }],
        )
        with patch("trading_master.output.daily_report.run_all_alerts", return_value=alerts), \
             patch("trading_master.output.daily_report.format_alert_report", return_value="1 alert"):
            report = _build_report_text()
        assert "SELL AAPL" in report

    def test_circuit_breaker_action(self, mock_env):
        alerts = _mock_alerts(
            alert_count=1,
            circuit_breaker={"triggered": True},
        )
        with patch("trading_master.output.daily_report.run_all_alerts", return_value=alerts), \
             patch("trading_master.output.daily_report.format_alert_report", return_value="CB triggered"):
            report = _build_report_text()
        assert "HALT ALL BUYING" in report

    def test_regime_change_action(self, mock_env):
        alerts = _mock_alerts(
            regime_changed=True,
            macro_regime="bear",
        )
        with patch("trading_master.output.daily_report.run_all_alerts", return_value=alerts), \
             patch("trading_master.output.daily_report.format_alert_report", return_value="Regime changed"):
            report = _build_report_text()
        assert "Regime changed" in report
        assert "BEAR" in report


# ── File output ────────────────────────────────────────────────────


class TestFileOutput:
    def test_generates_file(self, mock_env):
        tmp = mock_env["tmp_path"]
        filepath = generate_daily_report(output_dir=tmp)
        assert filepath.exists()
        assert filepath.suffix == ".txt"
        content = filepath.read_text(encoding="utf-8")
        assert "DAILY PORTFOLIO HEALTH REPORT" in content

    def test_filename_contains_date(self, mock_env):
        tmp = mock_env["tmp_path"]
        filepath = generate_daily_report(output_dir=tmp)
        assert "daily_report_" in filepath.name
        assert filepath.name.endswith(".txt")

    def test_creates_output_dir(self, mock_env):
        tmp = mock_env["tmp_path"] / "subdir" / "reports"
        filepath = generate_daily_report(output_dir=tmp)
        assert tmp.exists()
        assert filepath.exists()


# ── Cron report ────────────────────────────────────────────────────


class TestCronReport:
    def test_returns_string(self, mock_env):
        report = generate_cron_report()
        assert isinstance(report, str)
        assert "DAILY PORTFOLIO HEALTH REPORT" in report

    def test_no_file_created(self, mock_env):
        tmp = mock_env["tmp_path"]
        generate_cron_report()
        # No file should be created by cron report
        assert not list(tmp.glob("daily_report_*"))
