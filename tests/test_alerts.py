"""Tests for the unified alert system."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from trading_master.alerts import run_all_alerts, format_alert_report
from trading_master.models import MacroData, MarketRegime, PortfolioState


# ── Helpers ─────────────────────────────────────────────────────────


def _mock_macro(regime: MarketRegime = MarketRegime.BULL) -> MacroData:
    return MacroData(
        us_10yr_yield=4.25,
        us_2yr_yield=4.60,
        vix=18.5,
        vix_regime="normal",
        regime=regime,
        regime_signals=["test"],
        summary="Test.",
    )


def _mock_state(total_value: float = 100_000, cash: float = 20_000) -> PortfolioState:
    return PortfolioState(positions={}, cash=cash, total_value=total_value)


@pytest.fixture
def mock_all(tmp_path):
    """Mock all external dependencies for alert system."""
    mock_db = MagicMock()
    mock_db.cache_get.return_value = None  # no last regime stored

    mock_wm = MagicMock()
    mock_wm.check_alerts.return_value = []  # no watchlist alerts

    mock_slm = MagicMock()
    mock_slm.check_all.return_value = []  # no stop-loss triggers

    mock_tracker = MagicMock()
    mock_tracker.get_state.return_value = _mock_state()

    mock_cb = MagicMock()
    mock_cb.status_with_value.return_value = {
        "triggered": False,
        "hwm": 100_000,
        "current_dd_pct": 0.0,
        "threshold": 15.0,
    }

    macro = _mock_macro()

    with patch("trading_master.alerts.get_db", return_value=mock_db), \
         patch("trading_master.alerts.WatchlistManager", return_value=mock_wm), \
         patch("trading_master.alerts.StopLossMonitor", return_value=mock_slm), \
         patch("trading_master.alerts.PortfolioTracker", return_value=mock_tracker), \
         patch("trading_master.alerts.DrawdownCircuitBreaker", return_value=mock_cb), \
         patch("trading_master.alerts.fetch_macro_data", return_value=macro):
        yield {
            "db": mock_db,
            "wm": mock_wm,
            "slm": mock_slm,
            "tracker": mock_tracker,
            "cb": mock_cb,
            "macro": macro,
        }


# ── run_all_alerts: basic structure ────────────────────────────────


class TestRunAllAlertsStructure:
    def test_returns_dict(self, mock_all):
        result = run_all_alerts()
        assert isinstance(result, dict)

    def test_has_required_keys(self, mock_all):
        result = run_all_alerts()
        for key in ["watchlist_alerts", "stop_loss_alerts", "circuit_breaker",
                     "macro_regime", "regime_changed", "summary", "alert_count"]:
            assert key in result, f"Missing key: {key}"

    def test_all_clear_when_no_alerts(self, mock_all):
        result = run_all_alerts()
        assert result["alert_count"] == 0
        assert result["summary"] == "All clear — no alerts."

    def test_regime_is_bull(self, mock_all):
        result = run_all_alerts()
        assert result["macro_regime"] == "bull"

    def test_no_regime_change_on_first_run(self, mock_all):
        result = run_all_alerts()
        assert result["regime_changed"] is False


# ── Watchlist alerts ───────────────────────────────────────────────


class TestWatchlistAlerts:
    def test_watchlist_alerts_passed_through(self, mock_all):
        mock_all["wm"].check_alerts.return_value = [
            {"ticker": "AAPL", "alert_type": "price_above", "message": "AAPL hit $180"}
        ]
        result = run_all_alerts()
        assert len(result["watchlist_alerts"]) == 1
        assert result["alert_count"] == 1

    def test_watchlist_in_summary(self, mock_all):
        mock_all["wm"].check_alerts.return_value = [
            {"ticker": "AAPL", "alert_type": "price_above", "message": "AAPL hit $180"},
            {"ticker": "MSFT", "alert_type": "price_below", "message": "MSFT dropped"},
        ]
        result = run_all_alerts()
        assert "2 watchlist" in result["summary"]

    def test_cooldown_passed_to_watchlist(self, mock_all):
        run_all_alerts(cooldown_hours=48)
        mock_all["wm"].check_alerts.assert_called_with(cooldown_hours=48)


# ── Stop-loss alerts ───────────────────────────────────────────────


class TestStopLossAlerts:
    def test_triggered_stops_included(self, mock_all):
        mock_all["slm"].check_all.return_value = [
            {"ticker": "AAPL", "current_price": 140, "stop_price": 145,
             "triggered": True, "loss_pct": -8.0},
            {"ticker": "MSFT", "current_price": 350, "stop_price": 300,
             "triggered": False, "loss_pct": 5.0},
        ]
        result = run_all_alerts()
        assert len(result["stop_loss_alerts"]) == 1  # only triggered ones
        assert result["stop_loss_alerts"][0]["ticker"] == "AAPL"

    def test_stop_loss_in_summary(self, mock_all):
        mock_all["slm"].check_all.return_value = [
            {"ticker": "AAPL", "triggered": True, "current_price": 140,
             "stop_price": 145, "loss_pct": -8.0},
        ]
        result = run_all_alerts()
        assert "stop-loss" in result["summary"]

    def test_no_stops_no_alert(self, mock_all):
        mock_all["slm"].check_all.return_value = [
            {"ticker": "AAPL", "triggered": False, "loss_pct": 2.0},
        ]
        result = run_all_alerts()
        assert len(result["stop_loss_alerts"]) == 0


# ── Circuit breaker ────────────────────────────────────────────────


class TestCircuitBreaker:
    def test_circuit_breaker_not_triggered(self, mock_all):
        result = run_all_alerts()
        assert result["circuit_breaker"]["triggered"] is False
        assert result["alert_count"] == 0

    def test_circuit_breaker_triggered(self, mock_all):
        mock_all["cb"].status_with_value.return_value = {
            "triggered": True,
            "hwm": 100_000,
            "current_dd_pct": 16.0,
            "threshold": 15.0,
        }
        result = run_all_alerts()
        assert result["circuit_breaker"]["triggered"] is True
        assert result["alert_count"] == 1
        assert "circuit breaker" in result["summary"]

    def test_records_portfolio_value(self, mock_all):
        run_all_alerts()
        mock_all["cb"].record_portfolio_value.assert_called_once_with(100_000)


# ── Regime change detection ────────────────────────────────────────


class TestRegimeChange:
    def test_regime_change_detected(self, mock_all):
        # Simulate previous regime was "bull", now changed to "bear"
        mock_all["db"].cache_get.return_value = "bull"
        with patch("trading_master.alerts.fetch_macro_data",
                    return_value=_mock_macro(MarketRegime.BEAR)):
            result = run_all_alerts()
        assert result["regime_changed"] is True
        assert result["macro_regime"] == "bear"
        assert "regime changed" in result["summary"]
        assert result["alert_count"] == 1

    def test_no_change_same_regime(self, mock_all):
        mock_all["db"].cache_get.return_value = "bull"
        result = run_all_alerts()
        assert result["regime_changed"] is False

    def test_stores_regime_in_cache(self, mock_all):
        run_all_alerts()
        mock_all["db"].cache_set.assert_called_once()
        call_args = mock_all["db"].cache_set.call_args
        assert call_args[0][0] == "alerts:last_regime"
        assert call_args[0][1] == "bull"


# ── Combined alerts ────────────────────────────────────────────────


class TestCombinedAlerts:
    def test_multiple_alert_types(self, mock_all):
        mock_all["wm"].check_alerts.return_value = [{"ticker": "X", "alert_type": "t", "message": "m"}]
        mock_all["slm"].check_all.return_value = [
            {"ticker": "Y", "triggered": True, "current_price": 10, "stop_price": 12, "loss_pct": -5},
        ]
        mock_all["cb"].status_with_value.return_value = {
            "triggered": True, "hwm": 100000, "current_dd_pct": 20, "threshold": 15,
        }
        result = run_all_alerts()
        assert result["alert_count"] == 3  # watchlist + stop + cb

    def test_summary_joins_multiple(self, mock_all):
        mock_all["wm"].check_alerts.return_value = [{"ticker": "X", "alert_type": "t", "message": "m"}]
        mock_all["slm"].check_all.return_value = [
            {"ticker": "Y", "triggered": True, "current_price": 10, "stop_price": 12, "loss_pct": -5},
        ]
        result = run_all_alerts()
        assert ";" in result["summary"]  # multiple parts joined


# ── format_alert_report ────────────────────────────────────────────


class TestFormatAlertReport:
    def test_returns_string(self, mock_all):
        alerts = run_all_alerts()
        report = format_alert_report(alerts)
        assert isinstance(report, str)

    def test_contains_header(self, mock_all):
        alerts = run_all_alerts()
        report = format_alert_report(alerts)
        assert "ALERT REPORT" in report

    def test_contains_sections(self, mock_all):
        alerts = run_all_alerts()
        report = format_alert_report(alerts)
        assert "Watchlist Alerts" in report
        assert "Stop-Loss Alerts" in report
        assert "Circuit Breaker" in report
        assert "Macro Regime" in report

    def test_no_alerts_clean(self, mock_all):
        alerts = run_all_alerts()
        report = format_alert_report(alerts)
        assert "No watchlist alerts" in report
        assert "No stop-loss alerts" in report
        assert "OK" in report

    def test_stop_loss_alert_formatted(self, mock_all):
        alerts = {
            "watchlist_alerts": [],
            "stop_loss_alerts": [
                {"ticker": "AAPL", "current_price": 140.0, "stop_price": 145.0, "loss_pct": -8.0},
            ],
            "circuit_breaker": {"triggered": False, "hwm": 100000, "current_dd_pct": 0, "threshold": 15},
            "macro_regime": "bull",
            "regime_changed": False,
            "summary": "1 stop-loss triggered",
            "alert_count": 1,
        }
        report = format_alert_report(alerts)
        assert "AAPL" in report
        assert "$140.00" in report
        assert "$145.00" in report

    def test_regime_change_flagged(self, mock_all):
        alerts = {
            "watchlist_alerts": [],
            "stop_loss_alerts": [],
            "circuit_breaker": {"triggered": False, "hwm": 100000, "current_dd_pct": 0, "threshold": 15},
            "macro_regime": "bear",
            "regime_changed": True,
            "summary": "regime changed",
            "alert_count": 1,
        }
        report = format_alert_report(alerts)
        assert "REGIME CHANGED" in report
        assert "BEAR" in report

    def test_circuit_breaker_triggered_report(self, mock_all):
        alerts = {
            "watchlist_alerts": [],
            "stop_loss_alerts": [],
            "circuit_breaker": {"triggered": True, "hwm": 100000, "current_dd_pct": 18.5, "threshold": 15},
            "macro_regime": "bear",
            "regime_changed": False,
            "summary": "circuit breaker triggered",
            "alert_count": 1,
        }
        report = format_alert_report(alerts)
        assert "TRIGGERED" in report
        assert "18.5%" in report
