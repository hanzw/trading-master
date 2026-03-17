"""Tests for watchlist and unified alert system."""

from unittest.mock import patch, MagicMock

import pytest

from trading_master.db import Database
from trading_master.models import MarketData, MacroData, MarketRegime
from trading_master.portfolio.watchlist import WatchlistManager


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "test.db")


@pytest.fixture
def wm(db):
    return WatchlistManager(db=db)


# ── Add / Remove / Get ─────────────────────────────────────────────


class TestAddRemoveGet:
    def test_add_and_get(self, wm):
        wm.add("AAPL", target_price=180.0, thesis="Good value")
        items = wm.get_all()
        assert len(items) == 1
        assert items[0]["ticker"] == "AAPL"
        assert items[0]["target_price"] == 180.0
        assert items[0]["thesis"] == "Good value"

    def test_add_case_insensitive(self, wm):
        wm.add("aapl", target_price=180.0)
        items = wm.get_all()
        assert items[0]["ticker"] == "AAPL"

    def test_add_multiple(self, wm):
        wm.add("AAPL", target_price=180.0)
        wm.add("MSFT", target_price=400.0)
        wm.add("GOOG", max_pe=25.0)
        items = wm.get_all()
        assert len(items) == 3
        tickers = [i["ticker"] for i in items]
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert "GOOG" in tickers

    def test_remove(self, wm):
        wm.add("AAPL", target_price=180.0)
        wm.add("MSFT", target_price=400.0)
        wm.remove("AAPL")
        items = wm.get_all()
        assert len(items) == 1
        assert items[0]["ticker"] == "MSFT"

    def test_remove_nonexistent_is_noop(self, wm):
        wm.remove("XYZ")  # should not raise
        assert wm.get_all() == []

    def test_add_with_all_criteria(self, wm):
        wm.add("VZ", target_price=40.0, max_pe=10.0, min_yield=0.06, thesis="Telecom value")
        items = wm.get_all()
        assert len(items) == 1
        item = items[0]
        assert item["target_price"] == 40.0
        assert item["max_pe"] == 10.0
        assert item["min_yield"] == 0.06
        assert item["thesis"] == "Telecom value"

    def test_update_notes(self, wm):
        wm.add("AAPL")
        wm.update_notes("AAPL", "Earnings next week")
        items = wm.get_all()
        assert items[0]["notes"] == "Earnings next week"

    def test_readd_reactivates(self, wm):
        wm.add("AAPL", target_price=180.0)
        wm.remove("AAPL")
        assert wm.get_all() == []
        wm.add("AAPL", target_price=170.0)
        items = wm.get_all()
        assert len(items) == 1
        assert items[0]["target_price"] == 170.0


# ── Check Alerts ────────────────────────────────────────────────────


def _make_market(ticker, price=0.0, pe=None, div_yield=None):
    return MarketData(
        ticker=ticker,
        current_price=price,
        pe_ratio=pe,
        dividend_yield=div_yield,
    )


class TestCheckAlerts:
    def test_price_target_hit(self, wm):
        wm.add("AAPL", target_price=180.0)
        mock_market = _make_market("AAPL", price=175.0)

        with patch("trading_master.portfolio.watchlist.fetch_market_data", return_value=mock_market):
            alerts = wm.check_alerts()

        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "price_target"
        assert alerts[0]["ticker"] == "AAPL"
        assert alerts[0]["current_value"] == 175.0
        assert alerts[0]["target_value"] == 180.0

    def test_price_target_not_hit(self, wm):
        wm.add("AAPL", target_price=180.0)
        mock_market = _make_market("AAPL", price=195.0)

        with patch("trading_master.portfolio.watchlist.fetch_market_data", return_value=mock_market):
            alerts = wm.check_alerts()

        assert alerts == []

    def test_pe_target_hit(self, wm):
        wm.add("GOOG", max_pe=25.0)
        mock_market = _make_market("GOOG", price=150.0, pe=22.0)

        with patch("trading_master.portfolio.watchlist.fetch_market_data", return_value=mock_market):
            alerts = wm.check_alerts()

        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "pe_target"
        assert alerts[0]["current_value"] == 22.0

    def test_pe_target_not_hit(self, wm):
        wm.add("GOOG", max_pe=25.0)
        mock_market = _make_market("GOOG", price=150.0, pe=30.0)

        with patch("trading_master.portfolio.watchlist.fetch_market_data", return_value=mock_market):
            alerts = wm.check_alerts()

        assert alerts == []

    def test_yield_target_hit(self, wm):
        wm.add("VZ", min_yield=0.06)
        mock_market = _make_market("VZ", price=40.0, div_yield=0.065)

        with patch("trading_master.portfolio.watchlist.fetch_market_data", return_value=mock_market):
            alerts = wm.check_alerts()

        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "yield_target"
        assert alerts[0]["current_value"] == 0.065

    def test_yield_target_not_hit(self, wm):
        wm.add("VZ", min_yield=0.06)
        mock_market = _make_market("VZ", price=40.0, div_yield=0.04)

        with patch("trading_master.portfolio.watchlist.fetch_market_data", return_value=mock_market):
            alerts = wm.check_alerts()

        assert alerts == []

    def test_multiple_criteria_multiple_alerts(self, wm):
        wm.add("VZ", target_price=42.0, max_pe=12.0, min_yield=0.06)
        mock_market = _make_market("VZ", price=40.0, pe=10.0, div_yield=0.07)

        with patch("trading_master.portfolio.watchlist.fetch_market_data", return_value=mock_market):
            alerts = wm.check_alerts()

        # All three criteria should fire
        assert len(alerts) == 3
        types = {a["alert_type"] for a in alerts}
        assert types == {"price_target", "pe_target", "yield_target"}

    def test_empty_watchlist(self, wm):
        alerts = wm.check_alerts()
        assert alerts == []

    def test_fetch_failure_skips(self, wm):
        wm.add("BADTICKER", target_price=10.0)

        with patch(
            "trading_master.portfolio.watchlist.fetch_market_data",
            side_effect=Exception("API error"),
        ):
            alerts = wm.check_alerts()

        assert alerts == []


# ── run_all_alerts ──────────────────────────────────────────────────


class TestRunAllAlerts:
    def test_consolidation(self, db):
        """Test that run_all_alerts aggregates watchlist + stop-loss + circuit breaker + macro."""
        from trading_master.alerts import run_all_alerts

        # Set up watchlist
        wm = WatchlistManager(db=db)
        wm.add("AAPL", target_price=180.0)

        mock_market = _make_market("AAPL", price=175.0)
        mock_macro = MacroData(regime=MarketRegime.BULL)

        with patch("trading_master.portfolio.watchlist.fetch_market_data", return_value=mock_market), \
             patch("trading_master.alerts.fetch_macro_data", return_value=mock_macro), \
             patch("trading_master.alerts.get_db", return_value=db), \
             patch("trading_master.portfolio.watchlist.get_db", return_value=db), \
             patch("trading_master.portfolio.stop_loss.get_db", return_value=db), \
             patch("trading_master.alerts.PortfolioTracker") as MockTracker:

            # Mock tracker to return a portfolio state
            mock_state = MagicMock()
            mock_state.total_value = 100000.0
            MockTracker.return_value.get_state.return_value = mock_state

            result = run_all_alerts()

        assert "watchlist_alerts" in result
        assert "stop_loss_alerts" in result
        assert "circuit_breaker" in result
        assert "macro_regime" in result
        assert "regime_changed" in result
        assert "summary" in result
        assert "alert_count" in result

        # We should have at least the watchlist alert
        assert len(result["watchlist_alerts"]) == 1
        assert result["alert_count"] >= 1

    def test_regime_change_detection(self, db):
        """Test that regime_changed is detected on second call."""
        from trading_master.alerts import run_all_alerts, _LAST_REGIME_CACHE_KEY

        # Simulate a previous regime stored in cache
        db.cache_set(_LAST_REGIME_CACHE_KEY, "bull", ttl_hours=876_000)

        mock_macro = MacroData(regime=MarketRegime.BEAR)

        with patch("trading_master.alerts.fetch_macro_data", return_value=mock_macro), \
             patch("trading_master.alerts.get_db", return_value=db), \
             patch("trading_master.portfolio.watchlist.get_db", return_value=db), \
             patch("trading_master.portfolio.stop_loss.get_db", return_value=db), \
             patch("trading_master.alerts.PortfolioTracker") as MockTracker:

            mock_state = MagicMock()
            mock_state.total_value = 100000.0
            MockTracker.return_value.get_state.return_value = mock_state

            result = run_all_alerts()

        assert result["regime_changed"] is True
        assert result["macro_regime"] == "bear"

    def test_no_regime_change(self, db):
        """Test that regime_changed is False when regime stays the same."""
        from trading_master.alerts import run_all_alerts, _LAST_REGIME_CACHE_KEY

        db.cache_set(_LAST_REGIME_CACHE_KEY, "bull", ttl_hours=876_000)

        mock_macro = MacroData(regime=MarketRegime.BULL)

        with patch("trading_master.alerts.fetch_macro_data", return_value=mock_macro), \
             patch("trading_master.alerts.get_db", return_value=db), \
             patch("trading_master.portfolio.watchlist.get_db", return_value=db), \
             patch("trading_master.portfolio.stop_loss.get_db", return_value=db), \
             patch("trading_master.alerts.PortfolioTracker") as MockTracker:

            mock_state = MagicMock()
            mock_state.total_value = 100000.0
            MockTracker.return_value.get_state.return_value = mock_state

            result = run_all_alerts()

        assert result["regime_changed"] is False


# ── format_alert_report ─────────────────────────────────────────────


class TestFormatAlertReport:
    def test_format_with_alerts(self):
        from trading_master.alerts import format_alert_report

        data = {
            "watchlist_alerts": [
                {"alert_type": "price_target", "message": "AAPL hit $175", "ticker": "AAPL",
                 "current_value": 175.0, "target_value": 180.0}
            ],
            "stop_loss_alerts": [
                {"ticker": "MSFT", "current_price": 380.0, "stop_price": 390.0, "loss_pct": -5.0}
            ],
            "circuit_breaker": {"triggered": False, "hwm": 100000, "current_dd_pct": 2.0, "threshold": 15.0},
            "macro_regime": "bull",
            "regime_changed": False,
            "summary": "1 watchlist target(s) hit; 1 stop-loss(es) triggered",
            "alert_count": 2,
        }

        report = format_alert_report(data)
        assert "ALERT REPORT" in report
        assert "AAPL" in report
        assert "MSFT" in report
        assert "BULL" in report

    def test_format_no_alerts(self):
        from trading_master.alerts import format_alert_report

        data = {
            "watchlist_alerts": [],
            "stop_loss_alerts": [],
            "circuit_breaker": {"triggered": False, "hwm": 100000, "current_dd_pct": 0.0, "threshold": 15.0},
            "macro_regime": "sideways",
            "regime_changed": False,
            "summary": "All clear — no alerts.",
            "alert_count": 0,
        }

        report = format_alert_report(data)
        assert "0 alert(s)" in report
        assert "No watchlist alerts" in report
