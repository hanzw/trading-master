"""Tests for stop-loss monitoring."""

from unittest.mock import patch

import pytest

from trading_master.db import Database
from trading_master.models import Action
from trading_master.portfolio.stop_loss import StopLossMonitor
from trading_master.portfolio.tracker import PortfolioTracker


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "test.db")


@pytest.fixture
def monitor(db):
    return StopLossMonitor(db=db, default_stop_pct=8.0)


@pytest.fixture
def tracker(db):
    db.set_cash(100000.0)  # High cash so position limit checks pass
    return PortfolioTracker(db=db)


# ── set / get ────────────────────────────────────────────────────────


class TestSetGetStopLoss:
    def test_set_and_get(self, monitor):
        monitor.set_stop_loss("AAPL", 130.0)
        assert monitor.get_stop_loss("AAPL") == 130.0

    def test_get_missing(self, monitor):
        assert monitor.get_stop_loss("XYZ") is None

    def test_overwrite(self, monitor):
        monitor.set_stop_loss("AAPL", 130.0)
        monitor.set_stop_loss("AAPL", 125.0)
        assert monitor.get_stop_loss("AAPL") == 125.0

    def test_case_insensitive(self, monitor):
        monitor.set_stop_loss("aapl", 130.0)
        assert monitor.get_stop_loss("AAPL") == 130.0


# ── auto_set_stops ───────────────────────────────────────────────────


class TestAutoSetStops:
    def test_auto_set_for_new_positions(self, monitor, tracker):
        tracker.execute_action("AAPL", Action.BUY, 10, 150.0)
        tracker.execute_action("MSFT", Action.BUY, 5, 400.0)

        monitor.auto_set_stops(tracker)

        stop_aapl = monitor.get_stop_loss("AAPL")
        assert stop_aapl is not None
        assert pytest.approx(stop_aapl, abs=0.01) == 138.0  # 150 * 0.92

        stop_msft = monitor.get_stop_loss("MSFT")
        assert stop_msft is not None
        assert pytest.approx(stop_msft, abs=0.01) == 368.0  # 400 * 0.92

    def test_does_not_overwrite_existing(self, monitor, tracker):
        tracker.execute_action("AAPL", Action.BUY, 10, 150.0)
        monitor.set_stop_loss("AAPL", 100.0)

        monitor.auto_set_stops(tracker)

        assert monitor.get_stop_loss("AAPL") == 100.0  # unchanged


# ── check_all ────────────────────────────────────────────────────────


class TestCheckAll:
    def test_triggered(self, monitor, tracker):
        tracker.execute_action("AAPL", Action.BUY, 10, 150.0)
        monitor.set_stop_loss("AAPL", 140.0)

        mock_prices = {"AAPL": 135.0}
        with patch.object(
            StopLossMonitor, "_fetch_prices", return_value=mock_prices
        ):
            results = monitor.check_all()

        assert len(results) == 1
        assert results[0]["ticker"] == "AAPL"
        assert results[0]["triggered"] is True
        assert results[0]["current_price"] == 135.0
        assert results[0]["stop_price"] == 140.0

    def test_not_triggered(self, monitor, tracker):
        tracker.execute_action("AAPL", Action.BUY, 10, 150.0)
        monitor.set_stop_loss("AAPL", 130.0)

        mock_prices = {"AAPL": 155.0}
        with patch.object(
            StopLossMonitor, "_fetch_prices", return_value=mock_prices
        ):
            results = monitor.check_all()

        assert len(results) == 1
        assert results[0]["triggered"] is False

    def test_loss_pct_calculation(self, monitor, tracker):
        tracker.execute_action("AAPL", Action.BUY, 10, 200.0)
        monitor.set_stop_loss("AAPL", 180.0)

        mock_prices = {"AAPL": 180.0}
        with patch.object(
            StopLossMonitor, "_fetch_prices", return_value=mock_prices
        ):
            results = monitor.check_all()

        assert results[0]["loss_pct"] == -10.0  # (180-200)/200 * 100

    def test_no_positions(self, monitor):
        results = monitor.check_all()
        assert results == []

    def test_skips_positions_without_stop(self, monitor, tracker):
        tracker.execute_action("AAPL", Action.BUY, 10, 150.0)
        # No stop set

        mock_prices = {"AAPL": 100.0}
        with patch.object(
            StopLossMonitor, "_fetch_prices", return_value=mock_prices
        ):
            results = monitor.check_all()

        assert results == []


# ── Trailing stops ────────────────────────────────────────────────


class TestTrailingStops:
    def test_set_trailing_stop(self, monitor):
        """set_trailing_stop should compute stop = price - multiplier * ATR."""
        with patch.object(StopLossMonitor, "_fetch_atr", return_value=5.0):
            stop = monitor.set_trailing_stop("AAPL", atr_multiplier=2.0, current_price=100.0)

        assert stop == pytest.approx(90.0, abs=0.01)  # 100 - 2*5
        assert monitor.get_stop_loss("AAPL") == pytest.approx(90.0, abs=0.01)

    def test_set_trailing_stop_stores_metadata(self, monitor):
        """Trailing stop metadata should be retrievable."""
        with patch.object(StopLossMonitor, "_fetch_atr", return_value=5.0):
            monitor.set_trailing_stop("AAPL", atr_multiplier=2.5, current_price=100.0)

        meta = monitor.get_trailing_stop_meta("AAPL")
        assert meta is not None
        assert meta["atr_multiplier"] == 2.5
        assert meta["highest_price"] == 100.0
        assert meta["atr"] == 5.0

    def test_trailing_stop_ratchets_up(self, monitor, tracker):
        """When price rises, trailing stop should ratchet up."""
        tracker.execute_action("AAPL", Action.BUY, 10, 100.0)

        # Initially set trailing stop at price=100, ATR=5, mult=2 → stop=90
        with patch.object(StopLossMonitor, "_fetch_atr", return_value=5.0):
            monitor.set_trailing_stop("AAPL", atr_multiplier=2.0, current_price=100.0)

        assert monitor.get_stop_loss("AAPL") == pytest.approx(90.0, abs=0.01)

        # Price rises to 120 → new stop should be 120 - 2*5 = 110
        with patch.object(StopLossMonitor, "_fetch_prices", return_value={"AAPL": 120.0}), \
             patch.object(StopLossMonitor, "_fetch_atr", return_value=5.0):
            results = monitor.update_trailing_stops()

        assert len(results) == 1
        assert results[0]["ticker"] == "AAPL"
        assert results[0]["ratcheted"] is True
        assert results[0]["new_stop"] == pytest.approx(110.0, abs=0.01)
        assert monitor.get_stop_loss("AAPL") == pytest.approx(110.0, abs=0.01)

    def test_trailing_stop_does_not_ratchet_down(self, monitor, tracker):
        """When price drops, trailing stop should NOT decrease."""
        tracker.execute_action("AAPL", Action.BUY, 10, 100.0)

        with patch.object(StopLossMonitor, "_fetch_atr", return_value=5.0):
            monitor.set_trailing_stop("AAPL", atr_multiplier=2.0, current_price=100.0)

        original_stop = monitor.get_stop_loss("AAPL")  # 90.0

        # Price drops to 95 → candidate stop = 95 - 10 = 85, but should stay at 90
        with patch.object(StopLossMonitor, "_fetch_prices", return_value={"AAPL": 95.0}), \
             patch.object(StopLossMonitor, "_fetch_atr", return_value=5.0):
            results = monitor.update_trailing_stops()

        assert len(results) == 1
        assert results[0]["ratcheted"] is False
        assert results[0]["new_stop"] == pytest.approx(original_stop, abs=0.01)
        assert monitor.get_stop_loss("AAPL") == pytest.approx(original_stop, abs=0.01)

    def test_trailing_stop_no_positions_returns_empty(self, monitor):
        """With no positions, update_trailing_stops returns empty list."""
        results = monitor.update_trailing_stops()
        assert results == []

    def test_trailing_stop_skips_non_trailing(self, monitor, tracker):
        """Positions with fixed stops (no trailing metadata) are skipped."""
        tracker.execute_action("AAPL", Action.BUY, 10, 150.0)
        monitor.set_stop_loss("AAPL", 130.0)  # fixed stop, no trailing metadata

        with patch.object(StopLossMonitor, "_fetch_prices", return_value={"AAPL": 160.0}):
            results = monitor.update_trailing_stops()

        assert results == []  # AAPL has no trailing metadata

    def test_set_trailing_stop_zero_atr_raises(self, monitor):
        """If ATR is zero, set_trailing_stop should raise ValueError."""
        with patch.object(StopLossMonitor, "_fetch_atr", return_value=0.0):
            with pytest.raises(ValueError, match="ATR"):
                monitor.set_trailing_stop("AAPL", current_price=100.0)
