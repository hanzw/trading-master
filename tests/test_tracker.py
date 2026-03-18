"""Tests for PortfolioTracker — CRUD, trades, state management."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from trading_master.db import Database
from trading_master.models import Action, ActionSource, PortfolioState, Position
from trading_master.portfolio.tracker import PortfolioTracker


# ── Fixtures ────────────────────────────────────────────────────────


class _FakeConfig:
    class portfolio:
        db_path = "data/trading_master.db"
        snapshot_dir = "data/snapshots"
        default_cash = 10000.0

    class risk:
        max_position_pct = 8.0
        max_sector_pct = 20.0
        stop_loss_pct = 8.0
        holding_days = 20
        tail_multiplier = 2.0

    class circuit_breaker:
        max_drawdown_pct = 15.0

    project_root = Path(".")


@pytest.fixture
def db(tmp_path):
    """Fresh in-memory database for each test."""
    d = Database(db_path=tmp_path / "test.db")
    _ = d.conn  # trigger table creation
    return d


@pytest.fixture
def tracker(db):
    """Tracker with mocked yfinance prices."""
    t = PortfolioTracker(db=db)
    return t


@pytest.fixture(autouse=True)
def _mock_config():
    with patch("trading_master.portfolio.tracker.get_config", return_value=_FakeConfig(), create=True):
        with patch("trading_master.config.get_config", return_value=_FakeConfig()):
            yield


def _mock_prices(prices: dict[str, float]):
    """Context manager that mocks _fetch_prices to return given prices."""
    return patch.object(PortfolioTracker, "_fetch_prices", return_value=prices)


# ── get_state ──────────────────────────────────────────────────────


class TestGetState:
    def test_empty_portfolio(self, tracker):
        with _mock_prices({}):
            state = tracker.get_state()
        assert isinstance(state, PortfolioState)
        assert len(state.positions) == 0
        assert state.cash == 10000.0  # default
        assert state.total_value == 10000.0

    def test_with_positions(self, tracker, db):
        db.conn.execute(
            "INSERT INTO positions (ticker, quantity, avg_cost, sector, updated_at) VALUES (?, ?, ?, ?, datetime('now'))",
            ("AAPL", 10, 150.0, "Technology"),
        )
        db.conn.commit()

        with _mock_prices({"AAPL": 170.0}):
            state = tracker.get_state()

        assert "AAPL" in state.positions
        pos = state.positions["AAPL"]
        assert pos.quantity == 10
        assert pos.avg_cost == 150.0
        assert pos.current_price == 170.0
        assert pos.market_value == 1700.0
        assert pos.unrealized_pnl == pytest.approx(200.0)

    def test_recalculates_total(self, tracker, db):
        db.conn.execute(
            "INSERT INTO positions (ticker, quantity, avg_cost, updated_at) VALUES (?, ?, ?, datetime('now'))",
            ("AAPL", 10, 150.0),
        )
        db.conn.commit()

        with _mock_prices({"AAPL": 200.0}):
            state = tracker.get_state()

        # total = positions_value (2000) + cash (10000) = 12000
        assert state.total_value == pytest.approx(12000.0)

    def test_multiple_positions(self, tracker, db):
        db.conn.execute("INSERT INTO positions (ticker, quantity, avg_cost, updated_at) VALUES (?, ?, ?, datetime('now'))", ("AAPL", 10, 150.0))
        db.conn.execute("INSERT INTO positions (ticker, quantity, avg_cost, updated_at) VALUES (?, ?, ?, datetime('now'))", ("MSFT", 5, 300.0))
        db.conn.commit()

        with _mock_prices({"AAPL": 160.0, "MSFT": 310.0}):
            state = tracker.get_state()

        assert len(state.positions) == 2
        assert state.total_value == pytest.approx(10000 + 1600 + 1550)


# ── execute_action: BUY ────────────────────────────────────────────


class TestBuy:
    def test_basic_buy(self, tracker, db):
        # Buy $750 = 7.5% of $10k portfolio (under 8% limit)
        record = tracker.execute_action("AAPL", Action.BUY, 5, 150.0)
        assert record.ticker == "AAPL"
        assert record.action == Action.BUY
        assert record.quantity == 5
        assert record.price == 150.0

        pos = db.get_position("AAPL")
        assert pos["quantity"] == 5
        assert pos["avg_cost"] == 150.0

        assert db.get_cash() == pytest.approx(10000 - 750)

    def test_buy_updates_avg_cost(self, tracker, db):
        # Keep each buy well under 8%: 3*50=$150=1.5%, then 2*100=$200 ~ 3.5% cumulative
        tracker.execute_action("AAPL", Action.BUY, 3, 50.0)
        tracker.execute_action("AAPL", Action.BUY, 2, 100.0)

        pos = db.get_position("AAPL")
        assert pos["quantity"] == 5
        assert pos["avg_cost"] == pytest.approx((3*50 + 2*100) / 5)

    def test_buy_insufficient_cash_raises(self, tracker):
        with pytest.raises(ValueError, match="Insufficient cash"):
            tracker.execute_action("AAPL", Action.BUY, 1000, 100.0)  # $100,000 > $10,000

    def test_buy_position_limit_check(self, tracker, db):
        # Set max_position_pct = 8%, portfolio ~$10,000
        # Buying $900 = 9% of portfolio → should raise
        with pytest.raises(ValueError, match="max_position_pct"):
            tracker.execute_action("AAPL", Action.BUY, 9, 100.0)

    def test_buy_logs_action(self, tracker, db):
        record = tracker.execute_action("AAPL", Action.BUY, 5, 100.0, reasoning="Test buy")
        assert record.id is not None
        assert record.reasoning == "Test buy"
        assert record.source == ActionSource.MANUAL

    def test_buy_captures_before_after(self, tracker):
        record = tracker.execute_action("AAPL", Action.BUY, 5, 100.0)
        assert "cash" in record.portfolio_before
        assert "cash" in record.portfolio_after
        assert record.portfolio_after["cash"] < record.portfolio_before["cash"]

    def test_buy_uppercase_ticker(self, tracker, db):
        tracker.execute_action("aapl", Action.BUY, 5, 100.0)
        pos = db.get_position("AAPL")
        assert pos is not None


# ── execute_action: SELL ───────────────────────────────────────────


class TestSell:
    def test_basic_sell(self, tracker, db):
        tracker.execute_action("AAPL", Action.BUY, 5, 100.0)  # $500 = 5%
        record = tracker.execute_action("AAPL", Action.SELL, 3, 120.0)

        pos = db.get_position("AAPL")
        assert pos["quantity"] == 2
        assert pos["avg_cost"] == 100.0  # unchanged on sell

        # Cash: 10000 - 500 + 360 = 9860
        assert db.get_cash() == pytest.approx(9860.0)

    def test_sell_all(self, tracker, db):
        tracker.execute_action("AAPL", Action.BUY, 5, 100.0)
        tracker.execute_action("AAPL", Action.SELL, 5, 120.0)

        pos = db.get_position("AAPL")
        assert pos["quantity"] == 0

    def test_sell_more_than_owned_floors_at_zero(self, tracker, db):
        tracker.execute_action("AAPL", Action.BUY, 3, 100.0)
        tracker.execute_action("AAPL", Action.SELL, 10, 120.0)

        pos = db.get_position("AAPL")
        assert pos["quantity"] == 0  # floors at 0, not -5


# ── execute_action: HOLD ───────────────────────────────────────────


class TestHold:
    def test_hold_no_changes(self, tracker, db):
        initial_cash = db.get_cash()
        record = tracker.execute_action("AAPL", Action.HOLD, 0, 150.0)

        assert db.get_cash() == initial_cash
        assert record.action == Action.HOLD


# ── _calculate_avg_cost ────────────────────────────────────────────


class TestCalculateAvgCost:
    def test_first_purchase(self):
        avg = PortfolioTracker._calculate_avg_cost(0, 0, 10, 100.0)
        assert avg == pytest.approx(100.0)

    def test_add_at_same_price(self):
        avg = PortfolioTracker._calculate_avg_cost(10, 100.0, 10, 100.0)
        assert avg == pytest.approx(100.0)

    def test_add_at_higher_price(self):
        avg = PortfolioTracker._calculate_avg_cost(10, 100.0, 10, 200.0)
        assert avg == pytest.approx(150.0)

    def test_zero_total_quantity(self):
        avg = PortfolioTracker._calculate_avg_cost(0, 0, 0, 100.0)
        assert avg == 0.0


# ── _fetch_prices ──────────────────────────────────────────────────


class TestFetchPrices:
    def test_empty_tickers(self):
        result = PortfolioTracker._fetch_prices([])
        assert result == {}

    def test_returns_dict(self):
        with patch("trading_master.portfolio.tracker.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.fast_info = {"lastPrice": 150.0}
            mock_yf.Tickers.return_value.tickers = {"AAPL": mock_ticker}

            result = PortfolioTracker._fetch_prices(["AAPL"])
        assert "AAPL" in result


# ── get_position_weight ────────────────────────────────────────────


class TestGetPositionWeight:
    def test_no_position_zero_weight(self, tracker):
        with _mock_prices({}):
            weight = tracker.get_position_weight("AAPL")
        assert weight == 0.0

    def test_with_position(self, tracker, db):
        db.conn.execute(
            "INSERT INTO positions (ticker, quantity, avg_cost, updated_at) VALUES (?, ?, ?, datetime('now'))",
            ("AAPL", 50, 200.0),
        )
        db.conn.commit()

        with _mock_prices({"AAPL": 200.0}):
            weight = tracker.get_position_weight("AAPL")
        # market_value = 50 * 200 = 10000, total = 10000 + 10000 = 20000
        assert weight == pytest.approx(50.0)


# ── _snapshot_dict ─────────────────────────────────────────────────


class TestSnapshotDict:
    def test_snapshot_structure(self, tracker, db):
        db.conn.execute(
            "INSERT INTO positions (ticker, quantity, avg_cost, updated_at) VALUES (?, ?, ?, datetime('now'))",
            ("AAPL", 10, 150.0),
        )
        db.conn.commit()

        snap = tracker._snapshot_dict()
        assert "positions" in snap
        assert "cash" in snap
        assert "AAPL" in snap["positions"]
        assert snap["positions"]["AAPL"]["quantity"] == 10

    def test_empty_snapshot(self, tracker):
        snap = tracker._snapshot_dict()
        assert snap["positions"] == {}
        assert snap["cash"] == 10000.0
