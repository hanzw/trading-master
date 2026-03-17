"""Tests for portfolio tracker and CSV import."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from trading_master.db import Database
from trading_master.models import Action, ActionSource
from trading_master.portfolio.tracker import PortfolioTracker


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "test.db")


@pytest.fixture
def tracker(db):
    # Set high cash so position limit checks pass for test trades
    db.set_cash(100000.0)
    return PortfolioTracker(db=db)


def test_execute_buy(tracker):
    record = tracker.execute_action("AAPL", Action.BUY, 10, 150.0)
    assert record.ticker == "AAPL"
    assert record.action == Action.BUY
    assert record.quantity == 10

    pos = tracker.db.get_position("AAPL")
    assert pos["quantity"] == 10
    assert pos["avg_cost"] == 150.0


def test_execute_multiple_buys_avg_cost(tracker):
    tracker.execute_action("AAPL", Action.BUY, 10, 100.0)
    tracker.execute_action("AAPL", Action.BUY, 10, 200.0)

    pos = tracker.db.get_position("AAPL")
    assert pos["quantity"] == 20
    assert pos["avg_cost"] == 150.0  # weighted average


def test_execute_sell(tracker):
    tracker.execute_action("AAPL", Action.BUY, 10, 150.0)
    record = tracker.execute_action("AAPL", Action.SELL, 5, 175.0)

    assert record.action == Action.SELL
    pos = tracker.db.get_position("AAPL")
    assert pos["quantity"] == 5
    assert pos["avg_cost"] == 150.0  # unchanged on sell


def test_execute_sell_reduces_to_zero(tracker):
    tracker.execute_action("AAPL", Action.BUY, 10, 150.0)
    tracker.execute_action("AAPL", Action.SELL, 10, 175.0)

    pos = tracker.db.get_position("AAPL")
    assert pos["quantity"] == 0


def test_cash_updates_on_buy_sell(tracker):
    initial_cash = tracker.db.get_cash()
    tracker.execute_action("AAPL", Action.BUY, 10, 100.0)
    assert tracker.db.get_cash() == initial_cash - 1000.0

    tracker.execute_action("AAPL", Action.SELL, 5, 120.0)
    assert tracker.db.get_cash() == initial_cash - 1000.0 + 600.0


def test_action_logged_with_before_after(tracker):
    tracker.execute_action("AAPL", Action.BUY, 10, 150.0)
    actions = tracker.db.get_actions()
    assert len(actions) == 1
    assert actions[0]["portfolio_before"] != actions[0]["portfolio_after"]


def test_position_limit_enforced(tracker):
    """Buying a position that exceeds max_position_pct should raise ValueError."""
    # With $100k cash, a $50k buy = 50% > 8% limit
    with pytest.raises(ValueError, match="max_position_pct"):
        tracker.execute_action("AAPL", Action.BUY, 100, 500.0)


def test_csv_import(tracker, tmp_path):
    csv_file = tmp_path / "trades.csv"
    csv_file.write_text(
        "date,ticker,action,quantity,price\n"
        "2024-01-15,AAPL,BUY,5,150.00\n"
        "2024-02-01,MSFT,BUY,5,400.00\n"
    )

    from trading_master.portfolio.csv_import import import_csv
    records = import_csv(csv_file, tracker)
    assert len(records) == 2

    pos = tracker.db.get_position("AAPL")
    assert pos["quantity"] == 5

    pos = tracker.db.get_position("MSFT")
    assert pos["quantity"] == 5


def test_csv_import_json(tracker, tmp_path):
    import json
    json_file = tmp_path / "trades.json"
    json_file.write_text(json.dumps([
        {"date": "2024-01-15", "ticker": "AAPL", "action": "BUY", "quantity": "5", "price": "150"},
    ]))

    from trading_master.portfolio.csv_import import import_csv
    records = import_csv(json_file, tracker)
    assert len(records) == 1


def test_csv_import_skips_bad_rows(tracker, tmp_path):
    csv_file = tmp_path / "bad.csv"
    csv_file.write_text(
        "date,ticker,action,quantity,price\n"
        "2024-01-15,AAPL,BUY,5,150.00\n"
        "bad,row,data,x,y\n"
        "2024-02-01,MSFT,BUY,5,400.00\n"
    )

    from trading_master.portfolio.csv_import import import_csv
    records = import_csv(csv_file, tracker)
    assert len(records) == 2  # bad row skipped


def test_snapshot_and_diff(tracker):
    tracker.execute_action("AAPL", Action.BUY, 5, 150.0)

    from trading_master.portfolio.snapshot import take_snapshot
    snap = take_snapshot(tracker)
    assert "positions" in snap or "AAPL" in str(snap)
