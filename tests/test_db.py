"""Tests for the database layer."""

import tempfile
from pathlib import Path

import pytest

from trading_master.db import Database
from trading_master.models import Action, ActionRecord, ActionSource, Recommendation


@pytest.fixture
def db(tmp_path):
    """Create a temporary database."""
    db_path = tmp_path / "test.db"
    return Database(db_path)


def test_upsert_and_get_position(db):
    db.upsert_position("AAPL", 10, 150.0, "Technology")
    pos = db.get_position("AAPL")
    assert pos is not None
    assert pos["quantity"] == 10
    assert pos["avg_cost"] == 150.0
    assert pos["sector"] == "Technology"


def test_upsert_updates_existing(db):
    db.upsert_position("AAPL", 10, 150.0)
    db.upsert_position("AAPL", 15, 155.0)
    pos = db.get_position("AAPL")
    assert pos["quantity"] == 15
    assert pos["avg_cost"] == 155.0


def test_get_all_positions(db):
    db.upsert_position("AAPL", 10, 150.0)
    db.upsert_position("MSFT", 5, 400.0)
    db.upsert_position("GONE", 0, 100.0)  # zero qty, should be excluded
    positions = db.get_all_positions()
    assert len(positions) == 2
    tickers = {p["ticker"] for p in positions}
    assert tickers == {"AAPL", "MSFT"}


def test_cash_operations(db):
    # Default cash comes from config, but let's set it explicitly
    db.set_cash(50000.0)
    assert db.get_cash() == 50000.0
    db.set_cash(45000.0)
    assert db.get_cash() == 45000.0


def test_log_and_get_actions(db):
    record = ActionRecord(
        ticker="AAPL",
        action=Action.BUY,
        quantity=10,
        price=150.0,
        source=ActionSource.MANUAL,
        reasoning="Test buy",
    )
    action_id = db.log_action(record)
    assert action_id is not None

    actions = db.get_actions()
    assert len(actions) == 1
    assert actions[0]["ticker"] == "AAPL"
    assert actions[0]["action"] == "BUY"
    assert actions[0]["reasoning"] == "Test buy"


def test_get_actions_filtered_by_ticker(db):
    r1 = ActionRecord(ticker="AAPL", action=Action.BUY, quantity=10, price=150.0)
    r2 = ActionRecord(ticker="MSFT", action=Action.BUY, quantity=5, price=400.0)
    db.log_action(r1)
    db.log_action(r2)

    aapl_actions = db.get_actions(ticker="AAPL")
    assert len(aapl_actions) == 1
    assert aapl_actions[0]["ticker"] == "AAPL"


def test_save_and_get_recommendation(db):
    rec = Recommendation(
        ticker="AAPL",
        action=Action.BUY,
        confidence=80.0,
        summary="Strong buy signal",
        llm_tokens_used=1000,
        llm_cost_usd=0.05,
    )
    rec_id = db.save_recommendation(rec)
    assert rec_id is not None

    recs = db.get_recommendations()
    assert len(recs) == 1
    assert recs[0]["ticker"] == "AAPL"
    assert recs[0]["confidence"] == 80.0


def test_snapshot_save_and_get(db):
    portfolio = {"positions": {"AAPL": {"qty": 10}}, "cash": 5000}
    snap_id = db.save_snapshot(portfolio, source="test")
    assert snap_id is not None

    latest = db.get_latest_snapshot()
    assert latest is not None
    assert latest["portfolio_json"]["cash"] == 5000


def test_cache_operations(db):
    db.cache_set("test_key", {"value": 42}, ttl_hours=1)
    result = db.cache_get("test_key")
    assert result == {"value": 42}

    # Non-existent key
    assert db.cache_get("nonexistent") is None


def test_case_insensitive_ticker(db):
    db.upsert_position("aapl", 10, 150.0)
    pos = db.get_position("aapl")
    assert pos["ticker"] == "AAPL"
