"""Tests for CSV/JSON portfolio import."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from trading_master.db import Database
from trading_master.models import Action, ActionSource
from trading_master.portfolio.csv_import import import_csv, _read_file
from trading_master.portfolio.tracker import PortfolioTracker


class _FakeConfig:
    class portfolio:
        db_path = "data/trading_master.db"
        snapshot_dir = "data/snapshots"
        default_cash = 100000.0

    class risk:
        max_position_pct = 50.0  # high limit to avoid blocking test imports
        max_sector_pct = 80.0
        stop_loss_pct = 8.0
        holding_days = 20
        tail_multiplier = 2.0

    class circuit_breaker:
        max_drawdown_pct = 15.0

    project_root = Path(".")


@pytest.fixture
def db(tmp_path):
    d = Database(db_path=tmp_path / "test.db")
    _ = d.conn
    return d


@pytest.fixture
def tracker(db):
    return PortfolioTracker(db=db)


@pytest.fixture(autouse=True)
def _mock_config():
    with patch("trading_master.config.get_config", return_value=_FakeConfig()):
        yield


# ── _read_file ─────────────────────────────────────────────────────


class TestReadFile:
    def test_read_csv(self, tmp_path):
        csv_file = tmp_path / "trades.csv"
        csv_file.write_text("date,ticker,action,quantity,price\n2024-01-15,AAPL,BUY,10,150\n")
        rows = _read_file(csv_file)
        assert len(rows) == 1
        assert rows[0]["ticker"] == "AAPL"
        assert rows[0]["action"] == "BUY"

    def test_read_json(self, tmp_path):
        json_file = tmp_path / "trades.json"
        data = [{"date": "2024-01-15", "ticker": "AAPL", "action": "BUY", "quantity": "10", "price": "150"}]
        json_file.write_text(json.dumps(data))
        rows = _read_file(json_file)
        assert len(rows) == 1
        assert rows[0]["ticker"] == "AAPL"

    def test_json_non_array_raises(self, tmp_path):
        json_file = tmp_path / "bad.json"
        json_file.write_text('{"not": "an array"}')
        with pytest.raises(ValueError, match="Expected JSON array"):
            _read_file(json_file)

    def test_multiple_csv_rows(self, tmp_path):
        csv_file = tmp_path / "trades.csv"
        csv_file.write_text(
            "date,ticker,action,quantity,price\n"
            "2024-01-15,AAPL,BUY,10,150\n"
            "2024-01-16,MSFT,BUY,5,300\n"
            "2024-01-17,AAPL,SELL,3,160\n"
        )
        rows = _read_file(csv_file)
        assert len(rows) == 3


# ── import_csv: CSV ────────────────────────────────────────────────


class TestImportCSV:
    def test_basic_csv_import(self, tmp_path, tracker, db):
        csv_file = tmp_path / "trades.csv"
        csv_file.write_text("date,ticker,action,quantity,price\n2024-01-15,AAPL,BUY,5,100\n")
        records = import_csv(csv_file, tracker=tracker)
        assert len(records) == 1
        assert records[0].ticker == "AAPL"
        assert records[0].action == Action.BUY
        assert records[0].source == ActionSource.CSV_IMPORT
        pos = db.get_position("AAPL")
        assert pos["quantity"] == 5

    def test_multiple_trades(self, tmp_path, tracker, db):
        csv_file = tmp_path / "trades.csv"
        csv_file.write_text(
            "date,ticker,action,quantity,price\n"
            "2024-01-15,AAPL,BUY,5,100\n"
            "2024-01-16,MSFT,BUY,3,200\n"
        )
        records = import_csv(csv_file, tracker=tracker)
        assert len(records) == 2
        assert db.get_position("AAPL")["quantity"] == 5
        assert db.get_position("MSFT")["quantity"] == 3

    def test_buy_then_sell(self, tmp_path, tracker, db):
        csv_file = tmp_path / "trades.csv"
        csv_file.write_text(
            "date,ticker,action,quantity,price\n"
            "2024-01-15,AAPL,BUY,10,100\n"
            "2024-01-16,AAPL,SELL,4,120\n"
        )
        records = import_csv(csv_file, tracker=tracker)
        assert len(records) == 2
        pos = db.get_position("AAPL")
        assert pos["quantity"] == 6

    def test_cash_updated(self, tmp_path, tracker, db):
        initial_cash = db.get_cash()
        csv_file = tmp_path / "trades.csv"
        csv_file.write_text("date,ticker,action,quantity,price\n2024-01-15,AAPL,BUY,5,100\n")
        import_csv(csv_file, tracker=tracker)
        assert db.get_cash() == pytest.approx(initial_cash - 500)

    def test_ticker_uppercased(self, tmp_path, tracker, db):
        csv_file = tmp_path / "trades.csv"
        csv_file.write_text("date,ticker,action,quantity,price\n2024-01-15,aapl,buy,5,100\n")
        records = import_csv(csv_file, tracker=tracker)
        assert records[0].ticker == "AAPL"

    def test_reasoning_includes_filename(self, tmp_path, tracker):
        csv_file = tmp_path / "my_trades.csv"
        csv_file.write_text("date,ticker,action,quantity,price\n2024-01-15,AAPL,BUY,5,100\n")
        records = import_csv(csv_file, tracker=tracker)
        assert "my_trades.csv" in records[0].reasoning


# ── import_csv: JSON ───────────────────────────────────────────────


class TestImportJSON:
    def test_json_import(self, tmp_path, tracker, db):
        json_file = tmp_path / "trades.json"
        data = [{"ticker": "GOOGL", "action": "BUY", "quantity": "2", "price": "2800"}]
        json_file.write_text(json.dumps(data))
        records = import_csv(json_file, tracker=tracker)
        assert len(records) == 1
        assert records[0].ticker == "GOOGL"
        assert db.get_position("GOOGL")["quantity"] == 2


# ── Error handling ─────────────────────────────────────────────────


class TestImportErrors:
    def test_bad_row_skipped(self, tmp_path, tracker):
        csv_file = tmp_path / "trades.csv"
        csv_file.write_text(
            "date,ticker,action,quantity,price\n"
            "2024-01-15,AAPL,BUY,5,100\n"
            "2024-01-16,,,bad,row\n"  # invalid
            "2024-01-17,MSFT,BUY,3,200\n"
        )
        records = import_csv(csv_file, tracker=tracker)
        assert len(records) == 2  # bad row skipped

    def test_invalid_action_skipped(self, tmp_path, tracker):
        csv_file = tmp_path / "trades.csv"
        csv_file.write_text("date,ticker,action,quantity,price\n2024-01-15,AAPL,INVALID,5,100\n")
        records = import_csv(csv_file, tracker=tracker)
        assert len(records) == 0

    def test_empty_csv(self, tmp_path, tracker):
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("date,ticker,action,quantity,price\n")
        records = import_csv(csv_file, tracker=tracker)
        assert len(records) == 0


# ── Date parsing ───────────────────────────────────────────────────


class TestDateParsing:
    def test_iso_date(self, tmp_path, tracker):
        csv_file = tmp_path / "trades.csv"
        csv_file.write_text("date,ticker,action,quantity,price\n2024-01-15,AAPL,BUY,5,100\n")
        records = import_csv(csv_file, tracker=tracker)
        assert len(records) == 1  # no crash

    def test_us_date_format(self, tmp_path, tracker):
        csv_file = tmp_path / "trades.csv"
        csv_file.write_text("date,ticker,action,quantity,price\n01/15/2024,AAPL,BUY,5,100\n")
        records = import_csv(csv_file, tracker=tracker)
        assert len(records) == 1

    def test_no_date_column(self, tmp_path, tracker):
        csv_file = tmp_path / "trades.csv"
        csv_file.write_text("ticker,action,quantity,price\nAAPL,BUY,5,100\n")
        records = import_csv(csv_file, tracker=tracker)
        assert len(records) == 1
