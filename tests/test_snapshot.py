"""Tests for portfolio snapshot creation and diffing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from trading_master.models import PortfolioState, Position
from trading_master.portfolio.snapshot import (
    diff_snapshots,
    take_snapshot,
    detect_external_trades,
)


# ── diff_snapshots (pure function) ─────────────────────────────────


class TestDiffSnapshots:
    def test_identical_snapshots_no_changes(self):
        snap = {
            "cash": 10000.0,
            "positions": {"AAPL": {"quantity": 10, "avg_cost": 150.0}},
        }
        assert diff_snapshots(snap, snap) == []

    def test_new_position(self):
        old = {"cash": 10000, "positions": {}}
        new = {"cash": 9000, "positions": {"AAPL": {"quantity": 10, "avg_cost": 100}}}
        changes = diff_snapshots(old, new)
        assert len(changes) >= 1
        new_pos = [c for c in changes if c["change_type"] == "new_position"]
        assert len(new_pos) == 1
        assert new_pos[0]["ticker"] == "AAPL"
        assert new_pos[0]["details"]["quantity"] == 10

    def test_removed_position(self):
        old = {"cash": 9000, "positions": {"AAPL": {"quantity": 10, "avg_cost": 100}}}
        new = {"cash": 10000, "positions": {}}
        changes = diff_snapshots(old, new)
        removed = [c for c in changes if c["change_type"] == "removed_position"]
        assert len(removed) == 1
        assert removed[0]["ticker"] == "AAPL"

    def test_quantity_increase(self):
        old = {"cash": 10000, "positions": {"AAPL": {"quantity": 10, "avg_cost": 100}}}
        new = {"cash": 10000, "positions": {"AAPL": {"quantity": 20, "avg_cost": 100}}}
        changes = diff_snapshots(old, new)
        qty = [c for c in changes if c["change_type"] == "quantity_change"]
        assert len(qty) == 1
        assert qty[0]["details"]["delta"] == 10

    def test_quantity_decrease(self):
        old = {"cash": 10000, "positions": {"AAPL": {"quantity": 20, "avg_cost": 100}}}
        new = {"cash": 10000, "positions": {"AAPL": {"quantity": 5, "avg_cost": 100}}}
        changes = diff_snapshots(old, new)
        qty = [c for c in changes if c["change_type"] == "quantity_change"]
        assert len(qty) == 1
        assert qty[0]["details"]["delta"] == -15

    def test_cash_change(self):
        old = {"cash": 10000.0, "positions": {}}
        new = {"cash": 8000.0, "positions": {}}
        changes = diff_snapshots(old, new)
        cash = [c for c in changes if c["change_type"] == "cash_change"]
        assert len(cash) == 1
        assert cash[0]["details"]["delta"] == pytest.approx(-2000.0)

    def test_small_cash_change_ignored(self):
        old = {"cash": 10000.0, "positions": {}}
        new = {"cash": 10000.005, "positions": {}}
        changes = diff_snapshots(old, new)
        cash = [c for c in changes if c["change_type"] == "cash_change"]
        assert len(cash) == 0  # below 0.01 threshold

    def test_multiple_changes(self):
        old = {
            "cash": 10000,
            "positions": {
                "AAPL": {"quantity": 10, "avg_cost": 100},
                "MSFT": {"quantity": 5, "avg_cost": 300},
            },
        }
        new = {
            "cash": 8000,
            "positions": {
                "AAPL": {"quantity": 15, "avg_cost": 100},  # qty changed
                "GOOGL": {"quantity": 3, "avg_cost": 2800},  # new
                # MSFT removed
            },
        }
        changes = diff_snapshots(old, new)
        types = {c["change_type"] for c in changes}
        assert "new_position" in types
        assert "removed_position" in types
        assert "quantity_change" in types
        assert "cash_change" in types

    def test_empty_snapshots(self):
        assert diff_snapshots({}, {}) == []
        assert diff_snapshots({"cash": 0, "positions": {}}, {"cash": 0, "positions": {}}) == []

    def test_sorted_by_ticker(self):
        old = {"cash": 10000, "positions": {}}
        new = {"cash": 10000, "positions": {
            "MSFT": {"quantity": 5, "avg_cost": 300},
            "AAPL": {"quantity": 10, "avg_cost": 100},
        }}
        changes = diff_snapshots(old, new)
        tickers = [c["ticker"] for c in changes if c["change_type"] == "new_position"]
        assert tickers == sorted(tickers)


# ── take_snapshot ──────────────────────────────────────────────────


class TestTakeSnapshot:
    def test_returns_dict(self, tmp_path):
        mock_state = PortfolioState(
            positions={
                "AAPL": Position(
                    ticker="AAPL", quantity=10, avg_cost=150,
                    current_price=170, market_value=1700,
                ),
            },
            cash=8000,
            total_value=9700,
        )
        mock_tracker = MagicMock()
        mock_tracker.get_state.return_value = mock_state

        mock_db = MagicMock()

        with patch("trading_master.portfolio.snapshot.get_db", return_value=mock_db), \
             patch("trading_master.portfolio.snapshot.get_snapshot_dir", return_value=tmp_path):
            snap = take_snapshot(tracker=mock_tracker)

        assert "timestamp" in snap
        assert snap["cash"] == 8000
        assert snap["total_value"] == 9700
        assert "AAPL" in snap["positions"]
        assert snap["positions"]["AAPL"]["quantity"] == 10

    def test_saves_to_db(self, tmp_path):
        mock_state = PortfolioState(positions={}, cash=10000, total_value=10000)
        mock_tracker = MagicMock()
        mock_tracker.get_state.return_value = mock_state
        mock_db = MagicMock()

        with patch("trading_master.portfolio.snapshot.get_db", return_value=mock_db), \
             patch("trading_master.portfolio.snapshot.get_snapshot_dir", return_value=tmp_path):
            take_snapshot(tracker=mock_tracker)

        mock_db.save_snapshot.assert_called_once()

    def test_saves_json_file(self, tmp_path):
        mock_state = PortfolioState(positions={}, cash=10000, total_value=10000)
        mock_tracker = MagicMock()
        mock_tracker.get_state.return_value = mock_state
        mock_db = MagicMock()

        with patch("trading_master.portfolio.snapshot.get_db", return_value=mock_db), \
             patch("trading_master.portfolio.snapshot.get_snapshot_dir", return_value=tmp_path):
            take_snapshot(tracker=mock_tracker)

        json_files = list(tmp_path.glob("snapshot_*.json"))
        assert len(json_files) == 1

        import json
        content = json.loads(json_files[0].read_text())
        assert content["cash"] == 10000


# ── detect_external_trades ─────────────────────────────────────────


class TestDetectExternalTrades:
    def test_no_previous_snapshot(self, tmp_path):
        mock_tracker = MagicMock()
        mock_db = MagicMock()
        mock_db.get_latest_snapshot.return_value = None

        with patch("trading_master.portfolio.snapshot.get_db", return_value=mock_db):
            changes = detect_external_trades(tracker=mock_tracker)

        assert changes == []

    def test_detects_new_position(self, tmp_path):
        old_snap = {
            "portfolio_json": {"cash": 10000, "positions": {}},
        }
        new_state = PortfolioState(
            positions={"AAPL": Position(
                ticker="AAPL", quantity=10, avg_cost=100,
                current_price=100, market_value=1000,
            )},
            cash=9000, total_value=10000,
        )

        mock_tracker = MagicMock()
        mock_tracker.get_state.return_value = new_state
        mock_db = MagicMock()
        mock_db.get_latest_snapshot.return_value = old_snap

        with patch("trading_master.portfolio.snapshot.get_db", return_value=mock_db), \
             patch("trading_master.portfolio.snapshot.get_snapshot_dir", return_value=tmp_path):
            changes = detect_external_trades(tracker=mock_tracker)

        new_pos = [c for c in changes if c["change_type"] == "new_position"]
        assert len(new_pos) >= 1
