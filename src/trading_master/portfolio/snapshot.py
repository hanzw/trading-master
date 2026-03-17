"""Snapshot creation and diffing for portfolio state."""

from __future__ import annotations

import logging
from datetime import datetime

from ..config import get_snapshot_dir
from ..db import get_db
from .tracker import PortfolioTracker

logger = logging.getLogger(__name__)


def take_snapshot(tracker: PortfolioTracker | None = None) -> dict:
    """Capture the current portfolio state and persist it as a JSON snapshot.

    Returns the snapshot dict that was saved.
    """
    tracker = tracker or PortfolioTracker()
    state = tracker.get_state()

    snapshot = {
        "timestamp": state.timestamp.isoformat(),
        "cash": state.cash,
        "total_value": state.total_value,
        "positions": {
            ticker: {
                "quantity": pos.quantity,
                "avg_cost": pos.avg_cost,
                "current_price": pos.current_price,
                "market_value": pos.market_value,
            }
            for ticker, pos in state.positions.items()
        },
    }

    # Persist to database
    db = get_db()
    db.save_snapshot(snapshot, source="snapshot")

    # Also write to snapshot directory as a JSON file
    snap_dir = get_snapshot_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    snap_file = snap_dir / f"snapshot_{ts}.json"
    import json
    snap_file.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    logger.info("Snapshot saved to DB and %s", snap_file)

    return snapshot


def diff_snapshots(old: dict, new: dict) -> list[dict]:
    """Compare two snapshot dicts and return a list of detected changes.

    Each change is a dict with keys: ticker, change_type, details.
    change_type is one of: "new_position", "removed_position", "quantity_change",
                           "price_change", "cash_change".
    """
    changes: list[dict] = []
    old_positions = old.get("positions", {})
    new_positions = new.get("positions", {})

    all_tickers = set(old_positions) | set(new_positions)

    for ticker in sorted(all_tickers):
        old_pos = old_positions.get(ticker)
        new_pos = new_positions.get(ticker)

        if old_pos is None and new_pos is not None:
            changes.append({
                "ticker": ticker,
                "change_type": "new_position",
                "details": {
                    "quantity": new_pos["quantity"],
                    "avg_cost": new_pos.get("avg_cost", 0),
                },
            })
        elif new_pos is None and old_pos is not None:
            changes.append({
                "ticker": ticker,
                "change_type": "removed_position",
                "details": {
                    "old_quantity": old_pos["quantity"],
                },
            })
        else:
            # Both exist — check for quantity changes
            old_qty = old_pos["quantity"]  # type: ignore[index]
            new_qty = new_pos["quantity"]  # type: ignore[index]
            if abs(old_qty - new_qty) > 1e-9:
                changes.append({
                    "ticker": ticker,
                    "change_type": "quantity_change",
                    "details": {
                        "old_quantity": old_qty,
                        "new_quantity": new_qty,
                        "delta": new_qty - old_qty,
                    },
                })

    # Cash change
    old_cash = old.get("cash", 0.0)
    new_cash = new.get("cash", 0.0)
    if abs(old_cash - new_cash) > 0.01:
        changes.append({
            "ticker": "_CASH",
            "change_type": "cash_change",
            "details": {
                "old_cash": old_cash,
                "new_cash": new_cash,
                "delta": new_cash - old_cash,
            },
        })

    return changes


def detect_external_trades(tracker: PortfolioTracker | None = None) -> list[dict]:
    """Compare current live state against the last saved snapshot.

    Returns a list of discrepancies that may indicate external trades
    (e.g., trades made directly in a brokerage).
    """
    tracker = tracker or PortfolioTracker()
    db = get_db()

    last_snap = db.get_latest_snapshot()
    if last_snap is None:
        logger.info("No previous snapshot found; cannot detect external trades.")
        return []

    old = last_snap["portfolio_json"]
    current = take_snapshot(tracker)

    changes = diff_snapshots(old, current)
    if changes:
        logger.info("Detected %d discrepancies vs last snapshot", len(changes))
    else:
        logger.info("No discrepancies detected vs last snapshot")

    return changes
