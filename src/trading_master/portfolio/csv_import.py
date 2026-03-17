"""CSV and JSON import for portfolio actions."""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path

from ..models import Action, ActionRecord, ActionSource
from .tracker import PortfolioTracker

logger = logging.getLogger(__name__)


def import_csv(path: Path, tracker: PortfolioTracker | None = None) -> list[ActionRecord]:
    """Import trades from a CSV or JSON file and execute them through the tracker.

    CSV expected columns: date, ticker, action, quantity, price
    JSON expected format: array of objects with the same keys.

    Returns list of ActionRecords created.
    """
    path = Path(path)
    tracker = tracker or PortfolioTracker()

    raw_rows = _read_file(path)
    records: list[ActionRecord] = []

    for i, row in enumerate(raw_rows, start=1):
        try:
            ticker = row["ticker"].strip().upper()
            action = Action(row["action"].strip().upper())
            quantity = float(row["quantity"])
            price = float(row["price"])

            # Parse date if present (used for logging, execution is immediate)
            date_str = row.get("date", "")
            if date_str:
                try:
                    _parsed_date = datetime.fromisoformat(date_str.strip())
                except ValueError:
                    # Try common formats
                    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"):
                        try:
                            _parsed_date = datetime.strptime(date_str.strip(), fmt)
                            break
                        except ValueError:
                            continue

            record = tracker.execute_action(
                ticker=ticker,
                action=action,
                quantity=quantity,
                price=price,
                source=ActionSource.CSV_IMPORT,
                reasoning=f"Imported from {path.name} (row {i})",
            )
            records.append(record)
            logger.info("Imported row %d: %s %s %.2f @ %.2f", i, action.value, ticker, quantity, price)

        except (KeyError, ValueError) as exc:
            logger.warning("Skipping row %d: %s — %s", i, row, exc)
            continue

    logger.info("Import complete: %d/%d rows processed from %s", len(records), len(raw_rows), path.name)
    return records


def _read_file(path: Path) -> list[dict]:
    """Read rows from either CSV or JSON file."""
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            return data
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")

    # Default: treat as CSV (covers .csv and anything else)
    reader = csv.DictReader(text.splitlines())
    return list(reader)
