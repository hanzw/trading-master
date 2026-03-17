"""Portfolio update import: parse text dumps, diff, and apply changes."""

from __future__ import annotations

import logging
import re
from typing import Optional

from ..models import Action, ActionRecord, ActionSource

logger = logging.getLogger(__name__)

# Patterns for recognizing line types
_TICKER_RE = re.compile(r"^[A-Z]{1,5}(?:\.[A-Z])?$")
_DOLLAR_RE = re.compile(r"^\$[\d,]+\.?\d*$")
_SHARES_RE = re.compile(r"^[\d,]+\.?\d*$")
_CHANGE_PCT_RE = re.compile(r"^[+-]?[\d.]+%$")
_CHANGE_DOLLAR_RE = re.compile(r"^[+-]?\$[\d,.]+$")


def _clean_dollar(s: str) -> float:
    """Parse a dollar string like '$1,234.56' into a float."""
    return float(s.replace("$", "").replace(",", ""))


def _clean_number(s: str) -> float:
    """Parse a number string like '1,234.56' into a float."""
    return float(s.replace(",", ""))


def parse_portfolio_text(text: str) -> list[dict]:
    """Parse a text portfolio dump (from Fidelity/Robinhood/etc) into structured data.

    Handles formats like:
        TICKER
        Company Name
        SHARES
        $VALUE
        $PRICE
        CHANGE%
        $CHANGE

    Also handles cash entries (USD, FCASH) and tickers like BRK.B -> BRK-B.

    Returns list of {ticker, shares, value, price} dicts.
    Special entry with ticker='_CASH' for cash balances.
    """
    lines = [line.strip() for line in text.strip().splitlines()]
    lines = [line for line in lines if line]  # remove blanks

    positions: list[dict] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Skip known header/label lines
        if line.lower() in ("symbol", "description", "quantity", "last price",
                            "change", "change %", "current value", "cost basis total",
                            "today's gain/loss", "total gain/loss", "account total"):
            i += 1
            continue

        # Check if this is a cash entry
        if line.upper() in ("USD", "FCASH", "SPAXX", "FDRXX", "CORE"):
            # Look ahead for a dollar value
            j = i + 1
            while j < len(lines) and j < i + 5:
                if _DOLLAR_RE.match(lines[j]):
                    positions.append({
                        "ticker": "_CASH",
                        "shares": 0,
                        "value": _clean_dollar(lines[j]),
                        "price": 0,
                    })
                    i = j + 1
                    break
                j += 1
            else:
                i += 1
            continue

        # Check if this line is a ticker
        ticker_match = _TICKER_RE.match(line)
        if not ticker_match:
            # Could be a company name or other text, skip
            i += 1
            continue

        ticker = line.upper()
        # Convert BRK.B -> BRK-B style
        ticker = ticker.replace(".", "-")

        # Now look ahead for: company name, shares, value, price
        # We expect: name (text), shares (number), value ($), price ($), change%, change$
        j = i + 1
        shares = 0.0
        value = 0.0
        price = 0.0
        found_data = False

        # Scan up to 8 lines ahead for data belonging to this ticker
        scan_limit = min(j + 8, len(lines))
        while j < scan_limit:
            token = lines[j]

            # If we hit another ticker, stop
            if _TICKER_RE.match(token) and token != ticker.replace("-", "."):
                break

            # If this is a cash keyword, stop
            if token.upper() in ("USD", "FCASH", "SPAXX", "FDRXX", "CORE"):
                break

            if token.upper() == "N/A":
                j += 1
                continue

            # Try to parse as shares (plain number, not dollar)
            if not found_data and _SHARES_RE.match(token) and not _DOLLAR_RE.match(token):
                try:
                    shares = _clean_number(token)
                    found_data = True
                except ValueError:
                    pass
                j += 1
                continue

            # Dollar values: first is value, second is price
            if _DOLLAR_RE.match(token):
                val = _clean_dollar(token)
                if found_data and value == 0.0:
                    value = val
                elif found_data and price == 0.0:
                    price = val
                j += 1
                continue

            # Change percent or change dollar - skip
            if _CHANGE_PCT_RE.match(token) or _CHANGE_DOLLAR_RE.match(token):
                j += 1
                continue

            # Otherwise it's probably the company name or other text, skip
            j += 1

        if found_data and shares > 0:
            # If we have value but no price, calculate it
            if price == 0.0 and value > 0.0 and shares > 0:
                price = value / shares
            # If we have price but no value, calculate it
            if value == 0.0 and price > 0.0:
                value = price * shares

            positions.append({
                "ticker": ticker,
                "shares": shares,
                "value": value,
                "price": price,
            })

        i = j if j > i + 1 else i + 1

    return positions


def diff_portfolio(current_positions: dict, new_positions: list[dict]) -> dict:
    """Compare current DB positions against new snapshot.

    Args:
        current_positions: dict of {ticker: {"quantity": float, "avg_cost": float}}
        new_positions: list of dicts from parse_portfolio_text (excluding _CASH)

    Returns:
        {
            "added": [{"ticker": str, "shares": float, "price": float}, ...],
            "removed": [{"ticker": str, "shares": float}, ...],
            "changed": [{"ticker": str, "old_shares": float, "new_shares": float, "diff": float, "price": float}, ...],
            "unchanged": [{"ticker": str, "shares": float}, ...],
        }
    """
    new_map = {p["ticker"]: p for p in new_positions if p["ticker"] != "_CASH"}
    current_tickers = set(current_positions.keys())
    new_tickers = set(new_map.keys())

    added = []
    removed = []
    changed = []
    unchanged = []

    # New positions not in current
    for ticker in sorted(new_tickers - current_tickers):
        p = new_map[ticker]
        added.append({
            "ticker": ticker,
            "shares": p["shares"],
            "price": p["price"],
        })

    # Removed positions (in current but not in new)
    for ticker in sorted(current_tickers - new_tickers):
        removed.append({
            "ticker": ticker,
            "shares": current_positions[ticker]["quantity"],
        })

    # Positions in both — check for quantity changes
    for ticker in sorted(current_tickers & new_tickers):
        old_qty = current_positions[ticker]["quantity"]
        new_qty = new_map[ticker]["shares"]
        if abs(old_qty - new_qty) > 0.0001:
            changed.append({
                "ticker": ticker,
                "old_shares": old_qty,
                "new_shares": new_qty,
                "diff": new_qty - old_qty,
                "price": new_map[ticker]["price"],
            })
        else:
            unchanged.append({
                "ticker": ticker,
                "shares": old_qty,
            })

    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "unchanged": unchanged,
    }


def apply_portfolio_update(
    tracker,
    new_positions: list[dict],
    cash: Optional[float] = None,
) -> list[ActionRecord]:
    """Apply the diff: adjust positions to match new snapshot.

    For increased positions: log BUY action with source=EXTERNAL
    For decreased positions: log SELL action with source=EXTERNAL
    For new positions: log BUY
    For removed positions: log SELL (sell all)

    Returns list of ActionRecords created.
    """
    # Get current state from DB (lightweight, no price fetch)
    current_positions = {
        p["ticker"]: {"quantity": p["quantity"], "avg_cost": p["avg_cost"]}
        for p in tracker.db.get_all_positions()
    }

    # Separate cash entries from positions
    stock_positions = [p for p in new_positions if p["ticker"] != "_CASH"]

    diff = diff_portfolio(current_positions, stock_positions)
    records: list[ActionRecord] = []

    # Handle added positions (BUY)
    for item in diff["added"]:
        try:
            record = tracker.execute_action(
                ticker=item["ticker"],
                action=Action.BUY,
                quantity=item["shares"],
                price=item["price"],
                source=ActionSource.EXTERNAL,
                reasoning="Portfolio update import: new position",
            )
            records.append(record)
            logger.info(
                "Added position: BUY %s %.4f @ $%.2f",
                item["ticker"], item["shares"], item["price"],
            )
        except Exception as exc:
            logger.error("Failed to add %s: %s", item["ticker"], exc)

    # Handle removed positions (SELL all)
    for item in diff["removed"]:
        try:
            # Use avg_cost as price for removed positions (best we have)
            price = current_positions[item["ticker"]]["avg_cost"]
            record = tracker.execute_action(
                ticker=item["ticker"],
                action=Action.SELL,
                quantity=item["shares"],
                price=price,
                source=ActionSource.EXTERNAL,
                reasoning="Portfolio update import: position removed",
            )
            records.append(record)
            logger.info("Removed position: SELL %s %.4f", item["ticker"], item["shares"])
        except Exception as exc:
            logger.error("Failed to remove %s: %s", item["ticker"], exc)

    # Handle changed positions
    for item in diff["changed"]:
        diff_qty = item["diff"]
        try:
            if diff_qty > 0:
                record = tracker.execute_action(
                    ticker=item["ticker"],
                    action=Action.BUY,
                    quantity=diff_qty,
                    price=item["price"],
                    source=ActionSource.EXTERNAL,
                    reasoning=f"Portfolio update import: increased from {item['old_shares']:.4f} to {item['new_shares']:.4f}",
                )
            else:
                record = tracker.execute_action(
                    ticker=item["ticker"],
                    action=Action.SELL,
                    quantity=abs(diff_qty),
                    price=item["price"],
                    source=ActionSource.EXTERNAL,
                    reasoning=f"Portfolio update import: decreased from {item['old_shares']:.4f} to {item['new_shares']:.4f}",
                )
            records.append(record)
            logger.info(
                "Changed position: %s %s %.4f (was %.4f, now %.4f)",
                "BUY" if diff_qty > 0 else "SELL",
                item["ticker"], abs(diff_qty),
                item["old_shares"], item["new_shares"],
            )
        except Exception as exc:
            logger.error("Failed to update %s: %s", item["ticker"], exc)

    # Update cash if provided
    if cash is not None:
        tracker.db.set_cash(cash)
        logger.info("Updated cash balance to $%.2f", cash)

    return records
