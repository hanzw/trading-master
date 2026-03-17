"""Robinhood brokerage sync via robin_stocks (optional dependency)."""

from __future__ import annotations

import logging
from datetime import datetime

from ..models import Action, ActionRecord, ActionSource
from .tracker import PortfolioTracker

logger = logging.getLogger(__name__)


def is_available() -> bool:
    """Check if robin_stocks is installed and Robinhood credentials are configured."""
    try:
        import robin_stocks.robinhood as rh  # noqa: F401
    except ImportError:
        return False

    # Check for credentials in environment
    from ..config import get_env
    username = get_env("ROBINHOOD_USERNAME")
    password = get_env("ROBINHOOD_PASSWORD")
    return bool(username and password)


def sync_robinhood(tracker: PortfolioTracker | None = None) -> list[ActionRecord]:
    """Fetch current Robinhood positions and reconcile with local portfolio.

    Positions that differ from the local state are logged as external actions.
    Returns a list of ActionRecords for any detected changes.
    """
    tracker = tracker or PortfolioTracker()
    records: list[ActionRecord] = []

    try:
        import robin_stocks.robinhood as rh
    except ImportError:
        logger.warning("robin_stocks is not installed. Run: pip install robin-stocks")
        return records

    # Authenticate
    from ..config import get_env
    username = get_env("ROBINHOOD_USERNAME")
    password = get_env("ROBINHOOD_PASSWORD")
    mfa_code = get_env("ROBINHOOD_MFA")

    if not username or not password:
        logger.warning("Robinhood credentials not found in environment")
        return records

    try:
        login_kwargs = {"username": username, "password": password}
        if mfa_code:
            login_kwargs["mfa_code"] = mfa_code
        rh.login(**login_kwargs)
    except Exception as exc:
        logger.error("Robinhood login failed: %s", exc)
        return records

    try:
        # Fetch positions from Robinhood
        rh_positions = rh.get_open_stock_positions()
        rh_holdings: dict[str, dict] = {}

        for pos in rh_positions:
            try:
                instrument_url = pos.get("instrument", "")
                instrument = rh.get_instrument_by_url(instrument_url)
                ticker = instrument.get("symbol", "").upper()
                if not ticker:
                    continue
                quantity = float(pos.get("quantity", 0))
                avg_cost = float(pos.get("average_buy_price", 0))
                if quantity > 0:
                    rh_holdings[ticker] = {
                        "quantity": quantity,
                        "avg_cost": avg_cost,
                    }
            except Exception as exc:
                logger.warning("Error processing Robinhood position: %s", exc)
                continue

        # Get current local state
        local_positions = {
            p["ticker"]: p for p in tracker.db.get_all_positions()
        }

        # Reconcile: find differences
        all_tickers = set(rh_holdings) | set(local_positions)

        for ticker in sorted(all_tickers):
            rh_pos = rh_holdings.get(ticker)
            local_pos = local_positions.get(ticker)

            rh_qty = rh_pos["quantity"] if rh_pos else 0.0
            local_qty = local_pos["quantity"] if local_pos else 0.0
            delta = rh_qty - local_qty

            if abs(delta) < 1e-9:
                continue  # No change

            # Determine price for the action log
            price = rh_pos["avg_cost"] if rh_pos else (local_pos["avg_cost"] if local_pos else 0.0)

            if delta > 0:
                action = Action.BUY
                reasoning = f"External buy detected via Robinhood sync (RH: {rh_qty}, local: {local_qty})"
            else:
                action = Action.SELL
                delta = abs(delta)
                reasoning = f"External sell detected via Robinhood sync (RH: {rh_qty}, local: {local_qty})"

            record = tracker.execute_action(
                ticker=ticker,
                action=action,
                quantity=abs(delta),
                price=price,
                source=ActionSource.ROBINHOOD,
                reasoning=reasoning,
            )
            records.append(record)
            logger.info("Robinhood sync: %s %s %.2f shares @ %.2f", action.value, ticker, abs(delta), price)

        # Sync cash balance
        try:
            rh_cash = float(rh.load_account_profile().get("cash", 0))
            local_cash = tracker.db.get_cash()
            if abs(rh_cash - local_cash) > 0.01:
                logger.info("Updating cash: %.2f -> %.2f (from Robinhood)", local_cash, rh_cash)
                tracker.db.set_cash(rh_cash)
        except Exception as exc:
            logger.warning("Could not sync cash balance: %s", exc)

    except Exception as exc:
        logger.error("Robinhood sync failed: %s", exc)
    finally:
        try:
            rh.logout()
        except Exception:
            pass

    logger.info("Robinhood sync complete: %d actions recorded", len(records))
    return records
