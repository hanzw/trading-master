"""Recommendation and action history viewer."""

from __future__ import annotations

from ..db import get_db
from .report import print_actions, print_recommendations_list


def show_history(ticker: str | None = None, limit: int = 20) -> None:
    """Load recommendation history from DB and print it."""
    db = get_db()
    recs = db.get_recommendations(ticker=ticker, limit=limit)
    print_recommendations_list(recs)


def show_action_history(ticker: str | None = None, limit: int = 50) -> None:
    """Load action audit trail from DB and print it."""
    db = get_db()
    actions = db.get_actions(ticker=ticker, limit=limit)
    print_actions(actions)
