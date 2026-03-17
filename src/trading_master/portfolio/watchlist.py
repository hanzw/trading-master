"""Watchlist management — track tickers you want to buy with entry criteria."""

from __future__ import annotations

import logging
from datetime import datetime

from ..db import get_db
from ..data.market import fetch_market_data

logger = logging.getLogger(__name__)


class WatchlistManager:
    """Manage a watchlist of tickers with target entry criteria."""

    def __init__(self, db=None):
        self.db = db or get_db()
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create watchlist table if not exists."""
        self.db.conn.executescript("""
            CREATE TABLE IF NOT EXISTS watchlist (
                ticker TEXT PRIMARY KEY,
                added_date TEXT NOT NULL,
                thesis TEXT DEFAULT '',
                target_price REAL,
                max_pe REAL,
                min_yield REAL,
                notes TEXT DEFAULT '',
                active INTEGER DEFAULT 1
            );
        """)

    # ── CRUD ────────────────────────────────────────────────────────

    def add(
        self,
        ticker: str,
        target_price: float | None = None,
        thesis: str = "",
        max_pe: float | None = None,
        min_yield: float | None = None,
    ) -> None:
        """Add ticker to watchlist with optional entry criteria."""
        ticker = ticker.upper()
        self.db.conn.execute(
            """INSERT INTO watchlist (ticker, added_date, thesis, target_price, max_pe, min_yield)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(ticker) DO UPDATE SET
                 thesis = CASE WHEN excluded.thesis = '' THEN thesis ELSE excluded.thesis END,
                 target_price = COALESCE(excluded.target_price, target_price),
                 max_pe = COALESCE(excluded.max_pe, max_pe),
                 min_yield = COALESCE(excluded.min_yield, min_yield),
                 active = 1""",
            (ticker, datetime.now().isoformat(), thesis, target_price, max_pe, min_yield),
        )
        self.db.conn.commit()
        logger.info("Added %s to watchlist", ticker)

    def remove(self, ticker: str) -> None:
        """Remove from watchlist (sets active=0)."""
        ticker = ticker.upper()
        self.db.conn.execute(
            "UPDATE watchlist SET active = 0 WHERE ticker = ?", (ticker,)
        )
        self.db.conn.commit()
        logger.info("Removed %s from watchlist", ticker)

    def get_all(self) -> list[dict]:
        """Get all active watchlist items."""
        rows = self.db.conn.execute(
            "SELECT * FROM watchlist WHERE active = 1 ORDER BY ticker"
        ).fetchall()
        return [dict(r) for r in rows]

    def update_notes(self, ticker: str, notes: str) -> None:
        """Add notes to a watchlist item."""
        ticker = ticker.upper()
        self.db.conn.execute(
            "UPDATE watchlist SET notes = ? WHERE ticker = ?", (notes, ticker)
        )
        self.db.conn.commit()

    # ── Alert checking ──────────────────────────────────────────────

    def check_alerts(self) -> list[dict]:
        """Check all watchlist items against current market data.

        For each item, fetch current price/PE/yield from yfinance.
        Returns list of triggered alerts:
            {ticker, alert_type, message, current_value, target_value}
        Alert types: 'price_target' (price <= target), 'pe_target' (PE <= max_pe),
                     'yield_target' (yield >= min_yield)
        """
        items = self.get_all()
        if not items:
            return []

        alerts: list[dict] = []
        for item in items:
            ticker = item["ticker"]
            try:
                market = fetch_market_data(ticker)
            except Exception as exc:
                logger.warning("Failed to fetch data for %s: %s", ticker, exc)
                continue

            # Price target check
            if item["target_price"] is not None and market.current_price > 0:
                if market.current_price <= item["target_price"]:
                    alerts.append({
                        "ticker": ticker,
                        "alert_type": "price_target",
                        "message": (
                            f"{ticker} hit price target: "
                            f"${market.current_price:.2f} <= ${item['target_price']:.2f}"
                        ),
                        "current_value": market.current_price,
                        "target_value": item["target_price"],
                    })

            # PE target check
            if item["max_pe"] is not None and market.pe_ratio is not None:
                if market.pe_ratio <= item["max_pe"]:
                    alerts.append({
                        "ticker": ticker,
                        "alert_type": "pe_target",
                        "message": (
                            f"{ticker} PE ratio attractive: "
                            f"{market.pe_ratio:.1f} <= {item['max_pe']:.1f}"
                        ),
                        "current_value": market.pe_ratio,
                        "target_value": item["max_pe"],
                    })

            # Yield target check
            if item["min_yield"] is not None and market.dividend_yield is not None:
                if market.dividend_yield >= item["min_yield"]:
                    alerts.append({
                        "ticker": ticker,
                        "alert_type": "yield_target",
                        "message": (
                            f"{ticker} yield attractive: "
                            f"{market.dividend_yield:.2%} >= {item['min_yield']:.2%}"
                        ),
                        "current_value": market.dividend_yield,
                        "target_value": item["min_yield"],
                    })

        return alerts
