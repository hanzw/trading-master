"""Stop-loss monitoring and enforcement."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import yfinance as yf

from ..db import get_db

if TYPE_CHECKING:
    from .tracker import PortfolioTracker

logger = logging.getLogger(__name__)


class StopLossMonitor:
    """Track and enforce stop-loss levels for portfolio positions."""

    def __init__(self, db=None, default_stop_pct: float = 8.0):
        self.db = db or get_db()
        self.default_stop_pct = default_stop_pct

    # ── CRUD ───────────────────────────────────────────────────────

    def set_stop_loss(self, ticker: str, stop_price: float) -> None:
        """Store stop-loss price for a position in the cache table."""
        key = f"stop_loss:{ticker.upper()}"
        # Use a very long TTL (8760 h = 1 year) so stops persist
        self.db.cache_set(key, stop_price, ttl_hours=8760)

    def get_stop_loss(self, ticker: str) -> float | None:
        """Retrieve stop-loss price for a position."""
        key = f"stop_loss:{ticker.upper()}"
        val = self.db.cache_get(key)
        if val is not None:
            return float(val)
        return None

    # ── Monitoring ─────────────────────────────────────────────────

    def check_all(self) -> list[dict]:
        """Check all positions against their stop-losses.

        Fetches current prices via yfinance.
        Returns a list of dicts with keys:
            ticker, current_price, stop_price, triggered, loss_pct
        """
        positions = self.db.get_all_positions()
        if not positions:
            return []

        tickers = [p["ticker"] for p in positions]
        prices = self._fetch_prices(tickers)

        results: list[dict] = []
        for pos in positions:
            ticker = pos["ticker"]
            stop = self.get_stop_loss(ticker)
            if stop is None:
                continue

            current_price = prices.get(ticker, 0.0)
            triggered = current_price <= stop if current_price > 0 else False
            avg_cost = pos.get("avg_cost", 0.0)
            loss_pct = (
                ((current_price - avg_cost) / avg_cost * 100.0)
                if avg_cost > 0
                else 0.0
            )

            results.append(
                {
                    "ticker": ticker,
                    "current_price": current_price,
                    "stop_price": stop,
                    "triggered": triggered,
                    "loss_pct": round(loss_pct, 2),
                }
            )

        return results

    # ── Auto-set ───────────────────────────────────────────────────

    def auto_set_stops(self, tracker: PortfolioTracker) -> None:
        """For positions without a stop-loss, auto-set at ``avg_cost * (1 - default_stop_pct/100)``."""
        positions = self.db.get_all_positions()
        for pos in positions:
            ticker = pos["ticker"]
            if self.get_stop_loss(ticker) is not None:
                continue
            avg_cost = pos.get("avg_cost", 0.0)
            if avg_cost <= 0:
                continue
            stop_price = avg_cost * (1.0 - self.default_stop_pct / 100.0)
            self.set_stop_loss(ticker, round(stop_price, 4))
            logger.info("Auto-set stop-loss for %s at %.2f", ticker, stop_price)

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _fetch_prices(tickers: list[str]) -> dict[str, float]:
        """Fetch current prices for a list of tickers using yfinance."""
        prices: dict[str, float] = {}
        if not tickers:
            return prices
        try:
            data = yf.Tickers(" ".join(tickers))
            for ticker in tickers:
                try:
                    info = data.tickers[ticker].fast_info
                    prices[ticker] = float(
                        info.get("lastPrice", 0.0) or info.get("last_price", 0.0)
                    )
                except Exception:
                    prices[ticker] = 0.0
        except Exception:
            for ticker in tickers:
                prices[ticker] = 0.0
        return prices
