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

    # ── Trailing stops ──────────────────────────────────────────────

    def set_trailing_stop(
        self,
        ticker: str,
        atr_multiplier: float = 2.5,
        current_price: float | None = None,
    ) -> float:
        """Set a trailing stop at current_price - (atr_multiplier * ATR).

        Fetches ATR from technical data. Stores in cache as stop_loss:{ticker}.
        Also stores trailing stop metadata (atr_multiplier, highest_price) so
        update_trailing_stops can ratchet.

        Returns the stop price.
        """
        ticker = ticker.upper()

        # Fetch current price if not provided
        if current_price is None:
            prices = self._fetch_prices([ticker])
            current_price = prices.get(ticker, 0.0)
            if current_price <= 0:
                raise ValueError(f"Cannot fetch current price for {ticker}")

        # Fetch ATR from technical data
        atr = self._fetch_atr(ticker)
        if atr <= 0:
            raise ValueError(f"Cannot fetch ATR for {ticker}")

        stop_price = current_price - (atr_multiplier * atr)
        stop_price = round(max(stop_price, 0.0), 4)

        # Store stop price
        self.set_stop_loss(ticker, stop_price)

        # Store trailing stop metadata
        meta_key = f"trailing_stop_meta:{ticker}"
        self.db.cache_set(meta_key, {
            "atr_multiplier": atr_multiplier,
            "highest_price": current_price,
            "atr": atr,
        }, ttl_hours=8760)

        logger.info(
            "Set trailing stop for %s: price=%.2f, ATR=%.2f, multiplier=%.1f, stop=%.2f",
            ticker, current_price, atr, atr_multiplier, stop_price,
        )
        return stop_price

    def update_trailing_stops(self) -> list[dict]:
        """For all positions with trailing stops, ratchet UP if price has risen.

        New stop = max(current_stop, highest_price - k*ATR).
        Returns list of {ticker, old_stop, new_stop, ratcheted: bool}.
        """
        positions = self.db.get_all_positions()
        if not positions:
            return []

        tickers = [p["ticker"] for p in positions]
        prices = self._fetch_prices(tickers)

        results: list[dict] = []
        for pos in positions:
            ticker = pos["ticker"]
            meta_key = f"trailing_stop_meta:{ticker}"
            meta = self.db.cache_get(meta_key)
            if meta is None:
                continue

            current_price = prices.get(ticker, 0.0)
            if current_price <= 0:
                continue

            atr_multiplier = meta["atr_multiplier"]
            highest_price = meta["highest_price"]
            old_stop = self.get_stop_loss(ticker) or 0.0

            # Refresh ATR
            try:
                atr = self._fetch_atr(ticker)
            except Exception:
                atr = meta.get("atr", 0.0)

            if atr <= 0:
                continue

            # Update highest price
            new_highest = max(highest_price, current_price)

            # Compute new stop
            candidate_stop = round(new_highest - (atr_multiplier * atr), 4)
            candidate_stop = max(candidate_stop, 0.0)
            new_stop = max(old_stop, candidate_stop)

            ratcheted = new_stop > old_stop

            if ratcheted:
                self.set_stop_loss(ticker, new_stop)

            # Always update metadata with latest highest_price and atr
            self.db.cache_set(meta_key, {
                "atr_multiplier": atr_multiplier,
                "highest_price": new_highest,
                "atr": atr,
            }, ttl_hours=8760)

            results.append({
                "ticker": ticker,
                "old_stop": old_stop,
                "new_stop": new_stop,
                "ratcheted": ratcheted,
            })

        return results

    def get_trailing_stop_meta(self, ticker: str) -> dict | None:
        """Return trailing stop metadata for a ticker, or None if not a trailing stop."""
        meta_key = f"trailing_stop_meta:{ticker.upper()}"
        return self.db.cache_get(meta_key)

    @staticmethod
    def _fetch_atr(ticker: str) -> float:
        """Fetch current ATR(14) for a ticker using yfinance."""
        try:
            hist = yf.Ticker(ticker).history(period="1mo")
            if hist.empty or len(hist) < 15:
                return 0.0
            high = hist["High"].values
            low = hist["Low"].values
            close = hist["Close"].values
            tr = []
            for i in range(1, len(high)):
                tr.append(max(
                    high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]),
                ))
            if len(tr) < 14:
                return float(sum(tr) / len(tr)) if tr else 0.0
            # Simple moving average of last 14 TRs
            return float(sum(tr[-14:]) / 14)
        except Exception as exc:
            logger.warning("Could not fetch ATR for %s: %s", ticker, exc)
            return 0.0

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
