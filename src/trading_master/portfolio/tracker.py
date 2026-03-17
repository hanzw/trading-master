"""Portfolio CRUD and state management."""

from __future__ import annotations

import logging
from datetime import datetime

import yfinance as yf

import json as _json

from ..db import get_db
from ..models import Action, ActionRecord, ActionSource, Position, PortfolioState

logger = logging.getLogger(__name__)


class PortfolioTracker:
    """High-level wrapper around the Database for portfolio operations."""

    def __init__(self, db=None):
        self.db = db or get_db()

    # ── Read state ────────────────────────────────────────────────

    def get_state(self) -> PortfolioState:
        """Load all positions, fetch current prices via yfinance, return full state."""
        rows = self.db.get_all_positions()
        positions: dict[str, Position] = {}

        # Collect tickers for a single bulk fetch
        tickers = [r["ticker"] for r in rows]

        # Fetch current prices (batch call)
        prices = self._fetch_prices(tickers)

        for row in rows:
            ticker = row["ticker"]
            pos = Position(
                ticker=ticker,
                quantity=row["quantity"],
                avg_cost=row["avg_cost"],
                sector=row.get("sector", ""),
            )
            price = prices.get(ticker, 0.0)
            pos.update_market(price)
            positions[ticker] = pos

        cash = self.db.get_cash()
        state = PortfolioState(
            positions=positions,
            cash=cash,
            timestamp=datetime.now(),
        )
        state.recalculate()

        # Record portfolio value for circuit breaker high-water mark tracking
        try:
            from .circuit_breaker import DrawdownCircuitBreaker
            breaker = DrawdownCircuitBreaker(db=self.db)
            breaker.record_portfolio_value(state.total_value)
        except Exception:
            logger.debug("Circuit breaker recording failed (non-fatal)", exc_info=True)

        return state

    def get_position_weight(self, ticker: str) -> float:
        """Return the position's percentage weight in total portfolio value."""
        state = self.get_state()
        if state.total_value <= 0:
            return 0.0
        pos = state.positions.get(ticker.upper())
        if pos is None:
            return 0.0
        return (pos.market_value / state.total_value) * 100.0

    # ── Execute trades ────────────────────────────────────────────

    def execute_action(
        self,
        ticker: str,
        action: Action,
        quantity: float,
        price: float,
        source: ActionSource = ActionSource.MANUAL,
        reasoning: str = "",
    ) -> ActionRecord:
        """Execute a BUY/SELL, update position and cash, log with before/after state."""
        ticker = ticker.upper()

        # Capture before-state (lightweight, no price fetch)
        before_state = self._snapshot_dict()

        existing = self.db.get_position(ticker)
        old_qty = existing["quantity"] if existing else 0.0
        old_avg = existing["avg_cost"] if existing else 0.0

        if action == Action.BUY:
            cost = quantity * price
            cash = self.db.get_cash()

            # Guard: insufficient cash
            if cash < cost:
                raise ValueError(
                    f"Insufficient cash: ${cash:.2f} available, ${cost:.2f} required"
                )

            # Estimate total portfolio value = current positions value + cash.
            # Ideally we'd use live market prices, but fetching them here is too
            # slow for a pre-trade guard.  Compromise: use avg_cost * 1.5 safety
            # multiplier so that appreciated positions aren't under-counted,
            # which would let oversized positions slip through.  If a cached
            # price exists in the DB (from a recent `tm portfolio show` or
            # `tm stop-loss check`), prefer that instead.
            all_positions = self.db.get_all_positions()

            def _estimated_price(p: dict) -> float:
                """Return best available price estimate for position-limit math."""
                tkr = p["ticker"]
                avg = p["avg_cost"]
                # Try cached price from stop-loss / portfolio-show cache
                try:
                    import json as _j
                    row = self.db.conn.execute(
                        "SELECT value FROM cache WHERE key = ? AND expires_at > datetime('now')",
                        (f"price:{tkr}",),
                    ).fetchone()
                    if row:
                        cached = float(_j.loads(row[0]))
                        if cached > 0:
                            return cached
                except Exception:
                    pass
                # Fallback: avg_cost * 1.5 safety multiplier (overestimates to
                # prevent limit breaches on appreciated stocks)
                return avg * 1.5

            positions_value = sum(
                p["quantity"] * _estimated_price(p) for p in all_positions
            )
            total_value = positions_value + cash

            # Pre-trade validation: max position size check
            from ..config import get_config
            cfg = get_config()
            if total_value > 0:
                existing_value = old_qty * old_avg
                new_position_value = existing_value + cost
                position_pct = (new_position_value / total_value) * 100
                if position_pct > cfg.risk.max_position_pct:
                    raise ValueError(
                        f"Position size {position_pct:.1f}% would exceed "
                        f"max_position_pct limit of {cfg.risk.max_position_pct}%"
                    )

            # Pre-trade validation: max sector exposure check
            if total_value > 0:
                sector = ""
                if existing:
                    sector = existing.get("sector", "")
                if sector:
                    sector_value = sum(
                        p["quantity"] * _estimated_price(p)
                        for p in all_positions
                        if p.get("sector", "") == sector
                    )
                    new_sector_value = sector_value + cost
                    sector_pct = (new_sector_value / total_value) * 100
                    if sector_pct > cfg.risk.max_sector_pct:
                        raise ValueError(
                            f"Sector '{sector}' exposure {sector_pct:.1f}% would exceed "
                            f"max_sector_pct limit of {cfg.risk.max_sector_pct}%"
                        )

            # Warn if buy would leave cash below 5% of total portfolio value
            remaining_cash = cash - cost
            if total_value > 0 and remaining_cash < 0.05 * total_value:
                logger.warning(
                    "Low cash reserve after BUY: $%.2f remaining (%.1f%% of $%.2f portfolio)",
                    remaining_cash,
                    (remaining_cash / total_value) * 100,
                    total_value,
                )

            # Execute BUY in a SQLite transaction
            new_qty = old_qty + quantity
            new_avg = self._calculate_avg_cost(old_qty, old_avg, quantity, price)
            try:
                self.db.conn.execute("BEGIN")
                self.db.conn.execute(
                    """INSERT INTO cache (key, value, expires_at)
                       VALUES ('cash_balance', ?, '9999-12-31')
                       ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                    (_json.dumps(cash - cost),),
                )
                self.db.conn.execute(
                    """INSERT INTO positions (ticker, quantity, avg_cost, sector, updated_at)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(ticker) DO UPDATE SET
                         quantity = excluded.quantity,
                         avg_cost = excluded.avg_cost,
                         sector = CASE WHEN excluded.sector = '' THEN sector ELSE excluded.sector END,
                         updated_at = excluded.updated_at""",
                    (ticker, new_qty, new_avg, "", datetime.now().isoformat()),
                )
                self.db.conn.execute("COMMIT")
            except Exception:
                self.db.conn.execute("ROLLBACK")
                raise

        elif action == Action.SELL:
            new_qty = max(old_qty - quantity, 0.0)
            new_avg = old_avg  # avg cost unchanged on sells
            proceeds = quantity * price
            cash = self.db.get_cash()
            # Execute SELL in a SQLite transaction
            try:
                self.db.conn.execute("BEGIN")
                self.db.conn.execute(
                    """INSERT INTO cache (key, value, expires_at)
                       VALUES ('cash_balance', ?, '9999-12-31')
                       ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                    (_json.dumps(cash + proceeds),),
                )
                self.db.conn.execute(
                    """INSERT INTO positions (ticker, quantity, avg_cost, sector, updated_at)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(ticker) DO UPDATE SET
                         quantity = excluded.quantity,
                         avg_cost = excluded.avg_cost,
                         sector = CASE WHEN excluded.sector = '' THEN sector ELSE excluded.sector END,
                         updated_at = excluded.updated_at""",
                    (ticker, new_qty, new_avg, "", datetime.now().isoformat()),
                )
                self.db.conn.execute("COMMIT")
            except Exception:
                self.db.conn.execute("ROLLBACK")
                raise

        else:
            # HOLD: just log, no changes
            pass

        # Capture after-state
        after_state = self._snapshot_dict()

        record = ActionRecord(
            ticker=ticker,
            action=action,
            quantity=quantity,
            price=price,
            timestamp=datetime.now(),
            source=source,
            reasoning=reasoning,
            portfolio_before=before_state,
            portfolio_after=after_state,
        )
        record.id = self.db.log_action(record)
        return record

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _calculate_avg_cost(
        old_qty: float, old_avg: float, new_qty: float, new_price: float
    ) -> float:
        """Weighted average cost basis when buying more shares."""
        total_qty = old_qty + new_qty
        if total_qty <= 0:
            return 0.0
        return (old_qty * old_avg + new_qty * new_price) / total_qty

    @staticmethod
    def _fetch_prices(tickers: list[str]) -> dict[str, float]:
        """Fetch current prices for a list of tickers using yfinance."""
        prices: dict[str, float] = {}
        if not tickers:
            return prices
        try:
            # yfinance handles single or multiple tickers
            data = yf.Tickers(" ".join(tickers))
            for ticker in tickers:
                try:
                    info = data.tickers[ticker].fast_info
                    prices[ticker] = float(info.get("lastPrice", 0.0) or info.get("last_price", 0.0))
                except Exception:
                    logger.warning("Could not fetch price for %s", ticker)
                    prices[ticker] = 0.0
        except Exception:
            logger.warning("yfinance batch fetch failed, falling back to individual")
            for ticker in tickers:
                try:
                    t = yf.Ticker(ticker)
                    prices[ticker] = float(t.fast_info.get("lastPrice", 0.0) or t.fast_info.get("last_price", 0.0))
                except Exception:
                    logger.warning("Could not fetch price for %s", ticker)
                    prices[ticker] = 0.0
        return prices

    def _snapshot_dict(self) -> dict:
        """Lightweight snapshot of positions + cash for action logging."""
        positions = self.db.get_all_positions()
        return {
            "positions": {
                p["ticker"]: {"quantity": p["quantity"], "avg_cost": p["avg_cost"]}
                for p in positions
            },
            "cash": self.db.get_cash(),
        }
