"""Drawdown circuit breaker — blocks BUY recommendations when portfolio drawdown exceeds threshold."""

from __future__ import annotations

import logging
from datetime import datetime

from ..db import get_db
from ..models import Action

logger = logging.getLogger(__name__)

# TTL for high-water mark: ~100 years in hours
_HWM_TTL_HOURS = 876_000


class DrawdownCircuitBreaker:
    """Monitor portfolio drawdown and block BUY actions when it exceeds the configured threshold."""

    def __init__(self, max_drawdown_pct: float = 15.0, db=None):
        self.max_drawdown_pct = max_drawdown_pct
        self.db = db or get_db()

    def record_portfolio_value(self, total_value: float) -> None:
        """Store current portfolio value.  Update high-water mark if new peak."""
        current_hwm = self.get_high_water_mark()
        if total_value > current_hwm:
            self.db.cache_set("hwm", total_value, ttl_hours=_HWM_TTL_HOURS)

    def get_high_water_mark(self) -> float:
        """Get the highest recorded portfolio value."""
        hwm = self.db.cache_get("hwm")
        if hwm is not None:
            return float(hwm)
        return 0.0

    def get_current_drawdown(self, current_value: float) -> float:
        """Compute drawdown percentage from high-water mark.

        Returns 0.0 if no high-water mark is recorded or current_value >= hwm.
        """
        hwm = self.get_high_water_mark()
        if hwm <= 0:
            return 0.0
        if current_value >= hwm:
            return 0.0
        return ((hwm - current_value) / hwm) * 100.0

    def is_triggered(self, current_value: float) -> bool:
        """Returns True if drawdown exceeds max_drawdown_pct."""
        return self.get_current_drawdown(current_value) >= self.max_drawdown_pct

    def filter_recommendation(self, action: Action, current_value: float) -> Action:
        """If breaker is triggered and action is BUY, force HOLD.

        SELL and HOLD pass through.  Log a warning when triggered.
        """
        if self.is_triggered(current_value):
            dd = self.get_current_drawdown(current_value)
            hwm = self.get_high_water_mark()
            if action == Action.BUY:
                logger.warning(
                    "Circuit breaker TRIGGERED: drawdown %.1f%% (hwm=$%.2f, current=$%.2f) "
                    "— blocking BUY, forcing HOLD",
                    dd, hwm, current_value,
                )
                return Action.HOLD
            else:
                logger.warning(
                    "Circuit breaker TRIGGERED: drawdown %.1f%% (hwm=$%.2f, current=$%.2f) "
                    "— allowing %s through",
                    dd, hwm, current_value, action.value,
                )
        return action

    def status(self) -> dict:
        """Returns current circuit breaker status."""
        hwm = self.get_high_water_mark()
        return {
            "hwm": hwm,
            "current_dd_pct": 0.0,  # caller should supply current_value
            "triggered": False,
            "threshold": self.max_drawdown_pct,
        }

    def status_with_value(self, current_value: float) -> dict:
        """Returns circuit breaker status for a given portfolio value."""
        hwm = self.get_high_water_mark()
        dd = self.get_current_drawdown(current_value)
        return {
            "hwm": hwm,
            "current_dd_pct": dd,
            "triggered": dd >= self.max_drawdown_pct,
            "threshold": self.max_drawdown_pct,
        }
