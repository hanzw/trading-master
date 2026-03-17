"""Unified alert system — run ALL alert checks in one call."""

from __future__ import annotations

import json
import logging

from .db import get_db
from .portfolio.watchlist import WatchlistManager
from .portfolio.stop_loss import StopLossMonitor
from .portfolio.circuit_breaker import DrawdownCircuitBreaker
from .portfolio.tracker import PortfolioTracker
from .data.macro import fetch_macro_data

logger = logging.getLogger(__name__)

_LAST_REGIME_CACHE_KEY = "alerts:last_regime"


def run_all_alerts() -> dict:
    """Run ALL alert checks in one call.

    Returns:
        {
            'watchlist_alerts': [...],    # Watchlist targets hit
            'stop_loss_alerts': [...],    # Stop-losses triggered
            'circuit_breaker': {...},     # Drawdown status
            'macro_regime': str,          # Current regime
            'regime_changed': bool,       # True if regime differs from last check
            'summary': str,               # One-line summary
            'alert_count': int,           # Total alerts triggered
        }
    """
    db = get_db()

    # 1. Watchlist alerts
    wm = WatchlistManager(db=db)
    watchlist_alerts = wm.check_alerts()

    # 2. Stop-loss alerts
    slm = StopLossMonitor(db=db)
    stop_results = slm.check_all()
    stop_loss_alerts = [r for r in stop_results if r.get("triggered")]

    # 3. Circuit breaker
    tracker = PortfolioTracker(db=db)
    state = tracker.get_state()
    cb = DrawdownCircuitBreaker(db=db)
    cb.record_portfolio_value(state.total_value)
    cb_status = cb.status_with_value(state.total_value)

    # 4. Macro regime + change detection
    macro = fetch_macro_data()
    current_regime = macro.regime.value

    last_regime = db.cache_get(_LAST_REGIME_CACHE_KEY)
    regime_changed = last_regime is not None and last_regime != current_regime
    # Store current regime for next comparison (very long TTL)
    db.cache_set(_LAST_REGIME_CACHE_KEY, current_regime, ttl_hours=876_000)

    # 5. Count and summarize
    alert_count = len(watchlist_alerts) + len(stop_loss_alerts)
    if cb_status.get("triggered"):
        alert_count += 1
    if regime_changed:
        alert_count += 1

    summary_parts: list[str] = []
    if watchlist_alerts:
        summary_parts.append(f"{len(watchlist_alerts)} watchlist target(s) hit")
    if stop_loss_alerts:
        summary_parts.append(f"{len(stop_loss_alerts)} stop-loss(es) triggered")
    if cb_status.get("triggered"):
        summary_parts.append(
            f"circuit breaker ON (drawdown {cb_status['current_dd_pct']:.1f}%)"
        )
    if regime_changed:
        summary_parts.append(f"regime changed: {last_regime} -> {current_regime}")

    summary = "; ".join(summary_parts) if summary_parts else "All clear — no alerts."

    return {
        "watchlist_alerts": watchlist_alerts,
        "stop_loss_alerts": stop_loss_alerts,
        "circuit_breaker": cb_status,
        "macro_regime": current_regime,
        "regime_changed": regime_changed,
        "summary": summary,
        "alert_count": alert_count,
    }


def format_alert_report(alerts: dict) -> str:
    """Format alerts as a human-readable string for display or file output."""
    lines: list[str] = []

    lines.append(f"=== ALERT REPORT ({alerts['alert_count']} alert(s)) ===")
    lines.append(f"Summary: {alerts['summary']}")
    lines.append("")

    # Watchlist
    wa = alerts["watchlist_alerts"]
    lines.append(f"--- Watchlist Alerts ({len(wa)}) ---")
    if wa:
        for a in wa:
            lines.append(f"  [{a['alert_type']}] {a['message']}")
    else:
        lines.append("  No watchlist alerts.")
    lines.append("")

    # Stop-losses
    sl = alerts["stop_loss_alerts"]
    lines.append(f"--- Stop-Loss Alerts ({len(sl)}) ---")
    if sl:
        for s in sl:
            lines.append(
                f"  {s['ticker']}: ${s['current_price']:.2f} <= "
                f"stop ${s['stop_price']:.2f} (P&L {s['loss_pct']:+.1f}%)"
            )
    else:
        lines.append("  No stop-loss alerts.")
    lines.append("")

    # Circuit breaker
    cb = alerts["circuit_breaker"]
    status = "TRIGGERED" if cb.get("triggered") else "OK"
    lines.append(f"--- Circuit Breaker: {status} ---")
    lines.append(
        f"  HWM: ${cb.get('hwm', 0):,.2f} | "
        f"Drawdown: {cb.get('current_dd_pct', 0):.1f}% | "
        f"Threshold: {cb.get('threshold', 0):.1f}%"
    )
    lines.append("")

    # Macro
    lines.append(f"--- Macro Regime: {alerts['macro_regime'].upper()} ---")
    if alerts["regime_changed"]:
        lines.append("  ** REGIME CHANGED **")
    lines.append("")

    return "\n".join(lines)
