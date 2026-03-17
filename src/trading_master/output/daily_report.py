"""Daily portfolio health report generator for cron/scheduled use."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from ..alerts import run_all_alerts, format_alert_report
from ..db import get_db
from ..portfolio.tracker import PortfolioTracker
from ..portfolio.allocation import (
    compute_current_allocation,
    compute_drift,
    PRESET_MODELS,
    regime_adjusted_model,
    detect_overlaps,
)
from ..data.macro import fetch_macro_data

logger = logging.getLogger(__name__)


def generate_daily_report(output_dir: Path | None = None) -> Path:
    """Generate a comprehensive daily portfolio health report as a text file.

    Sections:
    1. Date + Portfolio Value + Cash
    2. Macro regime + key indicators (VIX, yields)
    3. Alert summary (watchlist + stops + circuit breaker)
    4. Allocation drift (vs regime-adjusted growth model)
    5. Top concerns (any triggered alerts, stops near trigger, overlaps)
    6. Action items (if any alerts fired, specific recommended actions)

    Saves to output_dir/daily_report_YYYY-MM-DD.txt
    Returns the file path.
    """
    report = _build_report_text()

    if output_dir is None:
        output_dir = Path("data/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    filepath = output_dir / f"daily_report_{date_str}.txt"
    filepath.write_text(report, encoding="utf-8")
    logger.info("Daily report saved to %s", filepath)
    return filepath


def generate_cron_report() -> str:
    """Generate the report as plain text (no file save), suitable for piping."""
    return _build_report_text()


def _build_report_text() -> str:
    """Build the full report as a plain-text string."""
    lines: list[str] = []
    now = datetime.now()

    # ── Section 1: Date + Portfolio Value + Cash ──────────────────
    db = get_db()
    tracker = PortfolioTracker(db=db)
    state = tracker.get_state()

    lines.append("=" * 60)
    lines.append(f"  DAILY PORTFOLIO HEALTH REPORT")
    lines.append(f"  {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Portfolio Value: ${state.total_value:,.2f}")
    lines.append(f"Cash:            ${state.cash:,.2f}")
    cash_pct = (state.cash / state.total_value * 100) if state.total_value > 0 else 0
    lines.append(f"Cash %:          {cash_pct:.1f}%")
    lines.append(f"Positions:       {len(state.positions)}")
    lines.append("")

    # ── Section 2: Macro regime + key indicators ──────────────────
    try:
        macro = fetch_macro_data()
        lines.append("-" * 60)
        lines.append("MACRO ENVIRONMENT")
        lines.append("-" * 60)
        lines.append(f"Regime:           {macro.regime.value.upper()}")
        if macro.vix is not None:
            lines.append(f"VIX:              {macro.vix:.1f}")
        if macro.us_10yr_yield is not None:
            lines.append(f"US 10Y Yield:     {macro.us_10yr_yield:.2f}%")
        if macro.us_2yr_yield is not None:
            lines.append(f"US 2Y Yield:      {macro.us_2yr_yield:.2f}%")
        if macro.us_10yr_yield is not None and macro.us_2yr_yield is not None:
            spread = macro.us_10yr_yield - macro.us_2yr_yield
            lines.append(f"Yield Spread:     {spread:+.2f}%")
            if spread < 0:
                lines.append("                  ** INVERTED YIELD CURVE **")
        if macro.sp500_above_sma200 is not None:
            status = "ABOVE" if macro.sp500_above_sma200 else "BELOW"
            lines.append(f"S&P 500 vs SMA200: {status}")
        regime_str = macro.regime.value
    except Exception as e:
        lines.append(f"Macro data unavailable: {e}")
        regime_str = "sideways"
    lines.append("")

    # ── Section 3: Alert summary ──────────────────────────────────
    try:
        alerts = run_all_alerts()
        lines.append("-" * 60)
        lines.append("ALERT SUMMARY")
        lines.append("-" * 60)
        lines.append(format_alert_report(alerts))
    except Exception as e:
        lines.append(f"Alert check failed: {e}")
        alerts = None
    lines.append("")

    # ── Section 4: Allocation drift ───────────────────────────────
    lines.append("-" * 60)
    lines.append("ALLOCATION DRIFT (vs regime-adjusted growth model)")
    lines.append("-" * 60)
    try:
        base_model = PRESET_MODELS["growth"]
        adjusted = regime_adjusted_model(base_model, regime_str)
        current_alloc = compute_current_allocation(state)
        drift_targets = compute_drift(current_alloc, adjusted)

        for t in drift_targets:
            if t.current_pct > 0.1 or abs(t.drift_pct) > 0.5:
                label = t.asset_class.value.replace("_", " ").title()
                drift_marker = ""
                if abs(t.drift_pct) > 5:
                    drift_marker = " ** NEEDS ATTENTION **"
                lines.append(
                    f"  {label:<25s}  current {t.current_pct:5.1f}%  "
                    f"target {t.target_pct:5.1f}%  drift {t.drift_pct:+5.1f}%{drift_marker}"
                )
    except Exception as e:
        lines.append(f"  Allocation check failed: {e}")
    lines.append("")

    # ── Section 5: Top concerns ───────────────────────────────────
    lines.append("-" * 60)
    lines.append("TOP CONCERNS")
    lines.append("-" * 60)
    concerns: list[str] = []

    if alerts and alerts.get("alert_count", 0) > 0:
        concerns.append(f"Active alerts: {alerts['summary']}")

    # Check for stops near trigger
    try:
        from ..portfolio.stop_loss import StopLossMonitor
        slm = StopLossMonitor(db=db)
        stop_results = slm.check_all()
        for s in stop_results:
            if not s.get("triggered") and s.get("loss_pct") is not None:
                if s["loss_pct"] < -5:  # within 5% of stop
                    concerns.append(
                        f"Stop near trigger: {s['ticker']} at {s['loss_pct']:+.1f}%"
                    )
    except Exception:
        pass

    # Check for overlaps
    try:
        position_values = {
            ticker: pos.market_value
            for ticker, pos in state.positions.items()
        }
        overlaps = detect_overlaps(position_values)
        for o in overlaps:
            concerns.append(
                f"Overlap: {', '.join(o['tickers'])} ({o['group']}) "
                f"combined {o['combined_pct']:.1f}% — {o['suggestion']}"
            )
    except Exception:
        pass

    if concerns:
        for c in concerns:
            lines.append(f"  - {c}")
    else:
        lines.append("  No concerns. Portfolio is healthy.")
    lines.append("")

    # ── Section 6: Action items ───────────────────────────────────
    lines.append("-" * 60)
    lines.append("ACTION ITEMS")
    lines.append("-" * 60)
    actions: list[str] = []

    if alerts:
        if alerts.get("watchlist_alerts"):
            for a in alerts["watchlist_alerts"]:
                actions.append(f"Review watchlist alert: {a.get('message', str(a))}")
        if alerts.get("stop_loss_alerts"):
            for s in alerts["stop_loss_alerts"]:
                actions.append(
                    f"SELL {s['ticker']}: stop-loss triggered at ${s['current_price']:.2f} "
                    f"(stop ${s['stop_price']:.2f})"
                )
        if alerts.get("circuit_breaker", {}).get("triggered"):
            actions.append(
                "HALT ALL BUYING: circuit breaker triggered — "
                "review portfolio drawdown before any new positions"
            )
        if alerts.get("regime_changed"):
            actions.append(
                f"Regime changed to {alerts['macro_regime'].upper()} — "
                f"review allocation targets and rebalance if needed"
            )

    if actions:
        for i, a in enumerate(actions, 1):
            lines.append(f"  {i}. {a}")
    else:
        lines.append("  No action required.")
    lines.append("")

    lines.append("=" * 60)
    lines.append("  END OF REPORT")
    lines.append("=" * 60)

    return "\n".join(lines)
