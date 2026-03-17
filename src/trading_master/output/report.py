"""Rich console formatted output for recommendations, portfolio, and actions."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..models import Action, Recommendation, PortfolioState

console = Console()

# ── Color helpers ─────────────────────────────────────────────────────

_ACTION_COLORS = {
    Action.BUY: "bold green",
    Action.SELL: "bold red",
    Action.HOLD: "bold yellow",
}

_SIGNAL_COLORS = {
    "STRONG_BUY": "bold green",
    "BUY": "green",
    "HOLD": "yellow",
    "SELL": "red",
    "STRONG_SELL": "bold red",
}


def _confidence_bar(confidence: float, width: int = 20) -> Text:
    """Render a confidence value (0-100) as a colored progress bar."""
    filled = int(confidence / 100 * width)
    empty = width - filled
    if confidence >= 70:
        color = "green"
    elif confidence >= 40:
        color = "yellow"
    else:
        color = "red"
    bar = Text()
    bar.append("█" * filled, style=color)
    bar.append("░" * empty, style="dim")
    bar.append(f" {confidence:.0f}%", style="bold")
    return bar


def _pnl_style(value: float) -> str:
    if value > 0:
        return "green"
    elif value < 0:
        return "red"
    return "dim"


# ── Recommendation ────────────────────────────────────────────────────

def print_recommendation(rec: Recommendation) -> None:
    """Print a full recommendation with analyst reports, risk, and debate notes."""
    action_style = _ACTION_COLORS.get(rec.action, "bold")

    # Header
    header = Text()
    header.append(f"{rec.ticker}", style="bold cyan")
    header.append("  ")
    header.append(rec.action.value, style=action_style)
    header.append("\n\n")
    header.append("Confidence: ")
    header.append_text(_confidence_bar(rec.confidence))
    header.append(f"\n\n{rec.summary}")

    console.print(Panel(header, title="Recommendation", border_style="cyan", padding=(1, 2)))

    # Analyst reports table
    if rec.analyst_reports:
        table = Table(title="Analyst Reports", show_header=True, header_style="bold magenta")
        table.add_column("Analyst", style="cyan", min_width=12)
        table.add_column("Signal", min_width=10)
        table.add_column("Confidence", min_width=12)
        table.add_column("Key Factors", ratio=1)

        for report in rec.analyst_reports:
            signal_style = _SIGNAL_COLORS.get(report.signal.value, "")
            signal_text = Text(report.signal.value, style=signal_style)
            conf_bar = _confidence_bar(report.confidence, width=10)
            factors = ", ".join(report.key_factors) if report.key_factors else report.summary[:80]
            revised = " (revised)" if report.revised else ""
            table.add_row(
                f"{report.analyst}{revised}",
                signal_text,
                conf_bar,
                factors,
            )

        console.print(table)

    # Risk assessment
    if rec.risk_assessment:
        ra = rec.risk_assessment
        risk_lines = Text()
        risk_color = "red" if ra.risk_score >= 70 else ("yellow" if ra.risk_score >= 40 else "green")
        risk_lines.append(f"Risk Score: {ra.risk_score:.0f}/100", style=risk_color)
        risk_lines.append(f"\nMax Position Size: {ra.max_position_size:.0f} shares")
        if ra.suggested_stop_loss is not None:
            risk_lines.append(f"\nSuggested Stop Loss: ${ra.suggested_stop_loss:.2f}")
        if ra.portfolio_impact:
            risk_lines.append(f"\nPortfolio Impact: {ra.portfolio_impact}")
        status = "APPROVED" if ra.approved else "REJECTED"
        status_style = "green" if ra.approved else "bold red"
        risk_lines.append(f"\nStatus: ")
        risk_lines.append(status, style=status_style)
        if ra.warnings:
            risk_lines.append("\n\nWarnings:", style="bold yellow")
            for w in ra.warnings:
                risk_lines.append(f"\n  - {w}")

        console.print(Panel(risk_lines, title="Risk Assessment", border_style="yellow", padding=(1, 2)))

    # Debate notes
    if rec.debate_notes:
        console.print(Panel(rec.debate_notes, title="Debate Notes", border_style="blue", padding=(1, 2)))

    # Cost info
    if rec.llm_tokens_used or rec.llm_cost_usd:
        cost = Text()
        cost.append(f"Tokens: {rec.llm_tokens_used:,}", style="dim")
        cost.append(f"  |  Cost: ${rec.llm_cost_usd:.4f}", style="dim")
        console.print(cost)

    console.print()


# ── Portfolio ─────────────────────────────────────────────────────────

def print_portfolio(state: PortfolioState) -> None:
    """Print portfolio positions as a rich table with totals."""
    table = Table(title="Portfolio", show_header=True, header_style="bold magenta", show_footer=True)
    table.add_column("Ticker", style="cyan", footer_style="bold")
    table.add_column("Qty", justify="right")
    table.add_column("Avg Cost", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Value", justify="right", footer_style="bold")
    table.add_column("P&L", justify="right")
    table.add_column("P&L%", justify="right")
    table.add_column("Weight", justify="right")

    total_value = 0.0
    total_pnl = 0.0

    for ticker, pos in sorted(state.positions.items()):
        pnl_s = _pnl_style(pos.unrealized_pnl)
        weight = (pos.market_value / state.total_value * 100) if state.total_value > 0 else 0.0
        total_value += pos.market_value
        total_pnl += pos.unrealized_pnl

        table.add_row(
            ticker,
            f"{pos.quantity:,.2f}",
            f"${pos.avg_cost:,.2f}",
            f"${pos.current_price:,.2f}",
            f"${pos.market_value:,.2f}",
            Text(f"${pos.unrealized_pnl:+,.2f}", style=pnl_s),
            Text(f"{pos.pnl_pct:+.1f}%", style=pnl_s),
            f"{weight:.1f}%",
        )

    # Cash row
    cash_weight = (state.cash / state.total_value * 100) if state.total_value > 0 else 0.0
    table.add_row(
        "CASH",
        "-",
        "-",
        "-",
        f"${state.cash:,.2f}",
        "-",
        "-",
        f"{cash_weight:.1f}%",
    )

    # Footer totals
    table.columns[0].footer = "TOTAL"
    table.columns[4].footer = f"${state.total_value:,.2f}"
    pnl_footer = Text(f"${total_pnl:+,.2f}", style=_pnl_style(total_pnl))
    table.columns[5].footer = pnl_footer

    console.print(table)
    console.print(f"  As of {state.timestamp:%Y-%m-%d %H:%M:%S}", style="dim")
    console.print()


# ── Actions ───────────────────────────────────────────────────────────

def print_actions(actions: list[dict]) -> None:
    """Print action audit trail as a rich table."""
    if not actions:
        console.print("[dim]No actions found.[/dim]")
        return

    table = Table(title="Action History", show_header=True, header_style="bold magenta")
    table.add_column("Date", style="dim", min_width=19)
    table.add_column("Ticker", style="cyan")
    table.add_column("Action", min_width=6)
    table.add_column("Qty", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Source")
    table.add_column("Reasoning", ratio=1)

    for a in actions:
        action_str = a.get("action", "")
        if action_str == "BUY":
            action_text = Text("BUY", style="green")
        elif action_str == "SELL":
            action_text = Text("SELL", style="red")
        else:
            action_text = Text(action_str, style="yellow")

        table.add_row(
            a.get("timestamp", "")[:19],
            a.get("ticker", ""),
            action_text,
            f"{a.get('quantity', 0):,.2f}",
            f"${a.get('price', 0):,.2f}",
            a.get("source", ""),
            (a.get("reasoning", "") or "")[:60],
        )

    console.print(table)
    console.print()


# ── Recommendations list ──────────────────────────────────────────────

def print_recommendations_list(recs: list[dict]) -> None:
    """Print a summary table of past recommendations."""
    if not recs:
        console.print("[dim]No recommendations found.[/dim]")
        return

    table = Table(title="Recommendations", show_header=True, header_style="bold magenta")
    table.add_column("Date", style="dim", min_width=19)
    table.add_column("Ticker", style="cyan")
    table.add_column("Action", min_width=6)
    table.add_column("Confidence", min_width=12)
    table.add_column("Summary", ratio=1)

    for r in recs:
        action_str = r.get("action", "")
        if action_str == "BUY":
            action_text = Text("BUY", style="green")
        elif action_str == "SELL":
            action_text = Text("SELL", style="red")
        else:
            action_text = Text(action_str, style="yellow")

        conf = r.get("confidence", 0)
        conf_bar = _confidence_bar(float(conf), width=10)

        table.add_row(
            r.get("timestamp", "")[:19],
            r.get("ticker", ""),
            action_text,
            conf_bar,
            (r.get("summary", "") or "")[:80],
        )

    console.print(table)
    console.print()
