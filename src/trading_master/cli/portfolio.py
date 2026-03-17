"""Portfolio management sub-app: show, sync, import, update, history, income, health."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..output.report import print_portfolio
from ..output.history import show_action_history
from ..portfolio.tracker import PortfolioTracker

console = Console()

portfolio_app = typer.Typer(help="Portfolio management commands")


@portfolio_app.command("show")
def portfolio_show() -> None:
    """Show current portfolio with live prices."""
    with console.status("[bold cyan]Fetching prices...", spinner="dots"):
        tracker = PortfolioTracker()
        state = tracker.get_state()
    print_portfolio(state)


@portfolio_app.command("sync")
def portfolio_sync() -> None:
    """Sync positions from Robinhood."""
    from ..portfolio.robinhood import sync_robinhood, is_available

    if not is_available():
        console.print("[red]Robinhood integration is not available. Check credentials.[/red]")
        raise typer.Exit(1)

    tracker = PortfolioTracker()
    with console.status("[bold cyan]Syncing from Robinhood...", spinner="dots"):
        sync_robinhood(tracker)
    console.print("[green]Portfolio synced from Robinhood.[/green]")
    state = tracker.get_state()
    print_portfolio(state)


@portfolio_app.command("import")
def portfolio_import(
    file: Path = typer.Argument(..., help="Path to CSV or JSON file", exists=True),
) -> None:
    """Import trades from a CSV or JSON file."""
    from ..portfolio.csv_import import import_csv

    tracker = PortfolioTracker()
    with console.status("[bold cyan]Importing trades...", spinner="dots"):
        records = import_csv(file, tracker)

    console.print(f"[green]Imported {len(records)} trades from {file.name}.[/green]")
    state = tracker.get_state()
    print_portfolio(state)


@portfolio_app.command("update")
def portfolio_update(
    file: Path = typer.Argument(..., help="Path to text file with portfolio snapshot", exists=True),
) -> None:
    """Import portfolio update from a text file, diff and apply changes."""
    from ..portfolio.update_import import parse_portfolio_text, diff_portfolio, apply_portfolio_update

    text = file.read_text(encoding="utf-8")
    tracker = PortfolioTracker()

    new_positions = parse_portfolio_text(text)
    if not new_positions:
        console.print("[red]No positions found in file.[/red]")
        raise typer.Exit(1)

    cash_entries = [p for p in new_positions if p["ticker"] == "_CASH"]
    cash = cash_entries[0]["value"] if cash_entries else None
    stock_positions = [p for p in new_positions if p["ticker"] != "_CASH"]

    console.print(f"[bold]Parsed {len(stock_positions)} positions from {file.name}[/bold]")
    if cash is not None:
        console.print(f"  Cash: ${cash:,.2f}")
    for p in stock_positions:
        console.print(f"  {p['ticker']:6s}  {p['shares']:>10.4f} shares  ${p['value']:>12,.2f}  @ ${p['price']:.2f}")

    current_positions = {
        p["ticker"]: {"quantity": p["quantity"], "avg_cost": p["avg_cost"]}
        for p in tracker.db.get_all_positions()
    }
    diff = diff_portfolio(current_positions, stock_positions)

    console.print()
    if diff["added"]:
        console.print("[green]New positions:[/green]")
        for item in diff["added"]:
            console.print(f"  + {item['ticker']:6s}  {item['shares']:.4f} shares @ ${item['price']:.2f}")
    if diff["removed"]:
        console.print("[red]Removed positions:[/red]")
        for item in diff["removed"]:
            console.print(f"  - {item['ticker']:6s}  {item['shares']:.4f} shares")
    if diff["changed"]:
        console.print("[yellow]Changed positions:[/yellow]")
        for item in diff["changed"]:
            direction = "+" if item["diff"] > 0 else ""
            console.print(
                f"  ~ {item['ticker']:6s}  {item['old_shares']:.4f} -> {item['new_shares']:.4f} "
                f"({direction}{item['diff']:.4f})"
            )
    if diff["unchanged"]:
        console.print(f"[dim]{len(diff['unchanged'])} positions unchanged[/dim]")

    if not diff["added"] and not diff["removed"] and not diff["changed"] and cash is None:
        console.print("[green]Portfolio is already up to date.[/green]")
        raise typer.Exit()

    typer.confirm("Apply these changes?", abort=True)

    with console.status("[bold cyan]Applying portfolio update...", spinner="dots"):
        records = apply_portfolio_update(tracker, new_positions, cash=cash)

    console.print(f"[green]Applied {len(records)} trades.[/green]")
    state = tracker.get_state()
    print_portfolio(state)


@portfolio_app.command("history")
def portfolio_history(
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Filter by ticker"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max rows"),
) -> None:
    """Show action audit trail."""
    show_action_history(ticker=ticker.upper() if ticker else None, limit=limit)


@portfolio_app.command("income")
def portfolio_income() -> None:
    """Show dividend income breakdown for the portfolio."""
    from ..data.dividends import compute_portfolio_income

    tracker = PortfolioTracker()
    with console.status("[bold cyan]Fetching dividend data...", spinner="dots"):
        state = tracker.get_state()

    if not state.positions:
        console.print("[yellow]No positions in portfolio.[/yellow]")
        raise typer.Exit()

    with console.status("[bold cyan]Computing income projections...", spinner="dots"):
        income = compute_portfolio_income(state.positions)

    table = Table(title="Dividend Income Breakdown", show_header=True, header_style="bold magenta")
    table.add_column("Ticker", style="cyan", min_width=8)
    table.add_column("Annual Income", justify="right")
    table.add_column("Yield", justify="right")
    table.add_column("Yield on Cost", justify="right")
    table.add_column("Sustainability", justify="right")
    table.add_column("Growth (5yr)", justify="right")
    table.add_column("Consec. Yrs", justify="right")

    for ticker_name, info in sorted(income["breakdown"].items()):
        sus = info["sustainability_score"]
        sus_color = "green" if sus >= 70 else ("yellow" if sus >= 40 else "red")

        growth = info.get("growth_rate_5yr")
        growth_str = f"{growth:+.1f}%" if growth is not None else "-"
        growth_color = "green" if (growth or 0) > 0 else ("red" if (growth or 0) < 0 else "dim")

        div_yield = info["dividend_yield"]
        yield_str = f"{div_yield:.2%}" if div_yield else "-"

        table.add_row(
            ticker_name,
            f"${info['annual_income']:,.2f}",
            yield_str,
            f"{info['yield_on_cost']:.2f}%",
            Text(f"{sus:.0f}/100", style=sus_color),
            Text(growth_str, style=growth_color),
            str(info["consecutive_increase_years"]),
        )

    console.print(table)

    summary = Text()
    summary.append("Total Annual Income: ", style="bold")
    summary.append(f"${income['total_annual_income']:,.2f}\n", style="bold green")
    summary.append("Monthly Average: ", style="bold")
    summary.append(f"${income['monthly_average']:,.2f}\n", style="green")
    summary.append("Yield on Cost: ", style="bold")
    summary.append(f"{income['yield_on_cost']:.2f}%\n")
    summary.append("Weighted Avg Growth: ", style="bold")
    growth_val = income["weighted_avg_growth_rate"]
    growth_style = "green" if growth_val > 0 else ("red" if growth_val < 0 else "dim")
    summary.append(f"{growth_val:+.2f}%\n", style=growth_style)
    summary.append("Projected 5yr Income: ", style="bold")
    summary.append(f"${income['projected_5yr_income']:,.2f}", style="cyan")

    console.print(Panel(summary, title="Income Summary", border_style="green", padding=(1, 2)))
    console.print()


@portfolio_app.command("health")
def portfolio_health(
    regime: Optional[str] = typer.Option(None, "--regime", "-r", help="Override regime: bull, sideways, bear, crisis. Auto-detected if omitted."),
) -> None:
    """Comprehensive portfolio health check — circuit breaker, stops, concentration, dividends, regime alerts, overlaps."""
    from ..portfolio.circuit_breaker import DrawdownCircuitBreaker
    from ..portfolio.stop_loss import StopLossMonitor
    from ..data.dividends import compute_portfolio_income
    from ..portfolio.allocation import (
        PRESET_MODELS, regime_allocation_alert, detect_overlaps,
    )
    from ..data.macro import fetch_macro_data

    tracker = PortfolioTracker()
    with console.status("[bold cyan]Running health check...", spinner="dots"):
        state = tracker.get_state()

    if not state.positions:
        console.print("[yellow]No positions in portfolio.[/yellow]")
        raise typer.Exit()

    console.rule("[bold cyan]Portfolio Health Check[/bold cyan]")

    # 1. Circuit Breaker Status
    breaker = DrawdownCircuitBreaker()
    cb_status = breaker.status_with_value(state.total_value)

    cb_text = Text()
    cb_text.append("High Water Mark: ", style="bold")
    cb_text.append(f"${cb_status['hwm']:,.2f}\n")
    cb_text.append("Current Value: ", style="bold")
    cb_text.append(f"${state.total_value:,.2f}\n")
    cb_text.append("Current Drawdown: ", style="bold")
    dd_color = "green" if cb_status["current_dd_pct"] < 5 else ("yellow" if cb_status["current_dd_pct"] < 10 else "red")
    cb_text.append(f"{cb_status['current_dd_pct']:.1f}%", style=dd_color)
    cb_text.append(f" (threshold: {cb_status['threshold']:.0f}%)\n")
    cb_text.append("Status: ", style="bold")
    if cb_status["triggered"]:
        cb_text.append("TRIGGERED — BUY orders blocked", style="bold red")
    else:
        cb_text.append("OK", style="bold green")

    console.print(Panel(cb_text, title="Circuit Breaker", border_style="red" if cb_status["triggered"] else "green"))

    # 2. Cash Reserve
    cash_pct = (state.cash / state.total_value * 100) if state.total_value > 0 else 0
    cash_color = "green" if cash_pct >= 10 else ("yellow" if cash_pct >= 5 else "red")
    cash_text = Text()
    cash_text.append(f"${state.cash:,.2f}", style="bold")
    cash_text.append(f" ({cash_pct:.1f}% of portfolio)", style=cash_color)
    if cash_pct < 5:
        cash_text.append("\n[Warning] Cash reserve below 5%!", style="bold red")

    console.print(Panel(cash_text, title="Cash Reserve", border_style=cash_color))

    # 3. Position Concentration
    conc_table = Table(show_header=True, header_style="bold magenta")
    conc_table.add_column("Ticker", style="cyan")
    conc_table.add_column("Weight", justify="right")
    conc_table.add_column("Value", justify="right")
    conc_table.add_column("Status")

    for ticker_name in sorted(state.positions.keys()):
        pos = state.positions[ticker_name]
        weight = (pos.market_value / state.total_value * 100) if state.total_value > 0 else 0
        if weight > 20:
            status = Text("OVERWEIGHT", style="bold red")
        elif weight > 10:
            status = Text("HEAVY", style="yellow")
        else:
            status = Text("OK", style="green")
        conc_table.add_row(ticker_name, f"{weight:.1f}%", f"${pos.market_value:,.2f}", status)

    console.print(Panel(conc_table, title="Position Concentration", border_style="cyan"))

    # 4. Sector Breakdown
    sectors: dict[str, float] = {}
    for pos in state.positions.values():
        sector = pos.sector or "Unknown"
        sectors[sector] = sectors.get(sector, 0) + pos.market_value

    if sectors:
        sec_table = Table(show_header=True, header_style="bold magenta")
        sec_table.add_column("Sector", style="cyan")
        sec_table.add_column("Value", justify="right")
        sec_table.add_column("Weight", justify="right")

        for sector, value in sorted(sectors.items(), key=lambda x: -x[1]):
            weight = (value / state.total_value * 100) if state.total_value > 0 else 0
            sec_table.add_row(sector, f"${value:,.2f}", f"{weight:.1f}%")

        console.print(Panel(sec_table, title="Sector Breakdown", border_style="blue"))

    # 5. Stop-Loss Summary
    monitor = StopLossMonitor()
    triggered_stops = []
    no_stop_tickers = []
    ok_stops = 0

    for ticker_name in state.positions:
        stop = monitor.get_stop_loss(ticker_name)
        if stop is None:
            no_stop_tickers.append(ticker_name)
        else:
            pos = state.positions[ticker_name]
            if pos.current_price > 0 and pos.current_price <= stop:
                triggered_stops.append(ticker_name)
            else:
                ok_stops += 1

    stop_text = Text()
    stop_text.append(f"Active stops: {ok_stops}  ", style="green")
    if triggered_stops:
        stop_text.append(f"Triggered: {len(triggered_stops)} ({', '.join(triggered_stops)})  ", style="bold red")
    if no_stop_tickers:
        stop_text.append(f"Missing: {len(no_stop_tickers)} ({', '.join(no_stop_tickers)})", style="yellow")

    stop_border = "red" if triggered_stops else ("yellow" if no_stop_tickers else "green")
    console.print(Panel(stop_text, title="Stop-Loss Status", border_style=stop_border))

    # 5b. Trailing Stop Status
    trailing_tickers = []
    for ticker_name in state.positions:
        meta = monitor.get_trailing_stop_meta(ticker_name)
        if meta is not None:
            trailing_tickers.append((ticker_name, meta))

    if trailing_tickers:
        trail_table = Table(show_header=True, header_style="bold magenta")
        trail_table.add_column("Ticker", style="cyan")
        trail_table.add_column("Stop Price", justify="right")
        trail_table.add_column("Highest", justify="right")
        trail_table.add_column("ATR Mult", justify="right")
        trail_table.add_column("ATR", justify="right")

        for ticker_name, meta in trailing_tickers:
            stop = monitor.get_stop_loss(ticker_name)
            trail_table.add_row(
                ticker_name,
                f"${stop:,.2f}" if stop else "-",
                f"${meta['highest_price']:,.2f}",
                f"{meta['atr_multiplier']:.1f}x",
                f"${meta['atr']:,.2f}",
            )

        console.print(Panel(trail_table, title="Trailing Stops (ATR-based)", border_style="cyan"))

    # 6. Regime-Adjusted Allocation Alerts
    if regime is None:
        try:
            with console.status("[bold cyan]Detecting market regime...", spinner="dots"):
                macro_data = fetch_macro_data()
            detected_regime = macro_data.regime.value
        except Exception:
            detected_regime = "sideways"
    else:
        detected_regime = regime.lower()

    console.print(f"\n[bold]Market regime:[/bold] [cyan]{detected_regime.upper()}[/cyan]")
    base_model = PRESET_MODELS.get("balanced")
    if base_model:
        alerts = regime_allocation_alert(state, base_model, detected_regime)
        if alerts:
            alert_text = "\n".join(f"  - {a}" for a in alerts)
            console.print(Panel(
                f"[bold yellow]Regime Alerts ({detected_regime.upper()}):[/bold yellow]\n{alert_text}",
                border_style="yellow",
                title="Regime-Adjusted Allocation",
            ))
        else:
            console.print(Panel(
                f"[bold green]Allocation is within regime-adjusted tolerances.[/bold green]",
                border_style="green",
                title="Regime-Adjusted Allocation",
            ))

    # 7. Overlap Detection
    position_values = {t: p.market_value for t, p in state.positions.items()}
    overlaps = detect_overlaps(position_values)
    if overlaps:
        overlap_text = Text()
        for ov in overlaps:
            tickers_str = ", ".join(ov["tickers"])
            overlap_text.append(f"{ov['group']}: ", style="bold cyan")
            overlap_text.append(f"{tickers_str} ")
            overlap_text.append(f"(combined {ov['combined_pct']:.1f}%)", style="yellow")
            overlap_text.append(f"\n  {ov['suggestion']}\n", style="dim")
        console.print(Panel(overlap_text, title="Overlap Detection", border_style="yellow"))

    # 8. Dividend Income Summary
    try:
        with console.status("[bold cyan]Fetching dividend data...", spinner="dots"):
            income = compute_portfolio_income(state.positions)
        inc_text = Text()
        inc_text.append(f"Annual Income: ${income['total_annual_income']:,.2f}", style="bold green")
        inc_text.append(f"  |  Monthly: ${income['monthly_average']:,.2f}")
        inc_text.append(f"  |  Yield on Cost: {income['yield_on_cost']:.2f}%")
        console.print(Panel(inc_text, title="Dividend Income", border_style="green"))
    except Exception:
        console.print(Panel("[dim]Could not fetch dividend data.[/dim]", title="Dividend Income", border_style="dim"))

    console.print()
