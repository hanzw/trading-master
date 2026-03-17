"""Stop-loss management sub-app: show, set, check, auto."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..output.report import _pnl_style
from ..portfolio.tracker import PortfolioTracker

console = Console()

stop_loss_app = typer.Typer(help="Stop-loss management commands")


@stop_loss_app.command("show")
def stop_loss_show() -> None:
    """Show all current stop-losses and their status."""
    from ..portfolio.stop_loss import StopLossMonitor

    tracker = PortfolioTracker()
    monitor = StopLossMonitor()

    with console.status("[bold cyan]Fetching portfolio data...", spinner="dots"):
        state = tracker.get_state()

    if not state.positions:
        console.print("[yellow]No positions in portfolio.[/yellow]")
        raise typer.Exit()

    table = Table(title="Stop-Loss Overview", show_header=True, header_style="bold magenta")
    table.add_column("Ticker", style="cyan", min_width=8)
    table.add_column("Avg Cost", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Stop Price", justify="right")
    table.add_column("Distance", justify="right")
    table.add_column("Status", min_width=10)

    for ticker in sorted(state.positions.keys()):
        pos = state.positions[ticker]
        stop = monitor.get_stop_loss(ticker)

        if stop is not None:
            distance_pct = ((pos.current_price - stop) / pos.current_price * 100) if pos.current_price > 0 else 0
            triggered = pos.current_price <= stop if pos.current_price > 0 else False
            status = Text("TRIGGERED", style="bold red") if triggered else Text("OK", style="green")
            dist_style = "red" if distance_pct < 3 else ("yellow" if distance_pct < 5 else "green")
            table.add_row(
                ticker,
                f"${pos.avg_cost:,.2f}",
                f"${pos.current_price:,.2f}",
                f"${stop:,.2f}",
                Text(f"{distance_pct:.1f}%", style=dist_style),
                status,
            )
        else:
            table.add_row(
                ticker,
                f"${pos.avg_cost:,.2f}",
                f"${pos.current_price:,.2f}",
                Text("NOT SET", style="dim"),
                "-",
                Text("NO STOP", style="dim yellow"),
            )

    console.print(table)
    console.print()


@stop_loss_app.command("set")
def stop_loss_set(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
    price: float = typer.Argument(..., help="Stop-loss price"),
) -> None:
    """Manually set a stop-loss price for a position."""
    from ..portfolio.stop_loss import StopLossMonitor

    ticker = ticker.upper()
    monitor = StopLossMonitor()
    monitor.set_stop_loss(ticker, price)
    console.print(f"[green]Stop-loss for {ticker} set at ${price:,.2f}.[/green]")


@stop_loss_app.command("check")
def stop_loss_check() -> None:
    """Check all stops against live prices and show triggered ones."""
    from ..portfolio.stop_loss import StopLossMonitor

    monitor = StopLossMonitor()

    with console.status("[bold cyan]Checking stop-losses against live prices...", spinner="dots"):
        results = monitor.check_all()

    if not results:
        console.print("[dim]No stop-losses are set.[/dim]")
        raise typer.Exit()

    triggered = [r for r in results if r["triggered"]]
    safe = [r for r in results if not r["triggered"]]

    if triggered:
        table = Table(title="TRIGGERED Stop-Losses", show_header=True, header_style="bold red")
        table.add_column("Ticker", style="cyan")
        table.add_column("Current Price", justify="right")
        table.add_column("Stop Price", justify="right")
        table.add_column("P&L %", justify="right")

        for r in triggered:
            pnl_style = _pnl_style(r["loss_pct"])
            table.add_row(
                r["ticker"],
                f"${r['current_price']:,.2f}",
                f"${r['stop_price']:,.2f}",
                Text(f"{r['loss_pct']:+.2f}%", style=pnl_style),
            )
        console.print(table)

        console.print(
            Panel(
                f"[bold red]{len(triggered)} stop-loss(es) triggered![/bold red] "
                "Consider selling these positions.",
                border_style="red",
            )
        )

    if safe:
        table = Table(title="Active Stop-Losses (OK)", show_header=True, header_style="bold green")
        table.add_column("Ticker", style="cyan")
        table.add_column("Current Price", justify="right")
        table.add_column("Stop Price", justify="right")
        table.add_column("P&L %", justify="right")

        for r in safe:
            pnl_style = _pnl_style(r["loss_pct"])
            table.add_row(
                r["ticker"],
                f"${r['current_price']:,.2f}",
                f"${r['stop_price']:,.2f}",
                Text(f"{r['loss_pct']:+.2f}%", style=pnl_style),
            )
        console.print(table)

    console.print()


@stop_loss_app.command("auto")
def stop_loss_auto() -> None:
    """Auto-set stops for positions that don't have one (default 8% below avg cost)."""
    from ..portfolio.stop_loss import StopLossMonitor

    tracker = PortfolioTracker()
    monitor = StopLossMonitor()

    with console.status("[bold cyan]Setting automatic stop-losses...", spinner="dots"):
        state = tracker.get_state()

        before = {t: monitor.get_stop_loss(t) for t in state.positions}
        monitor.auto_set_stops(tracker)
        after = {t: monitor.get_stop_loss(t) for t in state.positions}

    newly_set = {t: after[t] for t in after if before.get(t) is None and after[t] is not None}

    if newly_set:
        table = Table(title="Auto-Set Stop-Losses", show_header=True, header_style="bold magenta")
        table.add_column("Ticker", style="cyan")
        table.add_column("Avg Cost", justify="right")
        table.add_column("Stop Price", justify="right")
        table.add_column("Distance", justify="right")

        for ticker, stop in sorted(newly_set.items()):
            pos = state.positions.get(ticker)
            avg_cost = pos.avg_cost if pos else 0
            dist = ((avg_cost - stop) / avg_cost * 100) if avg_cost > 0 else 0
            table.add_row(
                ticker,
                f"${avg_cost:,.2f}",
                f"${stop:,.2f}",
                f"{dist:.1f}%",
            )
        console.print(table)
        console.print(f"[green]{len(newly_set)} stop-loss(es) auto-set.[/green]")
    else:
        console.print("[dim]All positions already have stop-losses set.[/dim]")
    console.print()
