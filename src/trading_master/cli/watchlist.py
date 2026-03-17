"""Watchlist and alerts CLI sub-app."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

watchlist_app = typer.Typer(help="Watchlist management commands")


@watchlist_app.command("add")
def watchlist_add(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
    target_price: Optional[float] = typer.Option(None, "--target-price", "-tp", help="Target entry price"),
    max_pe: Optional[float] = typer.Option(None, "--max-pe", help="Max PE ratio to trigger alert"),
    min_yield: Optional[float] = typer.Option(None, "--min-yield", help="Min dividend yield to trigger alert"),
    thesis: str = typer.Option("", "--thesis", "-t", help="Investment thesis"),
) -> None:
    """Add a ticker to the watchlist with optional entry criteria."""
    from ..portfolio.watchlist import WatchlistManager

    wm = WatchlistManager()
    wm.add(ticker.upper(), target_price=target_price, thesis=thesis, max_pe=max_pe, min_yield=min_yield)
    console.print(f"[green]Added {ticker.upper()} to watchlist.[/green]")
    if target_price:
        console.print(f"  Target price: ${target_price:,.2f}")
    if max_pe:
        console.print(f"  Max PE: {max_pe:.1f}")
    if min_yield:
        console.print(f"  Min yield: {min_yield:.2%}")
    if thesis:
        console.print(f"  Thesis: {thesis}")


@watchlist_app.command("remove")
def watchlist_remove(
    ticker: str = typer.Argument(..., help="Ticker symbol to remove"),
) -> None:
    """Remove a ticker from the watchlist."""
    from ..portfolio.watchlist import WatchlistManager

    wm = WatchlistManager()
    wm.remove(ticker.upper())
    console.print(f"[yellow]Removed {ticker.upper()} from watchlist.[/yellow]")


@watchlist_app.command("show")
def watchlist_show() -> None:
    """Show all watchlist items with current prices."""
    from ..portfolio.watchlist import WatchlistManager
    from ..data.market import fetch_market_data

    wm = WatchlistManager()
    items = wm.get_all()

    if not items:
        console.print("[dim]Watchlist is empty.[/dim]")
        raise typer.Exit()

    table = Table(title="Watchlist", show_header=True, header_style="bold magenta")
    table.add_column("Ticker", style="cyan", min_width=8)
    table.add_column("Current Price", justify="right")
    table.add_column("Target Price", justify="right")
    table.add_column("PE Ratio", justify="right")
    table.add_column("Max PE", justify="right")
    table.add_column("Yield", justify="right")
    table.add_column("Min Yield", justify="right")
    table.add_column("Thesis", max_width=30)
    table.add_column("Added", style="dim")

    for item in items:
        ticker = item["ticker"]
        try:
            market = fetch_market_data(ticker)
            price_str = f"${market.current_price:,.2f}" if market.current_price else "-"
            pe_str = f"{market.pe_ratio:.1f}" if market.pe_ratio else "-"
            yield_str = f"{market.dividend_yield:.2%}" if market.dividend_yield else "-"

            # Highlight if target hit
            if item["target_price"] and market.current_price and market.current_price <= item["target_price"]:
                price_str = Text(price_str, style="bold green")
        except Exception:
            price_str = Text("N/A", style="dim")
            pe_str = "-"
            yield_str = "-"

        target_str = f"${item['target_price']:,.2f}" if item["target_price"] else "-"
        max_pe_str = f"{item['max_pe']:.1f}" if item["max_pe"] else "-"
        min_yield_str = f"{item['min_yield']:.2%}" if item["min_yield"] else "-"
        added = item["added_date"][:10] if item["added_date"] else "-"

        table.add_row(
            ticker, price_str, target_str, pe_str, max_pe_str,
            yield_str, min_yield_str, item.get("thesis", "")[:30], added,
        )

    console.print(table)

    # Show notes if any
    notes_items = [i for i in items if i.get("notes")]
    if notes_items:
        console.print()
        for item in notes_items:
            console.print(f"  [cyan]{item['ticker']}[/cyan] notes: {item['notes']}")

    console.print()


@watchlist_app.command("check")
def watchlist_check() -> None:
    """Check all watchlist items against entry criteria."""
    from ..portfolio.watchlist import WatchlistManager

    wm = WatchlistManager()

    with console.status("[bold cyan]Checking watchlist alerts...", spinner="dots"):
        alerts = wm.check_alerts()

    if not alerts:
        console.print("[green]No watchlist alerts triggered.[/green]")
        raise typer.Exit()

    table = Table(title="Watchlist Alerts", show_header=True, header_style="bold green")
    table.add_column("Ticker", style="cyan")
    table.add_column("Alert Type", style="bold")
    table.add_column("Message")
    table.add_column("Current", justify="right")
    table.add_column("Target", justify="right")

    for a in alerts:
        table.add_row(
            a["ticker"],
            a["alert_type"],
            a["message"],
            f"{a['current_value']:.4g}",
            f"{a['target_value']:.4g}",
        )

    console.print(table)
    console.print(
        Panel(
            f"[bold green]{len(alerts)} alert(s) triggered![/bold green] "
            "Consider taking action on these tickers.",
            border_style="green",
        )
    )
    console.print()


@watchlist_app.command("notes")
def watchlist_notes(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
    note: str = typer.Argument(..., help="Note text to add"),
) -> None:
    """Add notes to a watchlist item."""
    from ..portfolio.watchlist import WatchlistManager

    wm = WatchlistManager()
    wm.update_notes(ticker.upper(), note)
    console.print(f"[green]Updated notes for {ticker.upper()}.[/green]")


# ── Top-level alerts command ────────────────────────────────────────


def register_alerts_command(app: typer.Typer) -> None:
    """Register the top-level 'alerts' command on the main app."""

    @app.command()
    def alerts() -> None:
        """Run ALL alerts: watchlist + stop-losses + circuit breaker + macro regime."""
        from ..alerts import run_all_alerts, format_alert_report

        with console.status("[bold cyan]Running all alert checks...", spinner="dots"):
            result = run_all_alerts()

        report = format_alert_report(result)

        if result["alert_count"] == 0:
            console.print(Panel(report, title="Alert Report", border_style="green"))
        else:
            console.print(Panel(report, title="Alert Report", border_style="red"))
