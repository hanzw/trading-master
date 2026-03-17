"""Action sub-app: buy, sell commands."""

from __future__ import annotations

import typer
from rich.console import Console

from ..models import Action, ActionSource
from ..portfolio.tracker import PortfolioTracker

console = Console()

action_app = typer.Typer(help="Manual trade logging")


@action_app.command("buy")
def action_buy(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
    quantity: float = typer.Argument(..., help="Number of shares"),
    price: float = typer.Option(..., "--price", "-p", help="Price per share"),
) -> None:
    """Log a manual BUY action."""
    tracker = PortfolioTracker()
    record = tracker.execute_action(
        ticker=ticker.upper(),
        action=Action.BUY,
        quantity=quantity,
        price=price,
        source=ActionSource.MANUAL,
        reasoning="Manual CLI buy",
    )
    console.print(f"[green]BUY {record.quantity} {record.ticker} @ ${record.price:.2f} recorded.[/green]")


@action_app.command("sell")
def action_sell(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
    quantity: float = typer.Argument(..., help="Number of shares"),
    price: float = typer.Option(..., "--price", "-p", help="Price per share"),
) -> None:
    """Log a manual SELL action."""
    tracker = PortfolioTracker()
    record = tracker.execute_action(
        ticker=ticker.upper(),
        action=Action.SELL,
        quantity=quantity,
        price=price,
        source=ActionSource.MANUAL,
        reasoning="Manual CLI sell",
    )
    console.print(f"[red]SELL {record.quantity} {record.ticker} @ ${record.price:.2f} recorded.[/red]")
