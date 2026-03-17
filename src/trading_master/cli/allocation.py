"""Asset allocation sub-app: show, rebalance."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..portfolio.tracker import PortfolioTracker

console = Console()

allocation_app = typer.Typer(help="Asset allocation and rebalancing commands")


@allocation_app.command("show")
def allocation_show(
    model: str = typer.Option("balanced", "--model", "-m", help="Allocation model: balanced, growth, conservative"),
    regime: Optional[str] = typer.Option(None, "--regime", "-r", help="Override regime: bull, sideways, bear, crisis. Auto-detected if omitted."),
) -> None:
    """Show current allocation vs regime-adjusted target model."""
    from ..portfolio.allocation import (
        PRESET_MODELS, compute_current_allocation, compute_drift, needs_rebalance,
        regime_adjusted_model, regime_allocation_alert,
    )
    from ..data.macro import fetch_macro_data

    if model not in PRESET_MODELS:
        console.print(f"[red]Unknown model '{model}'. Choose from: {', '.join(PRESET_MODELS)}[/red]")
        raise typer.Exit(1)

    base_model = PRESET_MODELS[model]
    tracker = PortfolioTracker()

    with console.status("[bold cyan]Fetching portfolio data...", spinner="dots"):
        state = tracker.get_state()

    if state.total_value <= 0:
        console.print("[yellow]Portfolio has no value.[/yellow]")
        raise typer.Exit()

    if regime is None:
        try:
            with console.status("[bold cyan]Detecting market regime...", spinner="dots"):
                macro = fetch_macro_data()
            detected_regime = macro.regime.value
        except Exception:
            detected_regime = "sideways"
    else:
        detected_regime = regime.lower()

    alloc_model = regime_adjusted_model(base_model, detected_regime)

    current = compute_current_allocation(state)
    drift_targets = compute_drift(current, alloc_model)

    console.print(f"\n[bold]Market regime:[/bold] [cyan]{detected_regime.upper()}[/cyan]")

    table = Table(
        title=f"Asset Allocation vs '{model}' Model ({detected_regime.upper()})  (Portfolio: ${state.total_value:,.2f})",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Asset Class", style="cyan", min_width=20)
    table.add_column("Target %", justify="right")
    table.add_column("Current %", justify="right")
    table.add_column("Drift", justify="right")
    table.add_column("Range", justify="right")
    table.add_column("Status", min_width=12)

    for t in drift_targets:
        if abs(t.drift_pct) <= 1.0:
            status = Text("OK", style="green")
        elif t.drift_pct > 0:
            status = Text("OVERWEIGHT", style="bold red")
        else:
            status = Text("UNDERWEIGHT", style="bold yellow")

        if abs(t.drift_pct) > alloc_model.rebalance_threshold_pct:
            drift_style = "bold red"
        elif abs(t.drift_pct) > 2.0:
            drift_style = "yellow"
        else:
            drift_style = "green"

        table.add_row(
            t.asset_class.value.replace("_", " ").title(),
            f"{t.target_pct:.1f}%",
            f"{t.current_pct:.1f}%",
            Text(f"{t.drift_pct:+.1f}%", style=drift_style),
            f"{t.min_pct:.0f}-{t.max_pct:.0f}%",
            status,
        )

    console.print(table)

    alerts = regime_allocation_alert(state, base_model, detected_regime)
    if alerts:
        alert_text = "\n".join(f"  - {a}" for a in alerts)
        console.print(Panel(
            f"[bold yellow]Regime Alerts:[/bold yellow]\n{alert_text}",
            border_style="yellow",
        ))

    rebalance_needed = needs_rebalance(drift_targets, alloc_model.rebalance_threshold_pct)
    if rebalance_needed:
        console.print(
            Panel(
                f"[bold yellow]Rebalancing recommended.[/bold yellow] "
                f"One or more asset classes have drifted beyond the "
                f"{alloc_model.rebalance_threshold_pct:.0f}% threshold.\n"
                f"Run [bold]tm allocation rebalance --model {model}[/bold] for specific suggestions.",
                border_style="yellow",
            )
        )
    else:
        console.print(
            Panel(
                "[bold green]Portfolio is within tolerance.[/bold green] No rebalancing needed.",
                border_style="green",
            )
        )
    console.print()


@allocation_app.command("rebalance")
def allocation_rebalance(
    model: str = typer.Option("balanced", "--model", "-m", help="Allocation model: balanced, growth, conservative"),
) -> None:
    """Show rebalance suggestions to bring portfolio back to target allocation."""
    from ..portfolio.allocation import PRESET_MODELS, suggest_rebalance

    if model not in PRESET_MODELS:
        console.print(f"[red]Unknown model '{model}'. Choose from: {', '.join(PRESET_MODELS)}[/red]")
        raise typer.Exit(1)

    alloc_model = PRESET_MODELS[model]
    tracker = PortfolioTracker()

    with console.status("[bold cyan]Computing rebalance suggestions...", spinner="dots"):
        state = tracker.get_state()

    if state.total_value <= 0:
        console.print("[yellow]Portfolio has no value.[/yellow]")
        raise typer.Exit()

    suggestions = suggest_rebalance(state, alloc_model)

    if not suggestions:
        console.print(
            Panel(
                "[bold green]Portfolio is well balanced.[/bold green] No trades needed.",
                border_style="green",
            )
        )
        console.print()
        return

    table = Table(
        title=f"Rebalance Suggestions — '{model}' Model  (Portfolio: ${state.total_value:,.2f})",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Asset Class", style="cyan", min_width=20)
    table.add_column("Action", min_width=6)
    table.add_column("Amount", justify="right")
    table.add_column("Current %", justify="right")
    table.add_column("Target %", justify="right")
    table.add_column("Drift", justify="right")
    table.add_column("Suggested Tickers", min_width=15)

    for s in suggestions:
        direction_style = "red" if s["direction"] == "SELL" else "green"
        table.add_row(
            s["asset_class"].replace("_", " ").title(),
            Text(s["direction"], style=f"bold {direction_style}"),
            f"${s['amount_usd']:,.2f}",
            f"{s['current_pct']:.1f}%",
            f"{s['target_pct']:.1f}%",
            Text(f"{s['drift_pct']:+.1f}%", style="yellow"),
            ", ".join(s["suggested_tickers"]) if s["suggested_tickers"] else "-",
        )

    console.print(table)

    # Display tax info for SELL suggestions
    sell_suggestions = [s for s in suggestions if s["direction"] == "SELL" and s.get("tax_warning")]
    if sell_suggestions:
        tax_lines = []
        for s in sell_suggestions:
            tax_lines.append(f"  - {s['tax_warning']}")
        console.print(Panel(
            "[bold yellow]Tax Impact Estimates:[/bold yellow]\n" + "\n".join(tax_lines),
            border_style="yellow",
        ))

    console.print(
        Panel(
            "[dim]These are suggestions only. Execute trades manually via [bold]tm action buy/sell[/bold].[/dim]",
            border_style="dim",
        )
    )
    console.print()
