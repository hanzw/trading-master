"""CLI commands for quantitative analysis (Monte Carlo, DCF, Black-Litterman)."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

import numpy as np

quant_app = typer.Typer(name="quant", help="Quantitative analysis tools")
console = Console()


@quant_app.command("monte-carlo")
def monte_carlo_cmd(
    simulations: int = typer.Option(10_000, "--sims", "-n", help="Number of simulations"),
    horizon: int = typer.Option(252, "--horizon", "-d", help="Horizon in trading days"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
) -> None:
    """Run Monte Carlo simulation on the current portfolio."""
    from ..quant.monte_carlo import simulate_portfolio_paths

    # Demo portfolio: 60/40 equities/bonds
    weights = np.array([0.6, 0.4])
    expected_returns = np.array([0.10, 0.04])
    cov_matrix = np.array([[0.04, 0.005], [0.005, 0.0025]])

    console.print("[bold]Running Monte Carlo simulation...[/bold]")
    result = simulate_portfolio_paths(
        weights=weights,
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        horizon_days=horizon,
        n_simulations=simulations,
        seed=seed,
    )

    table = Table(title="Monte Carlo Results (60/40 Portfolio)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Expected Value", f"${result['expected_value']:,.0f}")
    table.add_row("Worst Case (5%)", f"${result['worst_case_5pct']:,.0f}")
    table.add_row("Best Case (95%)", f"${result['best_case_95pct']:,.0f}")
    table.add_row("P(Loss)", f"{result['prob_loss']:.1%}")
    table.add_row("P(Gain > 20%)", f"{result['prob_gain_20pct']:.1%}")
    table.add_row("Median Max Drawdown", f"{result['max_drawdown_median']:.1%}")

    for pct, val in result["percentiles"].items():
        table.add_row(f"  {pct}th pctile", f"${val:,.0f}")

    console.print(table)


@quant_app.command("stress-test")
def stress_test_cmd() -> None:
    """Run historical stress test scenarios on the portfolio."""
    from ..quant.monte_carlo import stress_test_scenarios

    weights = np.array([0.6, 0.4])
    cov_matrix = np.array([[0.04, 0.005], [0.005, 0.0025]])
    portfolio_value = 1_000_000.0
    asset_classes = ["equities", "bonds"]

    results = stress_test_scenarios(
        weights=weights,
        cov_matrix=cov_matrix,
        portfolio_value=portfolio_value,
        asset_classes=asset_classes,
    )

    table = Table(title="Stress Test Scenarios ($1M Portfolio)")
    table.add_column("Scenario", style="cyan")
    table.add_column("Return %", style="red", justify="right")
    table.add_column("Dollar Impact", style="red", justify="right")

    for r in results:
        table.add_row(
            r["scenario"],
            f"{r['return_pct']:.1%}",
            f"${r['dollar_impact']:+,.0f}",
        )

    console.print(table)


@quant_app.command("dcf")
def dcf_cmd(
    ticker: str = typer.Argument(..., help="Stock ticker symbol"),
) -> None:
    """Run DCF valuation for a ticker using yfinance data."""
    from ..quant.dcf import auto_dcf

    console.print(f"[bold]Running DCF valuation for {ticker.upper()}...[/bold]")
    try:
        result = auto_dcf(ticker)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    table = Table(title=f"DCF Valuation — {result['ticker']}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Current Price", f"${result['current_price']:,.2f}")
    table.add_row("Intrinsic Value", f"${result['intrinsic_value']:,.2f}")
    table.add_row("With 25% Margin of Safety", f"${result['with_margin_of_safety']:,.2f}")
    if result["upside_pct"] is not None:
        table.add_row("Upside %", f"{result['upside_pct']:.1%}")
    table.add_row("Est. 5yr Growth", f"{result['growth_rate_5yr']:.1%}")
    table.add_row("Verdict", f"[bold]{result['verdict'].upper()}[/bold]")

    console.print(table)
