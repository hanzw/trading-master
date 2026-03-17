"""Risk analysis sub-app: dashboard, correlation, sizing."""

from __future__ import annotations

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..portfolio.tracker import PortfolioTracker

console = Console()

risk_app = typer.Typer(help="Risk analysis commands")


@risk_app.command("dashboard")
def risk_dashboard() -> None:
    """Show portfolio risk metrics: Sharpe, Sortino, VaR, CVaR, max drawdown, beta."""
    from ..portfolio.risk_metrics import portfolio_risk_dashboard
    from ..portfolio.correlation import fetch_returns

    tracker = PortfolioTracker()
    with console.status("[bold cyan]Fetching portfolio data...", spinner="dots"):
        state = tracker.get_state()

    if not state.positions:
        console.print("[yellow]No positions in portfolio.[/yellow]")
        raise typer.Exit()

    tickers = sorted(state.positions.keys())
    all_tickers = tickers + ["SPY"]

    with console.status("[bold cyan]Fetching historical returns...", spinner="dots"):
        returns, valid = fetch_returns(all_tickers)

    if returns is None or len(valid) < 1:
        console.print("[red]Could not fetch sufficient historical data.[/red]")
        raise typer.Exit(1)

    benchmark_returns = None
    if "SPY" in valid:
        spy_idx = valid.index("SPY")
        benchmark_returns = returns[:, spy_idx]
        pos_indices = [i for i, t in enumerate(valid) if t != "SPY"]
        valid_positions = [valid[i] for i in pos_indices]
        position_returns = returns[:, pos_indices] if pos_indices else returns
    else:
        valid_positions = valid
        position_returns = returns

    if not valid_positions:
        console.print("[red]No valid position data after filtering.[/red]")
        raise typer.Exit(1)

    total_position_value = sum(
        state.positions[t].market_value
        for t in valid_positions
        if t in state.positions
    )
    if total_position_value <= 0:
        console.print("[yellow]Portfolio has no market value.[/yellow]")
        raise typer.Exit()

    weights = np.array([
        state.positions[t].market_value / total_position_value
        if t in state.positions else 0.0
        for t in valid_positions
    ])

    dashboard = portfolio_risk_dashboard(
        position_returns, weights, benchmark_returns, state.total_value
    )

    table = Table(title="Risk Dashboard", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", min_width=20)
    table.add_column("Value", justify="right", min_width=15)

    sharpe_color = "green" if dashboard["sharpe"] > 1 else ("yellow" if dashboard["sharpe"] > 0 else "red")
    table.add_row("Sharpe Ratio", Text(f"{dashboard['sharpe']:.3f}", style=sharpe_color))

    sortino_color = "green" if dashboard["sortino"] > 1 else ("yellow" if dashboard["sortino"] > 0 else "red")
    table.add_row("Sortino Ratio", Text(f"{dashboard['sortino']:.3f}", style=sortino_color))

    table.add_row("VaR (95%)", Text(f"${dashboard['var_95']:,.2f}", style="yellow"))
    table.add_row("VaR (99%)", Text(f"${dashboard['var_99']:,.2f}", style="red"))
    table.add_row("CVaR (95%)", Text(f"${dashboard['cvar_95']:,.2f}", style="red"))

    dd_color = "green" if dashboard["max_dd"] < 0.1 else ("yellow" if dashboard["max_dd"] < 0.2 else "red")
    table.add_row("Max Drawdown", Text(f"{dashboard['max_dd']:.1%}", style=dd_color))

    table.add_row("Beta", f"{dashboard['beta']:.3f}")
    table.add_row("Calmar Ratio", f"{dashboard['calmar']:.3f}")
    table.add_row("Portfolio Value", f"${state.total_value:,.2f}")

    console.print(table)
    console.print()


@risk_app.command("correlation")
def risk_correlation() -> None:
    """Show correlation matrix and concentration risk for current holdings."""
    from ..portfolio.correlation import rolling_covariance, concentration_risk

    tracker = PortfolioTracker()
    with console.status("[bold cyan]Fetching portfolio data...", spinner="dots"):
        state = tracker.get_state()

    if not state.positions:
        console.print("[yellow]No positions in portfolio.[/yellow]")
        raise typer.Exit()

    tickers = sorted(state.positions.keys())

    with console.status("[bold cyan]Computing correlation matrix...", spinner="dots"):
        cov, corr, valid = rolling_covariance(tickers)

    if corr is None or len(valid) < 1:
        console.print("[red]Could not compute correlation matrix — insufficient data.[/red]")
        raise typer.Exit(1)

    table = Table(title="Correlation Matrix (60-day rolling)", show_header=True, header_style="bold magenta")
    table.add_column("", style="cyan", min_width=6)
    for t in valid:
        table.add_column(t, justify="right", min_width=8)

    for i, t in enumerate(valid):
        row_vals = []
        for j in range(len(valid)):
            val = corr[i, j]
            if i == j:
                style = "dim"
            elif abs(val) > 0.7:
                style = "bold red"
            elif abs(val) > 0.4:
                style = "yellow"
            else:
                style = "green"
            row_vals.append(Text(f"{val:.2f}", style=style))
        table.add_row(t, *row_vals)

    console.print(table)

    if cov is not None:
        cr = concentration_risk(cov)
        cr_table = Table(title="Concentration Risk", show_header=True, header_style="bold magenta")
        cr_table.add_column("Metric", style="cyan", min_width=25)
        cr_table.add_column("Value", justify="right", min_width=15)

        enb_color = "green" if cr["effective_num_bets"] > 3 else ("yellow" if cr["effective_num_bets"] > 1.5 else "red")
        cr_table.add_row("Effective # of Bets", Text(f"{cr['effective_num_bets']:.2f}", style=enb_color))

        dom_color = "red" if cr["top1_dominance"] > 0.6 else ("yellow" if cr["top1_dominance"] > 0.4 else "green")
        cr_table.add_row("Top Eigenvalue Dominance", Text(f"{cr['top1_dominance']:.1%}", style=dom_color))

        conc_text = Text("YES", style="bold red") if cr["concentrated"] else Text("NO", style="green")
        cr_table.add_row("Concentrated?", conc_text)

        console.print(cr_table)

        if cr["concentrated"]:
            console.print(
                Panel(
                    "[bold yellow]Warning:[/bold yellow] Portfolio is highly concentrated. "
                    "The top eigenvalue explains >60% of variance, meaning positions move together. "
                    "Consider diversifying into uncorrelated assets.",
                    border_style="yellow",
                )
            )
    console.print()


@risk_app.command("sizing")
def risk_sizing(
    ticker: str = typer.Argument(..., help="Ticker symbol to size"),
) -> None:
    """Show quantitative position size recommendation for a ticker."""
    from ..data.technical import fetch_technicals
    from ..portfolio.sizing import compute_position_size

    ticker = ticker.upper()
    tracker = PortfolioTracker()

    with console.status("[bold cyan]Fetching portfolio and technical data...", spinner="dots"):
        state = tracker.get_state()
        tech = fetch_technicals(ticker)

    if state.total_value <= 0:
        console.print("[red]Portfolio has no value.[/red]")
        raise typer.Exit(1)

    atr = tech.atr_14 if tech.atr_14 else 0.0
    price = tech.sma_20 if tech.sma_20 else 0.0

    if ticker in state.positions:
        price = state.positions[ticker].current_price or price

    if price <= 0:
        console.print(f"[red]Could not determine price for {ticker}.[/red]")
        raise typer.Exit(1)

    if atr <= 0:
        console.print(f"[yellow]Warning: ATR not available for {ticker}, sizing may be imprecise.[/yellow]")
        atr = price * 0.02

    sizing = compute_position_size(
        price=price,
        atr_14=atr,
        portfolio_value=state.total_value,
    )

    table = Table(title=f"Position Sizing: {ticker}", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", min_width=22)
    table.add_column("Value", justify="right", min_width=15)

    table.add_row("Current Price", f"${price:,.2f}")
    table.add_row("ATR (14-day)", f"${atr:,.2f}")
    table.add_row("Portfolio Value", f"${state.total_value:,.2f}")
    table.add_row("Recommended Shares", f"{sizing['shares']:,}")
    table.add_row("Dollar Amount", f"${sizing['dollar_amount']:,.2f}")
    table.add_row("% of Portfolio", f"{sizing['pct_of_portfolio']:.2f}%")
    table.add_row("Sizing Method", sizing["method"])

    console.print(table)
    console.print()
