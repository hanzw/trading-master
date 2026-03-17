"""Analyze, review, backtest, and macro commands (top-level)."""

from __future__ import annotations

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..budget import BudgetExceededError, CostBudget
from ..db import get_db
from ..output.report import print_recommendation
from ..output.history import show_history
from ..portfolio.tracker import PortfolioTracker

console = Console()


def register_commands(app: typer.Typer) -> None:
    """Register top-level commands on the main app."""

    @app.command()
    def analyze(
        ticker: Optional[str] = typer.Argument(None, help="Ticker symbol to analyze"),
        portfolio: bool = typer.Option(False, "--portfolio", "-p", help="Analyze all current holdings"),
        no_cache: bool = typer.Option(False, "--no-cache", help="Disable LLM response caching"),
    ) -> None:
        """Run the full AI analysis pipeline on a ticker or the entire portfolio."""
        if no_cache:
            from ..agents.cache import set_caching_enabled
            set_caching_enabled(False)

        if portfolio:
            _analyze_portfolio()
        elif ticker:
            budget = _analyze_single(ticker.upper())
            _print_cost_summary(budget)
        else:
            console.print("[red]Provide a TICKER or use --portfolio.[/red]")
            raise typer.Exit(1)

    @app.command()
    def review(
        ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Filter by ticker"),
        limit: int = typer.Option(20, "--limit", "-n", help="Max rows"),
    ) -> None:
        """Show pending/recent recommendations."""
        show_history(ticker=ticker.upper() if ticker else None, limit=limit)

    @app.command()
    def macro() -> None:
        """Show current macroeconomic indicators: VIX, yields, regime."""
        from ..data.macro import fetch_macro_data
        from ..models import MarketRegime

        with console.status("[bold cyan]Fetching macro data...", spinner="dots"):
            m = fetch_macro_data()

        regime_colors = {
            MarketRegime.BULL: "bold green",
            MarketRegime.BEAR: "bold red",
            MarketRegime.SIDEWAYS: "bold yellow",
            MarketRegime.CRISIS: "bold white on red",
        }
        regime_style = regime_colors.get(m.regime, "bold")

        table = Table(title="Macro Indicators", show_header=True, header_style="bold magenta")
        table.add_column("Indicator", style="cyan", min_width=24)
        table.add_column("Value", justify="right", min_width=18)

        if m.vix is not None:
            vix_color = "green" if m.vix < 15 else ("yellow" if m.vix <= 25 else ("red" if m.vix <= 35 else "bold red"))
            table.add_row("VIX", Text(f"{m.vix:.2f} ({m.vix_regime})", style=vix_color))
        else:
            table.add_row("VIX", Text("N/A", style="dim"))

        if m.us_10yr_yield is not None:
            table.add_row("10-Year Treasury Yield", f"{m.us_10yr_yield:.2f}%")
        else:
            table.add_row("10-Year Treasury Yield", Text("N/A", style="dim"))

        if m.us_2yr_yield is not None:
            table.add_row("13-Week T-Bill Rate", f"{m.us_2yr_yield:.2f}%")
        else:
            table.add_row("13-Week T-Bill Rate", Text("N/A", style="dim"))

        if m.yield_curve_spread is not None:
            spread_color = "red" if m.yield_curve_inverted else "green"
            spread_label = " (INVERTED)" if m.yield_curve_inverted else ""
            table.add_row(
                "Yield Curve Spread",
                Text(f"{m.yield_curve_spread:.2f}%{spread_label}", style=spread_color),
            )
        else:
            table.add_row("Yield Curve Spread", Text("N/A", style="dim"))

        if m.sp500_price is not None:
            above_below = "above" if m.sp500_above_sma200 else "BELOW"
            sp_color = "green" if m.sp500_above_sma200 else "red"
            sma_str = f"  (SMA200: {m.sp500_sma200:.2f})" if m.sp500_sma200 else ""
            table.add_row(
                "S&P 500",
                Text(f"{m.sp500_price:.2f} {above_below} SMA200{sma_str}", style=sp_color),
            )
        else:
            table.add_row("S&P 500", Text("N/A", style="dim"))

        console.print(table)

        regime_text = Text()
        regime_text.append(f"  {m.regime.value.upper()}  ", style=regime_style)
        border = "green" if m.regime == MarketRegime.BULL else (
            "red" if m.regime in (MarketRegime.BEAR, MarketRegime.CRISIS) else "yellow"
        )
        console.print(Panel(regime_text, title="Market Regime", border_style=border))

        if m.regime_signals:
            signals_text = "\n".join(f"  - {s}" for s in m.regime_signals)
            console.print(Panel(signals_text, title="Regime Signals", border_style="dim"))

        if m.regime == MarketRegime.CRISIS:
            console.print(
                Panel(
                    "[bold red]CRISIS regime detected![/bold red] "
                    "Consider reducing equity exposure and increasing cash/treasuries.",
                    border_style="red",
                )
            )

        console.print()


# ── Internal helpers ─────────────────────────────────────────────────

def _make_budget() -> CostBudget:
    """Create a CostBudget from the app config."""
    from ..config import get_config
    cfg = get_config().budget
    return CostBudget(
        max_cost_usd=cfg.max_cost_per_run,
        warn_cost_usd=cfg.warn_cost,
        max_tokens=cfg.max_tokens_per_run,
    )


def _print_cost_summary(budget: CostBudget) -> None:
    """Display cost summary after analysis."""
    s = budget.summary()
    console.print(
        f"\n[dim]Cost summary: ${s['cost']:.4f} spent "
        f"({s['tokens']:,} tokens, {s['calls']} LLM calls, "
        f"${s['remaining']:.2f} remaining)[/dim]"
    )


def _analyze_single(ticker: str, budget: CostBudget | None = None) -> CostBudget:
    """Run analysis for a single ticker with progress spinners."""
    from ..agents.graph import run_analysis

    if budget is None:
        budget = _make_budget()

    tracker = PortfolioTracker()
    db = get_db()

    with console.status("[bold cyan]Collecting data...", spinner="dots"):
        state = tracker.get_state()

    with console.status("[bold cyan]Running analysts... Debate round... Risk assessment... Synthesizing recommendation...", spinner="dots"):
        rec = asyncio.run(run_analysis(ticker, state))

    budget.record(rec.llm_tokens_used, rec.llm_cost_usd)

    print_recommendation(rec)
    db.save_recommendation(rec)
    console.print(f"[dim]Recommendation saved to database.[/dim]")
    return budget


def _analyze_portfolio() -> None:
    """Analyze every current holding."""
    from ..config import get_config

    tracker = PortfolioTracker()
    state = tracker.get_state()

    if not state.positions:
        console.print("[yellow]No positions in portfolio.[/yellow]")
        raise typer.Exit()

    budget = _make_budget()
    n_tickers = len(state.positions)
    model = get_config().llm.model
    estimated = budget.estimate_run_cost(n_tickers, model)

    console.print(
        f"[bold]Analyzing {n_tickers} holdings "
        f"(estimated cost: ${estimated:.4f})[/bold]"
    )

    if estimated > budget.warn_cost_usd:
        typer.confirm(
            f"Estimated cost ${estimated:.4f} exceeds warning threshold "
            f"${budget.warn_cost_usd:.2f}. Continue?",
            abort=True,
        )

    console.print()
    for ticker in sorted(state.positions):
        console.rule(f"[cyan]{ticker}[/cyan]")
        try:
            budget = _analyze_single(ticker, budget=budget)
        except BudgetExceededError as exc:
            console.print(f"[red bold]Budget exceeded: {exc}[/red bold]")
            break

    _print_cost_summary(budget)
