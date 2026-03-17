"""Backtest sub-app: accuracy (original) and walk-forward validation."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

backtest_app = typer.Typer(help="Backtest and walk-forward validation commands")


@backtest_app.command("accuracy")
def backtest_accuracy(
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Backtest specific ticker only"),
) -> None:
    """Run full backtest on all past recommendations (or a specific ticker)."""
    from ..portfolio.backtest import backtest_summary

    label = f"for {ticker.upper()}" if ticker else "for all tickers"
    with console.status(f"[bold cyan]Running backtest {label}...", spinner="dots"):
        summary = backtest_summary(ticker=ticker.upper() if ticker else None)

    n = summary["total_recommendations"]
    if n == 0:
        console.print("[yellow]No recommendations found to backtest.[/yellow]")
        raise typer.Exit()

    console.print(f"\n[bold]Backtest Results[/bold] — {n} recommendations\n")

    # Overall Hit Rates
    hr_table = Table(title="Hit Rates by Horizon", show_header=True, header_style="bold magenta")
    hr_table.add_column("Horizon", style="cyan", min_width=10)
    hr_table.add_column("Hit Rate", justify="right")
    hr_table.add_column("Avg Return", justify="right")
    hr_table.add_column("Total", justify="right")
    hr_table.add_column("Correct", justify="right")
    hr_table.add_column("Buy", justify="right")
    hr_table.add_column("Sell", justify="right")

    for label_h, key in [("30 days", "hit_rate_30d"), ("90 days", "hit_rate_90d"), ("180 days", "hit_rate_180d")]:
        hr = summary[key]
        if hr["n_total"] == 0:
            hr_table.add_row(label_h, "-", "-", "0", "-", "-", "-")
            continue
        rate = hr["hit_rate"]
        rate_style = "green" if rate and rate >= 60 else ("yellow" if rate and rate >= 45 else "red")
        ret = hr["avg_return"]
        ret_style = "green" if ret and ret > 0 else "red"
        hr_table.add_row(
            label_h,
            Text(f"{rate:.1f}%", style=rate_style) if rate is not None else "-",
            Text(f"{ret:+.2f}%", style=ret_style) if ret is not None else "-",
            str(hr["n_total"]),
            str(hr["n_correct"]),
            str(hr["n_buy"]),
            str(hr["n_sell"]),
        )

    console.print(hr_table)
    console.print()

    # Per-Agent Accuracy
    agent_acc = summary["agent_accuracy"]
    if agent_acc:
        ag_table = Table(title="Agent Accuracy (90-day)", show_header=True, header_style="bold magenta")
        ag_table.add_column("Agent", style="cyan", min_width=14)
        ag_table.add_column("Hit Rate", justify="right")
        ag_table.add_column("Avg Confidence", justify="right")
        ag_table.add_column("Samples", justify="right")

        for agent, data in sorted(agent_acc.items()):
            rate = data.get("hit_rate")
            rate_style = "green" if rate and rate >= 60 else ("yellow" if rate and rate >= 45 else "red")
            ag_table.add_row(
                agent.capitalize(),
                Text(f"{rate:.1f}%", style=rate_style) if rate is not None else "-",
                f"{data['avg_confidence']:.1f}%" if data.get("avg_confidence") is not None else "-",
                str(data.get("n", 0)),
            )

        console.print(ag_table)
        console.print()

    # Calibration
    cal = summary["calibration"]
    if cal:
        cal_table = Table(title="Calibration (90-day)", show_header=True, header_style="bold magenta")
        cal_table.add_column("Confidence Bucket", style="cyan", min_width=18)
        cal_table.add_column("Count", justify="right")
        cal_table.add_column("Actual Hit Rate", justify="right")
        cal_table.add_column("Avg Confidence", justify="right")
        cal_table.add_column("Calibration", justify="right")

        for row in cal:
            if row["count"] == 0:
                cal_table.add_row(row["bucket"], "0", "-", "-", "-")
                continue

            hit = row["hit_rate"]
            conf = row["avg_confidence"]
            gap = abs(hit - conf) if hit is not None and conf is not None else None
            if gap is not None:
                gap_style = "green" if gap < 10 else ("yellow" if gap < 20 else "red")
                gap_str = Text(f"{gap:+.1f}pp gap", style=gap_style)
            else:
                gap_str = Text("-")

            cal_table.add_row(
                row["bucket"],
                str(row["count"]),
                f"{hit:.1f}%" if hit is not None else "-",
                f"{conf:.1f}%" if conf is not None else "-",
                gap_str,
            )

        console.print(cal_table)

    console.print(f"\n[dim]Last updated: {summary['last_updated']}[/dim]\n")


@backtest_app.command("walk-forward")
def backtest_walk_forward(
    tickers: str = typer.Option(
        "AAPL,MSFT,GOOGL", "--tickers", "-t",
        help="Comma-separated list of tickers to test",
    ),
    windows: int = typer.Option(4, "--windows", "-w", help="Number of test windows"),
    train_days: int = typer.Option(252, "--train-days", help="Training window in trading days"),
    test_days: int = typer.Option(63, "--test-days", help="Test window in trading days"),
) -> None:
    """Walk-forward out-of-sample validation of the sizing pipeline.

    Tests the 7-layer sizing approach against equal-weight baseline
    across multiple train/test windows.
    """
    from ..portfolio.walk_forward import walk_forward_test

    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        console.print("[red]No tickers provided.[/red]")
        raise typer.Exit(1)

    console.print(
        f"[bold]Walk-Forward Backtest[/bold]\n"
        f"  Tickers: {', '.join(ticker_list)}\n"
        f"  Windows: {windows} x {test_days}d test (trained on {train_days}d)\n"
    )

    with console.status("[bold cyan]Running walk-forward validation...", spinner="dots"):
        result = walk_forward_test(
            tickers=ticker_list,
            train_days=train_days,
            test_days=test_days,
            n_windows=windows,
        )

    if not result["windows"]:
        console.print("[yellow]No results — insufficient data for walk-forward test.[/yellow]")
        if "error" in result.get("aggregate", {}):
            console.print(f"[dim]{result['aggregate']['error']}[/dim]")
        raise typer.Exit()

    # Window details table
    table = Table(
        title="Walk-Forward Results",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Window", style="cyan", min_width=8)
    table.add_column("Test Period", min_width=14)
    table.add_column("Strategy %", justify="right")
    table.add_column("Baseline %", justify="right")
    table.add_column("Excess %", justify="right")
    table.add_column("Strat Sharpe", justify="right")
    table.add_column("Base Sharpe", justify="right")
    table.add_column("Avg Corr", justify="right")

    for w in result["windows"]:
        excess = w["excess_return"]
        excess_style = "green" if excess > 0 else ("red" if excess < 0 else "dim")
        strat_style = "green" if w["strategy_return"] > 0 else "red"
        base_style = "green" if w["baseline_return"] > 0 else "red"

        table.add_row(
            f"#{w['window']}",
            w["test_period"],
            Text(f"{w['strategy_return']:+.2f}%", style=strat_style),
            Text(f"{w['baseline_return']:+.2f}%", style=base_style),
            Text(f"{excess:+.2f}%", style=excess_style),
            f"{w['strategy_sharpe']:.2f}",
            f"{w['baseline_sharpe']:.2f}",
            f"{w['avg_correlation']:.2f}",
        )

    console.print(table)

    # Aggregate summary
    agg = result["aggregate"]
    summary_text = Text()
    summary_text.append("Avg Strategy Return: ", style="bold")
    sr = agg["avg_strategy_return"]
    summary_text.append(f"{sr:+.2f}%\n", style="green" if sr > 0 else "red")
    summary_text.append("Avg Baseline Return: ", style="bold")
    br = agg["avg_baseline_return"]
    summary_text.append(f"{br:+.2f}%\n", style="green" if br > 0 else "red")
    summary_text.append("Strategy Wins: ", style="bold")
    summary_text.append(f"{agg['strategy_wins']}/{agg['n_windows']}\n")
    summary_text.append("Information Ratio: ", style="bold")
    ir = agg["information_ratio"]
    ir_style = "green" if ir > 0.5 else ("yellow" if ir > 0 else "red")
    summary_text.append(f"{ir:.3f}", style=ir_style)

    border = "green" if agg["strategy_wins"] > agg["n_windows"] / 2 else "yellow"
    console.print(Panel(summary_text, title="Aggregate", border_style=border))
    console.print()
