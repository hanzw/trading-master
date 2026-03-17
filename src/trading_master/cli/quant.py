"""CLI commands for quantitative analysis (Monte Carlo, DCF, Black-Litterman, HRP, Risk Parity, EVT)."""

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


@quant_app.command("bl")
def black_litterman_cmd(
    tickers: str = typer.Option(
        "",
        "--tickers",
        "-t",
        help="Comma-separated tickers (default: current portfolio holdings)",
    ),
    lookback: int = typer.Option(252, "--lookback", help="Lookback days for covariance"),
) -> None:
    """Run the Black-Litterman model: LLM analyst views -> optimal portfolio weights."""
    from ..db import get_db
    from ..portfolio.tracker import PortfolioTracker
    from ..quant.black_litterman import run_black_litterman

    tracker = PortfolioTracker()

    with console.status("[bold cyan]Loading portfolio...", spinner="dots"):
        state = tracker.get_state()

    # Determine tickers
    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    elif state.positions:
        ticker_list = sorted(state.positions.keys())
    else:
        console.print("[red]No tickers specified and no portfolio positions found.[/red]")
        console.print("Use: tm quant bl --tickers AAPL,MSFT,GOOGL")
        raise typer.Exit(1)

    # Build current weights from portfolio
    current_weights = None
    if state.positions:
        total_val = sum(
            state.positions[t].market_value
            for t in ticker_list
            if t in state.positions
        )
        if total_val > 0:
            current_weights = np.array([
                state.positions[t].market_value / total_val
                if t in state.positions else 0.0
                for t in ticker_list
            ])

    # Fetch latest analyst reports from DB
    db = get_db()
    analyst_reports: list[dict] = []
    for t in ticker_list:
        recs = db.get_recommendations(ticker=t, limit=1)
        if recs:
            rec = recs[0]
            reports_json = rec.get("analyst_reports", "[]")
            if isinstance(reports_json, str):
                try:
                    reports = json.loads(reports_json)
                except (json.JSONDecodeError, TypeError):
                    reports = []
            else:
                reports = reports_json
            for r in reports:
                analyst_reports.append({
                    "ticker": t,
                    "signal": r.get("signal", "HOLD"),
                    "confidence": r.get("confidence", 50),
                })

    if not analyst_reports:
        console.print("[yellow]No analyst reports found in DB — using neutral (HOLD) views.[/yellow]")

    with console.status("[bold cyan]Running Black-Litterman model...", spinner="dots"):
        result = run_black_litterman(
            tickers=ticker_list,
            analyst_reports=analyst_reports,
            current_weights=current_weights,
            lookback_days=lookback,
        )

    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
        raise typer.Exit(1)

    valid = result["tickers"]
    eq_ret = result["equilibrium_returns"]
    bl_ret = result["bl_returns"]
    w_cur = result["current_weights"]
    w_opt = result["optimal_weights"]
    trades = result["suggested_trades"]

    # ── Returns table ──
    ret_table = Table(
        title="Black-Litterman Returns (annualized)",
        show_header=True,
        header_style="bold magenta",
    )
    ret_table.add_column("Ticker", style="cyan", min_width=8)
    ret_table.add_column("Equilibrium", justify="right", min_width=12)
    ret_table.add_column("BL Posterior", justify="right", min_width=12)
    ret_table.add_column("Shift", justify="right", min_width=10)

    for i, t in enumerate(valid):
        eq = eq_ret[i] * 100
        bl = bl_ret[i] * 100
        shift = bl - eq
        shift_color = "green" if shift > 0.5 else ("red" if shift < -0.5 else "dim")
        ret_table.add_row(
            t,
            f"{eq:+.2f}%",
            f"{bl:+.2f}%",
            Text(f"{shift:+.2f}%", style=shift_color),
        )

    console.print(ret_table)
    console.print()

    # ── Weights table ──
    wt_table = Table(
        title="Portfolio Weights: Current vs Optimal",
        show_header=True,
        header_style="bold magenta",
    )
    wt_table.add_column("Ticker", style="cyan", min_width=8)
    wt_table.add_column("Current %", justify="right", min_width=10)
    wt_table.add_column("Optimal %", justify="right", min_width=10)
    wt_table.add_column("Direction", justify="center", min_width=10)

    for trade in trades:
        dir_style = (
            "bold green" if trade["direction"] == "BUY"
            else ("bold red" if trade["direction"] == "SELL" else "dim")
        )
        wt_table.add_row(
            trade["ticker"],
            f"{trade['current_pct']:.1f}%",
            f"{trade['target_pct']:.1f}%",
            Text(trade["direction"], style=dir_style),
        )

    console.print(wt_table)
    console.print()


@quant_app.command("hrp")
def hrp_cmd(
    tickers: str = typer.Option(
        "",
        "--tickers",
        "-t",
        help="Comma-separated tickers (default: current portfolio holdings)",
    ),
    lookback: int = typer.Option(252, "--lookback", help="Lookback days for covariance"),
    linkage: str = typer.Option("single", "--linkage", "-l", help="Linkage method: single, complete, average, ward"),
) -> None:
    """Run Hierarchical Risk Parity allocation on portfolio holdings."""
    from ..portfolio.tracker import PortfolioTracker
    from ..portfolio.correlation import fetch_returns
    from ..quant.hrp import hrp_allocation

    tracker = PortfolioTracker()

    with console.status("[bold cyan]Loading portfolio...", spinner="dots"):
        state = tracker.get_state()

    # Determine tickers
    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    elif state.positions:
        ticker_list = sorted(state.positions.keys())
    else:
        console.print("[red]No tickers specified and no portfolio positions found.[/red]")
        console.print("Use: tm quant hrp --tickers AAPL,MSFT,GOOGL,TLT")
        raise typer.Exit(1)

    if len(ticker_list) < 2:
        console.print("[red]HRP requires at least 2 tickers.[/red]")
        raise typer.Exit(1)

    # Fetch returns and compute covariance/correlation
    with console.status("[bold cyan]Fetching returns data...", spinner="dots"):
        returns_array, valid_tickers = fetch_returns(ticker_list, lookback_days=lookback)

    if returns_array is None or len(valid_tickers) < 2:
        console.print("[red]Could not fetch sufficient return data.[/red]")
        raise typer.Exit(1)

    cov_matrix = np.cov(returns_array, rowvar=False) * 252  # annualize
    corr_matrix = np.corrcoef(returns_array, rowvar=False)

    with console.status("[bold cyan]Running HRP optimization...", spinner="dots"):
        result = hrp_allocation(
            cov_matrix=cov_matrix,
            corr_matrix=corr_matrix,
            tickers=list(valid_tickers),
            linkage_method=linkage,
        )

    # ── Current weights (from portfolio) ──
    current_weights: dict[str, float] = {}
    if state.positions:
        total_val = sum(
            state.positions[t].market_value
            for t in valid_tickers
            if t in state.positions
        )
        if total_val > 0:
            for t in valid_tickers:
                if t in state.positions:
                    current_weights[t] = state.positions[t].market_value / total_val
                else:
                    current_weights[t] = 0.0

    # ── Weights table ──
    wt_table = Table(
        title=f"HRP Allocation ({linkage} linkage, {result.n_clusters} clusters)",
        show_header=True,
        header_style="bold magenta",
    )
    wt_table.add_column("Ticker", style="cyan", min_width=8)
    if current_weights:
        wt_table.add_column("Current %", justify="right", min_width=10)
    wt_table.add_column("HRP %", justify="right", min_width=10)
    if current_weights:
        wt_table.add_column("Direction", justify="center", min_width=10)

    wd = result.weight_dict
    for t in valid_tickers:
        hrp_pct = wd.get(t, 0.0) * 100
        row = [t]
        if current_weights:
            cur_pct = current_weights.get(t, 0.0) * 100
            row.append(f"{cur_pct:.1f}%")
        row.append(f"{hrp_pct:.1f}%")
        if current_weights:
            cur_pct = current_weights.get(t, 0.0) * 100
            diff = hrp_pct - cur_pct
            if diff > 1.0:
                direction = Text("BUY", style="bold green")
            elif diff < -1.0:
                direction = Text("SELL", style="bold red")
            else:
                direction = Text("HOLD", style="dim")
            row.append(direction)
        wt_table.add_row(*row)

    console.print(wt_table)
    console.print()


@quant_app.command("risk-parity")
def risk_parity_cmd(
    tickers: str = typer.Option(
        "",
        "--tickers",
        "-t",
        help="Comma-separated tickers (default: current portfolio holdings)",
    ),
    lookback: int = typer.Option(252, "--lookback", help="Lookback days for covariance"),
) -> None:
    """Run Risk Parity (Equal Risk Contribution) allocation on portfolio holdings."""
    from ..portfolio.tracker import PortfolioTracker
    from ..portfolio.correlation import fetch_returns
    from ..quant.risk_parity import risk_parity

    tracker = PortfolioTracker()

    with console.status("[bold cyan]Loading portfolio...", spinner="dots"):
        state = tracker.get_state()

    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    elif state.positions:
        ticker_list = sorted(state.positions.keys())
    else:
        console.print("[red]No tickers specified and no portfolio positions found.[/red]")
        console.print("Use: tm quant risk-parity --tickers AAPL,MSFT,GOOGL,TLT")
        raise typer.Exit(1)

    if len(ticker_list) < 2:
        console.print("[red]Risk Parity requires at least 2 tickers.[/red]")
        raise typer.Exit(1)

    with console.status("[bold cyan]Fetching returns data...", spinner="dots"):
        returns_array, valid_tickers = fetch_returns(ticker_list, lookback_days=lookback)

    if returns_array is None or len(valid_tickers) < 2:
        console.print("[red]Could not fetch sufficient return data.[/red]")
        raise typer.Exit(1)

    cov_matrix = np.cov(returns_array, rowvar=False) * 252

    with console.status("[bold cyan]Running Risk Parity optimization...", spinner="dots"):
        result = risk_parity(
            cov_matrix=cov_matrix,
            tickers=list(valid_tickers),
        )

    wt_table = Table(
        title=f"Risk Parity Allocation (ERC, {'converged' if result.converged else 'NOT converged'})",
        show_header=True,
        header_style="bold magenta",
    )
    wt_table.add_column("Ticker", style="cyan", min_width=8)
    wt_table.add_column("Weight %", justify="right", min_width=10)
    wt_table.add_column("Risk Contrib %", justify="right", min_width=14)
    wt_table.add_column("Target %", justify="right", min_width=10)

    wd = result.weight_dict
    rd = result.risk_dict
    target = 100.0 / len(valid_tickers)

    for t in valid_tickers:
        w_pct = wd.get(t, 0.0) * 100
        r_pct = rd.get(t, 0.0) * 100
        diff = abs(r_pct - target)
        rc_style = "green" if diff < 2.0 else ("yellow" if diff < 5.0 else "red")
        wt_table.add_row(
            t,
            f"{w_pct:.1f}%",
            Text(f"{r_pct:.1f}%", style=rc_style),
            f"{target:.1f}%",
        )

    console.print(wt_table)
    console.print(f"\n[dim]Portfolio volatility: {result.portfolio_volatility:.2%}[/dim]")
    console.print()


@quant_app.command("evt")
def evt_cmd(
    ticker: str = typer.Argument(..., help="Stock ticker symbol"),
    lookback: int = typer.Option(504, "--lookback", help="Lookback days (default 2yr)"),
    threshold: float = typer.Option(0.90, "--threshold", "-u", help="POT threshold quantile"),
) -> None:
    """Run Extreme Value Theory tail risk analysis for a ticker."""
    from ..portfolio.correlation import fetch_returns
    from ..quant.evt import evt_tail_risk

    ticker = ticker.upper()
    console.print(f"[bold]Running EVT tail risk analysis for {ticker}...[/bold]")

    try:
        returns_array, valid = fetch_returns([ticker], lookback_days=lookback)
    except Exception as exc:
        console.print(f"[red]Error fetching data:[/red] {exc}")
        raise typer.Exit(1)

    if returns_array is None or len(valid) == 0:
        console.print("[red]Could not fetch return data.[/red]")
        raise typer.Exit(1)

    returns = returns_array[:, 0]

    try:
        result = evt_tail_risk(returns, threshold_quantile=threshold)
    except Exception as exc:
        console.print(f"[red]EVT analysis failed:[/red] {exc}")
        raise typer.Exit(1)

    # Tail type styling
    tail_style = {"heavy": "bold red", "exponential": "yellow", "bounded": "green"}

    table = Table(title=f"EVT Tail Risk — {ticker}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Tail Type", Text(result.tail_type.upper(), style=tail_style.get(result.tail_type, "white")))
    table.add_row("Shape (xi)", f"{result.shape:.4f}")
    table.add_row("Scale (sigma)", f"{result.scale:.4f}")
    table.add_row("Threshold", f"{result.threshold:.4f}")
    table.add_row("Exceedances", f"{result.n_exceedances} / {result.n_total} ({result.exceedance_rate:.1%})")
    table.add_row("", "")
    table.add_row("VaR 95%", f"{result.var_95:.2%}")
    table.add_row("VaR 99%", f"{result.var_99:.2%}")
    table.add_row("CVaR 95%", f"{result.cvar_95:.2%}")
    table.add_row("CVaR 99%", f"{result.cvar_99:.2%}")

    if result.ks_pvalue is not None:
        ks_style = "green" if result.ks_pvalue > 0.05 else "red"
        table.add_row("", "")
        table.add_row("KS p-value", Text(f"{result.ks_pvalue:.4f}", style=ks_style))

    console.print(table)


@quant_app.command("regime")
def regime_cmd(
    ticker: str = typer.Argument("SPY", help="Ticker to analyze (default: SPY for market regime)"),
    lookback: int = typer.Option(504, "--lookback", help="Lookback days (default 2yr)"),
    regimes: int = typer.Option(3, "--regimes", "-k", help="Number of regimes (2 or 3)"),
) -> None:
    """Detect market regime using Hidden Markov Model."""
    from ..portfolio.correlation import fetch_returns
    from ..quant.regime import fit_regime_model

    ticker = ticker.upper()
    console.print(f"[bold]Fitting {regimes}-state HMM for {ticker}...[/bold]")

    try:
        returns_array, valid = fetch_returns([ticker], lookback_days=lookback)
    except Exception as exc:
        console.print(f"[red]Error fetching data:[/red] {exc}")
        raise typer.Exit(1)

    if returns_array is None or len(valid) == 0:
        console.print("[red]Could not fetch return data.[/red]")
        raise typer.Exit(1)

    with console.status("[bold cyan]Running EM (this may take a minute)...", spinner="dots"):
        try:
            result = fit_regime_model(returns_array[:, 0], n_regimes=regimes)
        except Exception as exc:
            console.print(f"[red]HMM fitting failed:[/red] {exc}")
            raise typer.Exit(1)

    regime_styles = {"bear": "bold red", "neutral": "yellow", "bull": "bold green"}

    table = Table(title=f"Regime Detection — {ticker} ({result.n_regimes}-state HMM)")
    table.add_column("Regime", style="cyan")
    table.add_column("Mean (daily)", justify="right")
    table.add_column("Volatility (ann.)", justify="right")
    table.add_column("Persistence", justify="right")
    table.add_column("Stationary %", justify="right")

    for label in result.regime_labels:
        info = result.regime_summary[label]
        style = regime_styles.get(label, "white")
        ann_vol = info["volatility"] * np.sqrt(252)
        table.add_row(
            Text(label.upper(), style=style),
            f"{info['mean']:.4%}",
            f"{ann_vol:.1%}",
            f"{info['persistence']:.1%}",
            f"{info['stationary_prob']:.1%}",
        )

    console.print(table)

    cur_style = regime_styles.get(result.current_label, "white")
    console.print(f"\nCurrent regime: [{cur_style}]{result.current_label.upper()}[/] "
                  f"(confidence: {result.current_probs[result.current_regime]:.0%})")
    console.print(f"[dim]Converged: {result.converged} | Iterations: {result.n_iterations} | "
                  f"Log-likelihood: {result.log_likelihood:.1f}[/dim]")
