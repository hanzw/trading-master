"""Value Investing Strategy Backtest — test quality+value factor returns.

Since we cannot backtest individual stock fundamentals over 20 years
without expensive data (Compustat/Bloomberg), we use factor ETFs as
proxies to find the optimal Quality+Value+Momentum blend:

Factor ETFs (all have 10+ year history):
  QUAL  — iShares MSCI USA Quality Factor (high ROE, stable earnings)
  VLUE  — iShares MSCI USA Value Factor (low PE, low PB)
  MTUM  — iShares MSCI USA Momentum Factor
  USMV  — iShares MSCI USA Min Volatility
  SIZE  — iShares MSCI USA Size Factor (small cap tilt)
  SPY   — S&P 500 benchmark

We test various blends of Quality+Value+Momentum and find
the combination with highest risk-adjusted return.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Result of a single strategy backtest."""
    name: str
    weights: dict[str, float]  # ETF -> weight
    ann_return: float
    ann_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float  # return / max_dd
    win_rate_monthly: float
    total_return: float
    n_years: float


@dataclass
class OptimizationResult:
    """Result of strategy optimization."""
    best_strategy: BacktestResult
    all_strategies: list[BacktestResult]
    benchmark: BacktestResult
    n_strategies_tested: int
    factor_correlations: dict[str, float]


def backtest_blend(
    returns: np.ndarray,
    weights: np.ndarray,
    trading_days_per_year: int = 252,
) -> dict:
    """Backtest a fixed-weight blend of factor returns.

    Parameters
    ----------
    returns : (T, N) daily returns for N factors
    weights : (N,) portfolio weights

    Returns
    -------
    dict with performance metrics.
    """
    port_ret = returns @ weights
    T = len(port_ret)
    n_years = T / trading_days_per_year

    # Annualized metrics
    total_ret = np.prod(1 + port_ret) - 1
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = port_ret.std() * np.sqrt(trading_days_per_year)

    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # Sortino
    downside = port_ret[port_ret < 0]
    down_vol = np.sqrt((downside ** 2).mean()) * np.sqrt(trading_days_per_year) if len(downside) > 0 else 1
    sortino = ann_ret / down_vol if down_vol > 0 else 0

    # Max drawdown
    equity = np.cumprod(1 + port_ret)
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    max_dd = abs(dd.min())

    calmar = ann_ret / max_dd if max_dd > 0 else 0

    # Monthly win rate
    monthly_ret = []
    for i in range(0, T - 21, 21):
        mr = np.prod(1 + port_ret[i:i + 21]) - 1
        monthly_ret.append(mr)
    win_rate = sum(1 for r in monthly_ret if r > 0) / len(monthly_ret) if monthly_ret else 0

    return {
        "ann_return": float(ann_ret),
        "ann_volatility": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "win_rate_monthly": float(win_rate),
        "total_return": float(total_ret),
        "n_years": float(n_years),
    }


def optimize_value_blend(
    factor_returns: dict[str, np.ndarray],
    benchmark_returns: np.ndarray | None = None,
    n_samples: int = 5000,
    seed: int = 42,
) -> OptimizationResult:
    """Find the optimal Quality + Value + Momentum blend.

    Tests n_samples random weight combinations and finds the one
    with the highest Sharpe ratio.

    Parameters
    ----------
    factor_returns : {"QUAL": returns_array, "VLUE": ..., "MTUM": ...}
    benchmark_returns : SPY returns for comparison
    n_samples : number of random portfolios to test
    seed : random seed

    Returns
    -------
    OptimizationResult with best strategy and all tested strategies.
    """
    factors = list(factor_returns.keys())
    N = len(factors)

    # Align lengths
    min_len = min(len(r) for r in factor_returns.values())
    if benchmark_returns is not None:
        min_len = min(min_len, len(benchmark_returns))
        benchmark_returns = benchmark_returns[-min_len:]

    ret_matrix = np.column_stack([
        factor_returns[f][-min_len:] for f in factors
    ])

    # Factor correlations
    corr = np.corrcoef(ret_matrix, rowvar=False)
    factor_corr = {}
    for i in range(N):
        for j in range(i + 1, N):
            factor_corr[f"{factors[i]}/{factors[j]}"] = float(corr[i, j])

    rng = np.random.default_rng(seed)

    # Generate random weight combinations
    all_results: list[BacktestResult] = []

    # Also test some structured portfolios
    structured = [
        ("Equal Weight", {f: 1.0 / N for f in factors}),
        ("Quality Only", {f: (1.0 if f == "QUAL" else 0.0) for f in factors}),
        ("Value Only", {f: (1.0 if f == "VLUE" else 0.0) for f in factors}),
    ]
    if "MTUM" in factors:
        structured.append(("Momentum Only", {f: (1.0 if f == "MTUM" else 0.0) for f in factors}))
    if "USMV" in factors:
        structured.append(("Min Vol Only", {f: (1.0 if f == "USMV" else 0.0) for f in factors}))

    # Buffett-style blends
    if "QUAL" in factors and "VLUE" in factors:
        structured.append(("Buffett 50/50 QV", {"QUAL": 0.5, "VLUE": 0.5, **{f: 0 for f in factors if f not in ("QUAL", "VLUE")}}))
    if "QUAL" in factors and "VLUE" in factors and "MTUM" in factors:
        structured.append(("Buffett+Momentum 40/40/20", {"QUAL": 0.4, "VLUE": 0.4, "MTUM": 0.2, **{f: 0 for f in factors if f not in ("QUAL", "VLUE", "MTUM")}}))
    if "QUAL" in factors and "VLUE" in factors and "USMV" in factors:
        structured.append(("Conservative QV+MinVol", {"QUAL": 0.35, "VLUE": 0.35, "USMV": 0.30, **{f: 0 for f in factors if f not in ("QUAL", "VLUE", "USMV")}}))

    for name, w_dict in structured:
        w = np.array([w_dict.get(f, 0) for f in factors])
        if w.sum() > 0:
            w = w / w.sum()
        metrics = backtest_blend(ret_matrix, w)
        all_results.append(BacktestResult(name=name, weights=w_dict, **metrics))

    # Random portfolios
    for _ in range(n_samples):
        raw = rng.dirichlet(np.ones(N))
        metrics = backtest_blend(ret_matrix, raw)
        w_dict = {f: float(raw[i]) for i, f in enumerate(factors)}
        all_results.append(BacktestResult(name="random", weights=w_dict, **metrics))

    # Sort by Sharpe
    all_results.sort(key=lambda x: x.sharpe, reverse=True)

    # Best strategy
    best = all_results[0]

    # Benchmark
    bench = None
    if benchmark_returns is not None:
        bm = backtest_blend(benchmark_returns.reshape(-1, 1), np.array([1.0]))
        bench = BacktestResult(name="SPY Benchmark", weights={"SPY": 1.0}, **bm)
    else:
        bench = BacktestResult(name="N/A", weights={}, ann_return=0, ann_volatility=0,
                               sharpe=0, sortino=0, max_drawdown=0, calmar=0,
                               win_rate_monthly=0, total_return=0, n_years=0)

    # Filter to named strategies + top random
    named = [r for r in all_results if r.name != "random"]
    top_random = [r for r in all_results if r.name == "random"][:5]
    for i, r in enumerate(top_random):
        r.name = f"Optimal Blend #{i+1}"

    display = sorted(named + top_random, key=lambda x: x.sharpe, reverse=True)

    return OptimizationResult(
        best_strategy=best,
        all_strategies=display,
        benchmark=bench,
        n_strategies_tested=len(all_results),
        factor_correlations=factor_corr,
    )
