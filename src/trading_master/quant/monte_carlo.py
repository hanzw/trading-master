"""Monte Carlo simulation for portfolio stress testing."""

from __future__ import annotations

import numpy as np


def simulate_portfolio_paths(
    weights: np.ndarray,
    expected_returns: np.ndarray,  # annual
    cov_matrix: np.ndarray,  # annual
    initial_value: float = 1_000_000.0,
    horizon_days: int = 252,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> dict:
    """Simulate portfolio paths using correlated GBM.

    Uses Cholesky decomposition for correlated random draws.

    Returns
    -------
    dict with keys:
        paths            – (n_simulations, horizon_days) array of portfolio values
        final_values     – (n_simulations,) terminal portfolio values
        percentiles      – {5, 25, 50, 75, 95} of terminal values
        prob_loss        – P(final < initial)
        prob_gain_20pct  – P(final > 1.2 * initial)
        expected_value   – mean terminal value
        worst_case_5pct  – 5th percentile terminal value
        best_case_95pct  – 95th percentile terminal value
        max_drawdown_median – median of per-path max drawdowns
    """
    rng = np.random.default_rng(seed)
    n_assets = len(weights)
    weights = np.asarray(weights, dtype=float)
    expected_returns = np.asarray(expected_returns, dtype=float)
    cov_matrix = np.asarray(cov_matrix, dtype=float)

    # Daily parameters
    dt = 1.0 / 252.0
    daily_mu = expected_returns * dt  # (n_assets,)
    daily_cov = cov_matrix * dt  # (n_assets, n_assets)

    # Cholesky decomposition for correlated draws
    L = np.linalg.cholesky(daily_cov)  # (n_assets, n_assets)

    # Portfolio drift and vol under GBM
    # For each simulation day, draw correlated asset returns and compute
    # weighted portfolio return.
    # drift-adjusted mu for log-normal: mu_i - 0.5 * sigma_i^2
    daily_var = np.diag(daily_cov)
    drift = daily_mu - 0.5 * daily_var  # (n_assets,)

    # Generate random normals: (n_simulations, horizon_days, n_assets)
    Z = rng.standard_normal((n_simulations, horizon_days, n_assets))

    # Correlate: multiply each (n_assets,) vector by L^T
    correlated = Z @ L.T  # (n_simulations, horizon_days, n_assets)

    # Log-returns per asset per day
    log_returns = drift[np.newaxis, np.newaxis, :] + correlated  # (n_sim, days, assets)

    # Asset price relatives (cumulative product)
    cum_log_returns = np.cumsum(log_returns, axis=1)  # (n_sim, days, assets)
    asset_values = np.exp(cum_log_returns)  # relative to day-0 = 1

    # Portfolio value = initial * sum(w_i * S_i(t))
    # where S_i(t) is the asset value relative to start
    portfolio_values = initial_value * (asset_values @ weights)  # (n_sim, days)

    # Insert initial value at the start for drawdown calculation
    paths_full = np.column_stack(
        [np.full(n_simulations, initial_value), portfolio_values]
    )  # (n_sim, days+1)

    # We return paths without the initial column to match (n_sim, horizon_days)
    paths = portfolio_values
    final_values = paths[:, -1]

    # Percentiles
    pct_keys = [5, 25, 50, 75, 95]
    pct_vals = np.percentile(final_values, pct_keys)
    percentiles = dict(zip(pct_keys, pct_vals.tolist()))

    # Probabilities
    prob_loss = float(np.mean(final_values < initial_value))
    prob_gain_20pct = float(np.mean(final_values > 1.2 * initial_value))

    # Max drawdown per path (using paths_full which includes initial value)
    running_max = np.maximum.accumulate(paths_full, axis=1)
    drawdowns = (running_max - paths_full) / running_max
    max_drawdowns = np.max(drawdowns, axis=1)
    max_drawdown_median = float(np.median(max_drawdowns))

    return {
        "paths": paths,
        "final_values": final_values,
        "percentiles": percentiles,
        "prob_loss": prob_loss,
        "prob_gain_20pct": prob_gain_20pct,
        "expected_value": float(np.mean(final_values)),
        "worst_case_5pct": float(pct_vals[0]),
        "best_case_95pct": float(pct_vals[-1]),
        "max_drawdown_median": max_drawdown_median,
    }


# ---------------------------------------------------------------------------
# Stress-test scenarios
# ---------------------------------------------------------------------------

# Each scenario maps asset-class keywords to a shock (fractional return).
# The convention: equities, treasuries/bonds, commodities.  We apply shocks
# to portfolio weights that match.

_SCENARIOS: list[dict] = [
    {
        "scenario": "2008 Financial Crisis",
        "shocks": {"equities": -0.38, "treasuries": 0.20, "bonds": 0.20, "commodities": -0.25},
    },
    {
        "scenario": "2020 COVID Crash",
        "shocks": {"equities": -0.34, "treasuries": 0.15, "bonds": 0.15, "commodities": 0.0},
    },
    {
        "scenario": "2022 Rate Shock",
        "shocks": {"equities": -0.19, "treasuries": -0.13, "bonds": -0.13, "commodities": 0.0},
    },
    {
        "scenario": "Flash Crash",
        "shocks": {"equities": -0.10, "treasuries": 0.0, "bonds": 0.0, "commodities": -0.10},
    },
    {
        "scenario": "Stagflation",
        "shocks": {"equities": -0.15, "treasuries": -0.10, "bonds": -0.10, "commodities": 0.30},
    },
]


def stress_test_scenarios(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    portfolio_value: float,
    asset_classes: list[str] | None = None,
) -> list[dict]:
    """Run historical stress scenarios.

    Parameters
    ----------
    weights : array of portfolio weights (summing to 1).
    cov_matrix : covariance matrix (unused in simple stress but kept for API consistency).
    portfolio_value : current portfolio dollar value.
    asset_classes : list of class labels per weight position, e.g.
        ["equities", "equities", "bonds", "commodities"].
        If *None*, all positions are treated as equities.

    Returns
    -------
    list of dicts with keys: scenario, return_pct, dollar_impact.
    """
    weights = np.asarray(weights, dtype=float)
    n = len(weights)

    if asset_classes is None:
        asset_classes = ["equities"] * n

    results: list[dict] = []
    for sc in _SCENARIOS:
        shocks = sc["shocks"]
        # Build per-asset shock vector
        shock_vec = np.array(
            [shocks.get(cls, 0.0) for cls in asset_classes], dtype=float
        )
        # Weighted portfolio return under this scenario
        port_return = float(weights @ shock_vec)
        dollar_impact = port_return * portfolio_value
        results.append(
            {
                "scenario": sc["scenario"],
                "return_pct": port_return,
                "dollar_impact": dollar_impact,
            }
        )
    return results
