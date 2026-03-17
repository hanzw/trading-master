"""Quantitative analysis tools — 16 modules.

Monte Carlo, DCF, Black-Litterman, Fama-French, GARCH, CAPM, Markowitz,
HRP, Risk Parity, EVT, Regime HMM, Cointegration, Sector Rotation,
Compare, Multi-Timeframe, Dashboard.
"""

# Public API re-exports for convenient imports:
#   from trading_master.quant import risk_parity, hrp_allocation, evt_tail_risk

from .black_litterman import run_black_litterman
from .capm import capm_regression, capm_expected_return
from .cointegration import cointegration_test, spread_zscore_series
from .compare import compare_allocations
from .dashboard import build_dashboard, compute_risk_score
from .dcf import auto_dcf
from .evt import evt_tail_risk, mean_excess_plot_data
from .fama_french import ff5_decompose, attribute_returns
from .garch import fit_garch, forecast_volatility
from .hrp import hrp_allocation
from .markowitz import (
    minimum_variance_portfolio,
    max_sharpe_portfolio,
    efficient_frontier,
)
from .monte_carlo import simulate_portfolio_paths, stress_test_scenarios
from .multi_timeframe import multi_timeframe_analysis
from .regime import fit_regime_model
from .risk_parity import risk_parity, inverse_volatility
from .sector_rotation import analyze_sectors

__all__ = [
    # Black-Litterman
    "run_black_litterman",
    # CAPM
    "capm_regression",
    "capm_expected_return",
    # Cointegration
    "cointegration_test",
    "spread_zscore_series",
    # Compare
    "compare_allocations",
    # Dashboard
    "build_dashboard",
    "compute_risk_score",
    # DCF
    "auto_dcf",
    # EVT
    "evt_tail_risk",
    "mean_excess_plot_data",
    # Fama-French
    "ff5_decompose",
    "attribute_returns",
    # GARCH
    "fit_garch",
    "forecast_volatility",
    # HRP
    "hrp_allocation",
    # Markowitz
    "minimum_variance_portfolio",
    "max_sharpe_portfolio",
    "efficient_frontier",
    # Monte Carlo
    "simulate_portfolio_paths",
    "stress_test_scenarios",
    # Multi-Timeframe
    "multi_timeframe_analysis",
    # Regime
    "fit_regime_model",
    # Risk Parity
    "risk_parity",
    "inverse_volatility",
    # Sector Rotation
    "analyze_sectors",
]
