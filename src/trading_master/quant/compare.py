"""Portfolio Optimizer Comparison — run Markowitz, HRP, and Risk Parity side by side.

Provides a unified view of three allocation methodologies on the same
asset universe, highlighting where they agree (consensus) and disagree
(model risk).

This helps investors understand:
  - Which assets all three methods overweight (high conviction)
  - Where methods diverge (model uncertainty → reduce position)
  - The risk/return tradeoffs of each approach
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .hrp import hrp_allocation
from .markowitz import max_sharpe_portfolio
from .risk_parity import risk_parity

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparing three allocation methods."""

    tickers: list[str]
    markowitz_weights: np.ndarray
    hrp_weights: np.ndarray
    risk_parity_weights: np.ndarray
    consensus_weights: np.ndarray     # average of all three
    dispersion: np.ndarray            # std across methods per asset
    max_dispersion_ticker: str        # highest model disagreement
    min_dispersion_ticker: str        # highest consensus

    @property
    def weight_table(self) -> list[dict]:
        """List of dicts for easy display/iteration."""
        rows = []
        for i, t in enumerate(self.tickers):
            rows.append({
                "ticker": t,
                "markowitz": float(self.markowitz_weights[i]),
                "hrp": float(self.hrp_weights[i]),
                "risk_parity": float(self.risk_parity_weights[i]),
                "consensus": float(self.consensus_weights[i]),
                "dispersion": float(self.dispersion[i]),
            })
        return rows


def compare_allocations(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    tickers: list[str] | None = None,
) -> ComparisonResult:
    """Run Markowitz, HRP, and Risk Parity on the same inputs.

    Parameters
    ----------
    expected_returns : (n,) annualized expected returns
    cov_matrix : (n, n) annualized covariance matrix
    tickers : asset names (optional)

    Returns
    -------
    ComparisonResult with all three weight vectors and consensus metrics.
    """
    mu = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)
    n = cov.shape[0]

    if tickers is None:
        tickers = [str(i) for i in range(n)]
    if len(tickers) != n:
        raise ValueError(f"Expected {n} tickers, got {len(tickers)}.")

    if n < 2:
        w = np.array([1.0])
        return ComparisonResult(
            tickers=tickers,
            markowitz_weights=w,
            hrp_weights=w,
            risk_parity_weights=w,
            consensus_weights=w,
            dispersion=np.array([0.0]),
            max_dispersion_ticker=tickers[0],
            min_dispersion_ticker=tickers[0],
        )

    # 1. Markowitz (max Sharpe)
    mk = max_sharpe_portfolio(mu, cov)
    mk_w = mk.weights

    # 2. HRP
    hrp_res = hrp_allocation(cov, tickers=tickers)
    hrp_w = hrp_res.weights

    # 3. Risk Parity
    rp_res = risk_parity(cov, tickers=tickers)
    rp_w = rp_res.weights

    # Consensus: simple average
    all_w = np.stack([mk_w, hrp_w, rp_w])  # (3, n)
    consensus = all_w.mean(axis=0)
    consensus = consensus / consensus.sum()  # re-normalize

    # Dispersion: std across methods
    dispersion = all_w.std(axis=0)

    max_disp_idx = int(np.argmax(dispersion))
    min_disp_idx = int(np.argmin(dispersion))

    return ComparisonResult(
        tickers=tickers,
        markowitz_weights=mk_w,
        hrp_weights=hrp_w,
        risk_parity_weights=rp_w,
        consensus_weights=consensus,
        dispersion=dispersion,
        max_dispersion_ticker=tickers[max_disp_idx],
        min_dispersion_ticker=tickers[min_disp_idx],
    )
