"""Hierarchical Risk Parity (HRP) — cluster-based portfolio allocation.

HRP (López de Prado, 2016) uses hierarchical clustering to build a
portfolio that doesn't require covariance matrix inversion:

1. Tree clustering: compute distance matrix from correlations, then
   hierarchical clustering (single/complete linkage).
2. Quasi-diagonalization: reorder assets so similar ones are adjacent.
3. Recursive bisection: allocate weights by splitting clusters based
   on inverse-variance within each branch.

Advantages over Markowitz:
  - No matrix inversion (numerically stable)
  - Better out-of-sample performance
  - Natural diversification through hierarchy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


@dataclass
class HRPResult:
    """Result of HRP allocation."""

    weights: np.ndarray
    tickers: list[str]
    cluster_order: list[int]  # quasi-diagonalized order
    n_clusters: int

    @property
    def weight_dict(self) -> dict[str, float]:
        """Mapping from ticker to weight."""
        return dict(zip(self.tickers, self.weights))


def _correlation_distance(corr_matrix: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to distance matrix: d = sqrt(0.5*(1-rho))."""
    corr = np.asarray(corr_matrix, dtype=float)
    # Ensure valid range
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - corr))
    # Zero diagonal
    np.fill_diagonal(dist, 0.0)
    return dist


def _quasi_diagonalize(link: np.ndarray, n: int) -> list[int]:
    """Reorder assets by traversing the dendrogram leaves.

    This places correlated assets together (quasi-diagonalization).
    """
    # Recursive extraction of leaf order from linkage matrix
    def _get_leaves(node_id: int) -> list[int]:
        if node_id < n:
            return [node_id]
        row = int(node_id - n)
        left = int(link[row, 0])
        right = int(link[row, 1])
        return _get_leaves(left) + _get_leaves(right)

    root = 2 * n - 2  # root node index
    return _get_leaves(root)


def _recursive_bisection(
    cov_matrix: np.ndarray,
    sorted_indices: list[int],
) -> np.ndarray:
    """Allocate weights by recursively bisecting the sorted asset list.

    At each step, split the list in half and allocate proportional
    to inverse cluster variance.
    """
    n = cov_matrix.shape[0]
    weights = np.ones(n)

    clusters = [sorted_indices]

    while clusters:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue

            # Split in half
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]

            # Compute inverse-variance weight for each sub-cluster
            left_var = _cluster_variance(cov_matrix, left)
            right_var = _cluster_variance(cov_matrix, right)

            total_inv_var = 1.0 / left_var + 1.0 / right_var
            alpha_left = (1.0 / left_var) / total_inv_var
            alpha_right = 1.0 - alpha_left

            # Scale weights
            for i in left:
                weights[i] *= alpha_left
            for i in right:
                weights[i] *= alpha_right

            if len(left) > 1:
                new_clusters.append(left)
            if len(right) > 1:
                new_clusters.append(right)

        clusters = new_clusters

    return weights


def _cluster_variance(cov_matrix: np.ndarray, indices: list[int]) -> float:
    """Compute the variance of an equal-weight portfolio of given assets."""
    cov_sub = cov_matrix[np.ix_(indices, indices)]
    n = len(indices)
    w = np.ones(n) / n
    var = float(w @ cov_sub @ w)
    return max(var, 1e-12)  # floor to avoid division by zero


def hrp_allocation(
    cov_matrix: np.ndarray,
    corr_matrix: np.ndarray | None = None,
    tickers: list[str] | None = None,
    linkage_method: str = "single",
) -> HRPResult:
    """Compute HRP portfolio weights.

    Parameters
    ----------
    cov_matrix : (n, n) covariance matrix (annualized or daily)
    corr_matrix : (n, n) correlation matrix. If None, derived from cov.
    tickers : asset names (optional, defaults to indices)
    linkage_method : 'single', 'complete', 'average', or 'ward'

    Returns
    -------
    HRPResult with weights, ordering, and cluster info.
    """
    cov = np.asarray(cov_matrix, dtype=float)
    n = cov.shape[0]

    if tickers is None:
        tickers = [str(i) for i in range(n)]

    if len(tickers) != n:
        raise ValueError(f"Expected {n} tickers, got {len(tickers)}.")

    if n == 1:
        return HRPResult(
            weights=np.array([1.0]),
            tickers=tickers,
            cluster_order=[0],
            n_clusters=1,
        )

    # Step 1: Correlation → distance matrix
    if corr_matrix is None:
        # Derive correlation from covariance
        std = np.sqrt(np.diag(cov))
        std[std < 1e-12] = 1e-12
        corr = cov / np.outer(std, std)
        corr = np.clip(corr, -1.0, 1.0)
    else:
        corr = np.asarray(corr_matrix, dtype=float)

    dist = _correlation_distance(corr)

    # Convert to condensed form for scipy
    dist_condensed = squareform(dist, checks=False)

    # Step 2: Hierarchical clustering
    link = linkage(dist_condensed, method=linkage_method)

    # Step 3: Quasi-diagonalization
    sorted_indices = _quasi_diagonalize(link, n)

    # Step 4: Recursive bisection
    weights = _recursive_bisection(cov, sorted_indices)

    # Normalize (should already sum to ~1, but ensure)
    weights = weights / weights.sum()

    # Count clusters (using a reasonable threshold)
    max_d = np.median(link[:, 2])
    clusters = fcluster(link, t=max_d, criterion="distance")
    n_clusters = len(set(clusters))

    return HRPResult(
        weights=weights,
        tickers=tickers,
        cluster_order=sorted_indices,
        n_clusters=n_clusters,
    )
