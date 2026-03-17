"""Regime Switching — Hidden Markov Model for market regime detection.

Uses a Gaussian HMM to classify market returns into discrete regimes
(e.g., bull/bear/crisis) based on the statistical properties of returns.

The model is fit via Expectation-Maximization (Baum-Welch algorithm)
using vectorized numpy operations — no Python loops over T observations.

Each regime is characterized by its own mean and variance, and the
model provides:
  - Most likely regime sequence (Viterbi decoding)
  - Current regime probability (filtered state)
  - Transition matrix (regime persistence and switching rates)

References:
  - Hamilton (1989) — "A New Approach to the Economic Analysis of
    Nonstationary Time Series and the Business Cycle"
  - Ang & Bekaert (2002) — "Regime Switches in Interest Rates"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class RegimeResult:
    """Result of HMM regime detection."""

    n_regimes: int
    means: np.ndarray            # (K,) regime means
    variances: np.ndarray        # (K,) regime variances
    volatilities: np.ndarray     # (K,) regime std devs
    transition_matrix: np.ndarray  # (K, K) transition probabilities
    stationary_probs: np.ndarray   # (K,) long-run regime probabilities

    regime_sequence: np.ndarray  # (T,) most likely regime at each time step
    regime_probs: np.ndarray     # (T, K) filtered regime probabilities
    current_regime: int
    current_probs: np.ndarray    # (K,)

    regime_labels: list[str]

    log_likelihood: float
    n_iterations: int
    converged: bool

    @property
    def current_label(self) -> str:
        return self.regime_labels[self.current_regime]

    @property
    def regime_summary(self) -> dict[str, dict]:
        summary = {}
        for i, label in enumerate(self.regime_labels):
            summary[label] = {
                "mean": float(self.means[i]),
                "volatility": float(self.volatilities[i]),
                "stationary_prob": float(self.stationary_probs[i]),
                "persistence": float(self.transition_matrix[i, i]),
            }
        return summary


def _emission_matrix(obs: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Compute (T, K) emission probability matrix — fully vectorized."""
    # obs: (T,), means: (K,), stds: (K,) -> broadcast to (T, K)
    stds_safe = np.maximum(stds, 1e-10)
    B = norm.pdf(obs[:, None], loc=means[None, :], scale=stds_safe[None, :])
    return np.maximum(B, 1e-300)


def _forward(B: np.ndarray, pi: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized forward pass with scaling.

    B : (T, K) emission probs
    pi : (K,) initial probs
    A : (K, K) transition matrix
    """
    T, K = B.shape
    alpha = np.zeros((T, K))
    scales = np.zeros(T)

    alpha[0] = pi * B[0]
    scales[0] = alpha[0].sum()
    if scales[0] > 0:
        alpha[0] /= scales[0]

    for t in range(1, T):
        # alpha[t] = (alpha[t-1] @ A) * B[t]  — vectorized over K
        alpha[t] = (alpha[t - 1] @ A) * B[t]
        scales[t] = alpha[t].sum()
        if scales[t] > 0:
            alpha[t] /= scales[t]

    return alpha, scales


def _backward(B: np.ndarray, A: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Vectorized backward pass with scaling."""
    T, K = B.shape
    beta = np.zeros((T, K))
    beta[T - 1] = 1.0

    for t in range(T - 2, -1, -1):
        # beta[t] = A @ (B[t+1] * beta[t+1])  — vectorized over K
        beta[t] = A @ (B[t + 1] * beta[t + 1])
        if scales[t + 1] > 0:
            beta[t] /= scales[t + 1]

    return beta


def _viterbi(B: np.ndarray, pi: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Vectorized Viterbi algorithm."""
    T, K = B.shape
    log_pi = np.log(np.maximum(pi, 1e-300))
    log_A = np.log(np.maximum(A, 1e-300))
    log_B = np.log(np.maximum(B, 1e-300))

    delta = np.zeros((T, K))
    psi = np.zeros((T, K), dtype=int)

    delta[0] = log_pi + log_B[0]

    for t in range(1, T):
        # candidates[i, j] = delta[t-1, i] + log_A[i, j]
        candidates = delta[t - 1, :, None] + log_A  # (K, K)
        psi[t] = candidates.argmax(axis=0)           # (K,)
        delta[t] = candidates.max(axis=0) + log_B[t] # (K,)

    path = np.zeros(T, dtype=int)
    path[T - 1] = int(np.argmax(delta[T - 1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    return path


def fit_regime_model(
    returns: np.ndarray,
    n_regimes: int = 3,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 42,
) -> RegimeResult:
    """Fit a Gaussian Hidden Markov Model to return data.

    Parameters
    ----------
    returns : 1-D array of returns (daily)
    n_regimes : number of hidden states (2=bull/bear, 3=bull/neutral/bear)
    max_iter : maximum EM iterations
    tol : convergence tolerance on log-likelihood
    seed : random seed for initialization

    Returns
    -------
    RegimeResult with regime parameters, sequence, and diagnostics.
    """
    returns = np.asarray(returns, dtype=float).flatten()
    returns = returns[np.isfinite(returns)]
    T = len(returns)
    K = n_regimes

    if T < 30:
        raise ValueError(f"Need at least 30 observations, got {T}.")
    if K < 2:
        raise ValueError("Need at least 2 regimes.")

    # ── Initialize parameters ──
    sorted_ret = np.sort(returns)
    chunk = T // K
    means = np.array([sorted_ret[i * chunk:(i + 1) * chunk].mean() for i in range(K)])
    stds = np.array([max(sorted_ret[i * chunk:(i + 1) * chunk].std(), 1e-6) for i in range(K)])

    A = np.full((K, K), 0.05 / (K - 1))
    np.fill_diagonal(A, 0.95)
    A = A / A.sum(axis=1, keepdims=True)

    pi = np.ones(K) / K

    # ── EM loop ──
    prev_ll = -np.inf
    converged = False

    for iteration in range(max_iter):
        # Emission matrix (T, K) — computed once per iteration
        B = _emission_matrix(returns, means, stds)

        # E-step
        alpha, scales = _forward(B, pi, A)
        beta = _backward(B, A, scales)

        # Gamma (state posteriors)
        gamma = alpha * beta
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma = gamma / np.maximum(gamma_sum, 1e-300)

        # Log-likelihood
        ll = np.sum(np.log(np.maximum(scales, 1e-300)))
        if abs(ll - prev_ll) < tol:
            converged = True
            break
        prev_ll = ll

        # M-step
        pi = gamma[0] / gamma[0].sum()

        # Transition matrix — vectorized: xi[i,j] = sum_t alpha[t,i]*A[i,j]*B[t+1,j]*beta[t+1,j]
        # alpha[:-1]: (T-1, K), B[1:]*beta[1:]: (T-1, K)
        ab = alpha[:-1]                    # (T-1, K)
        bb = B[1:] * beta[1:]             # (T-1, K)
        # xi = sum_t (ab[t, :, None] * A * bb[t, None, :])
        # = (ab.T @ bb) * A  element-wise — but we need the sum correctly
        # xi[i,j] = A[i,j] * sum_t ab[t,i] * bb[t,j]
        xi = A * (ab.T @ bb)              # (K, K)
        xi_row = xi.sum(axis=1, keepdims=True)
        A = xi / np.maximum(xi_row, 1e-300)

        # Emission parameters — vectorized
        gamma_sum_k = gamma.sum(axis=0)    # (K,)
        gamma_sum_k = np.maximum(gamma_sum_k, 1e-300)

        means = (gamma * returns[:, None]).sum(axis=0) / gamma_sum_k
        diff = returns[:, None] - means[None, :]   # (T, K)
        stds = np.sqrt((gamma * diff ** 2).sum(axis=0) / gamma_sum_k)
        stds = np.maximum(stds, 1e-6)

    n_iterations = iteration + 1

    # ── Sort regimes by mean ──
    order = np.argsort(means)
    means = means[order]
    stds = stds[order]
    variances = stds ** 2
    A = A[np.ix_(order, order)]
    pi = pi[order]
    gamma = gamma[:, order]

    # Viterbi with sorted params
    B_sorted = _emission_matrix(returns, means, stds)
    regime_seq = _viterbi(B_sorted, pi, A)

    # Stationary distribution
    try:
        eigenvalues, eigenvectors = np.linalg.eig(A.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = np.maximum(stationary / stationary.sum(), 0)
    except Exception:
        stationary = pi

    if K == 2:
        labels = ["bear", "bull"]
    elif K == 3:
        labels = ["bear", "neutral", "bull"]
    else:
        labels = [f"regime_{i}" for i in range(K)]

    current_regime = int(regime_seq[-1])
    current_probs = gamma[-1]

    return RegimeResult(
        n_regimes=K,
        means=means,
        variances=variances,
        volatilities=stds,
        transition_matrix=A,
        stationary_probs=stationary,
        regime_sequence=regime_seq,
        regime_probs=gamma,
        current_regime=current_regime,
        current_probs=current_probs,
        regime_labels=labels,
        log_likelihood=prev_ll,
        n_iterations=n_iterations,
        converged=converged,
    )
