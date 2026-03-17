"""Regime Switching — Hidden Markov Model for market regime detection.

Uses a Gaussian HMM to classify market returns into discrete regimes
(e.g., bull/bear/crisis) based on the statistical properties of returns.

The model is fit via Expectation-Maximization (Baum-Welch algorithm)
using only numpy/scipy — no external HMM libraries required.

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
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class RegimeResult:
    """Result of HMM regime detection."""

    n_regimes: int
    # Per-regime parameters (sorted by mean: lowest=bear/crisis, highest=bull)
    means: np.ndarray            # (K,) regime means
    variances: np.ndarray        # (K,) regime variances
    volatilities: np.ndarray     # (K,) regime std devs (sqrt of variances)
    transition_matrix: np.ndarray  # (K, K) transition probabilities
    stationary_probs: np.ndarray   # (K,) long-run regime probabilities

    # Sequence results
    regime_sequence: np.ndarray  # (T,) most likely regime at each time step
    regime_probs: np.ndarray     # (T, K) filtered regime probabilities
    current_regime: int          # most likely regime at last observation
    current_probs: np.ndarray    # (K,) probabilities for current regime

    # Labels
    regime_labels: list[str]     # human-readable labels

    # Diagnostics
    log_likelihood: float
    n_iterations: int
    converged: bool

    @property
    def current_label(self) -> str:
        return self.regime_labels[self.current_regime]

    @property
    def regime_summary(self) -> dict[str, dict]:
        """Summary of each regime's characteristics."""
        summary = {}
        for i, label in enumerate(self.regime_labels):
            summary[label] = {
                "mean": float(self.means[i]),
                "volatility": float(self.volatilities[i]),
                "stationary_prob": float(self.stationary_probs[i]),
                "persistence": float(self.transition_matrix[i, i]),
            }
        return summary


def _gaussian_emission(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Compute Gaussian emission probabilities, clipped for numerical stability."""
    probs = norm.pdf(x, loc=mu, scale=max(sigma, 1e-10))
    return np.maximum(probs, 1e-300)


def _forward(obs: np.ndarray, pi: np.ndarray, A: np.ndarray,
             means: np.ndarray, stds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward pass (alpha computation) with scaling.

    Returns (alpha_hat, scales) where alpha_hat are scaled forward probs.
    """
    T = len(obs)
    K = len(pi)
    alpha = np.zeros((T, K))
    scales = np.zeros(T)

    # t=0
    for j in range(K):
        alpha[0, j] = pi[j] * _gaussian_emission(obs[0], means[j], stds[j])
    scales[0] = alpha[0].sum()
    if scales[0] > 0:
        alpha[0] /= scales[0]

    # t=1..T-1
    for t in range(1, T):
        for j in range(K):
            alpha[t, j] = sum(alpha[t-1, i] * A[i, j] for i in range(K)) * \
                          _gaussian_emission(obs[t], means[j], stds[j])
        scales[t] = alpha[t].sum()
        if scales[t] > 0:
            alpha[t] /= scales[t]

    return alpha, scales


def _backward(obs: np.ndarray, A: np.ndarray,
              means: np.ndarray, stds: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Backward pass (beta computation) with scaling."""
    T = len(obs)
    K = A.shape[0]
    beta = np.zeros((T, K))

    beta[T-1, :] = 1.0

    for t in range(T-2, -1, -1):
        for i in range(K):
            beta[t, i] = sum(
                A[i, j] * _gaussian_emission(obs[t+1], means[j], stds[j]) * beta[t+1, j]
                for j in range(K)
            )
        if scales[t+1] > 0:
            beta[t] /= scales[t+1]

    return beta


def _viterbi(obs: np.ndarray, pi: np.ndarray, A: np.ndarray,
             means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Viterbi algorithm for most likely state sequence."""
    T = len(obs)
    K = len(pi)

    # Log domain for numerical stability
    log_pi = np.log(np.maximum(pi, 1e-300))
    log_A = np.log(np.maximum(A, 1e-300))

    delta = np.zeros((T, K))
    psi = np.zeros((T, K), dtype=int)

    for j in range(K):
        delta[0, j] = log_pi[j] + norm.logpdf(obs[0], means[j], max(stds[j], 1e-10))

    for t in range(1, T):
        for j in range(K):
            candidates = delta[t-1] + log_A[:, j]
            psi[t, j] = int(np.argmax(candidates))
            delta[t, j] = candidates[psi[t, j]] + norm.logpdf(obs[t], means[j], max(stds[j], 1e-10))

    # Backtrack
    path = np.zeros(T, dtype=int)
    path[T-1] = int(np.argmax(delta[T-1]))
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]

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

    rng = np.random.default_rng(seed)

    # ── Initialize parameters ──
    # Use quantile-based initialization for better convergence
    sorted_ret = np.sort(returns)
    chunk = T // K
    means = np.array([sorted_ret[i * chunk:(i+1) * chunk].mean() for i in range(K)])
    stds = np.array([max(sorted_ret[i * chunk:(i+1) * chunk].std(), 1e-6) for i in range(K)])

    # Transition matrix: high self-persistence
    A = np.full((K, K), 0.05 / (K - 1))
    np.fill_diagonal(A, 0.95)
    A = A / A.sum(axis=1, keepdims=True)

    # Initial state probabilities (uniform)
    pi = np.ones(K) / K

    # ── EM loop ──
    prev_ll = -np.inf
    converged = False

    for iteration in range(max_iter):
        # E-step: forward-backward
        alpha, scales = _forward(returns, pi, A, means, stds)
        beta = _backward(returns, A, means, stds, scales)

        # Compute gamma (state posteriors) and xi (transition posteriors)
        gamma = alpha * beta
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma_sum = np.maximum(gamma_sum, 1e-300)
        gamma = gamma / gamma_sum

        # Log-likelihood from scales
        ll = np.sum(np.log(np.maximum(scales, 1e-300)))

        if abs(ll - prev_ll) < tol:
            converged = True
            break
        prev_ll = ll

        # M-step: update parameters
        # Initial probabilities
        pi = gamma[0] / gamma[0].sum()

        # Transition matrix
        xi = np.zeros((K, K))
        for t in range(T - 1):
            for i in range(K):
                for j in range(K):
                    xi[i, j] += alpha[t, i] * A[i, j] * \
                                _gaussian_emission(returns[t+1], means[j], stds[j]) * beta[t+1, j]
        xi_row_sums = xi.sum(axis=1, keepdims=True)
        xi_row_sums = np.maximum(xi_row_sums, 1e-300)
        A = xi / xi_row_sums

        # Emission parameters
        gamma_sum_per_state = gamma.sum(axis=0)
        gamma_sum_per_state = np.maximum(gamma_sum_per_state, 1e-300)

        for j in range(K):
            means[j] = np.sum(gamma[:, j] * returns) / gamma_sum_per_state[j]
            diff = returns - means[j]
            stds[j] = np.sqrt(np.sum(gamma[:, j] * diff**2) / gamma_sum_per_state[j])
            stds[j] = max(stds[j], 1e-6)

    n_iterations = iteration + 1

    # ── Sort regimes by mean (lowest = bear/crisis, highest = bull) ──
    order = np.argsort(means)
    means = means[order]
    stds = stds[order]
    variances = stds ** 2
    A = A[np.ix_(order, order)]
    pi = pi[order]
    gamma = gamma[:, order]

    # Viterbi decoding with sorted parameters
    regime_seq = _viterbi(returns, pi, A, means, stds)
    # Remap viterbi output to sorted order
    remap = np.argsort(order)
    # Actually we need the inverse: if original regime i mapped to sorted position order[i],
    # we need sorted regime j to map from original regime order[j]
    # Since we already sorted A, means, stds, re-run viterbi with sorted params
    regime_seq = _viterbi(returns, pi, A, means, stds)

    # Stationary distribution from transition matrix
    try:
        eigenvalues, eigenvectors = np.linalg.eig(A.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        stationary = np.maximum(stationary, 0)
    except Exception:
        stationary = pi

    # Assign labels
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
