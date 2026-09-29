"""Correlation estimators for the snowball-pricer (P3).

Three estimators, three different jobs:

1. ``rolling_pearson`` / ``ewma_corr`` — historical (realized) correlation
   from constituent log-returns. Inputs to the P2 engine's ``corr`` matrix.
2. ``implied_correlation`` — forward-looking average correlation backed out
   of index vs constituent implied variances (a market-implied "diversification
   premium" gauge, useful as a stress anchor).
3. ``nearest_psd`` — repair helper: shocked or estimated matrices must be
   positive semi-definite for the Cholesky factor in the P2 engine.

All functions take/return plain numpy arrays; no internal state.
"""

from __future__ import annotations

from typing import List

import numpy as np


def nearest_psd(a: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return the nearest (in Frobenius sense, via eigenvalue clipping) PSD
    matrix with unit diagonal to ``a``.

    Steps: symmetrize → eigendecompose → clip eigenvalues below ``eps``
    → rescale diagonal back to 1. The unit-diagonal rescale is what makes
    the output a valid correlation matrix (1s on the diagonal), which is
    the form the P2 engine consumes.

    Idempotent: already-valid correlation matrices come back unchanged
    (up to numerical noise).
    """
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    assert a.shape == (n, n), "input must be square"
    a = 0.5 * (a + a.T)
    w, v = np.linalg.eigh(a)
    w = np.clip(w, eps, None)
    b = (v * w) @ v.T
    b = 0.5 * (b + b.T)
    d = np.sqrt(np.diag(b))
    out = b / np.outer(d, d)
    np.fill_diagonal(out, 1.0)
    return out


def rolling_pearson(returns: np.ndarray, window: int) -> np.ndarray:
    """Rolling-window Pearson correlation matrices.

    ``returns``: (n_obs, n_assets) log-returns. Returns
    (n_obs - window + 1, n_assets, n_assets); entry [t] uses
    ``returns[t:t+window]``. Rows are demeaned per window; zero-variance
    assets produce 0 off-diagonal correlation (not NaN).
    """
    r = np.asarray(returns, dtype=float)
    n_obs, n = r.shape
    assert n_obs >= window >= 2, "need at least `window` observations"
    n_win = n_obs - window + 1
    out = np.empty((n_win, n, n))
    for t in range(n_win):
        x = r[t : t + window] - r[t : t + window].mean(axis=0)
        cov = x.T @ x / (window - 1)
        sd = np.sqrt(np.diag(cov))
        with np.errstate(divide="ignore", invalid="ignore"):
            c = cov / np.outer(sd, sd)
        c[~np.isfinite(c)] = 0.0
        np.fill_diagonal(c, 1.0)
        out[t] = c
    return out


def ewma_corr(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics-style exponentially-weighted correlation matrix.

    Recurrence (per asset pair, starting from the sample moments of the
    first 10 observations):
        q[i,j,t] = lam * q[i,j,t-1] + (1-lam) * r[i,t] * r[j,t]
    correlation = q[i,j] / sqrt(q[i,i] * q[j,j]).

    ``lam=0.94`` is the RiskMetrics daily standard. Output is symmetric
    with unit diagonal; a ``nearest_psd`` pass is applied since the EWMA
    covariance estimate is PSD in exact arithmetic but can drift in
    floating point.
    """
    r = np.asarray(returns, dtype=float)
    n_obs, n = r.shape
    assert 0.0 < lam < 1.0, "lam must be in (0, 1)"
    assert n_obs >= 10, "need at least 10 observations for the warm start"
    seed = max(10, n_obs // 10)
    q = (r[:seed].T @ r[:seed]) / seed  # warm start: sample second moment
    w = 1.0 - lam
    for t in range(seed, n_obs):
        x = r[t][:, None]
        q = lam * q + w * (x @ x.T)
    d = np.sqrt(np.maximum(np.diag(q), 1e-300))
    corr = q / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return nearest_psd(corr)


def implied_correlation(index_var: float, comp_vars: np.ndarray,
                        weights: np.ndarray) -> float:
    """Average pairwise implied correlation from the index-vs-components
    variance identity.

    For a basket (index) I with weights w over components with implied
    variances σᵢ²:

        σ_I² = Σᵢ wᵢ² σᵢ² + Σ_{i≠j} wᵢ wⱼ σᵢ σⱼ ρᵢⱼ

    Assume a SINGLE average pairwise correlation ρ for all off-diagonal
    pairs (ρᵢⱼ = ρ). Solving:

        ρ = (σ_I² − Σᵢ wᵢ² σᵢ²) / (Σ_{i≠j} wᵢ wⱼ σᵢ σⱼ)

    Assumptions / limitations (documented, not hidden):
      - One average ρ describes every pair; pair structure is lost.
      - No idiosyncratic basis: index variance must be fully explained by
        the weighted components. If the denominator is ~0 or the formula
        yields |ρ| > 1, the data violates the model and we clip to
        [-0.99, 0.99] rather than raising — the caller should treat an
        extreme print as a signal, not a calibration.
      - Inputs should be same-tenor, same-convention implied variances
        (e.g. ATM IV²·T from the P1 surface at the same expiry).

    Returns the average implied pairwise correlation (float).
    """
    sigma_i = np.sqrt(np.maximum(np.asarray(comp_vars, dtype=float), 0.0))
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    assert len(sigma_i) == len(w), "lengths of comp_vars and weights differ"
    var_index = float(index_var)
    diag = float(np.sum((w * sigma_i) ** 2))
    cross = float(np.sum(np.outer(w * sigma_i, w * sigma_i))) - diag
    if abs(cross) < 1e-300:
        return 0.0
    rho = (var_index - diag) / cross
    return float(np.clip(rho, -0.99, 0.99))
