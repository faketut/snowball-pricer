"""Monte Carlo path simulator: Sobol QMC + Brownian bridge + correlation (P2).

Brownian driver
---------------
Per asset, per path we need Brownian increments on a fine time grid. With
QMC the *ordering* of the normals across Sobol dimensions matters: the
Brownian-bridge construction puts the highest-variance points (terminal
value, then successive bisection midpoints) on the leading Sobol dimensions,
which is where QMC uniformity is best. Any grid size works (bisection order,
not just powers of two).

Multi-asset: independent bridges per asset, then increments are correlated
per (path, step) with the Cholesky factor of the input correlation matrix:
``dW_corr[p, :, i] = L @ dW[p, :, i]``. Marginals are preserved.

Antithetic fallback: when ``qmc=False`` the second half of the batch mirrors
the first (``z -> -z``), giving the classic antithetic variance reduction.

Batching: paths are drawn in batches; one scrambled Sobol sequence object is
advanced across batches. Note this means the point set depends (mildly) on
``batch_size`` — documented, accepted; benchmarks fix the batch size.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc


def bridge_order(n_steps: int) -> List[Tuple[int, int, int]]:
    """Breadth-first bisection order: list of (mid, left, right) grid indices.

    The terminal index ``n_steps`` is handled separately (first dimension).
    Length of the returned list is ``n_steps - 1`` (all interior points).
    """
    order: List[Tuple[int, int, int]] = []
    dq: deque = deque([(0, n_steps)])
    while dq:
        a, b = dq.popleft()
        if b - a < 2:
            continue
        m = (a + b) // 2
        order.append((m, a, b))
        dq.append((a, m))
        dq.append((m, b))
    return order


def brownian_increments(n_steps: int, n_paths: int, n_assets: int, dt: float,
                        qmc_on: bool, seed: int) -> np.ndarray:
    """Correlated Brownian increments, shape (n_paths, n_assets, n_steps).

    Returned increments are *independent* across assets here; the caller
    applies the Cholesky factor (kept separate for testability).
    """
    d = n_assets * n_steps
    if qmc_on:
        sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
        u = sampler.random(n_paths)
        np.clip(u, 1e-12, 1.0 - 1e-12, out=u)
        z = ndtri(u).reshape(n_paths, n_assets, n_steps)
    else:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n_paths, n_assets, n_steps))
        half = n_paths // 2
        z[half:half + half] = -z[:half]  # antithetic mirror (even part)

    W = np.zeros((n_paths, n_assets, n_steps + 1))
    total_t = n_steps * dt
    W[:, :, n_steps] = np.sqrt(total_t) * z[:, :, 0]
    _fill_bridge(W, z, n_steps, dt)
    dW = np.diff(W, axis=2)
    del W, z
    return dW


def _fill_bridge(W: np.ndarray, z: np.ndarray, n_steps: int, dt: float) -> None:
    """Fill interior Brownian-bridge points of ``W`` in place.

    ``W``: (n_paths, n_assets, n_steps+1), zeros with the terminal column
    already set; ``z``: (n_paths, n_assets, n_steps) normals in bridge order
    (``z[..., 0]`` = terminal).

    Deliberately a scalar Python loop, not a vectorized gather/scatter: the
    (n_paths, n_assets) working set stays cache-resident, while the
    vectorized form streams ~3x more memory and measured SLOWER on this VM
    (documented in docs/p5_report.md). Identical to the pre-P5 loop.
    """
    for j, (m, a, b) in enumerate(bridge_order(n_steps), start=1):
        ta, tb, tm = a * dt, b * dt, m * dt
        wmean = ((tb - tm) * W[:, :, a] + (tm - ta) * W[:, :, b]) / (tb - ta)
        wvar = (tm - ta) * (tb - tm) / (tb - ta)
        W[:, :, m] = wmean + np.sqrt(wvar) * z[:, :, j]


def correlate_increments(dW: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Apply Cholesky correlation per (path, step)."""
    L = np.linalg.cholesky(corr)
    return np.einsum("ab,pbi->pai", L, dW)


@dataclass
class SimMarket:
    """Everything the simulator needs about the underlyings."""

    spots: np.ndarray    # (n_assets,)
    rates: np.ndarray    # (n_assets,) risk-free rates for drift
    divs: np.ndarray     # (n_assets,) dividend yields
    weights: np.ndarray  # (n_assets,) normalized basket weights
    corr: np.ndarray     # (n_assets, n_assets)
    lvs: tuple           # LocalVolSurface per asset (duck-typed: .local_vol(k, T))


class PathBatchDriver:
    """One batch of paths: pre-drawn increments, streamed step by step.

    Usage::

        driver = PathBatchDriver(market, tenor_years, steps_per_day,
                                 n_paths, qmc=True, seed=..., batch_size=...)
        for batch in driver.batches():
            S = batch.spots2d()          # (n_paths, n_assets), at t=0
            for i, t_next, S_new, sigma in batch.steps():
                ...                      # sigma: local vol used on [t_i, t_next]
    """

    def __init__(self, market: SimMarket, tenor_years: float,
                 steps_per_day: int = 1):
        self.market = market
        self.tenor_years = float(tenor_years)
        self.n_days = int(round(self.tenor_years * 252))
        self.steps_per_day = int(steps_per_day)
        self.n_steps = self.n_days * self.steps_per_day
        self.dt = 1.0 / (252 * self.steps_per_day)
        if self.n_steps < 1:
            raise ValueError("tenor too short for the given steps_per_day")
        # --- P5 hot-loop precomputations (per driver, done once) ---
        mkt = market
        n_a = len(mkt.spots)
        self._rd = mkt.rates - mkt.divs                     # (n_assets,)
        # Forward curve at each interval START t_i = i*dt: identical to the
        # per-step `spots * exp((rates-divs) * t)` the old loop computed.
        t_starts = np.arange(self.n_steps) * self.dt        # (n_steps,)
        self._F = (mkt.spots[None, :]
                   * np.exp(self._rd[None, :] * t_starts[:, None]))
        # Local-vol T-weights at each interval start, per asset surface.
        self._tw = [lv.t_weights(t_starts) for lv in mkt.lvs]
        # dW buffer reused across steps() calls within a batch is allocated
        # per batch (shape depends on batch size); nothing else per-step.

    def batches(self, n_paths: int, qmc_on: bool, seed: int,
                batch_size: int) -> Iterator["_Batch"]:
        n_assets = len(self.market.spots)
        sobol = qmc.Sobol(d=n_assets * self.n_steps, scramble=True,
                          seed=seed) if qmc_on else None
        remaining = n_paths
        while remaining > 0:
            nb = min(batch_size, remaining)
            if qmc_on:
                u = sobol.random(nb)
                np.clip(u, 1e-12, 1.0 - 1e-12, out=u)
                z = ndtri(u).reshape(nb, n_assets, self.n_steps)
            else:
                rng = np.random.default_rng(seed + n_paths - remaining)
                z = rng.standard_normal((nb, n_assets, self.n_steps))
                half = nb // 2
                z[half:half + half] = -z[:half]
            W = np.zeros((nb, n_assets, self.n_steps + 1))
            total_t = self.n_steps * self.dt
            W[:, :, self.n_steps] = np.sqrt(total_t) * z[:, :, 0]
            _fill_bridge(W, z, self.n_steps, self.dt)
            dW = np.diff(W, axis=2)
            del W, z
            dW = correlate_increments(dW, self.market.corr)
            yield _Batch(self, dW, nb)
            remaining -= nb


class _Batch:
    def __init__(self, driver: PathBatchDriver, dW: np.ndarray, n_paths: int):
        self.driver = driver
        self.dW = dW
        self.n_paths = n_paths

    def steps(self) -> Iterator[Tuple[int, float, np.ndarray, np.ndarray]]:
        """Yield (step_index_1based, t_next, S_new, sigma_used).

        ``S_new`` is a fresh (n_paths, n_assets) array each step; ``sigma_used``
        is the local vol applied over [t_i, t_next] (frozen at interval start).

        P5: the per-step body is algebraically identical to the old version —
        same forwards (precomputed), same local-vol lookup (precomputed
        T-weights, bitwise-identical lerp), same Euler update op order.
        """
        drv = self.driver
        mkt = drv.market
        dt = drv.dt
        rd = drv._rd
        F = drv._F
        tw = drv._tw
        lvs = mkt.lvs
        dW = self.dW
        n_a = len(mkt.spots)
        S = np.broadcast_to(mkt.spots, (self.n_paths, n_a)).copy()
        sigma = np.empty_like(S)
        for i in range(drv.n_steps):
            kk = np.log(S / F[i])
            for a, lv in enumerate(lvs):
                j0, j1, fT = tw[a]
                sigma[:, a] = lv.local_vol_at_weights(kk[:, a], j0[i],
                                                      j1[i], fT[i])
            drift = (rd - 0.5 * sigma ** 2) * dt
            # dW are true Brownian increments with Var = dt (NOT standard
            # normals), so no extra sqrt(dt) here.
            S = S * np.exp(drift + sigma * dW[:, :, i])
            yield i + 1, (i + 1) * dt, S, sigma


def simulate_terminal(market: SimMarket, tenor_years: float,
                      steps_per_day: int, n_paths: int,
                      qmc_on: bool, seed: int) -> np.ndarray:
    """Terminal spot values, shape (n_paths, n_assets). For tests/vanilla."""
    driver = PathBatchDriver(market, tenor_years, steps_per_day)
    S_T = None
    for batch in driver.batches(n_paths, qmc_on, seed, batch_size=n_paths):
        for _, _, S_new, _ in batch.steps():
            S_T = S_new
    return S_T
