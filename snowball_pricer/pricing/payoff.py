"""Basket-average snowball payoff (P2).

Contract (all economics in one place; no hardcoding in the engine):
- Basket: B(t) = sum_j w_j * S_j(t)/S_j(0), so B(0) = 1 by construction.
- Knock-out (autocall): observed every ``ko_obs_every_days`` trading days;
  first obs date with B >= ``ko_barrier`` redeems at
  ``notional * (1 + coupon_annual * t_ko)`` discounted to today.
- Knock-in: monitored every simulation step (at least daily); first touch of
  ``ki_barrier`` from above knocks in. Set ``ki_barrier <= 0`` to disable.
- Maturity (no KO): KI touched  -> ``notional * min(1, B(T))`` (principal at
  risk; the min() is economically redundant because the final date is itself
  a KO observation date, but it matches the P0 reference exactly);
  never KI -> ``notional * (1 + coupon_annual * T)``.

Brownian-bridge barrier corrections (optional, OFF by default):
- The contractual baseline monitors barriers exactly at their observation
  dates (monthly KO obs, daily KI) with no correction — this matches the P0
  reference and the term sheet.
- With ``ki_bridge``/``ko_bridge`` enabled, the engine instead approximates
  CONTINUOUS monitoring: P(touch | endpoints) via the Brownian-bridge formula
  with the instantaneous basket vol frozen per interval. This is a modeling
  variant (some desks mark KI as American-touch), NOT the contract. Note it
  creates a genuine local non-monotonicity of price vs spot at the KO
  barrier (touch-from-below > stay-above): discrete observation economics,
  documented in docs/p2_design.md.
- Basket vol: sigma_b^2 = x' C x where x are basket delta-weights and
  C_jk = sigma_j sigma_k rho_jk.

The correction uniforms come from a plain PRNG stream (seeded), not the
Sobol sequence: they add unbiased noise on top of the QMC driver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class UnderlyingSpec:
    """One basket constituent.

    ``surface``: a P1 surface object (VolSurface or compatible). None means
    "use the priced snapshot's surface" (the P2 default; per-name surfaces
    arrive with P3's correlation work). ``spot``/``rate``/``div_yield`` default
    to the surface's values when None.
    """

    name: str
    weight: float
    spot: Optional[float] = None
    rate: Optional[float] = None
    div_yield: Optional[float] = None
    surface: Optional[object] = None  # VolSurface-like; None -> snapshot surface


@dataclass(frozen=True)
class TermSheet:
    """Snowball economics. Barriers are fractions of the initial basket (=1)."""

    tenor_years: float = 2.0
    ko_barrier: float = 1.00
    ko_obs_every_days: int = 21
    ki_barrier: float = 0.80
    coupon_annual: float = 0.15
    notional: float = 1.0
    discount_rate: Optional[float] = None  # None -> first underlying's rate


def _basket_vol_from_perf(perf: np.ndarray, B: np.ndarray,
                         sigma: np.ndarray, weights: np.ndarray,
                         corr: np.ndarray) -> np.ndarray:
    """Basket vol from precomputed performance ``perf`` and basket ``B``.

    Single source of truth for the ``sqrt(y' C y)`` algebra; :func:`basket_vol`
    and the :func:`evaluate_snowball` hot loop both funnel through here so
    the loop does not recompute ``perf``/``B``.
    """
    x = (perf * weights) / B[:, None]          # basket delta weights
    y = x * sigma
    return np.sqrt(np.maximum(np.einsum("pj,jk,pk->p", y, corr, y), 0.0))


def basket_vol(sigma: np.ndarray, S: np.ndarray, spots: np.ndarray,
               weights: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Instantaneous basket vol per path. sigma, S: (n_paths, n_assets).

    P5: computed as ``sqrt(y' C y)`` with ``y = x * sigma`` (no (p,a,a)
    covariance cube materialized). Algebraically identical to the old
    ``x' (D C D) x`` form; FP association differs at ~1e-16.
    """
    perf = S / spots  # (p, a)
    B = np.maximum(perf @ weights, 1e-300)
    return _basket_vol_from_perf(perf, B, sigma, weights, corr)


def evaluate_snowball(driver, market, terms: TermSheet,
                      steps_cfg: Dict, corr_seed: int = 777,
                      norm_spots=None) -> Dict:
    """Stream batches from ``driver`` and accumulate snowball statistics.

    Returns dict with sum_pay, sum_pay_sq, n, ko_count (bridge-inclusive).

    ``norm_spots``: spots used to normalize performance (default: the
    market's own spots, so B(0) = 1). For spot-bump delta the caller passes
    the *base* spots here, which keeps barrier levels fixed in absolute
    terms while the simulated spot moves — that is the hedge delta. Without
    it the product would just be redefined (barriers as fractions of the new
    spot) and delta would be identically zero by homogeneity.
    """
    r_disc = terms.discount_rate if terms.discount_rate is not None else float(market.rates[0])
    T = terms.tenor_years
    df_T = float(np.exp(-r_disc * T))
    H_ki = terms.ki_barrier
    H_ko = terms.ko_barrier
    ki_on = H_ki > 0.0
    ki_bridge = steps_cfg.get("ki_bridge", False)
    ko_bridge = steps_cfg.get("ko_bridge", False)
    obs_every = terms.ko_obs_every_days * driver.steps_per_day
    dt = driver.dt
    norm = market.spots if norm_spots is None else np.asarray(norm_spots, dtype=float)
    B0 = float((market.spots / norm) @ market.weights)

    rng = np.random.default_rng(corr_seed)
    sum_pay = 0.0
    sum_pay_sq = 0.0
    n = 0
    ko_count = 0

    for batch in driver.batches(**steps_cfg["batch_kwargs"]):
        nb = batch.n_paths
        ki_hit = np.zeros(nb, dtype=bool)
        ko_idx = np.full(nb, -1, dtype=np.int64)  # 0-based obs index
        # Initial local vol at t=0: the bridge corrections on the first
        # interval need a basket vol; it is NOT zero (previous version left
        # these at 0, silently disabling the correction on interval 1).
        # P5: uses the precomputed t=0 weights (bitwise-identical lookup).
        S0_2d = np.broadcast_to(market.spots, (nb, len(market.spots))).copy()
        sigma0 = np.empty_like(S0_2d)
        tw = driver._tw
        for a, lv in enumerate(market.lvs):
            j0, j1, fT = tw[a]
            sigma0[:, a] = lv.local_vol_at_weights(np.zeros(nb), j0[0],
                                                   j1[0], fT[0])
        bv0_sq = basket_vol(sigma0, S0_2d, norm, market.weights,
                            market.corr) ** 2
        B_prev = np.full(nb, B0)
        bv2_prev = bv0_sq.copy()
        obs_B_prev = np.full(nb, B0)  # basket at previous KO obs (t=0 first)
        obs_bv2_prev = bv0_sq.copy()
        B_T = np.full(nb, B0)

        for i1, t_next, S, sigma in batch.steps():
            perf = S / norm
            B = perf @ market.weights
            B = np.maximum(B, 1e-300)
            # P5: reuse perf/B for the basket vol (no recompute inside
            # basket_vol); same numerics via _basket_vol_from_perf.
            bv = _basket_vol_from_perf(perf, B, sigma, market.weights,
                                       market.corr)
            bv2 = bv ** 2

            if ki_on:
                touched = B <= H_ki
                if ki_bridge:
                    m = (~ki_hit) & (B_prev > H_ki) & (B > H_ki) & (bv2_prev > 0) & (bv2 > 0)
                    if np.any(m):
                        num = ((np.log(B_prev[m]) - np.log(H_ki))
                               * (np.log(B[m]) - np.log(H_ki)))
                        den = 0.5 * (bv2_prev[m] + bv2[m]) * dt
                        p = np.exp(-2.0 * num / np.maximum(den, 1e-300))
                        ki_hit[m] |= rng.random(np.count_nonzero(m)) < np.minimum(p, 1.0)
                ki_hit |= touched

            if i1 % obs_every == 0:
                k = i1 // obs_every - 1  # 0-based
                hit_now = B >= H_ko
                first = (ko_idx < 0)
                if ko_bridge and np.any(first):
                    m = first & (obs_B_prev < H_ko) & (B < H_ko) & (obs_bv2_prev > 0) & (bv2 > 0)
                    if np.any(m):
                        dt_obs = obs_every * dt
                        num = ((np.log(H_ko) - np.log(obs_B_prev[m]))
                               * (np.log(H_ko) - np.log(B[m])))
                        den = 0.5 * (obs_bv2_prev[m] + bv2[m]) * dt_obs
                        p = np.exp(-2.0 * num / np.maximum(den, 1e-300))
                        bridged = rng.random(np.count_nonzero(m)) < np.minimum(p, 1.0)
                        idx = np.flatnonzero(m)[bridged]
                        ko_idx[idx] = k
                new_hits = first & (ko_idx < 0) & hit_now
                ko_idx[new_hits] = k
                obs_B_prev = B.copy()
                obs_bv2_prev = bv2.copy()

            B_prev = B
            bv2_prev = bv2
            B_T = B

        # --- vectorized payoff for the batch ---
        pay = np.empty(nb)
        ko_m = ko_idx >= 0
        t_ko = (ko_idx[ko_m] + 1) * terms.ko_obs_every_days / 252.0
        pay[ko_m] = (1.0 + terms.coupon_annual * t_ko) * np.exp(-r_disc * t_ko)
        rest = ~ko_m
        pay[rest] = np.where(ki_hit[rest],
                             np.minimum(B_T[rest], 1.0),
                             1.0 + terms.coupon_annual * T) * df_T
        pay *= terms.notional

        sum_pay += float(np.sum(pay))
        sum_pay_sq += float(np.sum(pay ** 2))
        n += nb
        ko_count += int(np.count_nonzero(ko_m))

    return {"sum_pay": sum_pay, "sum_pay_sq": sum_pay_sq, "n": n,
            "ko_count": ko_count}
