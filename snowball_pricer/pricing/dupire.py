"""Dupire local volatility from an implied total-variance surface (P2).

Given an arbitrage-free implied total-variance surface w(k, T) (k = log-strike),
Gatheral's form of Dupire's formula gives the local variance directly:

    sigma_loc^2(k, T) = (dw/dT) / D

    D = 1 - (k/w)(dw/dk) + 1/4(-1/4 - 1/w + (k/w)^2)(dw/dk)^2 + 1/2(d2w/dk2)

Two evaluation paths:
- Analytic (preferred): for P1's SSVI ``VolSurface`` the k-derivatives are
  closed-form in the SSVI parameters and dw/dT = (dw/dtheta)(dtheta/dT) with
  dtheta/dT exact on P1's piecewise-linear theta term structure. This avoids
  finite-difference noise, which matters because the denominator D involves
  second derivatives.
- Finite-difference fallback: for any other surface exposing
  ``total_var(k, T)`` (e.g. the ``FlatVol`` test shim). Central differences.

Because P1 guarantees an arbitrage-free surface, the Dupire numerator and
denominator are positive and the local variance is positive by construction.
Two safety rails remain (counted in diagnostics, not silent):
- floor: local vol floored at ``vol_floor`` (default 0.5%) — binds only from
  numerical noise or flat extrapolation beyond the last expiry;
- cap: local vol capped at ``vol_cap`` (default 150%) — SSVI wings can imply
  very high local vol far out of the money; the cap keeps the MC stable.

The surface is precomputed on a (k, T) grid; the simulator queries it with
vectorized bilinear interpolation (``local_vol(k, T)``). Outside the grid the
query clamps to the edge (documented; the grid spans k in [-4, 4], which
covers an 80% knock-in barrier at k ~ -0.22 with wide margin).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Test/shim surface: constant Black vol (NOT for production use)
# ---------------------------------------------------------------------------

class FlatVol:
    """Constant-volatility surface exposing the P1 surface protocol.

    Exists so the P2 engine/tests can run without a calibration step
    (vanilla-vs-Black validation, P0 cross-check). Production pricing must
    use a calibrated arbitrage-free surface from P1.
    """

    def __init__(self, vol: float, spot: float = 100.0, rate: float = 0.03,
                 div_yield: float = 0.0):
        self.vol = float(vol)
        self.spot = float(spot)
        self.rate = float(rate)
        self.div_yield = float(div_yield)

    def theta_at(self, T):
        T = np.asarray(T, dtype=float)
        return self.vol ** 2 * T

    def forward_at(self, T):
        T = np.asarray(T, dtype=float)
        return self.spot * np.exp((self.rate - self.div_yield) * T)

    def total_var(self, k, T):
        k = np.asarray(k, dtype=float)
        T = np.asarray(T, dtype=float)
        kb, Tb = np.broadcast_arrays(k, T)
        return np.full(kb.shape, self.vol ** 2, dtype=float) * Tb


# ---------------------------------------------------------------------------
# SSVI analytic local variance (Gatheral form of Dupire)
# ---------------------------------------------------------------------------

def _theta_prime(expiries: np.ndarray, thetas: np.ndarray, T: np.ndarray) -> np.ndarray:
    """dtheta/dT for P1's piecewise-linear theta term structure.

    Linear from the origin below the first expiry, linear between expiries,
    flat beyond the last expiry (matches VolSurface.theta_at).
    """
    T = np.asarray(T, dtype=float)
    out = np.empty_like(T)
    below = T < expiries[0]
    out[below] = thetas[0] / expiries[0]
    # At and beyond the last expiry use the last segment's slope (NOT the flat
    # extrapolation of theta_at): the MC grid ends at t_max <= last expiry and
    # the flat tail would imply zero local vol on the final slice, an artifact
    # of extrapolation, not of the market. Documented in docs/p2_design.md.
    last_slope = (thetas[-1] - thetas[-2]) / (expiries[-1] - expiries[-2]) \
        if len(expiries) > 1 else thetas[0] / expiries[0]
    rest = ~below
    if np.any(rest):
        idx = np.searchsorted(expiries, T[rest], side="right") - 1
        idx = np.clip(idx, 0, len(expiries) - 2)
        dT = expiries[idx + 1] - expiries[idx]
        out[rest] = (thetas[idx + 1] - thetas[idx]) / dT
        out[rest & (T >= expiries[-1])] = last_slope
    return out


def ssvi_local_var(k, T, rho: float, eta: float, gamma: float,
                   expiries, thetas) -> np.ndarray:
    """Analytic Dupire local variance for an SSVI surface.

    k, T broadcast; expiries/thetas are the ATM total-variance term structure.
    Returns local *variance* (may be tiny-negative from roundoff; the caller
    floors it).
    """
    k = np.asarray(k, dtype=float)
    T = np.asarray(T, dtype=float)
    kb, Tb = np.broadcast_arrays(k, T)
    expiries = np.asarray(expiries, dtype=float)
    thetas = np.asarray(thetas, dtype=float)

    # ATM total variance: linear from origin below first expiry (matches
    # VolSurface.theta_at), flat beyond the last expiry.
    theta = np.interp(Tb.ravel(), expiries, thetas).reshape(Tb.shape)
    below = Tb < expiries[0]
    theta = np.where(below, thetas[0] * Tb / expiries[0], theta)

    dtheta_dT = _theta_prime(expiries, thetas, Tb)

    # SSVI shape
    phi = eta / (np.power(theta, gamma) * np.power(1.0 + theta, 1.0 - gamma))
    u = phi * kb
    c = 1.0 - rho ** 2
    s = np.sqrt((u + rho) ** 2 + c)
    B = 1.0 + rho * u + s
    w = 0.5 * theta * B

    # k-derivatives (closed form)
    A = rho + (u + rho) / s                      # dB/du
    dw_dk = 0.5 * theta * phi * A
    d2w_dk2 = 0.5 * theta * phi ** 2 * (c / s ** 3)

    # theta-derivative (closed form)
    dphi_dtheta = phi * (-gamma / theta - (1.0 - gamma) / (1.0 + theta))
    dw_dtheta = 0.5 * B + 0.5 * theta * dphi_dtheta * kb * A
    dw_dT = dw_dtheta * dtheta_dT

    # Gatheral denominator
    k_over_w = kb / w
    D = (1.0 - k_over_w * dw_dk
         + 0.25 * (-0.25 - 1.0 / w + k_over_w ** 2) * dw_dk ** 2
         + 0.5 * d2w_dk2)
    return dw_dT / D


def fd_local_var(surface, k, T, hk: float = 1e-3, hT: float = 1e-4) -> np.ndarray:
    """Finite-difference Dupire local variance for a generic surface.

    Fallback for surfaces without analytic derivatives (e.g. FlatVol).
    """
    k = np.asarray(k, dtype=float)
    T = np.asarray(T, dtype=float)
    w = np.asarray(surface.total_var(k, T), dtype=float)
    dw_dk = (np.asarray(surface.total_var(k + hk, T), dtype=float)
             - np.asarray(surface.total_var(k - hk, T), dtype=float)) / (2 * hk)
    d2w_dk2 = (np.asarray(surface.total_var(k + hk, T), dtype=float)
               - 2.0 * w
               + np.asarray(surface.total_var(k - hk, T), dtype=float)) / hk ** 2
    Tp = np.maximum(T + hT, 1e-8)
    Tm = np.maximum(T - hT, 1e-8)
    dw_dT = (np.asarray(surface.total_var(k, Tp), dtype=float)
             - np.asarray(surface.total_var(k, Tm), dtype=float)) / (Tp - Tm)
    k_over_w = k / np.maximum(w, 1e-12)
    D = (1.0 - k_over_w * dw_dk
         + 0.25 * (-0.25 - 1.0 / np.maximum(w, 1e-12) + k_over_w ** 2) * dw_dk ** 2
         + 0.5 * d2w_dk2)
    return dw_dT / np.maximum(D, 1e-12)


# ---------------------------------------------------------------------------
# Gridded local-vol surface (what the simulator queries)
# ---------------------------------------------------------------------------

@dataclass
class LocalVolDiagnostics:
    frac_floored: float = 0.0
    frac_capped: float = 0.0
    min_raw_vol: float = float("nan")
    max_raw_vol: float = float("nan")


class LocalVolSurface:
    """Precomputed Dupire local-vol grid with vectorized bilinear lookup.

    Built from any surface exposing ``total_var(k, T)``; uses the analytic
    SSVI path when the surface carries SSVI parameters (P1 VolSurface).
    """

    def __init__(self, surface, t_max: float,
                 n_k: int = 161, n_t: int = 61,
                 k_min: float = -4.0, k_max: float = 4.0,
                 t_min: float = 1.0 / 365.0,
                 vol_floor: float = 0.005, vol_cap: float = 1.5):
        self.surface = surface
        self.t_max = float(t_max)
        self.k_min, self.k_max = k_min, k_max
        self.t_min = t_min
        self.vol_floor = vol_floor
        self.vol_cap = vol_cap

        k_grid = np.linspace(k_min, k_max, n_k)
        # denser near t=0 where local vol moves fastest: sqrt spacing
        t_grid = t_min + (t_max - t_min) * np.linspace(0.0, 1.0, n_t) ** 2
        kk, tt = np.meshgrid(k_grid, t_grid, indexing="ij")  # (n_k, n_t)

        if hasattr(surface, "rho") and hasattr(surface, "eta") and hasattr(surface, "gamma"):
            raw_var = ssvi_local_var(kk, tt, surface.rho, surface.eta,
                                     surface.gamma, surface.expiries,
                                     surface.thetas)
        else:
            raw_var = fd_local_var(surface, kk, tt)

        raw_vol = np.sqrt(np.maximum(raw_var, 0.0))
        diag = LocalVolDiagnostics()
        diag.min_raw_vol = float(np.min(raw_vol))
        diag.max_raw_vol = float(np.max(raw_vol))
        floored = raw_vol < vol_floor
        capped = raw_vol > vol_cap
        diag.frac_floored = float(np.mean(floored))
        diag.frac_capped = float(np.mean(capped))
        self.diagnostics = diag

        self._k_grid = k_grid
        self._t_grid = t_grid
        self._vol = np.clip(raw_vol, vol_floor, vol_cap)

    def local_vol(self, k, T) -> np.ndarray:
        """Vectorized bilinear interpolation. k, T broadcast together."""
        k = np.asarray(k, dtype=float)
        T = np.asarray(T, dtype=float)
        kb, Tb = np.broadcast_arrays(k, T)
        kc = np.clip(kb, self.k_min, self.k_max)
        Tc = np.clip(Tb, self.t_min, self.t_max)

        kg, tg = self._k_grid, self._t_grid
        dk = kg[1] - kg[0]
        fi = (kc - kg[0]) / dk
        i0 = np.clip(np.floor(fi).astype(int), 0, len(kg) - 2)
        fk = fi - i0

        # T grid is non-uniform: searchsorted then lerp
        j1 = np.clip(np.searchsorted(tg, Tc, side="left"), 1, len(tg) - 1)
        j0 = j1 - 1
        denom = tg[j1] - tg[j0]
        fT = np.where(denom > 0, (Tc - tg[j0]) / np.maximum(denom, 1e-300), 0.0)

        v = self._vol
        v00 = v[i0, j0]
        v10 = v[i0 + 1, j0]
        v01 = v[i0, j1]
        v11 = v[i0 + 1, j1]
        return ((1 - fk) * (1 - fT) * v00 + fk * (1 - fT) * v10
                + (1 - fk) * fT * v01 + fk * fT * v11)
