"""Volatility surface calibration (P1).

Two-stage design (rationale in docs/p1_design.md):
  Stage A — per-expiry raw-SVI fit (Gatheral 2004). Robust, expiry-local;
      gives a stable ATM total-variance seed per expiry.
  Stage B — global SSVI fit (Gatheral & Jacquier). The published surface is
      SSVI by construction, which admits checkable no-arbitrage sufficient
      conditions; the numeric gate in ``arbitrage.validate_surface`` is the
      final acceptance check before any snapshot is published.

Conventions:
- k = log-moneyness vs the forward, k = ln(K / F_T).
- w(k, T) = total implied variance = iv^2 * T.
- SSVI: w(k, theta) = theta/2 * (1 + rho*phi(theta)*k
                       + sqrt((phi(theta)*k + rho)^2 + (1 - rho^2)))
  with phi(theta) = eta / (theta^gamma * (1 + theta)^(1 - gamma)).
  theta_T is the ATM total variance for expiry T (w(0, T) = theta_T).
- Wings: SSVI wings are linear in |k| with slope theta*phi(theta)*(1 +/- rho)/2.
  No ad-hoc extrapolation is used outside the calibrated strike range — the
  SSVI formula itself is the extrapolator, and its asymptotic slope is bounded
  by Lee's moment formula (checked numerically in arbitrage.check_wings).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from . import iv as _iv
from .arbitrage import ArbitrageViolation, validate_surface

if TYPE_CHECKING:  # tick.py imports ssvi math from this module; keep runtime acyclic
    from .tick import OptionQuote


# ---------------------------------------------------------------------------
# Smile math
# ---------------------------------------------------------------------------

def ssvi_phi(theta, eta: float, gamma: float):
    """SSVI shape function. Vectorized in theta."""
    theta = np.asarray(theta, dtype=float)
    return eta / (np.power(theta, gamma) * np.power(1.0 + theta, 1.0 - gamma))


def ssvi_total_var(k, theta, rho: float, eta: float, gamma: float):
    """SSVI total variance w(k; theta). k and theta broadcast."""
    k = np.asarray(k, dtype=float)
    theta = np.asarray(theta, dtype=float)
    phi = ssvi_phi(theta, eta, gamma)
    pk = phi * k
    return theta * 0.5 * (
        1.0 + rho * pk + np.sqrt((pk + rho) ** 2 + (1.0 - rho**2))
    )


def raw_svi_total_var(k, a: float, b: float, rho: float, m: float, sigma: float):
    """Raw SVI (Gatheral 2004) total variance."""
    k = np.asarray(k, dtype=float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


# ---------------------------------------------------------------------------
# Stage A: per-expiry raw-SVI fit
# ---------------------------------------------------------------------------

def fit_raw_svi(
    k: np.ndarray, w: np.ndarray, weights: np.ndarray
) -> Dict[str, float]:
    """Least-squares raw-SVI fit in total-variance space.

    Returns dict(a, b, rho, m, sigma). Raises ValueError if the fit fails.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    weights = np.asarray(weights, dtype=float)
    sw = np.sqrt(np.maximum(weights, 1e-12))

    i_atm = int(np.argmin(np.abs(k)))
    w_atm = float(w[i_atm])
    x0 = np.array([w_atm * 0.5, 0.3, -0.4, 0.0, 0.3])
    lo = np.array([1e-9, 1e-8, -0.999, -2.0, 1e-3])
    hi = np.array([5.0, 5.0, 0.999, 2.0, 3.0])

    def resid(x):
        return sw * (raw_svi_total_var(k, *x) - w)

    res = least_squares(resid, x0, bounds=(lo, hi), max_nfev=4000)
    if not res.success:
        raise ValueError(f"raw-SVI fit failed: {res.message}")
    a, b, rho, m, sigma = (float(v) for v in res.x)
    return {"a": a, "b": b, "rho": rho, "m": m, "sigma": sigma}


def pava_isotonic(y: Sequence[float]) -> np.ndarray:
    """Pool-Adjacent-Violators Algorithm: smallest non-decreasing sequence
    dominating y in L2 (used to enforce no calendar-spread arbitrage on the
    ATM total-variance term structure)."""
    y = [float(v) for v in y]
    blocks: List[List[float]] = [[v] for v in y]
    i = 0
    while i < len(blocks) - 1:
        avg_i = sum(blocks[i]) / len(blocks[i])
        avg_j = sum(blocks[i + 1]) / len(blocks[i + 1])
        if avg_i <= avg_j:
            i += 1
        else:
            blocks[i] = blocks[i] + blocks[i + 1]
            del blocks[i + 1]
            if i > 0:
                i -= 1
    out: List[float] = []
    for b in blocks:
        avg = sum(b) / len(b)
        out.extend([avg] * len(b))
    return np.array(out)


# ---------------------------------------------------------------------------
# Stage B: global SSVI fit
# ---------------------------------------------------------------------------

def fit_ssvi_global(
    k: np.ndarray,
    theta_of_point: np.ndarray,
    w: np.ndarray,
    weights: np.ndarray,
) -> Tuple[float, float, float]:
    """Fit global SSVI (rho, eta, gamma) given per-point ATM total variance.

    Adds a soft penalty residual enforcing eta*(1+|rho|) <= 1.9, a
    conservative subset of the Gatheral-Jacquier no-butterfly-arbitrage
    sufficient conditions (the numeric Durrleman gate in arbitrage.py is the
    binding acceptance check).
    """
    k = np.asarray(k, dtype=float)
    theta_of_point = np.asarray(theta_of_point, dtype=float)
    w = np.asarray(w, dtype=float)
    sw = np.sqrt(np.maximum(np.asarray(weights, dtype=float), 1e-12))

    x0 = np.array([-0.4, 0.8, 0.5])
    lo = np.array([-0.95, 0.05, 0.05])
    hi = np.array([0.95, 3.0, 0.95])

    def resid(x):
        rho, eta, gamma = x
        model = ssvi_total_var(k, theta_of_point, rho, eta, gamma)
        r = sw * (model - w)
        pen = max(0.0, eta * (1.0 + abs(rho)) - 1.9)
        return np.append(r, [50.0 * pen])

    res = least_squares(resid, x0, bounds=(lo, hi), max_nfev=4000)
    if not res.success:
        raise ValueError(f"SSVI global fit failed: {res.message}")
    return float(res.x[0]), float(res.x[1]), float(res.x[2])


# ---------------------------------------------------------------------------
# Immutable published surface
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VolSurface:
    """Immutable, versioned-by-the-store volatility surface snapshot payload.

    All arrays are tuples (immutable). Queries interpolate the ATM
    total-variance term structure linearly in T; the SSVI formula with global
    (rho, eta, gamma) supplies the smile at any (k, T).
    """

    expiries: Tuple[float, ...]   # calibrated expiries, ascending
    thetas: Tuple[float, ...]     # ATM total variance per expiry, non-decreasing
    rho: float
    eta: float
    gamma: float
    forwards: Tuple[float, ...]   # per-expiry forward F_T, aligned with expiries
    spot: float
    rate: float
    div_yield: float

    def theta_at(self, T):
        """ATM total variance at T. Vectorized; linear from the origin below
        the first calibrated expiry, flat beyond the last."""
        scalar = np.ndim(T) == 0
        T_arr = np.atleast_1d(np.asarray(T, dtype=float))
        if np.any(T_arr <= 0.0):
            raise ValueError("T must be positive")
        Ts = np.array(self.expiries)
        th = np.array(self.thetas)
        out = np.interp(T_arr, Ts, th)  # flat outside [Tmin, Tmax]
        below = T_arr < Ts[0]
        out = np.where(below, th[0] * T_arr / Ts[0], out)
        return float(out[0]) if scalar else out

    def forward_at(self, T):
        """Forward at T. Vectorized; flat extrapolation outside."""
        scalar = np.ndim(T) == 0
        T_arr = np.atleast_1d(np.asarray(T, dtype=float))
        out = np.interp(T_arr, np.array(self.expiries), np.array(self.forwards))
        return float(out[0]) if scalar else out

    def total_var(self, k, T: float):
        return ssvi_total_var(k, self.theta_at(T), self.rho, self.eta, self.gamma)

    def iv(self, k, T: float):
        w = np.asarray(self.total_var(k, T), dtype=float)
        return np.sqrt(np.maximum(w, 0.0) / T)

    def iv_from_strike(self, K: float, T: float) -> float:
        F = self.forward_at(T)
        return float(self.iv(math.log(K / F), T))


# ---------------------------------------------------------------------------
# Builder: quotes -> calibrated, arbitrage-checked surface
# ---------------------------------------------------------------------------

def _invert_expiry(T: float, quotes: List[OptionQuote]) -> Dict:
    """Invert one expiry's quotes to (k, iv_mid, weight) points.

    Forward: put-call parity at the strike closest to the spot-implied
    forward among strikes quoting both a call and a put; falls back to the
    spot-implied forward if no pair exists.
    """
    by_strike: Dict[float, Dict[str, OptionQuote]] = {}
    for q in quotes:
        d = by_strike.setdefault(q.strike, {})
        d["call" if q.is_call else "put"] = q
    q0 = quotes[0]
    df = math.exp(-q0.rate * T)
    fwd_spot = q0.forward_spot()

    fwd = None
    paired = [
        (K, d["call"], d["put"])
        for K, d in by_strike.items()
        if "call" in d and "put" in d
    ]
    if paired:
        K_star = min(paired, key=lambda t: abs(t[0] - fwd_spot))[0]
        d = by_strike[K_star]
        fwd = _iv.forward_from_parity(d["call"].mid, d["put"].mid, K_star, df)
    if fwd is None or fwd <= 0.0:
        fwd = fwd_spot

    ks, ivs, weights = [], [], []
    for K, d in sorted(by_strike.items()):
        for side in ("call", "put"):
            q = d.get(side)
            if q is None:
                continue
            k = math.log(K / fwd)
            iv_mid = _iv.implied_vol_safe(q.mid, fwd, K, T, df, q.is_call)
            if not math.isfinite(iv_mid) or iv_mid <= 0.0:
                continue
            iv_bid = _iv.implied_vol_safe(q.bid, fwd, K, T, df, q.is_call)
            iv_ask = _iv.implied_vol_safe(q.ask, fwd, K, T, df, q.is_call)
            if math.isfinite(iv_bid) and math.isfinite(iv_ask) and iv_ask > iv_bid:
                wgt = 1.0 / max(iv_ask - iv_bid, 1e-3)
            else:
                wgt = 0.1  # single-sided / degenerate spread: down-weighted
            ks.append(k)
            ivs.append(iv_mid)
            weights.append(wgt)
    weights = np.array(weights)
    if weights.sum() > 0:
        weights = weights / weights.mean()  # normalize: mean weight = 1
    return {
        "T": T,
        "forward": fwd,
        "k": np.array(ks),
        "iv": np.array(ivs),
        "w": np.array(ivs) ** 2 * T,
        "weights": weights,
        "n": len(ks),
    }


def build_surface(
    quotes: Sequence[OptionQuote],
    asof: float,
    *,
    min_strikes: int = 5,
) -> Tuple[VolSurface, Dict]:
    """Calibrate an arbitrage-checked SSVI surface from raw quotes.

    Raises:
        ValueError: too few / bad quotes, or a calibration stage failed.
        ArbitrageViolation: the fitted surface failed the numeric
            no-arbitrage gate (never publish a torn surface).
    """
    groups: Dict[float, List[OptionQuote]] = {}
    for q in quotes:
        if q.is_valid():
            groups.setdefault(q.expiry, []).append(q)
    if not groups:
        raise ValueError("build_surface: no valid quotes")

    expiries = sorted(groups)
    per_exp: Dict[float, Dict] = {}
    for T in expiries:
        data = _invert_expiry(T, groups[T])
        if data["n"] < min_strikes:
            raise ValueError(
                f"build_surface: expiry T={T} has {data['n']} invertible "
                f"quotes (< {min_strikes})"
            )
        per_exp[T] = data

    # Stage A: raw-SVI per expiry -> ATM total-variance seeds.
    raw_params: Dict[float, Dict[str, float]] = {}
    theta_seed = []
    for T in expiries:
        d = per_exp[T]
        rp = fit_raw_svi(d["k"], d["w"], d["weights"])
        raw_params[T] = rp
        theta_seed.append(float(raw_svi_total_var(0.0, **rp)))
    thetas = pava_isotonic(theta_seed)  # calendar-arbitrage-free term structure

    # Stage B: global SSVI fit.
    ks = np.concatenate([per_exp[T]["k"] for T in expiries])
    ws = np.concatenate([per_exp[T]["w"] for T in expiries])
    wts = np.concatenate([per_exp[T]["weights"] for T in expiries])
    theta_idx = {T: i for i, T in enumerate(expiries)}
    theta_of_point = np.concatenate(
        [np.full(per_exp[T]["n"], thetas[theta_idx[T]]) for T in expiries]
    )
    rho, eta, gamma = fit_ssvi_global(ks, theta_of_point, ws, wts)

    q0 = quotes[0]
    surface = VolSurface(
        expiries=tuple(expiries),
        thetas=tuple(float(t) for t in thetas),
        rho=rho,
        eta=eta,
        gamma=gamma,
        forwards=tuple(float(per_exp[T]["forward"]) for T in expiries),
        spot=float(np.mean([q.underlying_price for q in quotes])),
        rate=float(q0.rate),
        div_yield=float(q0.div_yield),
    )

    report = validate_surface(surface)
    if not report["all_ok"]:
        raise ArbitrageViolation(report)

    # Diagnostics: calibration RMSE in vol points on the input mids.
    iv_model = np.sqrt(
        np.maximum(ssvi_total_var(ks, theta_of_point, rho, eta, gamma), 0.0)
        / np.concatenate(
            [np.full(per_exp[T]["n"], T) for T in expiries]
        )
    )
    iv_obs = np.concatenate([per_exp[T]["iv"] for T in expiries])
    rmse_iv = float(np.sqrt(np.mean((iv_model - iv_obs) ** 2)))

    diagnostics = {
        "asof": asof,
        "n_quotes": len(quotes),
        "n_used": int(sum(per_exp[T]["n"] for T in expiries)),
        "expiries": expiries,
        "thetas": [float(t) for t in thetas],
        "rho": rho,
        "eta": eta,
        "gamma": gamma,
        "rmse_iv": rmse_iv,
        "arbitrage": report,
    }
    return surface, diagnostics
