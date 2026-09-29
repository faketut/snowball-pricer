"""Implied-volatility inversion (P1).

Assumptions (documented, see SPEC.md §2):
- European exercise. SSE ETF options are European, so no de-Americanization
  is applied. American underlyings would need a de-Americanization step
  (e.g. binomial-tree implied vol) — explicitly out of scope for P1.
- Forward-first: the inverter takes the forward F as an input. The pipeline
  derives F from put-call parity when a call/put pair is quoted at the same
  strike/expiry, else from spot/rate/div (see ``forward_from_parity`` and
  ``surface._invert_expiry``).

Method: Newton-Raphson on the Black price with vega, falling back to
bisection on [1e-9, 20]. Price tolerance 1e-10.
"""
from __future__ import annotations

import math

_SQRT_2PI = math.sqrt(2.0 * math.pi)


class IVError(Exception):
    """Base class for IV inversion failures."""


class DegenerateQuoteError(IVError):
    """Quote violates no-arbitrage bounds or carries no time value to invert."""


class NoConvergenceError(IVError):
    """Neither Newton nor bisection converged (should not happen)."""


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _npdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def black_price(
    F: float, K: float, T: float, vol: float, df: float, is_call: bool
) -> float:
    """Undiscounted-forward Black price, discounted by df."""
    if vol <= 0.0:
        intrinsic = max(F - K, 0.0) if is_call else max(K - F, 0.0)
        return df * intrinsic
    sqt = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * vol * vol * T) / (vol * sqt)
    d2 = d1 - vol * sqt
    if is_call:
        return df * (F * _ncdf(d1) - K * _ncdf(d2))
    return df * (K * _ncdf(-d2) - F * _ncdf(-d1))


def black_vega(F: float, K: float, T: float, vol: float, df: float) -> float:
    sqt = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * vol * vol * T) / (vol * sqt)
    return F * df * _npdf(d1) * sqt


def forward_from_parity(call_mid: float, put_mid: float, K: float, df: float) -> float:
    """F = K + (C - P) / df from put-call parity on mids."""
    return K + (call_mid - put_mid) / df


def implied_vol(
    price: float,
    F: float,
    K: float,
    T: float,
    df: float,
    is_call: bool,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """Invert a European option price to Black implied vol.

    Raises:
        DegenerateQuoteError: non-positive inputs, price above the no-arb
            cap (df*F for calls, df*K for puts), or price strictly below
            intrinsic value.
        NoConvergenceError: both Newton and bisection failed.
    """
    if not (price > 0.0 and F > 0.0 and K > 0.0 and T > 0.0 and df > 0.0):
        raise DegenerateQuoteError(
            f"non-positive input: price={price} F={F} K={K} T={T} df={df}"
        )
    intrinsic = df * (max(F - K, 0.0) if is_call else max(K - F, 0.0))
    cap = df * (F if is_call else K)
    if price > cap * (1.0 + 1e-9):
        raise DegenerateQuoteError(
            f"price {price} above no-arb cap {cap} (crossed/stale quote?)"
        )
    if price < intrinsic - 1e-9:
        raise DegenerateQuoteError(
            f"price {price} below intrinsic {intrinsic}"
        )
    if price <= intrinsic + tol:
        # Time value below the inversion tolerance: vol is numerically
        # unidentifiable (deep ITM / extremely short-dated). Return 0 rather
        # than fit noise.
        return 0.0

    # Brenner-Subrahmanyam-style seed, clipped to a sane range.
    v = min(2.0, max(0.02, math.sqrt(2.0 * math.pi / T) * price / (F * df)))
    for _ in range(max_iter):
        p = black_price(F, K, T, v, df, is_call)
        diff = p - price
        if abs(diff) <= tol:
            return v
        vega = black_vega(F, K, T, v, df)
        if vega < 1e-14:
            break
        v -= diff / vega
        if not (1e-9 < v < 20.0):
            break
    else:
        # loop exhausted without break -> treat as non-converged below
        pass

    # Bisection fallback: f(lo) <= 0 <= f(hi) by the intrinsic/cap checks.
    lo, hi = 1e-9, 20.0
    f_lo = black_price(F, K, T, lo, df, is_call) - price
    f_hi = black_price(F, K, T, hi, df, is_call) - price
    if f_lo > 0.0 or f_hi < 0.0:
        raise NoConvergenceError("bisection bracket invalid")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f = black_price(F, K, T, mid, df, is_call) - price
        if abs(f) <= tol:
            return mid
        if f > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-13:
            return mid
    raise NoConvergenceError("bisection did not converge")


def implied_vol_safe(
    price: float,
    F: float,
    K: float,
    T: float,
    df: float,
    is_call: bool,
) -> float:
    """Non-raising variant: returns NaN when the quote cannot be inverted."""
    try:
        return implied_vol(price, F, K, T, df, is_call)
    except IVError:
        return math.nan
