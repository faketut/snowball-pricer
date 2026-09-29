"""No-arbitrage validation gates (P1).

A surface is publishable only if all three checks pass. These are numeric
gates on dense grids — they bind regardless of which parameterization was
fitted (SSVI sufficient conditions are enforced softly at fit time; these
checks are the hard gate).

Checks:
- calendar: total variance w(k, T) non-decreasing in T for every k.
- butterfly: Durrleman (2003) condition g(k) >= 0 for every expiry, where
      g(k) = (1 - k*w'(k)/(2*w(k)))^2 - w'(k)^2/4 * (1/w(k) + 1/4) + w''(k)/2
  with central finite differences on the surface's total_var.
- wings: asymptotic linear slope of w vs |k| bounded by Lee's moment
  formula: limsup w/|k| <= 2 as |k| -> infinity (checked at |k| = 9..10).
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np


class ArbitrageViolation(Exception):
    """Raised when a calibrated surface fails the no-arbitrage gate.

    Carries the full check report; the surface must be quarantined
    (never published) and the failure logged with the report.
    """

    def __init__(self, report: Dict):
        self.report = report
        failed = [k for k, v in report.items()
                  if k != "all_ok" and not v[0]]
        super().__init__(
            f"surface failed no-arbitrage gate: {failed}; "
            f"report={ {k: (v[0], v[1]) for k, v in report.items() if k != 'all_ok'} }"
        )


def durrleman_g(w_func: Callable[[float], float], k: float, h: float = 1e-4) -> float:
    """Durrleman's g(k) for a total-variance smile w(k). g >= 0 <=> no
    butterfly arbitrage (for that smile slice)."""
    w0 = float(w_func(k))
    if not np.isfinite(w0) or w0 <= 0.0:
        return float("nan")
    wp = (float(w_func(k + h)) - float(w_func(k - h))) / (2.0 * h)
    wpp = (float(w_func(k + h)) - 2.0 * w0 + float(w_func(k - h))) / (h * h)
    if not (np.isfinite(wp) and np.isfinite(wpp)):
        return float("nan")
    return (
        (1.0 - k * wp / (2.0 * w0)) ** 2
        - (wp**2) / 4.0 * (1.0 / w0 + 0.25)
        + wpp / 2.0
    )


def check_butterfly_smile(
    w_func: Callable[[float], float],
    k_grid: np.ndarray,
    tol: float = -1e-6,
) -> Tuple[bool, float]:
    """Butterfly check on one smile slice. Returns (ok, worst_g)."""
    gs = np.array([durrleman_g(w_func, float(k)) for k in k_grid])
    gs = gs[np.isfinite(gs)]
    if gs.size == 0:
        return False, float("nan")
    worst = float(np.min(gs))
    return worst >= tol, worst


def check_calendar_surface(
    surface, k_grid: np.ndarray, T_grid: np.ndarray, tol: float = -1e-9
) -> Tuple[bool, float]:
    """Calendar check: w(k, T) non-decreasing in T for every k.
    Returns (ok, worst_increment)."""
    worst = float("inf")
    for k in k_grid:
        w = np.asarray(surface.total_var(float(k), T_grid), dtype=float)
        inc = np.diff(w)
        if inc.size:
            worst = min(worst, float(np.min(inc)))
    if not np.isfinite(worst):
        return False, float("nan")
    return worst >= tol, worst


def check_butterfly_surface(
    surface, k_grid: np.ndarray, expiries, tol: float = -1e-6
) -> Tuple[bool, float]:
    """Butterfly check per calibrated expiry. Returns (ok, worst_g)."""
    worst = float("inf")
    for T in expiries:
        T = float(T)
        w_func = lambda k, T=T: float(surface.total_var(k, T))  # noqa: E731
        ok, g = check_butterfly_smile(w_func, k_grid, tol=tol)
        if not np.isfinite(g):
            return False, float("nan")
        worst = min(worst, g)
    return worst >= tol, worst


def check_wings(surface, expiries, tol: float = 1e-6) -> Tuple[bool, float]:
    """Lee's moment-formula bound: asymptotic slope of w vs |k| must be <= 2.
    Returns (ok, worst_excess_over_2)."""
    worst_excess = 0.0
    for T in expiries:
        T = float(T)
        slope_r = float(surface.total_var(10.0, T)) - float(surface.total_var(9.0, T))
        slope_l = float(surface.total_var(-9.0, T)) - float(surface.total_var(-10.0, T))
        for s in (slope_r, slope_l):
            if not np.isfinite(s):
                return False, float("nan")
            worst_excess = max(worst_excess, s - 2.0)
    return worst_excess <= tol, worst_excess


def validate_surface(
    surface,
    k_grid: np.ndarray = None,
    T_grid: np.ndarray = None,
) -> Dict:
    """Run all gates. Returns report dict; report['all_ok'] is the publish
    decision."""
    if k_grid is None:
        k_grid = np.linspace(-1.5, 1.0, 61)
    expiries = tuple(float(t) for t in surface.expiries)
    if T_grid is None:
        T_grid = np.linspace(expiries[0], expiries[-1], 25)

    cal_ok, cal_worst = check_calendar_surface(surface, k_grid, T_grid)
    bfly_ok, bfly_worst = check_butterfly_surface(surface, k_grid, expiries)
    wing_ok, wing_worst = check_wings(surface, expiries)
    all_ok = bool(cal_ok and bfly_ok and wing_ok)
    return {
        "calendar": (cal_ok, cal_worst),
        "butterfly": (bfly_ok, bfly_worst),
        "wings": (wing_ok, wing_worst),
        "all_ok": all_ok,
    }
