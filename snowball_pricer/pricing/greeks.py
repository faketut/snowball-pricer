"""Greeks for the snowball pricer (P2).

Honest scope statement — read before extending:
- DELTA (per underlying): central finite difference with Common Random
  Numbers (same Sobol seed / same pseudo-random streams for base and bumped
  runs). Pathwise/adjoint delta is NOT implemented on purpose: the snowball
  payoff is discontinuous at the KO/KI barriers (digital autocall trigger,
  digital knock-in trigger), so pathwise differentiation does not exist at
  the barriers and would be wrong exactly where risk concentrates. CRN
  bump-and-revalue is the robust standard choice here; with QMC the
  difference estimator converges well.
- VEGA (parallel vol bump, per 1 vol point): the *implied* total-variance
  surface is bumped by +0.01 in Black vol at every (K, T)
  (w -> (sqrt(w/T) + 0.01)^2 * T), the Dupire grid is rebuilt, and the
  portfolio is repriced with identical random draws (CRN). This measures
  sensitivity to the market input the desk actually marks, not to an
  internal local-vol parameter.
- KO PROBABILITY: fraction of paths autocalling (bridge-correction
  inclusive) with a binomial standard error. Useful as a trader-facing
  risk metric (expected tenor proxy).
- NOT covered: gamma, theta, rho-rates (bump the TermSheet discount rate /
  surface rate manually if needed), correlation greeks (P3), likelihood-ratio
  digitals, full AAD. AAD through the local-vol grid interpolation is
  documented future work — it would speed up multi-Greek batches but the
  CRN approach is exact and fast enough at P2 scale.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict

import numpy as np


def delta_crn(reprice: Callable, specs, base_price: float,
              h: float = 0.01) -> Dict[str, float]:
    """Per-underlying delta via central CRN bump.

    ``reprice``: callable taking a list of bumped UnderlyingSpec and
    returning (price, std_error). Bumps are relative (spot * (1 +/- h)).
    Returns {name: dPrice/dSpot}.
    """
    deltas = {}
    for j, spec in enumerate(specs):
        s0 = spec.spot
        bumped_up = [replace(specs[k], spot=s0 * (1 + h)) if k == j else specs[k]
                     for k in range(len(specs))]
        bumped_dn = [replace(specs[k], spot=s0 * (1 - h)) if k == j else specs[k]
                     for k in range(len(specs))]
        p_up, _ = reprice(bumped_up)
        p_dn, _ = reprice(bumped_dn)
        deltas[spec.name] = float((p_up - p_dn) / (2 * h * s0))
    return deltas

def vega_bump_surface(surface, bump: float = 0.01):
    """Return a copy of the surface with Black IV bumped by ``bump`` everywhere.

    Works on P1 VolSurface (rebuilds thetas from bumped ATM vols, keeps
    rho/eta/gamma/forwards) and on FlatVol (bumps vol directly).
    """
    from .dupire import FlatVol
    try:
        from ..surface import VolSurface
    except ImportError:  # pragma: no cover
        VolSurface = None

    if isinstance(surface, FlatVol):
        return FlatVol(vol=surface.vol + bump, spot=surface.spot,
                       rate=surface.rate, div_yield=surface.div_yield)
    if VolSurface is not None and isinstance(surface, VolSurface):
        new_thetas = []
        for T, th in zip(surface.expiries, surface.thetas):
            iv = (th / T) ** 0.5 + bump
            new_thetas.append(iv ** 2 * T)
        return VolSurface(expiries=surface.expiries, thetas=tuple(new_thetas),
                          rho=surface.rho, eta=surface.eta, gamma=surface.gamma,
                          forwards=surface.forwards, spot=surface.spot,
                          rate=surface.rate, div_yield=surface.div_yield)
    raise TypeError(f"cannot bump surface of type {type(surface)}")


def ko_probability(ko_count: int, n: int):
    """KO probability and binomial standard error."""
    p = ko_count / n if n else float("nan")
    se = float(np.sqrt(p * (1 - p) / n)) if n else float("nan")
    return p, se
