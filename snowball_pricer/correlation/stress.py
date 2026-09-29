"""Correlation stress scenarios + sensitivity table (P3).

Scenarios shock the off-diagonal of a base correlation matrix, then pass
through ``nearest_psd`` (clipping/shocks can push a matrix out of the PSD
cone, which the P2 engine's Cholesky factor requires).

``sensitivity_table`` reprices a 3-asset basket-AVERAGE snowball under the
base matrix and each scenario with the SAME engine seed, so the reported
deltas are driven by the correlation shock, not by MC noise
(common-random-numbers by construction).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .estimate import nearest_psd
from ..pricing.dupire import FlatVol
from ..pricing.engine import EngineConfig, price
from ..pricing.payoff import TermSheet, UnderlyingSpec
from ..snapshot import VolSurfaceSnapshot

SCENARIOS = ("plus_0.2", "minus_0.2", "crisis_1.0", "dispersion_0.0")


def shock_corr(corr: np.ndarray, scenario: str) -> np.ndarray:
    """Apply a named correlation stress to ``corr``.

    Scenarios:
      - ``"plus_0.2"``: off-diagonal ρ → ρ + 0.2
      - ``"minus_0.2"``: off-diagonal ρ → ρ − 0.2
      - ``"crisis_1.0"``: off-diagonal ρ → 0.95 (correlation spike)
      - ``"dispersion_0.0"``: off-diagonal ρ → 0.05 (dispersion trade /
        decorrelated regime)

    Off-diagonals are clipped to [-0.99, 0.99], the diagonal stays 1.0,
    and the result is PSD-repaired. Raises ValueError on unknown scenario.
    """
    base = np.asarray(corr, dtype=float)
    n = base.shape[0]
    assert base.shape == (n, n), "corr must be square"
    if scenario == "plus_0.2":
        off = base + 0.2
    elif scenario == "minus_0.2":
        off = base - 0.2
    elif scenario == "crisis_1.0":
        off = np.full_like(base, 0.95)
    elif scenario == "dispersion_0.0":
        off = np.full_like(base, 0.05)
    else:
        raise ValueError(f"unknown correlation scenario: {scenario!r}")
    off = np.clip(off, -0.99, 0.99)
    np.fill_diagonal(off, 1.0)
    return nearest_psd(off)


@dataclass
class ScenarioRow:
    scenario: str
    rho_offdiag: float       # mean off-diagonal of the shocked matrix
    price: float             # per unit notional
    std_error: float
    delta_price_bps: float   # vs base scenario, bps of notional
    ko_prob: float
    delta_ko_pp: float       # vs base scenario, percentage points
    seconds: float


def _flat_snapshot(vol: float = 0.25, spot: float = 100.0,
                   rate: float = 0.03, div: float = 0.0) -> VolSurfaceSnapshot:
    surf = FlatVol(vol=vol, spot=spot, rate=rate, div_yield=div)
    return VolSurfaceSnapshot(surface=surf, version=1, asof=0.0,
                              source="p3-stress", n_quotes=0, rmse_iv=0.0)


def sensitivity_table(
    base_corr: np.ndarray,
    scenarios: Sequence[str] = SCENARIOS,
    vol: float = 0.25,
    weights: Sequence[float] = (1 / 3, 1 / 3, 1 / 3),
    cfg: EngineConfig | None = None,
) -> List[ScenarioRow]:
    """Reprice a 3-asset basket-average snowball under base + shocked corr.

    Default contract: TermSheet() defaults (2Y, monthly KO @100%, daily KI
    @80%, 15% coupon), FlatVol 25% surface, equal weights, spot 100, rate 3%.
    ``cfg`` defaults to 40k QMC paths, no Greeks, fixed seed — same seed for
    every scenario (CRN deltas). Returns rows with base first, then each
    scenario in ``scenarios`` order.
    """
    base_corr = np.asarray(base_corr, dtype=float)
    if cfg is None:
        cfg = EngineConfig(n_paths=40_000, qmc=True, seed=20260929,
                           batch_size=20_000, compute_greeks=False)
    terms = TermSheet()
    snapshot = _flat_snapshot(vol=vol)
    underlyings = [
        UnderlyingSpec(name=f"A{i + 1}", weight=float(w))
        for i, w in enumerate(weights)
    ]
    rows: List[ScenarioRow] = []
    mats = {"base": nearest_psd(base_corr)}
    mats.update({s: shock_corr(base_corr, s) for s in scenarios})

    base_price = base_ko = None
    for name, mat in mats.items():
        t0 = time.perf_counter()
        res = price(snapshot, terms, underlyings, cfg=cfg, corr=mat)
        dt = time.perf_counter() - t0
        if name == "base":
            base_price, base_ko = res.price, res.ko_prob
        off = mat[np.triu_indices(len(mat), k=1)].mean()
        rows.append(ScenarioRow(
            scenario=name,
            rho_offdiag=float(off),
            price=res.price,
            std_error=res.std_error,
            delta_price_bps=float((res.price - base_price) * 1e4),
            ko_prob=res.ko_prob,
            delta_ko_pp=float((res.ko_prob - base_ko) * 100.0),
            seconds=dt,
        ))
    return rows


def format_table(rows: Sequence[ScenarioRow]) -> str:
    """Render sensitivity rows as an aligned text table."""
    head = (f"{'scenario':<14}{'rho_off':>9}{'price':>10}"
            f"{'SE':>9}{'dPrice_bps':>11}{'KO%':>8}{'dKO_pp':>8}{'sec':>7}")
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r.scenario:<14}{r.rho_offdiag:>9.3f}{r.price:>10.5f}"
            f"{r.std_error:>9.1e}{r.delta_price_bps:>+11.1f}"
            f"{r.ko_prob * 100:>8.2f}{r.delta_ko_pp:>+8.2f}{r.seconds:>7.1f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    n = 3
    base = np.full((n, n), 0.5)
    np.fill_diagonal(base, 1.0)
    rows = sensitivity_table(base)
    print(format_table(rows))
