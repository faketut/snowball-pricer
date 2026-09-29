"""P2 benchmarks: pricing throughput and QMC-vs-MC efficiency.

Measures (single-asset canonical 2Y snowball, flat 25% vol, daily steps):
- wall time for 100k and 1M paths (no Greeks), paths/sec
- QMC (Sobol+bridge) vs antithetic-MC variance ratio at fixed N
- Dupire grid build time (SSVI surface)
Results are appended to PERF.md by hand after review.
"""
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from snowball_pricer.pricing.dupire import FlatVol, LocalVolSurface
from snowball_pricer.pricing.engine import EngineConfig, price
from snowball_pricer.pricing.payoff import TermSheet, UnderlyingSpec
from snowball_pricer.snapshot import VolSurfaceSnapshot


def make_snap():
    surf = FlatVol(vol=0.25, spot=100.0, rate=0.03, div_yield=0.0)
    return VolSurfaceSnapshot(surface=surf, version=1, asof=0.0,
                              source="bench", n_quotes=0, rmse_iv=0.0)


def main():
    snap = make_snap()
    terms = TermSheet()
    specs = [UnderlyingSpec(name="S", weight=1.0)]

    for n in (100_000, 1_000_000):
        cfg = EngineConfig(n_paths=n, qmc=True, seed=5, steps_per_day=1,
                           batch_size=100_000, compute_greeks=False)
        t0 = time.perf_counter()
        res = price(snap, terms, specs, cfg)
        dt = time.perf_counter() - t0
        print(f"N={n:>8d}: price={res.price:.6f} se={res.std_error:.2e} "
              f"wall={dt:7.1f}s paths/sec={n / dt:,.0f}", flush=True)

    # QMC vs antithetic MC efficiency at fixed N
    n = 50_000
    ses = {}
    for qmc, tag in ((True, "qmc"), (False, "mc")):
        cfg = EngineConfig(n_paths=n, qmc=qmc, seed=9, steps_per_day=1,
                           batch_size=25_000, compute_greeks=False)
        t0 = time.perf_counter()
        res = price(snap, terms, specs, cfg)
        dt = time.perf_counter() - t0
        ses[tag] = res.std_error
        print(f"{tag}: se={res.std_error:.2e} wall={dt:.1f}s", flush=True)
    print(f"efficiency ratio (se_mc/se_qmc)^2 = {(ses['mc'] / ses['qmc']) ** 2:.1f}x",
          flush=True)

    # Dupire grid build on the synthetic SSVI surface
    from snowball_pricer.surface import VolSurface
    import math
    truth = dict(thetas=(0.0121, 0.0200, 0.0361, 0.0648),
                 expiries=(0.25, 0.5, 1.0, 2.0), rho=-0.45, eta=0.8,
                 gamma=0.5, spot=100.0, rate=0.03, div_yield=0.01)
    fwds = tuple(truth["spot"] * math.exp((truth["rate"] - truth["div_yield"]) * T)
                 for T in truth["expiries"])
    surf = VolSurface(expiries=truth["expiries"], thetas=truth["thetas"],
                      rho=truth["rho"], eta=truth["eta"], gamma=truth["gamma"],
                      forwards=fwds, spot=truth["spot"], rate=truth["rate"],
                      div_yield=truth["div_yield"])
    t0 = time.perf_counter()
    lv = LocalVolSurface(surf, t_max=2.0)
    print(f"Dupire grid build (161x61, analytic SSVI): "
          f"{time.perf_counter() - t0:.2f}s", flush=True)
    print("diag:", lv.diagnostics, flush=True)


if __name__ == "__main__":
    main()
