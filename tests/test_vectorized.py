"""P5 vectorization regression tests.

1. Fixed-seed equivalence: the vectorized engine must reproduce the
   pre-P5 reference prices (recorded 2026-09-29 from the unvectorized code,
   seed=5, 100k QMC paths, batch_size=100k) to ~1e-9 relative. The P5
   rewrite keeps the Sobol draws and every formula identical, so any
   deviation above FP noise (~1e-13 observed) is a real numerics bug —
   do NOT widen these tolerances; investigate instead.
2. Timing: the canonical benchmark workload must clear a paths/sec floor
   derived from the pre-P5 baseline (9,078 paths/sec, PERF.md). The floor
   is set at 11,000 paths/sec (~1.2x the old baseline) with margin for VM
   noise; the measured P5 factor on this VM is ~1.6-1.9x.
3. Unit equivalence of the two new fast paths (LV lookup, bridge fill).
"""

import time

import numpy as np
import pytest

from snowball_pricer.pricing.dupire import FlatVol, LocalVolSurface
from snowball_pricer.pricing.engine import EngineConfig, price
from snowball_pricer.pricing.mc import _fill_bridge, bridge_order
from snowball_pricer.pricing.payoff import TermSheet, UnderlyingSpec, basket_vol
from snowball_pricer.snapshot import VolSurfaceSnapshot

# Pre-P5 reference values (unvectorized code, 2026-09-29).
REF_PRICE_1A = 0.985908899218
REF_KO_1A = 0.88463
REF_PRICE_2A = 0.991773469635  # rho=0.5 basket
REF_SEED = 5
REF_N = 100_000


def _snap():
    surf = FlatVol(vol=0.25, spot=100.0, rate=0.03, div_yield=0.0)
    return VolSurfaceSnapshot(surface=surf, version=1, asof=0.0,
                              source="test", n_quotes=0, rmse_iv=0.0)


def _cfg(n=REF_N):
    return EngineConfig(n_paths=n, qmc=True, seed=REF_SEED, steps_per_day=1,
                        batch_size=min(n, 100_000), compute_greeks=False)


def test_reference_price_single_asset():
    res = price(_snap(), TermSheet(), [UnderlyingSpec(name="S", weight=1.0)],
                _cfg())
    assert res.price == pytest.approx(REF_PRICE_1A, rel=1e-9)
    assert res.ko_prob == pytest.approx(REF_KO_1A, abs=1e-8)


def test_reference_price_two_asset_correlated():
    specs = [UnderlyingSpec(name="A", weight=0.6),
             UnderlyingSpec(name="B", weight=0.4)]
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    res = price(_snap(), TermSheet(), specs, _cfg(), corr=corr)
    assert res.price == pytest.approx(REF_PRICE_2A, rel=1e-9)


def test_paths_per_sec_improved():
    # Timing gate: must beat the pre-P5 baseline (9,078 paths/sec) with
    # margin. Uses 50k paths to keep the suite fast.
    n = 50_000
    t0 = time.perf_counter()
    res = price(_snap(), TermSheet(), [UnderlyingSpec(name="S", weight=1.0)],
                _cfg(n))
    dt = time.perf_counter() - t0
    pps = n / dt
    assert res.n_paths == n
    assert pps > 11_000, f"only {pps:,.0f} paths/sec — vectorization regressed?"


def test_local_vol_fast_path_matches():
    rng = np.random.default_rng(0)
    lv = LocalVolSurface(FlatVol(vol=0.25), t_max=2.0)
    k = rng.uniform(-2, 2, size=(2000,))
    t = rng.uniform(0, 2, size=(2000,))
    j0, j1, fT = lv.t_weights(t)
    assert np.abs(lv.local_vol(k, t)
                  - lv.local_vol_at_weights(k, j0, j1, fT)).max() == 0.0


def test_bridge_fill_matches_scalar_loop():
    rng = np.random.default_rng(1)
    n_steps, dt = 63, 1 / 252
    z = rng.standard_normal((11, 2, n_steps))
    W_ref = np.zeros((11, 2, n_steps + 1))
    W_ref[:, :, n_steps] = np.sqrt(n_steps * dt) * z[:, :, 0]
    for j, (m, a, b) in enumerate(bridge_order(n_steps), start=1):
        ta, tb, tm = a * dt, b * dt, m * dt
        wmean = ((tb - tm) * W_ref[:, :, a]
                 + (tm - ta) * W_ref[:, :, b]) / (tb - ta)
        wvar = (tm - ta) * (tb - tm) / (tb - ta)
        W_ref[:, :, m] = wmean + np.sqrt(wvar) * z[:, :, j]
    W_new = np.zeros((11, 2, n_steps + 1))
    W_new[:, :, n_steps] = np.sqrt(n_steps * dt) * z[:, :, 0]
    _fill_bridge(W_new, z, n_steps, dt)
    assert np.abs(W_ref - W_new).max() == 0.0


def test_basket_vol_no_cube_regression():
    # basket_vol is the public entry; the hot loop uses _basket_vol_from_perf.
    # They must agree exactly (same inputs -> same outputs).
    from snowball_pricer.pricing.payoff import _basket_vol_from_perf
    rng = np.random.default_rng(2)
    S = rng.uniform(50, 150, size=(500, 3))
    sigma = rng.uniform(0.1, 0.5, size=(500, 3))
    spots = np.array([100.0, 100.0, 100.0])
    w = np.array([0.5, 0.3, 0.2])
    corr = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
    a = basket_vol(sigma, S, spots, w, corr)
    perf = S / spots
    B = np.maximum(perf @ w, 1e-300)
    b = _basket_vol_from_perf(perf, B, sigma, w, corr)
    assert np.abs(a - b).max() == 0.0
