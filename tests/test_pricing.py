"""P2 pricing engine tests.

(a) vanilla call/put vs Black-Scholes under flat vol (validates the
    simulator + discounting; Dupire is identity here by construction);
(b) flat-vol snowball vs the P0 reference price (cross-check);
(c) KO probability vs analytic single-barrier formula (degenerate 1-obs case);
(d) Greeks sanity (delta bounds, vega sign);
(e) convergence: std_error shrinks ~1/sqrt(N).
Plus unit tests for bridge ordering and the Dupire grid itself.
"""

import math

import numpy as np
import pytest
from scipy.stats import norm

from snowball_pricer.pricing.dupire import (
    FlatVol,
    LocalVolSurface,
    fd_local_var,
    ssvi_local_var,
)
from snowball_pricer.pricing.engine import (
    EngineConfig,
    price,
    price_vanilla,
)
from snowball_pricer.pricing.mc import bridge_order
from snowball_pricer.pricing.payoff import TermSheet, UnderlyingSpec, basket_vol
from snowball_pricer.snapshot import VolSurfaceSnapshot
from snowball_pricer.surface import VolSurface


# ---------------------------------------------------------------- helpers ---

def flat_snapshot(vol=0.25, spot=100.0, rate=0.03, div=0.0) -> VolSurfaceSnapshot:
    surf = FlatVol(vol=vol, spot=spot, rate=rate, div_yield=div)
    return VolSurfaceSnapshot(surface=surf, version=1, asof=0.0,
                              source="test", n_quotes=0, rmse_iv=0.0)


def ssvi_surface() -> VolSurface:
    truth = dict(thetas=(0.0121, 0.0200, 0.0361, 0.0648),
                 expiries=(0.25, 0.5, 1.0, 2.0), rho=-0.45, eta=0.8,
                 gamma=0.5, spot=100.0, rate=0.03, div_yield=0.01)
    fwds = tuple(truth["spot"] * math.exp((truth["rate"] - truth["div_yield"]) * T)
                 for T in truth["expiries"])
    return VolSurface(expiries=truth["expiries"], thetas=truth["thetas"],
                      rho=truth["rho"], eta=truth["eta"], gamma=truth["gamma"],
                      forwards=fwds, spot=truth["spot"], rate=truth["rate"],
                      div_yield=truth["div_yield"])


def bs_price(spot, strike, T, vol, rate, div, kind):
    F = spot * math.exp((rate - div) * T)
    df = math.exp(-rate * T)
    s = vol * math.sqrt(T)
    d1 = (math.log(F / strike) + 0.5 * s * s) / s
    d2 = d1 - s
    if kind == "call":
        return df * (F * norm.cdf(d1) - strike * norm.cdf(d2))
    return df * (strike * norm.cdf(-d2) - F * norm.cdf(-d1))


# ---------------------------------------------------------------- unit -------

def test_bridge_order_covers_interior_breadth_first():
    order = bridge_order(8)
    mids = [m for m, _, _ in order]
    assert sorted(mids) == list(range(1, 8))  # all interior points, once each
    assert mids[0] == 4  # coarsest first
    assert set(mids[1:3]) == {2, 6}  # next level
    assert len(bridge_order(504)) == 503


def test_dupire_recovers_flat_vol():
    lv = LocalVolSurface(FlatVol(vol=0.25), t_max=2.0)
    k = np.linspace(-3, 3, 61)
    T = np.linspace(0.01, 2.0, 20)
    kk, tt = np.meshgrid(k, T, indexing="ij")
    v = lv.local_vol(kk, tt)
    assert np.abs(v - 0.25).max() < 1e-9


def test_dupire_ssvi_positive_and_matches_fd():
    surf = ssvi_surface()
    lv = LocalVolSurface(surf, t_max=2.0)
    assert not np.isnan(lv._vol).any()
    assert (lv._vol > 0).all()
    assert lv.diagnostics.frac_floored == 0.0  # floor must not bind on arb-free surface
    k = np.linspace(-2, 2, 41)
    T = np.array([0.1, 0.3, 0.7, 1.5, 1.999])
    kk, tt = np.meshgrid(k, T, indexing="ij")
    va = ssvi_local_var(kk, tt, surf.rho, surf.eta, surf.gamma,
                        surf.expiries, surf.thetas)
    vf = fd_local_var(surf, kk, tt)
    assert np.max(np.abs(va - vf)) < 1e-4


def test_basket_vol_single_asset_equals_local_vol():
    rng = np.random.default_rng(0)
    S = rng.uniform(50, 150, size=(1000, 1))
    sigma = rng.uniform(0.1, 0.5, size=(1000, 1))
    bv = basket_vol(sigma, S, np.array([100.0]), np.array([1.0]),
                    np.eye(1))
    assert np.abs(bv - sigma[:, 0]).max() < 1e-12


# ------------------------------------------------- (a) vanilla vs Black -----

@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("strike", [90.0, 100.0, 110.0])
def test_vanilla_vs_black(kind, strike):
    snap = flat_snapshot(vol=0.25, rate=0.03, div=0.0)
    specs = [UnderlyingSpec(name="S", weight=1.0)]
    cfg = EngineConfig(n_paths=100_000, qmc=True, seed=7, steps_per_day=1,
                       batch_size=25_000, compute_greeks=False)
    r = price_vanilla(snap, strike, 1.0, kind, specs, cfg)
    ref = bs_price(100.0, strike, 1.0, 0.25, 0.03, 0.0, kind)
    assert abs(r["price"] - ref) < 0.0015, (r["price"], ref)  # 15 bps


# ------------------------------------------------- (b) P0 cross-check --------

def test_snowball_flat_vol_matches_p0():
    # P0 contract: S0=100, T=2Y, r=3%, flat 25% vol, coupon 15% p.a.,
    # monthly KO @100%, daily-close KI @80%. Reference: 0.98699 (N=10k, se 13bps).
    snap = flat_snapshot(vol=0.25, rate=0.03, div=0.0)
    terms = TermSheet(tenor_years=2.0, ko_barrier=1.00, ko_obs_every_days=21,
                      ki_barrier=0.80, coupon_annual=0.15, notional=1.0,
                      discount_rate=0.03)
    specs = [UnderlyingSpec(name="S", weight=1.0)]
    cfg = EngineConfig(n_paths=150_000, qmc=True, seed=11, steps_per_day=1,
                       batch_size=30_000, ki_bridge=False, ko_bridge=False,
                       compute_greeks=False)
    res = price(snap, terms, specs, cfg)
    assert abs(res.price - 0.98699) < 0.0035, res.price  # 35 bps incl. MC noise


# -------------------------------------- (c) KO prob vs analytic barrier ------

def _ko_test_setup(vol=0.20, rate=0.03, T=1.0, H=1.20, ko_bridge=False):
    snap = flat_snapshot(vol=vol, rate=rate, div=0.0)
    terms = TermSheet(tenor_years=T, ko_barrier=H, ko_obs_every_days=252,
                      ki_barrier=0.0, coupon_annual=0.15, notional=1.0,
                      discount_rate=rate)
    specs = [UnderlyingSpec(name="S", weight=1.0)]
    cfg = EngineConfig(n_paths=150_000, qmc=True, seed=13, steps_per_day=1,
                       batch_size=30_000, ki_bridge=False, ko_bridge=ko_bridge,
                       compute_greeks=False)
    return snap, terms, specs, cfg


def test_ko_probability_discrete_vs_digital():
    # Degenerate contractual case: one KO observation at maturity, no bridge.
    # KO prob = P(S_T >= H), lognormal tail.
    vol, rate, T, H = 0.20, 0.03, 1.0, 1.20
    snap, terms, specs, cfg = _ko_test_setup(ko_bridge=False)
    res = price(snap, terms, specs, cfg)
    mu = rate - 0.5 * vol ** 2
    analytic = norm.cdf((mu * T - math.log(H)) / (vol * math.sqrt(T)))
    assert abs(res.ko_prob - analytic) / analytic < 0.05, (res.ko_prob, analytic)


def test_ko_probability_bridge_vs_touch_analytic():
    # Continuous-touch variant: ko_bridge=True must recover the
    # reflection-principle touch probability. Validates the bridge math.
    vol, rate, T, H = 0.20, 0.03, 1.0, 1.20
    snap, terms, specs, cfg = _ko_test_setup(ko_bridge=True)
    res = price(snap, terms, specs, cfg)
    b = math.log(H)
    mu = rate - 0.5 * vol ** 2
    sigt = vol * math.sqrt(T)
    analytic = (norm.cdf((-b + mu * T) / sigt)
                + math.exp(2 * mu * b / vol ** 2) * norm.cdf((-b - mu * T) / sigt))
    assert abs(res.ko_prob - analytic) / analytic < 0.05, (res.ko_prob, analytic)


# ------------------------------------------------- (d) Greeks sanity ---------

def test_greeks_sanity():
    snap = flat_snapshot(vol=0.25, rate=0.03, div=0.0)
    terms = TermSheet()  # canonical 2Y snowball
    specs = [UnderlyingSpec(name="S", weight=1.0)]
    cfg = EngineConfig(n_paths=30_000, qmc=True, seed=17, steps_per_day=1,
                       batch_size=15_000, compute_greeks=True)
    res = price(snap, terms, specs, cfg)
    d = res.delta["S"]
    assert 0.0 < d < 1.5, d  # long basket exposure, bounded
    assert res.vega < 0.0, res.vega  # snowball is short vol (P0: -0.195%/pt)
    assert 0.0 < res.ko_prob < 1.0


# ------------------------------------------------- (e) convergence -----------

def test_std_error_shrinks_like_inv_sqrt_n():
    snap = flat_snapshot(vol=0.25, rate=0.03, div=0.0)
    terms = TermSheet()
    specs = [UnderlyingSpec(name="S", weight=1.0)]
    ses = []
    for n, seed in ((20_000, 21), (80_000, 22)):
        cfg = EngineConfig(n_paths=n, qmc=True, seed=seed, steps_per_day=1,
                           batch_size=20_000, compute_greeks=False)
        ses.append(price(snap, terms, specs, cfg).std_error)
    ratio = ses[0] / ses[1]
    assert 1.6 < ratio < 2.4, (ses, ratio)  # expect ~2.0


# ------------------------------------------------- multi-asset smoke ---------

def test_two_asset_basket_runs_with_correlation():
    snap = flat_snapshot(vol=0.25, rate=0.03, div=0.0)
    terms = TermSheet()
    specs = [UnderlyingSpec(name="A", weight=0.6),
             UnderlyingSpec(name="B", weight=0.4)]
    cfg = EngineConfig(n_paths=20_000, qmc=True, seed=23, steps_per_day=1,
                       batch_size=20_000, compute_greeks=False)
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    r = price(snap, terms, specs, cfg, corr=corr)
    assert 0.5 < r.price < 1.5
    assert r.ko_prob > 0
