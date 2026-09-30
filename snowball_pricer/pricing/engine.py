"""Pricing engine: snapshot -> snowball price (P2).

``price(snapshot, termsheet, underlyings, cfg)`` is the single entry point.
It consumes ONLY the P1 ``VolSurfaceSnapshot`` contract (``snapshot.surface``
plus metadata); the engine never reaches into P1's calibration internals.

Pipeline per call:
1. Resolve each UnderlyingSpec against the snapshot surface (spot/rate/div
   defaults) and build one ``LocalVolSurface`` per distinct surface object.
2. Stream MC batches: Sobol+bridge (or antithetic) paths under local vol,
   vectorized snowball evaluation with bridge barrier corrections.
3. Aggregate price / standard error / KO probability.
4. Optionally CRN Greeks (delta per underlying, parallel-bump vega).

``price_vanilla`` reuses the same simulator for European payoffs (validation).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .dupire import LocalVolSurface
from .greeks import delta_crn, ko_probability, vega_bump_surface
from .mc import PathBatchDriver, SimMarket, simulate_terminal
from .payoff import TermSheet, UnderlyingSpec, basket_vol, evaluate_snowball


@dataclass(frozen=True)
class EngineConfig:
    n_paths: int = 100_000
    qmc: bool = True
    seed: int = 20260929
    steps_per_day: int = 1
    batch_size: int = 20_000
    # Barrier monitoring: contractual discrete observation by default
    # (matches P0). The bridge flags select the continuous-touch variant.
    ki_bridge: bool = False
    ko_bridge: bool = False
    compute_greeks: bool = True
    delta_bump: float = 0.01


@dataclass
class PriceResult:
    price: float
    std_error: float
    ko_prob: float
    ko_prob_se: float
    delta: Dict[str, float]
    vega: float          # per 1 vol point parallel bump
    n_paths: int
    timing: Dict[str, float]
    surface_version: int
    tenor_years: float


@dataclass
class _ResolvedMarket:
    market: SimMarket
    specs: List[UnderlyingSpec]  # spots/rates/divs resolved


def _resolve_market(snapshot, underlyings: List[UnderlyingSpec],
                    tenor_years: float) -> _ResolvedMarket:
    surf0 = snapshot.surface
    spots, rates, divs, weights, lvs = [], [], [], [], []
    lv_cache: Dict[int, LocalVolSurface] = {}
    resolved = []
    for spec in underlyings:
        surf = spec.surface if spec.surface is not None else surf0
        spot = spec.spot if spec.spot is not None else float(surf.spot)
        rate = spec.rate if spec.rate is not None else float(surf.rate)
        div = spec.div_yield if spec.div_yield is not None else float(surf.div_yield)
        key = id(surf)
        if key not in lv_cache:
            lv_cache[key] = LocalVolSurface(surf, t_max=tenor_years)
        spots.append(spot)
        rates.append(rate)
        divs.append(div)
        weights.append(spec.weight)
        lvs.append(lv_cache[key])
        resolved.append(UnderlyingSpec(name=spec.name, weight=spec.weight,
                                       spot=spot, rate=rate, div_yield=div,
                                       surface=surf))
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    n = len(spots)
    corr = np.eye(n)  # P2 default: decorrelated; caller may override via set_corr
    market = SimMarket(spots=np.asarray(spots), rates=np.asarray(rates),
                       divs=np.asarray(divs), weights=weights, corr=corr,
                       lvs=tuple(lvs))
    return _ResolvedMarket(market=market, specs=resolved)


def set_correlation(resolved: _ResolvedMarket, corr: np.ndarray) -> _ResolvedMarket:
    """Attach a correlation matrix (P2: user-supplied; P3: calibrated)."""
    corr = np.asarray(corr, dtype=float)
    n = len(resolved.market.spots)
    assert corr.shape == (n, n), "corr must be (n_assets, n_assets)"
    mkt = resolved.market
    market = SimMarket(spots=mkt.spots, rates=mkt.rates, divs=mkt.divs,
                       weights=mkt.weights, corr=corr, lvs=mkt.lvs)
    return _ResolvedMarket(market=market, specs=resolved.specs)


def _price_core(snapshot, terms: TermSheet, resolved: _ResolvedMarket,
                cfg: EngineConfig, seed: int, norm_spots=None) -> Dict:
    t0 = time.perf_counter()
    market = resolved.market
    driver = PathBatchDriver(market, terms.tenor_years, cfg.steps_per_day)
    build_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    acc = evaluate_snowball(
        driver, market, terms,
        {"ki_bridge": cfg.ki_bridge, "ko_bridge": cfg.ko_bridge,
         "batch_kwargs": {"n_paths": cfg.n_paths, "qmc_on": cfg.qmc,
                          "seed": seed, "batch_size": cfg.batch_size}},
        corr_seed=seed + 999,
        norm_spots=norm_spots,
    )
    sim_s = time.perf_counter() - t1

    n = acc["n"]
    mean = acc["sum_pay"] / n
    var = max(acc["sum_pay_sq"] / n - mean ** 2, 0.0)
    se = float(np.sqrt(var / n))
    ko_p, ko_se = ko_probability(acc["ko_count"], n)
    return {"price": float(mean), "std_error": se, "ko_prob": ko_p,
            "ko_prob_se": ko_se, "n": n,
            "timing": {"build_s": build_s, "sim_s": sim_s}}


def price(snapshot, terms: TermSheet, underlyings: List[UnderlyingSpec],
          cfg: EngineConfig = EngineConfig(),
          corr: Optional[np.ndarray] = None,
          norm_spots: Optional[np.ndarray] = None) -> PriceResult:
    """Price a basket-average snowball from a P1 surface snapshot.

    ``norm_spots``: absolute spots used to normalize basket performance
    (default None -> the market's own spots, i.e. a freshly-struck contract
    with B(0) = 1). Pass the ORIGINAL trade-date spots to price a SEASONED
    position: barriers then stay fixed in absolute terms while the simulated
    spot moves (this is the hedge delta / P&L-explain convention; see
    SPEC.md section 7).
    """
    t0 = time.perf_counter()
    resolved = _resolve_market(snapshot, underlyings, terms.tenor_years)
    if corr is not None:
        resolved = set_correlation(resolved, corr)
    norm = None if norm_spots is None else np.asarray(norm_spots, dtype=float)

    core = _price_core(snapshot, terms, resolved, cfg, cfg.seed,
                       norm_spots=norm)

    delta: Dict[str, float] = {}
    vega = float("nan")
    greek_s = 0.0
    if cfg.compute_greeks:
        t1 = time.perf_counter()
        base_spots = resolved.market.spots.copy()
        # CRN delta: barriers fixed in absolute terms. With norm_spots
        # given (seasoned), they are already absolute; otherwise the base
        # spots play that role (fresh-contract convention).
        bump_norm = base_spots if norm is None else norm

        def reprice(specs):
            # CRN delta: same draws, barriers fixed in absolute terms via
            # norm_spots (see payoff.evaluate_snowball).
            r = _ResolvedMarket(market=_replace_specs(resolved, specs),
                                specs=specs)
            c = _price_core(snapshot, terms, r, cfg, cfg.seed,
                            norm_spots=bump_norm)
            return c["price"], c["std_error"]

        delta = delta_crn(reprice, resolved.specs, core["price"],
                          h=cfg.delta_bump)

        bumped_surf = vega_bump_surface(snapshot.surface, 0.01)
        bumped_snap = _bumped_snapshot(snapshot, bumped_surf)
        rb = _resolve_market(bumped_snap, underlyings, terms.tenor_years)
        if corr is not None:
            rb = set_correlation(rb, corr)
        vb = _price_core(bumped_snap, terms, rb, cfg, cfg.seed,
                         norm_spots=norm)
        vega = float(vb["price"] - core["price"])  # per 1 vol point
        greek_s = time.perf_counter() - t1

    timing = dict(core["timing"])
    timing["greeks_s"] = greek_s
    timing["total_s"] = time.perf_counter() - t0
    return PriceResult(price=core["price"], std_error=core["std_error"],
                       ko_prob=core["ko_prob"], ko_prob_se=core["ko_prob_se"],
                       delta=delta, vega=vega, n_paths=core["n"],
                       timing=timing, surface_version=snapshot.version,
                       tenor_years=terms.tenor_years)


def _replace_specs(resolved: _ResolvedMarket,
                   specs: List[UnderlyingSpec]) -> SimMarket:
    mkt = resolved.market
    return SimMarket(spots=np.array([s.spot for s in specs]),
                     rates=mkt.rates, divs=mkt.divs, weights=mkt.weights,
                     corr=mkt.corr, lvs=mkt.lvs)


def _bumped_snapshot(snapshot, bumped_surface):
    from ..snapshot import VolSurfaceSnapshot
    return VolSurfaceSnapshot(surface=bumped_surface, version=snapshot.version,
                              asof=snapshot.asof, source="vega-bump",
                              n_quotes=snapshot.n_quotes,
                              rmse_iv=snapshot.rmse_iv)


def price_vanilla(snapshot, strike: float, expiry_years: float, kind: str,
                  underlyings: List[UnderlyingSpec],
                  cfg: EngineConfig = EngineConfig()) -> Dict:
    """European call/put under the Dupire local-vol simulator (validation).

    Not a product path — used by tests/benchmarks to check the simulator
    against Black-Scholes.
    """
    assert kind in ("call", "put")
    resolved = _resolve_market(snapshot, underlyings, expiry_years)
    market = resolved.market
    assert len(market.spots) == 1, "vanilla validation is single-asset"
    r = float(market.rates[0])
    df = float(np.exp(-r * expiry_years))
    S_T = simulate_terminal(market, expiry_years, cfg.steps_per_day,
                            cfg.n_paths, cfg.qmc, cfg.seed)
    s = S_T[:, 0]
    if kind == "call":
        pay = np.maximum(s - strike, 0.0) * df
    else:
        pay = np.maximum(strike - s, 0.0) * df
    n = len(pay)
    mean = float(np.mean(pay))
    se = float(np.std(pay, ddof=1) / np.sqrt(n))
    return {"price": mean, "std_error": se, "n": n}
