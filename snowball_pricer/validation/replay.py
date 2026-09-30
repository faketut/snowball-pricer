"""Multi-regime synthetic tick feed + replay harness (P4).

``RegimeTickFeed`` emits ``OptionQuote`` ticks from a piecewise-constant SSVI
ground truth: each regime scales the ATM total-variance term structure to a
target ATM vol (e.g. 20% -> 45% -> 22%) while the spot follows a GBM with
regime-dependent realized vol. Quote generation reuses P1's math
(``iv.black_price`` + spread/noise microstructure from ``tick.py``);
``underlying_price`` is the current regime spot.

``run_replay`` drives the feed through the REAL ``run_pipeline`` (one
snapshot per sweep/day), then reprices every published snapshot with the
REAL ``engine.price``. Tenor ages across the replay (day d prices a
``tenor - d/252`` contract) so the P&L explain in ``hedge.py`` sees genuine
theta.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from ..iv import black_price
from ..pipeline import QuoteStore, run_pipeline
from ..pricing.engine import EngineConfig, price
from ..pricing.payoff import TermSheet, UnderlyingSpec
from ..snapshot import SnapshotStore, VolSurfaceSnapshot
from ..surface import ssvi_total_var
from ..tick import Feed, OptionQuote


@dataclass(frozen=True)
class Regime:
    """One market regime.

    ``n_ticks``: ticks spent in this regime (must be a multiple of the
        feed's sweep size so regime boundaries align with snapshot days).
    ``atm_vol``: ABSOLUTE target ATM Black vol for the regime (e.g. 0.20).
        The base thetas are scaled by ``(atm_vol / base_atm) ** 2`` where
        ``base_atm = sqrt(thetas[0] / expiries[0])`` of the base SSVI params.
    ``spot_vol``: ABSOLUTE GBM vol of the spot path in this regime.
    """

    n_ticks: int
    atm_vol: float
    spot_vol: float


_RegimeLike = Union[Regime, Tuple[int, float, float]]


def _as_regime(r: _RegimeLike) -> Regime:
    if isinstance(r, Regime):
        return r
    n_ticks, atm_vol, spot_vol = r
    return Regime(n_ticks=int(n_ticks), atm_vol=float(atm_vol),
                  spot_vol=float(spot_vol))


class RegimeTickFeed(Feed):
    """Synthetic multi-regime option-quote feed.

    Deterministic given ``seed``: the spot path is precomputed in
    ``__init__`` from ``np.random.default_rng(seed)``; quote microstructure
    noise uses an independent stream ``default_rng(seed + 7919)`` re-seeded
    on every ``subscribe()`` (same convention as ``SyntheticTickFeed``).

    Timing: one full quote sweep (``sweep_size`` ticks) = one trading day.
    The spot is constant within a sweep and follows a daily GBM across
    sweeps (regime-dependent vol) — this keeps each snapshot's calibration
    as clean as P1's static-spot case while the *regime* changes drive the
    validation. ``daily_close_spots()`` is the hedger's true path.
    """

    _NOISE_SEED_OFFSET = 7919

    def __init__(
        self,
        regimes: Sequence[_RegimeLike],
        *,
        thetas: Sequence[float],
        expiries: Sequence[float],
        rho: float,
        eta: float,
        gamma: float,
        spot0: float,
        rate: float,
        div_yield: float,
        underlying_id: str = "SYN",
        k_grid: Optional[Sequence[float]] = None,
        spread_bps: float = 150.0,
        noise_frac: float = 0.1,
        tick_size: float = 0.01,
        tick_dt: float = 0.05,
        seed: int = 0,
        t0: float = 1_700_000_000.0,
        atm_iv_tenor: float = 1.0,
    ) -> None:
        assert len(thetas) == len(expiries) and len(expiries) > 0
        self._regimes = [_as_regime(r) for r in regimes]
        assert self._regimes, "at least one regime required"
        assert all(r.n_ticks > 0 and r.atm_vol > 0 and r.spot_vol >= 0
                   for r in self._regimes)
        self._thetas = tuple(float(t) for t in thetas)
        self._expiries = tuple(float(e) for e in expiries)
        self._rho = float(rho)
        self._eta = float(eta)
        self._gamma = float(gamma)
        self._spot0 = float(spot0)
        self._rate = float(rate)
        self._div_yield = float(div_yield)
        self._underlying_id = underlying_id
        self._k_grid = tuple(k_grid) if k_grid is not None else tuple(
            np.linspace(-0.45, 0.25, 15)
        )
        self._spread_bps = float(spread_bps)
        self._noise_frac = float(noise_frac)
        self._tick_size = float(tick_size)
        self._tick_dt = float(tick_dt)
        self._seed = int(seed)
        self._t0 = float(t0)

        # One sweep = (#expiries x #strikes x call/put) ticks = one trading day.
        self._sweep_size = len(self._expiries) * len(self._k_grid) * 2
        total = sum(r.n_ticks for r in self._regimes)
        assert total % self._sweep_size == 0, \
            "total ticks must be a whole number of sweeps (days)"
        assert all(r.n_ticks % self._sweep_size == 0 for r in self._regimes), \
            "each regime's n_ticks must be a multiple of the sweep size"
        self._total_ticks = total
        self._n_days = total // self._sweep_size

        # Reference ATM vol of the base params at the tenor where the replay
        # reports ATM vol (atm_iv_tenor, default 1Y — same default as
        # run_replay): regime thetas are scaled so the regime's atm_vol
        # lands on the reported tenor.
        ref_idx = min(range(len(self._expiries)),
                      key=lambda i: abs(self._expiries[i] - atm_iv_tenor))
        self._base_atm = math.sqrt(self._thetas[ref_idx]
                                   / self._expiries[ref_idx])

        # Regime index per day (regimes align to sweep boundaries).
        bounds = np.cumsum([r.n_ticks for r in self._regimes])
        self._regime_of_day = np.array([
            int(np.searchsorted(bounds, d * self._sweep_size, side="right"))
            for d in range(self._n_days)
        ])

        # Precompute the true spot path: daily GBM, regime-dependent vol.
        # spots[d] = spot prevailing during sweep (day) d.
        rng = np.random.default_rng(self._seed)
        dt_day = 1.0 / 252.0
        spots = np.empty(self._n_days)
        spots[0] = self._spot0
        drift = self._rate - self._div_yield
        for d in range(1, self._n_days):
            sig = self._regimes[int(self._regime_of_day[d - 1])].spot_vol
            z = rng.standard_normal()
            spots[d] = spots[d - 1] * math.exp(
                (drift - 0.5 * sig * sig) * dt_day
                + sig * math.sqrt(dt_day) * z
            )
        self._spots = spots

        # Theta scale per day: (atm_vol / base_atm)^2.
        self._theta_scale = np.array([
            (self._regimes[int(self._regime_of_day[d])].atm_vol
             / self._base_atm) ** 2
            for d in range(self._n_days)
        ])

    # -- feed properties -------------------------------------------------
    @property
    def sweep_size(self) -> int:
        return self._sweep_size

    @property
    def total_ticks(self) -> int:
        return self._total_ticks

    @property
    def n_days(self) -> int:
        """Number of trading days (sweeps) in the replay."""
        return self._n_days

    @property
    def t0(self) -> float:
        return self._t0

    @property
    def tick_dt(self) -> float:
        return self._tick_dt

    @property
    def expiries(self) -> Tuple[float, ...]:
        return self._expiries

    @property
    def regimes(self) -> Tuple[Regime, ...]:
        return tuple(self._regimes)

    @property
    def seed(self) -> int:
        return self._seed

    def daily_close_spots(self) -> np.ndarray:
        """True spot path: spot prevailing during each day's sweep."""
        return self._spots.copy()

    def theta_scale_of_day(self, day: int) -> float:
        return float(self._theta_scale[day])

    # -- quote generation --------------------------------------------------
    def _true_iv(self, T: float, K: float, S: float, theta_scale: float) -> float:
        i = self._expiries.index(T)
        theta = self._thetas[i] * theta_scale
        fwd = S * math.exp((self._rate - self._div_yield) * T)
        k = math.log(K / fwd)
        w = float(ssvi_total_var(k, theta, self._rho, self._eta, self._gamma))
        return math.sqrt(max(w, 1e-12) / T)

    def subscribe(
        self, symbols: Optional[Sequence[str]] = None
    ) -> Iterator[OptionQuote]:
        rng = np.random.default_rng(self._seed + self._NOISE_SEED_OFFSET)
        ts = self._t0
        k_grid = self._k_grid
        tick = 0  # global tick index (one per quote, for the assert)
        # Listed-style strike grid: FIXED for the whole replay, centered on
        # the day-0 forward. Real listed strikes don't move with spot, and a
        # fixed grid keeps the pipeline's latest-per-key book fully
        # refreshed every sweep (no stale-strike accumulation across days).
        grid = {}  # T -> (df, fwd0, [K])
        for T in self._expiries:
            df = math.exp(-self._rate * T)
            fwd0 = self._spot0 * math.exp((self._rate - self._div_yield) * T)
            grid[T] = (df, [fwd0 * math.exp(k) for k in k_grid])
        for day in range(self._n_days):
            # Spot is constant within the sweep (day).
            S = float(self._spots[day])
            tscale = float(self._theta_scale[day])
            for T in self._expiries:
                df, strikes = grid[T]
                fwd = S * math.exp((self._rate - self._div_yield) * T)
                for K in strikes:
                    iv = self._true_iv(T, K, S, tscale)
                    for is_call in (True, False):
                        mid = black_price(fwd, K, T, iv, df, is_call)
                        spread = max(self._tick_size,
                                     self._spread_bps * 1e-4 * mid)
                        noisy = mid + rng.normal(0.0, self._noise_frac * spread)
                        bid = max(self._tick_size, noisy - 0.5 * spread)
                        ask = max(bid + self._tick_size, noisy + 0.5 * spread)
                        ts += self._tick_dt
                        yield OptionQuote(
                            ts=ts, underlying_id=self._underlying_id, expiry=T,
                            strike=K, is_call=is_call, bid=bid, ask=ask,
                            underlying_price=S, rate=self._rate,
                            div_yield=self._div_yield,
                        )
                        tick += 1
        assert tick == self._total_ticks


class ReplayResult(NamedTuple):
    """Output of ``run_replay``.

    ``rows``: one dict per published snapshot, ordered by day:
        {day, ts, spot, tenor_years, surface_version, price, std_error,
         delta ({name: dV/dS} of the LONG product), vega (per 1 vol point,
         LONG), ko_prob, atm_iv}.
    ``true_spots``: true daily close spots (the hedger's path), length n_days.
    ``snapshots``: the published ``VolSurfaceSnapshot`` objects (needed by
        ``pnl_explain`` for the frozen-market theta proxy).
    ``underlyings``: the ``UnderlyingSpec`` list the replay was priced with.
    ``norm_spots``: absolute normalization spots (seasoned position) or
        None (fresh contract each day).
    """

    rows: List[Dict]
    true_spots: List[float]
    snapshots: List[VolSurfaceSnapshot]
    terms: TermSheet
    underlyings: List[UnderlyingSpec]
    norm_spots: Optional[List[float]]
    engine_cfg: EngineConfig
    feed_spec: Dict


def run_replay(
    feed: RegimeTickFeed,
    terms: TermSheet,
    engine_cfg: EngineConfig,
    underlyings: Sequence[UnderlyingSpec],
    *,
    rebuild_every: Optional[int] = None,
    source: str = "replay",
    atm_iv_tenor: float = 1.0,
    norm_spots: Optional[Sequence[float]] = None,
) -> ReplayResult:
    """Run the feed through the real P1 pipeline, reprice every snapshot.

    Rebuild cadence defaults to one full sweep (one trading day). Each
    snapshot is repriced with the real ``engine.price`` at the day's true
    close spot and an AGING tenor (``terms.tenor_years - day/252``) so the
    P&L attribution sees genuine theta. The engine seed is fixed, so
    day-to-day price moves are pure market/surface effects, not MC noise
    resampling.

    ``norm_spots``: pass the trade-date (day-0) spots to price a SEASONED
    position — barriers stay fixed in absolute terms across the replay
    (the hedge-delta convention; SPEC.md section 7). None (default) prices
    a freshly-struck contract each day (B(0) = 1 at the current spot).
    """
    rebuild_every = rebuild_every or feed.sweep_size
    store = QuoteStore()
    snapshots = SnapshotStore()
    stats = run_pipeline(
        feed, n_ticks=feed.total_ticks, rebuild_every=rebuild_every,
        source=source, store=store, snapshots=snapshots,
    )
    snaps = snapshots.history()
    if not snaps:
        raise RuntimeError("run_replay: no snapshots published "
                           f"(stats={stats})")

    closes = feed.daily_close_spots()
    # Expiry nearest atm_iv_tenor for the ATM-vol series used by vega explain.
    ref_T = min(feed.expiries, key=lambda e: abs(e - atm_iv_tenor))

    rows: List[Dict] = []
    for snap in snaps:
        # Map snapshot -> replay day via its asof timestamp (robust to a
        # failed rebuild skipping a day).
        day = int(round((snap.asof - feed.t0)
                        / (feed.tick_dt * feed.sweep_size))) - 1
        if not (0 <= day < feed.n_days):
            raise RuntimeError(
                f"snapshot v{snap.version} asof maps to day {day}, "
                f"outside [0, {feed.n_days})")
        spot = float(closes[day])
        tenor_d = terms.tenor_years - day / 252.0
        terms_d = replace(terms, tenor_years=tenor_d)
        specs = [replace(u, spot=spot) for u in underlyings]
        res = price(snap, terms_d, list(specs), cfg=engine_cfg,
                    norm_spots=norm_spots)
        rows.append({
            "day": day,
            "ts": snap.asof,
            "spot": spot,
            "tenor_years": tenor_d,
            "surface_version": snap.version,
            "price": res.price,
            "std_error": res.std_error,
            "delta": dict(res.delta),
            "vega": res.vega,
            "ko_prob": res.ko_prob,
            "atm_iv": float(snap.surface.iv(0.0, ref_T)),
        })
    rows.sort(key=lambda r: r["day"])

    feed_spec = {
        "regimes": [{"n_ticks": r.n_ticks, "atm_vol": r.atm_vol,
                     "spot_vol": r.spot_vol} for r in feed.regimes],
        "n_days": feed.n_days,
        "sweep_size": feed.sweep_size,
        "seed": feed.seed,
        "rebuild_every": rebuild_every,
        "published_versions": stats["published_versions"],
        "rebuild_failures": stats["rebuild_failures"],
    }
    return ReplayResult(rows=rows, true_spots=[float(s) for s in closes],
                        snapshots=list(snaps), terms=terms,
                        underlyings=list(underlyings),
                        norm_spots=(None if norm_spots is None
                                    else [float(x) for x in norm_spots]),
                        engine_cfg=engine_cfg, feed_spec=feed_spec)
