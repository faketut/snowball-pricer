"""P5 real-time closed loop: feed -> pipeline -> vectorized reprice -> monitor.

Three feed modes (``--feed``): ``synthetic`` (default, RegimeTickFeed — the
P5 validation loop), ``questrade`` (live Questrade L1 option quotes via
``snowball_pricer.feeds.QuestradeFeed``; needs QUESTRADE_REFRESH_TOKEN and
--max-ticks to bound the unbounded stream), and ``yfinance`` (15-MINUTES
DELAYED Yahoo poll feed via ``snowball_pricer.feeds.YFinancePollFeed`` —
NOT real-time; every tick is_delayed=True).

Loop (synthetic mode, unattended):

    RegimeTickFeed (3 regimes: calm / stress / calm)
        -> QuoteStore.update per tick (fast path)
        -> rebuild_surface every ``rebuild_every`` ticks (slow path);
           failures are quarantined (last-good snapshot stays live) and
           reported to the Monitor
        -> on each published snapshot: reprice with the vectorized P2
           engine, feed {snapshot, price} to the Monitor
        -> Monitor: ARBITRAGE_QUARANTINE / REBUILD_FAILURE /
           GREEKS_DRIFT / STALENESS

Reprice sizing (documented): ``n_paths=20_000`` reprices in ~1.4 s wall on
this VM (P5 vectorized engine, no Greeks). Snapshots publish every
``rebuild_every=120`` ticks x ``tick_dt=0.05`` s = 6 sim-seconds, so a
reprice comfortably fits the cadence. Full Greeks (delta + vega = 4x the
core cost) run every ``greeks_every``-th snapshot — Greeks drift is only
meaningful on that slower cadence, and Greeks at 20k paths are CRN-quiet
enough for the 0.05/0.02 drift thresholds. All reprices share one fixed
seed: intentional common-random-numbers across time, so price/Greek moves
between snapshots reflect market moves, not MC noise.

``--inject-fault``: mid-run, a burst of crossed (bid > ask) quotes targeting
one strike (all expiries, call+put) for two full sweeps. Crossed quotes fail
``OptionQuote.is_valid()`` and are dropped by the QuoteStore — the drop
counter must spike and the book must stay clean (no quarantine needed, no
snapshot published from bad data). Proven in the log + results JSON.

Writes ``results/p5_live.json`` (price series + alert list + stats) and
prints a run summary.
"""
import argparse
import dataclasses
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from snowball_pricer.arbitrage import ArbitrageViolation
from snowball_pricer.monitoring import Monitor, MonitorConfig
from snowball_pricer.pipeline import QuoteStore, rebuild_surface
from snowball_pricer.pricing.engine import EngineConfig, price
from snowball_pricer.pricing.payoff import TermSheet, UnderlyingSpec
from snowball_pricer.snapshot import SnapshotStore
from snowball_pricer.validation.replay import Regime, RegimeTickFeed

try:
    from snowball_pricer.feeds import QuestradeFeed
except ImportError:  # pragma: no cover - websocket-client not installed
    QuestradeFeed = None

try:
    from snowball_pricer.feeds import YFinancePollFeed
except ImportError:  # pragma: no cover - yfinance not installed
    YFinancePollFeed = None

TRUTH = dict(thetas=[0.0121, 0.0200, 0.0361, 0.0648],
             expiries=[0.25, 0.5, 1.0, 2.0], rho=-0.45, eta=0.8, gamma=0.5)
RATE, DIV, SPOT0 = 0.03, 0.01, 100.0
TICK_DT = 0.05
SWEEP = len(TRUTH["expiries"]) * 15 * 2  # 4 expiries x 15 strikes x call/put


def build_feed(sim_seconds: float, seed: int) -> RegimeTickFeed:
    days = max(3, round(sim_seconds / (TICK_DT * SWEEP)))
    d_calm = max(1, int(days * 0.4))
    d_stress = max(1, int(days * 0.2))
    d_calm2 = max(1, days - d_calm - d_stress)
    regimes = [
        Regime(n_ticks=d_calm * SWEEP, atm_vol=0.20, spot_vol=0.18),
        Regime(n_ticks=d_stress * SWEEP, atm_vol=0.38, spot_vol=0.35),
        Regime(n_ticks=d_calm2 * SWEEP, atm_vol=0.22, spot_vol=0.18),
    ]
    return RegimeTickFeed(regimes, seed=seed, spot0=SPOT0, rate=RATE,
                          div_yield=DIV, tick_dt=TICK_DT, **TRUTH)


def main() -> None:
    ap = argparse.ArgumentParser(description="P5 real-time closed loop")
    ap.add_argument("--sim-seconds", type=float, default=300.0)
    ap.add_argument("--inject-fault", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-paths", type=int, default=20_000)
    ap.add_argument("--rebuild-every", type=int, default=0,
                    help="ticks between surface rebuilds (default per feed: "
                         "synthetic/questrade=SWEEP, yfinance=1000)")
    ap.add_argument("--poll-interval", type=float, default=180.0,
                    help="yfinance poll cadence in seconds (default 180; "
                         "Yahoo throttles aggressively, do not go low)")
    ap.add_argument("--greeks-every", type=int, default=5)
    ap.add_argument("--out", default="results/p5_live.json")
    ap.add_argument("--feed", choices=["synthetic", "questrade", "yfinance"],
                    default="synthetic",
                    help="tick source: synthetic regime feed (default), live "
                         "Questrade adapter (needs QUESTRADE_REFRESH_TOKEN), "
                         "or yfinance 15-MIN-DELAYED poll feed (NOT real-time)")
    ap.add_argument("--underlyings", default="SPY",
                    help="comma-separated underlyings for --feed "
                         "questrade/yfinance")
    ap.add_argument("--max-ticks", type=int, default=0,
                    help="stop after N ticks (required>0 for questrade and "
                         "yfinance modes; synthetic mode ends on its own)")
    ap.add_argument("--rate", type=float, default=RATE,
                    help="risk-free rate for questrade mode")
    ap.add_argument("--div", type=float, default=0.0,
                    help="dividend yield for questrade mode")
    args = ap.parse_args()

    underlyings = [u.strip().upper()
                   for u in args.underlyings.split(",") if u.strip()]
    if args.rebuild_every <= 0:
        args.rebuild_every = 1000 if args.feed == "yfinance" else SWEEP

    if args.feed == "questrade":
        if QuestradeFeed is None:
            ap.error("--feed questrade needs websocket-client "
                     "(pip install -r requirements.txt)")
        if args.max_ticks <= 0:
            ap.error("--feed questrade requires --max-ticks > 0")
        feed = QuestradeFeed(underlyings, rate=args.rate,
                             div_yield=args.div)
        specs = [UnderlyingSpec(name=u, weight=1.0 / len(underlyings))
                 for u in underlyings]
        n_ticks = args.max_ticks
        feed_health = feed.health
    elif args.feed == "yfinance":
        if YFinancePollFeed is None:
            ap.error("--feed yfinance needs yfinance "
                     "(pip install -r requirements.txt)")
        if args.max_ticks <= 0:
            ap.error("--feed yfinance requires --max-ticks > 0 "
                     "(the poll feed is unbounded)")
        print("NOTE: --feed yfinance is 15-MINUTES DELAYED market data, "
              "not real-time. Every tick is flagged is_delayed=True.",
              flush=True)
        feed = YFinancePollFeed(underlyings, rate=args.rate,
                                div_yield=args.div,
                                poll_interval_s=args.poll_interval)
        specs = [UnderlyingSpec(name=u, weight=1.0 / len(underlyings))
                 for u in underlyings]
        n_ticks = args.max_ticks
        feed_health = None
    else:
        feed = build_feed(args.sim_seconds, args.seed)
        n_ticks = feed.total_ticks
        specs = [UnderlyingSpec(name="SYN", weight=1.0)]
        feed_health = None
    store, snapshots = QuoteStore(), SnapshotStore()
    monitor = Monitor(MonitorConfig())
    terms = TermSheet()

    # Fault plan: one ATM-ish strike (middle of the k-grid) across all
    # expiries, call+put, crossed for two full sweeps starting mid-run.
    # Synthetic mode only — the live feed carries real quotes.
    fault_start, fault_end = -1, -1
    target_strikes = {}
    if args.feed == "synthetic":
        k_grid = list(np.linspace(-0.45, 0.25, 15))
        k_mid = k_grid[len(k_grid) // 2]
        target_strikes = {T: SPOT0 * math.exp((RATE - DIV) * T) * math.exp(k_mid)
                          for T in TRUTH["expiries"]}
        fault_start = n_ticks // 2
        fault_end = fault_start + 2 * SWEEP if args.inject_fault else -1

    stats = {"ticks": 0, "dropped": 0, "fault_dropped": 0, "rebuilds": 0,
             "rebuild_failures": 0, "published_versions": [], "reprices": 0,
             "greeks_runs": 0}
    series = []
    reprice_s_total = 0.0
    t_run = time.perf_counter()

    it = feed.subscribe()
    tick = 0
    while True:
        try:
            q = next(it)
        except StopIteration:
            break
        tick += 1
        stats["ticks"] += 1
        if args.max_ticks and tick >= args.max_ticks:
            break  # questrade mode is unbounded; stop on the tick budget
        faulted = False
        if fault_start <= tick <= fault_end and abs(
                q.strike - target_strikes[q.expiry]) < 1e-9:
            # crossed quote: bid > ask -> is_valid() False -> dropped
            q = dataclasses.replace(q, bid=q.ask * 1.2, ask=q.ask)
            faulted = True
        if not store.update(q):
            stats["dropped"] += 1
            stats["fault_dropped"] += 1 if faulted else 0
        if tick % args.rebuild_every == 0:
            stats["rebuilds"] += 1
            try:
                surface, diag = rebuild_surface(store, asof=q.ts,
                                                source="live-loop")
            except (ArbitrageViolation, ValueError) as exc:
                stats["rebuild_failures"] += 1
                monitor.on_rebuild_failure(exc, asof=q.ts)
                continue
            snap = snapshots.publish(surface, asof=q.ts, source="live-loop",
                                     n_quotes=diag["n_quotes"],
                                     rmse_iv=diag["rmse_iv"],
                                     diagnostics=diag)
            stats["published_versions"].append(snap.version)
            with_greeks = (snap.version % args.greeks_every == 0)
            cfg = EngineConfig(n_paths=args.n_paths, qmc=True,
                               seed=args.seed, steps_per_day=1,
                               batch_size=min(args.n_paths, 20_000),
                               compute_greeks=with_greeks)
            t0 = time.perf_counter()
            res = price(snap, terms, specs, cfg)
            dt = time.perf_counter() - t0
            reprice_s_total += dt
            stats["reprices"] += 1
            stats["greeks_runs"] += 1 if with_greeks else 0
            monitor.on_snapshot(snap, res)
            monitor.check_staleness(q.ts)  # expected None mid-run
            series.append({"ts": q.ts, "version": snap.version,
                           "price": res.price, "std_error": res.std_error,
                           "delta": res.delta, "vega": res.vega,
                           "ko_prob": res.ko_prob, "reprice_s": dt,
                           "greeks": with_greeks})
            print(f"[tick {tick:5d}] v{snap.version:3d} price={res.price:.6f} "
                  f"se={res.std_error:.1e} ko={res.ko_prob:.3f} "
                  f"delta={res.delta} vega={res.vega:.4f} "
                  f"reprice={dt:.2f}s", flush=True)

    wall_s = time.perf_counter() - t_run

    # Staleness self-test: probe the detector with a market time far past
    # the last snapshot (labeled; not part of the live run itself).
    last_ts = series[-1]["ts"] if series else 0.0
    probe = monitor.check_staleness(
        last_ts + monitor.config.staleness_seconds + 1.0)
    staleness_self_test = ("FIRED as expected" if probe is not None
                           else "DID NOT FIRE (unexpected)")

    results = {
        "config": {"feed": args.feed,
                   "underlyings": getattr(args, "underlyings", None),
                   "sim_seconds": args.sim_seconds,
                   "inject_fault": args.inject_fault, "seed": args.seed,
                   "n_paths": args.n_paths,
                   "rebuild_every": args.rebuild_every,
                   "greeks_every": args.greeks_every,
                   "n_ticks": n_ticks, "sweep_size": SWEEP,
                   "monitor_thresholds": dataclasses.asdict(
                       monitor.config)},
        "stats": stats,
        "avg_reprice_s": (reprice_s_total / stats["reprices"]
                          if stats["reprices"] else 0.0),
        "wall_s": wall_s,
        "price_series": series,
        "alerts": [dataclasses.asdict(a) for a in monitor.alerts],
        "staleness_self_test": staleness_self_test,
        "fault": ({"target": "one strike (k_mid) x all expiries x call/put",
                   "window_ticks": [fault_start, fault_end],
                   "crossed_dropped": stats["fault_dropped"]}
                  if args.inject_fault else None),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    n_alerts = len(monitor.alerts)
    kinds = {}
    for a in monitor.alerts:
        kinds[a.kind] = kinds.get(a.kind, 0) + 1
    print("\n==== P5 live-loop summary ====")
    print(f"ticks={stats['ticks']} dropped={stats['dropped']} "
          f"(fault-injected crossed: {stats['fault_dropped']})")
    print(f"rebuilds={stats['rebuilds']} failures={stats['rebuild_failures']} "
          f"published={len(stats['published_versions'])}")
    print(f"reprices={stats['reprices']} (with Greeks: {stats['greeks_runs']}) "
          f"avg_reprice={results['avg_reprice_s']:.2f}s wall={wall_s:.1f}s")
    if series:
        print(f"price: {series[0]['price']:.6f} -> {series[-1]['price']:.6f} "
              f"over {len(series)} snapshots")
    print(f"alerts: {n_alerts} {kinds}")
    for a in monitor.alerts:
        print(f"  [{a.kind}/{a.severity}] {a.message}")
    print(f"staleness self-test: {staleness_self_test}")
    if feed_health is not None:
        print(f"questrade feed health: {feed_health.as_dict()}")
    print(f"results -> {args.out}")


if __name__ == "__main__":
    main()
