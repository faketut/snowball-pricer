#!/usr/bin/env python3
"""P4 historical replay validation (runnable end-to-end, fixed seed).

Regime feed: calm (ATM 20%) -> vol spike (ATM 45%) -> calm (ATM 22%),
14 trading days, one snapshot per day. Each snapshot is repriced with the
real P2 engine (20k QMC paths, CRN Greeks); a SHORT snowball is delta-hedged
along the TRUE spot path; the realized P&L is attributed to delta / vega /
carry buckets plus a residual.

Writes results/p4_replay.json and prints a summary table. Unattended.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from snowball_pricer.pricing.engine import EngineConfig
from snowball_pricer.pricing.payoff import TermSheet, UnderlyingSpec
from snowball_pricer.validation import (
    RegimeTickFeed,
    pnl_explain,
    run_replay,
    simulate_hedge,
)

SEED = 20260929
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results",
                       "p4_replay.json")

# P1 ground-truth SSVI params (tests/conftest.py TRUTH).
TRUTH = dict(
    thetas=[0.0121, 0.0200, 0.0361, 0.0648],
    expiries=[0.25, 0.5, 1.0, 2.0],
    rho=-0.45,
    eta=0.8,
    gamma=0.5,
)

# (n_days, atm_vol, spot_vol): calm -> vol spike -> calm.
REGIMES = [(5, 0.20, 0.20), (5, 0.45, 0.45), (4, 0.22, 0.22)]

TERMS = TermSheet(
    tenor_years=2.0,
    ko_barrier=1.00,
    ko_obs_every_days=21,
    ki_barrier=0.80,
    coupon_annual=0.15,
    notional=1.0,
)
UNDERLYINGS = [UnderlyingSpec(name="SYN", weight=1.0)]
ENGINE_CFG = EngineConfig(
    n_paths=20_000, qmc=True, seed=SEED, steps_per_day=1,
    batch_size=20_000, compute_greeks=True,
)
RATE = 0.03


def main() -> None:
    t_start = time.perf_counter()
    sweep = len(TRUTH["expiries"]) * 15 * 2  # 4 expiries x 15 strikes x c/p
    feed = RegimeTickFeed(
        [(n_days * sweep, atm, svol) for n_days, atm, svol in REGIMES],
        seed=SEED, spot0=100.0, rate=RATE, div_yield=0.01, **TRUTH,
    )

    t0 = time.perf_counter()
    # Seasoned position: barriers fixed in absolute terms at the day-0
    # spot (the hedge-delta convention; SPEC.md section 7).
    replay = run_replay(feed, TERMS, ENGINE_CFG, UNDERLYINGS,
                        norm_spots=[100.0])
    replay_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    hedge = simulate_hedge(replay.rows, replay.true_spots, TERMS, RATE)
    hedge_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    explain = pnl_explain(replay, hedge)
    explain_s = time.perf_counter() - t0
    total_s = time.perf_counter() - t_start

    b = explain["buckets_bps"]
    out = {
        "seed": SEED,
        "regimes": [{"n_days": n, "atm_vol": a, "spot_vol": s}
                    for n, a, s in REGIMES],
        "terms": {f: getattr(TERMS, f) for f in
                  ("tenor_years", "ko_barrier", "ko_obs_every_days",
                   "ki_barrier", "coupon_annual", "notional")},
        "engine_cfg": {"n_paths": ENGINE_CFG.n_paths, "qmc": ENGINE_CFG.qmc,
                       "seed": ENGINE_CFG.seed,
                       "compute_greeks": ENGINE_CFG.compute_greeks},
        "feed_spec": replay.feed_spec,
        "rows": replay.rows,
        "true_spots": replay.true_spots,
        "hedge": {
            "realized_pnl": hedge.realized_pnl,
            "premium": hedge.premium,
            "final_value": hedge.final_value,
            "stock_pnl": hedge.stock_pnl,
            "interest_pnl": hedge.interest_pnl,
            "n_days": hedge.n_days,
            "notional": hedge.notional,
            "terminated": hedge.terminated,
            "names": hedge.names,
        },
        "pnl_explain_bps": b,
        "n_theta_repricings": explain["n_theta_repricings"],
        "daily": explain["daily"],
        "timings": {
            "replay_price_s": replay_s,
            "hedge_s": hedge_s,
            "explain_theta_s": explain["theta_seconds"],
            "explain_total_s": explain_s,
            "total_s": total_s,
        },
    }
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2)

    # ---- stdout summary ----
    print("=" * 72)
    print("P4 historical replay validation  (seed %d)" % SEED)
    print("=" * 72)
    print("regimes : " + " -> ".join(
        f"{n}d @ ATM {a:.0%}" for n, a, s in REGIMES))
    print("reprice : %d snapshots x %d QMC paths (Greeks on)" %
          (len(replay.rows), ENGINE_CFG.n_paths))
    print()
    print("day |    spot | atm_iv |   price |  delta |   vega/pt | ko_prob")
    print("----+---------+--------+---------+--------+-----------+--------")
    for r in replay.rows:
        d = r["delta"]
        d0 = d.get("SYN", next(iter(d.values())))
        print("%3d | %7.2f | %6.2f%% | %7.4f | %6.3f | %9.5f | %6.2f%%" % (
            r["day"], r["spot"], r["atm_iv"] * 100, r["price"], d0,
            r["vega"], r["ko_prob"] * 100))
    print()
    print("P&L explain, SHORT snowball, hedged (bps of notional):")
    print("  %-22s %+10.1f" % ("delta  (net hedge-model)", b["delta"]))
    print("  %-22s %+10.1f" % ("vega   (-v*dATMiv)", b["vega"]))
    print("  %-22s %+10.1f" % ("carry  (model theta)", b["carry_theta_model"]))
    print("  %-22s %+10.1f" % ("carry  (cash interest)", b["carry_interest"]))
    print("  %-22s %+10.1f" % ("residual", b["residual"]))
    print("  %-22s %+10.1f" % ("ACTUAL realized", b["actual"]))
    print()
    print("hedge: premium %.4f -> final value %.4f, terminated=%s" % (
        hedge.premium, hedge.final_value, hedge.terminated))
    print("timings: replay %.0fs | hedge %.1fs | theta %d x %.0fs | total %.0fs" % (
        replay_s, hedge_s, explain["n_theta_repricings"],
        explain["theta_seconds"], total_s))
    print("wrote", RESULTS)


if __name__ == "__main__":
    main()
