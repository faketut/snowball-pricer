"""P1 latency benchmark: fast tick path vs slow recalibration path.

Measures on the synthetic feed (reference workload: 4 expiries x 15 strikes
x call/put = 120 quotes per full sweep):
  fast path: QuoteStore.update + single-quote IV inversion, per tick
  slow path: full rebuild_surface (invert all + raw-SVI x4 + global SSVI fit
             + arbitrage gate on dense grids)

Writes results/p1_latency.json and prints a summary for PERF.md.
"""
import itertools
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from snowball_pricer.iv import implied_vol_safe
from snowball_pricer.pipeline import QuoteStore, rebuild_surface
from snowball_pricer.tick import SyntheticTickFeed

TRUTH = dict(
    thetas=[0.0121, 0.0200, 0.0361, 0.0648],
    expiries=[0.25, 0.5, 1.0, 2.0],
    rho=-0.45,
    eta=0.8,
    gamma=0.5,
    spot=100.0,
    rate=0.03,
    div_yield=0.01,
)


def percentiles(xs):
    xs = np.asarray(xs, dtype=float)
    return {
        "p50": float(np.percentile(xs, 50)),
        "p99": float(np.percentile(xs, 99)),
        "max": float(np.max(xs)),
        "n": int(xs.size),
    }


def main():
    feed = SyntheticTickFeed(**TRUTH, seed=7)
    store = QuoteStore()
    it = feed.subscribe()
    # Prime the book with one full sweep.
    first_sweep = [next(it) for _ in range(feed.sweep_size)]
    for q in first_sweep:
        store.update(q)

    # Fast path: per-tick update + IV inversion of that tick's mid.
    # (Forward from parity is cached per expiry in production; here we use
    # the spot-implied forward to isolate inversion cost.)
    fast_us = []
    for q in itertools.islice(it, 5000):
        t0 = time.perf_counter()
        store.update(q)
        df = math.exp(-q.rate * q.expiry)
        fwd = q.forward_spot()
        implied_vol_safe(q.mid, fwd, q.strike, q.expiry, df, q.is_call)
        fast_us.append((time.perf_counter() - t0) * 1e6)

    # Slow path: full surface rebuilds.
    slow_ms = []
    asof = first_sweep[-1].ts
    for _ in range(30):
        t0 = time.perf_counter()
        surface, diag = rebuild_surface(store, asof=asof)
        slow_ms.append((time.perf_counter() - t0) * 1e3)

    result = {
        "workload": "4 expiries x 15 strikes x call/put = 120 quotes/sweep",
        "fast_path_us_per_tick": percentiles(fast_us),
        "slow_path_rebuild_ms": percentiles(slow_ms),
        "slow_path_rmse_iv": diag["rmse_iv"],
        "note": "shared VM; timings include Python overhead, not a tuned build",
    }
    with open("results/p1_latency.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
