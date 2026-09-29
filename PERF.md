# PERF.md — snowball-pricer latency baselines

Baselines are honest measurements, not targets hit by gaming the benchmark.
Machine: shared Linux VM (this environment), CPython, numpy 1.26 / scipy 1.11.
Timings include Python overhead; a tuned build or Rust port would be faster,
but these numbers are the regression anchor for P1 — any change that moves
them significantly must be explainable.

## P1 pipeline (measured 2026-09-29, `scripts/p1_latency.py`)

Reference workload: 4 expiries × 15 strikes × call/put = 120 quotes/sweep,
synthetic feed.

| Path | What is timed | p50 | p99 | max | n |
|---|---|---|---|---|---|
| fast | `QuoteStore.update` + single-quote IV inversion, per tick | 8.2 µs | 43 µs | 299 µs | 5000 |
| slow | full `rebuild_surface`: invert 120-quote book + 4× raw-SVI fits + global SSVI fit + arbitrage gate on dense grids | 71 ms | 90 ms | 95 ms | 30 |

- P1 target was **< 100 ms p99 for a full rebuild: met** (90 ms).
- The two speeds are by design (SPEC.md §2.7): the fast path runs per tick;
  the slow path runs on cadence (every N ticks / seconds / on a large
  underlying move). A tick never waits for a calibration.
- Calibration quality at this speed: input-mid RMSE 0.009 vol pts on the
  reference workload (well under the 0.3 acceptance bar).

## How to re-run

```
python3 scripts/p1_latency.py   # writes results/p1_latency.json
python3 -m pytest tests/ -q     # 18 tests, includes e2e pipeline timing sanity
```

## P3 correlation (measured 2026-09-29)

Estimator timings (6000 daily obs, 3 assets, CPython):

| Function | Wall |
|---|---|
| `rolling_pearson(returns, window=252)` → 5751 matrices | 219 ms |
| `ewma_corr(returns)` (Python loop over obs) | 30.0 ms |
| `nearest_psd` (3×3) | 67.5 µs |
| 4× `shock_corr` (3×3) | 206 µs |

Stress reprice: 3-asset basket-average snowball, 40k QMC paths, no Greeks,
per scenario ≈ **13.1 s** (5 scenarios ≈ 66 s wall). The 8k-path test config
(`tests/test_correlation.py`) runs the base + 2-scenario table in ~15 s.

Measured sensitivity (base ρ=0.5, 40k QMC, SE ≈ 5.8 bps — signal ≫ noise):

| scenario | ρ off-diag | Δprice (bps of notional) | ΔKO (pp) |
|---|---|---|---|
| plus_0.2 | 0.700 | −47.8 | −0.61 |
| minus_0.2 | 0.300 | +54.6 | +0.71 |
| crisis_1.0 | 0.950 | −91.9 | −1.02 |
| dispersion_0.0 | 0.050 | +142.0 | +1.64 |

Headline: ±0.2 ρ moves the basket-average snowball by ~55 bps; extreme
scenarios by ≤142 bps. Correlation is a second-order driver here —
an order of magnitude below ε_stale (0.39–0.58%).

## Notes / known noise

- p99 fast-path includes GC / scheduling jitter of the shared VM; the max
  (299 µs) is VM noise, not a code path.
- Rebuild cost scales ~linearly in (#expiries × #strikes); the arbitrage
  gate's dense grids are the fixed overhead (~15–20 ms of the 71 ms).
  If the book grows past ~500 quotes/expiry, consider subsampling wings or
  moving the gate to a coarser grid with a documented tolerance.

## P2 pricing engine (2026-09-29)

Workload: canonical 2Y single-asset snowball, flat 25% vol, daily steps
(504 steps), contractual discrete barriers, no Greeks, batch_size=100k.

| N paths | wall | paths/sec | price | SE |
|---|---|---|---|---|
| 100k | 13.5 s | 7,382 | 0.985909 | 4.3e-04 |
| 1M | 110.2 s | 9,078 | 0.985837 | 1.4e-04 |

Dupire grid build (161×61, analytic SSVI path): <0.01 s — negligible vs
simulation. The per-step Python loop (~504 iterations of vector ops +
bilinear local-vol lookup per asset) dominates; a numba/Rust port of the
step loop is the obvious next step if reprice frequency demands it.

### QMC vs antithetic MC

- Vanilla call (smooth-ish payoff), N=20k, |err| vs Black-Scholes over
  4 seeds: QMC 1.2e-03 vs antithetic MC 1.0e-01 — QMC wins ~86x. The
  Sobol + Brownian-bridge machinery works as designed.
- Snowball (digital KO / KI-triggered put), N=50k, |err| vs 1M-path
  reference over 4 seeds: QMC 2.7e-04 vs antithetic MC 2.3e-04 — no gain.
  The barrier discontinuities blunt QMC's smoothness advantage (expected;
  cf. QMC literature on digital payoffs). The Sobol driver's value here is
  deterministic reproducibility, not variance. A bridge-probability-weighted
  ("smoothed") barrier estimator is candidate future work.

### How to re-run

```
python3 scripts/p2_benchmark.py
python3 -m pytest tests/test_pricing.py -q   # 16 P2 tests
```
