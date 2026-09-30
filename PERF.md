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
bilinear local-vol lookup per asset) dominates; P5 vectorized the hot loop
(~1.7x, see "P5 — MC hot-loop vectorization" below) — a numba/Rust/GPU
port of the step loop remains the obvious next step if reprice frequency
demands it.

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

## P5 — MC hot-loop vectorization (2026-09-29, this VM)

Same canonical workload as the P2 table above; `price()` signature and all
draws/formulas unchanged (`scripts/p2_benchmark.py` re-run 2026-09-29).

| N paths | before | after | speedup | paths/sec |
|---|---|---|---|---|
| 100k | 13.5 s | 7.4 s | **1.82x** | 13,459 |
| 1M | 110.2 s | 64.9 s | **1.70x** | 15,412 |

Prices identical to 1e-9 (0.985909 / 0.985837). Back-to-back A/B/A/B at
100k: old 10.4–14.5 s vs new 6.3–7.7 s → 1.6–1.9x; this VM's noisy-neighbor
jitter is large, so the honest claim is **~1.7x**.

What was attacked (cProfile): per-step `local_vol` (~33% of wall — now
precomputed T-weights + bitwise-identical fast lerp), per-step
`basket_vol` (no more (p,a,a) cube; `sqrt(y'Cy)`), per-step forwards
(hoisted). Deliberately NOT changed after measurement: the Brownian
bridge stays a scalar loop (the vectorized gather/scatter measured slower
— numpy fancy indexing on the strided axis runs ~50x slow and the
transpose-view form streams ~3x more memory), draws stay f64 (f32 `ndtri`
is not faster and breaks the clip guard). Remaining floor: Sobol draws +
`ndtri` ≈ 35% of wall. GPU: no GPU on this VM — future work, not faked.

Regression gate: `tests/test_vectorized.py` pins the pre-P5 reference
prices (seed 5) to 1e-9 relative and asserts > 11,000 paths/sec on the
canonical workload.

## P4 — historical replay validation (2026-09-29, this VM)

Canonical run: 14 trading days (5 calm / 5 vol-spike / 4 calm), 14
snapshots x 20,000 QMC paths with Greeks, seed 20260929
(`python3 scripts/p4_replay.py`; full numbers in `docs/p4_report.md`).

| stage | measured |
|-------|----------|
| Replay: P1 pipeline rebuild + P2 reprice, 14 snapshots x 20k QMC (Greeks on) | 96 s (~6.9 s / snapshot) |
| Hedge simulation (14 true-path days) | < 0.1 s |
| P&L explain: 13 frozen-market theta reprices x 20k QMC (no Greeks) | 44 s (~3.4 s / reprice) |
| **Total** | **140 s** |

Notes: Greeks roughly double the per-snapshot cost vs a price-only reprice
(compare ~3.4 s / 20k-path reprice without Greeks). The theta explain is
embarrassingly parallel across snapshot-days — halving wall-clock is one
`multiprocessing` pool away if P5's real-time loop needs it.
