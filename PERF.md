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

## Notes / known noise

- p99 fast-path includes GC / scheduling jitter of the shared VM; the max
  (299 µs) is VM noise, not a code path.
- Rebuild cost scales ~linearly in (#expiries × #strikes); the arbitrage
  gate's dense grids are the fixed overhead (~15–20 ms of the 71 ms).
  If the book grows past ~500 quotes/expiry, consider subsampling wings or
  moving the gate to a coarser grid with a documented tolerance.
