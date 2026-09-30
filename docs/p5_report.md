# P5 report — vectorized MC + real-time closed loop (2026-09-29)

Final phase of snowball-pricer. Two deliverables: (1) vectorize the P2 MC
hot loop (performance first), (2) close the loop — synthetic feed →
pipeline → reprice → monitoring with alerts.

## 1. Vectorization: profile → attack → measure

### Profile (cProfile, canonical workload: 2Y 1-asset snowball, flat 25% vol,
### 504 daily steps, 20k QMC paths, no Greeks)

| component | share of wall | note |
|---|---|---|
| per-step `LocalVolSurface.local_vol` | ~33% | numpy call overhead per step: `searchsorted` on the non-uniform T grid, 2× `clip`, per-asset Python loop |
| Sobol draws + `ndtri` | ~30% | `qmc.Sobol.random` + inverse-CDF transform of 50M uniforms per 100k batch |
| Brownian-bridge construction | ~7% | scalar Python loop, 503 iterations of cache-resident (p,a) ops |
| per-step `basket_vol` | ~6% | materialized a (p,a,a) covariance cube per step |
| Euler update + payoff bookkeeping | rest | already vectorized per step |

### Changes (`price()` signature unchanged; all draws/formulas preserved)

- **`dupire.py`**: new `LocalVolSurface.t_weights(t)` (T-grid weights,
  computed once) + `local_vol_at_weights(k, j0, j1, fT)` (per-step query
  with no `searchsorted`; k-lerp bitwise identical to `local_vol`).
- **`mc.py`**: `PathBatchDriver` precomputes per-step forwards and LV
  T-weights once; `_Batch.steps()` is a lean loop with the identical Euler
  update op order. `_fill_bridge` is a thin wrapper over the original
  scalar bridge loop (kept deliberately — see rejected ideas).
- **`payoff.py`**: `basket_vol` = `sqrt(y'Cy)`, `y = x·σ` — no (p,a,a)
  cube; the hot loop reuses already-computed `perf`/`B` through the single
  source of truth `_basket_vol_from_perf` (FP differs ~1e-16).

### Rejected after measurement (documented, not silent)

- **Vectorized Brownian bridge** (BFS depth-level gather/scatter): numpy
  fancy indexing `W[:, :, idx]` (mixed basic+advanced) runs at ~320 MB/s
  (~50x slow path); the transpose-view axis-0 form still streams ~3x more
  memory than the scalar loop and measured **slower** (743 ms vs ~250 ms
  at 30k paths) on this memory-constrained VM. The scalar loop's (p,a)
  working set stays cache-resident — it wins here.
- **float32 Sobol draws**: `ndtri` is *not* faster in f32, and `1-1e-12`
  rounds to `1.0` in float32, defeating the `ndtri` clip guard and
  producing `inf` normals — a correctness landmine. Draws stay f64.

### Measured speedup (honest numbers)

`scripts/p2_benchmark.py`, canonical 2Y/1-asset/daily workload,
`batch_size=100_000`, back-to-back old-vs-new on this VM:

| N paths | before (P2 baseline) | after (P5, measured 2026-09-29) | speedup |
|---|---|---|---|
| 100k | 13.5 s (7,382 paths/sec) | 7.4 s (13,459 paths/sec) | **1.82x** |
| 1M | 110.2 s (9,078 paths/sec) | 64.9 s (15,412 paths/sec) | **1.70x** |

Back-to-back A/B/A/B runs at 100k: old 10.4–14.5 s vs new 6.3–7.7 s
→ factor 1.6–1.9x depending on VM noise (this VM's noisy-neighbor jitter
is large; one old-code run spiked to 28 s). Reported factor: **~1.7x**.

Equivalence: the vectorized code reproduces the pre-P5 reference prices
(seed 5) to 3.9e-14 / 2.5e-13 relative (1-asset / 2-asset ρ=0.5) —
pinned at 1e-9 in `tests/test_vectorized.py`. The remaining floor is the
draws (~35% of wall: Sobol generation + `ndtri`); attacking that without
changing the QMC point set needs a faster inverse-CDF or a GPU — both
future work.

## 2. Live loop

`python3 scripts/live_loop.py --sim-seconds 300 --inject-fault --seed 7`
→ `results/p5_live.json`. P4's `RegimeTickFeed`, 3 regimes
(calm 20 d / stress 10 d / calm 20 d, 50 trading days, 6000 ticks),
rebuild every sweep (120 ticks), reprice every published snapshot
(20k QMC paths, fixed seed = CRN across time), full Greeks every 5th
snapshot. Ran unattended, wall 98.2 s.

| metric | measured |
|---|---|
| ticks | 6000 |
| dropped (all fault-injected crossed) | 16 |
| rebuilds / failures / published | 50 / 0 / 50 |
| reprices (with Greeks) | 50 (10) |
| avg reprice wall | 1.88 s (base ~1.2 s, Greeks runs ~4.6 s) |
| price v1 → v50 | 0.984916 → 0.980281 (stress regime repriced ~0.949 mid-run) |
| ko_prob | 0.898 → 0.894 (0.872 in stress) |
| alerts fired (live) | 0 — thresholds correctly not tripped |
| staleness self-test probe | FIRED as expected |

Reprice sizing held: a base reprice (~1.2 s) fits the 6-sim-second
snapshot cadence; Greeks runs (~4.6 s) fit the 30-sim-second Greeks
cadence. `GREEKS_DRIFT` did not fire — honestly reported, not gamed:
delta sits ~0.004 across regimes for this product (autocall economics
keep fresh-contract delta tiny and stable; the regime vol change shows up
in price/ko_prob, not delta), far below the 0.05 threshold. The detector
itself is proven by the 7 unit tests in `tests/test_monitoring.py`.

## 3. Fault-injection proof

`--inject-fault` crossed (bid > ask) quotes on one strike (all expiries,
call+put) for two sweeps mid-run (ticks 3000–3240). Quote from
`results/p5_live.json`:

```json
"stats": {"ticks": 6000, "dropped": 16, "fault_dropped": 16,
          "rebuilds": 50, "rebuild_failures": 0, ...},
"fault": {"target": "one strike (k_mid) x all expiries x call/put",
          "window_ticks": [3000, 3240], "crossed_dropped": 16}
```

All 16 crossed quotes failed `OptionQuote.is_valid()` and were dropped by
the `QuoteStore` — the drop counter spiked from its baseline of 0 to
exactly the 16 injected, the book stayed clean, and all 50 rebuilds
succeeded on valid data. No quarantine was needed and none fired: the
system correctly distinguished "bad ticks" (dropped) from "bad surface"
(quarantined).

The quarantine path is proven independently: a shorter `--sim-seconds 30`
run with the same fault left 4 stale book entries (dropped target quotes
kept the previous sweep's calm-regime values inside a stress-regime
book), the raw-SVI fit failed to converge, and the monitor fired

```text
[REBUILD_FAILURE/warning] rebuild failed with ValueError: raw-SVI fit
failed: The maximum number of function evaluations is exceeded.;
last-good snapshot v2 kept live
```

— i.e. the failure was quarantined, the previous good snapshot stayed
live, and the run continued. Mechanism verified by diffing the tick-360
book with/without the fault (4 keys differed, all fault-target strikes).

## 4. Monitoring

`Monitor` (`snowball_pricer/monitoring.py`): `on_snapshot` →
`GREEKS_DRIFT`; `on_rebuild_failure` → `ARBITRAGE_QUARANTINE` /
`REBUILD_FAILURE`; `check_staleness` → `STALENESS` (critical, 5-min
cooldown). Thresholds in SPEC.md §8.2. 7 unit tests green.

## 5. Honest limitations

- **GPU**: out of scope — no GPU on this VM. The step loop is the
  remaining prize; a GPU/CuPy port is future work, not faked.
- **Synthetic feed only**: no broker connected (`BrokerWsFeed` is still
  the P1 stub). RegimeTickFeed is deterministic and seeded — good for
  validation, not a market.
- **Greeks cost at scale**: delta+vega = 4x the core reprice (CRN bumps).
  The loop prices Greeks every 5th snapshot by design; per-snapshot
  Greeks at 1M paths would not fit any real-time cadence on CPU.
- **Draws are the CPU floor** (~35%): Sobol + `ndtri` dominate after P5.
- **VM noise**: all timings on a shared 2-core VM; factors are
  back-to-back medians, not lab conditions.
- **Single-asset demo**: the live loop prices a 1-asset basket; the
  3-asset correlation machinery (P3) is exercised in tests/stress, not
  in the loop.
