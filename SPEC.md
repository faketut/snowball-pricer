# SPEC.md — snowball-pricer (project level)

## 1. Product mechanics (reference snowball / autocallable)

Unless a term sheet overrides, the reference contract priced by this project is:

- **Underlying:** basket of N regional equity underlyings (constituents TBD per
  term sheet). **Basket linkage: CONFIRMED 2026-09-29 — basket average
  (weighted average of constituent performances), NOT worst-of.** Correlation
  sensitivity is therefore lower than worst-of; P3 correlation module still
  required but with milder stress scenarios.
  P0–P2 implement single-asset; multi-asset payoff linkage is a P3 decision that
  changes the correlation module, not the engine interface.
- **Tenor:** 2Y (configurable).
- **Knock-out (autocall):** observed **monthly** @ **100–105%** of initial level.
  On breach: redeem `notional × (1 + coupon × t_elapsed_years)`, terminate.
- **Knock-in:** observed on **daily closes** @ **75–80%** of initial level.
  Down-and-in; once hit, principal protection is lost permanently.
- **Maturity (never knocked out):**
  - never knocked in → `notional × (1 + coupon × T)` (principal + full coupon);
  - knocked in → `notional × min(1, S_T / S_0)` per linked underlying rule
    (investor absorbs the downside, keeps no upside beyond par).
- **Coupon:** 15% p.a. reference (term-sheet-driven).
- **State machine per path:** `Alive → {KnockedOut(t), KnockedIn, Matured}`.
  KO checked at monthly nodes; KI checked at daily closes (contractual discrete
  monitoring — never silently replaced by continuous monitoring; any continuous
  approximation must carry the BGK continuity correction, validated in P0).

## 2. Data contracts (P1 — implemented)

### 2.1 Tick schema: `OptionQuote` (frozen dataclass, `snowball_pricer/tick.py`)

| Field | Type | Meaning |
|---|---|---|
| `ts` | float | exchange timestamp, epoch seconds |
| `underlying_id` | str | e.g. `"510300"` / `"SYN"` (synthetic) |
| `expiry` | float | years to expiry, T > 0 |
| `strike` | float | K > 0 |
| `is_call` | bool | call / put |
| `bid`, `ask` | float | raw quote; `bid > 0`, `ask >= bid` (crossed quotes rejected by `is_valid()`) |
| `underlying_price` | float | spot S at `ts` |
| `rate` | float | cont-compounded risk-free r |
| `div_yield` | float | cont-compounded dividend yield q |
| `is_delayed` | bool | True when the venue flagged delayed data (default False) |

Derived: `mid`, `spread`, `key()` = `(underlying_id, expiry, strike, is_call)`,
`discount()`, `forward_spot()` (fallback forward).

### 2.2 Feed interface (`Feed.subscribe(symbols) -> Iterator[OptionQuote]`)

- Unbounded quote iterator; caller drives pacing, no hidden threads.
- `SyntheticTickFeed`: validation backbone — quotes from a known ground-truth
  SSVI surface + microstructure noise (spread = max(tick, spread_bps·mid),
  iid Gaussian mid noise, deterministic per seed). **No broker is connected.**
- `BrokerWsFeed` (P1 stub, raised `NotImplementedError`) is **retired** as of
  2026-09-30 and replaced by the real adapter `QuestradeFeed`
  (`snowball_pricer/feeds/questrade.py`) — **Questrade is the selected venue**.
  OAuth2 refresh-token auth (token from `QUESTRADE_REFRESH_TOKEN` env only,
  never logged/written), REST for symbol search + option chains + snapshots,
  WebSocket L1 streaming for quotes, reconnect with exponential backoff and
  universe re-resolve, `is_delayed` flag from the stream's `delay` field
  (US options L1 streaming needs the $9.95 CAD/mo package). The adapter
  contract from the old stub docstring is preserved: approved-flow
  credentials only, exchange/receive-ts normalization, heartbeat via
  `FeedHealth.last_msg_ts` (P5 STALENESS hook), reconnect with backoff,
  raw bid/ask forwarded (no mid fabrication), rate-limit pacing + 429
  backoff. Documented alternatives (not implemented): moomoo OpenAPI,
  IBKR TWS/Gateway. Setup: `docs/questrade_setup.md`.
- `YFinancePollFeed` (`snowball_pricer/feeds/yfinance.py`, Phase 1.5
  stepping stone, 2026-09-30): polls Yahoo option chains on a configurable
  cadence. **ALWAYS 15-minutes delayed — every quote has `is_delayed=True`
  hardcoded; never present it as real-time.** Missing bid/ask are emitted
  as None (never fabricated); each poll cycle is stamped with a cycle_id
  in `feed.cycle_stats`; errors/429s skip the cycle with backoff and never
  crash the loop. Quality baseline: `docs/yfinance_notes.md`
  (+ `results/yfinance_quality.json`, gitignored).

### 2.3 IV inversion (`snowball_pricer/iv.py`)

- European exercise assumed (SSE ETF options are European; de-Americanization
  out of scope — documented).
- Forward-first: put-call parity at the strike nearest the spot-implied
  forward among strikes quoting both sides; else spot-implied forward.
- Newton–Raphson on Black price (vega) with bisection fallback on [1e-9, 20];
  price tolerance 1e-10. `DegenerateQuoteError` on non-positive inputs,
  above-cap or below-intrinsic prices; price ≈ intrinsic ⇒ vol 0.0
  (numerically unidentifiable time value, not an error).
- Quotes that fail inversion are dropped from calibration (counted in
  diagnostics), never silently patched.

### 2.4 Surface: `VolSurface` (frozen dataclass, `snowball_pricer/surface.py`)

- Published form is **SSVI by construction**: per-expiry ATM total variances
  `thetas` (non-decreasing via PAVA isotonic regression — no calendar arb),
  global `(rho, eta, gamma)` with soft fit-time penalty
  `eta·(1+|rho|) ≤ 1.9`, per-expiry forwards, spot/rate/div.
- Queries: `total_var(k, T)`, `iv(k, T)`, `iv_from_strike(K, T)`; `theta_at(T)`
  interpolates linearly in T (linear-from-origin below first expiry, flat
  beyond last — documented extrapolation choice).
- Wings: no ad-hoc extrapolation — the SSVI formula is the extrapolator;
  asymptotic slope bounded by Lee's moment formula (numeric gate, §2.5).

### 2.5 No-arbitrage gate (`snowball_pricer/arbitrage.py`) — publish blocker

All three must pass on dense grids or the surface is quarantined
(`ArbitrageViolation`, never published; previous good snapshot stays live):
- **calendar**: `w(k,T)` non-decreasing in T ∀k;
- **butterfly**: Durrleman `g(k) ≥ 0` per expiry slice;
- **wings**: asymptotic slope of `w` vs `|k|` ≤ 2 (Lee).

### 2.6 Snapshot store (`snowball_pricer/snapshot.py`)

- `VolSurfaceSnapshot`: immutable `(surface, version, asof, source, n_quotes,
  rmse_iv)`; versions monotonic from 1 per store.
- `SnapshotStore.publish()`: single-publisher atomic swap (GIL-atomic ref
  assignment in CPython). **Lock-free contract for the future Rust port**
  (documented in module docstring): readers via `latest()` never block and
  always see a complete versioned snapshot — port must use `arc_swap` /
  seqlock equivalent. `publish_snapshot()` rejects stale versions.
- Bounded diagnostic history ring (ops only, not part of the publish contract).

### 2.7 Pipeline cadence (`snowball_pricer/pipeline.py`)

- **Fast path** (per tick): `QuoteStore.update` + single-quote IV — µs scale.
- **Slow path** (cadence: every N ticks / seconds / on large underlying move):
  full `rebuild_surface` = invert book + calibrate + gate + publish.
- Measured baselines in `PERF.md` (2026-09-29): fast p50 8.2µs / p99 43µs;
  full rebuild p50 71ms / p99 90ms (< 100ms p99 target ✅).

### 2.8 P1 acceptance (met 2026-09-29)

- 18/18 pytest green (`tests/`).
- Calibration recovery on synthetic ground truth: grid RMSE **0.0002** vol pts
  (target < 0.3); params recovered to ~3 decimals.
- Arbitrage suite: passes on calibrated surface; correctly rejects
  calendar-violating thetas, spiked-smile butterfly violation, steep wings;
  builder quarantines on gate failure without replacing the live snapshot.
- End-to-end: synthetic feed → ticks → IV → surface → snapshot publish,
  versions 1, 2, …; stale versions rejected.

## 3. Pricing engine contract (P2 — implemented 2026-09-29)

### 3.1 TermSheet (frozen dataclass, `snowball_pricer/pricing/payoff.py`)
| Field | Type | Default | Meaning |
|---|---|---|---|
| tenor_years | float | 2.0 | maturity in years (252 trading days/yr) |
| ko_barrier | float | 1.00 | autocall barrier, fraction of initial basket (=1) |
| ko_obs_every_days | int | 21 | KO observation spacing (trading days) |
| ki_barrier | float | 0.80 | knock-in barrier, fraction of initial basket; ≤0 disables KI |
| coupon_annual | float | 0.15 | autocall/maturity coupon, p.a., pro-rata |
| notional | float | 1.0 | price quoted per unit notional |
| discount_rate | float \| None | None | None → first underlying's rate |

Payoff (discounted): KO at obs j → `(1 + coupon·t_j)`; no KO + KI →
`min(1, B(T))`; no KO + no KI → `(1 + coupon·T)`. B(t) = Σ wⱼ·Sⱼ(t)/Sⱼ(0).

### 3.2 Basket (P2: average linkage confirmed 2026-09-29, NOT worst-of)
`UnderlyingSpec(name, weight, spot=None, rate=None, div_yield=None,
surface=None)` — None fields resolve from the priced snapshot's surface.
Weights normalized by the engine. Correlation: user-supplied matrix via
`price(..., corr=...)`; default identity. Calibrated correlation is P3.

### 3.3 Engine (`snowball_pricer/pricing/engine.py`)
`price(snapshot, termsheet, underlyings, cfg=EngineConfig(), corr=None)`
→ `PriceResult(price, std_error, ko_prob, ko_prob_se, delta{per name},
vega_per_vol_pt, n_paths, timing{}, surface_version, tenor_years)`.
Consumes ONLY the P1 `VolSurfaceSnapshot` contract. `EngineConfig`:
n_paths, qmc=True, seed, steps_per_day=1, batch_size, ki_bridge=False,
ko_bridge=False (contractual discrete monitoring by default; bridge flags
select the continuous-touch variant), compute_greeks, delta_bump.

### 3.4 Dynamics & simulation (see docs/p2_design.md)
- Dupire local vol from the snapshot (Gatheral form; analytic SSVI
  derivatives, FD fallback; vol floor 0.5%, cap 150% with diagnostics).
- Sobol QMC + Brownian-bridge path construction; Cholesky correlation;
  antithetic fallback. Batching: one Sobol sequence advanced across batches.
- Greeks: CRN central bump delta (barriers fixed absolute via norm_spots),
  +1pt parallel implied-surface bump vega, KO probability. Pathwise/AAD
  deliberately NOT implemented (barrier discontinuities; see §5 note in
  p2_design.md). AAD deferred as documented future work.

### 3.5 P2 acceptance (met 2026-09-29)
- Vanilla call/put vs Black-Scholes: max 3.1 bps (N=100k QMC, flat vol).
- Flat-vol snowball vs P0 reference 0.98699: within 35 bps (bridges off).
- KO prob: discrete vs lognormal-digital analytic <5%; bridge vs
  reflection-principle touch analytic <5%.
- Greeks: delta in continuation region ∈ (0, 1.5); vega < 0
  (bridges-off vega −0.203%/pt vs P0 −0.195%/pt).
- Convergence: SE(20k)/SE(80k) ∈ [1.6, 2.4] (≈2.0 expected).

## 4. Correlation contracts (P3 — implemented 2026-09-29)

### 4.1 Estimators (`snowball_pricer/correlation/estimate.py`)

| Function | Input | Output |
|---|---|---|
| `rolling_pearson(returns, window)` | `returns`: (n_obs, n_assets) log returns | ndarray (n_obs−window+1, n, n); `[t]` uses `returns[t:t+window]`; zero-variance assets yield 0 off-diagonal (never NaN); unit diagonal |
| `ewma_corr(returns, lam=0.94)` | same + decay `lam ∈ (0,1)` | (n, n) RiskMetrics correlation; warm start = sample second moment of first 10 obs; output passes `nearest_psd` |
| `implied_correlation(index_var, comp_vars, weights)` | index implied variance, per-component implied variances, weights | float: average pairwise implied ρ = (σ_I² − Σwᵢ²σᵢ²) / (Σ_{i≠j}wᵢwⱼσᵢσⱼ); clipped to [−0.99, 0.99] |
| `nearest_psd(a, eps=1e-8)` | square (n, n) array | nearest PSD correlation matrix: symmetrize → clip eigenvalues ≥ eps → rescale diagonal to 1; idempotent on valid input |

`implied_correlation` assumptions (not hidden): single average ρ for all
pairs; no idiosyncratic basis (index variance fully explained by
components); same-tenor/same-convention implied variances. An extreme
print (|ρ̂| near the clip bound) signals model violation — use as a stress
anchor, not a calibration (see docs/p3_design.md).

### 4.2 Stress scenarios (`snowball_pricer/correlation/stress.py`)

`shock_corr(corr, scenario)` shocks the off-diagonal only (diagonal stays
1.0), clips to [−0.99, 0.99], and PSD-repairs. Scenarios:

- `"plus_0.2"` / `"minus_0.2"`: ρ → ρ ± 0.2 (mild diversification-premium move)
- `"crisis_1.0"`: ρ → 0.95 (correlation spike; 0.95 not 1.0 to keep Cholesky non-singular)
- `"dispersion_0.0"`: ρ → 0.05 (dispersion / decorrelated regime)

Unknown scenario → `ValueError`.

### 4.3 Sensitivity table (`sensitivity_table`)

`SCENARIOS = ("plus_0.2", "minus_0.2", "crisis_1.0", "dispersion_0.0")`.
`sensitivity_table(base_corr, scenarios=SCENARIOS, vol, weights, cfg)`:
reprices the reference 3-asset basket-average snowball
(TermSheet defaults, FlatVol surface, default 40k QMC paths, no Greeks,
fixed seed) under base + each scenario with the **same engine seed**
(CRN deltas, not MC noise). Returns `ScenarioRow` list, base first:
`(scenario, rho_offdiag, price, std_error, delta_price_bps,
ko_prob, delta_ko_pp, seconds)`.

### 4.4 P3 acceptance (met 2026-09-29)

- 11/11 new pytest green (45 total with P1+P2 suite).
- Estimators recover synthetic ρ=0.6: rolling Pearson and EWMA
  (lam=0.995) within 0.05; default lam=0.94 within 0.15 (documented
  endpoint-estimator variance, see docs/p3_design.md).
- Implied correlation recovers ρ=0.6 on synthetic index-with-no-basis
  data within 0.01.
- Measured sensitivity (40k QMC, base ρ=0.5): ±0.2 ρ → ~55 bps price
  (|Δ| < 150 bps asserted, ~3× margin); crisis/dispersion extremes
  ≤ 142 bps. Basket variance provably monotonic in ρ. **No direction
  asserted**: KO/KI channels oppose; the sign is term-sheet-dependent.

## 5. Error budget (acceptance criteria)

| Class | Target | P0 measured | Gate |
|---|---|---|---|
| ε_num | < 0.1% notional | 0.017% @640k paths; N\*≈18k | P0 ✅ |
| ε_stale | < 0.3% notional | 0.39–0.58% on 2–3pt IV move w/ daily surface | P1+P5 must demonstrate shrink |
| barrier discretization | reported | −0.3 bps this contract; BGK residual −0.0 bps | P0 ✅ (machinery validated) |
| ε_model vs dealer quote | < 2% (1% stretch) | n/a | P4 replay |

## 6. Non-goals / constraints

- **Not** a latency system: ms–s pricing cadence is fine; no ns tail-latency
  requirement (explicitly out of scope — that is Legos' problem, a separate repo).
- No order execution in this repo (a future hedging executor would live with the
  execution stack, not here).
- No retail broker API keys in-repo; credentials via the approved connection
  flow only.


## 7. Historical replay validation (P4 — implemented 2026-09-29)

The "is the model honest" gate: replay a synthetic multi-regime history
through the REAL P1 pipeline, reprice every snapshot with the REAL P2
engine, delta-hedge the short position along the TRUE spot path, and
attribute the realized P&L. See `docs/p4_report.md` for the measured
numbers and the honest residual discussion.

### 7.1 Regime feed (`snowball_pricer/validation/replay.py`)

`RegimeTickFeed(Feed)`: piecewise-constant SSVI ground truth. Regimes are
`(n_ticks, atm_vol, spot_vol)` — `atm_vol` is the ABSOLUTE target ATM vol
at the reporting tenor (`atm_iv_tenor`, default 1Y — the same tenor
`run_replay` reports `atm_iv` at; base thetas scaled by
`(atm_vol/base_atm_1Y)^2` so the configured regime level matches the
measured 1Y ATM); `spot_vol` is the absolute GBM vol of the spot path. Quote generation reuses P1's math
(`iv.black_price` + spread/noise); `underlying_price` = current regime
spot. Deterministic given seed (spot stream `seed`, quote-noise stream
`seed + 7919`).

Two realism fixes made during P4 (documented, not hidden):
- Spot is constant within a sweep (one sweep = one trading day) and GBM
  across days. Per-tick spot movement smeared the smile inversion (the P1
  builder uses one forward per expiry); daily-constant spot keeps each
  snapshot's calibration as clean as P1's static case (recovered ATM vol
  within 0.1 vol pt of target, RMSE ≤ 0.01).
- The strike grid is FIXED for the whole replay (listed-style), centered
  on the day-0 forward. A per-day re-centered grid accumulated stale
  strikes in the pipeline's latest-per-key `QuoteStore` and biased the
  spike-regime calibration down by ~40%.

`run_replay(feed, terms, engine_cfg, underlyings, *, norm_spots=None)` →
`ReplayResult(rows, true_spots, snapshots, terms, underlyings, norm_spots,
engine_cfg, feed_spec)`:
drives the feed through the real `run_pipeline` (default cadence: one
sweep = one snapshot/day), reprices each published snapshot with
`engine.price` at the day's true close spot with an AGING tenor
(`tenor − day/252`, so the attribution sees genuine theta). The engine
seed is fixed across snapshots: day-to-day price moves are pure
market/surface effects, not MC resampling. `run_replay` takes
`norm_spots` (the trade-date spots): the canonical replay prices a
SEASONED position — barriers fixed in absolute terms at the day-0 spot.
A daily-rolled fresh contract would sit exactly at its KO barrier every
day (B(0) = 1 = ko_barrier by construction) and its delta would be
degenerate ≈ 0; the hedge's KO/KI checks use the same absolute barriers,
so pricing and hedging agree on the contract. Row:
`{day, ts, spot, tenor_years, surface_version, price, std_error, delta,
vega, ko_prob, atm_iv}` (`delta`/`vega` are the LONG product's).

### 7.2 Hedge + P&L explain (`snowball_pricer/validation/hedge.py`)

`simulate_hedge(rows, true_spots, terms, rate, notional=1.0)`:
delta-hedges a SHORT snowball. Sign convention: rows carry LONG delta
(> 0); the short's position delta is its negative, so the hedge holds
`shares = +delta_long × notional` (i.e. `hedge = −delta_short × notional`).
t0: receive premium, buy hedge. Daily: accrue cash at `rate`, check KO
observations on the true path (every `ko_obs_every_days` days, basket vs
`ko_barrier` → redeem `1 + coupon·t`, flatten), else rebalance to the
latest row delta (held between reprices). No KO: flatten + buy back at the
last model price at the end of the true path. Maturity economics (true-path
KI monitoring) implemented; the canonical 14-day replay never reaches it.
Internal identity `premium − final + stock + interest == realized` is
asserted in code.

`pnl_explain(replay, hedge, theta_cfg)`: SHORT-perspective attribution in
bps of notional, over the held day-to-day transitions —
- **delta**: Σ (h_d − delta_long_d)·dS_d — NET delta (hedge leg minus the
  position's model delta leg); ≈ 0 when the daily hedge tracks the model
  delta, nonzero from discrete/stale hedging;
- **vega**: Σ (−vega_long_d)·dATMiv_d (ATM iv from each snapshot's surface
  at a fixed tenor; vega is the engine's parallel +1pt bump, so this is an
  ATM-only approximation — named as such);
- **carry**: frozen-market model theta per snapshot (repriced at
  `tenor − 1 day`, same surface/spot/seed) with SHORT sign, plus hedge cash
  interest;
- **residual** = actual − (delta + vega + carry), by construction. It holds
  everything first-order misses: gamma (no gamma in the engine), discrete
  daily rebalancing, the vega ATM-only approximation, QMC noise in Greeks,
  surface recalibration noise, theta finite-difference noise.

### 7.3 P4 acceptance (met 2026-09-29)

- `python3 scripts/p4_replay.py` runs unattended → `results/p4_replay.json`
  + stdout summary table.
- 5 new pytest green (50 total with P1–P3 suite): replay completes with
  monotonic finite rows; hedge accounting identity exact; determinism
  (same seed → identical P&L table); autocall termination unit-tested;
  maturity payoff unit-tested.
- Measured attribution in `docs/p4_report.md` (not asserted — reported).

## 8. Real-time closed loop + monitoring (P5 — implemented 2026-09-29)

The "does it survive contact with a market" gate: a synthetic real-time
loop (feed → pipeline → vectorized reprice → monitor), plus the MC hot-loop
vectorization that makes per-snapshot repricing affordable.

### 8.1 Vectorized MC hot loop (measured, not claimed)

Profile-first (cProfile on the canonical 2Y/1-asset/daily workload):
per-step `LocalVolSurface.local_vol` was ~33% of wall (numpy call overhead:
`searchsorted` on the non-uniform T grid, double `clip`, per-asset Python
loop); Sobol draws + `ndtri` ~30%; Brownian-bridge construction ~7%;
per-step `basket_vol` (p,a,a) covariance cube ~6%.

What changed (all draws and formulas preserved; `price()` signature
unchanged):
- `LocalVolSurface.t_weights(t)` + `local_vol_at_weights(k, j0, j1, fT)`
  (`dupire.py`): T-grid weights precomputed once per driver step grid, so
  the per-step query skips `searchsorted`; the k-lerp is bitwise identical
  to `local_vol` (same clip, same indexing, same op order).
- `PathBatchDriver` precomputes per-step forwards and LV T-weights
  (`mc.py`); `_Batch.steps()` is a lean loop with identical Euler update.
- `basket_vol` is now `sqrt(y' C y)`, `y = x·σ` (`payoff.py`): no (p,a,a)
  cube; the hot loop reuses already-computed `perf`/`B` via
  `_basket_vol_from_perf` (single source of truth; FP differs ~1e-16).
- Rejected after measurement (documented, not silent): vectorized
  Brownian-bridge gather/scatter (numpy fancy-indexing on the strided axis
  runs ~50x slow; the transpose-view form streams ~3x more memory and
  measured SLOWER than the cache-resident scalar loop on this VM), and
  float32 Sobol draws (`ndtri` is not faster in f32, and `1-1e-12` rounds
  to `1.0` in f32 so the `ndtri` clip guard breaks → `inf` normals: a
  correctness landmine).

Measured on this VM (canonical benchmark, `scripts/p2_benchmark.py`,
100k QMC paths, batch 100k): **~1.6–1.9x speedup** (back-to-back
old-vs-new; VM noise is large, see PERF.md). Draws are now the floor
(~35% of wall). GPU is explicitly out of scope on this VM: no GPU
present; a GPU port of the step loop is future work, not faked here.
Regression: `tests/test_vectorized.py` pins the pre-P5 reference prices
(seed 5: 0.985908899218 1-asset, 0.991773469635 2-asset ρ=0.5) to 1e-9
relative — the vectorized code reproduces them to ~1e-13 — plus a
paths/sec timing gate.

### 8.2 Monitor (`snowball_pricer/monitoring.py`)

`Alert` dataclass `{ts, kind, severity, message}` (`ts` = market time when
known). `Monitor(config=MonitorConfig())`:

| hook | alert kind | severity | fires when |
|---|---|---|---|
| `on_snapshot(snapshot, price_result)` | `GREEKS_DRIFT` | warning | \|Δdelta\| or \|Δvega\| between consecutive *Greeked* reprices > threshold |
| `on_rebuild_failure(exc, asof=...)` | `ARBITRAGE_QUARANTINE` | warning | rebuild raised `ArbitrageViolation`; last-good snapshot kept |
| `on_rebuild_failure(exc, asof=...)` | `REBUILD_FAILURE` | warning | rebuild raised anything else (e.g. fit failure); last-good kept |
| `check_staleness(now_ts)` | `STALENESS` | critical | no published snapshot within `staleness_seconds` of market time |

Defaults (`MonitorConfig`): `delta_drift_threshold=0.05` (notional units;
typical delta ~0.9), `vega_drift_threshold=0.02` (per 1 vol pt; typical
vega ~−0.2), `staleness_seconds=30.0`, `staleness_cooldown_seconds=300.0`
(repeat STALENESS pages once per 5 min, not per tick). `GREEKS_DRIFT` only
compares reprices that both carry Greeks — the loop prices Greeks on a
slower cadence (below), so a Greeks→no-Greeks transition is expected and
must not alert. `check_staleness` before the first publish returns None
(quiet, not stale). All alerts accumulate in `monitor.alerts`.

### 8.3 Live loop (`scripts/live_loop.py`)

`python3 scripts/live_loop.py [--sim-seconds 300] [--inject-fault]
[--seed 7] [--n-paths 20000] [--rebuild-every 120] [--greeks-every 5]`:

- Feed: P4's `RegimeTickFeed`, 3 regimes (calm 40% / stress 20% / calm
  40% of the run; atm_vol 0.20/0.38/0.22), deterministic per `--seed`.
- Ticks → `QuoteStore.update` (fast path); `rebuild_surface` every
  `rebuild_every` ticks; failures quarantined → `monitor.on_rebuild_failure`.
- Each published snapshot is repriced with the vectorized engine
  (`n_paths=20000`, fixed seed = intentional CRN across time so moves are
  market moves, not MC noise); full Greeks every `greeks_every`-th
  snapshot → `monitor.on_snapshot`. `check_staleness` runs per snapshot.
- Sizing: 20k paths ≈ 1.4 s wall vs 6 sim-seconds between snapshots;
  Greeks runs ≈ 4x ≈ 5.6 s on a 30-sim-second cadence — a reprice always
  fits its cadence.
- `--inject-fault`: mid-run, two sweeps of crossed (bid > ask) quotes on
  one strike (all expiries, call+put). Crossed quotes fail
  `is_valid()` → dropped by the store; the drop counter must spike and the
  book must stay clean. Proven in the log and `results/p5_live.json`.
- Writes `results/p5_live.json` (config, stats, price series, alert list,
  fault record, staleness self-test) and prints a run summary. Runs
  unattended.

### 8.4 P5 acceptance (met 2026-09-29)

- 13 new pytest green (63 total): vectorized equivalence + timing gate,
  monitor alert lifecycle (all four kinds, thresholds, cooldown).
- `python3 scripts/p2_benchmark.py` re-run; before/after in PERF.md.
- `python3 scripts/live_loop.py --sim-seconds 300 --inject-fault`
  completes unattended; the injected fault is correctly alerted in
  `results/p5_live.json` (drop-counter spike; quarantine path also
  exercised — see `docs/p5_report.md`).
- Honest limitations (§8.1, `docs/p5_report.md`): GPU future work (no GPU
  on this VM), synthetic feed only (no broker connected — `BrokerWsFeed`
  still a stub), Greeks cost 4x the core reprice (cadence-limited at
  scale), draws are the remaining CPU floor.
