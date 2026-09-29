# SPEC.md — snowball-pricer (project level)

## 1. Product mechanics (reference snowball / autocallable)

Unless a term sheet overrides, the reference contract priced by this project is:

- **Underlying:** basket of N regional equity underlyings (constituents TBD per
  term sheet). **Basket vs worst-of linkage: TBD — user to confirm product form.**
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

Derived: `mid`, `spread`, `key()` = `(underlying_id, expiry, strike, is_call)`,
`discount()`, `forward_spot()` (fallback forward).

### 2.2 Feed interface (`Feed.subscribe(symbols) -> Iterator[OptionQuote]`)

- Unbounded quote iterator; caller drives pacing, no hidden threads.
- `SyntheticTickFeed`: validation backbone — quotes from a known ground-truth
  SSVI surface + microstructure noise (spread = max(tick, spread_bps·mid),
  iid Gaussian mid noise, deterministic per seed). **No broker is connected.**
- `BrokerWsFeed`: explicit stub (raises `NotImplementedError`). The adapter
  contract is documented in its docstring: approved-flow credentials only,
  exchange-ts normalization, heartbeat/sequence-gap health, reconnect with
  backoff + full snapshot resync, raw bid/ask forwarded (no mid fabrication).
  **Open: broker venue choice (MiniQMT / QMT / 恒生 / vendor) — needed from Jian.**

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

## 3. Pricing engine contract (P2 placeholder)

- Dynamics: Dupire local volatility calibrated from the snapshot (P2); SLV as
  follow-up. Model choice is recorded per price (model risk attribution).
- Path generation: Sobol QMC + Brownian bridge; barriers evaluated at
  contractual observation dates; BGK-corrected bridge on any coarsened step.
- Greeks: adjoint (AAD) / likelihood-ratio — no bump-and-revalue on the
  real-time path.
- Every price ships with: model id, surface version, N paths, SE estimate,
  ε-budget attribution.

## 4. Error budget (acceptance criteria)

| Class | Target | P0 measured | Gate |
|---|---|---|---|
| ε_num | < 0.1% notional | 0.017% @640k paths; N\*≈18k | P0 ✅ |
| ε_stale | < 0.3% notional | 0.39–0.58% on 2–3pt IV move w/ daily surface | P1+P5 must demonstrate shrink |
| barrier discretization | reported | −0.3 bps this contract; BGK residual −0.0 bps | P0 ✅ (machinery validated) |
| ε_model vs dealer quote | < 2% (1% stretch) | n/a | P4 replay |

## 5. Non-goals / constraints

- **Not** a latency system: ms–s pricing cadence is fine; no ns tail-latency
  requirement (explicitly out of scope — that is Legos' problem, a separate repo).
- No order execution in this repo (a future hedging executor would live with the
  execution stack, not here).
- No retail broker API keys in-repo; credentials via the approved connection
  flow only.
