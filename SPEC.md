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

## 2. Data contracts (P1 placeholder — to be specified before implementation)

- `VolSurfaceSnapshot`: **versioned, immutable**; fields: `asof` (exchange ts),
  per-expiry **SSVI parameters** (arbitrage-free by construction) *or* strike
  grid IVs, forward curve, discount curve, dividend assumptions, barrier-shift
  metadata. Exact schema TBD in P1 SPEC.
- **Publish/subscribe:** lock-free single-publisher snapshot ring; pricing
  threads always read a complete, versioned snapshot — never a torn surface.
  Tick-to-snapshot latency budget TBD in P1.
- **Invariants:** no calendar-spread arbitrage, no butterfly arbitrage across
  the published surface (P1 acceptance gate); wing extrapolation rule explicit
  (KI barrier lives in the deep-OTM wing — extrapolation choice is a priced
  assumption, logged per snapshot).

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
