# P1 design note — calibration choices

## Why raw-SVI per expiry + SSVI global (two stages)

- **Stage A (raw SVI per expiry)** is expiry-local and robust: 5 parameters,
  15 points, converges in milliseconds from an ATM-seeded start. Its job is
  not to be the published surface — it is to produce a stable **ATM total
  variance seed** `theta_T` per expiry without any cross-expiry coupling.
  Cross-expiry joint fits are fragile when one expiry's book is thin; stage A
  isolates that failure to one expiry (which then fails the `min_strikes`
  check loudly instead of poisoning the whole surface).
- **Stage B (global SSVI)** is the published form because it admits
  *checkable* no-arbitrage sufficient conditions (Gatheral–Jacquier) and has
  only 3 global parameters, so it cannot overfit one noisy expiry. The
  numeric Durrleman / calendar / Lee gates in `arbitrage.py` are the binding
  acceptance check — the fit-time penalty `eta·(1+|rho|) ≤ 1.9` is only a
  soft guardrail to keep the optimizer in the safe region.
- Alternative considered: single-stage joint SSVI fit of `(theta_T, rho, eta,
  gamma)`. Rejected for P1 — it couples a bad expiry's noise into the global
  shape parameters, and debugging which expiry broke the surface is harder.
  Revisit if stage A's seeds ever disagree with stage B's thetas materially
  (log both; currently they agree to ~1e-3).

## Forward handling

Put-call parity at the strike nearest the spot-implied forward, among strikes
quoting both a call and a put. Rationale: parity forwards remove the
rate/dividend assumption from the smile — the smile is then a pure function
of (K, F). Fallback is the spot-implied forward. Single-sided quotes still
invert against whichever forward was chosen (their weight is down-weighted
10× when no bid/ask IV spread is available).

## Wings and the knock-in barrier

The KI barrier (75–80%) lives in the deep-OTM-put wing — exactly where market
quotes are thinnest and any extrapolator is a *priced assumption*. P1 policy:
- No ad-hoc extrapolation (no flat-IV or linear wings pasted on). The SSVI
  formula itself extrapolates; its asymptotic slope is bounded by Lee's
  moment formula (`w/|k| ≤ 2`), enforced by the numeric wing gate.
- The extrapolation choice is logged per snapshot (SSVI params are the log).
- P2 (Dupire local vol) will inherit this wing shape directly — any wing
  mis-specification flows into the KI price. This is flagged as a model-risk
  item, not solved here.

## What breaks first in real data (expected failure modes, in order)

1. **Stale / crossed quotes.** Real books cross and go stale; `is_valid()`
   drops crossed, the inverter drops uninvertible, `min_strikes` fails loudly.
   What we do *not* yet have: feed-health awareness (how old is the newest
   quote per expiry?). The `BrokerWsFeed` contract requires heartbeat /
   sequence-gap events for exactly this reason — wire them into snapshot
   metadata (`asof` vs wall-clock skew alert) in P5.
2. **Wide wing spreads.** Real OTM wing quotes have huge relative spreads;
   the 1/spread weighting handles this gracefully (wings get low weight), but
   if the *entire* wing is wide the smile there is extrapolation, not data.
   Consider logging effective wing sample weight per snapshot.
3. **Discrete dividends / borrow.** We use cont-compounded `div_yield`; real
   single-stock dividends are discrete — parity forwards absorb most of this,
   but ex-div dates near short expiries will distort the smile. Known
   limitation, revisit when the underlying universe is fixed.
4. **American exercise.** Out of scope by design (SSE ETF options are
   European). If the universe ever includes American-style underlyings, a
   de-Americanization step must precede inversion.
5. **Corporate actions / expiry calendar.** Strike/expiry keys assume clean
   symbology; a real adapter must normalize splits and special dividends.

## Deliberate simplifications (documented, not hidden)

- Snapshot-level scalar `(spot, rate, div_yield)`; per-expiry forwards stored
  but no full forward/discount *curves*. Term-structure of rates is second
  order for a 2Y equity snowball — revisit in P2 if Dupire needs it.
- `theta_at(T)` flat beyond the last calibrated expiry. Pricing beyond 2Y
  uses the 2Y smile shape — conservative and explicit; do not silently
  extrapolate term structure.
- Least-squares in total-variance space with 1/IV-spread weights. Vega
  weighting is the textbook alternative; spread weighting won because it
  directly encodes quote quality (wide = uncertain).
