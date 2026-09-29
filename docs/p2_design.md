# P2 design notes — pricing engine

## 1. Dupire local volatility

The engine prices under Dupire local volatility calibrated to the P1
arbitrage-free SSVI snapshot. We use Gatheral's total-variance form:

    sigma_loc^2(k,T) = (dw/dT) / [1 - (k/w)(dw/dk)
                        + 1/4(-1/4 - 1/w + (k/w)^2)(dw/dk)^2 + 1/2(d2w/dk2)]

with k = log(K/F). For SSVI the k-derivatives are closed-form in
(rho, eta, gamma) and dw/dT = (dw/dtheta)(dtheta/dT) with dtheta/dT exact on
P1's piecewise-linear theta term structure. The analytic path was
cross-validated against finite differences (max abs diff 6e-7 on the
synthetic surface) and is the production path; a finite-difference fallback
covers non-SSVI surfaces (e.g. the FlatVol test shim).

Numerical notes (all verified empirically, see tests/test_pricing.py):
- Because P1 guarantees no-arbitrage, the Dupire denominator is positive and
  the raw local variance is non-negative everywhere on the grid (0% floored).
- SSVI wings imply very high local vol far out of the money (up to ~440% at
  k=±4, short T — genuine SSVI behavior, confirmed against FD, not a bug).
  The grid caps at 150% vol; the cap binds only at |k| ≳ 0.5 in the wings,
  a region where the snowball payoff is already decided (KO'd or deep KI),
  so the distortion is economically negligible. Diagnostics record the
  binding fractions.
- At/beyond the last expiry, P1's theta_at is flat, which would imply zero
  local vol on the final grid slice. We instead extend the last segment's
  slope — a deliberate, documented deviation (extrapolation artifact, not
  market information).
- Queries use vectorized bilinear interpolation on a 161×61 (k, T) grid,
  k ∈ [-4, 4]; the KI barrier sits at k ≈ -0.22, deep inside the grid.

## 2. Forward-skew limitation (the honest caveat)

Calibrating local vol to today's vanilla smile does NOT reproduce the
market's forward volatility dynamics. Local vol is a "sticky" model: its
forward smile is systematically flatter than what stochastic-vol markets
imply. The snowball's knock-in put is economically a *forward-starting*
option (it only matters conditional on surviving the autocall dates), so it
is exactly the piece most sensitive to forward skew.

Consequence: the P2 engine will price the KI leg with a flatter forward
skew than the market charges. Direction and rough size: LV typically
*underprices* forward skew exposure; for the canonical 2Y structure the
effect is order 0.5–2% of notional (to be quantified against dealer quotes
in P4). Mitigations, in order:
1. **Model reserve** (now): carry an explicit reserve against forward-skew
   mispricing; size it from the LV-vs-SLV spread once SLV exists.
2. **Stochastic local volatility (SLV)** (future): calibrate a
   vol-of-vol / mean-reversion pair to the (sparse) forward-skew market
   (forward-starting options, VIX-like instruments where available).
   This is the proper fix and is scoped as post-P3 work.

Real-time IV surface updates do not fix this: they refresh today's smile,
not the dynamics assumption. See docs/error_budget.md (epsilon_model).

## 3. Monte Carlo: Sobol + Brownian bridge

- QMC driver: scrambled Sobol, dimension = n_assets × n_steps. Normals are
  assigned to Brownian-bridge bisection order (terminal value first, then
  successive midpoints, breadth-first), which puts the highest-variance
  components on the leading Sobol dimensions where uniformity is best.
- Multi-asset: independent bridges per asset, correlated per (path, step)
  with the Cholesky factor of the input correlation matrix. Marginals are
  preserved exactly. (P2 default: identity; calibrated correlation is P3.)
- Antithetic fallback when QMC is off.
- Batching: one Sobol sequence advanced across batches; the point set
  therefore depends mildly on batch_size (documented; benchmarks fix it).
- The bridge *construction* (path generation) is unrelated to the optional
  bridge *barrier correction* (see §4) — similar name, different machinery.

## 4. Barrier monitoring: contractual vs continuous-touch

Contractual baseline (default): barriers are monitored exactly at their
observation dates — monthly KO observations, KI at every simulation step
(steps_per_day=1 ⇒ daily closes, matching P0). No correction.

Optional bridge correction (`ki_bridge`/`ko_bridge`): approximates
*continuous* monitoring via P(touch | endpoints) with the instantaneous
basket vol frozen per interval. This is a modeling variant, not the
contract. It is validated against the reflection-principle touch
probability (test_ko_probability_bridge_vs_touch_analytic).

Known feature of the continuous variant: price vs spot is locally
non-monotonic at the KO barrier (touch-from-below probability exceeds
stay-above probability near the barrier under discrete observation).
Delta too close to the barrier is therefore ill-behaved; the Greeks test
asserts delta in the continuation region instead. This is discrete-barrier
economics (cf. Broadie–Glasserman–Kou), not an implementation bug.

## 5. Greeks

- Delta: central finite difference with Common Random Numbers (identical
  Sobol draws and correction streams for base and bumped runs), barriers
  held fixed in absolute terms via `norm_spots` (without this, delta is
  identically zero by the product's homogeneity — documented in
  payoff.evaluate_snowball).
- Pathwise/adjoint delta is deliberately NOT implemented: the payoff is
  discontinuous at both barriers (digital autocall, digital KI trigger),
  so pathwise differentiation fails exactly where risk concentrates.
- Vega: +1 vol-point parallel bump of the *implied* surface (the quantity
  the desk marks), Dupire grid rebuilt, repriced under CRN. Validated:
  bridges-off vega = -0.203%/pt vs P0's -0.195%/pt (4% — MC noise).
- KO probability with binomial SE is reported as a trader-facing risk
  metric (expected-tenor proxy).
- Not covered: gamma, theta, rates rho, correlation Greeks (P3), full AAD
  (would speed multi-Greek batches; CRN is exact and fast enough at P2).

## 6. Performance

Pure-Python/numpy streaming simulator. Per-step cost is a handful of
vector ops plus one bilinear local-vol lookup per asset; the Python loop
over steps (≈504 for 2Y daily) dominates. See PERF.md for measured
throughput. A numba/Rust port of the step loop is the obvious next
performance step if repricing frequency demands it (P5).
