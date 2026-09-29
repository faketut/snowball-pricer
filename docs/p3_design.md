# P3 design — correlation module

## Why this module exists

The P2 engine prices basket-average snowballs and accepts a correlation
matrix, but correlation was either identity (default) or user-supplied.
P3 answers: *what should the matrix be, and how much does it matter?*
Since the product linkage is basket-AVERAGE (confirmed 2026-09-29, SPEC §1),
the module's job is to **quantify** a sensitivity that is expected to be
mild — not to assume it.

## Estimators (`correlation/estimate.py`)

Three estimators, three jobs:

| Estimator | Input | Job | Cost model |
|---|---|---|---|
| `rolling_pearson(returns, window)` | (n_obs, n) log returns | trailing realized corr, one matrix per window | one eig-less cov per window; pure-Python window loop |
| `ewma_corr(returns, lam=0.94)` | same | RiskMetrics exponential decay, one current matrix | O(n_obs·n²) Python loop |
| `implied_correlation(index_var, comp_vars, weights)` | index + constituent implied variances | forward-looking average ρ from the variance identity | O(n²) |

**Why both realized estimators?** Rolling Pearson is the transparent
baseline (each window is interpretable, window length is the honest tuning
knob). EWMA is the RiskMetrics production standard (lam=0.94 daily) —
smooth, but an *endpoint* estimator with effective N ≈ 1/(1−lam) ≈ 17
observations, hence high variance.

Measured sampling behavior (synthetic equicorrelated GBM, true ρ=0.6,
6000 obs, 12 seeds, 2026-09-29):

| lam | mean ρ̂ | std across seeds |
|---|---|---|
| 0.94 (default) | 0.53 | 0.13 |
| 0.985 | 0.58 | 0.03 |
| 0.995 | 0.59 | 0.018 |
| 0.999 | 0.60 | 0.011 |

Two lessons: (1) the estimator is **consistent** — bias and noise vanish as
decay slows, so the implementation is correct; (2) at the RiskMetrics
default the downward finite-sample bias and variance are large enough to
matter. The test suite pins both behaviors (`test_ewma_recovers_true_rho`
uses an honest ±0.15 tolerance at lam=0.94; `test_ewma_consistent_as_decay_slows`
checks ±0.05 at lam=0.995). Practical consequence: use lam=0.94 only for
fast regime detection; use slower decay (or rolling windows ≥ 1y) for a
pricing-grade input.

**Implied correlation** backs out the average pairwise ρ from
σ_I² = Σwᵢ²σᵢ² + Σ_{i≠j}wᵢwⱼσᵢσⱼρ. Assumptions are documented in the
docstring, not hidden: a single average ρ for all pairs (pair structure
lost), no idiosyncratic basis (index variance fully explained by
components), same-tenor/same-convention implied variances. On synthetic
index data with no basis it recovers ρ exactly (|err| < 0.01 in tests);
in production, an extreme print (|ρ̂| → clip bound 0.99) is a signal that
the model is violated — treat it as a stress anchor, not a calibration.

**`nearest_psd`** (eigenvalue clipping + unit-diagonal rescale) keeps every
estimated or shocked matrix valid for the engine's Cholesky factor.
Idempotent on already-valid matrices.

## Stress scenarios (`correlation/stress.py`)

- `plus_0.2` / `minus_0.2`: off-diagonal ±0.2 (a mild diversification-premium
  move), clipped to [−0.99, 0.99].
- `crisis_1.0`: off-diagonal → 0.95 (correlation spike; a real 1.0 would
  singularize the Cholesky).
- `dispersion_0.0`: off-diagonal → 0.05 (dispersion-trade / decorrelated regime).

Diagonal stays 1.0; every scenario exits through `nearest_psd`.

`sensitivity_table` reprices a 3-asset basket-average snowball (FlatVol 25%,
2Y, monthly KO @100%, daily KI @80%, 15% coupon, equal weights) under base +
scenarios **with the same engine seed** — deltas are CRN-driven, not MC noise.
Measured 2026-09-29 (40k QMC paths, base ρ=0.5):

| scenario | ρ (off-diag) | price | Δprice (bps) | KO prob | ΔKO (pp) |
|---|---|---|---|---|---|
| base | 0.500 | 0.99511 | +0.0 | 89.31% | +0.00 |
| plus_0.2 | 0.700 | 0.99033 | −47.8 | 88.70% | −0.61 |
| minus_0.2 | 0.300 | 1.00057 | +54.6 | 90.03% | +0.71 |
| crisis_1.0 | 0.950 | 0.98592 | −91.9 | 88.30% | −1.02 |
| dispersion_0.0 | 0.050 | 1.00931 | +142.0 | 90.96% | +1.64 |

MC noise (SE ≈ 5.8 bps per reprice) is an order of magnitude below the
signals, so these deltas are real.

## Why average-linkage sensitivity is mild

Basket variance is σ_B² = Σwᵢ²σᵢ² + Σ_{i≠j}wᵢwⱼσᵢσⱼρᵢⱼ — a **linear, convex
combination** in ρ. A ±0.2 move in ρ moves σ_B² by only ±(0.2·Σ_{i≠j}wᵢwⱼσᵢσⱼ),
and price moves through the basket vol channel only. Measured: ±0.2 in ρ
moves price by ~55 bps — inside the P0 model-reserve ballpark, and an
order of magnitude below the stale-surface error (ε_stale 0.39–0.58%,
SPEC §5). Even the extreme crisis/dispersion scenarios move price by
≤142 bps.

The sign is NOT asserted a priori: higher ρ raises basket vol, which
simultaneously raises KI touch probability (bad for the investor) and
lowers discrete ATM autocall probability under fixed risk-neutral drift
(good for the investor). In this parametrization the KO channel wins —
higher ρ → lower price. A different term sheet (e.g. deep ITM autocall,
KI closer to spot) could flip it. The tests assert monotonic basket
variance and a magnitude bound (150 bps for ±0.2, ~3× the measured
value), never a direction.

## What would change for worst-of

With worst-of linkage, correlation is a first-order driver: the payoff
depends on the minimum constituent, so higher ρ (co-movement) **lowers**
the chance that exactly one constituent craters — the opposite of the
KI channel above, and much larger in magnitude. A worst-of snowball would
need: pair-structure-preserving shocks (not a single average ρ),
tail-dependence stress (correlation alone misses joint-crash clustering),
and much tighter estimator tolerance. The P3 machinery (estimators,
shock_corr, sensitivity_table) generalizes to that case; the scenario
definitions and test bounds do not transfer. Not in scope — the product
is basket-average.

## Open items

- Feed choice for realized correlation (constituent total-return series
  source) is undecided; estimators take plain (n_obs, n) return arrays so
  any feed plugs in.
- P5 (calibration/replay vs dealer quotes) will decide whether the
  pricing-grade matrix is realized (EWMA/rolling) or implied.
