# P0 — Error-budget quantification

**Status:** done 2026-09-29 · script: `scripts/p0_error_scaling.py` ·
raw numbers: `results/p0_results.json` (regenerable, git-ignored)

## 1. Objective

Pin down what "pricing error within X% of the real market" means before building
anything, by measuring the three error classes separately on a simplified snowball.
The project's honest value proposition is shrinking **ε_stale**; ε_num is pure
engineering; ε_model dominates and is *not* fixed by data freshness.

## 2. Contract & assumptions (P0)

| Item | Value |
|---|---|
| Underlying | single asset, GBM, **flat σ = 25%** (P0 assumption) |
| S0 / T / r | 100 / 2Y / 3% flat |
| Knock-out | **monthly** @ **100%** of initial → autocall, payoff `(1 + 15%·t)` discounted |
| Knock-in | **daily closes** @ **80%** of initial |
| Maturity (no KO) | no-KI → `(1 + 15%·T)`; KI → `min(1, S_T/S0)` (discounted) |
| Notional | 1 (all errors quoted in %/bps of notional) |

Deferred to P2/P3: vol term-structure/skew, stochastic vol, basket/worst-of
correlation, discrete dividends, funding curve.

**Method:** antithetic variates throughout; batched simulation (20k pairs/batch);
`numpy` `default_rng` seeded (1234 / 777 / 4242) → reproducible. Runtime 46s CPU.

## 3. (a) Numerical error ε_num — solved problem

| N paths | Price | SE | SE·√N |
|---|---|---|---|
| 10,000 | 0.98699 | 0.001334 | 0.1334 |
| 40,000 | 0.98652 | 0.000667 | 0.1334 |
| 160,000 | 0.98621 | 0.000336 | 0.1345 |
| 640,000 | 0.98600 | 0.000169 | 0.1349 |

- **1/√N scaling verified**: SE·√N constant to 3 digits.
- **N ≈ 18,200 paths ⇒ SE < 0.1% of notional** (extrapolated; the 640k run
  achieved SE = 0.0169%, 6× below target).
- Implication: budget a few ×10⁴ paths per reprice and ε_num is negligible.
  P2 will push further with Sobol QMC, but there is no accuracy argument for
  more — only a latency/throughput one.

## 4. (b) Vega → staleness error ε_stale — the project's core claim

±1 vol-point bump with **common random numbers** (same normals re-scaled;
SE of the vega estimate only 0.003pp):

| Metric | Measured |
|---|---|
| P(26%) / P(24%) | 0.98391 / 0.98781 |
| **Vega** | **−0.195% of notional per vol point** (short vol, as expected for autocallables) |
| ε_stale, 2pt intraday IV move | **0.39%** of notional |
| ε_stale, 3pt intraday IV move | **0.58%** of notional |

Interpretation:

- A **daily-close surface** left stale through a volatile session misprices this
  contract by ~0.4–0.6%. A **real-time surface** (rebuilt on tick flow) compresses
  that toward the surface-build latency — this is the quantified "X%" the
  WebSocket feed buys.
- Measured vega sits at the low end of the 0.3–0.8%/pt rule of thumb because the
  100% monthly KO gives a **short expected tenor** (~2 months; ~50% knock out in
  month 1), which compresses all sensitivities. Two nuances for later phases:
  1. The *live book* (snowballs that survived = the risky ones) has larger vega
     than this at-inception average — staleness hurts most exactly where risk
     concentrates.
  2. Higher-KO structures (103–105%) live longer and carry proportionally larger
     vega; ε_stale is contract-dependent and must be re-measured per term sheet
     (P4).

## 5. (c) Discrete-barrier bias — negligible *for this contract*, machinery validated

| Variant | Price | Δ vs contractual discrete |
|---|---|---|
| Daily-close KI (contractual) | 0.98581 | — |
| Continuous KI via Brownian bridge | 0.98578 | **−0.3 bps** |
| Continuous + BGK continuity correction (B\* = 79.269) | 0.98580 | −0.0 bps |

- The discrete-vs-continuous gap is **−0.3 bps**: negligible here, *not* because
  the bridge is unnecessary, but because early KO (≈2-month expected life) means
  the 80% KI barrier is rarely threatened — few paths ever get close enough for
  intra-day touch probability to matter.
- The **BGK correction recovers the discrete price to −0.0 bps residual**,
  validating the bridge machinery that P2 needs anyway: for longer-tenor
  structures (higher KO / lower vol / nearer barriers) the textbook bias is
  **tens of bps**, and naive coarse-step simulation would silently eat it.
- Engineering rule going forward: always simulate at observation dates or finer,
  with bridge correction on any coarsened step — the P0 harness proves the
  implementation is correct.

## 6. ε_model — not quantifiable in P0 (deferred, by design)

Model-vs-dealer-quote deviation needs (i) a real calibrated surface with dynamics
(Dupire/SLV — P2), (ii) correlation assumptions (P3), and (iii) dealer quotes to
compare against (P4 replay). Target stays **< 2% of notional** (1% stretch), set
against a ~1–3% dealer bid-ask: tighter than that is fitting noise.

## 7. Summary: measured vs target

| Error class | Target | P0 measured | Verdict |
|---|---|---|---|
| ε_num (MC sampling) | < 0.10% | 0.017% @640k; N\*≈18k for 0.10% | ✅ solved |
| ε_stale (2–3pt IV move) | < 0.30% | 0.39–0.58% with daily surface | 🎯 this is what the real-time feed must kill |
| barrier discretization | — | −0.3 bps (contract-dependent) | ✅ machinery validated via BGK |
| ε_model (vs dealer quote) | < 2% | n/a in P0 | ⏳ P2–P4 |

**Bottom line:** P0 confirms the project thesis with numbers. Numerical error is
irrelevant at modest path counts; a stale surface costs ~0.5% on volatile days
(the thing we are building to eliminate); model error remains the dominant,
unmeasured term and is the real risk to the <2% target.
