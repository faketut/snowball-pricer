# snowball-pricer

Real-time implied-volatility-surface-driven **multi-asset (basket) snowball / autocallable
option pricer** via Monte Carlo path simulation.

## Positioning

This is a **standalone project, independent from [Legos](https://github.com/faketut/legos)**.
Legos solves a *latency* problem (nanosecond tick-to-trade, zero-overhead hot path);
this project solves an *accuracy* problem (compute-bound Monte Carlo pricing, millisecond
to second scale, throughput-oriented). Different objective, different constraints,
different optimization targets — forcing one into the other's architecture would hurt
both. What is shared is engineering discipline, not code: Spec-Driven Development,
`SPEC.md` before implementation, criterion-style regression anchors, `PERF.md` baselines.

Planned data flow (P1+):

```text
broker WebSocket ticks -> IV inversion -> arbitrage-free SSVI surface
    -> versioned VolSurface snapshot (lock-free publish)
    -> Dupire local-vol MC engine (QMC + Brownian bridge)
    -> price + Greeks -> RFQ / risk
```

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| **P0** | Define X: quantify the three error classes on a simplified snowball | ✅ done (`scripts/p0_error_scaling.py`, `docs/error_budget.md`) |
| **P1** | Data pipeline: WS tick ingest → per-quote IV inversion → SSVI arbitrage-free surface → versioned snapshot publish | planned |
| **P2** | Pricing engine: Dupire local vol + Brownian bridge + Sobol QMC; AAD Greeks | planned |
| **P3** | Correlation module (basket / worst-of) + stress scenarios | planned |
| **P4** | Historical replay validation: rebuild past surfaces from tick history, hedge-simulation P&L explain | planned |
| **P5** | Real-time loop + monitoring (surface arbitrage alerts, Greeks drift alerts) | planned |

## Error budget (targets)

| Error class | Source | Target |
|---|---|---|
| ε_num | Monte Carlo sampling error | < 0.1% of notional |
| ε_stale | Stale input: intraday IV moves not reflected in surface | < 0.3% of notional |
| ε_model | Model-vs-dealer-quote deviation (dynamics assumption, correlation, wings) | < 2% of notional |

Rationale and measured P0 numbers: see [`docs/error_budget.md`](docs/error_budget.md).
The honest claim of this project is shrinking **ε_stale** (real-time surface vs
yesterday's close) — ε_model dominates and is addressed by model choice (P2),
correlation handling (P3), and reserves, not by data freshness.

## Quickstart (P0)

```bash
python3 scripts/p0_error_scaling.py   # ~1 min CPU; writes results/p0_results.json
```

## Layout

```text
SPEC.md                  project-level spec (product mechanics, data contracts, error budget)
docs/error_budget.md     P0 quantitative error-budget analysis
scripts/p0_error_scaling.py   P0 Monte Carlo experiments
results/p0_results.json  measured P0 numbers (generated)
```
