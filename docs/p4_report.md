# P4 — Historical replay validation: regime replay + delta-hedge P&L explain

Date: 2026-09-29. Seed `20260929` (fixed; same seed ⇒ bit-identical rows
and attribution — covered by `tests/test_validation.py`).

## What was built

`snowball_pricer/validation/` — two modules, numpy/scipy only:

- **`replay.py`** — `RegimeTickFeed`: deterministic synthetic multi-regime
  option-quote feed (piecewise-constant SSVI ground truth; one full quote
  sweep = one trading day; spot constant within a sweep, daily GBM across
  sweeps with regime-dependent vol; separate RNG streams for spot vs
  quote-noise). `run_replay` drives the feed through the REAL P1 pipeline
  (`run_pipeline`, full surface rebuild per sweep) and reprices every
  published snapshot with the REAL P2 engine (`engine.price`, 20k QMC
  paths, Greeks on), aging the remaining tenor one day per sweep.
- **`hedge.py`** — `simulate_hedge`: delta-hedges a SHORT snowball along
  the TRUE spot path, daily rebalance to each snapshot's model delta
  (hold-last between reprices), cash accrual, true-path KO observation and
  KI monitoring, with an exact accounting identity
  `premium − final_value + stock_pnl + interest_pnl == realized_pnl`
  (unit-tested). `pnl_explain`: SHORT-perspective attribution into
  delta / vega / carry buckets + residual.

Two realism fixes made during P4 (documented, not hidden):
1. Spot is constant within each daily sweep. Per-tick spot motion smeared
   the smile inversion (P1 uses one forward per expiry); daily-constant
   spot keeps each snapshot's calibration as clean as P1's static case.
2. The strike grid is FIXED for the whole replay (listed-style). A
   per-day re-centered grid accumulated stale strikes in the pipeline's
   latest-per-key `QuoteStore` and biased the calibration.
3. The replayed contract is a SEASONED snowball: `engine.price` takes an
   optional `norm_spots` (day-0 spot), so barriers/observations stay fixed
   in absolute terms while the spot moves. A fresh contract re-struck each
   day sits exactly on its KO barrier (degenerate ~zero delta); the
   seasoned convention is the economically meaningful one.

## Canonical run (the numbers below)

- 14 trading days: 5 calm (ATM 20%) → 5 vol spike (ATM 45%) → 4 calm (ATM 22%)
- Canonical 2Y snowball: monthly KO @ 100%, daily KI @ 80%, 15% coupon,
  basket-average on one synthetic underlying, notional 1.0
- 14 snapshots × 20,000 QMC paths, Greeks on; seed 20260929
- Regime ATM reference = 1Y tenor (same tenor `atm_iv` is reported at), so
  the configured 20%/45%/22% match the measured 1Y ATM

### Replay surface quality — 14/14 snapshots published, no failures

| day | spot | 1Y ATM iv | price | delta | vega/pt | KO prob |
|----:|-----:|----------:|------:|------:|--------:|--------:|
| 0 | 100.00 | 20.00% | 0.9839 | 0.003 | −0.00186 | 89.66% |
| 1 | 100.80 | 20.02% | 0.9866 | 0.003 | −0.00176 | 90.79% |
| 2 | 101.07 | 19.98% | 0.9874 | 0.003 | −0.00162 | 91.24% |
| 3 | 100.50 | 20.01% | 0.9858 | 0.003 | −0.00181 | 90.42% |
| 4 | 101.88 | 20.00% | 0.9900 | 0.003 | −0.00165 | 92.27% |
| 5 | 101.87 | 45.03% | 0.9423 | 0.004 | −0.00191 | 87.54% |
| 6 | 100.64 | 45.04% | 0.9385 | 0.004 | −0.00196 | 86.54% |
| 7 |  94.20 | 44.96% | 0.9062 | 0.006 | −0.00171 | 79.86% |
| 8 |  95.65 | 45.01% | 0.9150 | 0.005 | −0.00162 | 81.66% |
| 9 |  93.14 | 44.99% | 0.8989 | 0.006 | −0.00173 | 78.36% |
| 10 | 91.64 | 22.00% | 0.9395 | 0.007 | −0.00232 | 71.35% |
| 11 | 93.03 | 21.97% | 0.9481 | 0.006 | −0.00239 | 74.91% |
| 12 | 92.77 | 21.99% | 0.9471 | 0.006 | −0.00235 | 74.23% |
| 13 | 93.87 | 22.02% | 0.9525 | 0.006 | −0.00217 | 76.92% |

- Calm → spike → calm recovered 1Y ATM: **20.00% → 45.03% → 22.00%**,
  matching the configured regime levels (regime ATM reference is the 1Y
  tenor by construction).
- The vol spike (day 4→5, +25 vol points) cut the long price
  0.9900 → 0.9423 (−477 bps); the collapse (day 9→10, −23 points) lifted
  it 0.8989 → 0.9395 (+406 bps). KO probability fell 92% → 71% as spot
  slid toward the KI barrier — the model reprices the barrier economics
  every day.
- Deltas are small (0.003–0.007): this contract is ~90% likely to
  autocall, so the KO redemption dominates and the mark is nearly
  spot-insensitive in the continuation region. Verified by brute force:
  ±1/±5 spot bumps on a fixed snapshot give ΔP/ΔS ≈ 0.002–0.003,
  consistent with the CRN delta — the small delta is real economics, not
  an engine bug.

### Delta-hedge P&L explain — SHORT snowball, hedged (bps of notional)

| bucket | bps | what it is |
|--------|----:|------------|
| delta (net hedge − model) | **+0.0** | Σ (h_d − δ_long,d)·dS_d; hedge replicates the model delta exactly |
| vega (−v·dATMiv) | **+14.0** | +413 on the spike, −398 on the collapse, ≈ net flat |
| carry (model theta) | **−15.2** | frozen spot+surface, tenor −1 day per snapshot, SHORT sign |
| carry (cash interest) | **+8.3** | 3% on the cash balance |
| **residual** | **+44.4** | everything first-order misses (see below) |
| **ACTUAL realized** | **+51.5** | premium 0.9839 → final mark 0.9525, no KO |

Attribution identity holds by construction:
0.0 + 14.0 − 15.2 + 8.3 + 44.4 = +51.5.

### Reading the residual honestly (+44.4 bps)

The residual is **not a pricing error** — P2 validates pricing separately
(3 bps vs Black-Scholes, 35 bps P0 cross-check). It is the unhedged
second-order risk of a delta-only hedge, in order of importance:

1. **Gamma (dominant).** The engine has no gamma and the hedge holds none.
   Day 6→7: spot −6.4% in one day; the long lost ~65 bps more than
   delta+vega+theta explain — that gamma P&L accrues to the residual
   (with the SHORT sign, a gain here).
2. **Vega convexity / ATM-only approximation.** Vega is state-dependent:
   it grew from −0.00165/pt (day 4, spot 101.88) to −0.00232/pt (day 10,
   spot 91.64, nearer KI). The bucket uses beginning-of-day ATM vega ×
   ΔATMiv, so vega-of-vega and skew changes land in the residual.
3. **QMC noise in Greeks** (20k paths; delta SE-scale ~1e-3 on a
   0.003–0.007 delta), **surface recalibration noise** (each snapshot is
   a fresh SSVI fit to noisy quotes), and **theta finite-difference
   noise** (20k-path reprices at T and T−1d).
4. No terminal settlement gap in this run (no KO; bought back at the
   model mark). A KO day's redemption-vs-mark gap would land here.

The delta bucket is 0.0 **by construction** — the hedge is rebalanced to
the model's own delta, so net delta P&L is zero and the test of the model
moves entirely into the residual. That is the point: first-order spot
risk was neutralized exactly; what remains is second-order.

## Timings (measured 2026-09-29, this VM)

- Replay (14 snapshots × 20k QMC, Greeks on): **96 s**
- Hedge simulation: < 0.1 s
- Theta explain (13 frozen-market reprices × 20k paths, no Greeks): **44 s**
- **Total: 140 s**

## How to reproduce

```
python3 scripts/p4_replay.py        # ~2.5 min; writes results/p4_replay.json
python3 -m pytest tests/test_validation.py -q
```

## Files

- `snowball_pricer/validation/replay.py`, `snowball_pricer/validation/hedge.py`
- `scripts/p4_replay.py`, `tests/test_validation.py`
- `SPEC.md` §7 (design + the three realism fixes)
