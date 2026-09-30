# yfinance delayed feed — quality notes (Phase 1.5)

## What this is

`YFinancePollFeed` (`snowball_pricer/feeds/yfinance.py`) polls Yahoo option
chains and feeds the **real P1 pipeline** (IV inversion → SSVI surface →
arbitrage gate). It is a **zero-cost stepping stone**, not a real-time feed:

- **Every quote has `is_delayed=True` hardcoded.** Yahoo data is ~15 minutes
  delayed. Nothing in this document or the code may be presented as
  real-time.
- Purpose: validate that the P1/P2 machinery works on real market-data
  *shapes* (wide chains, missing quotes, American-style underlyings) for
  free — before spending money on a real-time package.

## Measured numbers (2026-09-29 ~22:45–22:50 EDT, AFTER HOURS)

Run: `python3 scripts/yfinance_quality.py --underlyings SPY,QQQ,IWM
--cycles 3 --poll-interval 60 --max-expiries 4 --min-expiry-days 7`
(raw: `results/yfinance_quality.json`, gitignored).

| Underlying | Rows/cycle | Valid bid+ask | Completeness | Pipelines published | Calib. RMSE (IV pts) |
|---|---|---|---|---|---|
| SPY | 963 | 933 | **96.9%** | 3/3 | 0.0719 |
| QQQ | 1050 | 1004 | **95.6%** | 3/3 | 0.1326 |
| IWM | 472 | 431 | **91.3%** | 3/3 | 0.1093 |
| **Overall** | 2485 | 2368 | **95.3%** | **9/9 (quarantine 0%)** | — |

Setup per cycle: 4 expiries ≥ 7 days out, log-moneyness wing filter
k ∈ [-0.5, 0.3]; quotes with missing/zero bid/ask are dropped by the
QuoteStore (counted as incomplete, never fabricated).

## Caveats — read before citing these numbers

1. **After-hours lower bound.** The market was closed (22:45 EDT). Chains
   were static snapshots from the 16:00 ET close — all 3 cycles returned
   byte-identical rows, which is *expected*, not a bug. Off-session,
   market makers pull quotes, so the 91–97% completeness is a **lower
   bound**; a 09:30–16:00 ET re-run should do better.
2. **RMSE is fit quality, not market truth.** 0.07–0.13 IV points is the
   SSVI fit residual against the (delayed, possibly stale) mid quotes —
   it measures how well the surface explains the input, not pricing error
   vs the real market.
3. **American-style underlyings.** SPY/QQQ/IWM options are American; P1's
   inverter assumes European exercise. Short-dated deep-ITM put IVs carry
   early-exercise premium (we excluded <7d expiries, which mitigates this).
4. **What this does NOT prove.** The delay is fixed, so this experiment
   cannot measure the value of *removing* it (ε_stale). It proves the
   pipeline survives real quote shapes — nothing more.

## Incident log (honest)

- First run mixed all 3 underlyings into one QuoteStore → the arbitrage
  gate quarantined 3/3 cycles with a **butterfly violation**. Correct
  behavior: surfaces are per-underlying, and the experiment was fixed to
  run one pipeline per underlying (9/9 published). Lesson recorded, not
  hidden.
- One transient Yahoo timeout occurred during development (curl 30s);
  direct retry succeeded. The feed's per-symbol try/except + backoff
  covers this in production use.

## Recommendation

- Re-run `scripts/yfinance_quality.py` during **09:30–16:00 ET** for the
  fair completeness/RMSE measurement.
- Keep the 60s+ cadence and request pacing; Yahoo throttles aggressively.
- When the Questrade live adapter is smoke-tested, compare its surface
  RMSE against this baseline on the same underlyings — that is the
  apples-to-apples delayed-vs-live read.
