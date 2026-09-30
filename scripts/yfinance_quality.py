"""Phase 1.5 quality experiment: measure yfinance delayed-feed data quality.

Polls real Yahoo option chains (3 underlyings x N cycles), feeds each
cycle's quotes through the REAL P1 pipeline (IV inversion -> SSVI ->
arbitrage gate), and records:

- per-cycle quote completeness (% strikes with valid bid AND ask)
- arbitrage-gate pass / quarantine rate
- calibration RMSE where a surface publishes

Writes results/yfinance_quality.json (gitignored).

HONEST CAVEATS (also in docs/yfinance_notes.md):
- This feed is ALWAYS 15-minutes delayed. Nothing here is real-time.
- Run after-hours, bid/ask completeness is a LOWER BOUND on quality
  (market makers pull quotes off-session). Re-run 09:30-16:00 ET for the
  fair measurement.
- Be gentle with Yahoo: default cadence/pace is conservative.

Usage:
    python3 scripts/yfinance_quality.py \\
        --underlyings SPY,QQQ,IWM --cycles 3 --poll-interval 60 \\
        --max-expiries 4 --out results/yfinance_quality.json
"""
import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from snowball_pricer.arbitrage import ArbitrageViolation
from snowball_pricer.feeds.yfinance import YFinancePollFeed
from snowball_pricer.pipeline import QuoteStore, rebuild_surface


def main() -> None:
    ap = argparse.ArgumentParser(description="yfinance feed quality probe")
    ap.add_argument("--underlyings", default="SPY,QQQ,IWM")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--poll-interval", type=float, default=60.0)
    ap.add_argument("--max-expiries", type=int, default=4)
    ap.add_argument("--min-expiry-days", type=int, default=7,
                    help="skip expiries closer than N days (0DTE/weekly "
                         "chains are noise for surface calibration)")
    ap.add_argument("--k-band", default="-0.5,0.3",
                    help="log-moneyness wing filter, e.g. -0.5,0.3 "
                         "(empty string disables)")
    ap.add_argument("--rate", type=float, default=0.03)
    ap.add_argument("--div", type=float, default=0.0)
    ap.add_argument("--out", default="results/yfinance_quality.json")
    args = ap.parse_args()

    underlyings = [u.strip().upper()
                   for u in args.underlyings.split(",") if u.strip()]
    k_band = None
    if args.k_band.strip():
        lo, hi = args.k_band.split(",")
        k_band = (float(lo), float(hi))

    feed = YFinancePollFeed(
        underlyings, rate=args.rate, div_yield=args.div,
        poll_interval_s=args.poll_interval, max_expiries=args.max_expiries,
        min_expiry_days=args.min_expiry_days, k_band=k_band,
    )

    cycles = []
    t_run = time.perf_counter()
    for ci in range(args.cycles):
        quotes, cstats = feed.poll_cycle()
        # One pipeline PER UNDERLYING: mixing SPY/QQQ/IWM quotes into a
        # single book creates a torn butterfly (verified 2026-09-30) —
        # surfaces are per-underlying in production too.
        by_sym: Dict[str, List] = {u: [] for u in underlyings}
        for q in quotes:
            by_sym[q.underlying_id].append(q)
        pipelines = {}
        for sym, sq in by_sym.items():
            store = QuoteStore()
            dropped = sum(1 for q in sq if not store.update(q))
            outcome = {"status": "n/a", "dropped_by_store": dropped,
                       "n_rows": len(sq)}
            if sq:
                last_ts = max(q.ts for q in sq)
                try:
                    surface, diag = rebuild_surface(
                        store, asof=last_ts, source="yfinance-quality")
                    outcome.update({
                        "status": "published",
                        "rmse_iv": diag.get("rmse_iv"),
                        "n_quotes": diag.get("n_quotes"),
                        "n_expiries": (len(surface.expiries)
                                       if hasattr(surface, "expiries") else None),
                    })
                except ArbitrageViolation as exc:
                    outcome.update({"status": "quarantined",
                                    "reason": f"ArbitrageViolation: {exc}"[:300]})
                except ValueError as exc:
                    outcome.update({"status": "failed",
                                    "reason": f"ValueError: {exc}"[:300]})
            else:
                outcome.update({"status": "failed", "reason": "no quotes"})
            pipelines[sym] = outcome
        # Per-underlying completeness from the cycle stats.
        comp = {}
        for sym, s in cstats["underlyings"].items():
            comp[sym] = {
                "n_rows": s["n_rows"],
                "n_valid": s["n_valid"],
                "completeness": (s["n_valid"] / s["n_rows"]) if s["n_rows"] else 0.0,
                "expiries": s["expiries"],
                "error": s["error"],
                "pipeline": pipelines[sym]["status"],
            }
        cycles.append({
            "cycle_id": cstats["cycle_id"],
            "n_rows": cstats["n_rows"],
            "n_valid": cstats["n_valid"],
            "completeness": cstats["completeness"],
            "per_underlying": comp,
            "pipelines": pipelines,
        })
        n_pub = sum(1 for p in pipelines.values()
                      if p["status"] == "published")
        n_q = sum(1 for p in pipelines.values()
                  if p["status"] == "quarantined")
        print(f"[cycle {cstats['cycle_id']}] rows={cstats['n_rows']} "
              f"valid={cstats['n_valid']} "
              f"completeness={cstats['completeness']:.1%} "
              f"published={n_pub}/{len(pipelines)} quarantined={n_q}",
              flush=True)
        for sym, c in comp.items():
            rmse = pipelines[sym].get("rmse_iv")
            print(f"    {sym}: {c['n_valid']}/{c['n_rows']} "
                  f"({c['completeness']:.1%}), expiries={c['expiries']}, "
                  f"pipeline={c['pipeline']}"
                  + (f", rmse_iv={rmse:.4f}" if rmse is not None else "")
                  + (f" ERR: {c['error']}" if c['error'] else ""), flush=True)
        if ci < args.cycles - 1:
            time.sleep(args.poll_interval)

    wall_s = time.perf_counter() - t_run
    all_rows = sum(c["n_rows"] for c in cycles)
    all_valid = sum(c["n_valid"] for c in cycles)
    published = sum(1 for c in cycles for p in c["pipelines"].values()
                    if p["status"] == "published")
    quarantined = sum(1 for c in cycles for p in c["pipelines"].values()
                      if p["status"] == "quarantined")
    n_pipes = sum(len(c["pipelines"]) for c in cycles)
    rmses = [p["rmse_iv"] for c in cycles for p in c["pipelines"].values()
             if p.get("rmse_iv") is not None]
    rmse_by_sym: Dict[str, List[float]] = {}
    for c in cycles:
        for sym, p in c["pipelines"].items():
            if p.get("rmse_iv") is not None:
                rmse_by_sym.setdefault(sym, []).append(p["rmse_iv"])
    summary = {
        "underlyings": underlyings,
        "cycles": args.cycles,
        "poll_interval_s": args.poll_interval,
        "max_expiries": args.max_expiries,
        "min_expiry_days": args.min_expiry_days,
        "k_band": args.k_band,
        "total_rows": all_rows,
        "total_valid": all_valid,
        "overall_completeness": (all_valid / all_rows) if all_rows else 0.0,
        "published": published,
        "quarantined": quarantined,
        "n_pipelines": n_pipes,
        "quarantine_rate": quarantined / n_pipes if n_pipes else 0.0,
        "rmse_iv_values": rmses,
        "rmse_iv_by_underlying": {s: sum(v) / len(v)
                                  for s, v in rmse_by_sym.items()},
        "wall_s": wall_s,
        "is_delayed": True,
        "caveat": ("15-MINUTES DELAYED data; bid/ask completeness measured "
                   "after-hours is a LOWER BOUND — re-run 09:30-16:00 ET "
                   "for the fair measurement."),
    }
    payload = {"summary": summary, "cycles": cycles}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n==== yfinance quality summary ====")
    print(f"underlyings={underlyings} cycles={args.cycles}")
    print(f"overall completeness: {summary['overall_completeness']:.1%} "
          f"({all_valid}/{all_rows})")
    print(f"published={published}/{n_pipes} quarantined={quarantined} "
          f"(quarantine_rate={summary['quarantine_rate']:.0%})")
    for sym, r in summary["rmse_iv_by_underlying"].items():
        print(f"  {sym}: avg calibration rmse_iv={r:.4f}")
    print(f"wall={wall_s:.0f}s  results -> {args.out}")
    print("CAVEAT: " + summary["caveat"])


if __name__ == "__main__":
    main()
