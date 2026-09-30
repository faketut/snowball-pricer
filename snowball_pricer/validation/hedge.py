"""Delta-hedge simulator + P&L explain for the replay validation (P4).

``simulate_hedge`` delta-hedges a SHORT snowball position along the TRUE
spot path, rebalancing daily to the model deltas from the replay rows.
``pnl_explain`` decomposes the realized P&L into delta / vega / carry
buckets plus a residual — the residual is the "is the model honest" number.

Sign convention (documented once, used everywhere):
- Replay rows carry the LONG product's delta (dV/dS > 0 in the continuation
  region) and vega (per +1 vol point, < 0 for the snowball).
- The hedged position is SHORT one unit of notional: position delta is
  ``-delta_long``. The delta-neutral hedge therefore holds
  ``shares = +delta_long * notional`` of the underlying (buy stock against
  the short). I.e. ``hedge = -delta_short * notional``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from ..pricing.engine import EngineConfig, price
from ..pricing.payoff import TermSheet
from .replay import ReplayResult

DAYS_PER_YEAR = 252.0


@dataclass
class HedgeResult:
    """Result of ``simulate_hedge``. All P&L fields are per unit notional."""

    realized_pnl: float      # all-in realized P&L per notional
    premium: float           # received at t0 per notional
    final_value: float       # buyback mark / redemption / maturity payoff
    stock_pnl: float         # hedge stock leg: sum h*dS (per notional)
    interest_pnl: float      # cash accrual at `rate`
    n_days: int              # true-path days the hedge was held
    notional: float
    terminated: Optional[Dict]  # None, or {"reason", "day", ...}
    ledger: List[Dict]       # one entry per true-path day
    names: List[str]


def _as_paths(true_spots, names: List[str]) -> Dict[str, np.ndarray]:
    """true_spots: 1-D array-like (single underlying) or {name: array}."""
    if isinstance(true_spots, dict):
        paths = {n: np.asarray(true_spots[n], dtype=float) for n in names}
    else:
        if len(names) != 1:
            raise ValueError(
                "multi-asset hedge needs true_spots as {name: path} dict")
        paths = {names[0]: np.asarray(true_spots, dtype=float)}
    n = {len(p) for p in paths.values()}
    if len(n) != 1:
        raise ValueError("all true spot paths must have the same length")
    return paths


def simulate_hedge(
    rows: Sequence[Dict],
    true_spots: Union[Sequence[float], Dict[str, Sequence[float]]],
    terms: TermSheet,
    rate: float,
    *,
    notional: float = 1.0,
    weights: Optional[Dict[str, float]] = None,
) -> HedgeResult:
    """Delta-hedge a SHORT snowball along the true spot path.

    - t0 (close of day 0): receive ``premium = rows[0].price``, buy
      ``+delta_long * notional`` shares.
    - Each subsequent true-path day: accrue cash one day at ``rate``; check
      KO observations (every ``ko_obs_every_days`` days, basket vs
      ``ko_barrier`` on the true path); otherwise rebalance to the latest
      available row delta ("hold last delta between reprices").
    - End of the true path (no KO): flatten the hedge, buy back the short
      at the last model price. Maturity economics (KI on the true path) are
      implemented for completeness; the canonical replay is too short to
      reach them.
    """
    rows = sorted(rows, key=lambda r: r["day"])
    names = list(rows[0]["delta"].keys())
    for r in rows:
        if list(r["delta"].keys()) != names:
            raise ValueError("replay rows disagree on underlying names")
    if weights is None:
        weights = {n: (1.0 if len(names) == 1 else 1.0 / len(names))
                   for n in names}
    w = np.array([weights[n] for n in names])
    w = w / w.sum()

    paths = _as_paths(true_spots, names)
    n_days = len(next(iter(paths.values())))
    if n_days < 2:
        raise ValueError("need at least 2 true-path days to hedge")

    row_by_day = {r["day"]: r for r in rows}
    last_row = rows[0]

    def delta_at(day: int) -> Dict[str, float]:
        nonlocal last_row
        if day in row_by_day:
            last_row = row_by_day[day]
        return dict(last_row["delta"])

    def spot(name: str, day: int) -> float:
        return float(paths[name][day])

    def basket(day: int) -> float:
        return float(sum(w[i] * spot(nm, day) / spot(nm, 0)
                         for i, nm in enumerate(names)))

    premium = float(rows[0]["price"])
    # SHORT position: hedge = -position_delta * notional = +delta_long * notional
    shares = {n: float(delta_at(0)[n]) * notional for n in names}
    cash = premium * notional - sum(shares[n] * spot(n, 0) for n in names)
    stock_pnl = 0.0
    interest_pnl = 0.0
    ledger: List[Dict] = [{
        "day": 0, "action": "open",
        "spot": {n: spot(n, 0) for n in names},
        "delta_long": {n: float(delta_at(0)[n]) for n in names},
        "shares": dict(shares), "cash": cash,
        "stock_pnl_day": 0.0, "interest_day": 0.0,
    }]
    terminated: Optional[Dict] = None
    final_value = float(rows[-1]["price"])
    last_held_day = n_days - 1

    for day in range(1, n_days):
        # 1. cash accrual for one day
        growth = cash * (math.exp(rate / DAYS_PER_YEAR) - 1.0)
        interest_pnl += growth / notional
        cash *= math.exp(rate / DAYS_PER_YEAR)
        S = {n: spot(n, day) for n in names}
        S_prev = {n: spot(n, day - 1) for n in names}

        # 2. stock P&L on shares held over [day-1, day]
        sp_day = sum(shares[n] * (S[n] - S_prev[n]) for n in names) / notional
        stock_pnl += sp_day

        # 3. KO observation dates (absolute trading days from t0)
        if (day + 1) % terms.ko_obs_every_days == 0:
            if basket(day) >= terms.ko_barrier:
                t_ko = (day + 1) / DAYS_PER_YEAR
                redemption = notional * (1.0 + terms.coupon_annual * t_ko)
                cash += sum(shares[n] * S[n] for n in names) - redemption
                final_value = redemption / notional
                terminated = {"reason": "autocall", "day": day,
                              "t_ko_years": t_ko, "basket": basket(day)}
                last_held_day = day
                ledger.append({
                    "day": day, "action": "autocall",
                    "spot": S, "delta_long": delta_at(day),
                    "shares": {n: 0.0 for n in names}, "cash": cash,
                    "stock_pnl_day": sp_day, "interest_day": growth / notional,
                    "redemption": redemption / notional,
                })
                shares = {n: 0.0 for n in names}
                break

        # 4. maturity (original contract tenor reached)
        maturity_day = int(round(terms.tenor_years * DAYS_PER_YEAR))
        if day >= maturity_day:
            ki_hit = any(basket(d) <= terms.ki_barrier
                         for d in range(maturity_day + 1)) if terms.ki_barrier > 0 else False
            bT = basket(day)
            if ki_hit:
                payoff = notional * min(1.0, bT)
            else:
                payoff = notional * (1.0 + terms.coupon_annual * terms.tenor_years)
            cash += sum(shares[n] * S[n] for n in names) - payoff
            final_value = payoff / notional
            terminated = {"reason": "maturity", "day": day,
                          "knocked_in": ki_hit, "basket_T": bT}
            last_held_day = day
            ledger.append({
                "day": day, "action": "maturity",
                "spot": S, "delta_long": delta_at(day),
                "shares": {n: 0.0 for n in names}, "cash": cash,
                "stock_pnl_day": sp_day, "interest_day": growth / notional,
                "payoff": payoff / notional,
            })
            shares = {n: 0.0 for n in names}
            break

        # 5. rebalance to the latest model delta
        d = delta_at(day)
        new_shares = {n: float(d[n]) * notional for n in names}
        trade_cash = sum((new_shares[n] - shares[n]) * S[n] for n in names)
        cash -= trade_cash
        shares = new_shares
        ledger.append({
            "day": day, "action": "rebalance",
            "spot": S, "delta_long": {n: float(d[n]) for n in names},
            "shares": dict(shares), "cash": cash,
            "stock_pnl_day": sp_day, "interest_day": growth / notional,
        })

    else:
        # No KO / maturity: flatten hedge, buy back the short at model.
        S = {n: spot(n, n_days - 1) for n in names}
        cash += sum(shares[n] * S[n] for n in names)
        cash -= final_value * notional
        ledger.append({
            "day": n_days - 1, "action": "closeout",
            "spot": S, "delta_long": delta_at(n_days - 1),
            "shares": {n: 0.0 for n in names}, "cash": cash,
            "stock_pnl_day": 0.0, "interest_day": 0.0,
            "buyback": final_value,
        })

    realized = cash / notional
    # Accounting identity: premium - final_value + stock + interest == realized.
    identity = premium - final_value + stock_pnl + interest_pnl
    if abs(identity - realized) > 1e-9:
        raise AssertionError(
            f"hedge accounting broken: {identity} != {realized}")
    return HedgeResult(
        realized_pnl=realized, premium=premium, final_value=final_value,
        stock_pnl=stock_pnl, interest_pnl=interest_pnl,
        n_days=last_held_day + 1, notional=notional,
        terminated=terminated, ledger=ledger, names=names,
    )


def pnl_explain(
    replay: ReplayResult,
    hedge: HedgeResult,
    *,
    theta_cfg: Optional[EngineConfig] = None,
) -> Dict:
    """Decompose the realized (short) P&L into delta / vega / carry buckets.

    Buckets are SHORT-perspective, in bps of notional, over the held
    transitions d -> d+1 (d = 0..n_held-2); any terminal settlement gap
    (autocall redemption / maturity payoff vs the model mark) lands in the
    residual:
    - delta:   sum_d (h_d - delta_long_d)*dS_d  (NET delta: hedge leg minus
               the position's model delta leg; ~0 when the daily hedge
               tracks the model delta, nonzero from discrete/stale hedging)
    - vega:    sum_d (-vega_long_d) * d(ATMiv)_d  (ATM iv from each
               snapshot's surface at a fixed tenor; vega is the engine's
               parallel +1pt surface-bump vega, so this is an ATM-only
               approximation — named as such)
    - carry:   model theta (frozen spot+surface price change as tenor
               shrinks one day, repriced per snapshot, SHORT sign) + hedge
               cash interest
    - residual = actual - (delta + vega + carry), by construction. It holds
      everything the first-order attribution misses: gamma (no gamma hedge
      in the engine), the vega ATM-only approximation, QMC noise in
      Greeks, surface recalibration noise, theta finite-difference noise,
      and any terminal settlement gap.

    ``theta_cfg`` controls the frozen-market reprices (default: 20k QMC
    paths, no Greeks, fixed seed).
    """
    import time

    if theta_cfg is None:
        theta_cfg = EngineConfig(n_paths=20_000, qmc=True, seed=777001,
                                 compute_greeks=False, batch_size=20_000)

    rows = sorted(replay.rows, key=lambda r: r["day"])
    row_by_day = {r["day"]: r for r in rows}
    snap_by_version = {s.version: s for s in replay.snapshots}
    n_held = hedge.n_days  # true-path days 0..n_held-1 were held
    notional = hedge.notional
    names = hedge.names
    terms = replay.terms

    def row_at(day: int) -> Dict:
        # latest reprice at or before `day` (hold-last between reprices)
        cands = [d for d in row_by_day if d <= day]
        return row_by_day[max(cands)]

    t0 = time.perf_counter()
    # Shares held over each true day + spots, from the hedge ledger.
    shares_by_day = {e["day"]: e["shares"] for e in hedge.ledger
                     if e["action"] in ("open", "rebalance")}
    spot_by_day = {e["day"]: e["spot"] for e in hedge.ledger}

    # --- delta bucket: NET (hedge leg - position model-delta leg) ---
    delta_bps = 0.0
    vega_bps = 0.0
    for day in range(n_held - 1):
        r, r_next = row_at(day), row_at(day + 1)
        h = shares_by_day[day]
        s0, s1 = spot_by_day[day], spot_by_day[day + 1]
        for name in names:
            dS = s1[name] - s0[name]
            delta_bps += (h[name] / notional - r["delta"][name]) * dS * 1e4
        # --- vega bucket: short's vega P&L; vega per vol POINT ---
        d_iv_points = (r_next["atm_iv"] - r["atm_iv"]) / 0.01
        vega_bps += -r["vega"] * d_iv_points * 1e4

    # --- carry bucket: frozen-market theta per transition-day + interest ---
    theta_bps = 0.0
    n_theta = 0
    norm = replay.norm_spots
    held_days = sorted(d for d in row_by_day if d < n_held - 1)
    for day in held_days:
        r = row_by_day[day]
        snap = snap_by_version[r["surface_version"]]
        tenor_short = r["tenor_years"] - 1.0 / DAYS_PER_YEAR
        if tenor_short <= 0:
            continue
        specs = [replace(u, spot=r["spot"]) for u in replay.underlyings]
        terms_short = replace(terms, tenor_years=tenor_short)
        terms_full = replace(terms, tenor_years=r["tenor_years"])
        p_short = price(snap, terms_short, list(specs), cfg=theta_cfg,
                        norm_spots=norm).price
        p_full = price(snap, terms_full, list(specs), cfg=theta_cfg,
                       norm_spots=norm).price
        theta_long = p_short - p_full      # long P&L from one day of aging
        theta_bps += -theta_long * 1e4     # SHORT sign
        n_theta += 1
    theta_s = time.perf_counter() - t0
    interest_bps = hedge.interest_pnl * 1e4
    carry_bps = theta_bps + interest_bps

    actual_bps = hedge.realized_pnl * 1e4
    residual_bps = actual_bps - (delta_bps + vega_bps + carry_bps)

    # Per-day attribution table (for the report).
    daily = []
    for day in range(n_held - 1):
        r, r_next = row_at(day), row_at(day + 1)
        h = shares_by_day[day]
        s0, s1 = spot_by_day[day], spot_by_day[day + 1]
        net_d = sum((h[n] / notional - r["delta"][n]) * (s1[n] - s0[n])
                    for n in names)
        daily.append({
            "day": day,
            "spot": r["spot"],
            "dS": (r_next["spot"] - r["spot"]
                   if (day + 1) in row_by_day else None),
            "atm_iv": r["atm_iv"],
            "d_atm_iv": r_next["atm_iv"] - r["atm_iv"],
            "price": r["price"],
            "delta": r["delta"],
            "vega": r["vega"],
            "net_delta_bps": net_d * 1e4,
            "vega_day_bps": -r["vega"] * ((r_next["atm_iv"] - r["atm_iv"])
                                          / 0.01) * 1e4,
        })

    return {
        "buckets_bps": {
            "delta": delta_bps,
            "vega": vega_bps,
            "carry_theta_model": theta_bps,
            "carry_interest": interest_bps,
            "carry_total": carry_bps,
            "residual": residual_bps,
            "actual": actual_bps,
        },
        "n_theta_repricings": n_theta,
        "theta_seconds": theta_s,
        "n_held_days": n_held,
        "terminated": hedge.terminated,
        "daily": daily,
    }
