"""P4 validation tests: regime replay, hedge accounting, determinism."""
import math

import pytest

from snowball_pricer.pricing.engine import EngineConfig
from snowball_pricer.pricing.payoff import TermSheet, UnderlyingSpec
from snowball_pricer.validation import (
    RegimeTickFeed,
    pnl_explain,
    run_replay,
    simulate_hedge,
)

TRUTH = dict(
    thetas=[0.0121, 0.0200, 0.0361, 0.0648],
    expiries=[0.25, 0.5, 1.0, 2.0],
    rho=-0.45,
    eta=0.8,
    gamma=0.5,
)
TERMS = TermSheet()  # 2Y / KO 100% monthly / KI 80% daily / 15% coupon
UNDERLYINGS = [UnderlyingSpec(name="SYN", weight=1.0)]
# Small engine config keeps the validation tests fast; the canonical
# 20k-path config is exercised by scripts/p4_replay.py.
TEST_CFG = EngineConfig(n_paths=2000, qmc=True, seed=12345,
                        compute_greeks=True, batch_size=2000)
THETA_CFG = EngineConfig(n_paths=2000, qmc=True, seed=999,
                         compute_greeks=False, batch_size=2000)


def _tiny_feed(seed=7):
    sweep = len(TRUTH["expiries"]) * 15 * 2
    return RegimeTickFeed(
        [(2 * sweep, 0.20, 0.20), (2 * sweep, 0.45, 0.45)],
        seed=seed, spot0=100.0, rate=0.03, div_yield=0.01, **TRUTH)


def _tiny_replay(seed=7):
    # Seasoned like the canonical script: barriers fixed at day-0 spot.
    return run_replay(_tiny_feed(seed), TERMS, TEST_CFG, UNDERLYINGS,
                      norm_spots=[100.0])


def test_replay_completes_rows_monotonic_finite():
    replay = _tiny_replay()
    rows = replay.rows
    assert len(rows) == 4  # 2 regimes x 2 days, one snapshot/day
    ts = [r["ts"] for r in rows]
    assert all(b > a for a, b in zip(ts, ts[1:])), "rows not monotonic in ts"
    for r in rows:
        assert math.isfinite(r["price"]) and r["price"] > 0
        assert math.isfinite(r["std_error"])
        assert all(math.isfinite(v) for v in r["delta"].values())
        assert math.isfinite(r["vega"])
        assert 0.0 <= r["ko_prob"] <= 1.0
        assert math.isfinite(r["atm_iv"]) and r["atm_iv"] > 0
    # the vol-spike regime must show up in the calibrated surface
    assert rows[2]["atm_iv"] > rows[0]["atm_iv"] * 1.5
    # true path handed to the hedger matches the feed
    assert len(replay.true_spots) == 4


def test_hedge_accounting_identity_exact():
    replay = _tiny_replay()
    hedge = simulate_hedge(replay.rows, replay.true_spots, TERMS, 0.03)
    # internal identity: premium - final + stock + interest == realized
    lhs = hedge.premium - hedge.final_value + hedge.stock_pnl + hedge.interest_pnl
    assert abs(lhs - hedge.realized_pnl) < 1e-12
    explain = pnl_explain(replay, hedge, theta_cfg=THETA_CFG)
    b = explain["buckets_bps"]
    # attribution identity: buckets sum to actual by construction
    total = b["delta"] + b["vega"] + b["carry_total"] + b["residual"]
    assert abs(total - b["actual"]) < 1e-9
    assert math.isfinite(b["residual"])


def test_determinism_same_seed_identical_pnl():
    r1 = _tiny_replay(seed=99)
    r2 = _tiny_replay(seed=99)
    assert [x["price"] for x in r1.rows] == [x["price"] for x in r2.rows]
    h1 = simulate_hedge(r1.rows, r1.true_spots, TERMS, 0.03)
    h2 = simulate_hedge(r2.rows, r2.true_spots, TERMS, 0.03)
    e1 = pnl_explain(r1, h1, theta_cfg=THETA_CFG)
    e2 = pnl_explain(r2, h2, theta_cfg=THETA_CFG)
    assert e1["buckets_bps"] == e2["buckets_bps"]
    assert e1["daily"] == e2["daily"]


def _craft_rows(n_days, price=0.99, delta=0.5, vega=-0.002):
    rows = []
    for d in range(n_days):
        rows.append({
            "day": d, "ts": 1_700_000_000.0 + d * 86400.0,
            "spot": 100.0, "tenor_years": 2.0 - d / 252.0,
            "surface_version": d + 1, "price": price,
            "std_error": 1e-4, "delta": {"SYN": delta}, "vega": vega,
            "ko_prob": 0.5, "atm_iv": 0.20,
        })
    return rows


def test_autocall_terminates_and_pays_redemption():
    rows = _craft_rows(22)
    spots = [100.0] * 21 + [101.0]  # KO obs at day 20: (20+1) % 21 == 0
    hedge = simulate_hedge(rows, spots, TERMS, 0.03)
    assert hedge.terminated["reason"] == "autocall"
    assert hedge.terminated["day"] == 20
    # redemption = 1 + 15% * 21/252, hedge flattened
    assert hedge.final_value == pytest.approx(1.0 + 0.15 * 21 / 252.0)
    assert all(v == 0.0 for v in hedge.ledger[-1]["shares"].values())
    lhs = hedge.premium - hedge.final_value + hedge.stock_pnl + hedge.interest_pnl
    assert abs(lhs - hedge.realized_pnl) < 1e-12


def test_maturity_pays_full_coupon_when_never_knocked():
    n = 505
    rows = _craft_rows(n)
    spots = [100.0 * (1.0 - 0.0001 * d) for d in range(n)]  # drifts down, no KO
    hedge = simulate_hedge(rows, spots, TERMS, 0.03)
    assert hedge.terminated["reason"] == "maturity"
    assert hedge.terminated["knocked_in"] is False
    assert hedge.final_value == pytest.approx(1.0 + 0.15 * 2.0)
    lhs = hedge.premium - hedge.final_value + hedge.stock_pnl + hedge.interest_pnl
    assert abs(lhs - hedge.realized_pnl) < 1e-12
