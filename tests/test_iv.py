"""IV inverter tests: round-trip accuracy, parity forward, edge quotes."""
import itertools
import math

import pytest

from snowball_pricer.iv import (
    DegenerateQuoteError,
    black_price,
    black_vega,
    forward_from_parity,
    implied_vol,
    implied_vol_safe,
)


def test_round_trip_grid():
    """Black price -> implied_vol recovers the input vol to 1e-8,
    restricted to well-conditioned cases (vega large enough that a 1e-8
    vol move is visible at the 1e-10 price tolerance)."""
    worst = 0.0
    n = 0
    for F, K, T, vol, is_call in itertools.product(
        [100.0],
        [60.0, 80.0, 100.0, 120.0, 140.0, 200.0],
        [0.05, 0.25, 1.0, 2.0],
        [0.05, 0.2, 0.5, 1.5, 3.0, 5.0],
        [True, False],
    ):
        df = math.exp(-0.03 * T)
        price = black_price(F, K, T, vol, df, is_call)
        if price <= 1e-14:
            continue  # below inversion resolution; not a realistic quote
        if black_vega(F, K, T, vol, df) < 0.05:
            continue  # deep-ITM/short-dated: vol numerically unidentifiable
        rec = implied_vol(price, F, K, T, df, is_call)
        worst = max(worst, abs(rec - vol))
        n += 1
    assert n > 200, "grid too aggressively filtered"
    assert worst < 1e-8, f"worst round-trip error {worst}"


def test_newton_and_bisection_paths():
    """Extreme vols exercise both the Newton path and the bisection fallback."""
    F, K, T, df = 100.0, 100.0, 1.0, math.exp(-0.03)
    for vol in (0.02, 0.3, 4.0, 8.0):
        price = black_price(F, K, T, vol, df, True)
        assert abs(implied_vol(price, F, K, T, df, True) - vol) < 1e-8


def test_forward_from_parity():
    F_true, K, T = 105.0, 100.0, 0.5
    df = math.exp(-0.03 * T)
    vol = 0.25
    c = black_price(F_true, K, T, vol, df, True)
    p = black_price(F_true, K, T, vol, df, False)
    assert abs(forward_from_parity(c, p, K, df) - F_true) < 1e-10


def test_degenerate_quotes():
    F, K, T, df = 100.0, 100.0, 1.0, math.exp(-0.03)
    intrinsic_call = df * max(F - K, 0.0)

    with pytest.raises(DegenerateQuoteError):
        implied_vol(0.0, F, K, T, df, True)  # zero price
    with pytest.raises(DegenerateQuoteError):
        implied_vol(-1.0, F, K, T, df, True)  # negative price
    with pytest.raises(DegenerateQuoteError):
        implied_vol(df * F * 1.5, F, K, T, df, True)  # above no-arb cap
    with pytest.raises(DegenerateQuoteError):
        implied_vol(intrinsic_call - 0.5, 110.0, 100.0, T, df, True)  # below intrinsic

    # Price == intrinsic (ITM, positive intrinsic) -> vol 0, not an error.
    F_itm, K_itm = 110.0, 100.0
    intrinsic_itm = df * (F_itm - K_itm)
    assert implied_vol(intrinsic_itm, F_itm, K_itm, T, df, True) == 0.0
    # Zero price with zero intrinsic is degenerate (no information to invert).
    with pytest.raises(DegenerateQuoteError):
        implied_vol(0.0, F, K, T, df, True)

    # Non-raising variant returns NaN instead.
    assert math.isnan(implied_vol_safe(0.0, F, K, T, df, True))
    assert math.isnan(implied_vol_safe(-1.0, F, K, T, df, False))


def test_put_call_parity_consistent_ivs():
    """Same (F, K, T): call and put mids imply the same vol."""
    F, K, T = 100.0, 110.0, 0.75
    df = math.exp(-0.03 * T)
    vol = 0.3
    c = black_price(F, K, T, vol, df, True)
    p = black_price(F, K, T, vol, df, False)
    iv_c = implied_vol(c, F, K, T, df, True)
    iv_p = implied_vol(p, F, K, T, df, False)
    assert abs(iv_c - iv_p) < 1e-9
