"""Surface calibration tests: recovery of a known SSVI surface."""
import itertools
import math

import numpy as np
import pytest

from snowball_pricer.surface import build_surface, pava_isotonic
from tests.conftest import true_iv


def _collect(feed, sweeps=2):
    it = feed.subscribe()
    return [next(it) for _ in range(feed.sweep_size * sweeps)]


def test_pava_isotonic():
    y = pava_isotonic([0.3, 0.1, 0.2, 0.2, 0.05])
    assert all(b >= a for a, b in zip(y, y[1:]))
    np.testing.assert_allclose(pava_isotonic([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0])


def test_calibration_recovery(feed, truth):
    """Synthetic quotes -> calibrated surface; grid RMSE vs truth < 0.3 vol pts."""
    quotes = _collect(feed, sweeps=2)
    asof = quotes[-1].ts
    surface, diag = build_surface(quotes, asof)

    assert diag["rmse_iv"] < 0.3, f"input-mid RMSE {diag['rmse_iv']}"
    assert list(surface.expiries) == truth["expiries"]
    th = list(surface.thetas)
    assert all(b >= a for a, b in zip(th, th[1:])), "thetas must be non-decreasing"

    # Dense grid vs ground truth.
    errs = []
    for T in truth["expiries"]:
        fwd = truth["spot"] * math.exp((truth["rate"] - truth["div_yield"]) * T)
        for k in np.linspace(-0.4, 0.2, 25):
            K = fwd * math.exp(k)
            errs.append(surface.iv_from_strike(K, T) - true_iv(truth, K, T))
    rmse = float(np.sqrt(np.mean(np.square(errs))))
    assert rmse < 0.3, f"grid RMSE vs truth {rmse:.4f} vol pts"
    assert abs(surface.rho - truth["rho"]) < 0.25
    assert abs(surface.gamma - truth["gamma"]) < 0.3


def test_build_surface_rejects_thin_book(feed):
    quotes = _collect(feed, sweeps=1)[:3]  # far too few
    with pytest.raises(ValueError):
        build_surface(quotes, asof=quotes[-1].ts)


def test_build_surface_ignores_invalid_quotes(feed):
    quotes = _collect(feed, sweeps=2)
    bad = list(quotes)
    # Corrupt a handful into crossed quotes; builder must skip, not crash.
    import dataclasses

    for i in range(0, len(bad), 37):
        q = bad[i]
        bad[i] = dataclasses.replace(q, bid=q.ask + 1.0, ask=q.ask)
    surface, diag = build_surface(bad, asof=bad[-1].ts)
    assert diag["n_used"] < diag["n_quotes"]
    assert diag["rmse_iv"] < 0.3
