"""Arbitrage gate tests: pass on good surfaces, reject broken ones."""
import math

import numpy as np

from snowball_pricer.arbitrage import (
    ArbitrageViolation,
    check_butterfly_smile,
    check_calendar_surface,
    check_wings,
    validate_surface,
)
from snowball_pricer.surface import VolSurface, build_surface, ssvi_total_var


def _calibrated_surface(feed):
    it = feed.subscribe()
    quotes = [next(it) for _ in range(feed.sweep_size * 2)]
    surface, _ = build_surface(quotes, quotes[-1].ts)
    return surface


def test_gate_passes_on_calibrated_surface(feed):
    surface = _calibrated_surface(feed)
    report = validate_surface(surface)
    assert report["all_ok"], f"gate failed on good surface: {report}"
    assert report["calendar"][0] and report["butterfly"][0] and report["wings"][0]


def test_rejects_calendar_arbitrage():
    """Decreasing ATM total variance in T -> calendar check fails."""
    surface = VolSurface(
        expiries=(0.5, 1.0),
        thetas=(0.05, 0.03),  # decreasing: calendar arbitrage
        rho=-0.4,
        eta=0.8,
        gamma=0.5,
        forwards=(100.0, 100.0),
        spot=100.0,
        rate=0.03,
        div_yield=0.01,
    )
    ok, worst = check_calendar_surface(
        surface, np.linspace(-1.0, 0.5, 21), np.linspace(0.5, 1.0, 11)
    )
    assert not ok
    assert worst < 0.0


def test_rejects_butterfly_arbitrage():
    """A sharp spike in total variance implies negative density -> g < 0."""
    w_spike = lambda k: 0.04 * (1.0 + 8.0 * math.exp(-((k) / 0.02) ** 2))  # noqa: E731
    ok, worst = check_butterfly_smile(w_spike, np.linspace(-0.5, 0.5, 101))
    assert not ok, "spiked smile must violate butterfly"
    assert worst < 0.0

    # ...while a genuine SSVI smile passes.
    w_ssvi = lambda k: float(ssvi_total_var(k, 0.04, -0.4, 0.8, 0.5))  # noqa: E731
    ok2, worst2 = check_butterfly_smile(w_ssvi, np.linspace(-1.5, 1.0, 61))
    assert ok2, f"SSVI smile should pass, worst g={worst2}"


def test_rejects_wing_violation():
    """Duck-typed surface with asymptotic slope 3 > Lee bound 2."""

    class SteepWings:
        expiries = (1.0,)

        def total_var(self, k, T):
            return 3.0 * abs(k) + 0.04

    ok, worst = check_wings(SteepWings(), (1.0,))
    assert not ok
    assert worst > 0.0


def test_builder_quarantines_failing_surface(feed, monkeypatch):
    """If the gate fails, build_surface raises instead of returning."""
    import snowball_pricer.surface as surf_mod

    real_validate = surf_mod.validate_surface

    def bad_validate(surface, **kw):
        rep = real_validate(surface, **kw)
        rep = dict(rep)
        rep["butterfly"] = (False, -1.0)
        rep["all_ok"] = False
        return rep

    monkeypatch.setattr(surf_mod, "validate_surface", bad_validate)
    it = feed.subscribe()
    quotes = [next(it) for _ in range(feed.sweep_size)]
    try:
        surf_mod.build_surface(quotes, quotes[-1].ts)
    except ArbitrageViolation as e:
        assert "butterfly" in str(e)
    else:
        raise AssertionError("expected ArbitrageViolation")
