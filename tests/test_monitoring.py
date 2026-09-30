"""P5 monitoring tests: alert kinds, thresholds, staleness cooldown."""

import pytest

from snowball_pricer.arbitrage import ArbitrageViolation
from snowball_pricer.monitoring import (
    ARBITRAGE_QUARANTINE,
    GREEKS_DRIFT,
    REBUILD_FAILURE,
    STALENESS,
    Alert,
    Monitor,
    MonitorConfig,
)
from snowball_pricer.pricing.engine import PriceResult


def _snap(version, asof):
    class S:
        pass
    s = S()
    s.version = version
    s.asof = asof
    return s


def _res(delta, vega):
    return PriceResult(price=1.0, std_error=0.0, ko_prob=0.0, ko_prob_se=0.0,
                       delta=dict(delta), vega=vega, n_paths=1, timing={},
                       surface_version=1, tenor_years=2.0)


def test_first_snapshot_sets_baseline_no_alert():
    m = Monitor()
    assert m.on_snapshot(_snap(1, 100.0), _res({"SYN": 0.9}, -0.2)) == []
    assert m.alerts == []


def test_greeks_drift_fires_on_large_delta_move():
    m = Monitor()
    m.on_snapshot(_snap(1, 100.0), _res({"SYN": 0.90}, -0.20))
    fired = m.on_snapshot(_snap(2, 106.0), _res({"SYN": 0.80}, -0.20))
    assert len(fired) == 1
    assert fired[0].kind == GREEKS_DRIFT
    assert fired[0].severity == "warning"
    assert "0.1000" in fired[0].message  # |Δdelta| reported


def test_greeks_drift_fires_on_vega_move():
    m = Monitor()
    m.on_snapshot(_snap(1, 100.0), _res({"SYN": 0.90}, -0.20))
    fired = m.on_snapshot(_snap(2, 106.0), _res({"SYN": 0.90}, -0.17))
    assert len(fired) == 1 and fired[0].kind == GREEKS_DRIFT


def test_no_drift_alert_within_threshold_or_without_greeks():
    m = Monitor()
    m.on_snapshot(_snap(1, 100.0), _res({"SYN": 0.90}, -0.20))
    # small moves: quiet
    assert m.on_snapshot(_snap(2, 106.0), _res({"SYN": 0.901}, -0.201)) == []
    # missing Greeks (compute_greeks=False): delta={} vega=nan -> quiet, no crash
    assert m.on_snapshot(_snap(3, 112.0), _res({}, float("nan"))) == []


def test_quarantine_alert_kinds():
    m = Monitor()
    m.on_snapshot(_snap(3, 100.0), _res({"SYN": 0.9}, -0.2))
    a = m.on_rebuild_failure(ArbitrageViolation({"check": "calendar"}),
                             asof=106.0)
    assert a.kind == ARBITRAGE_QUARANTINE
    assert "v3 kept" in a.message
    b = m.on_rebuild_failure(ValueError("fit failed"), asof=112.0)
    assert b.kind == REBUILD_FAILURE
    assert [x.kind for x in m.alerts] == [ARBITRAGE_QUARANTINE,
                                          REBUILD_FAILURE]


def test_staleness_lifecycle():
    m = Monitor(MonitorConfig(staleness_seconds=30.0,
                              staleness_cooldown_seconds=300.0))
    assert m.check_staleness(1000.0) is None  # nothing published yet
    m.on_snapshot(_snap(1, 100.0), _res({"SYN": 0.9}, -0.2))
    assert m.check_staleness(129.9) is None
    a = m.check_staleness(130.1)
    assert a is not None and a.kind == STALENESS
    assert a.severity == "critical"
    # cooldown suppresses the repeat
    assert m.check_staleness(200.0) is None
    # ...but it fires again after the cooldown
    b = m.check_staleness(500.0)
    assert b is not None and b.kind == STALENESS


def test_custom_thresholds():
    m = Monitor(MonitorConfig(delta_drift_threshold=0.5))
    m.on_snapshot(_snap(1, 100.0), _res({"SYN": 0.90}, -0.20))
    assert m.on_snapshot(_snap(2, 106.0), _res({"SYN": 0.80}, -0.20)) == []
