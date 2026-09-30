"""Real-time monitoring: alerts on snapshot publish, rebuild failure, staleness (P5).

The monitor sits beside the pricing loop (``scripts/live_loop.py``):

- ``on_snapshot(snapshot, price_result)`` — called on every published
  surface; emits ``GREEKS_DRIFT`` when |Δdelta| or |Δvega| between
  consecutive reprices exceeds the configured threshold.
- ``on_rebuild_failure(exc, asof=...)`` — called when a surface rebuild is
  rejected/quarantined; emits ``ARBITRAGE_QUARANTINE`` (gate rejection,
  last-good snapshot kept) or ``REBUILD_FAILURE`` (any other rebuild
  error, e.g. fit failure — also quarantined, last-good kept).
- ``check_staleness(now_ts)`` — emits ``STALENESS`` when no snapshot has
  been published within ``staleness_seconds`` of market time.

All alerts are recorded in ``monitor.alerts`` in firing order. Thresholds
are configurable via ``MonitorConfig``; defaults are documented below and
in SPEC.md §8.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .arbitrage import ArbitrageViolation

# Alert kinds (SPEC.md §8.1)
ARBITRAGE_QUARANTINE = "ARBITRAGE_QUARANTINE"
REBUILD_FAILURE = "REBUILD_FAILURE"
GREEKS_DRIFT = "GREEKS_DRIFT"
STALENESS = "STALENESS"


@dataclass(frozen=True)
class Alert:
    """One monitoring event."""

    ts: float       # event time: market/exchange time (epoch seconds) when known
    kind: str       # one of the kinds above
    severity: str   # "info" | "warning" | "critical"
    message: str    # human-readable, includes the numbers that tripped it


@dataclass
class MonitorConfig:
    """Thresholds. Defaults (documented, SPEC.md §8.2):

    - ``delta_drift_threshold = 0.05``: |Δdelta| (per underlying, in
      notional units — delta ~1.0 for this product) between consecutive
      reprices that counts as drift.
    - ``vega_drift_threshold = 0.02``: |Δvega| per 1 vol-point parallel bump
      (vega ~ -0.2 for the reference snowball; 0.02 ≈ 10% of typical vega).
    - ``staleness_seconds = 30.0``: a published snapshot older than this vs
      market time is stale. The P1 rebuild cadence is daily in the replay
      harness but seconds-scale against a live broker feed; 30 s matches the
      "no tick within N seconds" convention for listed options.
    - ``staleness_cooldown_seconds = 300.0``: minimum gap between repeat
      STALENESS alerts so a dead feed pages once, not per tick.
    """

    delta_drift_threshold: float = 0.05
    vega_drift_threshold: float = 0.02
    staleness_seconds: float = 30.0
    staleness_cooldown_seconds: float = 300.0


class Monitor:
    """Stateful alert sink for the live pricing loop."""

    def __init__(self, config: Optional[MonitorConfig] = None):
        self.config = config or MonitorConfig()
        self._alerts: List[Alert] = []
        self._last_delta: Optional[Dict[str, float]] = None
        self._last_vega: Optional[float] = None
        self._last_version: Optional[int] = None
        self._last_snapshot_ts: Optional[float] = None
        self._last_staleness_ts: Optional[float] = None

    # -- accessors ------------------------------------------------------
    @property
    def alerts(self) -> List[Alert]:
        return list(self._alerts)

    def clear(self) -> None:
        self._alerts.clear()

    # -- hooks ----------------------------------------------------------
    def on_snapshot(self, snapshot, price_result) -> List[Alert]:
        """Consume a published snapshot + its price. Returns alerts fired."""
        fired: List[Alert] = []
        cfg = self.config
        cur_delta = dict(price_result.delta or {})
        cur_vega = float(price_result.vega)

        if self._last_delta is not None:
            # Drift is only meaningful between two reprices that both HAVE
            # Greeks (the loop prices Greeks on a slower cadence; a
            # Greeks->no-Greeks transition is expected, not drift).
            prev_has = bool(self._last_delta) or self._is_finite(
                self._last_vega)
            cur_has = bool(cur_delta) or self._is_finite(cur_vega)
            dmax, dvega = 0.0, 0.0
            if prev_has and cur_has:
                names = set(self._last_delta) | set(cur_delta)
                for n in names:
                    dmax = max(dmax, abs(cur_delta.get(n, 0.0)
                                        - self._last_delta.get(n, 0.0)))
                dvega = (abs(cur_vega - self._last_vega)
                         if self._is_finite(cur_vega)
                         and self._is_finite(self._last_vega) else 0.0)
            if (dmax > cfg.delta_drift_threshold
                    or dvega > cfg.vega_drift_threshold):
                fired.append(Alert(
                    ts=float(snapshot.asof),
                    kind=GREEKS_DRIFT,
                    severity="warning",
                    message=(
                        f"Greeks drift vs snapshot v{self._last_version}: "
                        f"|Δdelta|max={dmax:.4f} "
                        f"(thr {cfg.delta_drift_threshold}), "
                        f"|Δvega|={dvega:.4f} "
                        f"(thr {cfg.vega_drift_threshold}); "
                        f"now v{snapshot.version}"
                    ),
                ))

        self._last_delta = cur_delta
        self._last_vega = cur_vega
        self._last_version = int(snapshot.version)
        self._last_snapshot_ts = float(snapshot.asof)
        self._alerts.extend(fired)
        return fired

    def on_rebuild_failure(self, exc: BaseException,
                           asof: Optional[float] = None) -> Alert:
        """Record a quarantined rebuild. Returns the alert fired."""
        if isinstance(exc, ArbitrageViolation):
            kind, severity = ARBITRAGE_QUARANTINE, "warning"
            detail = f"arbitrage gate rejected the rebuild: {exc}"
        else:
            kind, severity = REBUILD_FAILURE, "warning"
            detail = (f"rebuild failed with {type(exc).__name__}: {exc}")
        alert = Alert(
            ts=float(asof) if asof is not None else time.time(),
            kind=kind,
            severity=severity,
            message=(f"{detail}; last-good snapshot "
                     f"v{self._last_version} kept live"),
        )
        self._alerts.append(alert)
        return alert

    def check_staleness(self, now_ts: float) -> Optional[Alert]:
        """Fire STALENESS if the live snapshot is older than the threshold.

        Returns the alert, or None. Cool-down applies to repeat firings.
        """
        if self._last_snapshot_ts is None:
            return None  # nothing published yet: not stale, just quiet
        cfg = self.config
        age = float(now_ts) - self._last_snapshot_ts
        if age <= cfg.staleness_seconds:
            return None
        if (self._last_staleness_ts is not None
                and float(now_ts) - self._last_staleness_ts
                < cfg.staleness_cooldown_seconds):
            return None
        alert = Alert(
            ts=float(now_ts),
            kind=STALENESS,
            severity="critical",
            message=(f"no published snapshot within {cfg.staleness_seconds:.0f}s "
                     f"of market time (last v{self._last_version} asof="
                     f"{self._last_snapshot_ts:.1f}, age={age:.1f}s)"),
        )
        self._alerts.append(alert)
        self._last_staleness_ts = float(now_ts)
        return alert

    @staticmethod
    def _is_finite(x) -> bool:
        try:
            return math.isfinite(float(x))
        except (TypeError, ValueError):
            return False
