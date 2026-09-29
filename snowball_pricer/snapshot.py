"""Versioned, atomic-swap snapshot store (P1).

Lock-free publish contract (this is the contract a future Rust port must
keep — see SPEC.md §2):
- Single publisher. ``publish()`` swaps one reference; readers calling
  ``latest()`` always observe a complete, versioned, immutable snapshot —
  never a torn / half-written surface.
- In CPython the swap is a single attribute assignment (GIL-atomic). The
  Rust port must use an equivalent primitive (e.g. ``arc_swap`` / seqlock):
  readers never block, never copy the surface, and always see a consistent
  (version, surface) pair.
- Versions are monotonic per store. ``publish()`` assigns the next version
  itself; ``publish_snapshot()`` (for externally-versioned snapshots)
  rejects any version <= the current one with ``StaleSnapshotError``.
- Snapshots are immutable (frozen dataclasses all the way down); the store
  additionally keeps a bounded diagnostic history ring (not part of the
  published contract — for ops/debugging only).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional

from .surface import VolSurface


@dataclass(frozen=True)
class VolSurfaceSnapshot:
    """One published surface version. Immutable."""

    surface: VolSurface
    version: int        # monotonic within the store, starts at 1
    asof: float         # exchange timestamp the surface was built from
    source: str         # e.g. "synthetic", "broker:<venue>"
    n_quotes: int       # quotes consumed by this calibration
    rmse_iv: float      # calibration RMSE vs input mids, vol points


class StaleSnapshotError(Exception):
    """An externally-versioned snapshot was not newer than current."""


class SnapshotStore:
    """Single-publisher atomic-swap store of VolSurfaceSnapshot."""

    def __init__(self, history_len: int = 32) -> None:
        self._current: Optional[VolSurfaceSnapshot] = None
        self._next_version: int = 1
        self._history: Deque[VolSurfaceSnapshot] = deque(maxlen=history_len)
        self._diagnostics: Deque[Dict[str, Any]] = deque(maxlen=history_len)

    def publish(
        self,
        surface: VolSurface,
        *,
        asof: float,
        source: str,
        n_quotes: int,
        rmse_iv: float,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> VolSurfaceSnapshot:
        """Calibrate-side entry point: assigns the next monotonic version
        and atomically swaps the published snapshot."""
        snap = VolSurfaceSnapshot(
            surface=surface,
            version=self._next_version,
            asof=asof,
            source=source,
            n_quotes=n_quotes,
            rmse_iv=rmse_iv,
        )
        self._current = snap  # atomic swap (see module docstring)
        self._history.append(snap)
        self._diagnostics.append(dict(diagnostics or {}))
        self._next_version += 1
        return snap

    def publish_snapshot(self, snap: VolSurfaceSnapshot) -> VolSurfaceSnapshot:
        """Publish a pre-versioned snapshot; rejects stale versions."""
        cur = self._current
        if cur is not None and snap.version <= cur.version:
            raise StaleSnapshotError(
                f"snapshot version {snap.version} <= current {cur.version}"
            )
        self._current = snap  # atomic swap (see module docstring)
        self._history.append(snap)
        self._next_version = max(self._next_version, snap.version + 1)
        return snap

    def latest(self) -> Optional[VolSurfaceSnapshot]:
        """Reader entry point: newest complete snapshot, or None."""
        return self._current

    def history(self) -> list:
        """Bounded diagnostic history (ops only, not the publish contract)."""
        return list(self._history)

    def latest_diagnostics(self) -> Optional[Dict[str, Any]]:
        return dict(self._diagnostics[-1]) if self._diagnostics else None
