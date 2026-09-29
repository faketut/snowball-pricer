"""P1 pipeline orchestration: two-speed market-data pipeline.

- Fast path (per tick): ``QuoteStore.update`` — O(1) latest-quote bookkeeping.
  Target: microseconds per tick (see PERF.md).
- Slow path (periodic): ``rebuild_surface`` — full IV inversion +
  SSVI calibration + arbitrage gate + snapshot publish. Runs on a cadence
  (every N ticks / seconds / on significant underlying move), NOT per tick.
  Target: <100ms p99 on the reference workload (see PERF.md).

``run_pipeline`` wires a feed through both paths; it is the harness used by
the end-to-end test and the latency benchmark.
"""
from __future__ import annotations

import math
import time
from typing import Dict, Iterator, List, Optional, Tuple

from .arbitrage import ArbitrageViolation
from .snapshot import SnapshotStore, VolSurfaceSnapshot
from .surface import VolSurface, build_surface
from .tick import Feed, OptionQuote


class QuoteStore:
    """Latest-quote-per-key book. Keys are
    (underlying_id, expiry, strike, is_call)."""

    def __init__(self) -> None:
        self._q: Dict[Tuple[str, float, float, bool], OptionQuote] = {}

    def update(self, quote: OptionQuote) -> bool:
        """Store the latest quote for its key. Returns False when the quote
        was dropped (structurally invalid or older than the stored one)."""
        if not quote.is_valid():
            return False
        k = quote.key
        old = self._q.get(k)
        if old is not None and quote.ts < old.ts:
            return False
        self._q[k] = quote
        return True

    def all(self) -> List[OptionQuote]:
        return list(self._q.values())

    def __len__(self) -> int:
        return len(self._q)


def rebuild_surface(
    store: QuoteStore, asof: float, source: str = "synthetic"
) -> Tuple[VolSurface, Dict]:
    """Slow path: invert the whole quote book, calibrate, gate, return.

    Raises ArbitrageViolation (quarantine: do NOT publish) or ValueError
    (too few quotes / fit failure).
    """
    quotes = store.all()
    return build_surface(quotes, asof)


def run_pipeline(
    feed: Feed,
    *,
    n_ticks: int,
    rebuild_every: int,
    source: str = "synthetic",
    store: Optional[QuoteStore] = None,
    snapshots: Optional[SnapshotStore] = None,
) -> Dict:
    """Drive ``n_ticks`` through the fast path, rebuilding + publishing the
    surface every ``rebuild_every`` ticks. Returns run stats.

    A rebuild that raises ArbitrageViolation/ValueError is counted in
    ``rebuild_failures`` and does NOT replace the published snapshot (the
    previous good snapshot stays live).
    """
    store = store or QuoteStore()
    snapshots = snapshots or SnapshotStore()
    stats: Dict = {
        "ticks": 0,
        "dropped": 0,
        "rebuilds": 0,
        "rebuild_failures": 0,
        "published_versions": [],
    }
    it = feed.subscribe()
    for i in range(n_ticks):
        try:
            quote = next(it)
        except StopIteration:
            break
        stats["ticks"] += 1
        if not store.update(quote):
            stats["dropped"] += 1
        if (i + 1) % rebuild_every == 0:
            stats["rebuilds"] += 1
            try:
                surface, diag = rebuild_surface(store, asof=quote.ts, source=source)
            except (ArbitrageViolation, ValueError):
                stats["rebuild_failures"] += 1
                continue
            snap = snapshots.publish(
                surface,
                asof=quote.ts,
                source=source,
                n_quotes=diag["n_quotes"],
                rmse_iv=diag["rmse_iv"],
                diagnostics=diag,
            )
            stats["published_versions"].append(snap.version)
    return stats
