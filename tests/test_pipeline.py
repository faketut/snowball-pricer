"""End-to-end pipeline test: synthetic feed -> ticks -> IV -> surface -> snapshot."""
import dataclasses

import pytest

from snowball_pricer.pipeline import QuoteStore, rebuild_surface, run_pipeline
from snowball_pricer.snapshot import (
    SnapshotStore,
    StaleSnapshotError,
    VolSurfaceSnapshot,
)
from snowball_pricer.surface import VolSurface


def test_quote_store_drops_bad_quotes(feed):
    store = QuoteStore()
    it = feed.subscribe()
    q = next(it)
    assert store.update(q) is True
    # Older timestamp for the same key is dropped.
    old = dataclasses.replace(q, ts=q.ts - 1.0)
    assert store.update(old) is False
    # Crossed quote is dropped.
    crossed = dataclasses.replace(q, bid=q.ask + 0.5, ask=q.ask, ts=q.ts + 1.0)
    assert store.update(crossed) is False
    assert len(store) == 1


def test_end_to_end_publish(feed):
    store = QuoteStore()
    snapshots = SnapshotStore()
    stats = run_pipeline(
        feed,
        n_ticks=feed.sweep_size * 2,
        rebuild_every=feed.sweep_size,
        store=store,
        snapshots=snapshots,
    )
    assert stats["rebuilds"] == 2
    assert stats["rebuild_failures"] == 0
    assert stats["published_versions"] == [1, 2]

    latest = snapshots.latest()
    assert latest is not None
    assert latest.version == 2
    assert isinstance(latest.surface, VolSurface)
    assert latest.rmse_iv < 0.3
    assert latest.source == "synthetic"


def test_rebuild_failure_keeps_last_good_snapshot(feed, monkeypatch):
    """A quarantined rebuild must not replace the live snapshot."""
    import snowball_pricer.surface as surf_mod
    from snowball_pricer.arbitrage import ArbitrageViolation

    store = QuoteStore()
    snapshots = SnapshotStore()
    stats = run_pipeline(
        feed, n_ticks=feed.sweep_size, rebuild_every=feed.sweep_size,
        store=store, snapshots=snapshots,
    )
    assert stats["published_versions"] == [1]

    def boom(*a, **k):
        raise ArbitrageViolation({"all_ok": False})

    monkeypatch.setattr("snowball_pricer.pipeline.build_surface", boom)
    stats2 = run_pipeline(
        feed, n_ticks=feed.sweep_size, rebuild_every=feed.sweep_size,
        store=store, snapshots=snapshots,
    )
    assert stats2["rebuild_failures"] == 1
    assert snapshots.latest().version == 1  # still the good one


def test_stale_snapshot_rejected():
    store = SnapshotStore()
    surf = VolSurface(
        expiries=(1.0,), thetas=(0.04,), rho=-0.4, eta=0.8, gamma=0.5,
        forwards=(100.0,), spot=100.0, rate=0.03, div_yield=0.01,
    )
    s1 = store.publish(surf, asof=1.0, source="t", n_quotes=10, rmse_iv=0.1)
    assert s1.version == 1
    stale = VolSurfaceSnapshot(
        surface=surf, version=1, asof=2.0, source="t", n_quotes=10, rmse_iv=0.1
    )
    with pytest.raises(StaleSnapshotError):
        store.publish_snapshot(stale)
    newer = VolSurfaceSnapshot(
        surface=surf, version=2, asof=2.0, source="t", n_quotes=10, rmse_iv=0.1
    )
    store.publish_snapshot(newer)
    assert store.latest().version == 2
