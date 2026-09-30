"""snowball-pricer: real-time IV-surface-driven snowball/autocallable pricer.

P1 delivers the market-data pipeline: ticks -> IV inversion -> arbitrage-free
SSVI surface -> versioned snapshot publish. See SPEC.md and docs/p1_design.md.
"""

from .arbitrage import (
    ArbitrageViolation,
    check_butterfly_smile,
    check_butterfly_surface,
    check_calendar_surface,
    check_wings,
    durrleman_g,
    validate_surface,
)
from .iv import (
    DegenerateQuoteError,
    IVError,
    NoConvergenceError,
    black_price,
    black_vega,
    forward_from_parity,
    implied_vol,
    implied_vol_safe,
)
from .pipeline import QuoteStore, rebuild_surface, run_pipeline
from .snapshot import SnapshotStore, StaleSnapshotError, VolSurfaceSnapshot
from .surface import (
    VolSurface,
    build_surface,
    fit_raw_svi,
    fit_ssvi_global,
    pava_isotonic,
    raw_svi_total_var,
    ssvi_phi,
    ssvi_total_var,
)
from .tick import Feed, OptionQuote, SyntheticTickFeed
from .feeds import QuestradeFeed
from .validation import (
    HedgeResult,
    Regime,
    RegimeTickFeed,
    ReplayResult,
    pnl_explain,
    run_replay,
    simulate_hedge,
)

__all__ = [
    "ArbitrageViolation",
    "DegenerateQuoteError",
    "Feed",
    "HedgeResult",
    "IVError",
    "NoConvergenceError",
    "OptionQuote",
    "QuoteStore",
    "QuestradeFeed",
    "Regime",
    "RegimeTickFeed",
    "ReplayResult",
    "SnapshotStore",
    "StaleSnapshotError",
    "VolSurface",
    "VolSurfaceSnapshot",
    "black_price",
    "black_vega",
    "build_surface",
    "check_butterfly_smile",
    "check_butterfly_surface",
    "check_calendar_surface",
    "check_wings",
    "durrleman_g",
    "fit_raw_svi",
    "fit_ssvi_global",
    "forward_from_parity",
    "implied_vol",
    "implied_vol_safe",
    "pava_isotonic",
    "pnl_explain",
    "raw_svi_total_var",
    "rebuild_surface",
    "run_pipeline",
    "run_replay",
    "simulate_hedge",
    "ssvi_phi",
    "ssvi_total_var",
    "SyntheticTickFeed",
]
