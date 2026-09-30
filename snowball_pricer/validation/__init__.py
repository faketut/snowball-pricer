"""Historical replay validation (P4).

``RegimeTickFeed`` is a synthetic multi-regime option-quote feed (calm ->
vol spike -> calm) used to validate the model the way a desk would: run the
real P1 pipeline on it, reprice every published snapshot with the real P2
engine, then delta-hedge the short position along the TRUE spot path and
explain the realized P&L (``hedge.py``).

Nothing here touches a broker; the "history" is synthetic but the pipeline,
the surface builder, and the pricing engine are all production code paths.
"""
from .hedge import HedgeResult, pnl_explain, simulate_hedge
from .replay import Regime, RegimeTickFeed, ReplayResult, run_replay

__all__ = [
    "HedgeResult",
    "Regime",
    "RegimeTickFeed",
    "ReplayResult",
    "pnl_explain",
    "run_replay",
    "simulate_hedge",
]
