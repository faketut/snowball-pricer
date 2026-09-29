"""snowball-pricer pricing engine (P2): Dupire local vol + QMC snowball pricer.

Consumes ONLY the P1 VolSurfaceSnapshot contract. See docs/p2_design.md.
"""

from .dupire import FlatVol, LocalVolSurface, fd_local_var, ssvi_local_var
from .engine import EngineConfig, PriceResult, price, price_vanilla, set_correlation
from .greeks import delta_crn, ko_probability, vega_bump_surface
from .mc import PathBatchDriver, SimMarket, bridge_order, simulate_terminal
from .payoff import TermSheet, UnderlyingSpec, basket_vol, evaluate_snowball

__all__ = [
    "EngineConfig",
    "FlatVol",
    "LocalVolSurface",
    "PathBatchDriver",
    "PriceResult",
    "SimMarket",
    "TermSheet",
    "UnderlyingSpec",
    "basket_vol",
    "bridge_order",
    "delta_crn",
    "evaluate_snowball",
    "fd_local_var",
    "ko_probability",
    "price",
    "price_vanilla",
    "set_correlation",
    "simulate_terminal",
    "ssvi_local_var",
    "vega_bump_surface",
]
