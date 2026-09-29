"""Correlation module (P3): estimation + stress scenarios.

Estimators (``estimate.py``):
  - ``rolling_pearson``: rolling window Pearson correlation matrices.
  - ``ewma_corr``: RiskMetrics-style exponentially-weighted correlation.
  - ``implied_correlation``: index-vs-components implied average correlation.
  - ``nearest_psd``: eigenvalue-clipping repair so shocked/estimated matrices
    stay valid for the Cholesky factor used by the P2 engine.

Stress (``stress.py``):
  - ``shock_corr``: apply named stress scenarios to a base matrix.
  - ``sensitivity_table``: reprice a basket snowball under base + scenarios
    and tabulate price / KO-probability deltas.

Product note: the basket linkage is basket-AVERAGE (weighted mean of
constituent performances, SPEC §1). Correlation sensitivity is therefore
mild; this module QUANTIFIES it rather than assuming it.
"""

from .estimate import (
    ewma_corr,
    implied_correlation,
    nearest_psd,
    rolling_pearson,
)
from .stress import SCENARIOS, sensitivity_table, shock_corr

__all__ = [
    "ewma_corr",
    "implied_correlation",
    "nearest_psd",
    "rolling_pearson",
    "SCENARIOS",
    "sensitivity_table",
    "shock_corr",
]
