"""P3 correlation module tests.

(a) Estimators recover a known synthetic rho = 0.6 (correlated GBM returns).
(b) Implied-correlation demo recovers the average rho on synthetic index data.
(c) shock_corr: scenario definitions + PSD repair.
(d) sensitivity_table: basket variance increases monotonically with rho;
    price sensitivity for basket-AVERAGE linkage is small:
    measured |dPrice| for a +/-0.2 rho move is ~55 bps at 40k QMC paths
    (2026-09-29), so the bound here is set at 150 bps (~3x margin).
    No direction is asserted: KO/KI effects oppose (see docs/p3_design.md).
"""

import numpy as np
import pytest

from snowball_pricer.correlation import (
    SCENARIOS,
    ewma_corr,
    implied_correlation,
    nearest_psd,
    rolling_pearson,
    sensitivity_table,
    shock_corr,
)
from snowball_pricer.correlation.stress import _flat_snapshot
from snowball_pricer.pricing.engine import EngineConfig, price
from snowball_pricer.pricing.payoff import TermSheet, UnderlyingSpec

TRUE_RHO = 0.6
N_ASSETS = 3
SEED = 20260929


def _gbm_returns(rho, n_obs=6000, vols=(0.20, 0.25, 0.30), seed=SEED):
    """Correlated log-returns from a known equicorrelated GBM."""
    rng = np.random.default_rng(seed)
    corr = np.full((N_ASSETS, N_ASSETS), rho)
    np.fill_diagonal(corr, 1.0)
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_obs, N_ASSETS)) @ L.T
    dt = 1 / 252
    return z * np.asarray(vols)[None, :] * np.sqrt(dt)


def _base_corr(rho=0.5):
    c = np.full((N_ASSETS, N_ASSETS), rho)
    np.fill_diagonal(c, 1.0)
    return c


# ---------------------------------------------------------- estimators -----

def test_ewma_recovers_true_rho():
    rets = _gbm_returns(TRUE_RHO)
    # Default lam=0.94 (RiskMetrics daily standard) is an endpoint estimator
    # with effective N ~ 1/(1-lam) = 17 observations: unbiased-ish but noisy
    # (sampling std ~0.13 across seeds). Tolerance is honest, not aspirational.
    est = ewma_corr(rets)
    assert est.shape == (N_ASSETS, N_ASSETS)
    np.testing.assert_allclose(np.diag(est), 1.0, atol=1e-12)
    assert np.linalg.eigvalsh(est).min() > -1e-9  # valid for Cholesky
    off = est[np.triu_indices(N_ASSETS, k=1)]
    assert abs(off.mean() - TRUE_RHO) < 0.15


def test_ewma_consistent_as_decay_slows():
    # Slower decay => larger effective sample => bias/noise vanish.
    # Verified empirically (2026-09-29): lam=0.995 gives mean 0.591, std 0.018
    # across 12 seeds; lam=0.999 gives mean 0.598, std 0.011.
    rets = _gbm_returns(TRUE_RHO)
    est = ewma_corr(rets, lam=0.995)
    off = est[np.triu_indices(N_ASSETS, k=1)]
    assert abs(off.mean() - TRUE_RHO) < 0.05


def test_rolling_pearson_recovers_true_rho():
    rets = _gbm_returns(TRUE_RHO)
    mats = rolling_pearson(rets, window=252)
    assert mats.shape == (6000 - 252 + 1, N_ASSETS, N_ASSETS)
    off = mats[:, 0, 1]
    assert abs(off.mean() - TRUE_RHO) < 0.05
    assert (np.abs(off) <= 1.0 + 1e-12).all()
    np.testing.assert_allclose(np.diagonal(mats, axis1=1, axis2=2), 1.0,
                               atol=1e-12)


def test_rolling_pearson_window_too_large():
    with pytest.raises(AssertionError):
        rolling_pearson(np.zeros((10, 2)), window=20)


def test_nearest_psd_repairs_and_is_idempotent():
    bad = np.array([[1.0, 0.99, 0.99],
                    [0.99, 1.0, 0.0],
                    [0.99, 0.0, 1.0]])  # not PSD
    assert np.linalg.eigvalsh(bad).min() < 0
    fixed = nearest_psd(bad)
    assert np.linalg.eigvalsh(fixed).min() > -1e-9
    np.testing.assert_allclose(np.diag(fixed), 1.0, atol=1e-12)
    good = _base_corr(0.5)
    np.testing.assert_allclose(nearest_psd(good), good, atol=1e-8)


# --------------------------------------------------- implied correlation ---

def test_implied_correlation_recovers_average_rho():
    # Index proxy built exactly from constituents (no idiosyncratic basis).
    vols = np.array([0.20, 0.25, 0.30])
    w = np.array([0.4, 0.35, 0.25])
    comp_vars = vols ** 2
    n = len(vols)
    corr = np.full((n, n), TRUE_RHO)
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(vols, vols) * corr
    index_var = float(w @ cov @ w)
    rho_hat = implied_correlation(index_var, comp_vars, w)
    assert abs(rho_hat - TRUE_RHO) < 0.01


def test_implied_correlation_clips_violations():
    # Index variance below the diagonal-only bound: model violated -> clip.
    rho_hat = implied_correlation(0.0001, np.array([0.04, 0.04]),
                                  np.array([0.5, 0.5]))
    assert -0.99 <= rho_hat <= 0.99


# --------------------------------------------------------- stress module ---

def test_shock_corr_scenarios():
    base = _base_corr(0.5)
    for s in SCENARIOS:
        m = shock_corr(base, s)
        assert m.shape == (N_ASSETS, N_ASSETS)
        np.testing.assert_allclose(np.diag(m), 1.0, atol=1e-12)
        assert np.linalg.eigvalsh(m).min() > -1e-9
        off = m[np.triu_indices(N_ASSETS, k=1)].mean()
        assert abs(off) <= 0.99 + 1e-9
    assert abs(shock_corr(base, "plus_0.2")[0, 1] - 0.7) < 1e-9
    assert abs(shock_corr(base, "minus_0.2")[0, 1] - 0.3) < 1e-9
    assert abs(shock_corr(base, "crisis_1.0")[0, 1] - 0.95) < 1e-6
    assert abs(shock_corr(base, "dispersion_0.0")[0, 1] - 0.05) < 1e-6


def test_shock_corr_unknown_scenario():
    with pytest.raises(ValueError, match="unknown correlation scenario"):
        shock_corr(_base_corr(), "whatever")


def _basket_var(corr, vols, weights):
    """Analytic instantaneous basket variance at equal spot."""
    cov = np.outer(vols, vols) * corr
    return float(weights @ cov @ weights)


def test_sensitivity_table_monotonic_variance_small_price_sensitivity():
    # Reduced config for test speed; bound set from the measured 40k-path
    # run (~55 bps per +/-0.2 move) with ~3x margin.
    cfg = EngineConfig(n_paths=8192, qmc=True, seed=SEED,
                       batch_size=8192, compute_greeks=False)
    rows = sensitivity_table(
        _base_corr(0.5), scenarios=("plus_0.2", "minus_0.2"),
        weights=(1 / 3, 1 / 3, 1 / 3), cfg=cfg)
    assert [r.scenario for r in rows] == ["base", "plus_0.2", "minus_0.2"]

    vols = np.array([0.25, 0.25, 0.25])
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    var_minus = _basket_var(shock_corr(_base_corr(0.5), "minus_0.2"), vols, w)
    var_base = _basket_var(_base_corr(0.5), vols, w)
    var_plus = _basket_var(shock_corr(_base_corr(0.5), "plus_0.2"), vols, w)
    assert var_minus < var_base < var_plus  # basket variance rises with rho

    base, plus, minus = rows
    for scen in (plus, minus):
        assert abs(scen.delta_price_bps) < 150.0  # average linkage: mild
        # signal must clear MC noise (same-seed repricing => CRN deltas)
        noise = 2.0 * np.hypot(base.std_error, scen.std_error) * 1e4
        assert abs(scen.delta_price_bps) > noise


def test_corr_actually_reaches_simulator():
    # A decorrelated vs crisis matrix must move the simulated basket vol:
    # smoke test that the corr argument is threaded through to paths.
    snapshot = _flat_snapshot()
    terms = TermSheet()
    underlyings = [UnderlyingSpec(name=f"A{i+1}", weight=1/3)
                   for i in range(3)]
    cfg = EngineConfig(n_paths=8192, qmc=True, seed=SEED,
                       batch_size=8192, compute_greeks=False)
    p0 = price(snapshot, terms, underlyings, cfg=cfg, corr=_base_corr(0.05))
    p1 = price(snapshot, terms, underlyings, cfg=cfg, corr=_base_corr(0.95))
    assert abs(p0.price - p1.price) > 0.0  # correlation bites somewhere
