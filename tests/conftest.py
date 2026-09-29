"""Shared fixtures: ground-truth SSVI surface for the synthetic feed."""
import math

import numpy as np
import pytest

from snowball_pricer.tick import SyntheticTickFeed

# Ground truth: realistic equity-index-like SSVI surface.
TRUTH = dict(
    thetas=[0.0121, 0.0200, 0.0361, 0.0648],  # ATM var for T = .25/.5/1/2
    expiries=[0.25, 0.5, 1.0, 2.0],
    rho=-0.45,
    eta=0.8,
    gamma=0.5,
    spot=100.0,
    rate=0.03,
    div_yield=0.01,
)


@pytest.fixture()
def feed():
    return SyntheticTickFeed(**TRUTH, seed=42)


@pytest.fixture()
def truth():
    return dict(TRUTH)


def true_iv(truth, K, T):
    """Ground-truth Black IV at strike K, expiry T."""
    from snowball_pricer.surface import ssvi_total_var

    i = truth["expiries"].index(T)
    theta = truth["thetas"][i]
    fwd = truth["spot"] * math.exp((truth["rate"] - truth["div_yield"]) * T)
    k = math.log(K / fwd)
    w = float(ssvi_total_var(k, theta, truth["rho"], truth["eta"], truth["gamma"]))
    return math.sqrt(w / T)
