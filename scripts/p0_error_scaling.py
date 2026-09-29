#!/usr/bin/env python3
"""P0 error-budget quantification for snowball-pricer.

Simplified single-asset snowball under flat vol (P0 assumption; term-structure /
skew / stochastic vol deferred to P2):
    S0 = 100, T = 2Y, r = 3% (flat), sigma = 25% (flat)
    Monthly knock-out @ 100% of initial -> autocall, payoff (1 + coupon*t), discounted
    Daily knock-in @ 80% of initial, monitored on daily closes (contractual)
    Coupon 15% p.a.
    At maturity (no KO): no-KI -> (1 + coupon*T); KI -> min(1, S_T/S0) (discounted)

Experiments:
  (a) Monte-Carlo standard error vs number of paths N -> 1/sqrt(N) scaling;
      find N such that SE < 0.1% of notional. Antithetic variates throughout.
  (b) Vega via +/-1 vol-point bump with common random numbers ->
      staleness error for a 2-3 point intraday IV move.
  (c) Discrete-vs-continuous barrier monitoring bias: price with contractual
      daily-close KI monitoring vs continuous monitoring via Brownian-bridge
      touch probabilities; validate the Broadie-Glasserman-Kou (BGK)
      continuity correction recovers the discrete price.

Results are printed and saved to results/p0_results.json.
"""
import json
import os
import time

import numpy as np

# ---------------------------------------------------------------- contract ---
S0 = 100.0
T_YEARS = 2.0
R = 0.03
SIGMA0 = 0.25          # P0 flat-vol assumption
COUPON = 0.15          # p.a., pro-rata on autocall
KO_LEVEL = 100.0       # % of initial, monthly observation
KI_LEVEL = 80.0        # % of initial, daily close observation

DPY = 252
N_DAYS = int(T_YEARS * DPY)                 # 504
OBS_EVERY = 21                              # monthly observation
KO_DAY = np.arange(OBS_EVERY, N_DAYS + 1, OBS_EVERY)  # 1-based day numbers 21..504
T_KO = KO_DAY / DPY
DISC_KO = np.exp(-R * T_KO)
DISC_T = np.exp(-R * T_YEARS)
DT = 1.0 / DPY
BGK_BETA = 0.5826      # -zeta(1/2)/sqrt(2*pi), Broadie-Glasserman-Kou
BATCH_PAIRS = 20000    # antithetic pairs per batch (memory-bounded)

DAY_IDX = np.arange(1, N_DAYS + 1)          # 1-based, for cutoff masking


# ------------------------------------------------------------ simulation ---
def simulate_pair_closes(n_pairs, sigma, seed):
    """Return (n_pairs, N_DAYS) antithetic pair of daily closes, legs stacked.

    Output shape is (2*n_pairs, N_DAYS): first n_pairs rows are the '+' leg,
    next n_pairs the mirrored '-' leg. Pairing is implicit by row index.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_pairs, N_DAYS))
    drift = (R - 0.5 * sigma * sigma) * DT
    vol = sigma * np.sqrt(DT)
    log_p = np.log(S0) + np.cumsum(drift + vol * z, axis=1)
    log_m = np.log(S0) + np.cumsum(drift - vol * z, axis=1)
    return np.exp(log_p), np.exp(log_m)


def first_ko_month(closes):
    """0-based index of first monthly KO, or -1 if never knocked out."""
    hit = closes[:, KO_DAY - 1] >= KO_LEVEL
    any_hit = hit.any(axis=1)
    return np.where(any_hit, hit.argmax(axis=1), -1), any_hit


def discounted_payoffs(Sp, Sm, ki_barrier=KI_LEVEL, bridge=False, bridge_seed=0):
    """Per-pair mean discounted payoff (notional = 1).

    bridge=False: KI monitored on daily closes (contractual).
    bridge=True:  KI monitored continuously; intra-day touch probability via
                  the Brownian bridge, drawn conditionally on the simulated
                  daily endpoints (pathwise-consistent MC).
    """
    n = Sp.shape[0]
    m_p, any_p = first_ko_month(Sp)
    m_m, any_m = first_ko_month(Sm)

    cut_p = np.where(any_p, KO_DAY[np.clip(m_p, 0, len(KO_DAY) - 1)], N_DAYS)
    cut_m = np.where(any_m, KO_DAY[np.clip(m_m, 0, len(KO_DAY) - 1)], N_DAYS)
    day_mask_p = DAY_IDX[None, :] <= cut_p[:, None]
    day_mask_m = DAY_IDX[None, :] <= cut_m[:, None]

    if bridge:
        rng = np.random.default_rng(bridge_seed)
        ki_p = _bridge_ki(Sp, ki_barrier, day_mask_p, rng)
        ki_m = _bridge_ki(Sm, ki_barrier, day_mask_m, rng)
    else:
        ki_p = ((Sp < ki_barrier) & day_mask_p).any(axis=1)
        ki_m = ((Sm < ki_barrier) & day_mask_m).any(axis=1)

    payoff_p = _settle(Sp, m_p, any_p, ki_p)
    payoff_m = _settle(Sm, m_m, any_m, ki_m)
    return 0.5 * (payoff_p + payoff_m)


def _settle(closes, m, any_ko, ki):
    n = closes.shape[0]
    out = np.empty(n)
    ko = any_ko
    mc = np.clip(m, 0, len(KO_DAY) - 1)
    out[ko] = (1.0 + COUPON * T_KO[mc[ko]]) * DISC_KO[mc[ko]]
    alive = ~ko
    s_t = closes[alive, -1]
    out[alive] = np.where(ki[alive],
                          np.minimum(1.0, s_t / S0) * DISC_T,
                          (1.0 + COUPON * T_YEARS) * DISC_T)
    return out


def _bridge_ki(closes, barrier, day_mask, rng):
    """Pathwise KI indicator under continuous monitoring.

    For each daily step, P(min < barrier | endpoints) via the Brownian bridge;
    a touch is drawn conditionally on the endpoints, so the joint law of
    (endpoints, touch indicators) is a valid Monte Carlo draw.
    """
    n = closes.shape[0]
    prev = np.concatenate([np.full((n, 1), S0), closes[:, :-1]], axis=1)
    a = np.log(prev / barrier)
    b = np.log(closes / barrier)
    # bridge touch probability; endpoints at/below barrier -> touch prob 1
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.exp(-2.0 * a * b / (SIGMA0 * SIGMA0 * DT))
    p = np.where((a <= 0) | (b <= 0), 1.0, np.minimum(p, 1.0))
    touched = rng.random((n, N_DAYS)) < p
    return (touched & day_mask).any(axis=1)


class Accum:
    """Online mean / SE accumulator over pair-mean payoffs."""

    def __init__(self):
        self.n = 0
        self.s = 0.0
        self.s2 = 0.0

    def add(self, x):
        self.n += x.size
        self.s += x.sum()
        self.s2 += np.square(x).sum()

    def price(self):
        return self.s / self.n

    def se(self):
        mean = self.price()
        var = max(self.s2 / self.n - mean * mean, 0.0)
        return np.sqrt(var / self.n)


def run_price(total_paths, sigma, seed0, ki_barrier=KI_LEVEL, bridge=False):
    """Price with antithetic pairs in batches; returns (price, se, n_pairs)."""
    acc = Accum()
    pairs_total = total_paths // 2
    done = 0
    b = 0
    while done < pairs_total:
        nb = min(BATCH_PAIRS, pairs_total - done)
        Sp, Sm = simulate_pair_closes(nb, sigma, seed0 + b)
        pm = discounted_payoffs(Sp, Sm, ki_barrier=ki_barrier, bridge=bridge,
                                bridge_seed=seed0 + 10_000 + b)
        acc.add(pm)
        done += nb
        b += 1
    return acc.price(), acc.se(), acc.n


# ------------------------------------------------------------ experiment (a)
def experiment_a():
    print("=== (a) standard error vs N ===", flush=True)
    rows = []
    for N in (10_000, 40_000, 160_000, 640_000):
        t = time.time()
        price, se, _ = run_price(N, SIGMA0, seed0=1234)
        rows.append({"N": N, "price": price, "se": se,
                     "se_times_sqrtN": se * np.sqrt(N),
                     "secs": time.time() - t})
        print(f"N={N:>8d}  price={price:.5f}  SE={se:.6f}  "
              f"SE*sqrt(N)={se * np.sqrt(N):.5f}", flush=True)
    # N* for SE < 0.001 (0.1% of notional), extrapolated from the largest N
    last = rows[-1]
    n_star = int(last["N"] * (last["se"] / 0.001) ** 2) + 1
    print(f"-> N for SE < 0.1% of notional: ~{n_star} paths "
          f"(extrapolated; 640k run achieved SE={last['se']:.6f})", flush=True)
    return {"rows": rows, "N_star_for_se_below_0_1pct": n_star}


# ------------------------------------------------------------ experiment (b)
def experiment_b():
    print("=== (b) vega via +-1pt bump (common random numbers) ===", flush=True)
    total_paths = 200_000
    pairs_total = total_paths // 2
    acc_up = Accum()
    acc_dn = Accum()
    acc_d = Accum()  # paired difference of pair-means
    done, b = 0, 0
    while done < pairs_total:
        nb = min(BATCH_PAIRS, pairs_total - done)
        # same normals for both bumps -> CRN
        Sp_u, Sm_u = simulate_pair_closes(nb, SIGMA0 + 0.01, 777 + b)
        Sp_d, Sm_d = simulate_pair_closes(nb, SIGMA0 - 0.01, 777 + b)
        pm_u = discounted_payoffs(Sp_u, Sm_u)
        pm_d = discounted_payoffs(Sp_d, Sm_d)
        acc_up.add(pm_u)
        acc_dn.add(pm_d)
        acc_d.add((pm_u - pm_d) / 2.0)   # per-1-vol-point sensitivity
        done += nb
        b += 1
    vega = acc_d.price()
    vega_se = acc_d.se()
    print(f"P(26%)={acc_up.price():.5f}  P(24%)={acc_dn.price():.5f}", flush=True)
    print(f"vega = {vega:.5f} per vol point (SE {vega_se:.6f}) "
          f"= {100 * vega:.3f}% of notional per vol point", flush=True)
    for mv in (2.0, 3.0):
        print(f"  staleness error for {mv:.0f}pt intraday IV move: "
              f"{100 * abs(vega) * mv:.3f}% of notional", flush=True)
    return {"price_up": acc_up.price(), "price_dn": acc_dn.price(),
            "vega_per_vol_point": vega, "vega_se": vega_se,
            "staleness_2pt_pct": 100 * abs(vega) * 2.0,
            "staleness_3pt_pct": 100 * abs(vega) * 3.0}


# ------------------------------------------------------------ experiment (c)
def experiment_c():
    print("=== (c) discrete vs continuous KI monitoring bias ===", flush=True)
    total_paths = 400_000
    p_disc, se_disc, _ = run_price(total_paths, SIGMA0, seed0=4242, bridge=False)
    p_cont, se_cont, _ = run_price(total_paths, SIGMA0, seed0=4242, bridge=True)
    b_star = KI_LEVEL * np.exp(-BGK_BETA * SIGMA0 * np.sqrt(DT))
    p_bgk, se_bgk, _ = run_price(total_paths, SIGMA0, seed0=4242,
                                 ki_barrier=b_star, bridge=True)
    bias_bps = 10000 * (p_cont - p_disc)
    resid_bps = 10000 * (p_bgk - p_disc)
    print(f"discrete (contractual) : {p_disc:.5f} (SE {se_disc:.6f})", flush=True)
    print(f"continuous (bridge)    : {p_cont:.5f} (SE {se_cont:.6f})", flush=True)
    print(f"continuous + BGK corr. : {p_bgk:.5f} (SE {se_bgk:.6f})  "
          f"[B*={b_star:.3f}]", flush=True)
    print(f"-> discrete-vs-continuous bias: {bias_bps:.1f} bps of notional", flush=True)
    print(f"-> BGK residual vs discrete    : {resid_bps:.1f} bps", flush=True)
    return {"price_discrete": p_disc, "se_discrete": se_disc,
            "price_continuous": p_cont, "se_continuous": se_cont,
            "price_bgk": p_bgk, "se_bgk": se_bgk,
            "bgk_barrier": b_star,
            "discrete_vs_continuous_bias_bps": bias_bps,
            "bgk_residual_bps": resid_bps}


def main():
    t0 = time.time()
    out = {"contract": {
        "S0": S0, "T_years": T_YEARS, "r": R, "sigma_flat": SIGMA0,
        "coupon_pa": COUPON, "KO_level_pct": KO_LEVEL, "KO_freq": "monthly",
        "KI_level_pct": KI_LEVEL, "KI_monitoring": "daily closes",
        "day_count": f"{DPY}/yr", "n_days": N_DAYS},
        "note": "P0 uses flat vol; term-structure/skew/stochastic-vol deferred to P2."}
    out["a_se_vs_N"] = experiment_a()
    out["b_vega"] = experiment_b()
    out["c_barrier_bias"] = experiment_c()
    out["elapsed_secs"] = time.time() - t0
    os.makedirs("results", exist_ok=True)
    with open("results/p0_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone in {out['elapsed_secs']:.1f}s. Results -> results/p0_results.json",
          flush=True)


if __name__ == "__main__":
    main()
