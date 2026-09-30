"""Render the SPY implied-vol smile from a live yfinance chain (RTH).

Uses the project's REAL P1 pipeline end-to-end:
    YFinancePollFeed -> build_surface (parity-forward IV inversion,
    raw-SVI per-expiry seeds, global SSVI fit, arbitrage gate)

The figure shows the fitted (published) SSVI smile per expiry as smooth
curves with the raw inverted mid-IV points as faint scatter. If the
arbitrage gate quarantines the surface, NO figure is produced.

Output:
    docs/assets/iv_smile_rth.png   (README figure)
    docs/assets/iv_smile_rth.json  (baked data for docs/demo.html)

Usage: python3 scripts/make_iv_smile.py
"""
import datetime as _dt
import json as _json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from snowball_pricer.arbitrage import ArbitrageViolation
from snowball_pricer.feeds.yfinance import YFinancePollFeed
from snowball_pricer.surface import _invert_expiry, build_surface


def main() -> None:
    feed = YFinancePollFeed(
        ["SPY"],
        rate=0.03,
        div_yield=0.0,
        max_expiries=4,
        min_expiry_days=7,
        k_band=(-0.5, 0.3),
        request_pace_s=1.0,
    )
    quotes, stats = feed.poll_cycle()
    print(f"polled {len(quotes)} quotes, stats={stats.get('underlyings')}")

    ts = max((q.ts for q in quotes), default=_dt.datetime.now().timestamp())
    try:
        surface, diag = build_surface(quotes, asof=ts)
    except (ArbitrageViolation, ValueError) as exc:
        raise SystemExit(f"surface quarantined/failed ({exc}) — no figure produced")
    print(f"surface OK: rmse_iv={diag.get('rmse_iv'):.4f}, n_quotes={diag.get('n_quotes')}")

    today = _dt.date.today()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    smile_data = {
        "asof": today.isoformat(),
        "source": "yfinance RTH (~15-min delayed); fitted SSVI via project P1 build_surface (arbitrage-gated)",
        "rmse_iv": round(float(diag.get("rmse_iv", 0.0)), 6),
        "smiles": [],
    }
    k_grid = np.linspace(-0.55, 0.35, 120)
    for T in sorted({q.expiry for q in quotes if q.is_valid()}):
        qs = [q for q in quotes if q.expiry == T and q.is_valid()]
        raw = _invert_expiry(T, qs)
        if raw["n"] < 5:
            continue
        iv_fit = np.array([surface.iv(float(k), T) for k in k_grid])
        days = int(round(T * 365))
        ax.scatter(raw["k"], raw["iv"], s=8, alpha=0.35)
        ax.plot(k_grid, iv_fit, lw=2, label=f"{days}d (n={raw['n']}, fit)")
        smile_data["smiles"].append({
            "T_years": round(float(T), 6),
            "days": days,
            "k_fit": [round(float(v), 5) for v in k_grid],
            "iv_fit": [round(float(v), 5) for v in iv_fit],
            "k_raw": [round(float(v), 5) for v in raw["k"]],
            "iv_raw": [round(float(v), 5) for v in raw["iv"]],
        })
    # 3D grid for the demo: fitted surface over k x T
    Ts = np.array(sorted({q.expiry for q in quotes if q.is_valid()}))
    k3 = np.linspace(-0.5, 0.3, 30)
    grid = [[round(float(surface.iv(float(k), float(T))), 5) for k in k3] for T in Ts]
    smile_data["grid"] = {
        "k": [round(float(v), 4) for v in k3],
        "T_days": [int(round(float(T) * 365)) for T in Ts],
        "iv": grid,
    }

    ax.axvline(0.0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("log-moneyness  log(K / forward)")
    ax.set_ylabel("implied volatility")
    ax.set_title(
        f"SPY implied-vol smile — fitted SSVI surface, live yfinance chain\n"
        f"{today.isoformat()} (RTH, ~15-min delayed) · calibration RMSE {diag.get('rmse_iv'):.4f} IV pts"
    )
    ax.legend(title="expiry", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    assets = os.path.join(os.path.dirname(__file__), "..", "docs", "assets")
    os.makedirs(assets, exist_ok=True)
    out = os.path.join(assets, "iv_smile_rth.png")
    fig.savefig(out, dpi=150)
    size = os.path.getsize(out)
    jout = os.path.join(assets, "iv_smile_rth.json")
    with open(jout, "w") as f:
        _json.dump(smile_data, f)
    print(f"wrote {out} ({size} bytes); data -> {jout}")
    if size < 20 * 1024:
        raise SystemExit(f"PNG suspiciously small ({size} bytes)")


if __name__ == "__main__":
    main()
