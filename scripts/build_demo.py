"""Build docs/demo.html — single self-contained interactive demo page.

Data is baked in at build time (no backend):
- Tab 1 (IV surface 3D): fitted SSVI grid from docs/assets/iv_smile_rth.json
  (live yfinance RTH SPY chain, P1 pipeline).
- Tab 2 (price sensitivity): P0 measured vega + P3 measured correlation table.
- Tab 3 (P&L explain waterfall): P4 replay measured buckets.

Every figure is labeled with its data source. No fabricated precision.

Usage: python3 scripts/build_demo.py
"""
import json
import os
import sys

ASSETS = os.path.join(os.path.dirname(__file__), "..", "docs", "assets")
OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "demo.html")

# --- P3 measured correlation sensitivity (docs/p3_design.md, exact) ---
# 3-asset basket-average snowball, FlatVol 25%, 2Y, monthly KO @100%,
# daily KI @80%, 15% coupon, equal weights; base rho=0.5, 40k QMC paths.
P3 = [
    ("dispersion\n\u03c1=0.05", 0.05, +142.0),
    ("minus_0.2\n\u03c1=0.30", 0.30, +54.6),
    ("base\n\u03c1=0.50", 0.50, 0.0),
    ("plus_0.2\n\u03c1=0.70", 0.70, -47.8),
    ("crisis\n\u03c1=0.95", 0.95, -91.9),
]

# --- P0 measured vega: -0.195%/vol-pt = -19.5 bps per vol point (docs/error_budget.md) ---
VEGA_BPS_PER_VOLPT = -19.5
VOL_BASE = 25.0

# --- P4 replay P&L explain buckets, bps (docs/p4_report.md, exact) ---
P4 = [
    ("delta\n(net hedge \u2212 model)", 0.0),
    ("vega\n(\u2212v\u00b7dATMiv)", 14.0),
    ("theta/carry\n(model)", -15.2),
    ("cash interest\n(3%)", 8.3),
    ("residual\n(unmodelled \u03b3/2nd-order)", 44.4),
]
P4_TOTAL = 51.5  # 0.0 + 14.0 - 15.2 + 8.3 + 44.4


def main() -> None:
    with open(os.path.join(ASSETS, "iv_smile_rth.json")) as f:
        smile = json.load(f)
    grid = smile["grid"]

    # Tab 2a: price-vs-vol line from measured vega (labeled as linear extrapolation)
    vols = [15.0 + 0.5 * i for i in range(41)]
    dprice = [VEGA_BPS_PER_VOLPT * (v - VOL_BASE) for v in vols]

    p3_labels = [r[0] for r in P3]
    p3_vals = [r[2] for r in P3]
    p4_labels = [r[0] for r in P4] + ["ACTUAL\nrealized"]
    # plotly waterfall: use relative measures then a total
    p4_measures = ["relative"] * len(P4) + ["total"]
    p4_y = [r[1] for r in P4] + [P4_TOTAL]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>snowball-pricer — interactive demo</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
         margin: 0; padding: 0 16px 40px; color: #1f2937; background: #fafafa; }}
  h1 {{ font-size: 22px; margin: 18px 0 4px; }}
  .sub {{ color: #6b7280; font-size: 13px; margin-bottom: 12px; }}
  .tabs {{ display: flex; gap: 6px; margin: 10px 0 16px; }}
  .tab-btn {{ padding: 8px 16px; border: 1px solid #d1d5db; background: #fff;
              cursor: pointer; border-radius: 6px; font-size: 14px; }}
  .tab-btn.active {{ background: #1f2937; color: #fff; border-color: #1f2937; }}
  .tab {{ display: none; }}
  .tab.active {{ display: block; }}
  .src {{ font-size: 12px; color: #6b7280; margin: 8px 0 4px; }}
  .note {{ font-size: 12px; color: #6b7280; max-width: 900px; }}
  .chart {{ width: 100%; max-width: 960px; height: 560px; }}
</style>
</head>
<body>
<h1>snowball-pricer — interactive demo</h1>
<div class="sub">Basket snowball / autocallable pricer: live-data surface, measured sensitivities, replay P&amp;L explain.</div>
<div class="tabs">
  <button class="tab-btn active" onclick="showTab(0)">1 · IV surface (live)</button>
  <button class="tab-btn" onclick="showTab(1)">2 · Price sensitivity</button>
  <button class="tab-btn" onclick="showTab(2)">3 · P&amp;L explain</button>
</div>

<div class="tab active" id="tab0">
  <div class="chart" id="surface3d"></div>
  <div class="src">Source: live yfinance SPY option chain, {smile['asof']} (RTH, ~15-min delayed) —
  fitted SSVI surface via the project's P1 pipeline (<code>build_surface</code>, arbitrage-gated);
  calibration RMSE {smile['rmse_iv']:.4f} IV pts.</div>
  <div class="note">Drag to rotate. The 3D surface is the exact object the pricing engine consumes
  (Dupire local-vol MC); nothing here is simulated.</div>
</div>

<div class="tab" id="tab1">
  <div class="chart" id="volsens" style="height:440px"></div>
  <div class="src">Source: P0 measured vega &minus;19.5 bps/vol-pt (flat 25% vol, 2Y snowball, monthly KO @100%,
  daily KI @80%, 15% coupon). Line is a linear extrapolation of the measured vega — local only.</div>
  <div class="chart" id="rhosens" style="height:440px"></div>
  <div class="src">Source: P3 measured (40k QMC paths, base &rho;=0.5, same seed — deltas are CRN-driven, not MC noise).
  &Delta;price in bps of notional. Sign is empirical for this term sheet (higher &rho; &rarr; higher basket vol &rarr; lower price).</div>
</div>

<div class="tab" id="tab2">
  <div class="chart" id="waterfall" style="height:480px"></div>
  <div class="src">Source: P4 14-day synthetic replay (calm &rarr; vol spike &rarr; calm), short snowball,
  delta-hedged daily. Attribution identity: 0.0 + 14.0 &minus; 15.2 + 8.3 + 44.4 = +51.5 bps.</div>
  <div class="note">The +44.4 bps residual is primarily unmodelled gamma / second-order risk (the engine has no
  gamma bucket) — it is an honesty term, not a pricing error.</div>
</div>

<script>
function showTab(i) {{
  document.querySelectorAll('.tab').forEach((t, j) => t.classList.toggle('active', i === j));
  document.querySelectorAll('.tab-btn').forEach((b, j) => b.classList.toggle('active', i === j));
}}

// ---- Tab 1: 3D IV surface (baked live data) ----
const grid = {json.dumps(grid)};
Plotly.newPlot('surface3d', [{{
  type: 'surface',
  x: grid.k, y: grid.T_days, z: grid.iv,
  colorscale: 'Viridis', showscale: true,
  colorbar: {{ title: 'IV' }},
  hovertemplate: 'log-moneyness=%{{x:.3f}}<br>expiry=%{{y}}d<br>IV=%{{z:.4f}}<extra></extra>'
}}], {{
  margin: {{ l: 0, r: 0, t: 30, b: 0 }},
  title: {{ text: 'SPY fitted SSVI implied-vol surface', font: {{ size: 15 }} }},
  scene: {{
    xaxis: {{ title: 'log-moneyness log(K/F)' }},
    yaxis: {{ title: 'expiry (days)' }},
    zaxis: {{ title: 'implied vol' }}
  }}
}}, {{ responsive: true }});

// ---- Tab 2a: price vs flat vol (measured vega, linear extrapolation) ----
const vols = {json.dumps(vols)};
const dprice = {json.dumps(dprice)};
Plotly.newPlot('volsens', [{{
  x: vols, y: dprice, mode: 'lines', name: '\\u0394price',
  line: {{ width: 2.5 }}
}}], {{
  margin: {{ l: 60, r: 20, t: 40, b: 50 }},
  title: {{ text: '\\u0394 snowball price vs flat vol (measured vega)', font: {{ size: 15 }} }},
  xaxis: {{ title: 'flat vol (%)' }},
  yaxis: {{ title: '\\u0394 price (bps of notional)' }},
  shapes: [{{ type: 'line', x0: {VOL_BASE}, x1: {VOL_BASE}, y0: Math.min(...dprice), y1: Math.max(...dprice),
             line: {{ dash: 'dash', color: '#9ca3af' }} }}],
  annotations: [{{ x: {VOL_BASE}, y: Math.max(...dprice), text: 'P0 base 25%',
                  showarrow: false, xanchor: 'left', font: {{ size: 11, color: '#6b7280' }} }}]
}}, {{ responsive: true }});

// ---- Tab 2b: price vs correlation (P3 measured) ----
const p3x = {json.dumps(p3_labels)};
const p3y = {json.dumps(p3_vals)};
Plotly.newPlot('rhosens', [{{
  x: p3x, y: p3y, type: 'bar',
  marker: {{ color: p3y.map(v => v >= 0 ? '#3b82f6' : '#ef4444') }},
  text: p3y.map(v => (v > 0 ? '+' : '') + v.toFixed(1) + ' bps'),
  textposition: 'outside'
}}], {{
  margin: {{ l: 60, r: 20, t: 40, b: 80 }},
  title: {{ text: '\\u0394 snowball price vs correlation (measured)', font: {{ size: 15 }} }},
  xaxis: {{ title: 'scenario' }},
  yaxis: {{ title: '\\u0394 price (bps of notional)' }}
}}, {{ responsive: true }});

// ---- Tab 3: P&L explain waterfall (P4 measured) ----
Plotly.newPlot('waterfall', [{{
  type: 'waterfall',
  x: {json.dumps(p4_labels)},
  y: {json.dumps(p4_y)},
  measure: {json.dumps(p4_measures)},
  text: {json.dumps([f"{{v:+.1f}}" for v in [r[1] for r in P4]] + [f"{{P4_TOTAL:+.1f}}"])},
  textposition: 'outside',
  connector: {{ line: {{ color: '#9ca3af' }} }},
  increasing: {{ marker: {{ color: '#3b82f6' }} }},
  decreasing: {{ marker: {{ color: '#ef4444' }} }},
  totals: {{ marker: {{ color: '#1f2937' }} }}
}}], {{
  margin: {{ l: 60, r: 20, t: 40, b: 80 }},
  title: {{ text: 'Short-snowball delta-hedged P&L explain (bps)', font: {{ size: 15 }} }},
  yaxis: {{ title: 'bps of notional' }}
}}, {{ responsive: true }});
</script>
</body>
</html>
"""
    with open(OUT, "w") as f:
        f.write(html)
    print(f"wrote {OUT} ({os.path.getsize(OUT)} bytes)")


if __name__ == "__main__":
    main()
