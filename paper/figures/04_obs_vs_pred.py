"""Figure 4: Observed vs predicted PM2.5 under LOSO-CV (pooled ensemble).

Hexbin density + 1:1 reference line + annotated metrics.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from _style import apply_style

apply_style()

HERE = Path(__file__).parent
R = json.loads((HERE / "results.json").read_text())

records = R["held_out_records"]
y_true = np.array([r["y_true"] for r in records])
y_pred = np.array([r["y_pred"] for r in records])

r2 = r2_score(y_true, y_pred)
rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
mae = float(mean_absolute_error(y_true, y_pred))

# Clip extreme outliers for display (>40 µg/m^3 wildfire-influenced days);
# keep them in metrics but crop axes.
lim = max(40, np.percentile(np.concatenate([y_true, y_pred]), 99.5))
lim = min(lim, 60)

fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.4))
hb = ax.hexbin(y_true, y_pred, gridsize=52, mincnt=1,
               cmap="viridis", linewidths=0,
               extent=(0, lim, 0, lim), bins="log")
cb = plt.colorbar(hb, ax=ax, shrink=0.85, pad=0.02)
cb.set_label("Sensor-days per bin (log)", fontsize=6.5)
cb.ax.tick_params(labelsize=6)

# 1:1 reference
ax.plot([0, lim], [0, lim], color="#EE6677", linewidth=0.9, linestyle="--",
        label="1:1 perfect", zorder=3)

# EPA standard reference lines
for thr, lbl in [(9.0, "EPA NAAQS (9 µg/m$^3$)")]:
    ax.axhline(thr, color="#AA3377", linewidth=0.5, linestyle=":", alpha=0.7)
    ax.axvline(thr, color="#AA3377", linewidth=0.5, linestyle=":", alpha=0.7)

# Metrics textbox
ax.text(0.03, 0.97,
        f"LOSO-CV Ensemble\n$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f} µg/m$^3$\nMAE  = {mae:.2f} µg/m$^3$\n$n$ = {len(y_true):,}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=7, family="serif",
        bbox=dict(facecolor="white", edgecolor="#CCC", linewidth=0.4, boxstyle="round,pad=0.3"))

ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel("Observed PM$_{2.5}$ (µg/m$^3$)")
ax.set_ylabel("Predicted PM$_{2.5}$ (µg/m$^3$)")
ax.set_aspect("equal")
ax.set_title("LOSO-CV predictions (pooled)", loc="left")
ax.legend(loc="lower right", fontsize=6.5)

plt.tight_layout()
out = HERE / "fig4_obs_vs_pred.pdf"
plt.savefig(out)
plt.close()
print(f"Saved {out}")
