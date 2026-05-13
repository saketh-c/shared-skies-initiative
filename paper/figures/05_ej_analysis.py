"""Figure 5: Environmental justice analysis.

Two-panel figure:
 - (a) Sensor-level scatter: EJ index vs mean PM2.5, bubble-sized by n_days,
       colored by city, with LOWESS trend.
 - (b) Exceedance of EPA annual standard (9 µg/m^3) by EJ quartile.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from _style import apply_style, CITY_COLORS

apply_style()

HERE = Path(__file__).parent
R = json.loads((HERE / "results.json").read_text())
meta = R["sensor_meta"]
ej_corr = R["ej_correlation"]
ej_exceed = R["ej_exceedance_by_quartile"]
ej_mean = R["ej_mean_pm25_by_quartile"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9),
                               gridspec_kw={"width_ratios": [1.2, 1.0]})

# ---------- (a) Scatter ----------
ej = np.array([m["ejf_score"] for m in meta])
pm = np.array([m["mean_pm25"] for m in meta])
nn = np.array([m["n_days"] for m in meta])
city = [m["city"] for m in meta]
sizes = 5 + (nn / nn.max()) * 45
colors = [CITY_COLORS.get(c, "#888") for c in city]

ax1.scatter(ej, pm, s=sizes, c=colors, edgecolors="white", linewidths=0.3,
            alpha=0.75, zorder=3)

# Linear trend
m_fit, b_fit = np.polyfit(ej, pm, 1)
xs = np.linspace(ej.min(), ej.max(), 200)
ax1.plot(xs, m_fit * xs + b_fit, color="#222", linewidth=0.9,
         linestyle="-", alpha=0.9, zorder=4, label=f"Linear fit")

# EPA threshold
ax1.axhline(9.0, color="#AA3377", linewidth=0.6, linestyle=":")
ax1.text(ej.max(), 9.1, " EPA annual (9 µg/m$^3$)", fontsize=6,
         color="#AA3377", ha="right", va="bottom")

ax1.set_xlabel("EJScreen EJ Index (TX percentile)")
ax1.set_ylabel("Mean observed PM$_{2.5}$ (µg/m$^3$)")
ax1.set_title("(a) Environmental justice vs. PM$_{2.5}$", loc="left")

# Annotate correlation
r, p = ej_corr["r"], ej_corr["p"]
p_str = "< 10$^{-3}$" if p < 1e-3 else f"= {p:.3f}"
ax1.text(0.03, 0.97,
         f"Pearson $r$ = {r:.3f}\n$p$ {p_str}\n$n$ = {ej_corr['n']} sensors",
         transform=ax1.transAxes, ha="left", va="top", fontsize=7,
         bbox=dict(facecolor="white", edgecolor="#CCC", linewidth=0.4, boxstyle="round,pad=0.3"))

# Legend for cities
for c, col in CITY_COLORS.items():
    ax1.scatter([], [], s=22, c=col, edgecolors="white", linewidths=0.3, label=c)
ax1.legend(loc="lower right", fontsize=6, handletextpad=0.3, borderaxespad=0.2)

# ---------- (b) Exceedance by quartile ----------
quartiles = ["1", "2", "3", "4"]
q_labels = ["Q1\n(low EJ)", "Q2", "Q3", "Q4\n(high EJ)"]
exceed_vals = [ej_exceed.get(q, 0) * 100 for q in quartiles]
mean_vals = [ej_mean.get(q, 0) for q in quartiles]

cmap_q = ["#4477AA", "#66CCEE", "#EE6677", "#AA3377"]
x = np.arange(len(quartiles))
bars = ax2.bar(x, exceed_vals, color=cmap_q, edgecolor="white", linewidth=0.4)
for bx, v, mv in zip(bars, exceed_vals, mean_vals):
    ax2.text(bx.get_x() + bx.get_width() / 2, v + 1,
             f"{v:.1f}%", ha="center", va="bottom", fontsize=7,
             fontweight="bold", color="#222")
    ax2.text(bx.get_x() + bx.get_width() / 2, -6,
             f"{mv:.2f}", ha="center", va="top", fontsize=6, color="#555")

ax2.set_xticks(x)
ax2.set_xticklabels(q_labels, fontsize=7)
ax2.set_ylabel("Tracts above EPA standard (%)")
ax2.set_ylim(-12, max(exceed_vals) * 1.25 if max(exceed_vals) > 0 else 100)
ax2.set_title("(b) Exceedance of EPA standard by EJ quartile", loc="left")
ax2.grid(axis="y", alpha=0.4)
ax2.text(0, -10, "mean µg/m$^3$:", fontsize=5.5, color="#555", ha="center")

plt.tight_layout()
out = HERE / "fig5_ej_analysis.pdf"
plt.savefig(out)
plt.close()
print(f"Saved {out}")
