"""Figure 2: Model comparison under Random 80/20 split vs. Leave-One-Sensor-Out CV.

This is the methodological headline figure: it shows the large performance gap
between location-leaked (random split) and location-honest (LOSO) evaluation,
and that tree-based ensembles are the most robust under spatial holdout.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import apply_style, MODEL_COLORS

apply_style()

HERE = Path(__file__).parent
R = json.loads((HERE / "results.json").read_text())
rs = R["random_split"]
lo = R["loso_summary"]

MODELS = ["MLR", "SVR", "RF", "LightGBM", "XGBoost", "Ensemble"]

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=False)

metrics = [("r2", "$R^2$", (0, 1.0)),
           ("rmse", "RMSE (µg/m$^3$)", None),
           ("mae", "MAE (µg/m$^3$)", None)]

for ax, (metric, label, ylim) in zip(axes, metrics):
    x = np.arange(len(MODELS))
    w = 0.38

    random_vals = []
    loso_vals = []
    for m in MODELS:
        random_vals.append(rs.get(m, {}).get(metric, np.nan))
        if m in lo:
            loso_vals.append(lo[m].get(f"{metric}_mean", np.nan))
        else:
            loso_vals.append(np.nan)

    random_vals = np.array(random_vals)
    loso_vals = np.array(loso_vals)

    bars_r = ax.bar(x - w / 2, random_vals, width=w,
                    color=["#bbb"] * len(MODELS),
                    edgecolor="white", linewidth=0.4, label="Random 80/20")
    bars_l = ax.bar(x + w / 2, loso_vals, width=w,
                    color=[MODEL_COLORS[m] for m in MODELS],
                    edgecolor="white", linewidth=0.4, label="LOSO-CV")

    # Value labels
    for bx, v in zip(bars_r, random_vals):
        if not np.isnan(v):
            ax.text(bx.get_x() + bx.get_width() / 2, v + (0.01 if metric == "r2" else 0.05),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=5.5, color="#555")
    for bx, v in zip(bars_l, loso_vals):
        if not np.isnan(v):
            ax.text(bx.get_x() + bx.get_width() / 2, v + (0.01 if metric == "r2" else 0.05),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=5.5, color="#222",
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=25, ha="right", fontsize=6.5)
    ax.set_ylabel(label)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(axis="y", alpha=0.4)

axes[0].set_title("(a) Coefficient of determination", loc="left")
axes[1].set_title("(b) Root-mean-square error", loc="left")
axes[2].set_title("(c) Mean absolute error", loc="left")

# Shared legend beneath
fig.legend(
    handles=[
        plt.Rectangle((0, 0), 1, 1, color="#bbb", label="Random 80/20 (location-leaked)"),
        plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS["XGBoost"], label="LOSO-CV (spatially honest)"),
    ],
    loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04),
    frameon=False, fontsize=7,
)

plt.tight_layout(rect=[0, 0.04, 1, 1])
out = HERE / "fig2_model_comparison.pdf"
plt.savefig(out)
plt.close()
print(f"Saved {out}")
