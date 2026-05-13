"""Figure 6: Fairness-stratified error analysis.

For each EJ quartile (Q1..Q4), reports:
 - LOSO-CV RMSE
 - LOSO-CV MAE
 - Signed bias (predicted - observed)

Reveals whether the model performs consistently across communities.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _style import apply_style

apply_style()

HERE = Path(__file__).parent
R = json.loads((HERE / "results.json").read_text())

df = pd.DataFrame(R["per_sensor_loso"])
df = df.dropna(subset=["ejf_score"])
df["ej_quartile"] = pd.qcut(df["ejf_score"], q=4, labels=[1, 2, 3, 4],
                            duplicates="drop")

agg = df.groupby("ej_quartile", observed=True).agg(
    rmse_mean=("rmse", lambda s: float(np.average(s, weights=df.loc[s.index, "n"]))),
    rmse_sem=("rmse", lambda s: float(s.std() / np.sqrt(len(s)))),
    mae_mean=("mae", lambda s: float(np.average(s, weights=df.loc[s.index, "n"]))),
    mae_sem=("mae", lambda s: float(s.std() / np.sqrt(len(s)))),
    bias_mean=("bias", lambda s: float(np.average(s, weights=df.loc[s.index, "n"]))),
    bias_sem=("bias", lambda s: float(s.std() / np.sqrt(len(s)))),
    n=("n", "sum"),
).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharex=True)

cmap_q = ["#4477AA", "#66CCEE", "#EE6677", "#AA3377"]
x = np.arange(len(agg))
x_labels = [f"Q{int(q)}" for q in agg["ej_quartile"]]

# RMSE
ax = axes[0]
ax.bar(x, agg["rmse_mean"], yerr=agg["rmse_sem"], color=cmap_q,
       edgecolor="white", linewidth=0.4, capsize=3, error_kw=dict(lw=0.6))
for xi, v in zip(x, agg["rmse_mean"]):
    ax.text(xi, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.set_ylabel("RMSE (µg/m$^3$)")
ax.set_title("(a) Prediction RMSE", loc="left")

# MAE
ax = axes[1]
ax.bar(x, agg["mae_mean"], yerr=agg["mae_sem"], color=cmap_q,
       edgecolor="white", linewidth=0.4, capsize=3, error_kw=dict(lw=0.6))
for xi, v in zip(x, agg["mae_mean"]):
    ax.text(xi, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.set_ylabel("MAE (µg/m$^3$)")
ax.set_title("(b) Prediction MAE", loc="left")

# Bias (signed)
ax = axes[2]
bars = ax.bar(x, agg["bias_mean"], yerr=agg["bias_sem"], color=cmap_q,
              edgecolor="white", linewidth=0.4, capsize=3, error_kw=dict(lw=0.6))
ax.axhline(0, color="#444", linewidth=0.5)
for xi, v in zip(x, agg["bias_mean"]):
    ax.text(xi, v + (0.04 if v >= 0 else -0.04), f"{v:+.2f}",
            ha="center", va="bottom" if v >= 0 else "top", fontsize=6.5)
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.set_ylabel("Signed bias (µg/m$^3$)")
ax.set_title("(c) Directional bias (pred $-$ obs)", loc="left")

# Shared x-label
fig.text(0.5, -0.02, "EJScreen EJ-index quartile (Q1 = low, Q4 = high)",
         ha="center", fontsize=7.5)

plt.tight_layout()
out = HERE / "fig6_fairness_error.pdf"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
