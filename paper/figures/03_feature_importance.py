"""Figure 3: Feature importance for the LightGBM model."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import apply_style

apply_style()

HERE = Path(__file__).parent
R = json.loads((HERE / "results.json").read_text())
fi = R["feature_importance"]

# Group features by category for color coding
CATEGORY_COLORS = {
    "temporal": "#4477AA",      # weather / time
    "meteorological": "#66CCEE",
    "pm25_history": "#EE6677",  # lag features
    "socio": "#228833",         # EJ indicators
    "spatial": "#CCBB44",       # lat/lon
}
CAT_OF = {
    "humidity": "meteorological",
    "temperature": "meteorological",
    "pressure": "meteorological",
    "ejf_score": "socio",
    "pct_people_of_color": "socio",
    "pct_low_income": "socio",
    "traffic_proximity": "socio",
    "superfund_proximity": "socio",
    "rmp_proximity": "socio",
    "diesel_pm_proximity": "socio",
    "pct_ling_isolated": "socio",
    "latitude": "spatial",
    "longitude": "spatial",
    "month": "temporal",
    "dow": "temporal",
    "day_of_year": "temporal",
    "pm25_lag1": "pm25_history",
    "pm25_lag7": "pm25_history",
    "pm25_roll7": "pm25_history",
}
PRETTY = {
    "humidity": "Humidity",
    "temperature": "Temperature",
    "pressure": "Pressure",
    "ejf_score": "EJScreen EJ Index",
    "pct_people_of_color": "% People of color",
    "pct_low_income": "% Low income",
    "traffic_proximity": "Traffic proximity",
    "superfund_proximity": "Superfund proximity",
    "rmp_proximity": "RMP proximity",
    "diesel_pm_proximity": "Diesel PM proximity",
    "pct_ling_isolated": "% Ling. isolated",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "month": "Month",
    "dow": "Day of week",
    "day_of_year": "Day of year",
    "pm25_lag1": "PM$_{2.5}$ lag (1 day)",
    "pm25_lag7": "PM$_{2.5}$ lag (7 day)",
    "pm25_roll7": "PM$_{2.5}$ rolling mean",
}
CAT_LABELS = {
    "pm25_history": "PM$_{2.5}$ autoregressive",
    "temporal": "Temporal",
    "meteorological": "Meteorological",
    "socio": "Socio/Env. justice",
    "spatial": "Spatial",
}

# Sort ascending for horizontal bars
fi_sorted = sorted(fi, key=lambda d: d["importance"])
names = [PRETTY.get(d["feature"], d["feature"]) for d in fi_sorted]
vals = [d["importance"] for d in fi_sorted]
cats = [CAT_OF.get(d["feature"], "temporal") for d in fi_sorted]
colors = [CATEGORY_COLORS[c] for c in cats]

fig, ax = plt.subplots(1, 1, figsize=(4.3, 3.6))
y = np.arange(len(names))
bars = ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.35)
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=7)
ax.set_xlabel("LightGBM gain importance")
ax.set_title("Feature importance (full-data fit)", loc="left")

total = sum(vals)
for bx, v in zip(bars, vals):
    ax.text(bx.get_width() + total * 0.007, bx.get_y() + bx.get_height() / 2,
            f"{v / total * 100:.1f}%", va="center", fontsize=6.2, color="#555")

# Category legend
from matplotlib.patches import Patch
handles = [Patch(facecolor=CATEGORY_COLORS[k], label=CAT_LABELS[k]) for k in
           ["pm25_history", "meteorological", "temporal", "socio", "spatial"]]
ax.legend(handles=handles, loc="lower right", fontsize=6.5,
          title="Feature group", title_fontsize=7)

ax.grid(axis="x", alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
out = HERE / "fig3_feature_importance.pdf"
plt.savefig(out)
plt.close()
print(f"Saved {out}")
