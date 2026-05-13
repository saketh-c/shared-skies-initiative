"""Figure 1: Study area and sensor distribution across four Texas cities.

Two-panel horizontal figure:
 - Left: Texas state outline + sensor points colored by city.
 - Right: Per-city histogram of sensor counts & mean PM2.5.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import apply_style, CITY_COLORS

apply_style()

HERE = Path(__file__).parent
RESULTS = json.loads((HERE / "results.json").read_text())
meta = RESULTS["sensor_meta"]
stats = RESULTS["summary_stats"]

fig, (ax_map, ax_bar) = plt.subplots(1, 2, figsize=(7.0, 2.8),
                                    gridspec_kw={"width_ratios": [1.55, 1.0]})

# ---------- LEFT: sensor map ----------
# Simple Texas bbox; no basemap dependency.
TX_BBOX = (-107.0, -93.0, 25.5, 36.8)  # (lon_min, lon_max, lat_min, lat_max)
for s in meta:
    city = s["city"]
    ax_map.scatter(s["lon"], s["lat"], s=14,
                   color=CITY_COLORS.get(city, "#888"),
                   edgecolors="white", linewidths=0.3,
                   alpha=0.85, zorder=3)

# Texas outline — crude but sufficient (straight-line approximation).
tx_outline = [
    (-106.65, 31.75), (-106.50, 31.75), (-104.53, 30.64), (-104.53, 29.39),
    (-103.11, 28.97), (-102.82, 29.78), (-101.25, 29.64), (-100.00, 28.15),
    (-99.10, 26.42), (-97.36, 25.84), (-96.48, 28.28), (-94.70, 29.64),
    (-93.85, 29.71), (-93.90, 31.00), (-94.04, 33.55), (-99.98, 34.00),
    (-100.00, 36.50), (-103.04, 36.50), (-103.04, 32.00), (-106.65, 31.75),
]
xs = [p[0] for p in tx_outline] + [tx_outline[0][0]]
ys = [p[1] for p in tx_outline] + [tx_outline[0][1]]
ax_map.plot(xs, ys, color="#444", linewidth=0.9, zorder=1)
ax_map.fill(xs, ys, color="#F5F5F5", zorder=0)

ax_map.set_xlim(TX_BBOX[0], TX_BBOX[1])
ax_map.set_ylim(TX_BBOX[2], TX_BBOX[3])
ax_map.set_aspect("equal")
ax_map.set_xlabel("Longitude")
ax_map.set_ylabel("Latitude")
ax_map.set_title("(a) Sensor network across Texas (n = 240)", loc="left")
ax_map.grid(False)

# Legend
for city, color in CITY_COLORS.items():
    ax_map.scatter([], [], s=22, color=color, edgecolors="white",
                   linewidths=0.3, label=city)
ax_map.legend(loc="lower left", ncol=1, fontsize=6.5,
              title="Metro", title_fontsize=7, handletextpad=0.4,
              borderaxespad=0.2)

# ---------- RIGHT: sensor + record counts per city ----------
cities = list(CITY_COLORS.keys())
sensor_counts = [stats["city_sensor_counts"].get(c, 0) for c in cities]
record_counts = [stats["city_counts"].get(c, 0) for c in cities]

y = np.arange(len(cities))
w = 0.38
colors = [CITY_COLORS[c] for c in cities]

ax_bar.barh(y - w / 2, sensor_counts, height=w, color=colors,
            edgecolor="white", linewidth=0.4, label="Sensors")
# Records on twin axis
ax2 = ax_bar.twiny()
ax2.barh(y + w / 2, record_counts, height=w, color=colors,
         alpha=0.45, edgecolor="white", linewidth=0.4, label="Sensor-days")

ax_bar.set_yticks(y)
ax_bar.set_yticklabels(cities, fontsize=7)
ax_bar.invert_yaxis()
ax_bar.set_xlabel("Sensor count", fontsize=7)
ax2.set_xlabel("Sensor-day records", fontsize=7)
ax_bar.set_title("(b) Coverage per metro", loc="left")
ax_bar.grid(axis="x", alpha=0.4)

# Annotate
for yi, sc, rc in zip(y, sensor_counts, record_counts):
    ax_bar.text(sc + 2, yi - w / 2, f"{sc}", va="center", fontsize=6.5)
    ax2.text(rc + 200, yi + w / 2, f"{rc:,}", va="center", fontsize=6.5)

plt.tight_layout()
out_path = HERE / "fig1_study_area.pdf"
plt.savefig(out_path)
plt.close()
print(f"Saved {out_path}")


def mpl_lighten(hex_color, amount=0.3):
    """Lighten an RGB hex color towards white."""
    import matplotlib.colors as mcolors
    c = np.array(mcolors.to_rgb(hex_color))
    return tuple(c + (1 - c) * amount)
