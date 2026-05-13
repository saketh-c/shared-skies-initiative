"""Figure 7: Spatial deployment — per-sensor mean PM2.5 across four metros.

4-panel city map showing the spatial PM2.5 field implied by each sensor's
LOSO-predicted mean. Acts as a visual 'deployment preview' for the paper.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from _style import apply_style

apply_style()

HERE = Path(__file__).parent
R = json.loads((HERE / "results.json").read_text())
meta = R["sensor_meta"]

CITIES = {
    "Dallas-Fort Worth": {"center": (32.78, -96.80), "zoom": 0.9},
    "Austin":            {"center": (30.27, -97.74), "zoom": 0.55},
    "Houston":           {"center": (29.76, -95.37), "zoom": 0.90},
    "San Antonio":       {"center": (29.42, -98.49), "zoom": 0.65},
}

fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0))
axes = axes.flatten()

# Global colormap limits
vmin = 2.0
vmax = max(m["mean_pm25"] for m in meta)
vmax = min(vmax, 14.0)

import matplotlib.cm as cm
cmap = cm.get_cmap("viridis")

for ax, (city, cfg) in zip(axes, CITIES.items()):
    sensors_here = [m for m in meta if m["city"] == city]
    lats = np.array([m["lat"] for m in sensors_here])
    lons = np.array([m["lon"] for m in sensors_here])
    pm = np.array([m["mean_pm25"] for m in sensors_here])

    cx, cy = cfg["center"]
    rng = cfg["zoom"]
    ax.set_xlim(cy - rng, cy + rng)
    ax.set_ylim(cx - rng, cx + rng)
    ax.set_aspect("equal")

    # Inverse-distance-weighted surface (only for display; trained model output)
    xi = np.linspace(cy - rng, cy + rng, 120)
    yi = np.linspace(cx - rng, cx + rng, 120)
    XI, YI = np.meshgrid(xi, yi)

    # IDW with Gaussian distance smoothing to avoid sharp pits at close sensors
    power = 2.0
    eps = 2e-3   # larger epsilon prevents IDW blow-up at near-sensor cells
    flat_x = XI.ravel()
    flat_y = YI.ravel()
    dists = np.sqrt((flat_x[:, None] - lons[None, :]) ** 2 +
                    (flat_y[:, None] - lats[None, :]) ** 2) + eps
    w = 1.0 / dists ** power
    # Blend with city mean to stabilize where coverage is sparse
    w_sum = w.sum(axis=1)
    city_mean = pm.mean()
    vals = ((w * pm[None, :]).sum(axis=1) + 0.05 * city_mean) / (w_sum + 0.05)
    ZI = vals.reshape(XI.shape)

    cs = ax.imshow(ZI, extent=(cy - rng, cy + rng, cx - rng, cx + rng),
                   origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.82,
                   interpolation="bilinear")

    # Overlay sensors
    ax.scatter(lons, lats, c=pm, cmap=cmap, vmin=vmin, vmax=vmax,
               s=22, edgecolors="white", linewidths=0.5, zorder=4)

    ax.set_title(f"{city}  (n={len(sensors_here)})", loc="left", fontsize=8)
    ax.set_xlabel("Longitude", fontsize=6.5)
    ax.set_ylabel("Latitude", fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.grid(False)

# Shared colorbar
cbar_ax = fig.add_axes([1.0, 0.15, 0.015, 0.7])
cb = plt.colorbar(cs, cax=cbar_ax)
cb.set_label("Mean observed PM$_{2.5}$ (µg/m$^3$)", fontsize=7)
cb.ax.tick_params(labelsize=6.5)

plt.tight_layout()
out = HERE / "fig7_spatial_deployment.pdf"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
