"""Shared publication style for all paper figures.

Loaded by each figure script via `from _style import apply_style`.
- Serif typography to match the NeurIPS body text.
- 300 DPI vector PDF output.
- Colourblind-safe 8-color palette (Paul Tol 'bright').
- Thin axis lines, no top/right spines, neutral grid.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Paul Tol 'bright' palette — colourblind-safe, prints well in BW too.
TOL_BRIGHT = [
    "#4477AA",  # blue
    "#EE6677",  # red
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#BBBBBB",  # grey
    "#000000",  # black
]

MODEL_COLORS = {
    "MLR":       "#BBBBBB",
    "SVR":       "#66CCEE",
    "RF":        "#228833",
    "LightGBM":  "#CCBB44",
    "XGBoost":   "#EE6677",
    "Ensemble":  "#4477AA",
    "LSTM":      "#AA3377",
}

CITY_COLORS = {
    "Dallas-Fort Worth": "#EE6677",
    "Austin":            "#228833",
    "Houston":           "#4477AA",
    "San Antonio":       "#CCBB44",
}


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # TrueType for embedded fonts
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.labelsize": 8,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.4,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "lines.linewidth": 1.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    mpl.rcParams["axes.prop_cycle"] = plt.cycler(color=TOL_BRIGHT)
