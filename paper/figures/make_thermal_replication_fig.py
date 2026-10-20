"""Thermal replication figure: 4+4 ramps, per-unit pre-registered predictions.

Collapse temperatures are extracted directly from the committed controller
logs (tools/data/ctl_thermal_*.log), so the figure regenerates from raw data.
Regenerate: twin/.venv/bin/python paper/figures/make_thermal_replication_fig.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "tools" / "data"

def collapse_temp(path):
    t, ts, g, T = [], [], [], []
    for ln in open(path):
        if ln.startswith("CTL,"):
            f = ln.strip().split(",")
            if len(f) == 16:
                try:
                    ts.append(float(f[4])); g.append(float(f[5])); T.append(float(f[15]))
                except ValueError:
                    pass
    ts, g, T = np.asarray(ts), np.asarray(g), np.asarray(T)
    green = (ts > 0) & (g > 0)
    idx = np.where(~green)[0]
    eps = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    return float(T[max(eps, key=len)[0]])

SERIES = {
    "A (own $\\kappa$)": (36.26, "#0072B2", "o",
        ["ctl_thermal_20260703_171730.log", "ctl_thermal_unitA_r1_20260719_155335.log",
         "ctl_thermal_unitA_r2_20260719_162612.log", "ctl_thermal_unitA_r3_20260719_165856.log"]),
    "B (own $\\kappa$)": (37.34, "#009E73", "o",
        ["ctl_thermal_unitB_r3_20260719_203827.log", "ctl_thermal_unitB_r4_20260719_210421.log",
         "ctl_thermal_unitB_r5_20260719_212957.log", "ctl_thermal_unitB_r6_20260719_215521.log"]),
    "B, A's $\\kappa$\n(control)": (36.26, "#CC79A7", "s",
        ["ctl_thermal_unitB_r1_20260719_180854.log", "ctl_thermal_unitB_r2_20260719_185957.log"]),
}

def draw_replication_panel(ax):
    """Replication summary: observed collapse temps vs per-unit pre-registered predictions."""
    for i, (label, (pred, color, marker, logs)) in enumerate(SERIES.items()):
        obs = [collapse_temp(DATA / L) for L in logs]
        print(label, "->", [f"{v:.2f}" for v in obs])
        x = np.full(len(obs), i) + np.linspace(-0.12, 0.12, len(obs))
        ax.plot(x, obs, marker, color=color, ms=5, mec="white", mew=0.6, zorder=3)
        ax.hlines(pred, i - 0.28, i + 0.28, color=color, ls=(0, (4, 2)), lw=1.2)
        # dashed line = prediction (stated in the caption); label just the value,
        # centered on the line, so it never crowds the frame
        ax.annotate(f"{pred:.2f}", (i + 0.30, pred), fontsize=6.5, color=color,
                    va="center", ha="left")
    ax.set_xticks(range(len(SERIES)))
    ax.set_xticklabels(list(SERIES.keys()), fontsize=7)
    ax.set_ylabel("observed collapse ($^\\circ$C)")
    ax.set_xlim(-0.5, 3.05)   # right room so the last category's value label clears the frame


def main():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "figure.dpi": 150,
        "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25,
    })
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    draw_replication_panel(ax)
    out = Path(__file__).parent / "thermal_replication.pdf"
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
