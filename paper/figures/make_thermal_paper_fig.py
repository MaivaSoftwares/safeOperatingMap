"""Paper-1 combined thermal figure (TIM page budget): two stacked panels.

(a) replication summary -- observed collapse temperatures for 4+4 ramps plus
    the cross-coefficient control, against each unit's pre-registered
    prediction (reuses make_thermal_replication_fig's panel).
(b) guard growth vs die temperature for the original unit-A ramp with the
    C2-theta prediction and the observed collapse (single-panel form of the
    thesis fig_hil_thermal, twin/analysis/figures.py).

Regenerate: twin/.venv/bin/python paper/figures/make_thermal_paper_fig.py
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_thermal_replication_fig import DATA, draw_replication_panel

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 150,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25,
})

BIAS_TC, EPS_ETA = 0.0755, 0.85          # mg/C, (1-delta)*eps in mg
T_COL = 25.0 + EPS_ETA / BIAS_TC         # predicted collapse: 36.26 C
CLR = "#0072B2"
CLR_OBS = "#D55E00"


def load_ramp():
    # Pinned to the original M7 ramp (2026-07-03), same glob caveat as the
    # thesis figure: unit-A annotations require the unit-A log.
    path = sorted(DATA.glob("ctl_thermal_20260703_*.log"))[-1]
    guard, tsleep, b0, tau, temp = [], [], [], [], []
    for ln in path.read_text().splitlines():
        if ln.startswith("CTL,"):
            f = ln.split(",")
            if len(f) == 16:
                try:
                    tsleep.append(float(f[4])); guard.append(float(f[5]))
                    b0.append(float(f[8])); tau.append(float(f[9]))
                    temp.append(float(f[15]))
                except ValueError:
                    pass
    return (np.asarray(guard), np.asarray(tsleep), np.asarray(b0),
            np.asarray(tau), np.asarray(temp), path.name)


def draw_guard_panel(ax):
    guard, tsleep, b0, tau, temp, src = load_ramp()
    green = (tsleep > 0) & (guard > 0)
    box = dict(fc="white", ec="none", alpha=0.85, pad=1.5)
    ax.plot(temp[green], guard[green], ".", color=CLR, ms=2.2, alpha=0.5,
            label="committed cycles")
    tt = np.linspace(20, T_COL - 0.02, 300)
    b0m, taum = np.median(b0[green]), np.median(tau[green])
    ax.plot(tt, taum * np.log(b0m / (EPS_ETA - BIAS_TC * np.abs(tt - 25.0))),
            ls="--", color="k", lw=1, label=r"C2-$\theta$ at median $\hat b_0,\hat\tau$")
    ax.axvline(T_COL, ls=":", color="k", lw=1)
    t_obs = float(temp[np.flatnonzero(~green)[0]])
    ax.axvline(t_obs, ls="-", color=CLR_OBS, lw=1)
    ax.annotate(f"predicted {T_COL:.2f}$\\,^\\circ$C", (T_COL - 0.5, 11.0),
                fontsize=6.5, color="0.25", ha="right", bbox=box)
    ax.annotate(f"observed {t_obs:.2f}$\\,^\\circ$C", (t_obs - 0.5, 22.4),
                fontsize=6.5, color=CLR_OBS, ha="right", bbox=box)
    ax.set(xlabel=r"die temp ($^\circ$C)", ylabel=r"$T_{guard}$ (ms)",
           xlim=(20, 40), ylim=(10, 29))   # top headroom so "observed" clears the frame
    ax.legend(loc="upper left", fontsize=6.5)
    print("guard panel <-", src)


fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.5, 3.55))
draw_replication_panel(ax_a)
draw_guard_panel(ax_b)
for ax, tag in [(ax_a, "(a)"), (ax_b, "(b)")]:
    ax.set_title(tag, loc="left", fontsize=8)
fig.subplots_adjust(hspace=0.48)

out = Path(__file__).parent / "thermal_paper.pdf"
fig.savefig(out)
print("wrote", out)
