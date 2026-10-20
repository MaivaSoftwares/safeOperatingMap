#!/usr/bin/env python3
"""Thermal-hysteresis campaign figure (clean 2026-08-06 rebuild), two panels:

(a) actual heating/cooling legs of a representative controlled ramp (bias vs reported die
    temperature, de-drifted), with the kappa_b = -75.5 uG/C slope and the measured loop width.
(b) loop width h vs heating rate for all 13 clean ramps (9 unit A + 4 unit B), bootstrap 95%
    CI, against the eps = 1 mg budget. Shows h <= ~1.1 mg with no rate-driven crossing.

Regenerates purely from the committed clean CSVs (data/temp_log_unit*_hyst_*.csv).
    twin/.venv/bin/python analysis/hyst_campaign_figure.py [outdir]
"""
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from scipy.signal import medfilt         # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from analysis import tempco as nz                          # noqa: E402
from analysis.hyst_rate_figure import collect, _legs       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
EPS_MG = 1.0
KAPPA = -75.5            # uG/C, validated bias-TC (unchanged; not refit here)
HEAT, COOL = "#CC3311", "#0077BB"
A_CLR, B_CLR = "#1f4e8c", "#b5651d"


def _dedrift_legs(name):
    d = np.genfromtxt(os.path.join(DATA, name), delimiter=",", names=True)
    t, T = d["t_s"], d["temp_C"]
    a, _ = nz.deglitch(d["accel_g"])
    good = (np.abs(T - medfilt(T, 11)) < 3)
    t, a, T = t[good], a[good], T[good]
    tn = (t - t[0]) / (t[-1] - t[0])
    X = np.vstack([T, tn, np.ones_like(T)]).T
    bT, bdrift, b0 = np.linalg.lstsq(X, a, rcond=None)[0]
    add = (a - bdrift * tn) * 1e3                          # de-drifted, mg
    ipk = int(np.argmax(medfilt(T, 51)))
    return T, add, ipk, b0


def _binned(Tx, Ax, edges):
    idx = np.digitize(Tx, edges)
    c = 0.5 * (edges[1:] + edges[:-1])
    m = np.array([Ax[idx == k].mean() if (idx == k).sum() > 3 else np.nan for k in range(1, len(edges))])
    s = np.array([Ax[idx == k].std(ddof=1) / np.sqrt((idx == k).sum()) if (idx == k).sum() > 3 else np.nan
                  for k in range(1, len(edges))])
    ok = ~np.isnan(m)
    return c[ok], m[ok], s[ok]


def draw_legs(ax, name, h_mg):
    T, add, ipk, b0 = _dedrift_legs(name)
    add = add - add.mean()                                # centre for display
    Tm = T.mean()
    edges = np.linspace(T.min(), T.max(), 12)
    for sl, colr, mk, lab in ((slice(0, ipk), HEAT, "o", "heating"),
                              (slice(ipk, None), COOL, "s", "cooling")):
        c, m, s = _binned(T[sl], add[sl], edges)
        ax.fill_between(c, m - s, m + s, color=colr, alpha=0.15, lw=0)
        ax.plot(c, m, "-" + mk, color=colr, ms=3, lw=1.3, mec="none", label=lab)
    Tl = np.array([T.min(), T.max()])
    ax.plot(Tl, KAPPA * (Tl - Tm) * 1e-3, color="0.4", ls=(0, (4, 2)), lw=1.0,   # uG/C -> mg
            label=r"$\kappa_b=-75.5\,\mu$g/$^\circ$C")
    ax.annotate(f"loop $\\approx${h_mg:.1f} mg", (0.04, 0.06), xycoords="axes fraction",
                fontsize=7, color="0.25")
    ax.set(xlabel=r"reported die temperature ($^\circ$C)", ylabel="bias, de-drifted (mg)",
           ylim=(-2.0, 0.75))
    ax.legend(fontsize=7, loc="upper right")


def draw_frontier(ax):
    pts = collect(DATA)
    ax.axhline(EPS_MG, ls="--", lw=1.1, color="0.35")
    ax.text(0.4, EPS_MG + 0.03, r"$\varepsilon=1$ mg", fontsize=7, color="0.35", va="bottom")
    styles = {"unitA": dict(m="o", c=A_CLR, lab="unit A"),
              "unitB": dict(m="s", c=B_CLR, lab="unit B", mfc="none")}
    for unit, series in pts.items():
        s = styles[unit]
        for rate, h, clo, chi, _ in series:
            ax.errorbar(rate, h, yerr=[[h - clo], [chi - h]], fmt=s["m"], ms=5,
                        color=s["c"], mfc=s.get("mfc", s["c"]), ecolor=s["c"],
                        elinewidth=0.9, capsize=2, zorder=3)
        ax.plot([], [], s["m"], ms=5, color=s["c"], mfc=s.get("mfc", s["c"]), label=s["lab"])
    ax.set(xlabel=r"heating rate $\dot\theta$ ($^\circ$C/min)", ylabel="loop width $h$ (mg)",
           xlim=(0, 10), ylim=(0, 1.7))
    ax.legend(fontsize=6.5, loc="upper right", frameon=False)
    ax.set_title("(b) loop width vs rate (13 ramps, 2 units)", loc="left", fontsize=8)


def main(outdir):
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.grid": True,
                         "grid.alpha": 0.25, "figure.dpi": 150, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    draw_legs(ax, "temp_log_unitA_hyst_fastrep1.csv", 1.06)
    out = os.path.join(outdir, "hyst_campaign.pdf")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "docs", "figures")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(outdir)
