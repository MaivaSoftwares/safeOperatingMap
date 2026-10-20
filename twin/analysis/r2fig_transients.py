"""N2 -- the 30 measured wake transients, raw, with the guard where physics puts it.
Regenerate: cd twin && .venv/bin/python -m analysis.r2fig_transients
Output: docs/figures/recovery_raw.{pdf,png}
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
FIGDIR = REPO / "docs" / "figures"
RAW = REPO / "tools/data/r1_burst_repeatability_unitA_r1_burst_pd_n30_reprocessed_20260706_203023/raw"
PD, RED = "#D55E00", "#d1495b"
T_FV, TAU, B0, GUARD, GLO, GHI = 6.174, 1.449, 82.6, 12.8, 12.54, 13.05


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "figure.dpi": 150,
        "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.18,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
    })
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for f in sorted(RAW.glob("burst_pd_*.txt")):
        d = pd.read_csv(f, skiprows=2, on_bad_lines="skip")
        d = d.apply(pd.to_numeric, errors="coerce").dropna()
        d = d[(d.drdy == 1) & (d.err == 0) & (d.accel_g.abs() < 1.0)]
        ped = d.accel_g.tail(len(d) // 4).median()
        t = d.t_us * 1e-3
        b = (d.accel_g - ped).abs() * 1e3
        m = t <= 24
        ax.plot(t[m], b[m], color=PD, lw=0.4, alpha=0.18, zorder=2)
    t = np.linspace(T_FV, 24, 300)
    ax.plot(t, B0 * np.exp(-(t - T_FV) / TAU), color="0.15", lw=1.2, zorder=4,
            label="median fit")
    ax.axvspan(0, T_FV, color="0.6", alpha=0.18, lw=0)
    ax.annotate("blind\nwindow", (1.4, 8), fontsize=6.4, color="0.35")
    ax.axhline(1.0, ls="--", color=RED, lw=0.9)
    ax.annotate(r"$\varepsilon$ = 1 mg", (17.5, 1.25), fontsize=6.4, color=RED)
    ax.axvspan(GLO, GHI, color="0.35", alpha=0.30, lw=0)
    ax.annotate("$T_{guard}$", (GUARD, 32), fontsize=6.4, color="0.25", ha="center")
    ax.set_xlim(0, 24)
    ax.set_yscale("log")
    ax.set_ylim(0.15, 120)
    ax.set_xlabel("time since wake command (ms)", fontsize=7.5, labelpad=1.5)
    ax.set_ylabel("post-wake |bias| (mg)", fontsize=7.5)
    ax.legend(loc="upper right", fontsize=6.4, framealpha=0.9)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"recovery_raw.{ext}")
    plt.close(fig)
    print("  recovery_raw     <- 30 raw bursts")


if __name__ == "__main__":
    main()
