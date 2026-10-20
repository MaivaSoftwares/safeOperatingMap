#!/usr/bin/env python3
"""Hysteresis loop-width h vs heating ramp rate — the r_max frontier figure.

Reads every saved unit-A / unit-B hysteresis capture (data/temp_log_unit*_hyst_*.csv),
recomputes h and its bootstrap 95% CI + the band-slope heating rate with the SAME code the
bench uses (analysis.tempco.loop_width_mg), and plots h(rate) with the ε = 1 mg C2 budget
line. r_max is the rate at which h crosses the budget; the crossing is bracketed by the data.

Reproducible: regenerates purely from the committed CSVs. See
docs/hysteresis_rate_experiment_plan_2026-08-03.md §10.

    twin/.venv/bin/python analysis/hyst_rate_figure.py
"""
import glob
import os
import re
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # twin/ on path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from scipy.signal import medfilt         # noqa: E402

from analysis import tempco as tc                        # noqa: E402
from analysis.tempco import _reject_temp_spikes, deglitch  # noqa: E402

EPS_MG = 1.0                              # C2 wake budget ε (mg)
EXCLUDE = ("overheat",)                   # discarded captures (see plan §10)
_RNG = np.random.default_rng(0)


def _legs(t, a, T):
    """Reproduce loop_width_mg's cleaning + rescue split, returning the up/down legs."""
    good = ~_reject_temp_spikes(T)
    a = deglitch(a)[0]
    t, a, T = t[good], a[good], T[good]
    k = max(3, min(51, (len(T) // 10) * 2 + 1))
    ipk = int(np.argmax(medfilt(T, k)))
    if min(ipk, len(T) - ipk) / len(T) < 0.15:            # early-peak: trim the dead tail
        base = np.median(T[(t - t[0]) <= 15.0])
        back = np.where((np.arange(len(T)) > ipk) & (T <= base + 0.5))[0]
        if len(back):
            cut = int(back[0]) + max(5, len(T) // 100)
            t, a, T = t[:cut], a[:cut], T[:cut]
            k = max(3, min(51, (len(T) // 10) * 2 + 1))
            ipk = int(np.argmax(medfilt(T, k)))
    Tu, au, tu = T[:ipk], a[:ipk], t[:ipk]
    Td, ad, td = T[ipk:], a[ipk:], t[ipk:]
    lo, hi = max(Tu.min(), Td.min()), min(Tu.max(), Td.max())
    return Tu, au, tu, Td, ad, td, lo, hi


def _h_of(Tu, au, tu, Td, ad, td, lo, hi):
    grid = np.linspace(lo, hi, 200)
    b_up = np.polyval(np.polyfit(Tu, au, 2), grid)
    b_dn = np.polyval(np.polyfit(Td, ad, 2), grid)
    diff = b_dn - b_up
    su, sd = np.argsort(Tu), np.argsort(Td)
    dgap = np.interp(grid, Td[sd], td[sd]) - np.interp(grid, Tu[su], tu[su])
    ds = np.linalg.lstsq(np.vstack([dgap, np.ones_like(dgap)]).T, diff, rcond=None)[0][0]
    return float(np.max(np.abs(diff - ds * dgap))) * 1e3


def _boot_ci(legs, n=400):
    Tu, au, tu, Td, ad, td, lo, hi = legs
    out = []
    for _ in range(n):
        iu = _RNG.integers(0, len(Tu), len(Tu))
        idn = _RNG.integers(0, len(Td), len(Td))
        out.append(_h_of(Tu[iu], au[iu], tu[iu], Td[idn], ad[idn], td[idn], lo, hi))
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


def collect(data_dir="data"):
    pts = {"unitA": [], "unitB": []}
    for path in sorted(glob.glob(os.path.join(data_dir, "temp_log_unit*_hyst_*.csv"))):
        name = os.path.basename(path)
        if any(x in name for x in EXCLUDE):
            continue
        unit = "unitA" if "unitA" in name else "unitB"
        d = np.genfromtxt(path, delimiter=",", names=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = tc.loop_width_mg(d["t_s"], d["accel_g"], d["temp_C"], warmup_s=0.0)
        if not np.isfinite(r["h_mg"]) or r["monotonic_unreliable"]:
            print(f"  skip {name}: h unresolved")
            continue
        legs = _legs(d["t_s"], d["accel_g"], d["temp_C"])
        clo, chi = _boot_ci(legs)
        label = re.sub(r"^temp_log_|\.csv$", "", name)
        pts[unit].append((r["rate_up_C_per_min"], r["h_mg"], clo, chi, label))
        print(f"  {label:22s} rate={r['rate_up_C_per_min']:.2f}  h={r['h_mg']:.2f}  CI[{clo:.2f},{chi:.2f}]")
    return pts


def make_figure(out="../../docs/figures/hyst_rate.pdf", title=True):
    pts = collect()
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.axhline(EPS_MG, ls="--", lw=1.2, color="0.35", zorder=1)
    ax.text(0.05, EPS_MG + 0.03, r"$\varepsilon = 1$ mg  (C2 wake budget)",
            fontsize=8, color="0.35", va="bottom")

    # r_max is bracketed: highest measured-safe rate .. first breach
    safe = [p[0] for u in pts.values() for p in u if p[3] < EPS_MG]
    breach = [p[0] for u in pts.values() for p in u if p[1] > EPS_MG]
    if safe and breach:
        lo_r, hi_r = max(safe), min(breach)
        ax.axvspan(lo_r, hi_r, color="orange", alpha=0.12, zorder=0)
        ax.text((lo_r + hi_r) / 2, 2.25, r"$r_{\max}$", ha="center", fontsize=10, color="darkorange")
        ax.annotate("", xy=(hi_r, 2.1), xytext=(lo_r, 2.1),
                    arrowprops=dict(arrowstyle="<->", color="darkorange", lw=1.2))

    styles = {"unitA": dict(marker="o", color="#1f4e8c", label="unit A"),
              "unitB": dict(marker="s", color="#b5651d", label="unit B", mfc="none")}
    for unit, series in pts.items():
        if not series:
            continue
        s = styles[unit]
        for rate, h, clo, chi, _ in series:
            ax.errorbar(rate, h, yerr=[[h - clo], [chi - h]], fmt=s["marker"],
                        ms=6, color=s["color"], mfc=s.get("mfc", s["color"]),
                        ecolor=s["color"], elinewidth=1, capsize=2.5, zorder=3)
        # a labelled proxy for the legend
        ax.plot([], [], s["marker"], ms=6, color=s["color"],
                mfc=s.get("mfc", s["color"]), label=s["label"])

    ax.set_xlabel(r"heating ramp rate $\dot\theta_{\uparrow}$  (°C/min)")
    ax.set_ylabel(r"loop-width hysteresis $h$  (mg)")
    ax.set_xlim(0, 3.7)
    ax.set_ylim(0, 2.5)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    if title:                                # standalone/docs view; the paper caption carries this
        ax.set_title("Thermal-lag hysteresis vs ramp rate (bars: 95% bootstrap CI)", fontsize=9)
    fig.tight_layout()
    here = os.path.dirname(os.path.abspath(__file__))
    dest = os.path.normpath(os.path.join(here, out))
    fig.savefig(dest)
    fig.savefig(dest.replace(".pdf", ".png"), dpi=150)
    print(f"wrote {dest}")


if __name__ == "__main__":
    make_figure()                                            # docs/ standalone (titled)
