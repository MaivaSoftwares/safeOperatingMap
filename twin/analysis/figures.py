"""Publication figures for the safe-operating-map thesis chapter — regenerate every figure from
COMMITTED data + the firmware C (libcore.so), one command, no screenshots:

    cd twin && .venv/bin/python -m analysis.figures        # -> docs/figures/*.pdf (+ .png)

Sources (printed in the manifest): measured HW captures in data/*.csv, the exact firmware map
engine via the cffi binding, and the two measured sleep modes (rail_off, pd). Deterministic:
no wall-clock / RNG. Tier 1 (this file) needs no simulator; the S2 boundary/timeline figures are
Tier 2 (added once the chirp simulator exists).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from scipy.signal import medfilt, welch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # twin/
from bindings import core_ffi as core    # noqa: E402
from analysis import noise as nz          # noqa: E402
from analysis import temperature as tp    # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
FIGDIR = Path(__file__).resolve().parent.parent.parent / "docs" / "figures"

# ---- measured coefficients: R1 campaign medians, unit A 2026-07-06/08 (SENSOR_COEFFICIENTS.md).
# b0 is WAKE-REFERENCED (b0_fit * exp(t_fv/tau): 82.6 mg at 6.17 ms, tau 1.449 ms -> 5.81 g),
# so the C's tau*ln(b0/((1-d)eps)) returns the 12.8 ms total guard from the wake command.
PON, FS, TGUARD, TPROBE = 0.067, 175.0, 0.0128, 0.005
COMMON = dict(b0=5.81, tau=0.001449, sigma_pw=0.00159, SF=1.001, SF_ref=1.0,
              fs=FS, T_probe=TPROBE, a_max_exp=1.0, range_limit=30.0, bias_tc=-75.5e-6)
MODES = {                                    # (P_sleep W, E_inrush J)  R1 medians
    "rail_off": dict(P_sleep=2e-5,    E_inrush=0.227e-3),
    "pd":       dict(P_sleep=0.632e-3, E_inrush=0.070e-3),
}
COLOR = {"rail_off": "#0072B2", "pd": "#D55E00",           # Okabe-Ito, fixed across all figures
         "GREEN": "#2ca25f", "YELLOW": "#f4b642", "RED": "#d1495b"}


def _style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.titlesize": 8,
        "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "figure.dpi": 150, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.25, "lines.linewidth": 1.4,
    })


def _save(fig, name, src):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name:16s} <- {src}")


def _sensor(mode, **over):
    p = dict(P_on=PON, **COMMON, **MODES[mode]); p.update(over)
    return core.sensor(**p)


# ================================================================================================
T_FV = 0.006174   # first trustworthy sample after the wake command (R1, pd)


# C-7 (2026-10-17): the scope shunt is 1.00 ohm (meter 1.1 ohm minus 0.1 ohm leads; bands 10 x 10.0 ohm
# +/-2 %). Energies, peaks and settled currents re-extracted at 1.00 ohm (r1_inrush_repeatability_*_shunt1p00_*).
SCOPE_SHUNT_OHM = 1.00
# C-7b (2026-10-23): energies integrated to electrical settling T_wake (unit A means of 10).
_INRUSH = {"rail_off": ("0.246 mJ", 81, 13.4), "pd": ("0.071 mJ", 51, 13.9)}


def _load_inrush(mode):
    """Representative R1 scope capture (Siglent CSV: header rows, then index,V)."""
    root = Path(__file__).resolve().parent.parent.parent / "tools" / "data"
    path = root / f"{mode}_scope" / "SDS814X_HD_CSV_C1_1.csv"
    d = np.genfromtxt(path, delimiter=",", skip_header=12, usecols=(0, 1))
    v = d[:, 1]
    dt_us = 0.2                                        # 200 ns sample interval
    # wake edge = end of the longest sustained low (sleep) segment before the surge
    k = max(1, int(20 / dt_us))                        # 20 us smoothing
    sm = np.convolve(v, np.ones(k) / k, mode="same")
    lo = sm < (np.percentile(sm, 10) + 0.15 * (sm.max() - np.percentile(sm, 10)))
    edges = np.flatnonzero(np.diff(np.r_[0, lo.astype(int), 0]))
    runs = list(zip(edges[::2], edges[1::2]))
    i0 = max(runs, key=lambda r: r[1] - r[0])[1] if runs else int(np.argmax(v))
    t = (np.arange(len(v)) - i0) * dt_us * 1e-3
    return t, v / SCOPE_SHUNT_OHM * 1e3                # ms (wake edge at 0), mA


def fig_inrush():
    """Both measured inrush surges (scope + 1.00 Ohm shunt, 5 MSa/s) — rail_off vs pd."""
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for m, (e, pk, _) in _INRUSH.items():
        t, i_mA = _load_inrush(m)
        ax.plot(t, i_mA, color=COLOR[m], lw=0.7, label=f"{m}: {e}, {pk} mA")
    ax.set(xlabel="time from wake (ms)", ylabel="rail current (mA)", xlim=(-0.3, 2.5),
           title="Inrush surge — rail_off vs pd (real scope)")
    ax.legend(loc="upper right")
    _save(fig, "inrush", "tools/data/{rail_off,pd}_scope capture 1 (R1, real HW)")


def _fig_inrush_one(mode):
    e, pk, settled = _INRUSH[mode]
    t, i_mA = _load_inrush(mode)
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.plot(t, i_mA, color=COLOR[mode], lw=0.7)
    ax.axhline(settled, ls=":", color="0.4", lw=1)
    ax.annotate(f"settled {settled} mA", (1.6, settled + 2), fontsize=7, color="0.3")
    ax.annotate(f"peak {i_mA.max():.0f} mA", (0.05, i_mA.max() - 4), fontsize=7)
    ax.set(xlabel="time from wake (ms)", ylabel="rail current (mA)", xlim=(-0.3, 2.5),
           title=f"Inrush — {mode} ($E$={e}, real scope)")
    _save(fig, f"inrush_{mode}", f"tools/data/{mode}_scope capture 1 (R1, real HW)")


def fig_inrush_rail_off():
    _fig_inrush_one("rail_off")


def fig_inrush_pd():
    _fig_inrush_one("pd")


def fig_noise():
    """Static noise PSD density (~55 ug/rtHz) + RMS-vs-bandwidth check, deglitched HW burst."""
    d = np.genfromtxt(DATA / "noise_burst.csv", delimiter=",", names=True)
    t = d["t_s"]; a, _ = nz.deglitch(d["accel_g"]); fs = 1 / np.median(np.diff(t))
    f, pxx = welch(a - a.mean(), fs=fs, nperseg=min(len(a), int(fs)))
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    dens = medfilt(np.sqrt(pxx[1:]) * 1e6, 11)
    ax.loglog(f[1:], dens, color=COLOR["pd"], lw=0.9)
    for lvl, lab in [(55, "measured ~55"), (85, "DS ≤85"), (25, "ADXL355 25")]:
        ax.axhline(lvl, ls=":", lw=1, color="0.5")
        ax.annotate(f"{lab} µg/√Hz", (1.2, lvl * 1.05), fontsize=6.5, color="0.35")
    ax.set(xlabel="frequency (Hz)", ylabel="noise density (µg/√Hz)",
           title="Noise spectral density", xlim=(1, fs / 2))
    ax.set_yticks([25, 55, 85])
    ax.set_yticklabels(["25", "55", "85"])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    _save(fig, "noise", "data/noise_burst.csv (real HW, deglitched)")


def fig_mode_crossover():
    """P_avg vs T_sleep for both modes; the ~257 ms pd<->rail_off crossover (auto-annotated from the curves)."""
    ts = np.geomspace(1e-3, 5, 500)
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    curves = {m: np.array([core.p_avg(_sensor(m), t, TGUARD) for t in ts]) * 1e3 for m in MODES}
    for m in MODES:
        ax.semilogx(ts * 1e3, curves[m], color=COLOR[m], label=m)
    xstar = ts[np.argmin(np.abs(curves["rail_off"] - curves["pd"]))] * 1e3
    ax.axvline(xstar, ls="--", color="0.4", lw=1)
    ax.annotate(f"crossover ≈{xstar:.0f} ms", (xstar * 1.1, 55), fontsize=7, color="0.3")
    ax.axhline(PON * 1e3, ls=":", color="k", lw=1); ax.annotate(f"continuous-ON {PON*1e3:.0f} mW", (1.2, 61), fontsize=7)
    ax.set(xlabel="$T_{sleep}$ (ms)", ylabel="average power (mW)",
           title="Mode trade-off (pd short sleeps, rail_off deep)")
    ax.legend(); _save(fig, "mode_crossover", "FFI sm_p_avg (measured coeffs)")


def fig_safemap():
    """The GREEN window on a log T_sleep axis: energy floor (ms) <-> monitoring-bandwidth ceiling."""
    ts = np.geomspace(3e-4, 2, 400)   # starts below the pd break-even so the RED floor is visible
    tol = core.tol(eps_allowed=0.001, sigma_allowed=0.0025, SF_allowed=0.0255,
                   B_required=0.5, D_max=100.0, S_min=0.10, delta=0.15)
    cls = {"RED": 0, "YELLOW": 1, "GREEN": 2}
    grid = np.array([[cls[core.classify(_sensor(m), tol, t)] for t in ts] for m in MODES])
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([COLOR["RED"], COLOR["YELLOW"], COLOR["GREEN"]])
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    ax.pcolormesh(ts * 1e3, [0.5, 1.5], grid, cmap=cmap, vmin=0, vmax=2, shading="nearest", edgecolors="face", linewidth=0.3, antialiased=False)
    ax.axhline(1.0, color="white", lw=3)
    ax.set_xscale("log"); ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(list(MODES))
    ax.set_ylim(0.0, 2.0); ax.set_xlabel("$T_{sleep}$ (ms, log)")
    # monitoring-bandwidth ceilings for several B_required (the window shrinks as apps demand more)
    for breq, c in [(0.5, "k"), (5.0, "0.35"), (28.0, "0.6")]:
        tmax = (1 / (2 * breq) - TGUARD - TPROBE) * 1e3
        if tmax > 1:
            ax.axvline(tmax, ls="--", color=c, lw=1)
            ax.annotate(f"$B_{{req}}$={breq:g}Hz", (tmax, 1.72), fontsize=6.5, color=c, ha="center")
    # top axis: B_eff
    axb = ax.secondary_xaxis("top", functions=(
        lambda x: 1e3 / (2 * (np.maximum(x, 1e-9) + (TGUARD + TPROBE) * 1e3)),
        lambda b: 1e3 / (2 * np.maximum(b, 1e-9)) - (TGUARD + TPROBE) * 1e3))
    axb.set_xlabel("$B_{eff}$ (Hz)")
    _save(fig, "safemap", "FFI sm_classify (measured coeffs)")


def fig_biastc():
    """Bias vs die-temperature (de-drifted), bivariate slope -75.5 ug/C, from HW heat-cool hump."""
    d = np.genfromtxt(DATA / "temp_log.csv", delimiter=",", names=True)
    t, T = d["t_s"], d["temp_C"]
    a, _ = nz.deglitch(d["accel_g"])
    good = ~((np.abs(d["accel_g"]) < 1e-5) & (np.abs(T - 25) < 1e-3)) & (np.abs(T - medfilt(T, 11)) < 3)
    t, a, T = t[good], a[good], T[good]
    res = a - medfilt(a, 151); good2 = np.abs(res - np.median(res)) < 6 * np.median(np.abs(res)) * 1.4826
    t, a, T = t[good2], a[good2], T[good2]
    tn = (t - t[0]) / (t[-1] - t[0])
    X = np.vstack([T, tn, np.ones_like(T)]).T
    bT, bdrift, b0 = np.linalg.lstsq(X, a, rcond=None)[0]
    a_dedrift = (a - bdrift * tn) * 1e3          # remove the settling drift -> pure T dependence
    ipk = int(np.argmax(medfilt(T, 51)))
    edges = np.linspace(T.min(), T.max(), 14)

    def binned(Tx, Ax):
        idx = np.digitize(Tx, edges); c = 0.5 * (edges[1:] + edges[:-1])
        m = np.array([np.median(Ax[idx == k]) if (idx == k).any() else np.nan for k in range(1, len(edges))])
        ok = ~np.isnan(m); return c[ok], m[ok]

    hc, hm = binned(T[:ipk], a_dedrift[:ipk])         # heating leg (cold -> hot)
    cc, cm = binned(T[ipk:], a_dedrift[ipk:])         # cooling leg (hot -> cold)

    def band(Tx, Ax):
        idx = np.digitize(Tx, edges); c = 0.5 * (edges[1:] + edges[:-1])
        sd = np.array([np.std(Ax[idx == k]) if (idx == k).sum() > 3 else np.nan
                       for k in range(1, len(edges))])
        ok = ~np.isnan(sd); return c[ok], sd[ok]

    HEAT, COOL = "#CC3311", "#0077BB"                 # warm heating, cool cooling
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for (bc, bm, Tx, Ax, colr, mk, lab, sgn) in (
            (hc, hm, T[:ipk], a_dedrift[:ipk], HEAT, "o", "heating", +1),
            (cc, cm, T[ipk:], a_dedrift[ipk:], COOL, "s", "cooling", -1)):
        sc, sd = band(Tx, Ax)
        ax.fill_between(sc, np.interp(sc, bc, bm) - sd, np.interp(sc, bc, bm) + sd,
                        color=colr, alpha=0.10, lw=0)
        ax.plot(bc, bm, "-" + mk, color=colr, ms=2.8, lw=1.2, mec="none", label=lab)
        k = len(bc) // 3                              # direction arrowhead on the branch
        dx, dy = sgn * (bc[k + 1] - bc[k]), sgn * (bm[k + 1] - bm[k])
        x0 = bc[k] if sgn > 0 else bc[k + 1]
        y0 = bm[k] if sgn > 0 else bm[k + 1]
        ax.annotate("", (x0 + 0.6 * dx, y0 + 0.6 * dy), (x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=colr, mutation_scale=16, lw=1.2))
    lo, hi = max(hc.min(), cc.min()), min(hc.max(), cc.max())
    hh = np.interp(cc, hc, hm); hyst = float(np.nanmean(np.abs(cm - hh)[(cc >= lo) & (cc <= hi)]))
    Tm = 0.5 * (lo + hi)
    y1, y2 = np.interp(Tm, hc, hm), np.interp(Tm, cc, cm)
    ax.annotate("", (Tm, y2), (Tm, y1), arrowprops=dict(arrowstyle="<->", color="0.25", lw=1))
    ax.annotate(f"{hyst:.1f} mg", (Tm + 0.8, 0.5 * (y1 + y2) - 0.4), fontsize=6.5, color="0.25")
    Tl = np.array([T.min(), T.max()])
    ax.plot(Tl, (bT * Tl + b0) * 1e3, color="0.35", ls=(0, (4, 2)), lw=1.0,
            label=f"fit $-${abs(bT)*1e6:.1f} µg/°C")
    ax.set(xlabel="die temperature (°C)", ylabel="bias, de-drifted (mg)")
    ax.legend(fontsize=7); _save(fig, "biastc", "data/temp_log.csv (real HW heat-cool hump)")


def fig_theta_reach():
    """GREEN half-width vs operating dT: uncompensated (~+-11 C) vs compensated (whole range)."""
    tol = core.tol(eps_allowed=0.001, sigma_allowed=0.0025, SF_allowed=0.0255,
                   B_required=0.5, D_max=2.0, S_min=0.0, delta=0.15)
    dT = np.arange(0, 60, 0.5)
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for lab, tc, c, ls in [("uncompensated", -75.5e-6, COLOR["RED"], "-"),
                           ("compensated (residual)", 3.0e-6, COLOR["GREEN"], (0, (4, 2)))]:
        w = []
        for x in dT:
            s = _sensor("rail_off", dT_op=float(x)); s.bias_tc = tc
            gi = core.green_interval(s, tol)
            w.append((gi[1] - gi[0]) if gi else 0.0)
        ax.plot(dT, w, color=c, ls=ls, lw=1.5, label=lab)
    ax.axvline(11, ls="--", color="0.4", lw=1); ax.annotate("±11 °C", (11.5, 0.1), fontsize=7)
    ax.set(xlabel="operating |ΔT| from calibration (°C)", ylabel="GREEN width (s)",
           title="Temperature reach of the safe map")
    ax.legend(); _save(fig, "theta_reach", "FFI green_interval vs dT_op")


def fig_frontier():
    """Energy-capture frontier: saving vs B_eff, both modes, wall/knee + T_lead regime bands."""
    ts = np.geomspace(1e-3, 20, 2000)
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    for m in MODES:
        beff = np.array([core.b_eff(_sensor(m), t, TGUARD) for t in ts])
        sav = np.array([core.saving(_sensor(m), t, TGUARD) for t in ts]) * 100
        ax.plot(beff, sav, color=COLOR[m], label=m)
    ax.scatter([FS / 2], [0], color="k", zorder=5, s=15)
    ax.annotate("continuous-ON", (FS / 2, 3), fontsize=6.5, ha="right")
    for b, lab in [(0.5, "1 s"), (5, "100 ms"), (28, "18 ms")]:
        ax.axvline(b, ls=":", color="0.6", lw=0.8)
    # regime tints, following the map's colours: C6-forbidden | pays | steep | wall
    for x0, x1, c in [(0.1, 0.5, COLOR["RED"]), (0.5, 5, COLOR["GREEN"]),
                      (5, 28, COLOR["YELLOW"]), (28, FS / 2, COLOR["RED"])]:
        ax.axvspan(x0, x1, color=c, alpha=0.07, lw=0, zorder=0)
    ax.annotate("wall", (26, 60), fontsize=7, color="0.3")
    ax.annotate("knee\n(~5 Hz, 80%)", (5.5, 40), fontsize=6.5, color="0.3")
    ax.set(xlabel="$B_{eff}$ (Hz)", ylabel="energy saving (%)", xlim=(0.1, FS / 2),
           ylim=(0, 100), title="Energy–capture frontier")
    ax.set_xscale("log"); ax.legend(loc="lower left")
    _save(fig, "frontier", "FFI sm_b_eff/sm_saving (measured coeffs)")


def fig_s2_timeline():
    """One gradual+detectable run: degradation, adaptive B_mon tracking the shrinking warning, hits."""
    from analysis import s2sim as s2
    r = s2.run_all(f_deg=1.0, detectable=True)["adaptive"]
    tl = r["timeline"]; t = np.array(tl["t"]) / 60.0
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    ax.plot(t, tl["b_mon"], color=COLOR["pd"], lw=1.0, label="$B_{mon}$ (adaptive)")
    ax.plot(t, [1 / (2 * (2.0 * (0.015 / 2.0) ** dd)) for dd in tl["d"]], "k--", lw=0.9,
            label="need $1/(2T_{lead})$")
    hit = np.array(tl["hit"])
    ax.scatter(t[~hit], np.array(tl["b_mon"])[~hit], s=14, color=COLOR["RED"], zorder=5, label="miss")
    ax.set(xlabel="time (min)", ylabel="bandwidth (Hz)", yscale="log",
           title=f"Adaptive tracks degradation (cover {r['coverage']*100:.0f}%, save {r['saving']*100:.0f}%)")
    ax.legend(fontsize=7, loc="upper left"); _save(fig, "s2_timeline", "s2sim (gradual+detectable)")


def fig_s2_boundary():
    """Winner (min-J) over degradation gradualness x precursor detectability."""
    from analysis import s2sim as s2
    fds = np.linspace(0.1, 1.0, 10)
    grid, order, _ = s2.boundary_map_robust(list(fds), [True, False], seeds=range(8))   # majority vote
    win_color = {"continuous": "#444444", "fixed_benign": COLOR["YELLOW"],
                 "fixed_worstcase": COLOR["rail_off"], "adaptive": COLOR["GREEN"]}
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([win_color[o] for o in order])
    fig, ax = plt.subplots(figsize=(3.8, 2.0))
    ax.pcolormesh(fds, [0, 1], grid, cmap=cmap, vmin=0, vmax=len(order) - 1, shading="nearest")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["detectable", "stealthy"])
    ax.set_xlabel("degradation gradualness (abrupt → gradual)")
    import matplotlib.patches as mp
    ax.legend(handles=[mp.Patch(color=win_color[o], label=o) for o in order],
              fontsize=6.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.set_title("Where adaptive $B_{mon}$ wins")
    _save(fig, "s2_boundary", "s2sim boundary_map (min-J winner)")




# ================================================================================================
# Tier 3: HIL campaign figures — regenerated from the COMMITTED bench logs in tools/data/
# (docs/hil_adaptive_experiment_plan_2026-07-01.md, Results). Same contract: no screenshots.
HILDATA = Path(__file__).resolve().parent.parent.parent / "tools" / "data"
CLR_LEARN, CLR_FROZEN = "#0072B2", "#D55E00"        # entity-fixed: online layer / frozen map


def _hil_log(pattern):
    """Latest committed log matching the pattern -> (ctl rows dict-of-lists, ok-fits list)."""
    path = sorted(HILDATA.glob(pattern))[-1]
    ctl = {k: [] for k in ("t_s", "guard_ms", "b0hat_mg", "tauhat_ms", "active", "tsleep_ms",
                           "temp_C")}
    fits = []
    for ln in path.read_text().splitlines():
        if ln.startswith("CTL,"):
            f = ln.split(",")
            if len(f) in (15, 16):
                try:
                    ctl["t_s"].append(float(f[2]) / 1e3); ctl["tsleep_ms"].append(float(f[4]))
                    ctl["guard_ms"].append(float(f[5])); ctl["active"].append(float(f[6]))
                    ctl["b0hat_mg"].append(float(f[8])); ctl["tauhat_ms"].append(float(f[9]))
                    ctl["temp_C"].append(float(f[15]) if len(f) == 16 else np.nan)
                except ValueError:
                    pass
        elif ln.startswith("CTLFIT") and ",ok," in ln:
            f = ln.split(",")
            fits.append((int(f[1]), float(f[3]), float(f[4])))   # cycle, b0_mg, tau_ms
    return {k: np.asarray(v) for k, v in ctl.items()}, np.asarray(fits), path.name


def fig_hil_ageing():
    """Exp 5 ablation on hardware: the online layer tracks the injected ageing and keeps the
    post-guard residual inside the C2 budget; the frozen map crosses it."""
    A, Afit, asrc = _hil_log("ctl_ageing_learn_*.log")
    B, Bfit, bsrc = _hil_log("ctl_ageing_frozen2_*.log")
    mA = A["t_s"] <= 1805
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.6, 3.6), sharex=True,
                                   gridspec_kw={"hspace": 0.12})
    ax1.plot(A["t_s"][mA] / 60, A["guard_ms"][mA], color=CLR_LEARN, label="online layer")
    ax1.plot(B["t_s"] / 60, B["guard_ms"], color=CLR_FROZEN, label="frozen map")
    ax1.set_ylabel(r"$T_{guard}$ (ms)")
    ax1.legend(loc="upper left")
    ax1.set_title(r"Ageing surrogate on hardware (spoof 100$\rightarrow$600%)")
    # post-guard residual vs the C2 budget: A at its live guard, B virtual at its frozen guard
    residA = A["b0hat_mg"][mA] * np.exp(-A["guard_ms"][mA] / A["tauhat_ms"][mA])
    gB = np.median(B["guard_ms"])
    residB = Bfit[:, 1] * np.exp(-gB / Bfit[:, 2])
    tB = Bfit[:, 0] / Bfit[:, 0].max() * 30.0                  # cycles -> minutes (uniform cadence)
    ax2.semilogy(A["t_s"][mA] / 60, residA, color=CLR_LEARN, lw=1.0)
    ax2.semilogy(tB, residB, ".", color=CLR_FROZEN, ms=2, alpha=0.5)
    ax2.axhline(1.0, ls=":", color="k", lw=1)
    box = dict(fc="white", ec="none", alpha=0.85, pad=1.5)
    ax2.annotate(r"$\varepsilon$ = 1 mg budget", (0.02, 0.90), xycoords="axes fraction",
                 fontsize=7, color="0.25", bbox=box)
    ax2.set(xlabel="time (min)", ylabel="post-guard bias (mg)")
    viol = float((residB > 1.0).mean())
    ax2.annotate(f"frozen map: {viol:.0%} of fits violate", (0.98, 0.76),
                 xycoords="axes fraction", ha="right", fontsize=7,
                 color=CLR_FROZEN, bbox=box)
    ax2.annotate("online layer: 0 violations", (0.98, 0.06), xycoords="axes fraction",
                 ha="right", fontsize=7, color=CLR_LEARN, bbox=box)
    _save(fig, "hil_ageing", f"{asrc} + {bsrc}")


def fig_hil_soak():
    """Exp 6: 12 h overnight soak — estimator stability and the wake record."""
    S, _, src = _hil_log("ctl_soak_*.log")
    hr = (S["t_s"] // 3600).astype(int)
    hours = np.unique(hr)
    act = np.array([S["active"][hr == h].sum() for h in hours])
    gmed = np.array([np.median(S["guard_ms"][hr == h]) for h in hours])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.6, 3.2), sharex=True,
                                   gridspec_kw={"hspace": 0.12})
    ax1.bar(hours, act, width=0.8, color=CLR_LEARN)
    ax1.set_ylabel("wakes / h")
    ax1.annotate("morning lab activity", (hours[-2] - 0.4, act[-2] + 8), fontsize=7,
                 color="0.25", ha="right")
    ax1.set_title(f"12 h soak: {len(S['t_s']):,} cycles, 0 log gaps")
    ax2.plot(hours + 0.5, gmed, color=CLR_LEARN)
    ax2.set(xlabel="hour", ylabel=r"$T_{guard}$ (ms)", ylim=(12, 18))
    _save(fig, "hil_soak", src)


def fig_hil_thermal():
    """Exp 4 thermal ramp: the C2 temperature term measured live — guard grows with the die
    temperature, GREEN collapses at the predicted 36.26 C singularity (fail-safe continuous),
    and re-admits on cooling."""
    # Pinned to the original M7 ramp (2026-07-03) -- a bare "ctl_thermal_*.log" glob
    # collides with the later per-unit replication campaign's ctl_thermal_unit{A,B}_r*.log
    # files, and _hil_log takes the alphabetically-last match, silently swapping in the
    # wrong unit's ramp under this figure's hardcoded 36.26 C (unit A) annotations.
    T, _, src = _hil_log("ctl_thermal_20260703_*.log")
    BIAS_TC, EPS_ETA = 0.0755, 0.85          # mg/C, (1-delta)*eps in mg
    t_col = 25.0 + EPS_ETA / BIAS_TC         # predicted collapse: 36.26 C
    tm = T["t_s"] / 60
    green = (T["tsleep_ms"] > 0) & (T["guard_ms"] > 0)
    c0, c1 = tm[~green].min(), tm[~green].max()

    fig = plt.figure(figsize=(4.2, 5.4))
    gs = fig.add_gridspec(3, 1, hspace=0.45, height_ratios=[1, 1, 1.3])
    ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2])
    box = dict(fc="white", ec="none", alpha=0.85, pad=1.5)
    for ax in (ax1, ax2):
        ax.axvspan(c0, c1, color=CLR_FROZEN, alpha=0.12, lw=0)

    ax1.plot(tm, T["temp_C"], color=CLR_LEARN, lw=1.0)
    ax1.axhline(t_col, ls=":", color="k", lw=1)
    ax1.annotate(r"predicted collapse 36.26$\,^\circ$C", (0.02, 0.86),
                 xycoords="axes fraction", fontsize=7, color="0.25", bbox=box)
    ax1.set_ylabel(r"die temp ($^\circ$C)")
    ax1.set_title("Thermal ramp on hardware (exp 4)")
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.plot(tm[green], T["guard_ms"][green], ".", color=CLR_LEARN, ms=2.2)
    ax2.plot(tm[~green], np.zeros((~green).sum()), ".", color=CLR_FROZEN, ms=2.2)
    ax2.annotate("GREEN rejected:\nfail-safe continuous", ((c0 + c1) / 2, 8),
                 fontsize=7, color=CLR_FROZEN, ha="center", bbox=box)
    ax2.set(xlabel="time (min)", ylabel=r"$T_{guard}$ (ms)", ylim=(-1.5, 27))

    ax3.plot(T["temp_C"][green], T["guard_ms"][green], ".", color=CLR_LEARN, ms=2.2,
             alpha=0.5, label="committed cycles")
    tt = np.linspace(20, t_col - 0.02, 300)
    b0, tau = np.median(T["b0hat_mg"][green]), np.median(T["tauhat_ms"][green])
    ax3.plot(tt, tau * np.log(b0 / (EPS_ETA - BIAS_TC * np.abs(tt - 25.0))),
             ls="--", color="k", lw=1, label=r"C2-$\theta$ at median $\hat b_0,\hat\tau$")
    ax3.axvline(t_col, ls=":", color="k", lw=1)
    t_obs = float(T["temp_C"][np.flatnonzero(~green)[0]])   # first rejected cycle = observed collapse
    ax3.axvline(t_obs, ls="-", color=CLR_FROZEN, lw=1)
    ax3.annotate(f"predicted {t_col:.2f}$\\,^\\circ$C", (t_col + 0.4, 11.0),
                 fontsize=6.5, color="0.25", bbox=box)
    ax3.annotate(f"observed {t_obs:.2f}$\\,^\\circ$C", (t_obs - 0.4, 25.3),
                 fontsize=6.5, color=CLR_FROZEN, ha="right", bbox=box)
    ax3.set(xlabel=r"die temp ($^\circ$C)", ylabel=r"$T_{guard}$ (ms)",
            xlim=(20, 40), ylim=(10, 27))
    ax3.legend(loc="upper left", fontsize=6.5)
    ax3.annotate(r"$\kappa\cdot$se margin", (0.03, 0.47),
                 xycoords="axes fraction", fontsize=6.5, color="0.25", bbox=box)
    _save(fig, "hil_thermal", src)


def fig_hil_campaign():
    """Measured energy saving per HIL experiment (INA219 through the sleeps)."""
    import json
    runs = [("quiet (30 min)",  "ctl_quiet_*.summary.json"),
            ("taps (15 min)",   "ctl_taps_20260702_213319.summary.json"),
            ("rate (10 min)",   "ctl_rate_*.summary.json"),
            ("thermal (25 min)", "ctl_thermal_20260703_*.summary.json"),
            ("ageing, online",  "ctl_ageing_learn_*.summary.json"),
            ("ageing, frozen",  "ctl_ageing_frozen2_*.summary.json"),
            ("soak (12 h)",     "ctl_soak_*.summary.json")]
    names, vals = [], []
    for label, pat in runs:
        d = json.loads(sorted(HILDATA.glob(pat))[-1].read_text())
        names.append(label); vals.append(d["saving"] * 100)
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, height=0.62, color=CLR_LEARN)
    for yi, v in zip(y, vals):
        ax.annotate(f"{v:.1f}%", (v + 1, yi), va="center", fontsize=7.5, color="0.2")
    ax.set(yticks=y, yticklabels=names, xlim=(0, 104),
           xlabel="measured energy saving vs continuous-ON (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("HIL campaign: measured saving per run")
    _save(fig, "hil_campaign", "tools/data/ctl_*.summary.json")


def main():
    _style()
    print(f"regenerating figures -> {FIGDIR}")
    for fn in (fig_inrush, fig_inrush_rail_off, fig_inrush_pd, fig_noise,
               fig_mode_crossover, fig_safemap, fig_biastc, fig_theta_reach, fig_frontier,
               fig_s2_timeline, fig_s2_boundary,
               fig_hil_ageing, fig_hil_soak, fig_hil_thermal, fig_hil_campaign):
        fn()
    print("done.")


if __name__ == "__main__":
    main()
