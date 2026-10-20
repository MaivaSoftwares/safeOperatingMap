"""A2 map transfer: unit A's safe operating map applied to unit B's physics.

The safe map is built FROM one unit's coefficients (campaign R1, unit A). A2
asks the transfer question: if that map -- its guard, its GREEN window -- is
applied unchanged to a second unit (B, characterized in campaign A1), where
does it mislead? Three classifications per (mode, T_sleep) cell:

  A_map      what the shipped map says          (A coefficients, A guard)
  B_truth    what B's own map would say         (B coefficients, B guard)
  B_under_A  B's physics OPERATED with A's map  (B coefficients, A's guard)

The dangerous direction is A_map GREEN where B_under_A is RED (the map
authorizes a point the second unit cannot deliver); the benign direction is
coverage loss. `classify_under_guard` mirrors core/safemap.c sm_classify
exactly except the guard is imposed instead of derived -- parity with the FFI
is pinned by a test at tg = the unit's own guard.

Warm vs cold: b0 is strongly history-dependent. The warm coefficients are the
A1 n=30 medians (board continuously powered); the cold-history variant is the
2026-07-13 R2 session (excluded=1) taken ~2 min after ~20 rail_off power
cycles, where the wake-referenced transient grows ~46x and even first_valid
moves. The transfer verdict flips there -- that is the section-10 argument in
one number.

Regenerate report + figure: cd twin && .venv/bin/python -m analysis.a2_transfer
Output: docs/figures/a2_transfer.{pdf,png}
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # twin/

import numpy as np
import pandas as pd
from bindings import core_ffi as core

FIGDIR = Path(__file__).resolve().parent.parent.parent / "docs" / "figures"

# Tolerance set of the published chapter map (figures.fig_safemap / safemap.pdf)
TOL = dict(eps_allowed=0.001, sigma_allowed=0.0025, SF_allowed=0.0255,
           B_required=0.5, D_max=100.0, S_min=0.10, delta=0.15)

EPS_EFF = (1.0 - TOL["delta"]) * TOL["eps_allowed"]   # 0.85 mg

# ---- unit A: R1 campaign medians 2026-07-06/08 (figures.py convention).
# b0 is WAKE-REFERENCED (b0_fit * exp(t_fv/tau): 82.6 mg at 6.174 ms -> 5.81 g).
A_T_FV_S = 0.006174
_A_COMMON = dict(P_on=0.0651, b0=5.81, tau=0.001449, sigma_pw=0.00159,
                 SF=1.001, SF_ref=1.0, fs=175.0, T_probe=0.005,
                 a_max_exp=1.0, range_limit=30.0, bias_tc=-75.5e-6, dT_op=0.0)
UNIT_A = {
    "rail_off": dict(_A_COMMON, P_sleep=2e-5, E_inrush=0.227e-3),
    "pd":       dict(_A_COMMON, P_sleep=0.632e-3, E_inrush=0.070e-3),
}

# ---- unit B warm: A1 campaign medians 2026-07-12/13 (commit d3c6015).
# burst pd n=30: b0_fit 27.14 mg at 6.195 ms, tau 1.297 ms; noise 1.70 mg;
# power n=30: P_on 66.12 mW, P_sleep pd 1.286 mW (rail_off at the 20 uW floor);
# scope inrush n=10 medians 94.1/288.8 uJ; SF wake dev pd -201 ppm (resolvable,
# unlike unit A's null). bias_tc measured 2026-07-16 thermal hump (+20 C,
# data/temp_log_unitB.csv): -68.9 +/- 4.8 ug/C, variant-stable -69..-72.
B_T_FV_S = 0.006195
B_B0_FIT_G, B_TAU_S = 27.14e-3, 0.001297
_B_COMMON = dict(P_on=0.066125, b0=B_B0_FIT_G * math.exp(B_T_FV_S / B_TAU_S),
                 tau=B_TAU_S, sigma_pw=0.00170, SF=1.0 - 201e-6, SF_ref=1.0,
                 fs=175.0, T_probe=0.005, a_max_exp=1.0, range_limit=30.0,
                 bias_tc=-68.9e-6, dT_op=0.0)
UNIT_B = {
    "rail_off": dict(_B_COMMON, P_sleep=2e-5, E_inrush=288.8e-6),
    "pd":       dict(_B_COMMON, P_sleep=1.286e-3, E_inrush=94.1e-6),
}

# ---- unit B cold-history: R2 session_20260713_210443 medians (excluded=1),
# ~2 min after ~20 rail_off power cycles. b0_fit 85.9 mg at 9.483 ms (first
# valid itself moved), tau 1.2715 ms -> wake-referenced ~149 g. Only the
# transient differs; power/inrush kept at the warm values.
BC_T_FV_S = 0.009483
BC_B0_FIT_G, BC_TAU_S = 85.9e-3, 0.0012715
B_COLD = {m: dict(c, b0=BC_B0_FIT_G * math.exp(BC_T_FV_S / BC_TAU_S),
                  tau=BC_TAU_S)
          for m, c in UNIT_B.items()}

MODES = ("rail_off", "pd")


def own_guard(coeffs: dict, tol: dict = TOL) -> float:
    """The unit's own minimum guard (sm_t_guard_min via the firmware C)."""
    return core.t_guard_min(core.sensor(**coeffs), core.tol(**tol))


def _recovery_pass(c: dict, tol: dict, tg: float, margin: bool) -> bool:
    k = (1.0 - tol["delta"]) if margin else 1.0
    bias = c["b0"] * math.exp(-tg / c["tau"]) + abs(c["bias_tc"] * c["dT_op"])
    return (bias <= k * tol["eps_allowed"] + 1e-9
            and c["sigma_pw"] <= k * tol["sigma_allowed"] + 1e-9
            and abs(c["SF"] - c["SF_ref"]) <= k * tol["SF_allowed"] + 1e-9)


def classify_under_guard(c: dict, tol: dict, ts: float, tg: float) -> str:
    """sm_classify with an imposed guard -- the transfer case (A's guard on
    B's physics). At tg = own_guard(c) this reproduces core.classify."""
    if not c["a_max_exp"] < c["range_limit"]:
        return "RED"
    if not math.isfinite(tg) or not _recovery_pass(c, tol, tg, False):
        return "RED"
    tc = ts + tg + c["T_probe"]
    p_avg = (c["P_sleep"] * ts + c["P_on"] * (tg + c["T_probe"]) + c["E_inrush"]) / tc
    s = 1.0 - p_avg / c["P_on"]
    b_eff = min(0.5 * c["fs"], 1.0 / (2.0 * tc))
    if not (s >= -1e-6 and ts <= tol["D_max"] + 1e-6
            and b_eff >= tol["B_required"] - 1e-6):
        return "RED"
    k = 1.0 - tol["delta"]
    green = (s >= tol["S_min"] and ts <= k * tol["D_max"]
             and b_eff >= tol["B_required"] / k
             and _recovery_pass(c, tol, tg, True))
    return "GREEN" if green else "YELLOW"


def rail_off_at(unit: dict, need_s: float) -> dict:
    """The unit's coefficient sets with rail_off at its own measured need (correction C-1): its
    amplitude is pd's delayed by (need - pd guard), so the C returns `need_s` as its guard."""
    delay = need_s - own_guard(unit["pd"])
    ro = dict(unit["rail_off"], b0=core.b0_delayed(unit["pd"]["b0"], unit["pd"]["tau"], delay))
    return dict(unit, rail_off=ro)


def transfer_table(ts_grid=None, cold: bool = False, unit_a: dict | None = None,
                   unit_b: dict | None = None) -> pd.DataFrame:
    """Per (mode, T_sleep): A_map, B_truth, B_under_A classifications."""
    ts_grid = np.geomspace(3e-4, 2.0, 800) if ts_grid is None else ts_grid
    t = core.tol(**TOL)
    unit_a = UNIT_A if unit_a is None else unit_a
    unit_b = (B_COLD if cold else UNIT_B) if unit_b is None else unit_b
    rows = []
    for mode in MODES:
        s_a = core.sensor(**unit_a[mode])
        s_b = core.sensor(**unit_b[mode])
        tg_a = core.t_guard_min(s_a, t)
        for ts in ts_grid:
            rows.append({
                "mode": mode, "T_sleep": float(ts),
                "A_map": core.classify(s_a, t, float(ts)),
                "B_truth": core.classify(s_b, t, float(ts)),
                "B_under_A": classify_under_guard(unit_b[mode], TOL, float(ts), tg_a),
            })
    return pd.DataFrame(rows)


def guard_transfer(cold: bool = False) -> dict:
    """Residual bias when unit B is operated with unit A's guard."""
    c = (B_COLD if cold else UNIT_B)["pd"]     # transient is mode-independent
    tg_a = own_guard(UNIT_A["pd"])
    resid = c["b0"] * math.exp(-tg_a / c["tau"])
    return {"tg_a_ms": tg_a * 1e3, "residual_mg": resid * 1e3,
            "eps_eff_mg": EPS_EFF * 1e3, "margin": EPS_EFF / resid,
            "c2_ok": resid <= EPS_EFF}


def summarize(df: pd.DataFrame) -> dict:
    """Transfer confusion counts per mode + the A-GREEN cells B can't deliver."""
    out = {}
    for mode in MODES:
        d = df[df["mode"] == mode]
        a_green = d[d.A_map == "GREEN"]
        out[mode] = {
            "n": len(d),
            "a_green": len(a_green),
            "unsafe": int((a_green.B_under_A == "RED").sum()),
            "degraded": int((a_green.B_under_A == "YELLOW").sum()),
            "coverage_lost": int(((d.A_map != "GREEN") & (d.B_truth == "GREEN")).sum()),
        }
    return out


def report() -> dict:
    print(f"A2 map transfer -- unit A's map on unit B (tol: chapter safemap set)")
    print(f"guards: A {own_guard(UNIT_A['pd'])*1e3:.2f} ms | B warm "
          f"{own_guard(UNIT_B['pd'])*1e3:.2f} ms | B cold-history "
          f"{own_guard(B_COLD['pd'])*1e3:.2f} ms")
    out = {}
    for label, cold in (("warm", False), ("cold-history", True)):
        df = transfer_table(cold=cold)
        gt = guard_transfer(cold=cold)
        s = summarize(df)
        out[label] = {"summary": s, "guard_transfer": gt}
        verdict = "PASS" if gt["c2_ok"] else "VIOLATED"
        print(f"\n[{label}] C2 under A's {gt['tg_a_ms']:.2f} ms guard: residual "
              f"{gt['residual_mg']:.3f} mg vs eps_eff {gt['eps_eff_mg']:.2f} mg "
              f"({'x%.1f margin' % gt['margin'] if gt['c2_ok'] else 'x%.1f over' % (1/gt['margin'])}) -> {verdict}")
        for mode in MODES:
            m = s[mode]
            print(f"  {mode:8s} A-GREEN cells {m['a_green']:4d}: "
                  f"unsafe {m['unsafe']}, energy-degraded {m['degraded']}; "
                  f"coverage lost {m['coverage_lost']}")
    return out


def fig(figdir: Path = FIGDIR) -> list[Path]:
    """(a) wake transients A / B-warm / B-cold vs A's guard; (b) map strips."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "figure.dpi": 150,
        "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25,
    })
    status = {"RED": "#d1495b", "YELLOW": "#f4b642", "GREEN": "#2ca25f"}
    # unit hues deliberately distinct from the mode colors used elsewhere
    units = [("unit A", UNIT_A["pd"], A_T_FV_S, "0.35"),
             ("unit B warm", UNIT_B["pd"], B_T_FV_S, "#009E73"),
             ("unit B cold-history", B_COLD["pd"], BC_T_FV_S, "#CC79A7")]

    fig_, (ax, axm) = plt.subplots(
        2, 1, figsize=(4.2, 4.6), height_ratios=[1.5, 1.0])

    tg_a = own_guard(UNIT_A["pd"])
    tt = np.linspace(0, 0.020, 600)
    for name, c, t_fv, col in units:
        b = c["b0"] * np.exp(-tt / c["tau"]) * 1e3
        pre = tt < t_fv                       # extrapolated: output not yet valid
        ax.semilogy(tt[pre] * 1e3, b[pre], color=col, lw=1.2, ls=(0, (1.5, 1.5)))
        ax.semilogy(tt[~pre] * 1e3, b[~pre], color=col, lw=1.4, label=name)
    ax.axhline(EPS_EFF * 1e3, color="k", lw=0.8, ls="--")
    ax.annotate(r"$(1{-}\delta)\,\epsilon$ = 0.85 mg", (0.3, EPS_EFF * 1e3 * 1.25),
                fontsize=6.5)
    ax.axvline(tg_a * 1e3, color="0.2", lw=0.9)
    ax.annotate(f"unit A map guard {tg_a*1e3:.1f} ms",
                (tg_a * 1e3 - 0.35, 1.6e-2), fontsize=6.5, color="0.2",
                ha="right")
    for name, c, _fv, col in units[1:]:
        r = c["b0"] * math.exp(-tg_a / c["tau"]) * 1e3
        ax.plot([tg_a * 1e3], [r], "o", color=col, ms=4, mec="white", mew=0.6)
        ax.annotate(f"{r:.2f} mg", (tg_a * 1e3 + 0.3, r * 1.1), fontsize=6.5, color=col)
    ax.set(xlabel="time from wake command (ms)", ylabel="|bias| (mg)",
           xlim=(0, 20), ylim=(1e-2, 3e5),
           title="Guard transfer: A's guard on B's wake transient")
    ax.legend(fontsize=6.5, loc="upper right")

    # map strips: A's map vs B's own map, per mode (dotted = extrapolated region
    # convention not needed here -- classification is per T_sleep cell)
    ts = np.geomspace(3e-4, 2.0, 400)
    strips = [("A map", UNIT_A), ("B map", UNIT_B)]
    code = {"RED": 0, "YELLOW": 1, "GREEN": 2}
    t = core.tol(**TOL)
    rows, labels = [], []
    for mode in MODES:
        for sname, coeffs in strips:
            s = core.sensor(**coeffs[mode])
            rows.append([code[core.classify(s, t, float(x))] for x in ts])
            labels.append(f"{mode}: {sname}")
    cmap = ListedColormap([status["RED"], status["YELLOW"], status["GREEN"]])
    axm.pcolormesh(ts * 1e3, np.arange(len(rows)) + 0.5, np.array(rows),
                   cmap=cmap, vmin=0, vmax=2, shading="nearest")
    for y in range(1, len(rows)):
        axm.axhline(y, color="white", lw=2 if y % 2 == 0 else 1)
    axm.set_ylim(0, len(rows))
    axm.set_xscale("log")
    axm.set_yticks(np.arange(len(rows)) + 0.5)
    axm.set_yticklabels(labels, fontsize=6.5)
    axm.set_xlabel("$T_{sleep}$ (ms, log)")
    axm.set_title("Shipped map (A) vs unit-B truth, warm", fontsize=8)
    axm.grid(False)

    fig_.subplots_adjust(hspace=0.42)
    figdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = figdir / f"a2_transfer.{ext}"
        fig_.savefig(p)
        paths.append(p)
    plt.close(fig_)
    return paths


def main() -> int:
    report()
    for p in fig():
        print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
