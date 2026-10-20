"""Paper 1 revision, Section VII: unit A's map applied to unit B, on the revised inputs.

As `a2_transfer.py` (three classifications per (mode, T_sleep) cell: A's map, B's own map, B's
physics under A's guard), with the coefficient sets of Table II as revised:

  - the slow term in the budget (decision D6, option b): b_slow = 0.21 mg on both units, carried
    into the firmware C (FW4), so every guard targets (1 - delta) eps - b_s = 0.64 mg;
  - E_inrush at the 1.00 ohm shunt (correction C-7), the mean of 10 captures:
    A 0.0770 / 0.2499 mJ, B 0.1039 / 0.3104 mJ;
  - unit B's P_on and pd sleep power as the mean of the 30 July windows, 66.09 and 1.269 mW,
    as unit A's (Table II revision T2-1; the A2 analysis of July used B's medians);
  - rail_off with each unit's own measured transient (correction C-1; unit B from run M1 of
    2026-10-12), so each mode has its own guard;
  - unit B's pd transient as Table II gives it (26.9 mg at 6.20 ms, 1.299 ms; the mean of the
    30 per-burst fits, as unit A's 82.6 mg).

`classify_under_guard` is a2_transfer's with b_slow added to the bias, as sm_classify now does;
its parity with the firmware C at each unit's own guard is checked on every cell before anything
is reported.

Run: twin/.venv/bin/python twin/analysis/p1_transfer_revision.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

TWIN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TWIN))
sys.path.insert(0, str(TWIN / "analysis"))
import a2_transfer as a2  # noqa: E402
from bindings import core_ffi as core  # noqa: E402

TOL = a2.TOL
B_SLOW_G = 0.21e-3
BUDGET_G = (1.0 - TOL["delta"]) * TOL["eps_allowed"] - B_SLOW_G      # 0.64 mg
GRID = np.geomspace(3e-4, 2.0, 800)                                    # as the submitted section


def wake_referenced(b0_fit_g: float, t_fv_s: float, tau_s: float) -> float:
    """Amplitude at the wake command from the fit at first valid (the C's convention)."""
    return b0_fit_g * math.exp(t_fv_s / tau_s)


def unit(p_on, p_pd, e_pd, e_ro, b0_pd, fv_pd, tau_pd, b0_ro, fv_ro, tau_ro, sigma, sf, btc) -> dict:
    common = dict(P_on=p_on, sigma_pw=sigma, SF=sf, SF_ref=1.0, fs=175.0, T_probe=0.005,
                  a_max_exp=1.0, range_limit=30.0, bias_tc=btc, dT_op=0.0, b_slow=B_SLOW_G)
    return {
        "pd": dict(common, P_sleep=p_pd, E_inrush=e_pd,
                   b0=wake_referenced(b0_pd, fv_pd, tau_pd), tau=tau_pd),
        "rail_off": dict(common, P_sleep=2e-5, E_inrush=e_ro,
                         b0=wake_referenced(b0_ro, fv_ro, tau_ro), tau=tau_ro),
    }


# Table II (pd and rail_off transients at their own first valid; energies at 1.00 ohm;
# powers and energies are means over the repeats, as Table II gives them)
UNIT_A = unit(p_on=65.07e-3, p_pd=0.632e-3, e_pd=0.0770e-3, e_ro=0.2499e-3,
              b0_pd=82.6e-3, fv_pd=6.17e-3, tau_pd=1.449e-3,
              b0_ro=7.53e-3, fv_ro=11.41e-3, tau_ro=1.575e-3,
              sigma=1.59e-3, sf=1.0 - 70e-6, btc=-75.5e-6)
UNIT_B = unit(p_on=66.0926e-3, p_pd=1.2689e-3, e_pd=0.1039e-3, e_ro=0.3104e-3,
              b0_pd=26.9e-3, fv_pd=6.20e-3, tau_pd=1.299e-3,
              b0_ro=1.50e-3, fv_ro=12.35e-3, tau_ro=1.019e-3,
              sigma=1.70e-3, sf=1.0 - 201e-6, btc=-68.9e-6)
MODES = ("pd", "rail_off")


def guard(c: dict) -> float:
    return core.t_guard_min(core.sensor(**c), core.tol(**TOL))


def transient_g(c: dict, tg: float) -> float:
    return c["b0"] * math.exp(-tg / c["tau"])


def _recovery_pass(c: dict, tg: float, margin: bool) -> bool:
    k = (1.0 - TOL["delta"]) if margin else 1.0
    bias = transient_g(c, tg) + abs(c["bias_tc"] * c["dT_op"]) + c["b_slow"]
    return (bias <= k * TOL["eps_allowed"] + 1e-9
            and c["sigma_pw"] <= k * TOL["sigma_allowed"] + 1e-9
            and abs(c["SF"] - c["SF_ref"]) <= k * TOL["SF_allowed"] + 1e-9)


def classify_under_guard(c: dict, ts: float, tg: float) -> str:
    """sm_classify with an imposed guard (a2_transfer's, with b_slow in the bias)."""
    if not c["a_max_exp"] < c["range_limit"]:
        return "RED"
    if not math.isfinite(tg) or not _recovery_pass(c, tg, False):
        return "RED"
    tc = ts + tg + c["T_probe"]
    p_avg = (c["P_sleep"] * ts + c["P_on"] * (tg + c["T_probe"]) + c["E_inrush"]) / tc
    s = 1.0 - p_avg / c["P_on"]
    b_eff = min(0.5 * c["fs"], 1.0 / (2.0 * tc))
    if not (s >= -1e-6 and ts <= TOL["D_max"] + 1e-6 and b_eff >= TOL["B_required"] - 1e-6):
        return "RED"
    k = 1.0 - TOL["delta"]
    green = (s >= TOL["S_min"] and ts <= k * TOL["D_max"] and b_eff >= TOL["B_required"] / k
             and _recovery_pass(c, tg, True))
    return "GREEN" if green else "YELLOW"


def check_parity() -> int:
    """classify_under_guard at a unit's own guard must equal the firmware C on every cell."""
    t, bad = core.tol(**TOL), 0
    for u in (UNIT_A, UNIT_B):
        for m in MODES:
            s, tg = core.sensor(**u[m]), guard(u[m])
            bad += sum(core.classify(s, t, float(x)) != classify_under_guard(u[m], float(x), tg)
                       for x in GRID)
    return bad


def saving(c: dict, ts: float, tg: float) -> float:
    tc = ts + tg + c["T_probe"]
    return 1.0 - (c["P_sleep"] * ts + c["P_on"] * (tg + c["T_probe"]) + c["E_inrush"]) / tc / c["P_on"]


def break_even(c: dict, tg: float) -> float:
    return brentq(lambda ts: saving(c, ts, tg), 1e-7, 10.0)


def at_bandwidth(c: dict, tg: float, b_hz: float) -> float:
    """Saving at monitoring bandwidth b (cycle 1/(2b))."""
    ts = 1.0 / (2.0 * b_hz) - tg - c["T_probe"]
    return saving(c, ts, tg) if ts > 0 else float("nan")


def crossover(u: dict, tg_pd: float, tg_ro: float) -> float:
    s_pd, s_ro = core.sensor(**u["pd"]), core.sensor(**u["rail_off"])
    f = lambda ts: core.p_avg(s_ro, ts, tg_ro) - core.p_avg(s_pd, ts, tg_pd)  # noqa: E731
    return brentq(f, 1e-4, 5.0, xtol=1e-6)


def report() -> None:
    bad = check_parity()
    print(f"parity of classify_under_guard with the firmware C at the own guards: {bad} mismatches")
    assert bad == 0
    t = core.tol(**TOL)
    g = {(n, m): guard(u[m]) for n, u in (("A", UNIT_A), ("B", UNIT_B)) for m in MODES}
    print(f"budget left to the transient {BUDGET_G*1e3:.2f} mg; guards (ms): "
          + ", ".join(f"{n} {m} {v*1e3:.2f}" for (n, m), v in g.items()))

    print("\nB's transient at A's guard, per mode (transient alone; with b_s; budget 0.85 margined)")
    for m in MODES:
        r = transient_g(UNIT_B[m], g[("A", m)])
        print(f"  {m:8s} A's guard {g[('A', m)]*1e3:.2f} ms: transient {r*1e3:.3f} mg; "
              f"+ b_s {(r + B_SLOW_G)*1e3:.3f} mg; margin to {BUDGET_G*1e3:.2f} mg x{BUDGET_G/r:.1f}")

    print(f"\ntransfer on a {len(GRID)}-point grid per mode (A's map; B under A's guard)")
    for m in MODES:
        s_a, s_b = core.sensor(**UNIT_A[m]), core.sensor(**UNIT_B[m])
        a_map = np.array([core.classify(s_a, t, float(x)) for x in GRID])
        b_own = np.array([core.classify(s_b, t, float(x)) for x in GRID])
        b_ua = np.array([classify_under_guard(UNIT_B[m], float(x), g[("A", m)]) for x in GRID])
        ag = a_map == "GREEN"
        print(f"  {m:8s} A-GREEN {ag.sum()}: B under A RED {np.sum(ag & (b_ua == 'RED'))}, "
              f"YELLOW {np.sum(ag & (b_ua == 'YELLOW'))} ({100*np.mean(b_ua[ag] == 'YELLOW'):.1f} %); "
              f"B-own GREEN {np.sum(b_own == 'GREEN')}; B GREEN under A {np.sum(b_ua == 'GREEN')}")
        if np.any(ag & (b_ua == "YELLOW")):
            x = GRID[ag & (b_ua == "YELLOW")]
            print(f"           degraded cells span T_sleep {x.min()*1e3:.2f} to {x.max()*1e3:.2f} ms")

    print("\nunit B: own guards and under A's guards")
    for label, gsrc in (("own", "B"), ("under A", "A")):
        for m in MODES:
            c, tg = UNIT_B[m], g[(gsrc, m)]
            tbe = break_even(c, tg)
            bmax = 1.0 / (2.0 * (tbe + tg + c["T_probe"]))
            sv = [100 * at_bandwidth(c, tg, b) for b in (0.5, 5.0, 20.0)]
            print(f"  {label:8s} {m:8s} guard {tg*1e3:.2f} ms: break-even {tbe*1e3:.2f} ms; "
                  f"largest bandwidth with a saving {bmax:.1f} Hz; saving at 0.5/5/20 Hz "
                  + " / ".join(f"{v:.1f}" for v in sv) + " %")
    for n, u in (("A", UNIT_A), ("B", UNIT_B)):
        print(f"  crossover unit {n} (own guards, firmware p_avg): "
              f"{crossover(u, g[(n, 'pd')], g[(n, 'rail_off')])*1e3:.0f} ms")
    print(f"  crossover unit B under A's guards: "
          f"{crossover(UNIT_B, g[('A', 'pd')], g[('A', 'rail_off')])*1e3:.0f} ms")

    print("\nratios B/A: E_inrush pd {:.3f}, rail_off {:.3f}; P_sleep pd {:.2f}; b0 pd {:.2f}; "
          "|kappa_b| {:.3f}".format(UNIT_B["pd"]["E_inrush"] / UNIT_A["pd"]["E_inrush"],
                                    UNIT_B["rail_off"]["E_inrush"] / UNIT_A["rail_off"]["E_inrush"],
                                    UNIT_B["pd"]["P_sleep"] / UNIT_A["pd"]["P_sleep"],
                                    26.9 / 82.6, 68.9 / 75.5))
    for n, u in (("A", UNIT_A), ("B", UNIT_B)):
        k = abs(u["pd"]["bias_tc"])
        print(f"  thermal collapse swing unit {n}: (1-delta) eps / |kappa| = {0.85e-3/k:.2f} C (as tested, "
              f"b_s = 0); with b_s {BUDGET_G/k:.2f} C")


if __name__ == "__main__":
    report()
