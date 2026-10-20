"""Paper 1, Table IV: an uncertainty budget for the guard in GUM form.

Measurand: T_guard = t_fv + tau ln(b0 / ((1 - delta) eps)), the time after the wake command at
which the fitted transient reaches the budget, set from the calibration day's bursts. eps and
delta are requirements, not measurements, and carry no uncertainty.

Rows from our own record (Type A) are computed here. Rows that rest on an instrument's
specification (Type B) take their limits from the documents named next to each constant; a
limit a is entered as a rectangular distribution, u = a / sqrt(3) (GUM 4.3.7). A limit that
could not be read from a document is None and its row is printed as missing: nothing is
filled in from memory.

The node's clock is the one row far larger than the rest, and only for the guard quoted in SI
seconds: the node measures the transient and times the guard with the same oscillator. So two
totals are printed, on the node's own clock and in SI seconds, and beside them a direct
comparison of the node's clock with the host computer's over the controller segments of run M3.

Run: twin/.venv/bin/python twin/analysis/p1_uncertainty_budget.py
"""
from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TWIN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TWIN))
sys.path.insert(0, str(TWIN / "analysis"))
import p1_benchmark as bm  # noqa: E402
from p1_recovery_models import RUNS as CAL_RUNS  # noqa: E402
from p1_revision_checks import record_runs, unit_of  # noqa: E402

BUDGET_MG = bm.BUDGET_MG
SQRT3 = math.sqrt(3.0)

# ---- Type B limits, each from a document ---------------------------------------------------------
# Scale factor: 16 devices between 119.58 and 120.61 mV/g about the nominal 120 (test report
# ZW8.3-0802-NJ10B, Table 4)
SF_LIMIT_REL = 0.005
# Logged acceleration is printed to 0.1 mg (firmware burst format "%.4f" g, Core/Src/main.c)
LOG_STEP_MG = 0.1
# Sensor's own digital step: 1864135 LSB/V x 0.120 V/g (MMS-30 data sheet, Table 1)
SENSOR_LSB_MG = 1e3 / (1864135 * 0.120)
# Node clock: the STM32L476's internal MSI oscillator, range 6 (4 MHz), free-running, with no PLL and
# no crystal reference (Core/Src/main.c, SystemClock_Config). Data sheet DS10198 Rev 11, Table 60,
# pp. 154-155: 3.96 to 4.04 MHz after factory calibration at 3 V and 30 degC; drift -3.5 % to +3 %
# over 0 to 85 degC; -0.8 % to +0.7 % over 2.4 to 3.6 V (ranges 4 to 7). Three independent limits.
CLOCK_LIMITS_REL = dict(factory=0.01, temperature=0.035, supply=0.008)
CLOCK_LIMIT_REL = math.sqrt(sum(v * v for v in CLOCK_LIMITS_REL.values()))
# Slow term of the recovery, unit A (r3_2_recovery_models_2026-10-08.md); not resolved on unit B
SLOW_MG = bm.SLOW_MG
# Bias temperature coefficient and its standard uncertainty, mg per degC (manuscript Table V)
KAPPA = {"A": (0.0755, 0.0030), "B": (0.0689, 0.0048)}
# Die-temperature read-out: 72 LSB per degC, non-linearity 2 % (MMS-30 data sheet, Table 1)
TEMP_STEP_C, TEMP_NONLIN_REL = 1.0 / 72.0, 0.02


def calibration_bursts(record: list, unit: str) -> pd.DataFrame:
    cal = [b for b in record if b.run == CAL_RUNS[unit] and b.normal]
    return pd.DataFrame(dict(b0=[b.b0 for b in cal], tau=[1e3 * b.tau for b in cal],
                             fv=[1e3 * b.fv for b in cal]))


def guard(b0: float, tau: float, fv: float, budget: float = BUDGET_MG) -> float:
    return fv + tau * math.log(b0 / budget)


def type_a(cal: pd.DataFrame, reps: int = 4000, seed: int = 20261013) -> dict:
    """The guard from the calibration medians, with its Type A standard uncertainty two ways:
    bootstrap over the bursts, and the GUM law of propagation with the b0-tau correlation."""
    n = len(cal)
    med = cal.median()
    g = guard(med.b0, med.tau, med.fv)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(reps):
        m = cal.iloc[rng.integers(0, n, n)].median()
        boot.append(guard(m.b0, m.tau, m.fv))
    c_b0, c_tau = med.tau / med.b0, math.log(med.b0 / BUDGET_MG)
    se = cal.std(ddof=1) / math.sqrt(n)
    r = float(cal.b0.corr(cal.tau))
    u_lpu = math.sqrt((c_b0 * se.b0) ** 2 + (c_tau * se.tau) ** 2 + 2 * c_b0 * c_tau * se.b0 * se.tau * r
                      + se.fv ** 2)
    per_wake = np.array([guard(r_.b0, r_.tau, r_.fv) for r_ in cal.itertuples()])
    return dict(n=n, b0=med.b0, tau=med.tau, fv=med.fv, guard=g, c_b0=c_b0, c_tau=c_tau,
                sd_b0=float(cal.b0.std(ddof=1)), sd_tau=float(cal.tau.std(ddof=1)),
                sd_fv=float(cal.fv.std(ddof=1)), se_b0=float(se.b0), se_tau=float(se.tau),
                se_fv=float(se.fv), corr=r, u_boot=float(np.std(boot, ddof=1)), u_lpu=u_lpu,
                per_wake_sd=float(per_wake.std(ddof=1)),
                per_wake_lo=float(np.percentile(per_wake, 2.5)), per_wake_hi=float(np.percentile(per_wake, 97.5)))


def between_sessions(record: list, unit: str, min_bursts: int = 10) -> dict:
    """The need of each run of the record (median over its bursts): how far the guard has to move
    from session to session. Many runs were taken in provoked states."""
    need = np.array([1e3 * np.median([b.need() for b in bs])
                     for bs in bm.by_run(record, unit, min_bursts).values()])
    return dict(runs=len(need), median=float(np.median(need)), sd=float(need.std(ddof=1)),
                lo=float(need.min()), hi=float(need.max()),
                p05=float(np.percentile(need, 5)), p95=float(np.percentile(need, 95)))


def settled_level(unit: str, min_bursts: int = 10) -> dict:
    """Wake-to-wake repeatability of the settled level: within each pd run of the record, the
    standard deviation over its bursts of the level each burst settles to (sleeps of 1 s)."""
    sds, n_b = [], 0
    for d in record_runs():
        f = d / "fits.csv"
        if unit_of(str(d.relative_to(bm.runs.DATA))) != unit or not f.exists():
            continue
        v = [1e3 * float(r["settled_g"]) for r in csv.DictReader(f.open())
             if r.get("mode", "pd") == "pd" and r.get("ok") == "True" and r.get("settled_g")]
        if len(v) >= min_bursts:
            sds.append(float(np.std(v, ddof=1)))
            n_b += len(v)
    return dict(runs=len(sds), bursts=n_b, median_sd=float(np.median(sds)) if sds else float("nan"),
                p95_sd=float(np.percentile(sds, 95)) if sds else float("nan"))


def budget(unit: str, a: dict) -> pd.DataFrame:
    """Rows: source, type, standard uncertainty of the input, sensitivity, contribution in ms."""
    g = a["guard"]
    rows = [
        ("b0 (amplitude at first valid), repeatability", "A", f"{a['se_b0']:.2f} mg", f"{a['c_b0']:.4f} ms/mg",
         a["c_b0"] * a["se_b0"]),
        ("tau (time constant), repeatability", "A", f"{a['se_tau']:.4f} ms", f"{a['c_tau']:.2f}", a["c_tau"] * a["se_tau"]),
        ("t_fv (first valid sample), repeatability", "A", f"{a['se_fv']:.4f} ms", "1", a["se_fv"]),
        (f"b0 and tau together (correlation {a['corr']:+.2f}), bootstrap over the bursts", "A", "", "", a["u_boot"]),
        ("scale factor (enters b0)", "B", f"{100 * SF_LIMIT_REL / SQRT3:.2f} %", f"{a['tau']:.3f} ms", a["tau"] * SF_LIMIT_REL / SQRT3),
        ("resolution of the logged samples (0.1 mg step)", "B", f"{LOG_STEP_MG / (2 * SQRT3):.3f} mg", f"{a['c_b0']:.4f} ms/mg",
         a["c_b0"] * LOG_STEP_MG / (2 * SQRT3)),
        ("node clock (value in SI seconds)", "B", "missing" if CLOCK_LIMIT_REL is None else f"{100 * CLOCK_LIMIT_REL / SQRT3:.2f} %",
         f"{g:.2f} ms", None if CLOCK_LIMIT_REL is None else g * CLOCK_LIMIT_REL / SQRT3),
    ]
    return pd.DataFrame(rows, columns=["source", "type", "standard uncertainty", "sensitivity", "contribution_ms"])


def clock_against_host(run_glob: str = "m3_real_unit*") -> dict | None:
    """Node time against host time over the finished controller segments of run M3: the last
    cycle's node time divided by the segment's requested host duration. The driver starts its
    clock about 0.4 s after the start command, and the last logged cycle falls within about a
    second of the end, so one segment reads to about 1 s in its length."""
    runs_ = sorted(bm.runs.DATA.glob(run_glob))
    if not runs_:
        return None
    ratio, temp, length = [], [], []
    for r in csv.DictReader((runs_[-1] / "events.csv").open()):
        m = re.match(r"controller (\d+) s", r["step"])
        log = bm.runs.DATA / Path(r["detail"]).name
        if not (m and r["rc"] == "0" and log.is_file()):
            continue
        last, temps, stopped = None, [], False
        for ln in log.read_text(errors="ignore").splitlines():
            if ln.startswith("CTL,"):
                f = ln.split(",")
                if len(f) == 16:
                    last = float(f[2])
                    temps.append(float(f[15]))
            elif ln.startswith("ACK: CTL STOP"):
                stopped = True
        if stopped and last:
            ratio.append(1e-3 * last / (float(m.group(1)) + 0.4))
            temp.append(float(np.median(temps)))
            length.append(float(m.group(1)))
    if len(ratio) < 2:
        return None
    ratio = np.array(ratio)
    return dict(run=runs_[-1].name, n=len(ratio), mean=float(ratio.mean()), sd=float(ratio.std(ddof=1)),
                lo=float(ratio.min()), hi=float(ratio.max()), reading=1.0 / float(np.median(length)),
                temp_lo=min(temp), temp_hi=max(temp))


def temperature_rows(unit: str, a: dict) -> pd.DataFrame:
    """The temperature term |kappa_b dtheta| takes part of the budget; its uncertainty moves the guard."""
    kappa, u_kappa = KAPPA[unit]
    rows = []
    for d in (0.0, 2.0, 5.0, 8.0):
        x = kappa * d
        u_theta = math.hypot(TEMP_STEP_C / (2 * SQRT3), TEMP_NONLIN_REL * d / SQRT3)
        u_x = math.hypot(d * u_kappa, kappa * u_theta)
        left = BUDGET_MG - x
        rows.append(dict(dtheta_C=d, term_mg=x, guard_ms=guard(a["b0"], a["tau"], a["fv"], left),
                         u_term_mg=u_x, u_guard_ms=a["tau"] / left * u_x))
    return pd.DataFrame(rows)


def report() -> None:
    record = bm.load_record("pd")
    pd.set_option("display.width", 200, "display.max_colwidth", 80)
    for unit in ("A", "B"):
        cal = calibration_bursts(record, unit)
        a = type_a(cal)
        print(f"\n=== unit {unit}: calibration set {CAL_RUNS[unit]} ({a['n']} bursts) ===")
        print(f"b0 {a['b0']:.2f} mg (sd {a['sd_b0']:.2f}), tau {a['tau']:.4f} ms (sd {a['sd_tau']:.4f}), "
              f"t_fv {a['fv']:.3f} ms (sd {a['sd_fv']:.4f}); correlation of b0 and tau {a['corr']:+.2f}")
        print(f"guard from the medians: {a['guard']:.2f} ms")
        b = budget(unit, a)
        print(b.assign(contribution_ms=b.contribution_ms.map(lambda v: "missing" if v is None or v != v else f"{v:.4f}"))
              .to_string(index=False))
        clock = b.contribution_ms.iloc[-1]
        own = math.sqrt(sum(v * v for v in b.contribution_ms.iloc[3:-1]))
        print(f"combined standard uncertainty on the node's own clock (bootstrap row and the Type B rows "
              f"without the clock): {own:.3f} ms; expanded, k = 2: {2 * own:.2f} ms")
        if clock is not None and clock == clock:
            si = math.hypot(own, clock)
            print(f"in SI seconds, with the clock's data-sheet limits: {si:.3f} ms; expanded, k = 2: {2 * si:.2f} ms")
        print(f"law of propagation with the correlation term, for comparison with the bootstrap: {a['u_lpu']:.4f} ms")
        print(f"\nnot uncertainty of the calibration value, reported next to it:")
        print(f"  need of a single wake within the calibration session: sd {a['per_wake_sd']:.2f} ms, "
              f"95 % of wakes {a['per_wake_lo']:.2f} to {a['per_wake_hi']:.2f} ms")
        s = between_sessions(record, unit)
        print(f"  need from session to session ({s['runs']} runs of 10 bursts or more, to 2026-10-03): median "
              f"{s['median']:.2f} ms, sd {s['sd']:.2f} ms, 5-95 % {s['p05']:.2f} to {s['p95']:.2f}, "
              f"range {s['lo']:.2f} to {s['hi']:.2f} ms")
        if SLOW_MG[unit]:
            print(f"  model form: a slow term of {SLOW_MG[unit]:.2f} mg; carried in the budget it lengthens the guard by "
                  f"{a['tau'] * math.log(BUDGET_MG / (BUDGET_MG - SLOW_MG[unit])):.2f} ms "
                  f"(to {guard(a['b0'], a['tau'], a['fv'], BUDGET_MG - SLOW_MG[unit]):.2f} ms)")
        else:
            print("  model form: no slow term resolved on this unit")
        lv = settled_level(unit)
        print(f"  settled level from wake to wake (1 s sleeps; {lv['runs']} runs, {lv['bursts']} bursts): "
              f"median within-run sd {lv['median_sd']:.3f} mg, 95th percentile {lv['p95_sd']:.3f} mg")
        print("\ntemperature term (static bias kappa_b x dtheta inside the budget):")
        print(temperature_rows(unit, a).round(3).to_string(index=False))
        k, uk = KAPPA[unit]
        print(f"collapse swing (1 - delta) eps / |kappa_b| = {BUDGET_MG / k:.2f} degC, standard uncertainty "
              f"{BUDGET_MG / k * uk / k:.2f} degC from kappa_b alone")


    c = clock_against_host()
    if c:
        print(f"\nnode clock against the host computer's ({c['run']}, {c['n']} segments, die {c['temp_lo']:.1f} to "
              f"{c['temp_hi']:.1f} degC): node/host {c['mean']:.5f} (sd {c['sd']:.5f}, {c['lo']:.5f} to {c['hi']:.5f}); "
              f"one segment reads to about {100 * c['reading']:.3f} %")


if __name__ == "__main__":
    report()
