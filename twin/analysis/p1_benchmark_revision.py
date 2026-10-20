"""Paper 1 revision, Section VI: the benchmark of `p1_benchmark.py` with the slow term (decision
D6, option b: 0.21 mg on both units, so the transient is left (1 - delta) eps - b_s = 0.64 mg)
and the 1.00 ohm scope shunt (correction C-7: E_inrush from the re-extraction at 1.00 ohm).

What changes from p1_benchmark:
  - the budget, 0.85 -> 0.64 mg, everywhere it enters: the guards of B4 and B5, the base
    threshold and the setting of the slope rule B6, and the waits of Result 3. The slow term
    is then in the budget, so SLOW_MG is zero (no separate "with slow-term budget" row);
  - E_inrush in the energy rows: the mean of each re-extraction at 1.00 ohm (Table II
    revision T2-2; unit A's means equal its medians at the printed precision);
  - two energy rows at the guards of Table II (13.22 ms unit A, 11.05 ms unit B), the values
    Section IV uses, beside the calibration-day medians of the benchmark.
The trust times of B1, B2, B3, 16 ms, 20 ms and 40 ms do not depend on the budget, so their
bias rows are those of the benchmark.

--check runs p1_benchmark as committed (0.85 mg, July energies), to be compared number by
number with the benchmark's first run.

--fine sets the slope rule's thresholds on a finer grid (factors 1 to 16 in steps of about
1.2 times, windows 4 to 24 samples), so that its calibration-day setting is not held off the
budget by the factor-of-two steps of the benchmark's grid.

--moving (implies --fine) gives Table VI's row "same setting, sensor moving" (R6S-4,
2026-10-21): the slope rule at its calibration-day setting with a simulated 50 mg, 20 Hz input,
in the table's own terms, for the phase draw of slope_under_input (seed 0) and five more.

Run: twin/.venv/bin/python twin/analysis/p1_benchmark_revision.py [--check | --fine | --moving]
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

TWIN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TWIN))
sys.path.insert(0, str(TWIN / "analysis"))
import p1_benchmark as bm  # noqa: E402

B_SLOW_MG = 0.21
TABLE_II_GUARD_S = {"A": 13.22e-3, "B": 11.05e-3}
FINE_WINDOWS = (4, 5, 6, 7, 8, 10, 12, 16, 24)
FINE_FACTORS = tuple(round(2 ** (k / 4), 3) for k in range(17))   # 1 to 16, x1.19 per step
DATA = TWIN.parent / "tools" / "data"
INRUSH_1P00 = "r1_inrush_repeatability_unit{u}_r1_inrush_{m}_n10_shunt1p00_"
MOVING_INPUT = (20.0, 50.0)            # Hz, mg: the input of Table VI's moving row
MOVING_SEEDS = range(6)                # seed 0 is slope_under_input's draw; 1 to 5 give the spread


def configure() -> dict:
    """Set the revised budget and energies in p1_benchmark; return the E_inrush used (mJ)."""
    bm.BUDGET_MG = 0.85 - B_SLOW_MG
    bm.SLOW_MG = {"A": 0.0, "B": 0.0}
    used, coeffs = {}, {}
    for u in ("A", "B"):
        coeffs[u] = {}
        for m in ("pd", "rail_off"):
            run = sorted(DATA.glob(INRUSH_1P00.format(u=u, m=m) + "*"))[-1]
            e = float(pd.read_csv(run / "fits.csv").E_inrush_mJ.mean())
            used[(u, m)] = (e, run.name)
            coeffs[u][m] = dict(bm.COEFFS[u][m], E_inrush=1e-3 * e)
    bm.COEFFS = coeffs
    return used


def slope_moving(record: list, unit: str, freq_hz: float, amp_mg: float, seed: int = 0) -> dict:
    """The slope rule at its calibration-day setting with a simulated input, in Table VI's terms:
    mean wait, run-mean bias at first trust, runs above tolerance by two standard errors, the
    share of wakes trusted with more than the tolerance of fitted transient, and the energy row
    at the mean wait.

    The input is that of bm.slope_under_input: a sinusoid of random phase added to each recorded
    wake, assuming the transient adds to it unchanged. Its phases are drawn in the same order
    (the draws of the inputs that come before this one in slope_under_input are skipped), so
    seed 0 reproduces its wait and share. The input is signal, so the error of the trusted
    reading is the wake's own deviation at that moment, read from the recorded burst.
    """
    c = bm.slope_calibrate(record, unit)
    groups = bm.by_run(record, unit)
    n_wakes = sum(len(bs) for bs in groups.values())
    inputs = [(0.0, 0.0)] + [(f, a) for f in bm.INPUT_FREQS_HZ for a in bm.INPUT_AMPS_MG]
    rng = np.random.default_rng(seed)
    for _ in range(inputs.index((freq_hz, amp_mg)) * n_wakes):
        rng.uniform(0, 2 * np.pi)
    w = 2 * np.pi * freq_hz
    trust = bm.slope_trust(c["thr"], c["window"])
    waits, over, undeclared, run_mean, run_se = [], [], 0, [], []
    for bs in groups.values():
        meas = []
        for b in bs:
            moved = replace(b, dev=b.dev + amp_mg * np.sin(w * b.t + rng.uniform(0, 2 * np.pi)))
            undeclared += bm.slope_detect(moved, c["thr"], c["window"]) < 0
            t = trust(moved, [])
            waits.append(t)
            over.append(b.fitted(t) > bm.TOL_MG)
            meas.append(b.sample(t)[1])
        if len(bs) >= bm.MIN_RUN_BURSTS and np.all(np.isfinite(meas)):
            run_mean.append(float(np.mean(meas)))
            run_se.append(float(np.std(meas, ddof=1) / np.sqrt(len(meas))))
    rm, rse = np.array(run_mean), np.array(run_se)
    wait = float(np.mean(waits))
    return dict(unit=unit, seed=seed, wait_mean_ms=1e3 * wait, undeclared=int(undeclared),
                wakes_over=float(np.mean(over)), meas_mean=float(rm.mean()),
                meas_mean_se=float(rm.std(ddof=1) / np.sqrt(len(rm))), n_runs_meas=len(rm),
                meas_over_2se=int(np.sum(np.abs(rm) - 2 * rse > bm.TOL_MG)),
                **{k: v for k, v in bm.energy_row(unit, "pd", wait).items() if k != "guard_ms"})


def main() -> None:
    if "--check" in sys.argv:
        print("CHECK: p1_benchmark as committed (budget 0.85 mg, July energies)")
        bm.report()
        return
    used = configure()
    if "--moving" in sys.argv:
        bm.SLOPE_WINDOWS, bm.SLOPE_FACTORS = FINE_WINDOWS, FINE_FACTORS
        f, a = MOVING_INPUT
        print(f"MOVING: slope rule (fine grid, calibration-day setting) with a simulated {a:g} mg, "
              f"{f:g} Hz input; budget {bm.BUDGET_MG:.2f} mg; E_inrush at 1.00 ohm")
        record = bm.load_record("pd")
        rows = [slope_moving(record, u, 0.0, 0.0) for u in ("A", "B")]
        rows += [slope_moving(record, u, f, a, s) for u in ("A", "B") for s in MOVING_SEEDS]
        print(pd.DataFrame(rows).round(4).to_string(index=False))
        return
    if "--fine" in sys.argv:
        bm.SLOPE_WINDOWS, bm.SLOPE_FACTORS = FINE_WINDOWS, FINE_FACTORS
        print(f"FINE GRID for the slope rule: windows {FINE_WINDOWS}, factors {FINE_FACTORS}")
    print(f"REVISION: budget {bm.BUDGET_MG:.2f} mg (slow term {B_SLOW_MG} mg on both units); "
          "E_inrush at 1.00 ohm (mean of 10):")
    for (u, m), (e, run) in used.items():
        print(f"  unit {u} {m:8s} {e:.4f} mJ   {run}")
    bm.report()
    print("\nenergy, pd mode, at the guards of Table II")
    rows = [dict(unit=u, method="map guard, Table II", **bm.energy_row(u, "pd", g))
            for u, g in TABLE_II_GUARD_S.items()]
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
