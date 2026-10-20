"""Paper 1, Sections II-C and VIII: what a temperature ramp does to
the deployed guard, and the bandwidth ceiling of II-D.

The firmware C has no rate input: sm_t_guard_min sizes the guard for (1 - delta) eps - |kappa
dT| - b_slow, i.e. eq. (2) with h = 0. This script reports, from Table II's coefficient sets
(p1_transfer_revision) and the committed hysteresis ramps:

  - each ramp's loop width h and heating rate, recomputed from twin/data with the bench's own
    tempco.loop_width_mg (as hyst_rate_figure.py), checked against the numbers in Section VIII;
  - r_max as revised in (5): the largest tested rate up to which every ramp kept h within
    (1 - delta) eps - b_s;
  - the first trusted reading under the steady-temperature guard, (1 - delta) eps + h, against eps;
  - the guard that carries the loop's bound below 4 C/min in the budget, as b_slow is carried
    (firmware C, b_slow + h);
  - 1 / (2 (T_guard + T_probe)), the largest effective bandwidth of (10).

Run from twin/: .venv/bin/python analysis/p1_ramp_guard.py
"""
from __future__ import annotations

import glob
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np

TWIN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TWIN))
sys.path.insert(0, str(TWIN / "analysis"))
import p1_transfer_revision as T  # noqa: E402
from analysis import tempco as tc  # noqa: E402

EPS_MG = T.TOL["eps_allowed"] * 1e3
DELTA = T.TOL["delta"]
BUDGET_MG = T.BUDGET_G * 1e3                     # (1 - delta) eps - b_s = 0.64 mg
RATE_LIMIT = 4.0                                 # C/min, the rate Section VIII quotes
EXCLUDE = ("overheat",)                          # as hyst_rate_figure.py


def ramps() -> list[tuple[str, str, float, float]]:
    out = []
    for path in sorted(glob.glob(str(TWIN / "data" / "temp_log_unit*_hyst_*.csv"))):
        name = os.path.basename(path)
        if any(x in name for x in EXCLUDE):
            continue
        d = np.genfromtxt(path, delimiter=",", names=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = tc.loop_width_mg(d["t_s"], d["accel_g"], d["temp_C"], warmup_s=0.0)
        if not np.isfinite(r["h_mg"]) or r["monotonic_unreliable"]:
            continue
        unit = "A" if "unitA" in name else "B"
        out.append((unit, name, float(r["rate_up_C_per_min"]), float(r["h_mg"])))
    return sorted(out, key=lambda x: (x[0], x[2]))


def r_max_bracket(rs: list[tuple[str, str, float, float]], unit: str) -> tuple[float, float]:
    """(largest tested rate with every slower ramp within budget, first rate that is not)."""
    ok, first_bad = 0.0, math.inf
    for u, _, rate, h in rs:
        if u != unit:
            continue
        if h > BUDGET_MG:
            first_bad = rate
            break
        ok = rate
    return ok, first_bad


def guard_ms(c: dict, extra_g: float = 0.0) -> float:
    return T.guard(dict(c, b_slow=c["b_slow"] + extra_g)) * 1e3


def main() -> int:
    rs = ramps()
    print(f"{len(rs)} ramps (budget left after b_s: {BUDGET_MG:.2f} mg)")
    for u, name, rate, h in rs:
        print(f"  {u}  {rate:5.2f} C/min  h = {h:.2f} mg  {name}")
    slow = [h for _, _, rate, h in rs if rate <= RATE_LIMIT]
    h_bound = max(slow)
    print(f"\nh for rate <= {RATE_LIMIT:.0f} C/min: max {h_bound:.2f} mg; all ramps: max "
          f"{max(h for *_, h in rs):.2f} mg")
    for u in ("A", "B"):
        ok, bad = r_max_bracket(rs, u)
        print(f"r_max, unit {u}: >= {ok:.2f} C/min" + (f" and < {bad:.2f}" if math.isfinite(bad) else
                                                         " (no tested ramp over the budget)"))

    print(f"\nfirst trusted reading under the steady-temperature guard, (1-delta) eps + h "
          f"(eps = {EPS_MG:.2f} mg):")
    for label, h in (("h = delta eps", DELTA * EPS_MG), (f"h <= {RATE_LIMIT:.0f} C/min", h_bound),
                     ("campaign max", max(h for *_, h in rs))):
        tot = (1 - DELTA) * EPS_MG + h
        print(f"  {label:16s} h = {h:.2f} mg -> {tot:.2f} mg ({tot - EPS_MG:+.2f} against eps)")
    within = [(u, rate, h) for u, _, rate, h in rs if h <= DELTA * EPS_MG + 1e-9]
    print(f"  ramps with h <= delta eps: {within}")

    h_quoted = round(h_bound, 2)                 # the bound as Section VIII quotes it
    print(f"\nguards (ms), steady (h = 0) and carrying h = {h_quoted:.2f} mg in the budget:")
    for name, u in (("A", T.UNIT_A), ("B", T.UNIT_B)):
        for m in T.MODES:
            g0, gh = guard_ms(u[m]), guard_ms(u[m], h_quoted * 1e-3)
            bmax = 1.0 / (2.0 * (g0 * 1e-3 + u[m]["T_probe"]))
            print(f"  {name} {m:8s} steady {g0:6.2f}   with loop {gh:6.2f}   "
                  f"1/(2(Tg+Tp)) = {bmax:5.2f} Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
