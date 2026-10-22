"""Paper 1 revision, Section VI (Reviewer 1, comment 6: "Has the precision improved?"): the scatter of
single readings after the guard against the settled output of the same wakes, and the wake-to-wake
repeatability of the settled level, from the 30-burst calibration sets of Table II.

For each burst the readings are taken about that burst's settled level (the mean of its second
half); the transient common to the 30 bursts is removed by subtracting their mean at each sample,
so what remains is the scatter of single readings at the same time after the wake. Its standard
deviation is compared over the 5 ms and the 20 ms after the guard of Table II with the settled
output from 0.3 s to the end of the capture. The ratio's 95% band for equal variances is the F
band of the two windows (readings taken as independent: the noise is white over the readout band,
Section III-C).

Run: twin/.venv/bin/python twin/analysis/p1_precision.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import f as fdist

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p1_recovery_models as M  # noqa: E402

GUARD_S = {"A": 13.22e-3, "B": 11.05e-3}          # Table II, pd
WINDOWS = {"guard to +5 ms": (0.0, 5e-3), "guard to +20 ms": (0.0, 20e-3)}
TAIL_FROM_S = 0.3                                  # settled output: 0.3 s after first valid to the end


def scatter(unit: str) -> dict:
    bursts = M.load_bursts(M.RUNS[unit])
    n = min(len(t) for t, _, _ in bursts)
    t = np.mean([b[0][:n] for b in bursts], axis=0)
    fv = float(np.mean([b[2] for b in bursts]))
    dev = np.array([b[1][:n] - b[1][:n][n // 2:].mean() for b in bursts]) * 1e3      # mg
    res = dev - dev.mean(axis=0)
    k = len(bursts) / (len(bursts) - 1)
    t0 = GUARD_S[unit] - fv

    def sd(lo: float, hi: float) -> tuple[float, int]:
        v = res[:, (t >= lo) & (t < hi)].ravel()
        return float(np.sqrt(k * np.mean(v ** 2))), int(v.size)

    out = {"bursts": len(bursts), "tail": sd(TAIL_FROM_S, t[-1] + 1.0)}
    for name, (a, b) in WINDOWS.items():
        out[name] = sd(t0 + a, t0 + b)
    level = np.array([b[1][:n][n // 2:].mean() for b in bursts]) * 1e3
    out["settled level SD"] = float(level.std(ddof=1))
    return out


def main() -> None:
    for u in ("A", "B"):
        r = scatter(u)
        st, nt = r["tail"]
        print(f"unit {u}: {r['bursts']} bursts; settled output (from {TAIL_FROM_S:g} s): "
              f"single-reading SD {st:.3f} mg ({nt} readings)")
        for name in WINDOWS:
            s, n = r[name]
            lo, hi = (float(np.sqrt(fdist.ppf(q, n - 1, nt - 1))) for q in (0.025, 0.975))
            print(f"  {name:16s} SD {s:.3f} mg ({n} readings); ratio to settled {s / st:.3f} "
                  f"(equal-variance 95% band {lo:.3f} to {hi:.3f})")
        print(f"  settled level from wake to wake: SD {r['settled level SD']:.3f} mg")


if __name__ == "__main__":
    main()
