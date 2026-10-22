"""Paper 1 revision, Table II and Sections III-D and III-E: the scale-factor wake deviation and the
bias temperature coefficient, from the committed captures.

  SF wake deviation   10 +/-1 g flip pairs per mode across sleep/wake cycles (r1_sf_wake_*):
                      mean, sample SD and the 95% t-interval of the mean, per unit and mode
  kappa_b             one heat-then-cool hump per unit (twin/data/temp_log*.csv), the bivariate fit
                      of tempco.bias_tc (bias against die temperature and time); +/- the standard
                      error of the fit. Unit B's log starts with 180 s of warm-up, dropped.

Run: twin/.venv/bin/python twin/analysis/p1_sf_tempco.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

TWIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TWIN))
from analysis import tempco  # noqa: E402

DATA = TWIN.parent / "tools" / "data"
SF_RUNS = {"A": "unitA_r1_sf_wake_pd_railoff_n10_20260708_213044",
           "B": "unitB_r1_sf_wake_pd_railoff_n10_20260713_211618"}
TC_LOGS = {"A": ("temp_log.csv", 0.0), "B": ("temp_log_unitB.csv", 180.0)}


def sf_wake(unit: str) -> pd.DataFrame:
    d = pd.read_csv(DATA / SF_RUNS[unit] / "sf_wake_cycles.csv")
    rows = []
    for m in ("pd", "rail_off"):
        v = d[f"sf_wake_dev_{m}_ppm"].dropna().to_numpy()
        n, mu, sd = len(v), float(v.mean()), float(v.std(ddof=1))
        h = float(tdist.ppf(0.975, n - 1) * sd / np.sqrt(n))
        rows.append(dict(unit=unit, mode=m, n=n, mean_ppm=mu, sd_ppm=sd, ci95_lo=mu - h, ci95_hi=mu + h))
    return pd.DataFrame(rows)


def main() -> None:
    print("III-D: scale-factor wake deviation (ppm), mean, sample SD, 95% t-interval of the mean")
    print(pd.concat([sf_wake(u) for u in SF_RUNS]).round(1).to_string(index=False))
    print("\nIII-E: bias temperature coefficient kappa_b (+/- standard error of the fit)")
    for u, (name, warm) in TC_LOGS.items():
        d = np.genfromtxt(TWIN / "data" / name, delimiter=",", names=True)
        print(f"-- unit {u}: {name}, warm-up dropped {warm:g} s")
        tempco.bias_tc(d["t_s"], d["accel_g"], d["temp_C"], warmup_s=warm)


if __name__ == "__main__":
    main()
