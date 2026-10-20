"""Paper 1 revision, Table II: the measured rows, mean +/- SD over the repeats named in the table.

Recomputes, from the committed runs, the rows of Table II that are statistics over repeats
(mean +/- SD):

  P_on, P_sleep pd      30 power windows of the pd run (r1_power_*: active / sleep power)
  b0, tau, sigma_pw, pd 30 bursts (r1_burst_*, the burst tool's per-burst fits)
  t_fv pd               the same 30 bursts (first valid sample, from the wake command)
  E_inrush pd, r.o.     10 scope captures per mode at the 1.00 ohm shunt (r1_inrush_*shunt1p00*)

The rows that are not statistics over repeats (rail_off b0 and tau, fitted to the mean of 30
bursts; b_s; the guards; kappa_b; the SF deviation) come from the scripts named in the table's
Source column.

Run: twin/.venv/bin/python twin/analysis/p1_table2.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "tools" / "data"
RUNS = {
    "A": dict(power="r1_power_repeatability_unitA_r1_power_pd_railoff_n30_20260706_214002",
              bursts="r1_burst_repeatability_unitA_r1_burst_pd_n30_reprocessed_20260706_203023",
              inrush_pd="r1_inrush_repeatability_unitA_r1_inrush_pd_n10_shunt1p00_20261017_022914",
              inrush_ro="r1_inrush_repeatability_unitA_r1_inrush_rail_off_n10_shunt1p00_20261017_022916"),
    "B": dict(power="r1_power_repeatability_unitB_r1_power_pd_railoff_n30_20260712_232816",
              bursts="r1_burst_repeatability_unitB_r1_burst_pd_n30_20260712_232025",
              inrush_pd="r1_inrush_repeatability_unitB_r1_inrush_pd_n10_shunt1p00_20261017_022917",
              inrush_ro="r1_inrush_repeatability_unitB_r1_inrush_rail_off_n10_shunt1p00_20261017_022919"),
}


def stat(x: pd.Series, scale: float = 1.0) -> tuple[float, float, int]:
    x = scale * x.astype(float)
    return float(x.mean()), float(x.std(ddof=1)), int(x.size)


def rows(unit: str) -> list[tuple[str, tuple[float, float, int], str]]:
    r = RUNS[unit]
    power = pd.read_csv(DATA / r["power"] / "runs.csv")
    pd_win = power[power["mode"] == "pd"]
    bursts = pd.read_csv(DATA / r["bursts"] / "fits.csv")
    bursts = bursts[bursts["ok"].astype(str) == "True"]
    e_pd = pd.read_csv(DATA / r["inrush_pd"] / "fits.csv")
    e_ro = pd.read_csv(DATA / r["inrush_ro"] / "fits.csv")
    return [
        ("P_on (mW)", stat(pd_win["active_power_W"], 1e3), "windows"),
        ("P_sleep pd (mW)", stat(pd_win["sleep_power_W"], 1e3), "windows"),
        ("E_inrush pd (mJ)", stat(e_pd["E_inrush_mJ"]), "captures"),
        ("E_inrush r.o. (mJ)", stat(e_ro["E_inrush_mJ"]), "captures"),
        ("t_fv pd (ms)", stat(bursts["first_valid_us"], 1e-3), "bursts"),
        ("b0 pd (mg)", stat(bursts["b0_g"], 1e3), "bursts"),
        ("tau pd (ms)", stat(bursts["tau_bias_s"], 1e3), "bursts"),
        ("sigma_pw (mg)", stat(bursts["noise_g"], 1e3), "bursts"),
    ]


def main() -> None:
    for unit in RUNS:
        print(f"unit {unit}")
        for name, (m, sd, n), what in rows(unit):
            print(f"  {name:20s} {m:9.4f} +/- {sd:.4f}   ({n} {what})")


if __name__ == "__main__":
    main()
