"""Paper 1 correction C-1: the map with rail_off at its own measured guard.

The submitted manuscript gives both sleep methods the pd guard (12.8 ms). The 30 rail_off bursts
taken with the calibration set show that rail_off reaches the 0.85 mg budget later (14.85 ms on
unit A; verification_findings_2026-10-08.md, 2.1). This script recomputes every number of the
manuscript's map section that depends on the rail_off guard, by the same bootstrap as before
(r3_uncertainty.py, same seed and pools) with one more pool: the bootstrap distribution of the
rail_off need. Each quantity is printed three ways:

  submitted    the value printed in the submitted manuscript
  shared       this code with one guard for both methods (must reproduce `submitted`)
  own guard    this code with rail_off at its own guard (correction C-1)

Outputs go to tools/data/r3_map_uncertainty_unitA_c1 and docs/figures/safemap_banded_c1.*; the
submitted artefacts (r3_map_uncertainty_unitA_r1, safemap_banded.*) are left as they are.

Run: twin/.venv/bin/python twin/analysis/p1_c1_railoff.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TWIN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TWIN))
sys.path.insert(0, str(TWIN / "analysis"))
import a2_transfer as a2  # noqa: E402
import r3_uncertainty as r3  # noqa: E402
from bindings import core_ffi as core  # noqa: E402

RO_RUN = "r1_burst_repeatability_unitA_r1_burst_railoff_n30_supporting_20260706_211949"
OUTDIR = r3.REPO / "tools" / "data" / "r3_map_uncertainty_unitA_c1"
BANDWIDTHS_HZ = (0.5, 5.0, 20.0)
T_PROBE = r3.COMMON["T_probe"]


def operating_draws(pools: dict, n: int = r3.N_SCALAR, seed: int = r3.SEED) -> pd.DataFrame:
    """Per draw and method: guard, saving at each monitoring bandwidth, and the bandwidth at
    which the cycle is all guard and probe (the manuscript's B_mon^max)."""
    idx = r3.draw_indices(np.random.default_rng(seed), pools, n)      # same draws as r3.scalar_draws
    tol = core.tol(**r3.TOL)
    rows = []
    for i in range(n):
        row = {}
        for mode in r3.MODES:
            s = r3.sensor_for(pools, idx, i, mode)
            tg = core.t_guard_min(s, tol)
            row[f"guard_{mode}"] = tg
            row[f"bmax_{mode}"] = 1.0 / (2.0 * (tg + T_PROBE))
            for b in BANDWIDTHS_HZ:
                ts = 1.0 / (2.0 * b) - tg - T_PROBE
                row[f"s{b:g}_{mode}"] = float(core.saving(s, ts, tg)) if ts > 0 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def run(pools: dict, grid: bool = True) -> dict:
    draws = r3.scalar_draws(pools)
    op = operating_draws(pools)
    pg = r3.pgreen_grid(pools) if grid else None
    return dict(draws=draws, op=op, pg=pg,
                robust={m: r3.robust_green(pg, m) for m in r3.MODES} if grid else {})


def fmt(v: pd.Series, scale: float = 1e3, nd: int = 2) -> str:
    v = v.dropna()
    return f"{scale * v.median():.{nd}f} [{scale * v.quantile(0.025):.{nd}f}, {scale * v.quantile(0.975):.{nd}f}]"


def table(shared: dict, own: dict) -> pd.DataFrame:
    """Every number of the map section that the rail_off guard touches."""
    d0, d1, o0, o1 = shared["draws"], own["draws"], shared["op"], own["op"]
    rows = [
        ("guard, pd (ms)", "12.82 [12.54, 13.05]", fmt(d0.T_guard_s), fmt(d1.T_guard_s)),
        ("guard, rail_off (ms)", "12.8 (same as pd)", fmt(d0.T_guard_rail_off_s), fmt(d1.T_guard_rail_off_s)),
        ("break-even, pd (ms)", "1.09 [1.08, 1.10]", fmt(d0.ts_min_energy_pd_s), fmt(d1.ts_min_energy_pd_s)),
        ("break-even, rail_off (ms)", "3.50 [3.46, 3.53]", fmt(d0.ts_min_energy_rail_off_s),
         fmt(d1.ts_min_energy_rail_off_s)),
        ("C6 edge, pd (ms)", "982.2 +/- 0.3", fmt(d0.green_hi_pd_s, nd=1), fmt(d1.green_hi_pd_s, nd=1)),
        ("C6 edge, rail_off (ms)", "982.2 +/- 0.3", fmt(d0.green_hi_rail_off_s, nd=1),
         fmt(d1.green_hi_rail_off_s, nd=1)),
        ("mode crossover (ms)", "250 [183, 1380]", fmt(d0.crossover_s, nd=0), fmt(d1.crossover_s, nd=0)),
    ]
    for b, sub in zip(BANDWIDTHS_HZ, ("97.8 (one figure)", "not given", "15-24 (both methods)")):
        for m in ("pd", "rail_off"):
            rows.append((f"saving at {b:g} Hz, {m} (%)", sub, fmt(o0[f"s{b:g}_{m}"], 100, 1),
                         fmt(o1[f"s{b:g}_{m}"], 100, 1)))
    for m in ("pd", "rail_off"):
        rows.append((f"B_mon^max, {m} (Hz)", "28.1 (one figure)", fmt(o0[f"bmax_{m}"], 1, 1),
                     fmt(o1[f"bmax_{m}"], 1, 1)))
    for m, sub in (("pd", "[3.3, 827]"), ("rail_off", "[5.9, 827]")):
        r0, r1 = shared["robust"].get(m), own["robust"].get(m)
        rows.append((f"robust GREEN, {m} (ms)", sub,
                     f"[{1e3 * r0[0]:.1f}, {1e3 * r0[1]:.0f}]" if r0 else "-",
                     f"[{1e3 * r1[0]:.1f}, {1e3 * r1[1]:.0f}]" if r1 else "-"))
    return pd.DataFrame(rows, columns=["quantity", "submitted", "shared guard (recomputed)", "own guard (C-1)"])


def transfer(need_a_ms: float, needs_b_ms: dict[str, float]) -> pd.DataFrame:
    """Unit A's map on unit B with each method at its own guard. Unit B's rail_off need is the
    one measured in run M1 (October 2026); its pd coefficients are the July calibration's."""
    a_c1 = a2.rail_off_at(a2.UNIT_A, 1e-3 * need_a_ms)
    rows = []
    cases = [("shared guard (submitted)", a2.UNIT_A, a2.UNIT_B)]
    cases += [(f"own guards; B rail_off need {k} {v:.2f} ms", a_c1, a2.rail_off_at(a2.UNIT_B, 1e-3 * v))
              for k, v in needs_b_ms.items()]
    for label, ua, ub in cases:
        summ = a2.summarize(a2.transfer_table(unit_a=ua, unit_b=ub))
        for mode in a2.MODES:
            ga, gb = a2.own_guard(ua[mode]), a2.own_guard(ub[mode])
            cb = ub[mode]
            rows.append(dict(case=label, mode=mode, guard_A_ms=1e3 * ga, need_B_ms=1e3 * gb,
                             residual_B_at_A_guard_mg=1e3 * cb["b0"] * np.exp(-ga / cb["tau"]),
                             **summ[mode]))
    return pd.DataFrame(rows)


def main() -> pd.DataFrame:
    pools = r3.load_pools()
    shared = run(pools)
    pools_c1 = dict(pools, ro_need=r3.ro_need_pool(RO_RUN))          # appended last: same pd draws
    own = run(pools_c1)
    need = pools_c1["ro_need"].need_s
    print(f"rail_off need, unit A ({RO_RUN}): {len(need)} bootstrap resamples, "
          f"median {1e3 * need.median():.2f} ms [{1e3 * need.quantile(0.025):.2f}, {1e3 * need.quantile(0.975):.2f}]")
    t = table(shared, own)
    pd.set_option("display.width", 220, "display.max_colwidth", 60)
    print(t.to_string(index=False))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    own["draws"].to_csv(OUTDIR / "draws.csv", index=False)
    own["op"].to_csv(OUTDIR / "operating_draws.csv", index=False)
    own["pg"].to_csv(OUTDIR / "pgreen.csv", index=False)
    t.to_csv(OUTDIR / "c1_numbers.csv", index=False)
    (OUTDIR / "summary.json").write_text(json.dumps(dict(
        seed=r3.SEED, rail_off_run=RO_RUN, tolerances=r3.TOL,
        rail_off_need_s=r3.band(need), robust_green_s=own["robust"],
        boundaries_s={c: r3.band(own["draws"][c]) for c in own["draws"].columns if c.endswith("_s")}),
        indent=2) + "\n")
    r3.fig_safemap_banded(own["pg"], own["draws"], name="safemap_banded_c1")
    print(f"-> {OUTDIR.relative_to(r3.REPO)}; figure docs/figures/safemap_banded_c1.pdf")
    # unit B's rail_off need: run M1 on build 33facb8 (central and 95 % bound), and the larger
    # 95 % bound of the same run on build 2943d47 an hour earlier
    tr = transfer(1e3 * float(need.median()), {"central": 12.93, "95 % bound": 13.24, "largest bound": 13.45})
    print("\nTransfer: unit A's map on unit B, 800 sleep durations per method")
    print(tr.round(3).to_string(index=False))
    tr.to_csv(OUTDIR / "transfer_c1.csv", index=False)
    return t


if __name__ == "__main__":
    main()
