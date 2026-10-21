"""Paper 1 revision, Section IV: the map and its bands with the slow term (decision D6, option b)
and the 1.00 ohm scope shunt (correction C-7), on top of correction C-1 (rail_off's own guard).

Same propagation as `p1_c1_railoff.py` (same seed, same pools, same draws), with three changes:

  - E_inrush pools from the re-extraction at 1.00 ohm (`*_shunt1p00_*`), same captures;
  - the rail_off need pool re-bootstrapped at the budget left by the slow term, (1-delta) eps - b_s;
  - b_s drawn in every draw from a normal distribution, 0.21 +/- 0.036 mg (Table IV), carried into
    the firmware C as `b_slow` (FW4).

`--check` runs the C-1 configuration through the same code path (b_s = 0, July energies, the
0.85 mg need pool) and must reproduce `tools/data/r3_map_uncertainty_unitA_c1/c1_numbers.csv`.

`--bandwidth` (correction C-12, 2026-10-19) reads the saved draws and writes the largest
monitoring bandwidth at which a method still saves energy, 1 / (2 (T_be + T_guard + T_probe)).
The `B_mon^max` rows of `numbers` are 1 / (2 (T_guard + T_probe)), where the sleep is zero and
the saving is negative (the surge is still paid); they are kept as computed.

`--beyond-2s` (2026-10-22) classifies the same draws on 19 more points of the map's log grid, from
2 s to 3 s, where the map figure's axis ends so that the crossover band fits; the first point,
2 s, repeats the last point of `pgreen.csv` and must match it.

Run: twin/.venv/bin/python twin/analysis/p1_map_revision.py [--check | --bandwidth | --beyond-2s]
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
import p1_c1_railoff as c1  # noqa: E402
import p1_recovery_models as models  # noqa: E402
import p1_revision_checks as chk  # noqa: E402
import r3_uncertainty as r3  # noqa: E402
from bindings import core_ffi as core  # noqa: E402

B_SLOW_MG, U_B_SLOW_MG = 0.21, 0.036
INRUSH_1P00 = {
    "inrush_pd": "r1_inrush_repeatability_unitA_r1_inrush_pd_n10_shunt1p00_",
    "inrush_ro": "r1_inrush_repeatability_unitA_r1_inrush_rail_off_n10_shunt1p00_",
}
OUTDIR = r3.REPO / "tools" / "data" / "r3_map_uncertainty_unitA_rev"
BUDGET_G = (1.0 - r3.TOL["delta"]) * r3.TOL["eps_allowed"]


def ro_need_pool_at(run: str, budget_g: float, reps: int = 1000, seed: int = r3.SEED) -> pd.DataFrame:
    """As r3.ro_need_pool (same resamples), with the need taken where the fitted single exponential
    of the mean rail_off transient reaches `budget_g`."""
    bursts = chk.burst_set(run, "rail_off")
    rng = np.random.default_rng(seed + 2)
    need = []
    for _ in range(reps):
        t, a, fv, _ = models.ensemble([bursts[j] for j in rng.integers(0, len(bursts), len(bursts))])
        p = models.fit("M1", t, a)["p"]
        b = abs(p[1])
        need.append(fv + (p[2] * np.log(b / budget_g) if b > budget_g else 0.0))
    need = np.array(need)
    return pd.DataFrame({"need_s": need[np.isfinite(need)]})


def sensor_for(pools: dict, idx: dict, i: int, mode: str):
    """r3.sensor_for with the slow term: b_s of the draw enters the firmware C as b_slow, and the
    rail_off amplitude is delayed against the pd guard computed with the same b_s."""
    tag = "ro" if mode == "rail_off" else "pd"
    b = pools["burst"].iloc[idx["burst"][i]]
    p = pools[f"power_{tag}"].iloc[idx[f"power_{tag}"][i]]
    e = pools[f"inrush_{tag}"].iloc[idx[f"inrush_{tag}"][i]]
    f = pools["sf"].iloc[idx["sf"][i]]
    bs = float(pools["bslow"].b_slow_g.iloc[idx["bslow"][i]]) if "bslow" in pools else 0.0
    b0 = b.b0_eff_g
    if mode == "rail_off" and "ro_need" in pools:
        tg_pd = b.tau_s * np.log(b.b0_eff_g / (BUDGET_G - bs))
        need = pools["ro_need"].need_s.iloc[idx["ro_need"][i]]
        b0 = core.b0_delayed(b0, b.tau_s, max(0.0, float(need - tg_pd)))
    return core.sensor(
        P_on=p.P_on_W, P_sleep=p.P_sleep_W, E_inrush=e.E_inrush_mJ * 1e-3,
        b0=b0, tau=b.tau_s, sigma_pw=b.sigma_g,
        SF=f[f"sf_post_{mode}_g_per_g"], SF_ref=f.sf_pre_g_per_g, b_slow=bs, **r3.COMMON,
    )


def pools_for(check: bool) -> dict:
    pools = r3.load_pools()
    if check:
        pools["ro_need"] = r3.ro_need_pool(c1.RO_RUN)                  # C-1 exactly
        return pools
    for key, prefix in INRUSH_1P00.items():                            # same keys, same order
        run = sorted(p for p in r3.REPO.joinpath("tools", "data").glob(prefix + "*"))[-1]
        pools[key] = pd.read_csv(run / "fits.csv")[["E_inrush_mJ"]].reset_index(drop=True)
    pools["ro_need"] = ro_need_pool_at(c1.RO_RUN, BUDGET_G - 1e-3 * B_SLOW_MG)
    rng = np.random.default_rng(r3.SEED + 3)
    pools["bslow"] = pd.DataFrame({"b_slow_g": np.clip(
        rng.normal(1e-3 * B_SLOW_MG, 1e-3 * U_B_SLOW_MG, r3.N_SCALAR), 0.0, None)})
    return pools


def numbers(res: dict) -> pd.DataFrame:
    d, o = res["draws"], res["op"]
    f = c1.fmt
    rows = [("guard, pd (ms)", f(d.T_guard_s)), ("guard, rail_off (ms)", f(d.T_guard_rail_off_s)),
            ("break-even, pd (ms)", f(d.ts_min_energy_pd_s)),
            ("break-even, rail_off (ms)", f(d.ts_min_energy_rail_off_s)),
            ("GREEN lower edge, pd (ms)", f(d.green_lo_pd_s)),
            ("GREEN lower edge, rail_off (ms)", f(d.green_lo_rail_off_s)),
            ("GREEN upper edge, pd (ms)", f(d.green_hi_pd_s, nd=1)),
            ("GREEN upper edge, rail_off (ms)", f(d.green_hi_rail_off_s, nd=1)),
            ("mode crossover (ms)", f(d.crossover_s, nd=0))]
    for b in c1.BANDWIDTHS_HZ:
        for m in ("pd", "rail_off"):
            rows.append((f"saving at {b:g} Hz, {m} (%)", f(o[f"s{b:g}_{m}"], 100, 1)))
    for m in ("pd", "rail_off"):
        rows.append((f"B_mon^max, {m} (Hz)", f(o[f"bmax_{m}"], 1, 1)))
        r = res["robust"].get(m)
        rows.append((f"robust GREEN, {m} (ms)", f"[{1e3 * r[0]:.1f}, {1e3 * r[1]:.0f}]" if r else "-"))
    return pd.DataFrame(rows, columns=["quantity", "median [2.5 %, 97.5 %]"])


def saving_bandwidth(draws: pd.DataFrame) -> pd.DataFrame:
    """Largest B_mon with a saving, per draw: the cycle whose sleep equals the energy break-even."""
    rows = []
    for m, g in (("pd", "T_guard_s"), ("rail_off", "T_guard_rail_off_s")):
        b = 1.0 / (2.0 * (draws[f"ts_min_energy_{m}_s"] + draws[g] + c1.T_PROBE))
        rows.append((f"largest B_mon with a saving, {m} (Hz)", c1.fmt(b, 1, 1)))
    return pd.DataFrame(rows, columns=["quantity", "median [2.5 %, 97.5 %]"])


TS_BEYOND = r3.TS_GRID[-1] * (r3.TS_GRID[1] / r3.TS_GRID[0]) ** np.arange(0, 20)   # 2.0 to 3.04 s


def beyond_2s() -> pd.DataFrame:
    """Classes of the map's draws from 2 s to 3 s, on the continuation of r3.TS_GRID."""
    r3.sensor_for = sensor_for
    pg = r3.pgreen_grid(pools_for(False), grid=TS_BEYOND)
    ref = pd.read_csv(OUTDIR / "pgreen.csv")
    cols = ["p_red", "p_yellow", "p_green"]
    for m in r3.MODES:                                  # the overlap point reproduces pgreen.csv
        a = pg[pg.method == m][cols].to_numpy()[0]
        b = ref[ref.method == m][cols].to_numpy()[-1]
        if not np.array_equal(a, b):
            raise SystemExit(f"beyond-2s: the 2 s point differs from pgreen.csv ({m}: {a} vs {b})")
    pg = pg[pg.T_sleep_s > r3.TS_GRID[-1] * (1 + 1e-9)].reset_index(drop=True)
    pg.to_csv(OUTDIR / "pgreen_beyond_2s.csv", index=False)
    print(pg.groupby("method")[cols].agg(["min", "max"]).to_string())
    print(f"-> {(OUTDIR / 'pgreen_beyond_2s.csv').relative_to(r3.REPO)}")
    return pg


def main() -> pd.DataFrame:
    if "--beyond-2s" in sys.argv:
        return beyond_2s()
    if "--bandwidth" in sys.argv:
        t = saving_bandwidth(pd.read_csv(OUTDIR / "draws.csv"))
        print(t.to_string(index=False))
        t.to_csv(OUTDIR / "bandwidth_with_saving.csv", index=False)
        print(f"-> {(OUTDIR / 'bandwidth_with_saving.csv').relative_to(r3.REPO)}")
        return t
    check = "--check" in sys.argv
    r3.sensor_for = sensor_for              # every draw of r3 and c1 goes through the revised builder
    pools = pools_for(check)
    res = c1.run(pools)
    t = numbers(res)
    pd.set_option("display.width", 200, "display.max_colwidth", 60)
    print(("CHECK (C-1 configuration through the revised code path)\n" if check else "") + t.to_string(index=False))
    if check:
        return t
    need = pools["ro_need"].need_s
    print(f"rail_off need at {1e3 * (BUDGET_G - 1e-3 * B_SLOW_MG):.2f} mg: median {1e3 * need.median():.2f} ms "
          f"[{1e3 * need.quantile(0.025):.2f}, {1e3 * need.quantile(0.975):.2f}]")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    res["draws"].to_csv(OUTDIR / "draws.csv", index=False)
    res["op"].to_csv(OUTDIR / "operating_draws.csv", index=False)
    res["pg"].to_csv(OUTDIR / "pgreen.csv", index=False)
    t.to_csv(OUTDIR / "numbers.csv", index=False)
    (OUTDIR / "summary.json").write_text(json.dumps(dict(
        seed=r3.SEED, rail_off_run=c1.RO_RUN, tolerances=r3.TOL, b_slow_mg=[B_SLOW_MG, U_B_SLOW_MG],
        inrush_runs=INRUSH_1P00, rail_off_need_s=r3.band(need), robust_green_s=res["robust"],
        boundaries_s={c: r3.band(res["draws"][c]) for c in res["draws"].columns if c.endswith("_s")}),
        indent=2) + "\n")
    print(f"-> {OUTDIR.relative_to(r3.REPO)}")
    return t


if __name__ == "__main__":
    main()
