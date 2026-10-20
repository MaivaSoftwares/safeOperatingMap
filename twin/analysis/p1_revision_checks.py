"""Paper 1 revision: checks of manuscript claims against every burst capture on disk.

Each function checks one statement of the paper against the record. Nothing here needs the bench.

  census()                 every ok burst of every run, with the guard that burst needed
  frozen_guard()           R3-1: what a frozen calibration guard would have left, day by day
  late_first_valid()       Sec. III-F: wakes whose first valid sample arrived late, and whether
                           the 20 ms start-up guard covered them
  rail_off_transient()     Sec. IV-A: the rail_off recovery, from the 30-burst mean
  slow_term_persistence()  R3-2: is the slow term of p1_recovery_models there in every run

The guard a burst needed is t_fv + tau ln(b0 / 0.85 mg), the manuscript's Eq. (12), from the
burst tool's own single-exponential fit (paper2_runs.load_run). Per-burst fits carry fit noise
(1.6 mg per sample on a 27 to 83 mg transient), so statements about a calibration use RUN
MEDIANS, never single bursts.

Unit attribution is by folder name, with the one known misfiling corrected (finding
F-1: the a3_cold_history_unitB sessions of 2026-07-28 19:25 to 2026-07-29 00:14 are unit A).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TWIN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TWIN / "analysis"))
import p1_recovery_models as models  # noqa: E402
import paper2_runs as runs  # noqa: E402

BUDGET_MG = 0.85
TOL_MG = 1.0
FROZEN_GUARD_MS = {"A": 12.79, "B": 10.69}      # manuscript Table IV (each unit's calibration)
STARTUP_GUARD_MS = 20.0                         # manuscript Sec. III-F
FV_BAND_MS = (6.12, 6.25)                       # normal sampling grid
TAU_BAND_MS = (0.8, 2.5)
RECORD_END = "20261003"          # last capture day of the record analysed for the revision
RAIL_OFF_RUN = "r1_burst_repeatability_unitA_r1_burst_railoff_n30_supporting_20260706_211949"


def record_runs() -> list[Path]:
    """Run folders of the record the submitted manuscript's revision analysis is built on:
    captures up to RECORD_END. Later captures (runs M1 and M3 of the plan) are reported apart."""
    out = []
    for raw in sorted(runs.DATA.rglob("raw")):
        stamps = re.findall(r"(?<!\d)(20\d{6})_\d{6}", str(raw.parent.relative_to(runs.DATA)))
        if not stamps or max(stamps) <= RECORD_END:
            out.append(raw.parent)
    return out


def unit_of(run: str) -> str:
    """Unit of a run folder (relative to tools/data), with finding F-1 applied."""
    r = run.lower()
    if "a3_cold_history_unitb/session_20260728_" in r or "a3_cold_history_unitb/session_20260729_00" in r:
        return "A"
    if "unitb" in r:
        return "B"
    if "unita" in r or "_a_" in r:
        return "A"
    return "?"


def guard_need_ms(b0_mg, tau_ms, fv_ms):
    """Wake-referenced guard at which a single-exponential transient reaches the budget."""
    return np.asarray(fv_ms) + np.asarray(tau_ms) * np.log(np.maximum(b0_mg, 1e-12) / BUDGET_MG)


def residual_mg(b0_mg, tau_ms, fv_ms, guard_ms):
    """Transient left at a wake-referenced guard."""
    return np.asarray(b0_mg) * np.exp(-(guard_ms - np.asarray(fv_ms)) / np.asarray(tau_ms))


def census() -> pd.DataFrame:
    rows = []
    for d in record_runs():
        try:
            df = runs.load_run(d)
        except FileNotFoundError:
            continue
        df = df[df["ok"]].copy()
        if not len(df):
            continue
        df["run"] = str(d.relative_to(runs.DATA))
        df["t"] = runs.stamp(d)
        rows.append(df)
    a = pd.concat(rows, ignore_index=True)
    a["unit"] = a["run"].map(unit_of)
    a["need_ms"] = guard_need_ms(a["b0_mg"], a["tau_ms"], a["fv_ms"])
    a["normal"] = (a["fv_ms"].between(*FV_BAND_MS)) & (a["tau_ms"].between(*TAU_BAND_MS))
    return a


def run_medians(a: pd.DataFrame, min_bursts: int = 5) -> pd.DataFrame:
    """One row per pd run with at least min_bursts normal-grid bursts."""
    p = a[(a["mode"] == "pd") & a["normal"]]
    g = p.groupby("run").agg(unit=("unit", "first"), t=("t", "first"), n=("b0_mg", "size"),
                             b0_mg=("b0_mg", "median"), tau_ms=("tau_ms", "median"),
                             fv_ms=("fv_ms", "median"))
    g = g[g["n"] >= min_bursts].copy()
    g["need_ms"] = guard_need_ms(g["b0_mg"], g["tau_ms"], g["fv_ms"])
    g["frozen_ms"] = g["unit"].map(FROZEN_GUARD_MS)
    g["res_frozen_mg"] = residual_mg(g["b0_mg"], g["tau_ms"], g["fv_ms"], g["frozen_ms"])
    g["res_startup_mg"] = residual_mg(g["b0_mg"], g["tau_ms"], g["fv_ms"], STARTUP_GUARD_MS)
    return g.sort_values("t")


def frozen_guard(a: pd.DataFrame) -> None:
    g = run_medians(a)
    print("\n== R3-1: a frozen calibration guard against the transient measured later (run medians)")
    for u in ("A", "B"):
        q = g[g["unit"] == u]
        if not len(q):
            continue
        days = q["t"].dt.normalize().nunique()
        print(f"unit {u}: {len(q)} runs on {days} days, {q['t'].min():%Y-%m-%d} to {q['t'].max():%Y-%m-%d}; "
              f"frozen guard {FROZEN_GUARD_MS[u]} ms")
        print(f"  b0 run medians {q['b0_mg'].min():.1f} to {q['b0_mg'].max():.1f} mg "
              f"(x{q['b0_mg'].max() / q['b0_mg'].min():.1f}); tau {q['tau_ms'].min():.3f} to {q['tau_ms'].max():.3f} ms")
        print(f"  guard needed {q['need_ms'].min():.2f} to {q['need_ms'].max():.2f} ms")
        over_b, over_t = (q["res_frozen_mg"] > BUDGET_MG), (q["res_frozen_mg"] > TOL_MG)
        print(f"  frozen guard leaves more than the 0.85 mg budget in {over_b.sum()} of {len(q)} runs, "
              f"more than the 1 mg tolerance in {over_t.sum()} ({100 * over_t.mean():.0f} %); "
              f"worst {q['res_frozen_mg'].max():.2f} mg")
        print(f"  20 ms start-up guard: worst residual {q['res_startup_mg'].max():.4f} mg")
        d = q.groupby(q["t"].dt.strftime("%m-%d")).agg(
            runs=("n", "size"), b0=("b0_mg", "median"), tau=("tau_ms", "median"),
            need_max=("need_ms", "max"), res_max=("res_frozen_mg", "max"))
        print(d.round(2).to_string())


def late_first_valid(a: pd.DataFrame) -> None:
    p = a[a["mode"] == "pd"]
    late = p[p["fv_ms"] > FV_BAND_MS[1]]
    print(f"\n== Sec. III-F: pd bursts {len(p)}; first valid later than {FV_BAND_MS[1]} ms in {len(late)}")
    print("  first-valid times (ms):", np.round(np.sort(late["fv_ms"].to_numpy()), 2))
    over = p[p["need_ms"] > STARTUP_GUARD_MS]
    print(f"  bursts needing more than the {STARTUP_GUARD_MS:.0f} ms start-up guard: {len(over)} "
          f"({(over['fv_ms'] > FV_BAND_MS[1]).sum()} of them with a late first valid)")
    cols = ["run", "raw_file", "b0_mg", "tau_ms", "fv_ms", "need_ms"]
    print(over.sort_values("need_ms", ascending=False)[cols].round(2).to_string(index=False))
    after_fv = p["need_ms"] - p["fv_ms"]
    ok_tau = p["tau_ms"].between(*TAU_BAND_MS)
    print(f"  time needed AFTER the first valid sample (tau in band): max {after_fv[ok_tau].max():.2f} ms, "
          f"99.9th pct {np.percentile(after_fv[ok_tau], 99.9):.2f} ms")


def burst_set(run: str, mode: str) -> list:
    """(time from first valid, acceleration, first valid) of every complete burst of one mode."""
    tool = runs._tool()
    bursts = []
    for f in sorted((runs.DATA / run / "raw").glob(f"burst_{mode}_*.txt")):
        lines = f.read_text(errors="ignore").splitlines()
        if not any(ln.startswith("BURST_END") for ln in lines):
            continue
        s, _ = tool.usable_samples(lines, require_err0=True)
        if len(s) < 100:
            continue
        t = np.array([v[0] for v in s])
        bursts.append((t - t[0], np.array([v[1] for v in s]), float(t[0])))
    return bursts


def budget_crossing_ms(bursts: list) -> float:
    """Time after the wake command at which the mean transient of the bursts reaches the budget."""
    t, a, fv, _ = models.ensemble(bursts)
    return 1e3 * (fv + models.guard_after_fv("M1", models.fit("M1", t, a)["p"], t[-1]))


def rail_off_delay(ro_run: str, pd_runs: tuple[str, ...], reps: int = 500, seed: int = 20261012) -> dict:
    """The rail_off delay of one session, by the rule fixed before run M1 was repeated.

    Need: the 95th percentile of the rail_off budget-crossing time over bootstrap resamples of
    the bursts. Delay: that need minus the lowest pd guard (crossing of the mean pd transient)
    among the pd blocks of the session, rounded up to 0.05 ms.
    """
    ro = burst_set(ro_run, "rail_off")
    rng = np.random.default_rng(seed)
    boot = np.array([budget_crossing_ms([ro[i] for i in rng.integers(0, len(ro), len(ro))])
                     for _ in range(reps)])
    boot = boot[np.isfinite(boot)]
    pd_guards = [budget_crossing_ms(burst_set(r, "pd")) for r in pd_runs]
    need95 = float(np.percentile(boot, 95))
    return dict(n=len(ro), need_ms=budget_crossing_ms(ro), need_p05=float(np.percentile(boot, 5)),
                need_p95=need95, n_boot=len(boot), pd_guards_ms=pd_guards,
                delay_ms=float(np.ceil((need95 - min(pd_guards)) / 0.05 - 1e-9) * 0.05))


def rail_off_transient() -> dict:
    bursts = burst_set(RAIL_OFF_RUN, "rail_off")
    t, a, fv, _ = models.ensemble(bursts)
    c = float(a[len(a) // 2:].mean())
    f1 = models.fit("M1", t, a)
    out = dict(n=len(bursts), fv_ms=1e3 * fv, b0_mg=1e3 * abs(f1["p"][1]), tau_ms=1e3 * f1["p"][2],
               noise_mg=1e3 * float(a[len(a) // 2:].std()),
               guard_ms=1e3 * (fv + models.guard_after_fv("M1", f1["p"], t[-1])))
    print(f"\n== Sec. IV-A: rail_off recovery, mean of {out['n']} bursts ({RAIL_OFF_RUN})")
    print(f"  first valid {out['fv_ms']:.2f} ms; amplitude there {out['b0_mg']:.2f} mg; tau {out['tau_ms']:.3f} ms; "
          f"noise of the mean {out['noise_mg']:.2f} mg")
    print(f"  reaches the 0.85 mg budget {out['guard_ms']:.2f} ms after the wake command")
    for g in (12.8, 14.6, 15.4, 20.0):
        r = 1e3 * abs(models.transient("M1", f1["p"], np.array([g * 1e-3 - fv]))[0])
        k = (t + fv >= g * 1e-3) & (t + fv < g * 1e-3 + 5e-3)
        out[f"res_{g}"] = r
        print(f"  at {g:5.1f} ms: fitted residual {r:.2f} mg; measured mean over the next 5 ms "
              f"{1e3 * (a[k].mean() - c):+.2f} mg (se {1e3 * a[k].std() / np.sqrt(k.sum()):.2f})")
    return out


def slow_term_persistence(min_bursts: int = 20) -> pd.DataFrame:
    rows = []
    for d in record_runs():
        name = str(d.relative_to(runs.DATA))
        try:
            b = [x for x in models.load_bursts(name)
                 if FV_BAND_MS[0] * 1e-3 <= x[2] <= FV_BAND_MS[1] * 1e-3 and len(x[0]) >= 1900]
        except (FileNotFoundError, ValueError):
            continue
        if len(b) < min_bursts:
            continue
        t, a, fv, _ = models.ensemble(b)
        f1, f2 = models.fit("M1", t, a), models.fit("M2", t, a)
        g1 = models.guard_after_fv("M1", f1["p"], t[-1])
        rows.append(dict(unit=unit_of(name), run=name, n=len(b), b0_mg=1e3 * abs(f1["p"][1]),
                         slow_mg=1e3 * abs(f2["p"][3]), tau2_ms=1e3 * f2["p"][4],
                         same_sign=bool(np.sign(f2["p"][3]) == np.sign(f2["p"][1])),
                         dbic=f2["bic"] - f1["bic"],
                         at_g1_mg=1e3 * abs(models.transient("M2", f2["p"], np.array([g1]))[0]),
                         dguard_ms=1e3 * (models.guard_after_fv("M2", f2["p"], t[-1]) - g1)))
    df = pd.DataFrame(rows)
    print(f"\n== R3-2: slow term in every pd run with at least {min_bursts} normal-grid bursts ({len(df)} runs)")
    for u in ("A", "B"):
        q = df[df["unit"] == u]
        if not len(q):
            continue
        sup = q[q["dbic"] < 0]
        print(f"unit {u}: {len(q)} runs; two-term model has the lower BIC in {len(sup)}; "
              f"slow term has the fast term's sign in {int(q['same_sign'].sum())}")
        if len(sup):
            print(f"  where supported: amplitude median {sup['slow_mg'].median():.2f} mg "
                  f"(10th to 90th pct {sup['slow_mg'].quantile(.1):.2f} to {sup['slow_mg'].quantile(.9):.2f}); "
                  f"tau2 median {sup['tau2_ms'].median():.0f} ms")
        print(f"  bias at the run's own single-exponential guard under the two-term model: "
              f"median {q['at_g1_mg'].median():.2f}, max {q['at_g1_mg'].max():.2f} mg")
    return df


if __name__ == "__main__":
    a = census()
    print(f"bursts {len(a)} in {a['run'].nunique()} runs, {a['t'].min():%Y-%m-%d} to {a['t'].max():%Y-%m-%d}; "
          f"by mode {a['mode'].value_counts().to_dict()}")
    frozen_guard(a)
    late_first_valid(a)
    rail_off_transient()
    slow_term_persistence()
