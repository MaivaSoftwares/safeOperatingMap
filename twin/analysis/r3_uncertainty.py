"""R3 — propagate the Campaign R1 coefficient distributions through the firmware C.

Every safe-map boundary today is one number from one measurement. R1 measured each
coefficient as a DISTRIBUTION (unit A, within-session, 2026-07-06..08). This module
bootstrap-samples those distributions and pushes every sample through the EXACT
firmware map engine (libcore.so via cffi — no re-implementation), so each boundary
becomes a band and GREEN becomes "GREEN at confidence 1-alpha":

    T_guard_min, T_sleep_min (break-even), green interval [lo, hi],
    the pd<->rail_off crossover, and P(GREEN | T_sleep) per mode.

Conventions (do not mix — SENSOR_COEFFICIENTS.md 2026-07-02):
  * b0 is WAKE-REFERENCED here: b0_eff = b0_fit * exp(t_first_valid / tau), so the
    C's T_guard = tau*ln(b0/((1-delta)*eps)) returns the TOTAL guard from the wake
    command (~12.8 ms this session), not the 6.6 ms past-first-sample component.
  * rail_off recovery (b0, tau, sigma) is tied to the pd-observed model (the
    rail_off transient hides behind the 11.4 ms init blind window); rail_off keeps
    its own measured P_sleep floor and E_inrush.
  * The band is WITHIN-SESSION repeatability only. Between-session drift (R2) is a
    separate, larger term and is deliberately NOT folded in.

Deterministic: fixed RNG seed, fixed output dir. Regenerate with
    cd twin && .venv/bin/python -m analysis.r3_uncertainty
Outputs: tools/data/r3_map_uncertainty_unitA_r1/{summary.json,draws.csv,pgreen.csv}
         docs/figures/safemap_banded.{pdf,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt           # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.optimize import brentq         # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # twin/
from bindings import core_ffi as core     # noqa: E402

TWIN = Path(__file__).resolve().parent.parent
REPO = TWIN.parent
FIGDIR = REPO / "docs" / "figures"
OUTDIR = REPO / "tools" / "data" / "r3_map_uncertainty_unitA_r1"

# ---- R1 source datasets (committed; campaign_r_uncertainty_results_2026-07-06.md) ----
R1 = {
    "burst":     REPO / "tools/data/r1_burst_repeatability_unitA_r1_burst_pd_n30_reprocessed_20260706_203023",
    "power":     REPO / "tools/data/r1_power_repeatability_unitA_r1_power_pd_railoff_n30_20260706_214002",
    "inrush_pd": REPO / "tools/data/r1_inrush_repeatability_unitA_r1_inrush_pd_n10_v2_20260707_204730",
    "inrush_ro": REPO / "tools/data/r1_inrush_repeatability_unitA_r1_inrush_railoff_n10_20260707_204927",
    "sf":        REPO / "tools/data/unitA_r1_sf_wake_pd_railoff_n10_20260708_213044",
}

MODES = ("rail_off", "pd")
SEED = 20260708
N_SCALAR = 10_000        # bootstrap draws for scalar boundaries
N_GRID = 2_000           # bootstrap draws for the P(GREEN | T_sleep) grid
TS_GRID = np.geomspace(3e-4, 2.0, 400)   # same axis as fig_safemap; starts below the pd break-even so RED is visible
ALPHA = 0.05             # robust GREEN = GREEN in >= 1-ALPHA of draws

# rail_off sleep power measured 0 by the rail monitor; keep the conservative
# leakage floor used by the chapter figures (figures.py MODES, campaign doc)
RAIL_OFF_PSLEEP_FLOOR = 2e-5

# fixed (non-R1) sensor fields — chapter figure conventions (figures.py COMMON)
COMMON = dict(fs=175.0, T_probe=0.005, a_max_exp=1.0, range_limit=30.0,
              bias_tc=-75.5e-6, dT_op=0.0)
# same tolerance set as fig_safemap (the chapter's Fig. 4)
TOL = dict(eps_allowed=0.001, sigma_allowed=0.0025, SF_allowed=0.0255,
           B_required=0.5, D_max=100.0, S_min=0.10, delta=0.15)

COLOR = {"rail_off": "#0072B2", "pd": "#D55E00",
         # colorblind-safe traffic light: blue-shifted green vs red-orange are
         # separable under deuteranopia/protanopia (unlike a pure green/red pair)
         "GREEN": "#009E73", "YELLOW": "#F0C808", "RED": "#CC3311"}


# ================================================================================================
def load_pools() -> dict[str, pd.DataFrame]:
    """Per-experiment row pools; joint row-resampling preserves within-run correlations."""
    burst = pd.read_csv(R1["burst"] / "fits.csv")
    burst = burst[burst["ok"] == True]  # noqa: E712
    pools = {
        "burst": pd.DataFrame({
            # wake-referenced initial bias (see module docstring)
            "b0_eff_g": burst.b0_g * np.exp(burst.first_valid_us * 1e-6 / burst.tau_bias_s),
            "tau_s": burst.tau_bias_s,
            "sigma_g": burst.noise_g,
        }).reset_index(drop=True),
        "sf": pd.read_csv(R1["sf"] / "sf_wake_cycles.csv")[
            ["sf_pre_g_per_g", "sf_post_pd_g_per_g", "sf_post_rail_off_g_per_g"]
        ].reset_index(drop=True),
    }
    power = pd.read_csv(R1["power"] / "runs.csv")
    for mode, tag in (("pd", "pd"), ("rail_off", "ro")):
        p = power[power["mode"] == mode]
        pools[f"power_{tag}"] = pd.DataFrame({
            "P_on_W": p.active_power_W,
            "P_sleep_W": p.sleep_power_W.clip(lower=RAIL_OFF_PSLEEP_FLOOR if mode == "rail_off" else 0.0),
        }).reset_index(drop=True)
        pools[f"inrush_{tag}"] = (
            pd.read_csv(R1[f"inrush_{tag}"] / "fits.csv")[["E_inrush_mJ"]].reset_index(drop=True)
        )
    return pools


def draw_indices(rng: np.random.Generator, pools: dict, n: int) -> dict[str, np.ndarray]:
    return {name: rng.integers(0, len(df), size=n) for name, df in pools.items()}


def sensor_for(pools: dict, idx: dict, i: int, mode: str):
    tag = "ro" if mode == "rail_off" else "pd"
    b = pools["burst"].iloc[idx["burst"][i]]
    p = pools[f"power_{tag}"].iloc[idx[f"power_{tag}"][i]]
    e = pools[f"inrush_{tag}"].iloc[idx[f"inrush_{tag}"][i]]
    f = pools["sf"].iloc[idx["sf"][i]]
    b0 = b.b0_eff_g
    if mode == "rail_off" and "ro_need" in pools:
        # correction C-1: rail_off has its own measured need. Its amplitude is pd's delayed by
        # (rail_off need - pd guard of this draw), so the C returns that need as rail_off's guard
        tg_pd = b.tau_s * np.log(b.b0_eff_g / ((1.0 - TOL["delta"]) * TOL["eps_allowed"]))
        need = pools["ro_need"].need_s.iloc[idx["ro_need"][i]]
        b0 = core.b0_delayed(b0, b.tau_s, max(0.0, float(need - tg_pd)))
    return core.sensor(
        P_on=p.P_on_W, P_sleep=p.P_sleep_W, E_inrush=e.E_inrush_mJ * 1e-3,
        b0=b0, tau=b.tau_s, sigma_pw=b.sigma_g,
        SF=f[f"sf_post_{mode}_g_per_g"], SF_ref=f.sf_pre_g_per_g, **COMMON,
    )


def ro_need_pool(run: str, reps: int = 1000, seed: int = SEED) -> pd.DataFrame:
    """Bootstrap distribution of the rail_off need (time from the wake command at which the mean
    rail_off transient reaches the budget), over resamples of the run's rail_off bursts."""
    sys.path.insert(0, str(TWIN / "analysis"))
    import p1_revision_checks as chk
    bursts = chk.burst_set(run, "rail_off")
    rng = np.random.default_rng(seed + 2)
    need = [chk.budget_crossing_ms([bursts[j] for j in rng.integers(0, len(bursts), len(bursts))])
            for _ in range(reps)]
    need = np.array(need)
    return pd.DataFrame({"need_s": 1e-3 * need[np.isfinite(need)]})


def crossover_s(s_ro, s_pd, tg_ro: float, tg_pd: float | None = None) -> float:
    """T_sleep where rail_off and pd average power curves cross (firmware sm_p_avg), each mode
    at its own guard. With one guard given, both modes use it (the submitted manuscript)."""
    tg_pd = tg_ro if tg_pd is None else tg_pd
    f = lambda ts: core.p_avg(s_ro, ts, tg_ro) - core.p_avg(s_pd, ts, tg_pd)  # noqa: E731
    lo, hi = 1e-4, 5.0
    if f(lo) * f(hi) > 0:
        return float("nan")
    return brentq(f, lo, hi, xtol=1e-6)


# ================================================================================================
def scalar_draws(pools: dict, n: int = N_SCALAR, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = draw_indices(rng, pools, n)
    t = core.tol(**TOL)
    rows = []
    for i in range(n):
        s_pd = sensor_for(pools, idx, i, "pd")
        s_ro = sensor_for(pools, idx, i, "rail_off")
        tg = core.t_guard_min(s_pd, t)     # pd guard; rail_off's equals it unless "ro_need" is given
        tg_ro = core.t_guard_min(s_ro, t)
        row = {"T_guard_s": tg, "T_guard_rail_off_s": tg_ro,
               "crossover_s": crossover_s(s_ro, s_pd, tg_ro, tg)}
        for mode, s in (("pd", s_pd), ("rail_off", s_ro)):
            row[f"ts_min_energy_{mode}_s"] = core.t_sleep_min(s)
            g = core.green_interval(s, t)
            row[f"green_lo_{mode}_s"] = g[0] if g else float("nan")
            row[f"green_hi_{mode}_s"] = g[1] if g else float("nan")
            row[f"feasible_{mode}"] = int(g is not None)
        rows.append(row)
    return pd.DataFrame(rows)


def pgreen_grid(pools: dict, n: int = N_GRID, seed: int = SEED) -> pd.DataFrame:
    """P(class | T_sleep) per mode over TS_GRID, via the firmware classifier."""
    rng = np.random.default_rng(seed + 1)
    idx = draw_indices(rng, pools, n)
    t = core.tol(**TOL)
    counts = {m: np.zeros((3, len(TS_GRID)), dtype=np.int64) for m in MODES}
    code = {"RED": 0, "YELLOW": 1, "GREEN": 2}
    for i in range(n):
        for mode in MODES:
            s = sensor_for(pools, idx, i, mode)
            for j, ts in enumerate(TS_GRID):
                counts[mode][code[core.classify(s, t, float(ts))], j] += 1
    rows = []
    for mode in MODES:
        for j, ts in enumerate(TS_GRID):
            rows.append({"method": mode, "T_sleep_s": float(ts),
                         "p_red": counts[mode][0, j] / n,
                         "p_yellow": counts[mode][1, j] / n,
                         "p_green": counts[mode][2, j] / n})
    return pd.DataFrame(rows)


def robust_green(pg: pd.DataFrame, mode: str, alpha: float = ALPHA) -> tuple | None:
    """Largest contiguous T_sleep run with P(GREEN) >= 1-alpha."""
    d = pg[pg.method == mode]
    ok = (d.p_green >= 1.0 - alpha).to_numpy()
    if not ok.any():
        return None
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    starts, ends = edges[::2], edges[1::2] - 1
    k = int(np.argmax(ends - starts))
    ts = d.T_sleep_s.to_numpy()
    return float(ts[starts[k]]), float(ts[ends[k]])


def band(series: pd.Series) -> dict:
    v = series.dropna()
    return {"median": float(v.median()), "p2_5": float(v.quantile(0.025)),
            "p97_5": float(v.quantile(0.975)), "p5": float(v.quantile(0.05)),
            "p95": float(v.quantile(0.95)), "mean": float(v.mean()),
            "std": float(v.std(ddof=1)), "n": int(len(v))}


# ================================================================================================
def fig_safemap_banded(pg: pd.DataFrame, draws: pd.DataFrame, name: str = "safemap_banded") -> None:
    """Chapter Fig. 4 with measurement-uncertainty bands: median-coefficient class map,
    hatched where the classification is uncertain, robust-GREEN edges annotated."""
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.titlesize": 8,
        "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "figure.dpi": 150, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.25, "lines.linewidth": 1.4,
    })
    fig, (ax, axb) = plt.subplots(
        2, 1, figsize=(3.5, 4.1), sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.32})

    # -- top: the map, hatched where uncertain ------------------------------------------------
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([COLOR["RED"], COLOR["YELLOW"], COLOR["GREEN"]])
    # modal class per point (majority across draws)
    grid = np.array([
        pg[pg.method == m][["p_red", "p_yellow", "p_green"]].to_numpy().argmax(axis=1)
        for m in MODES])
    ax.pcolormesh(TS_GRID * 1e3, [0.5, 1.5], grid, cmap=cmap, vmin=0, vmax=2, shading="nearest", edgecolors="face", linewidth=0.3, antialiased=False)
    ax.axhline(1.0, color="white", lw=3)
    ax.set_xscale("log")
    ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(list(MODES))
    ax.set_ylim(0.0, 2.0)
    for row, m in enumerate(MODES):
        d = pg[pg.method == m]
        conf = d[["p_red", "p_yellow", "p_green"]].to_numpy().max(axis=1)
        unc = conf < (1.0 - ALPHA)          # classification not settled at 1-alpha
        for st, en in _runs(unc):
            x0, x1 = TS_GRID[st] * 1e3, TS_GRID[en] * 1e3
            ax.add_patch(Rectangle((x0, row), x1 - x0, 1.0, facecolor="none",
                                   edgecolor="k", hatch="/////", linewidth=0.0))
        rg = robust_green(pg, m)
        if rg:
            ax.plot([rg[0] * 1e3, rg[1] * 1e3], [row + 0.5] * 2, color="k", lw=0.8,
                    marker="|", ms=5, solid_capstyle="butt")
    ax.set_title(f"Safe map with R1 measurement bands (hatched = class not settled at "
                 f"{1-ALPHA:.0%})", pad=6)

    # -- bottom: boundary distributions as median + 95% interval ------------------------------
    items = [
        ("guard $T_{g,min}$ (wake-ref.)", draws.T_guard_s, "0.2"),
        ("break-even pd", draws.ts_min_energy_pd_s, COLOR["pd"]),
        ("break-even rail_off", draws.ts_min_energy_rail_off_s, COLOR["rail_off"]),
        ("crossover pd$\\leftrightarrow$rail_off", draws.crossover_s, "0.4"),
        ("GREEN hi (C6/$B_{req}$) pd", draws.green_hi_pd_s, COLOR["pd"]),
    ]
    for k, (label, ser, c) in enumerate(items):
        b = band(ser)
        y = len(items) - 1 - k
        axb.plot([b["p2_5"] * 1e3, b["p97_5"] * 1e3], [y, y], color=c, lw=2.2,
                 solid_capstyle="butt")
        axb.plot(b["median"] * 1e3, y, "o", color=c, ms=4, mec="white", mew=0.5)
        ha = "right" if k == len(items) - 1 else "center"
        axb.annotate(f"{b['median']*1e3:.3g} ms", (b["median"] * 1e3, y + 0.20),
                     fontsize=7, ha=ha, color="0.25")
    axb.set_yticks(range(len(items)))
    axb.set_yticklabels([i[0] for i in items][::-1], fontsize=7.5)
    axb.set_ylim(-0.6, len(items) - 0.05)   # top headroom so the "12.8 ms" label clears the frame
    axb.set_xscale("log")
    axb.set_xlabel("$T_{sleep}$ (ms, log)")
    axb.set_title("Boundary bands: median and 95% interval (R1 bootstrap)", pad=4)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  safemap_banded   <- R1 bootstrap x firmware C ({N_GRID}/{N_SCALAR} draws)")


def _runs(mask: np.ndarray):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2] - 1))


# ================================================================================================
def main() -> dict:
    pools = load_pools()
    print(f"R3 pools: " + ", ".join(f"{k} n={len(v)}" for k, v in pools.items()))

    draws = scalar_draws(pools)
    pg = pgreen_grid(pools)

    summary = {
        "seed": SEED, "n_scalar": N_SCALAR, "n_grid": N_GRID, "alpha": ALPHA,
        "convention": "b0 wake-referenced (b0_eff = b0_fit*exp(t_fv/tau)); T_guard is total from wake command",
        "band_scope": "within-session repeatability (R1) only; between-session drift (R2) NOT folded in",
        "tolerances": TOL, "common_fields": COMMON,
        "rail_off_p_sleep_floor_W": RAIL_OFF_PSLEEP_FLOOR,
        "sources": {k: str(v.relative_to(REPO)) for k, v in R1.items()},
        "boundaries_s": {
            "T_guard_min": band(draws.T_guard_s),
            "crossover": band(draws.crossover_s),
            **{f"ts_min_energy_{m}": band(draws[f"ts_min_energy_{m}_s"]) for m in MODES},
            **{f"green_lo_{m}": band(draws[f"green_lo_{m}_s"]) for m in MODES},
            **{f"green_hi_{m}": band(draws[f"green_hi_{m}_s"]) for m in MODES},
        },
        "feasible_fraction": {m: float(draws[f"feasible_{m}"].mean()) for m in MODES},
        "robust_green_s": {m: robust_green(pg, m) for m in MODES},
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    draws.to_csv(OUTDIR / "draws.csv", index=False)
    pg.to_csv(OUTDIR / "pgreen.csv", index=False)
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig_safemap_banded(pg, draws)

    g = summary["boundaries_s"]["T_guard_min"]
    x = summary["boundaries_s"]["crossover"]
    print(f"T_guard_min (wake-ref): {g['median']*1e3:.2f} ms  [95%: {g['p2_5']*1e3:.2f}, {g['p97_5']*1e3:.2f}]")
    print(f"crossover:              {x['median']*1e3:.1f} ms  [95%: {x['p2_5']*1e3:.1f}, {x['p97_5']*1e3:.1f}]")
    for m in MODES:
        rg = summary["robust_green_s"][m]
        rg_txt = f"[{rg[0]*1e3:.2f} ms, {rg[1]*1e3:.0f} ms]" if rg else "none"
        print(f"robust GREEN {m}: {rg_txt}  (feasible in {summary['feasible_fraction'][m]:.1%} of draws)")
    print(f"Results -> {OUTDIR.relative_to(REPO)}")
    return summary


if __name__ == "__main__":
    main()
