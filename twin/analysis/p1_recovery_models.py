"""Paper 1, Section III-B: is the wake recovery a single exponential?

Fits three models to the raw post-wake burst captures behind the manuscript's (b0, tau):

  M1  c + b exp(-t/tau)                          the manuscript's model
  M2  c + b1 exp(-t/tau1) + b2 exp(-t/tau2)      two relaxation processes
  M3  c + b exp(-(t/tau)^beta)                   stretched exponential (distributed relaxation)

t is measured from the first valid sample; every model has a free settled level c and is fit
over the whole capture (the burst tool instead subtracts the mean of the second half and fits
the first 200 ms; fit_m1 reproduces its numbers to check the two agree).

Two views:
  per burst      residual RMS against the capture's own noise floor, AIC, BIC, and the guard
                 each model implies at the manuscript's budget (1 - delta) eps = 0.85 mg
  ensemble mean  the N bursts averaged sample-by-sample, which lowers the noise floor by
                 sqrt(N) and so exposes slow components a single burst cannot resolve

The model-free check does not depend on any fit: the ensemble-mean deviation from the settled
level after the manuscript's guard time, against the budget.

Limit: a component slower than the capture (0.82 s) is absorbed into c and is invisible here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, least_squares

TWIN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TWIN / "analysis"))
import paper2_runs as runs  # noqa: E402

BUDGET_G = 0.85e-3              # (1 - delta) eps, delta = 0.15, eps = 1 mg
RUNS = {
    "A": "r1_burst_repeatability_unitA_r1_burst_pd_n30_reprocessed_20260706_203023",
    "B": "r1_burst_repeatability_unitB_r1_burst_pd_n30_20260712_232025",
}
MODELS = ("M1", "M2", "M3")
NPAR = {"M1": 3, "M2": 5, "M3": 4}


def load_bursts(run: str) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """[(t_s from first valid, accel_g, first_valid_s)] for every complete pd burst of a run."""
    tool = runs._tool()
    out = []
    for f in sorted((runs.DATA / run / "raw").glob("burst_*.txt")):
        lines = f.read_text(errors="ignore").splitlines()
        if not any(ln.startswith("BURST_END") for ln in lines):
            continue
        s, _ = tool.usable_samples(lines, require_err0=True)
        if len(s) < 100:
            continue
        t = np.array([v[0] for v in s])
        a = np.array([v[1] for v in s])
        out.append((t - t[0], a, float(t[0])))
    return out


def transient(model: str, p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The decaying part of a model (without the settled level p[0])."""
    if model == "M1":
        return p[1] * np.exp(-t / p[2])
    if model == "M2":
        return p[1] * np.exp(-t / p[2]) + p[3] * np.exp(-t / p[4])
    return p[1] * np.exp(-np.power(t / p[2], p[3]))


def fit(model: str, t: np.ndarray, a: np.ndarray) -> dict:
    c0 = float(a[len(a) // 2:].mean())
    b0 = float(a[0] - c0)
    lo, hi = min(b0, -b0) * 5 - 1e-3, max(b0, -b0) * 5 + 1e-3
    if model == "M1":
        starts = [[c0, b0, 1.5e-3]]
        bounds = ([-1, lo, 1e-5], [1, hi, 1.0])
    elif model == "M2":
        # tau1 < tau2 is imposed by the bounds; several slow-term starts, best RSS kept
        starts = [[c0, 0.9 * b0, 1.2e-3, 0.1 * b0, ts] for ts in (5e-3, 2e-2, 1e-1, 3e-1)]
        bounds = ([-1, lo, 1e-5, lo, 3e-3], [1, hi, 3e-3, hi, 2.0])
    else:
        starts = [[c0, b0, 1.5e-3, be] for be in (1.0, 0.8, 0.6)]
        bounds = ([-1, lo, 1e-5, 0.2], [1, hi, 1.0, 2.0])
    best = None
    for x0 in starts:
        x0 = np.clip(x0, np.array(bounds[0]) + 1e-12, np.array(bounds[1]) - 1e-12)
        r = least_squares(lambda p: p[0] + transient(model, p, t) - a, x0, bounds=bounds,
                          x_scale="jac")
        if best is None or r.cost < best.cost:
            best = r
    res = best.fun
    n, k = len(t), NPAR[model]
    rss = float(res @ res)
    return dict(model=model, p=best.x, rss=rss, rms=float(np.sqrt(rss / n)),
                aic=n * np.log(rss / n) + 2 * k, bic=n * np.log(rss / n) + k * np.log(n),
                res=res)


def guard_after_fv(model: str, p: np.ndarray, t_max: float) -> float:
    """Last time (from first valid) at which |transient| equals the budget; 0 if never above."""
    tt = np.geomspace(1e-6, t_max, 4000)
    above = np.abs(transient(model, p, tt)) > BUDGET_G
    if not above.any():
        return 0.0
    i = int(np.flatnonzero(above)[-1])
    if i == len(tt) - 1:
        return float("nan")                     # still above the budget at the end of capture
    return float(brentq(lambda x: abs(transient(model, p, np.array([x]))[0]) - BUDGET_G,
                        tt[i], tt[i + 1]))


def per_burst(bursts) -> dict:
    rows = {m: [] for m in MODELS}
    noise = []
    for t, a, fv in bursts:
        noise.append(float(a[len(a) // 2:].std()))
        for m in MODELS:
            f = fit(m, t, a)
            f["guard"] = fv + guard_after_fv(m, f["p"], t[-1])
            rows[m].append(f)
    return dict(rows=rows, noise=np.array(noise))


def ensemble(bursts):
    """Sample-by-sample mean of the bursts (same firmware grid; index-aligned on first valid)."""
    n = min(len(t) for t, _, _ in bursts)
    t = np.mean([t[:n] for t, _, _ in bursts], axis=0)
    a = np.mean([a[:n] for _, a, _ in bursts], axis=0)
    jitter = float(np.max(np.std([t[:n] for t, _, _ in bursts], axis=0)))
    return t, a, float(np.mean([fv for _, _, fv in bursts])), jitter


def post_guard_deviation(t, a, fv, guard_s, windows=((0, 5e-3), (5e-3, 20e-3), (20e-3, 100e-3))):
    """Ensemble-mean deviation from the settled level in windows starting at the guard."""
    c = float(a[len(a) // 2:].mean())
    t0 = guard_s - fv
    out = []
    for lo, hi in windows:
        m = (t >= t0 + lo) & (t < t0 + hi)
        out.append((lo, hi, int(m.sum()), float(a[m].mean() - c), float(a[m].std() / np.sqrt(m.sum()))))
    return out


def report(unit: str) -> None:
    bursts = load_bursts(RUNS[unit])
    pb = per_burst(bursts)
    noise = pb["noise"]
    print(f"\n=== unit {unit}: {RUNS[unit]}")
    print(f"bursts {len(bursts)}; samples/burst {np.median([len(t) for t, _, _ in bursts]):.0f}; "
          f"span {1e3 * np.median([t[-1] for t, _, _ in bursts]):.0f} ms; "
          f"noise floor (2nd half std) {1e3 * np.median(noise):.2f} mg")
    m1 = pb["rows"]["M1"]
    print(f"M1 medians: b0 {1e3 * np.median([abs(f['p'][1]) for f in m1]):.1f} mg, "
          f"tau {1e3 * np.median([f['p'][2] for f in m1]):.3f} ms")
    print(f"{'model':5} {'rms/noise':>10} {'dAIC':>8} {'dBIC':>8} {'best AIC':>9} {'best BIC':>9} "
          f"{'guard ms (median, min..max)':>30}")
    aic = np.array([[f["aic"] for f in pb["rows"][m]] for m in MODELS])
    bic = np.array([[f["bic"] for f in pb["rows"][m]] for m in MODELS])
    for i, m in enumerate(MODELS):
        g = 1e3 * np.array([f["guard"] for f in pb["rows"][m]])
        r = np.array([f["rms"] for f in pb["rows"][m]]) / noise
        print(f"{m:5} {np.median(r):10.3f} {np.median(aic[i] - aic[0]):8.1f} "
              f"{np.median(bic[i] - bic[0]):8.1f} {int((aic.argmin(0) == i).sum()):9d} "
              f"{int((bic.argmin(0) == i).sum()):9d} "
              f"{np.nanmedian(g):12.2f} ({np.nanmin(g):.2f}..{np.nanmax(g):.2f})"
              f"{'  nan=' + str(int(np.isnan(g).sum())) if np.isnan(g).any() else ''}")
    m2 = pb["rows"]["M2"]
    print(f"M2 slow term: |b2| median {1e3 * np.median([abs(f['p'][3]) for f in m2]):.2f} mg, "
          f"tau2 median {1e3 * np.median([f['p'][4] for f in m2]):.1f} ms "
          f"(IQR {1e3 * np.percentile([f['p'][4] for f in m2], 25):.1f}.."
          f"{1e3 * np.percentile([f['p'][4] for f in m2], 75):.1f})")
    print(f"M3 beta: median {np.median([f['p'][3] for f in pb['rows']['M3']]):.3f} "
          f"(IQR {np.percentile([f['p'][3] for f in pb['rows']['M3']], 25):.3f}.."
          f"{np.percentile([f['p'][3] for f in pb['rows']['M3']], 75):.3f})")

    t, a, fv, jit = ensemble(bursts)
    tail = a[len(a) // 2:]
    print(f"-- ensemble mean of {len(bursts)} bursts (grid jitter <= {1e6 * jit:.0f} us); "
          f"noise floor {1e3 * tail.std():.3f} mg; first valid {1e3 * fv:.2f} ms")
    fits = {m: fit(m, t, a) for m in MODELS}
    for m in MODELS:
        f = fits[m]
        g = fv + guard_after_fv(m, f["p"], t[-1])
        early = f["res"][t < 20e-3]
        print(f"{m:5} rms/noise {f['rms'] / tail.std():.3f}  dAIC {f['aic'] - fits['M1']['aic']:8.1f}  "
              f"dBIC {f['bic'] - fits['M1']['bic']:8.1f}  guard {1e3 * g:.2f} ms  "
              f"max |res| in first 20 ms {1e3 * np.abs(early).max():.2f} mg  "
              f"params {np.array2string(f['p'], precision=5)}")
    g1 = fv + guard_after_fv("M1", fits["M1"]["p"], t[-1])
    print(f"-- model-free: ensemble-mean deviation from settled, after the M1 guard ({1e3 * g1:.2f} ms)")
    for lo, hi, n, d, se in post_guard_deviation(t, a, fv, g1):
        print(f"   guard+{1e3 * lo:.0f}..{1e3 * hi:.0f} ms: n={n:4d}  mean {1e3 * d:+.3f} mg "
              f"(se {1e3 * se:.3f}); budget {1e3 * BUDGET_G:.2f} mg")


def bootstrap_m2(unit: str, reps: int = 300, seed: int = 20261008) -> dict:
    """Resample bursts with replacement; refit M1 and M2 to each resampled ensemble mean."""
    bursts = load_bursts(RUNS[unit])
    rng = np.random.default_rng(seed)
    out = dict(b2=[], tau2=[], g1=[], g2=[], at_g1=[], daic=[])
    for _ in range(reps):
        pick = [bursts[i] for i in rng.integers(0, len(bursts), len(bursts))]
        t, a, fv, _ = ensemble(pick)
        f1, f2 = fit("M1", t, a), fit("M2", t, a)
        g1 = guard_after_fv("M1", f1["p"], t[-1])
        out["b2"].append(abs(f2["p"][3])); out["tau2"].append(f2["p"][4])
        out["g1"].append(fv + g1); out["g2"].append(fv + guard_after_fv("M2", f2["p"], t[-1]))
        out["at_g1"].append(abs(transient("M2", f2["p"], np.array([g1]))[0]))
        out["daic"].append(f2["aic"] - f1["aic"])
    return {k: np.array(v) for k, v in out.items()}


def report_bootstrap(unit: str) -> None:
    b = bootstrap_m2(unit)
    q = lambda x, s=1e3: "%.2f [%.2f, %.2f]" % tuple(s * np.nanpercentile(x, [50, 2.5, 97.5]))  # noqa: E731
    print(f"-- unit {unit} bootstrap over bursts (300 resamples), median [95 % interval]")
    print(f"   M2 slow amplitude |b2| {q(b['b2'])} mg; tau2 {q(b['tau2'])} ms")
    print(f"   guard M1 {q(b['g1'])} ms; guard M2 {q(b['g2'])} ms")
    print(f"   M2 residual bias at the M1 guard {q(b['at_g1'])} mg (budget 0.85, tolerance 1.00)")
    print(f"   dAIC(M2 - M1) {q(b['daic'], 1)}; M2 preferred in {100 * np.mean(b['daic'] < 0):.0f} %")


if __name__ == "__main__":
    for u in (sys.argv[1:] or list(RUNS)):
        report(u)
        report_bootstrap(u)
