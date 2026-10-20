"""B2 — deliberate C2 violation: measured bias at first trust vs the recovery physics.

Reads the committed truncated-guard sweep (CTLB2 records) and the committed R1
burst fits, subtracts the static pedestal (estimated from the baseline point,
where the predicted transient is ~0.1 mg), and plots the measured post-wake
bias at the first trusted samples against the b0*exp(-t/tau) prediction band
(one curve per R1 fit -> 95% band). Deterministic; regenerate with
    cd twin && .venv/bin/python -m analysis.b2_guard
Outputs: docs/figures/b2_guard_violation.{pdf,png}
         <sweep dir>/analysis.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # twin/

TWIN = Path(__file__).resolve().parent.parent
REPO = TWIN.parent
FIGDIR = REPO / "docs" / "figures"

SWEEP = REPO / "tools/data/unitA_b2_guard_sweep_20260709_082651"
R1_FITS = REPO / ("tools/data/r1_burst_repeatability_unitA_"
                  "r1_burst_pd_n30_reprocessed_20260706_203023/fits.csv")
BASELINE_US = 12400
EPS_MG = 1.0
GUARD_BAND_MS = (12.54, 12.82, 13.05)     # R3 (lo95, median, hi95), wake-referenced

COLOR = {"pd": "#D55E00", "RED": "#d1495b", "band": "#0072B2"}


def _style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.titlesize": 8,
        "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "figure.dpi": 150, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.25, "lines.linewidth": 1.4,
    })


def window_mean_pred(fits: pd.DataFrame, t0_ms: float, t1_ms: float) -> np.ndarray:
    """Trusted-sample model per R1 fit: mean of b0_signed*exp(-(t-t_fv)/tau) over the
    window, treating t < t_fv as unobservable (excluded)."""
    out = []
    for _, r in fits.iterrows():
        tfv, tau, b0 = r.first_valid_us / 1e3, r.tau_bias_s * 1e3, r.b0_signed_g * 1e3
        lo = max(t0_ms, tfv)
        if t1_ms <= lo:
            out.append(np.nan)
            continue
        a, b = (lo - tfv) / tau, (t1_ms - tfv) / tau
        out.append(b0 * (np.exp(-a) - np.exp(-b)) / max(b - a, 1e-9))
    return np.asarray(out)


def curves(fits: pd.DataFrame, t_ms: np.ndarray) -> np.ndarray:
    """|b(t)| per fit (trusted-sample model), NaN inside each fit's blind window."""
    c = np.full((len(fits), len(t_ms)), np.nan)
    for i, (_, r) in enumerate(fits.iterrows()):
        tfv, tau, b0 = r.first_valid_us / 1e3, r.tau_bias_s * 1e3, abs(r.b0_signed_g) * 1e3
        m = t_ms >= tfv
        c[i, m] = b0 * np.exp(-(t_ms[m] - tfv) / tau)
    return c


def main() -> dict:
    df = pd.read_csv(SWEEP / "b2_records.csv")
    fits = pd.read_csv(R1_FITS)
    ped = df[df.point_us == BASELINE_US].probe_mean_mg.mean()

    rows = []
    for us in sorted(df.point_us.unique(), reverse=True):
        g = df[df.point_us == us]
        tr = g.probe_mean_mg - ped
        t0, t1 = g.t_probe0_us.mean() / 1e3, g.t_probe1_us.mean() / 1e3
        pred = window_mean_pred(fits, t0, t1)
        pred = pred[~np.isnan(pred)]
        rows.append({
            "point_us": int(us), "n": int(len(g)),
            "t0_ms": t0, "t1_ms": t1,
            "transient_mean_mg": float(tr.mean()),
            "transient_sem_mg": float(tr.std(ddof=1) / np.sqrt(len(tr))),
            "transient_std_mg": float(tr.std(ddof=1)),
            "pred_median_mg": float(np.median(pred)) if len(pred) else None,
            "pred_p2_5_mg": float(np.percentile(pred, 2.5)) if len(pred) else None,
            "pred_p97_5_mg": float(np.percentile(pred, 97.5)) if len(pred) else None,
            "saturated": bool(tr.std(ddof=1) > 100.0),
        })

    # ---- figure -----------------------------------------------------------------
    _style()
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    t = np.geomspace(fits.first_valid_us.min() / 1e3, 22.0, 400)
    c = curves(fits, t)
    valid = ~np.all(np.isnan(c), axis=0)
    t, c = t[valid], c[:, valid]
    med = np.nanmedian(c, axis=0)
    lo = np.nanpercentile(c, 2.5, axis=0)
    hi = np.nanpercentile(c, 97.5, axis=0)
    ax.fill_between(t, lo, hi, color=COLOR["band"], alpha=0.20, lw=0,
                    label="$b_0e^{-t/\\tau}$ (R1, 95% band)")
    ax.plot(t, med, color=COLOR["band"], lw=1.2)

    tfv_med = fits.first_valid_us.median() / 1e3
    ax.axvspan(1.5, tfv_med, color="0.55", alpha=0.18, lw=0)
    ax.annotate("blind window\n(err frames)", (2.0, 20), fontsize=6.5, color="0.35")

    ax.axhline(EPS_MG, ls="--", color=COLOR["RED"], lw=1)
    ax.annotate("$\\epsilon_{allowed}$ = 1 mg", (22.5, 1.35), fontsize=6.5,
                color=COLOR["RED"], ha="right")
    ax.axvspan(GUARD_BAND_MS[0], GUARD_BAND_MS[2], color="0.4", alpha=0.25, lw=0)
    ax.annotate("$T_{guard}$", (GUARD_BAND_MS[1], 90), fontsize=6.5,
                color="0.3", ha="center")

    for r in rows:
        x = 0.5 * (max(r["t0_ms"], tfv_med) + r["t1_ms"])
        xerr = [[x - max(r["t0_ms"], tfv_med)], [r["t1_ms"] - x]]
        if r["saturated"]:
            ax.errorbar(x, 200.0, xerr=xerr, fmt="v", color=COLOR["pd"], ms=5,
                        capsize=2, lw=1)
            ax.annotate("saturated\n($\\sigma$ = 2.6 g)", (x, 250), fontsize=6.5,
                        color=COLOR["pd"], ha="center", va="bottom")
            continue
        y = abs(r["transient_mean_mg"])
        ax.errorbar(x, max(y, 0.05), xerr=xerr, yerr=r["transient_sem_mg"],
                    fmt="o", color=COLOR["pd"], ms=4, mec="white", mew=0.5,
                    capsize=2, lw=1,
                    label="measured (probe window)" if r["point_us"] == BASELINE_US else None)
        ax.annotate(f"{r['point_us']/1e3:g} ms", (x * 0.88, max(y, 0.05) * 1.6),
                    fontsize=6.5, color=COLOR["pd"], ha="right")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1.5, 24); ax.set_ylim(0.03, 1600)   # headroom so "saturated"/eps labels clear the frame
    ax.set_xticks([2, 4, 6, 10, 15, 20])
    ax.set_xticklabels(["2", "4", "6", "10", "15", "20"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("time after wake command (ms, log)")
    ax.set_ylabel("post-wake |bias| (mg, log)")
    ax.set_title("Bias at first trust vs guard truncation")
    ax.legend(loc="lower left", framealpha=0.9)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"b2_guard_violation.{ext}")
    plt.close(fig)

    result = {"sweep": str(SWEEP.relative_to(REPO)), "pedestal_mg": float(ped),
              "eps_allowed_mg": EPS_MG, "points": rows}
    (SWEEP / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print("  b2_guard_violation <-", SWEEP.name)
    for r in rows:
        tag = "SATURATED" if r["saturated"] else (
            f"{r['transient_mean_mg']:+7.2f} mg vs pred {r['pred_median_mg']:+7.2f}")
        print(f"    {r['point_us']:>6d} us: {tag}")
    return result


if __name__ == "__main__":
    main()
