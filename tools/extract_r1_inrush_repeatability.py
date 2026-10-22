#!/usr/bin/env python3
"""Extract R1 wake-inrush repeatability from scope CSV files.

Input is a folder of Siglent SDS814X HD CSV exports with metadata followed by:

  Second,Value

The scope voltage is assumed to be shunt voltage. The main reported quantities:

  E_inrush = integral from 0 to T_wake of max(Vshunt - Vsteady, 0) / Rshunt * Vrail dt
  T_wake   = electrical settling: the first time after the peak from which the excess current,
             smoothed over 20 us, stays within +-4 SD of its noise in the steady region for 1 ms
             (C-7b, 2026-10-23; integrating to the end of the capture added rectified noise)
  I_peak_abs = max(Vshunt) / Rshunt
  I_peak_excess = max(Vshunt - Vsteady) / Rshunt
  tau_eff = E_inrush / (Vrail * I_peak_excess)

The direct exponential fit is saved as diagnostic because real wake-current
waveforms can be multi-stage rather than a single clean RC decay.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as stats
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "twin"))

import numpy as np  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402


def finite(x: object) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def quantile(values: list[float], q: float) -> float:
    vals = sorted(v for v in values if finite(v))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def ci95(values: list[float]) -> dict:
    vals = [v for v in values if finite(v)]
    n = len(vals)
    if n == 0:
        return {"lo": float("nan"), "hi": float("nan"), "sem": float("nan")}
    if n == 1:
        return {"lo": vals[0], "hi": vals[0], "sem": float("nan")}
    sd = stats.stdev(vals)
    sem = sd / math.sqrt(n)
    try:
        from scipy.stats import t  # type: ignore

        k = float(t.ppf(0.975, n - 1))
    except Exception:  # noqa: BLE001
        k = 1.96
    m = stats.mean(vals)
    return {"lo": m - k * sem, "hi": m + k * sem, "sem": sem}


def summarize(values: list[float]) -> dict:
    vals = [v for v in values if finite(v)]
    if not vals:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "cv": float("nan"),
            "min": float("nan"),
            "p05": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
            "ci95_mean": ci95(vals),
        }
    mean = stats.mean(vals)
    sd = stats.stdev(vals) if len(vals) >= 2 else float("nan")
    return {
        "n": len(vals),
        "mean": mean,
        "std": sd,
        "cv": sd / mean if finite(sd) and abs(mean) > 0 else float("nan"),
        "min": min(vals),
        "p05": quantile(vals, 0.05),
        "p50": quantile(vals, 0.50),
        "p95": quantile(vals, 0.95),
        "max": max(vals),
        "ci95_mean": ci95(vals),
    }


def sort_key(path: Path) -> tuple[str, int]:
    stem = path.stem
    tail = stem.rsplit("_", 1)[-1]
    try:
        return stem[: -len(tail)], int(tail)
    except ValueError:
        return stem, 0


def read_siglent_csv(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    meta: dict[str, str] = {}
    rows = []
    data = False
    for line in path.read_text(errors="replace").splitlines():
        s = line.strip()
        if s == "Second,Value":
            data = True
            continue
        if not data:
            parts = [p.strip() for p in s.split(",")]
            if len(parts) >= 2:
                meta[parts[0]] = parts[1]
            continue
        parts = s.split(",")
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"no scope samples found in {path}")
    t = np.array([r[0] for r in rows], dtype=float)
    v = np.array([r[1] for r in rows], dtype=float)
    return t, v, meta


def fit_tau_diagnostic(t: np.ndarray, excess_v: np.ndarray, t_peak: float, peak_v: float) -> tuple[float, int]:
    m = (t >= t_peak) & (t < t_peak + 0.003) & (excess_v > max(0.05 * peak_v, 0.001))
    if int(m.sum()) < 8:
        return float("nan"), int(m.sum())
    x = t[m] - t_peak
    y = excess_v[m]
    try:
        with __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            pp, _ = curve_fit(
                lambda xx, amp, tau: amp * np.exp(-xx / np.maximum(tau, 1e-9)),
                x,
                y,
                p0=[peak_v, 400e-6],
                bounds=([0.0, 1e-6], [np.inf, 0.01]),
                maxfev=20000,
            )
        return abs(float(pp[1])), int(m.sum())
    except Exception:  # noqa: BLE001
        return float("nan"), int(m.sum())


def settling_time(t: np.ndarray, excess_a: np.ndarray, steady: np.ndarray, nsd: float = 4.0,
                  smooth_s: float = 20e-6, hold_s: float = 1e-3) -> float:
    """T_wake: first time after the smoothed peak from which |excess| stays within nsd SD of the
    smoothed steady-region noise for hold_s."""
    dt = float(np.median(np.diff(t)))
    k = max(1, int(round(smooth_s / dt)))
    sm = np.convolve(excess_a, np.ones(k) / k, "same")
    sd = float(np.std(sm[steady], ddof=1))
    i0 = int(np.argmax(np.where(t >= 0, sm, -np.inf)))
    bad = np.flatnonzero(np.abs(sm) > nsd * sd)
    nxt = np.r_[bad, len(t)]
    n = int(round(hold_s / dt))
    for c in np.r_[i0, bad[bad >= i0] + 1]:
        if nxt[np.searchsorted(nxt, c)] - c >= n:
            return float(t[c])
    return float("nan")


def extract_file(path: Path, *, shunt_ohm: float, voltage_v: float, pre_end_s: float, steady_start_s: float) -> dict:
    t, v, meta = read_siglent_csv(path)
    pre = v[t < pre_end_s]
    steady_region = v[t > steady_start_s]
    if len(pre) == 0:
        pre = v[: max(1, int(0.05 * len(v)))]
    if len(steady_region) == 0:
        steady_region = v[-max(1, int(0.10 * len(v))) :]

    baseline_v = float(np.median(pre))
    steady_v = float(np.median(steady_region))
    peak_idx = int(np.argmax(v))
    peak_v = float(v[peak_idx])
    t_peak = float(t[peak_idx])

    excess_v = v - steady_v
    t_wake = settling_time(t, excess_v / shunt_ohm, t > steady_start_s)
    if not math.isfinite(t_wake):
        raise SystemExit(f"{path.name}: no settling found")
    m_energy = (t >= 0.0) & (t <= t_wake)
    m_post = t >= 0.0
    power_excess = np.maximum(excess_v[m_energy], 0.0) / shunt_ohm * voltage_v
    e_j = float(np.trapezoid(power_excess, t[m_energy]))

    i_peak_abs_a = peak_v / shunt_ohm
    i_peak_excess_a = max(peak_v - steady_v, 0.0) / shunt_ohm
    tau_eff_s = e_j / (voltage_v * i_peak_excess_a) if i_peak_excess_a > 0 else float("nan")
    tau_fit_s, fit_n = fit_tau_diagnostic(t[m_post], excess_v[m_post], t_peak, max(peak_v - steady_v, 0.0))

    return {
        "file": path.name,
        "n_samples": int(len(t)),
        "sample_interval_s": float(meta.get("Sample Interval", "nan")),
        "pre_baseline_mV": baseline_v * 1e3,
        "steady_mV": steady_v * 1e3,
        "peak_mV": peak_v * 1e3,
        "t_peak_us": t_peak * 1e6,
        "E_inrush_mJ": e_j * 1e3,
        "t_wake_ms": t_wake * 1e3,
        "I_peak_abs_mA": i_peak_abs_a * 1e3,
        "I_peak_excess_mA": i_peak_excess_a * 1e3,
        "tau_eff_us": tau_eff_s * 1e6,
        "tau_fit_diag_us": tau_fit_s * 1e6,
        "tau_fit_n": fit_n,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract R1 inrush repeatability from scope CSVs")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--unit", default="A")
    ap.add_argument("--mode", required=True, choices=("pd", "rail_off", "vddio"))
    ap.add_argument("--shunt-ohm", type=float, default=1.1)
    ap.add_argument("--voltage-v", type=float, default=5.019)
    ap.add_argument("--pre-end-s", type=float, default=-0.001)
    ap.add_argument("--steady-start-s", type=float, default=0.006)
    ap.add_argument("--output-dir", default=str(ROOT / "tools" / "data"))
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    inp = Path(args.input_dir)
    files = sorted(inp.glob("*.csv"), key=sort_key)
    if not files:
        raise SystemExit(f"no CSV files found in {inp}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.label)
    outdir = Path(args.output_dir) / f"r1_inrush_repeatability_{safe_label}_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = [
        extract_file(
            p,
            shunt_ohm=args.shunt_ohm,
            voltage_v=args.voltage_v,
            pre_end_s=args.pre_end_s,
            steady_start_s=args.steady_start_s,
        )
        for p in files
    ]
    for i, row in enumerate(rows):
        row["run"] = i
        row["mode"] = args.mode

    with (outdir / "fits.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)

    metadata = {
        "label": args.label,
        "unit": args.unit,
        "mode": args.mode,
        "started_at": stamp,
        "input_dir": str(inp),
        "n": len(rows),
        "shunt_ohm": args.shunt_ohm,
        "voltage_v": args.voltage_v,
        "pre_end_s": args.pre_end_s,
        "steady_start_s": args.steady_start_s,
        "notes": args.notes,
        "processing": "E_inrush integrates positive excess above post-wake steady active shunt voltage from 0 to T_wake (smoothed excess within 4 SD of steady noise for 1 ms); tau_eff = E/(Vrail*I_peak_excess).",
    }

    summary = {
        "metadata": metadata,
        "mode": args.mode,
        "metrics": {
            "E_inrush_mJ": summarize([r["E_inrush_mJ"] for r in rows]),
            "t_wake_ms": summarize([r["t_wake_ms"] for r in rows]),
            "I_peak_abs_mA": summarize([r["I_peak_abs_mA"] for r in rows]),
            "I_peak_excess_mA": summarize([r["I_peak_excess_mA"] for r in rows]),
            "tau_eff_us": summarize([r["tau_eff_us"] for r in rows]),
            "tau_fit_diag_us": summarize([r["tau_fit_diag_us"] for r in rows]),
            "pre_baseline_mV": summarize([r["pre_baseline_mV"] for r in rows]),
            "steady_mV": summarize([r["steady_mV"] for r in rows]),
            "peak_mV": summarize([r["peak_mV"] for r in rows]),
            "t_peak_us": summarize([r["t_peak_us"] for r in rows]),
        },
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True))

    e = summary["metrics"]["E_inrush_mJ"]
    ip = summary["metrics"]["I_peak_abs_mA"]
    te = summary["metrics"]["tau_eff_us"]
    print(f"Wrote {outdir}")
    print(
        f"{args.mode}: n={len(rows)} "
        f"E={e['mean']:.6f}+/-{e['std']:.6f} mJ "
        f"Ipeak={ip['mean']:.3f}+/-{ip['std']:.3f} mA "
        f"tau_eff={te['mean']:.2f}+/-{te['std']:.2f} us"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
