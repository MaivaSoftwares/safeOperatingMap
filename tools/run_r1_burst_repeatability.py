#!/usr/bin/env python3
"""R1 burst repeatability campaign for safe-map uncertainty bands.

This is not just the pass/fail burst acceptance gate. It collects N firmware
BURST captures per mode, fits the post-wake acceleration transient, and writes
the coefficient distribution needed by Campaign R:

  b0, tau_bias, first_valid, burst_rate, T_guard = tau*ln(b0/((1-delta)*eps))

The primary fit intentionally uses only usable post-wake samples:
  drdy == 1, err == 0, |accel| < glitch_g

The MMS-30 can emit one or more early drdy samples with err=1 immediately after
wake. Those frames are useful diagnostics, but they are not valid measurement
samples and can include impossible saturated values. R1 uncertainty bands must
therefore be based on the first trustworthy sample, not the first toggling DRDY.

Primary use: Unit A, mode=pd, N=20-30. PD is the clean coefficient source because
the transient peak is visible. rail_off is peak-blinded by the MMS30_PowerUp
delay, so use rail_off only as supporting/tail-consistency evidence.

Bench setup:
  - close live_csv_monitor.py; the serial port is exclusive
  - board rigidly fixed, do not touch during the session
  - sensitive axis horizontal / near 0 g
  - leave board powered and VDDIO ON after the run

Example:
  twin/.venv/bin/python tools/run_r1_burst_repeatability.py \\
    --label unitA_r1_burst_pd_n30 \\
    --unit A \\
    --modes pd \\
    --runs 30 \\
    --sleep-ms 1000 \\
    --n 2000
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as stats
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "twin"))

from device.physical_device import PhysicalDevice  # noqa: E402
from analysis import hwcapture as hc  # noqa: E402


DEFAULT_EPS_G = 0.001
DEFAULT_DELTA = 0.15
DEFAULT_GLITCH_G = 1.0


def finite(x: object) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def mean_or_nan(values: list[float]) -> float:
    vals = [v for v in values if finite(v)]
    return stats.mean(vals) if vals else float("nan")


def stdev_or_nan(values: list[float]) -> float:
    vals = [v for v in values if finite(v)]
    return stats.stdev(vals) if len(vals) >= 2 else float("nan")


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


def guard_s(b0_g: float, tau_s: float, eps_g: float, delta: float) -> float:
    b0_g = abs(b0_g) if finite(b0_g) else b0_g
    target = (1.0 - delta) * eps_g
    if not (finite(b0_g) and finite(tau_s)) or b0_g <= target or tau_s <= 0:
        return float("nan")
    return tau_s * math.log(b0_g / target)


def parse_burst_rows(lines: list[str]) -> tuple[list[tuple[int, float, int, int]], dict]:
    """Parse raw firmware BURST rows as (t_us, accel_g, drdy, err)."""
    rows: list[tuple[int, float, int, int]] = []
    meta: dict = {}
    in_block = False
    for ln in lines:
        ln = ln.strip()
        if ln.startswith("BURST_BEGIN"):
            in_block = True
            meta.update(hc._kv(ln))  # noqa: SLF001 - shared parser helper
            continue
        if ln.startswith("BURST_END"):
            meta.update(hc._kv(ln))  # noqa: SLF001 - shared parser helper
            break
        if not in_block or ln.startswith("t_us"):
            continue
        p = ln.split(",")
        if len(p) != 4:
            continue
        try:
            rows.append((int(p[0]), float(p[1]), int(p[2]), int(p[3])))
        except ValueError:
            continue
    return rows, meta


def usable_samples(
    lines: list[str],
    *,
    require_err0: bool = True,
    glitch_g: float = DEFAULT_GLITCH_G,
) -> tuple[list[tuple[float, float]], dict]:
    rows, meta = parse_burst_rows(lines)
    out = []
    for t_us, accel_g, drdy, err in rows:
        if drdy != 1:
            continue
        if require_err0 and err != 0:
            continue
        if abs(accel_g) >= glitch_g:
            continue
        out.append((t_us / 1e6, accel_g))
    return out, meta


def burst_tail_metrics(lines: list[str]) -> dict:
    """Return raw-burst health metrics for usable samples only."""
    samples, _meta = usable_samples(lines)
    if not samples:
        return {}
    a = [v[1] for v in samples]
    t = [v[0] for v in samples]
    tail = a[int(0.9 * len(a)) :] if len(a) >= 10 else a
    return {
        "raw_valid_n": int(len(a)),
        "raw_peak_abs_g": float(max(abs(v) for v in a)) if len(a) else float("nan"),
        "tail_median_g": float(stats.median(tail)) if len(tail) else float("nan"),
        "tail_std_g": float(stats.pstdev(tail)) if len(tail) else float("nan"),
        "valid_window_s": float(t[-1] - t[0]) if len(t) >= 2 else float("nan"),
    }


def transient_fit_from_burst(
    lines: list[str],
    *,
    require_err0: bool = True,
    glitch_g: float = DEFAULT_GLITCH_G,
    settle_frac: float = 0.5,
    fit_window_s: float = 0.2,
) -> dict:
    """Fit b0*exp(-t/tau) after subtracting the settled level.

    b0 is signed in this helper; the R1 guard calculation uses abs(b0).
    """
    import numpy as np
    from scipy.optimize import curve_fit

    samples, meta = usable_samples(lines, require_err0=require_err0, glitch_g=glitch_g)
    if len(samples) < 10:
        return {
            "b0_g": float("nan"),
            "tau_bias_s": float("nan"),
            "noise_g": float("nan"),
            "settled_g": float("nan"),
            "first_valid_us": float("nan"),
            "burst_rate_hz": float(meta.get("rate_hz", "nan")),
            "n": len(samples),
        }

    t = np.array([v[0] for v in samples], dtype=float)
    a = np.array([v[1] for v in samples], dtype=float)
    first_valid_us = float(t[0] * 1e6)
    t = t - t[0]
    settled = float(a[t > t.max() * settle_frac].mean())
    dev = a - settled
    m = t < min(fit_window_s, t.max())
    try:
        with __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            pp, _ = curve_fit(
                lambda tt, b, tau: b * np.exp(-tt / np.maximum(tau, 1e-6)),
                t[m],
                dev[m],
                p0=[dev[m][0], 0.002],
                maxfev=20000,
            )
        b0, tau = float(pp[0]), abs(float(pp[1]))
    except Exception:  # noqa: BLE001
        b0, tau = float("nan"), float("nan")
    return {
        "b0_g": b0,
        "tau_bias_s": tau,
        "settled_g": settled,
        "first_valid_us": first_valid_us,
        "noise_g": float(dev[t > t.max() * settle_frac].std()),
        "burst_rate_hz": float(meta.get("rate_hz", "nan")),
        "n": int(len(t)),
    }


def one_burst(
    dev: PhysicalDevice,
    mode: str,
    sleep_ms: int,
    n: int,
    p_on_w: float,
    eps_g: float,
    delta: float,
    raw_path: Path,
) -> dict:
    lines = dev.run_burst(mode=mode, sleep_ms=sleep_ms, n=n)
    raw_path.write_text("\n".join(lines) + "\n")

    row: dict = {
        "mode": mode,
        "raw_file": raw_path.name,
        "ok": False,
        "error": "",
        "b0_g": float("nan"),
        "tau_bias_s": float("nan"),
        "guard_s": float("nan"),
        "noise_g": float("nan"),
        "burst_rate_hz": float("nan"),
        "first_valid_us": float("nan"),
    }
    row.update(burst_tail_metrics(lines))

    if not any(ln.startswith("BURST_END") for ln in lines):
        row["error"] = "missing BURST_END"
        return row

    try:
        fit = transient_fit_from_burst(lines, require_err0=True)
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"fit failed: {exc}"
        return row

    b0_signed = float(fit.get("b0_g", float("nan")))
    row.update(
        {
            "ok": finite(fit.get("b0_g")) and finite(fit.get("tau_bias_s")),
            "b0_g": abs(b0_signed) if finite(b0_signed) else float("nan"),
            "b0_signed_g": b0_signed,
            "tau_bias_s": float(fit.get("tau_bias_s", float("nan"))),
            "noise_g": float(fit.get("noise_g", float("nan"))),
            "burst_rate_hz": float(fit.get("burst_rate_hz", float("nan"))),
            "first_valid_us": float(fit.get("first_valid_us", float("nan"))),
            "settled_g": float(fit.get("settled_g", float("nan"))),
            "fit_n": int(fit.get("n", 0)),
        }
    )
    row["guard_s"] = guard_s(row["b0_g"], row["tau_bias_s"], eps_g, delta)

    # Also keep the non-strict de-trended fit for diagnostics only. This captures
    # the earlier err=1 transient shape but is not a valid measurement-state band.
    try:
        diag = transient_fit_from_burst(lines, require_err0=False)
        row["diag_b0_g"] = float(diag.get("b0_g", float("nan")))
        row["diag_tau_bias_s"] = float(diag.get("tau_bias_s", float("nan")))
        row["diag_noise_g"] = float(diag.get("noise_g", float("nan")))
        row["diag_first_valid_us"] = float(diag.get("first_valid_us", float("nan")))
    except Exception:  # noqa: BLE001
        pass

    return row


def summarize_mode(rows: list[dict]) -> dict:
    ok_rows = [r for r in rows if r.get("ok")]
    out = {"n_total": len(rows), "n_ok": len(ok_rows), "fit_fraction": len(ok_rows) / len(rows) if rows else 0.0}
    fields = {
        "b0_mg": [r["b0_g"] * 1e3 for r in ok_rows],
        "tau_ms": [r["tau_bias_s"] * 1e3 for r in ok_rows],
        "guard_ms": [r["guard_s"] * 1e3 for r in ok_rows],
        "noise_mg": [r["noise_g"] * 1e3 for r in ok_rows],
        "first_valid_ms": [r["first_valid_us"] * 1e-3 for r in ok_rows],
        "burst_rate_hz": [r["burst_rate_hz"] for r in ok_rows],
        "tail_std_mg": [r.get("tail_std_g", float("nan")) * 1e3 for r in ok_rows],
    }
    for name, vals in fields.items():
        clean = [v for v in vals if finite(v)]
        m = mean_or_nan(clean)
        sd = stdev_or_nan(clean)
        out[name] = {
            "mean": m,
            "std": sd,
            "cv": sd / m if finite(sd) and finite(m) and abs(m) > 0 else float("nan"),
            "min": min(clean) if clean else float("nan"),
            "p05": quantile(clean, 0.05),
            "p50": quantile(clean, 0.50),
            "p95": quantile(clean, 0.95),
            "max": max(clean) if clean else float("nan"),
            "ci95_mean": ci95(clean),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="R1 burst repeatability for safe-map uncertainty bands")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--label", required=True)
    ap.add_argument("--unit", default="A", help="physical unit id, e.g. A or B")
    ap.add_argument("--modes", default="pd", help="comma list, recommended first run: pd")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--sleep-ms", type=int, default=1000)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--gap-s", type=float, default=3.0)
    ap.add_argument("--pon-w", type=float, default=0.067)
    ap.add_argument("--eps-g", type=float, default=DEFAULT_EPS_G)
    ap.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    ap.add_argument("--output-dir", default=str(ROOT / "tools" / "data"))
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.label)
    outdir = Path(args.output_dir) / f"r1_burst_repeatability_{safe_label}_{stamp}"
    rawdir = outdir / "raw"
    rawdir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "label": args.label,
        "unit": args.unit,
        "started_at": stamp,
        "port": args.port,
        "baud": args.baud,
        "modes": [m.strip() for m in args.modes.split(",") if m.strip()],
        "runs": args.runs,
        "sleep_ms": args.sleep_ms,
        "n": args.n,
        "gap_s": args.gap_s,
        "p_on_W": args.pon_w,
        "eps_g": args.eps_g,
        "delta": args.delta,
        "notes": args.notes,
        "bench_setup": "axis horizontal/near 0 g; rigid mount; do not touch during run",
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"R1 burst repeatability output -> {outdir}")
    print("Close live monitor first. Keep board rigid and axis near 0 g.")

    all_rows: list[dict] = []
    summary = {"metadata": metadata, "modes": {}}

    with PhysicalDevice(args.port, args.baud) as dev:
        dev.ensure_streaming()
        for mode in metadata["modes"]:
            print(f"\n=== mode={mode} runs={args.runs} sleep_ms={args.sleep_ms} n={args.n} ===")
            mode_rows = []
            for i in range(args.runs):
                raw_path = rawdir / f"burst_{mode}_{i:03d}.txt"
                row = one_burst(dev, mode, args.sleep_ms, args.n, args.pon_w, args.eps_g, args.delta, raw_path)
                row["run"] = i
                mode_rows.append(row)
                all_rows.append(row)
                if row["ok"]:
                    print(
                        f"[{i+1:02d}/{args.runs}] "
                        f"b0={row['b0_g']*1e3:7.2f} mg "
                        f"tau={row['tau_bias_s']*1e3:6.3f} ms "
                        f"guard={row['guard_s']*1e3:6.2f} ms "
                        f"rate={row['burst_rate_hz']:6.0f} Hz "
                        f"first={row['first_valid_us']*1e-3:5.2f} ms"
                    )
                else:
                    print(f"[{i+1:02d}/{args.runs}] INVALID: {row['error']}")
                time.sleep(args.gap_s)
            summary["modes"][mode] = summarize_mode(mode_rows)

    fieldnames = sorted({k for r in all_rows for k in r.keys()})
    with (outdir / "fits.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True))

    print("\nSummary")
    for mode, sm in summary["modes"].items():
        b0 = sm["b0_mg"]
        tau = sm["tau_ms"]
        guard = sm["guard_ms"]
        print(
            f"{mode}: n_ok={sm['n_ok']}/{sm['n_total']} "
            f"b0={b0['mean']:.2f}+/-{b0['std']:.2f} mg "
            f"tau={tau['mean']:.3f}+/-{tau['std']:.3f} ms "
            f"guard={guard['mean']:.2f}+/-{guard['std']:.2f} ms "
            f"guard_p05_p95=[{guard['p05']:.2f},{guard['p95']:.2f}] ms"
        )
    print(f"\nWrote:\n  {outdir/'fits.csv'}\n  {outdir/'summary.json'}\n  {rawdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
