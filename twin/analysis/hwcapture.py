"""Ingest high-rate hardware captures (scope / Joulescope / PPK2 / MCU burst dump)
into the twin raw-sample schema, so the UNCHANGED recovery extraction
(`analysis.recovery.extract_recovery`, `choose_p`) runs on real silicon.

Two acquisition paths (see RECOVERY_CHARACTERIZATION.md):
- current at >=5 kHz  -> E_inrush, tau_inrush   (the fast instrument across the shunt)
- accel  at >=200 Hz  -> b0, tau_bias           (the MCU burst buffer)

The instrument CSV format is configurable (column name/index, unit scaling, and a
shunt-resistor option to turn a shunt voltage into current), since it depends on the
tool you end up using. Align t=0 to the wake instant (trigger/marker) before calling.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # twin/

import numpy as np
import pandas as pd

from analysis import recovery as rcv

SCHEMA = ("t_s", "accel_g", "current_A", "voltage_V", "temp_C", "drdy", "err", "phase")


def load_xy(path, x_col, y_col, x_scale: float = 1.0, y_scale: float = 1.0,
            comment: str = "#") -> tuple[np.ndarray, np.ndarray]:
    """Generic two-column loader for an instrument CSV. `*_col` is a column name or
    integer index; `*_scale` converts to SI (e.g. y_scale=1e-3 for mA->A, or
    1/R_shunt to turn a shunt voltage into current). Returns (x, y) as SI arrays."""
    df = pd.read_csv(path, comment=comment)
    def col(c):
        return df.iloc[:, c] if isinstance(c, int) else df[c]
    x = col(x_col).to_numpy(float) * x_scale
    y = col(y_col).to_numpy(float) * y_scale
    return x, y


def to_recovery_df(t: np.ndarray, current_A: np.ndarray | None = None,
                   accel_g: np.ndarray | None = None, voltage_V: float = 5.11,
                   t_accel: np.ndarray | None = None) -> pd.DataFrame:
    """Build a twin-schema recovery DataFrame from high-rate arrays. `current_A`
    sets the time base (for E_inrush); `accel_g` (optionally on its own `t_accel`)
    is interpolated onto it (for b0/tau_bias). Missing channels are filled neutrally
    so the corresponding extraction yields NaN rather than crashing."""
    t = np.asarray(t, float); t = t - t[0]
    n = len(t)
    cur = np.zeros(n) if current_A is None else np.asarray(current_A, float)
    if accel_g is None:
        acc = np.zeros(n)
    elif t_accel is not None:                      # different rate -> interpolate
        acc = np.interp(t, np.asarray(t_accel, float) - t_accel[0], np.asarray(accel_g, float))
    else:
        acc = np.asarray(accel_g, float)
    return pd.DataFrame({
        "t_s": t, "accel_g": acc, "current_A": cur, "voltage_V": voltage_V,
        "temp_C": np.nan, "drdy": 1, "err": 0, "phase": "recovery"}, columns=SCHEMA)


def extract_from_capture(current_csv: str, p_on_W: float, *,
                         t_col=0, i_col=1, t_scale: float = 1.0, i_scale: float = 1.0,
                         shunt_ohm: float | None = None, voltage_V: float = 5.11,
                         accel_csv: str | None = None, ta_col=0, a_col=1,
                         ta_scale: float = 1.0, a_scale: float = 1.0) -> dict:
    """Full path: load a high-rate current capture (+ optional accel capture) and run
    the twin's `extract_recovery` unchanged. `shunt_ohm` set => `i_col` is a shunt
    VOLTAGE and current = V/shunt_ohm. Returns E_inrush, tau_inrush, b0, tau_bias, ..."""
    t, iy = load_xy(current_csv, t_col, i_col, t_scale, i_scale)
    current_A = iy / shunt_ohm if shunt_ohm else iy
    accel = t_accel = None
    if accel_csv is not None:
        t_accel, accel = load_xy(accel_csv, ta_col, a_col, ta_scale, a_scale)
    df = to_recovery_df(t, current_A=current_A, accel_g=accel, voltage_V=voltage_V,
                        t_accel=t_accel)
    return rcv.extract_recovery(df, p_on_W)


# ----- firmware BURST block (accel transient -> b0, tau_bias) -----

def _kv(line: str) -> dict:
    return {k: v for tok in line.split()[1:] if "=" in tok
            for k, v in [tok.split("=", 1)]}


def parse_burst(lines) -> tuple[pd.DataFrame, dict]:
    """Parse a firmware BURST block (iterable of text lines, per the burst spec) ->
    (twin-schema recovery DataFrame of the VALID rows, meta dict)."""
    rows, meta, in_block = [], {}, False
    for ln in lines:
        ln = ln.strip()
        if ln.startswith("BURST_BEGIN"):
            in_block = True; meta.update(_kv(ln)); continue
        if ln.startswith("BURST_END"):
            meta.update(_kv(ln)); break
        if not in_block or ln.startswith("t_us"):
            continue
        p = ln.split(",")
        if len(p) == 4:
            try:
                rows.append((int(p[0]), float(p[1]), int(p[2]), int(p[3])))
            except ValueError:
                pass
    valid = [(t / 1e6, a) for (t, a, d, _e) in rows if d == 1]
    if valid:
        df = to_recovery_df(np.array([v[0] for v in valid]),
                            accel_g=np.array([v[1] for v in valid]))
    else:
        df = to_recovery_df(np.array([0.0]), accel_g=np.array([0.0]))
    return df, meta


def recovery_from_burst(lines, p_on_W: float) -> dict:
    """BURST block -> b0, tau_bias (E_inrush is NaN here; that comes from the scope)."""
    df, meta = parse_burst(lines)
    out = rcv.extract_recovery(df, p_on_W)
    out["burst_rate_hz"] = float(meta.get("rate_hz", "nan"))
    out["first_valid_us"] = float(meta.get("first_valid_us", "nan"))
    return out


def transient_from_burst(lines, glitch_g: float = 1.0, settle_frac: float = 0.5,
                         fit_window_s: float = 0.2) -> dict:
    """Robust post-wake bias transient from a BURST block: drop invalid glitch frames
    (|accel|>glitch_g), DE-TREND the static bias (subtract the settled mean), then fit
    the DEVIATION `b0*exp(-t/tau)` over the first `fit_window_s`. b0 is the transient
    amplitude above settled (not the static offset); returns NaN if there's no transient."""
    from scipy.optimize import curve_fit
    rows, in_block = [], False
    for ln in lines:
        ln = ln.strip()
        if ln.startswith("BURST_BEGIN"): in_block = True; continue
        if ln.startswith("BURST_END"): break
        if not in_block or ln.startswith("t_us"): continue
        p = ln.split(",")
        if len(p) == 4:
            try: rows.append((int(p[0]), float(p[1]), int(p[2])))
            except ValueError: pass
    v = [(t / 1e6, a) for (t, a, d) in rows if d == 1 and abs(a) < glitch_g]
    if len(v) < 10:
        return {"b0_g": float("nan"), "tau_bias_s": float("nan"), "n": len(v)}
    t = np.array([x[0] for x in v]); a = np.array([x[1] for x in v])
    first_valid_s = float(t[0]); t = t - t[0]
    settled = float(a[t > t.max() * settle_frac].mean())
    dev = a - settled
    m = t < min(fit_window_s, t.max())
    try:
        with __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            pp, _ = curve_fit(lambda tt, b, tau: b * np.exp(-tt / np.maximum(tau, 1e-6)),
                              t[m], dev[m], p0=[dev[m][0], 0.01], maxfev=20000)
        b0, tau = float(pp[0]), abs(float(pp[1]))
    except Exception:                       # noqa: BLE001
        b0, tau = float("nan"), float("nan")
    return {"b0_g": b0, "tau_bias_s": tau, "settled_g": settled,
            "first_valid_s": first_valid_s, "noise_g": float(dev[t > t.max() * settle_frac].std()),
            "n": int(len(t))}
