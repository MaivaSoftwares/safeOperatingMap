"""Raw burst-run access: every burst of a run directory as a table, from the raw captures.

A burst run directory holds raw/burst_<mode>_<k>.txt and, if the run finished, fits.csv. load_run()
refits the raw captures with the burst tool's own fit (tools/run_r1_burst_repeatability.py,
transient_fit_from_burst, imported, not copied) and uses fits.csv only for the columns the raw
files do not carry. (Extract for this bundle: the functions the paper's scripts use.)
"""
from __future__ import annotations

import importlib.util
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

TWIN = Path(__file__).resolve().parent.parent
REPO = TWIN.parent
DATA = REPO / "tools" / "data"


@lru_cache(maxsize=1)
def _tool():
    spec = importlib.util.spec_from_file_location(
        "run_r1_burst_repeatability", REPO / "tools" / "run_r1_burst_repeatability.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stamp(run_dir: Path) -> datetime | None:
    m = re.search(r"(\d{8}_\d{6})$", Path(run_dir).name)
    return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S") if m else None


@lru_cache(maxsize=None)
def _load(run_dir: str) -> pd.DataFrame:
    d = Path(run_dir)
    tool = _tool()
    rows = []
    for f in sorted((d / "raw").glob("burst_*.txt")):
        m = re.match(r"burst_(.+)_(\d+)\.txt", f.name)
        if not m:
            continue
        lines = f.read_text(errors="ignore").splitlines()
        complete = any(ln.startswith("BURST_END") for ln in lines)
        fit = tool.transient_fit_from_burst(lines, require_err0=True) if complete else {}
        b0, tau = fit.get("b0_g", np.nan), fit.get("tau_bias_s", np.nan)
        mode = m.group(1)
        for known in ("rail_off", "vddio", "pd"):          # burst_baseline_000 etc. are pd
            if mode.startswith(known):
                mode = known
                break
        else:
            mode = "pd"
        rows.append(dict(raw_file=f.name, mode=mode, k=int(m.group(2)),
                         ok=bool(complete and np.isfinite(b0) and np.isfinite(tau)),
                         b0_mg=1e3 * abs(b0), tau_ms=1e3 * tau,
                         fv_ms=1e-3 * fit.get("first_valid_us", np.nan),
                         settled_mg=1e3 * fit.get("settled_g", np.nan)))
    df = pd.DataFrame(rows, columns=["raw_file", "mode", "k", "ok", "b0_mg", "tau_ms", "fv_ms",
                                     "settled_mg"])
    fits = d / "fits.csv"
    if fits.exists() and fits.stat().st_size > 0 and len(df):
        fc = pd.read_csv(fits)
        keep = [c for c in ("raw_file", "phase", "point", "t_since_dose_s", "mode") if c in fc]
        fc = fc[keep].rename(columns={"mode": "mode_fits"})
        fc["_order"] = range(len(fc))             # fits.csv is in capture order; raw file
        df = df.merge(fc, on="raw_file", how="left")   # names sort by phase, not by time
        df = df.sort_values("_order", kind="stable", na_position="last").drop(columns="_order")
        df = df.reset_index(drop=True)
        if "mode_fits" in df:
            df["mode"] = df["mode_fits"].fillna(df["mode"])
            df = df.drop(columns="mode_fits")
    return df


def load_run(run_dir) -> pd.DataFrame:
    """Per-burst table of a run directory (path relative to tools/data, or absolute)."""
    d = Path(run_dir)
    if not d.is_absolute():
        d = DATA / d
    if not (d / "raw").is_dir():
        raise FileNotFoundError(f"no raw captures in {d}")
    return _load(str(d)).copy()
