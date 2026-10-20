"""Build the safe map from a coefficient set (via the cffi core) and validate
the energy model against a fixed-duty measurement. The map is the EXACT firmware
math; only the coefficients differ (measured vs placeholder)."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # twin/

import numpy as np
import pandas as pd
from bindings import core_ffi as core

# provenance of each coefficient
# Provenance of the virtual ground truth (see SENSOR_COEFFICIENTS.md). DS = MMS-30 datasheet,
# TM = to-measure on hardware (M7). The twin always *extracts* these from experiments; the tag
# records where the underlying truth comes from.
PROVENANCE = {
    "P_on": "to-measure (DS: low-power, no figure)",
    "sigma_pw": "DS-bounded (<=85 ug/rtHz)",
    "P_sleep": "to-measure (PD pin; no sleep-current spec)",
    "E_inrush": "to-measure (wake-recovery)",
    "b0": "to-measure (short-sleep transient; DS 30-min residual 0.5 mg)",
    "tau": "to-measure (wake-recovery)",
    "SF": "to-measure (scale-factor)", "SF_ref": "to-measure (scale-factor)", "fs": "config",
    "T_probe": "config", "a_max_exp": "application", "range_limit": "DS (+-30 g)",
    "bias_tc": "measured (-75.5 ug/C, 2026-06-30; DS <=200 ug/C)",
    "dT_op": "config (operating temperature deviation from reference)",
}

# sigma_allowed relaxed to 0.0025 (M7): the measured static noise is 1.74 mg, so an
# application C4 tolerance of 1.5 mg would be unattainable -- the real floor informs it.
DEFAULT_TOL = dict(eps_allowed=0.001, sigma_allowed=0.0025, SF_allowed=0.0255,
                   B_required=0.1, D_max=8.0, S_min=0.10, delta=0.15)


def coeffs_from_extraction(p_on_W: float, noise_g: float, p_sleep_W: float = 0.0005,
                           E_inrush_J: float = 0.005, b0_g: float = 0.0005,
                           tau_s: float = 0.05, SF: float = 1.0,
                           SF_ref: float = 1.0, bias_tc: float = -75.5e-6,
                           dT_op: float = 0.0, compensate: bool = False,
                           bias_tc_resid: float = 3.0e-6) -> dict:
    """Coefficients fully from data: P_on/noise (baseline), P_sleep (sleep-power),
    E_inrush/b0/tau (wake-recovery), SF/SF_ref (scale-factor). range is datasheet (C7).
    bias_tc is the measured static bias-TC (θ-axis); dT_op is the operating temperature
    deviation from reference (set per map to activate the C2 temperature term).

    compensate: temperature is KNOWN (the die-temp is in every frame), so the predictable
    bias_tc·dT can be subtracted. With compensate=True only the *residual* (calibration-limited)
    coefficient `bias_tc_resid` eats the C2 budget — defensibly the bias-TC fit SE (~3 µg/°C),
    vs the full 75.5 µg/°C uncompensated. This is what extends the safe region beyond ±13 °C."""
    eff_bias_tc = bias_tc_resid if compensate else bias_tc
    return dict(P_on=p_on_W, P_sleep=p_sleep_W, E_inrush=E_inrush_J, b0=b0_g,
                tau=tau_s, sigma_pw=noise_g, SF=SF, SF_ref=SF_ref, fs=350.0,
                T_probe=0.5, a_max_exp=1.0, range_limit=30.0,
                bias_tc=eff_bias_tc, dT_op=dT_op)


def build_map(coeffs: dict, tol_kw: dict | None = None,
              ts_grid=None) -> tuple[pd.DataFrame, tuple | None]:
    """Sweep T_sleep -> (P_avg, S, B_eff, class) DataFrame + green interval."""
    tol_kw = tol_kw or DEFAULT_TOL
    ts_grid = np.arange(0.05, 6.0, 0.02) if ts_grid is None else ts_grid
    s, t = core.sensor(**coeffs), core.tol(**tol_kw)
    tg = core.t_guard_min(s, t)
    rows = [dict(T_sleep=float(ts), P_avg_mW=core.p_avg(s, ts, tg) * 1e3,
                 S=core.saving(s, ts, tg), B_eff=core.b_eff(s, ts, tg),
                 klass=core.classify(s, t, ts)) for ts in ts_grid]
    return pd.DataFrame(rows), core.green_interval(s, t)


CLASS_CODE = {"RED": 0, "YELLOW": 1, "GREEN": 2}


def build_map_grid(coeffs_by_mode: dict, tol_kw: dict | None = None,
                   ts_grid=None) -> pd.DataFrame:
    """Classify (method x T_sleep) into RED/YELLOW/GREEN -- the full safe operating
    map, one row of coefficients per sleep mode. Long DataFrame: method, T_sleep, code."""
    tol_kw = tol_kw or DEFAULT_TOL
    ts_grid = np.arange(0.05, 6.0, 0.05) if ts_grid is None else ts_grid
    t = core.tol(**tol_kw)
    rows = []
    for mode, coeffs in coeffs_by_mode.items():
        s = core.sensor(**coeffs)
        for ts in ts_grid:
            rows.append({"method": mode, "T_sleep": float(ts),
                         "code": CLASS_CODE[core.classify(s, t, ts)]})
    return pd.DataFrame(rows)


def validate_energy(fd_df, tsleep: float, tprobe: float, *,
                    p_on: float, p_sleep: float, e_inrush: float = 0.0,
                    t_guard: float = 0.0) -> dict:
    """Validate the §3 energy model OUT OF SAMPLE: predict this fixed-duty run's
    P_avg from coefficients each measured in a SEPARATE experiment --
        P_on      from the baseline (continuous-ON) experiment,
        P_sleep   from the sleep-power experiment (the chosen mode),
        E_inrush  from the wake-recovery experiment (the chosen mode),
    -- then compare to the measured P_avg of THIS run, which the predictor never
    saw. The residual is therefore real (extraction error, ODR quantization,
    partial-cycle boundaries), not the algebraic identity you get when P_on/P_sleep
    are re-read from the run being checked."""
    measured = float((fd_df.current_A * fd_df.voltage_V).mean())

    s = core.sensor(P_on=p_on, P_sleep=p_sleep, E_inrush=e_inrush, T_probe=tprobe)
    predicted = float(core.p_avg(s, tsleep, t_guard))
    diff_pct = (measured - predicted) / predicted * 100.0

    Tc = tprobe + t_guard + tsleep
    return {"computed_W": predicted, "measured_W": measured,
            "p_on_W": p_on, "p_sleep_W": p_sleep, "e_inrush_J": e_inrush,
            "diff_pct": float(diff_pct),
            # per-cycle energy breakdown (J), so the residual can be attributed
            "E_on_J": p_on * tprobe, "E_inrush_J": e_inrush,
            "E_sleep_J": p_sleep * tsleep, "T_cycle_s": Tc}
