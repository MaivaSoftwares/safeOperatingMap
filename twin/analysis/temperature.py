"""Temperature axis (v2). Characterizes the coefficients across temperature and
shows the safe map is θ-dependent: the map is 3-D (method × T_sleep × θ); at a
given temperature it's the 2-D slice. GREEN shrinks at temperature extremes."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # twin/

import pandas as pd
from analysis import experiment as ex
from analysis import recovery as rcv
from analysis import mapbuild as mb

DEFAULT_TEMPS = (-40, -20, 0, 25, 50, 85)
RAIL_OFF_PSLEEP = 2e-5


def sweep_temperature(temps=DEFAULT_TEMPS, device: str = "virtual",
                      compensate: bool = False) -> pd.DataFrame:
    """Characterize (rail_off) coefficients at each temperature -> one row per θ. With
    compensate=True the known bias_tc·dT is subtracted, so only the residual eats the C2 budget
    and GREEN survives far wider in temperature (see coeffs_from_extraction)."""
    rows = []
    for T in temps:
        base = ex.extract_baseline(ex.run_baseline(device, n=2000, odr=100.0, temp=T))
        rec = rcv.extract_recovery(
            ex.run_recovery(device, offdur=2.0, recdur=1.0, odr=20000.0,
                            sleepmode="rail_off", temp=T), base["P_on_W"])
        sf = ex.extract_scalefactor(ex.run_scalefactor(device, offdur=2.0, odr=200.0, temp=T))
        coeffs = mb.coeffs_from_extraction(
            base["P_on_W"], base["noise_g"], RAIL_OFF_PSLEEP,
            rec["E_inrush_J"], rec["b0_g"], rec["tau_bias_s"],
            SF=sf["SF"]["rail_off"], SF_ref=sf["SF_ref"],
            dT_op=T - 25.0,        # activate the C2 static-bias temperature term at this θ
            compensate=compensate)
        _, gi = mb.build_map(coeffs)
        rows.append({"temp_C": T, "P_on_W": base["P_on_W"], "noise_mg": base["noise_g"] * 1e3,
                     "b0_mg": rec["b0_g"] * 1e3, "SF_dev_pct": sf["dev"]["rail_off"] * 100,
                     "green_lo": gi[0] if gi else None, "green_hi": gi[1] if gi else None,
                     "green_width_s": (gi[1] - gi[0]) if gi else 0.0})
    return pd.DataFrame(rows)


def theta_reach(bias_tc: float = -75.5e-6, bias_tc_resid: float = 3.0e-6,
                E_inrush: float = 0.0003, dT_max: float = 200.0) -> dict:
    """Max |ΔT_op| (°C from calibration) at which the map still has GREEN — uncompensated (full
    bias_tc) vs compensated (residual only). The bias-TC measurement thus both BOUNDS the
    uncompensated safe region and, via compensation, EXTENDS it across the operating range."""
    from bindings import core_ffi as core
    tol = core.tol(eps_allowed=0.001, sigma_allowed=0.0025, SF_allowed=0.0255,
                   B_required=0.5, D_max=2.0, S_min=0.0, delta=0.15)
    out = {}
    for label, tc in (("uncompensated", bias_tc), ("compensated", bias_tc_resid)):
        reach = 0.0
        dT = 0.0
        while dT <= dT_max:
            s = core.mms30(E_inrush=E_inrush, dT_op=dT)
            s.bias_tc = float(tc)
            if core.green_interval(s, tol) is None:
                break
            reach = dT
            dT += 0.5
        out[label] = reach
    return out
