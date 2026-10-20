"""Wake-recovery extraction (the crown jewel) + the instrument-design study.
Extracts E_inrush, b0, tau from a recovery capture, and shows how E_inrush
extraction degrades as the sampling rate drops."""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # twin/

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid

from analysis import experiment as ex

TRUTH = {"E_inrush_J": 0.00030, "tau_inrush_s": 43e-6, "b0_g": 0.085, "tau_bias_s": 0.0027}

# per-mode recovery ground truth (M3.1) -- deeper sleep, bigger/slower recovery
# b0_g/tau_bias_s CALIBRATED to hardware (M7, 2026-06-26, firmware BURST; recovery ms-scale).
# rail_off (0.30 mJ) + pd (0.057 mJ) E_inrush_J MEASURED (2026-06-29, scope CSV); vddio
# to-measure (set < pd, pending the jumper move + capture). Mirrors twin_engine.c rec_*.
TRUTH_BY_MODE = {
    "rail_off": {"E_inrush_J": 0.00030,  "b0_g": 0.0850, "tau_bias_s": 0.0027},
    "pd":       {"E_inrush_J": 0.000057, "b0_g": 0.0850, "tau_bias_s": 0.0027},
    "vddio":    {"E_inrush_J": 0.00002,  "b0_g": 0.0010, "tau_bias_s": 0.0020},
}


def _expdecay(t, A, tau):
    return A * np.exp(-t / np.maximum(tau, 1e-9))


# ----- §2.2.1 settled-power E_inrush: T_wake = first time |P-P_on| < p*P_on held
# for T_hold (capped at T_wake_max); choose p on the plateau where E is p-insensitive.

def _settle_index(within, hold_n):
    """First index where `within` stays True for hold_n consecutive samples (-1 if never)."""
    hold_n = max(1, int(hold_n))
    if len(within) < hold_n:
        return -1
    csum = np.concatenate([[0], np.cumsum(within.astype(int))])
    run = csum[hold_n:] - csum[:-hold_n]        # run[i] = within[i:i+hold_n].sum()
    idx = np.where(run == hold_n)[0]
    return int(idx[0]) if idx.size else -1


def t_wake(P, tw, p_on, p, t_hold=0.02, t_wake_max=0.5):
    """Settling endpoint: power within ±p*P_on for t_hold, capped at t_wake_max."""
    within = np.abs(P - p_on) < p * p_on
    dt = float(np.median(np.diff(tw))) if len(tw) > 1 else 0.01
    i = _settle_index(within, round(t_hold / dt) if dt > 0 else 1)
    tend = tw[i] if i >= 0 else t_wake_max
    return min(float(tend), t_wake_max)


def e_inrush_settled(P, tw, p_on, p, t_hold=0.02, t_wake_max=0.5):
    """E_inrush = integral of (P - P_on) over [0, T_wake(p)]."""
    tend = t_wake(P, tw, p_on, p, t_hold, t_wake_max)
    mask = tw <= tend
    if mask.sum() < 2:
        return 0.0, tend
    return float(trapezoid(P[mask] - p_on, tw[mask])), tend


def _flat_windows(E, Eref, tol, min_pts):
    """All maximal contiguous windows [i,j] whose E range <= tol*Eref (>= min_pts wide)."""
    n = len(E)
    out = []
    for i in range(n):
        j = i
        while j + 1 < n and (E[i:j + 2].max() - E[i:j + 2].min()) <= tol * Eref:
            j += 1
        if j - i + 1 >= min_pts:
            out.append((i, j))
    return out


def choose_p(P, tw, p_on, p_grid=None, t_hold=0.02, t_wake_max=0.5,
             flat_tol=0.01, min_pts=3):
    """Sweep p; pick p on the genuine settling plateau (§2.2.1).

    E_inrush(p) is non-increasing: a stricter p (small) integrates more of the
    surge; a looser p truncates it. The settling plateau is the FLAT tread (E
    range within `flat_tol` of the surge magnitude) with the HIGHEST E -- the
    range of p over which the full surge is captured and E is insensitive to p.
    Lower flat treads are premature-trigger truncations, not settling. p* is the
    CENTRE of that tread (robust, not an edge); `plateau_width`/`plateau_ok`
    report whether such a tread exists (it won't if the rate is too low)."""
    if p_grid is None:
        p_grid = np.linspace(0.005, 0.30, 40)
    E = np.array([e_inrush_settled(P, tw, p_on, p, t_hold, t_wake_max)[0] for p in p_grid])
    Eref = E.max() if E.size and E.max() > 0 else 1.0

    wins = _flat_windows(E, Eref, flat_tol, min_pts)
    if wins:
        i, j = max(wins, key=lambda w: E[w[0]:w[1] + 1].mean())   # highest-E flat tread
        c = (i + j) // 2
        p_star, plo, phi = float(p_grid[c]), float(p_grid[i]), float(p_grid[j])
        width, npts, ok = phi - plo, j - i + 1, True
    else:                                   # no insensitive region (rate too low)
        c = int(np.argmax(E))               # fall back to strictest p, flag it
        p_star = plo = phi = float(p_grid[c])
        width, npts, ok = 0.0, 1, False

    E_star, tw_star = e_inrush_settled(P, tw, p_on, p_star, t_hold, t_wake_max)
    return {"p": p_star, "E_inrush_J": E_star, "T_wake_s": tw_star,
            "T_hold_s": t_hold, "T_wake_max_s": t_wake_max,
            "plateau_lo": plo, "plateau_hi": phi, "plateau_width": float(width),
            "plateau_pts": int(npts), "plateau_ok": bool(ok),
            "sweep_p": p_grid, "sweep_E_mJ": E * 1e3}


def extract_recovery(df: pd.DataFrame, p_on_W: float) -> dict:
    """E_inrush = integral of (P - P_on) over recovery; b0,tau from the bias decay."""
    rec = df[df.phase == "recovery"].copy()
    tw = (rec.t_s - rec.t_s.iloc[0]).to_numpy()
    P = (rec.current_A * rec.voltage_V).to_numpy()
    a = rec.accel_g.to_numpy()

    # inrush energy via the §2.2.1 settled-power criterion with plateau-selected p
    sel = choose_p(P, tw, p_on_W)
    e_inrush = sel["E_inrush_J"]
    surge = np.maximum(P - p_on_W, 0.0)   # (kept for the tau_inrush fit below)

    # inrush time constant (fit the surge where it is appreciable)
    tau_inrush = float("nan")
    m = surge > 0.05 * surge.max() if surge.max() > 0 else np.zeros_like(surge, bool)
    if m.sum() >= 3:
        try:
            with warnings.catch_warnings():    # noiseless surge -> covariance warning
                warnings.simplefilter("ignore")
                pi, _ = curve_fit(_expdecay, tw[m], surge[m],
                                  p0=[surge.max(), 0.005], maxfev=20000)
            tau_inrush = abs(float(pi[1]))
        except Exception:                      # noqa: BLE001
            pass

    # bias recovery: accel = c + b0 * exp(-t/tau_bias). De-trend the settled level c BEFORE fitting
    # iff the tail is a genuine small static pedestal — |c| small vs the transient peak (e.g. the
    # temperature baseline bias_tc·dT, a few mg under an ~85 mg spike). If the tail is still a large
    # fraction of the peak the signal hasn't settled (slow tau / short window) so we leave it. This
    # discriminator needs no tau, so a pedestal can't fool a zero-asymptote fit into a wrong slope.
    b0 = tau_bias = float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            peak = float(np.max(np.abs(a)))
            tail = a[tw > 0.9 * tw[-1]]
            c0 = float(np.median(tail)) if len(tail) > 3 else 0.0
            a_use = a - c0 if (abs(c0) > 1e-9 and abs(c0) < 0.10 * peak) else a
            pb, _ = curve_fit(_expdecay, tw, a_use,
                              p0=[a_use[0] if a_use[0] > 0 else 0.005, 0.3], maxfev=20000)
        b0, tau_bias = abs(float(pb[0])), abs(float(pb[1]))
    except Exception:                          # noqa: BLE001
        pass

    settled = a[tw > 5 * tau_bias] if tau_bias == tau_bias and tau_bias > 0 else a
    noise = float(np.std(settled)) if len(settled) > 10 else float(np.std(a))

    return {"E_inrush_J": e_inrush, "tau_inrush_s": tau_inrush,
            "b0_g": b0, "tau_bias_s": tau_bias, "noise_g": noise,
            "p": sel["p"], "T_wake_s": sel["T_wake_s"], "T_hold_s": sel["T_hold_s"],
            "T_wake_max_s": sel["T_wake_max_s"],
            "plateau_lo": sel["plateau_lo"], "plateau_hi": sel["plateau_hi"],
            "plateau_width": sel["plateau_width"], "plateau_ok": sel["plateau_ok"],
            "sweep_p": sel["sweep_p"], "sweep_E_mJ": sel["sweep_E_mJ"]}


def characterize_recovery(odr: float = 500000.0, p_on_W: float = 0.067) -> dict:
    """Run the recovery experiment for EACH sleep mode and extract its coefficients.
    Returns {mode: {E_inrush_J, tau_inrush_s, b0_g, tau_bias_s, noise_g}}.
    Default odr is 500 kHz: the measured tau_inrush (~43 us) needs a fast instrument
    (a scope) to resolve the surge -- 20 kHz undersamples it (see rate_study)."""
    out = {}
    for mode in ("rail_off", "pd", "vddio"):
        # recdur=1.0 for the ms-scale bias-tau fit; offdur short (recovery is the costly phase at high odr)
        df = ex.run_recovery("virtual", offdur=0.05, recdur=1.0, odr=odr, sleepmode=mode)
        out[mode] = extract_recovery(df, p_on_W)
    return out


def rate_study(odrs=(8.6, 25, 50, 100, 250, 1000, 5000, 20000),
               p_on_W: float = 0.067, tau_inrush: float | None = None) -> pd.DataFrame:
    """Run the SAME recovery protocol at each rate; report E_inrush extraction error.
    `tau_inrush` (s) overrides the surge time constant -- the true E_inrush is unchanged
    (the surge integrates to E_inrush either way), so the error is purely a sampling
    effect: a shorter surge needs a higher rate. Use the measured tau to spec the rig."""
    rows = []
    for odr in odrs:
        df = ex.run_recovery("virtual", offdur=0.05, recdur=1.0, odr=float(odr),
                             tau_inrush=tau_inrush)
        ext = extract_recovery(df, p_on_W)
        err = (ext["E_inrush_J"] - TRUTH["E_inrush_J"]) / TRUTH["E_inrush_J"] * 100.0
        rows.append({"odr_hz": float(odr), "E_inrush_mJ": ext["E_inrush_J"] * 1e3,
                     "err_pct": float(err)})
    return pd.DataFrame(rows)
