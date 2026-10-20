"""Bias-vs-temperature characterization for the MMS-30 (the θ-axis / C2 temperature term).

The dominant temperature effect on the safe map is BIAS: the sensor's zero-g offset drifts
with die temperature (DS: up to ~200 µg/°C). Scale-factor TC (~100 ppm/°C) and noise TC need
an impractical ΔT to move the map, so we characterize **bias-TC** here.

Method: the sensor reports its own die temperature in every frame, so with the device held in a
FIXED orientation (gravity component constant) the acceleration reading varies with temperature
ONLY through the bias — so a regression of accel vs. die-temp gives the bias-TC slope directly.
We separate the *reversible* temperature effect from one-time warm-up / dielectric-charging drift
by fitting the heating and cooling legs independently: if both legs give the same slope (low
hysteresis), the effect is genuine temperature dependence, not time drift.

C2 link: the temperature-induced bias must stay inside the C2 wake budget ε. The map can tolerate
a swing of ΔT_crit = ε / bias_TC before C2 needs an explicit temperature term.
"""
from __future__ import annotations

import numpy as np

try:                                    # prefer the canonical implementation in noise.py ...
    from analysis.noise import deglitch
except (ImportError, AttributeError):   # ... but stay self-contained if noise.py is older/out of sync
    def deglitch(accel_g, n_sigma: float = 10.0, min_gap_mg: float = 8.0):
        """Local fallback: reject torn-SPI-read outliers by magnitude (see noise.deglitch)."""
        a = np.asarray(accel_g, float).copy()
        med = np.median(a)
        mad = float(np.median(np.abs(a - med))) * 1.4826
        thr = max(n_sigma * mad, min_gap_mg * 1e-3)
        bad = np.abs(a - med) > thr
        if bad.any():
            idx = np.arange(len(a))
            a[bad] = np.interp(idx[bad], idx[~bad], a[~bad])
        return a, bad


def _fit(T: np.ndarray, a: np.ndarray):
    """Least-squares accel = slope·T + intercept. Returns (slope_g_per_C, intercept, r2)."""
    if len(T) < 3 or np.ptp(T) < 1e-6:
        return float("nan"), float("nan"), float("nan")
    A = np.vstack([T, np.ones_like(T)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, a, rcond=None)
    resid = a - (slope * T + intercept)
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


def bias_tc(t_s, accel_g, temp_c, warmup_s: float = 0.0, eps_mg: float = 1.0) -> dict:
    """Regress bias against die temperature -> bias-TC (µg/°C). `warmup_s` drops a head so the
    one-time cold-start drift doesn't bias the fit; `eps_mg` is the C2 budget for ΔT_crit."""
    from scipy.signal import medfilt
    t = np.asarray(t_s, float)
    a_raw = np.asarray(accel_g, float)
    T = np.asarray(temp_c, float)
    # A torn SPI frame corrupts BOTH axes, so clean both. Reject: (1) null frames where the
    # whole frame read 0 (accel≈0 AND temp≈25.0 °C = 0 LSB), (2) temp spikes (die temp is
    # slow & smooth — it cannot jump >3 °C between samples), (3) accel magnitude outliers.
    null = (np.abs(a_raw) < 1e-5) & (np.abs(T - 25.0) < 1e-3)
    badT = np.abs(T - medfilt(T, 11)) > 3.0
    a_dg, badA = deglitch(a_raw)
    good = ~null & ~badT & ~badA
    bad_rate = 1.0 - float(good.mean())
    m = good & ((t - t[0]) >= warmup_s)
    t, a, T = t[m], a_dg[m], T[m]
    # second pass — LOCAL spike reject: accel legitimately tracks T here, so a global-median cut
    # can miss a one-sample jolt (e.g. a mechanical bump when the heat source is removed). Reject
    # residuals from a smooth baseline; these would otherwise drag the least-squares slope.
    if len(a) > 50:
        res = a - medfilt(a, min(151, (len(a) // 2) * 2 + 1))
        rmad = float(np.median(np.abs(res - np.median(res)))) * 1.4826
        loc = np.abs(res - np.median(res)) < 6 * max(rmad, 1e-4)
        t, a, T = t[loc], a[loc], T[loc]
    if bad_rate > 0.10:
        print(f"  !! DATA QUALITY: {bad_rate*100:.0f}% of frames were corrupt (null/torn) — the SPI link")
        print(f"     was failing during capture (EMI from the heat source? heat-disturbed connections?).")
        print(f"     Fix the capture before trusting this; cleaning cannot rescue a {bad_rate*100:.0f}%-bad run.")

    # BIVARIATE FIT  accel ≈ bias_TC·T + drift·(time) + b0.  On a heat-then-cool HUMP, temperature
    # is non-monotonic while time is monotonic, so the joint fit separates the *reversible*
    # temperature coefficient from the one-way settling drift. A univariate accel~T fit is
    # structurally wrong here — the drift leaks into the slope (that's the low-R² trap).
    tn = (t - t[0]) / max(t[-1] - t[0], 1e-9)
    X = np.vstack([T, tn, np.ones_like(T)]).T
    coef, *_ = np.linalg.lstsq(X, a, rcond=None)
    bT, b_drift, b0 = (float(c) for c in coef)
    resid = a - X @ coef
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    s2 = float(np.sum(resid ** 2)) / max(len(t) - 3, 1)
    try:
        se_ug = float(np.sqrt((s2 * np.linalg.inv(X.T @ X))[0, 0])) * 1e6
    except np.linalg.LinAlgError:
        se_ug = float("nan")
    bias_tc_ug = bT * 1e6
    drift_mg = b_drift * 1e3
    drift_rate = drift_mg / max((t[-1] - t[0]) / 60.0, 1e-9)        # mg/min

    # leg cross-check: split at the TEMPERATURE PEAK and fit each univariately — both legs should
    # land near the bivariate bias-TC (independent corroboration that the slope is reversible).
    kf = min(51, (len(T) // 10) * 2 + 1)
    ipk = int(np.argmax(medfilt(T, kf))) if len(T) > 20 else len(T) // 2
    slope_up = _fit(T[:ipk], a[:ipk])[0] * 1e6 if ipk > 10 else float("nan")
    slope_dn = _fit(T[ipk:], a[ipk:])[0] * 1e6 if len(T) - ipk > 10 else float("nan")
    hyst = abs(slope_up - slope_dn) if np.isfinite(slope_up) and np.isfinite(slope_dn) else float("nan")

    dT_span = float(np.ptp(T))
    dT_crit = eps_mg * 1e-3 / abs(bT) if np.isfinite(bT) and bT != 0 else float("inf")
    # reversibility: a usable run goes UP and DOWN. Peak in the interior -> both legs substantial.
    rev_frac = min(ipk, len(T) - ipk) / max(len(T), 1)
    monotonic = rev_frac < 0.15

    print(f"=== bias-TC: n={len(a)}, ΔT span={dT_span:.2f} °C ({T.min():.2f}–{T.max():.2f}) ===")
    if dT_span < 2.0:
        print(f"  !! ΔT span only {dT_span:.2f} °C — too small for a reliable slope; force a wider swing.")
    if monotonic:
        print(f"  !! TEMPERATURE WAS ~MONOTONIC (peak at the edge) — temp is collinear with time, so")
        print(f"     bias-TC cannot be separated from drift. Re-run: settle, then heat UP and cool DOWN.")
    flag = "   [UNRELIABLE — see warning]" if (monotonic or bad_rate > 0.10) else ""
    print(f"  bias-TC          = {bias_tc_ug:+.1f} ± {se_ug:.1f} µg/°C   (DS spec ≤ 200 µg/°C)   "
          f"R²={r2:.3f}{flag}")
    print(f"  drift (separated)= {drift_mg:+.2f} mg over run ({drift_rate:+.2f} mg/min)")
    legs_agree = (np.isfinite(slope_up) and np.isfinite(slope_dn)
                  and np.sign(slope_up) == np.sign(slope_dn))
    leg_note = ("same sign confirms it" if legs_agree
                else "OPPOSITE SIGNS — legs do not corroborate; treat bias-TC as unreliable here")
    print(f"  leg cross-check  : heating {slope_up:+.1f} | cooling {slope_dn:+.1f} µg/°C "
          f"(univariate, drift-affected — they BRACKET the bias-TC, {leg_note})")
    print(f"  C2: ΔT_crit (bias=ε={eps_mg:g} mg) = {dT_crit:.1f} °C  -> "
          f"{'C2 needs a temp term within normal swings' if dT_crit < 10 else 'C2 robust to ordinary ΔT'}")
    return {"bias_tc_ug_per_C": bias_tc_ug, "bias_tc_se_ug_per_C": se_ug, "r2": r2,
            "drift_mg": drift_mg, "drift_mg_per_min": drift_rate, "dT_span_C": dT_span,
            "slope_up_ug_per_C": slope_up, "slope_dn_ug_per_C": slope_dn,
            "hysteresis_ug_per_C": hyst, "dT_crit_C": dT_crit, "legs_same_sign": bool(legs_agree),
            "reversibility": float(rev_frac), "monotonic_unreliable": bool(monotonic),
            "bad_frame_rate": bad_rate, "intercept_g": b0, "n": int(len(a))}


def _reject_temp_spikes(T, win: int = 21, n_mad: float = 6.0, floor_C: float = 0.20,
                        phys_lo: float = 12.0, phys_hi: float = 55.0):
    """Torn-SPI-frame rejection on the die-temperature channel (Hampel filter).

    A torn SPI read corrupts a single frame's reported temperature (and the accel
    sampled in that same frame), so a spike is a LONE sample far from its neighbours;
    a real thermal ramp, however fast, moves smoothly across many frames. Flags a
    frame as bad when it is either outside the physically possible passive-heat range
    [phys_lo, phys_hi] °C, or more than n_mad LOCAL MADs from a rolling median. The
    threshold is adaptive (MAD-based) with an absolute floor: unlike the old fixed
    ±3 °C test it survives clusters of adjacent spikes (the rolling median stays put
    when most of its window is good) and does not clip a fast-but-real ramp (whose
    samples sit close to their own local median). Returns the boolean 'bad' mask; the
    caller drops those frames from BOTH channels, since the whole frame is suspect.
    phys_lo/phys_hi are this bench's passive-heat limits (die idles ~25 °C, warms to
    ~37 °C at the GREEN collapse; forced overheat has not exceeded ~44 °C) with margin —
    widen them for a different rig. Peak detection downstream is median-filtered, so a
    stray survivor cannot misplace the leg split even if one slips through.
    """
    from scipy.signal import medfilt
    T = np.asarray(T, float)
    win = int(win) | 1                                  # medfilt needs an odd window
    if len(T) < win:
        return ~np.isfinite(T) | (T < phys_lo) | (T > phys_hi)
    med = medfilt(T, win)
    dev = np.abs(T - med)
    mad = medfilt(dev, win) * 1.4826                    # local robust scale (spike-immune)
    thr = np.maximum(n_mad * mad, floor_C)              # floor so quiet stretches don't over-flag
    return (dev > thr) | (T < phys_lo) | (T > phys_hi) | ~np.isfinite(T)


def loop_width_mg(t_s, accel_g, temp_c, warmup_s: float = 0.0) -> dict:
    """Loop-width thermal hysteresis h (mg): the MAX bias offset between the heating
    and cooling legs at MATCHED reported die temperature, after the one-way settling
    drift is removed. This is the metric the C2 wake budget (ε ≈ 1 mg) is stated in —
    distinct from `bias_tc()`'s slope hysteresis (µg/°C). Physically h is opened by the
    thermal lag between the die-temperature *sensor* and the bias-generating structure:
    on a fast ramp the two sit at different temperatures, so at a matched *reported*
    temperature the legs disagree; as the ramp slows the lag closes and h -> 0. Measuring
    h vs |dθ/dt| is what grounds r_max (see docs/hysteresis_rate_experiment_plan_2026-08-03).

    Drift handling: at a matched temperature θ the raw leg difference is (lag loop) + (one-way
    settling drift accumulated over the TIME GAP between visiting θ on the up-leg vs the
    down-leg). That gap Δt(θ) is ~0 at the temperature peak and large at the edges, so the
    drift term is ∝ Δt(θ) while the reversible loop is ~independent of Δt. We therefore regress
    the raw difference on Δt(θ): the slope is the drift rate (removed), the residual is the loop.
    This separates drift from hysteresis WITHOUT a linear-in-time fit, which would misread the
    loop itself as drift (up-leg early, down-leg late sit on opposite sides of any time trend).
    """
    from scipy.signal import medfilt
    t = np.asarray(t_s, float)
    a_raw = np.asarray(accel_g, float)
    T = np.asarray(temp_c, float)
    n_raw = len(T)
    # frame cleaning: null frames, torn-SPI temperature spikes, accel-magnitude outliers.
    # A torn SPI read corrupts the WHOLE frame, so a flagged temperature spike drops that
    # frame's accel too (not just the temperature) — see _reject_temp_spikes.
    null = (np.abs(a_raw) < 1e-5) & (np.abs(T - 25.0) < 1e-3)
    badT = _reject_temp_spikes(T)
    n_spikes = int(badT.sum())
    spike_frac = n_spikes / max(n_raw, 1)
    a_dg, _ = deglitch(a_raw)
    good = ~null & ~badT
    m = good & ((t - t[0]) >= warmup_s)
    t, a, T = t[m], a_dg[m], T[m]
    n = len(t)
    nan = float("nan")
    spike = {"n_spikes": n_spikes, "spike_frac": float(spike_frac), "trimmed_tail_s": 0.0}
    if spike_frac > 0.02:
        print(f"  !! {n_spikes} torn temperature frames rejected ({100*spike_frac:.1f}% of the "
              f"capture) — check the heat source is passive (no fan/motor) and SPI wiring.")
    if n < 40 or np.ptp(T) < 2.0:
        return {"h_mg": nan, "rate_C_per_min": nan,
                "overlap_C": float(np.ptp(T)) if n else 0.0,
                "monotonic_unreliable": True, "n": int(n), **spike}
    kT = max(3, min(51, (n // 10) * 2 + 1))
    ipk = int(np.argmax(medfilt(T, kT)))                       # temperature peak = leg split
    rev_frac = min(ipk, n - ipk) / max(n, 1)
    if rev_frac < 0.15:
        # Rescue only: a fixed-duration capture (e.g. a 45-min run whose heat+cool hump
        # finishes in ~10 min) keeps logging at room temperature long after the die has
        # cooled back down. That flat baseline tail carries NO hysteresis — both legs have
        # already met — but by inflating n after an early peak it pushes rev_frac under the
        # guard and would discard an otherwise clean run. Drop the tail past the point where
        # the die returns to within 0.5 °C of its 15-s settle baseline, then re-split. This
        # fires ONLY when the run would otherwise be rejected, so runs that already yield an
        # h (the 2.25 / fast anchors) are never re-baselined; it never touches the peak or
        # the heating leg. On the slow unit-A run h is stable wherever the dead tail is cut.
        settle = (t - t[0]) <= 15.0
        base0 = float(np.median(T[settle])) if settle.sum() >= 5 else float(np.median(T[:max(5, n // 20)]))
        back = np.where((np.arange(n) > ipk) & (T <= base0 + 0.5))[0]
        if len(back):
            cut = min(n, int(back[0]) + max(5, n // 100))      # small margin past cool-back
            if cut < n - 5:
                spike["trimmed_tail_s"] = float(t[-1] - t[cut - 1])
                t, a, T = t[:cut], a[:cut], T[:cut]
                n = len(t)
                kT = max(3, min(51, (n // 10) * 2 + 1))
                ipk = int(np.argmax(medfilt(T, kT)))
                rev_frac = min(ipk, n - ipk) / max(n, 1)
                print(f"  (rescued: trimmed {spike['trimmed_tail_s']:.0f}s flat cool-back tail "
                      f"— die had returned to {base0 + 0.5:.1f} °C, no hysteresis beyond it)")
    if rev_frac < 0.15:                                        # peak at an edge -> ~monotonic
        return {"h_mg": nan, "rate_C_per_min": nan, "overlap_C": float(np.ptp(T)),
                "monotonic_unreliable": True, "n": int(n), **spike}
    Tu, au, tu = T[:ipk], a[:ipk], t[:ipk]                     # heating leg
    Td, ad, td = T[ipk:], a[ipk:], t[ipk:]                     # cooling leg
    lo, hi = max(Tu.min(), Td.min()), min(Tu.max(), Td.max())  # common θ overlap
    if hi - lo < 1.0:
        return {"h_mg": nan, "rate_C_per_min": nan, "overlap_C": float(hi - lo),
                "monotonic_unreliable": False, "n": int(n), **spike}
    # bias is a smooth function of temperature within a leg, so fit each leg as a low-order
    # polynomial in T; differencing the two fits gives a noise-free loop (a raw max|Δ| over
    # samples is an extreme-value statistic that noise inflates — the no-lag case must read 0).
    su, sd = np.argsort(Tu), np.argsort(Td)                    # ascending T for interp/polyfit
    grid = np.linspace(lo, hi, 200)
    b_up = np.polyval(np.polyfit(Tu, au, 2), grid)
    b_dn = np.polyval(np.polyfit(Td, ad, 2), grid)
    diff = b_dn - b_up
    # time gap between visiting θ on each leg; the one-way drift over the run scales with it,
    # the reversible loop does not -> regress out the drift-proportional-to-gap component.
    t_up = np.interp(grid, Tu[su], tu[su])
    t_dn = np.interp(grid, Td[sd], td[sd])
    dgap = t_dn - t_up
    G = np.vstack([dgap, np.ones_like(dgap)]).T
    (drift_slope, _icpt), *_ = np.linalg.lstsq(G, diff, rcond=None)
    loop = diff - drift_slope * dgap                           # drift-over-gap removed
    h_mg = float(np.max(np.abs(loop))) * 1e3
    # Per-leg ramp rate = robust linear slope of T vs t while each leg TRAVERSES the overlap
    # band [lo,hi] (the same band h is measured over). A median |dθ/dt| CANNOT be used here:
    # the die temperature is quantized to 0.01 °C, so on a slow ramp most central differences
    # are 0 or one quantum and the median collapses onto a ~2.2 °C/min quantization FLOOR that
    # is blind to the true rate (a 0.6 °C/min run read 2.24). A least-squares slope over the
    # band's hundreds of samples averages the quantum steps out and recovers the real rate.
    def _band_rate(Tl, tl):
        mb = (Tl >= lo) & (Tl <= hi)
        if mb.sum() < 10:
            return float("nan")
        A = np.vstack([tl[mb], np.ones(int(mb.sum()))]).T
        return abs(float(np.linalg.lstsq(A, Tl[mb], rcond=None)[0][0])) * 60.0
    rate_up = _band_rate(Tu, tu)                               # heating-leg ramp rate
    rate_dn = _band_rate(Td, td)                              # cooling-leg ramp rate
    rate = rate_up                                            # r_max x-axis = the HEATING rate
    # Cooling here is a passive coast, usually slower than active heating; h then reflects
    # mainly the heating-leg lag measured against the near-quasi-static cooling leg, so rate_up
    # is the rate to plot. Flag a large up/down asymmetry so a symmetric run (where both legs
    # lag and the offset ~doubles) is not pooled with an asymmetric one on the same h axis.
    asym = max(rate_up, rate_dn) / max(min(rate_up, rate_dn), 1e-9)
    rate_ambiguous = bool(asym > 2.0)
    print(f"=== loop-width hysteresis: h = {h_mg:.2f} mg  @ heating rate {rate_up:.2f} °C/min "
          f"(cooling {rate_dn:.2f}, overlap {hi - lo:.1f} °C, n={n}) ===")
    if rate_ambiguous:
        print(f"  !! ASYMMETRIC RAMP: heating/cooling differ {asym:.1f}x (passive coast) — h is the "
              f"heating-leg lag vs a slow cooling reference; plot it at rate_up, not a symmetric rate.")
    print(f"  {'h < 1 mg budget — rate OK for this ramp' if h_mg < 1.0 else 'h EXCEEDS the 1 mg budget at this ramp rate — slow the ramp'}")
    return {"h_mg": h_mg, "rate_C_per_min": rate, "rate_up_C_per_min": rate_up,
            "rate_dn_C_per_min": rate_dn, "rate_ambiguous": rate_ambiguous,
            "overlap_C": float(hi - lo), "monotonic_unreliable": False, "n": int(n), **spike}


if __name__ == "__main__":          # synthetic self-check: known bias-TC recovered
    rng = np.random.default_rng(0)
    n = 4000
    t = np.linspace(0, 900, n)                              # 15 min
    T = 25 + 8 * np.sin(np.pi * t / t[-1])                  # heat up then cool: 25->33->25 °C
    tc_true = 150e-6                                        # 150 µg/°C
    a = 0.5 + tc_true * (T - 25) + rng.normal(0, 0.7e-3, n) # 0g axis + bias-TC + noise
    r = bias_tc(t, a, T)
    print(f"\nself-check: target 150 µg/°C, recovered {r['bias_tc_ug_per_C']:.1f} µg/°C")

    # loop-width self-check: inject a die-sensor-to-structure lag (pure time shift) so the
    # bias follows a LAGGED temperature -> a known non-zero hysteresis loop; no lag -> ~0.
    T_struct = np.interp(t - 90.0, t, T, left=T[0], right=T[-1])   # structure lags 90 s
    a_lag = 0.5 + tc_true * (T_struct - 25) + rng.normal(0, 0.3e-3, n)
    h_none = loop_width_mg(t, a, T)["h_mg"]
    h_lag = loop_width_mg(t, a_lag, T)["h_mg"]
    print(f"self-check: loop width  no-lag {h_none:.2f} mg  |  90 s lag {h_lag:.2f} mg "
          f"({'OK: lag opens a loop' if h_lag > h_none + 0.3 else 'UNEXPECTED'})")

    # spike-rejection self-check: inject torn-SPI temperature spikes (and matching accel
    # corruption in the same frames) into the lagged run; h must be unchanged and the spikes
    # counted. Proves _reject_temp_spikes strips torn frames without moving the real loop.
    T_spk, a_spk = T.copy(), a_lag.copy()
    idx = rng.choice(n, size=40, replace=False)
    T_spk[idx] += rng.uniform(80, 120, size=idx.size)          # 100+ °C torn frames
    a_spk[idx] += rng.uniform(-2, 2, size=idx.size)            # same frames' accel is garbage too
    r_spk = loop_width_mg(t, a_spk, T_spk)
    print(f"self-check: with 40 injected spikes  h {r_spk['h_mg']:.2f} mg  ({r_spk['n_spikes']} "
          f"rejected) ({'OK: spikes stripped, h matches' if abs(r_spk['h_mg'] - h_lag) < 0.1 and r_spk['n_spikes'] >= 40 else 'UNEXPECTED'})")
