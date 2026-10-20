"""Paper 1 revision, Associate Editor point 1: the map guard against existing practice.

Every method below is a rule for when a freshly woken reading is first trusted. All are
evaluated on the same wakes: the pd burst captures of units A and B (2026-07-06 to
2026-10-03). A burst is a wake followed by continuous sampling, so each wake carries its own
settled level and the bias at any trust time is read from the capture without a new run.

  B0  continuous-ON              no wake; the reference
  B1  first valid sample         the woken sensor is treated as ideal
  B2  fixed wait from practice   a warm-up interval of about 1 s before data collection
                                 (Zauli et al. 2026), and the start-up of a whole switched
                                 platform, 0.92 s (Fu et al. 2018) and 0.95 s (Sarwar et al.
                                 2020); all longer than the 0.82 s capture
  B3  turn-on compensation       the calibration-day transient subtracted from the reading,
                                 trusted w ms after the first valid sample
  B4  frozen calibrated guard    the calibration day's guard, never updated
  B5  map guard, current         the guard computed from the same session's coefficients
  B6  slope rule, per wake       the output is trusted once its fitted slope stays below a
                                 threshold in three consecutive windows (the rule of Huang et
                                 al. 2026), window and threshold set on the calibration day

Bias at first trust, two estimates:
  measured   run mean of the first sample at or after the trust time, minus each burst's own
             settled level; no model; standard error = noise / sqrt(bursts in the run)
  fitted     each burst's fitted single-exponential transient at the trust time; low noise,
             blind to the slow term (r3_2_recovery_models_2026-10-08.md)
A single sample carries 1.6 to 1.7 mg of noise, so "above the 1 mg tolerance" is judged on
run means and on fitted values, never on one sample.

Limits, stated in the results file too:
  - B2's times are other nodes' figures applied to this sensor. Zauli et al. give their wait
    without a source or criterion and tie it to the start of an acquisition; Fu et al. and
    Sarwar et al. report how long their platform takes to start, not a wait they chose.
  - B3 is our implementation of the model-and-subtract idea on this transient, not a rerun of
    any published compensator.
  - B6 is our implementation of slope detection on this transient, not a rerun of Huang et
    al.'s detector: their low-pass filter, window and threshold belong to a drift of 18.8 s
    and do not carry over to one of 1.5 ms, and their window overlap is not stated. Its
    setting is chosen on the calibration day using the bias against the settled level, which
    is our characterisation; the threshold has no link to the tolerance of its own.
  - B5 here is the guard recomputed from each session's own bursts (the choice of trust time
    is in-sample; the measured reading is not). Tracking between sessions is evidenced by the
    hardware campaigns, not by this table.
  - Energy comes from the measured coefficients through the firmware's energy model
    (core/safemap.c), not from a metered run per method.
  - Many runs were taken in deliberately provoked states (cold starts, power-cycle tests), so
    fractions describe the range of states met, not time in deployment.

Run: twin/.venv/bin/python twin/analysis/p1_benchmark.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

TWIN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TWIN))
sys.path.insert(0, str(TWIN / "analysis"))
import a2_transfer as a2  # noqa: E402
import paper2_runs as runs  # noqa: E402
from bindings import core_ffi as core  # noqa: E402
from p1_recovery_models import RUNS as CAL_RUNS  # noqa: E402
from p1_revision_checks import FV_BAND_MS, TAU_BAND_MS, record_runs, unit_of  # noqa: E402

BUDGET_MG = 0.85                 # (1 - delta) eps
TOL_MG = 1.0                     # eps
PRACTICE_WAITS_S = {             # the waits reported by Zauli 2026, Fu 2018 and Sarwar 2020
    "B2 fixed warm-up, about 1 s (Zauli 2026)": 1.0,
    "B2 platform start-up, 0.92 s (Fu 2018)": 0.92,
    "B2 platform start-up, 0.95 s (Sarwar 2020)": 0.95,
}
STARTUP_GUARD_S = 0.020          # manuscript Sec. III-F
HEAD_S = 0.100                   # part of each capture kept (trust-time readings, slope rule)
COMP_WAITS_MS = (0.0, 2.0, 4.0)
BANDWIDTHS_HZ = (0.5, 5.0, 20.0)
MIN_RUN_BURSTS = 10              # run-mean noise <= about 0.5 mg
CAMPAIGN_TRUST_S = 0.016         # earliest first trust in the hardware campaigns (findings 2.2)
FLOOR_S = 0.040                  # main transient gone: what the measured estimate itself resolves
SLOW_MG = {"A": 0.21, "B": 0.0}  # slow term (r3_2_recovery_models); not resolved on unit B
SLOPE_RUN = 3                    # consecutive windows below the threshold (Huang et al. 2026)
SLOPE_WINDOWS = (4, 6, 8, 12, 16, 24)        # samples per window, one sample apart
SLOPE_FACTORS = (1.0, 2.0, 4.0, 8.0, 16.0)   # threshold as a multiple of budget / tau
INPUT_AMPS_MG = (10.0, 50.0, 200.0)          # simulated input added to the recorded wakes
INPUT_FREQS_HZ = (5.0, 20.0, 100.0)
COEFFS = {"A": a2.UNIT_A, "B": a2.UNIT_B}


@dataclass
class Burst:
    unit: str
    run: str
    t: np.ndarray        # s from the wake command, first HEAD_S of the capture
    dev: np.ndarray      # mg from the burst's settled level, signed so the transient is positive
    fv: float            # s, first valid sample
    b0: float            # mg, fitted amplitude at first valid
    tau: float           # s, fitted time constant

    @property
    def normal(self) -> bool:
        return (FV_BAND_MS[0] <= 1e3 * self.fv <= FV_BAND_MS[1]
                and TAU_BAND_MS[0] <= 1e3 * self.tau <= TAU_BAND_MS[1])

    def need(self, budget_mg: float | None = None) -> float:
        """Guard (s from the wake command) at which the fitted transient reaches the budget
        (BUDGET_MG as it stands when called, so a caller can change the module's budget)."""
        budget_mg = BUDGET_MG if budget_mg is None else budget_mg
        return self.fv + self.tau * np.log(max(self.b0, 1e-9) / budget_mg)

    def fitted(self, t_trust: float) -> float:
        return self.b0 * np.exp(-(t_trust - self.fv) / self.tau)

    def sample(self, t_trust: float) -> tuple[float, float]:
        """(time, deviation) of the first sample at or after t_trust; nan if past the head."""
        i = int(np.searchsorted(self.t, t_trust - 1e-9))
        if i >= len(self.t):
            return float("nan"), float("nan")
        return float(self.t[i]), float(self.dev[i])


def make_burst(unit: str, run: str, t: np.ndarray, a_g: np.ndarray, fit: dict) -> Burst | None:
    b0, tau = fit.get("b0_g", np.nan), fit.get("tau_bias_s", np.nan)
    if not (np.isfinite(b0) and np.isfinite(tau)) or b0 == 0 or tau <= 0:
        return None
    sign = 1.0 if b0 > 0 else -1.0
    head = t <= t[0] + HEAD_S
    return Burst(unit, run, t[head], 1e3 * sign * (a_g[head] - fit["settled_g"]), float(t[0]),
                 1e3 * abs(b0), float(tau))


def load_record(mode: str = "pd") -> list[Burst]:
    """Every complete burst of the record in one mode, refitted with the burst tool's own fit."""
    tool = runs._tool()
    out = []
    for d in record_runs():
        run, raw = str(d.relative_to(runs.DATA)), d / "raw"
        for f in sorted(raw.glob("burst_*.txt")):
            name = f.name[len("burst_"):]
            m = next((k for k in ("rail_off", "vddio", "pd") if name.startswith(k)), "pd")
            if m != mode:
                continue
            lines = f.read_text(errors="ignore").splitlines()
            if not any(ln.startswith("BURST_END") for ln in lines):
                continue
            s, _ = tool.usable_samples(lines, require_err0=True)
            if len(s) < 100:
                continue
            fit = tool.transient_fit_from_burst(lines, require_err0=True)
            b = make_burst(unit_of(run), run, np.array([v[0] for v in s]),
                           np.array([v[1] for v in s]), fit)
            if b is not None:
                out.append(b)
    return out


def calibration(record: list[Burst], unit: str) -> dict:
    """The calibration day's medians: what B3 subtracts and what B4 freezes."""
    cal = [b for b in record if b.run == CAL_RUNS[unit] and b.normal]
    b0 = float(np.median([b.b0 for b in cal]))
    tau = float(np.median([b.tau for b in cal]))
    fv = float(np.median([b.fv for b in cal]))
    return dict(b0=b0, tau=tau, fv=fv, guard=fv + tau * np.log(b0 / BUDGET_MG), n=len(cal))


def by_run(record: list[Burst], unit: str, min_bursts: int = 1) -> dict[str, list[Burst]]:
    out: dict[str, list[Burst]] = {}
    for b in record:
        if b.unit == unit and b.normal:
            out.setdefault(b.run, []).append(b)
    return {r: bs for r, bs in out.items() if len(bs) >= min_bursts}


def methods(cal: dict, slow_mg: float = 0.0) -> dict:
    """name -> (trust time for a burst in its run, subtracted model at a sample time or None)."""
    def comp(t):
        return cal["b0"] * np.exp(-(t - cal["fv"]) / cal["tau"])

    m = {"B1 first valid": (lambda b, bs: b.fv, None)}
    for w in COMP_WAITS_MS:
        m[f"B3 compensation, +{w:g} ms"] = (lambda b, bs, w=w: b.fv + 1e-3 * w, comp)
    m["B4 frozen guard"] = (lambda b, bs: cal["guard"], None)
    m["B5 map guard, current"] = (lambda b, bs: float(np.median([x.need() for x in bs])), None)
    if slow_mg:
        m["B5 with slow-term budget"] = (
            lambda b, bs: float(np.median([x.need(BUDGET_MG - slow_mg) for x in bs])), None)
    m["first trust 16 ms (campaigns)"] = (lambda b, bs: CAMPAIGN_TRUST_S, None)
    m["fixed 20 ms"] = (lambda b, bs: STARTUP_GUARD_S, None)
    m["floor: reading at 40 ms"] = (lambda b, bs: FLOOR_S, None)
    return m


def evaluate(groups: dict[str, list[Burst]], trust, model) -> dict:
    """Bias at first trust over runs: measured run means and fitted per-burst values (mg)."""
    run_mean, run_se, run_fit, burst_fit, guards = [], [], [], [], []
    for bs in groups.values():
        meas, fit = [], []
        for b in bs:
            tt = trust(b, bs)
            ts, d = b.sample(tt)
            sub_s = model(ts) if model else 0.0
            sub_f = model(tt) if model else 0.0
            meas.append(d - sub_s)
            fit.append(b.fitted(tt) - sub_f)
            guards.append(tt)
        burst_fit += fit
        run_fit.append(float(np.median(fit)))
        if len(bs) >= MIN_RUN_BURSTS and np.all(np.isfinite(meas)):
            run_mean.append(float(np.mean(meas)))
            run_se.append(float(np.std(meas, ddof=1) / np.sqrt(len(meas))))
    rm, rf, bf = np.array(run_mean), np.array(run_fit), np.array(burst_fit)
    return dict(
        guard_ms=1e3 * float(np.median(guards)),
        n_runs=len(rf), n_bursts=len(bf), n_runs_meas=len(rm),
        meas_mean=float(np.mean(rm)) if len(rm) else np.nan,
        meas_mean_se=float(np.std(rm, ddof=1) / np.sqrt(len(rm))) if len(rm) > 1 else np.nan,
        meas_med=float(np.median(np.abs(rm))) if len(rm) else np.nan,
        meas_p95=float(np.percentile(np.abs(rm), 95)) if len(rm) else np.nan,
        meas_max=float(np.max(np.abs(rm))) if len(rm) else np.nan,
        meas_se=float(np.median(run_se)) if len(rm) else np.nan,
        meas_over=int(np.sum(np.abs(rm) > TOL_MG)),
        meas_over_2se=int(np.sum(np.abs(rm) - 2 * np.array(run_se) > TOL_MG)),
        fit_med=float(np.median(np.abs(rf))), fit_max=float(np.max(np.abs(rf))),
        fit_min=float(np.min(rf)),
        runs_over=int(np.sum(np.abs(rf) > TOL_MG)),
        bursts_over=float(np.mean(np.abs(bf) > TOL_MG)),
    )


def bias_table(record: list[Burst], unit: str) -> pd.DataFrame:
    cal = calibration(record, unit)
    groups = by_run(record, unit)
    rows = []
    for name, (trust, model) in methods(cal, SLOW_MG[unit]).items():
        rows.append(dict(method=name, **evaluate(groups, trust, model)))
    return pd.DataFrame(rows)


def slope_detect(b: Burst, thr: float, window: int, step: int = 1, run: int = SLOPE_RUN) -> int:
    """Index of the sample at which a per-wake slope rule declares the output steady, -1 if never.

    A least-squares line is fitted to `window` consecutive samples, windows `step` samples
    apart. The output is declared steady when `run` consecutive windows have |slope| < thr
    (mg/s), at the last sample of the last of them.
    """
    n_win = (len(b.t) - window) // step + 1
    if n_win < run:
        return -1
    tw = sliding_window_view(b.t, window)[::step]
    xw = sliding_window_view(b.dev, window)[::step]
    tc = tw - tw.mean(axis=1, keepdims=True)
    ok = np.abs((tc * xw).sum(axis=1) / (tc ** 2).sum(axis=1)) < thr
    hit = ok[: n_win - run + 1].copy()
    for j in range(1, run):
        hit &= ok[j: n_win - run + 1 + j]
    first = np.flatnonzero(hit)
    return int((first[0] + run - 1) * step + window - 1) if len(first) else -1


def slope_trust(thr: float, window: int, step: int = 1, run: int = SLOPE_RUN):
    """Trust rule for `evaluate`: the first sample after the declaration (it took no part in
    it). A wake with no declaration inside the kept part of the capture is trusted at its end."""
    def trust(b: Burst, bs: list[Burst]) -> float:
        i = slope_detect(b, thr, window, step, run)
        return float(b.t[min(i + 1, len(b.t) - 1)]) if i >= 0 else float(b.t[-1])
    return trust


def slope_calibrate(record: list[Burst], unit: str) -> dict:
    """The slope rule as it would be set on the calibration day.

    Over SLOPE_WINDOWS x SLOPE_FACTORS: the setting with the shortest mean wait whose mean
    fitted bias at first trust, over the calibration-day wakes, is within the budget (the
    target the guard is built to). `grid` keeps every setting's calibration-day figures.
    """
    cal = calibration(record, unit)
    wakes = [b for b in record if b.run == CAL_RUNS[unit] and b.normal]
    thr0 = BUDGET_MG / cal["tau"]
    grid = []
    for w in SLOPE_WINDOWS:
        for k in SLOPE_FACTORS:
            tt = np.array([slope_trust(k * thr0, w)(b, wakes) for b in wakes])
            grid.append(dict(window=w, factor=k, thr=k * thr0, wait=float(tt.mean()),
                             bias=float(np.mean([b.fitted(t) for b, t in zip(wakes, tt)]))))
    best = min((g for g in grid if g["bias"] <= BUDGET_MG), key=lambda g: g["wait"])
    return dict(best, thr0=thr0, grid=grid)


def slope_rows(groups: dict[str, list[Burst]],
               settings: dict[str, tuple[float, int]]) -> pd.DataFrame:
    """Bias at first trust and the spread of the wait, for settings name -> (thr, window)."""
    rows = []
    for name, (thr, w) in settings.items():
        trust = slope_trust(thr, w)
        wakes = [b for bs in groups.values() for b in bs]
        tt = 1e3 * np.array([trust(b, []) for b in wakes])
        rows.append(dict(method=name, window=w, thr_mg_per_ms=1e-3 * thr,
                         wait_mean=float(tt.mean()), wait_p95=float(np.percentile(tt, 95)),
                         wait_max=float(tt.max()),
                         undeclared=int(sum(slope_detect(b, thr, w) < 0 for b in wakes)),
                         **evaluate(groups, trust, None)))
    return pd.DataFrame(rows)


def slope_table(record: list[Burst], unit: str) -> pd.DataFrame:
    """B6 on the whole record: the calibration-day setting, the setting whose threshold is the
    budget over tau, and the neighbouring settings in window and in threshold."""
    c = slope_calibrate(record, unit)
    w0, k0, thr0 = c["window"], c["factor"], c["thr0"]
    untuned = min((g for g in c["grid"] if g["factor"] == 1.0), key=lambda g: g["wait"])
    settings = {"B6 slope rule, calibration-day setting": (k0 * thr0, w0),
                f"B6 slope rule, threshold = budget / tau, {untuned['window']} samples":
                    (thr0, untuned["window"])}
    for w in SLOPE_WINDOWS:
        if w != w0:
            settings[f"  window {w}, threshold x{k0:g}"] = (k0 * thr0, w)
    for k in SLOPE_FACTORS:
        if k != k0:
            settings[f"  window {w0}, threshold x{k:g}"] = (k * thr0, w0)
    return slope_rows(by_run(record, unit), settings)


def compensation_wait(record: list[Burst], unit: str, model_unit: str | None = None) -> dict:
    """How long after the wake each approach must wait to cover the record's runs.

    Guard alone: the run needs `need()`. Compensation with the calibration-day model: the
    fitted residual |b0 exp(-(t-fv)/tau) - model(t)| must be within the budget from then on.
    model_unit takes the model from another unit (the transfer case).
    """
    cal = calibration(record, model_unit or unit)
    tt = np.arange(0.0, 0.040, 2e-5)
    guard, comp = [], []
    for bs in by_run(record, unit).values():
        b0 = float(np.median([b.b0 for b in bs]))
        tau = float(np.median([b.tau for b in bs]))
        fv = float(np.median([b.fv for b in bs]))
        guard.append(fv + tau * np.log(max(b0, 1e-9) / BUDGET_MG))
        t = fv + tt
        res = np.abs(b0 * np.exp(-(t - fv) / tau) - cal["b0"] * np.exp(-(t - cal["fv"]) / cal["tau"]))
        above = np.flatnonzero(res > BUDGET_MG)
        comp.append(fv if not len(above) else float(t[min(above[-1] + 1, len(t) - 1)]))
    g, c = 1e3 * np.array(guard), 1e3 * np.array(comp)
    return dict(n_runs=len(g), guard_all=float(g.max()), guard_p95=float(np.percentile(g, 95)),
                comp_all=float(c.max()), comp_p95=float(np.percentile(c, 95)),
                comp_med=float(np.median(c)), guard_med=float(np.median(g)))


def slope_under_input(record: list[Burst], unit: str, seed: int = 0) -> pd.DataFrame:
    """B6 at its calibration-day setting when the sensor is not at rest at the wake.

    Simulated, not measured: a sinusoidal acceleration of random phase is added to each recorded
    wake (every capture of the record was taken at rest), assuming the transient adds to the
    input unchanged. The input is signal, so the error of the first trusted reading is still
    the transient left at that moment. A guard fixed before the wake does not see the input:
    its row is the amplitude-zero row of `bias_table`. `over_rising` is the share of the wakes
    trusted above tolerance in which the input was rising, against the decay, when the rule
    declared (slope of the input at the middle of the last window).
    """
    c = slope_calibrate(record, unit)
    wakes = [b for bs in by_run(record, unit).values() for b in bs]
    rng = np.random.default_rng(seed)
    rows = []
    for f in (None, *INPUT_FREQS_HZ):
        for a in ((0.0,) if f is None else INPUT_AMPS_MG):
            tt, left, rising, undeclared = [], [], [], 0
            w = 2 * np.pi * (f or 0.0)
            for b in wakes:
                ph = rng.uniform(0, 2 * np.pi)
                moved = replace(b, dev=b.dev + a * np.sin(w * b.t + ph))
                i = slope_detect(moved, c["thr"], c["window"])
                undeclared += i < 0
                t = slope_trust(c["thr"], c["window"])(moved, [])
                tt.append(1e3 * t)
                left.append(b.fitted(t))
                rising.append(i >= 0 and np.cos(w * b.t[i - c["window"] // 2] + ph) > 0)
            tt, left, rising = np.array(tt), np.array(left), np.array(rising)
            over = left > TOL_MG
            rows.append(dict(freq_hz=f or 0.0, amp_mg=a, peak_slope_mg_per_ms=1e-3 * w * a,
                             wait_mean=float(tt.mean()), wait_p95=float(np.percentile(tt, 95)),
                             undeclared=int(undeclared), left_med=float(np.median(left)),
                             left_p95=float(np.percentile(left, 95)), over=float(over.mean()),
                             over_rising=(float(rising[over].mean()) if a and over.any()
                                          else np.nan)))
    return pd.DataFrame(rows)


def energy_row(unit: str, mode: str, guard_s: float) -> dict:
    """Saving and bandwidth for a guard, from the firmware's energy model and the unit's
    measured coefficients. One trusted probe window per cycle, cycle = 1 / (2 B)."""
    c = COEFFS[unit][mode]
    s = core.sensor(**c)
    row = dict(guard_ms=1e3 * guard_s, e_guard_mj=1e3 * c["P_on"] * guard_s,
               b_max_hz=1.0 / (2.0 * (core.t_sleep_min(s) + guard_s + c["T_probe"])))
    for b in BANDWIDTHS_HZ:
        ts = 1.0 / (2.0 * b) - guard_s - c["T_probe"]
        row[f"s_{b:g}"] = float(core.saving(s, ts, guard_s)) if ts > 0 else float("nan")
    return row


def energy_table(record: list[Burst], unit: str, mode: str = "pd") -> pd.DataFrame:
    cal = calibration(record, unit)
    rows = [dict(method="B0 continuous-ON", guard_ms=0.0, e_guard_mj=0.0, b_max_hz=np.nan,
                 **{f"s_{b:g}": 0.0 for b in BANDWIDTHS_HZ})]
    slope = slope_table(record, unit).iloc[:2]
    for name, g in (("B1 first valid", cal["fv"]),
                    *PRACTICE_WAITS_S.items(),
                    ("B3 compensation, +2 ms", cal["fv"] + 2e-3),
                    ("B4 / B5 map guard", cal["guard"]),
                    *((f"{r.method}, mean wait", 1e-3 * r.wait_mean) for r in slope.itertuples()),
                    ("first trust 16 ms (campaigns)", CAMPAIGN_TRUST_S),
                    ("fixed 20 ms", STARTUP_GUARD_S)):
        rows.append(dict(method=name, **energy_row(unit, mode, g)))
    return pd.DataFrame(rows)


def report() -> None:
    record = load_record("pd")
    pd.set_option("display.width", 200, "display.max_columns", 30)
    for unit in ("A", "B"):
        cal = calibration(record, unit)
        n = sum(len(v) for v in by_run(record, unit).values())
        print(f"\n=== unit {unit}: {n} normal-grid pd bursts in {len(by_run(record, unit))} runs ===")
        print(f"calibration ({cal['n']} bursts): b0 {cal['b0']:.1f} mg, tau {1e3 * cal['tau']:.3f} ms, "
              f"first valid {1e3 * cal['fv']:.2f} ms, guard {1e3 * cal['guard']:.2f} ms")
        print("\nbias at first trust (mg)")
        print(bias_table(record, unit).round(3).to_string(index=False))
        print("\nwait needed to cover the record (ms from the wake command)")
        for mu in (unit, "B" if unit == "A" else "A"):
            w = compensation_wait(record, unit, mu)
            print(f"  model from unit {mu}: guard alone median {w['guard_med']:.2f}, all runs "
                  f"{w['guard_all']:.2f}; compensation median {w['comp_med']:.2f}, 95 % "
                  f"{w['comp_p95']:.2f}, all runs {w['comp_all']:.2f}")
        c = slope_calibrate(record, unit)
        print(f"\nB6 slope rule; threshold x1 = budget / tau = {1e-3 * c['thr0']:.3f} mg/ms; "
              f"calibration-day setting: window {c['window']} samples, threshold x{c['factor']:g} "
              f"(calibration day: mean wait {1e3 * c['wait']:.2f} ms, "
              f"mean fitted bias {c['bias']:.2f} mg)")
        print("calibration-day mean wait (ms) / mean fitted bias (mg), window by threshold factor:")
        for w in SLOPE_WINDOWS:
            print(f"  {w:2d} samples  " + "   ".join(
                f"x{g['factor']:g}: {1e3 * g['wait']:.2f} / {g['bias']:.2f}"
                for g in c["grid"] if g["window"] == w))
        print(slope_table(record, unit).round(3).to_string(index=False))
        print(f"\nB6 with a simulated input at the wake (threshold {1e-3 * c['thr']:.2f} mg/ms; "
              f"wait and transient left at first trust; "
              f"undeclared = none within {1e3 * HEAD_S:.0f} ms)")
        print(slope_under_input(record, unit).round(3).to_string(index=False))
        print("\nenergy, pd mode")
        print(energy_table(record, unit).round(4).to_string(index=False))


if __name__ == "__main__":
    report()
