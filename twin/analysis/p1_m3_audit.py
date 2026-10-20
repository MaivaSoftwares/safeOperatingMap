"""Paper 1, Section V-E: the audit of the real-drift runs, fixed before the runs began.

The definitions are the pre-registered ones and are not tuned here:

  need of an hour (pd)   median over its valid bursts of t_fv + tau ln(b0 / budget); a burst is
                         valid if the tool's fit is ok and its first valid sample is at 6.12 to
                         6.25 ms; standard error = sd / sqrt(n); no value below 3 valid bursts
  need of a cycle        pd: the larger need of the two hours that bracket its segment;
                         rail_off: that plus the rail_off delay (2.10 ms, or the latest daily
                         delay by the rule of run M1 if larger)
  shortfall              need - guard in force (the guard_ms the firmware logs)
  violation              shortfall > tau ln(tolerance / budget); "resolved" if it still is after
                         taking two standard errors of the need off, else "unresolved"
  outside the budget     0 < shortfall <= that threshold; counted apart, not a violation
  excluded segment       sensor declared dead in it, or the tool ended with an error

Segments still running are skipped, so the audit can be read while the run goes on. A segment
cut short by a stop of the host (recorded by the driver when it resumes) has a log without a
stop line: its logged cycles are audited like any others and it is marked `cut`.

Run: twin/.venv/bin/python twin/analysis/p1_m3_audit.py [run directory] [--delays]
     (--delays recomputes the daily rail_off delay by bootstrap; about 15 s per day)
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TWIN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TWIN))
sys.path.insert(0, str(TWIN / "analysis"))
import p1_revision_checks as chk  # noqa: E402
import paper2_runs as runs  # noqa: E402

BUDGET_MG = 0.85
TOL_MG = 1.0
FV_BAND_MS = (6.12, 6.25)
MIN_BURSTS = 3
P_ON_MW = 67.0                   # as tools/run_ctl_experiment.py, for the saving of a segment
DELAY_MS = 2.10                  # unit B's; a run's own CTL RODELAY is used when recorded
CTL_FIELDS = ("cycle", "t_ms", "mode", "tsleep_ms", "guard_ms", "active", "lam_hat", "b0hat_mg",
              "tauhat_ms", "glo_ms", "ghi_ms", "recal", "e_mJ", "pavg_mW", "temp_C")


def burst_needs(fits_csv: Path) -> tuple[list[dict], int]:
    """Per valid burst: need (ms from the wake command), tau, b0, first valid. And the number dropped."""
    out, dropped = [], 0
    for r in csv.DictReader(fits_csv.open()):
        try:
            b0, tau, fv = 1e3 * float(r["b0_g"]), 1e3 * float(r["tau_bias_s"]), 1e-3 * float(r["first_valid_us"])
        except ValueError:
            dropped += 1
            continue
        if r["ok"] != "True" or not (FV_BAND_MS[0] <= fv <= FV_BAND_MS[1]) or b0 <= 0 or tau <= 0:
            dropped += 1
            continue
        out.append(dict(need=fv + tau * math.log(b0 / BUDGET_MG), tau=tau, b0=b0, fv=fv))
    return out, dropped


def hour_need(bursts: list[dict], min_bursts: int = MIN_BURSTS) -> dict | None:
    if len(bursts) < min_bursts:
        return None
    need = np.array([b["need"] for b in bursts])
    return dict(n=len(bursts), need=float(np.median(need)),
                se=float(need.std(ddof=1) / math.sqrt(len(need))),
                tau=float(np.median([b["tau"] for b in bursts])),
                b0=float(np.median([b["b0"] for b in bursts])))


def cycle_need(mode: str, before: dict | None, after: dict | None, delay_ms: float) -> dict | None:
    """The need a cycle is audited against, from the hours bracketing its segment."""
    have = [h for h in (before, after) if h is not None]
    if not have:
        return None
    h = max(have, key=lambda x: x["need"])
    return dict(need=h["need"] + (delay_ms if mode == "rail_off" else 0.0), se=h["se"], tau=h["tau"])


def classify(guard_ms: float, need: dict) -> str:
    short = need["need"] - guard_ms
    limit = need["tau"] * math.log(TOL_MG / BUDGET_MG)
    if short <= 0:
        return "ok"
    if short <= limit:
        return "outside_budget"
    return "resolved" if short - 2 * need["se"] > limit else "unresolved"


def read_segment(log: Path) -> dict:
    """Logged cycles of one controller segment and its stop line; `done` is False while it runs."""
    rows, stop, dead = [], None, False
    for ln in log.read_text(errors="ignore").splitlines():
        if ln.startswith("CTL,"):
            p = ln.split(",")
            if len(p) == len(CTL_FIELDS) + 1:
                try:
                    rows.append((p[3], float(p[5]), float(p[2]), float(p[15]), int(float(p[6])),
                                 float(p[14])))
                except ValueError:
                    pass
        elif ln.startswith("WARN: CTL SENSOR DEAD"):
            dead = True
        elif ln.startswith("ACK: CTL STOP"):
            m = re.search(r"cycles=(\d+) e_mJ=([\d.]+) ina_fail=(\d+) dead_episodes=(\d+)", ln)
            stop = dict(cycles=int(m.group(1)), e_mJ=float(m.group(2)), ina_fail=int(m.group(3)),
                        dead_episodes=int(m.group(4)))
    cyc = pd.DataFrame(rows, columns=["mode", "guard_ms", "t_ms", "temp_C", "active", "pavg_mW"])
    summ = log.with_suffix(".summary.json")
    s = json.loads(summ.read_text()) if summ.exists() else {}
    saving, wakes = s.get("saving"), s.get("active_cycles")
    if not s and len(cyc):                     # no summary (segment cut): the same figures from the log
        saving, wakes = 1.0 - float(cyc.pavg_mW.iloc[-1]) / P_ON_MW, int(cyc.active.sum())
    return dict(cycles=cyc, stop=stop, dead=dead or bool(stop and stop["dead_episodes"]),
                done=stop is not None, saving=saving, aborted=s.get("aborted"), false_wakes=wakes)


def load(run_dir: Path) -> dict:
    """What the driver did, by period: burst folders, controller logs, return codes."""
    state = json.loads((run_dir / "state.json").read_text())
    per: dict[int, dict] = {}
    for r in csv.DictReader((run_dir / "events.csv").open()):
        k = per.setdefault(int(r["period"]), dict(pd=[], rail_off=[], ctl=None, ctl_rc=None, bad=[]))
        step, name = r["step"], Path(r["detail"]).name
        if step.startswith("bursts"):
            mode = "rail_off" if "rail_off" in step else "pd"
            if r["rc"] == "0" and name:
                k[mode].append(name)
            else:
                k["bad"].append(step)
        elif step.startswith("controller") and "skipped" not in step:
            k["ctl"], k["ctl_rc"] = name, int(r["rc"])
    tag = f"m3_{state['args']['label']}_unit{state['args']['unit']}"
    cut = {i["cut_period"] for i in state.get("interruptions", []) if "cut_period" in i}
    for k in cut:                              # the tool never returned, so no event names its log
        logs = sorted(runs.DATA.glob(f"ctl_{tag}_p{k:03d}_*.log"))
        if logs and k in per and not per[k]["ctl"]:
            per[k]["ctl"], per[k]["ctl_rc"] = logs[-1].name, 0
    return dict(state=state, periods=per, cut=cut)


def daily_delay(p: dict, min_bursts: int = MIN_BURSTS) -> float | None:
    """rail_off delay of one daily block by the rule of run M1 (95 % upper bound of the need
    minus the lower pd guard of the blocks before and after, rounded up to 0.05 ms). A pd block
    without enough valid bursts (the sensor was not answering) is left out; None if no pd block
    or no rail_off block is usable."""
    pds = tuple(n for n in p["pd"] if len(burst_needs(runs.DATA / n / "fits.csv")[0]) >= min_bursts)
    if not (p["rail_off"] and pds):
        return None
    return chk.rail_off_delay(p["rail_off"][0], pds[:2])["delay_ms"]


def daily_delays(run: dict) -> dict[int, float]:
    out = {}
    for k, p in sorted(run["periods"].items()):
        d = daily_delay(p)
        if d is not None:
            out[k] = d
    return out


def audit(run_dir: Path, min_bursts: int = MIN_BURSTS, delays: dict[int, float] | None = None) -> dict:
    run = load(run_dir)
    base_delay = run["state"]["args"].get("rodelay_us", DELAY_MS * 1e3) / 1e3
    hours, dropped = {}, 0
    for k, p in sorted(run["periods"].items()):
        bursts = []
        for name in p["pd"]:
            b, d = burst_needs(runs.DATA / name / "fits.csv")
            bursts += b
            dropped += d
        hours[k] = hour_need(bursts, min_bursts)

    counts = {m: dict(ok=0, outside_budget=0, unresolved=0, resolved=0) for m in ("pd", "rail_off")}
    segs, excluded, all_cycles, worst = [], [], 0, None
    for k, p in sorted(run["periods"].items()):
        if not p["ctl"]:
            continue
        seg = read_segment(runs.DATA / p["ctl"])
        is_cut = not seg["done"] and k in run["cut"]
        if not seg["done"] and not is_cut:
            continue                                        # still running
        if is_cut:                                          # cut by an interruption: audit what was logged
            seg["stop"] = dict(cycles=len(seg["cycles"]), e_mJ=float("nan"), dead_episodes=0)
        if seg["dead"] or seg["aborted"] or p["ctl_rc"] != 0:
            excluded.append((k, "sensor dead" if seg["dead"] else f"tool ended with an error ({p['ctl_rc']})"))
            continue
        delay = max([base_delay] + [v for h, v in (delays or {}).items() if h <= k][-1:])
        row = dict(period=k, cut=is_cut, cycles=seg["stop"]["cycles"], logged=len(seg["cycles"]),
                   ina_fail=seg["stop"].get("ina_fail", 0),
                   saving=seg["saving"], false_wakes=seg["false_wakes"],
                   temp_C=float(seg["cycles"].temp_C.median()) if len(seg["cycles"]) else float("nan"))
        all_cycles += seg["stop"]["cycles"]
        for mode in ("pd", "rail_off"):
            g = seg["cycles"].guard_ms[seg["cycles"]["mode"] == mode]
            need = cycle_need(mode, hours.get(k), hours.get(k + 1), delay)
            row[f"guard_{mode}"] = float(g.median()) if len(g) else float("nan")
            row[f"guard_{mode}_min"] = float(g.min()) if len(g) else float("nan")
            row[f"need_{mode}"] = need["need"] if need else float("nan")
            if need is None or not len(g):
                row[f"audited_{mode}"] = 0
                continue
            row[f"audited_{mode}"] = len(g)
            for verdict, n in g.map(lambda x: classify(x, need)).value_counts().items():
                counts[mode][verdict] += int(n)
            margin = float(g.min() - need["need"])
            if worst is None or margin < worst[2]:
                worst = (k, mode, margin)
        segs.append(row)
    return dict(run=run, hours=hours, dropped_bursts=dropped, counts=counts, excluded=excluded,
                segments=pd.DataFrame(segs), all_cycles=all_cycles, worst=worst)


def model_free_check(run: dict, segments: pd.DataFrame, periods_per_day: int = 24) -> pd.DataFrame:
    """Per day, with no model: the bursts of the day pooled; the mean deviation from the settled
    level of the first sample at or after the day's median pd guard in force."""
    rows = []
    if not len(segments):
        return pd.DataFrame(rows)
    for day in sorted(set(segments.period // periods_per_day)):
        seg = segments[segments.period // periods_per_day == day]
        guard = float(seg.guard_pd.median())
        dev = []
        for k, p in run["periods"].items():
            if k // periods_per_day != day:
                continue
            for name in p["pd"]:
                for t, a, fv in chk.burst_set(name, "pd"):
                    settled = float(a[len(a) // 2:].mean())
                    i = int(np.searchsorted(t + fv, 1e-3 * guard - 1e-9))
                    if i < len(a):
                        dev.append(1e3 * (a[i] - settled) * (1.0 if a[0] >= settled else -1.0))
        if dev:
            rows.append(dict(day=int(day), guard_ms=guard, bursts=len(dev), mean_mg=float(np.mean(dev)),
                             se_mg=float(np.std(dev, ddof=1) / math.sqrt(len(dev))) if len(dev) > 1 else float("nan")))
    return pd.DataFrame(rows)


def report(run_dir: Path, with_delays: bool = False) -> None:
    run = load(run_dir)
    delays = daily_delays(run) if with_delays else None
    a = audit(run_dir, delays=delays)
    st = a["run"]["state"]
    print(f"run {run_dir.name}: started {st['t0']}; periods done {st['next_period']}; "
          f"rest {st.get('rest_started') or '-'} to {st.get('rest_ended') or '-'}; "
          f"interruptions {len(st['interruptions'])}; finished {st.get('finished', 'no')}")
    print(f"\nPRIMARY. Logged cycles audited, by verdict (all cycles in the audited segments: {a['all_cycles']}):")
    for mode, c in a["counts"].items():
        n = sum(c.values())
        print(f"  {mode:8s} audited {n:7d} | violations: resolved {c['resolved']}, unresolved {c['unresolved']} | "
              f"outside the budget, inside the tolerance {c['outside_budget']} | within the budget {c['ok']}")
    if a["worst"]:
        print(f"  smallest margin (lowest guard in a segment minus its need): {a['worst'][2]:+.2f} ms, "
              f"period {a['worst'][0]}, {a['worst'][1]}")
    print(f"  excluded segments: {a['excluded'] or 'none'}; bursts dropped as invalid: {a['dropped_bursts']}")
    if delays:
        print(f"  daily rail_off delay by the rule (ms): {delays}")
    h = pd.DataFrame([dict(period=k, **v) for k, v in a["hours"].items() if v])
    if len(h):
        print(f"\nHourly need, pd ({len(h)} hours with a value; {sum(v is None for v in a['hours'].values())} without): "
              f"min {h.need.min():.2f}, median {h.need.median():.2f}, max {h.need.max():.2f} ms; "
              f"median standard error {h.se.median():.2f} ms; b0 {h.b0.min():.1f} to {h.b0.max():.1f} mg")
    s = a["segments"]
    if len(s):
        pd.set_option("display.width", 220, "display.max_columns", 30)
        cols = ["period", "cycles", "logged", "need_pd", "guard_pd_min", "guard_pd", "need_rail_off",
                "guard_rail_off_min", "guard_rail_off", "saving", "false_wakes", "temp_C"]
        print("\nSegments (need and guard in ms):")
        print(s[cols].round(2).to_string(index=False, max_rows=40))
        print(f"\nSaving over the audited segments: median {100 * s.saving.median():.1f} %, "
              f"range {100 * s.saving.min():.1f} to {100 * s.saving.max():.1f} %; "
              f"false wakes {int(s.false_wakes.sum())} in {int(s.cycles.sum())} cycles")
        mf = model_free_check(a["run"], s)
        if len(mf):
            print("\nWith no model: first sample at or after the day's median pd guard, against the settled level")
            print(mf.round(3).to_string(index=False))


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    d = Path(args[0]) if args else sorted(runs.DATA.glob("m3_real_unit*"))[-1]
    report(d.resolve(), with_delays="--delays" in sys.argv)
