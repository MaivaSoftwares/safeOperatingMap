#!/usr/bin/env python3
"""Paper 1: the manuscript figures, drawn at their printed size.

Every figure here is drawn at exactly one IEEE column (3.5 in = 252 pt) so that
\\includegraphics[width=\\columnwidth] does not rescale it, with no text below 8 pt,
as a vector PDF with TrueType (Type 42) fonts and no in-plot title.

No data and no plotted number is changed. Each figure takes its data from the same
committed files, through the loading functions and constants of its existing
generator; only the drawing is rewritten here. Nothing outside paper/figures/rev/ is
written: the existing generators are never run (several of them save into
docs/figures or paper/figures, and b2_guard.main() also rewrites a file in tools/data).

    twin/.venv/bin/python paper/figures/make_revision_figures.py            # all
    twin/.venv/bin/python paper/figures/make_revision_figures.py inrush     # one

Figure                 data and computation taken from
inrush                 twin/analysis/figures.py            (_load_inrush, _INRUSH)
recovery_raw           twin/analysis/r2fig_transients.py   (constants; its burst loop)
recovery_terms         twin/analysis/p1_recovery_models.py (load_bursts, ensemble, fit M2)
hyst_campaign          twin/analysis/hyst_campaign_figure.py (_dedrift_legs, _binned)
safemap_banded         tools/data/r3_map_uncertainty_unitA_c1/{pgreen,draws}.csv with
                       twin/analysis/r3_uncertainty.py (band, robust_green, _runs).
                       This is correction C-1: rail_off at its own guard, so the lower
                       panel has one guard row per sleep method.
b2_guard_violation     twin/analysis/b2_guard.py           (curves, window_mean_pred); the
                       guard band of Table IV from r3_map_uncertainty_unitA_rev/summary.json
thermal_paper          paper/figures/make_thermal_replication_fig.py and the
                       load_ramp() of paper/figures/make_thermal_paper_fig.py; panel (c)
                       twin/analysis/p1_transfer_revision.py (UNIT_A) through libcore.so
transfer_paper         twin/analysis/p1_transfer_revision.py (Section VII, revised sets)
m3_drift               twin/analysis/p1_m3_audit.py        (audit: hours, segments, excluded)
testbed_photo_annotated  NOT built: see fig_testbed_photo_annotated().
"""
from __future__ import annotations

import ast
import json
import math
import sys
import warnings
from pathlib import Path

sys.dont_write_bytecode = True     # importing the existing generators must not leave .pyc files behind

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib import font_manager                   # noqa: E402
from matplotlib.colors import ListedColormap          # noqa: E402
from matplotlib.patches import Rectangle              # noqa: E402
from matplotlib.text import Text                      # noqa: E402
from matplotlib.ticker import FuncFormatter, NullLocator  # noqa: E402

HERE = Path(__file__).resolve().parent                # paper/figures
REPO = HERE.parents[1]
REV = HERE / "rev"                                    # the only directory written to
sys.path.insert(0, str(REPO / "twin"))
sys.path.insert(0, str(HERE))

W_IN = 3.5          # one IEEE column, 252 pt
FS = 8.0            # every text object is set at this size
MIN_PT = 8.0        # the build fails if any text object is smaller

# Text font. It must be a TrueType-outline font: the system's Nimbus Roman and TeX Gyre
# Termes are OpenType/CFF files, which matplotlib embeds as "CID Type 0C", not TrueType.
# STIXGeneral ships with matplotlib (same file on every machine) and is the text face of
# the "stix" math fonts, so text and symbols match.
SERIF = ["STIXGeneral", "Times New Roman", "Liberation Serif", "DejaVu Serif"]

INK, MUTED = "0.15", "0.30"
LS = 0.95             # line spacing of two-line labels


# ================================================================================================
def _style() -> None:
    plt.rcdefaults()
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif", "font.serif": SERIF, "mathtext.fontset": "stix",
        "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS, "legend.fontsize": FS,
        "xtick.labelsize": FS, "ytick.labelsize": FS,
        "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25, "lines.linewidth": 1.4,
        "axes.labelpad": 2.0, "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
        "legend.borderpad": 0.3, "legend.labelspacing": 0.25, "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.4, "legend.framealpha": 0.9,
    })


def _new(height_in: float, nrows: int = 1, **kw):
    """A figure exactly one column wide. Constrained layout keeps every label inside the
    page, so no bbox="tight" (which would change the page width) is needed."""
    fig, ax = plt.subplots(nrows, 1, figsize=(W_IN, height_in), layout="constrained", **kw)
    fig.get_layout_engine().set(w_pad=1.5 / 72, h_pad=1.5 / 72)
    return fig, ax


_PLAIN = FuncFormatter(lambda v, _pos: f"{v:g}")      # 0.1, 1, 10 instead of 10^-1, 10^0, 10^1


def _plain_log(axis) -> None:
    """Decade tick labels as plain numbers: an exponent would be set below 8 pt."""
    axis.set_major_formatter(_PLAIN)


def _undrawn_tick_labels(fig) -> set[int]:
    """Labels of ticks that lie outside their axis limits: they exist as objects but are not drawn."""
    hidden = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            tol = 1e-9 * max(abs(lo), abs(hi))
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo - tol <= tick.get_loc() <= hi + tol:
                    hidden.update((id(tick.label1), id(tick.label2)))
    return hidden


def _audit(fig) -> dict:
    """Smallest text size, and text that leaves the page or collides with other text."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    hidden = _undrawn_tick_labels(fig)
    texts = [t for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip() and id(t) not in hidden]
    page = fig.bbox
    boxes = [(t, t.get_window_extent(rend)) for t in texts]
    boxes = [(t, b) for t, b in boxes if b.width > 0 and b.height > 0]
    outside = [t.get_text() for t, b in boxes
               if b.x0 < page.x0 - 0.5 or b.y0 < page.y0 - 0.5 or b.x1 > page.x1 + 0.5 or b.y1 > page.y1 + 0.5]
    clash = []
    for i, (ti, bi) in enumerate(boxes):
        for tj, bj in boxes[i + 1:]:
            if bi.x0 < bj.x1 and bj.x0 < bi.x1 and bi.y0 < bj.y1 and bj.y0 < bi.y1:
                clash.append((ti.get_text(), tj.get_text()))
    return dict(n_text=len(texts), min_pt=min(t.get_fontsize() for t in texts),
                outside=outside, clash=clash)


def _finish(fig, name: str, src: str) -> dict:
    a = _audit(fig)
    if a["min_pt"] < MIN_PT - 1e-9:
        raise SystemExit(f"{name}: a text object is {a['min_pt']} pt (< {MIN_PT} pt)")
    out = REV / f"{name}.pdf"
    if out.parent.resolve() != (HERE / "rev").resolve():      # never write anywhere else
        raise SystemExit(f"refusing to write {out}")
    REV.mkdir(parents=True, exist_ok=True)
    w, h = fig.get_size_inches()
    fig.savefig(out)                                          # no bbox="tight": page = figure size
    plt.close(fig)
    print(f"  {name:20s} {w * 72:.1f} x {h * 72:.1f} pt   min text {a['min_pt']:.1f} pt "
          f"({a['n_text']} text objects)   <- {src}")
    for t in a["outside"]:
        print(f"      WARNING text outside the page: {t!r}")
    for p in a["clash"]:
        print(f"      WARNING text boxes overlap: {p[0]!r} / {p[1]!r}")
    return dict(name=name, path=str(out), width_pt=w * 72, height_pt=h * 72, **a)


def _defs_from_script(path: Path, names: set[str], env: dict) -> dict:
    """Run only the named top-level assignments and function definitions of a generator that
    cannot be imported because its module body draws and saves a figure. The code is the
    generator's own, so the computation cannot drift from it."""
    keep, found = [], set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            keep.append(node); found.add(node.name)
        elif isinstance(node, ast.Assign):
            tgt = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
            if tgt and tgt <= names:
                keep.append(node); found |= tgt
    if found != names:
        raise SystemExit(f"{path.name}: not found: {sorted(names - found)}")
    ns = dict(env)
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), ns)   # noqa: S102
    return ns


# ================================================================================================
def fig_inrush() -> dict:
    """Both measured wake-current surges (scope across the 1.00 ohm shunt; C-7)."""
    from analysis import figures as F
    fig, ax = _new(2.10)
    for m, (e, pk, _settled) in F._INRUSH.items():
        t, i_mA = F._load_inrush(m)
        ax.plot(t, i_mA, color=F.COLOR[m], lw=0.7, label=f"{m}: {e}, {pk} mA")
    ax.set(xlabel="time from wake (ms)", ylabel="rail current (mA)", xlim=(-0.3, 2.5))
    ax.legend(loc="upper right")
    return _finish(fig, "inrush", "tools/data/{rail_off,pd}_scope capture 1")


def fig_recovery_raw() -> dict:
    """The 30 measured wake transients, raw, with the median fit, the budget and the guard."""
    from analysis import r2fig_transients as R
    fig, ax = _new(2.10)
    files = sorted(R.RAW.glob("burst_pd_*.txt"))
    for f in files:                                     # the burst loop of r2fig_transients.main()
        d = pd.read_csv(f, skiprows=2, on_bad_lines="skip")
        d = d.apply(pd.to_numeric, errors="coerce").dropna()
        d = d[(d.drdy == 1) & (d.err == 0) & (d.accel_g.abs() < 1.0)]
        ped = d.accel_g.tail(len(d) // 4).median()
        t = d.t_us * 1e-3
        b = (d.accel_g - ped).abs() * 1e3
        m = t <= 24
        ax.plot(t[m], b[m], color=R.PD, lw=0.4, alpha=0.18, zorder=2)
    t = np.linspace(R.T_FV, 24, 300)
    ax.plot(t, R.B0 * np.exp(-(t - R.T_FV) / R.TAU), color="0.15", lw=1.2, zorder=4,
            label="median fit")
    ax.axvspan(0, R.T_FV, color="0.6", alpha=0.18, lw=0)
    ax.annotate("blind\nwindow", (1.4, 8), color=MUTED, linespacing=LS)
    ax.axhline(1.0, ls="--", color=R.RED, lw=0.9)
    # moved from x = 17.5 ms, where it lay across the traces, into the blind window (no data there)
    ax.annotate(r"$\varepsilon$ = 1 mg", (0.5, 1.25), color=R.RED)
    ax.axvspan(R.GLO, R.GHI, color="0.35", alpha=0.30, lw=0)
    ax.annotate(r"$T_{\mathrm{guard}}$", (R.GUARD, 32), color=MUTED, ha="center")
    ax.set_xlim(0, 24)
    ax.set_yscale("log")
    ax.set_ylim(0.15, 120)
    _plain_log(ax.yaxis)
    ax.set_xlabel("time since wake command (ms)")
    ax.set_ylabel("post-wake |bias| (mg)")
    ax.legend(loc="upper right")
    return _finish(fig, "recovery_raw", f"{len(files)} raw bursts, {R.RAW.parent.name}")


def fig_recovery_terms() -> dict:
    """Unit A's 30 pd bursts averaged: the fast and slow terms of the two-term fit, the guard of
    Table II and the margined tolerance (Section III-B; replaces recovery_raw in Fig. 3)."""
    from analysis import p1_recovery_models as RM
    run = RM.RUNS["A"]
    bursts = RM.load_bursts(run)
    t, a, fv, _jit = RM.ensemble(bursts)
    p = RM.fit("M2", t, a)["p"]
    c, sgn = p[0], np.sign(p[1])                        # the transient is negative: shown positive
    noise = float(a[len(a) // 2:].std())                # noise of the mean, per sample
    guard_ms = 6.17 + 1.449 * math.log(82.6 / (0.85 - 0.21))   # (11), Table II
    edges = np.geomspace(t[0] + 1e-5, t[-1], 40)
    tb, yb, eb = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (t >= lo) & (t < hi)
        if m.any():
            tb.append(math.exp(float(np.log(t[m]).mean())))
            yb.append(float((sgn * (a[m] - c)).mean()))
            eb.append(noise / math.sqrt(int(m.sum())))
    tb, yb, eb = np.array(tb), np.array(yb), np.array(eb)
    ok = yb > 0                                         # a log axis cannot show a bin at or below zero
    tt = np.geomspace(1e-5, t[-1], 800)
    fast = sgn * p[1] * np.exp(-tt / p[2])
    slow = sgn * p[3] * np.exp(-tt / p[4])
    fig, ax = _new(2.10)
    ax.errorbar(1e3 * (tb[ok] + fv), 1e3 * yb[ok], yerr=1e3 * np.minimum(eb[ok], 0.9 * yb[ok]),
                fmt="o", ms=2.6, color=INK, lw=0.8, capsize=0, zorder=4, label="mean of 30 bursts")
    ax.plot(1e3 * (tt + fv), 1e3 * fast, ls=(0, (4, 2)), color="#1f77b4", lw=1.2,
            label=rf"fast, $\tau$ = {1e3 * p[2]:.2f} ms")
    ax.plot(1e3 * (tt + fv), 1e3 * slow, ls=(0, (1, 1.5)), color="#d95f02", lw=1.6,
            label=rf"slow, $\tau$ = {p[4]:.2f} s")
    ax.plot(1e3 * (tt + fv), 1e3 * (fast + slow), color="0.45", lw=0.9, label="sum")
    ax.axhline(0.85, color=MUTED, lw=0.8)
    ax.annotate(r"$(1-\delta)\varepsilon$", (840, 1.0), ha="right", va="bottom", color=MUTED)
    ax.axvline(guard_ms, color=MUTED, lw=0.8, ls=(0, (5, 1.5, 1, 1.5)))
    ax.annotate(f"guard\n{guard_ms:.1f} ms", (guard_ms * 1.05, 60), va="top", color=MUTED,
                linespacing=LS)
    ax.set_xscale("log"); ax.set_yscale("log")
    _plain_log(ax.xaxis); _plain_log(ax.yaxis)
    ax.set_xlim(6, 850); ax.set_ylim(0.05, 150)
    ax.set_xlabel("time since wake command (ms)")
    ax.set_ylabel("bias from settled level (mg)")
    ax.legend(loc="upper right")
    return _finish(fig, "recovery_terms",
                   f"{run}: {len(bursts)} bursts; slow {1e3 * abs(p[3]):.3f} mg, {1e3 * p[4]:.0f} ms; "
                   f"{int((~ok).sum())} bins at or below zero not drawn")


def fig_hyst_campaign() -> dict:
    """Heating and cooling legs of the fastest controlled ramp, de-drifted."""
    from analysis import hyst_campaign_figure as H
    name, h_mg = "temp_log_unitA_hyst_fastrep1.csv", 1.06       # the arguments of H.main()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T, add, ipk, _b0 = H._dedrift_legs(name)
    add = add - add.mean()                                      # centre for display
    Tm = T.mean()
    edges = np.linspace(T.min(), T.max(), 12)
    fig, ax = _new(2.45)
    for sl, colr, mk, lab in ((slice(0, ipk), H.HEAT, "o", "heating"),
                              (slice(ipk, None), H.COOL, "s", "cooling")):
        c, m, s = H._binned(T[sl], add[sl], edges)
        ax.fill_between(c, m - s, m + s, color=colr, alpha=0.15, lw=0)
        ax.plot(c, m, "-" + mk, color=colr, ms=3, lw=1.3, mec="none", label=lab)
    Tl = np.array([T.min(), T.max()])
    ax.plot(Tl, H.KAPPA * (Tl - Tm) * 1e-3, color="0.4", ls=(0, (4, 2)), lw=1.0,
            label=rf"$\kappa_b$ = $-${abs(H.KAPPA):g} µg/°C")
    ax.annotate(f"loop {h_mg:.2f} mg", (0.04, 0.06), xycoords="axes fraction", color=MUTED)
    ax.set(xlabel="reported die temperature (°C)", ylabel="bias, de-drifted (mg)", ylim=(-2.0, 0.75))
    ax.legend(loc="upper right")
    return _finish(fig, "hyst_campaign", f"twin/data/{name}")


def fig_safemap_banded() -> dict:
    """The safe map with its measurement bands, from the revised bootstrap (C-1 rail_off guard,
    D6 slow term, C-7 surge energies; twin/analysis/p1_map_revision.py): the class map of each
    sleep method (top) and the boundary bands (bottom), one guard per method. Logic of r3_uncertainty.fig_safemap_banded; the row names moved from the left
    margin to beside each marker so that the axes keep the column width at 8 pt."""
    from analysis import r3_uncertainty as R
    src = REPO / "tools" / "data" / "r3_map_uncertainty_unitA_rev"
    pg, draws = pd.read_csv(src / "pgreen.csv"), pd.read_csv(src / "draws.csv")
    for m in R.MODES:                                   # the file's grid is the generator's grid
        if not np.allclose(pg[pg.method == m].T_sleep_s.to_numpy(), R.TS_GRID, rtol=1e-12, atol=0):
            raise SystemExit("pgreen.csv is not on r3_uncertainty.TS_GRID")
    C = R.COLOR
    fig, (ax, axb) = _new(2.60, nrows=2, sharex=True, gridspec_kw={"height_ratios": [0.78, 1.0]})

    # -- top: modal class per point, hatched where the class is not settled at 1-alpha ---------
    cmap = ListedColormap([C["RED"], C["YELLOW"], C["GREEN"]])
    grid = np.array([pg[pg.method == m][["p_red", "p_yellow", "p_green"]].to_numpy().argmax(axis=1)
                     for m in R.MODES])
    ax.pcolormesh(R.TS_GRID * 1e3, [0.5, 1.5], grid, cmap=cmap, vmin=0, vmax=2, shading="nearest",
                  edgecolors="face", linewidth=0.3, antialiased=False)
    ax.axhline(1.0, color="white", lw=3)
    ax.set_xscale("log")
    ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(list(R.MODES))
    ax.set_ylim(0.0, 2.0)
    for row, m in enumerate(R.MODES):
        d = pg[pg.method == m]
        conf = d[["p_red", "p_yellow", "p_green"]].to_numpy().max(axis=1)
        for st, en in R._runs(conf < (1.0 - R.ALPHA)):
            x0, x1 = R.TS_GRID[st] * 1e3, R.TS_GRID[en] * 1e3
            if x1 <= x0:
                # A run of one grid point (every run in this data set: one point at each lower
                # class edge) spans no width in the generator's rule, so it hatches nothing; the
                # zero-width patch only leaves stray hairline dashes in some PDF viewers.
                continue
            ax.add_patch(Rectangle((x0, row), x1 - x0, 1.0, facecolor="none", edgecolor="k",
                                   hatch="/////", linewidth=0.0))
        rg = R.robust_green(pg, m)
        if rg:
            ax.plot([rg[0] * 1e3, rg[1] * 1e3], [row + 0.5] * 2, color="k", lw=0.8, marker="|",
                    ms=5, solid_capstyle="butt")

    # -- bottom: boundary distributions as median and 2.5 to 97.5 percentile band -----------------
    items = [                                            # (name, draws, colour, side of the label)
        ("guard pd", draws.T_guard_s, C["pd"], "right"),
        ("guard rail_off", draws.T_guard_rail_off_s, C["rail_off"], "right"),
        ("break-even pd", draws.ts_min_energy_pd_s, C["pd"], "right"),
        ("break-even rail_off", draws.ts_min_energy_rail_off_s, C["rail_off"], "right"),
        ("crossover pd$\\leftrightarrow$rail_off", draws.crossover_s, "0.4", "left"),
        ("C6 edge pd", draws.green_hi_pd_s, C["pd"], "left"),
    ]
    hi_ms = 0.0
    for k, (label, ser, c, side) in enumerate(items):
        b = R.band(ser)
        y = len(items) - 1 - k
        lo, md, hi = b["p2_5"] * 1e3, b["median"] * 1e3, b["p97_5"] * 1e3
        hi_ms = max(hi_ms, hi)
        axb.plot([lo, hi], [y, y], color=c, lw=2.2, solid_capstyle="butt")
        axb.plot(md, y, "o", color=c, ms=4, mec="white", mew=0.5)
        text = f"{label}: {md:.3g} ms"
        if side == "right":
            axb.annotate(text, (max(hi, md), y), xytext=(5, 0), textcoords="offset points",
                         ha="left", va="center_baseline", color=INK)
        else:
            axb.annotate(text, (min(lo, md), y), xytext=(-5, 0), textcoords="offset points",
                         ha="right", va="center_baseline", color=INK)
    axb.set_yticks(range(len(items)))
    axb.set_yticklabels([])                              # the names sit beside the markers
    axb.tick_params(axis="y", length=0)
    axb.set_ylim(-0.6, len(items) - 0.4)
    axb.set_xscale("log")
    _plain_log(axb.xaxis)
    # the map grid is 0.3 ms to 2 s; the corrected crossover band ends later, so the axis does too
    # (the strip of the top panel beyond 2 s stays empty: no class was computed there)
    x_hi = R.TS_GRID[-1] * 1e3
    if hi_ms > x_hi:
        x_hi = 3000.0
        if hi_ms > x_hi:
            raise SystemExit(f"safemap_banded: a band ends at {hi_ms:.0f} ms, beyond the axis")
    axb.set_xlim(R.TS_GRID[0] * 1e3, x_hi)
    axb.set_xlabel(r"$T_{\mathrm{sleep}}$ (ms, log)")
    return _finish(fig, "safemap_banded", f"{src.relative_to(REPO)} (C-1, D6, C-7)")


def fig_b2_guard_violation() -> dict:
    """Deliberate guard truncation: measured bias at first trust on the b0*exp(-t/tau) band."""
    from analysis import b2_guard as B
    df = pd.read_csv(B.SWEEP / "b2_records.csv")
    fits = pd.read_csv(B.R1_FITS)
    ped = df[df.point_us == B.BASELINE_US].probe_mean_mg.mean()
    rows = []                                           # the per-point table of b2_guard.main()
    for us in sorted(df.point_us.unique(), reverse=True):
        g = df[df.point_us == us]
        tr = g.probe_mean_mg - ped
        rows.append({"point_us": int(us),
                     "t0_ms": g.t_probe0_us.mean() / 1e3, "t1_ms": g.t_probe1_us.mean() / 1e3,
                     "transient_mean_mg": float(tr.mean()),
                     "transient_sem_mg": float(tr.std(ddof=1) / np.sqrt(len(tr))),
                     "transient_std_mg": float(tr.std(ddof=1)),
                     "saturated": bool(tr.std(ddof=1) > 100.0)})
    # read-only check against the table b2_guard.main() committed beside the sweep
    committed = {p["point_us"]: p for p in json.loads((B.SWEEP / "analysis.json").read_text())["points"]}
    for r in rows:
        for key in ("t0_ms", "t1_ms", "transient_mean_mg", "transient_sem_mg"):
            if not math.isclose(r[key], committed[r["point_us"]][key], rel_tol=1e-9, abs_tol=1e-9):
                raise SystemExit(f"b2: {key} at {r['point_us']} us differs from the committed analysis.json")
        if r["saturated"] != committed[r["point_us"]]["saturated"]:
            raise SystemExit("b2: saturation flag differs from the committed analysis.json")

    fig, ax = _new(2.45)
    t = np.geomspace(fits.first_valid_us.min() / 1e3, 22.0, 400)
    c = B.curves(fits, t)
    valid = ~np.all(np.isnan(c), axis=0)
    t, c = t[valid], c[:, valid]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                 # all-NaN columns inside the blind window
        med = np.nanmedian(c, axis=0)
        lo = np.nanpercentile(c, 2.5, axis=0)
        hi = np.nanpercentile(c, 97.5, axis=0)
    ax.fill_between(t, lo, hi, color=B.COLOR["band"], alpha=0.20, lw=0,
                    label="$b_0e^{-t/\\tau}$ (Table II, 95% band)")
    ax.plot(t, med, color=B.COLOR["band"], lw=1.2)

    tfv_med = fits.first_valid_us.median() / 1e3
    ax.axvspan(1.5, tfv_med, color="0.55", alpha=0.18, lw=0)
    ax.annotate("blind window\n(err frames)", (1.6, 20), color=MUTED, linespacing=LS)
    ax.axhline(B.EPS_MG, ls="--", color=B.COLOR["RED"], lw=1)
    # was "eps_allowed = 1 mg" at the right, across the curve; the manuscript's symbol is plain epsilon
    ax.annotate(r"$\varepsilon$ = 1 mg", (1.6, 1.3), color=B.COLOR["RED"])
    # the guard of Table IV (R5S-2, 2026-10-19): the revised bootstrap with the slow term, in place of
    # b2_guard.GUARD_BAND_MS, the submitted version's 12.54-13.05 ms
    gb = json.loads((REPO / "tools" / "data" / "r3_map_uncertainty_unitA_rev" / "summary.json")
                    .read_text())["boundaries_s"]["T_guard_s"]
    ax.axvspan(1e3 * gb["p2_5"], 1e3 * gb["p97_5"], color="0.4", alpha=0.25, lw=0)
    ax.annotate(r"$T_{\mathrm{guard}}$", (1e3 * gb["median"], 330), color=MUTED, ha="center")

    for r in rows:
        x = 0.5 * (max(r["t0_ms"], tfv_med) + r["t1_ms"])
        xerr = [[x - max(r["t0_ms"], tfv_med)], [r["t1_ms"] - x]]
        if r["saturated"]:
            ax.errorbar(x, 200.0, xerr=xerr, fmt="v", color=B.COLOR["pd"], ms=5, capsize=2, lw=1)
            ax.annotate(f"saturated\n($\\sigma$ = {r['transient_std_mg'] / 1e3:.1f} g)", (x, 300),
                        color=B.COLOR["pd"], ha="center", va="bottom", linespacing=LS)
            continue
        y = max(abs(r["transient_mean_mg"]), 0.05)
        ax.errorbar(x, y, xerr=xerr, yerr=r["transient_sem_mg"], fmt="o", color=B.COLOR["pd"],
                    ms=4, mec="white", mew=0.5, capsize=2, lw=1,
                    label="measured (probe window)" if r["point_us"] == B.BASELINE_US else None)
        # each point's label sits just before the left end of its probe window, clear of the curve
        ax.annotate(f"{r['point_us'] / 1e3:g} ms", (max(r["t0_ms"], tfv_med), y), xytext=(-5, 0),
                    textcoords="offset points", color=B.COLOR["pd"], ha="right", va="center_baseline")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1.5, 24); ax.set_ylim(0.03, 1600)
    ax.set_xticks([2, 4, 6, 10, 15, 20])
    ax.set_xticklabels(["2", "4", "6", "10", "15", "20"])
    ax.xaxis.set_minor_locator(NullLocator())
    _plain_log(ax.yaxis)
    ax.set_xlabel("time after wake command (ms, log)")
    ax.set_ylabel("post-wake |bias| (mg, log)")
    ax.legend(loc="lower left")
    return _finish(fig, "b2_guard_violation", B.SWEEP.name)


def fig_thermal_paper() -> dict:
    """(a) observed collapse temperatures against each unit's prediction; (b) guard growth with
    die temperature on the original unit-A ramp; (c) unit A's pd map against die temperature
    (R5S-1, 2026-10-19), classified by the firmware C from Table II with b_s = 0, as the node held
    it in the ramps; the GREEN edge with the slow term dashed (R5S-6, 2026-10-20)."""
    import make_thermal_replication_fig as TR
    from analysis import p1_transfer_revision as T
    from bindings import core_ffi as core
    G = _defs_from_script(HERE / "make_thermal_paper_fig.py",
                          {"BIAS_TC", "EPS_ETA", "T_COL", "CLR", "CLR_OBS", "load_ramp"},
                          {"np": np, "DATA": TR.DATA})
    fig, (ax_a, ax_b, ax_c) = _new(5.65, nrows=3, gridspec_kw={"height_ratios": [1.0, 1.0, 1.0]})

    # -- (a) TR.draw_replication_panel ---------------------------------------------------------
    for i, (label, (pred, color, marker, logs)) in enumerate(TR.SERIES.items()):
        obs = [TR.collapse_temp(TR.DATA / L) for L in logs]
        x = np.full(len(obs), i) + np.linspace(-0.12, 0.12, len(obs))
        ax_a.plot(x, obs, marker, color=color, ms=5, mec="white", mew=0.6, zorder=3)
        ax_a.hlines(pred, i - 0.28, i + 0.28, color=color, ls=(0, (4, 2)), lw=1.2)
        ax_a.annotate(f"{pred:.2f}", (i + 0.30, pred), color=color, va="center", ha="left")
    ax_a.set_xticks(range(len(TR.SERIES)))
    for lab in ax_a.set_xticklabels(list(TR.SERIES.keys())):
        lab.set_linespacing(LS)
    ax_a.set_ylabel("observed collapse (°C)")
    ax_a.set_xlim(-0.5, 3.05)

    # -- (b) draw_guard_panel of make_thermal_paper_fig.py ---------------------------------------
    guard, tsleep, b0, tau, temp, src = G["load_ramp"]()
    T_COL, BIAS_TC, EPS_ETA = G["T_COL"], G["BIAS_TC"], G["EPS_ETA"]
    green = (tsleep > 0) & (guard > 0)
    ax_b.plot(temp[green], guard[green], ".", color=G["CLR"], ms=2.2, alpha=0.5, label="committed cycles")
    tt = np.linspace(20, T_COL - 0.02, 300)
    b0m, taum = np.median(b0[green]), np.median(tau[green])
    ax_b.plot(tt, taum * np.log(b0m / (EPS_ETA - BIAS_TC * np.abs(tt - 25.0))), ls="--", color="k", lw=1,
              label=r"C2-$\theta$ at median $\hat b_0,\hat\tau$")
    ax_b.axvline(T_COL, ls=":", color="k", lw=1)
    t_obs = float(temp[np.flatnonzero(~green)[0]])
    ax_b.axvline(t_obs, ls="-", color=G["CLR_OBS"], lw=1)
    # both values now sit in the empty strip to the right of the two lines (the "observed" label
    # used to lie under the legend)
    ax_b.annotate(f"observed\n{t_obs:.2f} °C", (t_obs + 0.35, 21.5), color=G["CLR_OBS"], ha="left",
                  va="center", linespacing=LS)
    ax_b.annotate(f"predicted\n{T_COL:.2f} °C", (T_COL + 0.38, 13.5), color=MUTED, ha="left",
                  va="center", linespacing=LS)
    ax_b.set(xlabel="die temp (°C)", ylabel=r"$T_{\mathrm{guard}}$ (ms)", xlim=(20, 40), ylim=(10, 29))
    ax_b.legend(loc="upper left")

    # -- (c) the pd map of unit A over monitoring bandwidth and die temperature --------------------
    # Each B_mon fixes the sleep, T_sleep = 1/(2 B_mon) - T_guard - T_probe (Section IV); the class
    # is sm_classify's at that sleep, with the guard at that die temperature. Grey: no sleep is left
    # (beyond the sampling limit). The dashed line is the GREEN edge with the slow term of Table II
    # (b_s = 0.21 mg); hatched, the GREEN the node held that the slow term closes.
    tol = core.tol(**T.TOL)
    t_probe = T.UNIT_A["pd"]["T_probe"]

    def cls(theta_c: float, b_hz: float, b_slow: float) -> int:
        s = core.sensor(**dict(T.UNIT_A["pd"], dT_op=theta_c - 25.0, b_slow=b_slow))
        tg = core.t_guard_min(s, tol)
        if not np.isfinite(tg):
            return 0
        t_s = 1.0 / (2.0 * b_hz) - tg - t_probe
        return -1 if t_s <= 0.0 else {"RED": 0, "YELLOW": 1, "GREEN": 2}[core.classify(s, tol, t_s)]

    theta = np.linspace(20.0, 40.0, 801)
    bmon = np.linspace(0.05, 30.0, 600)
    grid = np.array([[cls(th, b, T.B_SLOW_G) for th in theta] for b in bmon])
    grid0 = np.array([[cls(th, b, 0.0) for th in theta] for b in bmon])
    from analysis import r3_uncertainty as R
    C = R.COLOR                                         # the classes in Fig. 5's colours
    # filled class regions as vector paths (a pcolormesh of this grid would be 480 000 cells).
    # Coloured: the map with b_s = 0, as the node held it in the ramps, so its collapse sits under
    # the observed one (R5S-6, 2026-10-20; the first draw coloured the b_s = 0.21 map, which
    # painted RED over 33.5-36.3 C, where the node kept GREEN)
    ax_c.contourf(theta, bmon, grid0, levels=[-1.5, -0.5, 0.5, 1.5, 2.5],
                  colors=["0.88", C["RED"], C["YELLOW"], C["GREEN"]], antialiased=False)
    lost = ((grid0 == 2) & (grid != 2)).astype(float)   # GREEN the slow term closes
    with plt.rc_context({"hatch.color": "white", "hatch.linewidth": 0.6}):
        ax_c.contourf(theta, bmon, lost, levels=[0.5, 1.5], colors="none", hatches=["////"])
    ax_c.contour(theta, bmon, (grid == 2).astype(float), levels=[0.5], colors="k", linewidths=0.8,
                 linestyles=[(0, (4, 2))])
    obs_a = [TR.collapse_temp(TR.DATA / L) for L in next(iter(TR.SERIES.values()))[3]]
    ax_c.plot(obs_a, [28.6] * len(obs_a), "v", color="k", ms=3.5, ls="none")
    crit = 25.0 + ((1.0 - T.TOL["delta"]) * T.TOL["eps_allowed"] - T.B_SLOW_G) / abs(T.UNIT_A["pd"]["bias_tc"])
    crit0 = 25.0 + (1.0 - T.TOL["delta"]) * T.TOL["eps_allowed"] / abs(T.UNIT_A["pd"]["bias_tc"])
    ax_c.annotate("GREEN", (24.0, 10.0), color="white", ha="center", va="center")
    ax_c.annotate("RED", (38.4, 10.0), color="white", ha="center", va="center")
    ax_c.annotate("no cycle", (22.2, 28.3), color=INK, ha="center", va="center")
    ax_c.annotate(f"{crit:.2f} °C", (crit - 0.15, 3.0), color="white", ha="right", va="center")
    ax_c.annotate(f"{crit0:.2f} °C", (crit0 + 0.25, 3.0), color="white", ha="left", va="center")
    ax_c.annotate(f"$b_s={T.B_SLOW_G * 1e3:.2f}$ mg", (crit - 0.15, 12.5), color="white", ha="right",
                  va="center")
    ax_c.set(xlabel="die temp (°C)", ylabel=r"$B_{\mathrm{mon}}$ (Hz)", xlim=(20, 40), ylim=(0, 30))

    for ax, tag in ((ax_a, "(a)"), (ax_b, "(b)"), (ax_c, "(c)")):
        ax.set_title(tag, loc="left", pad=3)
    return _finish(fig, "thermal_paper", f"tools/data/ctl_thermal_*.log (4+4+2 ramps); panel (b) {src}; "
                   "panel (c) p1_transfer_revision.UNIT_A through libcore.so")


def fig_transfer_paper() -> dict:
    """Unit A's pd guard applied to both units' pd wake transients, on the revised coefficient
    sets of Section VII (Table II, the slow term in the budget, decision S7-8)."""
    from analysis import p1_transfer_revision as T
    fig, ax = _new(2.10)
    tg_a = T.guard(T.UNIT_A["pd"])
    tt = np.linspace(0, 0.018, 500)
    for name, c, t_fv, col in (("unit A", T.UNIT_A["pd"], 6.17e-3, "0.35"),
                               ("unit B", T.UNIT_B["pd"], 6.20e-3, "#009E73")):
        b = c["b0"] * np.exp(-tt / c["tau"]) * 1e3
        pre = tt < t_fv
        ax.semilogy(tt[pre] * 1e3, b[pre], color=col, lw=1.1, ls=(0, (1.5, 1.5)))
        ax.semilogy(tt[~pre] * 1e3, b[~pre], color=col, lw=1.3, label=name)
    ax.axhline(T.BUDGET_G * 1e3, color="k", lw=0.7, ls="--")
    ax.annotate(r"$(1{-}\delta)\,\varepsilon-b_s$", (0.4, T.BUDGET_G * 1e3 * 1.3))
    ax.axvline(tg_a * 1e3, color="0.2", lw=0.8)
    ax.annotate("unit-A guard", (tg_a * 1e3 - 0.3, 1.5e-2), color="0.2", ha="right")
    rB = T.transient_g(T.UNIT_B["pd"], tg_a) * 1e3
    ax.plot([tg_a * 1e3], [rB], "o", color="#009E73", ms=4, mec="white", mew=0.6)
    # was to the right of the marker, where the unit-A line crossed it
    ax.annotate(f"{rB:.2f} mg", (tg_a * 1e3, rB), xytext=(-7, -3), textcoords="offset points",
                color="#009E73", ha="right", va="top")
    ax.set(xlabel="time from wake command (ms)", ylabel="|bias| (mg)", xlim=(0, 18), ylim=(1e-2, 2e4))
    ax.set_xticks(range(0, 19, 2))
    _plain_log(ax.yaxis)
    ax.legend(loc="upper right")
    return _finish(fig, "transfer_paper", "analysis.p1_transfer_revision coefficient sets (units A and B, pd)")


def fig_testbed_photo_annotated() -> None:
    """Not built. The repository holds no unlabelled original of this photograph: both
    paper/figures/testbed_photo_annotated.jpg (text labels and arrows) and the JPEG inside
    paper/figures/testbed_photo.pdf (numbered red boxes) have their annotation in the pixels.
    Painting over either would be image manipulation, so the camera original is needed; with it,
    the labels can be drawn here as vector text over the photo at its native resolution."""
    print("  testbed_photo_annotated  NOT BUILT: no unlabelled original photograph in the repository")


# ================================================================================================
M3_RUNS = (("B", "m3_real_unitB_20261012_231635"), ("A", "m3_real_unitA_20261016_024528"))


def fig_m3_drift() -> dict:
    """Runs M3 (unit B) and M3-A (unit A): the hourly need of five independent pd wakes against the
    pd guard the firmware had in force in each hour's controller segment (lowest to median), by the
    pre-registered audit (twin/analysis/p1_m3_audit.py). Segments still running are left out, so the
    figure can be drawn while a run goes on."""
    from analysis import p1_m3_audit as M
    fig, axes = _new(3.0, nrows=2, sharex=True)
    c_need, c_guard = "#1f4e79", "#d9822b"
    for ax, (unit, name) in zip(axes, M3_RUNS):
        a = M.audit(M.runs.DATA / name)
        args = a["run"]["state"]["args"]
        for k, why in a["excluded"]:
            ax.axvspan(k, k + 1, color="0.55", alpha=0.25, lw=0)
        if a["run"]["state"].get("rest_ended"):
            r0 = args["rest_at"]
            ax.axvspan(r0, r0 + args["rest_len"], facecolor="none", edgecolor="0.45", hatch="////",
                       lw=0)
            ax.annotate("rest,\nsensor off", (r0 + 0.5 * args["rest_len"], 15.6), color=MUTED,
                        ha="center", va="top", linespacing=LS,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
        s = a["segments"]
        for row in s.itertuples():
            ax.fill_between([row.period, row.period + 1], row.guard_pd_min, row.guard_pd,
                            color=c_guard, alpha=0.55, lw=0)
        hrs = sorted((k, h) for k, h in a["hours"].items() if h is not None)
        ax.errorbar([k for k, _ in hrs], [h["need"] for _, h in hrs],
                    yerr=[h["se"] for _, h in hrs], fmt="o", ms=2.2, color=c_need, lw=0.7,
                    capsize=0, zorder=3)
        ax.set_ylim(9.5, 16.5)
        ax.set_ylabel("ms after wake")
        ax.annotate(f"unit {unit}", (0.015, 0.94), xycoords="axes fraction", va="top", color=INK)
    handles = [plt.Line2D([], [], marker="o", ms=3, ls="none", color=c_need),
               Rectangle((0, 0), 1, 1, color=c_guard, alpha=0.55, lw=0),
               Rectangle((0, 0), 1, 1, color="0.55", alpha=0.25, lw=0)]
    axes[0].legend(handles, ["need", "guard in force", "excluded"], loc="upper right", ncol=3,
                   handlelength=1.2, columnspacing=0.9)
    axes[1].set_xlim(0, 73)
    axes[1].set_xlabel("hours from the start of the run")
    return _finish(fig, "m3_drift", ", ".join(n for _, n in M3_RUNS))


FIGURES = {
    "testbed_photo_annotated": fig_testbed_photo_annotated,
    "inrush": fig_inrush,
    "recovery_raw": fig_recovery_raw,
    "recovery_terms": fig_recovery_terms,
    "hyst_campaign": fig_hyst_campaign,
    "safemap_banded": fig_safemap_banded,
    "b2_guard_violation": fig_b2_guard_violation,
    "thermal_paper": fig_thermal_paper,
    "transfer_paper": fig_transfer_paper,
    "m3_drift": fig_m3_drift,
}


def main(argv: list[str]) -> int:
    names = argv or list(FIGURES)
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        raise SystemExit(f"unknown figure(s) {unknown}; choose from {list(FIGURES)}")
    _style()
    face = Path(font_manager.findfont(font_manager.FontProperties(family="serif"))).name
    print(f"text font: {face}; output: {REV.relative_to(REPO)}/")
    done = [r for r in (FIGURES[n]() for n in names) if r]
    if done:
        print(f"{len(done)} figure(s) written; smallest text anywhere {min(r['min_pt'] for r in done):.1f} pt")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
