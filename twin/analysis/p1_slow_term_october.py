"""Paper 1 revision, D6: how much data the slow term rests on per unit, and the two-term fit of the
October bursts pooled (unit B: run M1 and run M3 before the fault; unit A: run M1 and run M3-A).
Same burst validity as slow_term_persistence(); BIC as the criterion; 200 resamples of the bursts.

Run: twin/.venv/bin/python twin/analysis/p1_slow_term_october.py
"""
import sys, re
from pathlib import Path
TW = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TW)); sys.path.insert(0, str(TW / 'analysis'))
import numpy as np
import p1_revision_checks as chk, p1_recovery_models as models, paper2_runs as runs

def valid(name):
    try:
        b = models.load_bursts(name)
    except (FileNotFoundError, ValueError):
        return []
    return [x for x in b if 6.12e-3 <= x[2] <= 6.25e-3 and len(x[0]) >= 1900]

# 1. record census (to 2026-10-03), pd runs only
cen = {'A': [0, 0, 0], 'B': [0, 0, 0]}
for d in chk.record_runs():
    name = str(d.relative_to(runs.DATA))
    u = chk.unit_of(name)
    if u not in cen or 'rail' in name.lower():
        continue
    b = valid(name)
    if not b:
        continue
    cen[u][0] += len(b)
    if len(b) >= 20:
        cen[u][1] += 1; cen[u][2] += len(b)
for u, (n, r20, n20) in cen.items():
    print(f'record unit {u}: valid pd bursts {n}; runs with >=20 bursts {r20} holding {n20} bursts')

def pooled(names, label, reps=200, seed=1):
    B = []
    for n in names:
        B += valid(n)
    t, a, fv, _ = models.ensemble(B)
    f1, f2 = models.fit('M1', t, a), models.fit('M2', t, a)
    c = a[len(a)//2:]
    noise = 1e3 * float(np.std(c))
    rng = np.random.default_rng(seed); amp, tau2, pref, sign = [], [], [], []
    for _ in range(reps):
        s = [B[i] for i in rng.integers(0, len(B), len(B))]
        tt, aa, _, _ = models.ensemble(s)
        g1, g2 = models.fit('M1', tt, aa), models.fit('M2', tt, aa)
        amp.append(1e3*abs(g2['p'][3])); tau2.append(1e3*g2['p'][4])
        pref.append(g2['bic'] < g1['bic']); sign.append(np.sign(g2['p'][3]) == np.sign(g2['p'][1]))
    k = (t >= 20e-3) & (t < 100e-3); late = 1e3*(a[k].mean() - c.mean())
    print(f'{label}: {len(B)} bursts; noise of the mean {noise:.3f} mg; main b0 {1e3*abs(f1["p"][1]):.1f} mg tau {1e3*f1["p"][2]:.3f} ms')
    print(f'   slow amp {1e3*abs(f2["p"][3]):.3f} mg [{np.percentile(amp,2.5):.3f}, {np.percentile(amp,97.5):.3f}]; '
          f'tau2 {1e3*f2["p"][4]:.0f} ms [{np.percentile(tau2,2.5):.0f}, {np.percentile(tau2,97.5):.0f}]; '
          f'dBIC {f2["bic"]-f1["bic"]:+.1f}; two-term preferred in {100*np.mean(pref):.0f} % of resamples; same sign {100*np.mean(sign):.0f} %')
    print(f'   mean deviation 20-100 ms after first valid: {late:+.3f} mg')

D = runs.DATA
b_oct = ['r1_burst_repeatability_unitB_m1b_fw33facb8_pd_n30_20261012_221930',
         'r1_burst_repeatability_unitB_m1b_fw33facb8_pd_close_n10_20261012_223147'] + \
        sorted(p.name for p in D.glob('r1_burst_repeatability_m3_real_unitB_p*_pd_*'))
a_oct = ['r1_burst_repeatability_unitA_m1b_fw33facb8_pd_n30_20261016_023012',
         'r1_burst_repeatability_unitA_m1b_fw33facb8_pd_close_n10_20261016_024151'] + \
        sorted(p.name for p in D.glob('r1_burst_repeatability_m3_real_unitA_p*_pd_*'))
pooled(['r1_burst_repeatability_unitB_r1_burst_pd_n30_20260712_232025'], 'unit B, July calibration set')
pooled(b_oct, 'unit B, October (M1 + run M3, before the fault)')
pooled(['r1_burst_repeatability_unitA_r1_burst_pd_n30_reprocessed_20260706_203023'], 'unit A, July calibration set')
pooled(a_oct, 'unit A, October (M1 + run M3-A so far)')
