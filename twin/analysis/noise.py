"""Noise-floor characterization for the MMS-30 (C4 + the detector threshold).

Determines whether the 1.74 mg RMS we measured is the INTRINSIC floor or
measurement/bandwidth-limited, by reporting the bandwidth-INDEPENDENT quantities:
  - noise spectral density (ug/sqrt(Hz))  -> comparable to the datasheet (<=85 ug/sqrt(Hz))
    and competitors (ADXL355 25, AXO314 15, SI1000 0.7 ug/sqrt(Hz)),
  - Allan deviation -> bias instability (ug) and velocity random walk (the
    standard inertial-grade metrics),
  - RMS-vs-bandwidth check -> confirms white-noise dominance (RMS scales as sqrt(BW)),
    so density (not RMS) is the right metric.

Feed it a STATIC capture (sensor dead still on an isolated mount): either a firmware
BURST block or a (t, accel) CSV. fs is inferred from the timestamps.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def _detrend(a: np.ndarray) -> np.ndarray:
    """Remove the static bias (mean) so we measure fluctuation, not offset."""
    return a - np.mean(a)


def deglitch(accel_g: np.ndarray, n_sigma: float = 10.0, min_gap_mg: float = 8.0):
    """Reject torn-SPI-read outliers before noise analysis. At high burst rates ~0.8% of
    MMS-30 reads collide with the sensor's internal register update and come back as
    *torn frames* that saturate toward the +-30 g rail. The 40-bit frame has no CRC and a
    torn read's drdy/err bits are themselves garbage, so these can only be caught by
    MAGNITUDE: real static noise is ~1-2 mg while torn reads are 10s-1000s of mg, leaving a
    clean bimodal GAP (~9.5 mg .. ~11 mg on the bench). We reject |a-median| beyond
    max(n_sigma*MAD, min_gap_mg) -- which lands in that gap -- and fill the sparse holes by
    interpolation. Returns (clean_array, bad_mask). Report bad_mask.mean() as the glitch
    rate: it is a data-quality metric, not noise."""
    a = np.asarray(accel_g, float).copy()
    med = np.median(a)
    mad = float(np.median(np.abs(a - med))) * 1.4826           # robust sigma
    thr = max(n_sigma * mad, min_gap_mg * 1e-3)
    bad = np.abs(a - med) > thr
    if bad.any():
        idx = np.arange(len(a))
        a[bad] = np.interp(idx[bad], idx[~bad], a[~bad])       # glitches are sparse (<1%)
    return a, bad


def psd_density(accel_g: np.ndarray, fs: float, band=(1.0, 100.0)) -> dict:
    """Welch PSD -> noise spectral density. Returns the median density in `band`
    (the flat white-noise floor) plus the full (f, density) arrays. Density is in
    g/sqrt(Hz); multiply by 1e6 for ug/sqrt(Hz)."""
    a = _detrend(np.asarray(accel_g, float))
    nperseg = min(len(a), max(256, int(fs)))          # ~1 s segments
    f, pxx = signal.welch(a, fs=fs, nperseg=nperseg)   # pxx in g^2/Hz
    dens = np.sqrt(pxx)                                 # g/sqrt(Hz)
    m = (f >= band[0]) & (f <= band[1])
    floor = float(np.median(dens[m])) if m.any() else float("nan")
    return {"f": f, "density_g": dens, "floor_g_rtHz": floor,
            "floor_ug_rtHz": floor * 1e6, "band": band}


def allan_deviation(accel_g: np.ndarray, fs: float, n_taus: int = 40) -> dict:
    """Overlapping Allan deviation of the (integrated) signal. For an accelerometer
    we Allan-analyze the acceleration directly: the -1/2 slope region is the
    velocity-random-walk / white-noise (noise density), and the flat minimum is the
    bias instability. Returns taus (s), adev (g), and the extracted metrics."""
    a = _detrend(np.asarray(accel_g, float))
    n = len(a)
    dt = 1.0 / fs
    taus = np.unique(np.floor(np.logspace(0, np.log10(n // 3), n_taus)).astype(int))
    taus = taus[taus >= 1]
    adev, tau_s = [], []
    cum = np.cumsum(a) * dt                              # integrate -> "velocity"
    for m in taus:
        if 2 * m >= n:
            break
        # overlapping Allan variance of the integrated signal
        y = cum[2 * m:] - 2 * cum[m:-m] + cum[:-2 * m]
        avar = np.mean(y ** 2) / (2 * (m * dt) ** 2)
        adev.append(np.sqrt(avar))
        tau_s.append(m * dt)
    tau_s, adev = np.array(tau_s), np.array(adev)
    bias_instability = float(adev.min()) if len(adev) else float("nan")
    tau_at_min = float(tau_s[int(np.argmin(adev))]) if len(adev) else float("nan")
    return {"tau_s": tau_s, "adev_g": adev,
            "bias_instability_g": bias_instability, "tau_at_min_s": tau_at_min}


def rms_vs_bandwidth(accel_g: np.ndarray, fs: float, bws=(5, 20, 50, 100)) -> list:
    """Low-pass at each BW and report RMS. White noise -> RMS ~ density*sqrt(BW),
    so RMS should scale as sqrt(BW). Deviation flags non-white content or limits."""
    a = _detrend(np.asarray(accel_g, float))
    out = []
    for bw in bws:
        if bw >= fs / 2:
            continue
        b, aa = signal.butter(4, bw / (fs / 2))
        rms = float(np.std(signal.filtfilt(b, aa, a)))
        out.append({"bw_hz": bw, "rms_g": rms, "rms_mg": rms * 1e3})
    return out


def analyze_static(t_s: np.ndarray, accel_g: np.ndarray) -> dict:
    """Full static-noise report. t_s, accel_g are equal-length arrays from a still
    capture. Returns density, Allan, RMS, and prints a datasheet-comparable summary."""
    t = np.asarray(t_s, float)
    fs = 1.0 / np.median(np.diff(t))
    a_raw = np.asarray(accel_g, float)
    a, bad = deglitch(a_raw)                       # remove torn-SPI-read outliers first
    glitch_rate = float(bad.mean())
    rms_full = float(np.std(_detrend(a)))
    psd = psd_density(a, fs)
    allan = allan_deviation(a, fs)
    rb = rms_vs_bandwidth(a, fs)
    print(f"=== static noise: n={len(a)}, fs={fs:.0f} Hz, span={(t[-1]-t[0]):.1f} s ===")
    if bad.any():
        print(f"  torn-read glitches    = {bad.sum()} ({glitch_rate*100:.2f}%)  rejected "
              f"[raw RMS {np.std(_detrend(a_raw))*1e3:.1f} mg was glitch-inflated]")
    print(f"  full-band RMS         = {rms_full*1e3:.3f} mg")
    print(f"  noise density (1-100Hz)= {psd['floor_ug_rtHz']:.1f} ug/sqrt(Hz)   "
          f"(DS <=85; ADXL355 25; AXO314 15; SI1000 0.7)")
    print(f"  bias instability       = {allan['bias_instability_g']*1e6:.1f} ug "
          f"(at tau={allan['tau_at_min_s']:.2f} s)")
    print("  RMS vs bandwidth (white noise -> ~sqrt(BW)):")
    for r in rb:
        print(f"    {r['bw_hz']:4.0f} Hz -> {r['rms_mg']:.3f} mg")
    return {"fs_hz": fs, "rms_full_mg": rms_full * 1e3, "glitch_rate": glitch_rate,
            **psd, **allan, "rms_vs_bw": rb}


if __name__ == "__main__":          # synthetic self-check: white noise of known density
    fs, dur, dens = 2400.0, 30.0, 85e-6        # 85 ug/sqrt(Hz) target
    n = int(fs * dur)
    rng = np.random.default_rng(0)
    a = rng.normal(0, dens * np.sqrt(fs / 2), n) + 0.5     # + static bias 0.5 g
    t = np.arange(n) / fs
    r = analyze_static(t, a)
    print(f"\nself-check: target 85 ug/sqrt(Hz), recovered {r['floor_ug_rtHz']:.1f} ug/sqrt(Hz)")
