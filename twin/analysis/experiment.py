"""Experiment façade -- device-agnostic. The dashboard calls these; selecting
virtual vs physical changes only which device produces the raw samples."""
from __future__ import annotations
import subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent      # twin/
ENGINE = ROOT / "twin-engine"


def run_baseline(device: str = "virtual", n: int = 500, odr: float = 100.0,
                 temp: float = 25.0, out: str | None = None) -> pd.DataFrame:
    """Continuous-ON baseline experiment -> raw-sample DataFrame."""
    if device == "virtual":
        out = out or str(ROOT / "data" / "baseline.csv")
        r = subprocess.run(
            [str(ENGINE), "--device", "virtual", "--exp", "baseline",
             "--n", str(int(n)), "--odr", str(float(odr)), "--temp", str(float(temp)),
             "--out", out],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"twin-engine failed: {r.stderr.strip()}")
        return pd.read_csv(out)

    if device == "physical":
        from device.physical_device import PhysicalDevice
        dev = PhysicalDevice().connect()
        try:
            return dev.run_baseline(dur=max(5.0, n / odr))
        finally:
            dev.close()

    raise ValueError(f"unknown device: {device}")


def extract_baseline(df: pd.DataFrame) -> dict:
    """Extract P_on and the noise floor from a baseline capture."""
    on = df[df.drdy == 1]
    p_on = float((on.current_A * on.voltage_V).mean())
    noise = float(on.accel_g.std())
    return {"P_on_W": p_on, "noise_g": noise, "n": int(len(on))}


def run_fixedduty(device: str = "virtual", tsleep: float = 2.0, tprobe: float = 0.5,
                  dur: float = 30.0, odr: float = 100.0, sleepmode: str = "rail_off",
                  out: str | None = None) -> pd.DataFrame:
    """Fixed duty-cycle experiment -> raw-sample DataFrame (probe/sleep phases)."""
    if device != "virtual":
        from device.physical_device import PhysicalDevice
        PhysicalDevice().connect()
    out = out or str(ROOT / "data" / "fixedduty.csv")
    r = subprocess.run(
        [str(ENGINE), "--device", "virtual", "--exp", "fixedduty",
         "--tsleep", str(tsleep), "--tprobe", str(tprobe), "--dur", str(dur),
         "--odr", str(odr), "--sleepmode", sleepmode, "--out", out],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"twin-engine failed: {r.stderr.strip()}")
    return pd.read_csv(out)


def run_sleeppower(device: str = "virtual", hold: float = 5.0, odr: float = 100.0,
                   out: str | None = None) -> pd.DataFrame:
    """Hold each sleep mode and log current -> raw-sample DataFrame (phase = mode)."""
    if device == "physical":
        from device.physical_device import PhysicalDevice
        dev = PhysicalDevice().connect()
        try:                                   # vddio excluded until the EN_VDDIO jumper is moved
            frames = [dev.run_sleeppower(hold=hold, mode=m) for m in ("rail_off", "pd")]
        finally:
            dev.close()
        return pd.concat(frames, ignore_index=True)
    out = out or str(ROOT / "data" / "sleeppower.csv")
    r = subprocess.run(
        [str(ENGINE), "--device", "virtual", "--exp", "sleeppower",
         "--hold", str(hold), "--odr", str(odr), "--out", out],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"twin-engine failed: {r.stderr.strip()}")
    return pd.read_csv(out)


def run_recovery(device: str = "virtual", offdur: float = 5.0, recdur: float = 2.0,
                 odr: float = 2000.0, sleepmode: str = "rail_off", temp: float = 25.0,
                 tau_inrush: float | None = None, out: str | None = None) -> pd.DataFrame:
    """Power-cycle experiment: baseline -> off (in `sleepmode`) -> recovery, at `odr`.
    High odr is needed to resolve the inrush; this is the instrument-design knob.
    `tau_inrush` (s) overrides the surge time constant (energy stays E_inrush) so the
    rate requirement can be re-derived once the real surge shape is measured."""
    if device == "physical":
        from device.physical_device import PhysicalDevice
        dev = PhysicalDevice().connect()
        try:                                   # bias/timing only -- INA219 can't resolve E_inrush
            return dev.run_recovery(offdur=offdur, recdur=recdur, mode=sleepmode)
        finally:
            dev.close()
    out = out or str(ROOT / "data" / "recovery.csv")
    cmd = [str(ENGINE), "--device", "virtual", "--exp", "recovery",
           "--offdur", str(offdur), "--recdur", str(recdur), "--odr", str(odr),
           "--sleepmode", sleepmode, "--temp", str(float(temp)), "--out", out]
    if tau_inrush is not None:
        cmd += ["--tauin", str(float(tau_inrush))]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"twin-engine failed: {r.stderr.strip()}")
    return pd.read_csv(out)


def extract_sleep_power(df: pd.DataFrame) -> dict:
    """P_sleep per mode = mean(I*V) over each mode's held segment."""
    return {str(mode): float((g.current_A * g.voltage_V).mean())
            for mode, g in df.groupby("phase")}


def run_scalefactor(device: str = "virtual", offdur: float = 2.0, odr: float = 200.0,
                    temp: float = 25.0, out: str | None = None) -> pd.DataFrame:
    """Known-stimulus (0 g / 1 g) before & after a power cycle, per mode -> SF(m)."""
    if device != "virtual":
        from device.physical_device import PhysicalDevice
        PhysicalDevice().connect()
    out = out or str(ROOT / "data" / "scalefactor.csv")
    r = subprocess.run(
        [str(ENGINE), "--device", "virtual", "--exp", "scalefactor",
         "--offdur", str(offdur), "--odr", str(odr), "--temp", str(float(temp)), "--out", out],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"twin-engine failed: {r.stderr.strip()}")
    return pd.read_csv(out)


def extract_scalefactor(df: pd.DataFrame) -> dict:
    """SF = reading@1g - reading@0g (the slope; bias cancels). Reference + per mode,
    plus the post-wake deviation |SF(m) - SF_ref| / SF_ref (constraint C5)."""
    def sf(seg0, seg1):
        return float(df[df.phase == seg1].accel_g.mean()
                     - df[df.phase == seg0].accel_g.mean())
    sf_ref = sf("ref_0g", "ref_1g")
    sf_mode = {m: sf(f"{m}_0g", f"{m}_1g") for m in ("rail_off", "pd", "vddio")}
    dev = {m: abs(sf_mode[m] - sf_ref) / sf_ref for m in sf_mode}
    return {"SF_ref": sf_ref, "SF": sf_mode, "dev": dev}


def measured_p_avg(df: pd.DataFrame) -> float:
    """Average power actually drawn over a run = mean(I*V)."""
    return float((df.current_A * df.voltage_V).mean())
