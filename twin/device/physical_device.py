"""Physical device driver (serial -> STM32 + MMS-30 + INA219).

M7: the real DAL implementation. Opens the firmware's UART CSV stream, maps each
firmware row to the twin's raw-sample schema, and drives the board (power / sleep
modes) so the SAME extraction pipeline (analysis.experiment / .recovery / .mapbuild)
runs unchanged against hardware.

Firmware contract (see docs/m7_hil_handoff_mms30_hardware_firmware_2026-06-24.md and
Core/Src/drivers/data_logger.c) -- 15-field CSV @115200 on /dev/ttyACM0:

  timestamp_ms,accel_g,temp_c,current_mA,voltage_V,power_mW,sleep_duration_s,
  actual_sleep_ms,drift_mg,mms30_valid,ina219_valid,mms30_status,ina219_status,drdy,err

mapped to the twin schema (t_s,accel_g,current_A,voltage_V,temp_C,drdy,err,phase).

Honest limits (handoff): INA219 measures the 5 V rail only at <=~200 Hz, so this path
gives steady-state / average power and bias-recovery TIMING, but NOT E_inrush (needs a
>=5 kHz current path). pd/vddio modes need the new control firmware flashed first.
"""
from __future__ import annotations
import time

import pandas as pd

try:
    import serial
except ImportError:                      # pyserial optional until hardware is used
    serial = None


class DeviceNotConnected(Exception):
    pass


# firmware CSV column order (Core/Src/drivers/data_logger.c)
_FW_COLS = ("timestamp_ms", "accel_g", "temp_c", "current_mA", "voltage_V", "power_mW",
            "sleep_duration_s", "actual_sleep_ms", "drift_mg", "mms30_valid",
            "ina219_valid", "mms30_status", "ina219_status", "drdy", "err")
_NCOLS = len(_FW_COLS)
# non-data lines the firmware interleaves (banner / acks / status)
_SKIP_PREFIX = ("INFO:", "WARN:", "ERROR:", "ACK:", "STATE:", "=", "timestamp_ms",
                "energy-aware", "MMS-30")
# twin raw-sample schema (matches twin-engine output)
SCHEMA = ("t_s", "accel_g", "current_A", "voltage_V", "temp_C", "drdy", "err", "phase")


def _parse_row(line: str):
    """Firmware CSV line -> dict in the twin schema, or None if not a data row."""
    s = line.strip()
    if not s or s.startswith(_SKIP_PREFIX):
        return None
    p = [x.strip() for x in s.split(",")]
    if len(p) != _NCOLS:
        return None
    try:
        return {
            "accel_g": float(p[1]), "temp_C": float(p[2]),
            "current_A": float(p[3]) / 1000.0, "voltage_V": float(p[4]),
            "mms30_valid": int(p[9]), "drdy": int(p[13]), "err": int(p[14]),
        }
    except ValueError:
        return None


class PhysicalDevice:
    """Serial DAL device. `connect()` then use the run_* helpers, which return
    DataFrames in the twin schema (same as the virtual twin-engine)."""
    kind = "physical"

    def __init__(self, port: str = "/dev/ttyACM0", baud: int = 115200, timeout: float = 0.3):
        self.port, self.baud, self.timeout = port, baud, timeout
        self.ser = None

    # ---- connection ----
    def connect(self):
        if serial is None:
            raise DeviceNotConnected("pyserial not installed (pip install pyserial)")
        try:
            ser = serial.Serial()                    # configure BEFORE open so DTR/RTS
            ser.port = self.port                     # don't pulse and reset the MCU
            ser.baudrate = self.baud
            ser.timeout = self.timeout
            ser.dtr = False
            ser.rts = False
            ser.open()
            self.ser = ser
        except Exception as e:                       # noqa: BLE001
            raise DeviceNotConnected(f"cannot open {self.port}: {e}") from e
        time.sleep(0.3)
        self.ser.reset_input_buffer()
        return self

    def close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            finally:
                self.ser = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    # ---- low-level ----
    def _cmd(self, command: str, settle: float = 0.15):
        """Send a firmware command (e.g. 'PWR OFF'); does not wait for the ACK row."""
        if self.ser is None:
            raise DeviceNotConnected("not connected")
        self.ser.write((command + "\n").encode("ascii"))
        self.ser.flush()
        time.sleep(settle)

    def collect(self, duration_s: float, phase: str, t0: float | None = None,
                on_row=None) -> list[dict]:
        """Stream rows for `duration_s` wall-clock seconds, labelled `phase`.
        t_s is host-monotonic (robust across power cycles).
        `on_row`, if given, is called with each parsed row dict as it arrives — used
        for live bench feedback (e.g. run_temp_hw's die-temperature readout). It runs
        in the read loop, so keep it cheap and self-throttling."""
        if self.ser is None:
            raise DeviceNotConnected("not connected")
        rows, start = [], time.monotonic()
        base = start if t0 is None else t0
        while time.monotonic() - start < duration_s:
            raw = self.ser.readline()
            if not raw:
                continue
            r = _parse_row(raw.decode("ascii", "replace"))
            if r is None:
                continue
            row = {"t_s": time.monotonic() - base, "accel_g": r["accel_g"],
                   "current_A": r["current_A"], "voltage_V": r["voltage_V"],
                   "temp_C": r["temp_C"], "drdy": r["drdy"], "err": r["err"],
                   "phase": phase}
            rows.append(row)
            if on_row is not None:
                on_row(row)
        return rows

    # ---- power / mode control ----
    def power(self, on: bool):
        self._cmd("PWR ON" if on else "PWR OFF", settle=0.05)

    def ensure_streaming(self):
        self._cmd("SET POLL ON")

    def set_sample_ms(self, ms: int):
        self._cmd(f"SET SAMPLE_MS {max(5, min(1000, int(ms)))}")

    def run_burst(self, mode: str = "rail_off", sleep_ms: int = 1000, n: int = 2000,
                  timeout_s: float = 30.0) -> list[str]:
        """Trigger a firmware BURST capture (high-rate post-wake accel transient) and
        return its lines, for analysis.hwcapture.recovery_from_burst -> b0, tau_bias."""
        if self.ser is None:
            raise DeviceNotConnected("not connected")
        self.ser.reset_input_buffer()
        self._cmd(f"BURST {mode} {int(sleep_ms)} {int(n)}", settle=0.0)
        lines, t0 = [], time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            raw = self.ser.readline()
            if not raw:
                continue
            ln = raw.decode("ascii", "replace").rstrip()
            lines.append(ln)
            if ln.startswith("BURST_END"):
                break
        return lines

    def _sleep(self, mode: str):
        """Enter the given sleep mode (verified polarities on the revA bench)."""
        if mode == "rail_off":  self.power(False)              # EN_MAIN low: rails -> 0 V
        elif mode == "pd":      self._cmd("PD ON", settle=0.05)   # PD HIGH = power-down
        elif mode == "vddio":   self._cmd("VDDIO OFF", settle=0.05)  # needs EN_VDDIO jumper moved
        else: raise ValueError(f"unknown mode {mode!r}")

    def _wake(self, mode: str):
        if mode == "rail_off":  self.power(True)
        elif mode == "pd":      self._cmd("PD OFF", settle=0.05)
        elif mode == "vddio":   self._cmd("VDDIO ON", settle=0.05)
        else: raise ValueError(f"unknown mode {mode!r}")

    # ---- experiment runners (twin-schema DataFrames; same as the virtual engine) ----
    def run_baseline(self, dur: float = 20.0, **_) -> pd.DataFrame:
        """Continuous-ON baseline -> P_on, noise. Read-only: collects the live stream
        (the board is already powered/streaming); does not toggle power."""
        return pd.DataFrame(self.collect(dur, "baseline"), columns=SCHEMA)

    def run_sleeppower(self, hold: float = 5.0, mode: str = "rail_off",
                       settle: float = 2.0, **_) -> pd.DataFrame:
        """Hold a sleep mode and log the rail current -> P_sleep. For pd/vddio the
        rails stay on; for vddio the EN_VDDIO jumper must be moved first."""
        self.power(True); self._wake(mode); self.collect(settle, "_warmup")  # rails on, active, settle
        self._sleep(mode)
        rows = self.collect(hold, mode)
        self._wake(mode)                                       # leave active
        return pd.DataFrame(rows, columns=SCHEMA)

    def run_recovery(self, offdur: float = 2.0, recdur: float = 1.0, base: float = 0.5,
                     mode: str = "rail_off", **_) -> pd.DataFrame:
        """baseline -> sleep -> wake -> recovery, for bias/timing (NOT E_inrush: INA219 too slow)."""
        self.power(True); self._wake(mode)
        rows = self.collect(base, "baseline")
        self._sleep(mode)
        rows += self.collect(offdur, "off")
        self._wake(mode)                                       # wake
        rows += self.collect(recdur, "recovery")
        return pd.DataFrame(rows, columns=SCHEMA)

    def run_scalefactor(self, dur: float = 4.0, modes=("rail_off", "pd"),
                        settle: float = 1.5, **_) -> pd.DataFrame:
        """Scale factor via the gravity stimulus (manual orientation): collect the sensor
        reading at 0 g (sensitive axis HORIZONTAL) and 1 g (axis VERTICAL, tip up) for the
        REFERENCE (continuous-ON) and after a wake in each mode. extract_scalefactor() then
        gives SF = mean(accel@1g) - mean(accel@0g) (bias cancels) and the C5 deviation
        |SF(m)-SF_ref|/SF_ref. Prompts you to reorient between phases; rest the board STILL
        on a flat/right-angle surface each time (don't hand-hold -- tremor adds tilt error)."""
        self.power(True); self._wake("rail_off"); self.ensure_streaming()
        self.collect(settle, "_warmup")
        rows = []

        def _stats(seg):
            n = len(seg); m = sum(seg) / n
            sd = (sum((x - m) ** 2 for x in seg) / n) ** 0.5
            return m, sd

        def oriented(phase, what, still_mg: float = 8.0):
            """Gate on stability: only log once the reading is steady (std < still_mg).
            The static noise floor is ~1.7 mg, so >8 mg std means the board is moving."""
            while True:
                input(f"\n>>> {what}\n    rest the board STILL (on a surface/fixture), press Enter...")
                time.sleep(1.0)                                  # let it settle after placing
                self.ser.reset_input_buffer()                    # DROP stale readings buffered while reorienting
                chk = [r["accel_g"] for r in self.collect(1.0, "_chk")]
                if not chk:
                    print("    no data — is it streaming? retrying..."); continue
                m, sd = _stats(chk)
                print(f"    reading: mean = {m * 1e3:7.1f} mg,  std = {sd * 1e3:5.1f} mg")
                if sd * 1e3 > still_mg:
                    print(f"    !! still MOVING (std {sd*1e3:.0f} mg > {still_mg:.0f}) — hold it dead "
                          f"still / clamp it, don't touch during logging. Retry.")
                    continue
                self.ser.reset_input_buffer()                    # fresh data only for the logged window
                rows.extend(self.collect(dur, phase))
                seg = [r["accel_g"] for r in rows if r["phase"] == phase]
                m2, sd2 = _stats(seg)
                print(f"    logged {phase}: mean = {m2 * 1e3:7.2f} mg, std = {sd2 * 1e3:.2f} mg  (n={len(seg)})")
                break

        oriented("ref_0g", "REFERENCE  0 g : sensitive axis HORIZONTAL (board flat)")
        oriented("ref_1g", "REFERENCE  1 g : sensitive axis VERTICAL (tip up 90 deg)")
        for m in modes:
            print(f"\n--- mode {m}: power-cycling (sleep -> wake) ---")
            self._sleep(m); time.sleep(2.0); self._wake(m); self.ensure_streaming()
            self.collect(settle, "_warmup")
            oriented(f"{m}_0g", f"{m} after wake,  0 g : axis HORIZONTAL")
            oriented(f"{m}_1g", f"{m} after wake,  1 g : axis VERTICAL")
        return pd.DataFrame(rows, columns=SCHEMA)
