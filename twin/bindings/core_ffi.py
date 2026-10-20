"""cffi binding to libcore.so -- the dashboard computes the safe map with the
EXACT C the STM32 firmware runs (no re-implementation, no drift)."""
from __future__ import annotations
from pathlib import Path
from cffi import FFI

_ffi = FFI()
_ffi.cdef(
    """
    typedef struct {
        float P_on, P_sleep, E_inrush, b0, tau, sigma_pw,
              SF, SF_ref, fs, T_probe, a_max_exp, range_limit,
              bias_tc, dT_op, b_slow;
    } sm_sensor_t;
    typedef struct {
        float eps_allowed, sigma_allowed, SF_allowed,
              B_required, D_max, S_min, delta;
    } sm_tol_t;

    float  sm_t_sleep_min(const sm_sensor_t *s);
    float  sm_t_guard_min(const sm_sensor_t *s, const sm_tol_t *t);
    float  sm_p_avg(const sm_sensor_t *s, float T_sleep, float T_guard);
    float  sm_b_eff(const sm_sensor_t *s, float T_sleep, float T_guard);
    float  sm_saving(const sm_sensor_t *s, float T_sleep, float T_guard);
    int    sm_classify(const sm_sensor_t *s, const sm_tol_t *t, float T_sleep);
    float  sm_temp_bias(const sm_sensor_t *s);
    float  sm_b0_delayed(float b0, float tau, float delay_s);
    _Bool  sm_green_interval(const sm_sensor_t *s, const sm_tol_t *t,
                             float *lo, float *hi);
    """
)

_LIB = _ffi.dlopen(str(Path(__file__).resolve().parent.parent / "libcore.so"))

# field order must match core/safemap.h exactly (ABI struct layout)
_SENSOR_FIELDS = ("P_on P_sleep E_inrush b0 tau sigma_pw "
                  "SF SF_ref fs T_probe a_max_exp range_limit bias_tc dT_op b_slow").split()
_TOL_FIELDS = "eps_allowed sigma_allowed SF_allowed B_required D_max S_min delta".split()

CLASS = {0: "RED", 1: "YELLOW", 2: "GREEN"}


def sensor(**kw):
    s = _ffi.new("sm_sensor_t *")
    for f in _SENSOR_FIELDS:
        setattr(s, f, float(kw.get(f, 0.0)))
    return s


def tol(**kw):
    t = _ffi.new("sm_tol_t *")
    for f in _TOL_FIELDS:
        setattr(t, f, float(kw.get(f, 0.0)))
    return t


def t_sleep_min(s):                  return _LIB.sm_t_sleep_min(s)
def t_guard_min(s, t):               return _LIB.sm_t_guard_min(s, t)
def p_avg(s, ts, tg):                return _LIB.sm_p_avg(s, ts, tg)
def b_eff(s, ts, tg):                return _LIB.sm_b_eff(s, ts, tg)
def saving(s, ts, tg):               return _LIB.sm_saving(s, ts, tg)
def classify(s, t, ts):              return CLASS[_LIB.sm_classify(s, t, ts)]
def temp_bias(s):                    return _LIB.sm_temp_bias(s)
def b0_delayed(b0, tau, delay_s):   return _LIB.sm_b0_delayed(b0, tau, delay_s)


def green_interval(s, t):
    lo = _ffi.new("float *")
    hi = _ffi.new("float *")
    ok = _LIB.sm_green_interval(s, t, lo, hi)
    return (lo[0], hi[0]) if ok else None


# the measured MMS-30 baseline (preliminary; SENSOR_COEFFICIENTS.md)
def mms30(E_inrush=0.005, dT_op=0.0):
    return sensor(P_on=0.067, P_sleep=0.00106, E_inrush=E_inrush, b0=0.0005,
                  tau=0.05, sigma_pw=0.0009, SF=1.0, SF_ref=1.0, fs=350.0,
                  T_probe=0.5, a_max_exp=1.0, range_limit=30.0,
                  bias_tc=-75.5e-6, dT_op=dT_op)  # measured bias-TC (2026-06-30)


def default_tol():
    return tol(eps_allowed=0.001, sigma_allowed=0.0015, SF_allowed=0.0255,
               B_required=0.1, D_max=8.0, S_min=0.10, delta=0.15)
