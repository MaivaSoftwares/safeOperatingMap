#include "safemap.h"
#include <math.h>

/* small slack so boundary equalities classify as "satisfied" under float error */
#define EPS 1e-9f

float sm_t_sleep_min(const sm_sensor_t *s)
{
    float dp = s->P_on - s->P_sleep;
    if (dp <= 0.0f) return INFINITY;   /* sleep saves nothing -> never beneficial */
    return s->E_inrush / dp;
}

float sm_bias_after(const sm_sensor_t *s, float T_guard)
{
    if (s->tau <= 0.0f) return s->b0;
    return s->b0 * expf(-T_guard / s->tau);
}

/* magnitude of the UNCOMPENSATED static temperature-induced bias (C2 theta-axis). This is a
 * constant offset (does not decay), so it eats the eps budget before the wake transient does. */
float sm_temp_bias(const sm_sensor_t *s)
{
    float tb = s->bias_tc * s->dT_op;
    return tb < 0.0f ? -tb : tb;
}

/* Wake-referenced amplitude of a transient that is another method's delayed by delay_s (rail_off:
 * the rail has to come up before the same recovery starts). With this b0 and the same tau,
 * sm_t_guard_min() is longer by exactly delay_s, and the planner sees the longer guard's cost. */
float sm_b0_delayed(float b0, float tau, float delay_s)
{
    if (tau <= 0.0f || delay_s <= 0.0f) return b0;
    return b0 * expf(delay_s / tau);
}

float sm_t_guard_min(const sm_sensor_t *s, const sm_tol_t *t)
{
    /* wait until the wake transient falls to the GREEN-margined tolerance (1-delta)*eps, MINUS
     * whatever the static temperature offset and the slow term already consume. If those alone
     * exceed the budget, no guard can recover the point -> C2 infeasible (RED). */
    float target = (1.0f - t->delta) * t->eps_allowed - sm_temp_bias(s) - s->b_slow;
    if (target <= 0.0f)                return INFINITY;   /* static temp bias alone busts budget */
    if (s->b0 <= target)               return 0.0f;       /* transient already in spec at wake */
    if (s->tau <= 0.0f)                return INFINITY;    /* never recovers */
    return s->tau * logf(s->b0 / target);
}

float sm_t_cycle(const sm_sensor_t *s, float T_sleep, float T_guard)
{
    return T_sleep + T_guard + s->T_probe;
}

float sm_p_avg(const sm_sensor_t *s, float T_sleep, float T_guard)
{
    float Tc = sm_t_cycle(s, T_sleep, T_guard);
    float E  = s->P_sleep * T_sleep
             + s->P_on    * T_guard
             + s->E_inrush
             + s->P_on    * s->T_probe;
    return E / Tc;
}

float sm_saving(const sm_sensor_t *s, float T_sleep, float T_guard)
{
    return 1.0f - sm_p_avg(s, T_sleep, T_guard) / s->P_on;
}

float sm_b_eff(const sm_sensor_t *s, float T_sleep, float T_guard)
{
    float within = 0.5f * s->fs;                              /* Nyquist within a probe */
    float across = 1.0f / (2.0f * sm_t_cycle(s, T_sleep, T_guard)); /* across cycles */
    return within < across ? within : across;
}

bool sm_precondition_ok(const sm_sensor_t *s)
{
    return s->a_max_exp < s->range_limit;   /* C7 */
}

/* --- derived / diagnostic quantities --- */

float sm_duty(const sm_sensor_t *s, float T_sleep, float T_guard)         /* §4.4 */
{
    return (T_guard + s->T_probe) / sm_t_cycle(s, T_sleep, T_guard);
}

float sm_e_wake_gross(const sm_sensor_t *s, float T_guard)                /* §4.5 */
{
    return s->P_on * T_guard + s->E_inrush;
}

float sm_e_wake_net(const sm_sensor_t *s)                                 /* §4.5 */
{
    return s->E_inrush;   /* the guard energy cancels vs continuous-ON */
}

float sm_e_trust(const sm_sensor_t *s, float T_guard)                     /* §4.6 */
{
    return s->P_on * T_guard;   /* powered-but-untrusted energy */
}

float sm_dpavg_dtguard(const sm_sensor_t *s, float T_sleep, float T_guard) /* §4.6 */
{
    float Tc = sm_t_cycle(s, T_sleep, T_guard);
    return (s->P_on - s->P_sleep) * (T_sleep - sm_t_sleep_min(s)) / (Tc * Tc);
}

float sm_e_obs(const sm_sensor_t *s, float T_sleep, float T_guard)        /* §4.6 */
{
    float Ecycle = sm_p_avg(s, T_sleep, T_guard) * sm_t_cycle(s, T_sleep, T_guard);
    return s->T_probe > 0.0f ? Ecycle / s->T_probe : INFINITY;
}

float sm_e_rec(const sm_sensor_t *s, const sm_tol_t *t)                   /* §4.6 */
{
    float Tg = sm_t_guard_min(s, t);
    return isfinite(Tg) ? s->P_on * Tg : INFINITY;
}

float sm_t_sleep_max_info(const sm_sensor_t *s, const sm_tol_t *t, float T_guard) /* §6 */
{
    if (0.5f * s->fs < t->B_required) return 0.0f;       /* C6 unattainable */
    float by_bw   = 1.0f / (2.0f * t->B_required) - T_guard - s->T_probe;
    float upper   = t->D_max < by_bw ? t->D_max : by_bw; /* min(C3, C6) */
    return upper > 0.0f ? upper : 0.0f;
}

float sm_p_miss(const sm_sensor_t *s, float T_sleep, float T_guard, float d_e) /* §4.7 */
{
    /* illustrative single-duration, uniform-arrival approximation.
     * The realistic L_miss comes from the Track-3 simulator, not this form. */
    float blind = T_sleep - d_e;
    if (blind < 0.0f) blind = 0.0f;
    return blind / sm_t_cycle(s, T_sleep, T_guard);
}

float sm_cost_J(const sm_sensor_t *s, float T_sleep, float T_guard,
                float lambda, float L_miss)                              /* §8 */
{
    return sm_p_avg(s, T_sleep, T_guard) + lambda * L_miss;
}

/* wake-recovery gate (C2/C4/C5); margin=true tightens by (1-delta). */
static bool recovery_pass(const sm_sensor_t *s, const sm_tol_t *t,
                          float Tg, bool margin)
{
    float k     = margin ? (1.0f - t->delta) : 1.0f;
    float sfdev = s->SF - s->SF_ref;
    if (sfdev < 0.0f) sfdev = -sfdev;

    bool c2 = sm_bias_after(s, Tg) + sm_temp_bias(s) + s->b_slow <= k * t->eps_allowed + EPS;
    bool c4 = s->sigma_pw          <= k * t->sigma_allowed + EPS;
    bool c5 = sfdev                <= k * t->SF_allowed     + EPS;
    return c2 && c4 && c5;
}

bool sm_green_interval(const sm_sensor_t *s, const sm_tol_t *t,
                       float *lo, float *hi)
{
    if (!sm_precondition_ok(s)) return false;                 /* C7 */

    float Tg = sm_t_guard_min(s, t);
    if (!isfinite(Tg) || !recovery_pass(s, t, Tg, false))
        return false;                                         /* per-method gate */

    /* lower: smallest T_sleep with S >= S_min (closed form; S_min=0 -> T_sleep_min) */
    float denom = (1.0f - t->S_min) * s->P_on - s->P_sleep;
    if (denom <= 0.0f) return false;
    float Tlo = (s->E_inrush + t->S_min * s->P_on * (Tg + s->T_probe)) / denom;

    /* upper: min(D_max, T_sleep where B_eff == B_required) */
    if (0.5f * s->fs < t->B_required) return false;           /* C6 impossible at all */
    float Thi = t->D_max;
    float Thi_bw = 1.0f / (2.0f * t->B_required) - Tg - s->T_probe;
    if (Thi_bw < Thi) Thi = Thi_bw;

    if (Tlo > Thi) return false;
    *lo = Tlo;
    *hi = Thi;
    return true;
}

sm_class_t sm_classify(const sm_sensor_t *s, const sm_tol_t *t, float T_sleep)
{
    if (!sm_precondition_ok(s)) return SM_RED;                /* C7 */

    float Tg = sm_t_guard_min(s, t);
    if (!isfinite(Tg))                       return SM_RED;
    if (!recovery_pass(s, t, Tg, false))     return SM_RED;   /* C2/C4/C5 hard gate */

    float S    = sm_saving(s, T_sleep, Tg);
    float Beff = sm_b_eff (s, T_sleep, Tg);

    /* hard feasibility (boundary). C1's hard form is energy-positive (S >= 0);
     * S_min is C1's margin parameter (definition doc S4.3: "replace the S = 0
     * target with S = S_min"), so 0 <= S < S_min is YELLOW, not RED. */
    bool hard = (S    >= -1e-6f)                  /* C1 */
             && (T_sleep <= t->D_max   + 1e-6f)   /* C3 */
             && (Beff >= t->B_required - 1e-6f);  /* C6 */
    if (!hard) return SM_RED;

    /* GREEN if every constraint holds with margin */
    float k     = 1.0f - t->delta;
    bool  green = (S    >= t->S_min)                /* C1 margined by S_min itself */
               && (T_sleep <= k * t->D_max)         /* C3 */
               && (Beff >= t->B_required / k)       /* C6 */
               && recovery_pass(s, t, Tg, true);    /* C2/C4/C5 */
    return green ? SM_GREEN : SM_YELLOW;
}

const char *sm_class_name(sm_class_t c)
{
    switch (c) {
    case SM_GREEN:  return "GREEN";
    case SM_YELLOW: return "YELLOW";
    default:        return "RED";
    }
}
