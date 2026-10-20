#ifndef SAFEMAP_H
#define SAFEMAP_H

#include <stdbool.h>

/* Safe operating map -- portable, freestanding core (float-only, no alloc,
 * no I/O). Implements the Rev 2.2 equations from
 * docs/safe_operating_map_definition. This file compiles unchanged on the
 * host simulator AND on the STM32 firmware. */

typedef struct {
    /* energy primitives */
    float P_on;        /* W : ACTIVE power (= continuous-ON baseline) */
    float P_sleep;     /* W : sleep power for this method             */
    float E_inrush;    /* J : one-time turn-on surge                  */
    /* wake-recovery coefficients */
    float b0;          /* g : initial post-wake bias b(0)            */
    float tau;         /* s : bias recovery time constant            */
    float sigma_pw;    /* g : post-guard noise std (C4)              */
    float SF;          /* - : post-wake scale factor (C5)            */
    float SF_ref;      /* - : reference scale factor                 */
    /* bandwidth + dynamic-range inputs */
    float fs;          /* Hz: sample rate                            */
    float T_probe;     /* s : probe window                           */
    float a_max_exp;   /* g : expected max acceleration (C7)         */
    float range_limit; /* g : sensor full scale (C7)                 */
    /* temperature (C2 theta-axis) */
    float bias_tc;     /* g/C: static bias temp coefficient (signed) */
    float dT_op;       /* C : operating temp deviation from ref      */
    /* model form (Paper 1 revision, decision D6, 2026-10-16): a slow component of the recovery
     * that does not decay within the guard; a constant term of the C2 budget. 0 = the single
     * exponential, which every initializer that does not name it gets. */
    float b_slow;      /* g : slow-term budget term                  */
} sm_sensor_t;

typedef struct {
    float eps_allowed;   /* g : post-wake bias tolerance (C2)   */
    float sigma_allowed; /* g : post-wake noise tolerance (C4)  */
    float SF_allowed;    /* - : scale-factor dev tolerance (C5) */
    float B_required;    /* Hz: needed effective bandwidth (C6) */
    float D_max;         /* s : max blind duration (C3)         */
    float S_min;         /* - : min energy-saving fraction (C1) */
    float delta;         /* - : GREEN/YELLOW margin fraction    */
} sm_tol_t;

typedef enum { SM_RED = 0, SM_YELLOW = 1, SM_GREEN = 2 } sm_class_t;

/* --- primitives (the Rev 2.2 closed forms) --- */
float sm_t_sleep_min(const sm_sensor_t *s);                    /* §4.3 E_inrush/(P_on-P_sleep) */
float sm_t_guard_min(const sm_sensor_t *s, const sm_tol_t *t); /* §5.2 tau*ln(b0/((1-d)eps - |tc*dT| - b_slow)) */
float sm_bias_after (const sm_sensor_t *s, float T_guard);     /* §4.6 b0*exp(-Tg/tau)         */
float sm_temp_bias  (const sm_sensor_t *s);                    /* C2 static |bias_tc*dT_op|    */
float sm_b0_delayed (float b0, float tau, float delay_s);      /* b0*exp(delay/tau): guard + delay */
float sm_t_cycle    (const sm_sensor_t *s, float T_sleep, float T_guard);  /* §3 */
float sm_p_avg      (const sm_sensor_t *s, float T_sleep, float T_guard);  /* §3 */
float sm_saving     (const sm_sensor_t *s, float T_sleep, float T_guard);  /* §4.1 S */
float sm_b_eff      (const sm_sensor_t *s, float T_sleep, float T_guard);  /* §5/C6 */
bool  sm_precondition_ok(const sm_sensor_t *s);                /* §5.3 C7 */

/* --- derived / diagnostic quantities (the rest of §4, §6, §8) --- */
float sm_duty        (const sm_sensor_t *s, float T_sleep, float T_guard); /* §4.4 D */
float sm_e_wake_gross(const sm_sensor_t *s, float T_guard);                /* §4.5 E_wake      */
float sm_e_wake_net  (const sm_sensor_t *s);                               /* §4.5 dE = E_inrush */
float sm_e_trust     (const sm_sensor_t *s, float T_guard);                /* §4.6 trust tax   */
float sm_dpavg_dtguard(const sm_sensor_t *s, float T_sleep, float T_guard);/* §4.6 dPavg/dTg   */
float sm_e_obs       (const sm_sensor_t *s, float T_sleep, float T_guard); /* §4.6 E/probe-sec */
float sm_e_rec       (const sm_sensor_t *s, const sm_tol_t *t);            /* §4.6 P_on*Tg_min */
float sm_t_sleep_max_info(const sm_sensor_t *s, const sm_tol_t *t, float T_guard); /* §6 upper bound */
float sm_p_miss      (const sm_sensor_t *s, float T_sleep, float T_guard, float d_e); /* §4.7 illustrative */
float sm_cost_J      (const sm_sensor_t *s, float T_sleep, float T_guard,
                      float lambda, float L_miss);             /* §8 J = P_avg + lambda*L_miss */

/* Green interval [lo,hi] on the T_sleep axis (hard bounds, no margin).
 * Returns false if the method is infeasible (recovery gate fails, or the
 * energy-positive lower bound exceeds the blindness/bandwidth upper bound). */
bool  sm_green_interval(const sm_sensor_t *s, const sm_tol_t *t,
                        float *lo, float *hi);

/* Classify one operating point (method s, sleep duration T_sleep). */
sm_class_t  sm_classify(const sm_sensor_t *s, const sm_tol_t *t, float T_sleep);
const char *sm_class_name(sm_class_t c);

#endif /* SAFEMAP_H */
