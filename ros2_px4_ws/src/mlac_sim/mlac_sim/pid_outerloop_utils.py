import numpy as np
import numba

# @numba.njit(cache=True)
# def get_force_jit(
#     state_p,
#     state_v,
#     goal_p,
#     goal_v,
#     goal_a,
#     maxPosErr,
#     maxVelErr,
#     mass,
#     gravity,
#     dt,
#     Kp,
#     Ki,
#     Kd,
#     eint_in
# ):
#     e = goal_p - state_p
#     edot = goal_v - state_v

#     # e_clamped_x = np.clip(e[0], -maxPosErr[0], maxPosErr[0])
#     # e_clamped_y = np.clip(e[1], -maxPosErr[1], maxPosErr[1])
#     # e_clamped_z = np.clip(e[2], -maxPosErr[2], maxPosErr[2])
#     e_clamped = np.clip(e, -maxPosErr, maxPosErr)

#     # edot_clamped_x = np.clip(edot[0], -maxVelErr[0], maxVelErr[0])
#     # edot_clamped_y = np.clip(edot[1], -maxVelErr[1], maxVelErr[1])
#     # edot_clamped_z = np.clip(edot[2], -maxVelErr[2], maxVelErr[2])
#     edot_clamped = np.clip(edot, -maxVelErr, maxVelErr)

#     ANTI_WINDUP_THRESHOLD = 0.2
#     eint_out = np.copy(eint_in)
#     if np.abs(e_clamped[0]) < ANTI_WINDUP_THRESHOLD: eint_out[0] += e_clamped[0] * dt
#     else: eint_out[0] = 0.0
#     if np.abs(e_clamped[1]) < ANTI_WINDUP_THRESHOLD: eint_out[1] += e_clamped[1] * dt
#     else: eint_out[1] = 0.0
#     if np.abs(e_clamped[2]) < ANTI_WINDUP_THRESHOLD: eint_out[2] += e_clamped[2] * dt
#     else: eint_out[2] = 0.0

#     a_fb = Kp * e_clamped + Ki * eint_out + Kd * edot_clamped
#     F_W = a_fb + mass * (goal_a - gravity)

#     return F_W, a_fb, e, edot, eint_out

# @numba.njit(cache=True)
# def get_attitude_jit():
#     pass

def convert_p_qbar(p):
    return np.sqrt(1/(1 - 1/p) - 1.1)

class IntegratorClass:
    def __init__(self):
        self.value_ = 0.0 

    def increment(self, inc, dt):
        self.value_ += inc * dt

    def reset(self):
        self.value_ = 0.0

    def value(self):
        return self.value_