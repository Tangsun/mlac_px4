import numpy as np
from scipy.spatial.transform import Rotation


def euler_to_rotation_matrix(rpy):
    roll, pitch, yaw = rpy
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    return R_z @ R_y @ R_x


def simulation_ode_euler(state, commands, mass, g_acc=9.81, hov_thrust=0.727):
    """
    Flying inverted pendulum dynamics with Euler angle state.
    state: [pos(3), vel(3), rpy(3)]
    commands: (thrust_norm, body_rates)
    """
    thrust_norm, omega_cmd = commands
    pos = state[0:3]
    vel = state[3:6]
    rpy = state[6:9]

    R = euler_to_rotation_matrix(rpy)
    f_d = thrust_norm * g_acc / hov_thrust
    acc = R @ np.array([0.0, 0.0, f_d]) - np.array([0.0, 0.0, g_acc])

    roll, pitch = rpy[0], rpy[1]
    E = np.array([
        [1.0,  0.0,            -np.sin(pitch)                  ],
        [0.0,  np.cos(roll),    np.cos(pitch) * np.sin(roll)   ],
        [0.0, -np.sin(roll),    np.cos(pitch) * np.cos(roll)   ],
    ])
    rpy_dot = np.linalg.solve(E, omega_cmd)

    return np.concatenate([vel, acc, rpy_dot])


def rk4_step(dynamics_fn, state, dt, *args, **kwargs):
    k1 = dynamics_fn(state, *args, **kwargs)
    k2 = dynamics_fn(state + 0.5 * dt * k1, *args, **kwargs)
    k3 = dynamics_fn(state + 0.5 * dt * k2, *args, **kwargs)
    k4 = dynamics_fn(state + dt * k3, *args, **kwargs)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
