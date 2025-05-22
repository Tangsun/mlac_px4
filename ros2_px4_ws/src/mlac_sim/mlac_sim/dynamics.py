"""
TODO description.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

# import jax
# import jax.numpy as jnp
import numpy as np
# from utils import hat, vee

# System constants
g_acc = 9.81    # gravitational acceleration
β = (0.1, 1.)   # drag coefficients


def prior(q, dq, g_acc=g_acc):
    """TODO: docstring."""
    nq = 3
    m = 1.5 # kg
    H = m*np.eye(nq)
    C = np.zeros((nq, nq))
    g = m*np.array([0., 0., g_acc])
    B = np.eye(nq)
    return H, C, g, B


def plant(q, dq, u, f_ext, prior=prior):
    """TODO: docstring."""
    H, C, g, B = prior(q, dq)
    ddq = np.linalg.solve(H, f_ext + B@u - C@dq - g)
    return ddq


def disturbance(q, dq, w, β=β):
    """TODO: docstring."""
    β = np.asarray(β)
    ϕ, dx, dy = q[2], dq[0], dq[1]
    sinϕ, cosϕ = np.sin(ϕ), np.cos(ϕ)
    R = np.array([
        [cosϕ, -sinϕ],
        [sinϕ,  cosϕ]
    ])
    v = R.T @ np.array([dx - w, dy])
    f_ext = - np.array([*(R @ (β * v * np.abs(v))), 0.])
    return f_ext

