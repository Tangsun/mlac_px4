import argparse
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np # For saving .npy file and some array operations

from utils import spline, random_ragged_spline

class Spline():
    def __init__(self, num_traj, T, dt, key, xmin_, ymin_, zmin_, xmax_, ymax_, zmax_,
                 z_offset=1.0,
                 min_xy_vel_for_yaw_calc=0.05,
                 initial_yaw_rad = 0.0
                 ):
        self.key = key
        self.xmin_ = xmin_
        self.ymin_ = ymin_
        self.zmin_ = zmin_
        self.xmax_ = xmax_
        self.ymax_ = ymax_
        self.zmax_ = zmax_
        self.z_offset_ = z_offset

        self.min_xy_vel_for_yaw_calc = min_xy_vel_for_yaw_calc
        self.initial_yaw_rad = initial_yaw_rad

        self.num_traj = num_traj
        self.T = T
        self.dt = dt
        
        num_knots_spatial = 6
        poly_orders_xyz = (9, 9, 9)
        deriv_orders_xyz = (4, 4, 4)
        min_step_xyz = jnp.array([-2, -2, -0.25])
        max_step_xyz = jnp.array([2, 2, 0.25])
        min_knot_xyz = jnp.array([self.xmin_, self.ymin_, self.zmin_ - self.z_offset_])
        max_knot_xyz = jnp.array([self.xmax_, self.ymax_, self.zmax_ - self.z_offset_])

        self.key, *subkeys = jax.random.split(self.key, 1 + self.num_traj)
        subkeys_stacked = jnp.vstack(subkeys)

        # vmap random_ragged_spline over the number of trajectories
        # This means self.t_knots_batch will have shape (num_traj, num_time_knots)
        # self.coefs_xyz_raw_batch will be a list of 3 arrays, each
        #   shape (num_traj, num_segments, poly_order+1)
        self.t_knots_batch, self.knots_xyz_raw_batch, self.coefs_xyz_raw_batch = jax.vmap(
            random_ragged_spline,
            in_axes=(0, None, None, None, None, None, None, None, None)
        )(
            subkeys_stacked, self.T, num_knots_spatial,
            poly_orders_xyz, deriv_orders_xyz,
            min_step_xyz, max_step_xyz, min_knot_xyz, max_knot_xyz
        )

    @partial(jax.jit, static_argnames=('self',))
    def _get_state_at_time_for_single_traj(self, t, t_knots_single, coefs_x, coefs_y, coefs_z_raw, prev_psi):
        """ Helper to get state for ONE trajectory at time t, given its specific t_knots and coefs. """
        def get_pos_vel_one_dim(time_val, tk, cf):
            pos = spline(time_val, tk, cf)
            vel = jax.grad(lambda tv: spline(tv, tk, cf))(time_val)
            return pos, vel

        x_pos, x_vel = get_pos_vel_one_dim(t, t_knots_single, coefs_x)
        y_pos, y_vel = get_pos_vel_one_dim(t, t_knots_single, coefs_y)
        z_raw_pos, z_vel = get_pos_vel_one_dim(t, t_knots_single, coefs_z_raw)

        z_pos = z_raw_pos + self.z_offset_
        x_pos_clipped = jnp.clip(x_pos, self.xmin_, self.xmax_)
        y_pos_clipped = jnp.clip(y_pos, self.ymin_, self.ymax_)
        z_pos_clipped = jnp.clip(z_pos, self.zmin_, self.zmax_)

        xy_vel_norm = jnp.sqrt(x_vel**2 + y_vel**2)
        calculated_psi = jnp.arctan2(y_vel, x_vel)
        
        psi = jax.lax.cond(
            xy_vel_norm < self.min_xy_vel_for_yaw_calc,
            lambda p_prev: p_prev,
            lambda _: calculated_psi,
            prev_psi
        )
        psi_normalized = (psi + jnp.pi) % (2 * jnp.pi) - jnp.pi

        return jnp.array([x_pos_clipped, y_pos_clipped, z_pos_clipped,
                          x_vel, y_vel, z_vel,
                          psi_normalized])

    def generate_all_trajectories_data_8col(self):
        """
        Generates the 8-column trajectory data for ALL trajectories generated in __init__.
        Output shape: (num_traj, num_time_steps, 8)
        """
        ts = jnp.arange(0, self.T + self.dt/2, self.dt) # Time vector, shape (num_time_steps,)

        # We need to apply the iterative psi calculation (scan) for EACH trajectory.
        # We can vmap the scan operation itself, or vmap the body of generate_trajectory_data_8col
        # from the previous version.

        # Let's define a function that processes one trajectory (given its unique knots and coeffs)
        # and then vmap this function over the batch of knots and coeffs.

        def process_one_trajectory(t_knots_single, x_coefs_single, y_coefs_single, z_coefs_raw_single):
            # Inner scan_body for this specific trajectory's data
            def scan_body_for_one_traj(carry_prev_psi, current_t):
                current_state_7_cols = self._get_state_at_time_for_single_traj(
                    current_t, t_knots_single,
                    x_coefs_single, y_coefs_single, z_coefs_raw_single,
                    carry_prev_psi
                )
                new_psi_for_carry = current_state_7_cols[-1]
                return new_psi_for_carry, current_state_7_cols

            # Initial carry for this trajectory's scan
            initial_pos_vel_psi_0 = self._get_state_at_time_for_single_traj(
                ts[0], t_knots_single,
                x_coefs_single, y_coefs_single, z_coefs_raw_single,
                self.initial_yaw_rad
            )
            initial_psi_carry = initial_pos_vel_psi_0[-1]

            _, pos_vel_psi_data_scan = jax.lax.scan(scan_body_for_one_traj, initial_psi_carry, ts)
            # pos_vel_psi_data_scan shape: (num_time_steps, 7)
            
            # Combine with time for this trajectory
            trajectory_array_single_jax = jnp.concatenate(
                (ts.reshape(-1, 1), pos_vel_psi_data_scan), axis=1
            ) # Shape: (num_time_steps, 8)
            return trajectory_array_single_jax

        # Vmap `process_one_trajectory` over the batch dimension of t_knots and coefficients.
        # self.t_knots_batch has shape (num_traj, num_time_knots)
        # self.coefs_xyz_raw_batch is a list of 3 arrays, each shape (num_traj, num_segments, poly_order+1)
        # We need to pass these batched inputs to the vmapped function.
        
        # Transpose coefs from list of [ (num_traj, ...), ... ] to [ (num_traj, ... x_coefs), (num_traj, ... y_coefs), ... ]
        # then we can vmap over the num_traj dimension.
        # No, random_ragged_spline already returns coefs_xyz_raw_batch as a list of [x_coefs_batch, y_coefs_batch, z_coefs_batch]
        # where x_coefs_batch is (num_traj, num_segments, poly_order_x+1) etc.

        all_trajectories_jax = jax.vmap(
            process_one_trajectory,
            in_axes=(0, 0, 0, 0) # Vmap over t_knots_batch, x_coefs_batch, y_coefs_batch, z_coefs_batch
        )(self.t_knots_batch, self.coefs_xyz_raw_batch[0], self.coefs_xyz_raw_batch[1], self.coefs_xyz_raw_batch[2])
        
        # all_trajectories_jax will have shape (num_traj, num_time_steps, 8)
        return np.asarray(all_trajectories_jax)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate multiple 3D spline trajectories and save as a single 8-column .npy file. "
                    "Relies on 'utils.py' for 'spline' and 'random_ragged_spline'."
    )
    parser.add_argument('--num_traj', type=int, default=50, # Changed default
                        help="Number of different random trajectories to generate and save in the batch.")
    parser.add_argument('--T_duration', type=float, default=30.0, help="Total duration of each trajectory (seconds).")
    parser.add_argument('--dt_step', type=float, default=0.02, help="Time step for sampling (seconds).")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for JAX PRNG.")
    
    parser.add_argument('--xmin', type=float, default=-2.0, help="Min X coordinate for trajectory knots.")
    parser.add_argument('--xmax', type=float, default=2.0, help="Max X coordinate for trajectory knots.")
    parser.add_argument('--ymin', type=float, default=-2.0, help="Min Y coordinate for trajectory knots.")
    parser.add_argument('--ymax', type=float, default=2.0, help="Max Y coordinate for trajectory knots.")
    parser.add_argument('--zmin', type=float, default=0.5, help="Min Z coordinate (final, after offset).")
    parser.add_argument('--zmax', type=float, default=3.5, help="Max Z coordinate (final, after offset).")
    parser.add_argument('--z_offset', type=float, default=2.0, help="Offset added to the raw Z-spline value.")
    parser.add_argument('--initial_yaw_deg', type=float, default=0.0, help="Initial yaw in degrees for t=0 if XY velocity is zero.")
    parser.add_argument('--min_xy_vel_for_yaw', type=float, default=0.05, help="Min XY-plane velocity (m/s) to calculate yaw from atan2; below this, previous yaw is held.")

    args = parser.parse_args()

    print(f"Generating {args.num_traj} trajectories: Duration={args.T_duration}s, dt={args.dt_step}s")

    initial_yaw_rad = np.deg2rad(args.initial_yaw_deg)

    spline_generator = Spline(
        num_traj=args.num_traj,
        T=args.T_duration,
        dt=args.dt_step,
        key=jax.random.PRNGKey(args.seed),
        xmin_=args.xmin, ymin_=args.ymin, zmin_=args.zmin,
        xmax_=args.xmax, ymax_=args.ymax, zmax_=args.zmax,
        z_offset=args.z_offset,
        min_xy_vel_for_yaw_calc=args.min_xy_vel_for_yaw,
        initial_yaw_rad=initial_yaw_rad
    )

    # Generate data for all trajectories
    batch_trajectory_data = spline_generator.generate_all_trajectories_data_8col()

    output_dir_default = '~/mlac_px4/ros2_px4_ws/src/mlac_sim/traj_data'
    filename = f"N{args.num_traj}_T{args.T_duration}_spline_8col.npy"

    output_dir = os.path.expanduser(output_dir_default)
    output_filepath = os.path.join(output_dir, filename)


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    np.save(output_filepath, batch_trajectory_data)
    print(f"Batch of {args.num_traj} trajectories saved to {output_filepath}")
    print(f"Shape of saved array: {batch_trajectory_data.shape}")
    if batch_trajectory_data.shape[0] > 0 and batch_trajectory_data.shape[1] > 0:
        print(f"First point of first trajectory (t,x,y,z,vx,vy,vz,psi):\n{batch_trajectory_data[0, 0, :]}")
        print(f"Last point of first trajectory:\n{batch_trajectory_data[0, -1, :]}")