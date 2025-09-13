import os
import numpy as np
import argparse
import re
import math

def generate_wind_velocity(seed, min_angle_deg=40.0, max_angle_deg=50.0, max_speed_mps=15.0, beta_a=2.0, beta_b=5.0):
    """
    Generates random wind velocity components based on specified constraints.

    Args:
        seed (int): Seed for the random number generator.
        min_angle_deg (float): Minimum wind angle in XY plane (degrees).
        max_angle_deg (float): Maximum wind angle in XY plane (degrees).
        max_speed_mps (float): Maximum wind speed in XY plane (m/s), used to scale beta distribution.
        beta_a (float): Shape parameter 'a' for the beta distribution.
        beta_b (float): Shape parameter 'b' for the beta distribution.

    Returns:
        tuple: (wind_x, wind_y, wind_z) components of the wind velocity.
    """
    np.random.seed(seed)
    
    # Sample angle in degrees, then convert to radians
    angle_deg = np.random.uniform(min_angle_deg, max_angle_deg)
    angle_rad = np.deg2rad(angle_deg)
    
    # Sample speed from beta distribution (0 to 1) and scale
    # Beta(a,b) gives values in [0,1]. Beta(2,5) peaks towards lower values.
    speed_xy_normalized = np.random.beta(beta_a, beta_b)
    speed_xy = speed_xy_normalized * max_speed_mps
    
    wind_x = speed_xy * np.cos(angle_rad)
    wind_y = speed_xy * np.sin(angle_rad)
    wind_z = 0.0  # Z component is always 0
    
    return wind_x, wind_y, wind_z

def modify_sdf_content(base_sdf_content, world_name_to_set, wind_x, wind_y, wind_z):
    """
    Modifies the base SDF content with the new world name and wind velocity.

    Args:
        base_sdf_content (str): The original SDF content.
        world_name_to_set (str): The new name for the world.
        wind_x (float): X component of wind velocity.
        wind_y (float): Y component of wind velocity.
        wind_z (float): Z component of wind velocity.

    Returns:
        str: The modified SDF content.
    """
    modified_sdf = base_sdf_content
    
    # 1. Modify world name
    # Use fr-string (raw f-string) with \g<group_number> for robust backreferences
    modified_sdf = re.sub(r"(<world name=')(\w+)(')", 
                          fr"\g<1>{world_name_to_set}\g<3>", 
                          modified_sdf, 
                          count=1) # Replace only the first occurrence
    
    # 2. Modify linear_velocity for wind
    wind_velocity_str = f"{wind_x:.6f} {wind_y:.6f} {wind_z:.6f}" # Format with 6 decimal places
    # Use fr-string (raw f-string) with \g<group_number> for robust backreferences
    modified_sdf = re.sub(r"(<linear_velocity>)(.*?)(</linear_velocity>)", 
                          fr"\g<1>{wind_velocity_str}\g<3>", 
                          modified_sdf, 
                          count=1) # Replace only the first occurrence under <wind>
    
    return modified_sdf

def main():
    parser = argparse.ArgumentParser(description="Generate a Gazebo SDF world file with custom world name and wind.")
    parser.add_argument('--world_name', type=str, default='windy_test', 
                        help="Name for the SDF world (will also be part of the filename if --output_filename is not set).")
    parser.add_argument('--output_filename', type=str, 
                        help="Output SDF filename (e.g., my_windy_world.sdf). If not provided, defaults to <world_name>.sdf.")
    parser.add_argument('--seed', type=int, default=42, 
                        help="Random seed for wind generation.")
    parser.add_argument('--min_angle', type=float, default=40.0, 
                        help="Minimum wind angle in XY plane (degrees).")
    parser.add_argument('--max_angle', type=float, default=50.0, 
                        help="Maximum wind angle in XY plane (degrees).")
    parser.add_argument('--max_speed', type=float, default=15.0, 
                        help="Maximum wind speed in XY plane (m/s), used to scale beta distribution.")
    parser.add_argument('--beta_a', type=float, default=4.0, 
                        help="Shape parameter 'a' for beta distribution of wind speed (e.g., 2.0).")
    parser.add_argument('--beta_b', type=float, default=5.0, 
                        help="Shape parameter 'b' for beta distribution of wind speed (e.g., 5.0).")
    parser.add_argument('--output_dir', type=str, 
                        default='~/mlac_px4/px4_src/PX4-Autopilot/Tools/simulation/gz/worlds/',
                        help="Directory to save the generated SDF file.")
    
    args = parser.parse_args()

    base_sdf_content = """<sdf version='1.9'>
  <world name='windy'>
    <physics type="ode">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>250</real_time_update_rate>
    </physics>
    <gravity>0 0 -9.8</gravity>
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    <atmosphere type='adiabatic'/>
    <scene>
      <grid>false</grid>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>true</shadows>
    </scene>
    <model name='ground_plane'>
      <static>true</static>
      <link name='link'>
        <collision name='collision'>
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>1 1</size>
            </plane>
          </geometry>
          <surface>
            <friction>
              <ode/>
            </friction>
            <bounce/>
            <contact/>
          </surface>
        </collision>
        <visual name='visual'>
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
            <specular>0.8 0.8 0.8 1</specular>
          </material>
        </visual>
        <pose>0 0 0 0 -0 0</pose>
        <inertial>
          <pose>0 0 0 0 -0 0</pose>
          <mass>1</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
        <enable_wind>true</enable_wind>
      </link>
      <pose>0 0 0 0 -0 0</pose>
      <self_collide>false</self_collide>
    </model>
    <light name='sunUTC' type='directional'>
      <pose>0 0 500 0 -0 0</pose>
      <cast_shadows>true</cast_shadows>
      <intensity>1</intensity>
      <direction>0.001 0.625 -0.78</direction>
      <diffuse>0.904 0.904 0.904 1</diffuse>
      <specular>0.271 0.271 0.271 1</specular>
      <attenuation>
        <range>2000</range>
        <linear>0</linear>
        <constant>1</constant>
        <quadratic>0</quadratic>
      </attenuation>
      <spot>
        <inner_angle>0</inner_angle>
        <outer_angle>0</outer_angle>
        <falloff>0</falloff>
      </spot>
    </light>
    <wind>
      <linear_velocity>5 2 0</linear_velocity>
    </wind>
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>47.397971057728974</latitude_deg>
      <longitude_deg> 8.546163739800146</longitude_deg>
      <elevation>0</elevation>
    </spherical_coordinates>
  </world>
</sdf>"""

    wind_x, wind_y, wind_z = generate_wind_velocity(
        args.seed, 
        args.min_angle, 
        args.max_angle, 
        args.max_speed,
        args.beta_a,
        args.beta_b
    )
    
    total_speed_xy = math.sqrt(wind_x**2 + wind_y**2)
    wind_angle_deg_actual = math.degrees(math.atan2(wind_y, wind_x))

    print(f"Seed used: {args.seed}")
    print(f"Generated wind velocity (X, Y, Z): {wind_x:.3f}, {wind_y:.3f}, {wind_z:.1f} m/s")
    print(f"Resulting XY plane speed: {total_speed_xy:.3f} m/s")
    print(f"Resulting XY plane angle: {wind_angle_deg_actual:.3f} degrees")
    
    modified_sdf_string = modify_sdf_content(base_sdf_content, args.world_name, wind_x, wind_y, wind_z)
    
    output_directory = os.path.expanduser(args.output_dir)
    
    if args.output_filename:
        output_filename_to_use = args.output_filename
        if not output_filename_to_use.endswith(".sdf"):
            output_filename_to_use += ".sdf"
    else:
        output_filename_to_use = f"{args.world_name}.sdf"
        
    output_filepath = os.path.join(output_directory, output_filename_to_use)
    
    try:
        os.makedirs(output_directory, exist_ok=True)
        with open(output_filepath, 'w') as f:
            f.write(modified_sdf_string)
        print(f"Successfully generated SDF file: {output_filepath}")
    except Exception as e:
        print(f"Error writing SDF file: {e}")

if __name__ == '__main__':
    main()
