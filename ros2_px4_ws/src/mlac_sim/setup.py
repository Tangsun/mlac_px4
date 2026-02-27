from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'mlac_sim'

# Helper function to recursively collect files and maintain directory structure for data_files
def package_files(directory_src, directory_dest_prefix):
    """
    Recursively collects files from directory_src and prepares them for data_files.
    Maintains the subdirectory structure relative to directory_src under directory_dest_prefix.
    """
    paths = []
    # Ensure the source directory exists to avoid errors if it's missing
    if not os.path.isdir(directory_src):
        return []
        
    for (path, directories, filenames) in os.walk(directory_src):
        for filename in filenames:
            # Full source path of the file
            source_path = os.path.join(path, filename)
            
            # Calculate the relative path from the base source directory
            # e.g., if directory_src is 'models' and path is 'models/sim_test',
            # then os.path.relpath(path, directory_src) gives 'sim_test'.
            # If path is 'models', relpath is '.'
            relative_subdir = os.path.relpath(path, directory_src)
            
            # Construct the destination directory
            if relative_subdir == '.': # Files directly in directory_src
                dest_dir = directory_dest_prefix
            else:
                dest_dir = os.path.join(directory_dest_prefix, relative_subdir)
            
            paths.append((dest_dir, [source_path]))
    return paths

# --- Construct the data_files list ---
# Static files
data_files_list = [
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    
    # Install all files from the traj_data directory (assuming flat structure needed here)
    (os.path.join('share', package_name, 'traj_data'), glob(os.path.join('traj_data', '*.npy'))),
    
    # Include launch files
    (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
]

# Dynamically add model files, preserving their directory structure
# This will install files like 'models/sim_test/test.pkl' to
# 'share/mlac_sim/models/sim_test/test.pkl'
model_files_list = package_files('models', os.path.join('share', package_name, 'models'))
onboard_model_files_list = package_files('onboard_models', os.path.join('share', package_name, 'onboard_models'))

# Combine all data_files entries
all_data_files = data_files_list + model_files_list + onboard_model_files_list

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=all_data_files, # Use the combined list
    install_requires=['setuptools', 'numpy', 'scipy'],
    zip_safe=True,
    maintainer='Your Name', #
    maintainer_email='your_email@example.com', #
    description='MLAC Adaptive Controller Package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mlac_mission_node = mlac_sim.mlac_mission_node:main',
            'repub_odom_node = mlac_sim.repub_odom:main',
            'yaw_sanity_check = mlac_sim.yaw_rotation_test:main',
            'bodyrate_sanity_check = mlac_sim.bodyrate_sanity_test:main',
            'frame_check = mlac_sim.frame_check_node:main',
        ],
    },
)