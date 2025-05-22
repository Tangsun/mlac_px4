from setuptools import find_packages, setup
import os
from glob import glob

# import sys
# # --- START HACK to force venv python for shebang ---
# # Check if we are in a virtual environment
# virtual_env_path = os.environ.get('VIRTUAL_ENV')
# if virtual_env_path:
#     # Construct the path to the python executable in the virtual environment
#     venv_python_executable = os.path.join(virtual_env_path, 'bin', 'python3')
#     # If the current executable is not the venv one (e.g., colcon is using system python to run setup.py)
#     # and the venv python exists, try to make setuptools use it.
#     if sys.executable != venv_python_executable and os.path.exists(venv_python_executable):
#         # This is a way to tell setuptools to use a specific executable for the scripts
#         # It's not a standard 'distutils' option, but setuptools might pick it up
#         # or we might need to modify how entry_points are generated.
#         # A more robust way would be to ensure colcon calls setup.py with the venv python.
#         # For now, let's just print a strong warning if this situation is detected.
#         print(f"WARNING: setup.py is being run by {sys.executable}, "
#               f"but VIRTUAL_ENV is {virtual_env_path}. Shebangs might be incorrect.")
#         print(f"Attempting to ensure shebang points to: {venv_python_executable}")
#         # One potential, more direct (but still hacky) way:
#         # If 'setuptools.command.install_scripts' is used, we could try to patch its behavior.
#         # However, for 'console_scripts' entry points, setuptools usually uses `sys.executable`.
#         # The most reliable fix is ensuring colcon uses the venv's python to *invoke* setup.py.
# # --- END HACK ---

from setuptools import find_packages, setup

package_name = 'mlac_sim'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install trajectory files
        (os.path.join('share', package_name, 'traj_data'), glob('traj_data/*.npy')),
        # Install model files for COML if they are in a 'models' directory
        (os.path.join('share', package_name, 'models'), glob('models/**/*.*', recursive=True)),
    ],
    install_requires=['setuptools', 'numpy', 'scipy'], # Add scipy if used by utils/dynamics indirectly
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='MLAC Adaptive Controller Package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mlac_mission_node = mlac_sim.mlac_mission_node:main', # Changed executable name here
            # If the filename is also mlac_mission_node.py, then:
            # 'mlac_mission_node = mlac_sim.mlac_mission_node:main',
        ],
    },
)