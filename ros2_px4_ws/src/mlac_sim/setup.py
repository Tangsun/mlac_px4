from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'mlac_sim'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # Correctly install all files from the traj_data directory
        (os.path.join('share', package_name, 'traj_data'), glob(os.path.join('traj_data', '*.npy'))),
        
        # Correctly install all files from the models directory (recursive)
        (os.path.join('share', package_name, 'models'), glob(os.path.join('models', '**', '*'), recursive=True)),
        
        # Also include launch files (standard practice)
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools', 'numpy', 'scipy'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='MLAC Adaptive Controller Package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mlac_mission_node = mlac_sim.mlac_mission_node:main',
            'repub_odom_node = mlac_sim.repub_odom:main',
        ],
    },
)