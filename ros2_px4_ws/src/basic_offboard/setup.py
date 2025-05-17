from setuptools import find_packages, setup

package_name = 'basic_offboard'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name', # Change this
    maintainer_email='your_email@example.com', # Change this
    description='Minimal offboard setpoint publisher example', # Change this
    license='Apache License 2.0', # Or your preferred license
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Format: 'executable_name = package_name.module_name:main_function'
            'setpoint_publisher = basic_offboard.setpoint_publisher_node:main',
            'mission_controller = basic_offboard.mission_controller_node:main',
            'attitude_setpoint = basic_offboard.attitude_publisher_node:main',
            'attitude_replay = basic_offboard.attitude_replay_node:main',
        ],
    }
)