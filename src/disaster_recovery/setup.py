from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'disaster_recovery'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') + glob('config/*.rviz') + glob('config/*.sdf') + glob('config/*.onnx') + glob('config/*.pth')),
        (os.path.join('share', package_name, 'models', 'turtlebot3_burger'), glob('models/turtlebot3_burger/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jong Woo Ro',
    maintainer_email='rjw13@knu.ac.kr',
    description='Autonomous Disaster Recovery and Waypoint Backtracking System for TurtleBot3 Burger',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'recovery_controller = disaster_recovery.recovery_controller:main',
            'aruco_docking = disaster_recovery.aruco_docking:main',
            'host_ai_pipeline = disaster_recovery.host_ai_pipeline:main',
            'gas_sensor_node = disaster_recovery.gas_sensor_node:main',
            'audio_capture_node = disaster_recovery.audio_capture_node:main',
            'blackbox_recorder = disaster_recovery.blackbox_recorder:main',
        ],
    },
)
