#!/usr/bin/env python3
"""
Launch file for all Robot Onboard nodes on Raspberry Pi 4.
Runs fully autonomous recovery, safety watchdog, ArUco visual servoing, and gas telemetry.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('disaster_recovery')
    param_file = os.path.join(pkg_dir, 'config', 'recovery_params.yaml')

    recovery_controller_node = Node(
        package='disaster_recovery',
        executable='recovery_controller',
        name='recovery_controller',
        output='screen',
        parameters=[param_file],
        emulate_tty=True
    )

    aruco_docking_node = Node(
        package='disaster_recovery',
        executable='aruco_docking',
        name='aruco_docking',
        output='screen',
        parameters=[param_file],
        emulate_tty=True
    )

    # MQ-2 -> 10k/20k divider -> ADS1115 (I2C bus 1, 0x48) -> RPi 4
    # OpenCR keeps the stock TurtleBot3 firmware (motors + /odom) untouched.
    gas_sensor_node = Node(
        package='disaster_recovery',
        executable='gas_sensor_node',
        name='gas_sensor_node',
        output='screen',
        parameters=[param_file],
        emulate_tty=True
    )

    blackbox_node = Node(
        package='disaster_recovery',
        executable='blackbox_recorder',
        name='blackbox_recorder',
        output='screen',
        parameters=[{'log_base_dir': '/home/ubuntu/blackbox_logs'}],
        emulate_tty=True
    )

    # Generic AB13X USB mic (plughw:1,0, 48 kHz S16_LE mono) -> /audio/raw (UInt8MultiArray)
    audio_node = Node(
        package='disaster_recovery',
        executable='audio_capture_node',
        name='audio_capture_node',
        output='screen',
        parameters=[param_file],
        emulate_tty=True
    )

    return LaunchDescription([
        recovery_controller_node,
        aruco_docking_node,
        gas_sensor_node,
        blackbox_node,
        audio_node
    ])
