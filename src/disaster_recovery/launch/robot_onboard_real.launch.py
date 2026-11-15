#!/usr/bin/env python3
"""
Launch file for physical TurtleBot3 Burger onboard nodes (Raspberry Pi 4).
Uses conservative Phase 2-B Track B profile (`recovery_params_real.yaml`) with
`real_tb3_policy_student.onnx`, stiction deadzone compensation, and 5 cm ArUco docking.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('disaster_recovery')
    default_param_file = os.path.join(pkg_dir, 'config', 'recovery_params_real.yaml')
    default_blackbox_dir = os.path.join(os.path.expanduser('~'), 'blackbox_logs')

    param_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_param_file,
        description='Full path to physical-robot ROS2 parameters YAML file.',
    )
    param_file = LaunchConfiguration('params_file')

    recovery_controller_node = Node(
        package='disaster_recovery',
        executable='recovery_controller',
        name='recovery_controller',
        output='screen',
        parameters=[param_file],
        emulate_tty=True,
    )

    aruco_docking_node = Node(
        package='disaster_recovery',
        executable='aruco_docking',
        name='aruco_docking',
        output='screen',
        parameters=[param_file],
        emulate_tty=True,
    )

    gas_sensor_node = Node(
        package='disaster_recovery',
        executable='gas_sensor_node',
        name='gas_sensor_node',
        output='screen',
        parameters=[param_file],
        emulate_tty=True,
    )

    blackbox_node = Node(
        package='disaster_recovery',
        executable='blackbox_recorder',
        name='blackbox_recorder',
        output='screen',
        parameters=[{'log_base_dir': default_blackbox_dir}],
        emulate_tty=True,
    )

    audio_node = Node(
        package='disaster_recovery',
        executable='audio_capture_node',
        name='audio_capture_node',
        output='screen',
        parameters=[param_file],
        emulate_tty=True,
    )

    return LaunchDescription([
        param_file_arg,
        recovery_controller_node,
        aruco_docking_node,
        gas_sensor_node,
        blackbox_node,
        audio_node,
    ])
