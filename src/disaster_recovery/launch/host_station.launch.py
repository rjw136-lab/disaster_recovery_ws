#!/usr/bin/env python3
"""
Launch file for Host Supervision Station.
Runs YOLO26n Vision AI, Faster-Whisper Audio AI, and 10Hz Heartbeat Watchdog.
Optionally opens RViz2 with the bundled disaster_station.rviz layout (use_rviz:=true).
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('disaster_recovery')
    rviz_config = os.path.join(pkg_share, 'config', 'disaster_station.rviz')
    semantic_map_params = os.path.join(pkg_share, 'config', 'semantic_map_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'yolo_model_path',
            default_value='yolo26n.pt',
            description='Path or name of the YOLO model weight (default: yolo26n.pt)'
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/camera/image_raw/compressed',
            description='Camera image topic name (e.g. /camera/image_raw or /camera/image_raw/compressed)'
        ),
        DeclareLaunchArgument(
            'use_compressed',
            default_value='true',
            description='Subscribe to CompressedImage'
        ),
        DeclareLaunchArgument(
            'show_window',
            default_value='true',
            description='Display OpenCV detection window'
        ),
        DeclareLaunchArgument(
            'heartbeat_rate_hz',
            default_value='10.0',
            description='Heartbeat transmission frequency in Hz'
        ),
        DeclareLaunchArgument(
            'confidence_threshold',
            default_value='0.40',
            description='Detection confidence threshold'
        ),
        DeclareLaunchArgument(
            'person_conf_threshold',
            default_value='0.55',
            description='Person detection confidence threshold to trigger survivor alert'
        ),
        DeclareLaunchArgument(
            'only_person_detection',
            default_value='false',
            description='Restrict YOLO detections strictly to person (COCO ID 0)'
        ),
        DeclareLaunchArgument(
            'min_consecutive_detections',
            default_value='2',
            description='Minimum consecutive frames of person detection to trigger survivor alert'
        ),
        DeclareLaunchArgument(
            'cuda_device',
            default_value='0',
            description='CUDA GPU device ID'
        ),
        DeclareLaunchArgument(
            'rotation_angle',
            default_value='270',
            description='Rotate camera image (0, 90, 180, 270)'
        ),
        DeclareLaunchArgument(
            'whisper_model_size',
            default_value='base',
            description='Faster-Whisper model size (tiny, base, small)'
        ),
        DeclareLaunchArgument(
            'whisper_device',
            default_value='cuda',
            description='Compute device for Faster-Whisper (cuda or cpu)'
        ),
        DeclareLaunchArgument(
            'whisper_compute_type',
            default_value='int8',
            description='Quantization compute type for Faster-Whisper (int8, int8_float32, float32)'
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='false',
            description='Also launch RViz2 with config/disaster_station.rviz'
        ),
        DeclareLaunchArgument(
            'enable_semantic_map',
            default_value='true',
            description='Launch real-time 2D disaster semantic mapping node (semantic_map_node)'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
        Node(
            package='disaster_recovery',
            executable='host_ai_pipeline',
            name='host_ai_pipeline',
            output='screen',
            parameters=[{
                'yolo_model_path': LaunchConfiguration('yolo_model_path'),
                'image_topic': LaunchConfiguration('image_topic'),
                'use_compressed': LaunchConfiguration('use_compressed'),
                'show_window': LaunchConfiguration('show_window'),
                'rotation_angle': LaunchConfiguration('rotation_angle'),
                'heartbeat_rate_hz': LaunchConfiguration('heartbeat_rate_hz'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                'person_conf_threshold': LaunchConfiguration('person_conf_threshold'),
                'only_person_detection': LaunchConfiguration('only_person_detection'),
                'min_consecutive_detections': LaunchConfiguration('min_consecutive_detections'),
                'cuda_device': LaunchConfiguration('cuda_device'),
                'whisper_model_size': LaunchConfiguration('whisper_model_size'),
                'whisper_device': LaunchConfiguration('whisper_device'),
                'whisper_compute_type': LaunchConfiguration('whisper_compute_type'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }],
            emulate_tty=True
        ),
        Node(
            package='disaster_recovery',
            executable='semantic_map_node',
            name='semantic_map_node',
            output='screen',
            parameters=[
                semantic_map_params,
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
            ],
            condition=IfCondition(LaunchConfiguration('enable_semantic_map')),
            emulate_tty=True
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='disaster_station_rviz',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_rviz'))
        )
    ])
