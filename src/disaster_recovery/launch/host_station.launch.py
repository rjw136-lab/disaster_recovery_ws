#!/usr/bin/env python3
"""
Launch file for Host Supervision Station on Windows 11 WSL2.
Runs YOLO26/v8n Vision AI, Faster-Whisper Audio AI, and 10Hz Heartbeat Watchdog.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    host_ai_node = Node(
        package='disaster_recovery',
        executable='host_ai_pipeline',
        name='host_ai_pipeline',
        output='screen',
        parameters=[{
            'heartbeat_rate_hz': 10.0,
            'yolo_model_path': 'yolov8n.pt',
            'whisper_model_size': 'base',
            'confidence_threshold': 0.50,
            'cuda_device': 0
        }],
        emulate_tty=True
    )

    return LaunchDescription([
        host_ai_node
    ])
