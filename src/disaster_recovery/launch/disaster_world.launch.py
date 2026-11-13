#!/usr/bin/env python3
"""
=================================================================================================
Project: Disaster Recovery TurtleBot3 Burger System
Launch : disaster_world.launch.py
Author : Autonomous Systems Laboratory / Jong Woo Ro
Usage  : ros2 launch disaster_recovery disaster_world.launch.py world:=<path_to_world>
=================================================================================================
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world = LaunchConfiguration('world')
    gui = LaunchConfiguration('gui', default='true')
    x_pose = LaunchConfiguration('x_pose', default='0.0')
    y_pose = LaunchConfiguration('y_pose', default='0.0')
    z_pose = LaunchConfiguration('z_pose', default='0.01')
    yaw_pose = LaunchConfiguration('yaw_pose', default='0.0')

    model = os.environ.get('TURTLEBOT3_MODEL', 'burger')
    local_sdf = os.path.join(get_package_share_directory('disaster_recovery'), 'models', f'turtlebot3_{model}', 'model.sdf')
    src_sdf = os.path.join('/home/lab/disaster_recovery_ws/src/disaster_recovery/models', f'turtlebot3_{model}', 'model.sdf')
    if os.path.exists(local_sdf):
        urdf_path = local_sdf
    elif os.path.exists(src_sdf):
        urdf_path = src_sdf
    else:
        urdf_path = os.path.join(pkg_tb3_gazebo, 'models', f'turtlebot3_{model}', 'model.sdf')

    declare_world_cmd = DeclareLaunchArgument('world', description='Full path to Gazebo world file')
    declare_gui_cmd = DeclareLaunchArgument('gui', default_value='true', description='Set to false for headless')
    declare_x_cmd = DeclareLaunchArgument('x_pose', default_value='0.0', description='Initial X position')
    declare_y_cmd = DeclareLaunchArgument('y_pose', default_value='0.0', description='Initial Y position')
    declare_z_cmd = DeclareLaunchArgument('z_pose', default_value='0.01', description='Initial Z position')
    declare_yaw_cmd = DeclareLaunchArgument('yaw_pose', default_value='0.0', description='Initial Yaw angle in radians')

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')),
        launch_arguments={'world': world}.items()
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')),
        condition=IfCondition(gui)
    )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_tb3_gazebo, 'launch', 'robot_state_publisher.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    spawn_entity_cmd = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', model,
            '-file', urdf_path,
            '-x', x_pose,
            '-y', y_pose,
            '-z', z_pose,
            '-Y', yaw_pose,
            '-timeout', '120.0'
        ],
        output='screen'
    )

    ld = LaunchDescription()
    ld.add_action(declare_world_cmd)
    ld.add_action(declare_gui_cmd)
    ld.add_action(declare_x_cmd)
    ld.add_action(declare_y_cmd)
    ld.add_action(declare_z_cmd)
    ld.add_action(declare_yaw_cmd)
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_entity_cmd)
    return ld
