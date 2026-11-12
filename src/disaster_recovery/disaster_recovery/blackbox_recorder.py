#!/usr/bin/env python3
"""
=================================================================================================
Project: Disaster Recovery TurtleBot3 Burger System
Node: blackbox_recorder.py
Author: Autonomous Systems Laboratory / Modified for Jong Woo Ro

Description:
  Onboard Local Blackbox Recorder.
  Triggered immediately upon network loss or emergency fault (/blackbox/trigger).
  Terminates unnecessary Wi-Fi outbound video streaming load and securely writes high-definition
  camera frames and synchronized sensor telemetry (/odom, /gas/level, /scan) to the local MicroSD.
=================================================================================================
"""

import os
import csv
import time
import math
import cv2
import numpy as np
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32
from cv_bridge import CvBridge, CvBridgeError


class BlackboxRecorderNode(Node):
    def __init__(self):
        super().__init__('blackbox_recorder')

        self.declare_parameter('log_base_dir', '/home/ubuntu/blackbox_logs')
        self.declare_parameter('video_fps', 20.0)
        self.declare_parameter('video_width', 640)
        self.declare_parameter('video_height', 480)

        self.base_dir = self.get_parameter('log_base_dir').value
        self.fps = self.get_parameter('video_fps').value
        self.width = self.get_parameter('video_width').value
        self.height = self.get_parameter('video_height').value

        self.is_recording = False
        self.bridge = CvBridge()
        self.video_writer = None
        self.csv_file = None
        self.csv_writer = None

        # Cached telemetry state
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_yaw = 0.0
        self.curr_lin_vel = 0.0
        self.curr_ang_vel = 0.0
        self.curr_gas = 0.0
        self.curr_min_scan = 99.0

        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.sub_trigger = self.create_subscription(
            Bool, '/blackbox/trigger', self.cb_trigger, qos_reliable
        )
        self.sub_image = self.create_subscription(
            Image, '/camera/image_raw', self.cb_image, qos_sub
        )
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self.cb_odom, qos_sub
        )
        self.sub_gas = self.create_subscription(
            Float32, '/gas/level', self.cb_gas, qos_reliable
        )
        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self.cb_scan, qos_sub
        )

        self.get_logger().info("Blackbox Recorder Node Standby.")

    def cb_trigger(self, msg: Bool):
        if msg.data and not self.is_recording:
            self.start_recording()

    def start_recording(self):
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = os.path.join(self.base_dir, f"session_{timestamp_str}")
        try:
            os.makedirs(target_dir, exist_ok=True)
        except Exception as e:
            self.get_logger().error(f"Failed to create blackbox log dir: {e}")
            target_dir = "/tmp"

        video_path = os.path.join(target_dir, "incident_video.mp4")
        csv_path = os.path.join(target_dir, "incident_telemetry.csv")

        # Initialize VideoWriter with MP4V codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, (self.width, self.height))

        # Initialize Telemetry CSV
        self.csv_file = open(csv_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "epoch_sec", "odom_x", "odom_y", "odom_yaw_deg",
            "linear_velocity", "angular_velocity", "gas_danger_level", "min_lidar_distance"
        ])

        self.is_recording = True
        self.get_logger().warn(f">>> [BLACKBOX ACTIVE] Logging video to {video_path} and data to {csv_path} <<<")

    def cb_odom(self, msg: Odometry):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        ori = msg.pose.pose.orientation
        siny_cosp = 2.0 * (ori.w * ori.z + ori.x * ori.y)
        cosy_cosp = 1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z)
        self.curr_yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
        self.curr_lin_vel = msg.twist.twist.linear.x
        self.curr_ang_vel = msg.twist.twist.angular.z

    def cb_gas(self, msg: Float32):
        self.curr_gas = msg.data

    def cb_scan(self, msg: LaserScan):
        valid = [r for r in msg.ranges if not (math.isnan(r) or math.isinf(r)) and r > 0.05]
        self.curr_min_scan = min(valid) if valid else 99.0

    def cb_image(self, msg: Image):
        if not self.is_recording:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

            # Add timestamp watermark
            cv2.putText(
                frame, f"BLACKBOX REC {datetime.now().strftime('%H:%M:%S.%f')[:-3]}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
            self.video_writer.write(frame)

            # Synchronous telemetry record
            self.csv_writer.writerow([
                f"{time.time():.4f}", f"{self.curr_x:.4f}", f"{self.curr_y:.4f}",
                f"{self.curr_yaw:.2f}", f"{self.curr_lin_vel:.3f}", f"{self.curr_ang_vel:.3f}",
                f"{self.curr_gas:.3f}", f"{self.curr_min_scan:.3f}"
            ])
            self.csv_file.flush()
        except Exception as e:
            self.get_logger().error(f"Blackbox write error: {e}")

    def destroy_node(self):
        if self.video_writer is not None:
            self.video_writer.release()
        if self.csv_file is not None:
            self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BlackboxRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
