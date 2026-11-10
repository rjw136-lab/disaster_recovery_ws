#!/usr/bin/env python3
"""
=================================================================================================
Project: Disaster Recovery TurtleBot3 Burger System
Node: aruco_docking.py
Author: Autonomous Systems Laboratory / Modified for Jongwoo Noh

Description:
  Precision Visual Docking Node using OpenCV solvePnP on ArUco Marker (ID: 0, 10x10 cm).
  Compensates for accumulated odometry slip/drift during dead-reckoning backtracking,
  guaranteeing landing within 15 cm of the base camp origin.

Key Algorithmic Steps:
  1. Activates only when triggered by /aruco/dock_trigger.
  2. Receives /camera/image_raw and reads intrinsic camera matrix (K) and distortion (D)
     from /camera/camera_info or local YAML configuration.
  3. Detects ArUco marker (DICT_4X4_50, Marker ID 0, length = 0.10 m).
  4. Computes 6-DoF pose (tvec: [tx, ty, tz], rvec: [rx, ry, rz]) using solvePnP with IPPE_SQUARE.
  5. Implements coupled P-Control:
     - Angular velocity (w): Steers to eliminate lateral offset (tx) and orientation error (yaw).
     - Linear velocity (v): Drives forward to achieve target stopping distance (tz = 0.20 m).
  6. Emits /aruco/dock_complete upon convergence (error < 2 cm, yaw < 3 deg) for >= 10 frames.

Camera Mounting Compensation:
  The Pi camera on this robot is mounted rolled 90 deg about its optical axis (the host station
  rotates the image by 270 deg to make it upright). solvePnP is run on the *unrotated* image so the
  calibrated intrinsics stay valid; the resulting pose is then re-expressed in the upright camera
  frame via `camera_rotation_deg` (0/90/180/270, same convention as host `rotation_angle`).
=================================================================================================
"""

import math
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool


class ArUcoDockingNode(Node):
    def __init__(self):
        super().__init__('aruco_docking')

        # -------------------------------------------------------------------------
        # Parameters
        # -------------------------------------------------------------------------
        self.declare_parameter('marker_id', 0)
        self.declare_parameter('marker_size_m', 0.10)          # 10 cm square
        self.declare_parameter('target_dock_dist_m', 0.20)     # Target forward distance
        self.declare_parameter('dist_tolerance_m', 0.02)       # 2 cm tolerance
        self.declare_parameter('lateral_tolerance_m', 0.02)    # 2 cm tolerance
        self.declare_parameter('yaw_tolerance_rad', 0.05)      # ~2.8 degrees
        self.declare_parameter('kp_linear', 0.40)
        self.declare_parameter('kp_lateral', 1.20)
        self.declare_parameter('kp_yaw', 0.80)
        self.declare_parameter('max_linear_speed', 0.08)       # Creep speed for precision
        self.declare_parameter('max_angular_speed', 0.35)
        self.declare_parameter('search_angular_speed', 0.20)
        self.declare_parameter('search_timeout_sec', 10.0)     # Fallback timeout if marker absent
        self.declare_parameter('camera_rotation_deg', 0)       # 0/90/180/270 (cv2.rotate convention)

        self.marker_id = self.get_parameter('marker_id').value
        self.marker_size = self.get_parameter('marker_size_m').value
        self.target_dist = self.get_parameter('target_dock_dist_m').value
        self.dist_tol = self.get_parameter('dist_tolerance_m').value
        self.lateral_tol = self.get_parameter('lateral_tolerance_m').value
        self.yaw_tol = self.get_parameter('yaw_tolerance_rad').value
        self.kp_lin = self.get_parameter('kp_linear').value
        self.kp_lat = self.get_parameter('kp_lateral').value
        self.kp_yaw = self.get_parameter('kp_yaw').value
        self.v_max = self.get_parameter('max_linear_speed').value
        self.w_max = self.get_parameter('max_angular_speed').value
        self.w_search = self.get_parameter('search_angular_speed').value
        self.search_timeout = float(self.get_parameter('search_timeout_sec').value)
        self.camera_rotation = int(self.get_parameter('camera_rotation_deg').value) % 360

        # Rotation mapping raw-sensor camera coordinates -> upright camera coordinates.
        # 90 : cv2.ROTATE_90_CLOCKWISE        -> x_u = -y_s, y_u =  x_s
        # 180: cv2.ROTATE_180                 -> x_u = -x_s, y_u = -y_s
        # 270: cv2.ROTATE_90_COUNTERCLOCKWISE -> x_u =  y_s, y_u = -x_s
        rot_map = {
            0: np.eye(3),
            90: np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            180: np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),
            270: np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        }
        if self.camera_rotation not in rot_map:
            self.get_logger().warn(
                f"Unsupported camera_rotation_deg={self.camera_rotation}; falling back to 0.")
            self.camera_rotation = 0
        self.R_sensor_to_upright = rot_map[self.camera_rotation]

        # -------------------------------------------------------------------------
        # Camera Intrinsics State
        # -------------------------------------------------------------------------
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_calibrated = False
        self.bridge = CvBridge()

        # Fallback default intrinsics for Raspberry Pi Camera Module V2 (640x480)
        # Will be overwritten as soon as /camera/camera_info arrives
        self.fallback_camera_matrix = np.array([
            [530.0, 0.0, 320.0],
            [0.0, 530.0, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.fallback_dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        # -------------------------------------------------------------------------
        # ArUco Configuration
        # -------------------------------------------------------------------------
        # OpenCV 4.x ArUco Dictionary Setup
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            self.aruco_params = cv2.aruco.DetectorParameters()
            # Adaptive thresholding optimizations for indoor disaster conditions
            self.aruco_params.adaptiveThreshWinSizeMin = 3
            self.aruco_params.adaptiveThreshWinSizeMax = 23
            self.aruco_params.adaptiveThreshWinSizeStep = 10
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        except AttributeError:
            # Legacy OpenCV fallback
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.aruco_detector = None

        # 3D coordinates of marker corners in marker's local frame
        # Order: top-left, top-right, bottom-right, bottom-left
        half = self.marker_size / 2.0
        self.obj_points = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0]
        ], dtype=np.float32)

        # -------------------------------------------------------------------------
        # Operational State
        # -------------------------------------------------------------------------
        self.docking_active = False
        self.docking_complete = False
        self.dock_trigger_time = None
        self.consecutive_aligned_frames = 0
        self.required_aligned_frames = 10
        self.last_seen_time = self.get_clock().now()
        self.search_sweep_direction = 1.0

        # -------------------------------------------------------------------------
        # Subscriptions & Publishers
        # -------------------------------------------------------------------------
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.sub_cam_info = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.cb_camera_info, qos_reliable
        )
        self.sub_image = self.create_subscription(
            Image, '/camera/image_raw', self.cb_image, qos_best_effort
        )
        self.sub_trigger = self.create_subscription(
            Bool, '/aruco/dock_trigger', self.cb_trigger, qos_reliable
        )

        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_complete = self.create_publisher(Bool, '/aruco/dock_complete', 10)
        self.pub_debug_img = self.create_publisher(Image, '/aruco/debug_image', 1)

        self.get_logger().info("ArUco Precision Docking Node Ready (Standing by for trigger).")

    # -----------------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------------
    def cb_trigger(self, msg: Bool):
        if msg.data and not self.docking_complete:
            self.docking_active = True
            self.dock_trigger_time = self.get_clock().now()
            self.get_logger().info(">>> Received Docking Trigger! Beginning Visual Servoing. <<<")

    def cb_camera_info(self, msg: CameraInfo):
        if not self.camera_calibrated:
            self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d, dtype=np.float64)
            self.camera_calibrated = True
            self.get_logger().info("Camera Intrinsics Successfully Loaded from /camera/camera_info.")

    def cb_image(self, msg: Image):
        if not self.docking_active or self.docking_complete:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        cam_mat = self.camera_matrix if self.camera_calibrated else self.fallback_camera_matrix
        dist_coeff = self.dist_coeffs if self.camera_calibrated else self.fallback_dist_coeffs

        # Marker Detection
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        if self.aruco_detector is not None:
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        target_marker_found = False
        target_idx = -1
        info_text = None

        if ids is not None:
            for i in range(len(ids)):
                if ids[i][0] == self.marker_id:
                    target_marker_found = True
                    target_idx = i
                    break

        if target_marker_found:
            self.last_seen_time = self.get_clock().now()
            corner = corners[target_idx][0]  # Shape: (4, 2)

            # SolvePnP for Marker Pose Estimation
            # Note: Using SOLVEPNP_IPPE_SQUARE designed specifically for planar markers
            success, rvec, tvec = cv2.solvePnP(
                self.obj_points,
                corner.astype(np.float32),
                cam_mat,
                dist_coeff,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if success:
                # Re-express pose in the upright camera frame (compensates sideways mounting)
                t_up = self.R_sensor_to_upright @ np.asarray(tvec, dtype=np.float64).reshape(3)
                tx = float(t_up[0])
                tz = float(t_up[2])

                # Compute Euler angles from rotation vector
                rot_sensor, _ = cv2.Rodrigues(rvec)
                rot_matrix = self.R_sensor_to_upright @ rot_sensor
                # Yaw around camera Y-axis: atan2(-R[0,2], R[2,2]) or asin(R[0,2])
                yaw = math.atan2(rot_matrix[0, 2], rot_matrix[2, 2])

                self.execute_docking_control(tx, tz, yaw)

                # Draw Visual Debug Overlays
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                cv2.drawFrameAxes(cv_image, cam_mat, dist_coeff, rvec, tvec, 0.05)
                info_text = f"X:{tx:.3f}m Z:{tz:.3f}m Yaw:{math.degrees(yaw):.1f}deg"
        else:
            # Marker not found in frame: Check active search timeout
            if self.dock_trigger_time is not None:
                elapsed_search = (self.get_clock().now() - self.dock_trigger_time).nanoseconds / 1e9
                if elapsed_search >= self.search_timeout:
                    self.get_logger().warn(
                        f"[ARUCO TIMEOUT] ArUco marker ID {self.marker_id} not detected within {elapsed_search:.1f}s. "
                        f"Finalizing mission at current pose (Fallback base camp arrival)."
                    )
                    self.stop_robot()
                    self.docking_complete = True
                    self.docking_active = False

                    complete_msg = Bool()
                    complete_msg.data = True
                    self.pub_complete.publish(complete_msg)
                    return

            # Perform active search sweep
            self.execute_search_sweep()

        # Publish debug visualization image (rotated upright for human viewing)
        if self.camera_rotation == 90:
            cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
        elif self.camera_rotation == 180:
            cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        elif self.camera_rotation == 270:
            cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if info_text:
            cv2.putText(cv_image, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            self.pub_debug_img.publish(debug_msg)
        except CvBridgeError:
            pass

    # -----------------------------------------------------------------------------
    # Visual Servoing Control
    # -----------------------------------------------------------------------------
    def execute_docking_control(self, tx: float, tz: float, yaw: float):
        """
        Closed-loop P-Controller for camera-to-marker alignment.
        """
        dist_err = tz - self.target_dist
        lateral_err = tx
        yaw_err = yaw

        # Convergence Check
        if (abs(dist_err) <= self.dist_tol and
                abs(lateral_err) <= self.lateral_tol and
                abs(yaw_err) <= self.yaw_tol):
            self.consecutive_aligned_frames += 1
            if self.consecutive_aligned_frames >= self.required_aligned_frames:
                self.stop_robot()
                self.docking_complete = True
                self.docking_active = False

                complete_msg = Bool()
                complete_msg.data = True
                self.pub_complete.publish(complete_msg)
                self.get_logger().info(
                    f"[DOCKING COMPLETE] Final Offset: Dist={dist_err*100:.1f}cm, "
                    f"Lat={lateral_err*100:.1f}cm, Yaw={math.degrees(yaw_err):.1f}°"
                )
                return
        else:
            self.consecutive_aligned_frames = max(0, self.consecutive_aligned_frames - 1)

        # Coupled P-Control velocities
        # Angular: counteract lateral shift (tx) and angle difference (yaw)
        w_cmd = -(self.kp_lat * lateral_err) - (self.kp_yaw * yaw_err)
        w_cmd = max(-self.w_max, min(self.w_max, w_cmd))

        # Linear: move towards target distance
        v_cmd = self.kp_lin * dist_err
        # Only move forward if heading is reasonably aligned
        if abs(lateral_err) > 0.08 or abs(yaw_err) > 0.30:
            v_cmd = 0.0  # Align orientation first
        else:
            v_cmd = max(-0.02, min(self.v_max, v_cmd))

        cmd = Twist()
        cmd.linear.x = float(v_cmd)
        cmd.angular.z = float(w_cmd)
        self.pub_cmd_vel.publish(cmd)

    def execute_search_sweep(self):
        """Slowly rotate in place to locate ArUco marker if temporarily occluded."""
        self.consecutive_aligned_frames = 0
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = self.search_sweep_direction * self.w_search
        self.pub_cmd_vel.publish(cmd)

    def stop_robot(self):
        cmd = Twist()
        self.pub_cmd_vel.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = ArUcoDockingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("ArUco Docking stopping due to KeyboardInterrupt.")
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
