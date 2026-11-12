#!/usr/bin/env python3
"""
=================================================================================================
Project: Disaster Recovery TurtleBot3 Burger System
Node: recovery_controller.py
Author: Autonomous Systems Laboratory / Modified for Jong Woo Ro

Description:
  Unified Fail-Safe, Odometry Logger, Watchdog, Obstacle Avoidance, and Autonomous Waypoint
  Backtracking State Machine for TurtleBot3 Burger. Operates 100% onboard Raspberry Pi 4.

Key Behaviors:
  1. Normal Exploration Mode:
     - Relays remote teleop commands (/cmd_vel_remote) to motors (/cmd_vel).
     - Monitors /odom and logs waypoints to a LIFO stack when:
       * Cumulative distance delta >= 0.3 m OR
       * Cumulative heading delta >= 25.0 deg (0.4363 rad).
  2. Safety Watchdog:
     - Monitors /heartbeat timeout (> 3.0 sec).
     - Monitors /gas/level for hazardous combustible gas concentration (threshold >= 0.40).
     - Upon fault detection:
       * Emits emergency stop (v=0, w=0).
       * Cuts off remote teleoperation.
       * Signals onboard blackbox recorder to initiate local sensor/video logging.
       * Transitions to AUTONOMOUS_BACKTRACKING mode.
  3. Autonomous Backtracking:
     - Pops waypoints sequentially from LIFO stack.
     - Implements Method A: Rotate-then-Translate 2-phase trajectory following.
     - Subscribes to LDS-01 /scan: Halts immediately if obstacles detected within +/-20 deg at 0.12~0.25m.
     - Upon reaching proximity of initial start point (< 0.50 m or empty stack), handoffs control
       to the ArUco precision docking node.
=================================================================================================
"""

import math
import collections
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header, Float32, Bool, String


class SystemState:
    INIT = 'INIT'
    NORMAL_TELEOP = 'NORMAL_TELEOP'
    EMERGENCY_STOP = 'EMERGENCY_STOP'
    BACKTRACKING_ROTATE = 'BACKTRACKING_ROTATE'
    BACKTRACKING_TRANSLATE = 'BACKTRACKING_TRANSLATE'
    OBSTACLE_HOLD = 'OBSTACLE_HOLD'
    HANDOVER_TO_ARUCO = 'HANDOVER_TO_ARUCO'
    ARUCO_DOCKING = 'ARUCO_DOCKING'
    MISSION_COMPLETE = 'MISSION_COMPLETE'


class Waypoint:
    def __init__(self, x: float, y: float, theta: float, scan_pts=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.scan_pts = scan_pts  # Nx2 numpy array of Cartesian laser points in robot frame

    def distance_to(self, other_x: float, other_y: float) -> float:
        return math.hypot(self.x - other_x, self.y - other_y)


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract Euler yaw angle from quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def icp_match_2d(src_pts, dst_pts, initial_guess=(0.0, 0.0, 0.0), max_iters=10, inlier_thresh=0.35, min_inliers=15):
    """
    Fast 2D Iterative Closest Point algorithm.
    Finds rigid transformation (dx, dy, dtheta) that aligns src_pts onto dst_pts.
    """
    if src_pts is None or dst_pts is None:
        return 0.0, 0.0, 0.0, False
    if len(src_pts) < min_inliers or len(dst_pts) < min_inliers:
        return 0.0, 0.0, 0.0, False

    dx0, dy0, dth0 = initial_guess
    c0, s0 = math.cos(dth0), math.sin(dth0)
    R = np.array([[c0, -s0], [s0, c0]], dtype=np.float32)
    t = np.array([dx0, dy0], dtype=np.float32)
    cur_pts = (src_pts @ R.T) + t

    for _ in range(max_iters):
        diff = cur_pts[:, np.newaxis, :] - dst_pts[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        indices = np.argmin(dist_sq, axis=1)
        min_dists = np.sqrt(dist_sq[np.arange(len(cur_pts)), indices])

        inliers = min_dists < inlier_thresh
        num_inliers = int(np.sum(inliers))
        if num_inliers < min_inliers:
            break

        p_src = cur_pts[inliers]
        p_dst = dst_pts[indices[inliers]]

        c_src = np.mean(p_src, axis=0)
        c_dst = np.mean(p_dst, axis=0)

        H = (p_src - c_src).T @ (p_dst - c_dst)
        U, S, Vt = np.linalg.svd(H)
        R_step = Vt.T @ U.T
        if np.linalg.det(R_step) < 0:
            Vt[-1, :] *= -1
            R_step = Vt.T @ U.T
        t_step = c_dst - (c_src @ R_step.T)

        cur_pts = (cur_pts @ R_step.T) + t_step
        R = R_step @ R
        t = (t @ R_step.T) + t_step

        if np.linalg.norm(t_step) < 1e-4 and abs(np.arctan2(R_step[1, 0], R_step[0, 0])) < 1e-4:
            break

    theta = float(np.arctan2(R[1, 0], R[0, 0]))
    diff = cur_pts[:, np.newaxis, :] - dst_pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    min_dists = np.sqrt(np.min(dist_sq, axis=1))
    inliers = min_dists < inlier_thresh
    mean_err = float(np.mean(min_dists[inliers])) if np.sum(inliers) >= min_inliers else 99.0

    success = (np.sum(inliers) >= min_inliers) and (mean_err < 0.15)
    return float(t[0]), float(t[1]), theta, success


class RecoveryController(Node):
    def __init__(self):
        super().__init__('recovery_controller')

        # -------------------------------------------------------------------------
        # ROS 2 Parameters Configuration
        # -------------------------------------------------------------------------
        self.declare_parameter('heartbeat_timeout_sec', 3.0)
        self.declare_parameter('gas_threshold', 0.40)
        self.declare_parameter('gas_stale_timeout_sec', 2.0)
        self.declare_parameter('gas_stale_triggers_recovery', False)
        self.declare_parameter('log_dist_threshold_m', 0.30)
        self.declare_parameter('log_angle_threshold_deg', 25.0)
        self.declare_parameter('waypoint_reach_radius_m', 0.08)
        self.declare_parameter('heading_tolerance_rad', 0.08)  # ~4.6 degrees
        self.declare_parameter('linear_speed_max', 0.12)       # Safe speed for TB3 Burger (m/s)
        self.declare_parameter('angular_speed_max', 0.45)      # Safe angular rate (rad/s)
        self.declare_parameter('kp_linear', 0.50)
        self.declare_parameter('kp_angular', 1.20)
        self.declare_parameter('lidar_min_range_m', 0.12)
        self.declare_parameter('lidar_max_danger_range_m', 0.25)
        self.declare_parameter('lidar_fov_half_deg', 20.0)
        self.declare_parameter('docking_start_distance_m', 0.50)
        self.declare_parameter('enable_scan_matching', True)
        self.declare_parameter('scan_match_weight', 0.40)
        self.declare_parameter('scan_match_max_dist_m', 0.40)
        self.declare_parameter('use_aruco_docking', True)
        self.declare_parameter('aruco_docking_timeout_sec', 12.0)

        # Retrieve parameters
        self.heartbeat_timeout = self.get_parameter('heartbeat_timeout_sec').value
        self.gas_threshold = self.get_parameter('gas_threshold').value
        self.gas_stale_timeout = self.get_parameter('gas_stale_timeout_sec').value
        self.gas_stale_triggers = self.get_parameter('gas_stale_triggers_recovery').value
        self.log_dist_th = self.get_parameter('log_dist_threshold_m').value
        self.log_angle_th = math.radians(self.get_parameter('log_angle_threshold_deg').value)
        self.wp_reach_radius = self.get_parameter('waypoint_reach_radius_m').value
        self.heading_tol = self.get_parameter('heading_tolerance_rad').value
        self.v_max = self.get_parameter('linear_speed_max').value
        self.w_max = self.get_parameter('angular_speed_max').value
        self.kp_lin = self.get_parameter('kp_linear').value
        self.kp_ang = self.get_parameter('kp_angular').value
        self.lidar_min_dist = self.get_parameter('lidar_min_range_m').value
        self.lidar_danger_dist = self.get_parameter('lidar_max_danger_range_m').value
        self.lidar_fov = math.radians(self.get_parameter('lidar_fov_half_deg').value)
        self.docking_dist_th = self.get_parameter('docking_start_distance_m').value
        self.enable_scan_matching = self.get_parameter('enable_scan_matching').value
        self.scan_match_weight = self.get_parameter('scan_match_weight').value
        self.scan_match_max_dist = self.get_parameter('scan_match_max_dist_m').value
        self.use_aruco_docking = self.get_parameter('use_aruco_docking').value
        self.aruco_timeout = self.get_parameter('aruco_docking_timeout_sec').value

        # -------------------------------------------------------------------------
        # Internal State Variables
        # -------------------------------------------------------------------------
        self.state = SystemState.INIT
        self.previous_state = SystemState.INIT

        # Robot Odometry State
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        self.odom_initialized = False

        # Raw Odometry and SE(2) Scan Matching Correction Offset
        self.raw_odom_x = 0.0
        self.raw_odom_y = 0.0
        self.raw_odom_theta = 0.0
        self.corr_x = 0.0
        self.corr_y = 0.0
        self.corr_th = 0.0

        # LiDAR Scan Matching State
        self.latest_scan_pts = None
        self.start_scan_pts = None
        self.scan_match_count = 0
        self.aruco_start_time = None

        # Waypoint Logging Stack (LIFO: Last-In, First-Out)
        self.waypoint_stack = collections.deque()
        self.last_logged_x = 0.0
        self.last_logged_y = 0.0
        self.last_logged_theta = 0.0
        self.start_pose = None

        # Watchdog Tracking
        self.last_heartbeat_time = self.get_clock().now()
        self.current_gas_level = 0.0
        self.last_gas_time = None  # None until gas_sensor_node finishes warm-up/baseline
        self.heartbeat_received_once = False
        self.fault_reason = ""

        # Sensor & Obstacle Flags
        self.obstacle_detected = False
        self.active_target_wp = None
        self.aruco_docking_complete = False

        # QoS Profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # -------------------------------------------------------------------------
        # Subscribers
        # -------------------------------------------------------------------------
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self.cb_odometry, sensor_qos
        )
        self.sub_heartbeat = self.create_subscription(
            Header, '/heartbeat', self.cb_heartbeat, reliable_qos
        )
        self.sub_gas = self.create_subscription(
            Float32, '/gas/level', self.cb_gas, reliable_qos
        )
        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self.cb_scan, sensor_qos
        )
        self.sub_remote_teleop = self.create_subscription(
            Twist, '/cmd_vel_remote', self.cb_remote_teleop, reliable_qos
        )
        self.sub_dock_complete = self.create_subscription(
            Bool, '/aruco/dock_complete', self.cb_dock_complete, reliable_qos
        )

        # -------------------------------------------------------------------------
        # Publishers
        # -------------------------------------------------------------------------
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_blackbox_trigger = self.create_publisher(Bool, '/blackbox/trigger', 10)
        self.pub_dock_trigger = self.create_publisher(Bool, '/aruco/dock_trigger', 10)
        self.pub_system_status = self.create_publisher(String, '/recovery/status', 10)

        # Main Control Loop Timer (10 Hz)
        self.timer = self.create_timer(0.10, self.control_loop)

        self.get_logger().info("Recovery Controller Node Initialized. Awaiting sensors and heartbeat...")

    # -----------------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------------
    def cb_odometry(self, msg: Odometry):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self.raw_odom_x = pos.x
        self.raw_odom_y = pos.y
        self.raw_odom_theta = quaternion_to_yaw(ori.x, ori.y, ori.z, ori.w)

        # Apply persistent SE(2) scan matching correction transform
        c_c = math.cos(self.corr_th)
        s_c = math.sin(self.corr_th)
        self.current_x = self.corr_x + self.raw_odom_x * c_c - self.raw_odom_y * s_c
        self.current_y = self.corr_y + self.raw_odom_x * s_c + self.raw_odom_y * c_c
        self.current_theta = normalize_angle(self.raw_odom_theta + self.corr_th)

        if not self.odom_initialized:
            self.start_pose = Waypoint(self.current_x, self.current_y, self.current_theta, scan_pts=self.latest_scan_pts)
            self.last_logged_x = self.current_x
            self.last_logged_y = self.current_y
            self.last_logged_theta = self.current_theta
            self.odom_initialized = True
            # Always ensure initial start pose is at the bottom of the stack
            self.waypoint_stack.append(self.start_pose)
            self.get_logger().info(f"Start Pose Anchor Registered at ({self.current_x:.3f}, {self.current_y:.3f})")

        # In NORMAL_TELEOP state, log waypoints according to delta thresholds
        if self.state == SystemState.NORMAL_TELEOP:
            delta_dist = math.hypot(self.current_x - self.last_logged_x, self.current_y - self.last_logged_y)
            delta_angle = abs(normalize_angle(self.current_theta - self.last_logged_theta))

            if delta_dist >= self.log_dist_th or delta_angle >= self.log_angle_th:
                wp = Waypoint(self.current_x, self.current_y, self.current_theta, scan_pts=self.latest_scan_pts)
                self.waypoint_stack.append(wp)
                self.last_logged_x = self.current_x
                self.last_logged_y = self.current_y
                self.last_logged_theta = self.current_theta
                self.get_logger().info(
                    f"[ODOM LOGGER] Recorded Waypoint #{len(self.waypoint_stack)}: "
                    f"x={self.current_x:.2f}, y={self.current_y:.2f}, th={math.degrees(self.current_theta):.1f}° "
                    f"(Scan Points: {len(self.latest_scan_pts) if self.latest_scan_pts is not None else 0})"
                )

    def cb_heartbeat(self, msg: Header):
        self.last_heartbeat_time = self.get_clock().now()
        if not self.heartbeat_received_once:
            self.heartbeat_received_once = True
            self.get_logger().info("First Heartbeat packet received from Host PC.")

    def cb_gas(self, msg: Float32):
        self.current_gas_level = msg.data
        self.last_gas_time = self.get_clock().now()

    def cb_scan(self, msg: LaserScan):
        """
        1. Fast vectorized obstacle detection within forward +/- lidar_fov.
        2. Cartesian point extraction for 2D LiDAR Scan Matching odometry correction.
        """
        if not msg.ranges:
            return

        ranges = np.array(msg.ranges, dtype=np.float32)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        # 1. Forward obstacle check
        norm_angles = np.array([normalize_angle(a) for a in angles], dtype=np.float32)
        fwd_mask = (np.abs(norm_angles) <= self.lidar_fov) & \
                   (ranges >= self.lidar_min_dist) & (ranges <= self.lidar_danger_dist) & \
                   np.isfinite(ranges)
        self.obstacle_detected = bool(np.any(fwd_mask))

        # 2. Extract 2D Cartesian scan points (downsample step=2 for fast ~180 points)
        sub_ranges = ranges[::2]
        sub_angles = norm_angles[::2]
        valid_mask = (sub_ranges >= self.lidar_min_dist) & (sub_ranges <= 3.5) & np.isfinite(sub_ranges)

        if np.sum(valid_mask) >= 20:
            r_valid = sub_ranges[valid_mask]
            a_valid = sub_angles[valid_mask]
            self.latest_scan_pts = np.column_stack((r_valid * np.cos(a_valid), r_valid * np.sin(a_valid))).astype(np.float32)

            if self.start_scan_pts is None and self.odom_initialized:
                self.start_scan_pts = self.latest_scan_pts
                if self.start_pose and self.start_pose.scan_pts is None:
                    self.start_pose.scan_pts = self.latest_scan_pts

    def cb_remote_teleop(self, msg: Twist):
        # Forward remote teleop to robot motors ONLY in normal teleop mode
        if self.state == SystemState.NORMAL_TELEOP:
            self.pub_cmd_vel.publish(msg)

    def cb_dock_complete(self, msg: Bool):
        if msg.data:
            self.aruco_docking_complete = True
            self.state = SystemState.MISSION_COMPLETE
            self.stop_robot()
            self.get_logger().info(">>> ArUco Precision Docking Verified Complete! Mission Success. <<<")

    # -----------------------------------------------------------------------------
    # State Machine and Control Logic
    # -----------------------------------------------------------------------------
    def control_loop(self):
        now = self.get_clock().now()
        dt_heartbeat = (now - self.last_heartbeat_time).nanoseconds / 1e9

        # Publish Diagnostic Status String
        status_msg = String()
        status_msg.data = f"STATE:{self.state}|FAULTS:{self.fault_reason}|STACK:{len(self.waypoint_stack)}"
        self.pub_system_status.publish(status_msg)

        # 1. State: INIT
        if self.state == SystemState.INIT:
            if self.odom_initialized and self.heartbeat_received_once:
                self.state = SystemState.NORMAL_TELEOP
                self.get_logger().info("System ready. Transitioned to NORMAL_TELEOP mode.")
            return

        # 2. Watchdog Fault Detection (Active in NORMAL_TELEOP)
        if self.state == SystemState.NORMAL_TELEOP:
            fault_triggered = False
            if dt_heartbeat > self.heartbeat_timeout:
                self.fault_reason = f"HEARTBEAT_TIMEOUT ({dt_heartbeat:.2f}s > {self.heartbeat_timeout}s)"
                fault_triggered = True
            elif self.current_gas_level >= self.gas_threshold:
                self.fault_reason = f"GAS_DANGER_EXCEEDED (Val={self.current_gas_level:.2f} >= {self.gas_threshold})"
                fault_triggered = True
            elif self.last_gas_time is not None:
                dt_gas = (now - self.last_gas_time).nanoseconds / 1e9
                if dt_gas > self.gas_stale_timeout:
                    if self.gas_stale_triggers:
                        self.fault_reason = f"GAS_SENSOR_STALE ({dt_gas:.1f}s)"
                        fault_triggered = True
                    else:
                        self.get_logger().warn(
                            f"/gas/level stale for {dt_gas:.1f}s (gas monitoring degraded)",
                            throttle_duration_sec=5.0)

            if fault_triggered:
                self.get_logger().error(f"[FAIL-SAFE TRIGGERED] {self.fault_reason}. Halting and beginning recovery!")
                self.trigger_emergency_transition()
                return

        # 3. State: EMERGENCY_STOP
        if self.state == SystemState.EMERGENCY_STOP:
            self.stop_robot()
            # Verify and pop immediate closest or current waypoint
            if not self.waypoint_stack:
                self.get_logger().warn("Waypoint stack is empty at emergency stop! Directly requesting ArUco docking.")
                self.state = SystemState.HANDOVER_TO_ARUCO
            else:
                self.active_target_wp = self.waypoint_stack.pop()
                self.get_logger().info(
                    f"Selected initial backtrack WP: ({self.active_target_wp.x:.2f}, {self.active_target_wp.y:.2f}). "
                    f"Remaining in stack: {len(self.waypoint_stack)}"
                )
                self.state = SystemState.BACKTRACKING_ROTATE
            return

        # 4. Obstacle Hold Check during Backtracking
        if self.state in [SystemState.BACKTRACKING_ROTATE, SystemState.BACKTRACKING_TRANSLATE]:
            if self.obstacle_detected:
                self.stop_robot()
                self.previous_state = self.state
                self.state = SystemState.OBSTACLE_HOLD
                self.get_logger().warn(
                    f"[LIDAR SAFETY] Obstacle detected within {self.lidar_danger_dist}m! Pausing backtracking."
                )
                return

        # State: OBSTACLE_HOLD
        if self.state == SystemState.OBSTACLE_HOLD:
            self.stop_robot()
            if not self.obstacle_detected:
                self.get_logger().info("[LIDAR SAFETY] Obstacle cleared. Resuming backtracking.")
                self.state = self.previous_state
            return

        # 5. Method A Backtracking Logic: Rotate-then-Translate
        if self.state in [SystemState.BACKTRACKING_ROTATE, SystemState.BACKTRACKING_TRANSLATE]:
            self.perform_scan_matching_correction()

        if self.state == SystemState.BACKTRACKING_ROTATE:
            if self.active_target_wp is None:
                self.select_next_waypoint()
                return

            dx = self.active_target_wp.x - self.current_x
            dy = self.active_target_wp.y - self.current_y
            target_heading = math.atan2(dy, dx)
            heading_diff = normalize_angle(target_heading - self.current_theta)

            if abs(heading_diff) > self.heading_tol:
                # In-place pure rotation
                cmd = Twist()
                cmd.linear.x = 0.0
                angular_speed = self.kp_ang * heading_diff
                # Clamp to safe angular speed limits
                cmd.angular.z = max(-self.w_max, min(self.w_max, angular_speed))
                self.pub_cmd_vel.publish(cmd)
            else:
                # Target heading aligned
                self.stop_robot()
                self.state = SystemState.BACKTRACKING_TRANSLATE
                self.get_logger().info("Heading aligned. Transitioning to BACKTRACKING_TRANSLATE.")

        elif self.state == SystemState.BACKTRACKING_TRANSLATE:
            if self.active_target_wp is None:
                self.select_next_waypoint()
                return

            dx = self.active_target_wp.x - self.current_x
            dy = self.active_target_wp.y - self.current_y
            dist_to_wp = math.hypot(dx, dy)
            target_heading = math.atan2(dy, dx)
            heading_diff = normalize_angle(target_heading - self.current_theta)

            # If heading diverged substantially, re-align
            if abs(heading_diff) > (self.heading_tol * 2.5):
                self.stop_robot()
                self.state = SystemState.BACKTRACKING_ROTATE
                return

            if dist_to_wp > self.wp_reach_radius:
                # Drive forward with minor steering correction
                cmd = Twist()
                lin_speed = self.kp_lin * dist_to_wp
                cmd.linear.x = max(0.03, min(self.v_max, lin_speed))
                cmd.angular.z = max(-self.w_max, min(self.w_max, self.kp_ang * 0.5 * heading_diff))
                self.pub_cmd_vel.publish(cmd)
            else:
                # Waypoint reached
                self.stop_robot()
                self.get_logger().info(
                    f"Reached Waypoint ({self.active_target_wp.x:.2f}, {self.active_target_wp.y:.2f})."
                )
                self.check_proximity_or_next_waypoint()

        # 6. Handover to ArUco Visual Docking
        elif self.state == SystemState.HANDOVER_TO_ARUCO:
            self.stop_robot()
            if not self.use_aruco_docking:
                self.get_logger().info("ArUco docking disabled (use_aruco_docking=False). MISSION_COMPLETE at base camp.")
                self.state = SystemState.MISSION_COMPLETE
                return

            self.aruco_start_time = self.get_clock().now()
            trigger_msg = Bool()
            trigger_msg.data = True
            self.pub_dock_trigger.publish(trigger_msg)
            self.get_logger().info("Triggered ArUco Precision Docking Node. Handing over control.")
            self.state = SystemState.ARUCO_DOCKING

        elif self.state == SystemState.ARUCO_DOCKING:
            # Backup safety timeout in case aruco node or camera stalls
            if self.aruco_start_time is not None:
                elapsed = (now - self.aruco_start_time).nanoseconds / 1e9
                if elapsed > self.aruco_timeout:
                    self.get_logger().warn(
                        f"[RECOVERY TIMEOUT] ArUco docking exceeded backup timeout ({elapsed:.1f}s > {self.aruco_timeout}s). "
                        "Marking MISSION_COMPLETE at current location."
                    )
                    self.stop_robot()
                    self.state = SystemState.MISSION_COMPLETE

        elif self.state == SystemState.MISSION_COMPLETE:
            self.stop_robot()

    def perform_scan_matching_correction(self):
        """
        Refines (self.current_x, self.current_y, self.current_theta) using 2D LiDAR scan matching.
        Eliminates accumulated wheel odometry slip during backtracking by maintaining a persistent SE(2) correction.
        """
        if not self.enable_scan_matching or self.latest_scan_pts is None or self.active_target_wp is None:
            return

        ref_pts = self.active_target_wp.scan_pts
        if ref_pts is None:
            if self.start_pose and self.start_scan_pts is not None:
                dist_orig = math.hypot(self.start_pose.x - self.current_x, self.start_pose.y - self.current_y)
                if dist_orig <= self.docking_dist_th:
                    ref_pts = self.start_scan_pts

        if ref_pts is None:
            return

        dist_to_wp = math.hypot(self.active_target_wp.x - self.current_x, self.active_target_wp.y - self.current_y)
        if dist_to_wp > self.scan_match_max_dist:
            return

        # Estimate initial guess in target WP frame (vector from WP to robot expressed in WP frame)
        dx_world = self.current_x - self.active_target_wp.x
        dy_world = self.current_y - self.active_target_wp.y
        wp_th = self.active_target_wp.theta
        c_th, s_th = math.cos(wp_th), math.sin(wp_th)
        dx_guess = c_th * dx_world + s_th * dy_world
        dy_guess = -s_th * dx_world + c_th * dy_world
        dth_guess = normalize_angle(self.current_theta - wp_th)

        dx_rel, dy_rel, dth_rel, ok = icp_match_2d(
            self.latest_scan_pts, ref_pts,
            initial_guess=(dx_guess, dy_guess, dth_guess),
            max_iters=15, inlier_thresh=0.35, min_inliers=15
        )

        if ok:
            # Reconstruct measured robot world pose
            x_meas = self.active_target_wp.x + (c_th * dx_rel - s_th * dy_rel)
            y_meas = self.active_target_wp.y + (s_th * dx_rel + c_th * dy_rel)
            th_meas = normalize_angle(wp_th + dth_rel)

            # Update persistent SE(2) correction transform T_corr = T_meas * T_raw^-1
            target_th_c = normalize_angle(th_meas - self.raw_odom_theta)
            c_tc, s_tc = math.cos(target_th_c), math.sin(target_th_c)
            target_x_c = x_meas - (self.raw_odom_x * c_tc - self.raw_odom_y * s_tc)
            target_y_c = y_meas - (self.raw_odom_x * s_tc + self.raw_odom_y * c_tc)

            # Outlier rejection on sudden jumps (< 0.25m jump allowed per correction)
            jump_dist = math.hypot(target_x_c - self.corr_x, target_y_c - self.corr_y)
            if jump_dist < 0.25:
                alpha = self.scan_match_weight
                self.corr_x = (1.0 - alpha) * self.corr_x + alpha * target_x_c
                self.corr_y = (1.0 - alpha) * self.corr_y + alpha * target_y_c
                self.corr_th = normalize_angle(self.corr_th + alpha * normalize_angle(target_th_c - self.corr_th))

                # Re-evaluate current_x, current_y, current_theta with updated transform
                c_c = math.cos(self.corr_th)
                s_c = math.sin(self.corr_th)
                self.current_x = self.corr_x + self.raw_odom_x * c_c - self.raw_odom_y * s_c
                self.current_y = self.corr_y + self.raw_odom_x * s_c + self.raw_odom_y * c_c
                self.current_theta = normalize_angle(self.raw_odom_theta + self.corr_th)

                self.scan_match_count += 1
                if self.scan_match_count % 10 == 1:
                    self.get_logger().info(
                        f"[LIDAR SCAN MATCH #{self.scan_match_count}] Drift Compensated: "
                        f"Pose=({self.current_x:.2f}, {self.current_y:.2f}, {math.degrees(self.current_theta):.1f}°) | "
                        f"Offset=({self.corr_x:.3f}, {self.corr_y:.3f}, {math.degrees(self.corr_th):.1f}°)"
                    )

    def check_proximity_or_next_waypoint(self):
        """Check if robot is within 50cm of origin, or fetch next backtrack waypoint."""
        if self.start_pose:
            dist_to_origin = math.hypot(self.start_pose.x - self.current_x, self.start_pose.y - self.current_y)
            if dist_to_origin <= self.docking_dist_th:
                self.get_logger().info(
                    f"Robot arrived within docking perimeter ({dist_to_origin:.2f}m <= {self.docking_dist_th}m). "
                    f"Switching to ArUco Visual Docking."
                )
                self.state = SystemState.HANDOVER_TO_ARUCO
                return

        if len(self.waypoint_stack) > 0:
            self.active_target_wp = self.waypoint_stack.pop()
            self.get_logger().info(
                f"Next backtrack WP: ({self.active_target_wp.x:.2f}, {self.active_target_wp.y:.2f}). "
                f"Remaining stack size: {len(self.waypoint_stack)}"
            )
            self.state = SystemState.BACKTRACKING_ROTATE
        else:
            self.get_logger().info("All backtracked waypoints exhausted. Initiating ArUco Visual Docking.")
            self.state = SystemState.HANDOVER_TO_ARUCO

    def select_next_waypoint(self):
        if len(self.waypoint_stack) > 0:
            self.active_target_wp = self.waypoint_stack.pop()
            self.state = SystemState.BACKTRACKING_ROTATE
        else:
            self.state = SystemState.HANDOVER_TO_ARUCO

    def trigger_emergency_transition(self):
        self.stop_robot()
        self.state = SystemState.EMERGENCY_STOP

        # Trigger onboard blackbox recorder to save sensor and camera data locally
        bb_msg = Bool()
        bb_msg.data = True
        self.pub_blackbox_trigger.publish(bb_msg)

    def stop_robot(self):
        stop_cmd = Twist()
        self.pub_cmd_vel.publish(stop_cmd)


def main(args=None):
    rclpy.init(args=args)
    node = RecoveryController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Recovery Controller stopping due to KeyboardInterrupt.")
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
