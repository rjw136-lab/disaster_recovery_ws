#!/usr/bin/env python3
"""
=================================================================================================
Project: Disaster Recovery TurtleBot3 Burger System
Node: recovery_controller.py
Author: Kyungpook National University / Jong Woo Ro

Description:
  Unified Fail-Safe, Odometry Logger, Watchdog, Obstacle Avoidance, and Autonomous Waypoint
  Backtracking State Machine for TurtleBot3 Burger. Operates onboard Raspberry Pi 4 without external compute.

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

import os
import json
import math
import time
import collections
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import Header, Float32, Bool, String
except ImportError:  # pragma: no cover - allows importing pure utility functions without ROS 2
    rclpy = None
    Node = object
    QoSProfile = ReliabilityPolicy = HistoryPolicy = DurabilityPolicy = None
    Twist = Odometry = LaserScan = Header = Float32 = Bool = String = Any

from disaster_recovery.td3_policy import (
    TD3LocalAvoidancePolicy,
    extract_state_vector,
    check_line_of_sight_clear
)
from disaster_recovery.local_grid_planner import ProposedRecoveryPlanner
from disaster_recovery.rl_context_policy import (
    ContextRLStudentPolicy,
    compute_homotopy_blend,
)
from disaster_recovery.dynamic_tracker import DynamicObstacleTracker, predict_track_trajectory
from disaster_recovery.cbf_safety_shield import DiscreteSoftCBFShield


class SystemState:
    INIT = 'INIT'
    NORMAL_TELEOP = 'NORMAL_TELEOP'
    EMERGENCY_STOP = 'EMERGENCY_STOP'
    BACKTRACKING_ROTATE = 'BACKTRACKING_ROTATE'
    BACKTRACKING_TRANSLATE = 'BACKTRACKING_TRANSLATE'
    BACKTRACKING_HYBRID = 'BACKTRACKING_HYBRID'
    LOCAL_TD3_AVOIDANCE = 'LOCAL_TD3_AVOIDANCE'
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


def apply_motor_deadzone(
    v_cmd: float,
    w_cmd: float,
    min_v: float = 0.0,
    min_w: float = 0.0,
    v_max: float = 0.20,
    w_max: float = 0.65,
    min_v_reverse: float = -0.10,
    suppress_v_boost: bool = False,
    suppress_w_boost: bool = False,
) -> tuple[float, float]:
    """
    Compensate OpenCR/Dynamixel stiction deadzone while keeping explicit zero holds untouched.
    When suppress_v_boost / suppress_w_boost is True (e.g., CBF decelerated that axis for safety),
    sub-threshold commands clamp to 0.0 instead of boosting upward across the safety barrier.
    """
    v_out = float(v_cmd)
    w_out = float(w_cmd)

    if abs(v_out) <= 1e-4 and abs(w_out) <= 1e-4:
        return 0.0, 0.0

    if min_v > 0.0:
        if abs(v_out) <= 0.005:
            v_out = 0.0
        elif abs(v_out) < min_v:
            v_out = 0.0 if suppress_v_boost else math.copysign(min_v, v_out)

    if min_w > 0.0:
        if abs(w_out) <= 0.02:
            w_out = 0.0
        elif abs(w_out) < min_w:
            w_out = 0.0 if suppress_w_boost else math.copysign(min_w, w_out)

    v_out = max(float(min_v_reverse), min(float(v_max), v_out))
    w_out = max(-float(w_max), min(float(w_max), w_out))
    return v_out, w_out


def resolve_config_asset(
    configured_path: str,
    default_filename: str,
    search_dirs: Optional[list[str]] = None,
) -> str:
    """Resolve an explicit asset path or filename in package config directories without silent sim fallback."""
    if search_dirs is None:
        search_dirs = []
        try:
            from ament_index_python.packages import get_package_share_directory
            search_dirs.append(os.path.join(get_package_share_directory("disaster_recovery"), "config"))
        except Exception:
            pass
        mod_dir = os.path.dirname(os.path.abspath(__file__))
        search_dirs.append(os.path.abspath(os.path.join(mod_dir, "..", "config")))

    req = (configured_path or "").strip()
    if req:
        if os.path.exists(req):
            return req
        if not os.path.isabs(req):
            for cfg_dir in search_dirs:
                cand = os.path.join(cfg_dir, req)
                if os.path.exists(cand):
                    return cand
        return ""

    for cfg_dir in search_dirs:
        cand = os.path.join(cfg_dir, default_filename)
        if os.path.exists(cand):
            return cand
    return ""


def resolve_semantic_rl_asset(
    semantic_onnx_path: str,
    fallback_onnx_path: str = "",
    default_semantic_filename: str = "sim_policy_student_semantic_v3.onnx",
    default_fallback_filename: str = "sim_policy_student.onnx",
    search_dirs: Optional[list[str]] = None,
) -> Tuple[str, bool]:
    """
    Resolve Phase 3 3-channel semantic ONNX model path, falling back gracefully to
    the Phase 2-B 1-channel ONNX model if the semantic model is not yet present.
    Returns (resolved_path, is_semantic_model).
    """
    req_sem = (semantic_onnx_path or "").strip()
    if req_sem:
        sem_resolved = resolve_config_asset(req_sem, default_semantic_filename, search_dirs=search_dirs)
        if sem_resolved:
            return sem_resolved, True
    else:
        base_hint = (fallback_onnx_path or "").strip()
        inferred_default = default_semantic_filename
        if "real_tb3" in os.path.basename(base_hint):
            inferred_default = "real_tb3_policy_student_semantic_v3.onnx"
        sem_resolved = resolve_config_asset("", inferred_default, search_dirs=search_dirs)
        if sem_resolved:
            return sem_resolved, True

    fb_resolved = resolve_config_asset(fallback_onnx_path, default_fallback_filename, search_dirs=search_dirs)
    return fb_resolved, False


class SemanticObjectCache:
    """
    Lightweight onboard semantic cache for RPi 4 / recovery_controller.
    Stores verified permanent objects received on /semantic_map/cached_objects (in odom frame)
    and locally triggered gas hazards (/gas/level >= gas_threshold) so Ch 1 (low-lying obstacles)
    and Ch 2 (gas hazard & survivor protection zones) persist across communication loss.
    """

    CH2_CLASSES = frozenset({"person", "gas_hazard", "audio_distress"})

    def __init__(
        self,
        dedup_radius_m: float = 0.45,
        default_ch1_radius_m: float = 0.22,
        default_ch2_radius_m: float = 0.55,
    ):
        self.dedup_radius_m = float(dedup_radius_m)
        self.default_ch1_radius_m = float(default_ch1_radius_m)
        self.default_ch2_radius_m = float(default_ch2_radius_m)
        self._entries: List[Dict[str, Any]] = []

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    @classmethod
    def infer_channel(cls, class_name: str, explicit_channel: Optional[int] = None) -> int:
        if explicit_channel in (1, 2):
            return int(explicit_channel)
        norm = str(class_name or "").strip().lower()
        return 2 if norm in cls.CH2_CLASSES else 1

    @classmethod
    def _dedup_group(cls, class_name: str, channel: int) -> str:
        """Separate Ch2 deduplication by semantic class group so GAS_HAZARD and person/AUDIO_DISTRESS never overwrite each other."""
        if int(channel) != 2:
            return "ch1"
        norm = str(class_name or "").strip().lower()
        if "gas" in norm:
            return "gas_hazard"
        if norm == "person":
            return "person"
        if "audio" in norm or "distress" in norm:
            return "audio_distress"
        return norm or "ch2_other"

    @classmethod
    def is_gas_hazard_class(cls, class_name: str) -> bool:
        return cls._dedup_group(class_name, 2) == "gas_hazard"

    def upsert_object(
        self,
        class_name: str,
        x: float,
        y: float,
        radius: Optional[float] = None,
        channel: Optional[int] = None,
        conf: float = 1.0,
        low_confidence: bool = False,
        marker_id: Optional[int] = None,
        update_position: bool = True,
        local_mq2: bool = False,
    ) -> Dict[str, Any]:
        if not (math.isfinite(x) and math.isfinite(y)):
            return {}
        ch = self.infer_channel(class_name, channel)
        default_r = self.default_ch2_radius_m if ch == 2 else self.default_ch1_radius_m
        r_val = float(radius) if (radius is not None and math.isfinite(radius) and radius > 0.0) else default_r
        cls_str = str(class_name or ("GAS_HAZARD" if ch == 2 else "obstacle"))
        grp = self._dedup_group(cls_str, ch)
        is_local_flag = bool(local_mq2) and (grp == "gas_hazard")

        # 1. Match by explicit marker_id when non-negative, same channel, and same class group
        if marker_id is not None and int(marker_id) >= 0:
            mid = int(marker_id)
            for entry in self._entries:
                if (
                    entry.get("marker_id") == mid
                    and entry.get("channel") == ch
                    and self._dedup_group(entry.get("class_name", ""), ch) == grp
                ):
                    entry["class_name"] = cls_str
                    if update_position:
                        entry["x"] = float(x)
                        entry["y"] = float(y)
                    entry["radius"] = float(r_val)
                    entry["conf"] = float(conf)
                    entry["low_confidence"] = bool(low_confidence)
                    entry["local_mq2"] = bool(entry.get("local_mq2", False) or is_local_flag)
                    return entry

        # 2. Spatial deduplication within dedup_radius_m on same channel AND same class group
        for entry in self._entries:
            if (
                entry.get("channel") == ch
                and self._dedup_group(entry.get("class_name", ""), ch) == grp
            ):
                dist = math.hypot(float(x) - float(entry["x"]), float(y) - float(entry["y"]))
                if dist <= self.dedup_radius_m:
                    entry["class_name"] = cls_str
                    if update_position:
                        entry["x"] = 0.5 * (float(entry["x"]) + float(x))
                        entry["y"] = 0.5 * (float(entry["y"]) + float(y))
                    entry["radius"] = max(float(entry["radius"]), float(r_val))
                    entry["conf"] = max(float(entry.get("conf", 0.5)), float(conf))
                    entry["low_confidence"] = bool(entry.get("low_confidence", False) and low_confidence)
                    entry["local_mq2"] = bool(entry.get("local_mq2", False) or is_local_flag)
                    if marker_id is not None and int(marker_id) >= 0:
                        entry["marker_id"] = int(marker_id)
                    return entry

        new_entry = {
            "marker_id": int(marker_id) if (marker_id is not None and int(marker_id) >= 0) else None,
            "class_name": cls_str,
            "x": float(x),
            "y": float(y),
            "radius": float(r_val),
            "channel": int(ch),
            "conf": float(conf),
            "low_confidence": bool(low_confidence),
            "local_mq2": is_local_flag,
        }
        self._entries.append(new_entry)
        return new_entry

    def sync_icp_correction(
        self,
        old_corr: Tuple[float, float, float],
        new_corr: Tuple[float, float, float],
    ) -> None:
        """Transform all cached Ch1/Ch2 world coordinates by delta_T = T_new * T_old^-1 on ICP updates."""
        ox, oy, oth = float(old_corr[0]), float(old_corr[1]), float(old_corr[2])
        nx, ny, nth = float(new_corr[0]), float(new_corr[1]), float(new_corr[2])
        d_th = normalize_angle(nth - oth)
        c_d = math.cos(d_th)
        s_d = math.sin(d_th)
        d_tx = nx - (c_d * ox - s_d * oy)
        d_ty = ny - (s_d * ox + c_d * oy)
        for entry in self._entries:
            ex = float(entry["x"])
            ey = float(entry["y"])
            entry["x"] = d_tx + c_d * ex - s_d * ey
            entry["y"] = d_ty + s_d * ex + c_d * ey

    def update_from_json(
        self,
        json_str: str,
        corr_se2: Optional[Tuple[float, float, float]] = None,
        logger=None,
    ) -> int:
        """
        Parse and upsert semantic objects from JSON.
        Coordinate policy:
          - frame_id in ('odom', '/odom', ''): coordinates are in raw odom frame and transformed by corr_se2.
          - Any other frame_id (e.g. 'map'): coordinates are stored verbatim without corr_se2 transform and a warning is logged.
        """
        if not json_str:
            return 0
        try:
            payload = json.loads(json_str)
        except Exception:
            return 0
        if not isinstance(payload, dict):
            return 0
        objects = payload.get("objects", [])
        if not isinstance(objects, list):
            return 0

        frame_id = str(payload.get("frame_id", "odom") or "odom").strip().lower()
        is_odom_frame = frame_id in ("odom", "/odom", "")
        if not is_odom_frame:
            msg_warn = (
                f"[SemanticObjectCache] Received non-odom frame_id='{frame_id}'; "
                "storing coordinates verbatim without corr_se2 transform."
            )
            if logger is not None:
                logger.warn(msg_warn)
            else:
                import logging
                logging.getLogger(__name__).warning(msg_warn)
        apply_corr = (corr_se2 is not None) and is_odom_frame
        if apply_corr:
            cx_c, cy_c, th_c = float(corr_se2[0]), float(corr_se2[1]), float(corr_se2[2])
            cos_c = math.cos(th_c)
            sin_c = math.sin(th_c)

        count = 0
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            try:
                ox = float(obj["x"])
                oy = float(obj["y"])
            except (KeyError, TypeError, ValueError):
                continue
            if apply_corr:
                ox, oy = (
                    cx_c + ox * cos_c - oy * sin_c,
                    cy_c + ox * sin_c + oy * cos_c,
                )
            cls_name = str(obj.get("class_name", "obstacle"))
            raw_ch = obj.get("channel", None)
            ch = int(raw_ch) if raw_ch in (1, 2) else None
            raw_r = obj.get("radius", None)
            rad = float(raw_r) if raw_r is not None else None
            conf = float(obj.get("conf", 1.0))
            low_conf = bool(obj.get("low_confidence", False))
            raw_mid = obj.get("marker_id", None)
            mid = int(raw_mid) if isinstance(raw_mid, (int, float)) else None
            self.upsert_object(
                class_name=cls_name,
                x=ox,
                y=oy,
                radius=rad,
                channel=ch,
                conf=conf,
                low_confidence=low_conf,
                marker_id=mid,
            )
            count += 1
        return count

    def record_local_gas_hazard(
        self,
        x: float,
        y: float,
        gas_level: float = 1.0,
        radius: Optional[float] = None,
    ) -> Dict[str, Any]:
        # Keep existing gas anchor fixed during 20 Hz updates (prevents dragging gas cloud with robot motion)
        return self.upsert_object(
            class_name="GAS_HAZARD",
            x=float(x),
            y=float(y),
            radius=float(radius) if radius is not None else self.default_ch2_radius_m,
            channel=2,
            conf=min(1.0, max(0.5, float(gas_level))),
            low_confidence=False,
            update_position=False,
            local_mq2=True,
        )

    def get_channel_tuples(
        self,
    ) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
        """Return (ch1_obstacles, ch2_hazards) as lists of (x, y, radius) in odom frame."""
        ch1: List[Tuple[float, float, float]] = []
        ch2: List[Tuple[float, float, float]] = []
        for entry in self._entries:
            tup = (float(entry["x"]), float(entry["y"]), float(entry["radius"]))
            if int(entry.get("channel", 1)) == 2:
                ch2.append(tup)
            else:
                ch1.append(tup)
        return ch1, ch2

    def get_cbf_channel_tuples(
        self,
    ) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float, bool, bool]]]:
        """
        Return (ch1_obstacles, ch2_cbf_hazards) for DiscreteSoftCBFShield.
        Each Ch2 item is (x, y, radius, allow_mq2_escape, is_local_mq2) where
        allow_mq2_escape is True ONLY for GAS_HAZARD (False for person/AUDIO_DISTRESS).
        """
        ch1: List[Tuple[float, float, float]] = []
        ch2: List[Tuple[float, float, float, bool, bool]] = []
        for entry in self._entries:
            if int(entry.get("channel", 1)) == 2:
                is_gas = self.is_gas_hazard_class(entry.get("class_name", ""))
                is_local = bool(entry.get("local_mq2", False)) and is_gas
                ch2.append((float(entry["x"]), float(entry["y"]), float(entry["radius"]), is_gas, is_local))
            else:
                ch1.append((float(entry["x"]), float(entry["y"]), float(entry["radius"])))
        return ch1, ch2


def compute_rl_blend_weight(
    front_min: float,
    side_clearance: float,
    used_tight_pass: bool,
    max_alpha: float = 0.70,
    min_side: Optional[float] = None,
    has_nearby_semantic: bool = False,
    semantic_proximity: Optional[float] = None,
    valve_deadband: float = 0.0,
    prev_alpha: float = 0.0,
) -> float:
    """Gate RL blend weight to 0 in narrow pinches/doorways while smoothly ramping authority near semantic hazards.

    `valve_deadband` shifts the zero-gates by +/-deadband depending on whether the
    blend valve was previously open (hysteresis against threshold chattering).
    """
    if semantic_proximity is None:
        sem_prox = 1.0 if has_nearby_semantic else 0.0
    else:
        sem_prox = max(0.0, min(1.0, float(semantic_proximity)))
    boost_cap = max(max_alpha, 0.82)
    eff_max_alpha = max_alpha + (boost_cap - max_alpha) * sem_prox
    off = float(valve_deadband) if prev_alpha <= 1e-6 else -float(valve_deadband)
    if used_tight_pass or front_min < 0.28 + off or side_clearance < 0.62 + off:
        return 0.0
    if min_side is not None and min_side < 0.19 + off:
        return 0.0
    front_gate = min(1.0, max(0.0, (front_min - 0.28) / (0.45 - 0.28)))
    side_gate = min(1.0, max(0.0, (side_clearance - 0.62) / (0.85 - 0.62)))
    alpha = float(eff_max_alpha * min(front_gate, side_gate))
    if min_side is not None and min_side < 0.32:
        alpha *= max(0.0, (min_side - 0.19) / (0.32 - 0.19))
    return float(alpha)


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract Euler yaw angle from quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def icp_match_2d(
    src_pts,
    dst_pts,
    initial_guess=(0.0, 0.0, 0.0),
    max_iters=10,
    inlier_thresh=0.35,
    min_inliers=15,
    max_points=120,
    return_stats=False,
    eval_inlier_thresh=None,
):
    """
    Fast 2D Iterative Closest Point algorithm.
    Finds rigid transformation (dx, dy, dtheta) that aligns src_pts onto dst_pts.
    Both point clouds are uniformly stride-subsampled to at most `max_points`
    points so the N x M correspondence search stays bounded in the control loop.
    When `return_stats=True`, returns a 5-tuple `(dx, dy, theta, success, stats)`
    containing `num_inliers`, `num_points`, `inlier_ratio`, and `mean_err`.
    """
    if src_pts is None or dst_pts is None:
        empty_stats = {"num_inliers": 0, "num_points": 0, "inlier_ratio": 0.0, "mean_err": 99.0}
        return (0.0, 0.0, 0.0, False, empty_stats) if return_stats else (0.0, 0.0, 0.0, False)
    if len(src_pts) < min_inliers or len(dst_pts) < min_inliers:
        empty_stats = {"num_inliers": 0, "num_points": int(len(src_pts)), "inlier_ratio": 0.0, "mean_err": 99.0}
        return (0.0, 0.0, 0.0, False, empty_stats) if return_stats else (0.0, 0.0, 0.0, False)
    if max_points > 0 and len(src_pts) > max_points:
        src_pts = src_pts[:: max(1, len(src_pts) // max_points)][: max_points]
    if max_points > 0 and len(dst_pts) > max_points:
        dst_pts = dst_pts[:: max(1, len(dst_pts) // max_points)][: max_points]

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
    eval_thresh = float(inlier_thresh if eval_inlier_thresh is None else eval_inlier_thresh)
    inliers = min_dists < eval_thresh
    num_inliers = int(np.sum(inliers))
    num_points = int(len(cur_pts))
    inlier_ratio = float(num_inliers) / float(max(1, num_points))
    mean_err = float(np.mean(min_dists[inliers])) if num_inliers >= min_inliers else 99.0

    success = (num_inliers >= min_inliers) and (mean_err < 0.15)
    if return_stats:
        stats = {
            "num_inliers": num_inliers,
            "num_points": num_points,
            "inlier_ratio": inlier_ratio,
            "mean_err": mean_err,
        }
        return float(t[0]), float(t[1]), theta, success, stats
    return float(t[0]), float(t[1]), theta, success


def estimate_scan_velocity_icp(
    prev_pts: Optional[np.ndarray],
    cur_pts: Optional[np.ndarray],
    dt: float,
    prev_vw: Tuple[float, float] = (0.0, 0.0),
    eig_ratio_thresh: float = 0.04,
) -> Tuple[float, float, bool]:
    """
    Stage 0.3 consecutive-scan 2D ICP velocity estimator with geometric degeneracy detection.
    Returns (v_scan_mps, w_scan_rps, icp_degenerate).
    Falls back to prev_vw with icp_degenerate=True in featureless corridors (lam_min / lam_max < thresh).
    """
    v_fb, w_fb = float(prev_vw[0]), float(prev_vw[1])
    if prev_pts is None or cur_pts is None or dt < 0.03 or dt > 0.60:
        return v_fb, w_fb, True
    if len(prev_pts) < 20 or len(cur_pts) < 20:
        return v_fb, w_fb, True

    # Geometric conditioning check via 2D point-cloud covariance eigenvalue ratio
    centered = cur_pts - np.mean(cur_pts, axis=0, keepdims=True)
    cov = (centered.T @ centered) / float(max(1, len(cur_pts)))
    tr = float(cov[0, 0] + cov[1, 1])
    det = float(cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0])
    disc = math.sqrt(max(0.0, 0.25 * tr * tr - det))
    lam_max = 0.5 * tr + disc
    lam_min = max(0.0, 0.5 * tr - disc)
    if lam_max < 1e-4 or (lam_min / lam_max) < float(eig_ratio_thresh):
        return v_fb, w_fb, True

    # Align cur_pts (t_k) onto prev_pts (t_{k-1}): transform gives robot motion (dx, dy, dth) in prev body frame
    dx_g = v_fb * dt
    dth_g = w_fb * dt
    dx, dy, dth, ok = icp_match_2d(
        cur_pts,
        prev_pts,
        initial_guess=(dx_g, 0.0, dth_g),
        max_iters=8,
        inlier_thresh=0.25,
        min_inliers=15,
    )
    if not ok or math.hypot(dx, dy) > 0.45 * max(1.0, dt * 5.0) or abs(dth) > 0.8:
        return v_fb, w_fb, True
    v_est = float(np.clip(dx / dt, -0.35, 0.35))
    w_est = float(np.clip(dth / dt, -1.50, 1.50))
    return v_est, w_est, False


@dataclass
class CrossingDecision:
    """Output of CrossingLaneScheduler for a single control tick."""

    active: bool
    mode: str  # "INACTIVE", "APPROACH_ALIGN", "HOLD", "DASH", "DASH_FALLBACK"
    v_ref: float = 0.0
    w_ref: float = 0.0
    target_xy: Optional[Tuple[float, float]] = None
    track_id: Optional[int] = None
    lane_offset_m: float = 0.0
    lane_candidates: Tuple[float, ...] = ()
    t_launch_sec: float = float("inf")
    min_clearance_m: float = 0.0
    refresh_watchdog: bool = False


class CrossingLaneScheduler:
    """
    Lightweight 2D (y_lane, t_launch) spatio-temporal crossing scheduler.

    Activates ONLY when the local planned path crosses a confirmed dynamic track's
    predicted sweep segment (`|d_path . u_hat| <= 0.55`). Purely sensor/grid/tracker
    driven (zero hardcoded world coordinates, gate dimensions, or scenario names):
      1. Measures static left/right wall clearances from local occupancy grids/LiDAR
         to derive two lateral lane candidates satisfying wall clearance >= 0.17 m.
      2. Rolls out IMM predictions over 30 steps at 0.15 s (4.5 s horizon) for both
         lateral lanes simultaneously to pick the earliest safe (y_lane, t_launch)
         satisfying center distance >= r_obs + 0.16 + sigma(tau).
      3. Executes Approach Pre-Alignment -> Hold (v_ref=0, aligned heading, no reverse
         rotation) -> uninterrupted Dash (v_ref=v_max), with a >= 1-period deadlock
         timeout fallback launching at the widest-clearance phase under CBF-QP.
    """

    def __init__(
        self,
        cross_align_max: float = 0.55,
        cross_align_hold_max: float = 0.80,
        cross_macro_align_max: float = 0.60,
        max_pinch_opening_width_m: float = 1.85,
        rollout_dt: float = 0.15,
        rollout_steps: int = 30,
        min_wall_clearance_m: float = 0.17,
        lane_wall_margin_m: float = 0.18,
        clearance_extra_m: float = 0.16,
        hold_standoff_m: float = 0.38,
        exit_clearance_m: float = 0.22,
        max_init_sigma_m: float = 0.22,
        max_approach_dist_m: float = 2.00,
        lane_switch_hysteresis_s: float = 0.60,
        dash_max_duration_s: float = 8.0,
        dash_coast_grace_s: float = 1.0,
        hold_coast_grace_s: float = 1.0,
        lateral_align_tol_m: float = 0.16,
        lateral_shift_max_s: float = 5.0,
        zone_hold_radius_m: float = 0.85,
        zone_hold_decay_gap_s: float = 6.0,
        max_zone_hold_s: float = 14.0,
        zone_cooldown_s: float = 4.0,
    ) -> None:
        self.cross_align_max = float(cross_align_max)
        self.cross_align_hold_max = max(float(cross_align_hold_max), self.cross_align_max)
        self.cross_macro_align_max = float(cross_macro_align_max)
        self.max_pinch_opening_width_m = float(max_pinch_opening_width_m)
        self.rollout_dt = float(rollout_dt)
        self.rollout_steps = int(rollout_steps)
        self.launch_search_steps = max(self.rollout_steps, int(round(6.75 / max(1e-3, self.rollout_dt))))
        self.min_wall_clearance_m = float(min_wall_clearance_m)
        self.lane_wall_margin_m = max(float(lane_wall_margin_m), self.min_wall_clearance_m)
        self.clearance_extra_m = float(clearance_extra_m)
        self.hold_standoff_m = float(hold_standoff_m)
        self.exit_clearance_m = float(exit_clearance_m)
        self.max_init_sigma_m = float(max_init_sigma_m)
        self.max_approach_dist_m = float(max_approach_dist_m)
        self.lane_switch_hysteresis_s = float(lane_switch_hysteresis_s)
        self.dash_max_duration_s = float(dash_max_duration_s)
        self.dash_coast_grace_s = float(dash_coast_grace_s)
        self.hold_coast_grace_s = float(hold_coast_grace_s)
        self.lateral_align_tol_m = float(lateral_align_tol_m)
        self.lateral_shift_max_s = float(lateral_shift_max_s)
        self.zone_hold_radius_m = float(zone_hold_radius_m)
        self.zone_hold_decay_gap_s = float(zone_hold_decay_gap_s)
        self.max_zone_hold_s = float(max_zone_hold_s)
        self.zone_cooldown_s = float(zone_cooldown_s)

        k_cross = self.rollout_steps + 1
        k_launch = self.launch_search_steps + 1
        self._taus_full = (
            np.arange(0, self.launch_search_steps + self.rollout_steps + 1, dtype=np.float64) * self.rollout_dt
        )
        self._s_grid = self._taus_full[:k_cross]
        self._win_idx = (
            np.arange(k_launch, dtype=np.int64)[:, None]
            + np.arange(k_cross, dtype=np.int64)[None, :]
        )

        self._mode: str = "INACTIVE"
        self._track_id: Optional[int] = None
        self._latched_lane_y: Optional[float] = None
        self._hold_start_time: Optional[float] = None
        self._dash_start_time: Optional[float] = None
        self._align_shift_start: Optional[float] = None
        self._latched_frame: Optional[
            Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], float, float, float, float]
        ] = None
        self._latched_is_pinch: bool = False
        self._zone_hold_anchor: Optional[Tuple[float, float]] = None
        self._zone_hold_accum_s: float = 0.0
        self._zone_hold_last_t: Optional[float] = None
        self._zone_cooldown_until: float = -1e9
        self.last_decision: CrossingDecision = CrossingDecision(active=False, mode="INACTIVE")

    @property
    def active(self) -> bool:
        return self._mode != "INACTIVE"

    @property
    def is_dashing(self) -> bool:
        return self._mode in ("DASH", "DASH_FALLBACK")

    def _clear_latch(self) -> None:
        """Clear active track latch without wiping cumulative spatial-zone HOLD budget."""
        self._mode = "INACTIVE"
        self._track_id = None
        self._latched_lane_y = None
        self._hold_start_time = None
        self._dash_start_time = None
        self._align_shift_start = None
        self._latched_frame = None
        self._latched_is_pinch = False
        self.last_decision = CrossingDecision(active=False, mode="INACTIVE")

    def reset(self, clear_zone_budget: bool = True) -> None:
        self._clear_latch()
        if clear_zone_budget:
            self._zone_hold_anchor = None
            self._zone_hold_accum_s = 0.0
            self._zone_hold_last_t = None
            self._zone_cooldown_until = -1e9

    def _sync_zone_hold_budget(
        self, rx: float, ry: float, now_sec: float, accumulate: bool
    ) -> float:
        """
        Track cumulative HOLD time spent within `zone_hold_radius_m` of an anchor spot
        across track ID flicker, coast resets, or `pc` jumps so HOLD/refresh_watchdog
        cannot starve indefinitely.
        """
        if self._zone_hold_anchor is not None:
            dist_z = math.hypot(rx - self._zone_hold_anchor[0], ry - self._zone_hold_anchor[1])
            gap_z = (
                (float(now_sec) - float(self._zone_hold_last_t))
                if self._zone_hold_last_t is not None
                else 0.0
            )
            if dist_z >= self.zone_hold_radius_m or gap_z > self.zone_hold_decay_gap_s:
                self._zone_hold_anchor = None
                self._zone_hold_accum_s = 0.0
                self._zone_hold_last_t = None

        if accumulate:
            if self._zone_hold_anchor is None:
                self._zone_hold_anchor = (float(rx), float(ry))
                self._zone_hold_accum_s = 0.0
                self._zone_hold_last_t = float(now_sec)
            else:
                dt_z = (
                    max(0.0, min(0.5, float(now_sec) - float(self._zone_hold_last_t)))
                    if self._zone_hold_last_t is not None
                    else 0.0
                )
                self._zone_hold_accum_s += dt_z
                self._zone_hold_last_t = float(now_sec)
        return self._zone_hold_accum_s

    def sync_icp_correction(
        self,
        old_corr: Tuple[float, float, float],
        new_corr: Tuple[float, float, float],
    ) -> None:
        """Transform latched dash frame under SE(2) ICP correction delta_T = T_new * T_old^-1."""
        if self._latched_frame is None:
            return
        ox, oy, oth = float(old_corr[0]), float(old_corr[1]), float(old_corr[2])
        nx, ny, nth = float(new_corr[0]), float(new_corr[1]), float(new_corr[2])
        d_th = normalize_angle(nth - oth)
        c_d = math.cos(d_th)
        s_d = math.sin(d_th)
        d_tx = nx - (c_d * ox - s_d * oy)
        d_ty = ny - (s_d * ox + c_d * oy)
        pc, ef, el, r_obs, x_exit, y_star, t_seen = self._latched_frame
        new_pc = (d_tx + c_d * pc[0] - s_d * pc[1], d_ty + s_d * pc[0] + c_d * pc[1])
        new_ef = (c_d * ef[0] - s_d * ef[1], s_d * ef[0] + c_d * ef[1])
        new_el = (c_d * el[0] - s_d * el[1], s_d * el[0] + c_d * el[1])
        self._latched_frame = (new_pc, new_ef, new_el, r_obs, x_exit, y_star, t_seen)

    @staticmethod
    def _build_path_polyline(
        rx: float,
        ry: float,
        path_world: Optional[Sequence[Tuple[float, float]]],
        waypoint_stack: Optional[Sequence[Any]],
        subgoal_xy: Optional[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        pts: List[Tuple[float, float]] = []
        if path_world and len(path_world) >= 2:
            n_p = len(path_world)
            step = max(1, n_p // 36)
            pts = [(float(path_world[i][0]), float(path_world[i][1])) for i in range(0, n_p, step)]
            last_pt = (float(path_world[-1][0]), float(path_world[-1][1]))
            if pts[-1] != last_pt:
                pts.append(last_pt)
            return pts
        pts.append((float(rx), float(ry)))
        if subgoal_xy is not None:
            pts.append((float(subgoal_xy[0]), float(subgoal_xy[1])))
        if waypoint_stack:
            wp_list = list(waypoint_stack)
            for wp in reversed(wp_list[-8:]):
                pts.append((float(wp.x), float(wp.y)))
        return pts

    def _detect_path_crossing(
        self,
        pts_path: Sequence[Tuple[float, float]],
        rx: float,
        ry: float,
        rth: float,
        pc: Tuple[float, float],
        u_hat: Tuple[float, float],
        amp: float,
        r_obs: float,
        align_max: Optional[float] = None,
        macro_align_max: Optional[float] = None,
        waypoint_stack: Optional[Sequence[Any]] = None,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float, float]]:
        """
        Return (e_fwd, e_lat, x_r, y_r) if `pts_path` transversely crosses the obstacle sweep
        segment (`|d_path . u_hat| <= align_max` and macro/downstream route is not parallel
        to `u_hat`), else None.
        """
        if len(pts_path) < 2:
            return None
        pcx, pcy = pc
        ux, uy = u_hat
        # Normal to sweep axis
        nx0, ny0 = -uy, ux

        arr = np.asarray(pts_path, dtype=np.float64)
        dx = arr[:, 0] - pcx
        dy = arr[:, 1] - pcy
        d_line = dx * nx0 + dy * ny0
        s_line = dx * ux + dy * uy
        sweep_reach = max(amp, 0.15) + r_obs + 0.30

        cross_idx: Optional[int] = None
        for i in range(len(pts_path) - 1):
            d0, d1 = float(d_line[i]), float(d_line[i + 1])
            if (d0 <= 0.0 <= d1) or (d1 <= 0.0 <= d0):
                denom = abs(d0 - d1)
                frac = abs(d0) / denom if denom > 1e-9 else 0.5
                s_c = float(s_line[i] + frac * (s_line[i + 1] - s_line[i]))
                if abs(s_c) <= sweep_reach:
                    cross_idx = i
                    break

        if cross_idx is None:
            # Check if path passes very close to sweep segment (e.g. inside crossing zone or clipped at grid edge)
            abs_d = np.abs(d_line)
            min_i = int(np.argmin(abs_d))
            if float(abs_d[min_i]) <= (r_obs + 0.35) and abs(float(s_line[min_i])) <= sweep_reach:
                cross_idx = min_i
            else:
                return None

        # Compute local path direction averaged over +/-0.35 m around crossing index
        i_lo = max(0, cross_idx - 2)
        i_hi = min(len(pts_path) - 1, cross_idx + 2)
        if i_lo == i_hi:
            i_lo = max(0, cross_idx - 1)
            i_hi = min(len(pts_path) - 1, cross_idx + 1)
        dpx = float(arr[i_hi, 0] - arr[i_lo, 0])
        dpy = float(arr[i_hi, 1] - arr[i_lo, 1])
        dn = math.hypot(dpx, dpy)
        if dn < 1e-4:
            dpx = float(arr[-1, 0] - arr[0, 0])
            dpy = float(arr[-1, 1] - arr[0, 1])
            dn = math.hypot(dpx, dpy)
        if dn < 1e-4:
            dpx, dpy = math.cos(rth), math.sin(rth)
        else:
            dpx /= dn
            dpy /= dn

        align_lim = self.cross_align_max if align_max is None else float(align_max)
        if abs(dpx * ux + dpy * uy) > align_lim:
            return None

        # Check macro path start->end direction against obstacle principal axis u_hat
        # to block axial corridors where local A* detour jogs briefly bend perpendicular to u_hat.
        mpx = float(arr[-1, 0] - arr[0, 0])
        mpy = float(arr[-1, 1] - arr[0, 1])
        mn = math.hypot(mpx, mpy)
        macro_lim = self.cross_macro_align_max if macro_align_max is None else float(macro_align_max)
        if mn >= 0.45:
            if abs((mpx / mn) * ux + (mpy / mn) * uy) > macro_lim:
                return None

        # Build right-handed crossing frame: e_fwd perpendicular to u_hat aligned with travel
        if (nx0 * dpx + ny0 * dpy) >= 0.0:
            efx, efy = nx0, ny0
        else:
            efx, efy = -nx0, -ny0
        elx, ely = -efy, efx  # 90 deg left of e_fwd (collinear with +/- u_hat)

        x_r = (rx - pcx) * efx + (ry - pcy) * efy
        y_r = (rx - pcx) * elx + (ry - pcy) * ely
        x_exit = r_obs + self.exit_clearance_m
        if x_r < -self.max_approach_dist_m or x_r > x_exit:
            return None
        if abs(y_r) > max(amp, 0.25) + r_obs + 0.90:
            return None

        # Downstream backtrack route check: reject L-corner turns where upcoming waypoints
        # run parallel to u_hat instead of genuinely crossing past pc along e_fwd.
        if waypoint_stack:
            cx_p = float(arr[cross_idx, 0])
            cy_p = float(arr[cross_idx, 1])
            p_down = None
            best_far = None
            best_far_d = 0.0
            for wp in reversed(waypoint_stack):
                wx_i = float(wp.x)
                wy_i = float(wp.y)
                x_wp_f = (wx_i - pcx) * efx + (wy_i - pcy) * efy
                if x_wp_f < -0.15:
                    continue
                d_from_c = math.hypot(wx_i - cx_p, wy_i - cy_p)
                if d_from_c > best_far_d:
                    best_far_d = d_from_c
                    best_far = (wx_i, wy_i)
                if d_from_c >= 1.50:
                    p_down = (wx_i, wy_i)
                    break
            if p_down is None and best_far is not None and best_far_d >= 0.80:
                p_down = best_far
            if p_down is not None:
                ddx = p_down[0] - cx_p
                ddy = p_down[1] - cy_p
                ddn = math.hypot(ddx, ddy)
                if ddn >= 0.60:
                    down_align = abs((ddx / ddn) * ux + (ddy / ddn) * uy)
                    x_down_f = (p_down[0] - pcx) * efx + (p_down[1] - pcy) * efy
                    if down_align > 0.65 and x_down_f < (x_exit + 0.45):
                        return None

        return (efx, efy), (elx, ely), float(x_r), float(y_r)

    def _measure_lateral_lanes(
        self,
        pc: Tuple[float, float],
        e_fwd: Tuple[float, float],
        e_lat: Tuple[float, float],
        x_hold: float,
        amp: float,
        r_obs: float,
        confirmed_tracks: Sequence[Any],
        local_grid: Optional[Any],
        dynamic_tracker: Optional[Any],
        scan_pts: Optional[np.ndarray],
        rx: float,
        ry: float,
        rth: float,
        return_opening: bool = False,
    ) -> Any:
        """
        Directly measure static left/right gate pillar clearances from local occupancy
        grids / LiDAR (excluding dynamic obstacle disks) to produce up to 2 lateral
        lane offsets satisfying wall clearance >= min_wall_clearance_m (0.17 m).
        When `return_opening=True`, returns `(lane_candidates, (y_wall_lo, y_wall_hi, hit_wall_lo, hit_wall_hi))`.
        """
        pcx, pcy = pc
        efx, efy = e_fwd
        elx, ely = e_lat

        y_search = max(1.50, amp + r_obs + 0.65)
        dy_step = 0.04
        n_half = int(round(y_search / dy_step))
        y_vals = np.arange(-n_half, n_half + 1, dtype=np.float64) * dy_step
        x_lo_tube = min(x_hold - 0.12, -0.32)
        x_hi_tube = max(0.24, r_obs + 0.06)
        x_vals = np.arange(x_lo_tube, x_hi_tube + 1e-6, 0.04, dtype=np.float64)

        wx = pcx + x_vals[:, None] * efx + y_vals[None, :] * elx
        wy = pcy + x_vals[:, None] * efy + y_vals[None, :] * ely
        occ_2d = np.zeros(wx.shape, dtype=bool)

        # Build confirmed dynamic-obstacle footprint + observed-trail mask for local_grid.occ
        # (moving cylinders decay slowly in local_grid.occ; masking observed physical obstacle
        # footprints prevents moving trails from posing as fake static gate walls while never
        # erasing real static pillars).
        dyn_disk = np.zeros(wx.shape, dtype=bool)
        for trk in confirmed_tracks:
            tx = float(getattr(trk, "x", 0.0))
            ty = float(getattr(trk, "y", 0.0))
            r_base = max(0.10, float(getattr(trk, "radius", 0.15)))
            trad = r_base + 0.04
            dx_d = wx - tx
            dy_d = wy - ty
            dyn_disk |= (dx_d * dx_d + dy_d * dy_d) <= (trad * trad)
            hist = getattr(trk, "pos_history", ())
            if hist:
                n_h = len(hist)
                stride_h = max(1, n_h // 8)
                trad_h2 = (r_base + 0.025) ** 2
                for hi in range(0, n_h, stride_h):
                    _, hx, hy = hist[hi]
                    dyn_disk |= ((wx - float(hx)) ** 2 + (wy - float(hy)) ** 2) <= trad_h2

        # Source 1: DynamicObstacleTracker static_occ (already dynamic-masked)
        if dynamic_tracker is not None and getattr(dynamic_tracker, "grid_initialized", False):
            s_res = float(dynamic_tracker.res)
            s_n = int(dynamic_tracker.n)
            six = np.floor(wx / s_res).astype(np.int32) - int(dynamic_tracker.origin_ix)
            siy = np.floor(wy / s_res).astype(np.int32) - int(dynamic_tracker.origin_iy)
            inb = (six >= 0) & (six < s_n) & (siy >= 0) & (siy < s_n)
            six_c = np.clip(six, 0, s_n - 1)
            siy_c = np.clip(siy, 0, s_n - 1)
            occ_2d |= inb & (dynamic_tracker.static_occ[six_c, siy_c] >= 0.50)

        # Source 2: Local occupancy grid (excluding dynamic obstacle disk & observed trail)
        if local_grid is not None and getattr(local_grid, "initialized", False):
            l_res = float(local_grid.res)
            l_n = int(local_grid.n)
            lix = np.floor(wx / l_res).astype(np.int32) - int(local_grid.origin_ix)
            liy = np.floor(wy / l_res).astype(np.int32) - int(local_grid.origin_iy)
            inb_l = (lix >= 0) & (lix < l_n) & (liy >= 0) & (liy < l_n)
            lix_c = np.clip(lix, 0, l_n - 1)
            liy_c = np.clip(liy, 0, l_n - 1)
            occ_2d |= inb_l & (local_grid.occ[lix_c, liy_c] >= 0.45) & (~dyn_disk)

        # Source 3: Current LiDAR scan points in crossing tube (excluding instantaneous dynamic disks)
        if scan_pts is not None and len(scan_pts) > 0:
            pts = np.asarray(scan_pts, dtype=np.float64)
            c_r, s_r = math.cos(rth), math.sin(rth)
            swx = rx + pts[:, 0] * c_r - pts[:, 1] * s_r
            swy = ry + pts[:, 0] * s_r + pts[:, 1] * c_r
            s_dyn = np.zeros(len(swx), dtype=bool)
            for trk in confirmed_tracks:
                tx = float(getattr(trk, "x", 0.0))
                ty = float(getattr(trk, "y", 0.0))
                trad = max(0.10, float(getattr(trk, "radius", 0.15))) + 0.06
                s_dyn |= ((swx - tx) ** 2 + (swy - ty) ** 2) <= (trad * trad)
            keep = ~s_dyn
            if np.any(keep):
                dx_s = swx[keep] - pcx
                dy_s = swy[keep] - pcy
                gx_s = dx_s * efx + dy_s * efy
                gy_s = dx_s * elx + dy_s * ely
                in_tube = (gx_s >= x_lo_tube) & (gx_s <= x_hi_tube) & (np.abs(gy_s) <= y_search)
                if np.any(in_tube):
                    y_bins = np.rint((gy_s[in_tube] + n_half * dy_step) / dy_step).astype(np.int32)
                    y_bins = np.clip(y_bins, 0, len(y_vals) - 1)
                    occ_2d[0, y_bins] = True

        blocked_y = np.any(occ_2d, axis=0)
        seed_idx = n_half  # y == 0.0
        if blocked_y[seed_idx]:
            free_idxs = np.flatnonzero(~blocked_y)
            if len(free_idxs) == 0:
                return ((0.0,), (0.0, 0.0, False, False)) if return_opening else (0.0,)
            near_i = int(free_idxs[np.argmin(np.abs(free_idxs - seed_idx))])
            if abs(float(y_vals[near_i])) > max(amp, 0.25) + 0.40:
                return ((0.0,), (0.0, 0.0, False, False)) if return_opening else (0.0,)
            seed_idx = near_i

        i_lo = seed_idx
        while i_lo > 0 and not blocked_y[i_lo - 1]:
            i_lo -= 1
        i_hi = seed_idx
        while i_hi < (len(y_vals) - 1) and not blocked_y[i_hi + 1]:
            i_hi += 1

        # Wall face sits between the outermost free sample and first blocked sample
        hit_wall_lo = bool(i_lo > 0)
        hit_wall_hi = bool(i_hi < (len(y_vals) - 1))
        y_wall_lo = float(y_vals[i_lo - 1] + 0.5 * dy_step) if hit_wall_lo else float(y_vals[0])
        y_wall_hi = float(y_vals[i_hi + 1] - 0.5 * dy_step) if hit_wall_hi else float(y_vals[-1])
        opening_info = (y_wall_lo, y_wall_hi, hit_wall_lo, hit_wall_hi)

        y_min_safe = y_wall_lo + self.lane_wall_margin_m
        y_max_safe = y_wall_hi - self.lane_wall_margin_m
        if y_min_safe > y_max_safe:
            y_min_safe = y_wall_lo + self.min_wall_clearance_m
            y_max_safe = y_wall_hi - self.min_wall_clearance_m
            if y_min_safe > y_max_safe:
                cands = (0.5 * (y_wall_lo + y_wall_hi),)
                return (cands, opening_info) if return_opening else cands

        y_des_span = max(0.32, amp + r_obs + 0.22)
        y_right = max(y_min_safe, -y_des_span)
        y_left = min(y_max_safe, y_des_span)
        if y_right > y_left:
            cands = (0.5 * (y_min_safe + y_max_safe),)
            return (cands, opening_info) if return_opening else cands
        if (y_left - y_right) < 0.08:
            cands = (0.5 * (y_right + y_left),)
            return (cands, opening_info) if return_opening else cands
        cands = (float(y_right), float(y_left))
        return (cands, opening_info) if return_opening else cands

    def _evaluate_lanes_2d(
        self,
        lane_candidates: Sequence[float],
        trk: Any,
        dynamic_tracker: Optional[Any],
        pc: Tuple[float, float],
        e_fwd: Tuple[float, float],
        e_lat: Tuple[float, float],
        x_r: float,
        y_r: float,
        x_hold: float,
        x_exit: float,
        r_obs: float,
        v_dash: float,
    ) -> Tuple[List[Dict[str, float]], np.ndarray]:
        """
        Evaluate 2D (y_lane, t_launch) over 30 steps at 0.15 s (4.5 s crossing rollout)
        for each lateral lane candidate simultaneously, accounting for 2D approach time.
        Returns (lane_eval_dicts, sigmas_cross_31).
        """
        pcx, pcy = pc
        efx, efy = e_fwd
        elx, ely = e_lat

        k_steps = self.rollout_steps + 1
        v_use = max(0.08, float(v_dash))
        approaching = x_r < (x_hold - 0.08)
        x_launch_start = x_hold if approaching else min(x_r, x_hold + 0.12)

        # Compute base arrival offset so the 4.5 s launch-search window starts at arrival
        if approaching:
            min_dist_hold = min(
                math.hypot(x_hold - x_r, float(yl) - y_r) for yl in lane_candidates
            )
            t_base = min(6.0, max(0.0, min_dist_hold / v_use))
        else:
            t_base = 0.0

        # Single vectorized IMM rollout covering both immediate [0..4.5] and arrival search window
        if t_base <= 1e-6:
            pred_comb = (
                dynamic_tracker.predict_trajectory(trk, self._taus_full)
                if dynamic_tracker is not None and hasattr(dynamic_tracker, "predict_trajectory")
                else predict_track_trajectory(trk, self._taus_full)
            )
            sig_cross = pred_comb.sigmas[:k_steps]
            centers_now = pred_comb.centers[:k_steps]
            centers_search = pred_comb.centers
        else:
            taus_comb = np.empty(k_steps + len(self._taus_full), dtype=np.float64)
            taus_comb[:k_steps] = self._s_grid
            taus_comb[k_steps:] = t_base + self._taus_full
            pred_comb = (
                dynamic_tracker.predict_trajectory(trk, taus_comb)
                if dynamic_tracker is not None and hasattr(dynamic_tracker, "predict_trajectory")
                else predict_track_trajectory(trk, taus_comb)
            )
            sig_cross = pred_comb.sigmas[:k_steps]
            centers_now = pred_comb.centers[:k_steps]
            centers_search = pred_comb.centers[k_steps:]

        req_dist = (r_obs + self.clearance_extra_m) + sig_cross
        dx_s = centers_search[:, 0] - pcx
        dy_s = centers_search[:, 1] - pcy
        ox_g = dx_s * efx + dy_s * efy
        oy_g = dx_s * elx + dy_s * ely
        ox_win = ox_g[self._win_idx]  # (K, K)
        oy_win = oy_g[self._win_idx]

        dx_n = centers_now[:, 0] - pcx
        dy_n = centers_now[:, 1] - pcy
        ox_now = dx_n * efx + dy_n * efy
        oy_now = dx_n * elx + dy_n * ely

        x_rob_s = x_launch_start + v_use * self._s_grid
        active_s = x_rob_s <= (x_exit + 0.15)
        active_s[0] = True

        # Realistic differential-drive lateral convergence speed during an unaligned launch
        max_lat_shift = (0.45 * v_use) * self._s_grid
        dx2_win = (x_rob_s[None, :] - ox_win) ** 2
        dx2_now_std = (x_rob_s - ox_now) ** 2

        results: List[Dict[str, float]] = []
        for y_lane in lane_candidates:
            y_l = float(y_lane)
            if approaching:
                t_arr_lane = math.hypot(x_hold - x_r, y_l - y_r) / v_use
                j_min = min(
                    self.launch_search_steps,
                    max(0, int(math.ceil((t_arr_lane - t_base) / self.rollout_dt))),
                )
            else:
                j_min = 0

            # Future launches (aligned to y_l)
            dy_win = y_l - oy_win
            dists_all = np.sqrt(dx2_win + dy_win * dy_win)
            margins_all = np.where(active_s[None, :], dists_all - req_dist[None, :], np.inf)
            raw_masked = np.where(active_s[None, :], dists_all, np.inf)

            # Immediate launch (j=0) respects differential-drive lateral rate limit
            dy_rem = y_l - y_r
            if (not approaching) and abs(dy_rem) > 0.08:
                y_rob_now = y_r + np.sign(dy_rem) * np.minimum(abs(dy_rem), max_lat_shift)
                # Penalize forward progress slightly during steep diagonal transition
                diag_scale = max(0.70, math.cos(math.atan2(min(abs(dy_rem), 0.30), 0.45)))
                x_rob_now = x_launch_start + (v_use * diag_scale) * self._s_grid
                act_now = x_rob_now <= (x_exit + 0.15)
                act_now[0] = True
                dx_n_c = x_rob_now - ox_now
                dy_n_c = y_rob_now - oy_now
                d_now = np.sqrt(dx_n_c * dx_n_c + dy_n_c * dy_n_c)
                margin_now = float(np.where(act_now, d_now - req_dist, np.inf).min())
                raw_now = float(np.where(act_now, d_now, np.inf).min())
            else:
                dy_n_s = y_l - oy_now
                d_now = np.sqrt(dx2_now_std + dy_n_s * dy_n_s)
                margin_now = float(np.where(active_s, d_now - req_dist, np.inf).min())
                raw_now = float(np.where(active_s, d_now, np.inf).min())

            min_margin_j = margins_all.min(axis=1)
            min_raw_j = raw_masked.min(axis=1)
            safe_js = np.flatnonzero(min_margin_j[j_min:] >= 0.0)
            if len(safe_js) > 0:
                j_best = int(safe_js[0]) + j_min
                t_launch = float(t_base + j_best * self.rollout_dt)
                best_margin = float(min_margin_j[j_best])
            else:
                t_launch = float("inf")
                best_margin = float(min_margin_j[j_min:].max())

            results.append(
                {
                    "y_lane": y_l,
                    "t_launch": t_launch,
                    "best_margin": best_margin,
                    "margin_now": margin_now,
                    "raw_now": raw_now,
                    "best_raw": float(min_raw_j.max()),
                }
            )
        return results, sig_cross

    def _coast_latched_frame(
        self,
        rx: float,
        ry: float,
        rth: float,
        v_max: float,
        w_max: float,
        now_sec: float,
    ) -> Optional[CrossingDecision]:
        """Gracefully coast on `_latched_frame` across brief occlusion/flicker without wiping timers."""
        if self._latched_frame is None:
            return None
        pc, ef, el, r_obs, x_exit, y_star, t_seen = self._latched_frame
        x_r = (rx - pc[0]) * ef[0] + (ry - pc[1]) * ef[1]
        if self.is_dashing:
            dash_elapsed = (
                (now_sec - self._dash_start_time) if self._dash_start_time is not None else 0.0
            )
            if (
                x_r <= x_exit
                and (now_sec - t_seen) <= self.dash_coast_grace_s
                and dash_elapsed <= self.dash_max_duration_s
            ):
                x_aim = min(x_exit + 0.45, max(x_r + 0.45, 0.15))
                aim_x = pc[0] + x_aim * ef[0] + y_star * el[0]
                aim_y = pc[1] + x_aim * ef[1] + y_star * el[1]
                ang_d = normalize_angle(math.atan2(aim_y - ry, aim_x - rx) - rth)
                v_ref = v_max if abs(ang_d) <= 0.25 else v_max * max(0.70, math.cos(ang_d))
                w_ref = max(-w_max, min(w_max, 1.8 * ang_d))
                self.last_decision = CrossingDecision(
                    active=True,
                    mode=self._mode,
                    v_ref=float(v_ref),
                    w_ref=float(w_ref),
                    target_xy=(aim_x, aim_y),
                    track_id=self._track_id,
                    lane_offset_m=float(y_star),
                    t_launch_sec=0.0,
                    refresh_watchdog=(dash_elapsed < max(1.0, self.dash_max_duration_s - 1.0)),
                )
                return self.last_decision
            return None

        if (
            self._mode in ("HOLD", "APPROACH_ALIGN")
            and (now_sec - t_seen) <= self.hold_coast_grace_s
            and (-self.max_approach_dist_m <= x_r <= x_exit)
        ):
            zone_accum = self._sync_zone_hold_budget(rx, ry, now_sec, accumulate=True)
            hold_el = (
                max(0.0, float(now_sec) - float(self._hold_start_time))
                if self._hold_start_time is not None
                else 0.0
            )
            eff_hold = max(hold_el, zone_accum)
            if zone_accum >= self.max_zone_hold_s:
                self._zone_cooldown_until = float(now_sec) + self.zone_cooldown_s
                return None
            theta_fwd = math.atan2(ef[1], ef[0])
            e_th_fwd = normalize_angle(theta_fwd - rth)
            w_cap = min(float(w_max), 0.45)
            w_hold = max(-w_cap, min(w_cap, 1.2 * e_th_fwd)) if abs(e_th_fwd) > 0.06 else 0.0
            x_hold = -(r_obs + self.hold_standoff_m)
            hold_x = pc[0] + x_hold * ef[0] + y_star * el[0]
            hold_y = pc[1] + x_hold * ef[1] + y_star * el[1]
            self.last_decision = CrossingDecision(
                active=True,
                mode="HOLD",
                v_ref=0.0,
                w_ref=float(w_hold),
                target_xy=(hold_x, hold_y),
                track_id=self._track_id,
                lane_offset_m=float(y_star),
                refresh_watchdog=(eff_hold < 10.0),
            )
            return self.last_decision
        return None

    def step(
        self,
        rx: float,
        ry: float,
        rth: float,
        v_max: float,
        w_max: float,
        now_sec: float,
        confirmed_tracks: Sequence[Any],
        path_world: Optional[Sequence[Tuple[float, float]]] = None,
        waypoint_stack: Optional[Sequence[Any]] = None,
        subgoal_xy: Optional[Tuple[float, float]] = None,
        local_grid: Optional[Any] = None,
        dynamic_tracker: Optional[Any] = None,
        scan_pts: Optional[np.ndarray] = None,
        scan_corrupted: bool = False,
    ) -> CrossingDecision:
        """Evaluate 2D lateral lane + IMM phase launch schedule for one control tick."""
        if scan_corrupted:
            self._clear_latch()
            return self.last_decision

        # Anti-starvation cooldown after exceeding max_zone_hold_s in the same spatial zone
        if (
            (not self.is_dashing)
            and self._zone_hold_anchor is not None
            and float(now_sec) < self._zone_cooldown_until
        ):
            if math.hypot(rx - self._zone_hold_anchor[0], ry - self._zone_hold_anchor[1]) < self.zone_hold_radius_m:
                self._clear_latch()
                return self.last_decision
            self._zone_hold_anchor = None
            self._zone_hold_accum_s = 0.0
            self._zone_hold_last_t = None
            self._zone_cooldown_until = -1e9

        # Handle brief track coasting during active HOLD/APPROACH_ALIGN/DASH without wiping timers
        if not confirmed_tracks:
            coast = self._coast_latched_frame(rx, ry, rth, v_max, w_max, now_sec)
            if coast is not None:
                return coast
            self._clear_latch()
            return self.last_decision

        pts_path = self._build_path_polyline(rx, ry, path_world, waypoint_stack, subgoal_xy)
        latched_pc = self._latched_frame[0] if self._latched_frame is not None else None
        has_static_sources = bool(
            (local_grid is not None and getattr(local_grid, "initialized", False))
            or (dynamic_tracker is not None and getattr(dynamic_tracker, "grid_initialized", False))
            or (scan_pts is not None and len(scan_pts) > 0)
        )

        # Find nearest transversely crossing confirmed dynamic track inside a bounded pinch opening
        # (hysteresis align 0.55->0.80 is only permitted when already latched to a verified pinch).
        chosen = None
        for trk in confirmed_tracks:
            if not getattr(trk, "is_confirmed", False):
                continue
            pred = (
                dynamic_tracker.predict_trajectory(trk, (0.0,))
                if dynamic_tracker is not None and hasattr(dynamic_tracker, "predict_trajectory")
                else predict_track_trajectory(trk, (0.0,))
            )
            r_obs = max(0.10, float(getattr(trk, "radius", 0.15)))
            tid = getattr(trk, "track_id", None)
            is_latched_id = (self._track_id is not None and tid == self._track_id)
            near_latched_pc = bool(
                self.active
                and latched_pc is not None
                and math.hypot(pred.p_c[0] - latched_pc[0], pred.p_c[1] - latched_pc[1]) <= 0.35
            )
            latched_pinch = bool(
                self.active and self._latched_is_pinch and (is_latched_id or near_latched_pc)
            )
            align_lim = self.cross_align_hold_max if latched_pinch else self.cross_align_max
            macro_lim = self.cross_align_hold_max if latched_pinch else self.cross_macro_align_max
            geom = self._detect_path_crossing(
                pts_path,
                rx,
                ry,
                rth,
                pred.p_c,
                pred.u_hat,
                pred.amplitude,
                r_obs,
                align_max=align_lim,
                macro_align_max=macro_lim,
                waypoint_stack=waypoint_stack,
            )
            if geom is None:
                continue
            ef, el, x_r, y_r = geom
            lane_cands = None
            if not (self.is_dashing and latched_pinch):
                x_hold_cand = -(r_obs + self.hold_standoff_m)
                lane_cands, open_info = self._measure_lateral_lanes(
                    pc=pred.p_c,
                    e_fwd=ef,
                    e_lat=el,
                    x_hold=x_hold_cand,
                    amp=pred.amplitude,
                    r_obs=r_obs,
                    confirmed_tracks=confirmed_tracks,
                    local_grid=local_grid,
                    dynamic_tracker=dynamic_tracker,
                    scan_pts=scan_pts,
                    rx=rx,
                    ry=ry,
                    rth=rth,
                    return_opening=True,
                )
                y_w_lo, y_w_hi, hit_w_lo, hit_w_hi = open_info
                opening_width = y_w_hi - y_w_lo
                is_pinch_opening = (
                    (not has_static_sources)
                    or (
                        hit_w_lo
                        and hit_w_hi
                        and (opening_width < self.max_pinch_opening_width_m)
                        and ((y_w_lo - 0.85) <= y_r <= (y_w_hi + 0.85))
                    )
                )
                if not is_pinch_opening:
                    continue
            prio = 0 if is_latched_id else (1 if near_latched_pc else 2)
            score = (prio, abs(x_r))
            if chosen is None or score < chosen[0]:
                chosen = (score, trk, pred, r_obs, ef, el, x_r, y_r, lane_cands)

        if chosen is None:
            if self._latched_is_pinch:
                coast = self._coast_latched_frame(rx, ry, rth, v_max, w_max, now_sec)
                if coast is not None:
                    return coast
            self._clear_latch()
            return self.last_decision

        _, trk, pred, r_obs, ef, el, x_r, y_r, lane_candidates = chosen
        tid = getattr(trk, "track_id", None)
        pc = pred.p_c
        x_hold = -(r_obs + self.hold_standoff_m)
        x_exit = r_obs + self.exit_clearance_m

        if self._track_id != tid:
            # Preserve state/timers if a re-ID'd track matches the same physical oscillating obstacle
            same_physical = bool(
                self.active
                and latched_pc is not None
                and math.hypot(pc[0] - latched_pc[0], pc[1] - latched_pc[1]) <= 0.35
            )
            if same_physical:
                self._track_id = tid
            elif (
                self.active
                and self._latched_is_pinch
                and self._latched_frame is not None
                and (now_sec - self._latched_frame[6]) <= self.hold_coast_grace_s
            ):
                # Do not let an unrelated transient track steal lock while latched track coasts briefly
                coast = self._coast_latched_frame(rx, ry, rth, v_max, w_max, now_sec)
                if coast is not None:
                    return coast
                self._track_id = tid
                self._mode = "INACTIVE"
                self._latched_lane_y = None
                self._hold_start_time = None
                self._dash_start_time = None
                self._align_shift_start = None
                self._latched_is_pinch = False
            else:
                self._track_id = tid
                self._mode = "INACTIVE"
                self._latched_lane_y = None
                self._hold_start_time = None
                self._dash_start_time = None
                self._align_shift_start = None
                self._latched_is_pinch = False
        self._latched_is_pinch = True

        # Continue committed DASH without interruption until exit boundary cleared
        if self.is_dashing:
            dash_elapsed = (
                (now_sec - self._dash_start_time) if self._dash_start_time is not None else 0.0
            )
            if x_r > x_exit or dash_elapsed > self.dash_max_duration_s:
                self.reset()
                return self.last_decision
            y_star = float(self._latched_lane_y) if self._latched_lane_y is not None else float(y_r)
            self._latched_frame = (pc, ef, el, r_obs, x_exit, y_star, now_sec)
            x_aim = min(x_exit + 0.45, max(x_r + 0.45, 0.15))
            aim_x = pc[0] + x_aim * ef[0] + y_star * el[0]
            aim_y = pc[1] + x_aim * ef[1] + y_star * el[1]
            ang_d = normalize_angle(math.atan2(aim_y - ry, aim_x - rx) - rth)
            v_ref = v_max if abs(ang_d) <= 0.25 else v_max * max(0.70, math.cos(ang_d))
            w_ref = max(-w_max, min(w_max, 1.8 * ang_d))
            self.last_decision = CrossingDecision(
                active=True,
                mode=self._mode,
                v_ref=float(v_ref),
                w_ref=float(w_ref),
                target_xy=(aim_x, aim_y),
                track_id=tid,
                lane_offset_m=y_star,
                t_launch_sec=0.0,
                refresh_watchdog=(dash_elapsed < max(1.0, self.dash_max_duration_s - 1.0)),
            )
            return self.last_decision

        # Reuse lateral lanes already measured during pinch verification
        if lane_candidates is None:
            lane_candidates = self._measure_lateral_lanes(
                pc=pc,
                e_fwd=ef,
                e_lat=el,
                x_hold=x_hold,
                amp=pred.amplitude,
                r_obs=r_obs,
                confirmed_tracks=confirmed_tracks,
                local_grid=local_grid,
                dynamic_tracker=dynamic_tracker,
                scan_pts=scan_pts,
                rx=rx,
                ry=ry,
                rth=rth,
            )

        # Simultaneous 2D (y_lane, t_launch) evaluation across 0.15 s x 30 steps (4.5 s)
        evals, sig_cross = self._evaluate_lanes_2d(
            lane_candidates=lane_candidates,
            trk=trk,
            dynamic_tracker=dynamic_tracker,
            pc=pc,
            e_fwd=ef,
            e_lat=el,
            x_r=x_r,
            y_r=y_r,
            x_hold=x_hold,
            x_exit=x_exit,
            r_obs=r_obs,
            v_dash=v_max,
        )
        best_eval = min(
            evals,
            key=lambda d: (
                d["t_launch"],
                -d["best_margin"],
                abs(d["y_lane"] - y_r),
            ),
        )

        # Lane commitment & hysteresis:
        # 1) Never lock onto an infeasible lane (t_launch == inf) when another candidate lane
        #    has a feasible safe crossing window (or >= 0.12 m better clearance margin).
        # 2) Once committed to a feasible lane at/near Hold, stick to it to prevent flip-flop.
        # 3) Farther upstream, require >= 0.60 s earlier launch time to switch between feasible lanes.
        at_or_near_hold = x_r >= (x_hold - 0.45)
        if at_or_near_hold and len(evals) >= 2:
            if self._latched_lane_y is not None:
                pref_eval = min(evals, key=lambda d: abs(d["y_lane"] - self._latched_lane_y))
            else:
                pref_eval = min(evals, key=lambda d: abs(d["y_lane"] - y_r))
            pref_feasible = math.isfinite(pref_eval["t_launch"])
            best_feasible = math.isfinite(best_eval["t_launch"])
            margin_gain = best_eval["best_margin"] - pref_eval["best_margin"]
            should_override_infeasible = (
                (not pref_feasible) and best_feasible and (margin_gain > 0.08)
            ) or (margin_gain > 0.15)
            if not should_override_infeasible:
                best_eval = pref_eval
        elif self._latched_lane_y is not None and len(evals) >= 2:
            prev_eval = min(evals, key=lambda d: abs(d["y_lane"] - self._latched_lane_y))
            if prev_eval is not best_eval:
                prev_t = prev_eval["t_launch"]
                best_t = best_eval["t_launch"]
                margin_gain = best_eval["best_margin"] - prev_eval["best_margin"]
                if math.isfinite(prev_t):
                    if not (best_t + self.lane_switch_hysteresis_s < prev_t):
                        best_eval = prev_eval
                elif math.isfinite(best_t):
                    # Avoid 2 mm discretization flip-flop on symmetric lanes when approaching;
                    # switch only when the other lane has a genuine clearance advantage.
                    if margin_gain <= 0.08:
                        best_eval = prev_eval
                else:
                    if (
                        margin_gain <= 0.12
                        and abs(prev_eval["y_lane"] - y_r) <= abs(best_eval["y_lane"] - y_r) + 0.10
                    ):
                        best_eval = prev_eval

        y_star = float(best_eval["y_lane"])
        self._latched_lane_y = y_star
        self._latched_frame = (pc, ef, el, r_obs, x_exit, y_star, now_sec)

        k_sig_chk = min(10, len(sig_cross) - 1)  # tau = 1.5 s
        imm_uncertain = float(sig_cross[k_sig_chk]) > self.max_init_sigma_m
        theta_fwd = math.atan2(ef[1], ef[0])
        e_th_fwd = normalize_angle(theta_fwd - rth)
        lat_err = abs(y_r - y_star)

        # Phase 1: Approach Pre-Alignment (upstream of Hold Point)
        if x_r < (x_hold - 0.08):
            self._mode = "APPROACH_ALIGN"
            # Hysteresis: only clear hold timer if robot backed well upstream of Hold Point
            if x_r < (x_hold - 0.25):
                self._hold_start_time = None
                self._align_shift_start = None
            dist_to_hold = max(0.0, x_hold - x_r)
            # Converge laterally onto y_star BEFORE reaching x_hold so the robot arrives
            # already straightened along e_fwd for an immediate straight-line Dash.
            if lat_err > 0.06 and dist_to_hold > 0.18:
                x_aim = min(x_hold - 0.12, x_r + 0.35)
            else:
                x_aim = min(x_hold + 0.25, x_r + 0.50)
            aim_x = pc[0] + x_aim * ef[0] + y_star * el[0]
            aim_y = pc[1] + x_aim * ef[1] + y_star * el[1]
            ang_err = normalize_angle(math.atan2(aim_y - ry, aim_x - rx) - rth)
            if abs(ang_err) > 1.15:
                v_ref = 0.0
                w_ref = max(-w_max, min(w_max, 1.5 * ang_err))
            else:
                v_ref = v_max * max(0.25, math.cos(ang_err))
                window_ready_ahead = (
                    (not imm_uncertain)
                    and best_eval["margin_now"] >= 0.0
                    and lat_err <= 0.12
                )
                if dist_to_hold < 0.26 and not window_ready_ahead:
                    v_ref = min(v_ref, max(0.06, 0.55 * dist_to_hold))
                w_ref = max(-w_max, min(w_max, 1.75 * ang_err))
            self.last_decision = CrossingDecision(
                active=True,
                mode="APPROACH_ALIGN",
                v_ref=float(v_ref),
                w_ref=float(w_ref),
                target_xy=(aim_x, aim_y),
                track_id=tid,
                lane_offset_m=y_star,
                lane_candidates=tuple(lane_candidates),
                t_launch_sec=float(best_eval["t_launch"]),
                min_clearance_m=float(best_eval["raw_now"]),
                refresh_watchdog=(lat_err > self.lateral_align_tol_m),
            )
            return self.last_decision

        # Phase 1b: Lateral Re-Alignment at Entry Boundary
        # If robot reached Hold Point laterally misaligned from the feasible side lane y_star
        # (and still safely upstream of obstacle sweep), shift laterally upstream of the gate
        # before locking HOLD (v_ref=0), capped at lateral_shift_max_s for deadlock safety.
        need_lateral_shift = (
            lat_err > self.lateral_align_tol_m
            and x_r < -(r_obs + 0.16)
            and (best_eval["margin_now"] < 0.0 or imm_uncertain)
            and math.isfinite(best_eval["t_launch"])
        )
        if need_lateral_shift:
            if self._align_shift_start is None:
                self._align_shift_start = now_sec
            shift_elapsed = max(0.0, now_sec - self._align_shift_start)
            if shift_elapsed < self.lateral_shift_max_s:
                self._mode = "APPROACH_ALIGN"
                x_shift = min(x_hold - 0.08, x_r - 0.02)
                aim_x = pc[0] + x_shift * ef[0] + y_star * el[0]
                aim_y = pc[1] + x_shift * ef[1] + y_star * el[1]
                dist_aim = math.hypot(aim_x - rx, aim_y - ry)
                ang_err = normalize_angle(math.atan2(aim_y - ry, aim_x - rx) - rth)
                if abs(ang_err) > 1.05:
                    v_ref = 0.0
                    w_ref = max(-w_max, min(w_max, 1.5 * ang_err))
                else:
                    v_ref = v_max * max(0.35, math.cos(ang_err))
                    v_ref = min(v_ref, max(0.07, 0.70 * dist_aim))
                    w_ref = max(-w_max, min(w_max, 1.75 * ang_err))
                self.last_decision = CrossingDecision(
                    active=True,
                    mode="APPROACH_ALIGN",
                    v_ref=float(v_ref),
                    w_ref=float(w_ref),
                    target_xy=(aim_x, aim_y),
                    track_id=tid,
                    lane_offset_m=y_star,
                    lane_candidates=tuple(lane_candidates),
                    t_launch_sec=float(best_eval["t_launch"]),
                    min_clearance_m=float(best_eval["raw_now"]),
                    refresh_watchdog=True,
                )
                return self.last_decision
        else:
            self._align_shift_start = None

        # Phase 2 / 3: At Entry Boundary -> HOLD or LAUNCH DASH (+ deadlock timeout fallback)
        if self._hold_start_time is None:
            self._hold_start_time = now_sec
        hold_elapsed = max(0.0, now_sec - self._hold_start_time)
        zone_accum_pre = self._sync_zone_hold_budget(rx, ry, now_sec, accumulate=False)
        eff_hold_elapsed = max(hold_elapsed, zone_accum_pre)

        t_period = (
            (2.0 * math.pi / pred.omega)
            if (pred.is_harmonic and pred.omega > 0.20)
            else 6.0
        )
        t_period = max(4.0, min(10.0, t_period))
        hold_timeout = max(5.5, min(10.0, 1.15 * t_period))

        window_open_now = (
            (not imm_uncertain)
            and (best_eval["margin_now"] >= 0.0)
            and (abs(e_th_fwd) <= 0.30)
        )
        timeout_launch = (
            (eff_hold_elapsed >= hold_timeout)
            and (abs(e_th_fwd) <= 0.35)
            and (
                best_eval["raw_now"] >= 0.85 * best_eval["best_raw"]
                or eff_hold_elapsed >= (hold_timeout + 0.5 * t_period)
            )
        )

        if window_open_now or timeout_launch:
            self._mode = "DASH" if window_open_now else "DASH_FALLBACK"
            self._dash_start_time = now_sec
            self._latched_frame = (pc, ef, el, r_obs, x_exit, y_star, now_sec)
            x_aim = min(x_exit + 0.45, max(x_r + 0.45, 0.15))
            aim_x = pc[0] + x_aim * ef[0] + y_star * el[0]
            aim_y = pc[1] + x_aim * ef[1] + y_star * el[1]
            ang_d = normalize_angle(math.atan2(aim_y - ry, aim_x - rx) - rth)
            v_ref = v_max if abs(ang_d) <= 0.25 else v_max * max(0.70, math.cos(ang_d))
            w_ref = max(-w_max, min(w_max, 1.8 * ang_d))
            self.last_decision = CrossingDecision(
                active=True,
                mode=self._mode,
                v_ref=float(v_ref),
                w_ref=float(w_ref),
                target_xy=(aim_x, aim_y),
                track_id=tid,
                lane_offset_m=y_star,
                lane_candidates=tuple(lane_candidates),
                t_launch_sec=0.0,
                min_clearance_m=float(best_eval["raw_now"]),
                refresh_watchdog=True,
            )
            return self.last_decision

        # HOLD: v_ref = 0, maintain lateral-lane forward heading alignment without reverse rotation
        zone_accum = self._sync_zone_hold_budget(rx, ry, now_sec, accumulate=True)
        eff_hold_elapsed = max(hold_elapsed, zone_accum)
        if zone_accum >= self.max_zone_hold_s:
            self._zone_cooldown_until = float(now_sec) + self.zone_cooldown_s
            self._clear_latch()
            return self.last_decision
        self._mode = "HOLD"
        w_cap_hold = min(float(w_max), 0.45)
        if abs(e_th_fwd) > 0.06:
            w_hold = max(-w_cap_hold, min(w_cap_hold, 1.2 * e_th_fwd))
        else:
            w_hold = 0.0
        hold_x = pc[0] + x_hold * ef[0] + y_star * el[0]
        hold_y = pc[1] + x_hold * ef[1] + y_star * el[1]
        self.last_decision = CrossingDecision(
            active=True,
            mode="HOLD",
            v_ref=0.0,
            w_ref=float(w_hold),
            target_xy=(hold_x, hold_y),
            track_id=tid,
            lane_offset_m=y_star,
            lane_candidates=tuple(lane_candidates),
            t_launch_sec=float(best_eval["t_launch"]),
            min_clearance_m=float(best_eval["raw_now"]),
            refresh_watchdog=(eff_hold_elapsed < hold_timeout + 2.0),
        )
        return self.last_decision


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
        self.declare_parameter('enable_td3_avoidance', True)
        self.declare_parameter('td3_model_path', '')
        self.declare_parameter('td3_clearance_dist_m', 0.38)
        self.declare_parameter('td3_max_avoid_steps', 150)
        self.declare_parameter('enable_hybrid_tracking', True)
        self.declare_parameter('lookahead_dist_m', 0.45)
        self.declare_parameter('recovery_strategy', 'baseline')
        self.declare_parameter('local_grid_size_m', 6.0)
        self.declare_parameter('local_grid_res_m', 0.04)
        self.declare_parameter('local_grid_inflation_m', 0.085)
        self.declare_parameter('shortcut_rejoin_dist_m', 3.2)
        self.declare_parameter('unstuck_window_sec', 2.8)
        self.declare_parameter('unstuck_min_disp_m', 0.06)
        self.declare_parameter('reset_corr_on_episode', False)
        self.declare_parameter('context_rl_onnx_path', '')
        self.declare_parameter('use_semantic_rl_policy', False)
        self.declare_parameter('semantic_rl_onnx_path', '')
        self.declare_parameter('phase4_rl_onnx_path', '')
        self.declare_parameter('enable_phase2b_envelope', True)
        self.declare_parameter('smoke_lidar_noise_prob', 0.0)
        self.declare_parameter('enable_lidar_despike', False)
        self.declare_parameter('clear_stack_on_link_restore', False)
        self.declare_parameter('enable_semantic_astar_cost', False)
        self.declare_parameter('enable_dynamic_headon_hold', False)
        self.declare_parameter('enable_teleop_rl_warmup', False)
        self.declare_parameter('enable_lidar_slip_residual', False)
        self.declare_parameter('min_effective_linear_vel', 0.0)
        self.declare_parameter('min_effective_angular_vel', 0.0)
        self.declare_parameter('enable_dynamic_tracker', False)
        self.declare_parameter('enable_cbf_safety_shield', False)
        self.declare_parameter('cbf_inner_loop_hz', 10.0)
        self.declare_parameter('cbf_front_offset_m', 0.020)
        self.declare_parameter('cbf_rear_offset_m', 0.060)
        self.declare_parameter('cbf_circle_radius_m', 0.098)
        self.declare_parameter('cbf_v_min', -0.10)
        self.declare_parameter('cbf_gamma_dynamic_s', 3.0)
        self.declare_parameter('cbf_gamma_static_s', 6.0)
        self.declare_parameter('cbf_gamma_gas_s', 8.0)
        self.declare_parameter('cbf_w_v', 1.0)
        self.declare_parameter('cbf_w_w', 0.05)
        self.declare_parameter('cbf_slip_margin_mps', 0.015)
        self.declare_parameter('cbf_lidar_voxel_m', 0.10)
        self.declare_parameter('cbf_lidar_max_points', 16)
        self.declare_parameter('cbf_enable_mq2_lifo_escape', True)
        self.declare_parameter('dynamic_graze_buffer_m', 0.06)
        self.declare_parameter('alpha_max_step_base', 0.15)
        self.declare_parameter('alpha_max_step_sem_gain', 0.10)
        self.declare_parameter('alpha_valve_deadband', 0.02)
        self.declare_parameter('enable_spatiotemporal_dynamic_control', True)

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
        self.enable_td3_avoidance = self.get_parameter('enable_td3_avoidance').value
        self.td3_model_path = self.get_parameter('td3_model_path').value
        self.td3_clearance_dist = self.get_parameter('td3_clearance_dist_m').value
        self.td3_max_avoid_steps = self.get_parameter('td3_max_avoid_steps').value
        self.enable_hybrid = self.get_parameter('enable_hybrid_tracking').value
        self.lookahead_dist = self.get_parameter('lookahead_dist_m').value
        self.recovery_strategy = str(self.get_parameter('recovery_strategy').value).lower()
        self.local_grid_size_m = float(self.get_parameter('local_grid_size_m').value)
        self.local_grid_res_m = float(self.get_parameter('local_grid_res_m').value)
        self.local_grid_inflation_m = float(self.get_parameter('local_grid_inflation_m').value)
        self.shortcut_rejoin_dist_m = float(self.get_parameter('shortcut_rejoin_dist_m').value)
        self.unstuck_window_sec = float(self.get_parameter('unstuck_window_sec').value)
        self.unstuck_min_disp_m = float(self.get_parameter('unstuck_min_disp_m').value)
        self.context_rl_onnx_path = str(self.get_parameter('context_rl_onnx_path').value)
        self.use_semantic_rl_policy = bool(self.get_parameter('use_semantic_rl_policy').value)
        self.semantic_rl_onnx_path = str(self.get_parameter('semantic_rl_onnx_path').value)
        self.phase4_rl_onnx_path = str(self.get_parameter('phase4_rl_onnx_path').value)
        self.enable_phase2b_envelope = bool(self.get_parameter('enable_phase2b_envelope').value)
        self.smoke_lidar_noise_prob = float(self.get_parameter('smoke_lidar_noise_prob').value)
        self.enable_lidar_despike = bool(self.get_parameter('enable_lidar_despike').value)
        self.clear_stack_on_link_restore = bool(self.get_parameter('clear_stack_on_link_restore').value)
        self.enable_semantic_astar_cost = bool(self.get_parameter('enable_semantic_astar_cost').value)
        self.enable_dynamic_headon_hold = bool(self.get_parameter('enable_dynamic_headon_hold').value)
        self.enable_teleop_rl_warmup = bool(self.get_parameter('enable_teleop_rl_warmup').value)
        self.enable_lidar_slip_residual = bool(self.get_parameter('enable_lidar_slip_residual').value)
        self.min_effective_linear_vel = float(self.get_parameter('min_effective_linear_vel').value)
        self.min_effective_angular_vel = float(self.get_parameter('min_effective_angular_vel').value)
        self.enable_dynamic_tracker = bool(self.get_parameter('enable_dynamic_tracker').value)
        self.enable_spatiotemporal_dynamic_control = bool(
            self.get_parameter('enable_spatiotemporal_dynamic_control').value
        )
        self.enable_cbf_safety_shield = bool(self.get_parameter('enable_cbf_safety_shield').value)
        self.cbf_inner_loop_hz = float(self.get_parameter('cbf_inner_loop_hz').value)
        self.cbf_front_offset_m = float(self.get_parameter('cbf_front_offset_m').value)
        self.cbf_rear_offset_m = float(self.get_parameter('cbf_rear_offset_m').value)
        self.cbf_circle_radius_m = float(self.get_parameter('cbf_circle_radius_m').value)
        self.cbf_v_min = float(self.get_parameter('cbf_v_min').value)
        self.cbf_gamma_dynamic_s = float(self.get_parameter('cbf_gamma_dynamic_s').value)
        self.cbf_gamma_static_s = float(self.get_parameter('cbf_gamma_static_s').value)
        self.cbf_gamma_gas_s = float(self.get_parameter('cbf_gamma_gas_s').value)
        self.cbf_w_v = float(self.get_parameter('cbf_w_v').value)
        self.cbf_w_w = float(self.get_parameter('cbf_w_w').value)
        self.cbf_slip_margin_mps = float(self.get_parameter('cbf_slip_margin_mps').value)
        self.cbf_lidar_voxel_m = float(self.get_parameter('cbf_lidar_voxel_m').value)
        self.cbf_lidar_max_points = int(self.get_parameter('cbf_lidar_max_points').value)
        self.cbf_enable_mq2_lifo_escape = bool(self.get_parameter('cbf_enable_mq2_lifo_escape').value)
        self.dynamic_graze_buffer_m = float(self.get_parameter('dynamic_graze_buffer_m').value)
        self.alpha_max_step_base = float(self.get_parameter('alpha_max_step_base').value)
        self.alpha_max_step_sem_gain = float(self.get_parameter('alpha_max_step_sem_gain').value)
        self.alpha_valve_deadband = float(self.get_parameter('alpha_valve_deadband').value)
        self.is_phase2b = (self.recovery_strategy == 'proposed_rl') and self.enable_phase2b_envelope
        self.semantic_cache = SemanticObjectCache()
        self._scan_tick_seq = 0
        self._prev_escape_count = 0
        self._rl_cooldown_until = 0.0
        self._rl_lockout_anchor = None
        self._rl_lockout_until = 0.0
        self._prev_front_min = None
        self._prev_front_time = None
        self._prev_alpha_rl = 0.0
        self._dynamic_hold_until = 0.0
        self._dynamic_hold_cooldown_until = 0.0
        self._prev_slip_scan_pts = None
        self._prev_slip_scan_stamp = None
        self._icp_done_scan_stamp = None
        self._icp_section_ms_total = 0.0
        self._icp_section_ms_max = 0.0
        self._icp_section_count = 0
        self._icp_section_last_log = 0.0
        self._icp_run_count = 0
        self._icp_skip_count = 0
        self.icp_degenerate = False
        self.scan_vel_v = 0.0
        self.scan_vel_w = 0.0
        self.reset_corr_on_episode = bool(self.get_parameter('reset_corr_on_episode').value) or (
            self.recovery_strategy in ('proposed', 'proposed_rl')
        )

        # -------------------------------------------------------------------------
        # Internal State Variables
        # -------------------------------------------------------------------------
        self.state = SystemState.INIT
        self.previous_state = SystemState.INIT

        # TD3 Local Reactive Avoidance Variables
        self.latest_raw_ranges = []
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0
        self.actual_v = 0.0
        self.actual_w = 0.0
        self.td3_avoid_step = 0
        self.recovery_step_count = 0

        # Initialize TD3 Local Avoidance Policy (Layer 2) — skipped for proposed / proposed_rl
        if self.enable_td3_avoidance and self.recovery_strategy not in ('proposed', 'proposed_rl'):
            model_path = self.td3_model_path
            if not model_path or not os.path.exists(model_path):
                candidates = []
                try:
                    from ament_index_python.packages import get_package_share_directory
                    pkg_cfg = os.path.join(get_package_share_directory("disaster_recovery"), "config")
                    candidates.extend([
                        os.path.join(pkg_cfg, "recovery_td3.onnx"),
                        os.path.join(pkg_cfg, "recovery_td3.pth"),
                    ])
                except Exception:
                    pass
                mod_dir = os.path.dirname(os.path.abspath(__file__))
                src_cfg = os.path.abspath(os.path.join(mod_dir, "..", "config"))
                candidates.extend([
                    os.path.join(src_cfg, "recovery_td3.onnx"),
                    os.path.join(src_cfg, "recovery_td3.pth"),
                ])
                for c in candidates:
                    if os.path.exists(c):
                        model_path = c
                        break
            # recovery_td3.onnx/.pth is trained with normalization constants (max_v=0.15, max_w=0.60);
            # keep policy scale fixed to model constants and apply runtime limits (self.v_max, self.w_max) on output clipping.
            self.td3_norm_max_v = 0.15
            self.td3_norm_max_w = 0.60
            self.td3_policy = TD3LocalAvoidancePolicy(
                model_path=model_path,
                max_v=self.td3_norm_max_v,
                min_v=0.03,
                max_w=self.td3_norm_max_w
            )
            self.get_logger().info(
                f"[TD3] Local Reactive Avoidance active (Engine: {self.td3_policy.model_type.upper()})"
            )
        else:
            self.td3_policy = None

        self.context_rl_policy = None
        if self.recovery_strategy in ('proposed', 'proposed_rl'):
            self.local_planner = ProposedRecoveryPlanner(
                size_m=self.local_grid_size_m,
                resolution_m=self.local_grid_res_m,
                inflation_radius_m=self.local_grid_inflation_m,
                lookahead_m=max(0.45, self.lookahead_dist),
                shortcut_rejoin_dist_m=self.shortcut_rejoin_dist_m,
                unstuck_window_sec=self.unstuck_window_sec,
                unstuck_min_disp_m=self.unstuck_min_disp_m,
                phase2b_mode=self.is_phase2b,
                enable_spatiotemporal_dynamic_control=self.enable_spatiotemporal_dynamic_control,
            )
            if self.recovery_strategy == 'proposed_rl':
                phase4_active = self.enable_cbf_safety_shield or self.enable_dynamic_tracker
                if self.use_semantic_rl_policy:
                    if phase4_active and (not self.semantic_rl_onnx_path.strip()) and self.phase4_rl_onnx_path.strip():
                        rl_path = resolve_config_asset(
                            self.phase4_rl_onnx_path,
                            "sim_policy_student_phase4.onnx",
                        )
                        if not rl_path:
                            rl_path, _ = resolve_semantic_rl_asset(
                                "",
                                fallback_onnx_path=self.context_rl_onnx_path,
                                default_semantic_filename="sim_policy_student_semantic_v3.onnx",
                                default_fallback_filename="sim_policy_student.onnx",
                            )
                    else:
                        rl_path, is_sem = resolve_semantic_rl_asset(
                            self.semantic_rl_onnx_path,
                            fallback_onnx_path=self.context_rl_onnx_path,
                            default_semantic_filename="sim_policy_student_semantic_v3.onnx",
                            default_fallback_filename="sim_policy_student.onnx",
                        )
                        if not is_sem and rl_path:
                            self.get_logger().warn(
                                f"[PROPOSED_RL] Semantic ONNX not found; falling back gracefully to 1ch {os.path.basename(rl_path)}"
                            )
                else:
                    rl_path = resolve_config_asset(self.context_rl_onnx_path, "sim_policy_student.onnx")
                if rl_path and os.path.exists(rl_path):
                    try:
                        self.context_rl_policy = ContextRLStudentPolicy(
                            onnx_path=rl_path,
                            v_max=self.v_max,
                            w_max=self.w_max,
                            phase2b_mode=self.is_phase2b,
                            v_min=self.cbf_v_min,
                        )
                        if self.context_rl_policy.patch_channels >= 4 and (
                            (not self.enable_cbf_safety_shield) or (not self.enable_dynamic_tracker)
                        ):
                            self.get_logger().warn(
                                f"[PROPOSED_RL SAFETY WARNING] Loaded 4-Ch Phase 4 ONNX ({os.path.basename(rl_path)}) "
                                f"with enable_cbf_safety_shield={self.enable_cbf_safety_shield}, "
                                f"enable_dynamic_tracker={self.enable_dynamic_tracker}! "
                                "Phase 4 4-Ch models omit internal SemanticCorridorSafetyHead and require CBF + DynamicTracker."
                            )
                        self.get_logger().info(
                            f"[PROPOSED_RL] Context-RL ONNX active: {os.path.basename(rl_path)} "
                            f"(channels={self.context_rl_policy.patch_channels}, v_min={self.cbf_v_min:+.2f}, semantic={self.use_semantic_rl_policy})"
                        )
                    except Exception as exc:
                        self.get_logger().warn(f"[PROPOSED_RL] ONNX load failed ({exc}), falling back to A*")
                else:
                    target_name = (
                        self.semantic_rl_onnx_path.strip()
                        or (self.phase4_rl_onnx_path.strip() if phase4_active else "sim_policy_student_semantic_v3.onnx")
                        if self.use_semantic_rl_policy
                        else self.context_rl_onnx_path.strip() or "sim_policy_student.onnx"
                    )
                    self.get_logger().warn(f"[PROPOSED_RL] {target_name} not found, falling back to A*")
            else:
                self.get_logger().info(
                    "[PROPOSED] Rolling 2D Grid A* + Topological Shortcut + Unstuck Escape active"
                )
        else:
            self.local_planner = None

        self.dynamic_tracker = (
            DynamicObstacleTracker(
                grid_size_m=self.local_grid_size_m,
                grid_res_m=self.local_grid_res_m,
                lidar_min_range_m=self.lidar_min_dist,
                lidar_offset_x_m=-0.032,
                enable_spatiotemporal_dynamic_control=self.enable_spatiotemporal_dynamic_control,
            )
            if self.enable_dynamic_tracker
            else None
        )
        self.cbf_shield = (
            DiscreteSoftCBFShield(
                l_f=self.cbf_front_offset_m,
                l_r=self.cbf_rear_offset_m,
                r_circle=self.cbf_circle_radius_m,
                v_min=self.cbf_v_min,
                v_max=self.v_max,
                w_max=self.w_max,
                gamma_dynamic=self.cbf_gamma_dynamic_s,
                gamma_static=self.cbf_gamma_static_s,
                gamma_gas=self.cbf_gamma_gas_s,
                w_v=self.cbf_w_v,
                w_w=self.cbf_w_w,
                slip_margin_mps=self.cbf_slip_margin_mps,
                lidar_voxel_m=self.cbf_lidar_voxel_m,
                lidar_max_points=self.cbf_lidar_max_points,
                lidar_min_range_m=self.lidar_min_dist,
                enable_mq2_lifo_escape=self.cbf_enable_mq2_lifo_escape,
                dynamic_graze_buffer_m=self.dynamic_graze_buffer_m,
                dt_default=1.0 / max(1.0, self.cbf_inner_loop_hz),
            )
            if self.enable_cbf_safety_shield
            else None
        )
        self.latest_cbf_result = None
        if self.cbf_shield is not None:
            self.get_logger().info(
                f"[STAGE2 CBF] 2-Circle Soft-CBF-QP Shield active "
                f"(l_f={self.cbf_front_offset_m:.3f}, l_r={self.cbf_rear_offset_m:.3f}, "
                f"r={self.cbf_circle_radius_m:.3f}, v_min={self.cbf_v_min:.2f}, "
                f"Q=diag({self.cbf_w_v:.2f},{self.cbf_w_w:.3f}))"
            )
        self.latest_scan_stamp_sec = None
        self.latest_scan_pose = None
        self.latest_scan_angle_min = 0.0
        self.latest_scan_angle_increment = None
        self.latest_dynamic_tracks = []
        self.latest_dynamic_ch3 = None
        self.crossing_scheduler = (
            CrossingLaneScheduler() if self.enable_spatiotemporal_dynamic_control else None
        )

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
        self.latest_scan_spike_ratio = 0.0
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
        self.td3_obstacle_detected = False
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
        self.sub_reset_episode = self.create_subscription(
            Bool, '/recovery/reset_episode', self.cb_reset_episode, reliable_qos
        )
        self.sub_semantic_cached = self.create_subscription(
            String, '/semantic_map/cached_objects', self.cb_semantic_cached_objects, reliable_qos
        )

        # -------------------------------------------------------------------------
        # Publishers
        # -------------------------------------------------------------------------
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_blackbox_trigger = self.create_publisher(Bool, '/blackbox/trigger', 10)
        self.pub_dock_trigger = self.create_publisher(Bool, '/aruco/dock_trigger', 10)
        self.pub_system_status = self.create_publisher(String, '/recovery/status', 10)

        # Main Control Loop Timer synchronized with cbf_inner_loop_hz
        control_period_s = 1.0 / max(1.0, float(self.cbf_inner_loop_hz))
        self.timer = self.create_timer(control_period_s, self.control_loop)

        self.get_logger().info("Recovery Controller Node Initialized. Awaiting sensors and heartbeat...")

    # -----------------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------------
    def cb_semantic_cached_objects(self, msg: String):
        """Cache verified permanent semantic objects (transformed into ICP-corrected frame) across comm loss."""
        if not msg.data:
            return
        self.semantic_cache.update_from_json(
            msg.data,
            corr_se2=(self.corr_x, self.corr_y, self.corr_th),
            logger=self.get_logger(),
        )

    def _get_active_semantic_channels(
        self,
    ) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
        """Return (ch1_obstacles, ch2_hazards) when semantic RL or 4-Ch ONNX is active, else ([], [])."""
        has_4ch_rl = self.context_rl_policy is not None and self.context_rl_policy.patch_channels >= 4
        if not (self.use_semantic_rl_policy or has_4ch_rl):
            return [], []
        return self.semantic_cache.get_channel_tuples()

    def _get_rl_actual_velocity(self) -> Tuple[float, float]:
        """Return ICP scan-based velocity when enable_lidar_slip_residual is active and non-degenerate, else wheel odom."""
        if self.enable_lidar_slip_residual and (not self.icp_degenerate) and self._prev_slip_scan_pts is not None:
            return float(self.scan_vel_v), float(self.scan_vel_w)
        return float(self.actual_v), float(self.actual_w)

    def _compute_homotopy_rl_command(
        self,
        v_base: float,
        w_base: float,
        alpha_rl: float,
        target_xy: Tuple[float, float],
        step_norm: float,
        sem_obs: Sequence[Tuple[float, float, float]],
        sem_haz: Sequence[Tuple[float, float, float]],
        sem_prox: float,
        left_min: float,
        right_min: float,
    ) -> Tuple[float, float, float]:
        """
        Compute RL / homotopy-blended nominal command (v_cmd, w_cmd, g_res).
        Evaluates homotopy activation OUTSIDE `alpha_rl > 0.0` so narrow passages (`alpha_rl == 0.0`)
        still transition smoothly via `compute_homotopy_blend` when Ch1/Ch2/Ch3 are active under CBF.
        Uses cached `context_rl_policy.last_ch123_peak` to avoid duplicate semantic projections.
        """
        act_v_rl, act_w_rl = self._get_rl_actual_velocity()
        homotopy_enabled = bool(self.enable_cbf_safety_shield) and (self.context_rl_policy.patch_channels >= 3)
        has_ch3 = (
            self.context_rl_policy.patch_channels >= 4
            and self.latest_dynamic_ch3 is not None
            and float(np.max(np.abs(self.latest_dynamic_ch3))) > 1e-4
        )
        has_sem_or_dyn = bool(sem_obs) or bool(sem_haz) or has_ch3

        if alpha_rl > 0.0 or (homotopy_enabled and has_sem_or_dyn):
            v_rl, w_rl = self.context_rl_policy.predict(
                scan_ranges=self.latest_raw_ranges,
                rx=self.current_x,
                ry=self.current_y,
                rth=self.current_theta,
                tx=target_xy[0],
                ty=target_xy[1],
                cmd_v=self.last_cmd_v,
                cmd_w=self.last_cmd_w,
                act_v=act_v_rl,
                act_w=act_w_rl,
                step_norm=step_norm,
                semantic_obstacles=sem_obs,
                semantic_hazards=sem_haz,
                dynamic_ch3=self.latest_dynamic_ch3,
                angle_min=self.latest_scan_angle_min,
                angle_increment=self.latest_scan_angle_increment,
            )
            if self.enable_cbf_safety_shield:
                ch123_peak = float(self.context_rl_policy.last_ch123_peak) if homotopy_enabled else 0.0
                return compute_homotopy_blend(
                    v_base=v_base,
                    w_base=w_base,
                    v_rl=v_rl,
                    w_rl=w_rl,
                    alpha_rl=alpha_rl,
                    ch123_peak=ch123_peak,
                    v_min=self.cbf_v_min,
                    v_max=self.v_max,
                    w_max=self.w_max,
                )
            alpha_astar = 1.0 - alpha_rl
            v_blend = max(-0.10, min(self.v_max, alpha_rl * v_rl + alpha_astar * v_base))
            w_blend = max(-self.w_max, min(self.w_max, alpha_rl * w_rl + alpha_astar * w_base))
            wall_push_guard = (
                (left_min < 0.32 and w_rl > w_base + 0.05)
                or (right_min < 0.32 and w_rl < w_base - 0.05)
            )
            opp_steer_conflict = (w_rl * w_base < -0.04) and (abs(w_rl - w_base) > 0.50)
            if self.is_phase2b and wall_push_guard:
                return min(v_blend, v_base), w_base, 0.0
            if self.is_phase2b and opp_steer_conflict:
                v_out = (1.0 - sem_prox) * min(v_blend, v_base) + sem_prox * v_blend
                w_out = (1.0 - sem_prox) * w_base + sem_prox * w_blend
                return v_out, w_out, 0.0
            return v_blend, w_blend, 0.0

        self.context_rl_policy.observe(
            scan_ranges=self.latest_raw_ranges,
            rx=self.current_x,
            ry=self.current_y,
            rth=self.current_theta,
            tx=target_xy[0],
            ty=target_xy[1],
            cmd_v=self.last_cmd_v,
            cmd_w=self.last_cmd_w,
            act_v=act_v_rl,
            act_w=act_w_rl,
            step_norm=step_norm,
            semantic_obstacles=sem_obs,
            semantic_hazards=sem_haz,
            dynamic_ch3=self.latest_dynamic_ch3,
            angle_min=self.latest_scan_angle_min,
            angle_increment=self.latest_scan_angle_increment,
        )
        return float(v_base), float(w_base), 0.0

    def _update_dynamic_tracker(self, now_sec: float) -> None:
        """
        Update Stage 1 dynamic tracker using 5 Hz scan.header.stamp gating and
        mid-tick damped-KF extrapolation (10 Hz / 20 Hz). Must run before
        local_planner.observe/step clears observed_free at new scan hit cells.
        """
        if self.dynamic_tracker is None or not self.odom_initialized or not self.latest_raw_ranges:
            return
        stamp_sec = self.latest_scan_stamp_sec if self.latest_scan_stamp_sec is not None else now_sec
        ext_free = None
        ext_ox = 0
        ext_oy = 0
        ext_res = self.local_grid_res_m
        if self.local_planner is not None and self.local_planner.grid.initialized:
            ext_free = self.local_planner.grid.observed_free
            ext_ox = self.local_planner.grid.origin_ix
            ext_oy = self.local_planner.grid.origin_iy
            ext_res = self.local_planner.grid.res
        srx, sry, srth = (
            self.latest_scan_pose
            if self.latest_scan_pose is not None
            else (self.current_x, self.current_y, self.current_theta)
        )
        self.latest_dynamic_tracks = self.dynamic_tracker.update(
            rx=srx,
            ry=sry,
            rth=srth,
            raw_ranges=self.latest_raw_ranges,
            scan_stamp_sec=stamp_sec,
            now_sec=now_sec,
            omega=self.actual_w,
            angle_min=self.latest_scan_angle_min,
            angle_increment=self.latest_scan_angle_increment,
            external_observed_free=ext_free,
            ext_origin_ix=ext_ox,
            ext_origin_iy=ext_oy,
            ext_res=ext_res,
            spike_ratio=float(getattr(self, "latest_scan_spike_ratio", 0.0)),
        )
        self.latest_dynamic_ch3 = self.dynamic_tracker.project_ch3(
            rx=self.current_x,
            ry=self.current_y,
            rth=self.current_theta,
            now_sec=now_sec,
        )

    def _dynamic_approach_active(self) -> bool:
        """True when a confirmed dynamic track is near the robot and closing on it."""
        for t in self.latest_dynamic_tracks:
            if not t.is_confirmed:
                continue
            dx = t.x - self.current_x
            dy = t.y - self.current_y
            if math.hypot(dx, dy) > 1.5:
                continue
            if t.speed < 0.05:
                continue
            if dx * t.vx + dy * t.vy < 0.0:
                return True
        return False

    def _headon_dynamic_closing_active(
        self,
        confirmed_tracks: Sequence[object],
        v_ref: float = 0.0,
        max_dist_m: float = 0.85,
        min_closing_mps: float = 0.25,
        min_obs_approach_mps: float = 0.05,
        min_front_cos: float = 0.50,
    ) -> bool:
        """
        True when a confirmed dynamic track is close ahead (d_dyn < 0.85 m within +/-60 deg
        of robot heading) and closing fast head-on (relative closing_speed > 0.25 m/s while
        the obstacle itself is approaching > 0.05 m/s; never triggers when receding).
        """
        if not bool(getattr(self, "enable_spatiotemporal_dynamic_control", True)):
            return False
        if not confirmed_tracks:
            return False
        fx = math.cos(self.current_theta)
        fy = math.sin(self.current_theta)
        v_rob = max(0.0, float(getattr(self, "actual_v", 0.0)), float(v_ref))
        vrx = v_rob * fx
        vry = v_rob * fy
        for t in confirmed_tracks:
            if not getattr(t, "is_confirmed", False):
                continue
            dx = float(t.x) - self.current_x
            dy = float(t.y) - self.current_y
            d_dyn = math.hypot(dx, dy)
            if d_dyn <= 1e-4 or d_dyn >= max_dist_m:
                continue
            ux = dx / d_dyn
            uy = dy / d_dyn
            if (ux * fx + uy * fy) < min_front_cos:
                continue
            tvx = float(t.vx)
            tvy = float(t.vy)
            obs_approach = -(ux * tvx + uy * tvy)
            if obs_approach <= min_obs_approach_mps:
                continue
            closing_speed = (vrx - tvx) * ux + (vry - tvy) * uy
            if closing_speed > min_closing_mps:
                return True
        return False

    def _apply_cbf_safety_shield(
        self,
        v_nom: float,
        w_nom: float,
        now_sec: Optional[float] = None,
        prefer_no_reverse: bool = False,
        suppress_tangential_steer: bool = False,
    ) -> Tuple[float, float]:
        """Filter nominal control (v_nom, w_nom) through Stage 2 2-Circle Soft-CBF-QP."""
        if self.cbf_shield is None or not self.odom_initialized:
            return float(v_nom), float(w_nom)
        ch1_obs, ch2_haz = self.semantic_cache.get_cbf_channel_tuples()
        dt_step = 1.0 / max(1.0, self.cbf_inner_loop_hz)
        has_ranges = bool(self.latest_raw_ranges)
        res = self.cbf_shield.filter(
            v_nom=float(v_nom),
            w_nom=float(w_nom),
            rx=self.current_x,
            ry=self.current_y,
            rth=self.current_theta,
            dt=dt_step,
            scan_ranges=(self.latest_raw_ranges if has_ranges else None),
            angle_min=self.latest_scan_angle_min,
            angle_increment=self.latest_scan_angle_increment,
            scan_pts_sensor=(None if has_ranges else self.latest_scan_pts),
            scan_pose=self.latest_scan_pose,
            dynamic_obstacles=(self.latest_dynamic_tracks if self.latest_dynamic_tracks else None),
            ch1_obstacles=(ch1_obs if ch1_obs else None),
            ch2_hazards=(ch2_haz if ch2_haz else None),
            lifo_waypoints=self.waypoint_stack,
            prefer_no_reverse=prefer_no_reverse,
            suppress_tangential_steer=suppress_tangential_steer,
        )
        self.latest_cbf_result = res
        return res.v, res.w

    def _compute_semantic_proximity(
        self,
        sem_obs: Sequence[Tuple[float, float, float]],
        sem_haz: Sequence[Tuple[float, float, float]],
        outer_reach_m: float = 1.20,
        inner_reach_m: float = 0.65,
        target_xy: Optional[Tuple[float, float]] = None,
    ) -> float:
        """
        Return continuous semantic proximity in [0.0, 1.0] from nearest cached obstacle/hazard
        clearance with forward-hemisphere / receding-object directional attenuation so rear
        objects already behind the robot (e.g., survivor at start of return) do not falsely
        saturate sem_prox and suppress nominal consistency guards.
        """
        max_prox = 0.0
        span = max(1e-6, outer_reach_m - inner_reach_m)
        fx = math.cos(float(self.current_theta))
        fy = math.sin(float(self.current_theta))
        has_tgt = False
        tx_u = 0.0
        ty_u = 0.0
        if target_xy is not None:
            tdx = float(target_xy[0]) - float(self.current_x)
            tdy = float(target_xy[1]) - float(self.current_y)
            td = math.hypot(tdx, tdy)
            if td > 1e-4:
                tx_u = tdx / td
                ty_u = tdy / td
                has_tgt = True
        for items in (sem_obs, sem_haz):
            for ox, oy, rad in items:
                dx = float(ox) - float(self.current_x)
                dy = float(oy) - float(self.current_y)
                dist = math.hypot(dx, dy)
                clearance = dist - float(rad)
                if clearance < outer_reach_m:
                    base_prox = max(0.0, min(1.0, (outer_reach_m - clearance) / span))
                    if clearance <= 0.15 or dist <= 1e-4:
                        dir_w = 1.0
                    else:
                        ux = dx / dist
                        uy = dy / dist
                        cos_hdg = ux * fx + uy * fy
                        if has_tgt:
                            cos_tgt = ux * tx_u + uy * ty_u
                            eff_cos = min(cos_hdg, cos_tgt) if cos_hdg < 0.25 else cos_hdg
                        else:
                            eff_cos = cos_hdg
                        if eff_cos >= 0.0:
                            dir_w = 1.0
                        elif eff_cos <= -0.35:
                            dir_w = 0.0
                        else:
                            dir_w = (eff_cos + 0.35) / 0.35
                    prox = base_prox * dir_w
                    if prox > max_prox:
                        max_prox = prox
        return float(max_prox)

    def _has_nearby_semantic_item(
        self,
        sem_obs: Sequence[Tuple[float, float, float]],
        sem_haz: Sequence[Tuple[float, float, float]],
        reach_m: float = 0.95,
    ) -> bool:
        return self._compute_semantic_proximity(
            sem_obs, sem_haz, outer_reach_m=reach_m, inner_reach_m=max(0.10, reach_m - 0.30)
        ) > 0.0

    def _reset_trial_runtime_state(
        self,
        clear_stack: bool = True,
        clear_sensors: bool = False,
        reset_corr: bool = True,
        await_fresh_odom: bool = False,
    ) -> None:
        """Unified inter-trial / link-restore runtime state reset helper."""
        self.stop_robot()
        get_clock_fn = getattr(self, "get_clock", None)
        if callable(get_clock_fn):
            try:
                self.last_heartbeat_time = get_clock_fn().now()
            except Exception:
                pass

        if await_fresh_odom:
            self.odom_initialized = False

        effective_reset_corr = bool(reset_corr) and bool(clear_stack)
        if effective_reset_corr:
            self.corr_x = 0.0
            self.corr_y = 0.0
            self.corr_th = 0.0
            self.scan_match_count = 0
            self.current_x = self.raw_odom_x
            self.current_y = self.raw_odom_y
            self.current_theta = self.raw_odom_theta

        if clear_sensors:
            self.current_gas_level = 0.0
            self.last_gas_time = None
            self.obstacle_detected = False
            self.td3_obstacle_detected = False
            self.latest_raw_ranges = []
            self.latest_scan_pts = None
            self.latest_scan_spike_ratio = 0.0

        if clear_stack:
            self.waypoint_stack.clear()
            if self.odom_initialized:
                pts_copy = self.latest_scan_pts.copy() if self.latest_scan_pts is not None else None
                self.start_pose = Waypoint(self.current_x, self.current_y, self.current_theta, pts_copy)
                self.start_scan_pts = pts_copy
                self.waypoint_stack.append(self.start_pose)
            else:
                self.start_pose = None
                self.start_scan_pts = None
            self.last_logged_x = self.current_x
            self.last_logged_y = self.current_y
            self.last_logged_theta = self.current_theta

        self.previous_state = SystemState.INIT
        self.active_target_wp = None
        self.aruco_docking_complete = False
        self.aruco_start_time = None
        self.td3_avoid_step = 0
        self.fault_reason = "NONE"
        self._link_restored_timer = None
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0
        self.actual_v = 0.0
        self.actual_w = 0.0
        self.recovery_step_count = 0
        self._prev_escape_count = 0
        self._rl_cooldown_until = 0.0
        self._rl_lockout_anchor = None
        self._rl_lockout_until = 0.0
        self._prev_front_min = None
        self._prev_front_time = None
        self._prev_alpha_rl = 0.0
        self._dynamic_hold_until = 0.0
        self._dynamic_hold_cooldown_until = 0.0
        self._prev_slip_scan_pts = None
        self._prev_slip_scan_stamp = None
        self._icp_done_scan_stamp = None
        self._icp_section_ms_total = 0.0
        self._icp_section_ms_max = 0.0
        self._icp_section_count = 0
        self._icp_section_last_log = 0.0
        self._icp_run_count = 0
        self._icp_skip_count = 0
        self._start_icp_diag_count = 0
        self.last_icp_stats = None
        self.icp_degenerate = False
        self.scan_vel_v = 0.0
        self.scan_vel_w = 0.0
        if clear_stack:
            self.semantic_cache.clear()
            if self.local_planner is not None:
                self.local_planner.reset()
        if self.context_rl_policy is not None:
            self.context_rl_policy.reset()
        if self.dynamic_tracker is not None:
            self.dynamic_tracker.reset()
        if self.cbf_shield is not None:
            self.cbf_shield.reset()
        cross_sched = getattr(self, "crossing_scheduler", None)
        if cross_sched is not None:
            cross_sched.reset()
        self.latest_cbf_result = None
        self.latest_scan_stamp_sec = None
        self.latest_scan_pose = None
        self.latest_dynamic_tracks = []
        self.latest_dynamic_ch3 = None

    def cb_reset_episode(self, msg: Bool):
        """Reset inter-trial odometry correction, waypoint stack, and local planner cleanly."""
        if not msg.data:
            return
        self._reset_trial_runtime_state(
            clear_stack=True,
            clear_sensors=True,
            reset_corr=True,
            await_fresh_odom=True,
        )
        self.state = (
            SystemState.NORMAL_TELEOP
            if (self.odom_initialized and self.heartbeat_received_once)
            else SystemState.INIT
        )
        self.get_logger().info(">>> Episode state reset via /recovery/reset_episode. <<<")

    def cb_odometry(self, msg: Odometry):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self.actual_v = float(msg.twist.twist.linear.x)
        self.actual_w = float(msg.twist.twist.angular.z)
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
            pts_copy = self.latest_scan_pts.copy() if self.latest_scan_pts is not None else None
            self.start_pose = Waypoint(self.current_x, self.current_y, self.current_theta, scan_pts=pts_copy)
            self.start_scan_pts = pts_copy
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
        self.current_gas_level = float(msg.data)
        self.last_gas_time = self.get_clock().now()
        if self.current_gas_level >= self.gas_threshold and self.odom_initialized:
            self.semantic_cache.record_local_gas_hazard(
                x=self.current_x,
                y=self.current_y,
                gas_level=self.current_gas_level,
            )

    @staticmethod
    def _despike_lidar_ranges(ranges: np.ndarray) -> np.ndarray:
        """
        Phase 2-B Layer-0 Spatio-Temporal Angular-Cluster Continuity Filter:
        Removes isolated 1-2 ray airborne smoke/dust backscatter spikes (< 2.2m)
        that lack spatial support (< 3 agreeing rays within +/-3 deg window),
        while preserving solid obstacles and ultra-close positive returns (0 < r < 0.10 -> 0.12).
        """
        n = len(ranges)
        if n < 7:
            return ranges
        pos_close = np.isfinite(ranges) & (ranges > 0.0) & (ranges < 0.10)
        clean = np.where(
            np.isfinite(ranges) & (ranges >= 0.10),
            ranges,
            np.where(pos_close, 0.12, 3.5),
        ).astype(np.float32)
        # Build 7-ray circular neighborhood matrix (7, N)
        neighbors = np.stack([np.roll(clean, shift) for shift in range(-3, 4)], axis=0)
        close_count = np.sum(np.abs(neighbors - clean[None, :]) < 0.24, axis=0)
        spike_mask = (clean < 2.20) & (close_count < 3) & (~pos_close) & np.isfinite(ranges) & (ranges > 0.0)
        if np.any(spike_mask):
            bg_val = np.percentile(neighbors, 75, axis=0).astype(np.float32)
            out = ranges.copy()
            out[spike_mask] = bg_val[spike_mask]
            return out
        return ranges

    def cb_scan(self, msg: LaserScan):
        """
        1. Fast vectorized obstacle detection within forward +/- lidar_fov.
        2. Cartesian point extraction for 2D LiDAR Scan Matching odometry correction.
        """
        if not msg.ranges:
            return

        ranges = np.array(msg.ranges, dtype=np.float32)
        self._scan_tick_seq += 1

        # Simulate fire-smoke / airborne dust LiDAR backscatter & dropout during emergency return
        if (
            self.smoke_lidar_noise_prob > 0.0
            and self.state not in (SystemState.INIT, SystemState.NORMAL_TELEOP)
        ):
            rng = np.random.default_rng(20261115 + self._scan_tick_seq * 131)
            n_rays = len(ranges)
            # Isolated single-ray backscatter spikes separated by >= 6 deg
            n_spikes = max(4, int(round(n_rays * self.smoke_lidar_noise_prob * 0.5)))
            base_idx = int(rng.integers(0, 6))
            stride = max(6, n_rays // max(1, n_spikes))
            spike_indices = (base_idx + np.arange(0, n_rays, stride)) % n_rays
            spike_dists = rng.uniform(0.18, 0.68, size=len(spike_indices)).astype(np.float32)
            ranges[spike_indices] = np.minimum(ranges[spike_indices], spike_dists)
            # Beam attenuation dropouts (~6%)
            drop_mask = rng.random(n_rays) < 0.06
            drop_mask[spike_indices] = False
            ranges[drop_mask] = np.inf

        if self.enable_lidar_despike:
            raw_pre_despike = ranges
            ranges = self._despike_lidar_ranges(ranges)
            diff_mask = np.isfinite(raw_pre_despike) & (raw_pre_despike > 0.0) & (
                (~np.isfinite(ranges)) | (np.abs(ranges - raw_pre_despike) > 1e-3)
            )
            self.latest_scan_spike_ratio = float(np.count_nonzero(diff_mask)) / float(max(1, len(ranges)))
        else:
            self.latest_scan_spike_ratio = 0.0

        self.latest_raw_ranges = list(ranges)
        if self.odom_initialized:
            self.latest_scan_pose = (self.current_x, self.current_y, self.current_theta)
        stamp = getattr(getattr(msg, 'header', None), 'stamp', None)
        if stamp is not None and (int(getattr(stamp, 'sec', 0)) != 0 or int(getattr(stamp, 'nanosec', 0)) != 0):
            self.latest_scan_stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        else:
            try:
                self.latest_scan_stamp_sec = self.get_clock().now().nanoseconds * 1e-9
            except Exception:
                self.latest_scan_stamp_sec = None
        self.latest_scan_angle_min = float(getattr(msg, 'angle_min', 0.0))
        ang_inc = float(getattr(msg, 'angle_increment', 0.0))
        self.latest_scan_angle_increment = ang_inc if ang_inc != 0.0 else None
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        # Preserve positive ultra-close returns (0 < r < lidar_min_dist -> lidar_min_dist) while excluding 0.0 dropouts
        eff_ranges = np.where(
            np.isfinite(ranges) & (ranges > 0.0) & (ranges < self.lidar_min_dist),
            np.float32(self.lidar_min_dist),
            ranges,
        )

        # 1. Forward obstacle check
        norm_angles = np.array([normalize_angle(a) for a in angles], dtype=np.float32)
        fwd_mask = (np.abs(norm_angles) <= self.lidar_fov) & \
                   (eff_ranges >= self.lidar_min_dist) & (eff_ranges <= self.lidar_danger_dist) & \
                   np.isfinite(eff_ranges)
        self.obstacle_detected = bool(np.any(fwd_mask))

        td3_mask = (np.abs(norm_angles) <= self.lidar_fov) & \
                   (eff_ranges >= self.lidar_min_dist) & (eff_ranges <= self.td3_clearance_dist) & \
                   np.isfinite(eff_ranges)
        self.td3_obstacle_detected = bool(np.any(td3_mask))

        # 2. Extract 2D Cartesian scan points (downsample step=2 for fast ~180 points)
        sub_ranges = eff_ranges[::2]
        sub_angles = norm_angles[::2]
        valid_mask = (sub_ranges >= self.lidar_min_dist) & (sub_ranges <= 3.5) & np.isfinite(sub_ranges)

        if np.sum(valid_mask) >= 20:
            r_valid = sub_ranges[valid_mask]
            a_valid = sub_angles[valid_mask]
            cur_pts = np.column_stack((r_valid * np.cos(a_valid), r_valid * np.sin(a_valid))).astype(np.float32)
            if self.enable_lidar_slip_residual and self._prev_slip_scan_pts is not None and self._prev_slip_scan_stamp is not None and self.latest_scan_stamp_sec is not None:
                dt_scan = self.latest_scan_stamp_sec - self._prev_slip_scan_stamp
                v_s, w_s, is_deg = estimate_scan_velocity_icp(
                    self._prev_slip_scan_pts,
                    cur_pts,
                    dt_scan,
                    prev_vw=(self.actual_v, self.actual_w),
                )
                self.scan_vel_v = v_s
                self.scan_vel_w = w_s
                self.icp_degenerate = is_deg
            self._prev_slip_scan_pts = cur_pts.copy()
            self._prev_slip_scan_stamp = self.latest_scan_stamp_sec
            self.latest_scan_pts = cur_pts

            if self.start_scan_pts is None and self.odom_initialized:
                self.start_scan_pts = self.latest_scan_pts.copy()
                if self.start_pose is not None and self.start_pose.scan_pts is None:
                    self.start_pose.scan_pts = self.start_scan_pts
        else:
            # Invalidate stale scan points so old scans never stamp ghost obstacles at a new pose
            self.latest_scan_pts = None

    def cb_remote_teleop(self, msg: Twist):
        # Forward remote teleop to robot motors ONLY in normal teleop mode
        if self.state == SystemState.NORMAL_TELEOP:
            self.last_cmd_v = float(msg.linear.x)
            self.last_cmd_w = float(msg.angular.z)
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

        # Publish Diagnostic Status String with ICP Corrected Pose
        status_msg = String()
        status_data = (
            f"STATE:{self.state}|FAULTS:{self.fault_reason}|STACK:{len(self.waypoint_stack)}"
            f"|POSE:{self.current_x:.3f},{self.current_y:.3f},{self.current_theta:.3f}"
        )
        if self.cbf_shield is not None:
            st = self.cbf_shield.get_stats()
            status_data += (
                f"|CBF:{int(st['total_steps'])},{int(st['intervention_steps'])},"
                f"{int(st['v_sign_reversals'])},{int(st['v_stall_toggles'])},"
                f"{int(st['reverse_steps'])},{st['max_slack_seen']:.4f},{st['mean_solve_ms']:.3f},"
                f"{st['max_solve_ms']:.3f},{int(st['mq2_escape_steps'])}"
            )
        n_conf_dyn = sum(
            1 for t in (getattr(self, "latest_dynamic_tracks", None) or ())
            if getattr(t, "is_confirmed", False)
        )
        cross_sched_st = getattr(self, "crossing_scheduler", None)
        if cross_sched_st is None:
            cross_mode = "NONE"
        else:
            last_dec_st = getattr(cross_sched_st, "last_decision", None)
            if last_dec_st is not None and hasattr(last_dec_st, "mode"):
                cross_mode = str(last_dec_st.mode)
            else:
                cross_mode = str(
                    getattr(
                        cross_sched_st,
                        "_mode",
                        "ACTIVE" if getattr(cross_sched_st, "active", False) else "INACTIVE",
                    )
                )
        status_data += f"|DYN:{n_conf_dyn}|CROSS:{cross_mode}"
        status_msg.data = status_data
        self.pub_system_status.publish(status_msg)

        # 0. Global Link Restoration Check
        # Exclude active GAS_-triggered recovery from auto-cancellation until MISSION_COMPLETE or explicit reset.
        if self.state not in [SystemState.NORMAL_TELEOP, SystemState.INIT]:
            is_active_gas_recovery = (
                str(self.fault_reason).startswith("GAS_")
                and self.state != SystemState.MISSION_COMPLETE
            )
            if (not is_active_gas_recovery) and dt_heartbeat < 0.5 and self.current_gas_level < self.gas_threshold:
                if not hasattr(self, '_link_restored_timer') or self._link_restored_timer is None:
                    self._link_restored_timer = now
                elif (now - self._link_restored_timer).nanoseconds / 1e9 > 1.2:
                    dist_home_now = (
                        math.hypot(self.start_pose.x - self.current_x, self.start_pose.y - self.current_y)
                        if self.start_pose is not None
                        else 0.0
                    )
                    should_clear_stack = (
                        self.state == SystemState.MISSION_COMPLETE
                        or self.clear_stack_on_link_restore
                        or dist_home_now <= self.docking_dist_th
                    )
                    self.state = SystemState.NORMAL_TELEOP
                    self._reset_trial_runtime_state(
                        clear_stack=should_clear_stack,
                        clear_sensors=False,
                        reset_corr=bool(self.reset_corr_on_episode) and should_clear_stack,
                    )
                    self.get_logger().info(
                        f">>> Heartbeat link restored! Controller reset to NORMAL_TELEOP mode (clear_stack={should_clear_stack}). <<<"
                    )
            else:
                self._link_restored_timer = None

        # 1. State: INIT
        if self.state == SystemState.INIT:
            if self.odom_initialized and self.heartbeat_received_once:
                self.state = SystemState.NORMAL_TELEOP
                self.get_logger().info("System ready. Transitioned to NORMAL_TELEOP mode.")
            return

        # 2. Watchdog Fault Detection (Active in NORMAL_TELEOP)
        if self.state == SystemState.NORMAL_TELEOP:
            if self.dynamic_tracker is not None and self.odom_initialized and self.latest_raw_ranges:
                self._update_dynamic_tracker(now.nanoseconds / 1e9)
            if (
                self.recovery_strategy in ('proposed', 'proposed_rl')
                and self.local_planner is not None
                and self.odom_initialized
                and self.latest_scan_pts is not None
            ):
                self.local_planner.observe(
                    self.current_x, self.current_y, self.current_theta, self.latest_scan_pts
                )
            if (
                self.enable_teleop_rl_warmup
                and self.context_rl_policy is not None
                and self.odom_initialized
                and self.latest_raw_ranges
            ):
                sem_obs_w, sem_haz_w = self._get_active_semantic_channels()
                tx_w = self.start_pose.x if self.start_pose is not None else self.current_x
                ty_w = self.start_pose.y if self.start_pose is not None else self.current_y
                act_v_w, act_w_w = self._get_rl_actual_velocity()
                self.context_rl_policy.observe(
                    scan_ranges=self.latest_raw_ranges,
                    rx=self.current_x,
                    ry=self.current_y,
                    rth=self.current_theta,
                    tx=tx_w,
                    ty=ty_w,
                    cmd_v=self.last_cmd_v,
                    cmd_w=self.last_cmd_w,
                    act_v=act_v_w,
                    act_w=act_w_w,
                    step_norm=0.0,
                    semantic_obstacles=sem_obs_w,
                    semantic_hazards=sem_haz_w,
                    dynamic_ch3=self.latest_dynamic_ch3,
                    angle_min=self.latest_scan_angle_min,
                    angle_increment=self.latest_scan_angle_increment,
                )
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
            # Discard any waypoints in the stack that are already within reach radius of current position
            while len(self.waypoint_stack) > 1 and self.waypoint_stack[-1].distance_to(self.current_x, self.current_y) <= self.wp_reach_radius:
                discarded = self.waypoint_stack.pop()
                self.get_logger().info(f"Skipping already-reached WP ({discarded.x:.2f}, {discarded.y:.2f})")

            # Verify and select immediate backtrack waypoint
            if not self.waypoint_stack:
                self.get_logger().warn("Waypoint stack is empty at emergency stop! Directly requesting ArUco docking.")
                self.state = SystemState.HANDOVER_TO_ARUCO
            else:
                self.active_target_wp = self.waypoint_stack[-1]
                dx = self.active_target_wp.x - self.current_x
                dy = self.active_target_wp.y - self.current_y
                target_heading = math.atan2(dy, dx)
                heading_diff = abs(normalize_angle(target_heading - self.current_theta))

                if heading_diff > self.heading_tol:
                    self.state = SystemState.BACKTRACKING_ROTATE
                else:
                    self.state = SystemState.BACKTRACKING_HYBRID if self.enable_hybrid else SystemState.BACKTRACKING_TRANSLATE

                self.get_logger().info(
                    f"Selected initial backtrack WP: ({self.active_target_wp.x:.2f}, {self.active_target_wp.y:.2f}). "
                    f"Remaining in stack: {len(self.waypoint_stack)} | Initial state: {self.state}"
                )
            return

        # 3b. Proposed / Proposed_RL Strategy: Rolling 2D Grid A* + Topological Shortcut (+ Context RL ONNX)
        if (
            self.recovery_strategy in ('proposed', 'proposed_rl')
            and self.local_planner is not None
            and self.state in [
                SystemState.BACKTRACKING_ROTATE,
                SystemState.BACKTRACKING_TRANSLATE,
                SystemState.BACKTRACKING_HYBRID,
                SystemState.LOCAL_TD3_AVOIDANCE,
                SystemState.OBSTACLE_HOLD,
            ]
        ):
            now_sec = now.nanoseconds / 1e9
            if self.active_target_wp is None:
                self.active_target_wp = (
                    self.waypoint_stack[-1] if len(self.waypoint_stack) > 0 else self.start_pose
                )
            _sec_t0 = time.perf_counter()
            self.perform_scan_matching_correction()
            if self.dynamic_tracker is not None:
                self._update_dynamic_tracker(now_sec)
            scan_corrupted = float(getattr(self, "latest_scan_spike_ratio", 0.0)) >= 0.03
            confirmed_dyn_tracks = (
                [t for t in self.latest_dynamic_tracks if getattr(t, "is_confirmed", False)]
                if (self.latest_dynamic_tracks and not scan_corrupted)
                else []
            )
            spatiotemporal_enabled = bool(
                getattr(self, "enable_spatiotemporal_dynamic_control", True)
            )
            if self.enable_semantic_astar_cost:
                astar_sem_obs, astar_sem_haz = self.semantic_cache.get_channel_tuples()
                astar_dyn_tracks = confirmed_dyn_tracks if confirmed_dyn_tracks else None
            else:
                astar_sem_obs, astar_sem_haz = None, None
                astar_dyn_tracks = (
                    confirmed_dyn_tracks
                    if (spatiotemporal_enabled and confirmed_dyn_tracks)
                    else None
                )
            status_tag, v_cmd, w_cmd, subgoal_xy = self.local_planner.step(
                rx=self.current_x,
                ry=self.current_y,
                rth=self.current_theta,
                waypoint_stack=self.waypoint_stack,
                start_pose=self.start_pose,
                scan_pts=self.latest_scan_pts,
                v_max=self.v_max,
                w_max=self.w_max,
                now_sec=now_sec,
                wp_reach_radius=self.wp_reach_radius,
                docking_dist_th=self.docking_dist_th,
                raw_ranges=self.latest_raw_ranges,
                semantic_obstacles=astar_sem_obs,
                semantic_hazards=astar_sem_haz,
                dynamic_obstacles=astar_dyn_tracks,
            )
            _sec_ms = (time.perf_counter() - _sec_t0) * 1e3
            self._icp_section_ms_total += _sec_ms
            if _sec_ms > self._icp_section_ms_max:
                self._icp_section_ms_max = _sec_ms
            self._icp_section_count += 1
            if _sec_ms > 50.0 and (now_sec - self._icp_section_last_log) > 2.0:
                self._icp_section_last_log = now_sec
                self.get_logger().warn(
                    f"[ICP/TRK/A*] section slow: {_sec_ms:.1f} ms "
                    f"(mean={self._icp_section_ms_total / max(1, self._icp_section_count):.1f} ms "
                    f"over {self._icp_section_count} loops, max={self._icp_section_ms_max:.1f} ms, "
                    f"icp run={self._icp_run_count} skip={self._icp_skip_count})"
                )
            if status_tag == "ARRIVED":
                cross_sched = getattr(self, "crossing_scheduler", None)
                if cross_sched is not None and cross_sched.active:
                    cross_sched.reset()
                dist_to_origin = (
                    math.hypot(self.start_pose.x - self.current_x, self.start_pose.y - self.current_y)
                    if self.start_pose is not None
                    else 0.0
                )
                if self.start_pose is None or dist_to_origin <= self.wp_reach_radius:
                    self.waypoint_stack.clear()
                    self.active_target_wp = None
                    self.stop_robot()
                    self.state = SystemState.MISSION_COMPLETE
                    return
                if self.use_aruco_docking and dist_to_origin <= self.docking_dist_th:
                    self.waypoint_stack.clear()
                    self.active_target_wp = None
                    self.stop_robot()
                    self.state = SystemState.HANDOVER_TO_ARUCO
                    return
                # Final precision approach to start_pose wrapped by CBF shield (never falls through to unshielded TD3)
                dx_a = self.start_pose.x - self.current_x
                dy_a = self.start_pose.y - self.current_y
                ang_a = normalize_angle(math.atan2(dy_a, dx_a) - self.current_theta)
                if abs(ang_a) > 0.85:
                    v_cmd = 0.0
                    w_cmd = max(-self.w_max, min(self.w_max, self.kp_ang * ang_a))
                else:
                    v_cmd = max(0.04, min(self.v_max, self.kp_lin * dist_to_origin)) * max(0.2, math.cos(ang_a))
                    w_cmd = max(-self.w_max, min(self.w_max, self.kp_ang * ang_a))
                v_pre_cbf, w_pre_cbf = float(v_cmd), float(w_cmd)
                if self.cbf_shield is not None:
                    v_cmd, w_cmd = self._apply_cbf_safety_shield(v_cmd, w_cmd, now_sec)
                cbf_int = bool(
                    self.cbf_shield is not None
                    and self.latest_cbf_result is not None
                    and self.latest_cbf_result.intervened
                )
                v_cmd, w_cmd = apply_motor_deadzone(
                    v_cmd,
                    w_cmd,
                    min_v=self.min_effective_linear_vel,
                    min_w=self.min_effective_angular_vel,
                    v_max=self.v_max,
                    w_max=self.w_max,
                    min_v_reverse=(self.cbf_v_min if self.enable_cbf_safety_shield else -0.10),
                    suppress_v_boost=(cbf_int and abs(v_cmd) < abs(v_pre_cbf) - 1e-4),
                    suppress_w_boost=(cbf_int and abs(w_cmd) < abs(w_pre_cbf) - 1e-4),
                )
                cmd = Twist()
                cmd.linear.x = float(v_cmd)
                cmd.angular.z = float(w_cmd)
                self.last_cmd_v = cmd.linear.x
                self.last_cmd_w = cmd.angular.z
                self.pub_cmd_vel.publish(cmd)
                return
            else:
                cross_sched = (
                    getattr(self, "crossing_scheduler", None)
                    if spatiotemporal_enabled
                    else None
                )
                cross_dec = None
                if cross_sched is not None:
                    if (
                        (not scan_corrupted)
                        and status_tag in ("ASTAR_TRACK", "ROTATE_ALIGN")
                        and (confirmed_dyn_tracks or cross_sched.active)
                    ):
                        path_world = (
                            self.local_planner.get_last_path_world()
                            if hasattr(self.local_planner, "get_last_path_world")
                            else None
                        )
                        cross_dec = cross_sched.step(
                            rx=self.current_x,
                            ry=self.current_y,
                            rth=self.current_theta,
                            v_max=self.v_max,
                            w_max=self.w_max,
                            now_sec=now_sec,
                            confirmed_tracks=confirmed_dyn_tracks,
                            path_world=path_world,
                            waypoint_stack=self.waypoint_stack,
                            subgoal_xy=subgoal_xy,
                            local_grid=self.local_planner.grid,
                            dynamic_tracker=self.dynamic_tracker,
                            scan_pts=self.latest_scan_pts,
                            scan_corrupted=scan_corrupted,
                        )
                    elif cross_sched.active:
                        cross_sched.reset()

                if cross_dec is not None and cross_dec.active:
                    v_cmd = float(cross_dec.v_ref)
                    w_cmd = float(cross_dec.w_ref)
                    if cross_dec.refresh_watchdog:
                        self.local_planner._anchor_time = now_sec
                        self.local_planner._anchor_x = self.current_x
                        self.local_planner._anchor_y = self.current_y
                    if self.recovery_strategy == 'proposed_rl' and self.context_rl_policy is not None:
                        self.recovery_step_count += 1
                        step_norm = min(1.0, self.recovery_step_count / 320.0)
                        target_xy_c = (
                            cross_dec.target_xy
                            if cross_dec.target_xy is not None
                            else (subgoal_xy if subgoal_xy is not None else (self.current_x, self.current_y))
                        )
                        front_min_c = self.local_planner.last_clearances[0]
                        sem_obs_c, sem_haz_c = self._get_active_semantic_channels()
                        act_v_c, act_w_c = self._get_rl_actual_velocity()
                        self._prev_alpha_rl = 0.0
                        self.context_rl_policy.observe(
                            scan_ranges=self.latest_raw_ranges,
                            rx=self.current_x,
                            ry=self.current_y,
                            rth=self.current_theta,
                            tx=target_xy_c[0],
                            ty=target_xy_c[1],
                            cmd_v=self.last_cmd_v,
                            cmd_w=self.last_cmd_w,
                            act_v=act_v_c,
                            act_w=act_w_c,
                            step_norm=step_norm,
                            semantic_obstacles=sem_obs_c,
                            semantic_hazards=sem_haz_c,
                            dynamic_ch3=self.latest_dynamic_ch3,
                            angle_min=self.latest_scan_angle_min,
                            angle_increment=self.latest_scan_angle_increment,
                        )
                        self._prev_front_min = front_min_c
                        self._prev_front_time = now_sec
                elif self.recovery_strategy == 'proposed_rl' and self.context_rl_policy is not None:
                    self.recovery_step_count += 1
                    step_norm = min(1.0, self.recovery_step_count / 320.0)
                    if subgoal_xy is not None:
                        target_xy = subgoal_xy
                    elif len(self.waypoint_stack) > 0:
                        target_xy = (self.waypoint_stack[-1].x, self.waypoint_stack[-1].y)
                    elif self.start_pose is not None:
                        target_xy = (self.start_pose.x, self.start_pose.y)
                    else:
                        target_xy = (self.current_x, self.current_y)

                    front_min, left_min, right_min, _ = self.local_planner.last_clearances
                    sem_obs, sem_haz = self._get_active_semantic_channels()
                    sem_prox = (
                        self._compute_semantic_proximity(sem_obs, sem_haz, target_xy=target_xy)
                        if (sem_obs or sem_haz)
                        else 0.0
                    )
                    df_dt = 0.0
                    if (
                        (self.is_phase2b or self.enable_dynamic_headon_hold)
                        and self._prev_front_min is not None
                        and self._prev_front_time is not None
                    ):
                        dt_f = now_sec - self._prev_front_time
                        if 0.03 <= dt_f <= 0.50:
                            df_dt = (front_min - self._prev_front_min) / dt_f

                    use_legacy_guards = self.is_phase2b and (not self.enable_cbf_safety_shield)
                    use_dynamic_hold = use_legacy_guards or self.enable_dynamic_headon_hold

                    if use_legacy_guards:
                        cur_esc = self.local_planner.escape_count
                        if cur_esc > self._prev_escape_count:
                            self._rl_cooldown_until = now_sec + 4.5
                            if self.local_planner._repeat_escape_count >= 1:
                                self._rl_lockout_anchor = (self.current_x, self.current_y)
                                self._rl_lockout_until = now_sec + 7.0
                        self._prev_escape_count = cur_esc
                        if self._rl_lockout_anchor is not None:
                            lockout_radius = (
                                0.50 if self.local_planner._repeat_escape_count >= 3 else 1.20
                            )
                            if now_sec >= self._rl_lockout_until or math.hypot(
                                self.current_x - self._rl_lockout_anchor[0],
                                self.current_y - self._rl_lockout_anchor[1],
                            ) >= lockout_radius:
                                self._rl_lockout_anchor = None

                    if use_dynamic_hold:
                        # Dynamic obstacle head-on detection: brief pause instead of backing into corners
                        if (
                            now_sec >= self._dynamic_hold_until
                            and now_sec >= self._dynamic_hold_cooldown_until
                            and max(abs(self.last_cmd_w), abs(self.actual_w)) < 0.35
                            and df_dt < -0.25
                            and 0.20 <= front_min <= 0.55
                        ):
                            self._dynamic_hold_until = now_sec + 1.2
                            self._dynamic_hold_cooldown_until = now_sec + 4.0

                    act_v_rl, act_w_rl = self._get_rl_actual_velocity()
                    if use_dynamic_hold and now_sec < self._dynamic_hold_until and 0.20 <= front_min <= 0.60:
                        v_cmd = 0.0
                        w_cmd = 0.0
                        self._prev_alpha_rl = 0.0
                        self.context_rl_policy.observe(
                            scan_ranges=self.latest_raw_ranges,
                            rx=self.current_x,
                            ry=self.current_y,
                            rth=self.current_theta,
                            tx=target_xy[0],
                            ty=target_xy[1],
                            cmd_v=self.last_cmd_v,
                            cmd_w=self.last_cmd_w,
                            act_v=act_v_rl,
                            act_w=act_w_rl,
                            step_norm=step_norm,
                            semantic_obstacles=sem_obs,
                            semantic_hazards=sem_haz,
                            dynamic_ch3=self.latest_dynamic_ch3,
                            angle_min=self.latest_scan_angle_min,
                            angle_increment=self.latest_scan_angle_increment,
                        )
                    elif status_tag == "ASTAR_TRACK" and subgoal_xy is not None:
                        if use_dynamic_hold and front_min < 0.20:
                            self._dynamic_hold_until = 0.0
                        if use_legacy_guards and (
                            now_sec < self._rl_cooldown_until
                            or self._rl_lockout_anchor is not None
                        ):
                            alpha_rl = 0.0
                        else:
                            alpha_rl = compute_rl_blend_weight(
                                front_min=front_min,
                                side_clearance=left_min + right_min,
                                used_tight_pass=self.local_planner.last_used_tight_pass,
                                min_side=(
                                    None
                                    if use_legacy_guards
                                    else min(left_min, right_min)
                                ),
                                semantic_proximity=sem_prox,
                                valve_deadband=(
                                    0.0 if use_legacy_guards else self.alpha_valve_deadband
                                ),
                                prev_alpha=(0.0 if use_legacy_guards else self._prev_alpha_rl),
                            )
                            if (
                                use_legacy_guards
                                and alpha_rl > 0.0
                                and max(abs(self.last_cmd_w), abs(self.actual_w)) < 0.35
                                and df_dt < -0.25
                            ):
                                alpha_rl *= max(0.0, 1.0 + df_dt / 0.50)

                        if use_legacy_guards:
                            if front_min < 0.24 or min(left_min, right_min) < 0.17:
                                alpha_rl = 0.0
                                self._prev_alpha_rl = alpha_rl
                            else:
                                delta_a = alpha_rl - self._prev_alpha_rl
                                max_step_a = self.alpha_max_step_base + self.alpha_max_step_sem_gain * sem_prox
                                if abs(delta_a) > max_step_a:
                                    alpha_rl = self._prev_alpha_rl + (max_step_a if delta_a > 0.0 else -max_step_a)
                                self._prev_alpha_rl = alpha_rl
                        elif self.enable_cbf_safety_shield:
                            if self._dynamic_approach_active():
                                delta_a = alpha_rl - self._prev_alpha_rl
                                max_step_a = self.alpha_max_step_base + self.alpha_max_step_sem_gain * sem_prox
                                if abs(delta_a) > max_step_a:
                                    alpha_rl = self._prev_alpha_rl + (max_step_a if delta_a > 0.0 else -max_step_a)
                            self._prev_alpha_rl = alpha_rl

                        v_astar_ref, w_astar_ref = float(v_cmd), float(w_cmd)
                        v_cmd, w_cmd, g_res = self._compute_homotopy_rl_command(
                            v_base=v_astar_ref,
                            w_base=w_astar_ref,
                            alpha_rl=alpha_rl,
                            target_xy=target_xy,
                            step_norm=step_norm,
                            sem_obs=sem_obs,
                            sem_haz=sem_haz,
                            sem_prox=sem_prox,
                            left_min=left_min,
                            right_min=right_min,
                        )
                        # Phase 4 Sim/Real nominal consistency guard (pure Ch3 dynamic regime without Ch1/Ch2):
                        # keep A* path heading authority anchored so Sim ONNX does not prematurely reverse
                        # and Real ONNX does not wander off-route when g_res saturates from distant Ch3 halos.
                        if self.enable_cbf_safety_shield and sem_prox < 0.10:
                            if g_res > 0.25 and (not sem_obs) and (not sem_haz):
                                w_cmd = 0.50 * w_cmd + 0.50 * w_astar_ref
                                if self._dynamic_approach_active():
                                    v_cap = 0.85 * v_astar_ref
                                    if (not scan_corrupted) and self._headon_dynamic_closing_active(
                                        confirmed_dyn_tracks, v_ref=v_cmd
                                    ):
                                        v_cap = min(v_cap, 0.09)
                                    v_cmd = min(v_cmd, v_cap)
                                elif front_min > 0.40 and v_cmd < 0.0 and v_astar_ref > 0.05:
                                    v_cmd = 0.60 * v_astar_ref
                            elif self._dynamic_approach_active():
                                if (
                                    (not sem_obs)
                                    and (not sem_haz)
                                    and (not scan_corrupted)
                                    and self._headon_dynamic_closing_active(
                                        confirmed_dyn_tracks, v_ref=v_cmd
                                    )
                                ):
                                    v_cmd = min(v_cmd, min(0.85 * v_astar_ref, 0.09))
                            elif front_min > 0.42:
                                if v_cmd < 0.0 and v_astar_ref > 0.05:
                                    v_cmd = 0.55 * v_astar_ref
                                if (w_cmd * w_astar_ref < -0.06) and (abs(w_cmd - w_astar_ref) > 0.55):
                                    w_cmd = 0.5 * (w_cmd + w_astar_ref)
                    else:
                        self._prev_alpha_rl = 0.0
                        self.context_rl_policy.observe(
                            scan_ranges=self.latest_raw_ranges,
                            rx=self.current_x,
                            ry=self.current_y,
                            rth=self.current_theta,
                            tx=target_xy[0],
                            ty=target_xy[1],
                            cmd_v=self.last_cmd_v,
                            cmd_w=self.last_cmd_w,
                            act_v=act_v_rl,
                            act_w=act_w_rl,
                            step_norm=step_norm,
                            semantic_obstacles=sem_obs,
                            semantic_hazards=sem_haz,
                            dynamic_ch3=self.latest_dynamic_ch3,
                            angle_min=self.latest_scan_angle_min,
                            angle_increment=self.latest_scan_angle_increment,
                        )
                    self._prev_front_min = front_min
                    self._prev_front_time = now_sec
                self.state = SystemState.BACKTRACKING_HYBRID
                self.active_target_wp = (
                    self.waypoint_stack[-1] if len(self.waypoint_stack) > 0 else self.start_pose
                )
                v_pre_cbf, w_pre_cbf = float(v_cmd), float(w_cmd)
                cross_protect_cbf = bool(
                    cross_dec is not None
                    and cross_dec.active
                    and cross_dec.mode in ("HOLD", "DASH", "DASH_FALLBACK")
                )
                if self.cbf_shield is not None:
                    v_cmd, w_cmd = self._apply_cbf_safety_shield(
                        v_cmd,
                        w_cmd,
                        now_sec,
                        prefer_no_reverse=cross_protect_cbf,
                        suppress_tangential_steer=cross_protect_cbf,
                    )
                cbf_int = bool(
                    self.cbf_shield is not None
                    and self.latest_cbf_result is not None
                    and self.latest_cbf_result.intervened
                )
                v_cmd, w_cmd = apply_motor_deadzone(
                    v_cmd,
                    w_cmd,
                    min_v=self.min_effective_linear_vel,
                    min_w=self.min_effective_angular_vel,
                    v_max=self.v_max,
                    w_max=self.w_max,
                    min_v_reverse=(self.cbf_v_min if self.enable_cbf_safety_shield else -0.10),
                    suppress_v_boost=(cbf_int and abs(v_cmd) < abs(v_pre_cbf) - 1e-4),
                    suppress_w_boost=(cbf_int and abs(w_cmd) < abs(w_pre_cbf) - 1e-4),
                )
                cmd = Twist()
                cmd.linear.x = float(v_cmd)
                cmd.angular.z = float(w_cmd)
                self.last_cmd_v = cmd.linear.x
                self.last_cmd_w = cmd.angular.z
                self.pub_cmd_vel.publish(cmd)
                return

        # 4. Obstacle Hold / TD3 Bypass Check during Backtracking
        if self.state in [SystemState.BACKTRACKING_ROTATE, SystemState.BACKTRACKING_TRANSLATE]:
            if self.enable_td3_avoidance and self.td3_policy is not None and self.td3_obstacle_detected:
                self.previous_state = self.state
                self.state = SystemState.LOCAL_TD3_AVOIDANCE
                self.td3_avoid_step = 0

                # Skip any waypoint in the stack that sits inside/too close to the fallen obstacle
                while (
                    self.active_target_wp is not None
                    and self.active_target_wp.distance_to(self.current_x, self.current_y) < 1.05
                    and len(self.waypoint_stack) > 0
                ):
                    self.active_target_wp = self.waypoint_stack.pop()

                self.get_logger().warn(
                    f"[HIERARCHICAL RECOVERY] Obstacle detected within {self.td3_clearance_dist}m! "
                    f"Engaging Layer 2: TD3 Local Reactive Avoidance -> Target WP: "
                    f"({self.active_target_wp.x:.2f}, {self.active_target_wp.y:.2f})"
                )
                return
            elif self.obstacle_detected:
                self.stop_robot()
                self.previous_state = self.state
                self.state = SystemState.OBSTACLE_HOLD
                self.get_logger().warn(
                    f"[LIDAR SAFETY] Obstacle detected within {self.lidar_danger_dist}m! Pausing backtracking (Baseline C)."
                )
                return

        # State: OBSTACLE_HOLD (Baseline C)
        if self.state == SystemState.OBSTACLE_HOLD:
            self.stop_robot()
            if not self.obstacle_detected:
                self.get_logger().info("[LIDAR SAFETY] Obstacle cleared. Resuming backtracking.")
                self.state = self.previous_state
            return

        # State: LOCAL_TD3_AVOIDANCE (Next-Gen Layer 2)
        if self.state == SystemState.LOCAL_TD3_AVOIDANCE:
            self.td3_avoid_step += 1
            self.perform_scan_matching_correction()

            if self.active_target_wp is None:
                self.select_next_waypoint()
                if self.active_target_wp is None:
                    self.state = SystemState.HANDOVER_TO_ARUCO
                    return

            dx = self.active_target_wp.x - self.current_x
            dy = self.active_target_wp.y - self.current_y
            dist_to_wp = math.hypot(dx, dy)

            # Condition 1: Waypoint reached during bypass maneuver
            if dist_to_wp <= self.wp_reach_radius * 1.5:
                self.get_logger().info(
                    f"[TD3 AVOIDANCE] Target WP reached ({dist_to_wp:.2f}m)! "
                    f"Applying Layer 3 ICP Drift Reset."
                )
                self.perform_scan_matching_correction()
                self.check_proximity_or_next_waypoint()
                return

            # Condition 2: Obstacle bypassed and line-of-sight to post-obstacle WP clear
            los_clear = check_line_of_sight_clear(
                self.latest_raw_ranges,
                self.current_x, self.current_y, self.current_theta,
                self.active_target_wp.x, self.active_target_wp.y,
                clearance_dist=self.td3_clearance_dist
            )
            if self.td3_avoid_step >= 10 and (not self.obstacle_detected) and los_clear:
                self.get_logger().info(
                    f"[TD3 AVOIDANCE] Obstacle bypassed after {self.td3_avoid_step} steps! "
                    f"Engaging Layer 3: 2D ICP Drift Reset & returning to breadcrumbs."
                )
                self.stop_robot()
                self.perform_scan_matching_correction()
                self.state = SystemState.BACKTRACKING_ROTATE
                return

            # Condition 3: Avoidance maneuver timeout
            if self.td3_avoid_step > self.td3_max_avoid_steps:
                self.get_logger().warn(
                    f"[TD3 AVOIDANCE] Maneuver reached step budget ({self.td3_avoid_step} steps). "
                    f"Advancing WP and resuming backtrack."
                )
                self.stop_robot()
                self.perform_scan_matching_correction()
                self.check_proximity_or_next_waypoint()
                self.state = SystemState.BACKTRACKING_ROTATE
                return

            # Execute TD3 inference (normalize state with policy training constants, clip outputs to runtime limits)
            td3_norm_v = float(getattr(self, "td3_norm_max_v", 0.15))
            td3_norm_w = float(getattr(self, "td3_norm_max_w", 0.60))
            state_vec = extract_state_vector(
                self.latest_raw_ranges,
                self.current_x, self.current_y, self.current_theta,
                self.active_target_wp.x, self.active_target_wp.y,
                self.last_cmd_v, self.last_cmd_w,
                max_v=td3_norm_v, max_w=td3_norm_w
            )
            v_cmd, w_cmd = self.td3_policy.predict(state_vec)

            cmd = Twist()
            cmd.linear.x = max(0.03, min(self.v_max, float(v_cmd)))
            cmd.angular.z = max(-self.w_max, min(self.w_max, float(w_cmd)))
            self.last_cmd_v = cmd.linear.x
            self.last_cmd_w = cmd.angular.z
            self.pub_cmd_vel.publish(cmd)
            return

        # 5. Method A Backtracking Logic: Rotate-then-Translate or Continuous Hybrid
        if self.state in [SystemState.BACKTRACKING_ROTATE, SystemState.BACKTRACKING_TRANSLATE, SystemState.BACKTRACKING_HYBRID]:
            self.perform_scan_matching_correction()
            if self.dynamic_tracker is not None:
                self._update_dynamic_tracker(now.nanoseconds / 1e9)

        if self.state == SystemState.BACKTRACKING_ROTATE:
            if self.active_target_wp is None:
                self.select_next_waypoint()
                return

            dx = self.active_target_wp.x - self.current_x
            dy = self.active_target_wp.y - self.current_y
            dist_to_wp = math.hypot(dx, dy)

            # Skip rotation and advance immediately if waypoint is already reached within reach radius
            if dist_to_wp <= self.wp_reach_radius:
                self.get_logger().info(
                    f"Target WP ({self.active_target_wp.x:.2f}, {self.active_target_wp.y:.2f}) reached "
                    f"({dist_to_wp:.2f}m <= {self.wp_reach_radius}m). Advancing to next WP."
                )
                self.check_proximity_or_next_waypoint()
                return

            target_heading = math.atan2(dy, dx)
            heading_diff = normalize_angle(target_heading - self.current_theta)

            if abs(heading_diff) > self.heading_tol:
                # In-place pure rotation
                cmd = Twist()
                cmd.linear.x = 0.0
                angular_speed = self.kp_ang * heading_diff
                # Clamp to safe angular speed limits
                cmd.angular.z = max(-self.w_max, min(self.w_max, angular_speed))
                self.last_cmd_v = cmd.linear.x
                self.last_cmd_w = cmd.angular.z
                self.pub_cmd_vel.publish(cmd)
            else:
                # Target heading aligned
                self.stop_robot()
                if self.enable_hybrid:
                    self.state = SystemState.BACKTRACKING_HYBRID
                    self.get_logger().info("Heading aligned. Transitioning to BACKTRACKING_HYBRID (Continuous Pure Pursuit + Residual RL).")
                else:
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
                self.last_cmd_v = cmd.linear.x
                self.last_cmd_w = cmd.angular.z
                self.pub_cmd_vel.publish(cmd)
            else:
                # Waypoint reached
                self.stop_robot()
                self.get_logger().info(
                    f"Reached Waypoint ({self.active_target_wp.x:.2f}, {self.active_target_wp.y:.2f})."
                )
                self.check_proximity_or_next_waypoint()

        elif self.state == SystemState.BACKTRACKING_HYBRID:
            # 1. Base Camp Arrival & Proximity Check
            dist_to_origin = 999.0
            if self.start_pose:
                dist_to_origin = math.hypot(self.start_pose.x - self.current_x, self.start_pose.y - self.current_y)

            # Terminal arrival: if within reach radius of base camp (<= wp_reach_radius), mission complete!
            if dist_to_origin <= self.wp_reach_radius:
                self.stop_robot()
                self.get_logger().info(
                    f">>> [HYBRID RECOVERY] Base camp origin reached (dist={dist_to_origin*100.0:.1f}cm <= {self.wp_reach_radius*100.0:.1f}cm). MISSION_COMPLETE! <<<"
                )
                self.state = SystemState.MISSION_COMPLETE
                return

            # Check ArUco visual docking handover
            if self.use_aruco_docking and dist_to_origin <= self.docking_dist_th:
                self.stop_robot()
                self.get_logger().info(
                    f"[HYBRID RECOVERY] Within docking perimeter ({dist_to_origin:.2f}m). Handing over to ArUco."
                )
                self.state = SystemState.HANDOVER_TO_ARUCO
                return

            # 2. Prune nearby passed waypoints while protecting the final base-camp anchor (start_pose)
            while len(self.waypoint_stack) > 1:
                top_wp = self.waypoint_stack[-1]
                d = top_wp.distance_to(self.current_x, self.current_y)
                dx_wp = top_wp.x - self.current_x
                dy_wp = top_wp.y - self.current_y
                forward_proj = dx_wp * math.cos(self.current_theta) + dy_wp * math.sin(self.current_theta)
                # Only prune waypoints within 0.22m or closely behind the robot (< 0.60m)
                if d < 0.22 or (d < 0.60 and forward_proj < -0.05):
                    self.waypoint_stack.pop()
                else:
                    break

            # 3. Target Waypoint Selection (Lookahead Pure Pursuit)
            target_wp = None
            if len(self.waypoint_stack) > 0:
                for wp in reversed(self.waypoint_stack):
                    if wp.distance_to(self.current_x, self.current_y) >= self.lookahead_dist:
                        target_wp = wp
                        break
                if target_wp is None:
                    target_wp = self.waypoint_stack[-1]
            elif self.start_pose:
                target_wp = self.start_pose

            if target_wp is None:
                self.stop_robot()
                self.state = SystemState.MISSION_COMPLETE
                return

            self.active_target_wp = target_wp

            dx = target_wp.x - self.current_x
            dy = target_wp.y - self.current_y
            dist_to_target = math.hypot(dx, dy)
            target_heading = math.atan2(dy, dx)
            alpha = normalize_angle(target_heading - self.current_theta)

            # If heading diverged substantially (> 95 deg), re-align with quick pivot
            if abs(alpha) > 1.65:
                self.stop_robot()
                self.state = SystemState.BACKTRACKING_ROTATE
                return

            # 4. Pure Pursuit Curvature and Regulated Velocity
            L_d = max(0.25, min(self.lookahead_dist, dist_to_target))
            curvature = 2.0 * math.sin(alpha) / L_d
            w_pp = self.v_max * curvature
            w_pp = max(-self.w_max, min(self.w_max, w_pp))

            # Regulated speed: keep robust velocity on tunnel straightaways while respecting v_max
            v_floor = min(0.14, 0.7 * self.v_max)
            v_regulated = self.v_max / (1.0 + 0.6 * abs(curvature))
            v_regulated = max(v_floor, min(self.v_max, v_regulated))

            # 5. Continuous Residual RL (TD3) Obstacle Avoidance & Wall Repulsion
            if self.enable_td3_avoidance and self.td3_policy is not None and len(self.latest_raw_ranges) > 0:
                td3_norm_v = float(getattr(self, "td3_norm_max_v", 0.15))
                td3_norm_w = float(getattr(self, "td3_norm_max_w", 0.60))
                state_vec = extract_state_vector(
                    self.latest_raw_ranges,
                    self.current_x, self.current_y, self.current_theta,
                    target_wp.x, target_wp.y,
                    self.last_cmd_v, self.last_cmd_w,
                    max_v=td3_norm_v, max_w=td3_norm_w
                )
                v_td3, w_td3 = self.td3_policy.predict(state_vec)

                if self.td3_obstacle_detected:
                    # Obstacle nearby: RL obstacle avoidance takes dominance
                    w_blend = 0.85 * w_td3 + 0.15 * w_pp
                    v_blend = min(v_regulated, max(v_floor, float(v_td3)))
                else:
                    # Clear corridor: Pure Pursuit line tracking with smooth RL wall repulsion & slip residual
                    w_blend = 0.55 * w_pp + 0.45 * w_td3
                    v_blend = v_regulated
            else:
                w_blend = w_pp
                v_blend = v_regulated

            # 6. Proportional Deceleration when approaching base camp
            if dist_to_origin <= self.docking_dist_th:
                v_blend = max(min(0.08, v_floor), min(v_blend, 0.50 * dist_to_origin))

            cmd = Twist()
            cmd.linear.x = max(min(0.08, v_floor), min(self.v_max, float(v_blend)))
            cmd.angular.z = max(-self.w_max, min(self.w_max, float(w_blend)))
            self.last_cmd_v = cmd.linear.x
            self.last_cmd_w = cmd.angular.z
            self.pub_cmd_vel.publish(cmd)

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
            # Transition back to NORMAL_TELEOP on link restoration is handled consistently in Section 0.

    def perform_scan_matching_correction(self):
        """
        Refines (self.current_x, self.current_y, self.current_theta) using 2D LiDAR scan matching.
        Eliminates accumulated wheel odometry slip during backtracking by maintaining a persistent SE(2) correction.
        """
        if not self.enable_scan_matching or self.latest_scan_pts is None or self.active_target_wp is None:
            self._icp_skip_count = getattr(self, "_icp_skip_count", 0) + 1
            return

        ref_wp = self.active_target_wp
        ref_pts = ref_wp.scan_pts
        is_start_match = False
        if self.start_pose and self.start_scan_pts is not None:
            dist_orig = math.hypot(self.start_pose.x - self.current_x, self.start_pose.y - self.current_y)
            if dist_orig <= max(self.docking_dist_th, 0.80):
                ref_wp = self.start_pose
                ref_pts = self.start_scan_pts
                is_start_match = True

        if ref_pts is None:
            self._icp_skip_count = getattr(self, "_icp_skip_count", 0) + 1
            return

        dist_to_wp = math.hypot(ref_wp.x - self.current_x, ref_wp.y - self.current_y)
        max_match_d = max(self.scan_match_max_dist, 0.80) if is_start_match else self.scan_match_max_dist
        if dist_to_wp > max_match_d:
            self._icp_skip_count = getattr(self, "_icp_skip_count", 0) + 1
            return

        # ICP cache: at most one ICP run per received scan (unifies the 2x/loop call sites)
        scan_stamp = getattr(self, "latest_scan_stamp_sec", None)
        if scan_stamp is not None and getattr(self, "_icp_done_scan_stamp", None) == scan_stamp:
            self._icp_skip_count = getattr(self, "_icp_skip_count", 0) + 1
            return

        # Filter out scan points belonging to confirmed dynamic obstacles so moving cylinders
        # cannot drag the static-background ICP estimate into drift.
        src_pts = self.latest_scan_pts
        dyn_tracks = getattr(self, "latest_dynamic_tracks", None)
        if dyn_tracks and src_pts is not None and len(src_pts) >= 20:
            c_r, s_r = math.cos(self.current_theta), math.sin(self.current_theta)
            keep_mask = np.ones(len(src_pts), dtype=bool)
            for trk in dyn_tracks:
                if not getattr(trk, "is_confirmed", False):
                    continue
                dxw = trk.x - self.current_x
                dyw = trk.y - self.current_y
                if dxw * dxw + dyw * dyw > 6.25:
                    continue
                xb = c_r * dxw + s_r * dyw
                yb = -s_r * dxw + c_r * dyw
                rad_m = float(getattr(trk, "radius", 0.18)) + 0.18
                keep_mask &= ((src_pts[:, 0] - xb) ** 2 + (src_pts[:, 1] - yb) ** 2) > (rad_m * rad_m)
            if int(np.count_nonzero(keep_mask)) >= 18:
                src_pts = src_pts[keep_mask]

        # Reject 1D-degenerate straight-wall scans (skip when zero-placeholder test fixture)
        real_scan_cloud = False
        y_eig_degenerate = False
        if src_pts is not None and len(src_pts) >= 15:
            centered = src_pts - np.mean(src_pts, axis=0, keepdims=True)
            cov = (centered.T @ centered) / float(len(src_pts))
            tr_c = float(cov[0, 0] + cov[1, 1])
            if tr_c > 1e-6:
                real_scan_cloud = True
                disc_c = math.sqrt(max(0.0, (float(cov[0, 0] - cov[1, 1])) ** 2 + 4.0 * float(cov[0, 1]) ** 2))
                lam_max = 0.5 * (tr_c + disc_c)
                lam_min = max(0.0, 0.5 * (tr_c - disc_c))
                if lam_max > 1e-6 and (lam_min / lam_max) < 0.045:
                    self._icp_skip_count = getattr(self, "_icp_skip_count", 0) + 1
                    return
                if is_start_match:
                    c_w = math.cos(self.current_theta)
                    s_w = math.sin(self.current_theta)
                    checked_h_yy = False
                    if len(src_pts) >= 16:
                        wx_p = c_w * src_pts[:, 0] - s_w * src_pts[:, 1]
                        wy_p = s_w * src_pts[:, 0] + c_w * src_pts[:, 1]
                        dwx = np.diff(wx_p)
                        dwy = np.diff(wy_p)
                        seg_d2 = dwx * dwx + dwy * dwy
                        valid_seg = (seg_d2 > 4e-4) & (seg_d2 < 0.1225)
                        if int(np.count_nonzero(valid_seg)) >= 10:
                            checked_h_yy = True
                            h_yy = float(np.mean((dwx[valid_seg] ** 2) / seg_d2[valid_seg]))
                            if h_yy < 0.04:
                                y_eig_degenerate = True
                    if not checked_h_yy:
                        wy_c = s_w * centered[:, 0] + c_w * centered[:, 1]
                        var_wy = float(np.mean(wy_c * wy_c))
                        if (var_wy / max(tr_c, 1e-6)) < 0.08:
                            y_eig_degenerate = True

        # Estimate initial guess in target WP frame (vector from WP to robot expressed in WP frame)
        dx_world = self.current_x - ref_wp.x
        dy_world = self.current_y - ref_wp.y
        wp_th = ref_wp.theta
        c_th, s_th = math.cos(wp_th), math.sin(wp_th)
        dx_guess = c_th * dx_world + s_th * dy_world
        dy_guess = -s_th * dx_world + c_th * dy_world
        dth_guess = normalize_angle(self.current_theta - wp_th)

        spike_ratio = float(getattr(self, "latest_scan_spike_ratio", 0.0))
        self._icp_run_count = getattr(self, "_icp_run_count", 0) + 1
        icp_thresh = 0.20 if real_scan_cloud else 0.35
        icp_min_inliers = (
            max(25, int(0.68 * min(len(src_pts), 120)))
            if real_scan_cloud
            else 15
        )
        icp_res = icp_match_2d(
            src_pts, ref_pts,
            initial_guess=(dx_guess, dy_guess, dth_guess),
            max_iters=8, inlier_thresh=icp_thresh, min_inliers=icp_min_inliers,
            return_stats=True,
        )
        if len(icp_res) >= 5:
            dx_rel, dy_rel, dth_rel, ok, icp_stats = icp_res[0], icp_res[1], icp_res[2], icp_res[3], icp_res[4]
        else:
            dx_rel, dy_rel, dth_rel, ok = icp_res
            icp_stats = {
                "num_inliers": icp_min_inliers if ok else 0,
                "num_points": min(len(src_pts), 120),
                "inlier_ratio": 1.0 if ok else 0.0,
                "mean_err": 0.0 if ok else 99.0,
            }

        # Coarse-to-fine rescue only for clean start-anchor scans with >20cm accumulated drift:
        # widen correspondence gate to 0.38m, then strictly evaluate final residual at icp_thresh (0.20m).
        if (not ok) and is_start_match and real_scan_cloud and (spike_ratio < 0.03):
            icp_res2 = icp_match_2d(
                src_pts, ref_pts,
                initial_guess=(dx_guess, dy_guess, dth_guess),
                max_iters=8, inlier_thresh=0.38, min_inliers=icp_min_inliers,
                return_stats=True, eval_inlier_thresh=icp_thresh,
            )
            if len(icp_res2) >= 5:
                dx_rel, dy_rel, dth_rel, ok, icp_stats = icp_res2[0], icp_res2[1], icp_res2[2], icp_res2[3], icp_res2[4]
            else:
                dx_rel, dy_rel, dth_rel, ok = icp_res2
                icp_stats = {
                    "num_inliers": icp_min_inliers if ok else 0,
                    "num_points": min(len(src_pts), 120),
                    "inlier_ratio": 1.0 if ok else 0.0,
                    "mean_err": 0.0 if ok else 99.0,
                }

        icp_num_inliers = int(icp_stats.get("num_inliers", 0))
        icp_inlier_ratio = float(icp_stats.get("inlier_ratio", 0.0))
        icp_mean_err = float(icp_stats.get("mean_err", 99.0))
        # Pure sensor/ICP observation classifier (no simulation-parameter reference)
        is_scan_corrupted = bool(
            real_scan_cloud
            and (
                spike_ratio >= 0.03
                or icp_inlier_ratio < 0.65
                or icp_mean_err > 0.08
            )
        )
        self.last_icp_stats = {
            "spike_ratio": spike_ratio,
            "num_inliers": icp_num_inliers,
            "inlier_ratio": icp_inlier_ratio,
            "mean_err": icp_mean_err,
            "is_scan_corrupted": is_scan_corrupted,
            "is_start_match": is_start_match,
            "ok": bool(ok),
        }

        if is_start_match and real_scan_cloud:
            diag_n = getattr(self, "_start_icp_diag_count", 0)
            if diag_n % 5 == 0:
                self.get_logger().info(
                    f"[ICP_START_DIAG #{diag_n}] ok={int(bool(ok))} corrupt={int(is_scan_corrupted)} "
                    f"spk={spike_ratio:.4f} inl={icp_inlier_ratio:.3f} ({icp_num_inliers}) "
                    f"merr={icp_mean_err:.4f} y_deg={int(y_eig_degenerate)}"
                )
            self._start_icp_diag_count = diag_n + 1

        if ok:
            is_clean_hq_start = bool(
                is_start_match
                and (not is_scan_corrupted)
                and (icp_num_inliers >= 25)
            )
            # Reconstruct measured robot world pose
            x_meas = ref_wp.x + (c_th * dx_rel - s_th * dy_rel)
            y_meas = ref_wp.y + (s_th * dx_rel + c_th * dy_rel)
            th_meas = normalize_angle(wp_th + dth_rel)

            # Robot-pose innovation outlier rejection:
            # - Corrupted/low-quality scans keep Step A conservative guard (0.12m jump / 0.16m drift cap)
            # - Clean high-quality start-anchor scans restore 0.38m jump / 0.42m drift cap for real drift
            pose_jump = math.hypot(x_meas - self.current_x, y_meas - self.current_y)
            yaw_jump = abs(normalize_angle(th_meas - self.current_theta))
            if is_start_match:
                max_p_jump = 0.38 if is_clean_hq_start else 0.12
                drift_cap = 0.42 if is_clean_hq_start else 0.16
            else:
                max_p_jump = 0.13
                drift_cap = 0.14
            max_y_jump = 0.14
            if pose_jump < max_p_jump and yaw_jump < max_y_jump:
                old_corr = (self.corr_x, self.corr_y, self.corr_th)
                alpha = self.scan_match_weight
                block_corr_y = bool(y_eig_degenerate or is_scan_corrupted)

                # Blend robot pose directly first, then invert T_corr = T_blend * T_raw^-1
                x_blend = (1.0 - alpha) * self.current_x + alpha * x_meas
                if block_corr_y:
                    y_blend = self.current_y
                    th_blend = self.current_theta
                    new_corr_th = self.corr_th
                else:
                    y_blend = (1.0 - alpha) * self.current_y + alpha * y_meas
                    th_blend = normalize_angle(self.current_theta + alpha * normalize_angle(th_meas - self.current_theta))
                    new_corr_th = normalize_angle(th_blend - self.raw_odom_theta)

                # Guard against cumulative false-ICP debris/wall-slip runaway (robot-frame displacement & yaw)
                robot_drift_old = math.hypot(self.current_x - self.raw_odom_x, self.current_y - self.raw_odom_y)
                robot_drift_new = math.hypot(x_blend - self.raw_odom_x, y_blend - self.raw_odom_y)
                if robot_drift_new > drift_cap and robot_drift_new > robot_drift_old:
                    self._icp_skip_count = getattr(self, "_icp_skip_count", 0) + 1
                    return
                if abs(new_corr_th) > 0.06 and abs(new_corr_th) > abs(self.corr_th):
                    self._icp_skip_count = getattr(self, "_icp_skip_count", 0) + 1
                    return

                c_c = math.cos(new_corr_th)
                s_c = math.sin(new_corr_th)
                new_corr_x = x_blend - (self.raw_odom_x * c_c - self.raw_odom_y * s_c)
                new_corr_y = self.corr_y if block_corr_y else (y_blend - (self.raw_odom_x * s_c + self.raw_odom_y * c_c))

                self.corr_x = new_corr_x
                self.corr_y = new_corr_y
                self.corr_th = new_corr_th
                new_corr = (self.corr_x, self.corr_y, self.corr_th)
                if self.dynamic_tracker is not None:
                    self.dynamic_tracker.sync_icp_correction(old_corr, new_corr)
                self.semantic_cache.sync_icp_correction(old_corr, new_corr)
                local_planner = getattr(self, "local_planner", None)
                if local_planner is not None and hasattr(local_planner, "sync_icp_correction"):
                    local_planner.sync_icp_correction(old_corr, new_corr)
                cross_sched = getattr(self, "crossing_scheduler", None)
                if cross_sched is not None and hasattr(cross_sched, "sync_icp_correction"):
                    cross_sched.sync_icp_correction(old_corr, new_corr)
                scan_pose = getattr(self, "latest_scan_pose", None)
                if scan_pose is not None:
                    ox_c, oy_c, oth_c = old_corr
                    nx_c, ny_c, nth_c = new_corr
                    d_th_c = normalize_angle(nth_c - oth_c)
                    c_dc = math.cos(d_th_c)
                    s_dc = math.sin(d_th_c)
                    d_tx_c = nx_c - (c_dc * ox_c - s_dc * oy_c)
                    d_ty_c = ny_c - (s_dc * ox_c + c_dc * oy_c)
                    srx, sry, srth = scan_pose
                    self.latest_scan_pose = (
                        d_tx_c + c_dc * srx - s_dc * sry,
                        d_ty_c + s_dc * srx + c_dc * sry,
                        normalize_angle(srth + d_th_c),
                    )

                self.current_x = x_blend
                self.current_y = y_blend
                self.current_theta = th_blend

                self.scan_match_count += 1
                self._icp_done_scan_stamp = scan_stamp
                if self.scan_match_count % 10 == 1:
                    self.get_logger().info(
                        f"[LIDAR SCAN MATCH #{self.scan_match_count}] Drift Compensated: "
                        f"Pose=({self.current_x:.2f}, {self.current_y:.2f}, {math.degrees(self.current_theta):.1f}°) | "
                        f"Offset=({self.corr_x:.3f}, {self.corr_y:.3f}, {math.degrees(self.corr_th):.1f}°) | "
                        f"spk={spike_ratio:.3f} inl={icp_inlier_ratio:.2f} merr={icp_mean_err:.3f} corrupt={int(is_scan_corrupted)}"
                    )

    def check_proximity_or_next_waypoint(self):
        """Check if robot is within origin, or fetch next backtrack waypoint."""
        if self.start_pose:
            dist_to_origin = math.hypot(self.start_pose.x - self.current_x, self.start_pose.y - self.current_y)
            if self.use_aruco_docking and dist_to_origin <= self.docking_dist_th:
                self.get_logger().info(
                    f"Robot arrived within docking perimeter ({dist_to_origin:.2f}m <= {self.docking_dist_th}m). "
                    f"Switching to ArUco Visual Docking."
                )
                self.state = SystemState.HANDOVER_TO_ARUCO
                return
            elif not self.use_aruco_docking and dist_to_origin <= self.wp_reach_radius:
                self.get_logger().info(
                    f"Robot arrived at base camp origin ({dist_to_origin:.2f}m <= {self.wp_reach_radius}m). "
                    f"MISSION_COMPLETE."
                )
                self.stop_robot()
                self.state = SystemState.MISSION_COMPLETE
                return

        if len(self.waypoint_stack) > 0:
            self.active_target_wp = self.waypoint_stack.pop()
            self.get_logger().info(
                f"Next backtrack WP: ({self.active_target_wp.x:.2f}, {self.active_target_wp.y:.2f}). "
                f"Remaining stack size: {len(self.waypoint_stack)}"
            )
            dx = self.active_target_wp.x - self.current_x
            dy = self.active_target_wp.y - self.current_y
            target_heading = math.atan2(dy, dx)
            heading_diff = abs(normalize_angle(target_heading - self.current_theta))
            if heading_diff > self.heading_tol:
                self.state = SystemState.BACKTRACKING_ROTATE
            else:
                self.state = SystemState.BACKTRACKING_HYBRID if self.enable_hybrid else SystemState.BACKTRACKING_TRANSLATE
        else:
            if not self.use_aruco_docking and self.start_pose and math.hypot(self.start_pose.x - self.current_x, self.start_pose.y - self.current_y) > self.wp_reach_radius:
                self.active_target_wp = self.start_pose
                self.state = SystemState.BACKTRACKING_ROTATE
            else:
                self.get_logger().info("All backtracked waypoints exhausted. Mission complete.")
                self.stop_robot()
                self.state = SystemState.MISSION_COMPLETE if not self.use_aruco_docking else SystemState.HANDOVER_TO_ARUCO

    def select_next_waypoint(self):
        if len(self.waypoint_stack) > 0:
            self.active_target_wp = self.waypoint_stack.pop()
            dx = self.active_target_wp.x - self.current_x
            dy = self.active_target_wp.y - self.current_y
            target_heading = math.atan2(dy, dx)
            heading_diff = abs(normalize_angle(target_heading - self.current_theta))
            if heading_diff > self.heading_tol:
                self.state = SystemState.BACKTRACKING_ROTATE
            else:
                self.state = SystemState.BACKTRACKING_HYBRID if self.enable_hybrid else SystemState.BACKTRACKING_TRANSLATE
        else:
            if not self.use_aruco_docking and self.start_pose and math.hypot(self.start_pose.x - self.current_x, self.start_pose.y - self.current_y) > self.wp_reach_radius:
                self.active_target_wp = self.start_pose
                self.state = SystemState.BACKTRACKING_ROTATE
            else:
                self.stop_robot()
                self.state = SystemState.MISSION_COMPLETE if not self.use_aruco_docking else SystemState.HANDOVER_TO_ARUCO

    def trigger_emergency_transition(self):
        self.stop_robot()
        self.state = SystemState.EMERGENCY_STOP
        self.recovery_step_count = 0
        if self.context_rl_policy is not None and (not self.enable_teleop_rl_warmup):
            self.context_rl_policy.reset()

        # Trigger onboard blackbox recorder to save sensor and camera data locally
        bb_msg = Bool()
        bb_msg.data = True
        self.pub_blackbox_trigger.publish(bb_msg)

    def stop_robot(self):
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0
        try:
            if rclpy.ok():
                stop_cmd = Twist()
                self.pub_cmd_vel.publish(stop_cmd)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = RecoveryController()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
