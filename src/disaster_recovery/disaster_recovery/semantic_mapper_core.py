#!/usr/bin/env python3
"""
Pure-Python / NumPy / OpenCV Semantic Mapping Core (ROS-independent).

Implements the mathematical and algorithmic foundations for real-time disaster
semantic SLAM on TurtleBot3 Burger:
  1. Image rotation unmapping (270/90/180/0 deg inverse transforms matching OpenCV).
  2. Pinhole camera intrinsics & 3D->2D projection (Paper Eq. 1-3 with Eq. 3 Y*fy fix).
  3. Pluggable Depth Estimation:
     - LidarProjectionDepth (2D LaserScan -> camera optical projection + foreground clustering)
     - GroundPlaneFallback (bbox bottom-center ray -> z_base=0 plane intersection)
     - RgbdDepthAdapter (aligned depth image median inside center ROI -> 3D point)
     - DepthEstimator / create_depth_estimator facade
  4. SemanticMarkerManager (Paper Algorithm 1 enhanced with TTL, min_hits, time-span,
     weighted-average merge deduplication, and multimodal GAS_HAZARD / AUDIO_DISTRESS events).
  5. 2D OccupancyGrid Map Overlay & Export (Paper Algorithm 2 with bottom-up vertical flip).

Reference:
  Lee & Kim (2025), "Development and Implementation of a SLAM-Based Real-Time
  Environment Mapping Algorithm Including Obstacle Information", J. Korea Robotics Soc. 20(2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


# =============================================================================
# Default Palettes & Multimodal Event Constants
# =============================================================================

EVENT_CLASS_IDS: Dict[str, int] = {
    "GAS_HAZARD": 1001,
    "AUDIO_DISTRESS": 1002,
}

# BGR color tuples for OpenCV rendering
DEFAULT_CLASS_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "person": (0, 0, 255),             # Red (Survivor)
    "GAS_HAZARD": (0, 165, 255),       # Orange
    "AUDIO_DISTRESS": (255, 0, 255),   # Magenta
    "chair": (255, 140, 0),            # Blue-orange
    "couch": (200, 100, 50),
    "potted plant": (0, 180, 0),       # Green
    "bed": (180, 105, 255),
    "dining table": (0, 200, 200),     # Yellow-teal
    "table": (0, 200, 200),
    "toilet": (255, 200, 0),
    "tv": (200, 50, 200),
    "bottle": (100, 220, 100),
    "backpack": (50, 150, 255),
    "suitcase": (120, 120, 255),
    "bench": (180, 105, 255),
    "door": (0, 255, 255),
    "fire-extinguisher": (0, 69, 255),
    "fire_extinguisher": (0, 69, 255),
    "stairs": (255, 255, 0),
    "debris": (42, 42, 165),
    "low_obstacle": (42, 42, 165),
}


# =============================================================================
# 0. Rigid-Body SE(3) & Time-Matching Helpers
# =============================================================================

def quat_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to a 3x3 rotation matrix."""
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = qx / norm, qy / norm, qz / norm, qw / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def pose_to_matrix(
    tx: float,
    ty: float,
    tz: float,
    qx: float = 0.0,
    qy: float = 0.0,
    qz: float = 0.0,
    qw: float = 1.0,
) -> np.ndarray:
    """Construct a 4x4 homogeneous transform matrix from translation and quaternion."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_rotation_matrix(qx, qy, qz, qw)
    T[:3, 3] = [float(tx), float(ty), float(tz)]
    return T


def rpy_to_matrix(
    tx: float,
    ty: float,
    tz: float,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
) -> np.ndarray:
    """Construct a 4x4 homogeneous transform matrix from translation and Roll-Pitch-Yaw (rad)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    R = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [float(tx), float(ty), float(tz)]
    return T


def default_t_base_cam(
    tx: float = 0.040,
    ty: float = 0.0,
    tz: float = 0.130,
) -> np.ndarray:
    """
    Return default 4x4 T_base_cam (camera_rgb_optical_frame -> base_link) for TurtleBot3 Burger.

    base_link -> camera_link: translation (tx, ty, tz), rpy (0, 0, 0)
    camera_link -> camera_rgb_optical_frame: translation (0, 0, 0), rpy (-pi/2, 0, -pi/2)
    """
    T_base_link = rpy_to_matrix(tx, ty, tz, 0.0, 0.0, 0.0)
    T_link_optical = rpy_to_matrix(0.0, 0.0, 0.0, -math.pi / 2.0, 0.0, -math.pi / 2.0)
    return T_base_link @ T_link_optical


def default_t_cam_scan(
    cam_xyz: Tuple[float, float, float] = (0.040, 0.0, 0.130),
    scan_xyz: Tuple[float, float, float] = (-0.032, 0.0, 0.172),
) -> np.ndarray:
    """
    Return default 4x4 T_cam_scan (base_scan -> camera_rgb_optical_frame) for TurtleBot3 Burger.
    """
    T_base_cam = default_t_base_cam(*cam_xyz)
    T_base_scan = rpy_to_matrix(scan_xyz[0], scan_xyz[1], scan_xyz[2], 0.0, 0.0, 0.0)
    return np.linalg.inv(T_base_cam) @ T_base_scan


def transform_point(pt: Union[Sequence[float], np.ndarray], T: np.ndarray) -> np.ndarray:
    """Transform a 3D point (X, Y, Z) using a 4x4 homogeneous matrix T."""
    p = np.asarray(pt, dtype=np.float64).reshape(3)
    T_mat = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return T_mat[:3, :3] @ p + T_mat[:3, 3]


def transform_points(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Transform (N, 3) 3D points using a 4x4 homogeneous matrix T."""
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, 3)
    if arr.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    T_mat = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return (arr @ T_mat[:3, :3].T) + T_mat[:3, 3]


def find_closest_timed_item(
    buffer: Sequence[Tuple[float, Any]],
    target_stamp_sec: float,
    max_delta_sec: float = 0.15,
) -> Optional[Any]:
    """
    Select item from a (stamp_sec, item) ring buffer with minimum |stamp_sec - target_stamp_sec|
    within max_delta_sec. Returns None if buffer is empty or closest delta > max_delta_sec.
    """
    if not buffer:
        return None
    best_item = None
    best_delta = float("inf")
    for stamp_sec, item in buffer:
        delta = abs(float(stamp_sec) - float(target_stamp_sec))
        if delta < best_delta:
            best_delta = delta
            best_item = item
    if best_delta <= max_delta_sec:
        return best_item
    return None


# =============================================================================
# 1. Rotation Unmapping
# =============================================================================

def unrotate_point(
    u_rot: float,
    v_rot: float,
    orig_width: int,
    orig_height: int,
    rotation_angle: int,
) -> Tuple[float, float]:
    """
    Map a pixel coordinate (u_rot, v_rot) from a rotated image back to the original
    pre-rotation camera frame (u, v) of size (orig_width, orig_height).

    Matches host_ai_pipeline.py OpenCV rotations:
      - 0 deg   : identity -> u = u_rot, v = v_rot
      - 90 deg  : cv2.ROTATE_90_CLOCKWISE -> u = v_rot, v = orig_height - 1 - u_rot
      - 180 deg : cv2.ROTATE_180 -> u = orig_width - 1 - u_rot, v = orig_height - 1 - v_rot
      - 270 deg : cv2.ROTATE_90_COUNTERCLOCKWISE -> u = orig_width - 1 - v_rot, v = u_rot
    """
    angle = int(rotation_angle) % 360
    u_r = float(u_rot)
    v_r = float(v_rot)
    w = int(orig_width)
    h = int(orig_height)

    if angle == 0:
        return u_r, v_r
    if angle == 270:
        return float(w - 1) - v_r, u_r
    if angle == 90:
        return v_r, float(h - 1) - u_r
    if angle == 180:
        return float(w - 1) - u_r, float(h - 1) - v_r

    raise ValueError(f"Unsupported rotation_angle={rotation_angle}. Expected 0, 90, 180, or 270.")


def unrotate_bbox_xyxy(
    bbox_xyxy: Tuple[float, float, float, float],
    orig_width: int,
    orig_height: int,
    rotation_angle: int,
) -> Tuple[float, float, float, float]:
    """
    Map an axis-aligned bounding box (x1, y1, x2, y2) from a rotated image back to
    an axis-aligned bounding box (x1_orig, y1_orig, x2_orig, y2_orig) in the original
    pre-rotation image frame (with x1_orig <= x2_orig and y1_orig <= y2_orig).
    """
    x1_r, y1_r, x2_r, y2_r = (
        float(bbox_xyxy[0]),
        float(bbox_xyxy[1]),
        float(bbox_xyxy[2]),
        float(bbox_xyxy[3]),
    )
    corners_rot = [
        (x1_r, y1_r),
        (x2_r, y1_r),
        (x2_r, y2_r),
        (x1_r, y2_r),
    ]
    unrotated = [
        unrotate_point(u_r, v_r, orig_width, orig_height, rotation_angle)
        for u_r, v_r in corners_rot
    ]
    us = [pt[0] for pt in unrotated]
    vs = [pt[1] for pt in unrotated]
    return float(min(us)), float(min(vs)), float(max(us)), float(max(vs))


# =============================================================================
# 2. Camera Intrinsics & 3D -> 2D Projection (Paper Eq. 1-3 with Y-typo fixed)
# =============================================================================

@dataclass
class CameraIntrinsics:
    """
    Pinhole camera intrinsic parameters and projection methods.

    Paper Eq. 1-3 (with Eq. 3 typo corrected from X*fy to Y*fy):
      u = (X * fx) / Z + cx
      v = (Y * fy) / Z + cy
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 640
    height: int = 480

    @classmethod
    def from_fov(
        cls,
        width: int = 640,
        height: int = 480,
        horizontal_fov: float = 1.085595,
    ) -> "CameraIntrinsics":
        """
        Construct CameraIntrinsics from image dimensions and horizontal FOV (radians).
        Default horizontal_fov=1.085595 matches turtlebot3_burger/model.sdf picam sensor.
        """
        w = int(width)
        h = int(height)
        hfov = float(horizontal_fov)
        if w <= 0 or h <= 0:
            raise ValueError("width and height must be positive integers.")
        if not (0.0 < hfov < math.pi):
            raise ValueError("horizontal_fov must be in (0, pi) radians.")
        fx = (w / 2.0) / math.tan(hfov / 2.0)
        fy = fx
        cx = w / 2.0
        cy = h / 2.0
        return cls(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy), width=w, height=h)

    @classmethod
    def from_camera_matrix(
        cls,
        camera_matrix: Union[Sequence[float], np.ndarray],
        width: int = 640,
        height: int = 480,
    ) -> "CameraIntrinsics":
        """Construct CameraIntrinsics from a 3x3 matrix or flat 9-element sequence."""
        K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        return cls(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            width=int(width),
            height=int(height),
        )

    @property
    def K(self) -> np.ndarray:
        """Return 3x3 intrinsic camera matrix K."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def project_points(
        self,
        pts_cam: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project 3D points in camera optical frame (X, Y, Z) onto 2D image coordinates (u, v)
        using Paper Eq. 1-3 with the Eq. 3 Y-axis correction:
          u = (X * fx) / Z + cx
          v = (Y * fy) / Z + cy

        Filters strictly for points in front of the camera (Z > 0 and finite).

        Args:
            pts_cam: Array of shape (N, 3) or (3,) with [X, Y, Z] in camera optical frame.

        Returns:
            Tuple of (u, v, valid_mask):
              - u: 1D float64 array of length M (projected horizontal pixel coords for valid points)
              - v: 1D float64 array of length M (projected vertical pixel coords for valid points)
              - valid_mask: 1D boolean array of length N indicating which input points had Z > 0
        """
        pts = np.asarray(pts_cam, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        if pts.shape[0] == 0:
            empty_f = np.empty((0,), dtype=np.float64)
            empty_b = np.empty((0,), dtype=bool)
            return empty_f, empty_f, empty_b

        valid_mask = np.all(np.isfinite(pts), axis=1) & (pts[:, 2] > 1e-6)
        valid_pts = pts[valid_mask]
        if valid_pts.shape[0] == 0:
            empty_f = np.empty((0,), dtype=np.float64)
            return empty_f, empty_f, valid_mask

        X = valid_pts[:, 0]
        Y = valid_pts[:, 1]
        Z = valid_pts[:, 2]

        u = (X * self.fx) / Z + self.cx
        v = (Y * self.fy) / Z + self.cy
        return u.astype(np.float64), v.astype(np.float64), valid_mask

    def project_point(self, pt_cam: Union[Sequence[float], np.ndarray]) -> Tuple[float, float]:
        """Project a single 3D point (X, Y, Z) with Z > 0 to (u, v)."""
        pt = np.asarray(pt_cam, dtype=np.float64).reshape(3)
        if pt[2] <= 1e-6:
            raise ValueError(f"Cannot project point with Z <= 0: {pt}")
        u = (pt[0] * self.fx) / pt[2] + self.cx
        v = (pt[1] * self.fy) / pt[2] + self.cy
        return float(u), float(v)

    def backproject_pixel(self, u: float, v: float, depth_m: float = 1.0) -> np.ndarray:
        """Back-project pixel (u, v) at depth Z = depth_m to 3D camera coordinates [X, Y, Z]."""
        z = float(depth_m)
        x = (float(u) - self.cx) * z / self.fx
        y = (float(v) - self.cy) * z / self.fy
        return np.array([x, y, z], dtype=np.float64)


# =============================================================================
# 3. Pluggable Depth Estimation
# =============================================================================

@dataclass
class DepthEstimate:
    """Result of 3D position / depth estimation in camera_rgb_optical_frame."""

    point_cam: np.ndarray
    depth_m: float
    method: str
    low_confidence: bool = False
    num_points: int = 1


class LidarProjectionDepth:
    """
    2D LiDAR + Monocular RGB Projection Fusion Depth Estimator.

    - Converts valid 2D LaserScan polar points (0.12 <= r <= 3.50 m, finite) into
      3D points in base_scan frame (z = 0).
    - Transforms points by 4x4 T_cam_scan into camera_rgb_optical_frame.
    - Projects points with Z_c > 0 onto image plane (u, v) via CameraIntrinsics.
    - Selects points inside the bounding box horizontal center 70% [x1 + 0.15*w, x2 - 0.15*w]
      (and vertical bounds [y1 - vertical_margin_px, y2 + vertical_margin_px] when enabled).
    - Finds min_z = min(Z_c), keeps foreground cluster Z_c <= min_z + 0.25 m to reject
      background wall points behind the object, and returns median (X_c, Y_c, Z_c) as P_camera.
    """

    def __init__(
        self,
        min_range_m: float = 0.12,
        max_range_m: float = 3.50,
        center_ratio: float = 0.70,
        foreground_margin_m: float = 0.25,
        vertical_margin_px: float = 40.0,
        check_vertical: bool = True,
    ) -> None:
        self.min_range_m = float(min_range_m)
        self.max_range_m = float(max_range_m)
        self.center_ratio = float(center_ratio)
        self.foreground_margin_m = float(foreground_margin_m)
        self.vertical_margin_px = float(vertical_margin_px)
        self.check_vertical = bool(check_vertical)

    def polar_to_cartesian(
        self,
        ranges: Union[Sequence[float], np.ndarray],
        angle_min: float = 0.0,
        angle_increment: float = 0.0,
        angles: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Convert 2D LaserScan polar ranges to valid (M, 3) Cartesian points [x, y, 0]
        in the laser scan frame, filtering min_range_m <= r <= max_range_m and finite values.
        """
        r_arr = np.asarray(ranges, dtype=np.float64).reshape(-1)
        if r_arr.size == 0:
            return np.empty((0, 3), dtype=np.float64)

        if angles is not None:
            theta = np.asarray(angles, dtype=np.float64).reshape(-1)
            if theta.size != r_arr.size:
                raise ValueError("angles and ranges must have the same length.")
        else:
            inc = float(angle_increment)
            if inc == 0.0 and r_arr.size > 1:
                inc = (2.0 * math.pi) / float(r_arr.size)
            theta = float(angle_min) + np.arange(r_arr.size, dtype=np.float64) * inc

        valid = np.isfinite(r_arr) & (r_arr >= self.min_range_m) & (r_arr <= self.max_range_m)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float64)

        r_v = r_arr[valid]
        th_v = theta[valid]
        x = r_v * np.cos(th_v)
        y = r_v * np.sin(th_v)
        z = np.zeros_like(x)
        return np.stack([x, y, z], axis=1)

    def estimate(
        self,
        bbox_xyxy: Tuple[float, float, float, float],
        intrinsics: CameraIntrinsics,
        T_cam_scan: Optional[np.ndarray] = None,
        ranges: Optional[Union[Sequence[float], np.ndarray]] = None,
        angle_min: float = 0.0,
        angle_increment: float = 0.0,
        angles: Optional[np.ndarray] = None,
        scan_points: Optional[np.ndarray] = None,
        scan_ranges: Optional[Union[Sequence[float], np.ndarray]] = None,
        scan_angles: Optional[np.ndarray] = None,
    ) -> Optional[DepthEstimate]:
        """
        Estimate 3D camera-frame coordinate P_camera for a 2D bounding box in the original
        (unrotated) image frame.
        """
        if ranges is None and scan_ranges is not None:
            ranges = scan_ranges
        if angles is None and scan_angles is not None:
            angles = scan_angles

        if T_cam_scan is None:
            T_mat = np.eye(4, dtype=np.float64)
        else:
            T_mat = np.asarray(T_cam_scan, dtype=np.float64).reshape(4, 4)

        if scan_points is not None:
            pts = np.asarray(scan_points, dtype=np.float64)
            if pts.ndim == 1:
                pts = pts.reshape(1, -1)
            if pts.shape[0] == 0:
                return None
            if pts.shape[1] == 2:
                pts = np.column_stack([pts, np.zeros(pts.shape[0], dtype=np.float64)])
            finite_mask = np.all(np.isfinite(pts), axis=1)
            pts = pts[finite_mask]
            if pts.shape[0] == 0:
                return None
            radii = np.hypot(pts[:, 0], pts[:, 1])
            # Filter by valid range when points are given in scan coordinates; if points are
            # already in camera coordinates (T_cam_scan is identity and z > 0), check 3D norm/z
            if not np.allclose(T_mat, np.eye(4)):
                r_mask = (radii >= self.min_range_m) & (radii <= self.max_range_m)
            else:
                norms = np.linalg.norm(pts, axis=1)
                r_mask = (norms >= self.min_range_m) & (norms <= self.max_range_m)
            pts_scan = pts[r_mask]
        elif ranges is not None:
            pts_scan = self.polar_to_cartesian(
                ranges=ranges,
                angle_min=angle_min,
                angle_increment=angle_increment,
                angles=angles,
            )
        else:
            return None

        if pts_scan.shape[0] == 0:
            return None

        pts_cam = transform_points(pts_scan, T_mat)
        u, v, valid_mask = intrinsics.project_points(pts_cam)
        if u.size == 0:
            return None
        pts_valid = pts_cam[valid_mask]

        x1 = min(float(bbox_xyxy[0]), float(bbox_xyxy[2]))
        y1 = min(float(bbox_xyxy[1]), float(bbox_xyxy[3]))
        x2 = max(float(bbox_xyxy[0]), float(bbox_xyxy[2]))
        y2 = max(float(bbox_xyxy[1]), float(bbox_xyxy[3]))
        w = x2 - x1
        if w <= 0.0:
            return None

        trim = 0.5 * (1.0 - self.center_ratio) * w
        u_min = x1 + trim
        u_max = x2 - trim

        in_box = (u >= u_min) & (u <= u_max)
        if self.check_vertical:
            in_box = in_box & (v >= (y1 - self.vertical_margin_px)) & (v <= (y2 + self.vertical_margin_px))

        if not np.any(in_box):
            return None

        box_pts = pts_valid[in_box]
        z_vals = box_pts[:, 2]
        min_z = float(np.min(z_vals))
        fg_mask = z_vals <= (min_z + self.foreground_margin_m)
        fg_pts = box_pts[fg_mask]
        if fg_pts.shape[0] == 0:
            return None

        median_pt = np.median(fg_pts, axis=0).astype(np.float64)
        return DepthEstimate(
            point_cam=median_pt,
            depth_m=float(median_pt[2]),
            method="lidar_fusion",
            low_confidence=False,
            num_points=int(fg_pts.shape[0]),
        )


class GroundPlaneFallback:
    """
    Ground-Plane Ray Intersection Depth Fallback.

    Casts a ray through the bounding box bottom-center pixel (u_c, y2) in
    camera_rgb_optical_frame and intersects it with the robot ground plane z_base = 0
    using the 4x4 transformation T_base_cam (camera_rgb_optical_frame -> base_link).
    Returns P_camera with low_confidence=True, rejecting upward/horizontal rays or
    intersections beyond max_range_m.
    """

    def __init__(
        self,
        max_range_m: float = 6.0,
        min_range_m: float = 0.10,
        ground_z_base: float = 0.0,
    ) -> None:
        self.max_range_m = float(max_range_m)
        self.min_range_m = float(min_range_m)
        self.ground_z_base = float(ground_z_base)

    def estimate(
        self,
        bbox_xyxy: Tuple[float, float, float, float],
        intrinsics: CameraIntrinsics,
        T_base_cam: Optional[np.ndarray] = None,
    ) -> Optional[DepthEstimate]:
        """
        Intersect ray through bottom-center (u_c, y2) with plane z_base = ground_z_base.
        """
        T_mat = default_t_base_cam() if T_base_cam is None else np.asarray(T_base_cam, dtype=np.float64).reshape(4, 4)

        x1 = min(float(bbox_xyxy[0]), float(bbox_xyxy[2]))
        x2 = max(float(bbox_xyxy[0]), float(bbox_xyxy[2]))
        y2 = max(float(bbox_xyxy[1]), float(bbox_xyxy[3]))
        u_c = 0.5 * (x1 + x2)
        v_b = y2

        # Normalized ray in camera optical frame at unit optical depth Z_c = 1.0
        ray_cam = np.array(
            [
                (u_c - intrinsics.cx) / intrinsics.fx,
                (v_b - intrinsics.cy) / intrinsics.fy,
                1.0,
            ],
            dtype=np.float64,
        )

        R_base_cam = T_mat[:3, :3]
        t_base_cam = T_mat[:3, 3]

        ray_base = R_base_cam @ ray_cam
        dz = float(ray_base[2])
        cam_z = float(t_base_cam[2]) - self.ground_z_base

        # Camera is above ground (cam_z > 0), so ray must point downward (dz < 0)
        if cam_z <= 0.0 or dz >= -1e-6:
            return None

        scale_z = -cam_z / dz
        if scale_z < self.min_range_m or scale_z > self.max_range_m:
            return None

        point_cam = (scale_z * ray_cam).astype(np.float64)
        # Also verify horizontal ground range in base_link frame does not exceed max_range_m
        point_base = R_base_cam @ point_cam + t_base_cam
        horiz_dist = float(math.hypot(point_base[0], point_base[1]))
        if horiz_dist > self.max_range_m or horiz_dist < self.min_range_m:
            return None

        return DepthEstimate(
            point_cam=point_cam,
            depth_m=float(point_cam[2]),
            method="ground_plane_fallback",
            low_confidence=True,
            num_points=1,
        )


class RgbdDepthAdapter:
    """
    Aligned RGB-D Depth Image Adapter (Paper Original 3D Depth Camera Slot).

    Accepts an aligned depth image in float32 (meters) or uint16 (millimeters),
    extracts the median valid depth inside the center region of bbox_xyxy, and
    back-projects the bounding box center (u_c, v_c) into 3D camera coordinates P_camera.
    """

    def __init__(
        self,
        min_range_m: float = 0.10,
        max_range_m: float = 6.0,
        center_ratio: float = 0.50,
        uint16_scale_m: float = 0.001,
    ) -> None:
        self.min_range_m = float(min_range_m)
        self.max_range_m = float(max_range_m)
        self.center_ratio = float(center_ratio)
        self.uint16_scale_m = float(uint16_scale_m)

    def estimate(
        self,
        bbox_xyxy: Tuple[float, float, float, float],
        intrinsics: CameraIntrinsics,
        depth_image: np.ndarray,
    ) -> Optional[DepthEstimate]:
        """Estimate 3D camera point from an aligned depth image."""
        if depth_image is None:
            return None
        depth_arr = np.asarray(depth_image)
        if depth_arr.ndim != 2 or depth_arr.size == 0:
            return None

        if np.issubdtype(depth_arr.dtype, np.integer):
            depth_m_img = depth_arr.astype(np.float64) * self.uint16_scale_m
        else:
            depth_m_img = depth_arr.astype(np.float64)

        img_h, img_w = depth_m_img.shape
        x1 = min(float(bbox_xyxy[0]), float(bbox_xyxy[2]))
        y1 = min(float(bbox_xyxy[1]), float(bbox_xyxy[3]))
        x2 = max(float(bbox_xyxy[0]), float(bbox_xyxy[2]))
        y2 = max(float(bbox_xyxy[1]), float(bbox_xyxy[3]))

        w = x2 - x1
        h = y2 - y1
        u_c = 0.5 * (x1 + x2)
        v_c = 0.5 * (y1 + y2)

        half_w = max(0.5, 0.5 * self.center_ratio * w)
        half_h = max(0.5, 0.5 * self.center_ratio * h)

        c0 = max(0, min(img_w - 1, int(math.floor(u_c - half_w))))
        c1 = max(c0 + 1, min(img_w, int(math.ceil(u_c + half_w)) + 1))
        r0 = max(0, min(img_h - 1, int(math.floor(v_c - half_h))))
        r1 = max(r0 + 1, min(img_h, int(math.ceil(v_c + half_h)) + 1))

        roi = depth_m_img[r0:r1, c0:c1]
        valid = np.isfinite(roi) & (roi >= self.min_range_m) & (roi <= self.max_range_m)
        if not np.any(valid):
            return None

        valid_depths = roi[valid]
        median_depth = float(np.median(valid_depths))
        point_cam = intrinsics.backproject_pixel(u_c, v_c, median_depth)
        return DepthEstimate(
            point_cam=point_cam,
            depth_m=median_depth,
            method="rgbd_depth",
            low_confidence=False,
            num_points=int(valid_depths.size),
        )


class DepthEstimator:
    """
    Pluggable Depth Estimator supporting:
      - depth_source='lidar_fusion' (LidarProjectionDepth + optional GroundPlaneFallback)
      - depth_source='rgbd_depth'   (RgbdDepthAdapter + optional fallback)
    """

    def __init__(
        self,
        depth_source: str = "lidar_fusion",
        use_ground_fallback: bool = True,
        lidar_estimator: Optional[LidarProjectionDepth] = None,
        ground_fallback: Optional[GroundPlaneFallback] = None,
        rgbd_adapter: Optional[RgbdDepthAdapter] = None,
    ) -> None:
        if depth_source not in ("lidar_fusion", "rgbd_depth"):
            raise ValueError(f"Unsupported depth_source='{depth_source}'. Expected 'lidar_fusion' or 'rgbd_depth'.")
        self.depth_source = depth_source
        self.use_ground_fallback = bool(use_ground_fallback)
        self.lidar_estimator = lidar_estimator or LidarProjectionDepth()
        self.ground_fallback = ground_fallback or GroundPlaneFallback()
        self.rgbd_adapter = rgbd_adapter or RgbdDepthAdapter()

    def estimate(
        self,
        bbox_xyxy: Tuple[float, float, float, float],
        intrinsics: CameraIntrinsics,
        T_cam_scan: Optional[np.ndarray] = None,
        ranges: Optional[Union[Sequence[float], np.ndarray]] = None,
        angle_min: float = 0.0,
        angle_increment: float = 0.0,
        angles: Optional[np.ndarray] = None,
        scan_points: Optional[np.ndarray] = None,
        T_base_cam: Optional[np.ndarray] = None,
        depth_image: Optional[np.ndarray] = None,
        scan_ranges: Optional[Union[Sequence[float], np.ndarray]] = None,
        scan_angles: Optional[np.ndarray] = None,
    ) -> Optional[DepthEstimate]:
        """Run configured depth estimation strategy for bbox_xyxy."""
        if ranges is None and scan_ranges is not None:
            ranges = scan_ranges
        if angles is None and scan_angles is not None:
            angles = scan_angles

        if self.depth_source == "rgbd_depth":
            if depth_image is not None:
                est = self.rgbd_adapter.estimate(
                    bbox_xyxy=bbox_xyxy,
                    intrinsics=intrinsics,
                    depth_image=depth_image,
                )
                if est is not None:
                    return est
            if ranges is not None or scan_points is not None:
                est = self.lidar_estimator.estimate(
                    bbox_xyxy=bbox_xyxy,
                    intrinsics=intrinsics,
                    T_cam_scan=T_cam_scan,
                    ranges=ranges,
                    angle_min=angle_min,
                    angle_increment=angle_increment,
                    angles=angles,
                    scan_points=scan_points,
                )
                if est is not None:
                    return est
        else:
            if ranges is not None or scan_points is not None:
                est = self.lidar_estimator.estimate(
                    bbox_xyxy=bbox_xyxy,
                    intrinsics=intrinsics,
                    T_cam_scan=T_cam_scan,
                    ranges=ranges,
                    angle_min=angle_min,
                    angle_increment=angle_increment,
                    angles=angles,
                    scan_points=scan_points,
                )
                if est is not None:
                    return est

        if self.use_ground_fallback and self.ground_fallback is not None:
            return self.ground_fallback.estimate(
                bbox_xyxy=bbox_xyxy,
                intrinsics=intrinsics,
                T_base_cam=T_base_cam,
            )
        return None


def create_depth_estimator(
    depth_source: str = "lidar_fusion",
    use_ground_fallback: bool = True,
    min_lidar_range_m: float = 0.12,
    max_lidar_range_m: float = 3.50,
    center_ratio: float = 0.70,
    foreground_margin_m: float = 0.25,
    vertical_margin_px: float = 40.0,
    check_vertical: bool = True,
    max_ground_range_m: float = 6.0,
    min_ground_range_m: float = 0.10,
    max_rgbd_range_m: float = 6.0,
    min_range_m: Optional[float] = None,
    max_range_m: Optional[float] = None,
    enable_ground_fallback: Optional[bool] = None,
    **kwargs: Any,
) -> DepthEstimator:
    """Factory function to create a configured DepthEstimator."""
    eff_min_lidar = float(min_range_m) if min_range_m is not None else float(min_lidar_range_m)
    eff_max_lidar = float(max_range_m) if max_range_m is not None else float(max_lidar_range_m)
    eff_ground_fb = (
        bool(enable_ground_fallback) if enable_ground_fallback is not None else bool(use_ground_fallback)
    )
    lidar = LidarProjectionDepth(
        min_range_m=eff_min_lidar,
        max_range_m=eff_max_lidar,
        center_ratio=center_ratio,
        foreground_margin_m=foreground_margin_m,
        vertical_margin_px=vertical_margin_px,
        check_vertical=check_vertical,
    )
    ground = GroundPlaneFallback(
        max_range_m=max_ground_range_m,
        min_range_m=min_ground_range_m,
    )
    rgbd = RgbdDepthAdapter(
        min_range_m=min_ground_range_m,
        max_range_m=max_rgbd_range_m,
    )
    return DepthEstimator(
        depth_source=depth_source,
        use_ground_fallback=eff_ground_fb,
        lidar_estimator=lidar,
        ground_fallback=ground,
        rgbd_adapter=rgbd,
    )


# =============================================================================
# 4. SemanticMarkerManager (Paper Algorithm 1 Enhanced)
# =============================================================================

@dataclass
class SemanticObservation:
    """A single localized semantic detection candidate in map/world coordinates."""

    class_id: int
    class_name: str
    x: float
    y: float
    z: float = 0.0
    conf: float = 1.0
    stamp_sec: float = 0.0
    low_confidence: bool = False
    method: str = "lidar_fusion"

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)


@dataclass
class PermanentMarker:
    """A verified permanent semantic marker on the 2D/3D disaster map."""

    marker_id: int
    class_id: int
    class_name: str
    x: float
    y: float
    z: float = 0.0
    conf: float = 1.0
    hits: int = 1
    first_seen_sec: float = 0.0
    last_seen_sec: float = 0.0
    low_confidence: bool = False
    method: str = "lidar_fusion"

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "marker_id": int(self.marker_id),
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
            "x": float(self.x),
            "y": float(self.y),
            "z": float(self.z),
            "conf": float(self.conf),
            "hits": int(self.hits),
            "first_seen_sec": float(self.first_seen_sec),
            "last_seen_sec": float(self.last_seen_sec),
            "low_confidence": bool(self.low_confidence),
            "method": str(self.method),
        }


class SemanticMarkerManager:
    """
    Paper Algorithm 1 Enhanced Marker Verification & Deduplication Engine.

    - Maintains temporary observations (`temp_markers`) with TTL expiration (default 10.0 s).
    - Promotes a cluster of `>= min_hits` (default 5) temporary observations of the same
      `class_id` within `cluster_radius_m` (default 0.45 m) spanning `>= min_span_sec`
      (default 1.0 s) to a permanent marker (`perm_markers`) at their mean position.
    - Merges re-observations or promoted clusters into any existing permanent marker of the
      same `class_id` within `merge_radius_m` (default 0.60 m) via weighted average position
      and maximum confidence.
    - Immediately registers / merges multimodal disaster events (`GAS_HAZARD`, `AUDIO_DISTRESS`)
      via `register_event(...)`.
    """

    def __init__(
        self,
        cluster_radius_m: float = 0.45,
        merge_radius_m: float = 0.60,
        min_hits: int = 5,
        min_span_sec: float = 1.0,
        temp_ttl_sec: float = 10.0,
        max_angular_vel_rad_s: float = 0.5,
    ) -> None:
        self.cluster_radius_m = float(cluster_radius_m)
        self.merge_radius_m = float(merge_radius_m)
        self.min_hits = int(min_hits)
        self.min_span_sec = float(min_span_sec)
        self.temp_ttl_sec = float(temp_ttl_sec)
        self.max_angular_vel_rad_s = float(max_angular_vel_rad_s)

        self.temp_markers: List[SemanticObservation] = []
        self.perm_markers: List[PermanentMarker] = []
        self._next_marker_id: int = 1

    def prune_expired(self, current_stamp_sec: float) -> int:
        """
        Remove temporary markers older than temp_ttl_sec relative to current_stamp_sec.
        Returns the number of expired observations removed.
        """
        now = float(current_stamp_sec)
        before = len(self.temp_markers)
        self.temp_markers = [
            obs for obs in self.temp_markers if (now - obs.stamp_sec) <= self.temp_ttl_sec
        ]
        return before - len(self.temp_markers)

    def is_motion_gated(self, angular_vel_rad_s: float) -> bool:
        """Return True if |angular_vel_rad_s| exceeds max_angular_vel_rad_s."""
        return abs(float(angular_vel_rad_s)) > self.max_angular_vel_rad_s

    def _find_closest_perm_marker(
        self,
        class_id: int,
        class_name: str,
        x: float,
        y: float,
        radius_m: Optional[float] = None,
    ) -> Optional[PermanentMarker]:
        r_limit = self.merge_radius_m if radius_m is None else float(radius_m)
        best_pm: Optional[PermanentMarker] = None
        best_dist = float("inf")
        for pm in self.perm_markers:
            if pm.class_id != class_id and pm.class_name != class_name:
                continue
            dist = math.hypot(pm.x - x, pm.y - y)
            if dist <= r_limit and dist < best_dist:
                best_dist = dist
                best_pm = pm
        return best_pm

    @staticmethod
    def _merge_into_perm_marker(
        pm: PermanentMarker,
        x: float,
        y: float,
        z: float,
        conf: float,
        weight: int,
        first_seen_sec: float,
        last_seen_sec: float,
        low_confidence: bool,
        method: str,
    ) -> PermanentMarker:
        w_old = max(1, int(pm.hits))
        w_new = max(1, int(weight))
        total = w_old + w_new
        pm.x = float((pm.x * w_old + x * w_new) / total)
        pm.y = float((pm.y * w_old + y * w_new) / total)
        pm.z = float((pm.z * w_old + z * w_new) / total)
        pm.conf = float(max(pm.conf, conf))
        pm.hits = int(total)
        pm.first_seen_sec = float(min(pm.first_seen_sec, first_seen_sec))
        pm.last_seen_sec = float(max(pm.last_seen_sec, last_seen_sec))
        if not low_confidence and pm.low_confidence:
            pm.low_confidence = False
            pm.method = method
        return pm

    def add_observation(
        self,
        obs_or_class_id: Optional[Union[SemanticObservation, int]] = None,
        class_name: str = "",
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        conf: float = 1.0,
        stamp_sec: float = 0.0,
        low_confidence: bool = False,
        method: str = "lidar_fusion",
        angular_vel_rad_s: float = 0.0,
        class_id: Optional[int] = None,
    ) -> Optional[PermanentMarker]:
        """
        Add a semantic observation in map coordinates and run Algorithm 1 verification.

        Can be called either with a `SemanticObservation` instance:
            manager.add_observation(obs)
        or with positional/keyword fields:
            manager.add_observation(class_id, class_name, x, y, z, conf, stamp_sec, ...)

        Returns the created or updated PermanentMarker if a promotion or merge occurred,
        or None if the observation remains in temp_markers (or was motion-gated).
        """
        if self.is_motion_gated(angular_vel_rad_s):
            return None

        if isinstance(obs_or_class_id, SemanticObservation):
            obs = obs_or_class_id
        else:
            cid = class_id if class_id is not None else (0 if obs_or_class_id is None else int(obs_or_class_id))
            obs = SemanticObservation(
                class_id=int(cid),
                class_name=str(class_name),
                x=float(x),
                y=float(y),
                z=float(z),
                conf=float(conf),
                stamp_sec=float(stamp_sec),
                low_confidence=bool(low_confidence),
                method=str(method),
            )

        # 1. Expire stale temporary observations
        self.prune_expired(obs.stamp_sec)

        # 2. If an existing PermanentMarker of the same class lies within merge_radius_m,
        #    update it via weighted average position and max confidence instead of duplicating.
        existing_pm = self._find_closest_perm_marker(
            class_id=obs.class_id,
            class_name=obs.class_name,
            x=obs.x,
            y=obs.y,
            radius_m=self.merge_radius_m,
        )
        if existing_pm is not None:
            return self._merge_into_perm_marker(
                pm=existing_pm,
                x=obs.x,
                y=obs.y,
                z=obs.z,
                conf=obs.conf,
                weight=1,
                first_seen_sec=obs.stamp_sec,
                last_seen_sec=obs.stamp_sec,
                low_confidence=obs.low_confidence,
                method=obs.method,
            )

        # 3. Append to temporary markers and check cluster within cluster_radius_m
        self.temp_markers.append(obs)

        cluster_indices: List[int] = []
        for idx, candidate in enumerate(self.temp_markers):
            if candidate.class_id != obs.class_id:
                continue
            if math.hypot(candidate.x - obs.x, candidate.y - obs.y) <= self.cluster_radius_m:
                cluster_indices.append(idx)

        if len(cluster_indices) < self.min_hits:
            return None

        cluster = [self.temp_markers[i] for i in cluster_indices]
        stamps = [c.stamp_sec for c in cluster]
        span_sec = max(stamps) - min(stamps)
        if span_sec < self.min_span_sec:
            return None

        # 4. Promotion condition satisfied (hits >= min_hits and span >= min_span_sec)
        avg_x = float(np.mean([c.x for c in cluster]))
        avg_y = float(np.mean([c.y for c in cluster]))
        avg_z = float(np.mean([c.z for c in cluster]))
        max_conf = float(max(c.conf for c in cluster))
        first_seen = float(min(stamps))
        last_seen = float(max(stamps))
        all_low_conf = all(c.low_confidence for c in cluster)
        chosen_method = next((c.method for c in cluster if not c.low_confidence), cluster[-1].method)

        index_set = set(cluster_indices)
        self.temp_markers = [
            m for i, m in enumerate(self.temp_markers) if i not in index_set
        ]

        # Check again whether the promoted cluster centroid falls within merge_radius_m
        # of an existing permanent marker
        merge_target = self._find_closest_perm_marker(
            class_id=obs.class_id,
            class_name=obs.class_name,
            x=avg_x,
            y=avg_y,
            radius_m=self.merge_radius_m,
        )
        if merge_target is not None:
            return self._merge_into_perm_marker(
                pm=merge_target,
                x=avg_x,
                y=avg_y,
                z=avg_z,
                conf=max_conf,
                weight=len(cluster),
                first_seen_sec=first_seen,
                last_seen_sec=last_seen,
                low_confidence=all_low_conf,
                method=chosen_method,
            )

        new_pm = PermanentMarker(
            marker_id=self._next_marker_id,
            class_id=obs.class_id,
            class_name=obs.class_name,
            x=avg_x,
            y=avg_y,
            z=avg_z,
            conf=max_conf,
            hits=len(cluster),
            first_seen_sec=first_seen,
            last_seen_sec=last_seen,
            low_confidence=all_low_conf,
            method=chosen_method,
        )
        self._next_marker_id += 1
        self.perm_markers.append(new_pm)
        return new_pm

    # Alias for convenience
    process_observation = add_observation

    def register_event(
        self,
        class_name: str,
        x: float,
        y: float,
        z: float = 0.0,
        conf: float = 1.0,
        stamp_sec: float = 0.0,
        class_id: Optional[int] = None,
        merge_radius_m: Optional[float] = None,
    ) -> PermanentMarker:
        """
        Immediately register or merge a multimodal disaster event (e.g. 'GAS_HAZARD',
        'AUDIO_DISTRESS') as a PermanentMarker.
        """
        cid = int(class_id) if class_id is not None else EVENT_CLASS_IDS.get(class_name, 1000)
        r_merge = self.merge_radius_m if merge_radius_m is None else float(merge_radius_m)

        existing = self._find_closest_perm_marker(
            class_id=cid,
            class_name=class_name,
            x=float(x),
            y=float(y),
            radius_m=r_merge,
        )
        if existing is not None:
            return self._merge_into_perm_marker(
                pm=existing,
                x=float(x),
                y=float(y),
                z=float(z),
                conf=float(conf),
                weight=1,
                first_seen_sec=float(stamp_sec),
                last_seen_sec=float(stamp_sec),
                low_confidence=False,
                method="event",
            )

        pm = PermanentMarker(
            marker_id=self._next_marker_id,
            class_id=cid,
            class_name=str(class_name),
            x=float(x),
            y=float(y),
            z=float(z),
            conf=float(conf),
            hits=1,
            first_seen_sec=float(stamp_sec),
            last_seen_sec=float(stamp_sec),
            low_confidence=False,
            method="event",
        )
        self._next_marker_id += 1
        self.perm_markers.append(pm)
        return pm

    def get_permanent_markers(self) -> List[PermanentMarker]:
        """Return a shallow copy of current permanent semantic markers."""
        return list(self.perm_markers)

    def to_dict(self) -> Dict[str, Any]:
        """Export current semantic map state as a JSON-serializable dictionary."""
        marker_dicts = [m.to_dict() for m in self.perm_markers]
        return {
            "num_permanent_markers": len(marker_dicts),
            "num_temp_markers": len(self.temp_markers),
            "parameters": {
                "cluster_radius_m": self.cluster_radius_m,
                "merge_radius_m": self.merge_radius_m,
                "min_hits": self.min_hits,
                "min_span_sec": self.min_span_sec,
                "temp_ttl_sec": self.temp_ttl_sec,
                "max_angular_vel_rad_s": self.max_angular_vel_rad_s,
            },
            "markers": marker_dicts,
            "objects": marker_dicts,
        }


# =============================================================================
# 5. 2D Map Overlay & Coordinate Transforms (Paper Algorithm 2)
# =============================================================================

def world_to_pixel(
    x: float,
    y: float,
    origin_x: float,
    origin_y: float,
    resolution: float,
    height: int,
) -> Tuple[int, int]:
    """
    Convert world/map coordinates (x, y) [m] to top-down OpenCV image pixel coordinates (px, py)
    following Paper Algorithm 2 with OccupancyGrid bottom-up vertical flip:
      px = int((x - origin_x) / resolution)
      py = height - 1 - int((y - origin_y) / resolution)
    """
    res = float(resolution)
    if res <= 0.0:
        raise ValueError("resolution must be strictly positive.")
    h = int(height)
    px = int((float(x) - float(origin_x)) / res)
    py = h - 1 - int((float(y) - float(origin_y)) / res)
    return px, py


def pixel_to_world(
    px: Union[int, float],
    py: Union[int, float],
    origin_x: float,
    origin_y: float,
    resolution: float,
    height: int,
) -> Tuple[float, float]:
    """
    Convert top-down OpenCV image pixel coordinates (px, py) back to world/map coordinates (x, y) [m]
    at the pixel cell center.
    """
    res = float(resolution)
    if res <= 0.0:
        raise ValueError("resolution must be strictly positive.")
    h = int(height)
    grid_col = float(px) + 0.5
    grid_row = float(h - 1 - float(py)) + 0.5
    x = float(origin_x) + grid_col * res
    y = float(origin_y) + grid_row * res
    return float(x), float(y)


def occupancy_grid_to_image(
    occupancy_grid: Union[Sequence[int], np.ndarray],
    width: int,
    height: int,
    flip_vertical: bool = True,
) -> np.ndarray:
    """
    Convert a ROS OccupancyGrid data array (-1=unknown, 0=free, 100=occupied) into a
    2D uint8 grayscale image (height, width), flipped vertically to match top-down OpenCV format.
    """
    w = int(width)
    h = int(height)
    arr = np.asarray(occupancy_grid)

    if arr.ndim == 3 and arr.shape[2] == 3:
        img = arr.astype(np.uint8).copy()
        return np.flipud(img).copy() if flip_vertical else img

    grid = arr.reshape((h, w))
    # If already a 0..255 uint8 grayscale image (with values > 100), keep intensities directly
    if grid.dtype == np.uint8 and int(np.max(grid)) > 100:
        gray = grid.copy()
    else:
        g_int = grid.astype(np.int16)
        gray = np.full((h, w), 205, dtype=np.uint8)  # Unknown (-1) -> 205 gray
        free_mask = (g_int >= 0) & (g_int < 25)
        occ_mask = g_int >= 65
        mid_mask = (g_int >= 25) & (g_int < 65)
        gray[free_mask] = 254
        gray[occ_mask] = 0
        if np.any(mid_mask):
            gray[mid_mask] = np.clip(
                (254.0 * (100 - g_int[mid_mask]) / 100.0), 0, 254
            ).astype(np.uint8)

    if flip_vertical:
        gray = np.flipud(gray).copy()
    return gray


def _parse_bgr_color(color_val: Any) -> Tuple[int, int, int]:
    """Normalize a palette color (tuple/list [B, G, R] or hex '#RRGGBB') into a BGR int tuple."""
    if isinstance(color_val, str):
        s = color_val.strip().lstrip("#")
        if len(s) == 6:
            r = int(s[0:2], 16)
            g = int(s[2:4], 16)
            b = int(s[4:6], 16)
            return (b, g, r)
    if isinstance(color_val, (tuple, list, np.ndarray)) and len(color_val) >= 3:
        return (int(color_val[0]), int(color_val[1]), int(color_val[2]))
    return (0, 200, 100)


def render_semantic_map_overlay(
    occupancy_grid: Union[Sequence[int], np.ndarray],
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    markers: Iterable[Union[PermanentMarker, SemanticObservation, Mapping[str, Any]]],
    palette: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    """
    Paper Algorithm 2 Implementation:
    Render permanent semantic markers (colored circles, class labels, and legend) onto
    a vertically flipped 2D OccupancyGrid BGR image.
    """
    w = int(width)
    h = int(height)
    base_gray_or_bgr = occupancy_grid_to_image(occupancy_grid, w, h, flip_vertical=True)
    if base_gray_or_bgr.ndim == 2:
        canvas = cv2.cvtColor(base_gray_or_bgr, cv2.COLOR_GRAY2BGR)
    else:
        canvas = base_gray_or_bgr.copy()

    color_map: Dict[str, Tuple[int, int, int]] = dict(DEFAULT_CLASS_PALETTE)
    if palette:
        for k, v in palette.items():
            color_map[str(k)] = _parse_bgr_color(v)

    seen_classes: List[str] = []

    for item in markers:
        if isinstance(item, Mapping):
            cls_name = str(item.get("class_name", "object"))
            mx = float(item.get("x", 0.0))
            my = float(item.get("y", 0.0))
            low_conf = bool(item.get("low_confidence", False))
        else:
            cls_name = str(getattr(item, "class_name", "object"))
            mx = float(getattr(item, "x", 0.0))
            my = float(getattr(item, "y", 0.0))
            low_conf = bool(getattr(item, "low_confidence", False))

        color = color_map.get(cls_name, color_map.get(cls_name.lower(), (0, 200, 100)))
        if cls_name not in seen_classes:
            seen_classes.append(cls_name)
            color_map[cls_name] = color

        px, py = world_to_pixel(mx, my, origin_x, origin_y, resolution, h)
        if 0 <= px < w and 0 <= py < h:
            radius = 4 if low_conf else 5
            cv2.circle(canvas, (px, py), radius, color, -1, lineType=cv2.LINE_AA)
            border_color = (180, 180, 180) if low_conf else (20, 20, 20)
            cv2.circle(canvas, (px, py), radius + 1, border_color, 1, lineType=cv2.LINE_AA)

            label = f"{cls_name}?" if low_conf else cls_name
            tx = min(max(2, px + radius + 3), max(2, w - 60))
            ty = min(max(12, py + 4), max(12, h - 4))
            cv2.putText(
                canvas,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (15, 15, 15),
                2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                lineType=cv2.LINE_AA,
            )

    # Render compact legend box in the top-left corner when markers are present and image is large enough
    if seen_classes and w >= 80 and h >= 40:
        row_h = 14
        box_h = min(h - 4, 8 + len(seen_classes) * row_h)
        box_w = min(w - 4, 125)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (3, 3), (3 + box_w, 3 + box_h), (245, 245, 245), -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
        cv2.rectangle(canvas, (3, 3), (3 + box_w, 3 + box_h), (80, 80, 80), 1)

        for idx, cls_name in enumerate(seen_classes):
            cy_leg = 12 + idx * row_h
            if cy_leg + 4 > h:
                break
            c_bgr = color_map.get(cls_name, (0, 200, 100))
            cv2.circle(canvas, (11, cy_leg), 4, c_bgr, -1, lineType=cv2.LINE_AA)
            cv2.putText(
                canvas,
                cls_name[:16],
                (19, cy_leg + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.33,
                (20, 20, 20),
                1,
                lineType=cv2.LINE_AA,
            )

    return canvas


def save_semantic_map_bundle(
    output_dir: str,
    occupancy_grid: Union[Sequence[int], np.ndarray],
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    marker_manager: SemanticMarkerManager,
    palette: Optional[Mapping[str, Any]] = None,
    origin_yaw: float = 0.0,
) -> Dict[str, str]:
    """
    Save the full 2D disaster semantic map artifact bundle into `output_dir`:
      1. raw_map.png
      2. raw_map.yaml
      3. semantic_objects.json
      4. semantic_map_overlay.png
    Returns a dictionary of generated file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_png_path = os.path.join(output_dir, "raw_map.png")
    raw_yaml_path = os.path.join(output_dir, "raw_map.yaml")
    json_path = os.path.join(output_dir, "semantic_objects.json")
    overlay_png_path = os.path.join(output_dir, "semantic_map_overlay.png")

    raw_img = occupancy_grid_to_image(occupancy_grid, width, height, flip_vertical=True)
    cv2.imwrite(raw_png_path, raw_img)

    yaml_content = (
        "image: raw_map.png\n"
        f"resolution: {float(resolution):.6f}\n"
        f"origin: [{float(origin_x):.6f}, {float(origin_y):.6f}, {float(origin_yaw):.6f}]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.196\n"
    )
    with open(raw_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    payload = marker_manager.to_dict()
    payload["map_metadata"] = {
        "width": int(width),
        "height": int(height),
        "resolution": float(resolution),
        "origin_x": float(origin_x),
        "origin_y": float(origin_y),
        "origin_yaw": float(origin_yaw),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    overlay_img = render_semantic_map_overlay(
        occupancy_grid=occupancy_grid,
        width=width,
        height=height,
        resolution=resolution,
        origin_x=origin_x,
        origin_y=origin_y,
        markers=marker_manager.perm_markers,
        palette=palette,
    )
    cv2.imwrite(overlay_png_path, overlay_img)

    return {
        "raw_map_png": raw_png_path,
        "raw_map_yaml": raw_yaml_path,
        "semantic_objects_json": json_path,
        "semantic_map_overlay_png": overlay_png_path,
    }


__all__ = [
    "EVENT_CLASS_IDS",
    "DEFAULT_CLASS_PALETTE",
    "quat_to_rotation_matrix",
    "pose_to_matrix",
    "rpy_to_matrix",
    "default_t_base_cam",
    "default_t_cam_scan",
    "transform_point",
    "transform_points",
    "find_closest_timed_item",
    "unrotate_point",
    "unrotate_bbox_xyxy",
    "CameraIntrinsics",
    "DepthEstimate",
    "LidarProjectionDepth",
    "GroundPlaneFallback",
    "RgbdDepthAdapter",
    "DepthEstimator",
    "create_depth_estimator",
    "SemanticObservation",
    "PermanentMarker",
    "SemanticMarkerManager",
    "world_to_pixel",
    "pixel_to_world",
    "occupancy_grid_to_image",
    "render_semantic_map_overlay",
    "save_semantic_map_bundle",
]
