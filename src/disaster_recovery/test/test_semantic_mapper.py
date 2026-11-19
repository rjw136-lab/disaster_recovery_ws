#!/usr/bin/env python3
"""
Unit tests for pure-Python / NumPy / OpenCV semantic_mapper_core.py.

Covers:
  1. Paper Eq. 1-3 3D->2D projection with Eq. 3 Y-axis fix (< 1 px error)
  2. 270 / 90 / 180 / 0 deg image rotation unmapping (< 1 px error)
  3. LiDAR 2D projection & foreground clustering vs background wall rejection
  4. GroundPlaneFallback & RgbdDepthAdapter & pluggable DepthEstimator
  5. Paper Algorithm 1: TTL expiration, min_hits + time-span promotion,
     weighted-average merge deduplication, GAS_HAZARD & AUDIO_DISTRESS events
  6. Paper Algorithm 2: world<->pixel transforms (< 1 px error) & 2D BGR map overlay rendering
"""

import json
import math
import os
import sys

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
import disaster_recovery as _dr_pkg
_SRC_PKG_DIR = os.path.join(_PKG_ROOT, "disaster_recovery")
if hasattr(_dr_pkg, "__path__") and _SRC_PKG_DIR not in _dr_pkg.__path__:
    _dr_pkg.__path__.insert(0, _SRC_PKG_DIR)

import cv2
import numpy as np
import pytest

from disaster_recovery.semantic_mapper_core import (
    CameraIntrinsics,
    DepthEstimate,
    DepthEstimator,
    GroundPlaneFallback,
    LidarProjectionDepth,
    PermanentMarker,
    RgbdDepthAdapter,
    SemanticMarkerManager,
    SemanticObservation,
    create_depth_estimator,
    default_t_base_cam,
    default_t_cam_scan,
    find_closest_timed_item,
    pixel_to_world,
    render_semantic_map_overlay,
    save_semantic_map_bundle,
    transform_point,
    unrotate_bbox_xyxy,
    unrotate_point,
    world_to_pixel,
)
import disaster_recovery.semantic_mapper_core as core_mod


def test_no_rclpy_dependency():
    """Verify semantic_mapper_core does not import rclpy."""
    assert "rclpy" not in core_mod.__dict__


# =============================================================================
# 1. Rotation Unmapping Tests (270, 90, 180, 0 deg)
# =============================================================================

@pytest.mark.parametrize(
    "angle,cv_rot_code",
    [
        (0, None),
        (90, cv2.ROTATE_90_CLOCKWISE),
        (180, cv2.ROTATE_180),
        (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ],
)
def test_unrotate_point_exact_match_with_opencv(angle, cv_rot_code):
    """Verify unrotate_point inverts OpenCV rotation with < 1 px (0.0 px) error."""
    orig_w, orig_h = 640, 480
    # Encode original (u, v) coordinates into a 2-channel array
    u_grid, v_grid = np.meshgrid(
        np.arange(orig_w, dtype=np.float32),
        np.arange(orig_h, dtype=np.float32),
    )
    coord_img = np.stack([u_grid, v_grid], axis=-1)

    if cv_rot_code is None:
        rot_img = coord_img
    else:
        rot_img = cv2.rotate(coord_img, cv_rot_code)

    rot_h, rot_w = rot_img.shape[:2]
    sample_us = [0, 1, rot_w // 4, rot_w // 2, (3 * rot_w) // 4, rot_w - 1]
    sample_vs = [0, 1, rot_h // 4, rot_h // 2, (3 * rot_h) // 4, rot_h - 1]

    for vr in sample_vs:
        for ur in sample_us:
            expected_u, expected_v = rot_img[vr, ur]
            u_rec, v_rec = unrotate_point(ur, vr, orig_w, orig_h, angle)
            err = math.hypot(u_rec - float(expected_u), v_rec - float(expected_v))
            assert err < 1.0, f"angle={angle}: err={err} >= 1 px"
            assert err == pytest.approx(0.0, abs=1e-6)


def test_unrotate_bbox_xyxy_all_angles():
    """Verify unrotate_bbox_xyxy preserves sorted bounds and < 1 px error across 0/90/180/270 deg."""
    orig_w, orig_h = 640, 480
    orig_box = (120.0, 80.0, 260.0, 300.0)

    # 270 deg CCW: u_rot = v, v_rot = W - 1 - u
    rot_270_box = (
        orig_box[1],
        (orig_w - 1) - orig_box[2],
        orig_box[3],
        (orig_w - 1) - orig_box[0],
    )
    rec_270 = unrotate_bbox_xyxy(rot_270_box, orig_w, orig_h, 270)
    assert rec_270[0] <= rec_270[2]
    assert rec_270[1] <= rec_270[3]
    for got, exp in zip(rec_270, orig_box):
        assert abs(got - exp) < 1.0
        assert got == pytest.approx(exp, abs=1e-6)

    # 90 deg CW: u_rot = H - 1 - v, v_rot = u
    rot_90_box = (
        (orig_h - 1) - orig_box[3],
        orig_box[0],
        (orig_h - 1) - orig_box[1],
        orig_box[2],
    )
    rec_90 = unrotate_bbox_xyxy(rot_90_box, orig_w, orig_h, 90)
    for got, exp in zip(rec_90, orig_box):
        assert abs(got - exp) < 1.0
        assert got == pytest.approx(exp, abs=1e-6)

    # 180 deg: u_rot = W - 1 - u, v_rot = H - 1 - v
    rot_180_box = (
        (orig_w - 1) - orig_box[2],
        (orig_h - 1) - orig_box[3],
        (orig_w - 1) - orig_box[0],
        (orig_h - 1) - orig_box[1],
    )
    rec_180 = unrotate_bbox_xyxy(rot_180_box, orig_w, orig_h, 180)
    for got, exp in zip(rec_180, orig_box):
        assert abs(got - exp) < 1.0
        assert got == pytest.approx(exp, abs=1e-6)

    # 0 deg
    rec_0 = unrotate_bbox_xyxy(orig_box, orig_w, orig_h, 0)
    for got, exp in zip(rec_0, orig_box):
        assert got == pytest.approx(exp, abs=1e-6)


# =============================================================================
# 2. CameraIntrinsics & Paper Eq. 1-3 (Y-axis fix) Tests
# =============================================================================

def test_camera_intrinsics_eq3_y_typo_fixed_and_subpixel_accuracy():
    """
    Verify Paper Eq. 1-3 projection uses v = (Y * fy) / Z + cy (not X * fy)
    and achieves < 1 px error against ground-truth pinhole projection.
    """
    intr = CameraIntrinsics(fx=530.1254, fy=529.8971, cx=318.4521, cy=239.1124, width=640, height=480)

    # Point strictly on Y-axis (X = 0, Y = 0.4 m, Z = 2.0 m):
    # With Y*fy fix: u = cx, v = (0.4 * 529.8971) / 2.0 + cy
    # With paper typo (X*fy): v would falsely equal cy!
    pts = np.array(
        [
            [0.0, 0.4, 2.0],
            [-0.5, 0.25, 1.5],
            [0.3, -0.2, -1.0],  # Behind camera (Z <= 0), must be filtered out
        ],
        dtype=np.float64,
    )
    u, v, valid_mask = intr.project_points(pts)
    assert valid_mask.tolist() == [True, True, False]
    assert u.shape == (2,)
    assert v.shape == (2,)

    expected_u0 = intr.cx
    expected_v0 = (0.4 * intr.fy) / 2.0 + intr.cy
    assert abs(u[0] - expected_u0) < 1.0
    assert abs(v[0] - expected_v0) < 1.0
    assert v[0] != pytest.approx(intr.cy)

    # Round-trip backproject -> project across image grid (< 1 px error)
    fov_intr = CameraIntrinsics.from_fov(width=640, height=480, horizontal_fov=1.085595)
    for u_gt in (50.0, 180.5, 320.0, 510.25, 610.0):
        for v_gt in (40.0, 120.75, 240.0, 390.5, 450.0):
            pt_3d = fov_intr.backproject_pixel(u_gt, v_gt, depth_m=2.3)
            u_proj, v_proj = fov_intr.project_point(pt_3d)
            pixel_err = math.hypot(u_proj - u_gt, v_proj - v_gt)
            assert pixel_err < 1.0
            assert pixel_err == pytest.approx(0.0, abs=1e-6)


# =============================================================================
# 3. Pluggable Depth Estimation Tests
# =============================================================================

def test_lidar_projection_depth_foreground_cluster_rejects_background_wall():
    """
    Verify LidarProjectionDepth selects center 70% horizontal rays, clusters foreground
    points within min_z + 0.25 m, and rejects background wall rays at 3.1 m.
    """
    intr = CameraIntrinsics.from_fov(640, 480, 1.085595)
    T_cam_scan = default_t_cam_scan()
    estimator = LidarProjectionDepth(
        min_range_m=0.12,
        max_range_m=3.50,
        center_ratio=0.70,
        foreground_margin_m=0.25,
        vertical_margin_px=40.0,
    )

    # Create 360-ray LaserScan around front of robot (angle 0 = +X_scan forward)
    num_rays = 360
    angle_min = -math.pi
    angle_inc = (2.0 * math.pi) / num_rays
    angles = angle_min + np.arange(num_rays) * angle_inc

    # Default background wall at 3.10 m everywhere
    ranges = np.full(num_rays, 3.10, dtype=np.float64)

    # Foreground survivor / obstacle directly ahead (|angle| <= 4 deg) at 1.50 m in scan frame
    fg_mask = np.abs(angles) <= math.radians(4.0)
    ranges[fg_mask] = 1.50

    # Inject invalid rays (< 0.12 m, > 3.50 m, NaN) to verify filtering
    ranges[0] = 0.05
    ranges[1] = 4.80
    ranges[2] = float("nan")

    # Bounding box covering [-10 deg, +10 deg] horizontally and the scan plane vertically
    # Notice rays between 4..7 deg hit the 3.10 m background wall INSIDE the 70% center box!
    bbox_xyxy = (220.0, 160.0, 420.0, 340.0)

    est = estimator.estimate(
        bbox_xyxy=bbox_xyxy,
        intrinsics=intr,
        T_cam_scan=T_cam_scan,
        ranges=ranges,
        angle_min=angle_min,
        angle_increment=angle_inc,
    )
    assert est is not None
    assert est.method == "lidar_fusion"
    assert est.low_confidence is False
    assert est.num_points >= 3

    # Expected camera optical Z_c = X_scan + (x_scan_in_base - x_cam_in_base) = 1.50 - 0.072 = 1.428 m
    expected_z_cam = 1.50 - 0.072
    assert est.depth_m == pytest.approx(expected_z_cam, abs=0.03)
    assert est.depth_m < 2.0, "Background wall at 3.10 m must be rejected by foreground clustering!"


def test_ground_plane_fallback_and_upward_ray_rejection():
    """
    Verify GroundPlaneFallback intersects bottom-center ray with z_base = 0 plane,
    marks low_confidence=True, and rejects upward rays or distances > max_range_m.
    """
    intr = CameraIntrinsics.from_fov(640, 480, 1.085595)
    T_base_cam = default_t_base_cam(tx=0.040, ty=0.0, tz=0.130)
    fallback = GroundPlaneFallback(max_range_m=5.0, min_range_m=0.10)

    # Known ground point in base_link frame: (X_base = 1.24 m, Y_base = -0.15 m, Z_base = 0.0 m)
    pt_base_gt = np.array([1.24, -0.15, 0.0], dtype=np.float64)
    T_cam_base = np.linalg.inv(T_base_cam)
    pt_cam_gt = transform_point(pt_base_gt, T_cam_base)
    u_gt, v_gt = intr.project_point(pt_cam_gt)

    bbox = (u_gt - 30.0, v_gt - 60.0, u_gt + 30.0, v_gt)
    est = fallback.estimate(bbox_xyxy=bbox, intrinsics=intr, T_base_cam=T_base_cam)
    assert est is not None
    assert est.low_confidence is True
    assert est.method == "ground_plane_fallback"
    assert np.allclose(est.point_cam, pt_cam_gt, atol=1e-4)

    # Transform back to base_link and verify Z_base == 0.0
    pt_base_rec = transform_point(est.point_cam, T_base_cam)
    assert np.allclose(pt_base_rec, pt_base_gt, atol=1e-4)

    # Upward ray (y2 < cy = 240.0) cannot intersect ground plane z_base=0 in front of camera
    upward_bbox = (280.0, 50.0, 360.0, 180.0)
    assert fallback.estimate(upward_bbox, intr, T_base_cam) is None

    # Nearly horizontal ray (y2 just barely below cy=240.0) exceeds max_range_m=5.0
    far_bbox = (280.0, 200.0, 360.0, 241.0)
    assert fallback.estimate(far_bbox, intr, T_base_cam) is None


def test_rgbd_depth_adapter_float32_and_uint16_and_pluggable_estimator():
    """Verify RgbdDepthAdapter handles both float32 (m) and uint16 (mm) depth maps and pluggable facade."""
    intr = CameraIntrinsics.from_fov(640, 480, 1.085595)
    adapter = RgbdDepthAdapter(center_ratio=0.50)

    # 1. Float32 meters
    depth_f32 = np.full((480, 640), 4.5, dtype=np.float32)
    depth_f32[190:290, 270:370] = 1.85
    bbox = (260.0, 180.0, 380.0, 300.0)

    est_f32 = adapter.estimate(bbox, intr, depth_f32)
    assert est_f32 is not None
    assert est_f32.method == "rgbd_depth"
    assert est_f32.low_confidence is False
    assert est_f32.depth_m == pytest.approx(1.85, abs=1e-4)

    # 2. Uint16 millimeters (1850 mm = 1.85 m)
    depth_u16 = (depth_f32 * 1000.0).astype(np.uint16)
    est_u16 = adapter.estimate(bbox, intr, depth_u16)
    assert est_u16 is not None
    assert est_u16.depth_m == pytest.approx(1.85, abs=1e-3)

    # 3. Pluggable DepthEstimator: lidar_fusion falling back to ground_plane_fallback when scan is empty
    estimator_lidar = create_depth_estimator(depth_source="lidar_fusion", use_ground_fallback=True)
    est_fb = estimator_lidar.estimate(
        bbox_xyxy=(280.0, 260.0, 360.0, 360.0),
        intrinsics=intr,
        ranges=np.zeros(360),  # All invalid (< 0.12 m)
        T_base_cam=default_t_base_cam(),
    )
    assert est_fb is not None
    assert est_fb.method == "ground_plane_fallback"
    assert est_fb.low_confidence is True

    # 4. Pluggable DepthEstimator: rgbd_depth mode
    estimator_rgbd = create_depth_estimator(depth_source="rgbd_depth")
    est_rgbd = estimator_rgbd.estimate(
        bbox_xyxy=bbox,
        intrinsics=intr,
        depth_image=depth_u16,
    )
    assert est_rgbd is not None
    assert est_rgbd.method == "rgbd_depth"
    assert est_rgbd.depth_m == pytest.approx(1.85, abs=1e-3)


# =============================================================================
# 4. SemanticMarkerManager (Paper Algorithm 1 Enhanced) Tests
# =============================================================================

def test_marker_manager_ttl_expiration():
    """Verify temporary observations older than temp_ttl_sec (10.0 s) expire and do not promote."""
    mgr = SemanticMarkerManager(
        cluster_radius_m=0.45,
        merge_radius_m=0.60,
        min_hits=5,
        min_span_sec=1.0,
        temp_ttl_sec=10.0,
    )
    # Add 3 observations at t = 0..1 s
    for i in range(3):
        res = mgr.add_observation(0, "person", 2.0, 1.0, 0.0, conf=0.80, stamp_sec=float(i) * 0.5)
        assert res is None
    assert len(mgr.temp_markers) == 3

    # Next observation arrives at t = 12.0 s (> 10.0 s TTL from t=0..1 s)
    res = mgr.add_observation(0, "person", 2.0, 1.0, 0.0, conf=0.85, stamp_sec=12.0)
    assert res is None
    assert len(mgr.perm_markers) == 0
    assert len(mgr.temp_markers) == 1


def test_marker_manager_promotion_requires_min_hits_and_time_span():
    """
    Verify promotion requires BOTH >= min_hits (5) within cluster_radius_m (0.45 m)
    AND time span >= min_span_sec (1.0 s).
    """
    mgr = SemanticMarkerManager(
        cluster_radius_m=0.45,
        merge_radius_m=0.60,
        min_hits=5,
        min_span_sec=1.0,
        temp_ttl_sec=10.0,
    )

    # 5 hits within 0.4 s (< 1.0 s min_span_sec) must NOT promote yet
    positions = [
        (3.00, -1.00),
        (3.04, -0.98),
        (2.96, -1.02),
        (3.02, -1.01),
        (2.98, -0.99),
    ]
    for i, (px, py) in enumerate(positions):
        pm = mgr.add_observation(
            0, "person", px, py, 0.0, conf=0.70 + 0.02 * i, stamp_sec=100.0 + i * 0.1
        )
        assert pm is None
    assert len(mgr.perm_markers) == 0
    assert len(mgr.temp_markers) == 5

    # 6th hit at t = 101.2 s (span = 1.2 s >= 1.0 s) triggers promotion!
    pm = mgr.add_observation(0, "person", 3.00, -1.00, 0.0, conf=0.92, stamp_sec=101.2)
    assert pm is not None
    assert len(mgr.perm_markers) == 1
    assert len(mgr.temp_markers) == 0
    assert pm.class_id == 0
    assert pm.class_name == "person"
    assert pm.hits == 6
    assert pm.conf == pytest.approx(0.92)
    assert pm.x == pytest.approx(3.00, abs=0.01)
    assert pm.y == pytest.approx(-1.00, abs=0.01)


def test_marker_manager_merge_deduplication_and_events():
    """
    Verify observations within merge_radius_m (0.60 m) of an existing permanent marker
    merge via weighted average position and max confidence, and GAS_HAZARD / AUDIO_DISTRESS
    register immediately.
    """
    mgr = SemanticMarkerManager(
        cluster_radius_m=0.45,
        merge_radius_m=0.60,
        min_hits=5,
        min_span_sec=1.0,
        temp_ttl_sec=10.0,
    )

    # Promote initial 'chair' permanent marker at (1.0, 2.0) with 5 hits
    for i in range(5):
        pm = mgr.add_observation(56, "chair", 1.0, 2.0, 0.0, conf=0.75, stamp_sec=float(i) * 0.3)
    assert pm is not None
    assert len(mgr.perm_markers) == 1
    assert pm.hits == 5

    # Re-observe same class within merge_radius_m (distance = 0.30 m <= 0.60 m) at (1.30, 2.0)
    merged = mgr.add_observation(56, "chair", 1.30, 2.0, 0.0, conf=0.95, stamp_sec=2.0)
    assert merged is not None
    assert len(mgr.perm_markers) == 1
    assert merged.marker_id == pm.marker_id
    assert merged.hits == 6
    assert merged.conf == pytest.approx(0.95)
    # Weighted average x = (1.0 * 5 + 1.30 * 1) / 6 = 1.05
    assert merged.x == pytest.approx(1.05, abs=1e-6)
    assert merged.y == pytest.approx(2.0, abs=1e-6)

    # Motion-gated observation (|omega| = 0.8 > 0.5 rad/s) is ignored
    gated = mgr.add_observation(
        56, "chair", 1.50, 2.0, 0.0, conf=0.99, stamp_sec=2.5, angular_vel_rad_s=0.8
    )
    assert gated is None
    assert merged.hits == 6

    # Register GAS_HAZARD and AUDIO_DISTRESS events
    gas_pm = mgr.register_event("GAS_HAZARD", x=-1.5, y=0.5, z=0.0, conf=1.0, stamp_sec=5.0)
    audio_pm = mgr.register_event("AUDIO_DISTRESS", x=0.2, y=-2.1, z=0.0, conf=0.9, stamp_sec=6.0)
    assert len(mgr.perm_markers) == 3
    assert gas_pm.class_name == "GAS_HAZARD"
    assert audio_pm.class_name == "AUDIO_DISTRESS"

    # Re-triggering GAS_HAZARD 0.2 m away merges into the existing GAS_HAZARD marker
    gas_pm2 = mgr.register_event("GAS_HAZARD", x=-1.3, y=0.5, z=0.0, conf=1.0, stamp_sec=6.5)
    assert len(mgr.perm_markers) == 3
    assert gas_pm2.marker_id == gas_pm.marker_id
    assert gas_pm2.hits == 2
    assert gas_pm2.x == pytest.approx(-1.4, abs=1e-6)

    # Verify JSON export dictionary
    export = mgr.to_dict()
    assert export["num_permanent_markers"] == 3
    json_str = json.dumps(export)
    assert "GAS_HAZARD" in json_str
    assert "AUDIO_DISTRESS" in json_str


# =============================================================================
# 5. Paper Algorithm 2: World <-> Pixel & 2D Map Overlay Rendering Tests
# =============================================================================

def test_world_to_pixel_and_pixel_to_world_roundtrip():
    """
    Verify Paper Algorithm 2 coordinate transforms:
      px = int((x - origin_x) / res)
      py = height - 1 - int((y - origin_y) / res)
    and round-trip error < 1 px.
    """
    origin_x, origin_y = -5.0, -4.0
    res = 0.05
    width, height = 200, 160

    test_points = [
        (-5.0, -4.0),
        (0.0, 0.0),
        (2.34, -1.18),
        (-2.75, 3.45),
        (4.92, 3.92),
    ]
    for wx, wy in test_points:
        px, py = world_to_pixel(wx, wy, origin_x, origin_y, res, height)
        expected_px = int((wx - origin_x) / res)
        expected_py = height - 1 - int((wy - origin_y) / res)
        assert px == expected_px
        assert py == expected_py

        wx_rec, wy_rec = pixel_to_world(px, py, origin_x, origin_y, res, height)
        px_rec, py_rec = world_to_pixel(wx_rec, wy_rec, origin_x, origin_y, res, height)
        assert abs(px_rec - px) < 1
        assert abs(py_rec - py) < 1
        assert abs(wx_rec - wx) / res < 1.0
        assert abs(wy_rec - wy) / res < 1.0


def test_render_semantic_map_overlay_and_save_bundle(tmp_path):
    """
    Verify render_semantic_map_overlay vertically flips bottom-up OccupancyGrid,
    draws colored marker circles & legend, and save_semantic_map_bundle writes all files.
    """
    width, height = 160, 120
    res = 0.05
    origin_x, origin_y = -4.0, -3.0

    # Create synthetic OccupancyGrid: free space (0), with an occupied wall (100) at grid row 0
    grid = np.zeros((height, width), dtype=np.int8)
    grid[0, :] = 100  # Bottom row in world coordinates (y = origin_y)

    mgr = SemanticMarkerManager(min_hits=1, min_span_sec=0.0)
    pm_person = mgr.add_observation(0, "person", 0.0, 0.0, 0.0, conf=0.91, stamp_sec=1.0)
    pm_gas = mgr.register_event("GAS_HAZARD", x=1.0, y=1.0, z=0.0, conf=1.0, stamp_sec=2.0)
    assert pm_person is not None
    assert pm_gas is not None

    overlay = render_semantic_map_overlay(
        occupancy_grid=grid.reshape(-1),
        width=width,
        height=height,
        resolution=res,
        origin_x=origin_x,
        origin_y=origin_y,
        markers=mgr.perm_markers,
    )
    assert isinstance(overlay, np.ndarray)
    assert overlay.shape == (height, width, 3)
    assert overlay.dtype == np.uint8

    # Grid row 0 (occupied=100 -> black 0) must end up at the bottom row (py = height - 1) after flip
    assert np.all(overlay[height - 1, 100] == 0)

    # Person marker at (0.0, 0.0) must be painted red (BGR = (0, 0, 255)) at (px, py)
    px_p, py_p = world_to_pixel(0.0, 0.0, origin_x, origin_y, res, height)
    bgr_at_person = overlay[py_p, px_p]
    assert int(bgr_at_person[2]) > 200 and int(bgr_at_person[0]) < 50

    # Save complete map bundle
    paths = save_semantic_map_bundle(
        output_dir=str(tmp_path),
        occupancy_grid=grid,
        width=width,
        height=height,
        resolution=res,
        origin_x=origin_x,
        origin_y=origin_y,
        marker_manager=mgr,
    )
    for key in ("raw_map_png", "raw_map_yaml", "semantic_objects_json", "semantic_map_overlay_png"):
        assert key in paths
        assert (tmp_path / paths[key]).exists()


def test_find_closest_timed_item():
    """Verify ring-buffer scan timestamp matching within +-0.15 s."""
    buf = [(10.00, "scan_0"), (10.20, "scan_1"), (10.40, "scan_2")]
    assert find_closest_timed_item(buf, 10.24, max_delta_sec=0.15) == "scan_1"
    assert find_closest_timed_item(buf, 10.80, max_delta_sec=0.15) is None


def test_semantic_map_node_integration_call_patterns():
    """Verify exact keyword-argument call patterns used by semantic_map_node.py."""
    intr = CameraIntrinsics.from_fov(640, 480, 1.085595)
    est = create_depth_estimator(
        depth_source="lidar_fusion",
        min_range_m=0.12,
        max_range_m=3.50,
        foreground_margin_m=0.25,
        enable_ground_fallback=True,
    )
    angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    ranges = np.full(360, 1.60, dtype=np.float64)
    res = est.estimate(
        bbox_xyxy=(240.0, 160.0, 400.0, 320.0),
        intrinsics=intr,
        scan_ranges=ranges,
        scan_angles=angles,
        T_cam_scan=default_t_cam_scan(),
        T_base_cam=default_t_base_cam(),
        depth_image=None,
    )
    assert res is not None
    assert res.method == "lidar_fusion"

    mgr = SemanticMarkerManager(min_hits=1, min_span_sec=0.0)
    obs = SemanticObservation(
        class_id=0,
        class_name="person",
        x=1.5,
        y=-0.5,
        z=0.0,
        conf=0.88,
        stamp_sec=1.0,
        low_confidence=False,
        method="lidar_fusion",
    )
    pm = mgr.add_observation(obs)
    assert pm is not None
    assert len(mgr.get_permanent_markers()) == 1

