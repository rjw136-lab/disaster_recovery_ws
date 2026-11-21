#!/usr/bin/env python3
"""
Unit & Integration Tests for Stage 1 (Phase 4-B) DynamicObstacleTracker:
1. Pure Python/NumPy dependency isolation (no rclpy / heavy libraries).
2. Zero false positives on static walls, fast rotation (|omega| > 0.45 rad/s), and corner reveals.
3. Close-range cylinder (d=0.45 m, r=0.21 m, >35 rays) detection without n_c<=35 truncation.
4. 0 deg / 359 deg wrap-around merge + 1-2 beam dropout tolerance + chord/radius filtering.
5. Surface arc centroid bias correction (Taubin circle fit + ray-direction fallback).
6. Damped-CV 4x4 covariance propagation, harmonic turnaround static_occ masking, and 2-tick coasting.
7. Full 5 Hz noisy LiDAR sinusoidal prediction gates (tau=0.5s & tau=1.0s FDE/ADE).
8. Ch3 (12x12) ego grid projection with max(r_eff - r_c, eps) denominator.
9. 5 Hz scan.header.stamp gating, non-mutating mid-tick KF extrapolation, and ICP SE(2) sync.
10. recovery_controller.py enable_dynamic_tracker (default False) parameter & wiring check.
11. CPU latency < 0.3 ms per tick.
"""

import math
import sys
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

from disaster_recovery.dynamic_tracker import (
    DynamicObstacleTracker,
    TrackedObstacle,
    correct_cluster_center_and_radius,
    damped_cv_matrices,
    max_eigenvalue_2x2,
    propagate_state_cov,
)


def _raycast_scene(
    rx: float,
    ry: float,
    rth: float,
    cylinders: Sequence[Tuple[float, float, float]] = (),
    boxes_wall_segments: Sequence[Tuple[float, float, float, float]] = (),
    n_rays: int = 360,
    max_range: float = 3.5,
    noise_std: float = 0.0,
    dropout_prob: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Exact 2D ray-scene intersection generator (walls + moving cylinders)."""
    ranges = np.full(n_rays, np.inf, dtype=np.float32)
    angs = np.linspace(0.0, 2.0 * math.pi, n_rays, endpoint=False, dtype=np.float64) + float(rth)
    dx_r = np.cos(angs)
    dy_r = np.sin(angs)

    for i in range(n_rays):
        dx = float(dx_r[i])
        dy = float(dy_r[i])
        best_t = max_range + 1.0

        # Ray-circle intersection
        for cx, cy, cr in cylinders:
            ox = cx - rx
            oy = cy - ry
            b = ox * dx + oy * dy
            c = ox * ox + oy * oy - cr * cr
            disc = b * b - c
            if disc >= 0.0:
                t = b - math.sqrt(disc)
                if 0.12 <= t < best_t:
                    best_t = t

        # Ray-segment intersection
        for x1, y1, x2, y2 in boxes_wall_segments:
            sx = x2 - x1
            sy = y2 - y1
            denom = dx * sy - dy * sx
            if abs(denom) < 1e-9:
                continue
            qpx = x1 - rx
            qpy = y1 - ry
            t = (qpx * sy - qpy * sx) / denom
            u = (qpx * dy - qpy * dx) / denom
            if 0.12 <= t < best_t and 0.0 <= u <= 1.0:
                best_t = t

        if best_t <= max_range:
            val = best_t
            if noise_std > 0.0 and rng is not None:
                val += float(rng.normal(0.0, noise_std))
            ranges[i] = np.float32(max(0.12, val))

    if dropout_prob > 0.0 and rng is not None:
        drop = rng.random(n_rays) < dropout_prob
        ranges[drop] = np.inf
    return ranges


def test_dynamic_tracker_pure_numpy_no_heavy_imports():
    """dynamic_tracker must stay pure Python + NumPy without rclpy, scipy, cv2, or torch."""
    import disaster_recovery.dynamic_tracker as dt_mod

    src_deps = ("rclpy", "scipy", "cv2", "torch")
    for dep in src_deps:
        assert dep not in dt_mod.__dict__, f"Forbidden heavy dependency {dep} bound in dynamic_tracker"


def test_static_walls_cornering_and_rotation_guard_zero_false_positives():
    """Rotating with |omega| > 0.45 rad/s and rounding a corner must yield 0 confirmed dynamic FPs."""
    rng = np.random.default_rng(101)
    # L-shaped corridor with corner blocking view of a side room initially
    walls = [
        (-1.0, -0.8, 3.0, -0.8),
        (-1.0, 0.8, 1.0, 0.8),
        (1.0, 0.8, 1.0, 2.5),
        (2.4, 0.8, 2.4, 2.5),
        (2.4, 0.8, 3.0, 0.8),
        (3.0, -0.8, 3.0, 0.8),
        (1.0, 2.5, 2.4, 2.5),
        (-1.0, -0.8, -1.0, 0.8),
    ]
    tracker = DynamicObstacleTracker()
    confirmed_fp_scans = 0
    tentative_fp_scans = 0

    for k in range(30):
        t = k * 0.20
        # Move forward up to (1.5, 0.0) while rotating fast (omega = 0.60 rad/s) to uncover (1.0..2.4, 0.8..2.5)
        if k < 10:
            rx, ry, rth, omega = 0.15 * k, 0.0, 0.0, 0.0
        else:
            rx, ry = 1.5, 0.0
            rth = 0.60 * ((k - 10) * 0.20)
            omega = 0.60
        ranges = _raycast_scene(rx, ry, rth, boxes_wall_segments=walls, noise_std=0.01, rng=rng)
        tracks = tracker.update(rx=rx, ry=ry, rth=rth, raw_ranges=ranges, scan_stamp_sec=t, omega=omega)
        if any(tr.is_confirmed for tr in tracks):
            confirmed_fp_scans += 1
        if len(tracks) > 0:
            tentative_fp_scans += 1

    assert confirmed_fp_scans == 0
    assert tentative_fp_scans <= 2  # Single-step grazing corner noise never promotes to confirmed


def test_close_range_cylinder_not_dropped_without_nc_cap():
    """Close cylinder (d=0.45 m, r=0.21 m, >35 rays) must be detected and accurately localized."""
    walls = [(-2.0, -2.0, 3.2, -2.0), (3.2, -2.0, 3.2, 2.0), (3.2, 2.0, -2.0, 2.0), (-2.0, 2.0, -2.0, -2.0)]
    tracker = DynamicObstacleTracker()

    # Empty background scans to establish observed_free in the room
    for k in range(3):
        ranges = _raycast_scene(0.0, 0.0, 0.0, boxes_wall_segments=walls)
        tracker.update(0.0, 0.0, 0.0, ranges, scan_stamp_sec=k * 0.20)

    # Moving cylinder passing close at d = 0.45 m ahead-ish (subtends ~55 deg = ~55 rays > 35)
    # Put two ticks at y = -0.05 -> +0.05 at x = 0.45 m (speed 0.50 m/s)
    positions = [(0.45, -0.08), (0.45, 0.00), (0.45, 0.08)]
    tracks: List[TrackedObstacle] = []
    for i, (cx, cy) in enumerate(positions):
        ranges = _raycast_scene(0.0, 0.0, 0.0, cylinders=[(cx, cy, 0.21)], boxes_wall_segments=walls)
        # Sanity check that cylinder occupies > 35 rays at d=0.45m
        n_close = int(np.sum((ranges < 0.70) & np.isfinite(ranges)))
        assert n_close > 35
        tracks = tracker.update(0.0, 0.0, 0.0, ranges, scan_stamp_sec=(3 + i) * 0.20)

    confirmed = [tr for tr in tracks if tr.is_confirmed]
    assert len(confirmed) == 1
    tr = confirmed[0]
    assert math.hypot(tr.x - 0.45, tr.y - 0.08) < 0.06
    assert 0.16 <= tr.radius <= 0.26
    assert tr.vy > 0.20


def test_wraparound_merge_beam_dropout_and_chord_radius_filters():
    """
    Verify:
      1) Cylinder straddling 0 deg / 359 deg seam with 1-2 dropped rays merges into 1 track.
      2) Long straight moving partition (L_chord > 0.70 m) is rejected from dynamic tracks.
      3) Tiny <3-ray spike cluster is rejected.
    """
    walls = [(-2.5, -2.5, 3.2, -2.5), (3.2, -2.5, 3.2, 2.5), (3.2, 2.5, -2.5, 2.5), (-2.5, 2.5, -2.5, -2.5)]
    tracker = DynamicObstacleTracker()
    for k in range(3):
        tracker.update(0.0, 0.0, 0.0, _raycast_scene(0.0, 0.0, 0.0, boxes_wall_segments=walls), scan_stamp_sec=k * 0.20)

    # Cylinder straddling 0/359 deg (moving along +x at y=0.0) with forced 2-ray dropout right at rays 0 & 1
    for i, cx in enumerate([1.60, 1.50, 1.40]):
        ranges = _raycast_scene(0.0, 0.0, 0.0, cylinders=[(cx, 0.0, 0.21)], boxes_wall_segments=walls)
        ranges[0] = np.inf
        ranges[1] = np.inf
        # Inject an isolated 2-ray clutter spike elsewhere
        ranges[90] = 1.0
        ranges[91] = 1.0
        tracks = tracker.update(0.0, 0.0, 0.0, ranges, scan_stamp_sec=(3 + i) * 0.20)

    confirmed = [tr for tr in tracks if tr.is_confirmed]
    assert len(tracks) == 1 and len(confirmed) == 1
    assert abs(confirmed[0].x - 1.40) < 0.06
    assert abs(confirmed[0].y) < 0.05

    # Long straight partition (1.3 m width -> L_chord > 0.70 m) must NOT create a track
    tracker_part = DynamicObstacleTracker()
    for k in range(3):
        tracker_part.update(0.0, 0.0, 0.0, _raycast_scene(0.0, 0.0, 0.0, boxes_wall_segments=walls), scan_stamp_sec=k * 0.20)
    for i, px in enumerate([1.60, 1.50, 1.40]):
        seg = [(px, -0.65, px, 0.65)]
        ranges = _raycast_scene(0.0, 0.0, 0.0, boxes_wall_segments=walls + seg)
        p_tracks = tracker_part.update(0.0, 0.0, 0.0, ranges, scan_stamp_sec=(3 + i) * 0.20)
        assert len(p_tracks) == 0


def test_surface_arc_bias_correction_and_ray_fallback():
    """Verify Taubin circle fit removes centroid bias and fallback shifts along outward ray."""
    # Partial narrow collinear-ish 3-point face (forces ill-conditioned circle fit fallback)
    pts_flat = np.array([[1.50, -0.15], [1.50, 0.00], [1.50, 0.15]], dtype=np.float64)
    est = correct_cluster_center_and_radius(pts_flat, robot_xy=(0.0, 0.0), kappa_arc=0.65)
    assert est is not None
    cx, cy, r_hat = est
    assert abs(r_hat - 0.15) < 1e-6
    assert cx > 1.50 + 0.09  # Shifted outward along +x away from robot
    assert abs(cy) < 1e-6


def test_damped_cv_exact_propagation_and_harmonic_turnaround_protection():
    """
    Verify:
      1) Continuous damping gamma_v(dt) = exp(-0.35 * dt) & 4x4 P(tau) = F P F^T + Q.
      2) Confirmed obstacle pausing at harmonic turnaround is NOT absorbed into static_occ.
      3) Up to 2 missed ticks keep confirmed track alive (coasting); 3rd miss removes it.
    """
    s0 = np.array([1.0, 0.0, 0.40, 0.0], dtype=np.float64)
    P0 = np.diag([0.01, 0.01, 0.05, 0.05])
    F, Q = damped_cv_matrices(0.50, beta=0.35, q_acc=0.30)
    assert abs(F[2, 2] - math.exp(-0.35 * 0.50)) < 1e-9
    s1, P1 = propagate_state_cov(s0, P0, tau=0.50, beta=0.35, q_acc=0.30)
    assert np.allclose(P1, F @ P0 @ F.T + Q, atol=1e-12)
    assert np.allclose(P1, P1.T, atol=1e-12)

    walls = [(-2.0, -2.0, 3.0, -2.0), (3.0, -2.0, 3.0, 2.0), (3.0, 2.0, -2.0, 2.0), (-2.0, 2.0, -2.0, -2.0)]
    tracker = DynamicObstacleTracker()
    for k in range(3):
        tracker.update(0.0, 0.0, 0.0, _raycast_scene(0.0, 0.0, 0.0, boxes_wall_segments=walls), scan_stamp_sec=k * 0.20)

    # Move into corridor -> confirm -> stop at turnaround for 8 ticks (1.6 s)
    ys = [-0.20, -0.10, 0.00] + [0.02] * 8
    for i, cy in enumerate(ys):
        ranges = _raycast_scene(0.0, 0.0, 0.0, cylinders=[(1.20, cy, 0.21)], boxes_wall_segments=walls)
        tracks = tracker.update(0.0, 0.0, 0.0, ranges, scan_stamp_sec=(3 + i) * 0.20)

    confirmed = [tr for tr in tracks if tr.is_confirmed]
    assert len(confirmed) == 1
    ix_c = int(math.floor(1.20 / tracker.res)) - tracker.origin_ix
    iy_c = int(math.floor(0.02 / tracker.res)) - tracker.origin_iy
    assert tracker.static_occ[ix_c, iy_c] < tracker.static_occ_thresh

    # Coasting test: occlude dynamic obstacle for 2 ticks -> kept; 3rd miss -> deleted
    empty_ranges = _raycast_scene(0.0, 0.0, 0.0, boxes_wall_segments=walls)
    t_miss1 = tracker.update(0.0, 0.0, 0.0, empty_ranges, scan_stamp_sec=(3 + len(ys)) * 0.20)
    assert any(tr.is_confirmed and tr.missed_ticks == 1 for tr in t_miss1)
    t_miss2 = tracker.update(0.0, 0.0, 0.0, empty_ranges, scan_stamp_sec=(4 + len(ys)) * 0.20)
    assert any(tr.is_confirmed and tr.missed_ticks == 2 for tr in t_miss2)
    t_miss3 = tracker.update(0.0, 0.0, 0.0, empty_ranges, scan_stamp_sec=(5 + len(ys)) * 0.20)
    assert len(t_miss3) == 0


def test_sinusoidal_crossfire_prediction_fde_ade_gates():
    """
    Full 5 Hz noisy 360-ray LiDAR simulation of dynamic_crossfire_alley harmonic cylinder
    (V_max=0.36 m/s, omega=0.95 rad/s, r=0.21 m, sigma_r=0.01 m, 6% dropouts).
    Must pass Stage 1 gates:
      tau = 0.5 s: mean FDE <= 0.12 m, mean ADE <= 0.09 m
      tau = 1.0 s: mean FDE <= 0.24 m, mean ADE <= 0.16 m
    """
    rng = np.random.default_rng(2026)
    walls = [(-1.5, -1.8, 3.0, -1.8), (3.0, -1.8, 3.0, 1.8), (3.0, 1.8, -1.5, 1.8), (-1.5, 1.8, -1.5, -1.8)]
    tracker = DynamicObstacleTracker()

    # Background warmup (2 scans)
    for k in range(2):
        r0 = _raycast_scene(0.0, 0.0, 0.0, boxes_wall_segments=walls, noise_std=0.01, rng=rng)
        tracker.update(0.0, 0.0, 0.0, r0, scan_stamp_sec=k * 0.20)

    v_max = 0.36
    w_osc = 0.95
    amp = v_max / w_osc
    cx_fixed = 1.45

    def gt_xy(t_obs: float) -> Tuple[float, float]:
        return cx_fixed, amp * math.sin(w_osc * t_obs)

    fde_05, ade_05, fde_10, ade_10 = [], [], [], []
    dt = 0.20
    for k in range(60):
        t = (2 + k) * dt
        t_motion = k * dt
        cx, cy = gt_xy(t_motion)
        ranges = _raycast_scene(
            0.0,
            0.0,
            0.0,
            cylinders=[(cx, cy, 0.21)],
            boxes_wall_segments=walls,
            noise_std=0.01,
            dropout_prob=0.06,
            rng=rng,
        )
        tracks = tracker.update(0.0, 0.0, 0.0, ranges, scan_stamp_sec=t)
        confirmed = [tr for tr in tracks if tr.is_confirmed]
        if confirmed:
            tr = min(confirmed, key=lambda c: math.hypot(c.x - cx, c.y - cy))
            px05, py05, _, _ = tracker.predict_future(tr, 0.5)
            px10, py10, _, _ = tracker.predict_future(tr, 1.0)
            gx05, gy05 = gt_xy(t_motion + 0.5)
            gx10, gy10 = gt_xy(t_motion + 1.0)
            fde_05.append(math.hypot(px05 - gx05, py05 - gy05))
            fde_10.append(math.hypot(px10 - gx10, py10 - gy10))

            e05 = []
            for tau in (0.1, 0.2, 0.3, 0.4, 0.5):
                px_t, py_t, _, _ = tracker.predict_future(tr, tau)
                gx_t, gy_t = gt_xy(t_motion + tau)
                e05.append(math.hypot(px_t - gx_t, py_t - gy_t))
            e10 = []
            for tau in (0.2, 0.4, 0.6, 0.8, 1.0):
                px_t, py_t, _, _ = tracker.predict_future(tr, tau)
                gx_t, gy_t = gt_xy(t_motion + tau)
                e10.append(math.hypot(px_t - gx_t, py_t - gy_t))
            ade_05.append(float(np.mean(e05)))
            ade_10.append(float(np.mean(e10)))

    assert len(fde_05) >= 50
    mean_fde05 = float(np.mean(fde_05))
    mean_ade05 = float(np.mean(ade_05))
    mean_fde10 = float(np.mean(fde_10))
    mean_ade10 = float(np.mean(ade_10))

    assert mean_fde05 <= 0.12, f"tau=0.5s FDE {mean_fde05:.4f} > 0.12 m"
    assert mean_ade05 <= 0.09, f"tau=0.5s ADE {mean_ade05:.4f} > 0.09 m"
    assert mean_fde10 <= 0.24, f"tau=1.0s FDE {mean_fde10:.4f} > 0.24 m"
    assert mean_ade10 <= 0.16, f"tau=1.0s ADE {mean_ade10:.4f} > 0.16 m"


def test_ch3_projection_denominator_fix():
    """Verify Ch3 12x12 projection matches exact max(r_eff - r_c, eps) formula without double counting r_c."""
    tracker = DynamicObstacleTracker(q_acc_base=0.0)
    # Zero covariance & zero velocity so all horizons stay centered at (0.375, 0.075) (near ego grid cell)
    tracker.tracks[1] = TrackedObstacle(
        track_id=1,
        x=0.375,
        y=0.075,
        vx=0.0,
        vy=0.0,
        radius=0.20,
        P_4x4=np.diag([0.04, 0.04, 0.0, 0.0]).astype(np.float64),  # std=0.20 => r_eff = 0.20 + 1.5*0.20 = 0.50
        last_update_stamp=1.0,
        age=3,
        hits=3,
        is_confirmed=True,
    )
    ch3 = tracker.project_ch3(rx=0.0, ry=0.0, rth=0.0, now_sec=1.0, horizons=(0.5,), weights=(1.0,))
    assert ch3.shape == (12, 12) and ch3.dtype == np.float32

    # Cell center (0.375, 0.075) is grid index (8, 6): d = 0 <= r_c => ch3 == 1.0
    assert abs(float(ch3[8, 6]) - 1.0) < 1e-5
    # Cell center at (-0.075, 0.075) is index (5, 6): d = 0.45 m, r_c = 0.20, r_eff = 0.50
    # Expected = 1 - (0.45 - 0.20) / (0.50 - 0.20) = 1 - 0.25 / 0.30 = 1/6
    assert abs(float(ch3[5, 6]) - (1.0 / 6.0)) < 1e-4

    # Zero-covariance edge case checks eps guard without NaN
    tracker.tracks[1].P_4x4 = np.zeros((4, 4), dtype=np.float64)
    ch3_zero = tracker.project_ch3(rx=0.0, ry=0.0, rth=0.0, now_sec=1.0, horizons=(0.5,), weights=(0.75,))
    assert np.all(np.isfinite(ch3_zero))
    assert abs(float(ch3_zero[8, 6]) - 0.75) < 1e-5


def test_5hz_stamp_gating_midtick_extrapolation_and_icp_se2_sync():
    """
    Verify:
      1) Unchanged scan_stamp skips update & performs non-mutating damped-KF extrapolation.
      2) sync_icp_correction shifts/rotates tracks seamlessly across waypoint ICP jumps.
    """
    walls = [(-2.0, -2.0, 3.0, -2.0), (3.0, -2.0, 3.0, 2.0), (3.0, 2.0, -2.0, 2.0), (-2.0, 2.0, -2.0, -2.0)]
    tracker = DynamicObstacleTracker()
    for k in range(2):
        tracker.update(0.0, 0.0, 0.0, _raycast_scene(0.0, 0.0, 0.0, boxes_wall_segments=walls), scan_stamp_sec=k * 0.20)

    # Scan updates at t=0.40, 0.60, 0.80
    for i, cy in enumerate([-0.16, -0.08, 0.00]):
        r = _raycast_scene(0.0, 0.0, 0.0, cylinders=[(1.20, cy, 0.21)], boxes_wall_segments=walls)
        tracker.update(0.0, 0.0, 0.0, r, scan_stamp_sec=0.40 + i * 0.20, now_sec=0.40 + i * 0.20)

    assert tracker.scan_update_count == 5
    anchor_y = tracker.tracks[1].y
    anchor_vy = tracker.tracks[1].vy
    assert anchor_vy > 0.25

    # Mid-tick at t=0.90 (same scan_stamp=0.80!) must NOT trigger scan update or mutate anchor track
    extrap_tracks = tracker.update(0.0, 0.0, 0.0, r, scan_stamp_sec=0.80, now_sec=0.90)
    assert tracker.last_did_update is False
    assert tracker.scan_update_count == 5
    assert tracker.extrap_tick_count == 1
    assert abs(tracker.tracks[1].y - anchor_y) < 1e-9  # Anchor untouched
    assert extrap_tracks[0].y > anchor_y + 0.02       # Extrapolated forward in +y

    # Apply ICP correction jump delta_T: shift (+0.10, -0.05) and rotate +0.08 rad
    old_corr = (0.0, 0.0, 0.0)
    new_corr = (0.10, -0.05, 0.08)
    px_before, py_before = tracker.tracks[1].x, tracker.tracks[1].y
    vx_before, vy_before = tracker.tracks[1].vx, tracker.tracks[1].vy
    tracker.sync_icp_correction(old_corr, new_corr)

    c_d, s_d = math.cos(0.08), math.sin(0.08)
    assert abs(tracker.tracks[1].x - (0.10 + c_d * px_before - s_d * py_before)) < 1e-9
    assert abs(tracker.tracks[1].y - (-0.05 + s_d * px_before + c_d * py_before)) < 1e-9
    assert abs(tracker.tracks[1].vx - (c_d * vx_before - s_d * vy_before)) < 1e-9
    assert abs(tracker.tracks[1].vy - (s_d * vx_before + c_d * vy_before)) < 1e-9


def test_recovery_controller_enable_dynamic_tracker_param_and_wiring():
    """Verify recovery_controller.py declares enable_dynamic_tracker (default False) and wires Stage 1."""
    try:
        import rclpy
        from std_msgs.msg import Bool
        from sensor_msgs.msg import LaserScan
        from disaster_recovery.recovery_controller import RecoveryController
    except ImportError:
        import pytest
        pytest.skip("rclpy unavailable in host Python environment")

    rclpy.init()
    try:
        node_default = RecoveryController()
        assert node_default.enable_dynamic_tracker is False
        assert node_default.dynamic_tracker is None
        node_default.destroy_node()
    finally:
        if rclpy.ok():
            rclpy.shutdown()

    # ROS2 CLI-style parameter override via rclpy.init
    rclpy.init(args=["--ros-args", "-p", "enable_dynamic_tracker:=true", "-p", "recovery_strategy:=proposed"])
    try:
        node = RecoveryController()
        assert node.enable_dynamic_tracker is True
        assert node.dynamic_tracker is not None

        node.odom_initialized = True
        node.current_x = 0.0
        node.current_y = 0.0
        node.current_theta = 0.0

        scan = LaserScan()
        scan.header.stamp.sec = 10
        scan.header.stamp.nanosec = 0
        scan.angle_min = 0.0
        scan.angle_increment = 2.0 * math.pi / 360.0
        scan.ranges = [2.0] * 360
        node.cb_scan(scan)
        assert abs(node.latest_scan_stamp_sec - 10.0) < 1e-6

        node._update_dynamic_tracker(10.0)
        assert node.dynamic_tracker.scan_update_count == 1
        assert node.latest_dynamic_ch3.shape == (12, 12)

        # Mid-tick unchanged stamp skips scan update and uses KF extrapolation
        node._update_dynamic_tracker(10.1)
        assert node.dynamic_tracker.scan_update_count == 1
        assert node.dynamic_tracker.extrap_tick_count == 1

        # Episode reset clears tracker cleanly
        node.cb_reset_episode(Bool(data=True))
        assert node.dynamic_tracker.scan_update_count == 0
        assert node.latest_dynamic_tracks == []
        node.destroy_node()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


def test_cpu_latency_under_0_3_ms():
    """Verify per-tick CPU latency < 0.3 ms (both mixed 5Hz/10Hz loop and 5 Hz scan updates)."""
    rng = np.random.default_rng(77)
    walls = [(-2.0, -2.0, 3.0, -2.0), (3.0, -2.0, 3.0, 2.0), (3.0, 2.0, -2.0, 2.0), (-2.0, 2.0, -2.0, -2.0)]
    tracker = DynamicObstacleTracker()

    # Pre-generate 20 scans with 2 moving cylinders
    scans = []
    for k in range(20):
        cy1 = 0.3 * math.sin(0.95 * k * 0.20)
        cy2 = -0.3 * math.sin(0.95 * k * 0.20)
        r = _raycast_scene(
            0.0, 0.0, 0.0,
            cylinders=[(1.20, cy1, 0.21), (2.10, cy2, 0.21)],
            boxes_wall_segments=walls,
            noise_std=0.01,
            rng=rng,
        )
        scans.append(r)

    # Warm-up
    for k in range(5):
        tracker.update(0.0, 0.0, 0.0, scans[k], scan_stamp_sec=k * 0.20)
        tracker.project_ch3(0.0, 0.0, 0.0, now_sec=k * 0.20)

    # Benchmark 200 alternating 10 Hz ticks (100 scan updates at 5 Hz + 100 KF extrapolations + Ch3)
    t0 = time.perf_counter()
    for tick in range(200):
        scan_idx = (tick // 2) % 20
        scan_stamp = 10.0 + (tick // 2) * 0.20
        now_t = 10.0 + tick * 0.10
        tracker.update(0.0, 0.0, 0.0, scans[scan_idx], scan_stamp_sec=scan_stamp, now_sec=now_t)
        tracker.project_ch3(0.0, 0.0, 0.0, now_sec=now_t)
    mean_ms = ((time.perf_counter() - t0) / 200.0) * 1000.0
    assert mean_ms < 0.30, f"Mean tracker+Ch3 latency {mean_ms:.4f} ms >= 0.30 ms"
