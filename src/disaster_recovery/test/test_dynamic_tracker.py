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

    # Moving cylinder passing close at d = 0.45 m ahead-ish (subtends > 35 deg = > 35 rays)
    # 5 consecutive moving ticks at speed 0.40 m/s ending at (0.45, 0.08)
    positions = [(0.45, -0.24), (0.45, -0.16), (0.45, -0.08), (0.45, 0.00), (0.45, 0.08)]
    tracks: List[TrackedObstacle] = []
    for i, (cx, cy) in enumerate(positions):
        ranges = _raycast_scene(0.0, 0.0, 0.0, cylinders=[(cx, cy, 0.21)], boxes_wall_segments=walls)
        # Sanity check that cylinder occupies > 35 rays at close range
        n_close = int(np.sum((ranges < 0.80) & np.isfinite(ranges)))
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
    for i, cx in enumerate([1.80, 1.70, 1.60, 1.50, 1.40]):
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

    # Move into corridor -> confirm at hit 5 -> stop at turnaround for 8 ticks (1.6 s)
    ys = [-0.40, -0.30, -0.20, -0.10, 0.00] + [0.02] * 8
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
        P_4x4=np.diag([0.04, 0.04, 0.0, 0.0]).astype(np.float64),  # std=0.20 => 1.5*0.20=0.30 capped to CH3_MAX_R_INFL=0.28
        last_update_stamp=1.0,
        age=3,
        hits=3,
        is_confirmed=True,
    )
    ch3 = tracker.project_ch3(rx=0.0, ry=0.0, rth=0.0, now_sec=1.0, horizons=(0.5,), weights=(1.0,))
    assert ch3.shape == (12, 12) and ch3.dtype == np.float32

    # Cell center (0.375, 0.075) is grid index (8, 6): d = 0 <= r_c => ch3 == 1.0
    assert abs(float(ch3[8, 6]) - 1.0) < 1e-5
    # Cell center at (-0.075, 0.075) is index (5, 6): d = 0.45 m, r_c = 0.20, r_infl = min(0.30, 0.28) = 0.28
    # Expected = 1 - (0.45 - 0.20) / 0.28 = 3/28
    assert abs(float(ch3[5, 6]) - (3.0 / 28.0)) < 1e-4

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


def test_imm_harmonic_crossfire_and_gauntlet_5hz_prediction():
    """
    Verify Step B 2-model (CV + Harmonic) IMM tracker at 5 Hz noisy 360-ray LiDAR:
      1) crossfire-type (A=0.38 m, omega=0.95 rad/s): PCA axis, omega/A, sweep endpoints,
         and predict_trajectory(tau in [0.0, 4.5] s) ADE/FDE accuracy.
      2) gauntlet-type (A=0.93 m, omega=0.45 rad/s): PCA axis, omega/A, sweep endpoints,
         and predict_trajectory(tau in [0.0, 4.5] s) ADE/FDE accuracy.
      3) Pure constant-velocity motion favors CV weight (mu_cv > mu_harm).
      4) Per-tick update + 30-step IMM predict_trajectory stays < 0.50 ms.
    """
    walls = [
        (-2.0, -2.2, 4.2, -2.2),
        (4.2, -2.2, 4.2, 2.2),
        (4.2, 2.2, -2.0, 2.2),
        (-2.0, 2.2, -2.0, -2.2),
    ]
    taus = np.arange(0, 31, dtype=np.float64) * 0.15  # 0.0 .. 4.5 s

    def _run_harmonic_scenario(
        seed: int,
        amp_gt: float,
        omega_gt: float,
        axis_gt: Tuple[float, float],
        center_gt: Tuple[float, float],
    ) -> dict:
        rng = np.random.default_rng(seed)
        tracker = DynamicObstacleTracker()
        for k in range(3):
            r0 = _raycast_scene(0.0, 0.0, 0.0, boxes_wall_segments=walls, noise_std=0.01, rng=rng)
            tracker.update(0.0, 0.0, 0.0, r0, scan_stamp_sec=k * 0.20)

        ades, fdes_45, mu_hs, omegas, amps, axis_dots, sweep_errs = [], [], [], [], [], [], []
        midtick_errs = []
        pregen_scans: List[np.ndarray] = []
        for k in range(75):
            tm = k * 0.20
            cx = center_gt[0] + axis_gt[0] * amp_gt * math.sin(omega_gt * tm)
            cy = center_gt[1] + axis_gt[1] * amp_gt * math.sin(omega_gt * tm)
            pregen_scans.append(
                _raycast_scene(
                    0.0,
                    0.0,
                    0.0,
                    cylinders=[(cx, cy, 0.21)],
                    boxes_wall_segments=walls,
                    noise_std=0.01,
                    dropout_prob=0.06,
                    rng=rng,
                )
            )

        for k in range(75):
            t = (3 + k) * 0.20
            tm = k * 0.20
            ranges = pregen_scans[k]
            tracks = tracker.update(0.0, 0.0, 0.0, ranges, scan_stamp_sec=t, now_sec=t)
            conf = [tr for tr in tracks if tr.is_confirmed]
            if conf:
                tr = conf[0]
                pred = tracker.predict_trajectory(tr, taus)
                if tm >= 5.0:
                    gt_x = center_gt[0] + axis_gt[0] * amp_gt * np.sin(omega_gt * (tm + taus))
                    gt_y = center_gt[1] + axis_gt[1] * amp_gt * np.sin(omega_gt * (tm + taus))
                    errs = np.hypot(pred.centers[:, 0] - gt_x, pred.centers[:, 1] - gt_y)
                    ades.append(float(errs.mean()))
                    fdes_45.append(float(errs[-1]))
                    mu_hs.append(pred.mu_harm)
                    omegas.append(pred.omega)
                    amps.append(pred.amplitude)
                    axis_dots.append(abs(pred.u_hat[0] * axis_gt[0] + pred.u_hat[1] * axis_gt[1]))
                    # Check sweep endpoints against true [center - A*axis, center + A*axis]
                    gt_s0 = (center_gt[0] - amp_gt * axis_gt[0], center_gt[1] - amp_gt * axis_gt[1])
                    gt_s1 = (center_gt[0] + amp_gt * axis_gt[0], center_gt[1] + amp_gt * axis_gt[1])
                    d_direct = 0.5 * (
                        math.hypot(pred.sweep_start[0] - gt_s0[0], pred.sweep_start[1] - gt_s0[1])
                        + math.hypot(pred.sweep_end[0] - gt_s1[0], pred.sweep_end[1] - gt_s1[1])
                    )
                    d_swap = 0.5 * (
                        math.hypot(pred.sweep_start[0] - gt_s1[0], pred.sweep_start[1] - gt_s1[1])
                        + math.hypot(pred.sweep_end[0] - gt_s0[0], pred.sweep_end[1] - gt_s0[1])
                    )
                    sweep_errs.append(min(d_direct, d_swap))

                    # Verify 10 Hz mid-tick extrapolation preserves IMM phase alignment
                    mid_tracks = tracker.update(0.0, 0.0, 0.0, ranges, scan_stamp_sec=t, now_sec=t + 0.10)
                    mid_conf = [mt for mt in mid_tracks if mt.is_confirmed]
                    assert len(mid_conf) == 1
                    pred_mid = tracker.predict_trajectory(mid_conf[0], taus)
                    gt_mx = center_gt[0] + axis_gt[0] * amp_gt * np.sin(omega_gt * (tm + 0.10 + taus))
                    gt_my = center_gt[1] + axis_gt[1] * amp_gt * np.sin(omega_gt * (tm + 0.10 + taus))
                    midtick_errs.append(
                        float(np.hypot(pred_mid.centers[:, 0] - gt_mx, pred_mid.centers[:, 1] - gt_my).mean())
                    )

        # Steady-state 10 Hz tick timing (alternating 5 Hz scan updates + 10 Hz mid-ticks + predict_trajectory)
        t_bench0 = time.perf_counter()
        n_bench = 120
        for tick in range(n_bench):
            s_idx = (tick // 2) % len(pregen_scans)
            s_stamp = 100.0 + (tick // 2) * 0.20
            now_t = 100.0 + tick * 0.10
            trks_b = tracker.update(0.0, 0.0, 0.0, pregen_scans[s_idx], scan_stamp_sec=s_stamp, now_sec=now_t)
            for tb in trks_b:
                if tb.is_confirmed:
                    tracker.predict_trajectory(tb, taus)
        mean_ms = ((time.perf_counter() - t_bench0) / float(n_bench)) * 1000.0

        return {
            "n_eval": len(ades),
            "ade": float(np.mean(ades)),
            "fde45": float(np.mean(fdes_45)),
            "mid_ade": float(np.mean(midtick_errs)),
            "mu_harm": float(np.mean(mu_hs)),
            "omega": float(np.mean(omegas)),
            "amp": float(np.mean(amps)),
            "axis_dot": float(np.mean(axis_dots)),
            "sweep_err": float(np.mean(sweep_errs)),
            "mean_ms": mean_ms,
        }

    # 1) Crossfire: A=0.38 m, omega=0.95 rad/s (lateral y-axis)
    cf = _run_harmonic_scenario(2026, 0.38, 0.95, (0.0, 1.0), (1.45, 0.0))
    assert cf["n_eval"] >= 45
    assert cf["axis_dot"] > 0.98
    assert cf["mu_harm"] > 0.90
    assert abs(cf["omega"] - 0.95) < 0.05
    assert abs(cf["amp"] - 0.38) < 0.04
    assert cf["sweep_err"] < 0.06
    assert cf["ade"] < 0.05, f"crossfire 0-4.5s ADE {cf['ade']:.4f} m >= 0.05 m"
    assert cf["fde45"] < 0.08, f"crossfire 4.5s FDE {cf['fde45']:.4f} m >= 0.08 m"
    assert cf["mid_ade"] < 0.05
    assert cf["mean_ms"] < 0.50, f"crossfire IMM tick time {cf['mean_ms']:.3f} ms >= 0.50 ms"

    # 2) Gauntlet: A=0.93 m, omega=0.45 rad/s (axial x-axis)
    gn = _run_harmonic_scenario(2027, 0.93, 0.45, (1.0, 0.0), (1.90, 0.0))
    assert gn["n_eval"] >= 45
    assert gn["axis_dot"] > 0.98
    assert gn["mu_harm"] > 0.85
    assert abs(gn["omega"] - 0.45) < 0.07
    assert abs(gn["amp"] - 0.93) < 0.12
    assert gn["sweep_err"] < 0.14
    assert gn["ade"] < 0.10, f"gauntlet 0-4.5s ADE {gn['ade']:.4f} m >= 0.10 m"
    assert gn["fde45"] < 0.20, f"gauntlet 4.5s FDE {gn['fde45']:.4f} m >= 0.20 m"
    assert gn["mid_ade"] < 0.10
    assert gn["mean_ms"] < 0.50, f"gauntlet IMM tick time {gn['mean_ms']:.3f} ms >= 0.50 ms"

    # 3) Pure constant-velocity obstacle must keep CV model dominant (mu_cv > mu_harm)
    rng_cv = np.random.default_rng(2028)
    trk_cv = DynamicObstacleTracker()
    for k in range(3):
        trk_cv.update(0.0, 0.0, 0.0, _raycast_scene(0.0, 0.0, 0.0, boxes_wall_segments=walls), scan_stamp_sec=k * 0.20)
    last_pred_cv = None
    for k in range(18):
        t = (3 + k) * 0.20
        cy = -1.6 + 0.18 * (k * 0.20)
        r = _raycast_scene(0.0, 0.0, 0.0, cylinders=[(1.50, cy, 0.21)], boxes_wall_segments=walls, noise_std=0.008, rng=rng_cv)
        tracks = trk_cv.update(0.0, 0.0, 0.0, r, scan_stamp_sec=t)
        conf = [tr for tr in tracks if tr.is_confirmed]
        if conf:
            last_pred_cv = trk_cv.predict_trajectory(conf[0], taus)
    assert last_pred_cv is not None
    assert last_pred_cv.mu_cv > last_pred_cv.mu_harm


def test_smoke_spike_ratio_and_promotion_guards_block_false_positives():
    """
    Verify Step B smoke false-positive promotion guards:
      1) spike_ratio >= 0.03 blocks track promotion completely (0 confirmed FPs).
      2) hits < 5 stays tentative; 5th consecutive hit promotes.
      3) Speed < 0.12 m/s stays tentative.
      4) Diameter outside [0.20, 0.55] m (e.g. r=0.07 m or r=0.34 m) stays tentative.
    """
    walls = [
        (-2.0, -2.0, 3.2, -2.0),
        (3.2, -2.0, 3.2, 2.0),
        (3.2, 2.0, -2.0, 2.0),
        (-2.0, 2.0, -2.0, -2.0),
    ]

    # 1) Smoke noise (spike_ratio >= 0.03) blocks promotion even if a moving cluster appears
    trk_smoke = DynamicObstacleTracker()
    for k in range(3):
        trk_smoke.update(0.0, 0.0, 0.0, _raycast_scene(0.0, 0.0, 0.0, boxes_wall_segments=walls), scan_stamp_sec=k * 0.20)
    rng = np.random.default_rng(909)
    for k in range(15):
        cy = -0.7 + 0.08 * k
        r = _raycast_scene(0.0, 0.0, 0.0, cylinders=[(1.40, cy, 0.21)], boxes_wall_segments=walls, noise_std=0.01, rng=rng)
        # Inject smoke-like random short spikes across 18 rays (~5%)
        spike_idx = rng.choice(360, size=18, replace=False)
        r[spike_idx] = rng.uniform(0.25, 0.90, size=18).astype(np.float32)
        tracks = trk_smoke.update(0.0, 0.0, 0.0, r, scan_stamp_sec=(3 + k) * 0.20, spike_ratio=0.035)
        assert not any(tr.is_confirmed for tr in tracks), "spike_ratio >= 0.03 must block track confirmation"

    # 2) Boundary check: spike_ratio = 0.030 blocks, spike_ratio = 0.029 allows confirmation at hit 5
    trk_clean = DynamicObstacleTracker()
    for k in range(3):
        trk_clean.update(0.0, 0.0, 0.0, _raycast_scene(0.0, 0.0, 0.0, boxes_wall_segments=walls), scan_stamp_sec=k * 0.20)
    for k in range(5):
        cy = -0.30 + 0.08 * k  # 0.40 m/s >= 0.12 m/s, diam 0.42 m in [0.20, 0.55]
        r = _raycast_scene(0.0, 0.0, 0.0, cylinders=[(1.40, cy, 0.21)], boxes_wall_segments=walls)
        tracks = trk_clean.update(0.0, 0.0, 0.0, r, scan_stamp_sec=(3 + k) * 0.20, spike_ratio=0.029)
        if k < 4:
            assert not any(tr.is_confirmed for tr in tracks), f"Must not confirm before hit 5 (k={k})"
        else:
            assert sum(1 for tr in tracks if tr.is_confirmed) == 1

    # 3) Slow speed (< 0.12 m/s, e.g., 0.06 m/s) stays unconfirmed
    trk_slow = DynamicObstacleTracker()
    for k in range(3):
        trk_slow.update(0.0, 0.0, 0.0, _raycast_scene(0.0, 0.0, 0.0, boxes_wall_segments=walls), scan_stamp_sec=k * 0.20)
    for k in range(8):
        cy = -0.10 + 0.012 * k  # 0.06 m/s < 0.12 m/s
        r = _raycast_scene(0.0, 0.0, 0.0, cylinders=[(1.40, cy, 0.21)], boxes_wall_segments=walls)
        tracks = trk_slow.update(0.0, 0.0, 0.0, r, scan_stamp_sec=(3 + k) * 0.20, spike_ratio=0.0)
        assert not any(tr.is_confirmed for tr in tracks), "Speed < 0.12 m/s must not confirm"

    # 4) Out-of-range diameters (< 0.20 m and > 0.55 m) stay unconfirmed
    for rad_bad in (0.07, 0.34):
        trk_diam = DynamicObstacleTracker()
        for k in range(3):
            trk_diam.update(0.0, 0.0, 0.0, _raycast_scene(0.0, 0.0, 0.0, boxes_wall_segments=walls), scan_stamp_sec=k * 0.20)
        for k in range(8):
            cy = -0.40 + 0.08 * k
            r = _raycast_scene(0.0, 0.0, 0.0, cylinders=[(1.40, cy, rad_bad)], boxes_wall_segments=walls)
            tracks = trk_diam.update(0.0, 0.0, 0.0, r, scan_stamp_sec=(3 + k) * 0.20, spike_ratio=0.0)
            assert not any(tr.is_confirmed for tr in tracks), f"Diameter {2*rad_bad:.2f} m outside [0.20, 0.55] must not confirm"


def test_static_partition_doorway_pass_and_fast_rotation_zero_false_positives():
    """
    Regression for Phase 4 static semantic world false positives:
      1) Passing through thin 0.16 m partition doorways (Partition 1 at x=1.45, Partition 2 at x=4.35)
         with grazing parallel rays + 180 deg fast rotation (|omega|=0.65 rad/s with 40 ms yaw skew)
         next to static survivor cylinder (5.85, 0.45, r=0.22) must yield 0 confirmed dynamic tracks.
      2) A genuine moving cylinder (r=0.21 m, v=0.36 m/s) oscillating in front of a 1.15 m pinch gate
         must still confirm promptly on hit 5 without delay.
    """
    def _box_segs(cx: float, cy: float, sx: float, sy: float) -> List[Tuple[float, float, float, float]]:
        hx, hy = 0.5 * sx, 0.5 * sy
        return [
            (cx - hx, cy - hy, cx + hx, cy - hy),
            (cx + hx, cy - hy, cx + hx, cy + hy),
            (cx + hx, cy + hy, cx - hx, cy + hy),
            (cx - hx, cy + hy, cx - hx, cy - hy),
        ]

    # Exact static wall + partition geometry of disaster_semantic_rescue_world.world
    sem_walls: List[Tuple[float, float, float, float]] = [
        (-1.11, -2.31, 7.51, -2.31),
        (-1.11, 2.31, 7.51, 2.31),
        (-1.11, -2.31, -1.11, 2.31),
        (7.51, -2.31, 7.51, 2.31),
    ]
    sem_walls += _box_segs(1.45, 1.67, 0.16, 1.45)   # p1_n_c
    sem_walls += _box_segs(1.45, -1.57, 0.16, 1.65)  # p1_s_c
    sem_walls += _box_segs(4.35, 1.57, 0.16, 1.65)   # p2_n_c
    sem_walls += _box_segs(4.35, -1.67, 0.16, 1.45)  # p2_s_c
    sem_walls += _box_segs(0.40, 1.65, 0.36, 0.36)   # pil_1
    sem_walls += _box_segs(3.00, -1.75, 0.40, 0.40)  # pil_2
    sem_walls += _box_segs(6.60, -1.55, 0.45, 0.45)  # pil_3
    static_survivor = [(5.85, 0.45, 0.22)]

    rng = np.random.default_rng(2419)
    tracker = DynamicObstacleTracker()
    max_confirmed = 0

    # Build outbound -> fast 180 deg turn (with 40 ms scan-odom yaw skew) -> return pose sequence
    poses: List[Tuple[float, float, float, float, float]] = []  # (rx, ry, rth_true, rth_odom, omega)
    # Outbound along north safe path: (0.0,0.0) -> (1.55,0.55) -> (4.74,0.28)
    for k in range(125):
        frac = k / 124.0
        rx = 4.74 * frac
        if rx < 1.55:
            ry = 0.55 * (rx / 1.55)
        elif rx < 3.95:
            ry = 0.55 + 0.04 * math.sin(math.pi * (rx - 1.55) / 2.40)
        else:
            ry = 0.55 - 0.27 * ((rx - 3.95) / 0.79)
        poses.append((rx, ry, 0.0, 0.0, 0.0))
    # Fast in-place 180 deg rotation at (4.74, 0.28) with omega=0.65 rad/s and 40 ms yaw skew (0.026 rad)
    for k in range(25):
        th_true = min(math.pi, (k + 1) * 0.65 * 0.20)
        omega_k = 0.65 if th_true < math.pi - 1e-3 else 0.0
        th_odom = th_true - omega_k * 0.04  # 40 ms latency skew
        poses.append((4.74, 0.28, th_true, th_odom, omega_k))
    # Return west through Partition 2 and Partition 1 back to (0.0, 0.0)
    for k in range(125):
        frac = k / 124.0
        rx = 4.74 * (1.0 - frac)
        ry = 0.28 + 0.40 * math.sin(math.pi * frac) if rx > 1.2 else 0.0
        poses.append((rx, ry, math.pi, math.pi, 0.0))

    for step_i, (rx, ry, rth_true, rth_odom, om) in enumerate(poses):
        ranges = _raycast_scene(
            rx,
            ry,
            rth_true,
            cylinders=static_survivor,
            boxes_wall_segments=sem_walls,
            noise_std=0.01,
            rng=rng,
        )
        tracks = tracker.update(
            rx=rx,
            ry=ry,
            rth=rth_odom,
            raw_ranges=ranges,
            scan_stamp_sec=step_i * 0.20,
            omega=om,
        )
        n_conf = sum(1 for tr in tracks if tr.is_confirmed)
        if n_conf > max_confirmed:
            max_confirmed = n_conf
    assert max_confirmed == 0, f"Static semantic world doorway+rotation produced {max_confirmed} confirmed FPs"

    # 2) Verify genuine moving cylinder near a 1.15 m pinch gate still confirms on hit 5
    gate_walls: List[Tuple[float, float, float, float]] = [
        (-1.0, -2.3, 5.0, -2.3),
        (-1.0, 2.3, 5.0, 2.3),
        (-1.0, -2.3, -1.0, 2.3),
        (5.0, -2.3, 5.0, 2.3),
    ]
    gate_walls += _box_segs(2.50, 1.4375, 0.20, 1.725)
    gate_walls += _box_segs(2.50, -1.4375, 0.20, 1.725)
    trk_gate = DynamicObstacleTracker()
    for k in range(3):
        r0 = _raycast_scene(0.80, 0.0, 0.0, boxes_wall_segments=gate_walls)
        trk_gate.update(0.80, 0.0, 0.0, r0, scan_stamp_sec=k * 0.20)
    confirmed_at = None
    for k in range(7):
        cy = -0.32 + 0.072 * k  # 0.36 m/s across pinch gate at x=2.15
        rk = _raycast_scene(0.80, 0.0, 0.0, cylinders=[(2.15, cy, 0.21)], boxes_wall_segments=gate_walls)
        trks = trk_gate.update(0.80, 0.0, 0.0, rk, scan_stamp_sec=(3 + k) * 0.20)
        if any(tr.is_confirmed for tr in trks):
            confirmed_at = k + 1
            break
    assert confirmed_at == 5, f"Expected real pinch-gate cylinder to confirm on hit 5, got hit {confirmed_at}"

    # 3) Close cylinder next to a perpendicular pinch-gate wall during active steering (|omega|=0.60 rad/s)
    #    must NOT be rejected as a static wall fragment and must confirm on hit 5 once disp >= 0.25 m.
    gate2_walls: List[Tuple[float, float, float, float]] = [
        (2.0, -2.3, 8.0, -2.3),
        (2.0, 2.3, 8.0, 2.3),
        (2.0, -2.3, 2.0, 2.3),
        (8.0, -2.3, 8.0, 2.3),
    ]
    gate2_walls += _box_segs(5.30, -1.075, 0.20, 2.45)  # g2_bot_c (y <= 0.15)
    gate2_walls += _box_segs(5.30, 1.80, 0.20, 1.00)    # g2_top_c (y >= 1.30)
    trk_steer = DynamicObstacleTracker()
    for k in range(3):
        r0 = _raycast_scene(4.45, 0.65, -0.35, boxes_wall_segments=gate2_walls)
        trk_steer.update(4.45, 0.65, -0.35, r0, scan_stamp_sec=k * 0.20, omega=0.0)
    confirmed_steer = None
    for k in range(7):
        cy = 0.56 - 0.075 * k  # 0.375 m/s passing right beside perpendicular g2_bot_c
        rk = _raycast_scene(4.45, 0.65, -0.35, cylinders=[(5.05, cy, 0.21)], boxes_wall_segments=gate2_walls)
        trks = trk_steer.update(4.45, 0.65, -0.35, rk, scan_stamp_sec=(3 + k) * 0.20, omega=-0.60)
        if any(tr.is_confirmed for tr in trks):
            confirmed_steer = k + 1
            break
    assert confirmed_steer == 5, (
        f"Expected moving cylinder next to perpendicular gate wall under |omega|=0.60 to confirm on hit 5, "
        f"got hit {confirmed_steer}"
    )
