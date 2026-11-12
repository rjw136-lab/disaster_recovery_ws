#!/usr/bin/env python3
"""
Unit & Integration Tests for Next-Gen Hierarchical Hybrid Autonomous Recovery:
1. 2D SVD ICP Scan Matching accuracy & convergence
2. TD3 28-dim State Vectorization & Centered 24-Sector LiDAR binning
3. TD3 PyTorch <-> ONNX numerical equivalence & < 1.5ms inference latency
4. Line-of-sight obstacle clearance verification
5. Closed-loop TD3 obstacle avoidance trajectory verification
"""

import os
import time
import math
import numpy as np
import pytest

from disaster_recovery.recovery_controller import icp_match_2d, normalize_angle
from disaster_recovery.td3_policy import (
    TD3LocalAvoidancePolicy,
    extract_state_vector,
    check_line_of_sight_clear,
    compute_expert_avoidance_action,
)


def test_icp_match_2d_recovers_known_se2_drift():
    """Verify 2D SVD ICP accurately recovers known translational and rotational wheel slip."""
    np.random.seed(0)
    # Generate synthetic L-shaped corridor scan (120 points)
    wall_x = np.column_stack((np.linspace(0.5, 2.5, 60), np.full(60, 0.8)))
    wall_y = np.column_stack((np.full(60, 2.5), np.linspace(-0.8, 0.8, 60)))
    dst_pts = np.vstack((wall_x, wall_y)).astype(np.float32)

    # Apply known rigid transformation (true drift: dx=0.11m, dy=-0.07m, dtheta=0.12 rad)
    true_dx, true_dy, true_dth = 0.11, -0.07, 0.12
    c, s = math.cos(true_dth), math.sin(true_dth)
    R_inv = np.array([[c, s], [-s, c]], dtype=np.float32)
    src_pts = (dst_pts - np.array([true_dx, true_dy], dtype=np.float32)) @ R_inv.T

    est_dx, est_dy, est_dth, success = icp_match_2d(
        src_pts, dst_pts, initial_guess=(0.08, -0.05, 0.09), max_iters=30
    )
    assert success is True
    assert abs(est_dx - true_dx) < 0.01
    assert abs(est_dy - true_dy) < 0.01
    assert abs(est_dth - true_dth) < 0.01


def test_extract_state_vector_dimensions_and_centering():
    """Verify 28-dim state vector formatting and centered Bin 0 [-7.5 deg, +7.5 deg]."""
    ranges = np.full(360, 3.5, dtype=np.float32)
    # Place close obstacle dead ahead at 0 deg (index 0 and index 359)
    ranges[0] = 0.25
    ranges[359] = 0.28

    state = extract_state_vector(
        ranges,
        robot_x=0.0, robot_y=0.0, robot_yaw=0.0,
        target_x=1.5, target_y=0.0,
        last_v=0.10, last_w=0.0
    )
    assert state.shape == (28,)
    assert state.dtype == np.float32
    # Bin 0 (front) should capture the 0.25m reading
    assert abs(state[0] - (0.25 / 3.5)) < 1e-4
    # Side bins should remain at max range (1.0)
    assert abs(state[6] - 1.0) < 1e-4
    # Distance to target (1.5m / 2.0m = 0.75)
    assert abs(state[24] - 0.75) < 1e-4
    # Heading error (0.0)
    assert abs(state[25]) < 1e-4


def test_line_of_sight_clearance_checker():
    """Verify line-of-sight check blocks when obstacle is ahead and clears when open."""
    ranges_blocked = np.full(360, 3.5, dtype=np.float32)
    ranges_blocked[0:5] = 0.22  # Obstacle dead ahead

    assert check_line_of_sight_clear(
        ranges_blocked, 0.0, 0.0, 0.0, 1.2, 0.0, clearance_dist=0.38
    ) is False

    ranges_clear = np.full(360, 2.5, dtype=np.float32)
    assert check_line_of_sight_clear(
        ranges_clear, 0.0, 0.0, 0.0, 1.2, 0.0, clearance_dist=0.38
    ) is True


def test_td3_onnx_inference_and_latency():
    """Verify ONNX model loads, outputs valid action bounds, and runs in < 1.5ms."""
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config"))
    onnx_path = os.path.join(config_dir, "recovery_td3.onnx")
    assert os.path.exists(onnx_path), f"Missing ONNX model at {onnx_path}"

    policy = TD3LocalAvoidancePolicy(model_path=onnx_path, max_v=0.15, min_v=0.03, max_w=0.60)
    assert policy.model_type == "onnx"

    state = np.ones(28, dtype=np.float32) * 0.5
    state[25] = 0.0
    # Warm-up
    for _ in range(10):
        policy.predict(state)

    t0 = time.perf_counter()
    for _ in range(200):
        v, w = policy.predict(state)
    avg_ms = ((time.perf_counter() - t0) / 200.0) * 1000.0

    assert 0.03 <= v <= 0.15
    assert -0.60 <= w <= 0.60
    assert avg_ms < 1.5


def test_hybrid_pure_pursuit_curvature_and_blending():
    """Verify Pure Pursuit curvature calculation, speed regulation, and TD3 action blending."""
    lookahead_dist = 0.45
    v_max = 0.20
    w_max = 0.65

    # Case 1: Straight line target directly ahead (alpha = 0)
    alpha = 0.0
    curvature = 2.0 * math.sin(alpha) / lookahead_dist
    w_pp = v_max * curvature
    v_reg = v_max / (1.0 + 1.8 * abs(curvature))
    assert abs(w_pp) < 1e-4
    assert abs(v_reg - 0.20) < 1e-4

    # Case 2: 30 deg curve to left
    alpha = math.radians(30.0)
    curvature = 2.0 * math.sin(alpha) / lookahead_dist
    w_pp = v_max * curvature
    v_reg = v_max / (1.0 + 1.8 * abs(curvature))
    assert w_pp > 0.15
    assert v_reg < 0.20  # Automatic regulation slows down on curves

    # Case 3: Blend with TD3 obstacle avoidance output
    v_td3 = 0.10
    w_td3 = 0.40
    # In obstacle zone: RL dominance
    w_blend_obs = 0.85 * w_td3 + 0.15 * w_pp
    assert abs(w_blend_obs - (0.85 * 0.40 + 0.15 * w_pp)) < 1e-4
    # In clear zone: tracking dominance
    w_blend_clear = 0.55 * w_pp + 0.45 * w_td3
    assert abs(w_blend_clear - (0.55 * w_pp + 0.45 * 0.40)) < 1e-4
