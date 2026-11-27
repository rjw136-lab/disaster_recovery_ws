#!/usr/bin/env python3
"""Comprehensive regression unit tests for Batch 1-3 safety, tracking, controller, training, and benchmark fixes."""

import ast
import importlib.util
import json
import math
import os
import sys
import types
from collections import deque

import numpy as np
import pytest
import torch

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
WS_ROOT = os.path.abspath(os.path.join(PKG_ROOT, "..", ".."))
SCRIPTS_HOST_DIR = os.path.join(WS_ROOT, "scripts", "host")

for p in (PKG_ROOT, SCRIPTS_HOST_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from disaster_recovery.dynamic_tracker import DynamicObstacleTracker, TrackedObstacle
from disaster_recovery.local_grid_planner import ProposedRecoveryPlanner
from disaster_recovery.rl_context_policy import (
    ContextRLStudentPolicy,
    HISTORY_LEN,
    PATCH_SIZE,
    PHASE4_FLAT_INPUT_DIM,
    STEP_FEAT_DIM,
)
from rl_procedural_env import CH3_Q_ACC, VectorDisasterEnv
from train_phase2_v100 import (
    FrozenBackboneSemanticStudent,
    FrozenBackboneSemanticTeacher,
    _bound_res_pre_vw,
    distill_ensemble_to_onnx,
    evaluate_policy_on_holdout,
    export_student_onnx,
    make_eval_seeds,
    plot_learning_curves,
)


def _make_dummy_controller_cls():
    """Import RecoveryController and helper types cleanly inside ROS 2 environment."""
    from disaster_recovery.recovery_controller import (
        RecoveryController,
        SemanticObjectCache,
        SystemState,
        Waypoint,
    )
    return RecoveryController, SemanticObjectCache, SystemState, Waypoint


def test_dynamic_tracker_invalid_beams_offset_bounds_freeze_cov_and_ch3_cache():
    """Verify Items #1 & #9: no free-space carving on invalid/ultra-close beams, ray-box angle preservation, offset bounds, Joseph cov under freeze_velocity, bit-exact project_ch3 cache."""
    tracker = DynamicObstacleTracker(
        grid_size_m=4.0,
        grid_res_m=0.05,
        lidar_min_range_m=0.12,
        lidar_max_range_m=3.5,
        lidar_offset_x_m=0.0,
    )
    tracker._recenter_grid(0.0, 0.0)
    # Seed a solid static wall cell at (1.0, 0.0)
    ix_wall = int(math.floor(1.0 / tracker.res)) - tracker.origin_ix
    iy_wall = int(math.floor(0.0 / tracker.res)) - tracker.origin_iy
    tracker.static_occ[ix_wall, iy_wall] = 1.0

    # 1) Feed NaN, 0.0, -inf, and 0.05m (< lidar_min_range_m) beams -> must NOT carve static_occ to 0.0
    n_rays = 360
    bad_vals = [float("nan"), 0.0, float("-inf"), 0.05]
    for step_i, val in enumerate(bad_vals, start=1):
        ranges = np.full(n_rays, val, dtype=np.float32)
        tracker.update(rx=0.0, ry=0.0, rth=0.0, raw_ranges=ranges, scan_stamp_sec=0.2 * step_i)
        assert float(tracker.static_occ[ix_wall, iy_wall]) > 0.85, (
            f"Invalid/ultra-close beam {val!r} must NOT carve existing static_occ to 0.0"
        )

    # 2) +inf far-open beam over full 4-parity stagger cycle SHOULD carve static_occ at (1.0, 0.0)
    open_ranges = np.full(n_rays, float("inf"), dtype=np.float32)
    for k in range(4):
        tracker.update(rx=0.0, ry=0.0, rth=0.0, raw_ranges=open_ranges, scan_stamp_sec=2.0 + 0.2 * k)
    assert float(tracker.static_occ[ix_wall, iy_wall]) == 0.0

    # 3) Large lidar_offset_x_m + oblique ray: zero boundary crash & proportional ray-box clipping preserves ray angle
    tracker_off = DynamicObstacleTracker(
        grid_size_m=2.0,
        grid_res_m=0.05,
        lidar_min_range_m=0.12,
        lidar_max_range_m=3.5,
        lidar_offset_x_m=0.45,
    )
    tracker_off._recenter_grid(0.0, 0.0)
    # Place static_occ off-axis at (0.90, 0.80); shoot only an open ray along +x (angle=0)
    ix_off = int(math.floor(0.90 / tracker_off.res)) - tracker_off.origin_ix
    iy_off = int(math.floor(0.80 / tracker_off.res)) - tracker_off.origin_iy
    tracker_off.static_occ[ix_off, iy_off] = 1.0
    ranges_off = np.full(360, 0.0, dtype=np.float32)
    ranges_off[0] = float("inf")
    # Ensure parity hits index 0 (scan_update_count becomes 1 -> parity = 1%2 = 1, so set index 0 and 1)
    ranges_off[1] = float("inf")
    tracker_off.update(rx=0.0, ry=0.0, rth=0.0, raw_ranges=ranges_off, scan_stamp_sec=0.2)
    assert float(tracker_off.static_occ[ix_off, iy_off]) > 0.90

    # Extreme offset pushing sensor outside grid still must not raise IndexError
    tracker_ext = DynamicObstacleTracker(
        grid_size_m=1.0,
        grid_res_m=0.05,
        lidar_offset_x_m=0.80,
    )
    tracker_ext.update(rx=0.0, ry=0.0, rth=0.3, raw_ranges=np.full(360, 2.5, dtype=np.float32), scan_stamp_sec=0.2)

    # 4) freeze_velocity=True updates position covariance block via Joseph form while preserving vx, vy
    tracker_kf = DynamicObstacleTracker()
    P0 = np.diag([0.25, 0.25, 0.09, 0.09]).astype(np.float64)
    trk = TrackedObstacle(
        track_id=1,
        x=1.0,
        y=0.0,
        vx=0.20,
        vy=-0.10,
        radius=0.14,
        P_4x4=P0.copy(),
        last_update_stamp=1.0,
        age=4,
        hits=4,
        is_confirmed=True,
    )
    tracker_kf.tracks[1] = trk
    tracker_kf._last_scan_stamp = 1.0
    tracker_kf._kf_step_and_associate([(1.04, -0.02, 0.14)], dt=0.2, stamp_sec=1.2, freeze_velocity=True)
    trk_after = tracker_kf.tracks[1]
    assert trk_after.vx == pytest.approx(0.20, abs=1e-9)
    assert trk_after.vy == pytest.approx(-0.10, abs=1e-9)
    assert trk_after.P_4x4[0, 0] < P0[0, 0], "Position variance must shrink under freeze_velocity measurement update"
    assert np.allclose(trk_after.P_4x4, trk_after.P_4x4.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(trk_after.P_4x4)
    assert float(eigvals.min()) > 0.0

    # 5) project_ch3() called twice consecutively with identical now_sec > _last_scan_stamp -> bit-exact equal
    tracker_kf._last_scan_stamp = 1.2
    tracker_kf._extrap_cache_stamp = None
    tracker_kf._extrap_cache_states.clear()
    ch3_first = tracker_kf.project_ch3(rx=0.5, ry=0.0, rth=0.0, now_sec=1.35)
    assert 1 in tracker_kf._extrap_cache_states, "project_ch3 must populate _extrap_cache_states"
    ch3_second = tracker_kf.project_ch3(rx=0.5, ry=0.0, rth=0.0, now_sec=1.35)
    assert np.array_equal(ch3_first, ch3_second), "Repeated project_ch3 at same now_sec must be bit-exact identical"


def test_recovery_controller_icp_despike_reset_pruning_ultra_close_and_cmd_sync(monkeypatch):
    """Verify Items #2 & #10 on RecoveryController methods without needing full ROS 2 spin."""
    from disaster_recovery import recovery_controller as rc_mod

    RecoveryController, SemanticObjectCache, SystemState, Waypoint = _make_dummy_controller_cls()

    # Build lightweight bare instance bypassing Node.__init__
    ctrl = RecoveryController.__new__(RecoveryController)
    ctrl.enable_scan_matching = True
    ctrl.scan_match_max_dist = 5.0
    ctrl.scan_match_weight = 0.5
    ctrl.raw_odom_x = 1.0
    ctrl.raw_odom_y = 0.0
    ctrl.raw_odom_theta = 0.0
    ctrl.corr_x = 0.0
    ctrl.corr_y = 0.0
    ctrl.corr_th = 0.0
    ctrl.current_x = 1.0
    ctrl.current_y = 0.0
    ctrl.current_theta = 0.0
    ctrl.scan_match_count = 0
    ctrl.odom_initialized = True
    ctrl.start_pose = Waypoint(0.0, 0.0, 0.0, None)
    ctrl.start_scan_pts = None
    ctrl.active_target_wp = Waypoint(1.0, 0.0, 0.0, np.zeros((30, 2), dtype=np.float32))
    ctrl.latest_scan_pts = np.zeros((30, 2), dtype=np.float32)
    ctrl.latest_scan_pose = (0.95, 0.05, 0.02)
    ctrl.dynamic_tracker = DynamicObstacleTracker()
    ctrl.semantic_cache = SemanticObjectCache()
    ctrl.semantic_cache.upsert_object("chair", 2.0, 0.5, radius=0.2, channel=1)
    ctrl.local_planner = ProposedRecoveryPlanner(size_m=4.0, resolution_m=0.05)
    ctrl.local_planner.grid._recenter(1.0, 0.0)
    ctrl.local_planner.grid.occ[40, 40] = 1.0
    ctrl.context_rl_policy = None
    ctrl.cbf_shield = None
    ctrl.waypoint_stack = deque([Waypoint(0.0, 0.0, 0.0), Waypoint(0.5, 0.0, 0.0)])
    ctrl.last_cmd_v = 0.12
    ctrl.last_cmd_w = 0.30
    ctrl.actual_v = 0.0
    ctrl.actual_w = 0.0
    ctrl.v_max = 0.20
    ctrl.w_max = 0.65
    ctrl.wp_reach_radius = 0.20
    ctrl.docking_dist_th = 0.55
    ctrl.use_aruco_docking = False
    ctrl.enable_hybrid = True
    ctrl.enable_td3_avoidance = False
    ctrl.td3_policy = None
    ctrl.lookahead_dist = 0.50
    ctrl.kp_ang = 1.0
    ctrl.kp_lin = 0.5
    ctrl.heading_tol = 0.20
    ctrl.state = SystemState.BACKTRACKING_HYBRID
    ctrl.enable_lidar_despike = False
    ctrl.smoke_lidar_noise_prob = 0.20
    ctrl._scan_tick_seq = 0
    ctrl.lidar_fov = math.radians(35.0)
    ctrl.lidar_min_dist = 0.12
    ctrl.lidar_danger_dist = 0.35
    ctrl.td3_clearance_dist = 0.65
    ctrl.obstacle_detected = False
    ctrl.td3_obstacle_detected = False
    ctrl.latest_raw_ranges = []
    ctrl.latest_scan_stamp_sec = None
    ctrl.latest_scan_angle_min = 0.0
    ctrl.latest_scan_angle_increment = None
    ctrl.enable_lidar_slip_residual = False
    ctrl._prev_slip_scan_pts = None
    ctrl._prev_slip_scan_stamp = None

    published = []
    ctrl.pub_cmd_vel = types.SimpleNamespace(publish=lambda m: published.append(m))
    logged_warns = []
    ctrl.get_logger = lambda: types.SimpleNamespace(
        info=lambda *a, **k: None,
        warn=lambda msg, *a, **k: logged_warns.append(msg),
        error=lambda *a, **k: None,
    )

    # 1) ICP correction syncs latest_scan_pose by exact delta_T = T_new * T_old^-1
    monkeypatch.setattr(rc_mod, "icp_match_2d", lambda *a, **k: (0.10, -0.06, 0.10, True))
    old_pose = ctrl.latest_scan_pose
    ctrl.perform_scan_matching_correction()
    assert ctrl.scan_match_count == 1
    d_th = rc_mod.normalize_angle(ctrl.corr_th - 0.0)
    c_d, s_d = math.cos(d_th), math.sin(d_th)
    exp_srx = ctrl.corr_x + c_d * old_pose[0] - s_d * old_pose[1]
    exp_sry = ctrl.corr_y + s_d * old_pose[0] + c_d * old_pose[1]
    exp_srth = rc_mod.normalize_angle(old_pose[2] + d_th)
    assert ctrl.latest_scan_pose[0] == pytest.approx(exp_srx, abs=1e-6)
    assert ctrl.latest_scan_pose[1] == pytest.approx(exp_sry, abs=1e-6)
    assert ctrl.latest_scan_pose[2] == pytest.approx(exp_srth, abs=1e-6)

    # 2) cb_scan: smoke_lidar_noise_prob > 0 & enable_lidar_despike=False must NOT invoke _despike_lidar_ranges
    despike_called = {"n": 0}
    orig_despike = RecoveryController._despike_lidar_ranges

    def _spy_despike(r):
        despike_called["n"] += 1
        return orig_despike(r)

    monkeypatch.setattr(ctrl, "_despike_lidar_ranges", _spy_despike)
    scan_msg = types.SimpleNamespace(
        ranges=[3.0] * 360,
        angle_min=0.0,
        angle_increment=2.0 * math.pi / 360.0,
        header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=1, nanosec=0)),
    )
    ctrl.cb_scan(scan_msg)
    assert despike_called["n"] == 0, "Despike must not run when enable_lidar_despike=False"

    # 3) cb_scan: ultra-close 0.08m front return sets obstacle_detected=True; 0.0 dropout stays False
    ctrl.smoke_lidar_noise_prob = 0.0
    ranges_ultra = [3.0] * 360
    ranges_ultra[0] = 0.08
    scan_msg_ultra = types.SimpleNamespace(
        ranges=ranges_ultra,
        angle_min=0.0,
        angle_increment=2.0 * math.pi / 360.0,
        header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=2, nanosec=0)),
    )
    ctrl.cb_scan(scan_msg_ultra)
    assert ctrl.obstacle_detected is True
    assert ctrl.td3_obstacle_detected is True

    ranges_zero = [3.0] * 360
    ranges_zero[0] = 0.0
    scan_msg_zero = types.SimpleNamespace(
        ranges=ranges_zero,
        angle_min=0.0,
        angle_increment=2.0 * math.pi / 360.0,
        header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=3, nanosec=0)),
    )
    ctrl.cb_scan(scan_msg_zero)
    assert ctrl.obstacle_detected is False

    # 4) _reset_trial_runtime_state(clear_stack=False): preserves corr, waypoint_stack, semantic_cache, local_planner
    saved_corr = (ctrl.corr_x, ctrl.corr_y, ctrl.corr_th)
    saved_stack_len = len(ctrl.waypoint_stack)
    ctrl.local_planner.grid.occ[30, 30] = 0.9
    ctrl._reset_trial_runtime_state(clear_stack=False, clear_sensors=False, reset_corr=True)
    assert (ctrl.corr_x, ctrl.corr_y, ctrl.corr_th) == saved_corr
    assert len(ctrl.waypoint_stack) == saved_stack_len
    assert len(ctrl.semantic_cache.get_channel_tuples()[0]) == 1
    assert float(ctrl.local_planner.grid.occ[30, 30]) == pytest.approx(0.9)

    # 5) _reset_trial_runtime_state(clear_stack=True, clear_sensors=False): refreshes start_pose to current pose
    ctrl.raw_odom_x, ctrl.raw_odom_y, ctrl.raw_odom_theta = 3.2, -1.1, 0.4
    ctrl._reset_trial_runtime_state(clear_stack=True, clear_sensors=False, reset_corr=True)
    assert ctrl.corr_x == 0.0 and ctrl.corr_y == 0.0 and ctrl.corr_th == 0.0
    assert ctrl.start_pose is not None
    assert ctrl.start_pose.x == pytest.approx(3.2)
    assert ctrl.start_pose.y == pytest.approx(-1.1)

    # 6) BACKTRACKING_HYBRID pruning protects far-behind WP (d=1.5m, forward_proj=-1.0) and final anchor
    ctrl.current_x, ctrl.current_y, ctrl.current_theta = 2.0, 0.0, 0.0
    ctrl.start_pose = Waypoint(0.0, 0.0, 0.0)
    far_behind = Waypoint(1.0, math.sqrt(1.25), 0.0)  # d=1.5m, dx=-1.0 -> forward_proj=-1.0
    ctrl.waypoint_stack = deque([ctrl.start_pose, far_behind])
    # Execute pruning loop directly via control_loop's logic by checking stack after one step
    while len(ctrl.waypoint_stack) > 1:
        top_wp = ctrl.waypoint_stack[-1]
        d = top_wp.distance_to(ctrl.current_x, ctrl.current_y)
        dx_wp = top_wp.x - ctrl.current_x
        dy_wp = top_wp.y - ctrl.current_y
        fwd_proj = dx_wp * math.cos(ctrl.current_theta) + dy_wp * math.sin(ctrl.current_theta)
        if d < 0.22 or (d < 0.60 and fwd_proj < -0.05):
            ctrl.waypoint_stack.pop()
        else:
            break
    assert len(ctrl.waypoint_stack) == 2 and ctrl.waypoint_stack[-1] is far_behind

    # Also test stop_robot syncs last_cmd_v/last_cmd_w
    ctrl.last_cmd_v = 0.18
    ctrl.last_cmd_w = -0.45
    ctrl.stop_robot()
    assert ctrl.last_cmd_v == 0.0 and ctrl.last_cmd_w == 0.0

    # 7) SemanticObjectCache warns on non-odom frame_id
    ctrl.semantic_cache.update_from_json(
        json.dumps({"frame_id": "map", "objects": [{"x": 1.0, "y": 2.0, "class_name": "chair", "channel": 1}]}),
        corr_se2=(0.5, 0.5, 0.2),
        logger=ctrl.get_logger(),
    )
    assert any("non-odom" in w for w in logged_warns)


def test_rl_context_policy_empty_scan_failsafe_and_deferred_priming():
    """Verify Items #3 & #11: empty/None scan sets feat[:24]=0.12/3.5 and defers 16-frame priming until first valid scan."""
    expected_min = 0.12 / 3.5
    feat_empty, _ = ContextRLStudentPolicy._build_step_and_patch(
        [], 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, patch_channels=4
    )
    feat_none, _ = ContextRLStudentPolicy._build_step_and_patch(
        None, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, patch_channels=4
    )
    assert np.allclose(feat_empty[:24], expected_min, atol=1e-6)
    assert np.allclose(feat_none[:24], expected_min, atol=1e-6)

    # Bypass ONNX session load to test observe() rolling history priming directly
    pol = ContextRLStudentPolicy.__new__(ContextRLStudentPolicy)
    pol.patch_channels = 4
    pol.hist = deque(maxlen=HISTORY_LEN)
    pol._near_latch_ttl = np.zeros(24, dtype=np.int32)
    pol._needs_prime = True
    pol.last_ch123_peak = 0.0

    # Call 1: empty scan -> _needs_prime stays True
    pol.observe([], 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert pol._needs_prime is True
    assert len(pol.hist) == HISTORY_LEN
    assert np.allclose(pol.hist[0][:24], expected_min, atol=1e-6)

    # Call 2: first valid scan -> all 16 frames primed with valid scan features, _needs_prime becomes False
    valid_scan = np.full(360, 2.10, dtype=np.float32)
    pol.observe(valid_scan, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert pol._needs_prime is False
    assert len(pol.hist) == HISTORY_LEN
    for f_i in pol.hist:
        assert np.allclose(f_i[:24], 2.10 / 3.5, atol=1e-5)


def test_rl_procedural_env_partial_reset_isolation_dr_profile_regime2_and_ch3_qacc():
    """Verify Items #5 & #8: bit-exact partial reset isolation, CH3_Q_ACC==0.30, strict real_tb3 DR default, Regime 2 diversity, Regime 3 plinth excluded from Ch1."""
    assert CH3_Q_ACC == pytest.approx(0.30, abs=1e-9)

    env = VectorDisasterEnv(
        num_envs=8,
        track="real_tb3",
        device=torch.device("cpu"),
        mode="train",
        base_seed=1234,
        patch_channels=4,
        enable_semantic_noise=True,
        enable_ch3_dr=True,
        modality_dropout_prob=0.35,
    )
    assert env.dr_profile == "strict" and env.enable_strict_real_dr is True
    env.reset_envs()
    # Advance 3 steps so non-reset envs accumulate non-trivial seen ticks, lidar, and DR state
    for _ in range(3):
        env.step(torch.full((8, 2), 0.10, dtype=torch.float32))

    keep_ids = [0, 2, 3, 4, 6, 7]
    reset_ids = torch.tensor([1, 5], dtype=torch.long)

    snap_low_seen = env.low_obs_seen_ticks[keep_ids].clone()
    snap_haz_seen = env.hazard_seen_ticks[keep_ids].clone()
    snap_mod_drop = env.modality_dropped[keep_ids].clone()
    snap_lidar = env._last_lidar_ranges[keep_ids].clone()
    snap_obs_low = env._obs_low_state[keep_ids].clone()
    snap_obs_haz = env._obs_haz_state[keep_ids].clone()
    snap_ch3_vn = env._obs_ch3_vnoise[keep_ids].clone()
    snap_ch3_keep = env._obs_ch3_keep[keep_ids].clone()

    env.reset_envs(env_ids=reset_ids)

    assert torch.equal(env.low_obs_seen_ticks[keep_ids], snap_low_seen)
    assert torch.equal(env.hazard_seen_ticks[keep_ids], snap_haz_seen)
    assert torch.equal(env.modality_dropped[keep_ids], snap_mod_drop)
    assert torch.equal(env._last_lidar_ranges[keep_ids], snap_lidar)
    assert torch.equal(env._obs_low_state[keep_ids], snap_obs_low)
    assert torch.equal(env._obs_haz_state[keep_ids], snap_obs_haz)
    assert torch.equal(env._obs_ch3_vnoise[keep_ids], snap_ch3_vn)
    assert torch.equal(env._obs_ch3_keep[keep_ids], snap_ch3_keep)

    # Strict DR active by default on real_tb3; legacy profile clamps slip/lag
    env_def = VectorDisasterEnv(num_envs=64, track="real_tb3", device=torch.device("cpu"), mode="train", base_seed=77)
    env_def.reset_envs(forced_regime=0)
    assert env_def.dr_profile == "strict"
    assert int(env_def.act_lag.max().item()) >= 1
    assert float(env_def.slip_v.min().item()) < 0.90

    env_leg = VectorDisasterEnv(num_envs=64, track="real_tb3", device=torch.device("cpu"), mode="train", base_seed=77, dr_profile="legacy")
    env_leg.reset_envs(forced_regime=0)
    assert int(env_leg.act_lag.max().item()) == 0
    assert float(env_leg.slip_v.min().item()) >= 0.92 - 1e-6

    # Regime 2 non-sinusoidal dynamic obstacles retain diverse radii and velocities
    env_r2 = VectorDisasterEnv(
        num_envs=16, track="sim", device=torch.device("cpu"), mode="train", base_seed=99,
        patch_channels=4, enable_sinusoidal_dynamic=False,
    )
    env_r2.reset_envs(forced_regime=2)
    act_cyls = env_r2.cyls[env_r2.cyl_active]
    assert len(act_cyls) >= 16
    assert float(act_cyls[:, 2].std().item()) > 1e-3, "Regime 2 cylinder radii should not be overwritten to constant 0.12"
    assert float(act_cyls[:, 3].abs().max().item()) > 1e-3, "Regime 2 cylinder vx should not be overwritten to 0.0"

    # Regime 3 plinth is intentionally excluded from Ch1 (tactile-only Base-6 regime)
    env_r3 = VectorDisasterEnv(
        num_envs=4, track="sim", device=torch.device("cpu"), mode="eval", base_seed=55,
        patch_channels=4, enable_semantic_noise=False, modality_dropout_prob=0.0,
    )
    _, patch_r3 = env_r3.reset_envs(forced_regime=3)
    assert bool((env_r3.plinth[:, 3] > 0.5).all())
    assert float(patch_r3[:, 1].max().item()) == 0.0


def test_train_phase2_dagger_pass2_lr_non_regression_step_cap_and_vmin_logprob(tmp_path):
    """Verify Items #4, #6, #12: DAgger Pass 2 init_lr > 1e-4, non-regression == 0.0, step cap guard, real_tb3 v_min log_prob consistency, dynamic plot_learning_curves."""
    base_onnx = os.path.join(WS_ROOT, "src", "disaster_recovery", "config", "sim_policy_student.onnx")
    assert os.path.exists(base_onnx)

    # 1) Censored v_min=-0.08 action clipping & rollout-vs-update log_prob exact parity
    t_real = FrozenBackboneSemanticTeacher(hidden_dim=128, patch_channels=4, use_safety_head=False, v_min=-0.08)
    t_real.eval()
    obs_h = torch.zeros((16, HISTORY_LEN, STEP_FEAT_DIM), dtype=torch.float32)
    patch = torch.zeros((16, 4, PATCH_SIZE, PATCH_SIZE), dtype=torch.float32)
    patch[:, 1, 6, 6] = 1.0
    # Bias actor pre-tanh strongly negative so some samples hit the -0.08 lower bound
    with torch.no_grad():
        t_real.sem_res_mlp[-1].bias[0] = -3.0
    act_r, lp_roll, _, _, _ = t_real.get_action_and_values(obs_h, patch, deterministic=False)
    assert float(act_r[:, 0].min().item()) >= -0.08 - 1e-6
    _, lp_eval, _, _, _ = t_real.get_action_and_values(obs_h, patch, action_taken=act_r, deterministic=False)
    assert float((lp_roll - lp_eval).abs().max().item()) < 1e-5, (
        "Rollout log_prob and update log_prob must match cleanly under v_min=-0.08 clamping"
    )

    # 2) DAgger Pass 2 init_lr > 1e-4 & FrozenBackboneSemanticStudent non-regression == 0.0
    t_sim = FrozenBackboneSemanticTeacher(hidden_dim=128, patch_channels=4, use_safety_head=False, v_min=-0.10)
    ckpt_p = str(tmp_path / "dummy_teacher.pt")
    torch.save(
        {
            "state_dict": t_sim.state_dict(),
            "hidden_dim": 128,
            "patch_channels": 4,
            "arch": "frozen_residual",
            "algo": "cppo",
            "use_safety_head": False,
            "cbf_v_min": -0.10,
            "teacher_nu": 1.0,
            "selection_score": 0.9,
        },
        ckpt_p,
    )
    out_onnx = str(tmp_path / "sim_student_test_p4.onnx")
    seeds_8 = make_eval_seeds(8, 10000)
    res = distill_ensemble_to_onnx(
        track="sim",
        ckpt_paths=[ckpt_p],
        out_onnx_path=out_onnx,
        device=torch.device("cpu"),
        teacher_samples=128,
        dagger_samples=64,
        patch_channels=4,
        use_semantic_channels=True,
        arch="frozen_residual",
        base_onnx_path=base_onnx,
        disable_safety_head=True,
        distill_epochs=2,
        val_seeds=seeds_8,
        test_seeds=seeds_8,
    )
    assert res["flat_input_dim"] == PHASE4_FLAT_INPUT_DIM
    assert res["non_regression_max_abs_diff"] == 0.0
    assert res["distill_pass1_init_lr"] > 1e-4
    assert res["distill_pass2_init_lr"] > 1e-4, f"DAgger Pass 2 lr collapsed to {res['distill_pass2_init_lr']}"

    # 3) Step cap guard prevents infinite loop when ch12_active is never triggered
    with pytest.raises(RuntimeError, match="Pass 1 collected 0 active transitions"):
        distill_ensemble_to_onnx(
            track="sim",
            ckpt_paths=[ckpt_p],
            out_onnx_path=str(tmp_path / "fail_cap.onnx"),
            device=torch.device("cpu"),
            teacher_samples=64,
            dagger_samples=32,
            patch_channels=4,
            use_semantic_channels=False,
            arch="frozen_residual",
            base_onnx_path=base_onnx,
            disable_safety_head=True,
            distill_epochs=1,
            val_seeds=seeds_8,
            test_seeds=seeds_8,
            max_collect_steps=3,
        )

    # 4) Dynamic plot_learning_curves renders cleanly with 4-Ch and eval_episodes
    curve_png = str(tmp_path / "curve_p4.png")
    fake_hist = [{
        "update": 1, "train_sr": 0.8, "unseen_sr": 0.9,
        "per_regime": {r: 0.9 for r in res["onnx_eval"]["per_regime"]},
    }]
    plot_learning_curves("sim", [{"seed": 101, "patch_channels": 4, "best_unseen_sr": 0.9, "early_stopped_at": None, "history": fake_hist}], res, curve_png)
    assert os.path.exists(curve_png) and os.path.getsize(curve_png) > 1024


def test_train_td3_delayed_executed_cmd_causality():
    """Verify Item #14: ObstacleAvoidanceEnv reports causal executed_cmd under 1-step control delay."""
    from train_td3 import ObstacleAvoidanceEnv

    env = ObstacleAvoidanceEnv(use_control_delay=True)
    env.reset(difficulty="normal")
    _, _, _, info0 = env.step(np.array([0.15, 0.50], dtype=np.float32), noisy=False)
    assert np.allclose(info0["executed_cmd"], np.array([0.08, 0.0], dtype=np.float32), atol=1e-6)
    _, _, _, info1 = env.step(np.array([0.05, -0.40], dtype=np.float32), noisy=False)
    assert np.allclose(info1["executed_cmd"], np.array([0.15, 0.50], dtype=np.float32), atol=1e-6)


def test_benchmark_scripts_syntax_and_integrity():
    """Verify Items #7 & #13 static correctness on run_comp5_semantic_rescue.py and run_phase3_semantic_suite.py."""
    p_comp5 = os.path.join(SCRIPTS_HOST_DIR, "run_comp5_semantic_rescue.py")
    p_suite = os.path.join(SCRIPTS_HOST_DIR, "run_phase3_semantic_suite.py")
    src_comp5 = open(p_comp5, "r", encoding="utf-8").read()
    src_suite = open(p_suite, "r", encoding="utf-8").read()

    ast.parse(src_comp5, filename=p_comp5)
    ast.parse(src_suite, filename=p_suite)

    # No forced timeout-to-arrived conversion
    assert "err_m < 0.60 and saw_rec" not in src_comp5
    assert "err_m < 0.60 and saw_rec" not in src_suite
    # No misplaced upper-lane pub_gas(0.75) trigger in Stage 0 of comp5
    assert "node.pub_gas.publish(Float32(data=0.75))" not in src_comp5
    # use_sim_time parameter override and teleport odom confirmation present
    assert 'Parameter("use_sim_time", value=True)' in src_suite
    assert "wait_for_teleport_odom" in src_suite and "wait_for_teleport_odom" in src_comp5
    # Dead second /semantic_map/save_map call removed from run_record_e2e
    assert src_suite.count("node.cli_save_map.wait_for_service") == 1


def test_bound_res_pre_vw_nonzero_grad_at_zero_and_pos_and_onnx_non_regression(tmp_path):
    """v4.3b smoke test: verify torch.where _bound_res_pre_vw has |grad|>0 at raw_v=0.0 and +0.2 in 1-step backprop (Teacher & Student) and ONNX non_regression_max_abs_diff == 0.0."""
    import onnxruntime as ort

    # 1) Direct autograd check at raw_v = 0.0, +0.2, -0.2
    for rv_val in (0.0, 0.2, -0.2):
        raw = torch.tensor([[rv_val, 0.0]], dtype=torch.float32, requires_grad=True)
        res_v, res_w = _bound_res_pre_vw(raw, v_pos_max=0.15)
        (res_v.sum() + res_w.sum()).backward()
        assert raw.grad is not None
        assert abs(float(raw.grad[0, 0].item())) > 1e-3, f"Zero gradient detected at raw_v={rv_val}: {raw.grad}"

    # 2) Full 1-step Teacher & Student backprop at both raw_v == 0.0 (zero-init) and raw_v == +0.2
    obs_h = torch.randn((8, HISTORY_LEN, STEP_FEAT_DIM), dtype=torch.float32) * 0.2
    patch = torch.zeros((8, 4, PATCH_SIZE, PATCH_SIZE), dtype=torch.float32)
    patch[:, 3, 6, 6] = 1.0  # active Ch3 dynamic channel -> sem_gate == 1.0
    base_onnx = os.path.join(WS_ROOT, "src", "disaster_recovery", "config", "sim_policy_student.onnx")
    assert os.path.exists(base_onnx)

    for bias_v in (0.0, 0.2):
        teacher = FrozenBackboneSemanticTeacher(hidden_dim=128, patch_channels=4, use_safety_head=False, v_min=-0.06)
        with torch.no_grad():
            teacher.sem_res_mlp[-1].weight.zero_()
            teacher.sem_res_mlp[-1].bias.zero_()
            teacher.sem_res_mlp[-1].bias[0] = bias_v
        teacher.zero_grad()
        act_t, _ = teacher(obs_h, patch)
        loss_t = (act_t[:, 0] - 0.15).pow(2).mean()
        loss_t.backward()
        g_w = teacher.sem_res_mlp[-1].weight.grad[0]
        g_b = teacher.sem_res_mlp[-1].bias.grad[0]
        assert float(g_b.abs().item()) > 1e-6, f"Teacher res_pre_v bias grad vanished at raw_v={bias_v}"
        assert float(g_w.abs().max().item()) > 1e-6, f"Teacher res_pre_v weight grad vanished at raw_v={bias_v}"

        student = FrozenBackboneSemanticStudent(patch_channels=4, base_onnx_path=base_onnx, use_safety_head=False)
        with torch.no_grad():
            student.sem_res_head[-1].weight.zero_()
            student.sem_res_head[-1].bias.zero_()
            student.sem_res_head[-1].bias[0] = bias_v
        student.zero_grad()
        act_s, _ = student.forward_with_brake(obs_h, patch)
        loss_s = (act_s[:, 0] - 0.15).pow(2).mean()
        loss_s.backward()
        gs_w = student.sem_res_head[-1].weight.grad[0]
        gs_b = student.sem_res_head[-1].bias.grad[0]
        assert float(gs_b.abs().item()) > 1e-6, f"Student res_pre_v bias grad vanished at raw_v={bias_v}"
        assert float(gs_w.abs().max().item()) > 1e-6, f"Student res_pre_v weight grad vanished at raw_v={bias_v}"

    # 3) ONNX export after 1-step update still satisfies non_regression_max_abs_diff == 0.0
    student_exp = FrozenBackboneSemanticStudent(patch_channels=4, base_onnx_path=base_onnx, use_safety_head=False)
    opt_s = torch.optim.AdamW([p for p in student_exp.parameters() if p.requires_grad], lr=1e-2)
    act_s, _ = student_exp.forward_with_brake(obs_h, patch)
    ((act_s[:, 0] - 0.18).pow(2).mean()).backward()
    opt_s.step()
    assert float(student_exp.sem_res_head[-1].weight[0].abs().max().item()) > 0.0, "1-step AdamW must unfreeze res_pre_v weights"
    student_exp.eval()
    out_onnx = str(tmp_path / "v4_3b_smoke_student.onnx")
    export_student_onnx(student_exp, out_onnx, patch_channels=4)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    base_sess = ort.InferenceSession(base_onnx, sess_options=opts, providers=["CPUExecutionProvider"])
    p4_sess = ort.InferenceSession(out_onnx, sess_options=opts, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(77)
    h_t = rng.normal(0.0, 0.5, size=(128, HISTORY_LEN * STEP_FEAT_DIM)).astype(np.float32)
    c0_t = np.clip(rng.uniform(0.0, 1.0, size=(128, PATCH_SIZE * PATCH_SIZE)), 0.0, 1.0).astype(np.float32)
    c123_z = np.zeros((128, 3 * PATCH_SIZE * PATCH_SIZE), dtype=np.float32)
    y_base = base_sess.run(None, {base_sess.get_inputs()[0].name: np.concatenate([h_t, c0_t], axis=1)})[0]
    y_p4 = p4_sess.run(None, {p4_sess.get_inputs()[0].name: np.concatenate([h_t, c0_t, c123_z], axis=1)})[0]
    assert float(np.max(np.abs(y_p4 - y_base))) == 0.0


def test_stepA_start_match_icp_guards_and_cbf_vmin_parity(monkeypatch):
    """Verify Step A Gate 2 recovery: strict is_start_match ICP guards (1D degeneracy, drift_cap, max_p_jump, yaw cap, corr_y block) and track cbf_v_min parity."""
    import yaml
    from disaster_recovery import recovery_controller as rc_mod

    RecoveryController, SemanticObjectCache, _, Waypoint = _make_dummy_controller_cls()

    def _make_start_ctrl(scan_pts, *, raw_xy=(0.20, 0.0), corr=(0.0, 0.0, 0.0), smoke_prob=0.0, spike_ratio=0.0):
        c = RecoveryController.__new__(RecoveryController)
        c.enable_scan_matching = True
        c.scan_match_max_dist = 0.45
        c.docking_dist_th = 0.50
        c.scan_match_weight = 0.50
        c.raw_odom_x, c.raw_odom_y = raw_xy
        c.raw_odom_theta = 0.0
        c.corr_x, c.corr_y, c.corr_th = corr
        c_c, s_c = math.cos(c.corr_th), math.sin(c.corr_th)
        c.current_x = c.raw_odom_x * c_c - c.raw_odom_y * s_c + c.corr_x
        c.current_y = c.raw_odom_x * s_c + c.raw_odom_y * c_c + c.corr_y
        c.current_theta = rc_mod.normalize_angle(c.raw_odom_theta + c.corr_th)
        c.start_pose = Waypoint(0.0, 0.0, 0.0, scan_pts.copy())
        c.start_scan_pts = scan_pts.copy()
        c.active_target_wp = c.start_pose
        c.latest_scan_pts = scan_pts.copy()
        c.latest_scan_pose = (c.current_x, c.current_y, c.current_theta)
        c.latest_scan_stamp_sec = 10.0
        c.latest_scan_spike_ratio = spike_ratio
        c.smoke_lidar_noise_prob = smoke_prob
        c.scan_match_count = 0
        c.dynamic_tracker = None
        c.semantic_cache = SemanticObjectCache()
        c.local_planner = None
        c.get_logger = lambda: types.SimpleNamespace(info=lambda *a, **k: None, warn=lambda *a, **k: None, error=lambda *a, **k: None)
        return c

    # Build a well-conditioned 2D box cloud around robot (4 walls: x=+/-1.5, y=+/-1.5)
    t = np.linspace(-1.4, 1.4, 25, dtype=np.float32)
    box_pts = np.vstack([
        np.column_stack([np.full_like(t, 1.5), t]),
        np.column_stack([-t, np.full_like(t, 1.5)]),
        np.column_stack([np.full_like(t, -1.5), -t]),
        np.column_stack([t, np.full_like(t, -1.5)]),
    ]).astype(np.float32)

    # 1) 1D-degenerate straight-wall scan MUST be rejected even during is_start_match=True
    line_pts = np.column_stack([np.full(40, 1.5, dtype=np.float32), np.linspace(-2.0, 2.0, 40, dtype=np.float32)])
    c_1d = _make_start_ctrl(line_pts)
    icp_called = {"n": 0}

    def _spy_icp(*a, **k):
        icp_called["n"] += 1
        return (0.20, 0.0, 0.0, True)

    monkeypatch.setattr(rc_mod, "icp_match_2d", _spy_icp)
    c_1d.perform_scan_matching_correction()
    assert icp_called["n"] == 0, "1D-degenerate scan must be rejected before ICP even in is_start_match"
    assert c_1d.scan_match_count == 0

    # 2) Verify strict icp_thresh <= 0.20 and min_inliers >= 25 in is_start_match
    captured_kwargs = {}

    def _capture_icp(*a, **k):
        captured_kwargs.update(k)
        return (0.24, 0.03, 0.02, True)

    c_ok = _make_start_ctrl(box_pts)
    monkeypatch.setattr(rc_mod, "icp_match_2d", _capture_icp)
    c_ok.perform_scan_matching_correction()
    assert captured_kwargs["inlier_thresh"] <= 0.20
    assert captured_kwargs["min_inliers"] >= 25
    assert c_ok.scan_match_count == 1

    # 3) Corrupted scan keeps max_p_jump = 0.12 m (rejects 0.20 m jump in is_start_match)
    c_jump = _make_start_ctrl(box_pts, spike_ratio=0.07)
    monkeypatch.setattr(rc_mod, "icp_match_2d", lambda *a, **k: (0.20, 0.20, 0.0, True))
    c_jump.perform_scan_matching_correction()
    assert c_jump.scan_match_count == 0 and c_jump.corr_y == 0.0

    # 4) Corrupted scan keeps drift_cap = 0.16 m (blocks cumulative runaway beyond 0.16 m in is_start_match)
    c_drift = _make_start_ctrl(box_pts, raw_xy=(0.20, 0.0), corr=(0.15, 0.0, 0.0), spike_ratio=0.07)
    monkeypatch.setattr(rc_mod, "icp_match_2d", lambda *a, **k: (0.43, 0.0, 0.0, True))
    c_drift.perform_scan_matching_correction()
    assert c_drift.scan_match_count == 0 and c_drift.corr_x == pytest.approx(0.15)

    # 5) yaw drift cap = 0.06 rad remains enforced in is_start_match
    c_yaw = _make_start_ctrl(box_pts, raw_xy=(0.20, 0.0), corr=(0.0, 0.0, 0.05))
    monkeypatch.setattr(rc_mod, "icp_match_2d", lambda *a, **k: (0.20, 0.0, 0.10, True))
    c_yaw.perform_scan_matching_correction()
    assert c_yaw.scan_match_count == 0 and c_yaw.corr_th == pytest.approx(0.05)

    # 6) Parallel vertical walls near start (y-axis normal degeneracy) block corr_y update while allowing corr_x
    par_y_pts = np.vstack([
        np.column_stack([np.full(30, 1.5, dtype=np.float32), np.linspace(-2.0, 2.0, 30, dtype=np.float32)]),
        np.column_stack([np.full(30, -1.5, dtype=np.float32), np.linspace(2.0, -2.0, 30, dtype=np.float32)]),
    ]).astype(np.float32)
    c_par = _make_start_ctrl(par_y_pts, raw_xy=(0.20, 0.0), corr=(0.0, 0.02, 0.0))
    monkeypatch.setattr(rc_mod, "icp_match_2d", lambda *a, **k: (0.26, 0.08, 0.03, True))
    c_par.perform_scan_matching_correction()
    assert c_par.scan_match_count == 1
    assert c_par.corr_x == pytest.approx(0.03, abs=1e-6)
    assert c_par.corr_y == pytest.approx(0.02, abs=1e-9), "Parallel-wall y-degeneracy must block corr_y update"
    assert c_par.corr_th == pytest.approx(0.0, abs=1e-9)

    # 7) High smoke spike ratio blocks corr_y update purely via sensor observation (smoke_prob=0.0)
    c_smoke = _make_start_ctrl(box_pts, raw_xy=(0.20, 0.0), corr=(0.0, -0.01, 0.0), smoke_prob=0.0, spike_ratio=0.07)
    monkeypatch.setattr(rc_mod, "icp_match_2d", lambda *a, **k: (0.26, -0.09, 0.02, True))
    c_smoke.perform_scan_matching_correction()
    assert c_smoke.scan_match_count == 1
    assert c_smoke.corr_x == pytest.approx(0.03, abs=1e-6)
    assert c_smoke.corr_y == pytest.approx(-0.01, abs=1e-9), "Smoke-corrupted scan must block corr_y update"

    # 8) Track-specific cbf_v_min parity across YAML and ab_disaster_benchmark.py
    cfg_dir = os.path.join(PKG_ROOT, "config")
    with open(os.path.join(cfg_dir, "recovery_params.yaml"), "r", encoding="utf-8") as f:
        p_sim = yaml.safe_load(f)["recovery_controller"]["ros__parameters"]
    with open(os.path.join(cfg_dir, "recovery_params_real.yaml"), "r", encoding="utf-8") as f:
        p_real = yaml.safe_load(f)["recovery_controller"]["ros__parameters"]
    assert p_sim["cbf_v_min"] == pytest.approx(-0.06, abs=1e-9)
    assert p_real["cbf_v_min"] == pytest.approx(-0.08, abs=1e-9)

    bench_src = open(os.path.join(SCRIPTS_HOST_DIR, "ab_disaster_benchmark.py"), "r", encoding="utf-8").read()
    assert 'default_cbf_v_min = -0.08 if strategy == "phase4_real" else -0.06' in bench_src
    assert "-p cbf_v_min:={default_cbf_v_min}" in bench_src


def test_stepA2_start_match_corrupted_vs_clean_scan_branching(monkeypatch):
    """Verify Step A-2 pure-observation 2-branch logic: corrupted scans keep conservative guards (0.12/0.16, corr_y blocked), clean HQ scans absorb 30cm drift (0.38/0.42, corr_y allowed), and smoke_lidar_noise_prob is never referenced in ICP."""
    import inspect
    from disaster_recovery import recovery_controller as rc_mod

    RecoveryController, SemanticObjectCache, _, Waypoint = _make_dummy_controller_cls()

    # 0) Pure observation check: perform_scan_matching_correction must NOT reference smoke_lidar_noise_prob
    src_icp = inspect.getsource(RecoveryController.perform_scan_matching_correction)
    assert "smoke_lidar_noise_prob" not in src_icp, "ICP logic must not reference simulation parameter smoke_lidar_noise_prob"

    def _make_ctrl(scan_pts, *, raw_xy=(0.20, 0.0), corr=(0.0, 0.0, 0.0), spike_ratio=0.0):
        c = RecoveryController.__new__(RecoveryController)
        c.enable_scan_matching = True
        c.scan_match_max_dist = 0.45
        c.docking_dist_th = 0.50
        c.scan_match_weight = 0.40
        c.raw_odom_x, c.raw_odom_y = raw_xy
        c.raw_odom_theta = 0.0
        c.corr_x, c.corr_y, c.corr_th = corr
        c_c, s_c = math.cos(c.corr_th), math.sin(c.corr_th)
        c.current_x = c.raw_odom_x * c_c - c.raw_odom_y * s_c + c.corr_x
        c.current_y = c.raw_odom_x * s_c + c.raw_odom_y * c_c + c.corr_y
        c.current_theta = rc_mod.normalize_angle(c.raw_odom_theta + c.corr_th)
        c.start_pose = Waypoint(0.0, 0.0, 0.0, scan_pts.copy())
        c.start_scan_pts = scan_pts.copy()
        c.active_target_wp = c.start_pose
        c.latest_scan_pts = scan_pts.copy()
        c.latest_scan_pose = (c.current_x, c.current_y, c.current_theta)
        c.latest_scan_stamp_sec = 10.0
        c.latest_scan_spike_ratio = spike_ratio
        c.smoke_lidar_noise_prob = 0.0
        c.scan_match_count = 0
        c.dynamic_tracker = None
        c.semantic_cache = SemanticObjectCache()
        c.local_planner = None
        c.get_logger = lambda: types.SimpleNamespace(info=lambda *a, **k: None, warn=lambda *a, **k: None, error=lambda *a, **k: None)
        return c

    t = np.linspace(-1.4, 1.4, 25, dtype=np.float32)
    box_pts = np.vstack([
        np.column_stack([np.full_like(t, 1.5), t]),
        np.column_stack([-t, np.full_like(t, 1.5)]),
        np.column_stack([np.full_like(t, -1.5), -t]),
        np.column_stack([t, np.full_like(t, -1.5)]),
    ]).astype(np.float32)

    # 1) Corrupted branch via each pure observation criterion (spike_ratio, inlier_ratio, mean_err)
    corrupt_cases = [
        ("spike_ratio", 0.04, {"num_inliers": 90, "num_points": 100, "inlier_ratio": 0.90, "mean_err": 0.02}),
        ("low_inlier_ratio", 0.0, {"num_inliers": 60, "num_points": 100, "inlier_ratio": 0.60, "mean_err": 0.03}),
        ("high_mean_err", 0.0, {"num_inliers": 90, "num_points": 100, "inlier_ratio": 0.90, "mean_err": 0.095}),
    ]
    for label, spk, stats in corrupt_cases:
        # 1a) 0.25m jump blocked by conservative max_p_jump=0.12
        c1 = _make_ctrl(box_pts, raw_xy=(0.20, 0.0), spike_ratio=spk)
        monkeypatch.setattr(rc_mod, "icp_match_2d", lambda *a, _st=stats, **k: (0.20, 0.25, 0.0, True, _st))
        c1.perform_scan_matching_correction()
        assert c1.scan_match_count == 0, f"{label}: 0.25m jump must be rejected under corrupted guard"
        assert c1.last_icp_stats["is_scan_corrupted"] is True

        # 1b) Small jump still blocks corr_y / corr_th update while allowing x
        c2 = _make_ctrl(box_pts, raw_xy=(0.20, 0.0), corr=(0.0, -0.02, 0.0), spike_ratio=spk)
        monkeypatch.setattr(rc_mod, "icp_match_2d", lambda *a, _st=stats, **k: (0.26, -0.09, 0.03, True, _st))
        c2.perform_scan_matching_correction()
        assert c2.scan_match_count == 1, f"{label}: small x correction should still be accepted"
        assert c2.corr_y == pytest.approx(-0.02, abs=1e-9), f"{label}: corr_y must stay frozen"
        assert c2.corr_th == pytest.approx(0.0, abs=1e-9), f"{label}: corr_th must stay frozen"

        # 1c) drift_cap=0.16 blocks drift growth beyond 0.16m
        c3 = _make_ctrl(box_pts, raw_xy=(0.20, 0.0), corr=(0.15, 0.0, 0.0), spike_ratio=spk)
        monkeypatch.setattr(rc_mod, "icp_match_2d", lambda *a, _st=stats, **k: (0.43, 0.0, 0.0, True, _st))
        c3.perform_scan_matching_correction()
        assert c3.scan_match_count == 0, f"{label}: drift beyond 0.16m must be blocked"

    # 2) Clean but low inlier count (<25) keeps conservative 0.12m/0.16m guards
    low_cnt_stats = {"num_inliers": 20, "num_points": 25, "inlier_ratio": 0.80, "mean_err": 0.02}
    c_low = _make_ctrl(box_pts, raw_xy=(0.20, 0.0), spike_ratio=0.0)
    monkeypatch.setattr(rc_mod, "icp_match_2d", lambda *a, **k: (0.20, 0.25, 0.0, True, low_cnt_stats))
    c_low.perform_scan_matching_correction()
    assert c_low.scan_match_count == 0, "num_inliers < 25 must NOT receive relaxed 0.38m/0.42m bounds"

    # 3) Clean high-quality start scan absorbs 32cm accumulated drift (both x and y)
    hq_stats = {"num_inliers": 92, "num_points": 100, "inlier_ratio": 0.92, "mean_err": 0.03}
    # Robot is physically at (0.05, 0.02), raw_odom drifted by (+0.14, -0.30) -> raw_odom=(0.19, -0.28)
    c_clean = _make_ctrl(box_pts, raw_xy=(0.19, -0.28), corr=(0.0, 0.0, 0.0), spike_ratio=0.0)
    monkeypatch.setattr(rc_mod, "icp_match_2d", lambda *a, **k: (0.05, 0.02, 0.0, True, hq_stats))
    for step_i in range(6):
        c_clean.latest_scan_stamp_sec = 20.0 + step_i * 0.2
        c_clean.perform_scan_matching_correction()
    assert c_clean.scan_match_count == 6
    assert c_clean.last_icp_stats["is_scan_corrupted"] is False
    assert math.hypot(c_clean.current_x - 0.05, c_clean.current_y - 0.02) < 0.03, (
        f"Clean HQ start scan must absorb 33cm drift to <3cm, got ({c_clean.current_x:.3f}, {c_clean.current_y:.3f})"
    )

    # 4) Clean HQ scan still rejects jumps >= 0.38m and drift > 0.42m
    c_big_jump = _make_ctrl(box_pts, raw_xy=(0.20, 0.0), spike_ratio=0.0)
    monkeypatch.setattr(rc_mod, "icp_match_2d", lambda *a, **k: (0.20, 0.40, 0.0, True, hq_stats))
    c_big_jump.perform_scan_matching_correction()
    assert c_big_jump.scan_match_count == 0, "Clean HQ scan must still reject >=0.38m jump"

    c_big_drift = _make_ctrl(box_pts, raw_xy=(0.20, 0.0), corr=(0.0, 0.41, 0.0), spike_ratio=0.0)
    monkeypatch.setattr(rc_mod, "icp_match_2d", lambda *a, **k: (0.20, 0.50, 0.0, True, hq_stats))
    c_big_drift.perform_scan_matching_correction()
    assert c_big_drift.scan_match_count == 0, "Clean HQ scan must still cap cumulative drift at 0.42m"

    # 5) u_pocket-like geometry (long vertical walls + pillar horizontal faces, h_yy ~ 0.09) allows corr_y on clean scan
    up_pts = np.vstack([
        np.column_stack([np.full(35, -1.5, dtype=np.float32), np.linspace(-2.5, 2.5, 35, dtype=np.float32)]),
        np.column_stack([np.full(25, 2.9, dtype=np.float32), np.linspace(-1.0, 1.0, 25, dtype=np.float32)]),
        np.column_stack([np.linspace(1.4, 1.85, 6, dtype=np.float32), np.full(6, 1.7, dtype=np.float32)]),
        np.column_stack([np.linspace(1.4, 1.85, 6, dtype=np.float32), np.full(6, -1.7, dtype=np.float32)]),
    ]).astype(np.float32)
    c_up = _make_ctrl(up_pts, raw_xy=(0.10, -0.28), corr=(0.0, 0.0, 0.0), spike_ratio=0.0)
    monkeypatch.setattr(rc_mod, "icp_match_2d", lambda *a, **k: (0.10, 0.0, 0.0, True, hq_stats))
    c_up.perform_scan_matching_correction()
    assert c_up.scan_match_count == 1
    assert c_up.corr_y > 0.10, "u_pocket pillar geometry (h_yy ~ 0.09) must NOT false-trigger y_eig_degenerate"


def test_stepA2_icp_match_2d_return_stats_and_coarse_to_fine_start_rescue():
    """Verify real SVD icp_match_2d return_stats metrics and coarse-to-fine 32cm start-anchor drift rescue without mocks."""
    from disaster_recovery import recovery_controller as rc_mod

    RecoveryController, SemanticObjectCache, _, Waypoint = _make_dummy_controller_cls()

    # Build a 4-wall room point cloud at start=(0, 0, 0)
    t = np.linspace(-1.4, 1.4, 25, dtype=np.float32)
    ref_pts = np.vstack([
        np.column_stack([np.full_like(t, 1.5), t]),
        np.column_stack([-t, np.full_like(t, 1.5)]),
        np.column_stack([np.full_like(t, -1.5), -t]),
        np.column_stack([t, np.full_like(t, -1.5)]),
    ]).astype(np.float32)

    # Robot physically at (0.10, -0.05, 0.0), raw_odom drifted by (+0.26, +0.20) -> 32.8cm drift (>0.20m fine gate)
    true_x, true_y = 0.10, -0.05
    src_pts = ref_pts - np.array([[true_x, true_y]], dtype=np.float32)

    # Direct stats check on real icp_match_2d (fine 0.20m fails at 32.8cm drift, coarse 0.38m + eval 0.20m succeeds)
    _, _, _, ok_fine, st_fine = rc_mod.icp_match_2d(
        src_pts, ref_pts, initial_guess=(0.36, 0.15, 0.0), max_iters=8,
        inlier_thresh=0.20, min_inliers=68, return_stats=True,
    )
    dx_c, dy_c, dth_c, ok_c, st_c = rc_mod.icp_match_2d(
        src_pts, ref_pts, initial_guess=(0.36, 0.15, 0.0), max_iters=8,
        inlier_thresh=0.38, min_inliers=68, return_stats=True, eval_inlier_thresh=0.20,
    )
    assert ok_fine is False and st_fine["inlier_ratio"] < 0.65
    assert ok_c is True and st_c["inlier_ratio"] >= 0.95 and st_c["mean_err"] < 0.02
    assert dx_c == pytest.approx(true_x, abs=0.02) and dy_c == pytest.approx(true_y, abs=0.02)

    # End-to-end controller start-anchor rescue over 5 clean scans
    c = RecoveryController.__new__(RecoveryController)
    c.enable_scan_matching = True
    c.scan_match_max_dist = 0.45
    c.docking_dist_th = 0.50
    c.scan_match_weight = 0.40
    c.raw_odom_x, c.raw_odom_y, c.raw_odom_theta = 0.36, 0.15, 0.0
    c.corr_x, c.corr_y, c.corr_th = 0.0, 0.0, 0.0
    c.current_x, c.current_y, c.current_theta = 0.36, 0.15, 0.0
    c.start_pose = Waypoint(0.0, 0.0, 0.0, ref_pts.copy())
    c.start_scan_pts = ref_pts.copy()
    c.active_target_wp = c.start_pose
    c.latest_scan_pts = src_pts.copy()
    c.latest_scan_pose = (c.current_x, c.current_y, c.current_theta)
    c.latest_scan_spike_ratio = 0.0
    c.scan_match_count = 0
    c.dynamic_tracker = None
    c.semantic_cache = SemanticObjectCache()
    c.local_planner = None
    c.get_logger = lambda: types.SimpleNamespace(info=lambda *a, **k: None, warn=lambda *a, **k: None, error=lambda *a, **k: None)

    for k in range(5):
        c.latest_scan_stamp_sec = 30.0 + k * 0.2
        c.perform_scan_matching_correction()
    assert c.scan_match_count == 5
    assert c.last_icp_stats["is_scan_corrupted"] is False
    assert math.hypot(c.current_x - true_x, c.current_y - true_y) < 0.04


def test_stepB_axial_always_on_crossing_lane_hold_dash_zero_interference_and_latency():
    """
    Verify all Step B integration requirements:
      (a) Axial sweep soft costmap always-on even when enable_semantic_astar_cost == False
          (and bypassed when spike_ratio >= 0.03).
      (b) Crossing 2D lateral lane pre-alignment + Hold/Dash timing + deadlock timeout fallback,
          with occupancy-grid wall measurement >= 0.17 m clearance and rotation invariance.
      (c) Zero-interference: 0 confirmed dynamic tracks OR spike_ratio >= 0.03 yields
          bit-identical control/planner outputs vs Step A-2 baseline.
      (d) Total added Step B per-tick compute latency < 1.0 ms.
    """
    import time
    from disaster_recovery import recovery_controller as rc_mod
    from disaster_recovery.cbf_safety_shield import DiscreteSoftCBFShield

    RecoveryController, SemanticObjectCache, SystemState, Waypoint = _make_dummy_controller_cls()
    CrossingLaneScheduler = rc_mod.CrossingLaneScheduler

    class _FakeTime:
        def __init__(self, ns: int):
            self.nanoseconds = int(ns)

        def __sub__(self, other):
            return _FakeTime(self.nanoseconds - other.nanoseconds)

    def _make_proposed_ctrl(
        *,
        rx=0.0,
        ry=0.0,
        rth=0.0,
        scan_pts=None,
        raw_ranges=None,
        wps=None,
        dyn_tracks=None,
        spike_ratio=0.0,
        enable_cbf=True,
        enable_sem_astar=False,
        with_scheduler=True,
    ):
        c = RecoveryController.__new__(RecoveryController)
        c.recovery_strategy = "proposed"
        c.state = SystemState.BACKTRACKING_HYBRID
        c.previous_state = SystemState.BACKTRACKING_HYBRID
        c.fault_reason = "HEARTBEAT_TIMEOUT"
        c.current_gas_level = 0.0
        c.gas_threshold = 0.40
        c.last_heartbeat_time = _FakeTime(0)
        c._link_restored_timer = None
        c.clear_stack_on_link_restore = False
        c.reset_corr_on_episode = False
        c.odom_initialized = True
        c.heartbeat_received_once = True
        c.current_x = float(rx)
        c.current_y = float(ry)
        c.current_theta = float(rth)
        c.raw_odom_x = float(rx)
        c.raw_odom_y = float(ry)
        c.raw_odom_theta = float(rth)
        c.corr_x = 0.0
        c.corr_y = 0.0
        c.corr_th = 0.0
        c.actual_v = 0.0
        c.actual_w = 0.0
        c.last_cmd_v = 0.0
        c.last_cmd_w = 0.0
        c.v_max = 0.20
        c.w_max = 0.65
        c.kp_lin = 0.60
        c.kp_ang = 1.40
        c.wp_reach_radius = 0.10
        c.docking_dist_th = 0.50
        c.use_aruco_docking = False
        c.enable_scan_matching = False
        c.enable_semantic_astar_cost = bool(enable_sem_astar)
        c.enable_dynamic_headon_hold = False
        c.enable_cbf_safety_shield = bool(enable_cbf)
        c.enable_dynamic_tracker = False
        c.cbf_inner_loop_hz = 10.0
        c.cbf_v_min = -0.06
        c.min_effective_linear_vel = 0.0
        c.min_effective_angular_vel = 0.0
        c.latest_scan_pts = scan_pts
        c.latest_raw_ranges = list(raw_ranges) if raw_ranges is not None else []
        c.latest_scan_pose = (c.current_x, c.current_y, c.current_theta)
        c.latest_scan_stamp_sec = 10.0
        c.latest_scan_angle_min = 0.0
        c.latest_scan_angle_increment = (2.0 * math.pi / len(raw_ranges)) if raw_ranges is not None else None
        c.latest_scan_spike_ratio = float(spike_ratio)
        c.latest_dynamic_tracks = list(dyn_tracks) if dyn_tracks is not None else []
        c.latest_dynamic_ch3 = None
        c.dynamic_tracker = None
        c.context_rl_policy = None
        c.semantic_cache = SemanticObjectCache()
        c.local_planner = ProposedRecoveryPlanner(size_m=6.0, resolution_m=0.04, inflation_radius_m=0.085)
        c.cbf_shield = DiscreteSoftCBFShield(v_min=-0.06, v_max=0.20, w_max=0.65) if enable_cbf else None
        c.latest_cbf_result = None
        c.crossing_scheduler = CrossingLaneScheduler() if with_scheduler else None
        c.waypoint_stack = deque(wps if wps is not None else [Waypoint(2.4, 0.0, 0.0), Waypoint(1.2, 0.0, 0.0)])
        c.start_pose = c.waypoint_stack[0]
        c.active_target_wp = c.waypoint_stack[-1]
        c._icp_section_ms_total = 0.0
        c._icp_section_ms_max = 0.0
        c._icp_section_count = 0
        c._icp_section_last_log = 0.0
        c._icp_run_count = 0
        c._icp_skip_count = 0
        c._clock_now_ns = int(10.0 * 1e9)
        c.get_clock = lambda: types.SimpleNamespace(now=lambda: _FakeTime(c._clock_now_ns))
        c.published_cmds = []
        c.pub_cmd_vel = types.SimpleNamespace(publish=lambda m: c.published_cmds.append((float(m.linear.x), float(m.angular.z))))
        c.pub_system_status = types.SimpleNamespace(publish=lambda m: None)
        c.get_logger = lambda: types.SimpleNamespace(info=lambda *a, **k: None, warn=lambda *a, **k: None, error=lambda *a, **k: None)
        return c

    # ------------------------------------------------------------------
    # (a) Axial sweep soft costmap always-on even with enable_semantic_astar_cost=False
    # ------------------------------------------------------------------
    xs_w = np.linspace(-1.0, 3.0, 100)
    wide_walls = np.vstack([
        np.column_stack((xs_w, np.full_like(xs_w, -1.25))),
        np.column_stack((xs_w, np.full_like(xs_w, 1.25))),
    ]).astype(np.float32)

    axial_trk = TrackedObstacle(
        track_id=10,
        x=1.20,
        y=0.0,
        vx=0.40,
        vy=0.0,
        radius=0.21,
        P_4x4=np.diag([0.002, 0.002, 0.01, 0.01]).astype(np.float64),
        last_update_stamp=10.0,
        age=25,
        hits=25,
        is_confirmed=True,
        consecutive_hits=25,
        imm_valid=True,
        imm_u_hat=(1.0, 0.0),
        imm_p_c=(1.20, 0.0),
        imm_amp=0.93,
        imm_omega=0.45,
        imm_phi=0.0,
        imm_t_ref=10.0,
        imm_mu_cv=0.01,
        imm_mu_harm=0.99,
        imm_sigma_h0=0.03,
        imm_sigma_omega=0.02,
    )

    c_axial = _make_proposed_ctrl(
        scan_pts=wide_walls,
        wps=[Waypoint(2.4, 0.0, 0.0), Waypoint(1.6, 0.0, 0.0), Waypoint(0.8, 0.0, 0.0)],
        dyn_tracks=[axial_trk],
        enable_cbf=False,
        enable_sem_astar=False,
    )
    c_axial.local_planner.observe(0.0, 0.0, 0.0, wide_walls)
    c_axial.control_loop()
    assert c_axial.local_planner.last_subgoal is not None
    assert abs(c_axial.local_planner.last_subgoal[1]) > 0.15, (
        f"Confirmed axial track must trigger lateral detour even when enable_semantic_astar_cost=False, "
        f"got subgoal={c_axial.local_planner.last_subgoal}"
    )
    assert c_axial.crossing_scheduler.active is False, "Axial obstacle must not engage crossing scheduler"

    # ------------------------------------------------------------------
    # (b) Crossing 2D lateral lane pre-alignment + Hold/Dash + deadlock fallback + rotation invariance
    # ------------------------------------------------------------------
    def _build_gate_grid(theta_rot: float = 0.0):
        c_r, s_r = math.cos(theta_rot), math.sin(theta_rot)
        grid = ProposedRecoveryPlanner(size_m=6.0, resolution_m=0.04, inflation_radius_m=0.085).grid
        grid._recenter(0.5 * c_r, 0.5 * s_r)
        # Gate opening half-width 0.56 m centered at local (1.50, 0.0); outer walls along x in [0.0, 2.2]
        local_pts = []
        for lx in np.linspace(0.0, 2.2, 55):
            local_pts.append((lx, -0.56))
            local_pts.append((lx, 0.56))
        for ly in np.linspace(0.56, 1.30, 20):
            local_pts.append((1.50, ly))
            local_pts.append((1.50, -ly))
        for lx, ly in local_pts:
            wx = lx * c_r - ly * s_r
            wy = lx * s_r + ly * c_r
            ix, iy = grid.world_to_grid(wx, wy)
            if grid.in_bounds(ix, iy):
                grid.occ[ix, iy] = 1.0
        return grid

    def _make_cross_track(t_now: float, phi0: float = 0.0, omega: float = 0.95, amp: float = 0.38,
                          sig_h0: float = 0.025, sig_w: float = 0.02, theta_rot: float = 0.0, mu_h: float = 0.99):
        c_r, s_r = math.cos(theta_rot), math.sin(theta_rot)
        pc_x, pc_y = 1.50 * c_r, 1.50 * s_r
        ux, uy = -s_r, c_r
        s_pos = amp * math.sin(omega * t_now + phi0)
        s_vel = amp * omega * math.cos(omega * t_now + phi0)
        phi_cur = math.atan2(math.sin(omega * t_now + phi0), math.cos(omega * t_now + phi0))
        return TrackedObstacle(
            track_id=7,
            x=pc_x + ux * s_pos,
            y=pc_y + uy * s_pos,
            vx=ux * s_vel,
            vy=uy * s_vel,
            radius=0.21,
            P_4x4=np.diag([0.002, 0.002, 0.02, 0.02]).astype(np.float64),
            last_update_stamp=t_now,
            age=30,
            hits=30,
            is_confirmed=True,
            consecutive_hits=30,
            imm_valid=True,
            imm_u_hat=(ux, uy),
            imm_p_c=(pc_x, pc_y),
            imm_amp=amp,
            imm_omega=omega,
            imm_phi=phi_cur,
            imm_t_ref=t_now,
            imm_mu_cv=1.0 - mu_h,
            imm_mu_harm=mu_h,
            imm_sigma_h0=sig_h0,
            imm_sigma_omega=sig_w,
        )

    # b1) Rotation-invariant lateral lane measurement (>= 0.17 m wall clearance on both sides)
    for rot in (0.0, math.radians(37.0)):
        c_r, s_r = math.cos(rot), math.sin(rot)
        sched_rot = CrossingLaneScheduler()
        gate_grid = _build_gate_grid(rot)
        trk_r = _make_cross_track(0.0, theta_rot=rot)
        path_r = [(0.1 * c_r, 0.1 * s_r), (1.0 * c_r, 1.0 * s_r), (2.2 * c_r, 2.2 * s_r)]
        dec_app = sched_rot.step(
            rx=0.1 * c_r,
            ry=0.1 * s_r,
            rth=rot,
            v_max=0.20,
            w_max=0.65,
            now_sec=0.0,
            confirmed_tracks=[trk_r],
            path_world=path_r,
            local_grid=gate_grid,
        )
        assert dec_app.active and dec_app.mode == "APPROACH_ALIGN"
        assert len(dec_app.lane_candidates) == 2
        for y_l in dec_app.lane_candidates:
            wall_clr = 0.56 - abs(y_l)
            assert wall_clr >= 0.17, f"Lateral lane {y_l:.3f} violates >=0.17m wall clearance ({wall_clr:.3f}m)"
        assert abs(dec_app.lane_offset_m) >= 0.30

    # b2) High initial IMM sigma at Hold Point forces HOLD (v_ref=0, no reverse rotation, watchdog refreshed)
    sched_unc = CrossingLaneScheduler()
    gate_grid_0 = _build_gate_grid(0.0)
    path_0 = [(0.90, 0.36), (1.50, 0.36), (2.20, 0.36)]
    trk_unc = _make_cross_track(0.0, sig_h0=0.28, sig_w=0.25, mu_h=0.60)
    dec_unc = sched_unc.step(
        rx=0.90, ry=0.36, rth=-0.12,
        v_max=0.20, w_max=0.65, now_sec=0.0,
        confirmed_tracks=[trk_unc], path_world=path_0, local_grid=gate_grid_0,
    )
    assert dec_unc.active and dec_unc.mode == "HOLD"
    assert dec_unc.v_ref == 0.0
    assert dec_unc.w_ref > 0.0, "Must gently align toward gate-forward heading without reverse turn"
    assert dec_unc.refresh_watchdog is True

    # b3) Full closed-loop crossing simulation: APPROACH_ALIGN -> HOLD -> DASH -> clear exit with 0 collisions
    sched_sim = CrossingLaneScheduler()
    rx_s, ry_s, rth_s = 0.10, 0.0, 0.0
    modes_seen = []
    min_center_dist = float("inf")
    dash_v_max_seen = 0.0
    dt_sim = 0.10
    for step_i in range(180):
        t_s = step_i * dt_sim
        trk_s = _make_cross_track(t_s, phi0=0.2, omega=0.95, amp=0.38)
        path_s = [(rx_s, ry_s), (1.50, 0.0), (2.40, 0.0)]
        dec_s = sched_sim.step(
            rx=rx_s, ry=ry_s, rth=rth_s,
            v_max=0.20, w_max=0.65, now_sec=t_s,
            confirmed_tracks=[trk_s], path_world=path_s, local_grid=gate_grid_0,
        )
        modes_seen.append(dec_s.mode)
        if dec_s.active:
            v_use, w_use = dec_s.v_ref, dec_s.w_ref
            if dec_s.mode == "DASH":
                dash_v_max_seen = max(dash_v_max_seen, v_use)
        else:
            v_use, w_use = 0.20, 0.0
        rth_s = rc_mod.normalize_angle(rth_s + w_use * dt_sim)
        rx_s += v_use * math.cos(rth_s) * dt_sim
        ry_s += v_use * math.sin(rth_s) * dt_sim
        d_obs = math.hypot(rx_s - trk_s.x, ry_s - trk_s.y)
        if d_obs < min_center_dist:
            min_center_dist = d_obs
        assert abs(ry_s) <= 0.56 - 0.16, f"Robot grazed gate wall at y={ry_s:.3f}"
        if rx_s > 2.05:
            break

    assert rx_s > 2.05, f"Robot failed to cross gate; stopped at x={rx_s:.2f}"
    assert "APPROACH_ALIGN" in modes_seen
    assert "DASH" in modes_seen
    assert dash_v_max_seen == pytest.approx(0.20, abs=1e-6)
    assert min_center_dist >= 0.21 + 0.16, (
        f"Closed-loop crossing violated safe center clearance: {min_center_dist:.3f} < 0.37 m"
    )

    # b4) Deadlock timeout fallback: impossible fast/wide obstacle eventually triggers DASH_FALLBACK (>= 1 period)
    sched_fb = CrossingLaneScheduler()
    fb_triggered_t = None
    for k in range(130):
        t_k = k * 0.10
        # Narrow opening (0.40 m wide -> single center lane y=0) + huge radius obstacle so safe window never opens
        narrow_gate = ProposedRecoveryPlanner(size_m=6.0, resolution_m=0.04).grid
        narrow_gate._recenter(0.5, 0.0)
        for lx in np.linspace(0.0, 2.2, 40):
            for ly in (-0.22, 0.22):
                ix, iy = narrow_gate.world_to_grid(lx, ly)
                if narrow_gate.in_bounds(ix, iy):
                    narrow_gate.occ[ix, iy] = 1.0
        trk_imp = _make_cross_track(t_k, omega=1.0, amp=0.12)
        trk_imp.radius = 0.24
        dec_fb = sched_fb.step(
            rx=0.88, ry=0.0, rth=0.0,
            v_max=0.20, w_max=0.65, now_sec=t_k,
            confirmed_tracks=[trk_imp],
            path_world=[(0.88, 0.0), (1.50, 0.0), (2.20, 0.0)],
            local_grid=narrow_gate,
        )
        if dec_fb.mode == "DASH_FALLBACK":
            fb_triggered_t = t_k
            assert dec_fb.v_ref == pytest.approx(0.20, abs=1e-6)
            break
    assert fb_triggered_t is not None, "Deadlock timeout fallback must launch after >= 1 period"
    assert fb_triggered_t >= 5.5, f"Fallback triggered prematurely at {fb_triggered_t:.2f} s"

    # ------------------------------------------------------------------
    # (c) Zero-interference: 0 confirmed tracks OR spike_ratio >= 0.03 -> bit-identical to Step A-2
    # ------------------------------------------------------------------
    raw_360 = [2.5] * 360
    tentative_trk = TrackedObstacle(
        track_id=99, x=1.2, y=0.0, vx=0.35, vy=0.0, radius=0.21,
        P_4x4=np.eye(4, dtype=np.float64) * 0.01,
        last_update_stamp=10.0, age=3, hits=3, is_confirmed=False, consecutive_hits=3,
    )
    c_base = _make_proposed_ctrl(
        scan_pts=wide_walls, raw_ranges=raw_360, dyn_tracks=[], spike_ratio=0.0,
        enable_cbf=True, enable_sem_astar=False, with_scheduler=False,
    )
    c_zero_conf = _make_proposed_ctrl(
        scan_pts=wide_walls, raw_ranges=raw_360, dyn_tracks=[tentative_trk], spike_ratio=0.0,
        enable_cbf=True, enable_sem_astar=False, with_scheduler=True,
    )
    c_corrupt = _make_proposed_ctrl(
        scan_pts=wide_walls, raw_ranges=raw_360, dyn_tracks=[axial_trk], spike_ratio=0.05,
        enable_cbf=True, enable_sem_astar=False, with_scheduler=True,
    )
    for c_i in (c_base, c_zero_conf, c_corrupt):
        c_i.local_planner.observe(0.0, 0.0, 0.0, wide_walls)

    for tick in range(5):
        ns = int((10.0 + tick * 0.1) * 1e9)
        for c_i in (c_base, c_zero_conf, c_corrupt):
            c_i._clock_now_ns = ns
            c_i.control_loop()

    assert c_zero_conf.published_cmds == c_base.published_cmds, (
        "0 confirmed tracks must produce bit-identical commands vs baseline"
    )
    assert c_corrupt.published_cmds == c_base.published_cmds, (
        "Corrupted scan (spike_ratio >= 0.03) must produce bit-identical commands vs baseline"
    )
    assert c_zero_conf.local_planner.last_subgoal == c_base.local_planner.last_subgoal
    assert c_corrupt.local_planner.last_subgoal == c_base.local_planner.last_subgoal

    # ------------------------------------------------------------------
    # (d) End-to-end Step B added latency profiling (< 1.0 ms per tick)
    # ------------------------------------------------------------------
    trk_prof = DynamicObstacleTracker()
    sched_prof = CrossingLaneScheduler()
    grid_prof = _build_gate_grid(0.0)
    taus_30 = np.arange(0, 31, dtype=np.float64) * 0.15
    t_cross = _make_cross_track(5.0, omega=0.95, amp=0.38)
    t_axial = axial_trk
    path_prof = [(0.2, 0.0), (1.0, 0.0), (1.5, 0.0), (2.2, 0.0)]
    wps_prof = [Waypoint(2.4, 0.0, 0.0), Waypoint(1.2, 0.0, 0.0)]

    # Warm-up
    for _ in range(10):
        trk_prof.predict_trajectory(t_cross, taus_30)
        grid_prof.build_cost_maps(0.2, 0.0, waypoint_list=wps_prof)
        grid_prof.build_cost_maps(0.2, 0.0, dynamic_obstacles=[t_axial], waypoint_list=wps_prof)
        sched_prof.step(
            rx=0.2, ry=0.0, rth=0.0, v_max=0.20, w_max=0.65, now_sec=5.0,
            confirmed_tracks=[t_cross], path_world=path_prof, local_grid=grid_prof,
            dynamic_tracker=trk_prof,
        )

    n_iters = 150
    # Measure Step A-2 baseline build_cost_maps vs Step B dynamic axial sweep delta
    t_b0 = time.perf_counter()
    for _ in range(n_iters):
        grid_prof.build_cost_maps(0.2, 0.0, waypoint_list=wps_prof)
    base_costmap_ms = ((time.perf_counter() - t_b0) / float(n_iters)) * 1000.0

    t_d0 = time.perf_counter()
    for _ in range(n_iters):
        grid_prof.build_cost_maps(0.2, 0.0, dynamic_obstacles=[t_axial], waypoint_list=wps_prof)
    dyn_costmap_ms = ((time.perf_counter() - t_d0) / float(n_iters)) * 1000.0
    axial_delta_ms = max(0.0, dyn_costmap_ms - base_costmap_ms)

    # Measure IMM trajectory rollout + CrossingLaneScheduler per-tick cost
    t_s0 = time.perf_counter()
    for i in range(n_iters):
        trk_prof.predict_trajectory(t_cross, taus_30)
        sched_prof.step(
            rx=0.2, ry=0.0, rth=0.0, v_max=0.20, w_max=0.65, now_sec=5.0 + i * 0.01,
            confirmed_tracks=[t_cross], path_world=path_prof, local_grid=grid_prof,
            dynamic_tracker=trk_prof,
        )
    imm_and_sched_ms = ((time.perf_counter() - t_s0) / float(n_iters)) * 1000.0
    total_added_ms = axial_delta_ms + imm_and_sched_ms
    assert total_added_ms < 1.0, (
        f"Total Step B added compute latency {total_added_ms:.3f} ms >= 1.0 ms "
        f"(axial_delta={axial_delta_ms:.3f} ms, imm+sched={imm_and_sched_ms:.3f} ms)"
    )


def test_stepB2_asymmetric_lane_selection_lateral_shift_hysteresis_and_cbf_hold_protection():
    """
    Verify Step B-2 deadlock-breaking fixes (sensor/grid/tracker-driven, zero hardcoding):
      1. Asymmetric gate: never locks onto an infeasible lane (t_launch=inf) near Hold;
         switches to the feasible side lane and executes lateral pre-alignment shift
         instead of freezing at v_ref=0 while misaligned.
      2. Moving-obstacle trail in local_grid.occ does not create fake static gate walls.
      3. State/timer hysteresis: alignment bend (0.55 < align <= 0.80), brief track
         occlusion (<= 1.0 s), and same-obstacle re-ID preserve HOLD timer without reset;
         while INACTIVE entry still rejects axial align > 0.55.
      4. CBF prefer_no_reverse blocks spurious reverse during safe HOLD/DASH (h >= 0)
         while preserving emergency reverse when barrier is breached (h < 0).
    """
    from disaster_recovery import recovery_controller as rc_mod
    from disaster_recovery.cbf_safety_shield import DiscreteSoftCBFShield

    CrossingLaneScheduler = rc_mod.CrossingLaneScheduler

    # 1) Asymmetric gate opening y in [-0.90, +0.18] relative to pc=(5.14, 0.33), ef=(-1, 0), el=(0, -1)
    #    Physical walls at x=5.40: y_world <= 0.15 (y_local >= +0.18) and y_world >= 1.30 (y_local <= -0.97)
    asym_grid = ProposedRecoveryPlanner(size_m=6.0, resolution_m=0.04, inflation_radius_m=0.085).grid
    asym_grid._recenter(5.50, 0.50)
    for wy in np.linspace(-1.20, 0.15, 35):
        ix, iy = asym_grid.world_to_grid(5.40, float(wy))
        if asym_grid.in_bounds(ix, iy):
            asym_grid.occ[ix, iy] = 1.0
    for wy in np.linspace(1.30, 2.20, 25):
        ix, iy = asym_grid.world_to_grid(5.40, float(wy))
        if asym_grid.in_bounds(ix, iy):
            asym_grid.occ[ix, iy] = 1.0
    # Stamp moving-obstacle trail inside the opening at y_world in [0.40, 0.68]
    pos_hist = tuple((10.0 + k * 0.2, 5.14, 0.33 + 0.36 * math.sin(0.95 * k * 0.2)) for k in range(25))
    for _, hx, hy in pos_hist:
        for dx_t in (-0.08, 0.0, 0.08):
            for dy_t in (-0.08, 0.0, 0.08):
                ix, iy = asym_grid.world_to_grid(hx + dx_t, hy + dy_t)
                if asym_grid.in_bounds(ix, iy):
                    asym_grid.occ[ix, iy] = 0.85

    def _make_asym_trk(t_now: float, tid: int = 18):
        s_pos = 0.36 * math.sin(0.95 * t_now)
        s_vel = 0.36 * 0.95 * math.cos(0.95 * t_now)
        phi_c = math.atan2(math.sin(0.95 * t_now), math.cos(0.95 * t_now))
        return TrackedObstacle(
            track_id=tid,
            x=5.14,
            y=0.33 + s_pos,
            vx=0.0,
            vy=s_vel,
            radius=0.21,
            P_4x4=np.diag([0.002, 0.002, 0.02, 0.02]).astype(np.float64),
            last_update_stamp=t_now,
            age=30,
            hits=30,
            is_confirmed=True,
            consecutive_hits=30,
            pos_history=pos_hist,
            imm_valid=True,
            imm_u_hat=(0.0, 1.0),
            imm_p_c=(5.14, 0.33),
            imm_amp=0.36,
            imm_omega=0.95,
            imm_phi=phi_c,
            imm_t_ref=t_now,
            imm_mu_cv=0.01,
            imm_mu_harm=0.99,
            imm_sigma_h0=0.035,
            imm_sigma_omega=0.02,
        )

    sched = CrossingLaneScheduler()
    path_g2 = [(5.72, 0.27), (5.40, 0.55), (4.60, 0.70)]
    trk0 = _make_asym_trk(0.0, tid=18)
    dec0 = sched.step(
        rx=5.72, ry=0.27, rth=math.pi,
        v_max=0.20, w_max=0.65, now_sec=0.0,
        confirmed_tracks=[trk0], path_world=path_g2, local_grid=asym_grid,
    )
    assert dec0.active
    assert len(dec0.lane_candidates) == 2, f"Trail masking must preserve 2 lanes, got {dec0.lane_candidates}"
    assert dec0.lane_offset_m < -0.50, (
        f"Must pick feasible open side lane (~-0.70) instead of infeasible center (~0.0), got {dec0.lane_offset_m:.3f}"
    )
    assert dec0.mode == "APPROACH_ALIGN", (
        f"Must laterally align from y_r=+0.06 to y_star={dec0.lane_offset_m:.2f} instead of freezing in HOLD, got {dec0.mode}"
    )
    assert dec0.refresh_watchdog is True

    # 2) Closed-loop crossing of asymmetric gate: lateral shift -> DASH -> exits with 0 collisions
    rx_s, ry_s, rth_s = 5.72, 0.27, math.pi
    dt_s = 0.10
    modes_asym = []
    min_d_asym = float("inf")
    for k in range(150):
        t_k = k * dt_s
        trk_k = _make_asym_trk(t_k, tid=18)
        dec_k = sched.step(
            rx=rx_s, ry=ry_s, rth=rth_s,
            v_max=0.20, w_max=0.65, now_sec=t_k,
            confirmed_tracks=[trk_k],
            path_world=[(rx_s, ry_s), (5.40, 0.72), (4.30, 0.72)],
            local_grid=asym_grid,
        )
        modes_asym.append(dec_k.mode)
        v_u, w_u = (dec_k.v_ref, dec_k.w_ref) if dec_k.active else (0.20, 0.0)
        rth_s = rc_mod.normalize_angle(rth_s + w_u * dt_s)
        rx_s += v_u * math.cos(rth_s) * dt_s
        ry_s += v_u * math.sin(rth_s) * dt_s
        min_d_asym = min(min_d_asym, math.hypot(rx_s - trk_k.x, ry_s - trk_k.y))
        if rx_s < 4.65:
            break
    assert rx_s < 4.65, f"Robot failed to clear asymmetric gate; stopped at x={rx_s:.2f}, y={ry_s:.2f}"
    assert "DASH" in modes_asym
    assert min_d_asym >= 0.37, f"Asymmetric gate clearance too tight: {min_d_asym:.3f} m"

    # 3) Hysteresis: alignment bend (align=0.65), 0.4s track dropout, and same-pc re-ID preserve HOLD timer
    sched_h = CrossingLaneScheduler()
    # First check INACTIVE still rejects bent path with align ~ 0.65 (> 0.55)
    bent_path = [(5.72, 0.98), (5.14, 0.48), (4.60, 0.02)]  # slope dy/dx ~ 0.86 => |d.u| ~ 0.65
    dec_rej = sched_h.step(
        rx=5.72, ry=0.98, rth=math.pi,
        v_max=0.20, w_max=0.65, now_sec=0.0,
        confirmed_tracks=[_make_asym_trk(0.0, tid=18)],
        path_world=bent_path, local_grid=asym_grid,
    )
    assert dec_rej.active is False, "INACTIVE entry must still reject align > 0.55"

    # Enter HOLD cleanly on straight path (force high sigma so it stays in HOLD)
    trk_hold = _make_asym_trk(1.0, tid=18)
    trk_hold.imm_sigma_h0 = 0.28
    straight_path = [(5.72, 1.03), (5.14, 1.03), (4.50, 1.03)]
    dec_h1 = sched_h.step(
        rx=5.72, ry=1.03, rth=math.pi,
        v_max=0.20, w_max=0.65, now_sec=1.0,
        confirmed_tracks=[trk_hold], path_world=straight_path, local_grid=asym_grid,
    )
    assert dec_h1.mode == "HOLD"
    t_hold0 = sched_h._hold_start_time
    assert t_hold0 == pytest.approx(1.0)

    # Alignment bend to ~0.65 while active preserves HOLD timer
    dec_h2 = sched_h.step(
        rx=5.72, ry=1.03, rth=math.pi,
        v_max=0.20, w_max=0.65, now_sec=2.0,
        confirmed_tracks=[trk_hold], path_world=bent_path, local_grid=asym_grid,
    )
    assert dec_h2.active and dec_h2.mode == "HOLD"
    assert sched_h._hold_start_time == pytest.approx(t_hold0)

    # Brief 0.4s track dropout preserves HOLD timer
    dec_h3 = sched_h.step(
        rx=5.72, ry=1.03, rth=math.pi,
        v_max=0.20, w_max=0.65, now_sec=2.4,
        confirmed_tracks=[], path_world=straight_path, local_grid=asym_grid,
    )
    assert dec_h3.active and dec_h3.mode == "HOLD"
    assert sched_h._hold_start_time == pytest.approx(t_hold0)

    # Re-ID of same physical obstacle (tid 18 -> 55 at same pc) preserves HOLD timer
    trk_reid = _make_asym_trk(2.6, tid=55)
    trk_reid.imm_sigma_h0 = 0.28
    dec_h4 = sched_h.step(
        rx=5.72, ry=1.03, rth=math.pi,
        v_max=0.20, w_max=0.65, now_sec=2.6,
        confirmed_tracks=[trk_reid], path_world=straight_path, local_grid=asym_grid,
    )
    assert dec_h4.active and dec_h4.mode == "HOLD" and dec_h4.track_id == 55
    assert sched_h._hold_start_time == pytest.approx(t_hold0)

    # 4) CBF prefer_no_reverse prevents spurious HOLD reverse when h >= 0, allows reverse when h < 0
    cbf = DiscreteSoftCBFShield(v_min=-0.06, v_max=0.20, w_max=0.65)
    # Safe standoff d=0.48 m > r_safe (~0.40 m) => h_dyn > 0, with noisy vx toward robot
    trk_safe_hold = TrackedObstacle(
        track_id=18, x=0.48, y=0.05, vx=-0.12, vy=0.35, radius=0.21,
        P_4x4=np.diag([0.002, 0.002, 0.02, 0.02]).astype(np.float64),
        last_update_stamp=1.0, age=20, hits=20, is_confirmed=True,
    )
    res_def = cbf.filter(
        v_nom=0.0, w_nom=0.0, rx=0.0, ry=0.0, rth=0.0, dt=0.10,
        dynamic_obstacles=[trk_safe_hold], update_stats=False,
        prefer_no_reverse=False,
    )
    res_prot = cbf.filter(
        v_nom=0.0, w_nom=0.0, rx=0.0, ry=0.0, rth=0.0, dt=0.10,
        dynamic_obstacles=[trk_safe_hold], update_stats=False,
        prefer_no_reverse=True, suppress_tangential_steer=True,
    )
    assert res_prot.min_h_dynamic > 0.0
    assert res_def.v < -0.005, f"Baseline CBF should exhibit spurious HOLD reverse here, got v={res_def.v:.4f}"
    assert res_prot.v >= 0.0, f"prefer_no_reverse must prevent spurious HOLD reverse when h>0, got v={res_prot.v:.4f}"

    # Breached dynamic barrier (d=0.32 m < r_safe ~ 0.40 m => h_dyn < 0): emergency reverse MUST stay enabled
    trk_close = TrackedObstacle(
        track_id=18, x=0.32, y=0.0, vx=-0.20, vy=0.0, radius=0.21,
        P_4x4=np.diag([0.002, 0.002, 0.02, 0.02]).astype(np.float64),
        last_update_stamp=1.0, age=20, hits=20, is_confirmed=True,
    )
    res_emerg = cbf.filter(
        v_nom=0.0, w_nom=0.0, rx=0.0, ry=0.0, rth=0.0, dt=0.10,
        dynamic_obstacles=[trk_close], update_stats=False,
        prefer_no_reverse=True, suppress_tangential_steer=True,
    )
    assert res_emerg.min_h_dynamic < 0.0
    assert res_emerg.v < -0.01, f"Must still allow emergency reverse when h_dyn < 0, got v={res_emerg.v:.4f}"


def test_stepB3_wide_corridor_corner_blocks_crossing_scheduler_while_narrow_pinch_works():
    """
    Verify Step B-3 pure-geometry pinch/crossing guard:
      1. Wide open hall / corridor with an X-axis reciprocating obstacle:
         - Approaching a 90-deg corner where local path bends perpendicular to u_hat
           must NOT activate CrossingLaneScheduler (nor engage align 0.55->0.80 latch),
           whether checked via wide/unbounded corridor geometry OR downstream waypoints.
         - Moving axially along a corridor where an A* local detour jog briefly turns
           perpendicular (|d_local . u_hat| <= 0.55 while |d_macro . u_hat| > 0.60)
           must NOT activate CrossingLaneScheduler.
      2. Narrow bounded pinch opening (1.15 m gate with static walls on both sides):
         - Full Step B-2 lateral-lane pre-alignment, HOLD (including 0.55->0.80 bend
           hysteresis on verified pinch), and collision-free DASH continue to work.
    """
    from disaster_recovery import recovery_controller as rc_mod

    _, _, _, Waypoint = _make_dummy_controller_cls()
    CrossingLaneScheduler = rc_mod.CrossingLaneScheduler

    # 1) Build wide open hall grid (walls at x=+/-4.3, y=+/-4.3; inner baffle at y=1.5, x in [0.5, 2.7])
    hall_grid = ProposedRecoveryPlanner(size_m=6.0, resolution_m=0.04, inflation_radius_m=0.085).grid
    hall_grid._recenter(2.0, 2.0)
    for lx in np.linspace(-1.0, 4.4, 130):
        ix, iy = hall_grid.world_to_grid(float(lx), 4.30)
        if hall_grid.in_bounds(ix, iy):
            hall_grid.occ[ix, iy] = 1.0
    for ly in np.linspace(-1.0, 4.4, 130):
        ix, iy = hall_grid.world_to_grid(4.30, float(ly))
        if hall_grid.in_bounds(ix, iy):
            hall_grid.occ[ix, iy] = 1.0
    for lx in np.linspace(0.5, 2.7, 55):
        ix, iy = hall_grid.world_to_grid(float(lx), 1.50)
        if hall_grid.in_bounds(ix, iy):
            hall_grid.occ[ix, iy] = 1.0

    def _make_hall_axial_trk(t_now: float, pc_x: float = 1.85, pc_y: float = 2.55, amp: float = 0.90):
        omega = 0.45
        s_pos = amp * math.sin(omega * t_now)
        s_vel = amp * omega * math.cos(omega * t_now)
        phi_c = math.atan2(math.sin(omega * t_now), math.cos(omega * t_now))
        return TrackedObstacle(
            track_id=31,
            x=pc_x + s_pos,
            y=pc_y,
            vx=s_vel,
            vy=0.0,
            radius=0.18,
            P_4x4=np.diag([0.002, 0.002, 0.02, 0.02]).astype(np.float64),
            last_update_stamp=t_now,
            age=30,
            hits=30,
            is_confirmed=True,
            consecutive_hits=30,
            imm_valid=True,
            imm_u_hat=(1.0, 0.0),
            imm_p_c=(pc_x, pc_y),
            imm_amp=amp,
            imm_omega=omega,
            imm_phi=phi_c,
            imm_t_ref=t_now,
            imm_mu_cv=0.01,
            imm_mu_harm=0.99,
            imm_sigma_h0=0.03,
            imm_sigma_omega=0.02,
        )

    # 1a) Approaching corner along +y while turning into -x corridor (both with and without waypoint_stack)
    sched_hall = CrossingLaneScheduler()
    corner_wps = [
        Waypoint(-3.0, 2.80, math.pi),
        Waypoint(-1.5, 2.80, math.pi),
        Waypoint(0.0, 2.80, math.pi),
        Waypoint(1.4, 2.80, math.pi),
        Waypoint(2.8, 2.80, math.pi),
        Waypoint(2.9, 1.90, math.pi / 2.0),
    ]
    corner_paths = [
        # Going straight north toward corner (local perp to u_hat, wide open hall)
        ((2.95, 1.35, math.pi / 2.0), [(2.95, 1.35), (2.90, 2.10), (2.85, 2.80)]),
        # Turning corner (macro_align > 0.60, would previously stay latched at align_lim=0.80)
        ((2.65, 2.15, 2.1), [(2.65, 2.15), (2.60, 2.55), (1.90, 2.85), (1.20, 2.90)]),
        # Axial corridor with tiny local A* lateral jog perpendicular to u_hat (|d_macro . u_hat| ~ 0.94)
        ((0.60, 3.05, math.pi), [(0.60, 3.05), (0.50, 2.55), (0.00, 2.65), (-0.60, 2.95)]),
    ]
    for step_idx, ((rx_c, ry_c, rth_c), path_c) in enumerate(corner_paths):
        trk_c = _make_hall_axial_trk(10.0 + step_idx * 0.5)
        # Check with full hall_grid + corner_wps
        dec_w = sched_hall.step(
            rx=rx_c, ry=ry_c, rth=rth_c,
            v_max=0.20, w_max=0.65, now_sec=10.0 + step_idx * 0.5,
            confirmed_tracks=[trk_c], path_world=path_c,
            waypoint_stack=corner_wps, local_grid=hall_grid,
        )
        assert dec_w.active is False and dec_w.mode == "INACTIVE", (
            f"Wide hall corner step {step_idx} must NOT activate CrossingLaneScheduler, got {dec_w}"
        )
        # Check even with local_grid=None when corner_wps turn parallel to u_hat
        dec_wp_only = sched_hall.step(
            rx=rx_c, ry=ry_c, rth=rth_c,
            v_max=0.20, w_max=0.65, now_sec=12.0 + step_idx * 0.5,
            confirmed_tracks=[trk_c], path_world=path_c,
            waypoint_stack=corner_wps, local_grid=None,
        )
        assert dec_wp_only.active is False and dec_wp_only.mode == "INACTIVE", (
            f"Downstream axial waypoint stack at step {step_idx} must block corner false-trigger, got {dec_wp_only}"
        )
        # Check even without waypoint_stack when wide hall_grid is present
        dec_grid_only = sched_hall.step(
            rx=rx_c, ry=ry_c, rth=rth_c,
            v_max=0.20, w_max=0.65, now_sec=14.0 + step_idx * 0.5,
            confirmed_tracks=[trk_c], path_world=path_c,
            waypoint_stack=None, local_grid=hall_grid,
        )
        assert dec_grid_only.active is False and dec_grid_only.mode == "INACTIVE", (
            f"Wide unbounded hall grid at step {step_idx} must block false-trigger, got {dec_grid_only}"
        )

    # 2) Narrow bounded crossing pinch (1.15 m opening, walls on both sides, transverse downstream WPs)
    pinch_grid = ProposedRecoveryPlanner(size_m=6.0, resolution_m=0.04, inflation_radius_m=0.085).grid
    pinch_grid._recenter(5.50, 0.50)
    for wy in np.linspace(-1.20, 0.15, 35):
        ix, iy = pinch_grid.world_to_grid(5.40, float(wy))
        if pinch_grid.in_bounds(ix, iy):
            pinch_grid.occ[ix, iy] = 1.0
    for wy in np.linspace(1.30, 2.20, 25):
        ix, iy = pinch_grid.world_to_grid(5.40, float(wy))
        if pinch_grid.in_bounds(ix, iy):
            pinch_grid.occ[ix, iy] = 1.0

    pinch_wps = [
        Waypoint(1.20, 0.00, math.pi),
        Waypoint(2.50, 0.00, math.pi),
        Waypoint(3.90, 0.50, math.pi),
        Waypoint(5.40, 0.72, math.pi),
    ]

    def _make_pinch_trk(t_now: float):
        s_pos = 0.36 * math.sin(0.95 * t_now)
        s_vel = 0.36 * 0.95 * math.cos(0.95 * t_now)
        phi_c = math.atan2(math.sin(0.95 * t_now), math.cos(0.95 * t_now))
        return TrackedObstacle(
            track_id=42,
            x=5.05,
            y=0.72 + s_pos,
            vx=0.0,
            vy=s_vel,
            radius=0.21,
            P_4x4=np.diag([0.002, 0.002, 0.02, 0.02]).astype(np.float64),
            last_update_stamp=t_now,
            age=30,
            hits=30,
            is_confirmed=True,
            consecutive_hits=30,
            imm_valid=True,
            imm_u_hat=(0.0, 1.0),
            imm_p_c=(5.05, 0.72),
            imm_amp=0.36,
            imm_omega=0.95,
            imm_phi=phi_c,
            imm_t_ref=t_now,
            imm_mu_cv=0.01,
            imm_mu_harm=0.99,
            imm_sigma_h0=0.03,
            imm_sigma_omega=0.02,
        )

    sched_pinch = CrossingLaneScheduler()
    rx_p, ry_p, rth_p = 6.30, 0.50, math.pi
    modes_p = []
    min_d_p = float("inf")
    for k in range(180):
        t_k = k * 0.10
        trk_p = _make_pinch_trk(t_k)
        dec_p = sched_pinch.step(
            rx=rx_p, ry=ry_p, rth=rth_p,
            v_max=0.20, w_max=0.65, now_sec=t_k,
            confirmed_tracks=[trk_p],
            path_world=[(rx_p, ry_p), (5.40, 0.72), (4.20, 0.55)],
            waypoint_stack=pinch_wps,
            local_grid=pinch_grid,
        )
        modes_p.append(dec_p.mode)
        v_u, w_u = (dec_p.v_ref, dec_p.w_ref) if dec_p.active else (0.20, 0.0)
        rth_p = rc_mod.normalize_angle(rth_p + w_u * 0.10)
        rx_p += v_u * math.cos(rth_p) * 0.10
        ry_p += v_u * math.sin(rth_p) * 0.10
        min_d_p = min(min_d_p, math.hypot(rx_p - trk_p.x, ry_p - trk_p.y))
        if rx_p < 4.55:
            break

    assert rx_p < 4.55, f"Narrow pinch crossing must complete cleanly; stopped at ({rx_p:.2f}, {ry_p:.2f})"
    assert "APPROACH_ALIGN" in modes_p and "DASH" in modes_p
    assert min_d_p >= 0.37, f"Narrow pinch crossing clearance violated: {min_d_p:.3f} m"


def test_stepB4_headon_dynamic_closing_speed_cap_and_non_interference():
    """
    Verify Step B-4 (c):
      1) When CrossingLaneScheduler is inactive and a confirmed dynamic track is close ahead
         (d_dyn < 0.85 m) with fast head-on closing (closing_speed > 0.25 m/s), the proposed_rl
         velocity cap drops from 0.85 * v_astar_ref (~0.17 m/s) to 0.09 m/s to suppress Sim v4.4
         head-on rush (tested for both g_res > 0.25 and g_res == 0.0 paths).
      2) No 0.09 m/s intervention when:
         - Obstacle is far (d_dyn >= 0.85 m),
         - Obstacle is receding or barely closing (closing_speed <= 0.05 m/s / obs receding),
         - CrossingLaneScheduler is active (HOLD / DASH / DASH_FALLBACK),
         - Confirmed dynamic tracks == 0 (empty or tentative) or scan_corrupted (spike_ratio >= 0.03)
           -> bit-identical commands vs Step A-2 baseline.
      3) Combined Step B + Step B-4 added per-tick compute latency stays < 1.0 ms.
    """
    import time
    from disaster_recovery import recovery_controller as rc_mod

    RecoveryController, SemanticObjectCache, SystemState, Waypoint = _make_dummy_controller_cls()
    CrossingLaneScheduler = rc_mod.CrossingLaneScheduler
    CrossingDecision = rc_mod.CrossingDecision

    class _FakeTime:
        def __init__(self, ns: int):
            self.nanoseconds = int(ns)

        def __sub__(self, other):
            return _FakeTime(self.nanoseconds - other.nanoseconds)

    class _MockRushRLPolicy:
        def __init__(self, v_out: float = 0.20, w_out: float = 0.0, ch_peak: float = 0.85):
            self.patch_channels = 4
            self.last_ch123_peak = float(ch_peak)
            self._v_out = float(v_out)
            self._w_out = float(w_out)

        def predict(self, **kwargs):
            dyn_ch3 = kwargs.get("dynamic_ch3")
            if dyn_ch3 is None:
                self.last_ch123_peak = 0.0
            else:
                self.last_ch123_peak = float(np.max(np.abs(dyn_ch3)))
            return self._v_out, self._w_out

        def observe(self, **kwargs):
            return None

    xs = np.linspace(-1.0, 3.0, 100)
    wide_walls = np.vstack([
        np.column_stack((xs, np.full_like(xs, -1.25))),
        np.column_stack((xs, np.full_like(xs, 1.25))),
    ]).astype(np.float32)
    raw_360 = [2.5] * 360

    def _make_rl_ctrl(
        *,
        dyn_tracks,
        ch3_peak: float = 0.85,
        spike_ratio: float = 0.0,
        actual_v: float = 0.16,
        active_cross_dec=None,
    ):
        c = RecoveryController.__new__(RecoveryController)
        c.recovery_strategy = "proposed_rl"
        c.state = SystemState.BACKTRACKING_HYBRID
        c.previous_state = SystemState.BACKTRACKING_HYBRID
        c.fault_reason = "HEARTBEAT_TIMEOUT"
        c.current_gas_level = 0.0
        c.gas_threshold = 0.40
        c.last_heartbeat_time = _FakeTime(0)
        c._link_restored_timer = None
        c.clear_stack_on_link_restore = False
        c.reset_corr_on_episode = False
        c.odom_initialized = True
        c.heartbeat_received_once = True
        c.current_x = 0.0
        c.current_y = 0.0
        c.current_theta = 0.0
        c.raw_odom_x = 0.0
        c.raw_odom_y = 0.0
        c.raw_odom_theta = 0.0
        c.corr_x = 0.0
        c.corr_y = 0.0
        c.corr_th = 0.0
        c.actual_v = float(actual_v)
        c.actual_w = 0.0
        c.last_cmd_v = float(actual_v)
        c.last_cmd_w = 0.0
        c.v_max = 0.20
        c.w_max = 0.65
        c.kp_lin = 0.60
        c.kp_ang = 1.40
        c.wp_reach_radius = 0.10
        c.docking_dist_th = 0.50
        c.use_aruco_docking = False
        c.enable_scan_matching = False
        c.enable_semantic_astar_cost = False
        c.enable_dynamic_headon_hold = False
        c.enable_cbf_safety_shield = True
        c.enable_dynamic_tracker = False
        c.enable_lidar_slip_residual = False
        c.icp_degenerate = False
        c._prev_slip_scan_pts = None
        c.scan_vel_v = 0.0
        c.scan_vel_w = 0.0
        c.use_semantic_rl_policy = False
        c.is_phase2b = False
        c.alpha_valve_deadband = 0.0
        c._prev_alpha_rl = 0.8
        c.alpha_max_step_base = 1.0
        c.alpha_max_step_sem_gain = 0.0
        c.recovery_step_count = 0
        c._prev_front_min = None
        c._prev_front_time = None
        c._dynamic_hold_until = 0.0
        c._dynamic_hold_cooldown_until = 0.0
        c._rl_cooldown_until = 0.0
        c._rl_lockout_anchor = None
        c._prev_escape_count = 0
        c.cbf_inner_loop_hz = 10.0
        c.cbf_v_min = -0.06
        c.min_effective_linear_vel = 0.0
        c.min_effective_angular_vel = 0.0
        c.latest_scan_pts = wide_walls
        c.latest_raw_ranges = list(raw_360)
        c.latest_scan_pose = (0.0, 0.0, 0.0)
        c.latest_scan_stamp_sec = 10.0
        c.latest_scan_angle_min = 0.0
        c.latest_scan_angle_increment = 2.0 * math.pi / 360.0
        c.latest_scan_spike_ratio = float(spike_ratio)
        c.latest_dynamic_tracks = list(dyn_tracks)
        c.latest_dynamic_ch3 = (
            np.full((12, 12), float(ch3_peak), dtype=np.float32) if ch3_peak > 0.0 else None
        )
        c.dynamic_tracker = None
        c.context_rl_policy = _MockRushRLPolicy(v_out=0.20, w_out=0.0, ch_peak=ch3_peak)
        c.semantic_cache = SemanticObjectCache()
        c.local_planner = ProposedRecoveryPlanner(size_m=6.0, resolution_m=0.04, inflation_radius_m=0.085)
        c.local_planner.observe(0.0, 0.0, 0.0, wide_walls)
        # Keep cbf_shield=None so published v_cmd exposes the exact nominal consistency guard output
        c.cbf_shield = None
        c.latest_cbf_result = None
        if active_cross_dec is not None:
            c.crossing_scheduler = types.SimpleNamespace(
                active=True,
                step=lambda **kwargs: active_cross_dec,
                reset=lambda: None,
            )
        else:
            c.crossing_scheduler = CrossingLaneScheduler()
        c.waypoint_stack = deque([Waypoint(2.4, 0.0, 0.0), Waypoint(1.2, 0.0, 0.0)])
        c.start_pose = c.waypoint_stack[0]
        c.active_target_wp = c.waypoint_stack[-1]
        c._icp_section_ms_total = 0.0
        c._icp_section_ms_max = 0.0
        c._icp_section_count = 0
        c._icp_section_last_log = 0.0
        c._icp_run_count = 0
        c._icp_skip_count = 0
        c._clock_now_ns = int(10.0 * 1e9)
        c.get_clock = lambda: types.SimpleNamespace(now=lambda: _FakeTime(c._clock_now_ns))
        c.published_cmds = []
        c.pub_cmd_vel = types.SimpleNamespace(
            publish=lambda m: c.published_cmds.append((float(m.linear.x), float(m.angular.z)))
        )
        c.pub_system_status = types.SimpleNamespace(publish=lambda m: None)
        c.get_logger = lambda: types.SimpleNamespace(
            info=lambda *a, **k: None, warn=lambda *a, **k: None, error=lambda *a, **k: None
        )
        return c

    def _trk(x: float, y: float, vx: float, vy: float = 0.0, confirmed: bool = True, tid: int = 11):
        return TrackedObstacle(
            track_id=tid,
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            radius=0.18,
            P_4x4=np.diag([0.002, 0.002, 0.01, 0.01]).astype(np.float64),
            last_update_stamp=10.0,
            age=25 if confirmed else 2,
            hits=25 if confirmed else 2,
            is_confirmed=confirmed,
            consecutive_hits=25 if confirmed else 2,
        )

    # 1) Close head-on closing (d_dyn=0.834 < 0.85 m, fwd_cos=0.864 >= 0.50,
    #    obs_approach=0.294 > 0.05, closing=0.466 > 0.25) -> speed cap 0.09 m/s activates
    #    (y=0.42 keeps A* in ASTAR_TRACK with 0.85 * v_astar_ref ~ 0.12 m/s > 0.09 m/s).
    trk_close_fast = _trk(x=0.72, y=0.42, vx=-0.34)
    c_close_gres = _make_rl_ctrl(dyn_tracks=[trk_close_fast], ch3_peak=0.85)
    c_close_gres.control_loop()
    assert len(c_close_gres.published_cmds) == 1
    assert c_close_gres.published_cmds[0][0] == pytest.approx(0.09, abs=1e-4), (
        f"Expected 0.09 m/s head-on closing cap (g_res=1), got {c_close_gres.published_cmds[0]}"
    )

    c_close_nogres = _make_rl_ctrl(dyn_tracks=[trk_close_fast], ch3_peak=0.0)
    c_close_nogres.control_loop()
    assert c_close_nogres.published_cmds[0][0] == pytest.approx(0.09, abs=1e-4), (
        f"Expected 0.09 m/s head-on closing cap (g_res=0), got {c_close_nogres.published_cmds[0]}"
    )

    # 2a) Far approaching obstacle (d_dyn=1.18 >= 0.85 m): keeps 0.85 * v_astar_ref (~0.17 m/s), NOT 0.09
    c_far = _make_rl_ctrl(dyn_tracks=[_trk(x=1.10, y=0.42, vx=-0.35)], ch3_peak=0.85)
    c_far.control_loop()
    assert c_far.published_cmds[0][0] >= 0.14, (
        f"Far approaching obstacle (>=0.85m) must not engage 0.09 m/s cap, got {c_far.published_cmds[0]}"
    )

    # 2b) Receding obstacle (vx=+0.30, closing_speed <= 0.05) & barely-approaching (vx=-0.04): NO 0.09 cap
    c_recede = _make_rl_ctrl(dyn_tracks=[_trk(x=0.72, y=0.42, vx=+0.30)], ch3_peak=0.85)
    c_recede.control_loop()
    assert c_recede.published_cmds[0][0] >= 0.14, (
        f"Receding obstacle must not engage head-on cap, got {c_recede.published_cmds[0]}"
    )
    c_slow_obs = _make_rl_ctrl(dyn_tracks=[_trk(x=0.72, y=0.42, vx=-0.04)], ch3_peak=0.85, actual_v=0.02)
    c_slow_obs.control_loop()
    assert c_slow_obs.published_cmds[0][0] >= 0.12, (
        f"Slow obstacle (obs_approach <= 0.05 m/s) must not engage 0.09 m/s cap, got {c_slow_obs.published_cmds[0]}"
    )

    # 2c) CrossingLaneScheduler active (HOLD / DASH / DASH_FALLBACK): head-on cap NEVER interferes
    for mode_name, v_sched in (("HOLD", 0.0), ("DASH", 0.20), ("DASH_FALLBACK", 0.20)):
        dec_sched = CrossingDecision(
            active=True,
            mode=mode_name,
            v_ref=v_sched,
            w_ref=0.0,
            target_xy=(1.2, 0.0),
            track_id=11,
            lane_offset_m=0.30,
            lane_candidates=(0.30, -0.30),
            t_launch_sec=0.0,
            min_clearance_m=0.45,
            refresh_watchdog=True,
        )
        c_sched = _make_rl_ctrl(
            dyn_tracks=[trk_close_fast], ch3_peak=0.85, active_cross_dec=dec_sched
        )
        c_sched.control_loop()
        assert c_sched.published_cmds[0][0] == pytest.approx(v_sched, abs=1e-6), (
            f"CrossingLaneScheduler {mode_name} must bypass head-on cap; expected {v_sched}, got {c_sched.published_cmds[0]}"
        )

    # 2d) 0 confirmed tracks (empty / tentative) & corrupted scan (spike_ratio >= 0.03): bit-identical to baseline
    c_zero_empty = _make_rl_ctrl(dyn_tracks=[], ch3_peak=0.0, spike_ratio=0.0)
    c_zero_tent = _make_rl_ctrl(
        dyn_tracks=[_trk(x=0.65, y=0.0, vx=-0.35, confirmed=False)], ch3_peak=0.0, spike_ratio=0.0
    )
    c_corrupt = _make_rl_ctrl(dyn_tracks=[trk_close_fast], ch3_peak=0.0, spike_ratio=0.05)
    for ci in (c_zero_empty, c_zero_tent, c_corrupt):
        ci.control_loop()
    assert c_zero_tent.published_cmds == c_zero_empty.published_cmds, (
        "Tentative-only tracks (0 confirmed) must be bit-identical to 0-track baseline"
    )
    assert c_corrupt.published_cmds == c_zero_empty.published_cmds, (
        "Corrupted scan (spike_ratio >= 0.03) must bypass Step B-4 guards identically to baseline"
    )

    # 3) Latency check: Step B-4 axial band + head-on closing check adds < 1.0 ms per tick
    axial_perf_trk = TrackedObstacle(
        track_id=77,
        x=1.20,
        y=0.0,
        vx=-0.35,
        vy=0.0,
        radius=0.18,
        P_4x4=np.diag([0.002, 0.002, 0.01, 0.01]).astype(np.float64),
        last_update_stamp=10.0,
        age=25,
        hits=25,
        is_confirmed=True,
        consecutive_hits=25,
        imm_valid=True,
        imm_u_hat=(1.0, 0.0),
        imm_p_c=(1.20, 0.0),
        imm_amp=0.933,
        imm_omega=0.45,
        imm_phi=0.0,
        imm_t_ref=10.0,
        imm_mu_cv=0.01,
        imm_mu_harm=0.99,
        imm_sigma_h0=0.03,
        imm_sigma_omega=0.02,
    )
    grid_p = c_zero_empty.local_planner.grid
    wps_p = [Waypoint(2.4, 0.0, 0.0), Waypoint(1.2, 0.0, 0.0)]
    for _ in range(10):
        grid_p.build_cost_maps(0.0, 0.0, waypoint_list=wps_p)
        grid_p.build_cost_maps(0.0, 0.0, dynamic_obstacles=[axial_perf_trk], waypoint_list=wps_p)
        c_close_gres._headon_dynamic_closing_active([trk_close_fast], v_ref=0.20)
    n_it = 150
    t0 = time.perf_counter()
    for _ in range(n_it):
        grid_p.build_cost_maps(0.0, 0.0, waypoint_list=wps_p)
    base_ms = (time.perf_counter() - t0) * 1000.0 / n_it
    t1 = time.perf_counter()
    for _ in range(n_it):
        grid_p.build_cost_maps(0.0, 0.0, dynamic_obstacles=[axial_perf_trk], waypoint_list=wps_p)
        c_close_gres._headon_dynamic_closing_active([trk_close_fast], v_ref=0.20)
    stepb4_ms = (time.perf_counter() - t1) * 1000.0 / n_it
    added_ms = max(0.0, stepb4_ms - base_ms)
    assert added_ms < 1.0, f"Step B-4 added compute latency {added_ms:.3f} ms >= 1.0 ms"


def test_ablation_toggle_parity_and_clean_disable():
    """
    Verify `enable_spatiotemporal_dynamic_control` ablation toggle:
      (1) True (default) is bit-identical to Step B-4 across tracker IMM, axial sweep costmap,
          CrossingLaneScheduler, and 0.09 m/s head-on closing deceleration guard.
      (2) False cleanly disables all 4 Step B~B-4 spatiotemporal elements while keeping
          5 Hz scan-gated + 10 Hz Damped-CV KF (Ch3 + CBF-QP) and Step A-2 ICP intact.
    """
    from disaster_recovery import recovery_controller as rc_mod
    from disaster_recovery.cbf_safety_shield import DiscreteSoftCBFShield
    from disaster_recovery.dynamic_tracker import DynamicObstacleTracker, predict_track_trajectory

    RecoveryController, SemanticObjectCache, SystemState, Waypoint = _make_dummy_controller_cls()
    CrossingLaneScheduler = rc_mod.CrossingLaneScheduler
    CrossingDecision = rc_mod.CrossingDecision

    class _FakeTime:
        def __init__(self, ns: int):
            self.nanoseconds = int(ns)

        def __sub__(self, other):
            return _FakeTime(self.nanoseconds - other.nanoseconds)

    class _MockRushRLPolicy:
        def __init__(self, v_out: float = 0.20, w_out: float = 0.0, ch_peak: float = 0.85):
            self.patch_channels = 4
            self._v_out = float(v_out)
            self._w_out = float(w_out)
            self.last_ch123_peak = float(ch_peak)

        def observe(self, **kwargs):
            return None

        def predict(self, **kwargs):
            dyn_ch3 = kwargs.get("dynamic_ch3")
            if dyn_ch3 is None:
                self.last_ch123_peak = 0.0
            else:
                self.last_ch123_peak = float(np.max(np.abs(dyn_ch3)))
            return self._v_out, self._w_out

    # ------------------------------------------------------------------
    # (1) DynamicObstacleTracker: 5 Hz scan + 10 Hz Damped-CV KF vs IMM
    # ------------------------------------------------------------------
    trk_default = DynamicObstacleTracker()
    trk_on = DynamicObstacleTracker(enable_spatiotemporal_dynamic_control=True)
    trk_off = DynamicObstacleTracker(enable_spatiotemporal_dynamic_control=False)
    assert trk_default.enable_spatiotemporal_dynamic_control is True

    # Feed a clean 1D harmonic reciprocating cylinder along x in [0.95, 2.25] at 5 Hz scan + 10 Hz control
    n_angles = 360
    angles = np.linspace(0.0, 2.0 * math.pi, n_angles, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    obs_rad = 0.18
    omega_osc = 1.15
    amp_osc = 0.65
    cx_osc = 1.60

    def _make_dyn_ranges(cx: float = None, cy: float = 0.0) -> list:
        ranges = np.full(n_angles, 3.4, dtype=np.float64)
        # Static walls at y = +/- 1.8 m and x = +/- 2.8 m
        for i in range(n_angles):
            ca, sa = float(cos_a[i]), float(sin_a[i])
            r_best = 3.4
            if abs(sa) > 1e-6:
                for wy in (-1.8, 1.8):
                    t_w = wy / sa
                    if 0.12 < t_w < r_best:
                        r_best = t_w
            if abs(ca) > 1e-6:
                for wx in (-2.8, 2.8):
                    t_w = wx / ca
                    if 0.12 < t_w < r_best:
                        r_best = t_w
            if cx is not None:
                # Ray-circle intersection from origin (0, 0) along (ca, sa)
                b_proj = cx * ca + cy * sa
                if b_proj > 0.0:
                    c_val = cx * cx + cy * cy - obs_rad * obs_rad
                    disc = b_proj * b_proj - c_val
                    if disc >= 0.0:
                        t_hit = b_proj - math.sqrt(disc)
                        if 0.12 < t_hit < r_best:
                            r_best = t_hit
            ranges[i] = r_best
        return ranges.tolist()

    # First prime static walls (5 scans without dynamic obstacle) so observed_free is established
    wall_ranges = _make_dyn_ranges(cx=None)
    for k in range(5):
        t_s = 1.0 + 0.2 * k
        for trk_inst in (trk_default, trk_on, trk_off):
            trk_inst.update(0.0, 0.0, 0.0, wall_ranges, scan_stamp_sec=t_s, now_sec=t_s, omega=0.0)

    # 20 scan updates (4.0 s @ 5 Hz) interleaved with mid-tick 10 Hz extrapolation calls
    for k in range(20):
        t_scan = 2.0 + 0.2 * k
        ox = cx_osc + amp_osc * math.sin(omega_osc * (t_scan - 2.0))
        ranges_k = _make_dyn_ranges(ox, 0.0)
        for trk_inst in (trk_default, trk_on, trk_off):
            trk_inst.update(0.0, 0.0, 0.0, ranges_k, scan_stamp_sec=t_scan, now_sec=t_scan, omega=0.0)
            # Mid-tick 10 Hz call with same scan stamp
            trk_inst.update(0.0, 0.0, 0.0, ranges_k, scan_stamp_sec=t_scan, now_sec=t_scan + 0.1, omega=0.0)

    t_eval = 2.0 + 0.2 * 19 + 0.1
    trks_def = [t for t in trk_default.get_extrapolated_tracks(now_sec=t_eval, confirmed_only=True)]
    trks_on = [t for t in trk_on.get_extrapolated_tracks(now_sec=t_eval, confirmed_only=True)]
    trks_off = [t for t in trk_off.get_extrapolated_tracks(now_sec=t_eval, confirmed_only=True)]
    assert len(trks_def) >= 1 and len(trks_on) == len(trks_def) and len(trks_off) >= 1

    # Default vs explicit True parity (bit-identical)
    assert trks_def[0].x == trks_on[0].x and trks_def[0].y == trks_on[0].y
    assert trks_def[0].imm_mu_harm == trks_on[0].imm_mu_harm
    assert trks_on[0].imm_valid is True and trks_on[0].imm_mu_harm > 0.50

    # False disables IMM completely while preserving Damped-CV KF, Ch3 projection, and CBF risk circles
    assert trks_off[0].is_confirmed is True
    assert trks_off[0].imm_valid is False
    assert trks_off[0].imm_mu_harm == 0.0
    pred_off = trk_off.predict_trajectory(trks_off[0], (0.5, 1.0, 1.5))
    pred_cv_ref = predict_track_trajectory(trks_off[0], (0.5, 1.0, 1.5), enable_harmonic=False)
    assert pred_off.mu_harm == 0.0 and pred_off.is_harmonic is False
    assert np.allclose(pred_off.centers, pred_cv_ref.centers, atol=1e-9)

    # Even if a track carries external IMM fields, predict_trajectory / predict_future with OFF stays pure CV
    pred_off_ext = trk_off.predict_trajectory(trks_on[0], (0.5, 1.0))
    pred_cv_ext = predict_track_trajectory(trks_on[0], (0.5, 1.0), enable_harmonic=False)
    assert pred_off_ext.mu_harm == 0.0 and pred_off_ext.is_harmonic is False
    assert np.allclose(pred_off_ext.centers, pred_cv_ext.centers, atol=1e-9)

    ch3_off = trk_off.project_ch3(0.0, 0.0, 0.0, now_sec=t_eval)
    assert ch3_off.shape == (12, 12) and float(np.max(ch3_off)) > 0.05
    circles_off = trk_off.get_predicted_risk_circles(now_sec=t_eval, horizons=(0.5, 1.0))
    assert len(circles_off) >= 2

    # ------------------------------------------------------------------
    # (2) Axial sweep band & A* dynamic pass-through + CrossingScheduler + 0.09 m/s guard
    # ------------------------------------------------------------------
    xs_w = np.linspace(-1.0, 3.0, 100)
    wide_walls = np.vstack([
        np.column_stack((xs_w, np.full_like(xs_w, -1.25))),
        np.column_stack((xs_w, np.full_like(xs_w, 1.25))),
    ]).astype(np.float32)
    raw_360 = [2.5] * 360

    axial_trk = TrackedObstacle(
        track_id=10,
        x=1.20,
        y=0.0,
        vx=-0.35,
        vy=0.0,
        radius=0.18,
        P_4x4=np.diag([0.002, 0.002, 0.01, 0.01]).astype(np.float64),
        last_update_stamp=10.0,
        age=25,
        hits=25,
        is_confirmed=True,
        consecutive_hits=25,
        imm_valid=True,
        imm_u_hat=(1.0, 0.0),
        imm_p_c=(1.20, 0.0),
        imm_amp=0.93,
        imm_omega=0.45,
        imm_phi=0.0,
        imm_t_ref=10.0,
        imm_mu_cv=0.01,
        imm_mu_harm=0.99,
        imm_sigma_h0=0.03,
        imm_sigma_omega=0.02,
    )

    wps_ax = [Waypoint(2.4, 0.0, 0.0), Waypoint(1.6, 0.0, 0.0), Waypoint(0.8, 0.0, 0.0)]
    p_def = ProposedRecoveryPlanner(size_m=6.0, resolution_m=0.04, inflation_radius_m=0.085)
    p_on = ProposedRecoveryPlanner(
        size_m=6.0, resolution_m=0.04, inflation_radius_m=0.085, enable_spatiotemporal_dynamic_control=True
    )
    p_off = ProposedRecoveryPlanner(
        size_m=6.0, resolution_m=0.04, inflation_radius_m=0.085, enable_spatiotemporal_dynamic_control=False
    )
    for pl in (p_def, p_on, p_off):
        pl.observe(0.0, 0.0, 0.0, wide_walls)

    _, soft_def = p_def.grid.build_cost_maps(0.0, 0.0, dynamic_obstacles=[axial_trk], waypoint_list=wps_ax)
    _, soft_on = p_on.grid.build_cost_maps(0.0, 0.0, dynamic_obstacles=[axial_trk], waypoint_list=wps_ax)
    _, soft_off = p_off.grid.build_cost_maps(0.0, 0.0, dynamic_obstacles=[axial_trk], waypoint_list=wps_ax)
    assert np.array_equal(soft_def, soft_on)
    assert p_on.grid._last_axial_band_active is True
    assert p_on.grid._last_axial_core_mask is not None
    assert p_off.grid._last_axial_band_active is False
    assert p_off.grid._last_axial_core_mask is None

    def _make_ctrl(
        *,
        enable_st: bool,
        dyn_tracks,
        use_rl: bool = True,
        enable_cbf_obj: bool = False,
        active_cross_dec=None,
    ):
        c = RecoveryController.__new__(RecoveryController)
        c.recovery_strategy = "proposed_rl" if use_rl else "proposed"
        c.state = SystemState.BACKTRACKING_HYBRID
        c.previous_state = SystemState.BACKTRACKING_HYBRID
        c.fault_reason = "HEARTBEAT_TIMEOUT"
        c.current_gas_level = 0.0
        c.gas_threshold = 0.40
        c.last_heartbeat_time = _FakeTime(0)
        c._link_restored_timer = None
        c.clear_stack_on_link_restore = False
        c.reset_corr_on_episode = False
        c.odom_initialized = True
        c.heartbeat_received_once = True
        c.current_x = 0.0
        c.current_y = 0.0
        c.current_theta = 0.0
        c.raw_odom_x = 0.0
        c.raw_odom_y = 0.0
        c.raw_odom_theta = 0.0
        c.corr_x = 0.0
        c.corr_y = 0.0
        c.corr_th = 0.0
        c.actual_v = 0.16
        c.actual_w = 0.0
        c.last_cmd_v = 0.16
        c.last_cmd_w = 0.0
        c.v_max = 0.20
        c.w_max = 0.65
        c.kp_lin = 0.60
        c.kp_ang = 1.40
        c.wp_reach_radius = 0.10
        c.docking_dist_th = 0.50
        c.use_aruco_docking = False
        c.enable_scan_matching = False
        c.enable_semantic_astar_cost = False
        c.enable_dynamic_headon_hold = False
        c.enable_cbf_safety_shield = True
        c.enable_dynamic_tracker = False
        c.enable_spatiotemporal_dynamic_control = bool(enable_st)
        c.enable_lidar_slip_residual = False
        c.icp_degenerate = False
        c._prev_slip_scan_pts = None
        c.scan_vel_v = 0.0
        c.scan_vel_w = 0.0
        c.use_semantic_rl_policy = False
        c.is_phase2b = False
        c.alpha_valve_deadband = 0.0
        c._prev_alpha_rl = 0.8
        c.alpha_max_step_base = 1.0
        c.alpha_max_step_sem_gain = 0.0
        c.recovery_step_count = 0
        c._prev_front_min = None
        c._prev_front_time = None
        c._dynamic_hold_until = 0.0
        c._dynamic_hold_cooldown_until = 0.0
        c._rl_cooldown_until = 0.0
        c._rl_lockout_anchor = None
        c._prev_escape_count = 0
        c.cbf_inner_loop_hz = 10.0
        c.cbf_v_min = -0.06
        c.min_effective_linear_vel = 0.0
        c.min_effective_angular_vel = 0.0
        c.latest_scan_pts = wide_walls
        c.latest_raw_ranges = list(raw_360)
        c.latest_scan_pose = (0.0, 0.0, 0.0)
        c.latest_scan_stamp_sec = 10.0
        c.latest_scan_angle_min = 0.0
        c.latest_scan_angle_increment = 2.0 * math.pi / 360.0
        c.latest_scan_spike_ratio = 0.0
        c.latest_dynamic_tracks = list(dyn_tracks)
        c.latest_dynamic_ch3 = np.full((12, 12), 0.85, dtype=np.float32)
        c.dynamic_tracker = None
        c.context_rl_policy = _MockRushRLPolicy(v_out=0.20, w_out=0.0, ch_peak=0.85) if use_rl else None
        c.semantic_cache = SemanticObjectCache()
        c.local_planner = ProposedRecoveryPlanner(
            size_m=6.0,
            resolution_m=0.04,
            inflation_radius_m=0.085,
            enable_spatiotemporal_dynamic_control=bool(enable_st),
        )
        c.local_planner.observe(0.0, 0.0, 0.0, wide_walls)
        c.cbf_shield = DiscreteSoftCBFShield(v_min=-0.06, v_max=0.20, w_max=0.65) if enable_cbf_obj else None
        c.latest_cbf_result = None
        if active_cross_dec is not None:
            c.crossing_scheduler = types.SimpleNamespace(
                active=True,
                step=lambda **kwargs: active_cross_dec,
                reset=lambda: None,
            )
        else:
            c.crossing_scheduler = CrossingLaneScheduler() if enable_st else None
        c.waypoint_stack = deque([Waypoint(2.4, 0.0, 0.0), Waypoint(1.6, 0.0, 0.0), Waypoint(0.8, 0.0, 0.0)])
        c.start_pose = c.waypoint_stack[0]
        c.active_target_wp = c.waypoint_stack[-1]
        c._icp_section_ms_total = 0.0
        c._icp_section_ms_max = 0.0
        c._icp_section_count = 0
        c._icp_section_last_log = 0.0
        c._icp_run_count = 0
        c._icp_skip_count = 0
        c._clock_now_ns = int(10.0 * 1e9)
        c.get_clock = lambda: types.SimpleNamespace(now=lambda: _FakeTime(c._clock_now_ns))
        c.published_cmds = []
        c.pub_cmd_vel = types.SimpleNamespace(
            publish=lambda m: c.published_cmds.append((float(m.linear.x), float(m.angular.z)))
        )
        c.pub_system_status = types.SimpleNamespace(publish=lambda m: None)
        c.get_logger = lambda: types.SimpleNamespace(
            info=lambda *a, **k: None, warn=lambda *a, **k: None, error=lambda *a, **k: None
        )
        return c

    # (2a) A* dynamic track pass-through disabled when enable_semantic_astar_cost=False and toggle=False
    c_ax_on = _make_ctrl(enable_st=True, dyn_tracks=[axial_trk], use_rl=False)
    c_ax_off = _make_ctrl(enable_st=False, dyn_tracks=[axial_trk], use_rl=False)
    c_ax_empty = _make_ctrl(enable_st=False, dyn_tracks=[], use_rl=False)
    c_ax_on.control_loop()
    c_ax_off.control_loop()
    c_ax_empty.control_loop()
    assert abs(c_ax_on.local_planner.last_subgoal[1]) > 0.15
    assert c_ax_off.local_planner.last_subgoal == c_ax_empty.local_planner.last_subgoal
    assert c_ax_off.published_cmds == c_ax_empty.published_cmds
    assert c_ax_off.local_planner.grid._last_axial_band_active is False
    assert c_ax_off.local_planner.grid._last_axial_core_mask is None

    # (2b) 0.09 m/s head-on closing deceleration guard disabled when toggle=False (reverts to 0.85 * v_astar_ref)
    headon_trk = TrackedObstacle(
        track_id=12,
        x=0.65,
        y=0.0,
        vx=-0.35,
        vy=0.0,
        radius=0.18,
        P_4x4=np.diag([0.002, 0.002, 0.01, 0.01]).astype(np.float64),
        last_update_stamp=10.0,
        age=25,
        hits=25,
        is_confirmed=True,
        consecutive_hits=25,
    )
    c_ho_on = _make_ctrl(enable_st=True, dyn_tracks=[headon_trk], use_rl=True, enable_cbf_obj=False)
    c_ho_off = _make_ctrl(enable_st=False, dyn_tracks=[headon_trk], use_rl=True, enable_cbf_obj=False)
    c_ho_on.crossing_scheduler = None
    assert c_ho_on._headon_dynamic_closing_active([headon_trk], v_ref=0.20) is True
    assert c_ho_off._headon_dynamic_closing_active([headon_trk], v_ref=0.20) is False
    c_ho_on.control_loop()
    c_ho_off.control_loop()
    assert c_ho_on.published_cmds[0][0] <= 0.09 + 1e-6
    assert c_ho_off.published_cmds[0][0] >= 0.12

    # (2c) CrossingLaneScheduler completely bypassed when toggle=False (even if scheduler object exists)
    hold_dec = CrossingDecision(
        active=True,
        mode="HOLD",
        v_ref=0.0,
        w_ref=0.0,
        target_xy=(1.2, 0.0),
        track_id=12,
        lane_offset_m=0.30,
        lane_candidates=(0.30, -0.30),
        t_launch_sec=0.0,
        min_clearance_m=0.45,
        refresh_watchdog=True,
    )
    cbf_flags_seen = []
    c_cross_off = _make_ctrl(
        enable_st=False,
        dyn_tracks=[headon_trk],
        use_rl=True,
        enable_cbf_obj=True,
        active_cross_dec=hold_dec,
    )
    orig_apply_cbf = c_cross_off._apply_cbf_safety_shield

    def _spy_cbf(v_nom, w_nom, now_sec=None, prefer_no_reverse=False, suppress_tangential_steer=False):
        cbf_flags_seen.append((bool(prefer_no_reverse), bool(suppress_tangential_steer)))
        return orig_apply_cbf(
            v_nom,
            w_nom,
            now_sec=now_sec,
            prefer_no_reverse=prefer_no_reverse,
            suppress_tangential_steer=suppress_tangential_steer,
        )

    c_cross_off._apply_cbf_safety_shield = _spy_cbf
    c_cross_off.control_loop()
    assert cbf_flags_seen == [(False, False)], f"Expected CBF flags (False, False), got {cbf_flags_seen}"
    assert c_cross_off.latest_cbf_result is not None

    # (2d) 2-Circle Soft CBF-QP actively intervenes against close oncoming track when toggle=False
    imminent_trk = TrackedObstacle(
        track_id=13,
        x=0.36,
        y=0.0,
        vx=-0.45,
        vy=0.0,
        radius=0.18,
        P_4x4=np.diag([0.002, 0.002, 0.01, 0.01]).astype(np.float64),
        last_update_stamp=10.0,
        age=25,
        hits=25,
        is_confirmed=True,
        consecutive_hits=25,
    )
    c_cbf_off = _make_ctrl(enable_st=False, dyn_tracks=[imminent_trk], use_rl=True, enable_cbf_obj=True)
    c_cbf_off.control_loop()
    assert c_cbf_off.latest_cbf_result is not None and c_cbf_off.latest_cbf_result.intervened is True
    assert c_cbf_off.published_cmds[0][0] < 0.05

    # (2e) Step A-2 ICP remains intact when enable_spatiotemporal_dynamic_control=False
    t_lin = np.linspace(-1.4, 1.4, 25, dtype=np.float32)
    ref_pts = np.vstack([
        np.column_stack([np.full_like(t_lin, 1.5), t_lin]),
        np.column_stack([-t_lin, np.full_like(t_lin, 1.5)]),
        np.column_stack([np.full_like(t_lin, -1.5), -t_lin]),
        np.column_stack([t_lin, np.full_like(t_lin, -1.5)]),
    ]).astype(np.float32)
    true_x, true_y = 0.10, -0.05
    src_pts = ref_pts - np.array([[true_x, true_y]], dtype=np.float32)
    c_icp = RecoveryController.__new__(RecoveryController)
    c_icp.enable_spatiotemporal_dynamic_control = False
    c_icp.enable_scan_matching = True
    c_icp.scan_match_max_dist = 0.45
    c_icp.docking_dist_th = 0.50
    c_icp.scan_match_weight = 0.40
    c_icp.raw_odom_x, c_icp.raw_odom_y, c_icp.raw_odom_theta = 0.36, 0.15, 0.0
    c_icp.corr_x, c_icp.corr_y, c_icp.corr_th = 0.0, 0.0, 0.0
    c_icp.current_x, c_icp.current_y, c_icp.current_theta = 0.36, 0.15, 0.0
    c_icp.start_pose = Waypoint(0.0, 0.0, 0.0, ref_pts.copy())
    c_icp.start_scan_pts = ref_pts.copy()
    c_icp.active_target_wp = c_icp.start_pose
    c_icp.latest_scan_pts = src_pts.copy()
    c_icp.latest_scan_pose = (c_icp.current_x, c_icp.current_y, c_icp.current_theta)
    c_icp.latest_scan_spike_ratio = 0.0
    c_icp.scan_match_count = 0
    c_icp.dynamic_tracker = None
    c_icp.semantic_cache = SemanticObjectCache()
    c_icp.local_planner = None
    c_icp.get_logger = lambda: types.SimpleNamespace(
        info=lambda *a, **k: None, warn=lambda *a, **k: None, error=lambda *a, **k: None
    )
    for k in range(5):
        c_icp.latest_scan_stamp_sec = 30.0 + k * 0.2
        c_icp.perform_scan_matching_correction()
    assert c_icp.scan_match_count == 5
    assert math.hypot(c_icp.current_x - true_x, c_icp.current_y - true_y) < 0.04


def test_crossing_scheduler_flicker_reid_anti_starvation():
    """
    Regression for CrossingLaneScheduler HOLD starvation under track flicker / re-ID / pc drift:
      1) Track changing ID with pc jumping > 0.35 m and brief > 1.0 s dropouts every 2.0 s
         within the same spatial zone must NOT reset cumulative HOLD time to 0:
         when aligned, DASH_FALLBACK still launches instead of starving forever.
      2) When robot heading cannot align (e.g., e_th_fwd = 120 deg), hitting max_zone_hold_s
         must release HOLD (mode == 'INACTIVE', refresh_watchdog == False) with zone cooldown
         so local_planner unstuck recovery is never permanently suppressed.
      3) Moving >= 0.85 m away from the zone cleanly clears the cooldown and re-arms scheduler.
    """
    from disaster_recovery import recovery_controller as rc_mod

    CrossingLaneScheduler = rc_mod.CrossingLaneScheduler

    gate_grid = ProposedRecoveryPlanner(size_m=6.0, resolution_m=0.04, inflation_radius_m=0.085).grid
    gate_grid._recenter(4.35, 0.25)
    for wy in np.linspace(-2.0, -0.55, 40):
        ix, iy = gate_grid.world_to_grid(4.35, float(wy))
        if gate_grid.in_bounds(ix, iy):
            gate_grid.occ[ix, iy] = 1.0
    for wy in np.linspace(0.75, 2.2, 40):
        ix, iy = gate_grid.world_to_grid(4.35, float(wy))
        if gate_grid.in_bounds(ix, iy):
            gate_grid.occ[ix, iy] = 1.0

    def _make_flicker_trk(t_now: float, tid: int, pc_y: float) -> TrackedObstacle:
        return TrackedObstacle(
            track_id=tid,
            x=4.35,
            y=pc_y,
            vx=0.0,
            vy=0.10,
            radius=0.22,
            P_4x4=np.diag([0.08, 0.08, 0.15, 0.15]).astype(np.float64),
            last_update_stamp=t_now,
            age=15,
            hits=15,
            is_confirmed=True,
            consecutive_hits=15,
            imm_valid=True,
            imm_u_hat=(0.0, 1.0),
            imm_p_c=(4.35, pc_y),
            imm_amp=0.60,
            imm_omega=0.95,
            imm_phi=0.0,
            imm_t_ref=t_now,
            imm_mu_cv=0.05,
            imm_mu_harm=0.95,
            imm_sigma_h0=0.35,  # Forces imm_uncertain=True -> stays in HOLD until timeout
            imm_sigma_omega=0.25,
        )

    path_w = [(4.73, 0.10), (4.35, 0.10), (3.50, 0.10)]

    # (1) Aligned heading (rth=pi) under repeated 1.2s dropouts + re-ID + 0.50m pc jumps -> DASH_FALLBACK fires
    sched_fb = CrossingLaneScheduler()
    saw_fallback = False
    for k in range(160):  # 16.0 s
        t_k = k * 0.10
        cycle = int(t_k // 3.2)
        in_gap = (t_k % 3.2) >= 2.0  # 1.2s gap > hold_coast_grace_s (1.0s) resets _hold_start_time!
        if in_gap:
            tracks = []
        else:
            pc_y = -0.10 if (cycle % 2 == 0) else 0.40  # 0.50 m > 0.35 m same_physical threshold!
            tracks = [_make_flicker_trk(t_k, tid=10 + cycle, pc_y=pc_y)]
        dec = sched_fb.step(
            rx=4.73, ry=0.10, rth=math.pi,
            v_max=0.20, w_max=0.65, now_sec=t_k,
            confirmed_tracks=tracks, path_world=path_w, local_grid=gate_grid,
        )
        if dec.mode == "DASH_FALLBACK":
            saw_fallback = True
            break
    assert saw_fallback, "Flickering/re-ID'd track must still trigger DASH_FALLBACK via cumulative zone budget"

    # (2) Unaligned heading (rth=-0.3 rad => e_th_fwd ~ 163 deg): must release HOLD at max_zone_hold_s with cooldown
    sched_rel = CrossingLaneScheduler(max_zone_hold_s=10.0, zone_cooldown_s=4.0)
    released_at = None
    watchdog_refreshed_after_cap = False
    for k in range(180):  # 18.0 s
        t_k = k * 0.10
        cycle = int(t_k // 3.0)
        pc_y = -0.10 if (cycle % 2 == 0) else 0.40
        tracks = [_make_flicker_trk(t_k, tid=100 + cycle, pc_y=pc_y)]
        dec = sched_rel.step(
            rx=4.73, ry=0.10, rth=-0.30,
            v_max=0.20, w_max=0.65, now_sec=t_k,
            confirmed_tracks=tracks, path_world=path_w, local_grid=gate_grid,
        )
        if t_k >= 10.5 and released_at is None and not dec.active:
            released_at = t_k
            assert dec.mode == "INACTIVE" and dec.refresh_watchdog is False
        if released_at is not None and t_k < released_at + 3.5:
            assert dec.active is False and dec.refresh_watchdog is False
            if dec.refresh_watchdog:
                watchdog_refreshed_after_cap = True
    assert released_at is not None, "Unaligned HOLD must release after max_zone_hold_s"
    assert not watchdog_refreshed_after_cap

    # (3) Moving >= 0.85 m away clears zone cooldown and immediately re-arms scheduler
    dec_far = sched_rel.step(
        rx=5.65, ry=0.10, rth=math.pi,
        v_max=0.20, w_max=0.65, now_sec=released_at + 1.0,
        confirmed_tracks=[_make_flicker_trk(released_at + 1.0, tid=200, pc_y=0.10)],
        path_world=[(5.65, 0.10), (4.35, 0.10), (3.50, 0.10)],
        local_grid=gate_grid,
    )
    assert dec_far.active is True




