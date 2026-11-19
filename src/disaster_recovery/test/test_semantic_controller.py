#!/usr/bin/env python3
"""
Unit & Integration Tests for Phase 3 Semantic-Aware Recovery Controller & Context-RL Policy:
1. 1-channel (688) vs 3-channel (976) ContextRLStudentPolicy ONNX input shape auto-detection
   and 100% identical 1-channel behavior when patch_channels == 1.
2. Exact numerical equivalence of ContextRLStudentPolicy.project_semantic_channels (Ch 1 / Ch 2
   12x12 ego-centric grid projection) against VectorDisasterEnv._compute_semantic_channels in
   scripts/host/rl_procedural_env.py.
3. SemanticObjectCache deduplication, local gas hazard caching, and persistence across
   communication/heartbeat loss.
4. use_semantic_rl_policy ON/OFF switch, semantic_rl_onnx_path resolution & graceful fallback,
   and recovery_params.yaml / recovery_params_real.yaml configuration checks.
"""

import json
import math
import os
import sys
import tempfile

import numpy as np
import pytest
import torch
import yaml

PKG_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WS_ROOT = os.path.abspath(os.path.join(PKG_SRC_DIR, "..", ".."))
HOST_SCRIPTS_DIR = os.path.join(WS_ROOT, "scripts", "host")
for p in (HOST_SCRIPTS_DIR, PKG_SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from disaster_recovery.recovery_controller import (  # noqa: E402
    SemanticObjectCache,
    compute_rl_blend_weight,
    resolve_semantic_rl_asset,
)
from disaster_recovery.rl_context_policy import (  # noqa: E402
    ContextRLStudentPolicy,
    DistilledEdgeStudent,
    FLAT_INPUT_DIM,
    HISTORY_LEN,
    PATCH_SIZE,
    SEMANTIC_FLAT_INPUT_DIM,
    STEP_FEAT_DIM,
    TB3_MAX_V,
    TB3_MAX_W,
    TB3_MIN_V,
    TemporalContextActorCritic,
    compute_flat_input_dim,
)
from rl_procedural_env import VectorDisasterEnv  # noqa: E402


def _export_temp_student_onnx(out_path: str, patch_channels: int) -> None:
    student = DistilledEdgeStudent(patch_channels=patch_channels).eval()
    flat_dim = compute_flat_input_dim(patch_channels)
    dummy = torch.randn(1, flat_dim, dtype=torch.float32)
    torch.onnx.export(
        student,
        dummy,
        out_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["obs_flat"],
        output_names=["cmd_vel"],
        dynamic_axes={"obs_flat": {0: "batch"}, "cmd_vel": {0: "batch"}},
    )


def test_1ch_and_3ch_models_and_onnx_autodetection():
    """Verify 1ch (688) and 3ch (976) architecture shapes and ContextRLStudentPolicy auto-detection."""
    assert compute_flat_input_dim(1) == FLAT_INPUT_DIM == 688
    assert compute_flat_input_dim(3) == SEMANTIC_FLAT_INPUT_DIM == 976

    # Verify TemporalContextActorCritic supports patch_channels=1 and patch_channels=3
    obs_hist = torch.randn(4, HISTORY_LEN, STEP_FEAT_DIM, dtype=torch.float32)
    patch_1ch = torch.rand(4, 1, PATCH_SIZE, PATCH_SIZE, dtype=torch.float32)
    patch_3ch = torch.rand(4, 3, PATCH_SIZE, PATCH_SIZE, dtype=torch.float32)

    teacher_1ch = TemporalContextActorCritic(patch_channels=1).eval()
    teacher_3ch = TemporalContextActorCritic(patch_channels=3).eval()
    act1, val1 = teacher_1ch(obs_hist, patch_1ch)
    act3, val3 = teacher_3ch(obs_hist, patch_3ch)
    assert act1.shape == (4, 2) and val1.shape == (4,)
    assert act3.shape == (4, 2) and val3.shape == (4,)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_1ch = os.path.join(tmpdir, "student_1ch.onnx")
        onnx_3ch = os.path.join(tmpdir, "student_3ch.onnx")
        _export_temp_student_onnx(onnx_1ch, patch_channels=1)
        _export_temp_student_onnx(onnx_3ch, patch_channels=3)

        pol_1ch = ContextRLStudentPolicy(onnx_path=onnx_1ch)
        pol_3ch = ContextRLStudentPolicy(onnx_path=onnx_3ch)

        assert pol_1ch.patch_channels == 1
        assert pol_1ch.flat_input_dim == 688
        assert pol_3ch.patch_channels == 3
        assert pol_3ch.flat_input_dim == 976

        scan_ranges = [2.0] * 360
        scan_ranges[0] = 0.45
        sem_obs = [(0.50, 0.15, 0.22)]
        sem_haz = [(0.65, -0.25, 0.55)]

        # 1ch policy must keep 100% identical behavior even if semantic lists are passed
        pol_1ch.reset()
        patch_a = pol_1ch.observe(scan_ranges, 0.0, 0.0, 0.0, 1.2, 0.0, 0.1, 0.0, 0.1, 0.0, step_norm=0.2)
        v_a, w_a = pol_1ch.predict(scan_ranges, 0.0, 0.0, 0.0, 1.2, 0.0, 0.1, 0.0, 0.1, 0.0, step_norm=0.3)

        pol_1ch.reset()
        patch_b = pol_1ch.observe(
            scan_ranges, 0.0, 0.0, 0.0, 1.2, 0.0, 0.1, 0.0, 0.1, 0.0,
            step_norm=0.2, semantic_obstacles=sem_obs, semantic_hazards=sem_haz,
        )
        v_b, w_b = pol_1ch.predict(
            scan_ranges, 0.0, 0.0, 0.0, 1.2, 0.0, 0.1, 0.0, 0.1, 0.0,
            step_norm=0.3, semantic_obstacles=sem_obs, semantic_hazards=sem_haz,
        )
        assert patch_a.shape == (12, 12)
        assert np.array_equal(patch_a, patch_b)
        assert v_a == v_b and w_a == w_b

        # 3ch policy produces (3, 12, 12) patch and populates Ch 1 & Ch 2
        patch_3 = pol_3ch.observe(
            scan_ranges, 0.0, 0.0, 0.0, 1.2, 0.0, 0.1, 0.0, 0.1, 0.0,
            step_norm=0.2, semantic_obstacles=sem_obs, semantic_hazards=sem_haz,
        )
        assert patch_3.shape == (3, 12, 12)
        assert np.allclose(patch_3[0], patch_a)
        assert float(patch_3[1].max()) > 0.5
        assert float(patch_3[2].max()) > 0.5

        v_3, w_3 = pol_3ch.predict(
            scan_ranges, 0.0, 0.0, 0.0, 1.2, 0.0, 0.1, 0.0, 0.1, 0.0,
            step_norm=0.3, semantic_obstacles=sem_obs, semantic_hazards=sem_haz,
        )
        assert TB3_MIN_V <= v_3 <= TB3_MAX_V
        assert -TB3_MAX_W <= w_3 <= TB3_MAX_W


def test_ch1_ch2_grid_projection_matches_rl_procedural_env_exactly():
    """Verify ContextRLStudentPolicy.project_semantic_channels matches rl_procedural_env.py formula to <1e-6."""
    env = VectorDisasterEnv(
        num_envs=1,
        track="sim",
        device=torch.device("cpu"),
        mode="eval",
        patch_channels=3,
        use_semantic_channels=True,
        modality_dropout_prob=0.0,
        enable_semantic_noise=False,
        semantic_reg_delay_steps=0,
    )
    env.reset_envs()

    rx, ry, rth = 1.25, -0.85, 0.62
    env.rx[0] = rx
    env.ry[0] = ry
    env.rth[0] = rth

    # Clear all env semantic slots and populate known obstacles & hazards
    env.low_obs_active.fill_(False)
    env.low_obs.zero_()
    env.plinth.zero_()
    env.hazard_active.fill_(False)
    env.hazard_zones.zero_()
    env.low_obs_reg_delay.zero_()
    env.low_obs_seen_ticks.fill_(10)
    env.hazard_reg_delay.zero_()
    env.hazard_seen_ticks.fill_(10)

    sem_obs = [
        (1.65, -0.60, 0.22),
        (0.95, -1.10, 0.18),
        (1.40, -0.35, 0.25),
    ]
    sem_haz = [
        (1.70, -0.95, 0.55),
        (1.05, -0.45, 0.48),
    ]

    for i, (ox, oy, rad) in enumerate(sem_obs):
        env.low_obs_active[0, i] = True
        env.low_obs[0, i, 0] = ox
        env.low_obs[0, i, 1] = oy
        env.low_obs[0, i, 2] = rad

    for j, (hx, hy, rad) in enumerate(sem_haz):
        env.hazard_active[0, j] = True
        env.hazard_zones[0, j, 0] = hx
        env.hazard_zones[0, j, 1] = hy
        env.hazard_zones[0, j, 2] = rad

    env_ch1, env_ch2 = env._compute_semantic_channels()
    env_ch1_np = env_ch1[0, 0].cpu().numpy()
    env_ch2_np = env_ch2[0, 0].cpu().numpy()

    pol_ch1, pol_ch2 = ContextRLStudentPolicy.project_semantic_channels(
        rx=rx,
        ry=ry,
        rth=rth,
        semantic_obstacles=sem_obs,
        semantic_hazards=sem_haz,
    )

    assert pol_ch1.shape == (12, 12) and pol_ch2.shape == (12, 12)
    assert np.allclose(pol_ch1, env_ch1_np, atol=1e-6), f"Ch1 mismatch max={np.abs(pol_ch1 - env_ch1_np).max()}"
    assert np.allclose(pol_ch2, env_ch2_np, atol=1e-6), f"Ch2 mismatch max={np.abs(pol_ch2 - env_ch2_np).max()}"


def test_semantic_cache_dedup_persistence_and_local_gas():
    """Verify SemanticObjectCache deduplicates objects, records local gas hazards, and persists across comm loss."""
    cache = SemanticObjectCache(dedup_radius_m=0.45, default_ch1_radius_m=0.22, default_ch2_radius_m=0.55)

    msg_json_1 = json.dumps({
        "frame_id": "odom",
        "objects": [
            {"marker_id": 1, "class_name": "chair", "x": 1.20, "y": -0.40, "radius": 0.22, "channel": 1, "conf": 0.82},
            {"marker_id": 2, "class_name": "person", "x": 2.10, "y": 0.50, "radius": 0.55, "channel": 2, "conf": 0.91},
        ],
    })
    assert cache.update_from_json(msg_json_1) == 2
    assert len(cache) == 2

    # Second update shifts marker #1 slightly and adds a duplicate chair within 0.45m without marker_id
    msg_json_2 = json.dumps({
        "frame_id": "odom",
        "objects": [
            {"marker_id": 1, "class_name": "chair", "x": 1.24, "y": -0.38, "radius": 0.22, "channel": 1, "conf": 0.88},
            {"class_name": "chair", "x": 1.26, "y": -0.36, "radius": 0.24, "channel": 1, "conf": 0.90},
        ],
    })
    cache.update_from_json(msg_json_2)
    assert len(cache) == 2  # Deduped!

    # Local onboard gas hazard when /gas/level >= gas_threshold after comm loss
    cache.record_local_gas_hazard(x=0.75, y=-0.15, gas_level=0.65)
    assert len(cache) == 3

    # Simulate communication loss (no more messages or empty/corrupt message): cached objects persist!
    cache.update_from_json("")
    ch1_list, ch2_list = cache.get_channel_tuples()
    assert len(ch1_list) == 1
    assert len(ch2_list) == 2
    assert abs(ch1_list[0][2] - 0.24) < 1e-6
    assert any(math.hypot(hx - 2.10, hy - 0.50) < 1e-4 for hx, hy, _ in ch2_list)
    assert any(math.hypot(hx - 0.75, hy - (-0.15)) < 1e-4 for hx, hy, _ in ch2_list)


def test_use_semantic_rl_policy_switch_and_yaml_configs():
    """Verify use_semantic_rl_policy ON/OFF switch, ONNX asset fallback, and YAML defaults."""
    cfg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))

    for yaml_name in ("recovery_params.yaml", "recovery_params_real.yaml"):
        yaml_path = os.path.join(cfg_dir, yaml_name)
        assert os.path.exists(yaml_path)
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        rc_params = data["recovery_controller"]["ros__parameters"]
        assert "use_semantic_rl_policy" in rc_params
        assert rc_params["use_semantic_rl_policy"] is False
        assert "semantic_rl_onnx_path" in rc_params
        assert rc_params["semantic_rl_onnx_path"] == ""

    with tempfile.TemporaryDirectory() as tmpdir:
        fb_1ch_path = os.path.join(tmpdir, "sim_policy_student.onnx")
        _export_temp_student_onnx(fb_1ch_path, patch_channels=1)

        # 1. When semantic v3 ONNX is not yet in tmpdir, resolve_semantic_rl_asset falls back gracefully to 1ch
        resolved_fb, is_sem_fb = resolve_semantic_rl_asset(
            semantic_onnx_path="",
            fallback_onnx_path="",
            search_dirs=[tmpdir],
        )
        assert resolved_fb == fb_1ch_path
        assert is_sem_fb is False

        # 2. When sim_policy_student_semantic_v3.onnx is present, it is selected automatically
        sem_3ch_path = os.path.join(tmpdir, "sim_policy_student_semantic_v3.onnx")
        _export_temp_student_onnx(sem_3ch_path, patch_channels=3)
        resolved_sem, is_sem = resolve_semantic_rl_asset(
            semantic_onnx_path="",
            fallback_onnx_path="",
            search_dirs=[tmpdir],
        )
        assert resolved_sem == sem_3ch_path
        assert is_sem is True

        # 3. Real TB3 track auto-infers real_tb3_policy_student_semantic_v3.onnx when context_rl_onnx_path is real_tb3
        real_sem_path = os.path.join(tmpdir, "real_tb3_policy_student_semantic_v3.onnx")
        _export_temp_student_onnx(real_sem_path, patch_channels=3)
        resolved_real, is_sem_real = resolve_semantic_rl_asset(
            semantic_onnx_path="",
            fallback_onnx_path="real_tb3_policy_student.onnx",
            search_dirs=[tmpdir],
        )
        assert resolved_real == real_sem_path
        assert is_sem_real is True

    # 4. Verify compute_rl_blend_weight preserves Ch0 safety while boosting open-corridor blend for nearby semantic objects
    assert compute_rl_blend_weight(0.24, 1.2, False, min_side=0.40, has_nearby_semantic=True) == 0.0
    assert compute_rl_blend_weight(0.60, 1.2, False, min_side=0.16, has_nearby_semantic=True) == 0.0
    w_base = compute_rl_blend_weight(0.60, 1.10, False, min_side=0.45, has_nearby_semantic=False)
    w_sem = compute_rl_blend_weight(0.60, 1.10, False, min_side=0.45, has_nearby_semantic=True)
    assert abs(w_base - 0.70) < 1e-6
    assert w_sem > w_base
