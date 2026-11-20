#!/usr/bin/env python3
"""
Unit Tests for Phase 3 Semantic 3-Channel RL Environment, Modality Dropout,
Perception Noise, Teacher/Student 1ch->3ch Warm-Start, and Semantic v3 ONNX Export.
"""

import os
import sys
import tempfile

import numpy as np
import onnxruntime as ort
import pytest
import torch

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.abspath(os.path.join(TEST_DIR, "..", "..", ".."))
HOST_SCRIPTS_DIR = os.path.join(WS_ROOT, "scripts", "host")
PKG_SRC_DIR = os.path.join(WS_ROOT, "src", "disaster_recovery")
for p in (HOST_SCRIPTS_DIR, PKG_SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from rl_procedural_env import (  # noqa: E402
    BASE_REGIME_NAMES,
    HISTORY_LEN,
    NUM_BASE_REGIMES,
    NUM_REGIMES,
    PATCH_SIZE,
    REGIME_NAMES,
    SEMANTIC_REGIME_NAMES,
    STEP_FEAT_DIM,
    TB3_MAX_V,
    TB3_MAX_W,
    TB3_MIN_V,
    VectorDisasterEnv,
)
from train_phase2_v100 import (  # noqa: E402
    ContextRLStudent,
    DistilledEdgeStudent,
    FLAT_INPUT_DIM,
    FrozenBackboneSemanticStudent,
    FrozenBackboneSemanticTeacher,
    SEMANTIC_FLAT_INPUT_DIM,
    SEMANTIC_PATCH_CHANNELS,
    TemporalContextActorCritic,
    _flatten_obs,
    compute_flat_input_dim,
    export_student_onnx,
    resolve_student_onnx_filename,
    warm_start_1ch_to_3ch,
    warm_start_teacher_from_phase2b,
)


def test_3ch_obs_shape_and_zero_nans_across_all_scenarios():
    """Verify (B, 3, 12, 12) patch shape, (B, 16, 34) history shape, and zero NaNs across all 8 regimes."""
    assert len(BASE_REGIME_NAMES) == 6
    assert len(SEMANTIC_REGIME_NAMES) == 2
    assert len(REGIME_NAMES) == 8
    assert "low_obstacle_field" in REGIME_NAMES
    assert "gas_victim_corridor" in REGIME_NAMES

    device = torch.device("cpu")
    num_envs = 32
    for track in ("sim", "real_tb3"):
        env = VectorDisasterEnv(
            num_envs=num_envs,
            track=track,
            device=device,
            max_steps=64,
            mode="train",
            base_seed=12345,
            patch_channels=3,
            use_semantic_channels=True,
        )
        forced_regs = [i % NUM_REGIMES for i in range(num_envs)]
        obs_hist, local_patch = env.reset_envs(forced_regimes=forced_regs)

        assert obs_hist.shape == (num_envs, HISTORY_LEN, STEP_FEAT_DIM)
        assert local_patch.shape == (num_envs, 3, PATCH_SIZE, PATCH_SIZE)
        assert torch.isfinite(obs_hist).all()
        assert torch.isfinite(local_patch).all()
        assert float(local_patch.min().item()) >= 0.0
        assert float(local_patch.max().item()) <= 1.0

        # Ensure all 8 regimes are present in batch
        assert set(env.regime.tolist()) == set(range(NUM_REGIMES))

        for step_idx in range(35):
            # Mix normal actions and edge-case actions
            act = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
            act[:, 0] = 0.16 if step_idx % 5 != 0 else -0.06
            act[:, 1] = 0.25 * float(np.sin(step_idx * 0.4))
            obs_hist, local_patch, rewards, dones, info = env.step(act)

            assert obs_hist.shape == (num_envs, HISTORY_LEN, STEP_FEAT_DIM)
            assert local_patch.shape == (num_envs, 3, PATCH_SIZE, PATCH_SIZE)
            assert rewards.shape == (num_envs,)
            assert dones.shape == (num_envs,)
            assert torch.isfinite(obs_hist).all(), f"NaN/Inf in obs_hist at step {step_idx}"
            assert torch.isfinite(local_patch).all(), f"NaN/Inf in local_patch at step {step_idx}"
            assert torch.isfinite(rewards).all(), f"NaN/Inf in rewards at step {step_idx}"
            assert float(local_patch.min().item()) >= 0.0
            assert float(local_patch.max().item()) <= 1.0

            done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            if len(done_ids) > 0:
                obs_hist, local_patch = env.reset_envs(env_ids=done_ids)
                assert torch.isfinite(obs_hist).all()
                assert torch.isfinite(local_patch).all()


def test_ch0_backward_compatibility_and_exact_equivalence():
    """Verify Ch 0 is bit-for-bit identical when semantic channels are disabled (patch_channels=1 vs 3)."""
    device = torch.device("cpu")
    num_envs = 12
    seeds = [10000 + i for i in range(num_envs)]
    base_regs = [i % NUM_BASE_REGIMES for i in range(num_envs)]

    torch.manual_seed(777)
    env_1ch = VectorDisasterEnv(
        num_envs=num_envs,
        track="sim",
        device=device,
        mode="eval",
        base_seed=50000,
        patch_channels=1,
        use_semantic_channels=False,
    )
    h1, p1 = env_1ch.reset_envs(eval_seeds=seeds, forced_regimes=base_regs)
    assert p1.shape == (num_envs, 1, PATCH_SIZE, PATCH_SIZE)

    torch.manual_seed(777)
    env_3ch_zero = VectorDisasterEnv(
        num_envs=num_envs,
        track="sim",
        device=device,
        mode="eval",
        base_seed=50000,
        patch_channels=3,
        use_semantic_channels=False,
    )
    h3_zero, p3_zero = env_3ch_zero.reset_envs(eval_seeds=seeds, forced_regimes=base_regs)
    assert p3_zero.shape == (num_envs, 3, PATCH_SIZE, PATCH_SIZE)
    assert torch.all(p3_zero[:, 1:, :, :] == 0.0)
    assert torch.allclose(h1, h3_zero, atol=1e-6)
    assert torch.allclose(p1[:, 0], p3_zero[:, 0], atol=1e-6)

    torch.manual_seed(777)
    env_3ch_sem = VectorDisasterEnv(
        num_envs=num_envs,
        track="sim",
        device=device,
        mode="eval",
        base_seed=50000,
        patch_channels=3,
        use_semantic_channels=True,
        include_semantic_regimes=False,
    )
    h3_sem, p3_sem = env_3ch_sem.reset_envs(eval_seeds=seeds, forced_regimes=base_regs)
    assert torch.allclose(h1, h3_sem, atol=1e-6)
    assert torch.allclose(p1[:, 0], p3_sem[:, 0], atol=1e-6)

    # Step all three environments with identical actions and verify Ch 0 & obs_hist equivalence
    act = torch.tensor([[0.15, 0.10]] * num_envs, dtype=torch.float32, device=device)
    for _ in range(15):
        torch.manual_seed(888)
        h1, p1, r1, d1, _ = env_1ch.step(act)
        torch.manual_seed(888)
        h3_zero, p3_zero, r3_zero, d3_zero, _ = env_3ch_zero.step(act)
        torch.manual_seed(888)
        h3_sem, p3_sem, r3_sem, d3_sem, _ = env_3ch_sem.step(act)

        assert torch.all(p3_zero[:, 1:, :, :] == 0.0)
        assert torch.allclose(h1, h3_zero, atol=1e-6)
        assert torch.allclose(p1[:, 0], p3_zero[:, 0], atol=1e-6)
        assert torch.allclose(r1, r3_zero, atol=1e-6)
        assert torch.equal(d1, d3_zero)
        assert torch.allclose(p1[:, 0], p3_sem[:, 0], atol=1e-6)


def test_modality_dropout_zeroes_ch1_and_ch2_cleanly():
    """Verify Modality Dropout zeroes Ch 1 & Ch 2 at ~25-30% rate while leaving Ch 0 intact."""
    device = torch.device("cpu")
    num_envs = 128

    # 1. 100% dropout -> Ch 1 & Ch 2 must be strictly zero, Ch 0 non-zero
    env_full_drop = VectorDisasterEnv(
        num_envs=16,
        track="sim",
        device=device,
        mode="train",
        base_seed=2026,
        patch_channels=3,
        use_semantic_channels=True,
        modality_dropout_prob=1.0,
        enable_semantic_noise=False,
    )
    _, patch_drop = env_full_drop.reset_envs(forced_regimes=[6, 7] * 8)
    assert torch.all(env_full_drop.modality_dropped)
    assert torch.all(patch_drop[:, 1:, :, :] == 0.0)
    assert float(patch_drop[:, 0, :, :].sum().item()) > 0.0

    # 2. 0% dropout -> Ch 1 (regime 6) and Ch 2 (regime 7) must be active
    env_no_drop = VectorDisasterEnv(
        num_envs=16,
        track="sim",
        device=device,
        mode="train",
        base_seed=2026,
        patch_channels=3,
        use_semantic_channels=True,
        modality_dropout_prob=0.0,
        enable_semantic_noise=False,
    )
    _, patch_nodrop = env_no_drop.reset_envs(forced_regimes=[6, 7] * 8)
    assert not bool(env_no_drop.modality_dropped.any())
    assert float(patch_nodrop[0::2, 1, :, :].max().item()) > 0.5  # Regime 6 Ch 1 active
    assert float(patch_nodrop[1::2, 2, :, :].max().item()) > 0.5  # Regime 7 Ch 2 active

    # 3. Default ~28% dropout in training mode -> verify empirical rate in [20%, 36%] and clean zeroing
    env_stoch = VectorDisasterEnv(
        num_envs=num_envs,
        track="sim",
        device=device,
        mode="train",
        base_seed=3030,
        patch_channels=3,
        use_semantic_channels=True,
        modality_dropout_prob=0.28,
        enable_semantic_noise=False,
    )
    env_stoch.reset_envs(forced_regimes=[6, 7] * (num_envs // 2))
    drop_counts = 0
    total_counts = 0
    act = torch.tensor([[0.12, 0.0]] * num_envs, dtype=torch.float32, device=device)

    for _ in range(25):
        _, patch_s, _, _, info = env_stoch.step(act)
        dropped_mask = info["modality_dropped"]
        drop_counts += int(dropped_mask.sum().item())
        total_counts += num_envs
        if bool(dropped_mask.any()):
            assert torch.all(patch_s[dropped_mask, 1:, :, :] == 0.0)

    empirical_rate = drop_counts / float(total_counts)
    assert 0.20 <= empirical_rate <= 0.36, f"Expected ~28% modality dropout, got {empirical_rate:.3f}"


def test_low_obstacle_field_scenario_and_contact_penalty():
    """Verify Regime 6 low obstacles are invisible to 2D LiDAR (Ch 0), visible in Ch 1, and penalize contact."""
    device = torch.device("cpu")
    env = VectorDisasterEnv(
        num_envs=2,
        track="sim",
        device=device,
        mode="eval",
        base_seed=42,
        patch_channels=3,
        use_semantic_channels=True,
        modality_dropout_prob=0.0,
        enable_semantic_noise=False,
    )
    env.reset_envs(forced_regime=6)
    assert int(env.regime[0].item()) == 6
    assert bool(env.low_obs_active[0].any())

    # Configure a controlled comparison:
    # Env 0 has a sub-LiDAR low obstacle directly ahead at (0.32m, 0.0m, r=0.15m).
    # Env 1 has no low obstacle ahead.
    env.box_active.fill_(False)
    env.cyl_active.fill_(False)
    env.plinth.fill_(0.0)
    env.low_obs_active.fill_(False)
    env.hazard_active.fill_(False)
    env.lidar_noise_std.fill_(0.0)
    env.lidar_drop_prob.fill_(0.0)
    env.slip_v.fill_(1.0)
    env.slip_w.fill_(1.0)
    env.turn_bias.fill_(0.0)
    env.act_lag.fill_(0)

    env.rx.fill_(0.0)
    env.ry.fill_(0.0)
    env.rth.fill_(0.0)
    env.odom_x.fill_(0.0)
    env.odom_y.fill_(0.0)
    env.odom_th.fill_(0.0)
    env.goal_x.fill_(2.0)
    env.goal_y.fill_(0.0)
    env.nav_wp[:, 0, 0] = 2.0
    env.nav_wp[:, 0, 1] = 0.0
    env.nav_wp[:, 1, 0] = 2.0
    env.nav_wp[:, 1, 1] = 0.0
    env.prev_dist.fill_(2.0)
    env.min_dist_seen.fill_(2.0)

    env.low_obs[0, 0] = torch.tensor([0.32, 0.0, 0.15], device=device)
    env.low_obs_active[0, 0] = True

    feat, patch = env._compute_obs_step()
    # Low obstacle must NOT appear in 2D LiDAR Ch 0 (arena walls are >1m away, outside 0.9m patch)
    assert float(patch[0, 0].max().item()) == 0.0
    # Low obstacle MUST appear strongly in semantic Ch 1 ahead of the robot (ix >= 6)
    assert float(patch[0, 1, 6:, :].max().item()) > 0.85
    assert float(patch[1, 1].max().item()) == 0.0

    # Drive both robots straight forward into (0.32, 0.0): Env 0 hits low obstacle and incurs penalty
    act = torch.tensor([[0.20, 0.0], [0.20, 0.0]], dtype=torch.float32, device=device)
    contact_seen = False
    total_r0 = 0.0
    total_r1 = 0.0
    for _ in range(10):
        _, _, rew, _, info = env.step(act)
        total_r0 += float(rew[0].item())
        total_r1 += float(rew[1].item())
        if bool(info["low_obs_contact"][0].item()):
            contact_seen = True
            assert float(info["low_contact_penalty"][0].item()) <= -0.60

    assert contact_seen, "Robot 0 should have contacted the low obstacle dead ahead"
    assert not bool(info["ever_hit_low_obs"][1].item())
    assert total_r0 < total_r1 - 3.0, f"Expected contact penalty to reduce return ({total_r0:.2f} vs {total_r1:.2f})"


def test_gas_victim_corridor_scenario_and_zone_rewards():
    """Verify Regime 7 gas & victim zones appear in Ch 2 soft field and apply zone entry penalty / clearance reward."""
    device = torch.device("cpu")
    env = VectorDisasterEnv(
        num_envs=2,
        track="sim",
        device=device,
        mode="eval",
        base_seed=99,
        patch_channels=3,
        use_semantic_channels=True,
        modality_dropout_prob=0.0,
        enable_semantic_noise=False,
    )
    env.reset_envs(forced_regime=7)
    assert int(env.regime[0].item()) == 7
    assert bool(env.hazard_active[0].any())

    # Controlled setup: place a gas hazard zone at (0.30, 0.0, r=0.25)
    # Env 0 drives straight through the hazard zone (y=0.0); Env 1 drives parallel outside the zone (y=0.42).
    env.box_active.fill_(False)
    env.cyl_active.fill_(False)
    env.low_obs_active.fill_(False)
    env.hazard_active.fill_(False)
    env.slip_v.fill_(1.0)
    env.slip_w.fill_(1.0)
    env.turn_bias.fill_(0.0)
    env.act_lag.fill_(0)

    env.rx[:] = torch.tensor([0.0, 0.0], device=device)
    env.ry[:] = torch.tensor([0.0, 0.42], device=device)
    env.rth.fill_(0.0)
    env.odom_x[:] = env.rx.clone()
    env.odom_y[:] = env.ry.clone()
    env.odom_th.fill_(0.0)
    env.goal_x[:] = torch.tensor([2.0, 2.0], device=device)
    env.goal_y[:] = torch.tensor([0.0, 0.42], device=device)
    env.nav_wp[0] = torch.tensor([[2.0, 0.0], [2.0, 0.0]], device=device)
    env.nav_wp[1] = torch.tensor([[2.0, 0.42], [2.0, 0.42]], device=device)
    env.prev_dist.fill_(2.0)
    env.min_dist_seen.fill_(2.0)

    env.hazard_zones[:, 0] = torch.tensor([[0.30, 0.0, 0.25, 0.0], [0.30, 0.0, 0.25, 1.0]], device=device)
    env.hazard_active[:, 0] = True

    _, patch = env._compute_obs_step()
    # Soft safety field must be visible in Ch 2 with smooth intermediate values (0 < val < 1) as well as core 1.0
    ch2_0 = patch[0, 2]
    assert float(ch2_0.max().item()) > 0.90
    assert bool(((ch2_0 > 0.05) & (ch2_0 < 0.95)).any()), "Ch 2 should form a soft radial safety field"
    assert float(patch[0, 1].max().item()) == 0.0

    act = torch.tensor([[0.20, 0.0], [0.20, 0.0]], dtype=torch.float32, device=device)
    entered_0 = False
    clear_rew_1 = 0.0
    pen_0 = 0.0
    for _ in range(15):
        _, _, _, _, info = env.step(act)
        if bool(info["hazard_zone_entered"][0].item()):
            entered_0 = True
            pen_0 += float(info["hazard_entry_penalty"][0].item())
        clear_rew_1 += float(info["hazard_clearance_reward"][1].item())

    assert entered_0, "Env 0 should have entered the gas hazard zone"
    assert pen_0 < -2.0, f"Expected negative hazard zone entry penalty, got {pen_0}"
    assert not bool(info["ever_entered_hazard"][1].item()), "Env 1 bypassed outside zone"
    assert clear_rew_1 > 0.15, f"Expected positive hazard clearance reward for safe bypass, got {clear_rew_1}"


def test_semantic_perception_noise_injection():
    """Verify registration delay, position jitter, missed detections, and non-blocking false positives."""
    device = torch.device("cpu")

    # 1. Registration delay: with reg_delay_steps=3, marker is withheld on initial reset (tick 1)
    # and appears after >3 ticks in range.
    env_delay = VectorDisasterEnv(
        num_envs=4,
        track="sim",
        device=device,
        mode="train",
        base_seed=555,
        patch_channels=3,
        use_semantic_channels=True,
        modality_dropout_prob=0.0,
        enable_semantic_noise=True,
        semantic_pos_jitter_std=0.0,
        semantic_reg_delay_steps=3,
        semantic_miss_prob=0.0,
        semantic_fp_prob=0.0,
    )
    _, p_init = env_delay.reset_envs(forced_regime=6)
    # On reset, seen_ticks == 1 <= reg_delay (1..3); at least envs with delay >= 1 withhold markers initially
    env_delay.low_obs_reg_delay.fill_(3)
    env_delay.low_obs_seen_ticks.fill_(0)
    _, p_t1 = env_delay._compute_obs_step()  # seen_ticks becomes 1 <= 3
    assert float(p_t1[:, 1].max().item()) == 0.0
    for _ in range(3):
        _, p_t4 = env_delay._compute_obs_step()  # seen_ticks reaches 4 > 3
    assert float(p_t4[:, 1].max().item()) > 0.5

    # 2. Position jitter & false positives: verify non-blocking peripheral placement & zero NaNs
    env_noisy = VectorDisasterEnv(
        num_envs=32,
        track="real_tb3",
        device=device,
        mode="train",
        base_seed=777,
        patch_channels=3,
        use_semantic_channels=True,
        modality_dropout_prob=0.0,
        enable_semantic_noise=True,
        semantic_pos_jitter_std=0.08,
        semantic_reg_delay_steps=0,
        semantic_miss_prob=0.10,
        semantic_fp_prob=1.0,  # Force peripheral false positive every step
    )
    # Empty world with only false positives enabled -> robot center (ix=5..6, iy=5..6) must never be blocked
    env_noisy.reset_envs(forced_regime=0)
    env_noisy.low_obs_active.fill_(False)
    env_noisy.hazard_active.fill_(False)
    env_noisy.plinth.fill_(0.0)
    _, p_fp = env_noisy._compute_obs_step()
    assert torch.isfinite(p_fp).all()
    sem_sum = float((p_fp[:, 1].sum() + p_fp[:, 2].sum()).item())
    assert sem_sum > 0.0, "False positive markers should appear in Ch 1 / Ch 2"
    # Immediate robot footprint center cells (5:7, 5:7) must remain clear of false positives
    assert float(p_fp[:, 1:, 5:7, 5:7].max().item()) == 0.0


def test_warm_start_weight_transfer_1ch_to_3ch_preserves_ch0_outputs():
    """Verify 1-ch -> 3-ch Warm-Start preserves Ch 0 driving policy at t=0 and allows Ch 1/2 gradients."""
    device = torch.device("cpu")
    torch.manual_seed(101)

    # 1. Test Teacher Actor-Critic warm-start (from existing checkpoint if present, else synthetic 1-ch teacher)
    teacher_1ch = TemporalContextActorCritic(hidden_dim=128, patch_channels=1).to(device)
    real_ckpt_path = "/hdd/rl_archives/sim/best_teacher_seed101.pt"
    if os.path.exists(real_ckpt_path):
        ckpt = torch.load(real_ckpt_path, map_location=device)
        teacher_1ch.load_state_dict(ckpt["state_dict"])
    teacher_1ch.eval()

    teacher_3ch = TemporalContextActorCritic(hidden_dim=128, patch_channels=3).to(device)
    ws_meta = warm_start_teacher_from_phase2b(teacher_3ch, teacher_1ch, ch12_init_std=0.0)
    teacher_3ch.eval()

    assert "spatial_conv.0.weight" in ws_meta["expanded_conv_keys"]
    w1 = teacher_1ch.spatial_conv[0].weight.detach()
    w3 = teacher_3ch.spatial_conv[0].weight.detach()
    assert w1.shape == (16, 1, 3, 3)
    assert w3.shape == (16, 3, 3, 3)
    assert torch.allclose(w3[:, 0:1, :, :], w1, atol=1e-7)
    assert torch.all(w3[:, 1:, :, :] == 0.0)

    # Verify exact policy output preservation on arbitrary 3-channel inputs at start of training
    obs_hist = torch.randn((8, HISTORY_LEN, STEP_FEAT_DIM), dtype=torch.float32, device=device)
    patch_3ch = torch.rand((8, 3, PATCH_SIZE, PATCH_SIZE), dtype=torch.float32, device=device)
    patch_1ch = patch_3ch[:, 0:1, :, :]

    with torch.no_grad():
        act_1, val_1 = teacher_1ch(obs_hist, patch_1ch)
        act_3, val_3 = teacher_3ch(obs_hist, patch_3ch)
    assert torch.allclose(act_1, act_3, atol=1e-6)
    assert torch.allclose(val_1, val_3, atol=1e-6)

    # Verify near-zero init (ch12_init_std=1e-4) also preserves exact output when Ch 1 & Ch 2 are zeroed
    teacher_3ch_near = TemporalContextActorCritic(hidden_dim=128, patch_channels=3).to(device)
    teacher_3ch_near.load_warm_start(teacher_1ch, ch12_init_std=1e-4)
    teacher_3ch_near.eval()
    patch_3ch_zero12 = patch_3ch.clone()
    patch_3ch_zero12[:, 1:, :, :] = 0.0
    with torch.no_grad():
        act_3n, val_3n = teacher_3ch_near(obs_hist, patch_3ch_zero12)
    assert torch.allclose(act_1, act_3n, atol=1e-6)
    assert torch.allclose(val_1, val_3n, atol=1e-6)

    # Verify non-zero gradients flow immediately into Ch 1 & Ch 2 conv weights on step 1
    teacher_3ch.train()
    act_out, val_out = teacher_3ch(obs_hist, patch_3ch)
    loss = act_out.sum() + val_out.sum()
    loss.backward()
    grad_w3 = teacher_3ch.spatial_conv[0].weight.grad
    assert grad_w3 is not None
    assert float(grad_w3[:, 1, :, :].abs().sum().item()) > 0.0
    assert float(grad_w3[:, 2, :, :].abs().sum().item()) > 0.0

    # 2. Test ContextRLStudent / DistilledEdgeStudent 1ch -> 3ch warm-start
    student_1ch = ContextRLStudent(patch_channels=1).to(device).eval()
    student_3ch = ContextRLStudent(patch_channels=3).to(device)
    warm_start_1ch_to_3ch(student_3ch, student_1ch, ch12_init_std=0.0)
    student_3ch.eval()

    flat_1 = _flatten_obs(obs_hist, patch_1ch)
    flat_3 = _flatten_obs(obs_hist, patch_3ch)
    assert flat_1.shape == (8, FLAT_INPUT_DIM)
    assert flat_3.shape == (8, SEMANTIC_FLAT_INPUT_DIM)
    with torch.no_grad():
        s_act1 = student_1ch(flat_1)
        s_act3 = student_3ch(flat_3)
        s_act3_pair = student_3ch(obs_hist, patch_3ch)
    assert torch.allclose(s_act1, s_act3, atol=1e-6)
    assert torch.allclose(s_act3, s_act3_pair, atol=1e-6)


def test_semantic_v3_onnx_export_and_non_overwriting_filenames():
    """Verify *_semantic_v3.onnx filename separation and 3-channel ONNX export shape & ORT equivalence."""
    assert resolve_student_onnx_filename("sim", patch_channels=1) == "sim_policy_student.onnx"
    assert resolve_student_onnx_filename("real_tb3", patch_channels=1) == "real_tb3_policy_student.onnx"
    assert resolve_student_onnx_filename("sim", patch_channels=3) == "sim_policy_student_semantic_v3.onnx"
    assert resolve_student_onnx_filename("real_tb3", patch_channels=3) == "real_tb3_policy_student_semantic_v3.onnx"
    assert compute_flat_input_dim(1) == 688
    assert compute_flat_input_dim(3) == 976
    assert SEMANTIC_PATCH_CHANNELS == 3

    student_3ch = ContextRLStudent(patch_channels=3).eval()
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_fname = resolve_student_onnx_filename("sim", patch_channels=3)
        assert onnx_fname.endswith("_semantic_v3.onnx")
        out_path = os.path.join(tmpdir, onnx_fname)
        meta = export_student_onnx(student_3ch, out_path, patch_channels=3)

        assert os.path.exists(out_path)
        assert meta["flat_input_dim"] == 976
        assert meta["patch_channels"] == 3
        assert meta["output_dim"] == 2

        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        in_info = sess.get_inputs()[0]
        out_info = sess.get_outputs()[0]
        assert in_info.name == "obs_flat"
        assert out_info.name == "cmd_vel"
        assert in_info.shape[1] == 976
        assert out_info.shape[1] == 2

        obs_hist = torch.randn((4, HISTORY_LEN, STEP_FEAT_DIM), dtype=torch.float32)
        patch_3ch = torch.rand((4, 3, PATCH_SIZE, PATCH_SIZE), dtype=torch.float32)
        flat_3 = _flatten_obs(obs_hist, patch_3ch)

        with torch.no_grad():
            pt_out = student_3ch(flat_3).numpy()
        ort_out = sess.run(["cmd_vel"], {"obs_flat": flat_3.numpy().astype(np.float32)})[0]

        assert ort_out.shape == (4, 2)
        assert np.all(np.isfinite(ort_out))
        assert np.all(ort_out[:, 0] >= TB3_MIN_V - 1e-5) and np.all(ort_out[:, 0] <= TB3_MAX_V + 1e-5)
        assert np.all(np.abs(ort_out[:, 1]) <= TB3_MAX_W + 1e-5)
        assert np.allclose(pt_out, ort_out, atol=1e-5)


def test_frozen_residual_option2_exact_phase2b_parity_and_semantic_grad():
    """
    Verify Option 2 (Frozen Phase 2-B Backbone + Zero-Gated Residual Semantic Adapter):
      1. Teacher freezes Phase 2-B base.* params, preserves exact output when Ch1=Ch2=0,
         and propagates gradients only into residual/semantic modules when Ch1/Ch2 > 0.
      2. Student loads frozen Phase 2-B 1-Ch ONNX base, exports cleanly to ONNX, and matches
         sim_policy_student.onnx within float32 epsilon whenever Ch1=Ch2=0.
    """
    device = torch.device("cpu")
    torch.manual_seed(77)

    base_teacher = TemporalContextActorCritic(hidden_dim=128, patch_channels=1).to(device)
    ckpt_1ch = "/hdd/rl_v2_exp/archives/sim/best_teacher_seed101.pt"
    if os.path.exists(ckpt_1ch):
        obj = torch.load(ckpt_1ch, map_location=device)
        base_teacher.load_state_dict(obj["state_dict"])
    base_teacher.eval()

    opt2_teacher = FrozenBackboneSemanticTeacher(hidden_dim=128, patch_channels=3).to(device)
    opt2_teacher.load_warm_start(base_teacher)
    opt2_teacher.enable_semantic_kernel(1.0)
    opt2_teacher.eval()

    assert all(not p.requires_grad for p in opt2_teacher.base.parameters())
    assert any(p.requires_grad for p in opt2_teacher.sem_res_mlp.parameters())

    obs_hist = torch.randn((8, HISTORY_LEN, STEP_FEAT_DIM), dtype=torch.float32, device=device)
    patch_zero12 = torch.zeros((8, 3, PATCH_SIZE, PATCH_SIZE), dtype=torch.float32, device=device)
    patch_zero12[:, 0:1, :, :] = torch.rand((8, 1, PATCH_SIZE, PATCH_SIZE), dtype=torch.float32)

    with torch.no_grad():
        base_act_1ch, _ = base_teacher(obs_hist, patch_zero12[:, 0:1])
        opt2_act_zero, _ = opt2_teacher(obs_hist, patch_zero12)
    assert torch.allclose(base_act_1ch, opt2_act_zero, atol=1e-5)

    # Zero-gate must prevent semantic gradients on Ch1=Ch2=0 while allowing active gradients on Ch1/Ch2 > 0
    opt2_teacher.train()
    patch_sem = patch_zero12.clone()
    patch_sem[:, 1:3, 4:8, 4:8] = 0.85
    with torch.no_grad():
        torch.nn.init.normal_(opt2_teacher.sem_res_mlp[-1].weight, mean=0.0, std=0.02)
    act_sem, val_sem = opt2_teacher(obs_hist, patch_sem)
    (act_sem.sum() + val_sem.sum()).backward()
    assert opt2_teacher.sem_spatial[0].weight.grad is not None
    assert float(opt2_teacher.sem_spatial[0].weight.grad.abs().sum().item()) > 0.0
    assert all(p.grad is None for p in opt2_teacher.base.parameters())

    # Student ONNX exact parity check against Phase 2-B sim_policy_student.onnx
    base_onnx = os.path.join(WS_ROOT, "src", "disaster_recovery", "config", "sim_policy_student.onnx")
    if not os.path.exists(base_onnx):
        pytest.skip("sim_policy_student.onnx not present")

    opt2_student = FrozenBackboneSemanticStudent(patch_channels=3, base_onnx_path=base_onnx).to(device)
    opt2_student.enable_semantic_kernel(1.0)
    with torch.no_grad():
        torch.nn.init.normal_(opt2_student.sem_res_head[-1].weight, mean=0.0, std=0.05)
    opt2_student.eval()
    assert all(not p.requires_grad for p in opt2_student.base.parameters())

    with tempfile.TemporaryDirectory() as tmpdir:
        out_onnx = os.path.join(tmpdir, "opt2_test.onnx")
        export_student_onnx(opt2_student, out_onnx, patch_channels=3)

        base_sess = ort.InferenceSession(base_onnx, providers=["CPUExecutionProvider"])
        opt2_sess = ort.InferenceSession(out_onnx, providers=["CPUExecutionProvider"])

        flat_1 = _flatten_obs(obs_hist, patch_zero12[:, 0:1]).numpy().astype(np.float32)
        flat_3_zero = _flatten_obs(obs_hist, patch_zero12).numpy().astype(np.float32)
        flat_3_sem = _flatten_obs(obs_hist, patch_sem).numpy().astype(np.float32)

        y_base = base_sess.run(None, {base_sess.get_inputs()[0].name: flat_1})[0]
        y_opt2_zero = opt2_sess.run(None, {opt2_sess.get_inputs()[0].name: flat_3_zero})[0]
        y_opt2_sem = opt2_sess.run(None, {opt2_sess.get_inputs()[0].name: flat_3_sem})[0]

        assert float(np.max(np.abs(y_opt2_zero - y_base))) < 1e-5
        assert float(np.max(np.abs(y_opt2_sem - y_base))) > 1e-3
