#!/usr/bin/env python3
"""
Stage 3 (Phase 4-C & 4-A) Unit Tests:
  1. 4-channel patch (1120 flat dim, STEP_FEAT_DIM=34 invariant, zero outer clamping on Ch2/Ch3).
  2. Sinusoidal reciprocating crossfire regime, Ch3 DR (0.2~0.4s delay, vel noise, 10~20% occlusion),
     and PLICP 5 Hz sample-hold velocity noise DR.
  3. Pure physical safety cost c_t separated from BatchSoftCBFShield intervention/slack reward penalty.
  4. Exact module-level non-regression: 4-Ch ONNX (Ch1=Ch2=Ch3=0) vs Phase 2-B 1-Ch ONNX max_abs_diff == 0.0,
     plus ContextRLStudentPolicy auto-detection of 4-channel (1120-dim) ONNX.
  5. Continuous homotopy unification u_nom = (1 - g_res) * Blend(u_base, u_rl) + g_res * u_rl.
  6. CPPO (Rockafellar-Uryasev nu + G_C > nu tail weighting + NO independent std norm),
     M3 expected_cost mode, PID Lagrangian, and CBF transition Actor-freeze Critic-reheat.
"""

import os
import sys
import tempfile

import numpy as np
import onnxruntime as ort
import pytest
import torch

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HOST_SCRIPTS = os.path.join(WS_ROOT, "scripts", "host")
for _p in (PKG_ROOT, HOST_SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from disaster_recovery.rl_context_policy import (  # noqa: E402
    ContextRLStudentPolicy,
    HISTORY_LEN,
    PATCH_SIZE,
    PHASE4_FLAT_INPUT_DIM,
    PHASE4_PATCH_CHANNELS,
    STEP_FEAT_DIM,
    TB3_MAX_V,
    TB3_MAX_W,
    TB3_MIN_V,
    compute_flat_input_dim,
    compute_homotopy_blend,
    compute_homotopy_gate,
)
from rl_procedural_env import (  # noqa: E402
    BatchSoftCBFShield,
    CH3_MAX_R_INFL,
    COST_DYN_MARGIN_M,
    COST_W_COL,
    COST_W_DYN,
    COST_W_GAS,
    COST_W_LOW,
    NUM_REGIMES,
    PATCH_CHANNELS,
    TB3_RADIUS,
    VectorDisasterEnv,
)
from train_phase2_v100 import (  # noqa: E402
    CPPORiskEstimator,
    FrozenBackboneSemanticStudent,
    FrozenBackboneSemanticTeacher,
    PIDLagrangian,
    evaluate_policy_on_holdout,
    export_student_onnx,
    make_eval_seeds,
    resolve_student_onnx_filename,
)


def test_4ch_dimensions_step_feat_34_and_no_outer_clamping():
    """Verify STEP_FEAT_DIM=34, PATCH_CHANNELS=4 (1120 dim), Ch3 formula, and ZERO outer clamping on Ch2/Ch3."""
    assert STEP_FEAT_DIM == 34
    assert PATCH_CHANNELS == 4
    assert PHASE4_PATCH_CHANNELS == 4
    assert PHASE4_FLAT_INPUT_DIM == 1120
    assert compute_flat_input_dim(4) == 1120
    assert resolve_student_onnx_filename("sim", patch_channels=4) == "sim_policy_student_phase4.onnx"
    assert resolve_student_onnx_filename("real_tb3", patch_channels=4) == "real_tb3_policy_student_phase4.onnx"
    assert resolve_student_onnx_filename("sim", patch_channels=3) == "sim_policy_student_semantic_v3.onnx"
    assert resolve_student_onnx_filename("sim", patch_channels=1) == "sim_policy_student.onnx"

    env = VectorDisasterEnv(
        num_envs=4,
        track="sim",
        device=torch.device("cpu"),
        mode="eval",
        patch_channels=4,
        use_semantic_channels=True,
        modality_dropout_prob=0.0,
        enable_semantic_noise=False,
        enable_ch3_dr=False,
        enable_plicp_vel_noise_dr=False,
        semantic_reg_delay_steps=0,
    )
    obs_h, patch = env.reset_envs()
    assert obs_h.shape == (4, HISTORY_LEN, 34)
    assert patch.shape == (4, 4, PATCH_SIZE, PATCH_SIZE)
    assert torch.isfinite(obs_h).all() and torch.isfinite(patch).all()
    assert float(patch.min().item()) >= 0.0 and float(patch.max().item()) <= 1.0

    # Place a gas hazard (Ch2) and dynamic obstacle (Ch3) far outside the +-0.9m patch:
    # Must produce strictly 0.0 everywhere (no outer border clamping).
    env.rx.zero_()
    env.ry.zero_()
    env.rth.zero_()
    env.low_obs_active.fill_(False)
    env.plinth.zero_()
    env.hazard_active.fill_(False)
    env.hazard_zones.zero_()
    env.hazard_active[0, 0] = True
    env.hazard_zones[0, 0] = torch.tensor([2.8, 0.0, 0.25, 0.0])  # far outside 0.9m patch
    env.hazard_reg_delay.zero_()
    env.hazard_seen_ticks.fill_(10)

    env.cyl_active.fill_(False)
    env.cyl_sin_active.fill_(False)
    env.cyls.zero_()
    env.cyl_active[0, 0] = True
    env.cyls[0, 0] = torch.tensor([2.8, 0.0, 0.14, 0.0, 0.20])  # moves in +y at x=2.8m (outside patch)
    env.cyl_hist_fifo.zero_()
    env.cyl_hist_fifo[0, :, 0, :5] = env.cyls[0, 0]
    env.cyl_hist_fifo[0, :, 0, 5] = 1.0
    env.dyn_seen_ticks.fill_(10)

    _, patch_far = env._compute_obs_step()
    assert float(patch_far[0, 2].max().item()) == 0.0, "Ch2 must not clamp out-of-patch hazards to border"
    assert float(patch_far[0, 3].max().item()) == 0.0, "Ch3 must not clamp out-of-patch dynamic obstacles to border"

    # Now place dynamic obstacle entering ego patch over H={0.5, 1.0, 1.5}s -> Ch3 must activate inside patch
    env.cyls[0, 0] = torch.tensor([0.80, -0.30, 0.14, -0.35, 0.30])
    env.cyl_hist_fifo[0, :, 0, :5] = env.cyls[0, 0]
    env.cyl_hist_fifo[0, :, 0, 5] = 1.0
    _, patch_near = env._compute_obs_step()
    assert float(patch_near[0, 3].max().item()) > 0.50


def test_sinusoidal_reciprocating_regime_and_ch3_plicp_dr():
    """Verify Regime 2 50% sinusoidal crossfire kinematics, Ch3 DR (delay/noise/dropout), and PLICP 5 Hz hold DR."""
    env = VectorDisasterEnv(
        num_envs=64,
        track="sim",
        device=torch.device("cpu"),
        mode="train",
        base_seed=2026,
        patch_channels=4,
        use_semantic_channels=True,
    )
    env.reset_envs(forced_regime=2)
    sin_env_frac = float(env.cyl_sin_active.any(dim=1).float().mean().item())
    assert 0.25 <= sin_env_frac <= 0.75, f"Expected ~50% sinusoidal episodes in regime 2, got {sin_env_frac:.2f}"

    # Verify Ch3 DR parameter ranges
    assert int(env.ch3_delay_ticks.min().item()) >= 2
    assert int(env.ch3_delay_ticks.max().item()) <= 4
    assert float(env.ch3_vel_noise_std.min().item()) >= 0.05 - 1e-6
    assert float(env.ch3_vel_noise_std.max().item()) <= 0.12 + 1e-6
    assert float(env.ch3_drop_prob.min().item()) >= 0.10 - 1e-6
    assert float(env.ch3_drop_prob.max().item()) <= 0.20 + 1e-6

    # Verify Ch3 inflation radius cap (prevents 1.5s horizon from saturating 1.8m patch)
    assert float(env.ch3_denom_tau.max().item()) <= CH3_MAX_R_INFL + 1e-6

    # Verify PLICP velocity noise DR parameter ranges (§0.3 realistic bounds) & 5 Hz (2-tick) sample-hold behavior
    assert float(env.plicp_v_std.min().item()) >= 0.01 - 1e-6
    assert float(env.plicp_v_std.max().item()) <= 0.03 + 1e-6
    assert float(env.plicp_w_std.min().item()) >= 0.02 - 1e-6
    assert float(env.plicp_w_std.max().item()) <= 0.05 + 1e-6

    # Verify sinusoidal obstacle velocity follows v(t) = V_max * cos(omega * t + phi) and changes sign over cycle
    sin_env_ids = torch.nonzero(env.cyl_sin_active.any(dim=1), as_tuple=False).squeeze(-1)
    assert len(sin_env_ids) > 0
    e0 = int(sin_env_ids[0].item())
    c0 = int(torch.nonzero(env.cyl_sin_active[e0], as_tuple=False)[0].item())
    ay0 = float(env.cyl_sin_params[e0, c0, 3].item())
    om0 = float(env.cyl_sin_params[e0, c0, 4].item())
    ph0 = float(env.cyl_sin_params[e0, c0, 5].item())
    n_cycle_ticks = max(45, int(np.ceil(np.pi / max(abs(om0), 0.15) / env.dt)) + 5)
    vy_trace = []
    vw_trace = []
    act = torch.zeros((64, 2), dtype=torch.float32)
    act[:, 0] = 0.05
    for tick in range(1, n_cycle_ticks + 1):
        env.step(act)
        vy_now = float(env.cyls[e0, c0, 4].item())
        vy_expected = ay0 * om0 * float(np.cos(om0 * (tick * env.dt) + ph0))
        assert abs(vy_now - vy_expected) < 1e-4
        vy_trace.append(vy_now)
        if tick <= 5:
            vw_trace.append(env.actual_vw.clone())
    assert min(vy_trace) < -0.05 and max(vy_trace) > 0.05, "Sinusoidal obstacle must reciprocate (change sign)"

    # At least half of the envs hold identical actual_vw between adjacent ticks due to 5 Hz sample-hold
    same_between_t1_t2 = torch.allclose(vw_trace[1], vw_trace[2], atol=0.0)
    held_frac = float(((vw_trace[1] - vw_trace[2]).abs().max(dim=1).values == 0.0).float().mean().item())
    assert same_between_t1_t2 or held_frac >= 0.30

    # Verify Ch3 confirmation delay explicitly: with delay=3 and zero occlusion/noise, Ch3 stays 0 for ticks 0..2
    env_d = VectorDisasterEnv(
        num_envs=1,
        track="sim",
        device=torch.device("cpu"),
        mode="train",
        patch_channels=4,
        enable_ch3_dr=True,
        ch3_delay_ticks_range=(3, 3),
        ch3_vel_noise_std_range=(0.0, 0.0),
        ch3_occlusion_drop_range=(0.0, 0.0),
        enable_plicp_vel_noise_dr=False,
        enable_semantic_noise=False,
        modality_dropout_prob=0.0,
    )
    env_d.reset_envs(forced_regime=0)
    env_d.rx.zero_()
    env_d.ry.zero_()
    env_d.rth.zero_()
    env_d.boxes.zero_()
    env_d.box_active.fill_(False)
    env_d.cyl_active.fill_(False)
    env_d.cyl_sin_active.fill_(False)
    env_d.cyls.zero_()
    env_d.cyl_active[0, 0] = True
    env_d.cyls[0, 0] = torch.tensor([0.55, 0.0, 0.14, 0.0, 0.15])
    env_d.cyl_hist_fifo.zero_()
    env_d.cyl_hist_fifo[0, :, 0, :5] = env_d.cyls[0, 0]
    env_d.cyl_hist_fifo[0, :, 0, 5] = 1.0
    env_d.dyn_seen_ticks.zero_()

    zero_act = torch.zeros((1, 2), dtype=torch.float32)
    _, p0 = env_d._compute_obs_step()
    assert float(p0[0, 3].max().item()) == 0.0  # seen=0 < delay=3
    _, p1, _, _, _ = env_d.step(zero_act)
    assert float(p1[0, 3].max().item()) == 0.0  # seen=1 < delay=3
    _, p2, _, _, _ = env_d.step(zero_act)
    assert float(p2[0, 3].max().item()) == 0.0  # seen=2 < delay=3
    _, p3, _, _, _ = env_d.step(zero_act)
    assert float(p3[0, 3].max().item()) > 0.30  # seen=3 >= delay=3 -> confirmed!


def test_pure_physical_cost_and_batch_soft_cbf_shield_v_min():
    """Verify c_t is pure physical violation cost and BatchSoftCBFShield enforces runtime v_min cleanly."""
    env = VectorDisasterEnv(
        num_envs=2,
        track="sim",
        device=torch.device("cpu"),
        mode="eval",
        patch_channels=4,
        enable_ch3_dr=False,
        enable_plicp_vel_noise_dr=False,
        enable_semantic_noise=False,
        modality_dropout_prob=0.0,
        v_min=-0.10,
    )
    env.reset_envs(forced_regime=0)
    # Clear obstacles and place controlled dynamic obstacle + gas hazard for env 0, clear space for env 1
    env.box_active.fill_(False)
    env.cyl_active.fill_(False)
    env.cyl_sin_active.fill_(False)
    env.low_obs_active.fill_(False)
    env.hazard_active.fill_(False)
    env.plinth.zero_()
    env.rx.zero_()
    env.ry.zero_()
    env.rth.zero_()

    # Env 0: moving obstacle with center at (0.34, 0.0), r=0.14 -> surface dist ~0.20 < 0.28 -> positive cost_dyn
    env.cyl_active[0, 0] = True
    env.cyls[0, 0] = torch.tensor([0.34, 0.0, 0.14, -0.20, 0.0])
    # Env 0: also inside gas zone centered at (0.0, 0.0), r=0.25 -> positive cost_gas
    env.hazard_active[0, 0] = True
    env.hazard_zones[0, 0] = torch.tensor([0.0, 0.0, 0.25, 0.0])
    env._compute_obs_step()

    # Check BatchSoftCBFShield with runtime v_min=-0.10 vs v_min=0.0
    u_nom = torch.tensor([[0.20, 0.0], [0.20, 0.0]], dtype=torch.float32)
    cbf_neg = BatchSoftCBFShield(dt=0.10, v_min=-0.10, v_max=TB3_MAX_V, w_max=TB3_MAX_W)
    cbf_zero = BatchSoftCBFShield(dt=0.10, v_min=0.0, v_max=TB3_MAX_V, w_max=TB3_MAX_W)

    u_safe_neg, slack_neg, int_neg = cbf_neg.filter_batch(env, u_nom)
    u_safe_zero, slack_zero, int_zero = cbf_zero.filter_batch(env, u_nom)

    assert torch.isfinite(u_safe_neg).all() and torch.isfinite(slack_neg).all()
    assert bool(int_neg[0].item()) is True, "CBF must intervene on impending oncoming dynamic obstacle"
    assert float(u_safe_neg[0, 0].item()) < float(u_nom[0, 0].item())
    assert float(u_safe_neg[0, 0].item()) >= -0.10 - 1e-6
    assert float(u_safe_zero[0, 0].item()) >= 0.0 - 1e-6, "CBF with v_min=0.0 must never command reverse"
    assert bool(int_neg[1].item()) is False, "CBF must not intervene in open clear space"
    assert torch.allclose(u_safe_neg[1], u_nom[1], atol=1e-5)

    # Step env and verify pure physical cost formula
    _, _, _, _, info = env.step(torch.zeros((2, 2), dtype=torch.float32))
    assert float(info["cost"][0].item()) > 0.0
    assert float(info["cost_dyn"][0].item()) > 0.0
    assert float(info["cost_gas"][0].item()) > 0.0
    assert float(info["cost"][1].item()) == 0.0
    expected_c0 = (
        info["cost_col"][0] + info["cost_low"][0] + info["cost_gas"][0] + info["cost_dyn"][0]
    )
    assert abs(float(info["cost"][0].item()) - float(expected_c0.item())) < 1e-6


def test_phase4_4ch_exact_non_regression_max_abs_diff_zero_onnx():
    """
    Mandatory Gate:
      When Ch1=Ch2=Ch3=0, 4-Ch Phase 4 Student ONNX (1120 dim, safety_head removed) MUST match
      Phase 2-B 1-Ch Student ONNX with strict max_abs_diff == 0.0, even after residual weights
      are perturbed to non-zero values.
    """
    base_onnx = os.path.join(WS_ROOT, "src", "disaster_recovery", "config", "sim_policy_student.onnx")
    if not os.path.exists(base_onnx):
        pytest.skip(f"Phase 2-B base ONNX not found: {base_onnx}")

    student = FrozenBackboneSemanticStudent(
        patch_channels=4,
        base_onnx_path=base_onnx,
        use_safety_head=False,
    )
    # Perturb trainable residual adapter weights to simulate a fully trained Phase 4 student
    torch.manual_seed(123)
    with torch.no_grad():
        for p in student.sem_res_head.parameters():
            p.normal_(0.0, 0.25)
        for p in student.sem_spatial.parameters():
            p.normal_(0.0, 0.25)
    student.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_onnx = os.path.join(tmpdir, "sim_policy_student_phase4.onnx")
        meta = export_student_onnx(student, out_onnx, patch_channels=4)
        assert meta["flat_input_dim"] == 1120
        assert meta["patch_channels"] == 4

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        base_sess = ort.InferenceSession(base_onnx, sess_options=opts, providers=["CPUExecutionProvider"])
        p4_sess = ort.InferenceSession(out_onnx, sess_options=opts, providers=["CPUExecutionProvider"])
        base_in = base_sess.get_inputs()[0].name
        p4_in = p4_sess.get_inputs()[0].name

        rng = np.random.default_rng(2026)
        h_test = rng.normal(0.0, 0.5, size=(256, HISTORY_LEN * STEP_FEAT_DIM)).astype(np.float32)
        c0_test = np.clip(rng.uniform(0.0, 1.0, size=(256, PATCH_SIZE * PATCH_SIZE)), 0.0, 1.0).astype(np.float32)
        c123_zero = np.zeros((256, 3 * PATCH_SIZE * PATCH_SIZE), dtype=np.float32)

        out_1ch = base_sess.run(None, {base_in: np.concatenate([h_test, c0_test], axis=1)})[0]
        out_4ch_zero = p4_sess.run(None, {p4_in: np.concatenate([h_test, c0_test, c123_zero], axis=1)})[0]
        max_abs_diff = float(np.max(np.abs(out_4ch_zero - out_1ch)))
        assert max_abs_diff == 0.0, f"Expected exact non-regression max_abs_diff == 0.0, got {max_abs_diff}"

        # Non-zero Ch3 must activate residual output
        c123_ch3 = np.zeros((256, 3 * PATCH_SIZE * PATCH_SIZE), dtype=np.float32)
        c123_ch3[:, 2 * PATCH_SIZE * PATCH_SIZE :] = 0.80  # Ch3 active
        out_4ch_dyn = p4_sess.run(None, {p4_in: np.concatenate([h_test, c0_test, c123_ch3], axis=1)})[0]
        assert float(np.max(np.abs(out_4ch_dyn - out_1ch))) > 1e-3

        # Verify ContextRLStudentPolicy auto-detects 4-channel (1120-dim) ONNX and runs predict with dynamic_ch3
        wrapper = ContextRLStudentPolicy(out_onnx)
        assert wrapper.patch_channels == 4
        assert wrapper.flat_input_dim == 1120
        scan = np.full(360, 2.0, dtype=np.float32)
        ch3 = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        ch3[7, 6] = 0.9
        v, w = wrapper.predict(scan, 0.0, 0.0, 0.0, 2.0, 0.0, 0.1, 0.0, 0.1, 0.0, dynamic_ch3=ch3)
        assert TB3_MIN_V <= v <= TB3_MAX_V
        assert -TB3_MAX_W <= w <= TB3_MAX_W


def test_homotopy_continuous_interpolation():
    """Verify u_nom = (1 - g_res) * Blend(u_base, u_rl) + g_res * u_rl continuous homotopy unification."""
    v_base, w_base = 0.12, 0.40
    v_rl, w_rl = 0.18, -0.35
    alpha_rl = 0.50

    # Pure LiDAR (Ch1=Ch2=Ch3=0): g_res == 0 -> exact Blend(u_base, u_rl)
    v0, w0, g0 = compute_homotopy_blend(v_base, w_base, v_rl, w_rl, alpha_rl, ch123_peak=0.0)
    assert g0 == 0.0
    assert abs(v0 - 0.15) < 1e-6
    assert abs(w0 - 0.025) < 1e-6

    # Active Ch1/2/3 (>= 0.0401): g_res == 1 -> exact u_rl (homotopy unified)
    v1, w1, g1 = compute_homotopy_blend(v_base, w_base, v_rl, w_rl, alpha_rl, ch123_peak=0.05)
    assert g1 == 1.0
    assert abs(v1 - v_rl) < 1e-6
    assert abs(w1 - w_rl) < 1e-6

    # Intermediate peak: strictly continuous linear interpolation with no hard jump
    peaks = np.linspace(0.0, 0.05, 51)
    ws = [compute_homotopy_blend(v_base, w_base, v_rl, w_rl, alpha_rl, float(pk))[1] for pk in peaks]
    max_step = float(np.max(np.abs(np.diff(ws))))
    assert max_step < 0.02, f"Expected smooth transition without torque spike, max_step={max_step}"
    assert compute_homotopy_gate(0.0201) == pytest.approx(0.5, abs=1e-3)


def test_cppo_nu_tail_weighting_no_std_norm_and_pid_lagrangian():
    """Verify Rockafellar-Uryasev nu, non-masking low_obs baseline + per-regime nu, adv_c stability, and PID Lagrangian."""
    est = CPPORiskEstimator(alpha=0.10, nu_lr=0.05, cost_adv_scale=1.0, cvar_mean_mix=0.25, adv_c_clip=3.0)
    rng = np.random.default_rng(7)
    # Heavy-tailed distribution: 90% low cost (~0.1), 10% tail cost (~2.0)
    ep_costs = np.concatenate([rng.uniform(0.0, 0.2, 900), rng.uniform(1.5, 2.5, 100)])
    for _ in range(40):
        stats = est.update_nu(ep_costs)
    q90 = float(np.quantile(ep_costs, 0.90))
    assert abs(est.nu - q90) < 0.35
    assert stats["cvar"] > stats["mean_cost"]

    # 1) Low-cost regime (e.g. low_obs cost=1.5 < global nu=5.0) MUST NOT be masked to zero,
    #    while NO independent std normalization blows up tiny adv_c=1e-4 to O(1).
    est.nu = 5.0
    adv_r = torch.tensor([1.0, -1.0, 2.0, -2.0], dtype=torch.float32)
    adv_c_small = torch.tensor([1e-4, 1e-4, 1e-4, 1e-4], dtype=torch.float32)
    ep_c = torch.tensor([1.5, 2.0, 6.0, 10.0], dtype=torch.float32)  # first 2 < global nu=5.0, last 2 in tail
    lam = 2.0

    comb_cppo = est.combine_advantages(adv_r, adv_c_small, ep_c, lam=lam, algo="cppo")
    adv_r_norm = (adv_r - adv_r.mean()) / (adv_r.std(unbiased=False) + 1e-8)
    non_tail_delta = comb_cppo[:2] - adv_r_norm[:2] / (1.0 + lam)
    tail_delta = comb_cppo[2:] - adv_r_norm[2:] / (1.0 + lam)
    # Non-tail receives cvar_mean_mix=0.25 baseline (non-zero, NOT masked out!)
    expected_non_tail = -(lam / (1.0 + lam)) * 0.25 * 1e-4
    assert torch.allclose(non_tail_delta, torch.full_like(non_tail_delta, expected_non_tail), atol=1e-7)
    # Tail receives stronger penalty than non-tail, without std-norm explosion
    expected_tail = -(lam / (1.0 + lam)) * (0.25 + 0.75 * 3.0) * 1e-4
    assert torch.allclose(tail_delta, torch.full_like(tail_delta, expected_tail), atol=1e-7)
    assert float(tail_delta.abs().mean().item()) > float(non_tail_delta.abs().mean().item())

    # 2) Scale stability: huge positive & negative adv_c in tail are bounded and negative adv_c gets no 1/alpha bonus blowup
    adv_c_extreme = torch.tensor([0.5, 10.0, 10.0, -10.0], dtype=torch.float32)
    comb_ext = est.combine_advantages(adv_r, adv_c_extreme, ep_c, lam=lam, algo="cppo")
    delta_ext = comb_ext - adv_r_norm / (1.0 + lam)
    max_possible = (lam / (1.0 + lam)) * 3.0 + 1e-5
    assert float(delta_ext.abs().max().item()) <= max_possible
    # Negative adv_c inside tail (index 3) is capped at adv_c_clip (and not multiplied by 1/alpha=10)
    adv_c_neg = torch.tensor([-0.5, -0.5, -0.5, -0.5], dtype=torch.float32)
    comb_neg = est.combine_advantages(adv_r, adv_c_neg, ep_c, lam=lam, algo="cppo")
    neg_tail_delta = float((comb_neg[2] - adv_r_norm[2] / (1.0 + lam)).item())
    assert neg_tail_delta == pytest.approx((lam / (1.0 + lam)) * 0.5, abs=1e-6)

    # 3) Per-regime nu tracking activates tail for low_obs regime 6 even when global nu is high
    reg_costs = np.concatenate([np.full(40, 10.0), np.linspace(0.0, 2.0, 40)])
    reg_ids = np.concatenate([np.zeros(40, dtype=np.int64), np.full(40, 6, dtype=np.int64)])
    est_r = CPPORiskEstimator(alpha=0.10, nu_lr=0.05, cvar_mean_mix=0.25)
    est_r.update_nu(reg_costs, regimes=reg_ids)
    assert est_r.nu > 5.0
    assert est_r.regime_nu[6] < 2.0
    comb_reg = est_r.combine_advantages(
        adv_r,
        torch.ones(4),
        torch.tensor([1.9, 1.9, 1.9, 1.9]),
        lam=1.0,
        algo="cppo",
        regimes=torch.tensor([0, 0, 6, 6], dtype=torch.long),
    )
    # Regime 6 (cost=1.9 > regime_nu[6]) receives stronger penalty than Regime 0 (cost=1.9 < regime_nu[0]=10.0)
    delta_reg = comb_reg - adv_r_norm / 2.0
    assert float(delta_reg[2].item()) < float(delta_reg[0].item()) - 0.5

    # M3 expected_cost mode applies cost advantage uniformly across all episodes
    comb_m3 = est.combine_advantages(adv_r, adv_c_small, ep_c, lam=lam, algo="expected_cost")
    expected_m3 = (adv_r_norm - lam * adv_c_small) / (1.0 + lam)
    assert torch.allclose(comb_m3, expected_m3, atol=1e-7)

    # PID Lagrangian update, CBF transition decay, and non-saturation check for CVaR defaults
    pid = PIDLagrangian(kp=0.40, ki=0.08, kd=0.10, lambda_max=5.0, i_max=25.0)
    l1 = pid.update(risk_metric=1.35, d_thresh=0.35)  # delta = 1.0 -> P=0.40, I=0.08, D=0 -> 0.48
    assert l1 == pytest.approx(0.48, abs=1e-6)
    l2 = pid.update(risk_metric=1.85, d_thresh=0.35)  # delta = 1.5, I=2.5, D=0.5 -> 0.60+0.20+0.05 = 0.85
    assert l2 == pytest.approx(0.85, abs=1e-6)
    prev_lam, prev_int = pid.lam, pid.integral
    pid.decay_for_cbf_transition(0.3)
    assert pid.lam == pytest.approx(0.3 * prev_lam, abs=1e-6)
    assert pid.integral == pytest.approx(0.3 * prev_int, abs=1e-6)
    assert pid.has_prev is False

    pid_cvar = PIDLagrangian(kp=0.06, ki=0.012, kd=0.02, lambda_max=5.0, i_max=150.0)
    for _ in range(10):
        lam_now = pid_cvar.update(risk_metric=9.0, d_thresh=1.0)
    assert 0.5 < lam_now < 5.0, f"CVaR PID should control smoothly without early saturation, got {lam_now}"


def test_cbf_transition_actor_freeze_critic_reheat():
    """Verify FrozenBackboneSemanticTeacher has scalar cost_critic_net and freezes Actor during Critic reheat."""
    model = FrozenBackboneSemanticTeacher(hidden_dim=64, patch_channels=4, use_safety_head=False)
    assert hasattr(model, "cost_critic_net")
    assert not hasattr(model, "quantile_cost_critic")

    # Verify exact zero residual & zero actor grad when Ch1=Ch2=Ch3=0
    obs = torch.randn(4, HISTORY_LEN, STEP_FEAT_DIM)
    patch_zero = torch.zeros(4, 4, PATCH_SIZE, PATCH_SIZE)
    patch_zero[:, 0] = torch.rand(4, PATCH_SIZE, PATCH_SIZE)
    act_full, _ = model(obs, patch_zero)
    act_base, _ = model.forward_base(obs, patch_zero)
    assert torch.equal(act_full, act_base)

    # Freeze actor for Critic re-warming (CBF transition)
    model.set_actor_frozen(True)
    actor_before = [p.detach().clone() for p in list(model.sem_spatial.parameters()) + list(model.sem_res_mlp.parameters())]
    critic_before = [p.detach().clone() for p in list(model.critic_net.parameters()) + list(model.cost_critic_net.parameters())]
    base_before = [p.detach().clone() for p in model.base.parameters()]

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    patch_act = patch_zero.clone()
    patch_act[:, 3] = 0.6  # Ch3 active
    _, _, _, val_r, val_c = model.get_action_and_values(obs, patch_act)
    loss = (val_r - 1.0).pow(2).mean() + (val_c - 0.5).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

    # Actor and base MUST remain bit-for-bit identical; Critics MUST update
    for p_old, p_new in zip(actor_before, list(model.sem_spatial.parameters()) + list(model.sem_res_mlp.parameters())):
        assert torch.equal(p_old, p_new)
    for p_old, p_new in zip(base_before, model.base.parameters()):
        assert torch.equal(p_old, p_new)
    critic_changed = any(
        not torch.equal(p_old, p_new)
        for p_old, p_new in zip(critic_before, list(model.critic_net.parameters()) + list(model.cost_critic_net.parameters()))
    )
    assert critic_changed, "Reward and Cost Critics must update during CBF re-warming"

    # Unfreeze actor and verify base remains frozen
    model.set_actor_frozen(False)
    assert all(p.requires_grad for p in model.sem_res_mlp.parameters())
    assert all(not p.requires_grad for p in model.base.parameters())


def test_hit_low_now_no_forward_sliding_and_cost_scale():
    """Verify hit_low_now blocks forward penetration at obstacle rim without +0.025m/+0.045m slide assist, and COST_W_COL=10."""
    assert COST_W_COL == pytest.approx(10.0)
    assert COST_W_LOW == pytest.approx(1.5)
    assert COST_W_GAS == pytest.approx(1.0)

    env = VectorDisasterEnv(
        num_envs=1,
        track="sim",
        device=torch.device("cpu"),
        mode="eval",
        base_seed=123,
        patch_channels=4,
        use_semantic_channels=True,
        modality_dropout_prob=0.0,
        enable_semantic_noise=False,
        enable_ch3_dr=False,
        enable_plicp_vel_noise_dr=False,
    )
    env.reset_envs(forced_regime=6)
    env.box_active.fill_(False)
    env.cyl_active.fill_(False)
    env.cyl_sin_active.fill_(False)
    env.plinth.zero_()
    env.low_obs_active.fill_(False)
    env.hazard_active.fill_(False)
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
    env.low_obs[0, 0] = torch.tensor([0.32, 0.0, 0.15])
    env.low_obs_active[0, 0] = True

    rim_x_max = 0.32 - (0.15 + TB3_RADIUS + 0.02) + 0.005
    act = torch.tensor([[0.22, 0.0]], dtype=torch.float32)
    hit_count = 0
    cum_low_cost = 0.0
    for _ in range(15):
        _, _, _, _, info = env.step(act)
        if bool(info["low_obs_contact"][0].item()):
            hit_count += 1
            step_cl = float(info["cost_low"][0].item())
            cum_low_cost += step_cl
            if hit_count == 1:
                assert step_cl == pytest.approx(COST_W_LOW, abs=1e-5)
            else:
                assert step_cl == pytest.approx(0.015 * COST_W_LOW, abs=1e-5)
            # Robot MUST be stopped at rim (no +0.025m forward jump, no +0.045m lateral slide)
            assert float(env.rx[0].item()) <= rim_x_max
            assert abs(float(env.ry[0].item())) < 1e-4
    assert hit_count >= 5, f"Expected sustained rim impedance stop against low obstacle, got {hit_count} hits"
    assert COST_W_LOW <= cum_low_cost < COST_W_COL, f"Non-fatal rim contact cost ({cum_low_cost}) must stay below COST_W_COL ({COST_W_COL})"


def test_context_rl_student_policy_v_min_clipping():
    """Verify ContextRLStudentPolicy clips reverse velocity to configured v_min (e.g. -0.08 for real_tb3)."""

    class _ConstReverseModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros((x.shape[0], 2), dtype=x.dtype, device=x.device)
            out[:, 0] = -0.10
            out[:, 1] = 0.15
            return out

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_p = os.path.join(tmpdir, "const_rev_4ch.onnx")
        dummy = torch.zeros((1, PHASE4_FLAT_INPUT_DIM), dtype=torch.float32)
        torch.onnx.export(
            _ConstReverseModel(),
            dummy,
            onnx_p,
            input_names=["obs_flat"],
            output_names=["action"],
            opset_version=14,
            dynamic_axes={"obs_flat": {0: "batch"}, "action": {0: "batch"}},
        )
        scan = np.full(360, 2.0, dtype=np.float32)
        pol_real = ContextRLStudentPolicy(onnx_p, v_min=-0.08)
        v_real, w_real = pol_real.predict(scan, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert v_real == pytest.approx(-0.08, abs=1e-6)
        assert w_real == pytest.approx(0.15, abs=1e-6)

        pol_sim = ContextRLStudentPolicy(onnx_p, v_min=-0.10)
        v_sim, _ = pol_sim.predict(scan, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert v_sim == pytest.approx(-0.10, abs=1e-6)


def test_expanded_frozen_student_capacity_non_regression_and_cpu_latency():
    """Verify expanded 4-Ch FrozenBackboneSemanticStudent (24ch / 48 / 192->96->2, brake 96->1) maintains max_abs_diff == 0.0 and p95 <= 0.15 ms CPU."""
    import time

    base_onnx = os.path.join(WS_ROOT, "src", "disaster_recovery", "config", "sim_policy_student.onnx")
    if not os.path.exists(base_onnx):
        pytest.skip(f"Phase 2-B base ONNX not found: {base_onnx}")

    student = FrozenBackboneSemanticStudent(patch_channels=4, base_onnx_path=base_onnx, use_safety_head=False)
    assert student.sem_spatial[2].out_channels == 24
    assert student.sem_spatial[-2].out_features == 48
    assert student.sem_res_head[0].out_features == 192
    assert student.sem_res_head[2].out_features == 96
    assert student.sem_brake_head.in_features == 96

    torch.manual_seed(99)
    with torch.no_grad():
        for p in list(student.sem_spatial.parameters()) + list(student.sem_res_head.parameters()) + list(student.sem_brake_head.parameters()):
            p.normal_(0.0, 0.20)
    student.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_onnx = os.path.join(tmpdir, "expanded_p4.onnx")
        export_student_onnx(student, out_onnx, patch_channels=4)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        base_sess = ort.InferenceSession(base_onnx, sess_options=opts, providers=["CPUExecutionProvider"])
        p4_sess = ort.InferenceSession(out_onnx, sess_options=opts, providers=["CPUExecutionProvider"])

        rng = np.random.default_rng(314)
        h_t = rng.normal(0.0, 0.5, size=(128, HISTORY_LEN * STEP_FEAT_DIM)).astype(np.float32)
        c0_t = np.clip(rng.uniform(0.0, 1.0, size=(128, PATCH_SIZE * PATCH_SIZE)), 0.0, 1.0).astype(np.float32)
        c123_z = np.zeros((128, 3 * PATCH_SIZE * PATCH_SIZE), dtype=np.float32)
        y_base = base_sess.run(None, {base_sess.get_inputs()[0].name: np.concatenate([h_t, c0_t], axis=1)})[0]
        y_p4 = p4_sess.run(None, {p4_sess.get_inputs()[0].name: np.concatenate([h_t, c0_t, c123_z], axis=1)})[0]
        assert float(np.max(np.abs(y_p4 - y_base))) == 0.0

        sample = np.zeros((1, PHASE4_FLAT_INPUT_DIM), dtype=np.float32)
        in_k = p4_sess.get_inputs()[0].name
        for _ in range(50):
            p4_sess.run(None, {in_k: sample})
        lats = []
        for _ in range(500):
            t0 = time.perf_counter()
            p4_sess.run(None, {in_k: sample})
            lats.append((time.perf_counter() - t0) * 1000.0)
        p95_ms = float(np.percentile(lats, 95))
        assert p95_ms <= 0.15, f"Expected single-thread CPU p95 latency <= 0.15 ms, got {p95_ms:.4f} ms"


def test_eval_determinism_regime_balance_cbf_and_safe_unseen_sr():
    """Verify bit-exact eval determinism, balanced 8-regime distribution for 400 episodes, CBF eval, and safe_unseen_sr."""
    # 1) Balanced 8-regime distribution for N=400 eval seeds (no >=90 hardcoding bias)
    seeds_400 = make_eval_seeds(400, 10000)
    env_bal = VectorDisasterEnv(
        num_envs=400,
        track="sim",
        device=torch.device("cpu"),
        mode="eval",
        base_seed=10000,
        patch_channels=4,
        include_semantic_regimes=True,
    )
    env_bal.reset_envs(eval_seeds=seeds_400)
    counts = torch.bincount(env_bal.regime, minlength=NUM_REGIMES).tolist()
    assert counts == [50] * NUM_REGIMES, f"Expected exact 50/regime across 8 regimes, got {counts}"

    # 2) Bit-exact determinism across two independent evaluate_policy_on_holdout calls & CBF ON eval + safe_unseen_sr
    eval_16 = make_eval_seeds(16, 10000)

    def simple_policy(h: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        del p
        goal_sin = h[:, -1, 25]
        goal_cos = h[:, -1, 26]
        heading_err = torch.atan2(goal_sin, goal_cos)
        v = torch.full((h.shape[0],), 0.16, dtype=torch.float32, device=h.device)
        w = torch.clamp(1.2 * heading_err, -TB3_MAX_W, TB3_MAX_W)
        return torch.stack([v, w], dim=-1)

    res1 = evaluate_policy_on_holdout(
        "sim",
        torch.device("cpu"),
        simple_policy,
        max_steps=80,
        patch_channels=4,
        eval_seeds=eval_16,
        use_cbf=False,
    )
    res2 = evaluate_policy_on_holdout(
        "sim",
        torch.device("cpu"),
        simple_policy,
        max_steps=80,
        patch_channels=4,
        eval_seeds=eval_16,
        use_cbf=False,
    )
    assert res1["unseen_sr"] == res2["unseen_sr"]
    assert res1["safe_unseen_sr"] == res2["safe_unseen_sr"]
    assert res1["dropout_fallback_sr"] == res2["dropout_fallback_sr"]
    assert res1["cvar10_cost"] == res2["cvar10_cost"]
    assert res1["mean_cost"] == res2["mean_cost"]
    assert 0.0 <= res1["safe_unseen_sr"] <= res1["unseen_sr"] + 1e-9

    res_cbf = evaluate_policy_on_holdout(
        "sim",
        torch.device("cpu"),
        simple_policy,
        max_steps=80,
        patch_channels=4,
        eval_seeds=eval_16,
        use_cbf=True,
        cbf_v_min=-0.10,
    )
    assert res_cbf["use_cbf"] is True
    assert 0.0 <= res_cbf["safe_unseen_sr"] <= res_cbf["unseen_sr"] + 1e-9
    assert np.isfinite(res_cbf["cvar10_cost"]) and np.isfinite(res_cbf["cbf_intervention_rate"])


def test_batch_soft_cbf_observed_state_and_strict_real_dr():
    """Verify BatchSoftCBFShield avoids GT dynamic leakage under Ch3 DR and enable_strict_real_dr toggles real_tb3 clamps."""
    env = VectorDisasterEnv(
        num_envs=1,
        track="sim",
        device=torch.device("cpu"),
        mode="train",
        patch_channels=4,
        enable_ch3_dr=True,
        ch3_delay_ticks_range=(3, 3),
        ch3_vel_noise_std_range=(0.0, 0.0),
        ch3_occlusion_drop_range=(0.0, 0.0),
        enable_semantic_noise=False,
        modality_dropout_prob=0.0,
    )
    env.reset_envs(forced_regime=0)
    env.box_active.fill_(False)
    env.cyl_active.fill_(False)
    env.cyl_sin_active.fill_(False)
    env.low_obs_active.fill_(False)
    env.hazard_active.fill_(False)
    env.plinth.zero_()
    env.rx.zero_()
    env.ry.zero_()
    env.rth.zero_()
    env._last_lidar_ranges.fill_(3.5)

    # Dynamic obstacle close ahead in GT & FIFO, but unconfirmed (dyn_seen_ticks=0 < delay=3)
    env.cyl_active[0, 0] = True
    env.cyls[0, 0] = torch.tensor([0.34, 0.0, 0.14, -0.20, 0.0])
    env.cyl_hist_fifo.zero_()
    env.cyl_hist_fifo[0, :, 0, :5] = env.cyls[0, 0]
    env.cyl_hist_fifo[0, :, 0, 5] = 1.0
    env.dyn_seen_ticks.zero_()
    env._obs_ch3_vnoise.zero_()
    env._obs_ch3_keep.fill_(True)

    cbf = BatchSoftCBFShield(dt=0.10, v_min=-0.10, use_observed_state=True)
    u_nom = torch.tensor([[0.20, 0.0]], dtype=torch.float32)
    _, _, int_unconfirmed = cbf.filter_batch(env, u_nom, use_observed_state=True)
    assert bool(int_unconfirmed[0].item()) is False, "CBF must not leak unconfirmed GT dynamic obstacle during training"

    env.dyn_seen_ticks.fill_(3)
    u_conf, _, int_confirmed = cbf.filter_batch(env, u_nom, use_observed_state=True)
    assert bool(int_confirmed[0].item()) is True
    assert float(u_conf[0, 0].item()) < 0.20

    # Strict real_tb3 DR toggle check
    env_clamped = VectorDisasterEnv(num_envs=64, track="real_tb3", device=torch.device("cpu"), mode="train", base_seed=77, enable_strict_real_dr=False)
    env_clamped.reset_envs(forced_regime=0)
    assert int(env_clamped.act_lag.max().item()) == 0
    assert float(env_clamped.slip_v.min().item()) >= 0.92 - 1e-6

    env_strict = VectorDisasterEnv(num_envs=64, track="real_tb3", device=torch.device("cpu"), mode="train", base_seed=77, enable_strict_real_dr=True)
    env_strict.reset_envs(forced_regime=0)
    assert int(env_strict.act_lag.max().item()) >= 1
    assert float(env_strict.slip_v.min().item()) < 0.90


def test_context_rl_edge_lidar_safety_angles_priming_and_peak_cache():
    """Verify Part A: angle_min=-pi, n=359 non-multiple scan, 0<r<0.12 preservation, invalid fallback, latch->ch0, 16-tick priming."""
    import math

    # 1) angle_min=-pi with n=359 non-multiple scan: front obstacle at 0 rad (middle index ~179-180) must land in front bin 0
    n = 359
    angle_min = -math.pi
    angle_inc = 2.0 * math.pi / float(n)
    scan_359 = np.full(n, 3.0, dtype=np.float32)
    mid_idx = int(round((0.0 - angle_min) / angle_inc))
    scan_359[mid_idx] = 0.40
    feat_359, patch_359 = ContextRLStudentPolicy._build_step_and_patch(
        scan_359, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        patch_channels=1, angle_min=angle_min, angle_increment=angle_inc,
    )
    assert abs(float(feat_359[0]) - (0.40 / 3.5)) < 1e-4, f"Expected front bin 0 == 0.40/3.5, got {feat_359[0]}"
    assert abs(float(feat_359[12]) - (3.0 / 3.5)) < 1e-4, "Rear bin 12 must not receive front obstacle"
    assert float(patch_359[6:, :].max()) > 0.0

    # 2) Positive ultra-close 0.0 < r < 0.12 preserved as 0.12 (not 3.5) in both bins_24 and ch0
    scan_close = np.full(360, 2.5, dtype=np.float32)
    scan_close[0] = 0.07
    feat_cl, patch_cl = ContextRLStudentPolicy._build_step_and_patch(
        scan_close, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, patch_channels=1,
    )
    assert abs(float(feat_cl[0]) - (0.12 / 3.5)) < 1e-5
    assert float(patch_cl[6, 5:7].max()) > 0.0, "Ultra-close hit (0.07 -> 0.12 m) must populate front ch0 cell"

    # 3) All-invalid scan (NaN / 0.0 / -inf) must NEVER fill bins_24 with 1.0 open space
    for bad_val in (float("nan"), 0.0, float("-inf")):
        scan_bad = np.full(360, bad_val, dtype=np.float32)
        feat_b, patch_b = ContextRLStudentPolicy._build_step_and_patch(
            scan_bad, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, patch_channels=1,
        )
        assert float(feat_b[:24].max()) < 0.20, f"All-invalid scan ({bad_val}) must not report open space"
        assert float(patch_b.max()) > 0.0

    # 4) Near-latch TTL consistency between bins_24 and ch0
    ttl = np.zeros(24, dtype=np.int32)
    scan_arm = np.full(360, 2.5, dtype=np.float32)
    scan_arm[0] = 0.15
    ContextRLStudentPolicy._build_step_and_patch(
        scan_arm, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, near_latch_ttl=ttl, patch_channels=1,
    )
    assert ttl[0] == 15
    scan_drop = np.full(360, 2.5, dtype=np.float32)
    # Entire front bin 0 drops to 0.0
    scan_drop[:8] = 0.0
    scan_drop[353:] = 0.0
    feat_lat, patch_lat = ContextRLStudentPolicy._build_step_and_patch(
        scan_drop, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, near_latch_ttl=ttl, patch_channels=1,
    )
    assert abs(float(feat_lat[0]) - (0.12 / 3.5)) < 1e-5
    assert float(patch_lat[6, 5:7].max()) > 0.0, "Near-latched sector must also project into ch0 patch"

    # 5) Cold-start 16-tick priming + last_ch123_peak caching
    base_onnx = os.path.join(WS_ROOT, "src", "disaster_recovery", "config", "sim_policy_student_phase4.onnx")
    if not os.path.exists(base_onnx):
        base_onnx = os.path.join(WS_ROOT, "src", "disaster_recovery", "config", "sim_policy_student.onnx")
    if os.path.exists(base_onnx):
        pol = ContextRLStudentPolicy(base_onnx, v_min=-0.08)
        assert pol._needs_prime is True
        ch3 = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        ch3[7, 6] = 0.65
        pol.observe(scan_close, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, dynamic_ch3=ch3)
        assert pol._needs_prime is False
        hist_mat = np.stack(list(pol.hist), axis=0)
        assert hist_mat.shape == (HISTORY_LEN, STEP_FEAT_DIM)
        assert np.allclose(hist_mat, hist_mat[0:1], atol=0.0), "First observe after reset must prime all 16 slots"
        if pol.patch_channels >= 4:
            assert pol.last_ch123_peak == pytest.approx(0.65, abs=1e-5)


def test_recovery_controller_homotopy_icp_se2_cache_gas_and_deadzone():
    """Verify Part B/D runtime controller fixes: alpha_rl==0 homotopy, far-origin SE(2) ICP, cache sync/no-drag, gas guard, deadzone."""
    import json
    import math
    from disaster_recovery.recovery_controller import (
        RecoveryController,
        SemanticObjectCache,
        SystemState,
        Waypoint,
        apply_motor_deadzone,
        estimate_scan_velocity_icp,
        normalize_angle,
    )
    from disaster_recovery.local_grid_planner import ProposedRecoveryPlanner

    # 1) CBF-decelerated deadzone boost suppression
    v_sup, w_sup = apply_motor_deadzone(
        0.012, -0.04, min_v=0.025, min_w=0.08, v_max=0.15, w_max=0.55,
        suppress_v_boost=True, suppress_w_boost=True,
    )
    assert v_sup == 0.0 and w_sup == 0.0, "CBF-decelerated sub-threshold command must stay 0.0, never boost"
    v_norm, w_norm = apply_motor_deadzone(
        0.012, -0.04, min_v=0.025, min_w=0.08, v_max=0.15, w_max=0.55,
        suppress_v_boost=False, suppress_w_boost=False,
    )
    assert abs(v_norm - 0.025) < 1e-6 and abs(w_norm + 0.08) < 1e-6

    # 2) SemanticObjectCache: marker_id==0 support, ICP SE(2) sync, corr_se2 ingestion, 20 Hz gas non-dragging
    cache = SemanticObjectCache(dedup_radius_m=0.45, default_ch2_radius_m=0.55)
    cache.upsert_object("chair", 1.0, 0.0, radius=0.20, channel=1, marker_id=0)
    cache.upsert_object("chair", 1.05, 0.0, radius=0.22, channel=1, marker_id=0)
    assert len(cache) == 1, "marker_id=0 must deduplicate cleanly"

    cache.record_local_gas_hazard(x=2.00, y=1.00, gas_level=0.50)
    for k in range(20):
        # Robot moves 0.02m/tick within dedup_radius_m (20 Hz gas callback): anchor must NOT drift!
        cache.record_local_gas_hazard(x=2.00 + 0.015 * k, y=1.00, gas_level=0.70)
    _, ch2_list = cache.get_channel_tuples()
    assert len(ch2_list) == 1
    assert abs(ch2_list[0][0] - 2.00) < 1e-6 and abs(ch2_list[0][1] - 1.00) < 1e-6

    # Sync ICP jump (dx=+0.5, dy=-0.2, dth=pi/2)
    cache.sync_icp_correction((0.0, 0.0, 0.0), (0.5, -0.2, 0.5 * math.pi))
    ch1_s, ch2_s = cache.get_channel_tuples()
    # (1.05, 0.0) rotated +90 deg -> (0.0, 1.05) + (0.5, -0.2) = (0.5, 0.85)
    assert abs(ch1_s[0][0] - 0.50) < 1e-5 and abs(ch1_s[0][1] - 0.85) < 1e-5
    # Incoming odom JSON transformed by same corr_se2 lands in identical corrected frame
    cache.update_from_json(
        json.dumps({"frame_id": "odom", "objects": [{"marker_id": 5, "class_name": "chair", "x": 1.05, "y": 0.0, "channel": 1}]}),
        corr_se2=(0.5, -0.2, 0.5 * math.pi),
    )
    ch1_after, _ = cache.get_channel_tuples()
    assert len(ch1_after) == 1, "Transformed odom object must deduplicate with ICP-synced cached entry"

    # 3) Homotopy blending reached even when alpha_rl == 0.0 with active Ch3 > 0.05
    class _DummyRLPolicy:
        patch_channels = 4
        last_ch123_peak = 0.0
        predict_calls = 0

        def predict(self, *args, **kwargs):
            self.predict_calls += 1
            ch3 = kwargs.get("dynamic_ch3")
            self.last_ch123_peak = float(np.max(ch3)) if ch3 is not None else 0.0
            return -0.06, 0.45

        def observe(self, *args, **kwargs):
            self.last_ch123_peak = 0.0

    ctrl = object.__new__(RecoveryController)
    ctrl.enable_cbf_safety_shield = True
    ctrl.enable_lidar_slip_residual = False
    ctrl.icp_degenerate = False
    ctrl._prev_slip_scan_pts = None
    ctrl.actual_v = 0.10
    ctrl.actual_w = 0.0
    ctrl.last_cmd_v = 0.10
    ctrl.last_cmd_w = 0.0
    ctrl.current_x = 0.0
    ctrl.current_y = 0.0
    ctrl.current_theta = 0.0
    ctrl.cbf_v_min = -0.08
    ctrl.v_max = 0.20
    ctrl.w_max = 0.65
    ctrl.is_phase2b = True
    ctrl.latest_raw_ranges = [2.0] * 360
    ctrl.latest_scan_angle_min = 0.0
    ctrl.latest_scan_angle_increment = None
    ctrl.context_rl_policy = _DummyRLPolicy()
    ctrl.latest_dynamic_ch3 = np.zeros((12, 12), dtype=np.float32)
    ctrl.latest_dynamic_ch3[7, 6] = 0.80  # > 0.05 => g_res == 1.0

    v_out, w_out, g_res = ctrl._compute_homotopy_rl_command(
        v_base=0.18, w_base=-0.10, alpha_rl=0.0, target_xy=(1.0, 0.0), step_norm=0.1,
        sem_obs=[], sem_haz=[], sem_prox=0.0, left_min=0.25, right_min=0.25,
    )
    assert ctrl.context_rl_policy.predict_calls == 1, "predict() must run even when alpha_rl==0.0 if Ch3 active"
    assert g_res == pytest.approx(1.0, abs=1e-6)
    assert v_out == pytest.approx(-0.06, abs=1e-6) and w_out == pytest.approx(0.45, abs=1e-6)

    # 4) Far-from-origin SE(2) ICP must NOT reject valid small pose correction due to origin lever-arm
    ctrl.enable_scan_matching = True
    ctrl.scan_match_weight = 0.50
    ctrl.scan_match_max_dist = 0.60
    ctrl.docking_dist_th = 0.50
    ctrl.raw_odom_x = 12.0
    ctrl.raw_odom_y = 0.0
    ctrl.raw_odom_theta = 0.0
    ctrl.corr_x = 0.0
    ctrl.corr_y = 0.0
    ctrl.corr_th = 0.0
    ctrl.current_x = 12.0
    ctrl.current_y = 0.0
    ctrl.current_theta = 0.0
    ctrl.scan_match_count = 0
    ctrl.dynamic_tracker = None
    ctrl.semantic_cache = SemanticObjectCache()
    ctrl.semantic_cache.upsert_object("chair", 13.0, 0.0, channel=1)
    ctrl.start_pose = None
    ctrl.start_scan_pts = None

    class _DummyLogger:
        def info(self, *_a, **_k): pass
        def warn(self, *_a, **_k): pass

    ctrl.get_logger = lambda: _DummyLogger()
    # Build well-conditioned 4-pillar 2D scan in ref frame, rotated by +0.08 rad (lever arm at x=12m would be 0.96m!)
    ang_p = np.linspace(0.0, 2.0 * math.pi, 20, endpoint=False, dtype=np.float32)
    pillar_centers = [(1.4, 0.9), (-1.3, 1.1), (-1.1, -1.2), (1.5, -0.8)]
    ref_pts = np.vstack([
        np.column_stack([cx + 0.12 * np.cos(ang_p), cy + 0.12 * np.sin(ang_p)])
        for cx, cy in pillar_centers
    ]).astype(np.float32)
    dth_true = 0.08
    c_t, s_t = math.cos(dth_true), math.sin(dth_true)
    # Latest scan observed at robot rotated +0.08 rad relative to ref_pts => R(-dth) @ ref_pts
    R_inv = np.array([[c_t, s_t], [-s_t, c_t]], dtype=np.float32)
    ctrl.latest_scan_pts = (ref_pts @ R_inv.T).astype(np.float32)
    ctrl.active_target_wp = Waypoint(12.0, 0.0, 0.0, scan_pts=ref_pts)
    ctrl.perform_scan_matching_correction()
    assert ctrl.scan_match_count == 1, "Far-origin valid yaw ICP correction must not be rejected by origin lever-arm"
    assert abs(ctrl.current_theta - 0.5 * dth_true) < 0.015
    assert math.hypot(ctrl.current_x - 12.0, ctrl.current_y - 0.0) < 0.05
    ch1_synced, _ = ctrl.semantic_cache.get_channel_tuples()
    assert len(ch1_synced) == 1 and abs(ch1_synced[0][1] - math.sin(ctrl.current_theta)) < 0.03

    # 5) Consecutive-scan ICP velocity degeneracy detection (featureless parallel walls vs structured room)
    wall_pts = np.column_stack([np.linspace(-2.0, 2.0, 60, dtype=np.float32), np.full(60, 0.5, dtype=np.float32)])
    _, _, deg_wall = estimate_scan_velocity_icp(wall_pts, wall_pts, dt=0.10, prev_vw=(0.12, 0.0))
    assert deg_wall is True, "Collinear featureless wall must trigger icp_degenerate=True"
    shifted_pts = ref_pts - np.array([0.02, 0.0], dtype=np.float32)
    v_s, w_s, deg_ok = estimate_scan_velocity_icp(ref_pts, shifted_pts, dt=0.10, prev_vw=(0.18, 0.0))
    assert deg_ok is False and abs(v_s - 0.20) < 0.04 and abs(w_s) < 0.08

    # 6) Stage 0.1 semantic A* cost detour in ProposedRecoveryPlanner
    planner = ProposedRecoveryPlanner(size_m=4.0, resolution_m=0.05, inflation_radius_m=0.08)
    wp_stack = [Waypoint(1.5, 0.0, 0.0)]
    tag, v_p, w_p, sub_xy = planner.step(
        rx=0.0, ry=0.0, rth=0.0, waypoint_stack=wp_stack, start_pose=Waypoint(1.5, 0.0, 0.0),
        scan_pts=None, v_max=0.20, w_max=0.65, now_sec=1.0, wp_reach_radius=0.10, docking_dist_th=0.30,
        semantic_obstacles=[(0.55, 0.0, 0.18)], semantic_hazards=[(0.90, -0.15, 0.35)],
    )
    assert tag in ("ASTAR_TRACK", "ROTATE_ALIGN") and sub_xy is not None
    assert abs(sub_xy[1]) > 0.08, f"Semantic A* should detour laterally around Ch1/Ch2 obstacle, got subgoal={sub_xy}"

    # 7) GAS_-triggered recovery must NOT be auto-cancelled in Section 0 while active
    ctrl_src_path = os.path.join(PKG_ROOT, "disaster_recovery", "recovery_controller.py")
    with open(ctrl_src_path, "r", encoding="utf-8") as f:
        ctrl_src = f.read()
    assert 'str(self.fault_reason).startswith("GAS_")' in ctrl_src


def test_dynamic_tracker_ch3_r_infl_cap_and_cbf_ultra_close_scan():
    """Verify Part C: dynamic_tracker r_infl <= 0.28 cap and cbf_safety_shield 0<r<0.12 preservation."""
    from disaster_recovery.dynamic_tracker import CH3_MAX_R_INFL, DynamicObstacleTracker, TrackedObstacle
    from disaster_recovery.cbf_safety_shield import transform_scan_to_base_footprint

    assert CH3_MAX_R_INFL == pytest.approx(0.28)
    tracker = DynamicObstacleTracker()
    trk = TrackedObstacle(
        track_id=1,
        x=0.45,
        y=0.0,
        vx=0.0,
        vy=0.0,
        radius=0.14,
        P_4x4=np.diag([0.50, 0.50, 0.25, 0.25]).astype(np.float64),  # huge cov => uncapped 1.5*sqrt(lam) > 1.0
        last_update_stamp=1.0,
        age=5,
        hits=5,
        is_confirmed=True,
    )
    tracker.tracks[1] = trk
    _, _, r_eff_15, _ = tracker.predict_future(trk, 1.5)
    assert r_eff_15 - trk.radius <= CH3_MAX_R_INFL + 1e-6

    ch3 = tracker.project_ch3(rx=0.0, ry=0.0, rth=0.0, now_sec=1.0)
    # Center (0.45, 0.0), r_c=0.14, max r_eff = 0.42 -> cell at (-0.675, 0.0) is > 1.1m away => strictly 0.0!
    assert float(ch3[1, 6]) == 0.0, "Capped r_infl<=0.28 must prevent huge covariance from saturating 1.8m patch"
    assert float(ch3[8, 6]) > 0.5

    # CBF scan transform: 0 < r < 0.12 preserved at 0.12 m; NaN / -inf / 0.0 / -0.05 / >max_range excluded
    scan = np.array([0.07, 0.0, -0.05, float("nan"), float("-inf"), float("inf"), 4.0, 0.50], dtype=np.float64)
    pts = transform_scan_to_base_footprint(
        scan_ranges=scan, angle_min=0.0, angle_increment=0.1,
        lidar_offset_x=-0.032, lidar_offset_y=0.0, min_range=0.12, max_range=3.5,
    )
    assert pts.shape == (2, 2), f"Expected exactly 2 valid points (0.07->0.12 and 0.50), got shape {pts.shape}"
    assert pts[0, 0] == pytest.approx(0.12 - 0.032, abs=1e-6)
    assert pts[0, 1] == pytest.approx(0.0, abs=1e-6)


def test_env_single_raycast_subgoal_obs_and_eval_noisy_dr():
    """Verify Part D: single _cast_lidar per step, lookahead subgoal dist_norm, and v_min/Noisy-DR holdout eval."""
    env = VectorDisasterEnv(
        num_envs=4,
        track="sim",
        device=torch.device("cpu"),
        mode="train",
        base_seed=42,
        patch_channels=4,
        subgoal_obs_prob=1.0,
        subgoal_lookahead_range=(0.45, 0.80),
        odom_icp_alpha=0.05,
        odom_icp_interval=5,
    )
    obs_h, _ = env.reset_envs()
    # With subgoal_obs_prob=1.0 and lookahead in [0.45, 0.80], dist_norm <= 0.80 / 4.0 = 0.20
    assert float(obs_h[:, -1, 24].max().item()) <= 0.80 / 4.0 + 1e-4

    call_count = {"n": 0}
    orig_cast = env._cast_lidar

    def _counted_cast():
        call_count["n"] += 1
        return orig_cast()

    env._cast_lidar = _counted_cast
    env.step(torch.zeros((4, 2), dtype=torch.float32))
    assert call_count["n"] == 1, f"step() must call _cast_lidar() exactly once, got {call_count['n']}"

    # Verify evaluate_policy_on_holdout forwards v_min and supports deterministic Noisy-DR eval
    eval_8 = make_eval_seeds(8, 10000)
    def zero_pol(h: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        del p
        return torch.zeros((h.shape[0], 2), dtype=torch.float32, device=h.device)

    r_ndr1 = evaluate_policy_on_holdout(
        "real_tb3", torch.device("cpu"), zero_pol, max_steps=10,
        patch_channels=4, eval_seeds=eval_8, v_min=-0.08, enable_noisy_dr=True,
    )
    r_ndr2 = evaluate_policy_on_holdout(
        "real_tb3", torch.device("cpu"), zero_pol, max_steps=10,
        patch_channels=4, eval_seeds=eval_8, v_min=-0.08, enable_noisy_dr=True,
    )
    assert r_ndr1["v_min"] == pytest.approx(-0.08, abs=1e-6)
    assert r_ndr1["enable_noisy_dr"] is True
    assert r_ndr1["mean_cost"] == r_ndr2["mean_cost"] and r_ndr1["cvar10_cost"] == r_ndr2["cvar10_cost"]
