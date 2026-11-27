#!/usr/bin/env python3
"""
High-Throughput Vectorized Procedural Disaster Simulator (PyTorch CUDA / CPU).

Supports Phase 2 V100-32GB Dual-Track Context-Adaptive RL:
  - Track A ('sim'): Gazebo-matched physical & LiDAR randomization for Sim-to-Sim.
  - Track B ('real_tb3'): Aggressive real TurtleBot3 physical imperfections
    (wheel odometry drift, L/R slip, motor deadzone, 1-3 tick lag, LDS-01/LD08
    noise/dropout/glare) for Sim-to-Real Raspberry Pi deployment.

Zero Data Leakage Guarantee:
  - Training batches generate fresh randomized procedural episodes (seeds >= 100,000)
    or replay hard training seeds via Prioritized Level Replay (PLR).
  - Evaluation mode evaluates strictly isolated holdout seeds (10,000 .. 10,059),
    completely disjoint from training & official Gazebo 5 benchmark worlds.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


BASE_REGIME_NAMES = (
    "collapsed_debris",
    "tight_corridor",
    "dynamic_crossing",
    "plinth_slip_drift",
    "u_pocket_culdesac",
    "chicane_door_pinch",
)
SEMANTIC_REGIME_NAMES = (
    "low_obstacle_field",
    "gas_victim_corridor",
)
REGIME_NAMES = BASE_REGIME_NAMES + SEMANTIC_REGIME_NAMES
NUM_BASE_REGIMES = len(BASE_REGIME_NAMES)
NUM_REGIMES = len(REGIME_NAMES)

TB3_MAX_V = 0.20
TB3_MIN_V = -0.10  # Allow short unpin/reverse escape moves!
TB3_MAX_W = 0.65
TB3_RADIUS = 0.095  # Physical half-width 0.089m + slight corner envelope
COLLISION_DIST = 0.108
GOAL_REACH_RADIUS = 0.18
MAX_RANGE_M = 3.5

# Observation constants
NUM_RAYS = 72
NUM_LIDAR_BINS = 24
PATCH_SIZE = 12
PATCH_CHANNELS = 4  # Ch0: LiDAR, Ch1: Low obs, Ch2: Gas/Survivor, Ch3: Dynamic future occupancy
PATCH_SPAN_M = 1.8  # +-0.9m around robot center -> 0.15m per cell
STEP_FEAT_DIM = 34  # 24 lidar + 3 goal polar + 2 cmd + 2 actual vel + 2 tracking err + 1 regime/step norm
HISTORY_LEN = 16    # 1.6 seconds at 10 Hz

# Stage 3/4 pure physical safety cost weights (c_t excludes CBF intervention & slack)
COST_W_COL = 10.0
COST_W_LOW = 1.5
COST_W_GAS = 1.0
COST_W_DYN = 1.0
COST_DYN_MARGIN_M = 0.28

# Stage 1 / Stage 3 Ch3 future-occupancy prediction constants (aligned with dynamic_tracker.py q_acc_base=0.30)
CH3_HORIZONS_S = (0.5, 1.0, 1.5)
CH3_WEIGHTS = (1.0, 0.75, 0.50)
CH3_DAMPING_BETA = 0.35
CH3_Q_ACC = 0.30
CH3_EPS = 1e-3
CH3_MAX_R_INFL = 0.28


@dataclass
class TrackConfig:
    name: str
    lidar_noise_std_range: Tuple[float, float]
    lidar_dropout_prob_range: Tuple[float, float]
    lidar_glare_prob: float
    odom_scale_range: Tuple[float, float]
    odom_yaw_bias_range: Tuple[float, float]  # rad per meter traveled
    slip_gain_range: Tuple[float, float]      # physical speed scale
    turn_bias_range: Tuple[float, float]      # unwanted yaw drift per linear vel
    deadzone_v: float
    deadzone_w: float
    act_lag_max: int                          # max ticks delay (0..3)


SIM_TRACK = TrackConfig(
    name="sim",
    lidar_noise_std_range=(0.008, 0.018),
    lidar_dropout_prob_range=(0.01, 0.07),
    lidar_glare_prob=0.02,
    odom_scale_range=(0.97, 1.03),
    odom_yaw_bias_range=(-0.012, 0.012),
    slip_gain_range=(0.88, 1.04),
    turn_bias_range=(-0.04, 0.04),
    deadzone_v=0.0,
    deadzone_w=0.0,
    act_lag_max=1,
)

REAL_TB3_TRACK = TrackConfig(
    name="real_tb3",
    lidar_noise_std_range=(0.012, 0.028),
    lidar_dropout_prob_range=(0.03, 0.08),
    lidar_glare_prob=0.015,
    odom_scale_range=(0.95, 1.05),
    odom_yaw_bias_range=(-0.030, 0.030),
    slip_gain_range=(0.74, 1.04),
    turn_bias_range=(-0.10, 0.10),
    deadzone_v=0.015,
    deadzone_w=0.045,
    act_lag_max=2,
)


class VectorDisasterEnv:
    """
    Batched 2D/2.5D procedural disaster navigation simulator running natively on PyTorch CUDA/CPU.
    Simulates B parallel TurtleBot3 robots with:
      - Rectangular walls/partitions (up to MAX_BOXES=8 per env)
      - Static + moving circular debris obstacles (up to MAX_CYLS=14 per env)
      - Sub-LiDAR low plinths & low-obstacle fields (invisible on 2D LiDAR, visible in Ch 1)
      - Gas hazard & survivor protection zones (soft safety field in Ch 2)
      - Modality dropout (~25-30% Ch 1/2 blackout) & realistic semantic perception noise
      - Domain-randomized slip, actuator lag/deadzone, odom drift, and LiDAR artifacts
    """

    MAX_BOXES = 8
    MAX_CYLS = 14
    MAX_LOW_OBS = 8
    MAX_HAZARDS = 6

    def __init__(
        self,
        num_envs: int = 256,
        track: str = "sim",
        device: Optional[torch.device] = None,
        max_steps: int = 320,
        dt: float = 0.10,
        mode: str = "train",
        base_seed: int = 100_000,
        patch_channels: int = PATCH_CHANNELS,
        use_semantic_channels: Optional[bool] = None,
        include_semantic_regimes: Optional[bool] = None,
        modality_dropout_prob: Optional[float] = None,
        enable_semantic_noise: Optional[bool] = None,
        semantic_pos_jitter_std: float = 0.06,
        semantic_reg_delay_steps: int = 3,
        semantic_miss_prob: float = 0.08,
        semantic_fp_prob: float = 0.10,
        enable_sinusoidal_dynamic: Optional[bool] = None,
        enable_ch3_dr: Optional[bool] = None,
        ch3_delay_ticks_range: Tuple[int, int] = (2, 4),
        ch3_vel_noise_std_range: Tuple[float, float] = (0.05, 0.12),
        ch3_occlusion_drop_range: Tuple[float, float] = (0.10, 0.20),
        enable_plicp_vel_noise_dr: Optional[bool] = None,
        plicp_v_noise_std_range: Tuple[float, float] = (0.01, 0.03),
        plicp_w_noise_std_range: Tuple[float, float] = (0.02, 0.05),
        plicp_hold_ticks: int = 2,
        v_min: float = TB3_MIN_V,
        enable_strict_real_dr: Optional[bool] = None,
        dr_profile: Optional[str] = None,
        subgoal_obs_prob: Optional[float] = None,
        subgoal_lookahead_range: Tuple[float, float] = (0.45, 1.50),
        odom_icp_alpha: float = 0.30,
        odom_icp_interval: int = 1,
    ):
        self.B = int(num_envs)
        self.track_cfg = REAL_TB3_TRACK if track == "real_tb3" else SIM_TRACK
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.mode = mode
        self.base_seed = int(base_seed)
        self.rng = np.random.RandomState(self.base_seed)
        self.v_min = float(v_min)
        if dr_profile is not None:
            dp = str(dr_profile).strip().lower()
            if dp not in ("legacy", "strict"):
                raise ValueError(f"dr_profile must be 'legacy' or 'strict', got {dr_profile!r}")
            self.dr_profile = dp
            self.enable_strict_real_dr = (dp == "strict")
        elif enable_strict_real_dr is not None:
            self.enable_strict_real_dr = bool(enable_strict_real_dr)
            self.dr_profile = "strict" if self.enable_strict_real_dr else "legacy"
        else:
            # Default: Track B ('real_tb3') enables strict DR automatically so real-robot DR is never silently disabled;
            # Track A ('sim') defaults to 'legacy' for Phase 2-B/3 regression parity.
            self.enable_strict_real_dr = (self.track_cfg.name == "real_tb3")
            self.dr_profile = "strict" if self.enable_strict_real_dr else "legacy"
        self.odom_icp_alpha = float(np.clip(odom_icp_alpha, 0.0, 1.0))
        self.odom_icp_interval = int(max(1, odom_icp_interval))

        if patch_channels not in (1, 3, 4):
            raise ValueError(f"patch_channels must be 1, 3, or 4, got {patch_channels}")
        self.patch_channels = int(patch_channels)
        if use_semantic_channels is None:
            self.use_semantic_channels = self.patch_channels >= 3
        else:
            self.use_semantic_channels = bool(use_semantic_channels) and (self.patch_channels >= 3)

        if include_semantic_regimes is None:
            self.include_semantic_regimes = bool(self.use_semantic_channels)
        else:
            self.include_semantic_regimes = bool(include_semantic_regimes)
        self.num_active_regimes = NUM_REGIMES if self.include_semantic_regimes else NUM_BASE_REGIMES

        if modality_dropout_prob is None:
            self.modality_dropout_prob = 0.28 if mode == "train" else 0.0
        else:
            self.modality_dropout_prob = float(np.clip(modality_dropout_prob, 0.0, 1.0))

        if enable_semantic_noise is None:
            self.enable_semantic_noise = (mode == "train")
        else:
            self.enable_semantic_noise = bool(enable_semantic_noise)

        self.semantic_pos_jitter_std = float(max(0.0, semantic_pos_jitter_std))
        self.semantic_reg_delay_steps = int(max(0, semantic_reg_delay_steps))
        self.semantic_miss_prob = float(np.clip(semantic_miss_prob, 0.0, 1.0))
        self.semantic_fp_prob = float(np.clip(semantic_fp_prob, 0.0, 1.0))
        self.ch2_clean_env_frac = 0.0
        self._clean_ch2_env_mask = torch.zeros(self.B, dtype=torch.bool, device=self.device)

        # Phase 4 dynamic/sensor DR toggles (default active for 4-Ch training; zero impact on 1-Ch/3-Ch legacy)
        self.enable_sinusoidal_dynamic = (
            (self.patch_channels >= 4) if enable_sinusoidal_dynamic is None else bool(enable_sinusoidal_dynamic)
        )
        self.enable_ch3_dr = (
            ((mode == "train") and (self.patch_channels >= 4)) if enable_ch3_dr is None else bool(enable_ch3_dr)
        )
        self.ch3_delay_ticks_range = (
            int(max(0, min(4, ch3_delay_ticks_range[0]))),
            int(max(0, min(4, ch3_delay_ticks_range[1]))),
        )
        self.ch3_vel_noise_std_range = (float(max(0.0, ch3_vel_noise_std_range[0])), float(max(0.0, ch3_vel_noise_std_range[1])))
        self.ch3_occlusion_drop_range = (
            float(np.clip(ch3_occlusion_drop_range[0], 0.0, 1.0)),
            float(np.clip(ch3_occlusion_drop_range[1], 0.0, 1.0)),
        )
        self.enable_plicp_vel_noise_dr = (
            ((mode == "train") and (self.patch_channels >= 4))
            if enable_plicp_vel_noise_dr is None
            else bool(enable_plicp_vel_noise_dr)
        )
        self.plicp_v_noise_std_range = (float(max(0.0, plicp_v_noise_std_range[0])), float(max(0.0, plicp_v_noise_std_range[1])))
        self.plicp_w_noise_std_range = (float(max(0.0, plicp_w_noise_std_range[0])), float(max(0.0, plicp_w_noise_std_range[1])))
        self.plicp_hold_ticks = int(max(1, plicp_hold_ticks))
        if subgoal_obs_prob is None:
            self.subgoal_obs_prob = 0.50 if ((mode == "train") and (self.patch_channels >= 4)) else 0.0
        else:
            self.subgoal_obs_prob = float(np.clip(subgoal_obs_prob, 0.0, 1.0))
        self.subgoal_lookahead_range = (
            float(max(0.25, subgoal_lookahead_range[0])),
            float(max(subgoal_lookahead_range[0], subgoal_lookahead_range[1])),
        )

        # Dedicated LiDAR, semantic & Phase 4 RNGs for exact determinism and stream isolation
        lidar_seed = (self.base_seed ^ 0x1D4A7C3B) & 0x7FFFFFFF
        self.lidar_torch_gen = torch.Generator(device=self.device)
        self.lidar_torch_gen.manual_seed(lidar_seed)

        sem_seed = (self.base_seed ^ 0x5E4A471C) & 0x7FFFFFFF
        self.sem_rng = np.random.RandomState(sem_seed)
        self.sem_torch_gen = torch.Generator(device=self.device)
        self.sem_torch_gen.manual_seed(sem_seed)

        p4_seed = (self.base_seed ^ 0x4A5E4D2B) & 0x7FFFFFFF
        self.p4_rng = np.random.RandomState(p4_seed)
        self.p4_torch_gen = torch.Generator(device=self.device)
        self.p4_torch_gen.manual_seed(p4_seed)

        # Precompute Stage 1 damped-CV Ch3 horizon displacement & capped covariance inflation constants
        ch3_a_list: List[float] = []
        ch3_denom_list: List[float] = []
        beta = CH3_DAMPING_BETA
        p0_pos = 0.02 * 0.02
        p0_vel = 0.10 * 0.10
        for tau in CH3_HORIZONS_S:
            g = math.exp(-beta * tau)
            a_t = (1.0 - g) / beta
            g2 = g * g
            q_pp = (CH3_Q_ACC / (beta * beta)) * (tau - 2.0 * (1.0 - g) / beta + (1.0 - g2) / (2.0 * beta))
            lam_max = max(0.0, p0_pos + (a_t * a_t) * p0_vel + max(0.0, q_pp))
            r_infl = min(1.5 * math.sqrt(lam_max), CH3_MAX_R_INFL)
            ch3_a_list.append(a_t)
            ch3_denom_list.append(max(r_infl, CH3_EPS))
        self.ch3_a_tau = torch.tensor(ch3_a_list, dtype=torch.float32, device=self.device)  # (H,)
        self.ch3_w_tau = torch.tensor(CH3_WEIGHTS, dtype=torch.float32, device=self.device)  # (H,)
        self.ch3_denom_tau = torch.tensor(ch3_denom_list, dtype=torch.float32, device=self.device)  # (H,)

        # Precompute ray unit vectors in robot frame (NUM_RAYS = 72, centered so bin 0 ~ 0 deg)
        # Matching TD3 extract_state_vector: rays from 0..2pi, rolled by half_bin when binning 72->24.
        ray_angs = np.linspace(0.0, 2.0 * math.pi, NUM_RAYS, endpoint=False, dtype=np.float32)
        self.ray_cos_local = torch.tensor(np.cos(ray_angs), device=self.device)  # (72,)
        self.ray_sin_local = torch.tensor(np.sin(ray_angs), device=self.device)  # (72,)

        # Precompute 12x12 ego patch cell centers in robot frame (+x forward, +y left)
        cell_res = PATCH_SPAN_M / PATCH_SIZE
        half_span = 0.5 * PATCH_SPAN_M
        centers_1d = torch.linspace(
            -half_span + 0.5 * cell_res,
            half_span - 0.5 * cell_res,
            PATCH_SIZE,
            dtype=torch.float32,
            device=self.device,
        )
        gx_2d, gy_2d = torch.meshgrid(centers_1d, centers_1d, indexing="ij")
        self.ego_grid_x = gx_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, 12, 12)
        self.ego_grid_y = gy_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, 12, 12)

        # Arena bounding walls (x_min, x_max, y_min, y_max)
        self.bounds = torch.zeros((self.B, 4), dtype=torch.float32, device=self.device)

        # Boxes: (B, MAX_BOXES, 4) -> (cx, cy, hx, hy) axis-aligned rectangles
        self.box_active = torch.zeros((self.B, self.MAX_BOXES), dtype=torch.bool, device=self.device)
        self.boxes = torch.zeros((self.B, self.MAX_BOXES, 4), dtype=torch.float32, device=self.device)

        # Cylinders (static debris + moving dynamic obstacles): (B, MAX_CYLS, 5) -> (cx, cy, r, vx, vy)
        self.cyl_active = torch.zeros((self.B, self.MAX_CYLS), dtype=torch.bool, device=self.device)
        self.cyls = torch.zeros((self.B, self.MAX_CYLS, 5), dtype=torch.float32, device=self.device)
        # Sinusoidal reciprocating kinematics (Stage 3): (x0, y0, amp_x, amp_y, omega, phi)
        self.cyl_sin_active = torch.zeros((self.B, self.MAX_CYLS), dtype=torch.bool, device=self.device)
        self.cyl_sin_params = torch.zeros((self.B, self.MAX_CYLS, 6), dtype=torch.float32, device=self.device)
        # Ch3 0.2~0.4s (2~4 ticks) delay ring buffer: (B, 5, MAX_CYLS, 6) -> (cx, cy, r, vx, vy, is_dyn)
        self.cyl_hist_fifo = torch.zeros((self.B, 5, self.MAX_CYLS, 6), dtype=torch.float32, device=self.device)
        self.dyn_seen_ticks = torch.zeros((self.B, self.MAX_CYLS), dtype=torch.long, device=self.device)
        self.ch3_delay_ticks = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.ch3_vel_noise_std = torch.zeros(self.B, dtype=torch.float32, device=self.device)
        self.ch3_drop_prob = torch.zeros(self.B, dtype=torch.float32, device=self.device)
        # PLICP 5 Hz sample-hold velocity noise DR state
        self.plicp_v_std = torch.zeros(self.B, dtype=torch.float32, device=self.device)
        self.plicp_w_std = torch.zeros(self.B, dtype=torch.float32, device=self.device)
        self.plicp_phase = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.plicp_held_vw = torch.zeros((self.B, 2), dtype=torch.float32, device=self.device)
        # Train-deploy dist_norm alignment: optional lookahead subgoal target observation (0.45..1.50 m)
        self.use_subgoal_obs = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self.subgoal_lookahead_m = torch.full((self.B,), 1.0, dtype=torch.float32, device=self.device)
        # Cached latest noisy LiDAR scan and observed Ch1/Ch2/Ch3 states for BatchSoftCBFShield
        self._last_lidar_ranges = torch.full((self.B, NUM_RAYS), MAX_RANGE_M, dtype=torch.float32, device=self.device)
        self._obs_ch3_vnoise = torch.zeros((self.B, self.MAX_CYLS, 2), dtype=torch.float32, device=self.device)
        self._obs_ch3_keep = torch.ones((self.B, self.MAX_CYLS), dtype=torch.bool, device=self.device)
        self._obs_low_state = torch.zeros((self.B, self.MAX_LOW_OBS, 4), dtype=torch.float32, device=self.device)
        self._obs_haz_state = torch.zeros((self.B, self.MAX_HAZARDS, 4), dtype=torch.float32, device=self.device)

        # Sub-LiDAR invisible plinth (at most 1 per env for regime 3): (cx, cy, r, active)
        self.plinth = torch.zeros((self.B, 4), dtype=torch.float32, device=self.device)

        # Ch 1: Sub-LiDAR low-lying obstacles (B, MAX_LOW_OBS, 3) -> (cx, cy, r)
        self.low_obs_active = torch.zeros((self.B, self.MAX_LOW_OBS), dtype=torch.bool, device=self.device)
        self.low_obs = torch.zeros((self.B, self.MAX_LOW_OBS, 3), dtype=torch.float32, device=self.device)

        # Ch 2: Gas hazard (kind=0) & survivor protection zones (kind=1): (B, MAX_HAZARDS, 4) -> (cx, cy, r, kind)
        self.hazard_active = torch.zeros((self.B, self.MAX_HAZARDS), dtype=torch.bool, device=self.device)
        self.hazard_zones = torch.zeros((self.B, self.MAX_HAZARDS, 4), dtype=torch.float32, device=self.device)

        # Semantic perception noise & modality dropout state tensors
        # Per-episode semantic perception noise state (Ch1 tracks MAX_LOW_OBS only; Regime 3 plinth is tactile-only)
        n_low_slots = self.MAX_LOW_OBS
        self.low_obs_bias = torch.zeros((self.B, n_low_slots, 2), dtype=torch.float32, device=self.device)
        self.low_obs_reg_delay = torch.zeros((self.B, n_low_slots), dtype=torch.long, device=self.device)
        self.low_obs_seen_ticks = torch.zeros((self.B, n_low_slots), dtype=torch.long, device=self.device)

        self.hazard_bias = torch.zeros((self.B, self.MAX_HAZARDS, 2), dtype=torch.float32, device=self.device)
        self.hazard_reg_delay = torch.zeros((self.B, self.MAX_HAZARDS), dtype=torch.long, device=self.device)
        self.hazard_seen_ticks = torch.zeros((self.B, self.MAX_HAZARDS), dtype=torch.long, device=self.device)

        self.ep_modality_dropped = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self.modality_dropped = torch.zeros(self.B, dtype=torch.bool, device=self.device)

        # Robot true & estimated (odom-drifted) state
        self.rx = torch.zeros(self.B, dtype=torch.float32, device=self.device)
        self.ry = torch.zeros(self.B, dtype=torch.float32, device=self.device)
        self.rth = torch.zeros(self.B, dtype=torch.float32, device=self.device)

        self.odom_x = torch.zeros(self.B, dtype=torch.float32, device=self.device)
        self.odom_y = torch.zeros(self.B, dtype=torch.float32, device=self.device)
        self.odom_th = torch.zeros(self.B, dtype=torch.float32, device=self.device)

        self.goal_x = torch.zeros(self.B, dtype=torch.float32, device=self.device)
        self.goal_y = torch.zeros(self.B, dtype=torch.float32, device=self.device)

        # Episode metadata & curriculum descriptors
        self.regime = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.ep_seed = np.zeros(self.B, dtype=np.int64)
        self.steps = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.prev_dist = torch.zeros(self.B, dtype=torch.float32, device=self.device)
        self.min_dist_seen = torch.zeros(self.B, dtype=torch.float32, device=self.device)
        self.stall_steps = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.bypass_bonus_given = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self.plinth_escape_bonus_given = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self.plinth_unpin_bonus_given = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self.hit_plinth = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self.hit_low_obs = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self.prev_hit_low = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self.low_clear_bonus_given = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self.entered_hazard = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self.prev_in_hazard = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self.hazard_clear_bonus_given = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self.gate1_x = torch.full((self.B,), 999.0, dtype=torch.float32, device=self.device)
        self.gate2_x = torch.full((self.B,), 999.0, dtype=torch.float32, device=self.device)
        self.gate1_done = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self.gate2_done = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self.nav_wp = torch.zeros((self.B, 2, 2), dtype=torch.float32, device=self.device)
        self.wp0_seg_half_y = torch.zeros(self.B, dtype=torch.float32, device=self.device)
        self.nav_stage = torch.zeros(self.B, dtype=torch.long, device=self.device)

        # Per-env randomized physics/sensor params
        self.lidar_noise_std = torch.zeros(self.B, dtype=torch.float32, device=self.device)
        self.lidar_drop_prob = torch.zeros(self.B, dtype=torch.float32, device=self.device)
        self.odom_scale = torch.ones(self.B, dtype=torch.float32, device=self.device)
        self.odom_yaw_bias = torch.zeros(self.B, dtype=torch.float32, device=self.device)
        self.slip_v = torch.ones(self.B, dtype=torch.float32, device=self.device)
        self.slip_w = torch.ones(self.B, dtype=torch.float32, device=self.device)
        self.turn_bias = torch.zeros(self.B, dtype=torch.float32, device=self.device)
        self.act_lag = torch.zeros(self.B, dtype=torch.long, device=self.device)

        # Action lag FIFO (B, 4, 2)
        self.act_fifo = torch.zeros((self.B, 4, 2), dtype=torch.float32, device=self.device)
        self.last_cmd = torch.zeros((self.B, 2), dtype=torch.float32, device=self.device)
        self.actual_vw = torch.zeros((self.B, 2), dtype=torch.float32, device=self.device)

        # Temporal observation buffer (B, HISTORY_LEN, STEP_FEAT_DIM) & latest 12x12 local patch (B, C, 12, 12)
        self.obs_hist = torch.zeros((self.B, HISTORY_LEN, STEP_FEAT_DIM), dtype=torch.float32, device=self.device)
        self.local_patch = torch.zeros(
            (self.B, self.patch_channels, PATCH_SIZE, PATCH_SIZE), dtype=torch.float32, device=self.device
        )

        # PLR hard-case pool (training mode only)
        self.hard_seeds: List[Tuple[int, int]] = []  # (seed, regime)

    # ------------------------------------------------------------------
    # Procedural Episode Generation (Pure Geometry, Zero Leakage)
    # ------------------------------------------------------------------
    def _build_single_layout(self, seed: int, forced_regime: Optional[int] = None) -> dict:
        r = np.random.RandomState(int(seed))
        regime = int(forced_regime) if forced_regime is not None else int(r.randint(0, self.num_active_regimes))

        bounds = np.array([-2.2, 4.2, -1.8, 1.8], dtype=np.float32)
        boxes = np.zeros((self.MAX_BOXES, 4), dtype=np.float32)
        box_act = np.zeros(self.MAX_BOXES, dtype=bool)
        cyls = np.zeros((self.MAX_CYLS, 5), dtype=np.float32)
        cyl_act = np.zeros(self.MAX_CYLS, dtype=bool)
        cyl_sin_act = np.zeros(self.MAX_CYLS, dtype=bool)
        cyl_sin_params = np.zeros((self.MAX_CYLS, 6), dtype=np.float32)
        plinth = np.zeros(4, dtype=np.float32)
        low_obs = np.zeros((self.MAX_LOW_OBS, 3), dtype=np.float32)
        low_obs_act = np.zeros(self.MAX_LOW_OBS, dtype=bool)
        hazards = np.zeros((self.MAX_HAZARDS, 4), dtype=np.float32)
        hazard_act = np.zeros(self.MAX_HAZARDS, dtype=bool)

        rx, ry = 0.0, float(r.uniform(-0.12, 0.12))
        rth = float(r.uniform(-0.35, 0.35))
        gx, gy = 2.2, 0.0
        wp1 = None
        wp2 = None
        wp0_seg_half_y = 0.0

        if regime == 0:
            # Regime 0: Collapsed open clutter + U-pockets (3..10 rocks)
            half_w = float(r.uniform(1.3, 2.3))
            bounds = np.array([-1.2, 4.2, -half_w, half_w], dtype=np.float32)
            gx = float(r.uniform(2.1, 3.4))
            gy = float(r.uniform(-half_w * 0.55, half_w * 0.55))

            n_rocks = int(r.randint(4, 11))
            placed = []
            for k in range(n_rocks):
                for _ in range(25):
                    rad = float(r.uniform(0.14, 0.23))
                    cx = float(r.uniform(0.42, gx - 0.35))
                    cy = float(r.uniform(-half_w + rad + 0.14, half_w - rad - 0.14))
                    if math.hypot(cx - rx, cy - ry) < rad + 0.32:
                        continue
                    if math.hypot(cx - gx, cy - gy) < rad + 0.32:
                        continue
                    if all(math.hypot(cx - px, cy - py) >= rad + pr + 0.24 for px, py, pr in placed):
                        placed.append((cx, cy, rad))
                        cyls[k] = [cx, cy, rad, 0.0, 0.0]
                        cyl_act[k] = True
                        break

        elif regime == 1:
            # Regime 1: Tight indoor corridor / doorway pinch (width 0.85..1.25m) + rocks leaving >=0.34m gap
            corr_w = float(r.uniform(0.85, 1.25))
            half_c = 0.5 * corr_w
            turn_corner = bool(r.rand() < 0.45)

            if not turn_corner:
                bounds = np.array([-0.9, 3.6, -half_c, half_c], dtype=np.float32)
                rx, ry = 0.0, float(r.uniform(-half_c * 0.25, half_c * 0.25))
                rth = float(r.uniform(-0.20, 0.20))
                gx = float(r.uniform(2.0, 2.9))
                gy = float(r.uniform(-half_c * 0.30, half_c * 0.30))

                # Doorway jamb pair (leaves >= 0.48m opening)
                door_x = float(r.uniform(1.05, 1.65))
                gap_w = float(r.uniform(0.48, min(0.70, corr_w * 0.85)))
                gap_y = float(r.uniform(-half_c + 0.5 * gap_w + 0.04, half_c - 0.5 * gap_w - 0.04))
                wp1 = (door_x, gap_y)
                # Top jamb box
                top_h = half_c - (gap_y + 0.5 * gap_w)
                if top_h > 0.03:
                    boxes[0] = [door_x, gap_y + 0.5 * gap_w + 0.5 * top_h, 0.08, 0.5 * top_h]
                    box_act[0] = True
                # Bottom jamb box
                bot_h = (gap_y - 0.5 * gap_w) - (-half_c)
                if bot_h > 0.03:
                    boxes[1] = [door_x, -half_c + 0.5 * bot_h, 0.08, 0.5 * bot_h]
                    box_act[1] = True

                # Place 1..2 rocks offset so AT LEAST one side has gap >= 0.34m!
                rock_idx = 0
                for rx_pos in (float(r.uniform(0.50, door_x - 0.32)), float(r.uniform(door_x + 0.32, gx - 0.32))):
                    if rock_idx >= 2:
                        break
                    rad = float(r.uniform(0.14, 0.18))
                    side = 1.0 if r.rand() < 0.5 else -1.0
                    min_gap = float(r.uniform(0.34, max(0.35, corr_w - 2.0 * rad - 0.04)))
                    if corr_w - 2.0 * rad >= 0.34:
                        max_center_y = half_c - rad - 0.02
                        min_center_y = -half_c + min_gap + rad
                        if max_center_y >= min_center_y:
                            cy = side * float(r.uniform(max(0.0, min_center_y), max_center_y))
                            if side > 0:
                                cy = max(cy, -half_c + 0.34 + rad)
                            else:
                                cy = min(cy, half_c - 0.34 - rad)
                            cyls[rock_idx] = [rx_pos, cy, rad, 0.0, 0.0]
                            cyl_act[rock_idx] = True
                            rock_idx += 1
            else:
                # L-shaped right-angle turn corridor
                bounds = np.array([-0.8, 2.8, -half_c, 2.8], dtype=np.float32)
                rx, ry = 0.0, 0.0
                rth = float(r.uniform(-0.20, 0.20))
                corner_x = float(r.uniform(1.25, 1.75))
                wp1 = (corner_x, 0.0)
                boxes[0] = [0.5 * (-0.8 + (corner_x - half_c)), 0.5 * (half_c + 2.8), 0.5 * ((corner_x - half_c) - (-0.8)), 0.5 * (2.8 - half_c)]
                box_act[0] = True
                far_x = corner_x + half_c
                boxes[1] = [0.5 * (far_x + 2.8), 0.5 * (-half_c + 2.8), 0.5 * (2.8 - far_x), 0.5 * (2.8 + half_c)]
                box_act[1] = True
                gx = corner_x + float(r.uniform(-half_c * 0.25, half_c * 0.25))
                gy = float(r.uniform(1.5, 2.25))
                if r.rand() < 0.75:
                    rad = float(r.uniform(0.14, 0.18))
                    cyls[0] = [float(r.uniform(0.50, corner_x - 0.3)), float(r.uniform(0.05, half_c - rad)) * (1 if r.rand() < 0.5 else -1), rad, 0.0, 0.0]
                    if corr_w - 2.0 * rad >= 0.34:
                        cyl_act[0] = True

        elif regime == 2:
            # Regime 2: Generalized 3-way high-speed dynamic curriculum (continuous DR, zero world hardcoding)
            #   - 20% (or 100% if sinusoidal disabled): linear moving obstacles (v=0.10..0.20 m/s) in maze partitions
            #   - 40%: narrow gate (0.85..1.50 m) orthogonal reciprocating crossing (+-0.35 m, r=0.15..0.23, vy=0.20..0.45, w=0.7..1.2)
            #   - 40%: baffle/corridor axial head-on reciprocating (r=0.15..0.23, amp_x=0.70..1.30, vx=0.25..0.45)
            u_dyn = float(r.rand()) if self.enable_sinusoidal_dynamic else 0.0
            if u_dyn < 0.20:
                half_w = float(r.uniform(1.35, 1.90))
                bounds = np.array([-1.0, 3.4, -half_w, half_w], dtype=np.float32)
                gx = float(r.uniform(2.05, 2.70))
                gy = float(r.uniform(-half_w * 0.45, half_w * 0.45))

                w1x = float(r.uniform(0.68, 1.00))
                w1_side = 1.0 if r.rand() < 0.5 else -1.0
                boxes[0] = [w1x, w1_side * (half_w * 0.66), 0.06, half_w * 0.34]
                box_act[0] = True
                w2x = float(r.uniform(1.38, 1.72))
                boxes[1] = [w2x, -w1_side * (half_w * 0.66), 0.06, half_w * 0.34]
                box_act[1] = True
                wp1 = (w1x, -w1_side * (half_w * 0.55))
                wp2 = (w2x, w1_side * (half_w * 0.55))

                n_dyn = int(r.randint(2, 4))
                for k in range(n_dyn):
                    rad = float(r.uniform(0.14, 0.18))
                    cx, cy = 1.2, (0.62 * half_w) if (k % 2 == 0) else (-0.62 * half_w)
                    for _ in range(20):
                        cx_c = float(r.uniform(0.75, gx - 0.35))
                        cy_c = float(r.uniform(-half_w * 0.65, half_w * 0.65))
                        if math.hypot(cx_c - rx, cy_c - ry) >= rad + 0.68 and math.hypot(cx_c - gx, cy_c - gy) >= rad + 0.42:
                            cx, cy = cx_c, cy_c
                            break
                    if abs(cy) < 0.45:
                        cy = (0.62 * half_w) if (k % 2 == 0) else (-0.62 * half_w)
                    speed = float(r.uniform(0.10, 0.20))
                    vang = float(r.uniform(-math.pi, math.pi))
                    cyls[k] = [cx, cy, rad, speed * math.cos(vang), speed * math.sin(vang)]
                    cyl_act[k] = True
            elif u_dyn < 0.60:
                # 40%: Narrow gate orthogonal reciprocating crossing (calibrated for Ch3 observability & physical passability)
                half_w = float(r.uniform(1.35, 1.85))
                gate_w = float(r.uniform(1.05, 1.50))
                gate_x = float(r.uniform(1.10, 1.55))
                max_gy = max(0.05, min(0.35, half_w - 0.5 * gate_w - 0.18))
                gate_y = float(r.uniform(-max_gy, max_gy))
                gx = float(r.uniform(gate_x + 1.55, min(3.85, gate_x + 2.00)))
                gy = float(r.uniform(-half_w * 0.35, half_w * 0.35))
                bounds = np.array([-1.0, max(4.2, gx + 0.55), -half_w, half_w], dtype=np.float32)
                rx, ry = 0.0, float(r.uniform(-0.12, 0.12))
                rth = float(r.uniform(-0.25, 0.25))

                top_h = half_w - (gate_y + 0.5 * gate_w)
                if top_h > 0.04:
                    boxes[0] = [gate_x, gate_y + 0.5 * gate_w + 0.5 * top_h, 0.06, 0.5 * top_h]
                    box_act[0] = True
                bot_h = (gate_y - 0.5 * gate_w) - (-half_w)
                if bot_h > 0.04:
                    boxes[1] = [gate_x, -half_w + 0.5 * bot_h, 0.06, 0.5 * bot_h]
                    box_act[1] = True
                wp1 = (gate_x, gate_y)
                wp2 = (gx, gy)
                wp0_seg_half_y = max(0.0, 0.5 * gate_w - 0.18)

                base_phi = float(r.uniform(-math.pi, math.pi))
                for k in range(2):
                    rad = float(r.uniform(0.15, 0.22))
                    if k == 0:
                        min_off = max(0.24, rad + 0.08)
                        off_x = float(r.uniform(min_off, 0.36))
                        x0 = gate_x - off_x
                    else:
                        # Place 2nd crossing obstacle downstream with staging pocket so it enters +-0.90m Ch3 cleanly after gate
                        off_x = float(r.uniform(0.80, 1.05))
                        x0 = gate_x + off_x
                    y0 = gate_y
                    v_y = float(r.uniform(0.24, 0.42))
                    omega = float(r.uniform(0.65, 1.05))
                    max_amp_y = max(0.34, half_w - abs(y0) - rad - 0.08)
                    amp_y = min(max(v_y / omega, rad + 0.15), max_amp_y)
                    phi = base_phi + k * math.pi + float(r.uniform(-0.35, 0.35))
                    cx0 = x0
                    cy0 = y0 + amp_y * math.sin(phi)
                    if math.hypot(cx0 - rx, cy0 - ry) < rad + 0.55:
                        phi += 0.5 * math.pi
                        cy0 = y0 + amp_y * math.sin(phi)
                    cyls[k] = [cx0, cy0, rad, 0.0, amp_y * omega * math.cos(phi)]
                    cyl_act[k] = True
                    cyl_sin_act[k] = True
                    cyl_sin_params[k] = [x0, y0, 0.0, amp_y, omega, phi]
            else:
                # 40%: Baffle/corridor axial head-on reciprocating encounter
                half_w = float(r.uniform(1.38, 1.82))
                baffle_d = float(r.uniform(0.24, 0.40))
                inner_hw = half_w - baffle_d
                gx = float(r.uniform(2.55, 3.35))
                gy = float(r.uniform(-inner_hw * 0.45, inner_hw * 0.45))
                bounds = np.array([-1.0, 4.4, -half_w, half_w], dtype=np.float32)
                rx, ry = 0.0, float(r.uniform(-0.12, 0.12))
                rth = float(r.uniform(-0.22, 0.22))

                w1x = float(r.uniform(0.80, 1.30))
                w1_side = 1.0 if r.rand() < 0.5 else -1.0
                boxes[0] = [w1x, w1_side * (half_w - 0.5 * baffle_d), 0.07, 0.5 * baffle_d]
                box_act[0] = True
                w2x = float(r.uniform(1.75, 2.35))
                boxes[1] = [w2x, -w1_side * (half_w - 0.5 * baffle_d), 0.07, 0.5 * baffle_d]
                box_act[1] = True
                wp1 = (w1x, -w1_side * (inner_hw * 0.35))
                wp2 = (w2x, w1_side * (inner_hw * 0.35))

                rads = [float(r.uniform(0.15, 0.23)), float(r.uniform(0.15, 0.23))]
                y0_0 = float(r.uniform(-0.20, 0.20))
                side_1 = -1.0 if y0_0 >= 0.0 else 1.0
                y1_near = y0_0 + side_1 * (rads[0] + rads[1] + 0.12)
                y1_far = side_1 * (inner_hw - rads[1] - 0.08)
                if abs(y1_far) <= abs(y1_near):
                    y0_1 = y1_far
                else:
                    y0_1 = float(r.uniform(min(y1_far, y1_near), max(y1_far, y1_near)))
                y0_list = [y0_0, y0_1]

                for k in range(2):
                    rad = rads[k]
                    y0 = y0_list[k]
                    amp_x = float(r.uniform(0.70, 1.30))
                    v_x = float(r.uniform(0.25, 0.45))
                    omega = v_x / amp_x
                    x_lo = max(1.15, float(bounds[0]) + rad + 0.15 + amp_x)
                    x_hi = min(2.45, float(bounds[1]) - rad - 0.15 - amp_x)
                    x0 = float(r.uniform(x_lo, max(x_lo + 0.05, x_hi)))
                    if k == 0:
                        phi = float(r.uniform(0.35 * math.pi, 0.90 * math.pi))
                    else:
                        phi = (
                            float(r.uniform(0.25 * math.pi, 0.95 * math.pi))
                            if r.rand() < 0.75
                            else float(r.uniform(-0.15 * math.pi, 0.25 * math.pi))
                        )
                    cx0 = x0 + amp_x * math.sin(phi)
                    if math.hypot(cx0 - rx, y0 - ry) < rad + 0.65:
                        phi = 0.55 * math.pi
                        cx0 = x0 + amp_x * math.sin(phi)
                    vx0 = amp_x * omega * math.cos(phi)
                    cyls[k] = [cx0, y0, rad, vx0, 0.0]
                    cyl_act[k] = True
                    cyl_sin_act[k] = True
                    cyl_sin_params[k] = [x0, y0, amp_x, 0.0, omega, phi]

        elif regime == 3:
            # Regime 3: Low-profile invisible plinth + heavy asymmetric slip
            half_w = float(r.uniform(1.1, 1.7))
            bounds = np.array([-1.0, 3.4, -half_w, half_w], dtype=np.float32)
            gx = float(r.uniform(2.0, 2.9))
            gy = float(r.uniform(-0.35, 0.35))

            # 2..4 visible rocks
            for k in range(int(r.randint(2, 5))):
                rad = float(r.uniform(0.15, 0.20))
                cx = float(r.uniform(0.55, gx - 0.35))
                cy = float(r.uniform(-half_w + 0.3, half_w - 0.3))
                if math.hypot(cx - rx, cy - ry) > 0.45:
                    cyls[k] = [cx, cy, rad, 0.0, 0.0]
                    cyl_act[k] = True
            # 1 sub-LiDAR invisible low plinth directly between start and goal
            px = float(r.uniform(0.65, 1.35))
            py = float(r.uniform(-0.18, 0.18))
            pr = float(r.uniform(0.14, 0.20))
            plinth = np.array([px, py, pr, 1.0], dtype=np.float32)

        elif regime == 4:
            # Regime 4: U-pocket cul-de-sac (open mouth facing robot) + flank debris -> must reverse & loop around
            half_w = float(r.uniform(1.25, 1.70))
            bounds = np.array([-1.4, 3.6, -half_w, half_w], dtype=np.float32)
            rx, ry = 0.0, float(r.uniform(-0.10, 0.10))
            rth = float(r.uniform(-0.22, 0.22))
            gx = float(r.uniform(2.25, 3.05))
            gy = float(r.uniform(-half_w * 0.30, half_w * 0.30))

            ux_back = float(r.uniform(1.20, 1.55))
            uy = float(r.uniform(-0.22, 0.22))
            pocket_hw = float(r.uniform(0.36, 0.48))
            arm_len = float(r.uniform(0.40, 0.56))
            wall_t = 0.06

            # Back wall & two side arms forming a 'ㄷ' trap open toward -x
            boxes[0] = [ux_back, uy, wall_t, pocket_hw + wall_t]
            box_act[0] = True
            boxes[1] = [ux_back - 0.5 * arm_len, uy + pocket_hw, 0.5 * arm_len, wall_t]
            box_act[1] = True
            boxes[2] = [ux_back - 0.5 * arm_len, uy - pocket_hw, 0.5 * arm_len, wall_t]
            box_act[2] = True

            # Optional 1..2 outer flank rocks keeping >= 0.42m bypass clearance on at least one side
            open_side = 1.0 if uy <= 0.0 else -1.0
            bypass_y = float(np.clip(uy + open_side * (pocket_hw + 0.46), -half_w + 0.26, half_w - 0.26))
            # Half the time lure robot straight toward goal inside 'ㄷ' trap without bypass hint;
            # half the time give flank bypass WP so both decoy escape and flank navigation are learned.
            if r.rand() < 0.50:
                wp1 = (gx, gy)
            else:
                wp1 = (ux_back + 0.15, bypass_y)

            placed_r = []
            for k in range(2):
                rad = float(r.uniform(0.13, 0.17))
                for _ in range(15):
                    cx = float(r.uniform(0.55, gx - 0.35))
                    cy = float(r.uniform(-half_w + rad + 0.18, half_w - rad - 0.18))
                    # Keep mouth entry and chosen bypass lane passable
                    if abs(cy - bypass_y) < rad + 0.34 and abs(cx - ux_back) < 0.70:
                        continue
                    if math.hypot(cx - rx, cy - ry) < rad + 0.36 or math.hypot(cx - gx, cy - gy) < rad + 0.34:
                        continue
                    if abs(cy - uy) < pocket_hw + rad + 0.16 and cx < ux_back + 0.20:
                        continue
                    if all(math.hypot(cx - px, cy - py) >= rad + pr + 0.25 for px, py, pr in placed_r):
                        placed_r.append((cx, cy, rad))
                        cyls[k] = [cx, cy, rad, 0.0, 0.0]
                        cyl_act[k] = True
                        break

        elif regime == 5:
            # Regime 5: Double S-chicane doorways + asymmetric debris pinch (house-matched)
            corr_w = float(r.uniform(1.05, 1.55))
            half_c = 0.5 * corr_w
            bounds = np.array([-0.8, 3.8, -half_c, half_c], dtype=np.float32)
            rx, ry = 0.0, float(r.uniform(-half_c * 0.25, half_c * 0.25))
            rth = float(r.uniform(-0.20, 0.20))

            d1_x = float(r.uniform(0.85, 1.20))
            d2_x = float(r.uniform(d1_x + 0.85, min(2.45, d1_x + 1.25)))
            gx = float(r.uniform(d2_x + 0.65, 3.25))
            gy = float(r.uniform(-half_c * 0.35, half_c * 0.35))

            side1 = 1.0 if r.rand() < 0.5 else -1.0
            gap1_w = float(r.uniform(0.38, min(0.52, corr_w * 0.50)))
            gap1_y = side1 * float(r.uniform(0.12, max(0.14, half_c - 0.5 * gap1_w - 0.05)))

            gap2_w = float(r.uniform(0.38, min(0.52, corr_w * 0.50)))
            gap2_y = -side1 * float(r.uniform(0.12, max(0.14, half_c - 0.5 * gap2_w - 0.05)))

            wp1 = (d1_x, gap1_y)
            wp2 = (d2_x, gap2_y)

            b_idx = 0
            for dx, g_y, g_w in ((d1_x, gap1_y, gap1_w), (d2_x, gap2_y, gap2_w)):
                top_h = half_c - (g_y + 0.5 * g_w)
                if top_h > 0.03 and b_idx < self.MAX_BOXES:
                    boxes[b_idx] = [dx, g_y + 0.5 * g_w + 0.5 * top_h, 0.07, 0.5 * top_h]
                    box_act[b_idx] = True
                    b_idx += 1
                bot_h = (g_y - 0.5 * g_w) - (-half_c)
                if bot_h > 0.03 and b_idx < self.MAX_BOXES:
                    boxes[b_idx] = [dx, -half_c + 0.5 * bot_h, 0.07, 0.5 * bot_h]
                    box_act[b_idx] = True
                    b_idx += 1

            # Place 1..2 pinch rocks between zones while preserving >= 0.32m corridor clearance
            rock_xs = (0.5 * (rx + d1_x), 0.5 * (d1_x + d2_x))
            for k, rx_p in enumerate(rock_xs):
                rad = float(r.uniform(0.13, 0.16))
                if corr_w - 2.0 * rad < 0.34:
                    continue
                r_side = 1.0 if r.rand() < 0.5 else -1.0
                if r_side > 0:
                    cy = float(r.uniform(max(-half_c + 0.34 + rad, -0.05), half_c - rad - 0.02))
                else:
                    cy = float(r.uniform(-half_c + rad + 0.02, min(half_c - 0.34 - rad, 0.05)))
                cyls[k] = [rx_p, cy, rad, 0.0, 0.0]
                cyl_act[k] = True

        elif regime == 6:
            # Regime 6: Low-obstacle field (sub-LiDAR low obstacles invisible in Ch 0, visible in Ch 1)
            half_w = float(r.uniform(1.25, 1.80))
            bounds = np.array([-1.0, 3.8, -half_w, half_w], dtype=np.float32)
            rx, ry = 0.0, float(r.uniform(-0.12, 0.12))
            rth = float(r.uniform(-0.22, 0.22))
            gx = float(r.uniform(2.20, 3.10))
            gy = float(r.uniform(-half_w * 0.40, half_w * 0.40))

            # Primary low obstacles along start->goal line forcing proactive Ch 1 slalom avoidance
            lx0 = float(r.uniform(0.65, 1.00))
            ly0 = float(r.uniform(-0.16, 0.16))
            lr0 = float(r.uniform(0.14, 0.19))
            low_obs[0] = [lx0, ly0, lr0]
            low_obs_act[0] = True

            side0 = -1.0 if ly0 > 0.0 else 1.0
            wp1_y = float(np.clip(ly0 + side0 * (lr0 + 0.36), -half_w + 0.28, half_w - 0.28))
            wp1 = (lx0, wp1_y)

            lx1 = float(r.uniform(1.32, 1.75))
            ly1 = float(np.clip(-side0 * float(r.uniform(0.05, 0.24)), -half_w + 0.38, half_w - 0.38))
            lr1 = float(r.uniform(0.14, 0.19))
            low_obs[1] = [lx1, ly1, lr1]
            low_obs_act[1] = True

            wp2_y = float(np.clip(ly1 - side0 * (lr1 + 0.36), -half_w + 0.28, half_w - 0.28))
            wp2 = (lx1, wp2_y)

            placed_low = [(lx0, ly0, lr0), (lx1, ly1, lr1)]
            n_low = int(r.randint(4, min(7, self.MAX_LOW_OBS + 1)))
            for k in range(2, n_low):
                for _ in range(25):
                    rad = float(r.uniform(0.13, 0.18))
                    cx = float(r.uniform(0.50, gx - 0.38))
                    cy = float(r.uniform(-half_w + rad + 0.16, half_w - rad - 0.16))
                    if math.hypot(cx - rx, cy - ry) < rad + 0.36 or math.hypot(cx - gx, cy - gy) < rad + 0.36:
                        continue
                    if math.hypot(cx - wp1[0], cy - wp1[1]) < rad + 0.28 or math.hypot(cx - wp2[0], cy - wp2[1]) < rad + 0.28:
                        continue
                    if all(math.hypot(cx - px, cy - py) >= rad + pr + 0.36 for px, py, pr in placed_low):
                        placed_low.append((cx, cy, rad))
                        low_obs[k] = [cx, cy, rad]
                        low_obs_act[k] = True
                        break

            # Place 1..2 normal LiDAR-visible flank pillars so Ch 0 and Ch 1 coexist
            for k in range(2):
                for _ in range(20):
                    rad = float(r.uniform(0.14, 0.18))
                    cx = float(r.uniform(0.55, gx - 0.38))
                    flank_side = 1.0 if (k % 2 == 0) else -1.0
                    cy = flank_side * float(r.uniform(half_w * 0.55, half_w - rad - 0.12))
                    if math.hypot(cx - wp1[0], cy - wp1[1]) < rad + 0.32 or math.hypot(cx - wp2[0], cy - wp2[1]) < rad + 0.32:
                        continue
                    if all(math.hypot(cx - px, cy - py) >= rad + pr + 0.32 for px, py, pr in placed_low):
                        cyls[k] = [cx, cy, rad, 0.0, 0.0]
                        cyl_act[k] = True
                        break

        else:
            # Regime 7: Gas + victim corridor (gas leak & survivor zones visible in Ch 2 soft safety field)
            corr_w = float(r.uniform(1.50, 2.10))
            half_c = 0.5 * corr_w
            bounds = np.array([-0.9, 3.8, -half_c, half_c], dtype=np.float32)
            rx, ry = 0.0, float(r.uniform(-half_c * 0.22, half_c * 0.22))
            rth = float(r.uniform(-0.20, 0.20))
            gx = float(r.uniform(2.35, 3.20))
            gy = float(r.uniform(-half_c * 0.32, half_c * 0.32))

            # Zone 0: Gas leak hazard zone (kind=0.0)
            hx0 = float(r.uniform(0.72, 1.12))
            side0 = 1.0 if r.rand() < 0.5 else -1.0
            hr0 = float(r.uniform(0.24, 0.32))
            hy0 = side0 * float(r.uniform(0.08, max(0.10, half_c - hr0 - 0.08)))
            hazards[0] = [hx0, hy0, hr0, 0.0]
            hazard_act[0] = True

            # Zone 1: Survivor protection zone (kind=1.0) on opposite side
            hx1 = float(r.uniform(1.48, 1.92))
            side1 = -side0
            hr1 = float(r.uniform(0.24, 0.31))
            hy1 = side1 * float(r.uniform(0.08, max(0.10, half_c - hr1 - 0.08)))
            hazards[1] = [hx1, hy1, hr1, 1.0]
            hazard_act[1] = True

            wp1_y = float(np.clip(hy0 - side0 * (hr0 + 0.34), -half_c + 0.22, half_c - 0.22))
            wp2_y = float(np.clip(hy1 - side1 * (hr1 + 0.34), -half_c + 0.22, half_c - 0.22))
            wp1 = (hx0, wp1_y)
            wp2 = (hx1, wp2_y)

            # Optional Zone 2 further down corridor
            if gx - hx1 > 0.85 and r.rand() < 0.70:
                hx2 = float(r.uniform(hx1 + 0.50, gx - 0.38))
                side2 = side0
                hr2 = float(r.uniform(0.20, 0.26))
                hy2 = side2 * float(r.uniform(0.14, max(0.16, half_c - hr2 - 0.06)))
                hazards[2] = [hx2, hy2, hr2, float(r.randint(0, 2))]
                hazard_act[2] = True

            # Place 1..2 structural wall/pillar obstacles on the hazard-side wall so Ch 0 has corridor cues
            boxes[0] = [hx0, side0 * (half_c - 0.07), 0.10, 0.07]
            box_act[0] = True
            boxes[1] = [hx1, side1 * (half_c - 0.07), 0.10, 0.07]
            box_act[1] = True

        if wp1 is None:
            wp1 = (gx, gy)
        if wp2 is None:
            wp2 = (gx, gy)

        # Ensure physical corridor passability for plinth and S-chicane regimes
        half_arena_w = float(bounds[3])
        if regime == 3:
            px_v, py_v, pr_v, _ = plinth
            for k in range(self.MAX_CYLS):
                if cyl_act[k]:
                    cx_v, cy_v, rad_v = float(cyls[k, 0]), float(cyls[k, 1]), float(cyls[k, 2])
                    if math.hypot(cx_v - px_v, cy_v - py_v) < pr_v + rad_v + 0.60 or abs(cy_v) < 0.46:
                        cyls[k, 1] = (0.65 * half_arena_w) if (k % 2 == 0) else (-0.65 * half_arena_w)
        elif regime == 5:
            d1_x_v, gap1_y_v = wp1
            d2_x_v, gap2_y_v = wp2
            gap_w_v = min(0.58, half_arena_w * 1.10)
            box_act[:] = False
            b_idx = 0
            for dx_v, gy_v in ((d1_x_v, gap1_y_v * 0.50), (d2_x_v, gap2_y_v * 0.50)):
                top_h = half_arena_w - (gy_v + 0.5 * gap_w_v)
                if top_h > 0.04 and b_idx < self.MAX_BOXES:
                    boxes[b_idx] = [dx_v, gy_v + 0.5 * gap_w_v + 0.5 * top_h, 0.06, 0.5 * top_h]
                    box_act[b_idx] = True
                    b_idx += 1
                bot_h = (gy_v - 0.5 * gap_w_v) - (-half_arena_w)
                if bot_h > 0.04 and b_idx < self.MAX_BOXES:
                    boxes[b_idx] = [dx_v, -half_arena_w + 0.5 * bot_h, 0.06, 0.5 * bot_h]
                    box_act[b_idx] = True
                    b_idx += 1
            for k in range(self.MAX_CYLS):
                if cyl_act[k] and abs(float(cyls[k, 1])) < 0.38:
                    cyl_act[k] = False

        return {
            "regime": regime,
            "bounds": bounds,
            "boxes": boxes,
            "box_act": box_act,
            "cyls": cyls,
            "cyl_act": cyl_act,
            "cyl_sin_act": cyl_sin_act,
            "cyl_sin_params": cyl_sin_params,
            "plinth": plinth,
            "low_obs": low_obs,
            "low_obs_act": low_obs_act,
            "hazards": hazards,
            "hazard_act": hazard_act,
            "pose": (rx, ry, rth),
            "goal": (gx, gy),
            "nav_wps": np.array([wp1, wp2], dtype=np.float32),
            "wp0_seg_half_y": float(wp0_seg_half_y),
        }

    def reset_envs(
        self,
        env_ids: Optional[torch.Tensor] = None,
        eval_seeds: Optional[Sequence[int]] = None,
        plr_prob: float = 0.25,
        forced_regime: Optional[int] = None,
        forced_regimes: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reset all (or masked subset) of parallel environments cleanly."""
        if env_ids is None:
            ids_np = np.arange(self.B, dtype=np.int64)
            if eval_seeds is not None and self.mode == "eval":
                # Guarantee bit-exact reproducibility across repeated eval resets
                self.rng = np.random.RandomState(self.base_seed)
                self.lidar_torch_gen.manual_seed((self.base_seed ^ 0x1D4A7C3B) & 0x7FFFFFFF)
                sem_seed = (self.base_seed ^ 0x5E4A471C) & 0x7FFFFFFF
                self.sem_rng = np.random.RandomState(sem_seed)
                self.sem_torch_gen.manual_seed(sem_seed)
                p4_seed = (self.base_seed ^ 0x4A5E4D2B) & 0x7FFFFFFF
                self.p4_rng = np.random.RandomState(p4_seed)
                self.p4_torch_gen.manual_seed(p4_seed)
        else:
            ids_np = env_ids.detach().cpu().numpy()

        cfg = self.track_cfg
        n_res = len(ids_np)
        b_regime = np.zeros(n_res, dtype=np.int64)
        b_bounds = np.zeros((n_res, 4), dtype=np.float32)
        b_boxes = np.zeros((n_res, self.MAX_BOXES, 4), dtype=np.float32)
        b_box_act = np.zeros((n_res, self.MAX_BOXES), dtype=bool)
        b_cyls = np.zeros((n_res, self.MAX_CYLS, 5), dtype=np.float32)
        b_cyl_act = np.zeros((n_res, self.MAX_CYLS), dtype=bool)
        b_cyl_sin_act = np.zeros((n_res, self.MAX_CYLS), dtype=bool)
        b_cyl_sin_params = np.zeros((n_res, self.MAX_CYLS, 6), dtype=np.float32)
        b_plinth = np.zeros((n_res, 4), dtype=np.float32)
        b_low_obs = np.zeros((n_res, self.MAX_LOW_OBS, 3), dtype=np.float32)
        b_low_act = np.zeros((n_res, self.MAX_LOW_OBS), dtype=bool)
        b_hazards = np.zeros((n_res, self.MAX_HAZARDS, 4), dtype=np.float32)
        b_haz_act = np.zeros((n_res, self.MAX_HAZARDS), dtype=bool)
        b_rx = np.zeros(n_res, dtype=np.float32)
        b_ry = np.zeros(n_res, dtype=np.float32)
        b_rth = np.zeros(n_res, dtype=np.float32)
        b_gx = np.zeros(n_res, dtype=np.float32)
        b_gy = np.zeros(n_res, dtype=np.float32)
        b_nwps = np.zeros((n_res, 2, 2), dtype=np.float32)
        b_wp0_seg_hy = np.zeros(n_res, dtype=np.float32)
        b_d0 = np.zeros(n_res, dtype=np.float32)
        b_lnoise = np.zeros(n_res, dtype=np.float32)
        b_ldrop = np.zeros(n_res, dtype=np.float32)
        b_oscale = np.ones(n_res, dtype=np.float32)
        b_obias = np.zeros(n_res, dtype=np.float32)
        b_sv = np.ones(n_res, dtype=np.float32)
        b_sw = np.ones(n_res, dtype=np.float32)
        b_tbias = np.zeros(n_res, dtype=np.float32)
        b_lag = np.zeros(n_res, dtype=np.int64)

        for k_idx, i in enumerate(ids_np):
            explicit_reg: Optional[int] = None
            if forced_regimes is not None:
                explicit_reg = int(forced_regimes[k_idx % len(forced_regimes)])
            elif forced_regime is not None:
                explicit_reg = int(forced_regime)

            if eval_seeds is not None:
                seed = int(eval_seeds[k_idx % len(eval_seeds)])
                if explicit_reg is not None:
                    f_reg = explicit_reg
                elif self.mode == "eval":
                    if self.include_semantic_regimes:
                        f_reg = seed % NUM_REGIMES
                    else:
                        f_reg = seed % NUM_BASE_REGIMES
                else:
                    f_reg = None
                layout = self._build_single_layout(seed, forced_regime=f_reg)
            elif self.mode == "train" and explicit_reg is None and self.hard_seeds and self.rng.rand() < plr_prob:
                pick = self.hard_seeds[int(self.rng.randint(0, len(self.hard_seeds)))]
                seed, f_reg = int(pick[0]), int(pick[1])
                layout = self._build_single_layout(seed, forced_regime=f_reg)
            else:
                seed = int(self.rng.randint(100_000, 999_999_999))
                if self.mode == "train" and explicit_reg is None and self.num_active_regimes > 2:
                    # Upweight regime 2 (dynamic_crossing) from 12.5% -> 28% during training
                    if self.rng.rand() < 0.28:
                        rand_reg = 2
                    else:
                        other_idx = int(self.rng.randint(0, self.num_active_regimes - 1))
                        rand_reg = other_idx if other_idx < 2 else other_idx + 1
                else:
                    rand_reg = int((i + self.rng.randint(0, self.num_active_regimes)) % self.num_active_regimes)
                f_reg = explicit_reg if explicit_reg is not None else rand_reg
                layout = self._build_single_layout(seed, forced_regime=f_reg)

            self.ep_seed[i] = seed
            b_regime[k_idx] = layout["regime"]
            b_bounds[k_idx] = layout["bounds"]
            b_boxes[k_idx] = layout["boxes"]
            b_box_act[k_idx] = layout["box_act"]
            b_cyls[k_idx] = layout["cyls"]
            b_cyl_act[k_idx] = layout["cyl_act"]
            b_cyl_sin_act[k_idx] = layout["cyl_sin_act"]
            b_cyl_sin_params[k_idx] = layout["cyl_sin_params"]
            b_plinth[k_idx] = layout["plinth"]
            b_low_obs[k_idx] = layout["low_obs"]
            b_low_act[k_idx] = layout["low_obs_act"]
            b_hazards[k_idx] = layout["hazards"]
            b_haz_act[k_idx] = layout["hazard_act"]

            rx, ry, rth = layout["pose"]
            gx, gy = layout["goal"]
            nwps = layout["nav_wps"]
            seg_hy = float(layout.get("wp0_seg_half_y", 0.0))
            b_rx[k_idx], b_ry[k_idx], b_rth[k_idx] = rx, ry, rth
            b_gx[k_idx], b_gy[k_idx] = gx, gy
            b_nwps[k_idx] = nwps
            b_wp0_seg_hy[k_idx] = seg_hy
            wp0_y_c = float(np.clip(ry, float(nwps[0, 1]) - seg_hy, float(nwps[0, 1]) + seg_hy))
            d_01 = math.hypot(float(nwps[0, 0]) - rx, wp0_y_c - ry)
            d_12 = math.hypot(float(nwps[1, 0]) - float(nwps[0, 0]), float(nwps[1, 1]) - float(nwps[0, 1]))
            d_2g = math.hypot(gx - float(nwps[1, 0]), gy - float(nwps[1, 1]))
            b_d0[k_idx] = d_01 + d_12 + d_2g

            b_lnoise[k_idx] = float(self.rng.uniform(*cfg.lidar_noise_std_range))
            b_ldrop[k_idx] = float(self.rng.uniform(*cfg.lidar_dropout_prob_range))
            b_oscale[k_idx] = float(self.rng.uniform(*cfg.odom_scale_range))
            b_obias[k_idx] = float(self.rng.uniform(*cfg.odom_yaw_bias_range))
            slip_lo, slip_hi = cfg.slip_gain_range
            if layout["regime"] == 3:
                slip_lo *= 0.72
            if self.dr_profile == "strict":
                b_sv[k_idx] = float(self.rng.uniform(slip_lo, slip_hi))
                b_sw[k_idx] = float(self.rng.uniform(slip_lo, slip_hi))
                b_tbias[k_idx] = float(self.rng.uniform(*cfg.turn_bias_range))
                b_lag[k_idx] = int(self.rng.randint(0, cfg.act_lag_max + 1))
            else:
                # Legacy Phase 2-B/3 DR profile (consumes matching RNG draws while keeping slip/lag clamped)
                b_sv[k_idx] = float(self.rng.uniform(max(0.92, slip_lo), slip_hi))
                b_sw[k_idx] = float(self.rng.uniform(max(0.92, slip_lo), slip_hi))
                b_tbias[k_idx] = float(self.rng.uniform(*cfg.turn_bias_range)) * 0.25
                _ = int(self.rng.randint(0, cfg.act_lag_max + 1))
                b_lag[k_idx] = 0

        idx_t = torch.from_numpy(ids_np).to(self.device)
        self.regime[idx_t] = torch.from_numpy(b_regime).to(self.device)
        self.bounds[idx_t] = torch.from_numpy(b_bounds).to(self.device)
        self.boxes[idx_t] = torch.from_numpy(b_boxes).to(self.device)
        self.box_active[idx_t] = torch.from_numpy(b_box_act).to(self.device)
        self.cyls[idx_t] = torch.from_numpy(b_cyls).to(self.device)
        self.cyl_active[idx_t] = torch.from_numpy(b_cyl_act).to(self.device)
        self.cyl_sin_active[idx_t] = torch.from_numpy(b_cyl_sin_act).to(self.device)
        self.cyl_sin_params[idx_t] = torch.from_numpy(b_cyl_sin_params).to(self.device)
        self.plinth[idx_t] = torch.from_numpy(b_plinth).to(self.device)

        # Initialize Ch3 delay FIFO buffer & Phase 4 Ch3/PLICP DR parameters via dedicated p4_rng
        init_is_dyn = (
            self.cyl_active[idx_t]
            & (
                self.cyl_sin_active[idx_t]
                | ((self.cyls[idx_t, :, 3].abs() + self.cyls[idx_t, :, 4].abs()) > 1e-4)
            )
        ).float().unsqueeze(-1)
        init_cyl_snap = torch.cat([self.cyls[idx_t], init_is_dyn], dim=-1)  # (n_res, MAX_CYLS, 6)
        self.cyl_hist_fifo[idx_t] = init_cyl_snap.unsqueeze(1).expand(-1, 5, -1, -1).clone()
        self.dyn_seen_ticks[idx_t] = 0
        if self.enable_ch3_dr:
            d_lo, d_hi = self.ch3_delay_ticks_range
            b_ch3_delay = self.p4_rng.randint(d_lo, d_hi + 1, size=n_res).astype(np.int64)
            b_ch3_vstd = self.p4_rng.uniform(*self.ch3_vel_noise_std_range, size=n_res).astype(np.float32)
            b_ch3_drop = self.p4_rng.uniform(*self.ch3_occlusion_drop_range, size=n_res).astype(np.float32)
            self.ch3_delay_ticks[idx_t] = torch.from_numpy(b_ch3_delay).to(self.device)
            self.ch3_vel_noise_std[idx_t] = torch.from_numpy(b_ch3_vstd).to(self.device)
            self.ch3_drop_prob[idx_t] = torch.from_numpy(b_ch3_drop).to(self.device)
        else:
            self.ch3_delay_ticks[idx_t] = 0
            self.ch3_vel_noise_std[idx_t] = 0.0
            self.ch3_drop_prob[idx_t] = 0.0

        if self.enable_plicp_vel_noise_dr:
            b_pv = self.p4_rng.uniform(*self.plicp_v_noise_std_range, size=n_res).astype(np.float32)
            b_pw = self.p4_rng.uniform(*self.plicp_w_noise_std_range, size=n_res).astype(np.float32)
            b_pph = self.p4_rng.randint(0, self.plicp_hold_ticks, size=n_res).astype(np.int64)
            self.plicp_v_std[idx_t] = torch.from_numpy(b_pv).to(self.device)
            self.plicp_w_std[idx_t] = torch.from_numpy(b_pw).to(self.device)
            self.plicp_phase[idx_t] = torch.from_numpy(b_pph).to(self.device)
        else:
            self.plicp_v_std[idx_t] = 0.0
            self.plicp_w_std[idx_t] = 0.0
            self.plicp_phase[idx_t] = 0
        self.plicp_held_vw[idx_t] = 0.0
        if self.subgoal_obs_prob > 0.0:
            b_sub_use = self.p4_rng.rand(n_res) < self.subgoal_obs_prob
            b_sub_la = self.p4_rng.uniform(*self.subgoal_lookahead_range, size=n_res).astype(np.float32)
            self.use_subgoal_obs[idx_t] = torch.from_numpy(b_sub_use).to(self.device)
            self.subgoal_lookahead_m[idx_t] = torch.from_numpy(b_sub_la).to(self.device)
        else:
            self.use_subgoal_obs[idx_t] = False
            self.subgoal_lookahead_m[idx_t] = 1.0
        self.low_obs[idx_t] = torch.from_numpy(b_low_obs).to(self.device)
        self.low_obs_active[idx_t] = torch.from_numpy(b_low_act).to(self.device)
        self.hazard_zones[idx_t] = torch.from_numpy(b_hazards).to(self.device)
        self.hazard_active[idx_t] = torch.from_numpy(b_haz_act).to(self.device)

        # Initialize semantic perception noise & episode-level modality dropout via dedicated sem_rng
        n_low_slots = self.MAX_LOW_OBS
        if self.enable_semantic_noise and self.semantic_pos_jitter_std > 0.0:
            low_b = np.clip(
                self.sem_rng.normal(0.0, self.semantic_pos_jitter_std * 0.75, size=(n_res, n_low_slots, 2)),
                -0.14,
                0.14,
            ).astype(np.float32)
            haz_b = np.clip(
                self.sem_rng.normal(0.0, self.semantic_pos_jitter_std * 0.75, size=(n_res, self.MAX_HAZARDS, 2)),
                -0.14,
                0.14,
            ).astype(np.float32)
            self.low_obs_bias[idx_t] = torch.from_numpy(low_b).to(self.device)
            self.hazard_bias[idx_t] = torch.from_numpy(haz_b).to(self.device)
        else:
            self.low_obs_bias[idx_t] = 0.0
            self.hazard_bias[idx_t] = 0.0

        if self.enable_semantic_noise and self.semantic_reg_delay_steps > 0:
            low_d = self.sem_rng.randint(1, self.semantic_reg_delay_steps + 1, size=(n_res, n_low_slots)).astype(np.int64)
            haz_d = self.sem_rng.randint(1, self.semantic_reg_delay_steps + 1, size=(n_res, self.MAX_HAZARDS)).astype(np.int64)
            self.low_obs_reg_delay[idx_t] = torch.from_numpy(low_d).to(self.device)
            self.hazard_reg_delay[idx_t] = torch.from_numpy(haz_d).to(self.device)
        else:
            self.low_obs_reg_delay[idx_t] = 0
            self.hazard_reg_delay[idx_t] = 0

        self.low_obs_seen_ticks[idx_t] = 0
        self.hazard_seen_ticks[idx_t] = 0

        if self.modality_dropout_prob >= 1.0:
            self.ep_modality_dropped[idx_t] = True
        elif self.modality_dropout_prob <= 0.0:
            self.ep_modality_dropped[idx_t] = False
        else:
            ep_drop_p = 0.5 * self.modality_dropout_prob
            ep_drop = (self.sem_rng.rand(n_res) < ep_drop_p)
            self.ep_modality_dropped[idx_t] = torch.from_numpy(ep_drop).to(self.device)

        rx_t = torch.from_numpy(b_rx).to(self.device)
        ry_t = torch.from_numpy(b_ry).to(self.device)
        rth_t = torch.from_numpy(b_rth).to(self.device)
        d0_t = torch.from_numpy(b_d0).to(self.device)
        self.rx[idx_t] = rx_t
        self.ry[idx_t] = ry_t
        self.rth[idx_t] = rth_t
        self.odom_x[idx_t] = rx_t
        self.odom_y[idx_t] = ry_t
        self.odom_th[idx_t] = rth_t
        self.goal_x[idx_t] = torch.from_numpy(b_gx).to(self.device)
        self.goal_y[idx_t] = torch.from_numpy(b_gy).to(self.device)
        self.nav_wp[idx_t] = torch.from_numpy(b_nwps).to(self.device)
        self.wp0_seg_half_y[idx_t] = torch.from_numpy(b_wp0_seg_hy).to(self.device)
        self.nav_stage[idx_t] = 0
        self.prev_dist[idx_t] = d0_t
        self.min_dist_seen[idx_t] = d0_t
        self.steps[idx_t] = 0
        self.stall_steps[idx_t] = 0
        self.bypass_bonus_given[idx_t] = False
        self.plinth_escape_bonus_given[idx_t] = False
        self.plinth_unpin_bonus_given[idx_t] = False
        self.hit_plinth[idx_t] = False
        self.hit_low_obs[idx_t] = False
        self.prev_hit_low[idx_t] = False
        self.low_clear_bonus_given[idx_t] = False
        self.entered_hazard[idx_t] = False
        self.prev_in_hazard[idx_t] = False
        self.hazard_clear_bonus_given[idx_t] = False
        g1 = np.where(
            ((b_regime == 2) & b_box_act[:, 0]) | ((b_regime == 4) & b_box_act[:, 0]),
            b_boxes[:, 0, 0] + 0.14,
            np.where((b_regime == 5) | (b_regime == 6) | (b_regime == 7), b_nwps[:, 0, 0] + 0.10, 999.0),
        ).astype(np.float32)
        g2 = np.where(
            (b_regime == 2) & b_box_act[:, 1],
            b_boxes[:, 1, 0] + 0.14,
            np.where((b_regime == 5) | (b_regime == 6) | (b_regime == 7), b_nwps[:, 1, 0] + 0.10, 999.0),
        ).astype(np.float32)
        self.gate1_x[idx_t] = torch.from_numpy(g1).to(self.device)
        self.gate2_x[idx_t] = torch.from_numpy(g2).to(self.device)
        self.gate1_done[idx_t] = False
        self.gate2_done[idx_t] = False

        self.lidar_noise_std[idx_t] = torch.from_numpy(b_lnoise).to(self.device)
        self.lidar_drop_prob[idx_t] = torch.from_numpy(b_ldrop).to(self.device)
        self.odom_scale[idx_t] = torch.from_numpy(b_oscale).to(self.device)
        self.odom_yaw_bias[idx_t] = torch.from_numpy(b_obias).to(self.device)
        self.slip_v[idx_t] = torch.from_numpy(b_sv).to(self.device)
        self.slip_w[idx_t] = torch.from_numpy(b_sw).to(self.device)
        self.turn_bias[idx_t] = torch.from_numpy(b_tbias).to(self.device)
        self.act_lag[idx_t] = torch.from_numpy(b_lag).to(self.device)

        self.act_fifo[idx_t] = 0.0
        self.last_cmd[idx_t] = 0.0
        self.actual_vw[idx_t] = 0.0
        self.obs_hist[idx_t] = 0.0
        self.local_patch[idx_t] = 0.0
        self._obs_ch3_vnoise[idx_t] = 0.0
        self._obs_ch3_keep[idx_t] = True
        self._obs_low_state[idx_t] = 0.0
        self._obs_haz_state[idx_t] = 0.0

        # Prime initial frame across history buffer without mutating non-reset envs during partial resets
        feat, patch = self._compute_obs_step(update_ids=(idx_t if env_ids is not None else None))
        self.obs_hist[idx_t] = feat[idx_t].unsqueeze(1).expand(-1, HISTORY_LEN, -1)
        self.local_patch[idx_t] = patch[idx_t]
        return self.obs_hist, self.local_patch

    # ------------------------------------------------------------------
    # Batched Analytical Raycasting & Signed Distance Check (GPU)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _cast_lidar(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (noisy_72_ranges (B,72), true_min_obstacle_clearance (B,))."""
        c = torch.cos(self.rth)[:, None]  # (B, 1)
        s = torch.sin(self.rth)[:, None]
        ca = c * self.ray_cos_local[None, :] - s * self.ray_sin_local[None, :]  # (B, 72)
        sa = s * self.ray_cos_local[None, :] + c * self.ray_sin_local[None, :]  # (B, 72)

        rx = self.rx[:, None]  # (B, 1)
        ry = self.ry[:, None]
        ranges = torch.full((self.B, NUM_RAYS), MAX_RANGE_M, dtype=torch.float32, device=self.device)

        # 1. Outer arena axis-aligned walls (x_min, x_max, y_min, y_max)
        eps = 1e-6
        ca_safe = torch.where(ca.abs() > eps, ca, torch.where(ca >= 0, eps, -eps))
        sa_safe = torch.where(sa.abs() > eps, sa, torch.where(sa >= 0, eps, -eps))

        tx1 = (self.bounds[:, 0:1] - rx) / ca_safe
        tx2 = (self.bounds[:, 1:2] - rx) / ca_safe
        ty1 = (self.bounds[:, 2:3] - ry) / sa_safe
        ty2 = (self.bounds[:, 3:4] - ry) / sa_safe
        for t_cand in (tx1, tx2, ty1, ty2):
            valid = t_cand > 0.03
            ranges = torch.where(valid, torch.minimum(ranges, t_cand), ranges)

        # Robot true clearance to arena walls
        wall_clear = torch.minimum(
            torch.minimum(self.rx - self.bounds[:, 0], self.bounds[:, 1] - self.rx),
            torch.minimum(self.ry - self.bounds[:, 2], self.bounds[:, 3] - self.ry),
        )

        # 2. Boxes: slab intersection (B, K, R) -> vectorized!
        # boxes: (B, K, 4) = (bcx, bcy, bhx, bhy)
        bcx = self.boxes[:, :, 0:1] - rx[:, :, None]  # (B, K, 1)
        bcy = self.boxes[:, :, 1:2] - ry[:, :, None]
        bhx = self.boxes[:, :, 2:3]
        bhy = self.boxes[:, :, 3:4]
        b_act = self.box_active[:, :, None]  # (B, K, 1)

        inv_ca = 1.0 / ca_safe[:, None, :]  # (B, 1, 72)
        inv_sa = 1.0 / sa_safe[:, None, :]

        # Ray origin relative to box center = -b_rel
        ox_box = -bcx
        oy_box = -bcy
        tx_a = (-bhx - ox_box) * inv_ca
        tx_b = (bhx - ox_box) * inv_ca
        tmin_x = torch.minimum(tx_a, tx_b)
        tmax_x = torch.maximum(tx_a, tx_b)

        ty_a = (-bhy - oy_box) * inv_sa
        ty_b = (bhy - oy_box) * inv_sa
        tmin_y = torch.minimum(ty_a, ty_b)
        tmax_y = torch.maximum(ty_a, ty_b)

        t_enter = torch.maximum(tmin_x, tmin_y)
        t_exit = torch.minimum(tmax_x, tmax_y)
        hit_box = b_act & (t_exit >= torch.clamp(t_enter, min=0.0)) & (t_exit > 0.03)
        t_box = torch.where(t_enter > 0.03, t_enter, t_exit)
        t_box = torch.where(hit_box, t_box, MAX_RANGE_M)
        ranges = torch.minimum(ranges, t_box.min(dim=1).values)

        # Box signed surface distance from robot
        dx_box = torch.clamp(torch.abs(self.rx[:, None] - self.boxes[:, :, 0]) - self.boxes[:, :, 2], min=0.0)
        dy_box = torch.clamp(torch.abs(self.ry[:, None] - self.boxes[:, :, 1]) - self.boxes[:, :, 3], min=0.0)
        dist_boxes = torch.hypot(dx_box, dy_box)
        dist_boxes = torch.where(self.box_active, dist_boxes, 10.0).min(dim=1).values

        # 3. Cylinders (static + dynamic obstacles): (B, C, R)
        dx_c = self.cyls[:, :, 0:1] - rx[:, :, None]  # (B, C, 1)
        dy_c = self.cyls[:, :, 1:2] - ry[:, :, None]
        r_c = self.cyls[:, :, 2:3]
        c_act = self.cyl_active[:, :, None]

        b_proj = dx_c * ca[:, None, :] + dy_c * sa[:, None, :]  # (B, C, 72)
        c_term = dx_c * dx_c + dy_c * dy_c - r_c * r_c
        disc = b_proj * b_proj - c_term
        hit_c = c_act & (disc >= 0.0) & (b_proj > 0.0)
        t_cyl = b_proj - torch.sqrt(torch.clamp(disc, min=0.0))
        hit_c = hit_c & (t_cyl > 0.03)
        t_cyl = torch.where(hit_c, t_cyl, MAX_RANGE_M)
        ranges = torch.minimum(ranges, t_cyl.min(dim=1).values)

        dist_cyls = torch.hypot(self.rx[:, None] - self.cyls[:, :, 0], self.ry[:, None] - self.cyls[:, :, 1]) - self.cyls[:, :, 2]
        dist_cyls = torch.where(self.cyl_active, dist_cyls, 10.0).min(dim=1).values

        min_clear = torch.minimum(wall_clear, torch.minimum(dist_boxes, dist_cyls))

        # 4. Inject Track A / Track B sensor noise, dropout, and glare via dedicated deterministic generator
        noise = (
            torch.randn(ranges.shape, dtype=ranges.dtype, device=ranges.device, generator=self.lidar_torch_gen)
            * self.lidar_noise_std[:, None]
        )
        noisy = torch.clamp(ranges + noise, 0.12, MAX_RANGE_M)

        drop_mask = (
            torch.rand(ranges.shape, dtype=ranges.dtype, device=ranges.device, generator=self.lidar_torch_gen)
            < self.lidar_drop_prob[:, None]
        )
        noisy = torch.where(drop_mask, MAX_RANGE_M, noisy)

        if self.track_cfg.lidar_glare_prob > 0.0:
            glare_mask = (
                torch.rand(ranges.shape, dtype=ranges.dtype, device=ranges.device, generator=self.lidar_torch_gen)
                < self.track_cfg.lidar_glare_prob
            )
            glare_val = (
                torch.rand(ranges.shape, dtype=ranges.dtype, device=ranges.device, generator=self.lidar_torch_gen)
                * 0.45
                + 0.15
            )
            noisy = torch.where(glare_mask, glare_val, noisy)

        return noisy, min_clear

    def set_ch2_clean_frac(self, frac: float) -> None:
        """Mix a fraction of parallel training envs with clean (zero dropout/miss) Ch2 visibility for CBF alignment."""
        f = float(np.clip(frac, 0.0, 1.0))
        self.ch2_clean_env_frac = f
        n_clean = int(round(f * self.B))
        self._clean_ch2_env_mask = torch.arange(self.B, device=self.device) < n_clean

    @torch.no_grad()
    def _compute_semantic_channels(
        self,
        upd_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute semantic ego patch channels (B, 1, 12, 12) each:
          - Ch 1: low-lying sub-LiDAR obstacles (`low_obs` only; Regime 3 plinth is intentionally
            excluded from Ch 1 as a Base-6 tactile blind-unpin regime).
          - Ch 2: gas hazard & survivor protection zones (soft safety field).
        Includes Modality Dropout (~25-30%) and realistic semantic perception noise
        (position jitter, registration delay, missed detections, non-blocking false positives).
        When `upd_mask` is provided (partial reset), persistent state tensors are only updated on `upd_mask` rows.
        """
        c = torch.cos(self.rth)[:, None]  # (B, 1)
        s = torch.sin(self.rth)[:, None]

        all_low = self.low_obs  # (B, MAX_LOW_OBS, 3)
        all_low_act = self.low_obs_active  # (B, MAX_LOW_OBS)
        m1 = self.MAX_LOW_OBS

        # True distances for registration delay check (markers confirm inside 2.5m range)
        true_low_dist = torch.hypot(all_low[:, :, 0] - self.rx[:, None], all_low[:, :, 1] - self.ry[:, None])
        in_low_range = all_low_act & (true_low_dist <= 2.5)
        new_low_seen = torch.where(in_low_range, self.low_obs_seen_ticks + 1, self.low_obs_seen_ticks)
        if upd_mask is None:
            self.low_obs_seen_ticks = new_low_seen
        else:
            self.low_obs_seen_ticks = torch.where(upd_mask[:, None], new_low_seen, self.low_obs_seen_ticks)

        true_haz_dist = torch.hypot(
            self.hazard_zones[:, :, 0] - self.rx[:, None],
            self.hazard_zones[:, :, 1] - self.ry[:, None],
        )
        in_haz_range = self.hazard_active & (true_haz_dist <= 2.8)
        new_haz_seen = torch.where(in_haz_range, self.hazard_seen_ticks + 1, self.hazard_seen_ticks)
        if upd_mask is None:
            self.hazard_seen_ticks = new_haz_seen
        else:
            self.hazard_seen_ticks = torch.where(upd_mask[:, None], new_haz_seen, self.hazard_seen_ticks)

        low_wx = all_low[:, :, 0]
        low_wy = all_low[:, :, 1]
        haz_wx = self.hazard_zones[:, :, 0]
        haz_wy = self.hazard_zones[:, :, 1]

        low_vis = all_low_act & (self.low_obs_seen_ticks > self.low_obs_reg_delay)
        haz_vis = self.hazard_active & (self.hazard_seen_ticks > self.hazard_reg_delay)

        if self.enable_semantic_noise:
            if self.semantic_pos_jitter_std > 0.0:
                step_j_low = torch.clamp(
                    torch.randn((self.B, m1, 2), dtype=torch.float32, device=self.device, generator=self.sem_torch_gen)
                    * (self.semantic_pos_jitter_std * 0.55),
                    -0.10,
                    0.10,
                )
                step_j_haz = torch.clamp(
                    torch.randn(
                        (self.B, self.MAX_HAZARDS, 2),
                        dtype=torch.float32,
                        device=self.device,
                        generator=self.sem_torch_gen,
                    )
                    * (self.semantic_pos_jitter_std * 0.55),
                    -0.10,
                    0.10,
                )
                low_wx = low_wx + self.low_obs_bias[:, :, 0] + step_j_low[:, :, 0]
                low_wy = low_wy + self.low_obs_bias[:, :, 1] + step_j_low[:, :, 1]
                haz_wx = haz_wx + self.hazard_bias[:, :, 0] + step_j_haz[:, :, 0]
                haz_wy = haz_wy + self.hazard_bias[:, :, 1] + step_j_haz[:, :, 1]

            if self.semantic_miss_prob > 0.0:
                fast_rot_boost = torch.where(self.actual_vw[:, 1:2].abs() > 0.50, 0.12, 0.0)
                eff_miss_low = torch.clamp(self.semantic_miss_prob + fast_rot_boost, 0.0, 0.75)
                u_low = torch.rand((self.B, m1), dtype=torch.float32, device=self.device, generator=self.sem_torch_gen)
                u_haz = torch.rand(
                    (self.B, self.MAX_HAZARDS), dtype=torch.float32, device=self.device, generator=self.sem_torch_gen
                )
                low_vis = low_vis & (u_low >= eff_miss_low)
                if self.ch2_clean_env_frac > 0.0:
                    haz_vis = torch.where(
                        self._clean_ch2_env_mask[:, None],
                        haz_vis,
                        haz_vis & (u_haz >= eff_miss_low),
                    )
                else:
                    haz_vis = haz_vis & (u_haz >= eff_miss_low)

        # Project Ch 1 low obstacles into 12x12 ego frame
        dx_low = low_wx - self.rx[:, None]
        dy_low = low_wy - self.ry[:, None]
        ego_low_x = c * dx_low + s * dy_low
        ego_low_y = -s * dx_low + c * dy_low
        low_r = all_low[:, :, 2]

        dist_low_grid = torch.hypot(
            self.ego_grid_x - ego_low_x[:, :, None, None],
            self.ego_grid_y - ego_low_y[:, :, None, None],
        )  # (B, M1, 12, 12)
        core_low = low_r[:, :, None, None] + 0.04
        val_low = torch.clamp(1.0 - torch.clamp(dist_low_grid - core_low, min=0.0) / 0.16, 0.0, 1.0)
        val_low = torch.where(low_vis[:, :, None, None], val_low, 0.0)
        ch1 = val_low.max(dim=1).values  # (B, 12, 12)

        # Project Ch 2 gas hazard & survivor protection zones as soft safety field into 12x12 ego frame
        dx_haz = haz_wx - self.rx[:, None]
        dy_haz = haz_wy - self.ry[:, None]
        ego_haz_x = c * dx_haz + s * dy_haz
        ego_haz_y = -s * dx_haz + c * dy_haz
        haz_r = torch.clamp(self.hazard_zones[:, :, 2], min=0.15)

        dist_haz_grid = torch.hypot(
            self.ego_grid_x - ego_haz_x[:, :, None, None],
            self.ego_grid_y - ego_haz_y[:, :, None, None],
        )  # (B, MAX_HAZARDS, 12, 12)
        r_inner = 0.45 * haz_r[:, :, None, None]
        r_span = torch.clamp(0.75 * haz_r[:, :, None, None], min=0.12)
        val_haz = torch.clamp(1.0 - torch.clamp(dist_haz_grid - r_inner, min=0.0) / r_span, 0.0, 1.0)
        val_haz = torch.where(haz_vis[:, :, None, None], val_haz, 0.0)
        ch2 = val_haz.max(dim=1).values  # (B, 12, 12)

        # Occasional non-blocking false positive semantic markers on periphery
        if self.enable_semantic_noise and self.semantic_fp_prob > 0.0:
            fp_roll = torch.rand((self.B, 5), dtype=torch.float32, device=self.device, generator=self.sem_torch_gen)
            has_fp = fp_roll[:, 0] < self.semantic_fp_prob
            if bool(has_fp.any()):
                tx_w = torch.where(
                    self.nav_stage == 0,
                    self.nav_wp[:, 0, 0],
                    torch.where(self.nav_stage == 1, self.nav_wp[:, 1, 0], self.goal_x),
                )
                ty_w = torch.where(
                    self.nav_stage == 0,
                    self.nav_wp[:, 0, 1],
                    torch.where(self.nav_stage == 1, self.nav_wp[:, 1, 1], self.goal_y),
                )
                tdx = tx_w - self.rx
                tdy = ty_w - self.ry
                t_ego_x = torch.cos(self.rth) * tdx + torch.sin(self.rth) * tdy
                t_ego_y = -torch.sin(self.rth) * tdx + torch.cos(self.rth) * tdy
                t_ang = torch.atan2(t_ego_y, t_ego_x)

                # Offset angle >= 100 deg away from target direction at radius 0.60..0.82m so path is never blocked
                fp_side = torch.where(fp_roll[:, 1] < 0.5, 1.0, -1.0)
                fp_ang = t_ang + fp_side * (1.85 + 0.75 * fp_roll[:, 2])
                fp_dist = 0.60 + 0.22 * fp_roll[:, 3]
                fp_x = (fp_dist * torch.cos(fp_ang))[:, None, None]
                fp_y = (fp_dist * torch.sin(fp_ang))[:, None, None]

                gx_2d = self.ego_grid_x.squeeze(0)  # (1, 12, 12)
                gy_2d = self.ego_grid_y.squeeze(0)
                d_fp = torch.hypot(gx_2d - fp_x, gy_2d - fp_y)  # (B, 12, 12)
                fp_blob = torch.clamp(1.0 - d_fp / 0.18, 0.0, 1.0) * 0.60
                fp_to_ch1 = has_fp & (fp_roll[:, 4] < 0.5)
                fp_to_ch2 = has_fp & (fp_roll[:, 4] >= 0.5)
                ch1 = torch.where(fp_to_ch1[:, None, None], torch.maximum(ch1, fp_blob), ch1)
                ch2 = torch.where(fp_to_ch2[:, None, None], torch.maximum(ch2, fp_blob), ch2)

        # Modality Dropout (~25-30% of episodes/steps zero out Ch 1 & Ch 2 cleanly)
        if self.modality_dropout_prob >= 1.0:
            new_mod_drop = torch.ones(self.B, dtype=torch.bool, device=self.device)
        elif self.modality_dropout_prob <= 0.0:
            new_mod_drop = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        else:
            p_ep = 0.5 * self.modality_dropout_prob
            p_step = (self.modality_dropout_prob - p_ep) / max(1e-6, 1.0 - p_ep)
            u_drop = torch.rand(self.B, dtype=torch.float32, device=self.device, generator=self.sem_torch_gen)
            new_mod_drop = self.ep_modality_dropped | (u_drop < p_step)

        if upd_mask is None:
            self.modality_dropped = new_mod_drop
        else:
            self.modality_dropped = torch.where(upd_mask, new_mod_drop, self.modality_dropped)

        # Cache observed delayed/noisy/visibility-masked Ch1 & Ch2 state for BatchSoftCBFShield alignment
        not_dropped = (~self.modality_dropped)[:, None]
        if self.ch2_clean_env_frac > 0.0:
            haz_not_dropped = (~(self.modality_dropped & (~self._clean_ch2_env_mask)))[:, None]
        else:
            haz_not_dropped = not_dropped
        new_obs_low = torch.stack(
            [
                low_wx[:, : self.MAX_LOW_OBS],
                low_wy[:, : self.MAX_LOW_OBS],
                low_r[:, : self.MAX_LOW_OBS],
                (low_vis[:, : self.MAX_LOW_OBS] & not_dropped).float(),
            ],
            dim=-1,
        )
        new_obs_haz = torch.stack(
            [
                haz_wx,
                haz_wy,
                haz_r,
                (haz_vis & haz_not_dropped).float(),
            ],
            dim=-1,
        )
        if upd_mask is None:
            self._obs_low_state.copy_(new_obs_low)
            self._obs_haz_state.copy_(new_obs_haz)
        else:
            u3 = upd_mask[:, None, None]
            self._obs_low_state = torch.where(u3, new_obs_low, self._obs_low_state)
            self._obs_haz_state = torch.where(u3, new_obs_haz, self._obs_haz_state)

        if bool(self.modality_dropped.any()):
            keep_low = not_dropped[:, :, None]
            keep_haz = haz_not_dropped[:, :, None]
            ch1 = torch.where(keep_low, ch1, 0.0)
            ch2 = torch.where(keep_haz, ch2, 0.0)

        ch1 = torch.clamp(torch.nan_to_num(ch1, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).unsqueeze(1)
        ch2 = torch.clamp(torch.nan_to_num(ch2, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).unsqueeze(1)
        return ch1, ch2

    @torch.no_grad()
    def _compute_dynamic_ch3(
        self,
        upd_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute 12x12 ego-centric Ch 3 (dynamic obstacle future occupancy over H={0.5, 1.0, 1.5}s)
        matching Stage 1 dynamic_tracker.py formula with ZERO outer border clamping:
          ch3[i,j] = max_{k, tau} w_tau * clip(1 - max(d - r_c, 0) / max(r_eff(tau) - r_c, eps), 0, 1)
        Injects Stage 3 Ch3 DR (0.2~0.4s confirmation delay, velocity noise, 10~20% occlusion dropout)
        when enable_ch3_dr=True.
        """
        batch_arange = torch.arange(self.B, device=self.device)
        # Fetch delayed snapshot from 5-tick FIFO (0..4 ticks ago)
        delay_idx = torch.clamp(self.ch3_delay_ticks, 0, 4)
        snap = self.cyl_hist_fifo[batch_arange, delay_idx]  # (B, MAX_CYLS, 6)
        cx = snap[:, :, 0]
        cy = snap[:, :, 1]
        rc = snap[:, :, 2]
        vx = snap[:, :, 3]
        vy = snap[:, :, 4]
        vis = (snap[:, :, 5] > 0.5) & (self.dyn_seen_ticks >= delay_idx[:, None])

        if self.enable_ch3_dr:
            v_std = self.ch3_vel_noise_std[:, None]
            if bool((v_std > 0.0).any()):
                dvx = torch.randn((self.B, self.MAX_CYLS), dtype=torch.float32, device=self.device, generator=self.p4_torch_gen) * v_std
                dvy = torch.randn((self.B, self.MAX_CYLS), dtype=torch.float32, device=self.device, generator=self.p4_torch_gen) * v_std
                new_vn = torch.stack([dvx, dvy], dim=-1)
                if upd_mask is None:
                    self._obs_ch3_vnoise.copy_(new_vn)
                else:
                    self._obs_ch3_vnoise = torch.where(upd_mask[:, None, None], new_vn, self._obs_ch3_vnoise)
                vx = vx + self._obs_ch3_vnoise[:, :, 0]
                vy = vy + self._obs_ch3_vnoise[:, :, 1]
            else:
                if upd_mask is None:
                    self._obs_ch3_vnoise.zero_()
                else:
                    self._obs_ch3_vnoise = torch.where(
                        upd_mask[:, None, None], torch.zeros_like(self._obs_ch3_vnoise), self._obs_ch3_vnoise
                    )
            drop_p = self.ch3_drop_prob[:, None]
            if bool((drop_p > 0.0).any()):
                u_keep = torch.rand((self.B, self.MAX_CYLS), dtype=torch.float32, device=self.device, generator=self.p4_torch_gen)
                keep_m = u_keep >= drop_p
                if upd_mask is None:
                    self._obs_ch3_keep = keep_m
                else:
                    self._obs_ch3_keep = torch.where(upd_mask[:, None], keep_m, self._obs_ch3_keep)
                vis = vis & self._obs_ch3_keep
            else:
                if upd_mask is None:
                    self._obs_ch3_keep.fill_(True)
                else:
                    self._obs_ch3_keep = torch.where(
                        upd_mask[:, None], torch.ones_like(self._obs_ch3_keep), self._obs_ch3_keep
                    )
        else:
            if upd_mask is None:
                self._obs_ch3_vnoise.zero_()
                self._obs_ch3_keep.fill_(True)
            else:
                self._obs_ch3_vnoise = torch.where(
                    upd_mask[:, None, None], torch.zeros_like(self._obs_ch3_vnoise), self._obs_ch3_vnoise
                )
                self._obs_ch3_keep = torch.where(
                    upd_mask[:, None], torch.ones_like(self._obs_ch3_keep), self._obs_ch3_keep
                )

        if not bool(vis.any()):
            return torch.zeros((self.B, 1, PATCH_SIZE, PATCH_SIZE), dtype=torch.float32, device=self.device)

        c = torch.cos(self.rth)[:, None, None]  # (B, 1, 1)
        s = torch.sin(self.rth)[:, None, None]
        # Predict damped-CV future positions: (B, MAX_CYLS, H)
        a_tau = self.ch3_a_tau[None, None, :]  # (1, 1, H)
        px_w = cx[:, :, None] + vx[:, :, None] * a_tau
        py_w = cy[:, :, None] + vy[:, :, None] * a_tau
        dx_w = px_w - self.rx[:, None, None]
        dy_w = py_w - self.ry[:, None, None]
        ego_x = c * dx_w + s * dy_w  # (B, MAX_CYLS, H)
        ego_y = -s * dx_w + c * dy_w

        # Pure ego-patch projection (NO outer clamping!)
        # Broadcast to (B, MAX_CYLS, H, 12, 12)
        gx = self.ego_grid_x.unsqueeze(2)  # (1, 1, 1, 12, 12)
        gy = self.ego_grid_y.unsqueeze(2)
        dist = torch.hypot(gx - ego_x[:, :, :, None, None], gy - ego_y[:, :, :, None, None])
        r_c_b = rc[:, :, None, None, None]
        denom_b = self.ch3_denom_tau[None, None, :, None, None]
        w_b = self.ch3_w_tau[None, None, :, None, None]

        ratio = torch.clamp(dist - r_c_b, min=0.0) / denom_b
        val = w_b * torch.clamp(1.0 - ratio, 0.0, 1.0)
        val = torch.where(vis[:, :, None, None, None], val, 0.0)
        ch3 = val.amax(dim=(1, 2))  # (B, 12, 12)
        ch3 = torch.clamp(torch.nan_to_num(ch3, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).unsqueeze(1)
        return ch3

    @torch.no_grad()
    def _compute_obs_step(
        self,
        precomputed_lidar: Optional[torch.Tensor] = None,
        update_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute current tick feature (B, STEP_FEAT_DIM) & ego local costmap patch (B, C, 12, 12)."""
        if precomputed_lidar is None:
            noisy_72, _ = self._cast_lidar()
        else:
            noisy_72 = precomputed_lidar
        if update_ids is None:
            self._last_lidar_ranges = noisy_72
            upd_mask: Optional[torch.Tensor] = None
        else:
            self._last_lidar_ranges[update_ids] = noisy_72[update_ids]
            upd_mask = torch.zeros(self.B, dtype=torch.bool, device=self.device)
            upd_mask[update_ids] = True

        # Roll by 1 ray (NUM_RAYS // NUM_LIDAR_BINS // 2 = 1) so Bin 0 centers on straight-ahead!
        rolled = torch.roll(noisy_72, shifts=1, dims=1)
        bins_24 = rolled.view(self.B, NUM_LIDAR_BINS, NUM_RAYS // NUM_LIDAR_BINS).min(dim=2).values / MAX_RANGE_M

        # Build 12x12 robot-centric local occupancy patch Ch 0 (1=close obstacle hit, 0=clear)
        px = noisy_72 * self.ray_cos_local[None, :]  # (B, 72) in robot frame (+x forward)
        py = noisy_72 * self.ray_sin_local[None, :]  # (+y left)
        cell_res = PATCH_SPAN_M / PATCH_SIZE
        half = 0.5 * PATCH_SPAN_M
        valid_hit = (noisy_72 < MAX_RANGE_M - 0.08) & (px > -half) & (px < half) & (py > -half) & (py < half)
        ix = torch.clamp(((px + half) / cell_res).long(), 0, PATCH_SIZE - 1)
        iy = torch.clamp(((py + half) / cell_res).long(), 0, PATCH_SIZE - 1)

        ch0_flat = torch.zeros((self.B, PATCH_SIZE * PATCH_SIZE), dtype=torch.float32, device=self.device)
        flat_idx = ix * PATCH_SIZE + iy
        vals = torch.where(valid_hit, 1.0, 0.0)
        ch0_flat.scatter_add_(1, flat_idx, vals)
        ch0 = torch.clamp(ch0_flat, 0.0, 1.0).view(self.B, 1, PATCH_SIZE, PATCH_SIZE)

        if self.patch_channels == 1:
            patch = ch0
            if update_ids is None:
                self.modality_dropped.fill_(False)
            else:
                self.modality_dropped[update_ids] = False
        elif not self.use_semantic_channels:
            zeros_sem = torch.zeros(
                (self.B, self.patch_channels - 1, PATCH_SIZE, PATCH_SIZE),
                dtype=torch.float32,
                device=self.device,
            )
            patch = torch.cat([ch0, zeros_sem], dim=1)
            if update_ids is None:
                self.modality_dropped.fill_(True)
            else:
                self.modality_dropped[update_ids] = True
        elif self.patch_channels == 3:
            ch1, ch2 = self._compute_semantic_channels(upd_mask=upd_mask)
            patch = torch.cat([ch0, ch1, ch2], dim=1)
        else:
            ch1, ch2 = self._compute_semantic_channels(upd_mask=upd_mask)
            ch3 = self._compute_dynamic_ch3(upd_mask=upd_mask)
            patch = torch.cat([ch0, ch1, ch2, ch3], dim=1)

        # Target polar in ESTIMATED (odom-drifted) robot frame!
        # Optionally supply nav_wp / 0.45..1.50 m lookahead subgoal during 4-Ch training to align dist_norm with A* subgoals.
        dx_est = self.goal_x - self.odom_x
        dy_est = self.goal_y - self.odom_y
        if bool(self.use_subgoal_obs.any()):
            wp_x = torch.where(
                self.nav_stage == 0,
                self.nav_wp[:, 0, 0],
                torch.where(self.nav_stage == 1, self.nav_wp[:, 1, 0], self.goal_x),
            )
            wp_y = torch.where(
                self.nav_stage == 0,
                self.nav_wp[:, 0, 1],
                torch.where(self.nav_stage == 1, self.nav_wp[:, 1, 1], self.goal_y),
            )
            sdx = wp_x - self.odom_x
            sdy = wp_y - self.odom_y
            sdist = torch.hypot(sdx, sdy)
            scale_la = torch.clamp(self.subgoal_lookahead_m / torch.clamp(sdist, min=1e-4), max=1.0)
            sdx_la = sdx * scale_la
            sdy_la = sdy * scale_la
            dx_est = torch.where(self.use_subgoal_obs, sdx_la, dx_est)
            dy_est = torch.where(self.use_subgoal_obs, sdy_la, dy_est)
        dist_est = torch.hypot(dx_est, dy_est)
        ang_est = torch.atan2(dy_est, dx_est) - self.odom_th
        ang_est = torch.atan2(torch.sin(ang_est), torch.cos(ang_est))

        dist_norm = torch.clamp(dist_est / 4.0, 0.0, 1.0)
        sin_err = torch.sin(ang_est)
        cos_err = torch.cos(ang_est)

        v_cmd_norm = self.last_cmd[:, 0] / TB3_MAX_V
        w_cmd_norm = self.last_cmd[:, 1] / TB3_MAX_W
        v_act_norm = self.actual_vw[:, 0] / TB3_MAX_V
        w_act_norm = self.actual_vw[:, 1] / TB3_MAX_W
        err_v = v_cmd_norm - v_act_norm
        err_w = w_cmd_norm - w_act_norm
        step_norm = self.steps.float() / float(self.max_steps)

        feat = torch.cat(
            [
                bins_24,
                dist_norm[:, None],
                sin_err[:, None],
                cos_err[:, None],
                v_cmd_norm[:, None],
                w_cmd_norm[:, None],
                v_act_norm[:, None],
                w_act_norm[:, None],
                err_v[:, None],
                err_w[:, None],
                step_norm[:, None],
            ],
            dim=-1,
        )
        feat = torch.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)
        patch = torch.nan_to_num(patch, nan=0.0, posinf=1.0, neginf=0.0)
        return feat, patch

    # ------------------------------------------------------------------
    # Vectorized Step with Strong Positive Terminal/Progress/Bypass Rewards
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Step all B environments on GPU with exact differential-drive + plinth/low-obs/hazard/odom-drift physics.
        action: Tensor(B, 2) -> (v in [-0.10, 0.20], w in [-0.65, 0.65])
        """
        self.steps += 1

        clean_act = torch.nan_to_num(action, nan=0.0, posinf=TB3_MAX_V, neginf=self.v_min)
        raw_v = torch.clamp(clean_act[:, 0], self.v_min, TB3_MAX_V)
        raw_w = torch.clamp(clean_act[:, 1], -TB3_MAX_W, TB3_MAX_W)
        cmd = torch.stack([raw_v, raw_w], dim=-1)

        # Push into lag FIFO & pop delayed command according to per-env act_lag
        self.act_fifo = torch.roll(self.act_fifo, shifts=1, dims=1)
        self.act_fifo[:, 0, :] = cmd
        batch_arange = torch.arange(self.B, device=self.device)
        delayed = self.act_fifo[batch_arange, self.act_lag]

        # Apply Dynamixel motor deadzone (Track B realism)
        v_eff = torch.where(delayed[:, 0].abs() < self.track_cfg.deadzone_v, 0.0, delayed[:, 0])
        w_eff = torch.where(delayed[:, 1].abs() < self.track_cfg.deadzone_w, 0.0, delayed[:, 1])

        # Apply floor slip & wheel bias -> TRUE physical displacement
        v_true = v_eff * self.slip_v
        w_true = w_eff * self.slip_w + self.turn_bias * v_eff.abs()

        prev_rx = self.rx.clone()
        prev_ry = self.ry.clone()
        prev_rth = self.rth.clone()

        cand_th = prev_rth + w_true * self.dt
        cand_th = torch.atan2(torch.sin(cand_th), torch.cos(cand_th))
        cand_x = prev_rx + v_true * torch.cos(cand_th) * self.dt
        cand_y = prev_ry + v_true * torch.sin(cand_th) * self.dt

        # Sub-LiDAR invisible low plinth collision check: freezes forward travel into plinth!
        p_act = self.plinth[:, 3] > 0.5
        dx_p = cand_x - self.plinth[:, 0]
        dy_p = cand_y - self.plinth[:, 1]
        d_p = torch.hypot(dx_p, dy_p)
        hit_plinth_now = p_act & (d_p < self.plinth[:, 2] + TB3_RADIUS + 0.02)
        self.hit_plinth |= hit_plinth_now

        # Push robot cleanly out onto plinth rim with tangential deflection so it slides around on contact
        push_r = self.plinth[:, 2] + TB3_RADIUS + 0.022
        ux_p = dx_p / torch.clamp(d_p, min=1e-4)
        uy_p = dy_p / torch.clamp(d_p, min=1e-4)
        side_p = torch.where(dy_p >= 0.0, 1.0, -1.0)
        cand_x = torch.where(hit_plinth_now, self.plinth[:, 0] + ux_p * push_r + 0.025, cand_x)
        cand_y = torch.where(hit_plinth_now, self.plinth[:, 1] + uy_p * push_r + side_p * 0.045, cand_y)

        # Ch 1 Sub-LiDAR low-obstacle field physical contact & impedance check
        has_any_low = bool(self.low_obs_active.any())
        if has_any_low:
            dx_lo = cand_x[:, None] - self.low_obs[:, :, 0]  # (B, MAX_LOW_OBS)
            dy_lo = cand_y[:, None] - self.low_obs[:, :, 1]
            d_lo = torch.hypot(dx_lo, dy_lo)
            surf_lo = torch.where(self.low_obs_active, d_lo - self.low_obs[:, :, 2], 10.0)
            min_idx = surf_lo.argmin(dim=1)
            min_surf_lo = surf_lo[batch_arange, min_idx]
            min_low_clear = min_surf_lo - TB3_RADIUS
            hit_low_now = min_surf_lo < (TB3_RADIUS + 0.02)

            near_cx = self.low_obs[batch_arange, min_idx, 0]
            near_cy = self.low_obs[batch_arange, min_idx, 1]
            near_r = self.low_obs[batch_arange, min_idx, 2]
            near_d = torch.clamp(d_lo[batch_arange, min_idx], min=1e-4)
            push_lo_r = near_r + TB3_RADIUS + 0.022
            # Block forward penetration on the obstacle rim using pre-step radial direction (no +x/lateral slide assist)
            dx_lo_ref = prev_rx - near_cx
            dy_lo_ref = prev_ry - near_cy
            d_lo_ref = torch.hypot(dx_lo_ref, dy_lo_ref)
            use_cand_ref = d_lo_ref < 1e-4
            ux_lo = torch.where(use_cand_ref, (cand_x - near_cx) / near_d, dx_lo_ref / torch.clamp(d_lo_ref, min=1e-4))
            uy_lo = torch.where(use_cand_ref, (cand_y - near_cy) / near_d, dy_lo_ref / torch.clamp(d_lo_ref, min=1e-4))
            cand_x = torch.where(hit_low_now, near_cx + ux_lo * push_lo_r, cand_x)
            cand_y = torch.where(hit_low_now, near_cy + uy_lo * push_lo_r, cand_y)
            new_low_hit = hit_low_now & (~self.hit_low_obs)
            self.hit_low_obs |= hit_low_now
        else:
            hit_low_now = torch.zeros(self.B, dtype=torch.bool, device=self.device)
            new_low_hit = torch.zeros(self.B, dtype=torch.bool, device=self.device)
            min_low_clear = torch.full((self.B,), 10.0, dtype=torch.float32, device=self.device)

        self.rx = cand_x
        self.ry = cand_y
        self.rth = cand_th

        actual_ds = torch.hypot(self.rx - prev_rx, self.ry - prev_ry)
        # Signed forward speed inferred by chassis displacement (negative if reversing)
        fwd_proj = (self.rx - prev_rx) * torch.cos(self.rth) + (self.ry - prev_ry) * torch.sin(self.rth)
        v_meas = fwd_proj / self.dt
        dth_meas = torch.atan2(torch.sin(self.rth - prev_rth), torch.cos(self.rth - prev_rth))
        w_meas = dth_meas / self.dt
        if self.enable_plicp_vel_noise_dr:
            # Stage 3 PLICP 5 Hz sample-hold velocity noise DR
            upd_mask = ((self.steps + self.plicp_phase) % self.plicp_hold_ticks == 0) | (self.steps == 1)
            n_v = torch.randn(self.B, dtype=torch.float32, device=self.device, generator=self.p4_torch_gen) * self.plicp_v_std
            n_w = torch.randn(self.B, dtype=torch.float32, device=self.device, generator=self.p4_torch_gen) * self.plicp_w_std
            new_vw = torch.stack([v_meas + n_v, w_meas + n_w], dim=-1)
            self.plicp_held_vw = torch.where(upd_mask[:, None], new_vw, self.plicp_held_vw)
            self.actual_vw = self.plicp_held_vw.clone()
        else:
            self.actual_vw = torch.stack([v_meas, w_meas], dim=-1)

        # Update ODOM-drifted pose estimate with configurable onboard 2D ICP scan-matching drift bounding
        self.odom_th = self.odom_th + (w_eff * self.dt) + self.odom_yaw_bias * (v_eff.abs() * self.dt)
        self.odom_th = torch.atan2(torch.sin(self.odom_th), torch.cos(self.odom_th))
        self.odom_x = self.odom_x + (v_eff * self.odom_scale) * torch.cos(self.odom_th) * self.dt
        self.odom_y = self.odom_y + (v_eff * self.odom_scale) * torch.sin(self.odom_th) * self.dt
        if self.odom_icp_alpha > 0.0:
            if self.odom_icp_interval <= 1:
                a_icp = self.odom_icp_alpha
                self.odom_x = (1.0 - a_icp) * self.odom_x + a_icp * self.rx
                self.odom_y = (1.0 - a_icp) * self.odom_y + a_icp * self.ry
                self.odom_th = torch.atan2(
                    (1.0 - a_icp) * torch.sin(self.odom_th) + a_icp * torch.sin(self.rth),
                    (1.0 - a_icp) * torch.cos(self.odom_th) + a_icp * torch.cos(self.rth),
                )
            else:
                do_icp = (self.steps % self.odom_icp_interval) == 0
                a_t = torch.where(do_icp, self.odom_icp_alpha, 0.0)
                self.odom_x = (1.0 - a_t) * self.odom_x + a_t * self.rx
                self.odom_y = (1.0 - a_t) * self.odom_y + a_t * self.ry
                self.odom_th = torch.atan2(
                    (1.0 - a_t) * torch.sin(self.odom_th) + a_t * torch.sin(self.rth),
                    (1.0 - a_t) * torch.cos(self.odom_th) + a_t * torch.cos(self.rth),
                )

        # Advance moving dynamic obstacles: sinusoidal reciprocating crossfire OR linear wall-bounce
        sin_mask = self.cyl_active & self.cyl_sin_active
        lin_dyn_mask = self.cyl_active & (~self.cyl_sin_active) & (
            (self.cyls[:, :, 3].abs() + self.cyls[:, :, 4].abs()) > 1e-4
        )
        if bool(sin_mask.any()):
            t_now = self.steps.float()[:, None] * self.dt
            x0 = self.cyl_sin_params[:, :, 0]
            y0 = self.cyl_sin_params[:, :, 1]
            ax = self.cyl_sin_params[:, :, 2]
            ay = self.cyl_sin_params[:, :, 3]
            om = self.cyl_sin_params[:, :, 4]
            ph = self.cyl_sin_params[:, :, 5]
            ang = om * t_now + ph
            s_cx = x0 + ax * torch.sin(ang)
            s_cy = y0 + ay * torch.sin(ang)
            s_vx = ax * om * torch.cos(ang)
            s_vy = ay * om * torch.cos(ang)
            self.cyls[:, :, 0] = torch.where(sin_mask, s_cx, self.cyls[:, :, 0])
            self.cyls[:, :, 1] = torch.where(sin_mask, s_cy, self.cyls[:, :, 1])
            self.cyls[:, :, 3] = torch.where(sin_mask, s_vx, self.cyls[:, :, 3])
            self.cyls[:, :, 4] = torch.where(sin_mask, s_vy, self.cyls[:, :, 4])

        if bool(lin_dyn_mask.any()):
            next_cx = self.cyls[:, :, 0] + self.cyls[:, :, 3] * self.dt
            next_cy = self.cyls[:, :, 1] + self.cyls[:, :, 4] * self.dt
            bounce_y = (next_cy > (self.bounds[:, 3:4] - 0.25)) | (next_cy < (self.bounds[:, 2:3] + 0.25))
            bounce_x = (next_cx > (self.bounds[:, 1:2] - 0.25)) | (next_cx < (self.bounds[:, 0:1] + 0.25))
            self.cyls[:, :, 3] = torch.where(lin_dyn_mask & bounce_x, -self.cyls[:, :, 3], self.cyls[:, :, 3])
            self.cyls[:, :, 4] = torch.where(lin_dyn_mask & bounce_y, -self.cyls[:, :, 4], self.cyls[:, :, 4])
            self.cyls[:, :, 0] = torch.where(lin_dyn_mask, next_cx, self.cyls[:, :, 0])
            self.cyls[:, :, 1] = torch.where(lin_dyn_mask, next_cy, self.cyls[:, :, 1])

        # Push updated dynamic cylinder snapshot into Ch3 delay FIFO (index 0 = newest)
        is_dyn_now = sin_mask | lin_dyn_mask
        cur_dyn_dist = torch.hypot(self.cyls[:, :, 0] - self.rx[:, None], self.cyls[:, :, 1] - self.ry[:, None])
        in_dyn_range = is_dyn_now & (cur_dyn_dist <= 3.2)
        self.dyn_seen_ticks = torch.where(in_dyn_range, self.dyn_seen_ticks + 1, torch.zeros_like(self.dyn_seen_ticks))
        self.cyl_hist_fifo = torch.roll(self.cyl_hist_fifo, shifts=1, dims=1)
        self.cyl_hist_fifo[:, 0, :, :5] = self.cyls
        self.cyl_hist_fifo[:, 0, :, 5] = is_dyn_now.float()

        # Single LiDAR raycast per step (reused by _compute_obs_step to avoid 2x raycast overhead)
        noisy_72, min_clear = self._cast_lidar()
        cur_dist = torch.hypot(self.goal_x - self.rx, self.goal_y - self.ry)

        collision = min_clear < COLLISION_DIST
        reached = (cur_dist < GOAL_REACH_RADIUS) & (~collision)
        timeout = self.steps >= self.max_steps
        dones = collision | reached | timeout

        # Advance navigable intermediate stage cleanly when passing doorway/partition opening
        # v4.4: for gate_orth (wp0_seg_half_y > 0), measure shortest distance to the safe opening segment
        # x = gate_x, y in [gate_y - (0.5*gate_w - 0.18), gate_y + (0.5*gate_w - 0.18)] so lateral lane
        # staging inside the gate opening is unpenalized while outer-wall excursions remain penalized.
        wp0_y_clamped = torch.clamp(
            self.ry,
            self.nav_wp[:, 0, 1] - self.wp0_seg_half_y,
            self.nav_wp[:, 0, 1] + self.wp0_seg_half_y,
        )
        d_to_wp0 = torch.hypot(self.nav_wp[:, 0, 0] - self.rx, wp0_y_clamped - self.ry)
        d_to_wp1 = torch.hypot(self.nav_wp[:, 1, 0] - self.rx, self.nav_wp[:, 1, 1] - self.ry)
        adv0 = (self.nav_stage == 0) & ((d_to_wp0 < 0.28) | (self.rx > self.nav_wp[:, 0, 0] + 0.12))
        self.nav_stage = torch.where(adv0, 1, self.nav_stage)
        adv1 = (self.nav_stage == 1) & ((d_to_wp1 < 0.28) | (self.rx > self.nav_wp[:, 1, 0] + 0.12))
        self.nav_stage = torch.where(adv1, 2, self.nav_stage)

        d_01 = torch.hypot(self.nav_wp[:, 1, 0] - self.nav_wp[:, 0, 0], self.nav_wp[:, 1, 1] - self.nav_wp[:, 0, 1])
        d_1g = torch.hypot(self.goal_x - self.nav_wp[:, 1, 0], self.goal_y - self.nav_wp[:, 1, 1])
        nav_dist = torch.where(
            self.nav_stage == 0,
            d_to_wp0 + d_01 + d_1g,
            torch.where(self.nav_stage == 1, d_to_wp1 + d_1g, cur_dist),
        )

        # Strong Positive + Navigable Context-Aware Reward Formulation
        progress = torch.clamp(self.prev_dist - nav_dist, -0.30, 0.30) * 14.0
        self.prev_dist = nav_dist

        # Positive reward for tightening closest navigable goal record (prevents oscillation farming)
        record_gain = torch.clamp(self.min_dist_seen - nav_dist, min=0.0, max=0.40) * 8.0
        self.min_dist_seen = torch.minimum(self.min_dist_seen, nav_dist)

        # Positive reward for safely threading narrow passages (<0.34m clearance without crashing)
        tight_safe = (min_clear >= COLLISION_DIST + 0.015) & (min_clear < 0.34) & (actual_ds > 0.008) & (~self.bypass_bonus_given)
        bypass_bonus = torch.where(tight_safe, 3.5, 0.0)
        self.bypass_bonus_given |= tight_safe

        # S-bend Maze gate passage rewards (prevents Euclidean heuristic local minima!)
        g1_pass = (~self.gate1_done) & (self.rx > self.gate1_x)
        g2_pass = (~self.gate2_done) & (self.rx > self.gate2_x)
        gate_bonus = torch.where(g1_pass, 2.0, 0.0) + torch.where(g2_pass, 2.0, 0.0)
        self.gate1_done |= g1_pass
        self.gate2_done |= g2_pass

        # Tactile feedback + unpin/escape rewards on invisible sub-LiDAR plinth
        unpin_now = self.hit_plinth & (~hit_plinth_now) & (d_p > self.plinth[:, 2] + TB3_RADIUS + 0.05) & (~self.plinth_unpin_bonus_given)
        unpin_bonus = torch.where(unpin_now, 2.0, 0.0)
        self.plinth_unpin_bonus_given |= unpin_now

        cleared_plinth = self.hit_plinth & (~hit_plinth_now) & (d_p > self.plinth[:, 2] + TB3_RADIUS + 0.08) & (~self.plinth_escape_bonus_given) & (self.rx > self.plinth[:, 0] + 0.05)
        plinth_bonus = torch.where(cleared_plinth, 6.0, 0.0)
        self.plinth_escape_bonus_given |= cleared_plinth

        # Ch 1 Low-obstacle field contact/collision penalty & proactive contact-free clearance bonus
        low_contact_pen = (
            torch.where(new_low_hit, -6.0, 0.0)
            + torch.where(hit_low_now, -1.50, 0.0)
            + torch.where((~hit_low_now) & (min_low_clear < 0.08), (min_low_clear - 0.08) * 5.0, 0.0)
        )
        low_clear_pass = (
            (self.regime == 6)
            & (~self.hit_low_obs)
            & (~self.low_clear_bonus_given)
            & (self.rx > self.nav_wp[:, 1, 0] + 0.15)
        )
        low_clear_bonus = torch.where(low_clear_pass, 4.5, 0.0)
        self.low_clear_bonus_given |= low_clear_pass

        # Ch 2 Gas hazard & survivor protection zone entry penalty & safe clearance reward
        has_any_haz = bool(self.hazard_active.any())
        if has_any_haz:
            dx_hz = self.rx[:, None] - self.hazard_zones[:, :, 0]
            dy_hz = self.ry[:, None] - self.hazard_zones[:, :, 1]
            d_hz = torch.hypot(dx_hz, dy_hz)
            hz_r = torch.clamp(self.hazard_zones[:, :, 2], min=0.15)
            bound_hz = torch.where(self.hazard_active, d_hz - (hz_r + TB3_RADIUS), 10.0)
            min_hazard_clear = bound_hz.min(dim=1).values
            in_hazard_now = min_hazard_clear < 0.0
            self.entered_hazard |= in_hazard_now

            pen_depth = torch.clamp(-min_hazard_clear / 0.30, 0.0, 1.0)
            hazard_entry_pen = (
                torch.where(in_hazard_now, -1.60 - 2.60 * pen_depth, 0.0)
                + torch.where(
                    (~in_hazard_now) & (min_hazard_clear < 0.08),
                    (min_hazard_clear - 0.08) * 3.0,
                    0.0,
                )
            )

            near_safe_bypass = (
                (~in_hazard_now)
                & (min_hazard_clear >= 0.06)
                & (min_hazard_clear < 0.55)
                & (actual_ds > 0.006)
            )
            hazard_pass_bonus = (
                (self.regime == 7)
                & (~self.entered_hazard)
                & (~self.hazard_clear_bonus_given)
                & (self.rx > self.nav_wp[:, 1, 0] + 0.18)
            )
            self.hazard_clear_bonus_given |= hazard_pass_bonus
            hazard_clear_rew = torch.where(near_safe_bypass, 0.24, 0.0) + torch.where(hazard_pass_bonus, 8.0, 0.0)
        else:
            in_hazard_now = torch.zeros(self.B, dtype=torch.bool, device=self.device)
            min_hazard_clear = torch.full((self.B,), 10.0, dtype=torch.float32, device=self.device)
            pen_depth = torch.zeros(self.B, dtype=torch.float32, device=self.device)
            hazard_entry_pen = torch.zeros(self.B, dtype=torch.float32, device=self.device)
            hazard_clear_rew = torch.zeros(self.B, dtype=torch.float32, device=self.device)

        # Stage 3 pure physical safety cost c_t (excludes CBF intervention & slack)
        if bool(is_dyn_now.any()):
            dx_dyn = self.cyls[:, :, 0] - self.rx[:, None]
            dy_dyn = self.cyls[:, :, 1] - self.ry[:, None]
            d_dyn_center = torch.hypot(dx_dyn, dy_dyn)
            d_dyn_all = d_dyn_center - self.cyls[:, :, 2]
            d_dyn_masked = torch.where(is_dyn_now, d_dyn_all, 10.0)
            min_dyn_dist = d_dyn_masked.min(dim=1).values

            # General physical yield condition (zero scenario/regime hardcoding):
            # Front nearby dynamic obstacle (surface dist <= 0.90 m) actively approaching or crossing robot's forward corridor
            cos_th_d = torch.cos(self.rth)[:, None]
            sin_th_d = torch.sin(self.rth)[:, None]
            ego_x_dyn = dx_dyn * cos_th_d + dy_dyn * sin_th_d
            ego_y_dyn = -dx_dyn * sin_th_d + dy_dyn * cos_th_d
            cvx_dyn = self.cyls[:, :, 3]
            cvy_dyn = self.cyls[:, :, 4]
            ego_vy_dyn = -cvx_dyn * sin_th_d + cvy_dyn * cos_th_d
            closing_speed = -(dx_dyn * cvx_dyn + dy_dyn * cvy_dyn) / torch.clamp(d_dyn_center, min=1e-4)
            is_approaching = closing_speed > 0.02
            in_fwd_corridor = ego_y_dyn.abs() <= (self.cyls[:, :, 2] + TB3_RADIUS + 0.18)
            moving_into_corridor = (ego_y_dyn * ego_vy_dyn < 0.0) & (ego_y_dyn.abs() <= 0.80)
            is_crossing = (ego_vy_dyn.abs() > 0.05) & (in_fwd_corridor | moving_into_corridor)
            front_dyn_yield = (
                is_dyn_now
                & (d_dyn_all <= 0.90)
                & (ego_x_dyn > 0.0)
                & (is_approaching | is_crossing)
            ).any(dim=1)
        else:
            min_dyn_dist = torch.full((self.B,), 10.0, dtype=torch.float32, device=self.device)
            front_dyn_yield = torch.zeros(self.B, dtype=torch.bool, device=self.device)

        # Onset + bounded sustained cost prevents multi-tick low_obs/gas accumulation from inverting COST_W_COL=10.0
        low_onset = hit_low_now & (~self.prev_hit_low)
        low_sustain = hit_low_now & self.prev_hit_low
        self.prev_hit_low = hit_low_now.clone()

        haz_onset = in_hazard_now & (~self.prev_in_hazard)
        self.prev_in_hazard = in_hazard_now.clone()

        cost_col = COST_W_COL * collision.float()
        cost_low = COST_W_LOW * low_onset.float() + (0.015 * COST_W_LOW) * low_sustain.float()
        cost_gas = COST_W_GAS * haz_onset.float() + (0.12 * COST_W_GAS) * torch.where(in_hazard_now, pen_depth, 0.0)
        cost_dyn = COST_W_DYN * torch.clamp(COST_DYN_MARGIN_M - min_dyn_dist, min=0.0)
        cost_total = torch.nan_to_num(cost_col + cost_low + cost_gas + cost_dyn, nan=0.0, posinf=20.0, neginf=0.0)

        # Safety clearance & smoothness shaping (tactile penalty when blindly pushing into plinth!)
        danger_pen = torch.where(min_clear < 0.20, (min_clear - 0.20) * 5.0, 0.0)
        plinth_push_pen = torch.where(hit_plinth_now & (v_eff > 0.01), -0.25, 0.0)
        tgt_wp_x = torch.where(
            self.nav_stage == 0,
            self.nav_wp[:, 0, 0],
            torch.where(self.nav_stage == 1, self.nav_wp[:, 1, 0], self.goal_x),
        )
        tgt_wp_y = torch.where(
            self.nav_stage == 0,
            wp0_y_clamped,
            torch.where(self.nav_stage == 1, self.nav_wp[:, 1, 1], self.goal_y),
        )
        wp_ang = torch.atan2(tgt_wp_y - self.ry, tgt_wp_x - self.rx)
        heading_err = torch.atan2(torch.sin(wp_ang - self.rth), torch.cos(wp_ang - self.rth))
        facing_wp = heading_err.abs() <= (0.5 * math.pi)
        safe_dyn_wait = front_dyn_yield & (raw_v <= 0.03) & facing_wp
        stall_pen = torch.where(
            (actual_ds < 0.003) & (~hit_plinth_now) & (~hit_low_now) & (~safe_dyn_wait),
            -0.08,
            0.0,
        )
        unneeded_rev_pen = torch.where(
            (raw_v < -0.01) & (min_clear > 0.25) & (min_dyn_dist > 0.45),
            -0.08,
            0.0,
        )
        smooth_pen = -0.020 * torch.abs(raw_w - self.last_cmd[:, 1]) - 0.008 * torch.abs(raw_w)

        # Strong terminal positive reward + speed bonus + stall timeout discouragement
        time_bonus = torch.clamp(1.0 - self.steps.float() / float(self.max_steps), min=0.0) * 20.0
        terminal_r = (
            torch.where(reached, 75.0 + time_bonus, 0.0)
            + torch.where(collision, -22.0, 0.0)
            + torch.where(timeout & (~reached) & (~collision), -18.0, 0.0)
        )

        rewards = (
            progress
            + record_gain
            + bypass_bonus
            + gate_bonus
            + unpin_bonus
            + plinth_bonus
            + low_contact_pen
            + low_clear_bonus
            + hazard_entry_pen
            + hazard_clear_rew
            + danger_pen
            + plinth_push_pen
            + stall_pen
            + unneeded_rev_pen
            + smooth_pen
            + terminal_r
        )
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=-25.0)

        self.last_cmd = cmd
        feat, patch = self._compute_obs_step(precomputed_lidar=noisy_72)
        self.obs_hist = torch.roll(self.obs_hist, shifts=-1, dims=1)
        self.obs_hist[:, -1, :] = feat
        self.local_patch = patch

        # Track failed training seeds into PLR pool (before resetting done envs)
        done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
        info = {
            "reached": reached.clone(),
            "collision": collision.clone(),
            "timeout": timeout.clone(),
            "regimes": self.regime.clone(),
            "min_clear": min_clear.clone(),
            "low_obs_contact": hit_low_now.clone(),
            "ever_hit_low_obs": self.hit_low_obs.clone(),
            "min_low_obs_clear": min_low_clear.clone(),
            "low_contact_penalty": low_contact_pen.clone(),
            "hazard_zone_entered": in_hazard_now.clone(),
            "ever_entered_hazard": self.entered_hazard.clone(),
            "min_hazard_clear": min_hazard_clear.clone(),
            "hazard_entry_penalty": hazard_entry_pen.clone(),
            "hazard_clearance_reward": hazard_clear_rew.clone(),
            "modality_dropped": self.modality_dropped.clone(),
            "cost": cost_total.clone(),
            "cost_col": cost_col.clone(),
            "cost_low": cost_low.clone(),
            "cost_gas": cost_gas.clone(),
            "cost_dyn": cost_dyn.clone(),
            "min_dyn_dist": min_dyn_dist.clone(),
        }

        if len(done_ids) > 0 and self.mode == "train":
            done_np = done_ids.detach().cpu().numpy()
            fail_mask = (collision | timeout).detach().cpu().numpy()
            reg_np = self.regime.detach().cpu().numpy()
            for idx in done_np:
                if fail_mask[idx]:
                    self.hard_seeds.append((int(self.ep_seed[idx]), int(reg_np[idx])))
            if len(self.hard_seeds) > 2000:
                self.hard_seeds = self.hard_seeds[-2000:]

        return self.obs_hist, self.local_patch, rewards, dones, info


class BatchSoftCBFShield:
    """
    GPU-vectorized 2-Circle Covering Soft-CBF-QP Safety Shield (pure PyTorch 2x2 PDAS + Armijo).
    Mirrors Stage 2 cbf_safety_shield.py geometry, continuous decay rates, asymmetric Q, and runtime v_min:
      - Front circle (+l_f = +0.020 m), Rear circle (-l_r = -0.060 m), r_circle = 0.098 m
      - gamma_dyn = 3.0 < gamma_static = 6.0 <= gamma_gas = 8.0 [s^-1]
      - Penalties: p_dyn = 5000 > p_static = 2000 > p_gas = 400
      - Asymmetric Q = diag(w_v=1.0, w_w=0.05), input box U = [v_min, v_max] x [-w_max, w_max]
      - Inside Ch2 (h < 0): deactivates that Ch2 radial constraint to prevent divergence.
    """

    def __init__(
        self,
        dt: float = 0.10,
        v_min: float = TB3_MIN_V,
        v_max: float = TB3_MAX_V,
        w_max: float = TB3_MAX_W,
        l_f: float = 0.020,
        l_r: float = 0.060,
        r_circle: float = 0.098,
        gamma_dyn: float = 3.0,
        gamma_static: float = 6.0,
        gamma_gas: float = 8.0,
        w_v: float = 1.0,
        w_w: float = 0.05,
        p_dyn: float = 5000.0,
        p_static: float = 2000.0,
        p_gas: float = 400.0,
        lidar_max_points: int = 16,
        slip_margin_v: float = 0.015,
        dyn_max_accel: float = 0.40,
        max_iters: int = 12,
        use_observed_state: Optional[bool] = None,
    ):
        self.dt = float(dt)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.l_f = float(l_f)
        self.l_r = float(l_r)
        self.r_circle = float(r_circle)
        self.gamma_dyn = float(gamma_dyn)
        self.gamma_static = float(gamma_static)
        self.gamma_gas = float(gamma_gas)
        self.alpha_dyn_dt = (1.0 - math.exp(-self.gamma_dyn * self.dt)) / self.dt
        self.alpha_static_dt = (1.0 - math.exp(-self.gamma_static * self.dt)) / self.dt
        self.alpha_gas_dt = (1.0 - math.exp(-self.gamma_gas * self.dt)) / self.dt
        self.w_v = float(w_v)
        self.w_w = float(w_w)
        self.p_dyn = float(p_dyn)
        self.p_static = float(p_static)
        self.p_gas = float(p_gas)
        self.lidar_max_points = int(lidar_max_points)
        self.slip_margin_v = float(slip_margin_v)
        self.dyn_max_accel = float(dyn_max_accel)
        self.max_iters = int(max_iters)
        self.use_observed_state = use_observed_state

    @torch.no_grad()
    def solve_qp_batch(
        self,
        u_nom: torch.Tensor,
        a_v: torch.Tensor,
        a_w: torch.Tensor,
        b_rhs: torch.Tensor,
        p_wt: torch.Tensor,
        v_min_batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Solve batched 2D soft-CBF QP via exact slack elimination + 2x2 PDAS (Semismooth Newton)
        + vectorized Armijo 1D backtracking:
          min_{u in U} 0.5*(u - u_nom)^T Q (u - u_nom) + 0.5*sum_m p_m * max(0, a_m^T u - b_m)^2
        Returns: (u_safe (B, 2), slack_sq_sum (B,), intervened (B,) bool)
        """
        u_nom = torch.nan_to_num(u_nom, nan=0.0, posinf=self.v_max, neginf=self.v_min)
        v_lo = (
            v_min_batch.to(dtype=u_nom.dtype, device=u_nom.device)
            if v_min_batch is not None
            else torch.full((u_nom.shape[0],), self.v_min, dtype=u_nom.dtype, device=u_nom.device)
        )
        v_hi = torch.full_like(v_lo, self.v_max)
        u0_v = torch.minimum(torch.maximum(u_nom[:, 0], v_lo), v_hi)
        u0_w = torch.clamp(u_nom[:, 1], -self.w_max, self.w_max)
        uv = u0_v.clone()
        uw = u0_w.clone()

        def _obj(v_cur: torch.Tensor, w_cur: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            res = a_v * v_cur[:, None] + a_w * w_cur[:, None] - b_rhs
            viol = torch.where(p_wt > 0.0, torch.clamp(res, min=0.0), 0.0)
            phi = (
                0.5 * self.w_v * (v_cur - u0_v) ** 2
                + 0.5 * self.w_w * (w_cur - u0_w) ** 2
                + 0.5 * (p_wt * viol * viol).sum(dim=1)
            )
            return phi, viol

        phi_cur, viol_cur = _obj(uv, uw)
        if float(viol_cur.max().item()) <= 1e-7:
            intervened = (u0_v - u_nom[:, 0]).abs() + (u0_w - u_nom[:, 1]).abs() > 1e-6
            return torch.stack([uv, uw], dim=-1), torch.zeros_like(uv), intervened

        for _ in range(self.max_iters):
            act = (viol_cur > 0.0) & (p_wt > 0.0)
            pw_act = torch.where(act, p_wt, 0.0)
            gv = self.w_v * (uv - u0_v) + (pw_act * viol_cur * a_v).sum(dim=1)
            gw = self.w_w * (uw - u0_w) + (pw_act * viol_cur * a_w).sum(dim=1)

            # Projected-gradient KKT residual check
            pg_v = uv - torch.minimum(torch.maximum(uv - gv * 1e-3, v_lo), v_hi)
            pg_w = uw - torch.clamp(uw - gw * 1e-3, -self.w_max, self.w_max)
            active_env = (pg_v * pg_v + pg_w * pg_w) > 1e-16
            if not bool(active_env.any()):
                break

            h00 = self.w_v + (pw_act * a_v * a_v).sum(dim=1) + 1e-9
            h01 = (pw_act * a_v * a_w).sum(dim=1)
            h11 = self.w_w + (pw_act * a_w * a_w).sum(dim=1) + 1e-9

            fix_v = ((uv <= v_lo + 1e-9) & (gv > 0.0)) | ((uv >= self.v_max - 1e-9) & (gv < 0.0))
            fix_w = ((uw <= -self.w_max + 1e-9) & (gw > 0.0)) | ((uw >= self.w_max - 1e-9) & (gw < 0.0))

            det = torch.clamp(h00 * h11 - h01 * h01, min=1e-12)
            dv_full = -(h11 * gv - h01 * gw) / det
            dw_full = -(-h01 * gv + h00 * gw) / det

            dv = torch.where(fix_v, 0.0, torch.where(fix_w, -gv / h00, dv_full))
            dw = torch.where(fix_w, 0.0, torch.where(fix_v, -gw / h11, dw_full))
            dv = torch.where(active_env, dv, 0.0)
            dw = torch.where(active_env, dw, 0.0)

            # Vectorized Armijo backtracking line search
            accepted = torch.zeros_like(active_env)
            best_v = uv
            best_w = uw
            best_phi = phi_cur
            best_viol = viol_cur
            for step_sz in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
                cand_v = torch.minimum(torch.maximum(uv + step_sz * dv, v_lo), v_hi)
                cand_w = torch.clamp(uw + step_sz * dw, -self.w_max, self.w_max)
                cand_phi, cand_viol = _obj(cand_v, cand_w)
                imp = active_env & (~accepted) & (cand_phi < best_phi - 1e-10)
                if bool(imp.any()):
                    best_v = torch.where(imp, cand_v, best_v)
                    best_w = torch.where(imp, cand_w, best_w)
                    best_phi = torch.where(imp, cand_phi, best_phi)
                    best_viol = torch.where(imp[:, None], cand_viol, best_viol)
                    accepted |= imp
                if bool((accepted | (~active_env)).all()):
                    break
            if not bool(accepted.any()):
                break
            uv, uw, phi_cur, viol_cur = best_v, best_w, best_phi, best_viol

        slack_sq_sum = (viol_cur * viol_cur).sum(dim=1)
        u_safe = torch.stack([uv, uw], dim=-1)
        intervened = ((u_safe - u_nom).abs().sum(dim=-1) > 1e-4)
        return u_safe, slack_sq_sum, intervened

    @torch.no_grad()
    def filter_batch(
        self,
        env: VectorDisasterEnv,
        u_nom: torch.Tensor,
        use_observed_state: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build 2-Circle CBF constraints from env's current LiDAR/dynamic/Ch1/Ch2 state and solve batched QP.
        When use_observed_state is True (default during training; False during eval unless explicitly set):
          - Dynamic obstacles use delayed/confirmed cyl_hist_fifo snapshots (+ Ch3 velocity noise/occlusion DR)
            whenever env.enable_ch3_dr is True (prevents GT leakage during training).
          - Ch1/Ch2 use delayed/noisy/visibility-masked semantic detections whenever env.enable_semantic_noise is True.
        """
        b = env.B
        dev = env.device
        c = torch.cos(env.rth)[:, None]
        s = torch.sin(env.rth)[:, None]
        eff_use_obs = (
            bool(use_observed_state)
            if use_observed_state is not None
            else (bool(self.use_observed_state) if self.use_observed_state is not None else (env.mode == "train"))
        )

        # 1) Nearest K LiDAR points in ego frame
        ranges = env._last_lidar_ranges  # (B, 72)
        k_lid = min(self.lidar_max_points, ranges.shape[1])
        top_r, top_idx = torch.topk(ranges, k=k_lid, dim=1, largest=False)
        lid_valid = top_r < (MAX_RANGE_M - 0.15)
        lid_x = top_r * env.ray_cos_local[top_idx]
        lid_y = top_r * env.ray_sin_local[top_idx]
        lid_r = torch.full_like(lid_x, self.r_circle + 0.015 + 0.020)  # 0.133m
        lid_vx = torch.zeros_like(lid_x)
        lid_vy = torch.zeros_like(lid_x)
        lid_alpha = torch.full_like(lid_x, self.alpha_static_dt)
        lid_p = torch.where(lid_valid, self.p_static, 0.0)
        lid_aomax = torch.zeros_like(lid_x)
        lid_is_gas = torch.zeros_like(lid_valid)

        # 2) Dynamic cylinders in ego frame (delayed/confirmed tracker estimate under Ch3 DR when use_observed_state=True)
        if eff_use_obs and env.enable_ch3_dr:
            batch_arange = torch.arange(b, device=dev)
            delay_idx = torch.clamp(env.ch3_delay_ticks, 0, 4)
            snap = env.cyl_hist_fifo[batch_arange, delay_idx]  # (B, MAX_CYLS, 6)
            d_cx = snap[:, :, 0]
            d_cy = snap[:, :, 1]
            d_rc = snap[:, :, 2]
            d_vx_w = snap[:, :, 3] + env._obs_ch3_vnoise[:, :, 0]
            d_vy_w = snap[:, :, 4] + env._obs_ch3_vnoise[:, :, 1]
            dyn_act = (snap[:, :, 5] > 0.5) & (env.dyn_seen_ticks >= delay_idx[:, None]) & env._obs_ch3_keep
        else:
            dyn_act = env.cyl_active & (
                env.cyl_sin_active | ((env.cyls[:, :, 3].abs() + env.cyls[:, :, 4].abs()) > 1e-4)
            )
            d_cx = env.cyls[:, :, 0]
            d_cy = env.cyls[:, :, 1]
            d_rc = env.cyls[:, :, 2]
            d_vx_w = env.cyls[:, :, 3]
            d_vy_w = env.cyls[:, :, 4]
        dx_d = d_cx - env.rx[:, None]
        dy_d = d_cy - env.ry[:, None]
        dyn_x = c * dx_d + s * dy_d
        dyn_y = -s * dx_d + c * dy_d
        dyn_vx = c * d_vx_w + s * d_vy_w
        dyn_vy = -s * d_vx_w + c * d_vy_w
        dyn_r = self.r_circle + d_rc + 0.04 + 1.2 * 0.025
        dyn_alpha = torch.full_like(dyn_x, self.alpha_dyn_dt)
        dyn_p = torch.where(dyn_act, self.p_dyn, 0.0)
        dyn_aomax = torch.full_like(dyn_x, self.dyn_max_accel)
        dyn_is_gas = torch.zeros_like(dyn_act)

        # 3) Ch1 Low obstacles in ego frame (delayed/noisy detections when use_observed_state=True & enable_semantic_noise)
        if eff_use_obs and env.enable_semantic_noise and env.use_semantic_channels:
            l_wx = env._obs_low_state[:, :, 0]
            l_wy = env._obs_low_state[:, :, 1]
            l_rc = env._obs_low_state[:, :, 2]
            low_vis_mask = env._obs_low_state[:, :, 3] > 0.5
        else:
            l_wx = env.low_obs[:, :, 0]
            l_wy = env.low_obs[:, :, 1]
            l_rc = env.low_obs[:, :, 2]
            low_vis_mask = env.low_obs_active & (~env.modality_dropped[:, None])
        dx_l = l_wx - env.rx[:, None]
        dy_l = l_wy - env.ry[:, None]
        low_x = c * dx_l + s * dy_l
        low_y = -s * dx_l + c * dy_l
        low_r = self.r_circle + l_rc + 0.05
        low_vx = torch.zeros_like(low_x)
        low_vy = torch.zeros_like(low_x)
        low_alpha = torch.full_like(low_x, self.alpha_static_dt)
        low_p = torch.where(low_vis_mask, self.p_static, 0.0)
        low_aomax = torch.zeros_like(low_x)
        low_is_gas = torch.zeros_like(env.low_obs_active)

        # 4) Ch2 Gas / Survivor hazards in ego frame (delayed/noisy detections when use_observed_state=True & enable_semantic_noise)
        if eff_use_obs and env.enable_semantic_noise and env.use_semantic_channels:
            h_wx = env._obs_haz_state[:, :, 0]
            h_wy = env._obs_haz_state[:, :, 1]
            h_rc = torch.clamp(env._obs_haz_state[:, :, 2], min=0.15)
            haz_vis_mask = env._obs_haz_state[:, :, 3] > 0.5
        else:
            h_wx = env.hazard_zones[:, :, 0]
            h_wy = env.hazard_zones[:, :, 1]
            h_rc = torch.clamp(env.hazard_zones[:, :, 2], min=0.15)
            haz_vis_mask = env.hazard_active & (~env.modality_dropped[:, None])
        dx_h = h_wx - env.rx[:, None]
        dy_h = h_wy - env.ry[:, None]
        haz_x = c * dx_h + s * dy_h
        haz_y = -s * dx_h + c * dy_h
        haz_r = self.r_circle + h_rc + 0.05
        haz_vx = torch.zeros_like(haz_x)
        haz_vy = torch.zeros_like(haz_x)
        haz_alpha = torch.full_like(haz_x, self.alpha_gas_dt)
        haz_p = torch.where(haz_vis_mask, self.p_gas, 0.0)
        haz_aomax = torch.zeros_like(haz_x)
        haz_is_gas = torch.ones_like(env.hazard_active)

        # Concatenate all obstacle sources along M_obs dimension
        ox = torch.cat([lid_x, dyn_x, low_x, haz_x], dim=1)
        oy = torch.cat([lid_y, dyn_y, low_y, haz_y], dim=1)
        rsafe = torch.cat([lid_r, dyn_r, low_r, haz_r], dim=1)
        ovx = torch.cat([lid_vx, dyn_vx, low_vx, haz_vx], dim=1)
        ovy = torch.cat([lid_vy, dyn_vy, low_vy, haz_vy], dim=1)
        alp = torch.cat([lid_alpha, dyn_alpha, low_alpha, haz_alpha], dim=1)
        p_obs = torch.cat([lid_p, dyn_p, low_p, haz_p], dim=1)
        aomax = torch.cat([lid_aomax, dyn_aomax, low_aomax, haz_aomax], dim=1)
        is_gas = torch.cat([lid_is_gas, dyn_is_gas, low_is_gas, haz_is_gas], dim=1)
        is_dyn = torch.cat(
            [
                torch.zeros_like(lid_valid),
                dyn_act,
                torch.zeros_like(low_vis_mask),
                torch.zeros_like(haz_vis_mask),
            ],
            dim=1,
        )

        # Lateral-exit reverse-pullback guard & CPA-consistent velocity sensitivity on dynamic obstacles:
        # When robot heading is already laterally exiting an approaching dynamic obstacle's
        # trajectory lane, v < 0 pulls back into the lane (clamp v_min >= 0), and braking to v=0
        # freezes the robot on the tracks while CPA sensitivity d(d_cpa^2)/dv > 0 favors clearing laterally.
        spd_all = torch.hypot(ovx, ovy)
        dist_c_all = torch.hypot(ox, oy)
        ux_all = ovx / torch.clamp(spd_all, min=1e-6)
        uy_all = ovy / torch.clamp(spd_all, min=1e-6)
        d_lat_all = ox * uy_all - oy * ux_all
        abs_d_lat_all = d_lat_all.abs()
        approaching_all = (ox * ovx + oy * ovy) < 0.0
        exit_align_all = torch.where(
            abs_d_lat_all > 1e-4,
            (d_lat_all * (-uy_all)) / torch.clamp(abs_d_lat_all, min=1e-4),
            0.0,
        )
        lat_exit_mask = (
            is_dyn
            & (dist_c_all <= (rsafe + 1.20))
            & (spd_all > 0.05)
            & approaching_all
            & (abs_d_lat_all > 0.02)
            & (abs_d_lat_all < (rsafe + 0.25))
            & (exit_align_all > 0.20)
        )
        v_min_batch: Optional[torch.Tensor] = None
        if self.v_min < 0.0 and bool(lat_exit_mask.any()):
            lat_pullback = lat_exit_mask.any(dim=1)
            v_min_batch = torch.where(
                lat_pullback,
                torch.zeros(b, dtype=u_nom.dtype, device=dev),
                torch.full((b,), self.v_min, dtype=u_nom.dtype, device=dev),
            )

        # Build constraints for both front (+l_f) and rear (-l_r) covering circles
        av_list: List[torch.Tensor] = []
        aw_list: List[torch.Tensor] = []
        b_list: List[torch.Tensor] = []
        p_list: List[torch.Tensor] = []
        front_app_dyn = is_dyn & (ox > self.l_f) & (((ox - self.l_f) * ovx + oy * ovy) < 0.0)
        t_cpa0 = torch.clamp(
            -(ox * ovx + oy * ovy) / torch.clamp(spd_all * spd_all, min=1e-4),
            0.0,
            0.6,
        )
        av_cpa = -2.0 * t_cpa0 * d_lat_all * (-uy_all)
        for k_idx, l_k in enumerate((self.l_f, -self.l_r)):
            dpx = l_k - ox
            dpy = -oy
            dist_sq = dpx * dpx + dpy * dpy
            dist = torch.sqrt(torch.clamp(dist_sq, min=1e-12))
            h = dist_sq - rsafe * rsafe
            av_k = -2.0 * dpx
            aw_k = -2.0 * l_k * dpy
            if k_idx == 1:
                # Project rear-circle a_w -> 0 for front-approaching or laterally-exited dynamic obstacles
                # so rear leverage (-l_r) never counter-steers the nose into the obstacle.
                aw_k = torch.where(front_app_dyn | lat_exit_mask, 0.0, aw_k)
            use_cpa_v = lat_exit_mask & (exit_align_all > 0.25) & (dist > (rsafe - 0.03))
            av_k = torch.where(use_cpa_v, av_cpa, av_k)
            eps_inter = dist * (
                abs(self.v_max * self.w_max) + abs(l_k) * (self.w_max * self.w_max) + aomax
            ) * self.dt
            eps_slip = 2.0 * dist * self.slip_margin_v
            b_k = alp * h - 2.0 * (dpx * ovx + dpy * ovy) - eps_slip - eps_inter
            # Deactivate distant (> rsafe + 1.2m) or inside-Ch2 (h < 0) constraints
            near_m = dist <= (rsafe + 1.20)
            gas_exempt = is_gas & (h < 0.0)
            p_k = torch.where(near_m & (~gas_exempt), p_obs, 0.0)
            av_list.append(av_k)
            aw_list.append(aw_k)
            b_list.append(b_k)
            p_list.append(p_k)

        a_v = torch.cat(av_list, dim=1)
        a_w = torch.cat(aw_list, dim=1)
        b_rhs = torch.cat(b_list, dim=1)
        p_wt = torch.cat(p_list, dim=1)
        return self.solve_qp_batch(u_nom, a_v, a_w, b_rhs, p_wt, v_min_batch=v_min_batch)
