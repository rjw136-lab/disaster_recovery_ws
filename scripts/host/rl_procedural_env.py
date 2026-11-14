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
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


REGIME_NAMES = (
    "collapsed_debris",
    "tight_corridor",
    "dynamic_crossing",
    "plinth_slip_drift",
)

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
PATCH_SPAN_M = 1.8  # +-0.9m around robot center -> 0.15m per cell
STEP_FEAT_DIM = 34  # 24 lidar + 3 goal polar + 2 cmd + 2 actual vel + 2 tracking err + 1 regime/step norm
HISTORY_LEN = 16    # 1.6 seconds at 10 Hz


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
    lidar_dropout_prob_range=(0.01, 0.04),
    lidar_glare_prob=0.005,
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
      - Sub-LiDAR low plinths (invisible on 2D LiDAR, detected via cmd-vs-disp stall)
      - Domain-randomized slip, actuator lag/deadzone, odom drift, and LiDAR artifacts
    """

    MAX_BOXES = 8
    MAX_CYLS = 14

    def __init__(
        self,
        num_envs: int = 256,
        track: str = "sim",
        device: Optional[torch.device] = None,
        max_steps: int = 320,
        dt: float = 0.10,
        mode: str = "train",
        base_seed: int = 100_000,
    ):
        self.B = int(num_envs)
        self.track_cfg = REAL_TB3_TRACK if track == "real_tb3" else SIM_TRACK
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.mode = mode
        self.rng = np.random.RandomState(base_seed)

        # Precompute ray unit vectors in robot frame (NUM_RAYS = 72, centered so bin 0 ~ 0 deg)
        # Matching TD3 extract_state_vector: rays from 0..2pi, rolled by half_bin when binning 72->24.
        ray_angs = np.linspace(0.0, 2.0 * math.pi, NUM_RAYS, endpoint=False, dtype=np.float32)
        self.ray_cos_local = torch.tensor(np.cos(ray_angs), device=self.device)  # (72,)
        self.ray_sin_local = torch.tensor(np.sin(ray_angs), device=self.device)  # (72,)

        # Arena bounding walls (x_min, x_max, y_min, y_max)
        self.bounds = torch.zeros((self.B, 4), dtype=torch.float32, device=self.device)

        # Boxes: (B, MAX_BOXES, 4) -> (cx, cy, hx, hy) axis-aligned rectangles
        self.box_active = torch.zeros((self.B, self.MAX_BOXES), dtype=torch.bool, device=self.device)
        self.boxes = torch.zeros((self.B, self.MAX_BOXES, 4), dtype=torch.float32, device=self.device)

        # Cylinders (static debris + moving dynamic obstacles): (B, MAX_CYLS, 5) -> (cx, cy, r, vx, vy)
        self.cyl_active = torch.zeros((self.B, self.MAX_CYLS), dtype=torch.bool, device=self.device)
        self.cyls = torch.zeros((self.B, self.MAX_CYLS, 5), dtype=torch.float32, device=self.device)

        # Sub-LiDAR invisible plinth (at most 1 per env for regime 3): (cx, cy, r, active)
        self.plinth = torch.zeros((self.B, 4), dtype=torch.float32, device=self.device)

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
        self.gate1_x = torch.full((self.B,), 999.0, dtype=torch.float32, device=self.device)
        self.gate2_x = torch.full((self.B,), 999.0, dtype=torch.float32, device=self.device)
        self.gate1_done = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self.gate2_done = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self.nav_wp = torch.zeros((self.B, 2, 2), dtype=torch.float32, device=self.device)
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

        # Temporal observation buffer (B, HISTORY_LEN, STEP_FEAT_DIM) & latest 12x12 local patch (B, 1, 12, 12)
        self.obs_hist = torch.zeros((self.B, HISTORY_LEN, STEP_FEAT_DIM), dtype=torch.float32, device=self.device)
        self.local_patch = torch.zeros((self.B, 1, PATCH_SIZE, PATCH_SIZE), dtype=torch.float32, device=self.device)

        # PLR hard-case pool (training mode only)
        self.hard_seeds: List[Tuple[int, int]] = []  # (seed, regime)

    # ------------------------------------------------------------------
    # Procedural Episode Generation (Pure Geometry, Zero Leakage)
    # ------------------------------------------------------------------
    def _build_single_layout(self, seed: int, forced_regime: Optional[int] = None) -> dict:
        r = np.random.RandomState(int(seed))
        regime = int(forced_regime) if forced_regime is not None else int(r.randint(0, 4))

        bounds = np.array([-2.2, 4.2, -1.8, 1.8], dtype=np.float32)
        boxes = np.zeros((self.MAX_BOXES, 4), dtype=np.float32)
        box_act = np.zeros(self.MAX_BOXES, dtype=bool)
        cyls = np.zeros((self.MAX_CYLS, 5), dtype=np.float32)
        cyl_act = np.zeros(self.MAX_CYLS, dtype=bool)
        plinth = np.zeros(4, dtype=np.float32)

        rx, ry = 0.0, float(r.uniform(-0.12, 0.12))
        rth = float(r.uniform(-0.35, 0.35))
        gx, gy = 2.2, 0.0
        wp1 = None
        wp2 = None

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
            # Regime 2: Maze partitions (clear side openings) + 2..3 Dynamic Moving Obstacles
            half_w = float(r.uniform(1.35, 1.90))
            bounds = np.array([-1.0, 3.4, -half_w, half_w], dtype=np.float32)
            gx = float(r.uniform(2.05, 2.70))
            gy = float(r.uniform(-half_w * 0.45, half_w * 0.45))

            w1x = float(r.uniform(0.68, 1.00))
            w1_side = 1.0 if r.rand() < 0.5 else -1.0
            # Partition leaves >=0.45*half_w clear channel on opposite side (clearly visible!)
            boxes[0] = [w1x, w1_side * (half_w * 0.48), 0.07, half_w * 0.52]
            box_act[0] = True
            w2x = float(r.uniform(1.38, 1.72))
            boxes[1] = [w2x, -w1_side * (half_w * 0.48), 0.07, half_w * 0.52]
            box_act[1] = True
            wp1 = (w1x, -w1_side * (half_w * 0.55))
            wp2 = (w2x, w1_side * (half_w * 0.55))

            n_dyn = int(r.randint(2, 4))
            for k in range(n_dyn):
                rad = float(r.uniform(0.14, 0.17))
                cx = float(r.uniform(0.55, gx - 0.35))
                cy = float(r.uniform(-half_w * 0.65, half_w * 0.65))
                speed = float(r.uniform(0.08, 0.16))
                vang = float(r.uniform(-math.pi, math.pi))
                cyls[k] = [cx, cy, rad, speed * math.cos(vang), speed * math.sin(vang)]
                cyl_act[k] = True

        else:
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

        if wp1 is None:
            wp1 = (gx, gy)
        if wp2 is None:
            wp2 = (gx, gy)

        return {
            "regime": regime,
            "bounds": bounds,
            "boxes": boxes,
            "box_act": box_act,
            "cyls": cyls,
            "cyl_act": cyl_act,
            "plinth": plinth,
            "pose": (rx, ry, rth),
            "goal": (gx, gy),
            "nav_wps": np.array([wp1, wp2], dtype=np.float32),
        }

    def reset_envs(
        self,
        env_ids: Optional[torch.Tensor] = None,
        eval_seeds: Optional[List[int]] = None,
        plr_prob: float = 0.25,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reset all (or masked subset) of parallel environments cleanly."""
        if env_ids is None:
            ids_np = np.arange(self.B, dtype=np.int64)
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
        b_plinth = np.zeros((n_res, 4), dtype=np.float32)
        b_rx = np.zeros(n_res, dtype=np.float32)
        b_ry = np.zeros(n_res, dtype=np.float32)
        b_rth = np.zeros(n_res, dtype=np.float32)
        b_gx = np.zeros(n_res, dtype=np.float32)
        b_gy = np.zeros(n_res, dtype=np.float32)
        b_nwps = np.zeros((n_res, 2, 2), dtype=np.float32)
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
            if eval_seeds is not None:
                seed = int(eval_seeds[k_idx % len(eval_seeds)])
                forced_regime = seed % 4 if self.mode == "eval" else None
                layout = self._build_single_layout(seed, forced_regime=forced_regime)
            elif self.mode == "train" and self.hard_seeds and self.rng.rand() < plr_prob:
                pick = self.hard_seeds[int(self.rng.randint(0, len(self.hard_seeds)))]
                seed, forced_regime = int(pick[0]), int(pick[1])
                layout = self._build_single_layout(seed, forced_regime=forced_regime)
            else:
                seed = int(self.rng.randint(100_000, 999_999_999))
                forced_regime = int((i + self.rng.randint(0, 4)) % 4)
                layout = self._build_single_layout(seed, forced_regime=forced_regime)

            self.ep_seed[i] = seed
            b_regime[k_idx] = layout["regime"]
            b_bounds[k_idx] = layout["bounds"]
            b_boxes[k_idx] = layout["boxes"]
            b_box_act[k_idx] = layout["box_act"]
            b_cyls[k_idx] = layout["cyls"]
            b_cyl_act[k_idx] = layout["cyl_act"]
            b_plinth[k_idx] = layout["plinth"]

            rx, ry, rth = layout["pose"]
            gx, gy = layout["goal"]
            nwps = layout["nav_wps"]
            b_rx[k_idx], b_ry[k_idx], b_rth[k_idx] = rx, ry, rth
            b_gx[k_idx], b_gy[k_idx] = gx, gy
            b_nwps[k_idx] = nwps
            d_01 = math.hypot(float(nwps[0, 0]) - rx, float(nwps[0, 1]) - ry)
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
            b_sv[k_idx] = float(self.rng.uniform(slip_lo, slip_hi))
            b_sw[k_idx] = float(self.rng.uniform(slip_lo, slip_hi))
            b_tbias[k_idx] = float(self.rng.uniform(*cfg.turn_bias_range))
            b_lag[k_idx] = int(self.rng.randint(0, cfg.act_lag_max + 1))

        idx_t = torch.from_numpy(ids_np).to(self.device)
        self.regime[idx_t] = torch.from_numpy(b_regime).to(self.device)
        self.bounds[idx_t] = torch.from_numpy(b_bounds).to(self.device)
        self.boxes[idx_t] = torch.from_numpy(b_boxes).to(self.device)
        self.box_active[idx_t] = torch.from_numpy(b_box_act).to(self.device)
        self.cyls[idx_t] = torch.from_numpy(b_cyls).to(self.device)
        self.cyl_active[idx_t] = torch.from_numpy(b_cyl_act).to(self.device)
        self.plinth[idx_t] = torch.from_numpy(b_plinth).to(self.device)

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
        self.nav_stage[idx_t] = 0
        self.prev_dist[idx_t] = d0_t
        self.min_dist_seen[idx_t] = d0_t
        self.steps[idx_t] = 0
        self.stall_steps[idx_t] = 0
        self.bypass_bonus_given[idx_t] = False
        self.plinth_escape_bonus_given[idx_t] = False
        self.plinth_unpin_bonus_given[idx_t] = False
        self.hit_plinth[idx_t] = False
        g1 = np.where((b_regime == 2) & b_box_act[:, 0], b_boxes[:, 0, 0] + 0.12, 999.0).astype(np.float32)
        g2 = np.where((b_regime == 2) & b_box_act[:, 1], b_boxes[:, 1, 0] + 0.12, 999.0).astype(np.float32)
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

        # Prime initial frame across history buffer
        feat, patch = self._compute_obs_step()
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

        # 4. Inject Track A / Track B sensor noise, dropout, and glare
        noise = torch.randn_like(ranges) * self.lidar_noise_std[:, None]
        noisy = torch.clamp(ranges + noise, 0.12, MAX_RANGE_M)

        drop_mask = torch.rand_like(ranges) < self.lidar_drop_prob[:, None]
        noisy = torch.where(drop_mask, MAX_RANGE_M, noisy)

        if self.track_cfg.lidar_glare_prob > 0.0:
            glare_mask = torch.rand_like(ranges) < self.track_cfg.lidar_glare_prob
            glare_val = torch.rand_like(ranges) * 0.45 + 0.15
            noisy = torch.where(glare_mask, glare_val, noisy)

        return noisy, min_clear

    @torch.no_grad()
    def _compute_obs_step(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute current tick feature (B, STEP_FEAT_DIM) & ego local costmap patch (B, 1, 12, 12)."""
        noisy_72, _ = self._cast_lidar()

        # Roll by 1 ray (NUM_RAYS // NUM_LIDAR_BINS // 2 = 1) so Bin 0 centers on straight-ahead!
        rolled = torch.roll(noisy_72, shifts=1, dims=1)
        bins_24 = rolled.view(self.B, NUM_LIDAR_BINS, NUM_RAYS // NUM_LIDAR_BINS).min(dim=2).values / MAX_RANGE_M

        # Build 12x12 robot-centric local occupancy patch (1=close obstacle hit, 0=clear)
        px = noisy_72 * self.ray_cos_local[None, :]  # (B, 72) in robot frame (+x forward)
        py = noisy_72 * self.ray_sin_local[None, :]  # (+y left)
        cell_res = PATCH_SPAN_M / PATCH_SIZE
        half = 0.5 * PATCH_SPAN_M
        valid_hit = (noisy_72 < MAX_RANGE_M - 0.08) & (px > -half) & (px < half) & (py > -half) & (py < half)
        ix = torch.clamp(((px + half) / cell_res).long(), 0, PATCH_SIZE - 1)
        iy = torch.clamp(((py + half) / cell_res).long(), 0, PATCH_SIZE - 1)

        patch = torch.zeros((self.B, PATCH_SIZE * PATCH_SIZE), dtype=torch.float32, device=self.device)
        flat_idx = ix * PATCH_SIZE + iy
        vals = torch.where(valid_hit, 1.0, 0.0)
        patch.scatter_add_(1, flat_idx, vals)
        patch = torch.clamp(patch, 0.0, 1.0).view(self.B, 1, PATCH_SIZE, PATCH_SIZE)

        # Target polar in ESTIMATED (odom-drifted) robot frame!
        # This forces Track B to notice when odom drifts away from physical LiDAR geometry.
        dx_est = self.goal_x - self.odom_x
        dy_est = self.goal_y - self.odom_y
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
        return feat, patch

    # ------------------------------------------------------------------
    # Vectorized Step with Strong Positive Terminal/Progress/Bypass Rewards
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Step all B environments on GPU with exact differential-drive + plinth/slip/odom-drift physics.
        action: Tensor(B, 2) -> (v in [-0.10, 0.20], w in [-0.65, 0.65])
        """
        self.steps += 1

        raw_v = torch.clamp(action[:, 0], TB3_MIN_V, TB3_MAX_V)
        raw_w = torch.clamp(action[:, 1], -TB3_MAX_W, TB3_MAX_W)
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

        # Push robot cleanly out onto plinth rim so it stalls physically until it reverses/steers around!
        push_r = self.plinth[:, 2] + TB3_RADIUS + 0.022
        ux_p = dx_p / torch.clamp(d_p, min=1e-4)
        uy_p = dy_p / torch.clamp(d_p, min=1e-4)
        self.rx = torch.where(hit_plinth_now, self.plinth[:, 0] + ux_p * push_r, cand_x)
        self.ry = torch.where(hit_plinth_now, self.plinth[:, 1] + uy_p * push_r, cand_y)
        self.rth = cand_th

        actual_ds = torch.hypot(self.rx - prev_rx, self.ry - prev_ry)
        # Signed forward speed inferred by chassis displacement (negative if reversing)
        fwd_proj = (self.rx - prev_rx) * torch.cos(self.rth) + (self.ry - prev_ry) * torch.sin(self.rth)
        v_meas = fwd_proj / self.dt
        dth_meas = torch.atan2(torch.sin(self.rth - prev_rth), torch.cos(self.rth - prev_rth))
        w_meas = dth_meas / self.dt
        self.actual_vw = torch.stack([v_meas, w_meas], dim=-1)

        # Update ODOM-drifted pose estimate (Track B drifts over distance; Track A near-exact)
        self.odom_th = self.odom_th + (w_eff * self.dt) + self.odom_yaw_bias * (v_eff.abs() * self.dt)
        self.odom_th = torch.atan2(torch.sin(self.odom_th), torch.cos(self.odom_th))
        self.odom_x = self.odom_x + (v_eff * self.odom_scale) * torch.cos(self.odom_th) * self.dt
        self.odom_y = self.odom_y + (v_eff * self.odom_scale) * torch.sin(self.odom_th) * self.dt

        # Advance moving dynamic obstacles inside arena bounds with wall bounce
        dyn_mask = self.cyl_active & ((self.cyls[:, :, 3].abs() + self.cyls[:, :, 4].abs()) > 1e-4)
        next_cx = self.cyls[:, :, 0] + self.cyls[:, :, 3] * self.dt
        next_cy = self.cyls[:, :, 1] + self.cyls[:, :, 4] * self.dt
        bounce_y = (next_cy > (self.bounds[:, 3:4] - 0.25)) | (next_cy < (self.bounds[:, 2:3] + 0.25))
        bounce_x = (next_cx > (self.bounds[:, 1:2] - 0.25)) | (next_cx < (self.bounds[:, 0:1] + 0.25))
        self.cyls[:, :, 3] = torch.where(dyn_mask & bounce_x, -self.cyls[:, :, 3], self.cyls[:, :, 3])
        self.cyls[:, :, 4] = torch.where(dyn_mask & bounce_y, -self.cyls[:, :, 4], self.cyls[:, :, 4])
        self.cyls[:, :, 0] = torch.where(dyn_mask, next_cx, self.cyls[:, :, 0])
        self.cyls[:, :, 1] = torch.where(dyn_mask, next_cy, self.cyls[:, :, 1])

        # Compute signed clearance & next observations
        _, min_clear = self._cast_lidar()
        cur_dist = torch.hypot(self.goal_x - self.rx, self.goal_y - self.ry)

        collision = min_clear < COLLISION_DIST
        reached = (cur_dist < GOAL_REACH_RADIUS) & (~collision)
        timeout = self.steps >= self.max_steps
        dones = collision | reached | timeout

        # Advance navigable intermediate stage cleanly when passing doorway/partition opening
        d_to_wp0 = torch.hypot(self.nav_wp[:, 0, 0] - self.rx, self.nav_wp[:, 0, 1] - self.ry)
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

        # Safety clearance & smoothness shaping (tactile penalty when blindly pushing into plinth!)
        danger_pen = torch.where(min_clear < 0.20, (min_clear - 0.20) * 5.0, 0.0)
        plinth_push_pen = torch.where(hit_plinth_now & (v_eff > 0.01), -0.25, 0.0)
        stall_pen = torch.where((actual_ds < 0.003) & (~hit_plinth_now), -0.08, 0.0)
        smooth_pen = -0.020 * torch.abs(raw_w - self.last_cmd[:, 1]) - 0.008 * torch.abs(raw_w)

        # Strong terminal positive reward + speed bonus + stall timeout discouragement
        time_bonus = torch.clamp(1.0 - self.steps.float() / float(self.max_steps), min=0.0) * 20.0
        terminal_r = (
            torch.where(reached, 75.0 + time_bonus, 0.0)
            + torch.where(collision, -22.0, 0.0)
            + torch.where(timeout & (~reached) & (~collision), -10.0, 0.0)
        )

        rewards = progress + record_gain + bypass_bonus + gate_bonus + unpin_bonus + plinth_bonus + danger_pen + plinth_push_pen + stall_pen + smooth_pen + terminal_r

        self.last_cmd = cmd
        feat, patch = self._compute_obs_step()
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
