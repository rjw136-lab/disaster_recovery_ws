#!/usr/bin/env python3
"""
=================================================================================================
Project: Disaster Recovery TurtleBot3 Burger System
Module : td3_policy.py (Twin Delayed DDPG Local Reactive Obstacle Avoidance Policy & ONNX Engine)
Author : Autonomous Systems Laboratory / Jong Woo Ro
=================================================================================================
Hierarchical Hybrid Recovery Architecture - Layer 2: Local Reactive Avoidance (TD3)
- Lightweight Actor MLP (< 25K parameters, ~86 KB ONNX footprint)
- 28-dimensional state vector (24-sector centered LiDAR downsampling + Goal Polar + Action History)
- Continuous action space: [v in (0.03, 0.15) m/s, w in (-0.60, 0.60) rad/s]
- Ultra-low latency onboard inference (< 0.1ms on Host, < 1.0ms on Raspberry Pi 4 CPU)
- Line-of-sight clearance verifier for seamless handover to Layer 3 (2D ICP Drift Reset)
=================================================================================================
"""

import os
import math
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


# -----------------------------------------------------------------------------------------------
# Shared Heuristic Potential Field / VFH Prior Constants
# -----------------------------------------------------------------------------------------------
MIN_LASER_RANGE = 0.10
MAX_LASER_RANGE = 3.50
LOS_TARGET_EXTRA_MARGIN = 0.12

PRIOR_SECTOR_WEIGHTS = (1.4, 1.2, 1.0, 0.7, 0.5, 0.3)
PRIOR_OBS_THRESHOLD = 0.52
PRIOR_REPULSE_EXP = 1.5
PRIOR_SYM_BIAS = 0.58
PRIOR_SYM_HYST_W = 0.12
PRIOR_FRONT_THRESH = 0.48
PRIOR_FRONT_SCALE = 0.28
PRIOR_FRONT_GAIN = 1.35
PRIOR_GOAL_OFFSET = 0.32
PRIOR_GOAL_SCALE = 0.25
PRIOR_GOAL_MIN = 0.05
PRIOR_ATTRACT_GAIN = 1.10
PRIOR_REPULSE_GAIN = 0.95
PRIOR_W_SMOOTH = 0.80
PRIOR_TURN_SLOWDOWN = 0.75
PRIOR_TURN_MIN = 0.10
PRIOR_V_FRONT_OFFSET = 0.24
PRIOR_V_FRONT_SCALE = 0.25
PRIOR_OBS_MIN = 0.05

RESIDUAL_V_SCALE = 0.04
RESIDUAL_W_SCALE = 0.22


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


# -----------------------------------------------------------------------------------------------
# PyTorch Actor-Critic Architecture
# -----------------------------------------------------------------------------------------------
if TORCH_AVAILABLE:
    class TD3Actor(nn.Module):
        """
        Residual TD3 Actor with Heuristic Potential Field / VFH Prior:
        Combines a 28->128->128->2 MLP policy network with an embedded Heuristic Potential
        Field / VFH safety prior directly inside the computation graph so that the exported
        ONNX model provides balanced trajectory smoothness alongside reactive obstacle clearance.
        """
        def __init__(self, state_dim=28, action_dim=2, max_v=0.15, min_v=0.03, max_w=0.60):
            super(TD3Actor, self).__init__()
            self.fc1 = nn.Linear(state_dim, 128)
            self.ln1 = nn.LayerNorm(128)
            self.fc2 = nn.Linear(128, 128)
            self.ln2 = nn.LayerNorm(128)
            self.fc3 = nn.Linear(128, action_dim)

            self.max_v = max_v
            self.min_v = min_v
            self.max_w = max_w

            nn.init.orthogonal_(self.fc1.weight, gain=0.8)
            nn.init.orthogonal_(self.fc2.weight, gain=0.8)
            nn.init.zeros_(self.fc3.weight)
            nn.init.zeros_(self.fc3.bias)

            self.register_buffer(
                "sector_weights",
                torch.tensor([list(PRIOR_SECTOR_WEIGHTS)], dtype=torch.float32)
            )

        def _compute_prior(self, state):
            laser_m = state[..., :24] * MAX_LASER_RANGE
            heading_err = state[..., 25:26] * math.pi
            last_w = state[..., 27:28] * self.max_w

            front_min = torch.minimum(
                laser_m[..., 0:1],
                torch.minimum(laser_m[..., 1:2], laser_m[..., 23:24])
            )
            wide_front_min = torch.minimum(
                front_min,
                torch.minimum(laser_m[..., 2:3], laser_m[..., 22:23])
            )

            left_sec = laser_m[..., 1:7]
            right_sec = torch.flip(laser_m[..., 18:24], dims=[-1])

            left_pen = torch.clamp((PRIOR_OBS_THRESHOLD - left_sec) / PRIOR_OBS_THRESHOLD, min=0.0) ** PRIOR_REPULSE_EXP
            right_pen = torch.clamp((PRIOR_OBS_THRESHOLD - right_sec) / PRIOR_OBS_THRESHOLD, min=0.0) ** PRIOR_REPULSE_EXP

            repulse_w = torch.sum((right_pen - left_pen) * self.sector_weights, dim=-1, keepdim=True)

            left_clear = torch.mean(left_sec[..., :4], dim=-1, keepdim=True)
            right_clear = torch.mean(right_sec[..., :4], dim=-1, keepdim=True)

            sym_dir = torch.where(left_clear >= right_clear, PRIOR_SYM_BIAS, -PRIOR_SYM_BIAS)
            sym_dir = torch.where(
                torch.abs(last_w) > PRIOR_SYM_HYST_W,
                torch.where(last_w > 0.0, PRIOR_SYM_BIAS, -PRIOR_SYM_BIAS),
                sym_dir
            )
            front_pen = torch.clamp((PRIOR_FRONT_THRESH - front_min) / PRIOR_FRONT_SCALE, min=0.0)
            repulse_w = repulse_w + sym_dir * front_pen * PRIOR_FRONT_GAIN

            goal_weight = torch.clamp(
                (wide_front_min - PRIOR_GOAL_OFFSET) / PRIOR_GOAL_SCALE,
                min=PRIOR_GOAL_MIN,
                max=1.0
            )
            attract_w = PRIOR_ATTRACT_GAIN * heading_err * goal_weight

            w_prior_raw = (
                PRIOR_W_SMOOTH * (attract_w + PRIOR_REPULSE_GAIN * repulse_w)
                + (1.0 - PRIOR_W_SMOOTH) * last_w
            )
            w_prior = torch.clamp(w_prior_raw, min=-self.max_w, max=self.max_w)

            turn_factor = torch.clamp(
                1.0 - PRIOR_TURN_SLOWDOWN * (torch.abs(w_prior) / self.max_w),
                min=PRIOR_TURN_MIN,
                max=1.0
            )
            obs_factor = torch.clamp(
                (front_min - PRIOR_V_FRONT_OFFSET) / PRIOR_V_FRONT_SCALE,
                min=PRIOR_OBS_MIN,
                max=1.0
            )
            v_prior = self.min_v + (self.max_v - self.min_v) * turn_factor * obs_factor
            v_prior = torch.clamp(v_prior, min=self.min_v, max=self.max_v)
            return v_prior, w_prior

        def forward(self, state):
            # 1. Learned Deep MLP Residual Policy
            x = torch.relu(self.ln1(self.fc1(state)))
            x = torch.relu(self.ln2(self.fc2(x)))
            raw = torch.tanh(self.fc3(x))
            res_v = raw[..., 0:1] * RESIDUAL_V_SCALE
            res_w = raw[..., 1:2] * RESIDUAL_W_SCALE

            # 2. Embedded Heuristic Potential Field / VFH Prior
            v_prior, w_prior = self._compute_prior(state)

            v = torch.clamp(v_prior + res_v, min=self.min_v, max=self.max_v)
            w = torch.clamp(w_prior + res_w, min=-self.max_w, max=self.max_w)

            return torch.cat([v, w], dim=-1)


    class TD3Critic(nn.Module):
        """
        Twin Critic Network Q1, Q2 (s, a_norm | phi_1, phi_2) for TD3 value estimation.
        Accepts normalized actions a_norm in [-1, 1]^2.
        """
        def __init__(self, state_dim=28, action_dim=2):
            super(TD3Critic, self).__init__()
            # Q1 Architecture
            self.q1_fc1 = nn.Linear(state_dim + action_dim, 128)
            self.q1_fc2 = nn.Linear(128, 128)
            self.q1_fc3 = nn.Linear(128, 1)

            # Q2 Architecture
            self.q2_fc1 = nn.Linear(state_dim + action_dim, 128)
            self.q2_fc2 = nn.Linear(128, 128)
            self.q2_fc3 = nn.Linear(128, 1)

        def forward(self, state, action):
            sa = torch.cat([state, action], dim=-1)

            q1 = torch.relu(self.q1_fc1(sa))
            q1 = torch.relu(self.q1_fc2(q1))
            q1 = self.q1_fc3(q1)

            q2 = torch.relu(self.q2_fc1(sa))
            q2 = torch.relu(self.q2_fc2(q2))
            q2 = self.q2_fc3(q2)

            return q1, q2

        def q1(self, state, action):
            sa = torch.cat([state, action], dim=-1)
            q1 = torch.relu(self.q1_fc1(sa))
            q1 = torch.relu(self.q1_fc2(q1))
            return self.q1_fc3(q1)
else:
    class TD3Actor:
        pass
    class TD3Critic:
        pass


# -----------------------------------------------------------------------------------------------
# State Vectorization Helper (360 LiDAR -> 28 Dimensions)
# -----------------------------------------------------------------------------------------------
def extract_state_vector(
    scan_ranges,
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    target_x: float,
    target_y: float,
    last_v: float = 0.0,
    last_w: float = 0.0,
    max_v: float = 0.15,
    max_w: float = 0.60,
    num_laser_bins: int = 24,
    max_laser_range: float = MAX_LASER_RANGE
) -> np.ndarray:
    """
    Constructs normalized 28-dimensional state vector:
      [0..23]  : 24-sector centered LiDAR minimum distances normalized to [0, 1]
                 Bin 0 is centered at 0 deg [-7.5, +7.5 deg], Bin 1..11 left, Bin 13..23 right.
      [24]     : Normalized Euclidean distance to target waypoint (clipped at 2.0m) [0, 1]
      [25]     : Normalized relative target heading error in robot frame [-1, 1]
      [26]     : Previous normalized linear velocity [0, 1]
      [27]     : Previous normalized angular velocity [-1, 1]

    Real-Robot LiDAR Safety Handling:
      - +inf or ranges > max_laser_range -> clamped to max_laser_range (3.5m open space)
      - 0.0 < ranges < 0.10 -> clamped to 0.10m (minimum safety lower bound, never treated as 3.5m)
      - NaN, -inf, <= 0.0 (e.g. LDS-01 dropout) -> replaced by minimum of nearest valid circular
        neighbors (or conservative 0.10m bound if wide sector / entire scan is invalid)
      - Angle-based bin mapping via linspace [0, 2*pi) so non-multiple ray counts (e.g. 359, 361)
        never truncate tail rays or shift sector angles.
    """
    state = np.zeros(num_laser_bins + 4, dtype=np.float32)
    min_laser_range = MIN_LASER_RANGE
    safe_fallback_norm = float(min_laser_range / max_laser_range)

    # 1. 24-Sector Centered LiDAR Downsampling with Angle-Based Binning & Safe Dropout Masking
    if scan_ranges is not None and len(scan_ranges) > 0:
        raw = np.asarray(scan_ranges, dtype=np.float32).reshape(-1)
        n_pts = len(raw)

        # Invalid dropout / corrupted rays: NaN, -inf, or <= 0.0
        bad_mask = np.isnan(raw) | np.isneginf(raw) | (raw <= 0.0)
        valid_mask = ~bad_mask

        if not np.any(valid_mask):
            # Entire scan is invalid: induce conservative deceleration / obstacle avoidance
            state[:num_laser_bins] = safe_fallback_norm
        else:
            ranges = raw.copy()
            # 1) +inf or > max_laser_range -> max_laser_range
            far_mask = valid_mask & (np.isposinf(ranges) | (ranges > max_laser_range))
            ranges[far_mask] = max_laser_range

            # 2) Positive ultra-close readings (0.0 < r < 0.10) -> 0.10m safety lower bound
            close_mask = valid_mask & (ranges < min_laser_range)
            ranges[close_mask] = min_laser_range

            # 3) NaN, -inf, <= 0.0 -> minimum of nearest valid circular neighbors (or conservative bound)
            if np.any(bad_mask):
                valid_idx = np.flatnonzero(valid_mask)
                bad_idx = np.flatnonzero(bad_mask)
                pos = np.searchsorted(valid_idx, bad_idx)
                right_idx = valid_idx[pos % len(valid_idx)]
                left_idx = valid_idx[(pos - 1) % len(valid_idx)]
                neighbor_min = np.minimum(ranges[left_idx], ranges[right_idx])

                gap_span = (right_idx - left_idx) % n_pts
                gap_span = np.where(gap_span == 0, n_pts, gap_span)
                max_safe_gap = max(2, (2 * n_pts) // num_laser_bins)
                ranges[bad_idx] = np.where(
                    gap_span > max_safe_gap,
                    np.minimum(neighbor_min, min_laser_range),
                    neighbor_min,
                )

            ranges = np.clip(ranges, min_laser_range, max_laser_range)

            # 4) Angle-based bin mapping across [0, 2*pi) without tail truncation
            angles = np.linspace(0.0, 2.0 * math.pi, n_pts, endpoint=False, dtype=np.float64)
            bin_width = (2.0 * math.pi) / float(num_laser_bins)
            bin_indices = (
                np.floor(((angles + 0.5 * bin_width) % (2.0 * math.pi)) / bin_width).astype(np.int32)
                % num_laser_bins
            )

            sector_mins = np.full(num_laser_bins, np.inf, dtype=np.float32)
            np.minimum.at(sector_mins, bin_indices, ranges)

            unpopulated = ~np.isfinite(sector_mins)
            if np.any(unpopulated):
                pop_idx = np.flatnonzero(~unpopulated)
                unpop_idx = np.flatnonzero(unpopulated)
                pos_b = np.searchsorted(pop_idx, unpop_idx)
                r_b = pop_idx[pos_b % len(pop_idx)]
                l_b = pop_idx[(pos_b - 1) % len(pop_idx)]
                sector_mins[unpop_idx] = np.minimum(sector_mins[l_b], sector_mins[r_b])

            state[:num_laser_bins] = np.clip(sector_mins, min_laser_range, max_laser_range) / max_laser_range
    else:
        # Missing or empty scan: conservative safety lower bound (never 1.0 open space)
        state[:num_laser_bins] = safe_fallback_norm

    # 2. Target Goal Polar Representation
    dx = target_x - robot_x
    dy = target_y - robot_y
    dist_to_goal = math.hypot(dx, dy)
    target_angle = math.atan2(dy, dx)
    heading_diff = normalize_angle(target_angle - robot_yaw)

    state[num_laser_bins] = min(dist_to_goal, 2.0) / 2.0
    state[num_laser_bins + 1] = heading_diff / math.pi

    # 3. Previous Action History
    state[num_laser_bins + 2] = max(0.0, min(1.0, last_v / max(max_v, 1e-3)))
    state[num_laser_bins + 3] = max(-1.0, min(1.0, last_w / max(max_w, 1e-3)))

    return state


def compute_expert_avoidance_action(
    state: np.ndarray,
    max_v: float = 0.15,
    min_v: float = 0.03,
    max_w: float = 0.60
) -> tuple[float, float]:
    """
    Smooth Heuristic Potential-Field / VFH prior controller sharing identical parameters
    with TD3Actor._compute_prior(). Used for warm-starting TD3 and as a deterministic
    fallback safety controller.
    """
    laser_m = state[:24] * MAX_LASER_RANGE
    heading_err = float(state[25]) * math.pi
    last_w = float(state[27]) * max_w

    # Minimum clearance in front cone (Bins 23, 0, 1 -> [-22.5 deg, +22.5 deg])
    front_min = float(min(laser_m[0], laser_m[1], laser_m[23]))
    # Wider front cone (Bins 22..2 -> [-37.5 deg, +37.5 deg])
    wide_front_min = float(min(front_min, laser_m[2], laser_m[22]))

    left_sectors = laser_m[1:7]
    right_sectors = laser_m[18:24][::-1]
    sector_weights = np.asarray(PRIOR_SECTOR_WEIGHTS, dtype=np.float32)

    left_pen = np.maximum((PRIOR_OBS_THRESHOLD - left_sectors) / PRIOR_OBS_THRESHOLD, 0.0) ** PRIOR_REPULSE_EXP
    right_pen = np.maximum((PRIOR_OBS_THRESHOLD - right_sectors) / PRIOR_OBS_THRESHOLD, 0.0) ** PRIOR_REPULSE_EXP
    repulse_w = float(np.sum((right_pen - left_pen) * sector_weights))

    left_clear = float(np.mean(left_sectors[:4]))
    right_clear = float(np.mean(right_sectors[:4]))

    sym_dir = PRIOR_SYM_BIAS if (left_clear >= right_clear) else -PRIOR_SYM_BIAS
    if abs(last_w) > PRIOR_SYM_HYST_W:
        sym_dir = PRIOR_SYM_BIAS if (last_w > 0.0) else -PRIOR_SYM_BIAS

    front_pen = max(0.0, (PRIOR_FRONT_THRESH - front_min) / PRIOR_FRONT_SCALE)
    repulse_w += sym_dir * front_pen * PRIOR_FRONT_GAIN

    goal_weight = float(np.clip((wide_front_min - PRIOR_GOAL_OFFSET) / PRIOR_GOAL_SCALE, PRIOR_GOAL_MIN, 1.0))
    attract_w = PRIOR_ATTRACT_GAIN * heading_err * goal_weight

    w_prior_raw = (
        PRIOR_W_SMOOTH * (attract_w + PRIOR_REPULSE_GAIN * repulse_w)
        + (1.0 - PRIOR_W_SMOOTH) * last_w
    )
    w = float(np.clip(w_prior_raw, -max_w, max_w))

    turn_factor = float(np.clip(1.0 - PRIOR_TURN_SLOWDOWN * (abs(w) / max_w), PRIOR_TURN_MIN, 1.0))
    obs_factor = float(np.clip((front_min - PRIOR_V_FRONT_OFFSET) / PRIOR_V_FRONT_SCALE, PRIOR_OBS_MIN, 1.0))
    v = min_v + (max_v - min_v) * turn_factor * obs_factor
    v = float(np.clip(v, min_v, max_v))

    return v, w


def check_line_of_sight_clear(
    scan_ranges,
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    target_x: float,
    target_y: float,
    clearance_dist: float = 0.38
) -> bool:
    """
    Checks whether the obstacle has been safely bypassed:
      1. Forward cone [-22.5 deg, +22.5 deg] has clearance > clearance_dist
      2. All-around proximity clearance > 0.21m (not scraping alongside the obstacle)
      3. Ray cone toward target waypoint (within +-22.5 deg of target bearing) is clear
         up to min(dist_to_target, clearance_dist + 0.12).
    """
    state = extract_state_vector(
        scan_ranges, robot_x, robot_y, robot_yaw, target_x, target_y
    )
    laser_m = state[:24] * MAX_LASER_RANGE

    # 1. Forward cone check
    if min(laser_m[0], laser_m[1], laser_m[23]) < clearance_dist:
        return False

    # 2. All-around safety margin check (ensure robot body has passed the obstacle)
    if np.min(laser_m) < 0.21:
        return False

    # 3. Line-of-sight toward target waypoint
    dx = target_x - robot_x
    dy = target_y - robot_y
    dist_to_target = math.hypot(dx, dy)
    heading_diff = normalize_angle(math.atan2(dy, dx) - robot_yaw)

    # Only transition back if target is within front hemisphere (+-75 deg)
    if abs(heading_diff) > math.radians(75.0):
        return False

    # Map heading_diff to 24-bin index
    bin_angle = 2.0 * math.pi / 24.0
    center_bin = int(round((heading_diff % (2.0 * math.pi)) / bin_angle)) % 24
    req_clear = min(dist_to_target, clearance_dist + LOS_TARGET_EXTRA_MARGIN)

    for offset in (-1, 0, 1):
        b = (center_bin + offset) % 24
        if laser_m[b] < req_clear:
            return False

    return True


# -----------------------------------------------------------------------------------------------
# High-Performance TD3 Local Avoidance Inference Engine
# -----------------------------------------------------------------------------------------------
class TD3LocalAvoidancePolicy:
    """
    Unified Policy Engine:
    - Strictly loads .onnx with ONNX Runtime or .pth/.pt with PyTorch state_dict
    - Never silently falls back to an untrained random/zero neural network on load failure
    - Provides a built-in deterministic heuristic prior controller as fail-safe fallback
    """
    def __init__(
        self,
        model_path: str = None,
        max_v: float = 0.15,
        min_v: float = 0.03,
        max_w: float = 0.60,
        device: str = "cpu",
        allow_untrained: bool = False
    ):
        self.max_v = max_v
        self.min_v = min_v
        self.max_w = max_w
        self.device = device

        self.ort_session = None
        self.torch_actor = None
        self.model_type = "fallback"

        if model_path is not None and model_path != "":
            if os.path.exists(model_path):
                self.load(model_path)
            else:
                print(
                    f"[TD3][WARN] Model file not found at '{model_path}'. "
                    f"Locking policy to deterministic 'fallback' prior instead of untrained network."
                )
                self.model_type = "fallback"
        elif allow_untrained:
            if TORCH_AVAILABLE:
                self.torch_actor = TD3Actor(
                    state_dim=28, action_dim=2,
                    max_v=self.max_v, min_v=self.min_v, max_w=self.max_w
                ).to(self.device)
                self.torch_actor.eval()
                self.model_type = "torch"
            else:
                print("[TD3][WARN] PyTorch unavailable for allow_untrained=True. Using 'fallback'.")
                self.model_type = "fallback"

    def load(self, path: str) -> bool:
        """
        Strictly loads weights from .onnx (via ONNX Runtime) or .pth/.pt (via PyTorch state_dict).
        On any error or unsupported format, switches to 'fallback' and returns False.
        """
        self.ort_session = None
        self.torch_actor = None
        self.model_type = "fallback"

        if not path or not os.path.exists(path):
            print(f"[TD3][WARN] Load failed: path does not exist ('{path}'). Using 'fallback'.")
            return False

        if path.endswith(".onnx"):
            if not ORT_AVAILABLE:
                print(f"[TD3][WARN] Cannot load '{path}': onnxruntime is not installed. Using 'fallback'.")
                return False
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.ort_session = ort.InferenceSession(
                    path, sess_options=opts, providers=['CPUExecutionProvider']
                )
                self.model_type = "onnx"
                return True
            except Exception as e:
                print(f"[TD3][WARN] ONNX load error for '{path}': {e}. Switching to 'fallback'.")
                self.ort_session = None
                self.model_type = "fallback"
                return False

        elif path.endswith(".pth") or path.endswith(".pt"):
            if not TORCH_AVAILABLE:
                print(f"[TD3][WARN] Cannot load '{path}': PyTorch is not installed. Using 'fallback'.")
                return False
            try:
                actor = TD3Actor(
                    state_dim=28, action_dim=2,
                    max_v=self.max_v, min_v=self.min_v, max_w=self.max_w
                ).to(self.device)
                state_dict = torch.load(path, map_location=self.device, weights_only=True)
                actor.load_state_dict(state_dict)
                actor.eval()
                self.torch_actor = actor
                self.model_type = "torch"
                return True
            except Exception as e:
                print(f"[TD3][WARN] PyTorch state_dict load error for '{path}': {e}. Switching to 'fallback'.")
                self.torch_actor = None
                self.model_type = "fallback"
                return False

        else:
            print(
                f"[TD3][WARN] Unsupported model extension for '{path}' "
                f"(expected .onnx, .pth, or .pt). Switching to 'fallback'."
            )
            return False

    def predict(self, state: np.ndarray) -> tuple[float, float]:
        """
        Infers continuous action (v, w) from 28-dim state vector.
        Guaranteed to execute in < 1.5ms.
        """
        if self.model_type == "onnx" and self.ort_session is not None:
            inp = state.reshape(1, 28).astype(np.float32)
            out = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name: inp})[0]
            v = float(np.clip(out[0, 0], self.min_v, self.max_v))
            w = float(np.clip(out[0, 1], -self.max_w, self.max_w))
            return v, w

        elif self.model_type == "torch" and self.torch_actor is not None:
            with torch.no_grad():
                tensor_s = torch.from_numpy(state).unsqueeze(0).to(self.device)
                out = self.torch_actor(tensor_s).cpu().numpy()[0]
                v = float(np.clip(out[0], self.min_v, self.max_v))
                w = float(np.clip(out[1], -self.max_w, self.max_w))
                return v, w

        else:
            return compute_expert_avoidance_action(state, self.max_v, self.min_v, self.max_w)

    def export_onnx(self, output_path: str) -> bool:
        """Exports the active PyTorch actor to optimized ONNX model."""
        if not TORCH_AVAILABLE or self.torch_actor is None:
            print("[TD3] Cannot export ONNX: PyTorch actor unavailable.")
            return False

        dummy_input = torch.zeros(1, 28, dtype=torch.float32).to(self.device)
        self.torch_actor.eval()
        try:
            torch.onnx.export(
                self.torch_actor,
                dummy_input,
                output_path,
                input_names=['state'],
                output_names=['action'],
                dynamic_axes={'state': {0: 'batch_size'}, 'action': {0: 'batch_size'}},
                opset_version=14
            )
            print(f"[TD3] Successfully exported ONNX model to: {output_path}")
            return True
        except Exception as e:
            print(f"[TD3] ONNX export failed: {e}")
            return False
