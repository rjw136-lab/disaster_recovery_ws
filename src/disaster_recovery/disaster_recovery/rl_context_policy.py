#!/usr/bin/env python3
"""
Phase 2 Context-Adaptive Dual-Track RL Policy Architecture & Edge ONNX Runtime Wrapper.

Components:
  - TemporalContextActorCritic: V100 GPU Teacher combining:
      1) 1.6s Temporal GRU Context/Belief Encoder over (LiDAR + Goal + cmd vs actual disp)
      2) 2D Ego-Centric 12x12 Occupancy Costmap CNN
      3) Bounded continuous Actor (v in [-0.10, 0.20], w in [-0.65, 0.65]) & Value Critic
  - DistilledEdgeStudent: Pure feedforward 1D-TCN + 2D-CNN + MLP ONNX Student
    distilled from a 3-seed Early-Stopped Teacher Ensemble for <1.5 ms RPi4 10-20 Hz inference.
  - ContextRLStudentPolicy: Real-time ROS 2 rolling-history ONNX wrapper (zero PyTorch needed on edge).
"""

import collections
import math
from typing import Deque, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


HISTORY_LEN = 16
STEP_FEAT_DIM = 34
PATCH_SIZE = 12
PATCH_SPAN_M = 1.8
FLAT_INPUT_DIM = HISTORY_LEN * STEP_FEAT_DIM + PATCH_SIZE * PATCH_SIZE  # 544 + 144 = 688
SEMANTIC_PATCH_CHANNELS = 3
SEMANTIC_FLAT_INPUT_DIM = HISTORY_LEN * STEP_FEAT_DIM + SEMANTIC_PATCH_CHANNELS * PATCH_SIZE * PATCH_SIZE  # 544 + 432 = 976

TB3_MIN_V = -0.10
TB3_MAX_V = 0.20
TB3_MAX_W = 0.65


def compute_flat_input_dim(patch_channels: int = 1) -> int:
    """Return flattened observation dimension for DistilledEdgeStudent / ContextRLStudentPolicy."""
    return int(HISTORY_LEN * STEP_FEAT_DIM + int(patch_channels) * PATCH_SIZE * PATCH_SIZE)


def _build_ego_grid_xy() -> Tuple[np.ndarray, np.ndarray]:
    """Precompute 12x12 ego patch cell centers in robot frame (+x forward, +y left), matching rl_procedural_env.py."""
    cell_res = PATCH_SPAN_M / PATCH_SIZE
    half_span = 0.5 * PATCH_SPAN_M
    centers_1d = np.linspace(
        -half_span + 0.5 * cell_res,
        half_span - 0.5 * cell_res,
        PATCH_SIZE,
        dtype=np.float32,
    )
    gx_2d, gy_2d = np.meshgrid(centers_1d, centers_1d, indexing="ij")
    return gx_2d.astype(np.float32), gy_2d.astype(np.float32)


_EGO_GRID_X, _EGO_GRID_Y = _build_ego_grid_xy()


def _wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


if HAS_TORCH:

    def scale_raw_action(raw_tanh: torch.Tensor) -> torch.Tensor:
        """Map tanh output in [-1, 1]^2 smoothly onto TB3 (v in [-0.10, 0.20], w in [-0.65, 0.65])."""
        v_mid = 0.5 * (TB3_MIN_V + TB3_MAX_V)
        v_half = 0.5 * (TB3_MAX_V - TB3_MIN_V)
        v = v_mid + v_half * raw_tanh[..., 0]
        w = TB3_MAX_W * raw_tanh[..., 1]
        return torch.stack([v, w], dim=-1)

    class TemporalContextActorCritic(nn.Module):
        """Teacher Actor-Critic with 1.6s GRU context encoder + 12x12 spatial CNN."""

        def __init__(self, hidden_dim: int = 64, patch_channels: int = 1):
            super().__init__()
            self.hidden_dim = int(hidden_dim)
            self.patch_channels = int(patch_channels)
            self.spatial_conv = nn.Sequential(
                nn.Conv2d(self.patch_channels, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),  # 12x12 -> 6x6
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((3, 3)),  # 24 * 9 = 216
                nn.Flatten(),
                nn.Linear(216, 48),
                nn.Tanh(),
            )
            self.temporal_proj = nn.Sequential(
                nn.Linear(STEP_FEAT_DIM, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
            )
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

            fused_dim = STEP_FEAT_DIM + hidden_dim + 48  # 146
            self.actor_net = nn.Sequential(
                nn.Linear(fused_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 2),
            )
            self.log_std = nn.Parameter(torch.tensor([-0.85, -0.75], dtype=torch.float32))

            self.critic_net = nn.Sequential(
                nn.Linear(fused_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
            )

        def _align_patch_channels(self, local_patch: torch.Tensor) -> torch.Tensor:
            if local_patch.ndim == 3:
                local_patch = local_patch.unsqueeze(1)
            c_in = local_patch.shape[1]
            if c_in == self.patch_channels:
                return local_patch
            if c_in < self.patch_channels:
                pad_c = self.patch_channels - c_in
                zeros = torch.zeros(
                    (local_patch.shape[0], pad_c, local_patch.shape[2], local_patch.shape[3]),
                    dtype=local_patch.dtype,
                    device=local_patch.device,
                )
                return torch.cat([local_patch, zeros], dim=1)
            return local_patch[:, : self.patch_channels, :, :]

        def encode(self, obs_hist: torch.Tensor, local_patch: torch.Tensor) -> torch.Tensor:
            aligned_patch = self._align_patch_channels(local_patch)
            sp_emb = self.spatial_conv(aligned_patch)
            t_emb = self.temporal_proj(obs_hist)
            _, h_n = self.gru(t_emb)
            z_t = h_n.squeeze(0)
            cur = obs_hist[:, -1, :]
            return torch.cat([cur, z_t, sp_emb], dim=-1)

        def forward(self, obs_hist: torch.Tensor, local_patch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            fused = self.encode(obs_hist, local_patch)
            mean_raw = self.actor_net(fused)
            action = scale_raw_action(torch.tanh(mean_raw))
            value = self.critic_net(fused).squeeze(-1)
            return action, value

        def get_action_and_value(
            self,
            obs_hist: torch.Tensor,
            local_patch: torch.Tensor,
            action_taken: Optional[torch.Tensor] = None,
            deterministic: bool = False,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            fused = self.encode(obs_hist, local_patch)
            mean_pre = torch.clamp(self.actor_net(fused), -3.5, 3.5)
            std = torch.exp(torch.clamp(self.log_std, -2.5, 0.0)).unsqueeze(0)
            dist = torch.distributions.Normal(mean_pre, std)

            v_mid = 0.5 * (TB3_MIN_V + TB3_MAX_V)
            v_half = 0.5 * (TB3_MAX_V - TB3_MIN_V)

            if action_taken is None:
                if deterministic:
                    u = mean_pre
                else:
                    u = dist.rsample()
                raw_tanh = torch.tanh(u)
                action = scale_raw_action(raw_tanh)
            else:
                v_raw = torch.clamp((action_taken[..., 0] - v_mid) / v_half, -0.9995, 0.9995)
                w_raw = torch.clamp(action_taken[..., 1] / TB3_MAX_W, -0.9995, 0.9995)
                raw_tanh = torch.stack([v_raw, w_raw], dim=-1)
                u = 0.5 * torch.log((1.0 + raw_tanh) / (1.0 - raw_tanh))
                action = action_taken

            # Exact change-of-variables Jacobian correction for squashed tanh Gaussian
            log_prob = (dist.log_prob(u) - torch.log(torch.clamp(1.0 - raw_tanh * raw_tanh, min=1e-6))).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            value = self.critic_net(fused).squeeze(-1)
            return action, log_prob, entropy, value

    class DistilledEdgeStudent(nn.Module):
        """
        Lightweight static-shape edge Student ONNX model (1D-TCN + 2D-CNN + MLP).
        Input: flat float32 tensor of shape (B, 688) for 1-channel or (B, 976) for 3-channel
               = [obs_hist(16*34), local_patch(C*12*12)].
        Output: float32 tensor (B, 2) = [v_cmd in [-0.10, 0.20], w_cmd in [-0.65, 0.65]].
        """

        def __init__(self, patch_channels: int = 1):
            super().__init__()
            self.patch_channels = int(patch_channels)
            self.flat_input_dim = compute_flat_input_dim(self.patch_channels)
            self.tcn = nn.Sequential(
                nn.Conv1d(STEP_FEAT_DIM, 48, kernel_size=4, stride=2),  # L=16 -> 7
                nn.ReLU(),
                nn.Conv1d(48, 64, kernel_size=3, stride=2),             # L=7 -> 3
                nn.ReLU(),
                nn.Flatten(),                                           # 64*3 = 192
                nn.Linear(192, 64),
                nn.Tanh(),
            )
            self.spatial = nn.Sequential(
                nn.Conv2d(self.patch_channels, 12, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(12, 16, kernel_size=3, stride=2, padding=1),  # 6x6
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((3, 3)),                           # 16*9 = 144
                nn.Flatten(),
                nn.Linear(144, 48),
                nn.Tanh(),
            )
            self.head = nn.Sequential(
                nn.Linear(STEP_FEAT_DIM + 64 + 48, 96),
                nn.Tanh(),
                nn.Linear(96, 64),
                nn.Tanh(),
                nn.Linear(64, 2),
            )

        def forward(
            self,
            flat_obs: torch.Tensor,
            local_patch: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            b = flat_obs.shape[0]
            hist_dim = HISTORY_LEN * STEP_FEAT_DIM
            if local_patch is not None:
                hist = flat_obs.reshape(b, HISTORY_LEN, STEP_FEAT_DIM)
                if local_patch.ndim == 3:
                    local_patch = local_patch.unsqueeze(1)
                c_in = local_patch.shape[1]
                if c_in == self.patch_channels:
                    patch = local_patch
                elif c_in < self.patch_channels:
                    zeros = torch.zeros(
                        (b, self.patch_channels - c_in, PATCH_SIZE, PATCH_SIZE),
                        dtype=local_patch.dtype,
                        device=local_patch.device,
                    )
                    patch = torch.cat([local_patch, zeros], dim=1)
                else:
                    patch = local_patch[:, : self.patch_channels, :, :]
            else:
                hist = flat_obs[:, :hist_dim].view(b, HISTORY_LEN, STEP_FEAT_DIM)
                patch = flat_obs[:, hist_dim:].view(b, self.patch_channels, PATCH_SIZE, PATCH_SIZE)

            t_feat = self.tcn(hist.transpose(1, 2))
            s_feat = self.spatial(patch)
            cur = hist[:, -1, :]
            out_pre = self.head(torch.cat([cur, t_feat, s_feat], dim=-1))
            return scale_raw_action(torch.tanh(out_pre))


class ContextRLStudentPolicy:
    """
    Edge inference wrapper maintaining 1.6s rolling observation context and executing
    distilled ONNX policy at 10-20 Hz (<1.5ms CPU latency).
    Supports both 1-channel (688-dim) Phase 2-B ONNX models and 3-channel (976-dim)
    Phase 3 semantic ONNX models via automatic input shape detection.
    """

    def __init__(
        self,
        onnx_path: str,
        v_max: float = TB3_MAX_V,
        w_max: float = TB3_MAX_W,
        phase2b_mode: bool = False,
        patch_channels: Optional[int] = None,
    ):
        if not HAS_ORT:
            raise RuntimeError("onnxruntime is required for ContextRLStudentPolicy")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
        inp_meta = self.sess.get_inputs()[0]
        self.input_name = inp_meta.name
        self.onnx_path = str(onnx_path)

        detected_dim = FLAT_INPUT_DIM
        if inp_meta.shape and len(inp_meta.shape) >= 2 and isinstance(inp_meta.shape[-1], int):
            detected_dim = int(inp_meta.shape[-1])
        if patch_channels is not None:
            self.patch_channels = int(patch_channels)
        elif detected_dim == SEMANTIC_FLAT_INPUT_DIM:
            self.patch_channels = 3
        else:
            patch_elems = max(PATCH_SIZE * PATCH_SIZE, detected_dim - HISTORY_LEN * STEP_FEAT_DIM)
            self.patch_channels = max(1, patch_elems // (PATCH_SIZE * PATCH_SIZE))
        self.flat_input_dim = compute_flat_input_dim(self.patch_channels)

        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.phase2b_mode = bool(phase2b_mode)
        self._near_latch_ttl = np.zeros(24, dtype=np.int32)
        self.hist: Deque[np.ndarray] = collections.deque(maxlen=HISTORY_LEN)
        self.reset()

    def reset(self) -> None:
        self.hist.clear()
        self._near_latch_ttl.fill(0)
        zero_step = np.zeros(STEP_FEAT_DIM, dtype=np.float32)
        zero_step[:24] = 1.0
        for _ in range(HISTORY_LEN):
            self.hist.append(zero_step.copy())

    @staticmethod
    def project_semantic_channels(
        rx: float,
        ry: float,
        rth: float,
        semantic_obstacles: Optional[Sequence[Tuple[float, float, float]]] = None,
        semantic_hazards: Optional[Sequence[Tuple[float, float, float]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project world/odom-frame semantic objects into 12x12 ego-centric Ch 1 and Ch 2 grids,
        matching rl_procedural_env.py (_compute_semantic_channels) geometry 100%:
          - Ch 1 (low-lying sub-LiDAR obstacles):
              core_low = r + 0.04
              val_low = clip(1.0 - clip(dist - core_low, min=0.0) / 0.16, 0.0, 1.0)
          - Ch 2 (gas hazard & survivor protection zones):
              haz_r = max(r, 0.15)
              r_inner = 0.45 * haz_r
              r_span = max(0.75 * haz_r, 0.12)
              val_haz = clip(1.0 - clip(dist - r_inner, min=0.0) / r_span, 0.0, 1.0)
        """
        ch1 = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        ch2 = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

        c = float(math.cos(rth))
        s = float(math.sin(rth))

        if semantic_obstacles:
            obs_arr = np.asarray(semantic_obstacles, dtype=np.float32)
            if obs_arr.ndim == 2 and obs_arr.shape[0] > 0 and obs_arr.shape[1] >= 3:
                dx_low = obs_arr[:, 0] - float(rx)
                dy_low = obs_arr[:, 1] - float(ry)
                ego_low_x = c * dx_low + s * dy_low
                ego_low_y = -s * dx_low + c * dy_low
                low_r = obs_arr[:, 2]

                dist_low_grid = np.hypot(
                    _EGO_GRID_X[None, :, :] - ego_low_x[:, None, None],
                    _EGO_GRID_Y[None, :, :] - ego_low_y[:, None, None],
                )
                core_low = low_r[:, None, None] + np.float32(0.04)
                val_low = np.clip(
                    np.float32(1.0) - np.maximum(dist_low_grid - core_low, np.float32(0.0)) / np.float32(0.16),
                    np.float32(0.0),
                    np.float32(1.0),
                )
                ch1 = np.nan_to_num(val_low.max(axis=0), nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
                ch1 = np.clip(ch1, 0.0, 1.0)

        if semantic_hazards:
            haz_arr = np.asarray(semantic_hazards, dtype=np.float32)
            if haz_arr.ndim == 2 and haz_arr.shape[0] > 0 and haz_arr.shape[1] >= 3:
                dx_haz = haz_arr[:, 0] - float(rx)
                dy_haz = haz_arr[:, 1] - float(ry)
                ego_haz_x = c * dx_haz + s * dy_haz
                ego_haz_y = -s * dx_haz + c * dy_haz
                haz_r = np.maximum(haz_arr[:, 2], np.float32(0.15))

                dist_haz_grid = np.hypot(
                    _EGO_GRID_X[None, :, :] - ego_haz_x[:, None, None],
                    _EGO_GRID_Y[None, :, :] - ego_haz_y[:, None, None],
                )
                r_inner = np.float32(0.45) * haz_r[:, None, None]
                r_span = np.maximum(np.float32(0.75) * haz_r[:, None, None], np.float32(0.12))
                val_haz = np.clip(
                    np.float32(1.0) - np.maximum(dist_haz_grid - r_inner, np.float32(0.0)) / r_span,
                    np.float32(0.0),
                    np.float32(1.0),
                )
                ch2 = np.nan_to_num(val_haz.max(axis=0), nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
                ch2 = np.clip(ch2, 0.0, 1.0)

        return ch1, ch2

    @staticmethod
    def _build_step_and_patch(
        scan_ranges: Optional[Sequence[float]],
        rx: float,
        ry: float,
        rth: float,
        tx: float,
        ty: float,
        cmd_v: float,
        cmd_w: float,
        act_v: float,
        act_w: float,
        step_norm: float = 0.0,
        near_latch_ttl: Optional[np.ndarray] = None,
        patch_channels: int = 1,
        semantic_obstacles: Optional[Sequence[Tuple[float, float, float]]] = None,
        semantic_hazards: Optional[Sequence[Tuple[float, float, float]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        feat = np.zeros(STEP_FEAT_DIM, dtype=np.float32)
        ch0 = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

        if scan_ranges is not None and len(scan_ranges) >= 24:
            r = np.asarray(scan_ranges, dtype=np.float32).copy()
            if near_latch_ttl is not None:
                finite_pos = np.isfinite(r) & (r > 0.0)
                r[~finite_pos] = 3.5
                r = np.clip(r, 0.12, 3.5)
                n = len(r)
                step = max(1, n // 24)
                usable = step * 24
                rolled = np.roll(r[:usable], step // 2)
                rolled_fin = np.roll(finite_pos[:usable], step // 2)
                sec_r = rolled.reshape(24, step)
                sec_fin = rolled_fin.reshape(24, step)
                bins_24 = sec_r.min(axis=1) / 3.5
                for s in range(24):
                    if np.any(sec_fin[s]):
                        s_min = float(sec_r[s].min())
                        if s_min < 0.18:
                            near_latch_ttl[s] = 15
                        else:
                            near_latch_ttl[s] = 0
                    elif near_latch_ttl[s] > 0:
                        bins_24[s] = 0.12 / 3.5
                        near_latch_ttl[s] -= 1
            else:
                r[(~np.isfinite(r)) | (r < 0.12)] = 3.5
                r = np.clip(r, 0.12, 3.5)
                n = len(r)
                step = max(1, n // 24)
                usable = step * 24
                rolled = np.roll(r[:usable], step // 2)
                bins_24 = rolled.reshape(24, step).min(axis=1) / 3.5
            feat[:24] = bins_24

            angs = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False, dtype=np.float32)
            half = 0.5 * PATCH_SPAN_M
            cell_res = PATCH_SPAN_M / PATCH_SIZE
            px = r * np.cos(angs)
            py = r * np.sin(angs)
            mask = (r < 3.42) & (px > -half) & (px < half) & (py > -half) & (py < half)
            if np.any(mask):
                ix = np.clip(((px[mask] + half) / cell_res).astype(np.int32), 0, PATCH_SIZE - 1)
                iy = np.clip(((py[mask] + half) / cell_res).astype(np.int32), 0, PATCH_SIZE - 1)
                np.add.at(ch0, (ix, iy), 1.0)
                ch0 = np.clip(ch0, 0.0, 1.0)
        else:
            feat[:24] = 1.0

        dx = tx - rx
        dy = ty - ry
        dist_norm = min(math.hypot(dx, dy) / 4.0, 1.0)
        ang_err = _wrap_angle(math.atan2(dy, dx) - rth)
        feat[24] = dist_norm
        feat[25] = math.sin(ang_err)
        feat[26] = math.cos(ang_err)
        feat[27] = cmd_v / TB3_MAX_V
        feat[28] = cmd_w / TB3_MAX_W
        feat[29] = act_v / TB3_MAX_V
        feat[30] = act_w / TB3_MAX_W
        feat[31] = (cmd_v - act_v) / TB3_MAX_V
        feat[32] = (cmd_w - act_w) / TB3_MAX_W
        feat[33] = float(np.clip(step_norm, 0.0, 1.0))

        if int(patch_channels) <= 1:
            patch = ch0
        else:
            ch1, ch2 = ContextRLStudentPolicy.project_semantic_channels(
                rx=rx,
                ry=ry,
                rth=rth,
                semantic_obstacles=semantic_obstacles,
                semantic_hazards=semantic_hazards,
            )
            patch = np.stack([ch0, ch1, ch2], axis=0).astype(np.float32)

        return feat, patch

    def observe(
        self,
        scan_ranges: Optional[Sequence[float]],
        rx: float,
        ry: float,
        rth: float,
        tx: float,
        ty: float,
        cmd_v: float,
        cmd_w: float,
        act_v: float,
        act_w: float,
        step_norm: float = 0.0,
        semantic_obstacles: Optional[Sequence[Tuple[float, float, float]]] = None,
        semantic_hazards: Optional[Sequence[Tuple[float, float, float]]] = None,
    ) -> np.ndarray:
        """Append the current step feature vector to the rolling context buffer."""
        feat, patch = self._build_step_and_patch(
            scan_ranges, rx, ry, rth, tx, ty, cmd_v, cmd_w, act_v, act_w, step_norm,
            near_latch_ttl=(self._near_latch_ttl if self.phase2b_mode else None),
            patch_channels=self.patch_channels,
            semantic_obstacles=semantic_obstacles,
            semantic_hazards=semantic_hazards,
        )
        self.hist.append(feat)
        return patch

    def predict(
        self,
        scan_ranges: Optional[Sequence[float]],
        rx: float,
        ry: float,
        rth: float,
        tx: float,
        ty: float,
        cmd_v: float,
        cmd_w: float,
        act_v: float,
        act_w: float,
        step_norm: float = 0.0,
        semantic_obstacles: Optional[Sequence[Tuple[float, float, float]]] = None,
        semantic_hazards: Optional[Sequence[Tuple[float, float, float]]] = None,
    ) -> Tuple[float, float]:
        patch = self.observe(
            scan_ranges, rx, ry, rth, tx, ty, cmd_v, cmd_w, act_v, act_w, step_norm,
            semantic_obstacles=semantic_obstacles,
            semantic_hazards=semantic_hazards,
        )
        hist_arr = np.stack(list(self.hist), axis=0).reshape(1, -1)
        patch_arr = patch.reshape(1, -1)
        flat = np.concatenate([hist_arr, patch_arr], axis=1).astype(np.float32)
        out = self.sess.run(None, {self.input_name: flat})[0][0]
        v = float(np.clip(out[0], TB3_MIN_V, self.v_max))
        w = float(np.clip(out[1], -self.w_max, self.w_max))
        return v, w
