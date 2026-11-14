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

TB3_MIN_V = -0.10
TB3_MAX_V = 0.20
TB3_MAX_W = 0.65


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

        def __init__(self, hidden_dim: int = 64):
            super().__init__()
            self.spatial_conv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
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

        def encode(self, obs_hist: torch.Tensor, local_patch: torch.Tensor) -> torch.Tensor:
            sp_emb = self.spatial_conv(local_patch)
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
        Input: flat float32 tensor of shape (B, 688) = [obs_hist(16*34), local_patch(12*12)].
        Output: float32 tensor (B, 2) = [v_cmd in [-0.10, 0.20], w_cmd in [-0.65, 0.65]].
        """

        def __init__(self):
            super().__init__()
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
                nn.Conv2d(1, 12, kernel_size=3, padding=1),
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

        def forward(self, flat_obs: torch.Tensor) -> torch.Tensor:
            b = flat_obs.shape[0]
            hist = flat_obs[:, : HISTORY_LEN * STEP_FEAT_DIM].view(b, HISTORY_LEN, STEP_FEAT_DIM)
            patch = flat_obs[:, HISTORY_LEN * STEP_FEAT_DIM :].view(b, 1, PATCH_SIZE, PATCH_SIZE)

            t_feat = self.tcn(hist.transpose(1, 2))
            s_feat = self.spatial(patch)
            cur = hist[:, -1, :]
            out_pre = self.head(torch.cat([cur, t_feat, s_feat], dim=-1))
            return scale_raw_action(torch.tanh(out_pre))


class ContextRLStudentPolicy:
    """
    Edge inference wrapper maintaining 1.6s rolling observation context and executing
    distilled ONNX policy at 10-20 Hz (<1.5ms CPU latency).
    """

    def __init__(
        self,
        onnx_path: str,
        v_max: float = TB3_MAX_V,
        w_max: float = TB3_MAX_W,
    ):
        if not HAS_ORT:
            raise RuntimeError("onnxruntime is required for ContextRLStudentPolicy")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.hist: Deque[np.ndarray] = collections.deque(maxlen=HISTORY_LEN)
        self.reset()

    def reset(self) -> None:
        self.hist.clear()
        zero_step = np.zeros(STEP_FEAT_DIM, dtype=np.float32)
        zero_step[:24] = 1.0
        for _ in range(HISTORY_LEN):
            self.hist.append(zero_step.copy())

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
    ) -> Tuple[np.ndarray, np.ndarray]:
        feat = np.zeros(STEP_FEAT_DIM, dtype=np.float32)
        patch = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

        if scan_ranges is not None and len(scan_ranges) >= 24:
            r = np.asarray(scan_ranges, dtype=np.float32).copy()
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
                np.add.at(patch, (ix, iy), 1.0)
                patch = np.clip(patch, 0.0, 1.0)
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
        return feat, patch

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
    ) -> Tuple[float, float]:
        feat, patch = self._build_step_and_patch(
            scan_ranges, rx, ry, rth, tx, ty, cmd_v, cmd_w, act_v, act_w, step_norm
        )
        self.hist.append(feat)
        hist_arr = np.stack(list(self.hist), axis=0).reshape(1, -1)
        patch_arr = patch.reshape(1, -1)
        flat = np.concatenate([hist_arr, patch_arr], axis=1).astype(np.float32)
        out = self.sess.run(None, {self.input_name: flat})[0][0]
        v = float(np.clip(out[0], TB3_MIN_V, self.v_max))
        w = float(np.clip(out[1], -self.w_max, self.w_max))
        return v, w
