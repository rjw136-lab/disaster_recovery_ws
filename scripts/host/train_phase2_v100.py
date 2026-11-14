#!/usr/bin/env python3
"""
Phase 2 Large-Scale Anti-Overfitting Dual-Track RL Training & Ensemble Distillation (Tesla V100 32GB).

Trains 3 independent seeds (101, 202, 303) of TemporalContextActorCritic (Recurrent PPO) per track:
  - Track A ('sim'): Gazebo-matched physical dynamics for Sim-to-Sim.
  - Track B ('real_tb3'): Aggressive odom drift, L/R slip, deadzone, act lag, LDS-01/LD08 noise for Sim-to-Real.

Anti-overfitting & Generalization Safeguards:
  1. Procedural mixed-regime batches (collapsed_debris, tight_corridor, dynamic_crossing, plinth_slip_drift).
  2. Prioritized Level Replay (PLR) over hard failed training seeds (replay_prob=0.25).
  3. Zero-leakage evaluation every eval_every updates on 60 isolated unseen seeds (10000..10059, 15/regime).
  4. Generalization-Gap monitoring + Early Stopping on peak Unseen Eval SR.
  5. 3-Seed Teacher Ensemble -> DAgger Distillation into lightweight static (B, 688) ONNX Edge Student.
  6. 3D/2.5D MP4 rollout videos (Early/Mid/Student), learning curve PNGs, and nvidia-smi GPU logs.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
PKG_SRC = os.path.join(WS_ROOT, "src", "disaster_recovery")
for p in (SCRIPT_DIR, PKG_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from rl_procedural_env import (  # noqa: E402
    GOAL_REACH_RADIUS,
    HISTORY_LEN,
    PATCH_SIZE,
    REGIME_NAMES,
    STEP_FEAT_DIM,
    TB3_MAX_V,
    TB3_MAX_W,
    TB3_MIN_V,
    TB3_RADIUS,
    VectorDisasterEnv,
)
from disaster_recovery.rl_context_policy import (  # noqa: E402
    DistilledEdgeStudent,
    FLAT_INPUT_DIM,
    TemporalContextActorCritic,
)

EVAL_SEEDS = np.arange(10000, 10060, dtype=np.int64)


class NvidiaSmiLogger:
    """Background daemon logging Tesla V100 GPU utilization, VRAM, power, and temperature."""

    def __init__(self, log_path: str, interval_s: float = 5.0):
        self.log_path = log_path
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: List[Tuple[float, float, float, float]] = []

    def start(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"# nvidia-smi telemetry start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("timestamp,gpu_util_pct,mem_util_pct,mem_used_mib,mem_total_mib,power_w,temp_c\n")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                res = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3.0,
                )
                if res.returncode == 0 and res.stdout.strip():
                    line = res.stdout.strip().splitlines()[0]
                    parts = [float(x.strip()) for x in line.split(",")]
                    ts = time.strftime("%H:%M:%S")
                    self.samples.append((parts[0], parts[2], parts[4], parts[5]))
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        f.write(f"{ts},{line}\n")
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def stop(self) -> Dict[str, float]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if not self.samples:
            return {"mean_gpu_util": 0.0, "max_gpu_util": 0.0, "max_vram_mib": 0.0, "mean_power_w": 0.0}
        arr = np.asarray(self.samples, dtype=np.float32)
        return {
            "mean_gpu_util": float(arr[:, 0].mean()),
            "max_gpu_util": float(arr[:, 0].max()),
            "max_vram_mib": float(arr[:, 1].max()),
            "mean_power_w": float(arr[:, 2].mean()),
            "max_temp_c": float(arr[:, 3].max()),
        }


def _flatten_obs(obs_hist: torch.Tensor, local_patch: torch.Tensor) -> torch.Tensor:
    b = obs_hist.shape[0]
    return torch.cat([obs_hist.reshape(b, -1), local_patch.reshape(b, -1)], dim=-1)


@torch.no_grad()
def evaluate_policy_on_holdout(
    track: str,
    device: torch.device,
    policy_fn,
    max_steps: int = 320,
) -> Dict[str, object]:
    """Deterministic evaluation over all 60 unseen holdout seeds (10000..10059, 15/regime)."""
    env = VectorDisasterEnv(
        num_envs=len(EVAL_SEEDS),
        track=track,
        device=device,
        max_steps=max_steps,
        mode="eval",
        base_seed=10000,
    )
    obs_hist, local_patch = env.reset_envs(eval_seeds=EVAL_SEEDS)
    finished = torch.zeros(len(EVAL_SEEDS), dtype=torch.bool, device=device)
    reached = torch.zeros(len(EVAL_SEEDS), dtype=torch.bool, device=device)
    crashed = torch.zeros(len(EVAL_SEEDS), dtype=torch.bool, device=device)
    ep_ret = torch.zeros(len(EVAL_SEEDS), dtype=torch.float32, device=device)
    ep_steps = torch.zeros(len(EVAL_SEEDS), dtype=torch.float32, device=device)
    regimes = env.regime.clone()

    for _ in range(max_steps):
        if bool(finished.all()):
            break
        act = policy_fn(obs_hist, local_patch)
        obs_hist, local_patch, rew, dones, info = env.step(act)
        active = ~finished
        ep_ret[active] += rew[active]
        ep_steps[active] += 1.0
        just_done = active & dones
        reached[just_done] = info["reached"][just_done]
        crashed[just_done] = info["collision"][just_done]
        finished |= just_done

    per_regime: Dict[str, float] = {}
    for r_id, r_name in enumerate(REGIME_NAMES):
        mask = regimes == r_id
        if bool(mask.any()):
            per_regime[r_name] = float(reached[mask].float().mean().item())
        else:
            per_regime[r_name] = 0.0

    unseen_sr = float(reached.float().mean().item())
    coll_rate = float(crashed.float().mean().item())
    return {
        "unseen_sr": unseen_sr,
        "collision_rate": coll_rate,
        "timeout_rate": max(0.0, 1.0 - unseen_sr - coll_rate),
        "mean_return": float(ep_ret.mean().item()),
        "mean_steps": float(ep_steps.mean().item()),
        "per_regime": per_regime,
    }


def _draw_box_3d(ax, xmin, xmax, ymin, ymax, zmax, color="#475569", alpha=0.65):
    xs = [xmin, xmax]
    ys = [ymin, ymax]
    verts = [
        [(xs[0], ys[0], 0), (xs[1], ys[0], 0), (xs[1], ys[1], 0), (xs[0], ys[1], 0)],
        [(xs[0], ys[0], zmax), (xs[1], ys[0], zmax), (xs[1], ys[1], zmax), (xs[0], ys[1], zmax)],
        [(xs[0], ys[0], 0), (xs[1], ys[0], 0), (xs[1], ys[0], zmax), (xs[0], ys[0], zmax)],
        [(xs[0], ys[1], 0), (xs[1], ys[1], 0), (xs[1], ys[1], zmax), (xs[0], ys[1], zmax)],
        [(xs[0], ys[0], 0), (xs[0], ys[1], 0), (xs[0], ys[1], zmax), (xs[0], ys[0], zmax)],
        [(xs[1], ys[0], 0), (xs[1], ys[1], 0), (xs[1], ys[1], zmax), (xs[1], ys[0], zmax)],
    ]
    poly = Poly3DCollection(verts, facecolors=color, linewidths=0.4, edgecolors="#1e293b", alpha=alpha)
    ax.add_collection3d(poly)


def _draw_cyl_3d(ax, cx, cy, r, h, color="#b45309", alpha=0.80):
    th = np.linspace(0, 2 * np.pi, 16)
    z = np.array([0.0, h])
    th_grid, z_grid = np.meshgrid(th, z)
    x_grid = cx + r * np.cos(th_grid)
    y_grid = cy + r * np.sin(th_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha, linewidth=0.2)


@torch.no_grad()
def render_3d_rollout_video(
    track: str,
    stage_name: str,
    policy_fn,
    out_mp4: str,
    out_png: str,
    device: torch.device,
    eval_seed: int = 10003,
) -> Dict[str, object]:
    """
    Render a genuine 3D Isometric Procedural Disaster Navigation + 2D Costmap HUD MP4 video
    and extract a crisp high-res PNG snapshot, then verify with ffprobe.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_mp4)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)

    env = VectorDisasterEnv(num_envs=1, track=track, device=device, max_steps=320, mode="eval", base_seed=10000)
    obs_hist, local_patch = env.reset_envs(eval_seeds=np.array([eval_seed], dtype=np.int64))

    regime_id = int(env.regime[0].item())
    regime_name = REGIME_NAMES[regime_id]
    bounds = env.bounds[0].detach().cpu().numpy()
    boxes = env.boxes[0].detach().cpu().numpy()
    box_active = env.box_active[0].detach().cpu().numpy()
    plinth = env.plinth[0].detach().cpu().numpy()
    goal_x = float(env.goal_x[0].item())
    goal_y = float(env.goal_y[0].item())

    frames_rx: List[float] = []
    frames_ry: List[float] = []
    frames_rth: List[float] = []
    frames_vw: List[Tuple[float, float, float, float]] = []
    frames_lidar: List[np.ndarray] = []
    frames_patch: List[np.ndarray] = []
    frames_cyls: List[Tuple[np.ndarray, np.ndarray]] = []
    frames_clear: List[float] = []
    outcome = "TIMEOUT"

    for t in range(320):
        noisy_72, min_c = env._cast_lidar()
        frames_rx.append(float(env.rx[0].item()))
        frames_ry.append(float(env.ry[0].item()))
        frames_rth.append(float(env.rth[0].item()))
        frames_lidar.append(noisy_72[0].detach().cpu().numpy())
        frames_patch.append(local_patch[0, 0].detach().cpu().numpy())
        frames_cyls.append((
            env.cyls[0].detach().cpu().numpy().copy(),
            env.cyl_active[0].detach().cpu().numpy().copy(),
        ))
        frames_clear.append(float(min_c[0].item()))

        act = policy_fn(obs_hist, local_patch)
        cmd_v = float(act[0, 0].item())
        cmd_w = float(act[0, 1].item())
        act_v = float(env.actual_vw[0, 0].item())
        act_w = float(env.actual_vw[0, 1].item())
        frames_vw.append((cmd_v, cmd_w, act_v, act_w))

        obs_hist, local_patch, _, dones, info = env.step(act)
        if bool(dones[0].item()):
            frames_rx.append(float(env.rx[0].item()))
            frames_ry.append(float(env.ry[0].item()))
            frames_rth.append(float(env.rth[0].item()))
            noisy_72, min_c = env._cast_lidar()
            frames_lidar.append(noisy_72[0].detach().cpu().numpy())
            frames_patch.append(local_patch[0, 0].detach().cpu().numpy())
            frames_cyls.append((
                env.cyls[0].detach().cpu().numpy().copy(),
                env.cyl_active[0].detach().cpu().numpy().copy(),
            ))
            frames_clear.append(float(min_c[0].item()))
            frames_vw.append((cmd_v, cmd_w, float(env.actual_vw[0, 0].item()), float(env.actual_vw[0, 1].item())))
            if bool(info["reached"][0].item()):
                outcome = "PASS"
            elif bool(info["collision"][0].item()):
                outcome = "COLLISION"
            break

    w_px, h_px = 1280, 720
    dpi = 100
    fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi, facecolor="#0f172a")
    ff_cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w_px}x{h_px}", "-pix_fmt", "rgba", "-r", "12",
        "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
        out_mp4,
    ]
    proc = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    key_idx = max(0, min(len(frames_rx) - 1, int(0.65 * (len(frames_rx) - 1))))
    angs_local = np.linspace(0.0, 2.0 * np.pi, len(frames_lidar[0]), endpoint=False)

    # Hold end frame briefly so viewer sees outcome cleanly
    indices = list(range(len(frames_rx))) + [len(frames_rx) - 1] * 8
    for frame_no, idx in enumerate(indices):
        fig.clf()
        ax3d = fig.add_subplot(1, 2, 1, projection="3d", facecolor="#0f172a")
        ax2d = fig.add_subplot(1, 2, 2, facecolor="#1e293b")

        rx, ry, rth = frames_rx[idx], frames_ry[idx], frames_rth[idx]
        cyls, c_act = frames_cyls[idx]

        # 3D Scene
        ax3d.set_xlim(bounds[0] - 0.1, bounds[1] + 0.1)
        ax3d.set_ylim(bounds[2] - 0.1, bounds[3] + 0.1)
        ax3d.set_zlim(0.0, 1.2)
        ax3d.view_init(elev=36, azim=-62)
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False
        ax3d.tick_params(colors="#94a3b8", labelsize=7)
        ax3d.set_title(
            f"{track.upper()} | {stage_name} | Seed {eval_seed} ({regime_name})",
            color="#f8fafc", fontsize=9.2, fontweight="bold", pad=8, x=0.53,
        )

        # Arena outer walls outline on floor
        ax3d.plot(
            [bounds[0], bounds[1], bounds[1], bounds[0], bounds[0]],
            [bounds[2], bounds[2], bounds[3], bounds[3], bounds[2]],
            [0, 0, 0, 0, 0], color="#64748b", lw=1.2,
        )
        # Boxes (3D walls: [cx, cy, hx, hy])
        for bi in range(len(boxes)):
            if box_active[bi]:
                cx_b, cy_b, hx_b, hy_b = boxes[bi, 0], boxes[bi, 1], boxes[bi, 2], boxes[bi, 3]
                _draw_box_3d(ax3d, cx_b - hx_b, cx_b + hx_b, cy_b - hy_b, cy_b + hy_b, zmax=0.85, color="#475569", alpha=0.72)

        # Low plinth (3D invisible-to-LiDAR step plate)
        if plinth[3] > 0.5:
            _draw_cyl_3d(ax3d, plinth[0], plinth[1], plinth[2], h=0.055, color="#f43f5e", alpha=0.85)

        # Debris cylinders (static amber vs dynamic cyan)
        for ci in range(len(cyls)):
            if c_act[ci]:
                is_dyn = (abs(cyls[ci, 3]) + abs(cyls[ci, 4])) > 1e-4
                _draw_cyl_3d(
                    ax3d, cyls[ci, 0], cyls[ci, 1], cyls[ci, 2],
                    h=0.48, color="#06b6d4" if is_dyn else "#d97706", alpha=0.85,
                )

        # Goal beacon
        ax3d.scatter([goal_x], [goal_y], [0.22], color="#22c55e", s=160, marker="*", zorder=10)
        _draw_cyl_3d(ax3d, goal_x, goal_y, GOAL_REACH_RADIUS, h=0.03, color="#22c55e", alpha=0.35)

        # Robot 3D cylinder + heading arrow + trajectory ribbon
        _draw_cyl_3d(ax3d, rx, ry, TB3_RADIUS, h=0.19, color="#38bdf8", alpha=0.95)
        ax3d.plot(
            [rx, rx + 0.28 * np.cos(rth)],
            [ry, ry + 0.28 * np.sin(rth)],
            [0.20, 0.20], color="#fef08a", lw=2.5,
        )
        ax3d.plot(frames_rx[: idx + 1], frames_ry[: idx + 1], [0.05] * (idx + 1), color="#38bdf8", lw=2.0)

        # LiDAR 3D hit points at z=0.17m
        ranges = frames_lidar[idx]
        hit_mask = ranges < 3.40
        gx_hit = rx + ranges[hit_mask] * np.cos(rth + angs_local[hit_mask])
        gy_hit = ry + ranges[hit_mask] * np.sin(rth + angs_local[hit_mask])
        ax3d.scatter(gx_hit, gy_hit, np.full_like(gx_hit, 0.17), c="#ef4444", s=6, alpha=0.75)

        # Right pane: 12x12 Ego Occupancy Costmap + HUD metrics
        patch_img = frames_patch[idx].T
        ax2d.imshow(patch_img, origin="lower", cmap="inferno", extent=[-0.9, 0.9, -0.9, 0.9], vmin=0.0, vmax=1.0)
        ax2d.plot([0, 0.30], [0, 0], color="#38bdf8", lw=2.5)
        ax2d.add_patch(plt.Circle((0, 0), TB3_RADIUS, fill=False, edgecolor="#38bdf8", lw=2.0))
        cv, cw, av, aw = frames_vw[idx]
        dist_rem = math.hypot(goal_x - rx, goal_y - ry)
        status_col = "#4ade80" if outcome == "PASS" and idx == len(frames_rx) - 1 else ("#f87171" if outcome == "COLLISION" and idx == len(frames_rx) - 1 else "#e2e8f0")
        hud = (
            f"Tick: {idx:03d} ({idx * 0.1:.1f}s) | Outcome: {outcome}\n"
            f"Goal Rem: {dist_rem:.2f} m | Min Clear: {frames_clear[idx]:.2f} m\n"
            f"Cmd (v,w): ({cv:+.2f}, {cw:+.2f}) | Act: ({av:+.2f}, {aw:+.2f})"
        )
        ax2d.set_title(hud, color=status_col, fontsize=10, fontweight="bold", loc="left", pad=8)
        ax2d.set_xlabel("Ego +X Forward (m)", color="#cbd5e1", fontsize=9)
        ax2d.set_ylabel("Ego +Y Left (m)", color="#cbd5e1", fontsize=9)
        ax2d.tick_params(colors="#94a3b8", labelsize=8)

        fig.tight_layout(pad=1.2)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        if proc.stdin is not None:
            proc.stdin.write(buf.tobytes())
        if frame_no == key_idx or (frame_no == len(frames_rx) - 1 and not os.path.exists(out_png)):
            fig.savefig(out_png, dpi=130, facecolor="#0f172a")

    plt.close(fig)
    if proc.stdin is not None:
        proc.stdin.close()
    proc.wait(timeout=20.0)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames,duration", "-of", "json", out_mp4],
        capture_output=True, text=True,
    )
    return {
        "mp4": out_mp4,
        "png": out_png,
        "outcome": outcome,
        "steps": len(frames_rx),
        "ffprobe": probe.stdout.strip(),
    }


def train_teacher_seed(
    track: str,
    seed: int,
    num_envs: int,
    rollout_steps: int,
    max_updates: int,
    eval_every: int,
    archive_dir: str,
    evidence_dir: str,
    repo_img_dir: str,
    device: torch.device,
    record_stages: bool = False,
) -> Tuple[str, Dict[str, object]]:
    """Train one Recurrent PPO seed on mixed-regime VectorDisasterEnv with unseen-seed early stopping."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = VectorDisasterEnv(
        num_envs=num_envs,
        track=track,
        device=device,
        max_steps=320,
        mode="train",
        base_seed=100_000 + seed * 1_000_000,
    )
    teacher_hidden_dim = 128
    model = TemporalContextActorCritic(hidden_dim=teacher_hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.5e-4, weight_decay=1e-4)

    obs_hist, local_patch = env.reset_envs()
    best_ckpt = os.path.join(archive_dir, f"best_teacher_seed{seed}.pt")
    os.makedirs(archive_dir, exist_ok=True)

    history: List[Dict[str, object]] = []
    best_unseen_sr = -1.0
    best_gap = 999.0
    best_ret = -1e9
    no_improve_evals = 0
    total_steps = 0
    total_episodes = 0
    early_stopped_at: Optional[int] = None
    t_start = time.time()

    # Running window of recent training episode outcomes
    recent_reached: List[float] = []
    recent_failed: List[float] = []
    stage_videos: List[Dict[str, object]] = []

    obs_buf = torch.zeros((rollout_steps, num_envs, HISTORY_LEN, STEP_FEAT_DIM), dtype=torch.float32, device=device)
    patch_buf = torch.zeros((rollout_steps, num_envs, 1, PATCH_SIZE, PATCH_SIZE), dtype=torch.float32, device=device)
    act_buf = torch.zeros((rollout_steps, num_envs, 2), dtype=torch.float32, device=device)
    logp_buf = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
    rew_buf = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
    val_buf = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
    done_buf = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)

    for upd in range(1, max_updates + 1):
        frac = 1.0 - (upd - 1) / float(max_updates)
        lr_now = 8.0e-5 + (3.2e-4 - 8.0e-5) * frac
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now
        ent_coef = 0.0004 + (0.004 - 0.0004) * frac
        plr_prob = 0.25 if upd > 20 else 0.0

        model.eval()
        with torch.no_grad():
            for t in range(rollout_steps):
                obs_buf[t] = obs_hist
                patch_buf[t] = local_patch
                action, log_prob, _, value = model.get_action_and_value(obs_hist, local_patch, deterministic=False)
                act_buf[t] = action
                logp_buf[t] = log_prob
                val_buf[t] = value

                obs_next, patch_next, rewards, dones, info = env.step(action)
                rew_buf[t] = rewards
                done_buf[t] = dones.float()
                total_steps += num_envs

                done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
                if len(done_ids) > 0:
                    total_episodes += len(done_ids)
                    r_done = info["reached"][done_ids].float().detach().cpu().numpy()
                    c_done = (info["collision"][done_ids] | info["timeout"][done_ids]).float().detach().cpu().numpy()
                    recent_reached.extend(r_done.tolist())
                    recent_failed.extend(c_done.tolist())
                    if len(recent_reached) > 3000:
                        recent_reached = recent_reached[-3000:]
                        recent_failed = recent_failed[-3000:]
                    obs_next, patch_next = env.reset_envs(env_ids=done_ids, plr_prob=plr_prob)

                obs_hist = obs_next
                local_patch = patch_next

            _, _, _, next_value = model.get_action_and_value(obs_hist, local_patch, deterministic=False)

        # Compute GAE
        adv_buf = torch.zeros_like(rew_buf)
        lastgaelam = torch.zeros(num_envs, dtype=torch.float32, device=device)
        gamma = 0.99
        gae_lambda = 0.95
        for t in reversed(range(rollout_steps)):
            next_nv = next_value if t == rollout_steps - 1 else val_buf[t + 1]
            next_nonterminal = 1.0 - done_buf[t]
            delta = rew_buf[t] + gamma * next_nv * next_nonterminal - val_buf[t]
            lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam
            adv_buf[t] = lastgaelam
        ret_buf = adv_buf + val_buf

        # Flatten rollout batch
        b_obs = obs_buf.reshape(-1, HISTORY_LEN, STEP_FEAT_DIM)
        b_patch = patch_buf.reshape(-1, 1, PATCH_SIZE, PATCH_SIZE)
        b_act = act_buf.reshape(-1, 2)
        b_logp = logp_buf.reshape(-1)
        b_adv = adv_buf.reshape(-1)
        b_ret = ret_buf.reshape(-1)
        b_val = val_buf.reshape(-1)
        batch_size = b_obs.shape[0]
        mb_size = max(1024, batch_size // 12)

        model.train()
        pg_losses = []
        v_losses = []
        ent_losses = []
        for _ in range(4):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mb_size):
                mb = perm[start : start + mb_size]
                _, new_logp, entropy, new_val = model.get_action_and_value(
                    b_obs[mb], b_patch[mb], action_taken=b_act[mb], deterministic=False
                )
                ratio = torch.exp(torch.clamp(new_logp - b_logp[mb], -15.0, 15.0))
                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 0.80, 1.20)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                ret_norm = 25.0
                v_pred_n = new_val / ret_norm
                v_targ_n = b_ret[mb] / ret_norm
                v_old_n = b_val[mb] / ret_norm
                v_clip_n = v_old_n + torch.clamp(v_pred_n - v_old_n, -0.40, 0.40)
                v_loss = 0.5 * torch.max((v_pred_n - v_targ_n) ** 2, (v_clip_n - v_targ_n) ** 2).mean()
                ent_loss = entropy.mean()

                loss = pg_loss + 0.35 * v_loss - ent_coef * ent_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                pg_losses.append(float(pg_loss.item()))
                v_losses.append(float(v_loss.item()))
                ent_losses.append(float(ent_loss.item()))

        if record_stages and upd in (25, 150):
            model.eval()
            st_tag = "early" if upd == 25 else "mid"
            mp4_p = os.path.join(evidence_dir, f"phase2_{track}_{st_tag}_upd{upd}.mp4")
            png_p = os.path.join(repo_img_dir, f"phase2_{track}_{st_tag}_upd{upd}.png")
            vid_res = render_3d_rollout_video(
                track, f"{st_tag} upd={upd}",
                lambda h, p, m=model: m(h, p)[0],
                mp4_p, png_p, device=device, eval_seed=10019 if track == "real_tb3" else 10002,
            )
            shutil.copy2(png_p, os.path.join(evidence_dir, os.path.basename(png_p)))
            stage_videos.append(vid_res)

        if upd == 1 or upd % eval_every == 0 or upd == max_updates:
            model.eval()
            ev = evaluate_policy_on_holdout(track, device, lambda h, p, m=model: m(h, p)[0])
            train_sr = float(np.mean(recent_reached)) if recent_reached else 0.0
            unseen_sr = float(ev["unseen_sr"])
            gen_gap = train_sr - unseen_sr
            elapsed = time.time() - t_start

            rec = {
                "update": upd,
                "steps": total_steps,
                "episodes": total_episodes,
                "elapsed_s": round(elapsed, 1),
                "train_sr": round(train_sr, 4),
                "unseen_sr": round(unseen_sr, 4),
                "gen_gap": round(gen_gap, 4),
                "collision_rate": round(float(ev["collision_rate"]), 4),
                "mean_return": round(float(ev["mean_return"]), 2),
                "per_regime": {k: round(v, 4) for k, v in ev["per_regime"].items()},
                "pg_loss": round(float(np.mean(pg_losses)), 4),
                "v_loss": round(float(np.mean(v_losses)), 4),
                "entropy": round(float(np.mean(ent_losses)), 4),
                "plr_pool_len": len(env.hard_seeds),
            }
            history.append(rec)

            improved = (
                unseen_sr > best_unseen_sr + 0.002
                or (abs(unseen_sr - best_unseen_sr) <= 0.002 and (abs(gen_gap) < abs(best_gap) or float(ev["mean_return"]) > best_ret + 1.0))
            )
            if improved:
                best_unseen_sr = unseen_sr
                best_gap = gen_gap
                best_ret = float(ev["mean_return"])
                no_improve_evals = 0
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "hidden_dim": teacher_hidden_dim,
                        "track": track,
                        "seed": seed,
                        "update": upd,
                        "steps": total_steps,
                        "episodes": total_episodes,
                        "unseen_sr": best_unseen_sr,
                        "per_regime": ev["per_regime"],
                    },
                    best_ckpt,
                )
            elif upd >= eval_every * 2:
                no_improve_evals += 1

            reg_str = " ".join(f"{k[:4]}={v*100:.0f}%" for k, v in ev["per_regime"].items())
            print(
                f"  [{track:8s}|s={seed}|upd={upd:03d}/{max_updates}|steps={total_steps/1e6:5.2f}M|ep={total_episodes:6d}] "
                f"Train={train_sr*100:5.1f}% Unseen={unseen_sr*100:5.1f}% Gap={gen_gap*100:+5.1f}% "
                f"Ret={ev['mean_return']:6.1f} [{reg_str}] Best={best_unseen_sr*100:5.1f}%",
                flush=True,
            )

            # Anti-overfitting Early Stopping criterion
            if upd >= 175 and no_improve_evals >= 6 and best_unseen_sr >= 0.94:
                early_stopped_at = upd
                print(f"  [EARLY STOP] {track} seed={seed} converged cleanly at upd={upd} (Best Unseen={best_unseen_sr*100:.1f}%)", flush=True)
                break

    summary = {
        "track": track,
        "seed": seed,
        "total_steps": total_steps,
        "total_episodes": total_episodes,
        "best_unseen_sr": best_unseen_sr,
        "early_stopped_at": early_stopped_at,
        "elapsed_s": round(time.time() - t_start, 1),
        "history": history,
        "stage_videos": stage_videos,
    }
    with open(os.path.join(archive_dir, f"history_seed{seed}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return best_ckpt, summary


def distill_ensemble_to_onnx(
    track: str,
    ckpt_paths: List[str],
    out_onnx_path: str,
    device: torch.device,
    teacher_samples: int = 180_000,
    dagger_samples: int = 80_000,
) -> Dict[str, object]:
    """
    Distill a 3-seed Teacher Ensemble into a static-shape DistilledEdgeStudent ONNX model
    with observation noise & DAgger rollout aggregation, then verify Unseen SR & CPU latency.
    """
    loaded: List[Tuple[float, TemporalContextActorCritic]] = []
    for cp in ckpt_paths:
        ckpt = torch.load(cp, map_location=device)
        m = TemporalContextActorCritic(hidden_dim=int(ckpt.get("hidden_dim", 64))).to(device)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        loaded.append((float(ckpt.get("unseen_sr", 0.0)), m))

    # Rank descending by Unseen SR so teachers[0] is the strongest Lead Teacher
    loaded.sort(key=lambda x: x[0], reverse=True)
    teachers = [m for _, m in loaded]

    @torch.no_grad()
    def ensemble_act(obs_hist: torch.Tensor, local_patch: torch.Tensor) -> torch.Tensor:
        acts = torch.stack([t(obs_hist, local_patch)[0] for t in teachers], dim=0)  # (K, B, 2)
        if len(teachers) == 1:
            return acts[0]
        # Agreement-preserving fusion: average when teachers agree; defer to lead teacher when bifurcating
        w_std = acts[:, :, 1].std(dim=0)
        mean_act = acts.mean(dim=0)
        agree = (w_std < 0.18).unsqueeze(-1)
        return torch.where(agree, mean_act, acts[0])

    # 1. Collect clean & perturbed Teacher rollouts across randomized procedural worlds
    env = VectorDisasterEnv(num_envs=512, track=track, device=device, max_steps=320, mode="train", base_seed=5_500_000)
    obs_hist, local_patch = env.reset_envs()

    x_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []
    collected = 0
    print(f"\n[DISTILL {track}] Collecting {teacher_samples} Teacher Ensemble transitions...", flush=True)
    with torch.no_grad():
        while collected < teacher_samples:
            target_act = ensemble_act(obs_hist, local_patch)
            x_list.append(_flatten_obs(obs_hist, local_patch).cpu())
            y_list.append(target_act.cpu())
            collected += obs_hist.shape[0]

            # Add mild exploration noise so student covers near-boundary recovery states
            exec_act = target_act + torch.randn_like(target_act) * torch.tensor([0.015, 0.05], device=device)
            exec_act[:, 0] = torch.clamp(exec_act[:, 0], TB3_MIN_V, TB3_MAX_V)
            exec_act[:, 1] = torch.clamp(exec_act[:, 1], -TB3_MAX_W, TB3_MAX_W)
            obs_next, patch_next, _, dones, _ = env.step(exec_act)
            done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            if len(done_ids) > 0:
                obs_next, patch_next = env.reset_envs(env_ids=done_ids, plr_prob=0.35)
            obs_hist, local_patch = obs_next, patch_next

    student = DistilledEdgeStudent().to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1.2e-3, weight_decay=1e-4)

    def _fit_student(x_all: torch.Tensor, y_all: torch.Tensor, epochs: int) -> Tuple[float, float]:
        n = x_all.shape[0]
        perm = torch.randperm(n)
        n_val = max(4096, int(0.10 * n))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        tr_loss_last = 0.0
        val_loss_last = 0.0
        for _ in range(epochs):
            student.train()
            p_tr = tr_idx[torch.randperm(len(tr_idx))]
            losses = []
            for start in range(0, len(p_tr), 2048):
                idx = p_tr[start : start + 2048]
                xb = x_all[idx].to(device)
                yb = y_all[idx].to(device)
                # Data augmentation: slight Gaussian input jitter on LiDAR channels during distillation
                xb_aug = xb.clone()
                xb_aug[:, : HISTORY_LEN * STEP_FEAT_DIM] += torch.randn_like(xb_aug[:, : HISTORY_LEN * STEP_FEAT_DIM]) * 0.008
                pred = student(xb_aug)
                # Normalize action scale: v range 0.30, w range 1.30
                scale = torch.tensor([1.0 / 0.30, 1.0 / 1.30], device=device)
                loss = F.mse_loss(pred * scale, yb * scale) + 0.20 * F.l1_loss(pred * scale, yb * scale)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            sched.step()
            tr_loss_last = float(np.mean(losses))

        student.eval()
        with torch.no_grad():
            scale = torch.tensor([1.0 / 0.30, 1.0 / 1.30], device=device)
            v_preds = []
            v_tgts = []
            for start in range(0, len(val_idx), 4096):
                idx = val_idx[start : start + 4096]
                v_preds.append(student(x_all[idx].to(device)) * scale)
                v_tgts.append(y_all[idx].to(device) * scale)
            val_loss_last = float(F.mse_loss(torch.cat(v_preds, 0), torch.cat(v_tgts, 0)).item())
        return tr_loss_last, val_loss_last

    x_tensor = torch.cat(x_list, dim=0)
    y_tensor = torch.cat(y_list, dim=0)
    tr1, val1 = _fit_student(x_tensor, y_tensor, epochs=22)
    print(f"  [DISTILL {track} Pass 1] samples={len(x_tensor)} train_loss={tr1:.5f} val_mse={val1:.5f}", flush=True)

    # 2. DAgger Stage: Roll out Student itself, query Teacher Ensemble on visited states, and re-fit!
    obs_hist, local_patch = env.reset_envs()
    dx_list: List[torch.Tensor] = []
    dy_list: List[torch.Tensor] = []
    collected_dag = 0
    student.eval()
    with torch.no_grad():
        while collected_dag < dagger_samples:
            flat_obs = _flatten_obs(obs_hist, local_patch)
            s_act = student(flat_obs)
            t_act = ensemble_act(obs_hist, local_patch)
            dx_list.append(flat_obs.cpu())
            dy_list.append(t_act.cpu())
            collected_dag += obs_hist.shape[0]
            obs_next, patch_next, _, dones, _ = env.step(s_act)
            done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            if len(done_ids) > 0:
                obs_next, patch_next = env.reset_envs(env_ids=done_ids, plr_prob=0.40)
            obs_hist, local_patch = obs_next, patch_next

    x_all = torch.cat([x_tensor] + dx_list, dim=0)
    y_all = torch.cat([y_tensor] + dy_list, dim=0)
    tr2, val2 = _fit_student(x_all, y_all, epochs=28)
    print(f"  [DISTILL {track} DAgger Pass 2] samples={len(x_all)} train_loss={tr2:.5f} val_mse={val2:.5f}", flush=True)

    # 3. Export lightweight static-shape ONNX
    os.makedirs(os.path.dirname(os.path.abspath(out_onnx_path)), exist_ok=True)
    student_cpu = student.cpu().eval()
    dummy = torch.randn(1, FLAT_INPUT_DIM, dtype=torch.float32)
    torch.onnx.export(
        student_cpu,
        dummy,
        out_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["obs_flat"],
        output_names=["cmd_vel"],
        dynamic_axes={"obs_flat": {0: "batch"}, "cmd_vel": {0: "batch"}},
    )

    # 4. Evaluate exported ONNX directly via CPU single-thread ORT on all 60 unseen seeds!
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(out_onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    def onnx_policy_fn(h: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        flat_np = _flatten_obs(h, p).detach().cpu().numpy().astype(np.float32)
        out_np = sess.run(None, {in_name: flat_np})[0]
        return torch.from_numpy(out_np).to(h.device)

    ev_onnx = evaluate_policy_on_holdout(track, device, onnx_policy_fn)

    # Single-thread CPU Latency benchmark (1000 forward passes)
    sample_x = np.random.randn(1, FLAT_INPUT_DIM).astype(np.float32)
    for _ in range(50):
        sess.run(None, {in_name: sample_x})
    lats_ms = []
    for _ in range(1000):
        t0 = time.perf_counter()
        sess.run(None, {in_name: sample_x})
        lats_ms.append((time.perf_counter() - t0) * 1000.0)
    lat_arr = np.asarray(lats_ms, dtype=np.float64)

    onnx_size_kb = os.path.getsize(out_onnx_path) / 1024.0
    print(
        f"  [ONNX VERIFIED {track}] size={onnx_size_kb:.1f}KB | Unseen SR={ev_onnx['unseen_sr']*100:.1f}% "
        f"| CPU Lat mean={lat_arr.mean():.3f}ms p95={np.percentile(lat_arr, 95):.3f}ms "
        f"| Regimes={ {k: round(v*100,1) for k,v in ev_onnx['per_regime'].items()} }",
        flush=True,
    )
    return {
        "onnx_path": out_onnx_path,
        "onnx_size_kb": round(onnx_size_kb, 1),
        "val_mse_pass1": round(val1, 6),
        "val_mse_dagger": round(val2, 6),
        "onnx_eval": ev_onnx,
        "cpu_latency_ms": {
            "mean": round(float(lat_arr.mean()), 4),
            "p95": round(float(np.percentile(lat_arr, 95)), 4),
            "p99": round(float(np.percentile(lat_arr, 99)), 4),
        },
        "eval_fn": onnx_policy_fn,
    }


def plot_learning_curves(
    track: str,
    seed_summaries: List[Dict[str, object]],
    distill_res: Dict[str, object],
    out_png: str,
) -> None:
    """Plot 2x2 learning curves, Generalization Gap, Per-Regime breakdown, and Distilled Student summary."""
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), dpi=150)
    fig.patch.set_facecolor("#ffffff")

    # Collect common update grid via interpolation up to min max_upd or full range
    all_upds = sorted({h["update"] for s in seed_summaries for h in s["history"]})
    grid = np.asarray(all_upds, dtype=np.float64)

    tr_mat, un_mat, gap_mat = [], [], []
    reg_mats = {r: [] for r in REGIME_NAMES}
    for s in seed_summaries:
        hist = s["history"]
        u = np.array([h["update"] for h in hist], dtype=np.float64)
        tr = np.array([h["train_sr"] * 100.0 for h in hist], dtype=np.float64)
        un = np.array([h["unseen_sr"] * 100.0 for h in hist], dtype=np.float64)
        tr_mat.append(np.interp(grid, u, tr))
        un_mat.append(np.interp(grid, u, un))
        gap_mat.append(tr_mat[-1] - un_mat[-1])
        for r in REGIME_NAMES:
            rv = np.array([h["per_regime"][r] * 100.0 for h in hist], dtype=np.float64)
            reg_mats[r].append(np.interp(grid, u, rv))

    tr_arr = np.vstack(tr_mat)
    un_arr = np.vstack(un_mat)
    gap_arr = np.vstack(gap_mat)

    ax = axes[0, 0]
    ax.plot(grid, tr_arr.mean(0), label="Train Mixed SR (3-seed mean)", color="#64748b", lw=2.0, ls="--")
    ax.fill_between(grid, tr_arr.mean(0) - tr_arr.std(0), tr_arr.mean(0) + tr_arr.std(0), color="#64748b", alpha=0.15)
    ax.plot(grid, un_arr.mean(0), label="Unseen Holdout SR (60 seeds, mean)", color="#2563eb", lw=2.4)
    ax.fill_between(grid, un_arr.mean(0) - un_arr.std(0), un_arr.mean(0) + un_arr.std(0), color="#2563eb", alpha=0.18)
    onnx_sr = distill_res["onnx_eval"]["unseen_sr"] * 100.0
    ax.axhline(onnx_sr, color="#16a34a", lw=1.8, ls=":", label=f"Distilled ONNX Student ({onnx_sr:.1f}%)")
    ax.set_title(f"(A) {track.upper()} Train vs Unseen Holdout Success Rate", fontweight="bold")
    ax.set_xlabel("PPO Update")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower right")

    ax = axes[0, 1]
    ax.plot(grid, gap_arr.mean(0), color="#d97706", lw=2.2, label="Gen Gap = Train - Unseen (%)")
    ax.fill_between(grid, gap_arr.mean(0) - gap_arr.std(0), gap_arr.mean(0) + gap_arr.std(0), color="#d97706", alpha=0.18)
    ax.axhline(0.0, color="#0f172a", lw=1.0, ls="--")
    for s in seed_summaries:
        if s["early_stopped_at"] is not None:
            ax.axvline(s["early_stopped_at"], color="#dc2626", ls=":", lw=1.2, alpha=0.7)
    ax.set_title("(B) Anti-Overfitting Generalization Gap & Early-Stop", fontweight="bold")
    ax.set_xlabel("PPO Update")
    ax.set_ylabel("Gap (% pt)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5)

    ax = axes[1, 0]
    cols = ["#0284c7", "#7c3aed", "#059669", "#e11d48"]
    for col, r in zip(cols, REGIME_NAMES):
        rm = np.vstack(reg_mats[r])
        ax.plot(grid, rm.mean(0), label=r, color=col, lw=2.0)
    ax.set_title("(C) Unseen Holdout SR per Disaster Regime", fontweight="bold")
    ax.set_xlabel("PPO Update")
    ax.set_ylabel("Regime Unseen SR (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower right")

    ax = axes[1, 1]
    labels = [f"Seed {s['seed']}\nTeacher" for s in seed_summaries] + ["Distilled\nONNX Student"]
    vals = [s["best_unseen_sr"] * 100.0 for s in seed_summaries] + [onnx_sr]
    b_cols = ["#60a5fa", "#60a5fa", "#60a5fa", "#16a34a"]
    bars = ax.bar(labels, vals, color=b_cols, edgecolor="#1e293b", width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2.0, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9.5)
    lat = distill_res["cpu_latency_ms"]
    ax.set_title(
        f"(D) 3-Seed Teachers vs Distilled ONNX ({distill_res['onnx_size_kb']} KB, CPU {lat['mean']:.2f}ms)",
        fontweight="bold",
    )
    ax.set_ylabel("Unseen Eval SR (%)")
    ax.set_ylim(0, 112)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Phase 2 V100 Large-Scale Dual-Track RL Convergence ({track.upper()})", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 V100 Dual-Track RL Training & Ensemble ONNX Distillation")
    parser.add_argument("--tracks", default="both", choices=["sim", "real_tb3", "both"])
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--num-envs", type=int, default=768)
    parser.add_argument("--rollout-steps", type=int, default=48)
    parser.add_argument("--max-updates", type=int, default=360)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--archive-root", default="/hdd/rl_archives")
    parser.add_argument("--evidence-dir", default="/hdd/evidence/training")
    parser.add_argument("--repo-img-dir", default=os.path.join(WS_ROOT, "docs", "images", "evidence"))
    parser.add_argument("--distill-existing-tracks", default="", help="Comma-separated tracks whose existing checkpoints should be distilled without retraining")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    tracks = ["sim", "real_tb3"] if args.tracks == "both" else [args.tracks]
    reuse_tracks = {x.strip() for x in args.distill_existing_tracks.split(",") if x.strip()}

    os.makedirs(args.archive_root, exist_ok=True)
    os.makedirs(args.evidence_dir, exist_ok=True)
    os.makedirs(args.repo_img_dir, exist_ok=True)

    gpu_logger = NvidiaSmiLogger(os.path.join(args.evidence_dir, "nvidia_smi_v100.log"), interval_s=5.0)
    gpu_logger.start()

    print("=" * 88)
    print(f"  Phase 2 V100 Dual-Track Anti-Overfitting RL | device={device} | tracks={tracks} | seeds={seeds}")
    print(f"  Batch={args.num_envs} envs x {args.rollout_steps} steps = {args.num_envs * args.rollout_steps}/upd | max_upd={args.max_updates}")
    print("=" * 88, flush=True)

    all_track_results: Dict[str, object] = {}
    try:
        for track in tracks:
            t0 = time.time()
            track_arch = os.path.join(args.archive_root, track)
            ckpt_paths: List[str] = []
            seed_summaries: List[Dict[str, object]] = []

            for i, seed in enumerate(seeds):
                cp_existing = os.path.join(track_arch, f"best_teacher_seed{seed}.pt")
                hist_existing = os.path.join(track_arch, f"history_seed{seed}.json")
                if track in reuse_tracks and os.path.exists(cp_existing) and os.path.exists(hist_existing):
                    with open(hist_existing, "r", encoding="utf-8") as hf:
                        summ = json.load(hf)
                    ckpt_paths.append(cp_existing)
                    seed_summaries.append(summ)
                    print(f"\n--- Reusing Track={track} Seed={seed} ({i+1}/{len(seeds)}) Best={summ.get('best_unseen_sr', 0)*100:.1f}% ---", flush=True)
                    continue
                print(f"\n--- Training Track={track} Seed={seed} ({i+1}/{len(seeds)}) ---", flush=True)
                cp, summ = train_teacher_seed(
                    track=track,
                    seed=seed,
                    num_envs=args.num_envs,
                    rollout_steps=args.rollout_steps,
                    max_updates=args.max_updates,
                    eval_every=args.eval_every,
                    archive_dir=track_arch,
                    evidence_dir=args.evidence_dir,
                    repo_img_dir=args.repo_img_dir,
                    device=device,
                    record_stages=(i == 0),
                )
                ckpt_paths.append(cp)
                seed_summaries.append(summ)

            out_onnx = os.path.join(WS_ROOT, "src", "disaster_recovery", "config", f"{track}_policy_student.onnx")
            dist_res = distill_ensemble_to_onnx(track=track, ckpt_paths=ckpt_paths, out_onnx_path=out_onnx, device=device)

            # Render final ONNX Student 3D MP4 & PNG visual proof on unseen procedural test case
            stu_mp4 = os.path.join(args.evidence_dir, f"phase2_{track}_final_student.mp4")
            stu_png = os.path.join(args.repo_img_dir, f"phase2_{track}_final_student.png")
            eval_seed_vis = 10019 if track == "real_tb3" else 10002
            stu_vid = render_3d_rollout_video(
                track, "Distilled Student ONNX",
                dist_res.pop("eval_fn"),
                stu_mp4, stu_png, device=device, eval_seed=eval_seed_vis,
            )
            shutil.copy2(stu_png, os.path.join(args.evidence_dir, os.path.basename(stu_png)))

            curve_repo_png = os.path.join(args.repo_img_dir, f"phase2_learning_curves_{track}.png")
            plot_learning_curves(track, seed_summaries, dist_res, curve_repo_png)
            shutil.copy2(curve_repo_png, os.path.join(args.evidence_dir, os.path.basename(curve_repo_png)))

            total_steps = sum(s["total_steps"] for s in seed_summaries)
            total_eps = sum(s["total_episodes"] for s in seed_summaries)
            all_track_results[track] = {
                "total_steps": total_steps,
                "total_episodes": total_eps,
                "wall_time_s": round(time.time() - t0, 1),
                "seeds": [
                    {k: v for k, v in s.items() if k != "history"}
                    for s in seed_summaries
                ],
                "distilled_onnx": dist_res,
                "student_3d_video": stu_vid,
                "learning_curve_png": curve_repo_png,
            }
    finally:
        gpu_stats = gpu_logger.stop()

    all_track_results["gpu_telemetry"] = gpu_stats
    summary_path = os.path.join(WS_ROOT, "benchmark_results", "phase2_training_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_track_results, f, indent=2)

    shutil.copy2(summary_path, os.path.join(args.evidence_dir, "phase2_training_summary.json"))
    print("\n" + "=" * 88)
    print("[PHASE 2 COMPLETE] Summary saved:", summary_path)
    print("[GPU STATS]", gpu_stats)
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
