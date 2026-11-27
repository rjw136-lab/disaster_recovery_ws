#!/usr/bin/env python3
"""
=================================================================================================
Project: Disaster Recovery TurtleBot3 Burger System
Script : train_td3.py (TD3 Deep Reinforcement Learning Local Avoidance Trainer & Benchmark)
Author : Autonomous Systems Laboratory / Jong Woo Ro
Usage  : python3 scripts/host/train_td3.py [--episodes 300] [--benchmark]
=================================================================================================
Trains the Residual Twin Delayed DDPG (TD3+BC) local reactive avoidance actor-critic in
normalized [-1, 1] action space on a realistic multi-obstacle corridor environment (1~3 variable-
radius obstacles, narrow corridors, [-0.6, 0.6] rad initial heading error, acceleration limits,
and 1-step control delay), verifies PyTorch-ONNX numerical equivalence, performs a 3-way ablation
study (Untrained Prior Only vs Trained TD3 PyTorch vs Trained TD3 ONNX), and benchmarks edge
inference latency.
=================================================================================================
"""

import os
import sys
import time
import math
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

ws_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/disaster_recovery"))
if ws_path not in sys.path:
    sys.path.insert(0, ws_path)

from disaster_recovery.td3_policy import (
    TD3Actor,
    TD3Critic,
    extract_state_vector,
    compute_expert_avoidance_action,
    TD3LocalAvoidancePolicy,
    normalize_angle
)


# -----------------------------------------------------------------------------------------------
# Action Space Normalization Helpers: Physical [v, w] <-> Normalized [-1, 1]^2
# -----------------------------------------------------------------------------------------------
DEFAULT_MIN_V = 0.03
DEFAULT_MAX_V = 0.15
DEFAULT_MAX_W = 0.60


def normalize_action(action, min_v=DEFAULT_MIN_V, max_v=DEFAULT_MAX_V, max_w=DEFAULT_MAX_W):
    """
    Maps physical action [v in (min_v, max_v), w in (-max_w, max_w)] to normalized [-1, 1]^2.
    Supports both torch.Tensor and np.ndarray / sequence.
    """
    v_mid = 0.5 * (max_v + min_v)
    v_half = 0.5 * (max_v - min_v)
    if isinstance(action, torch.Tensor):
        v_norm = (action[..., 0:1] - v_mid) / v_half
        w_norm = action[..., 1:2] / max_w
        return torch.clamp(torch.cat([v_norm, w_norm], dim=-1), -1.0, 1.0)
    else:
        arr = np.asarray(action, dtype=np.float32)
        out = np.empty_like(arr, dtype=np.float32)
        out[..., 0] = (arr[..., 0] - v_mid) / v_half
        out[..., 1] = arr[..., 1] / max_w
        return np.clip(out, -1.0, 1.0)


def denormalize_action(action_norm, min_v=DEFAULT_MIN_V, max_v=DEFAULT_MAX_V, max_w=DEFAULT_MAX_W):
    """
    Maps normalized action [-1, 1]^2 back to physical [v in (min_v, max_v), w in (-max_w, max_w)].
    Supports both torch.Tensor and np.ndarray / sequence.
    """
    v_mid = 0.5 * (max_v + min_v)
    v_half = 0.5 * (max_v - min_v)
    if isinstance(action_norm, torch.Tensor):
        norm_c = torch.clamp(action_norm, -1.0, 1.0)
        v = v_mid + norm_c[..., 0:1] * v_half
        w = norm_c[..., 1:2] * max_w
        return torch.cat([v, w], dim=-1)
    else:
        arr = np.clip(np.asarray(action_norm, dtype=np.float32), -1.0, 1.0)
        out = np.empty_like(arr, dtype=np.float32)
        out[..., 0] = v_mid + arr[..., 0] * v_half
        out[..., 1] = arr[..., 1] * max_w
        return out


# -----------------------------------------------------------------------------------------------
# Realistic Simulation Environment for DRL Training & Ablation Evaluation
# -----------------------------------------------------------------------------------------------
class ObstacleAvoidanceEnv:
    """
    Kinematic 2D differential drive environment modeling TurtleBot3 Burger with 360-deg LiDAR.
    Includes:
      - Explicit robot_radius (0.105m) + safety_margin (0.015m) collision boundary
      - 1~3 multi-obstacles with variable radius [0.10, 0.25] m
      - Narrow corridor widths and wide initial heading error [-0.6, 0.6] rad
      - Per-step acceleration limits (max_dv, max_dw) and 1-step control delay option
    """
    def __init__(
        self,
        max_v=DEFAULT_MAX_V,
        min_v=DEFAULT_MIN_V,
        max_w=DEFAULT_MAX_W,
        dt=0.10,
        max_dv=0.045,
        max_dw=0.28,
        use_control_delay=True,
        difficulty="normal"
    ):
        self.max_v = max_v
        self.min_v = min_v
        self.max_w = max_w
        self.dt = dt
        self.max_dv = max_dv
        self.max_dw = max_dw
        self.use_control_delay = use_control_delay
        self.difficulty = difficulty

        self.robot_radius = 0.105   # TB3 Burger physical radius (m)
        self.safety_margin = 0.015  # Minimum safety buffer from hull to surface (m)
        self.danger_radius = self.robot_radius + self.safety_margin  # 0.120 m from robot center
        self.max_steps = 240

        self.obstacles = []
        self.obstacle = (0.8, 0.0, 0.18)
        self.reset()

    def reset(self, difficulty=None):
        mode = difficulty if difficulty is not None else self.difficulty
        self.robot_x = 0.0
        self.robot_y = random.uniform(-0.10, 0.10)
        self.robot_yaw = random.uniform(-0.60, 0.60)
        self.last_v = 0.08
        self.last_w = 0.0
        self.delayed_cmd = (self.last_v, self.last_w)
        self.step_count = 0

        if mode == "unseen_hard":
            self.target_dist = random.uniform(1.45, 2.00)
            self.target_x = self.target_dist
            self.target_y = random.uniform(-0.28, 0.28)
            half_width = random.uniform(0.56, 0.80)
            num_obs = random.choice([2, 3])
            r_min, r_max = 0.14, 0.25
        else:
            self.target_dist = random.uniform(1.30, 1.90)
            self.target_x = self.target_dist
            self.target_y = random.uniform(-0.26, 0.26)
            half_width = random.uniform(0.60, 1.10)
            num_obs = random.choices([1, 2, 3], weights=[0.30, 0.45, 0.25], k=1)[0]
            r_min, r_max = 0.10, 0.25

        self.wall_y_top = half_width + random.uniform(-0.04, 0.04)
        self.wall_y_bot = -half_width + random.uniform(-0.04, 0.04)
        self.target_y = float(np.clip(self.target_y, self.wall_y_bot + 0.30, self.wall_y_top - 0.30))

        # Place 1~3 obstacles with guaranteed navigable corridor clearance
        self.obstacles = []
        x_slots = np.linspace(0.52, self.target_dist - 0.46, num_obs)
        prev_side = random.choice([-1, 1])

        for idx in range(num_obs):
            r_obs = random.uniform(r_min, r_max)
            ox = float(x_slots[idx] + random.uniform(-0.06, 0.06))
            ox = float(np.clip(ox, 0.48, self.target_dist - 0.44))

            # Line of sight y at ox
            los_y = self.robot_y + (self.target_y - self.robot_y) * (ox / max(self.target_dist, 1e-3))

            if idx == 0:
                # Primary obstacle near line-of-sight blocking direct path
                oy = los_y + random.uniform(-0.14, 0.14)
            else:
                # Subsequent obstacles staggered laterally to create slalom / gate
                prev_side = -prev_side
                oy = los_y + prev_side * random.uniform(0.16, 0.34)

            # Ensure at least one side between obstacle and corridor wall has >= 0.44m passage
            min_pass = 2.0 * self.danger_radius + 0.20  # 0.44m center-Pass clearance
            top_gap = (self.wall_y_top - (oy + r_obs))
            bot_gap = ((oy - r_obs) - self.wall_y_bot)
            if top_gap < min_pass and bot_gap < min_pass:
                if abs(self.wall_y_top - oy) >= abs(oy - self.wall_y_bot):
                    oy = self.wall_y_top - min_pass - r_obs
                else:
                    oy = self.wall_y_bot + min_pass + r_obs

            oy = float(np.clip(oy, self.wall_y_bot + r_obs + 0.08, self.wall_y_top - r_obs - 0.08))

            # Ensure non-overlapping passage with previously placed obstacles
            valid_obs = True
            for px, py, pr in self.obstacles:
                c_dist = math.hypot(ox - px, oy - py)
                min_c_dist = r_obs + pr + 2.0 * self.danger_radius + 0.16
                if c_dist < min_c_dist:
                    # Shift laterally away from previous obstacle
                    direction = 1.0 if oy >= py else -1.0
                    dy_req = math.sqrt(max(0.0, min_c_dist ** 2 - (ox - px) ** 2))
                    oy_cand = py + direction * dy_req
                    if (self.wall_y_bot + r_obs + 0.08) <= oy_cand <= (self.wall_y_top - r_obs - 0.08):
                        oy = oy_cand
                    else:
                        oy_cand2 = py - direction * dy_req
                        if (self.wall_y_bot + r_obs + 0.08) <= oy_cand2 <= (self.wall_y_top - r_obs - 0.08):
                            oy = oy_cand2
                        else:
                            valid_obs = False
                            break

            # Ensure start and goal are not inside obstacle + buffer
            if math.hypot(ox - self.robot_x, oy - self.robot_y) < (r_obs + self.danger_radius + 0.20):
                valid_obs = False
            if math.hypot(ox - self.target_x, oy - self.target_y) < (r_obs + self.danger_radius + 0.20):
                valid_obs = False

            if valid_obs:
                self.obstacles.append((ox, oy, r_obs))

        if len(self.obstacles) == 0:
            r_obs = 0.16
            ox = 0.5 * self.target_dist
            oy = 0.0
            self.obstacles.append((ox, oy, r_obs))

        self.obstacle = self.obstacles[0]
        self.prev_dist = math.hypot(self.target_x - self.robot_x, self.target_y - self.robot_y)
        return self._get_obs()

    def _simulate_lidar(self, num_rays=360, noisy=True):
        """Vectorized 360-ray 2D LiDAR raycast against all cylinder obstacles and corridor walls."""
        ranges = np.full(num_rays, 3.5, dtype=np.float32)
        angles = self.robot_yaw + np.linspace(0.0, 2.0 * math.pi, num_rays, endpoint=False)

        cos_a = np.cos(angles)
        sin_a = np.sin(angles)

        # Ray-circle intersection across all 1~3 obstacles
        for ox, oy, r_obs in self.obstacles:
            dx = ox - self.robot_x
            dy = oy - self.robot_y

            b = dx * cos_a + dy * sin_a
            c = dx * dx + dy * dy - r_obs * r_obs
            disc = b * b - c

            valid = (disc >= 0.0) & (b > 0.0)
            if np.any(valid):
                sqrt_disc = np.sqrt(np.maximum(disc[valid], 0.0))
                dist_circle = b[valid] - sqrt_disc
                ranges[valid] = np.where(
                    dist_circle > 0.05,
                    np.minimum(ranges[valid], dist_circle),
                    ranges[valid]
                )

        # Vectorized Ray-Wall intersection (y = wall_y_top and y = wall_y_bot)
        up_mask = sin_a > 1e-3
        d_top = (self.wall_y_top - self.robot_y) / np.where(up_mask, sin_a, 1.0)
        ranges = np.where(up_mask & (d_top > 0.0), np.minimum(ranges, d_top), ranges)

        dn_mask = sin_a < -1e-3
        d_bot = (self.wall_y_bot - self.robot_y) / np.where(dn_mask, sin_a, -1.0)
        ranges = np.where(dn_mask & (d_bot > 0.0), np.minimum(ranges, d_bot), ranges)

        if noisy:
            ranges += np.random.normal(0.0, 0.01, size=num_rays).astype(np.float32)
        return np.clip(ranges, 0.08, 3.5)

    def _get_obs(self, noisy=True):
        ranges = self._simulate_lidar(noisy=noisy)
        return extract_state_vector(
            ranges,
            self.robot_x, self.robot_y, self.robot_yaw,
            self.target_x, self.target_y,
            self.last_v, self.last_w,
            max_v=self.max_v, max_w=self.max_w
        )

    def step(self, action, noisy=True):
        self.step_count += 1
        cmd_v = float(np.clip(action[0], self.min_v, self.max_v))
        cmd_w = float(np.clip(action[1], -self.max_w, self.max_w))

        # 1-Step Control Delay Option
        if self.use_control_delay:
            target_v, target_w = self.delayed_cmd
            self.delayed_cmd = (cmd_v, cmd_w)
        else:
            target_v, target_w = cmd_v, cmd_w

        # Acceleration / Deceleration Rate Limits (max_dv, max_dw)
        v = float(np.clip(target_v, self.last_v - self.max_dv, self.last_v + self.max_dv))
        w = float(np.clip(target_w, self.last_w - self.max_dw, self.last_w + self.max_dw))
        v = float(np.clip(v, self.min_v, self.max_v))
        w = float(np.clip(w, -self.max_w, self.max_w))

        self.robot_x += v * math.cos(self.robot_yaw) * self.dt
        self.robot_y += v * math.sin(self.robot_yaw) * self.dt
        self.robot_yaw = normalize_angle(self.robot_yaw + w * self.dt)

        cur_dist = math.hypot(self.target_x - self.robot_x, self.target_y - self.robot_y)

        # Minimum distance from robot center to any obstacle surface & hull clearance
        dist_to_obs_surface = min(
            math.hypot(ox - self.robot_x, oy - self.robot_y) - r_obs
            for ox, oy, r_obs in self.obstacles
        )
        hull_to_obs_clearance = dist_to_obs_surface - self.robot_radius
        wall_top_clearance = (self.wall_y_top - self.robot_y) - self.robot_radius
        wall_bot_clearance = (self.robot_y - self.wall_y_bot) - self.robot_radius

        # Collision occurs when robot hull (robot_radius=0.105m) breaches safety_margin (0.015m)
        collision = (
            (hull_to_obs_clearance < self.safety_margin)
            or (wall_top_clearance < self.safety_margin)
            or (wall_bot_clearance < self.safety_margin)
        )

        reached = (cur_dist < 0.16) and (not collision)

        # Reward formulation: progress + clearance margin + smoothness + terminal bonus/penalty
        progress = (self.prev_dist - cur_dist) * 10.0
        self.prev_dist = cur_dist

        min_surface_dist = min(
            dist_to_obs_surface,
            self.wall_y_top - self.robot_y,
            self.robot_y - self.wall_y_bot
        )
        clearance_penalty = 0.0
        if min_surface_dist < 0.30:
            clearance_penalty = (min_surface_dist - 0.30) * 5.0

        smoothness_penalty = -0.025 * abs(w) - 0.04 * abs(w - self.last_w) - 0.02
        reward = progress + clearance_penalty + smoothness_penalty

        done = False
        if reached:
            reward += 30.0
            done = True
        elif collision:
            reward -= 30.0
            done = True
        elif self.step_count >= self.max_steps:
            reward -= 5.0
            done = True

        dw = abs(w - self.last_w)
        self.last_v = v
        self.last_w = w

        return self._get_obs(noisy=noisy), reward, done, {
            'reached': reached,
            'collision': collision,
            'dw': dw,
            'min_obs_dist': dist_to_obs_surface,
            'hull_clearance': hull_to_obs_clearance,
            'executed_cmd': np.array([target_v, target_w], dtype=np.float32),
            'executed_vw': np.array([v, w], dtype=np.float32),
        }


# -----------------------------------------------------------------------------------------------
# Replay Buffer with Causal Action Storage & Selective Expert BC Masking
# -----------------------------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=60000, state_dim=28, action_dim=2):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.is_expert = np.zeros((capacity, 1), dtype=np.float32)
        self.expert_actions = np.zeros((capacity, action_dim), dtype=np.float32)

    def add(self, state, action, reward, next_state, done, is_expert=0.0, expert_action=None):
        """
        Stores strictly the executed `action` passed to env.step(action) in self.actions
        to preserve Markov transition causality, while storing `is_expert` and `expert_action`
        separately for selective BC regularization on successful expert demonstrations.
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        self.is_expert[self.ptr] = float(is_expert)
        self.expert_actions[self.ptr] = action if expert_action is None else expert_action

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size=128):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.next_states[idx]),
            torch.from_numpy(self.dones[idx]),
            torch.from_numpy(self.is_expert[idx]),
            torch.from_numpy(self.expert_actions[idx]),
        )


# -----------------------------------------------------------------------------------------------
# TD3+BC Trainer (Normalized [-1, 1] Action Space + Decaying Selective BC)
# -----------------------------------------------------------------------------------------------
class TD3Trainer:
    def __init__(
        self,
        state_dim=28,
        action_dim=2,
        min_v=DEFAULT_MIN_V,
        max_v=DEFAULT_MAX_V,
        max_w=DEFAULT_MAX_W,
        alpha=2.5,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        self.min_v = min_v
        self.max_v = max_v
        self.max_w = max_w
        self.alpha = alpha
        print(f"[TD3] Training device: {self.device} | TD3+BC alpha: {self.alpha}")

        self.actor = TD3Actor(state_dim, action_dim, max_v=max_v, min_v=min_v, max_w=max_w).to(self.device)
        self.actor_target = TD3Actor(state_dim, action_dim, max_v=max_v, min_v=min_v, max_w=max_w).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = TD3Critic(state_dim, action_dim).to(self.device)
        self.critic_target = TD3Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.tau = 0.005
        # Target policy smoothing parameters in normalized [-1, 1] action space
        self.policy_noise = 0.15
        self.noise_clip = 0.35
        self.policy_freq = 2
        self.total_it = 0

    def select_action(self, state, noise_norm=0.10):
        """
        Infers physical action from actor and applies isotropic exploration noise
        in [-1, 1] normalized action space before denormalizing back to [v, w].
        """
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_phys = self.actor(state_tensor).cpu().numpy()[0]
        if noise_norm > 0.0:
            act_norm = normalize_action(action_phys, self.min_v, self.max_v, self.max_w)
            act_norm = np.clip(
                act_norm + np.random.normal(0.0, noise_norm, size=2).astype(np.float32),
                -1.0,
                1.0
            )
            action_phys = denormalize_action(act_norm, self.min_v, self.max_v, self.max_w)
        return action_phys

    def train_step(self, replay_buffer, batch_size=128, bc_weight=0.5):
        self.total_it += 1

        states, actions, rewards, next_states, dones, is_expert, expert_actions = replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        is_expert = is_expert.to(self.device)
        expert_actions = expert_actions.to(self.device)

        # Normalize executed actions and expert reference actions to [-1, 1]^2
        actions_norm = normalize_action(actions, self.min_v, self.max_v, self.max_w)
        expert_actions_norm = normalize_action(expert_actions, self.min_v, self.max_v, self.max_w)

        with torch.no_grad():
            # Target policy smoothing entirely in [-1, 1] normalized space
            next_phys = self.actor_target(next_states)
            next_norm = normalize_action(next_phys, self.min_v, self.max_v, self.max_w)
            noise = (torch.randn_like(next_norm) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_norm = (next_norm + noise).clamp(-1.0, 1.0)

            target_q1, target_q2 = self.critic_target(next_states, next_norm)
            target_q = torch.min(target_q1, target_q2)
            target_y = rewards + ((1.0 - dones) * self.gamma * target_q)

        current_q1, current_q2 = self.critic(states, actions_norm)
        critic_loss = nn.MSELoss()(current_q1, target_y) + nn.MSELoss()(current_q2, target_y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            pred_phys = self.actor(states)
            pred_norm = normalize_action(pred_phys, self.min_v, self.max_v, self.max_w)

            q_val = self.critic.q1(states, pred_norm)
            # Fujimoto & Gu (2021) TD3+BC scale normalization: lmbda = alpha / mean(|Q|)
            lmbda = self.alpha / (q_val.abs().mean().detach() + 1e-5)
            rl_loss = -lmbda * q_val.mean()

            # Selective BC loss in [-1, 1] normalized space ONLY on verified expert/successful transitions
            expert_count = is_expert.sum()
            if expert_count > 0.0 and bc_weight > 0.0:
                per_sample_mse = ((pred_norm - expert_actions_norm) ** 2).mean(dim=-1, keepdim=True)
                bc_loss = (per_sample_mse * is_expert).sum() / (expert_count + 1e-6)
            else:
                bc_loss = torch.zeros(1, device=self.device).squeeze()

            actor_loss = rl_loss + bc_weight * bc_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


# -----------------------------------------------------------------------------------------------
# Fixed-Seed Policy Evaluation Helper for 3-Way Ablation
# -----------------------------------------------------------------------------------------------
def evaluate_policy_ablation(policy: TD3LocalAvoidancePolicy, n_episodes: int = 120, base_seed: int = 10000):
    """
    Evaluates a policy on `n_episodes` fixed-seed episodes (50% normal multi-obstacle,
    50% unseen_hard narrow multi-obstacle scenarios).
    Returns detailed metrics dictionary.
    """
    env = ObstacleAvoidanceEnv()
    successes = 0
    collisions = 0
    timeouts = 0
    reached_steps = []
    all_dw = []
    ep_min_obs_dists = []
    res_v_mags = []
    res_w_mags = []

    for i in range(n_episodes):
        ep_seed = base_seed + i
        random.seed(ep_seed)
        np.random.seed(ep_seed)

        diff_mode = "unseen_hard" if (i >= n_episodes // 2) else "normal"
        s = env.reset(difficulty=diff_mode)
        done = False
        min_obs_ep = float("inf")

        while not done:
            v_prior, w_prior = compute_expert_avoidance_action(s, env.max_v, env.min_v, env.max_w)
            v, w = policy.predict(s)
            res_v_mags.append(abs(v - v_prior))
            res_w_mags.append(abs(w - w_prior))

            s, _, done, info = env.step((v, w), noisy=True)
            all_dw.append(info['dw'])
            if info['min_obs_dist'] < min_obs_ep:
                min_obs_ep = info['min_obs_dist']

        ep_min_obs_dists.append(min_obs_ep)
        if info['reached']:
            successes += 1
            reached_steps.append(env.step_count)
        elif info['collision']:
            collisions += 1
        else:
            timeouts += 1

    return {
        'n_episodes': n_episodes,
        'success_rate': (successes / n_episodes) * 100.0,
        'collision_rate': (collisions / n_episodes) * 100.0,
        'timeout_rate': (timeouts / n_episodes) * 100.0,
        'avg_steps': float(np.mean(reached_steps)) if len(reached_steps) > 0 else float(env.max_steps),
        'avg_dw': float(np.mean(all_dw)) if len(all_dw) > 0 else 0.0,
        'avg_min_obs_dist': float(np.mean(ep_min_obs_dists)) if len(ep_min_obs_dists) > 0 else 0.0,
        'avg_res_v': float(np.mean(res_v_mags)) if len(res_v_mags) > 0 else 0.0,
        'avg_res_w': float(np.mean(res_w_mags)) if len(res_w_mags) > 0 else 0.0,
    }


# -----------------------------------------------------------------------------------------------
# Main Training, ONNX Export, Equivalence Check & 3-Way Ablation Benchmark
# -----------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TD3 DRL Obstacle Avoidance Trainer")
    parser.add_argument("--episodes", type=int, default=300, help="Training episodes")
    parser.add_argument("--benchmark", action="store_true", help="Run evaluation & latency benchmark only")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    config_dir = os.path.join(ws_path, "config")
    os.makedirs(config_dir, exist_ok=True)
    pth_path = os.path.join(config_dir, "recovery_td3.pth")
    onnx_path = os.path.join(config_dir, "recovery_td3.onnx")

    env = ObstacleAvoidanceEnv()
    trainer = TD3Trainer(alpha=2.5)
    buffer = ReplayBuffer(capacity=60000)

    if not args.benchmark:
        print(f"\n==========================================================================")
        print(f"  🚀 TD3+BC Local Avoidance DRL Training Started ({args.episodes} Episodes)  ")
        print(f"==========================================================================")

        start_time = time.time()
        window_success = []
        collision_count = 0

        for ep in range(1, args.episodes + 1):
            progress_ratio = ep / float(max(args.episodes, 1))
            diff_mode = "unseen_hard" if (random.random() < 0.35 * progress_ratio) else "normal"
            state = env.reset(difficulty=diff_mode)
            ep_reward = 0.0
            done = False

            # Warm-start curriculum: first 25% episodes collect prior + exploration transitions;
            # remaining 75% roll out policy with decaying exploration noise
            use_expert_rollout = (progress_ratio <= 0.25) or (
                progress_ratio <= 0.50 and random.random() < 0.20
            )
            expl_noise = max(0.03, 0.14 * (1.0 - 0.75 * progress_ratio))
            # Decay BC weight from 0.40 down to 0.02 so RL value optimization dominates
            bc_weight = max(0.02, 0.40 * ((1.0 - progress_ratio) ** 1.5))

            ep_transitions = []
            step_idx = 0

            while not done:
                v_ref, w_ref = compute_expert_avoidance_action(state, env.max_v, env.min_v, env.max_w)
                ref_action = np.array([v_ref, w_ref], dtype=np.float32)

                if use_expert_rollout:
                    ref_norm = normalize_action(ref_action, env.min_v, env.max_v, env.max_w)
                    noisy_norm = np.clip(
                        ref_norm + np.random.normal(0.0, 0.08, size=2).astype(np.float32),
                        -1.0,
                        1.0
                    )
                    action = denormalize_action(noisy_norm, env.min_v, env.max_v, env.max_w)
                else:
                    action = trainer.select_action(state, noise_norm=expl_noise)

                next_state, reward, done, info = env.step(action)
                # Under 1-step control delay, store the causally executed delayed command (executed_cmd)
                # so (state, a_exec) -> (reward, next_state) satisfies Markov causality in the replay buffer.
                a_causal = np.asarray(info.get('executed_cmd', action), dtype=np.float32).copy()
                # Step 0 under delay executes the initial zero buffer rather than the expert policy.
                allow_exp = not (env.use_control_delay and step_idx == 0)
                ep_transitions.append((state, a_causal, reward, next_state, done, ref_action, allow_exp))

                state = next_state
                ep_reward += reward
                step_idx += 1

            # Causal buffer commit: always store actual executed `a_exec` in buffer.actions;
            # only mark `is_expert = 1.0` if episode succeeded without collision!
            ep_succeeded = bool(info['reached'] and not info['collision'])
            for s_t, a_exec, r_t, ns_t, d_t, a_ref, allow_exp in ep_transitions:
                is_exp_flag = 1.0 if (ep_succeeded and use_expert_rollout and allow_exp) else 0.0
                buffer.add(
                    s_t,
                    a_exec,
                    r_t,
                    ns_t,
                    d_t,
                    is_expert=is_exp_flag,
                    expert_action=a_ref,
                )
                if buffer.size >= 256:
                    trainer.train_step(buffer, batch_size=128, bc_weight=bc_weight)

            window_success.append(1 if info['reached'] else 0)
            if len(window_success) > 50:
                window_success.pop(0)
            if info['collision']:
                collision_count += 1

            if ep % 50 == 0 or ep == args.episodes:
                recent_rate = np.mean(window_success) * 100.0
                elapsed = time.time() - start_time
                print(
                    f"[Episode {ep:3d}/{args.episodes}] Reward: {ep_reward:6.1f} | "
                    f"Recent Success: {recent_rate:5.1f}% | Collisions: {collision_count:2d} | "
                    f"BC Weight: {bc_weight:.3f} | Elapsed: {elapsed:4.1f}s"
                )

        print(f"\n[INFO] Saving trained PyTorch weights to: {pth_path}")
        torch.save(trainer.actor.state_dict(), pth_path)

        print(f"[INFO] Exporting optimized ONNX model to: {onnx_path}")
        export_policy = TD3LocalAvoidancePolicy(model_path=pth_path, device="cpu")
        assert export_policy.model_type == "torch", "Failed to load saved .pth weights for ONNX export"
        export_ok = export_policy.export_onnx(onnx_path)
        assert export_ok, "ONNX export failed"

    # -------------------------------------------------------------------------
    # 1. PyTorch vs ONNX Numerical Equivalence Verification (100 Random States)
    # -------------------------------------------------------------------------
    prior_policy = TD3LocalAvoidancePolicy(model_path=None, device="cpu", allow_untrained=True)
    torch_policy = TD3LocalAvoidancePolicy(model_path=pth_path, device="cpu")
    onnx_policy = TD3LocalAvoidancePolicy(model_path=onnx_path, device="cpu")

    assert prior_policy.model_type == "torch", "Untrained Prior policy must initialize TD3Actor"
    assert torch_policy.model_type == "torch", f"Expected torch engine for {pth_path}, got {torch_policy.model_type}"
    assert onnx_policy.model_type == "onnx", f"Expected onnx engine for {onnx_path}, got {onnx_policy.model_type}"

    rng = np.random.default_rng(12345)
    rand_states = rng.uniform(0.0, 1.0, size=(100, 28)).astype(np.float32)
    rand_states[:, 25] = rng.uniform(-1.0, 1.0, size=100).astype(np.float32)
    rand_states[:, 27] = rng.uniform(-1.0, 1.0, size=100).astype(np.float32)

    torch_preds = np.array([torch_policy.predict(s) for s in rand_states], dtype=np.float32)
    onnx_preds = np.array([onnx_policy.predict(s) for s in rand_states], dtype=np.float32)
    max_abs_diff = float(np.max(np.abs(torch_preds - onnx_preds)))
    mean_abs_diff = float(np.mean(np.abs(torch_preds - onnx_preds)))
    assert max_abs_diff < 1e-4, f"PyTorch vs ONNX max_abs_diff ({max_abs_diff:.2e}) exceeded 1e-4!"

    # -------------------------------------------------------------------------
    # 2. 3-Way Fixed-Seed Ablation Study (120 Episodes: 60 Normal + 60 Unseen Hard)
    # -------------------------------------------------------------------------
    n_eval = 120
    ablation_models = [
        ("1. Untrained Prior Only", prior_policy),
        ("2. Trained TD3 (PyTorch)", torch_policy),
        ("3. Trained TD3 (ONNX)", onnx_policy),
    ]
    ablation_results = {}
    for label, pol in ablation_models:
        ablation_results[label] = evaluate_policy_ablation(pol, n_episodes=n_eval, base_seed=10000)

    print(f"\n==========================================================================================================")
    print(f"  📊 3-Way Ablation Study ({n_eval} Fixed-Seed Episodes: 60 Standard + 60 Unseen Hard Multi-Obstacle)  ")
    print(f"==========================================================================================================")
    header = (
        f"{'Model Variant':<26} | {'Succ(%)':>7} | {'Coll(%)':>7} | {'Time(%)':>7} | "
        f"{'Steps':>6} | {'dw(rad/s)':>9} | {'MinDist(m)':>10} | {'|res_v|':>8} | {'|res_w|':>8}"
    )
    print(header)
    print("-" * len(header))
    for label, _ in ablation_models:
        r = ablation_results[label]
        print(
            f"{label:<26} | {r['success_rate']:7.1f} | {r['collision_rate']:7.1f} | {r['timeout_rate']:7.1f} | "
            f"{r['avg_steps']:6.1f} | {r['avg_dw']:9.4f} | {r['avg_min_obs_dist']:10.4f} | "
            f"{r['avg_res_v']:8.4f} | {r['avg_res_w']:8.4f}"
        )
    print(f"==========================================================================================================")

    # -------------------------------------------------------------------------
    # 3. Edge Inference Latency & Dynamic Specification Benchmark
    # -------------------------------------------------------------------------
    print(f"\n==========================================================================")
    print(f"  ⚡ TD3 ONNX Closed-Loop Evaluation & Edge Latency Benchmark  ⚡")
    print(f"==========================================================================")

    size_kb = os.path.getsize(onnx_path) / 1024.0 if os.path.exists(onnx_path) else float("inf")
    size_status = "PASS" if size_kb < 500.0 else "FAIL"
    diff_status = "PASS" if max_abs_diff < 1e-4 else "FAIL"

    dummy_state = rand_states[0]
    for _ in range(50):
        onnx_policy.predict(dummy_state)

    latencies = []
    for _ in range(1000):
        t0 = time.perf_counter()
        onnx_policy.predict(dummy_state)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    avg_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    p99_lat = float(np.percentile(latencies, 99))
    lat_status = "PASS" if p99_lat < 1.500 else "FAIL"

    onnx_res = ablation_results["3. Trained TD3 (ONNX)"]
    prior_res = ablation_results["1. Untrained Prior Only"]
    improv_status = (
        "PASS"
        if (
            onnx_res['success_rate'] > prior_res['success_rate']
            or (
                onnx_res['success_rate'] == prior_res['success_rate']
                and onnx_res['avg_min_obs_dist'] > prior_res['avg_min_obs_dist']
            )
        )
        else "FAIL"
    )

    print(f"  • Engine Active          : {onnx_policy.model_type.upper()}")
    print(f"  • ONNX Model File Size   : {size_kb:.2f} KB (Spec: < 500.0 KB -> {size_status})")
    print(f"  • PyTorch-ONNX Max Diff  : {max_abs_diff:.2e} (Mean: {mean_abs_diff:.2e}, Spec: < 1e-4 -> {diff_status})")
    print(f"  • Closed-Loop Success    : {onnx_res['success_rate']:.1f}% (Prior: {prior_res['success_rate']:.1f}% -> {improv_status})")
    print(f"  • Collision Rate         : {onnx_res['collision_rate']:.1f}% (Prior: {prior_res['collision_rate']:.1f}%)")
    print(f"  • Timeout Rate           : {onnx_res['timeout_rate']:.1f}% (Prior: {prior_res['timeout_rate']:.1f}%)")
    print(f"  • Steering Smoothness    : {onnx_res['avg_dw']:.4f} rad/s/step")
    print(f"  • Mean Min Obs Clearance : {onnx_res['avg_min_obs_dist']:.4f} m (Prior: {prior_res['avg_min_obs_dist']:.4f} m)")
    print(f"  • Average 1-Step Latency : {avg_lat:.3f} ms")
    print(f"  • 95th Percentile        : {p95_lat:.3f} ms")
    print(f"  • 99th Percentile        : {p99_lat:.3f} ms (Target: < 1.500 ms -> {lat_status})")
    print(f"==========================================================================\n")


if __name__ == "__main__":
    main()
