#!/usr/bin/env python3
"""
Fixed-Variable A/B Benchmark Runner for Disaster Autonomous Recovery:
Compares:
  1. Baseline: Current 4-layer hybrid recovery (recovery_strategy:=baseline)
  2. Proposed: Upgraded Layer-2 Rolling 2D Grid A* + Topological Shortcut
               + Unstuck Escape + Episode SE(2) Reset (recovery_strategy:=proposed)

Across identical fixed seeds, target depths, goal pools, and identical fallen-rock
coordinates across 5 distinct disaster environments/situations:
  - collapsed       : Collapsed building (12-20m, 3-6 rocks)
  - collapsed_heavy : Severe secondary collapse (14-20m, 6-9 rocks)
  - tunnel          : Underground disaster tunnel (10-16m, 3-6 rocks)
  - house           : Multi-room indoor apartment (7-12m, 2-4 rocks)
  - maze            : Tight inner maze + 2 dynamic moving obstacles (6-10m, 2-4 rocks)
"""

import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Tuple

os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
os.environ.setdefault("GAZEBO_IP", "127.0.0.1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Header, String
from std_srvs.srv import Empty

WS_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS_ROOT, "src", "disaster_recovery"))
from debris_utils import (  # noqa: E402
    delete_debris_entities,
    select_random_debris_positions,
    spawn_random_debris_cluster,
)
from disaster_recovery.local_grid_planner import RollingLocalGrid, astar_2d  # noqa: E402


def normalize_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


@dataclass
class ScenarioConfig:
    name: str
    label_ko: str
    world_file: str
    reset_pose: Tuple[float, float, float]
    max_step_size: float
    explore_x_range: Tuple[float, float]
    explore_y_range: Tuple[float, float]
    min_depth: float
    max_depth: float
    debris_min: int
    debris_max: int
    min_spacing: float
    outbound_timeout_sim: float
    return_timeout_base: float
    funnel_collapsed: bool = False
    use_guided_outbound: bool = False
    min_blackout_dist: float = 0.0
    min_checkpoints: int = 0
    doorway_keepouts: Tuple[Tuple[float, float, float], ...] = ()


WS_WORLDS = os.path.join(WS_ROOT, "src", "disaster_recovery", "worlds")

SCENARIOS: Dict[str, ScenarioConfig] = {
    "collapsed": ScenarioConfig(
        name="collapsed",
        label_ko="붕괴 건물 (기본 재난환경)",
        world_file=f"{WS_WORLDS}/disaster_collapsed_building.world",
        reset_pose=(0.0, -4.0, 1.5708),
        max_step_size=0.004,
        explore_x_range=(-2.0, 10.0),
        explore_y_range=(2.0, 12.0),
        min_depth=12.0,
        max_depth=20.0,
        debris_min=3,
        debris_max=6,
        min_spacing=1.20,
        outbound_timeout_sim=150.0,
        return_timeout_base=240.0,
        funnel_collapsed=True,
    ),
    "collapsed_heavy": ScenarioConfig(
        name="collapsed_heavy",
        label_ko="붕괴 건물 (고밀도 다중 낙석)",
        world_file=f"{WS_WORLDS}/disaster_collapsed_building.world",
        reset_pose=(0.0, -4.0, 1.5708),
        max_step_size=0.004,
        explore_x_range=(-2.0, 10.0),
        explore_y_range=(2.0, 12.0),
        min_depth=14.0,
        max_depth=20.0,
        debris_min=6,
        debris_max=9,
        min_spacing=0.90,
        outbound_timeout_sim=160.0,
        return_timeout_base=260.0,
        funnel_collapsed=True,
    ),
    "tunnel": ScenarioConfig(
        name="tunnel",
        label_ko="지하 터널 (협소 험지·차량 잔해)",
        world_file=f"{WS_WORLDS}/disaster_tunnel_world.world",
        reset_pose=(-0.5, -3.0, 1.5708),
        max_step_size=0.001,
        explore_x_range=(-2.5, 2.5),
        explore_y_range=(2.0, 22.0),
        min_depth=10.0,
        max_depth=16.0,
        debris_min=3,
        debris_max=6,
        min_spacing=1.05,
        outbound_timeout_sim=140.0,
        return_timeout_base=220.0,
        funnel_collapsed=False,
    ),
    "house": ScenarioConfig(
        name="house",
        label_ko="실내 가옥 (다실 문틀·직각 복도)",
        world_file=f"{WS_WORLDS}/disaster_house_world.world",
        reset_pose=(-2.0, -0.5, 0.0),
        max_step_size=0.001,
        explore_x_range=(-6.6, 6.6),
        explore_y_range=(-4.2, 4.8),
        min_depth=14.8,
        max_depth=17.8,
        debris_min=2,
        debris_max=4,
        min_spacing=0.95,
        outbound_timeout_sim=150.0,
        return_timeout_base=240.0,
        funnel_collapsed=False,
        use_guided_outbound=True,
        min_blackout_dist=4.2,
        min_checkpoints=6,
        doorway_keepouts=(
            (1.10, -0.17, 0.65),
            (-5.15, 3.30, 0.60),
            (-6.30, 0.92, 0.65),
            (2.30, 4.63, 0.65),
            (6.22, -0.17, 0.65),
        ),
    ),
    "maze": ScenarioConfig(
        name="maze",
        label_ko="미로 격벽 (동적 이동 장애물 포함)",
        world_file=f"{WS_WORLDS}/disaster_maze_world.world",
        reset_pose=(0.0, 0.0, 0.0),
        max_step_size=0.001,
        explore_x_range=(-2.0, 2.0),
        explore_y_range=(-2.0, 2.0),
        min_depth=9.0,
        max_depth=12.5,
        debris_min=2,
        debris_max=3,
        min_spacing=0.90,
        outbound_timeout_sim=130.0,
        return_timeout_base=210.0,
        funnel_collapsed=False,
        use_guided_outbound=True,
        min_blackout_dist=1.65,
        min_checkpoints=7,
        doorway_keepouts=(
            (-1.03, -1.50, 0.55),
            (1.25, 0.94, 0.55),
        ),
    ),
}

# Verified multi-doorway room traversal routes for turtlebot3_house (0.25m+ wall clearance)
HOUSE_ROUTES: List[List[Tuple[float, float]]] = [
    [(1.10, -0.65), (1.10, 0.38), (-1.40, 0.38), (-2.60, 2.80), (-5.15, 3.30), (-6.35, 3.20), (-6.30, 0.92), (-6.35, -2.10)],
    [(1.10, -0.65), (1.10, 0.38), (4.20, 0.38), (5.80, 2.00), (5.20, 4.10), (2.30, 4.63), (1.15, 4.30), (1.15, 2.50), (1.15, 4.20)],
    [(1.10, -0.65), (1.10, 0.38), (6.22, 0.38), (6.22, -0.17), (6.22, -4.10), (5.65, -2.10), (6.65, -3.70)],
    [(1.10, -0.65), (1.10, 0.38), (-1.40, 0.38), (-3.50, 3.50), (-5.15, 3.30), (-6.35, 4.10), (-6.35, 1.80), (-6.30, 0.92), (-6.35, -2.50)],
]

# Verified inner-partition traversal routes for turtlebot3_dqn_world maze (0.25m+ wall clearance)
MAZE_ROUTES: List[List[Tuple[float, float]]] = [
    [(-0.55, -0.80), (-1.03, -1.50), (-1.03, -1.92), (-1.75, -1.92), (-1.03, -1.92), (-1.03, -1.50), (-1.80, -0.75), (-1.80, 0.25), (-1.80, 1.82), (-0.45, 1.94), (0.60, 1.92), (0.60, 0.94), (1.80, 0.94), (1.80, 1.82)],
    [(0.55, 0.45), (0.60, 0.94), (1.80, 0.94), (1.80, 1.82), (1.80, 0.94), (0.60, 0.94), (0.60, 1.92), (-0.45, 1.94), (-1.80, 1.82), (-1.80, -0.60), (-1.03, -1.50), (-1.03, -1.92), (-1.75, -1.92)],
    [(0.30, -0.95), (0.30, -1.86), (1.72, -1.86), (1.80, -0.15), (0.55, -0.15), (0.60, 0.94), (1.80, 0.94), (1.80, 1.82), (1.80, 0.94), (0.60, 0.94), (0.60, 1.92), (-0.45, 1.94), (-1.80, 1.82)],
    [(-0.55, -0.85), (-1.80, -0.85), (-1.80, 1.82), (-0.45, 1.94), (0.60, 1.92), (0.60, 0.94), (1.80, 0.94), (1.80, 1.82), (1.80, 0.94), (0.55, 0.45), (0.55, -0.15), (1.80, -0.15), (1.72, -1.86), (0.30, -1.86)],
]


class SimHarnessNode(Node):
    """ROS 2 sim-time client node for benchmark driving, watchdog, and telemetry."""

    def __init__(self):
        super().__init__(
            "ab_benchmark_harness",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.pub_cmd_remote = self.create_publisher(Twist, "/cmd_vel_remote", 10)
        self.pub_heartbeat = self.create_publisher(Header, "/heartbeat", 10)
        self.pub_reset_ep = self.create_publisher(Bool, "/recovery/reset_episode", reliable_qos)
        self.cli_reset_world = self.create_client(Empty, "/reset_world")

        self.sub_odom = self.create_subscription(Odometry, "/odom", self._cb_odom, sensor_qos)
        self.sub_scan = self.create_subscription(LaserScan, "/scan", self._cb_scan, sensor_qos)
        self.sub_status = self.create_subscription(String, "/recovery/status", self._cb_status, 10)

        self.odom_ready = False
        self.scan_ready = False
        self.cur_x = 0.0
        self.cur_y = 0.0
        self.cur_th = 0.0

        self.ctrl_state = "INIT"
        self.ctrl_stack = 0
        self.ctrl_pose = (0.0, 0.0, 0.0)
        self.last_status_ts = 0.0

        self.scan_front_min = 10.0
        self.scan_left_min = 10.0
        self.scan_right_min = 10.0
        self.scan_ranges: Optional[np.ndarray] = None
        self.scan_angles: Optional[np.ndarray] = None
        self.scan_pts_robot: Optional[np.ndarray] = None
        self.scan_seq = 0
        self.last_grid_scan_seq = -1
        self.outbound_grid = RollingLocalGrid(
            size_m=7.2,
            resolution_m=0.06,
            inflation_radius_m=0.12,
            soft_radius_m=0.24,
        )

        self.heartbeat_enabled = True
        self.hb_seq = 0
        self.hb_timer = self.create_timer(0.10, self._hb_tick)

    def sim_now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def sim_sleep(self, sim_sec: float) -> None:
        t0 = self.sim_now()
        while self.sim_now() - t0 < sim_sec:
            rclpy.spin_once(self, timeout_sec=0.01)

    def _hb_tick(self):
        if self.heartbeat_enabled:
            hdr = Header()
            hdr.stamp = self.get_clock().now().to_msg()
            hdr.frame_id = f"ab_bench_hb_{self.hb_seq}"
            self.pub_heartbeat.publish(hdr)
            self.hb_seq += 1

    def _cb_odom(self, msg: Odometry):
        self.cur_x = msg.pose.pose.position.x
        self.cur_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.cur_th = math.atan2(siny, cosy)
        self.odom_ready = True

    def _cb_scan(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        angles = (angles + math.pi) % (2.0 * math.pi) - math.pi

        clean = np.where((ranges >= 0.10) & (ranges <= 3.5) & np.isfinite(ranges), ranges, 3.5)
        valid = (ranges >= 0.10) & (ranges <= 3.5) & np.isfinite(ranges)
        fwd = valid & (np.abs(angles) <= math.radians(25.0))
        left = valid & (angles > math.radians(25.0)) & (angles <= math.radians(85.0))
        right = valid & (angles < -math.radians(25.0)) & (angles >= -math.radians(85.0))

        self.scan_front_min = float(np.min(ranges[fwd])) if np.any(fwd) else 10.0
        self.scan_left_min = float(np.min(ranges[left])) if np.any(left) else 10.0
        self.scan_right_min = float(np.min(ranges[right])) if np.any(right) else 10.0
        self.scan_ranges = clean
        self.scan_angles = angles
        pts_mask = (ranges >= 0.13) & (ranges <= 3.5) & np.isfinite(ranges)
        if np.any(pts_mask):
            rx_pts = ranges[pts_mask] * np.cos(angles[pts_mask])
            ry_pts = ranges[pts_mask] * np.sin(angles[pts_mask])
            self.scan_pts_robot = np.column_stack([rx_pts, ry_pts]).astype(np.float32)
        else:
            self.scan_pts_robot = None
        self.scan_seq += 1
        self.scan_ready = True

    def _cb_status(self, msg: String):
        self.last_status_ts = self.sim_now()
        for part in msg.data.split("|"):
            if part.startswith("STATE:"):
                self.ctrl_state = part.split(":", 1)[1]
            elif part.startswith("STACK:"):
                try:
                    self.ctrl_stack = int(part.split(":", 1)[1])
                except ValueError:
                    pass
            elif part.startswith("POSE:"):
                try:
                    xs, ys, ths = part.split(":", 1)[1].split(",")
                    self.ctrl_pose = (float(xs), float(ys), float(ths))
                except ValueError:
                    pass

    def drive_raw(self, v: float, w: float):
        tw = Twist()
        tw.linear.x = float(v)
        tw.angular.z = float(w)
        self.pub_cmd_remote.publish(tw)

    def stop_drive(self):
        for _ in range(3):
            self.drive_raw(0.0, 0.0)
            rclpy.spin_once(self, timeout_sec=0.01)

    def set_heartbeat(self, enabled: bool):
        self.heartbeat_enabled = enabled
        if enabled:
            self._hb_tick()

    def trigger_episode_reset(self):
        msg = Bool()
        msg.data = True
        self.pub_reset_ep.publish(msg)

    def reset_gazebo_world(self, timeout_s: float = 3.0) -> None:
        if self.cli_reset_world.wait_for_service(timeout_sec=2.0):
            fut = self.cli_reset_world.call_async(Empty.Request())
            t0 = time.time()
            while not fut.done() and (time.time() - t0) < timeout_s:
                rclpy.spin_once(self, timeout_sec=0.02)

    def choose_open_gap_cmd(self, goal_x: float, goal_y: float) -> Tuple[float, float]:
        if self.scan_ranges is None or self.scan_angles is None:
            return 0.14, 0.0
        if self.scan_front_min < 0.28:
            w_turn = 0.55 if self.scan_left_min >= self.scan_right_min else -0.55
            return -0.08, w_turn

        goal_rel = normalize_angle(math.atan2(goal_y - self.cur_y, goal_x - self.cur_x) - self.cur_th)
        cand_angles = np.deg2rad(np.arange(-95, 96, 5))
        half_win = math.radians(18.0)

        best_score = -1e9
        best_ang = 0.0
        for cand in cand_angles:
            mask = np.abs(((self.scan_angles - cand + math.pi) % (2.0 * math.pi)) - math.pi) <= half_win
            sector_min = float(np.min(self.scan_ranges[mask])) if np.any(mask) else 3.5
            if sector_min < 0.34:
                continue
            goal_align = math.cos(normalize_angle(float(cand) - goal_rel))
            score = min(sector_min, 2.5) + 0.70 * goal_align - 0.12 * abs(float(cand))
            if score > best_score:
                best_score = score
                best_ang = float(cand)

        if best_score < -1e8:
            w_turn = 0.55 if self.scan_left_min >= self.scan_right_min else -0.55
            return 0.0, w_turn

        w_cmd = max(-0.62, min(0.62, 1.35 * best_ang))
        v_cmd = 0.20 if self.scan_front_min > 0.65 else 0.14
        if abs(best_ang) > 0.85:
            v_cmd = 0.06
        return v_cmd, w_cmd

    def reset_outbound_grid(self):
        self.outbound_grid.reset()
        self.last_grid_scan_seq = -1

    def choose_guided_astar_cmd(self, goal_x: float, goal_y: float) -> Tuple[float, float]:
        """Follow doorway/partition checkpoints cleanly using rolling 2D occupancy + A*."""
        if self.scan_seq != self.last_grid_scan_seq and self.scan_pts_robot is not None:
            self.outbound_grid.update_scan(self.cur_x, self.cur_y, self.cur_th, self.scan_pts_robot)
            self.last_grid_scan_seq = self.scan_seq

        if not self.outbound_grid.initialized:
            return self.choose_open_gap_cmd(goal_x, goal_y)

        hard_blocked, soft_cost = self.outbound_grid.build_cost_maps(self.cur_x, self.cur_y)
        sx, sy = self.outbound_grid.world_to_grid(self.cur_x, self.cur_y)
        sx = max(1, min(self.outbound_grid.n - 2, sx))
        sy = max(1, min(self.outbound_grid.n - 2, sy))
        hard_blocked[sx, sy] = False

        half = 0.5 * self.outbound_grid.size_m - 0.35
        dx = goal_x - self.cur_x
        dy = goal_y - self.cur_y
        scale = 1.0
        if abs(dx) > half:
            scale = min(scale, half / abs(dx))
        if abs(dy) > half:
            scale = min(scale, half / abs(dy))
        gx, gy = self.outbound_grid.world_to_grid(self.cur_x + dx * scale, self.cur_y + dy * scale)
        gx = max(1, min(self.outbound_grid.n - 2, gx))
        gy = max(1, min(self.outbound_grid.n - 2, gy))

        if hard_blocked[gx, gy]:
            best_ij = None
            best_d2 = 1e9
            for rad in range(1, 13):
                for di in range(-rad, rad + 1):
                    for dj in range(-rad, rad + 1):
                        nx, ny = gx + di, gy + dj
                        if self.outbound_grid.in_bounds(nx, ny, margin=1) and not hard_blocked[nx, ny]:
                            d2 = di * di + dj * dj
                            if d2 < best_d2:
                                best_d2 = d2
                                best_ij = (nx, ny)
                if best_ij is not None:
                    break
            if best_ij is not None:
                gx, gy = best_ij

        path_ij, _ = astar_2d(hard_blocked, soft_cost, (sx, sy), (gx, gy), max_expansions=7000)
        if not path_ij:
            return self.choose_open_gap_cmd(goal_x, goal_y)

        look_cells = max(4, int(round(0.50 / self.outbound_grid.res)))
        sub_ij = path_ij[min(look_cells, len(path_ij) - 1)]
        sub_x, sub_y = self.outbound_grid.grid_to_world(sub_ij[0], sub_ij[1])
        alpha = normalize_angle(math.atan2(sub_y - self.cur_y, sub_x - self.cur_x) - self.cur_th)

        if abs(alpha) > 1.05:
            return 0.0, float(max(-0.62, min(0.62, 1.55 * alpha)))

        if self.scan_front_min < 0.19:
            w_dir = 1.0 if alpha >= 0.0 else -1.0
            if abs(alpha) < 0.15:
                w_dir = 1.0 if self.scan_left_min >= self.scan_right_min else -1.0
            return 0.0, float(0.55 * w_dir)

        ang_scale = max(0.20, math.cos(alpha))
        if self.scan_front_min < 0.34:
            prox = max(0.25, (self.scan_front_min - 0.18) / (0.34 - 0.18))
            v_cmd = 0.14 * prox * ang_scale
        else:
            v_cmd = 0.22 * ang_scale
        w_cmd = max(-0.65, min(0.65, 1.65 * alpha))
        return float(v_cmd), float(w_cmd)


def get_gazebo_true_pose(model_name: str = "burger") -> Optional[Tuple[float, float, float]]:
    try:
        out = subprocess.check_output(
            f"gz model -m {model_name} -p", shell=True, stderr=subprocess.DEVNULL, timeout=2.0
        ).decode().strip()
        vals = [float(x) for x in out.split()]
        if len(vals) >= 6:
            return vals[0], vals[1], vals[5]
    except Exception:
        pass
    return None


def run_cmd_bounded(cmd: str, timeout_s: float = 4.0) -> int:
    try:
        return subprocess.call(cmd, shell=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return -1


def _kill_matching_gazebo_procs(gz_master_uri: str) -> None:
    """Terminate leftover Gazebo/ROS processes matching only this worker's GAZEBO_MASTER_URI."""
    target_env = f"GAZEBO_MASTER_URI={gz_master_uri}".encode()
    targets = ("gzserver", "gzclient", "recovery_controller", "robot_state_publish")
    my_pid = os.getpid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == my_pid:
            continue
        try:
            with open(f"/proc/{pid}/comm", "rb") as f:
                comm = f.read().decode(errors="ignore").strip()
            if not any(comm.startswith(t[:15]) for t in targets):
                continue
            with open(f"/proc/{pid}/environ", "rb") as f:
                env_entries = f.read().split(b"\x00")
            if target_env in env_entries:
                os.kill(pid, signal.SIGKILL)
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue


def launch_stack(
    cfg: ScenarioConfig,
    strategy: str,
    rtf: float,
    gz_port: int = 11345,
    ros_domain_id: Optional[int] = None,
    kill_stale: bool = False,
):
    gz_master_uri = f"http://127.0.0.1:{gz_port}"
    if kill_stale:
        subprocess.call(
            "killall -9 gzserver gzclient recovery_controller 2>/dev/null || true",
            shell=True,
        )
    else:
        _kill_matching_gazebo_procs(gz_master_uri)
    time.sleep(1.0)

    os.makedirs("/tmp/disaster_recovery", exist_ok=True)
    log_gz = open(f"/tmp/disaster_recovery/ab_gz_{cfg.name}_{strategy}_{gz_port}.log", "w")
    log_ctrl = open(f"/tmp/disaster_recovery/ab_ctrl_{cfg.name}_{strategy}_{gz_port}.log", "w")

    rx, ry, ryaw = cfg.reset_pose
    gazebo_models = os.path.expanduser("~/.gazebo/models")
    pkg_models = os.path.join(WS_ROOT, "src", "disaster_recovery", "models")
    domain_export = f"ROS_DOMAIN_ID={ros_domain_id} " if ros_domain_id is not None else ""
    env_prefix = (
        f"export ROS_LOCALHOST_ONLY=1 {domain_export}GAZEBO_IP=127.0.0.1 "
        f"GAZEBO_MASTER_URI={gz_master_uri} "
        "GAZEBO_MODEL_DATABASE_URI='' "
        f"GAZEBO_MODEL_PATH='{gazebo_models}:{pkg_models}:/opt/ros/humble/share/turtlebot3_gazebo/models' "
        "TURTLEBOT3_MODEL=burger && "
        "source /opt/ros/humble/setup.bash && "
        f"source {WS_ROOT}/install/setup.bash && "
    )

    gz_cmd = (
        env_prefix
        + f"ros2 launch disaster_recovery disaster_world.launch.py "
        f"x_pose:={rx} y_pose:={ry} yaw_pose:={ryaw} world:={cfg.world_file} gui:=false"
    )
    p_gz = subprocess.Popen(
        ["bash", "-c", gz_cmd],
        stdout=log_gz,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    # Allow Gazebo world & mesh loading cleanly without poking physics or spawning slow ros2 CLI subprocesses
    for _ in range(12):
        time.sleep(0.5)
        if p_gz.poll() is not None:
            raise RuntimeError(f"Gazebo launch exited prematurely for {cfg.name}/{strategy}")

    params_file = os.path.join(WS_ROOT, "src", "disaster_recovery", "config", "recovery_params.yaml")
    ctrl_cmd = (
        env_prefix
        + "ros2 run disaster_recovery recovery_controller --ros-args "
        f"--params-file {params_file} "
        f"-p use_sim_time:=true -p use_aruco_docking:=false -p recovery_strategy:={strategy}"
    )
    p_ctrl = subprocess.Popen(
        ["bash", "-c", ctrl_cmd],
        stdout=log_ctrl,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    time.sleep(1.5)
    return [p_gz, p_ctrl], [log_gz, log_ctrl]


def stop_stack(procs, logs, gz_port: int = 11345):
    for p in procs:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            pass
    time.sleep(0.5)
    for p in procs:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
    _kill_matching_gazebo_procs(f"http://127.0.0.1:{gz_port}")
    for f in logs:
        try:
            f.close()
        except Exception:
            pass
    time.sleep(0.5)


def generate_fixed_scenarios(cfg: ScenarioConfig, trials: int, base_seed: int) -> List[dict]:
    """Pre-generate deterministic seed, target depth, and goal pool per trial."""
    specs = []
    for i in range(1, trials + 1):
        trial_seed = base_seed + i * 101
        rng = random.Random(trial_seed)
        target_depth = round(rng.uniform(cfg.min_depth, cfg.max_depth), 2)
        if cfg.use_guided_outbound:
            routes = HOUSE_ROUTES if cfg.name == "house" else MAZE_ROUTES
            base_route = routes[(i - 1) % len(routes)]
            full_route = list(base_route) + list(reversed(base_route[:-1])) + list(base_route[1:])
            goal_pool = [(round(px, 3), round(py, 3)) for px, py in full_route]
        else:
            goal_pool = [
                (
                    round(rng.uniform(*cfg.explore_x_range), 3),
                    round(rng.uniform(*cfg.explore_y_range), 3),
                )
                for _ in range(30)
            ]
        specs.append({
            "trial": i,
            "seed": trial_seed,
            "target_depth": target_depth,
            "goal_pool": goal_pool,
            "debris_positions": None,  # Populated during baseline run, replayed in proposed run
        })
    return specs


def _write_json_atomic(path: str, payload: dict) -> None:
    """Atomically write JSON via temporary file + os.replace to protect checkpoints."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _summarize_results(cfg: ScenarioConfig, strategy: str, results: List[dict]) -> dict:
    """Aggregate summary statistics across completed trials for one strategy."""
    ordered = sorted(results, key=lambda r: int(r["trial"]))
    passes = [r for r in ordered if r["success"]]
    summary = {
        "scenario": cfg.name,
        "label_ko": cfg.label_ko,
        "strategy": strategy,
        "trials": len(ordered),
        "passes": len(passes),
        "success_rate_pct": round(100.0 * len(passes) / max(1, len(ordered)), 1),
        "mean_error_cm_all": round(float(np.mean([r["terminal_error_cm"] for r in ordered])), 2),
        "mean_error_cm_pass": round(float(np.mean([r["terminal_error_cm"] for r in passes])), 2) if passes else None,
        "mean_time_s_all": round(float(np.mean([r["return_duration_s"] for r in ordered])), 2),
        "mean_time_s_pass": round(float(np.mean([r["return_duration_s"] for r in passes])), 2) if passes else None,
        "mean_ret_path_m_pass": round(float(np.mean([r["return_path_len_m"] for r in passes])), 2) if passes else None,
        "mean_explored_m": round(float(np.mean([r["explored_path_m"] for r in ordered])), 2),
    }
    return {"summary": summary, "trials": ordered}


def run_strategy_suite(
    cfg: ScenarioConfig,
    strategy: str,
    scenario_specs: List[dict],
    rtf: float,
    save_debris_into_specs: bool,
    initial_results: Optional[List[dict]] = None,
    on_trial_done: Optional[Callable[[dict, List[dict]], None]] = None,
    gz_port: int = 11345,
    ros_domain_id: Optional[int] = None,
    kill_stale: bool = False,
) -> dict:
    results: List[dict] = list(initial_results) if initial_results else []
    done_trials = {int(r["trial"]) for r in results}
    pending_specs = [s for s in scenario_specs if int(s["trial"]) not in done_trials]

    print(f"\n{'='*86}")
    print(
        f"  [BENCHMARK] Scenario={cfg.name} ({cfg.label_ko}) | Strategy={strategy.upper()} | "
        f"Trials={len(scenario_specs)} (cached={len(done_trials)}, pending={len(pending_specs)})"
    )
    print(f"{'='*86}")
    if not pending_specs:
        return _summarize_results(cfg, strategy, results)

    gz_master_uri = f"http://127.0.0.1:{gz_port}"
    procs, logs = launch_stack(
        cfg,
        strategy,
        rtf,
        gz_port=gz_port,
        ros_domain_id=ros_domain_id,
        kill_stale=kill_stale,
    )
    rclpy.init()
    node = SimHarnessNode()

    # Wait for sim clock, odom, scan, and controller status
    t_wait = time.time()
    while time.time() - t_wait < 60.0:
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.odom_ready and node.scan_ready and node.sim_now() > 0.1:
            break
    if not (node.odom_ready and node.scan_ready):
        stop_stack(procs, logs, gz_port=gz_port)
        node.destroy_node()
        rclpy.shutdown()
        raise RuntimeError("Sensors failed to become ready in benchmark suite")

    u_rate = int(round(rtf / cfg.max_step_size))
    run_cmd_bounded(
        f"GAZEBO_MASTER_URI={gz_master_uri} timeout -k 1 2 gz physics -u {u_rate} >/dev/null 2>&1",
        timeout_s=4.0,
    )

    base_x, base_y, base_yaw = cfg.reset_pose

    try:
        for spec in pending_specs:
            t_wall_trial0 = time.time()
            t_sim_trial0 = node.sim_now()
            t_idx = spec["trial"]
            rng = random.Random(spec["seed"])
            target_depth = spec["target_depth"]
            goal_pool = spec["goal_pool"]
            goal_ptr = 0

            # Clean inter-trial reset: reset controller state BEFORE teleporting Gazebo world
            node.set_heartbeat(True)
            node.stop_drive()
            node.trigger_episode_reset()
            node.sim_sleep(0.2)
            delete_debris_entities(max_count=12)
            node.reset_gazebo_world(timeout_s=3.0)
            run_cmd_bounded(
                f"GAZEBO_MASTER_URI={gz_master_uri} timeout -k 1 2 gz physics -u {u_rate} >/dev/null 2>&1",
                timeout_s=4.0,
            )
            node.sim_sleep(0.6)
            node.trigger_episode_reset()
            node.sim_sleep(0.8)

            # Confirm controller ready in NORMAL_TELEOP near reset_pose
            wait_t0 = node.sim_now()
            while node.sim_now() - wait_t0 < 6.0:
                rclpy.spin_once(node, timeout_sec=0.02)
                if node.ctrl_state == "NORMAL_TELEOP" and math.hypot(node.cur_x - base_x, node.cur_y - base_y) < 0.20:
                    break
            node.trigger_episode_reset()
            node.reset_outbound_grid()
            node.sim_sleep(0.4)

            # Phase 1: Outbound Exploration (Open-Gap or Guided Doorway/Partition A*)
            goal_x, goal_y = goal_pool[goal_ptr]
            goal_ptr += 1
            checkpoints_done = 0
            cp_start_t = node.sim_now()
            outbound_path: List[List[float]] = [[round(node.cur_x, 3), round(node.cur_y, 3)]]
            explored_m = 0.0
            prev_x, prev_y = node.cur_x, node.cur_y
            anchor_x, anchor_y = node.cur_x, node.cur_y
            anchor_t = node.sim_now()
            t_exp_start = node.sim_now()

            def _phase1_done() -> bool:
                if (node.sim_now() - t_exp_start) >= cfg.outbound_timeout_sim:
                    return True
                if explored_m < target_depth:
                    return False
                if cfg.min_blackout_dist > 0.0 and math.hypot(node.cur_x - base_x, node.cur_y - base_y) < cfg.min_blackout_dist:
                    return False
                if cfg.min_checkpoints > 0 and checkpoints_done < cfg.min_checkpoints:
                    return False
                return True

            while not _phase1_done():
                t_tick = node.sim_now()
                while node.sim_now() - t_tick < 0.05:
                    rclpy.spin_once(node, timeout_sec=0.01)

                step_d = math.hypot(node.cur_x - prev_x, node.cur_y - prev_y)
                explored_m += step_d
                prev_x, prev_y = node.cur_x, node.cur_y
                if len(outbound_path) == 0 or math.hypot(node.cur_x - outbound_path[-1][0], node.cur_y - outbound_path[-1][1]) >= 0.12:
                    outbound_path.append([round(node.cur_x, 3), round(node.cur_y, 3)])

                if cfg.funnel_collapsed and node.cur_y < 1.2:
                    target_h = max(1.10, min(2.04, 1.57 + 0.45 * (node.cur_x - (-0.75))))
                    h_diff = normalize_angle(target_h - node.cur_th)
                    w_cmd = max(-0.35, min(0.35, 1.2 * h_diff + 0.25 * (node.scan_right_min - node.scan_left_min)))
                    v_cmd = 0.20
                    if node.scan_front_min < 0.45:
                        v_cmd = 0.08
                        w_cmd = 0.45 if node.scan_left_min > node.scan_right_min else -0.45
                elif cfg.use_guided_outbound:
                    if math.hypot(node.cur_x - goal_x, node.cur_y - goal_y) < 0.55:
                        goal_x, goal_y = goal_pool[goal_ptr % len(goal_pool)]
                        goal_ptr += 1
                        checkpoints_done += 1
                        cp_start_t = node.sim_now()
                    elif node.sim_now() - cp_start_t > 15.0:
                        goal_x, goal_y = goal_pool[goal_ptr % len(goal_pool)]
                        goal_ptr += 1
                        cp_start_t = node.sim_now()

                    if node.sim_now() - anchor_t > 3.5:
                        if math.hypot(node.cur_x - anchor_x, node.cur_y - anchor_y) < 0.12:
                            turn_d = 0.60 if node.scan_left_min >= node.scan_right_min else -0.60
                            esc_t0 = node.sim_now()
                            while node.sim_now() - esc_t0 < 0.9:
                                node.drive_raw(-0.08, turn_d)
                                rclpy.spin_once(node, timeout_sec=0.01)
                        anchor_x, anchor_y = node.cur_x, node.cur_y
                        anchor_t = node.sim_now()

                    v_cmd, w_cmd = node.choose_guided_astar_cmd(goal_x, goal_y)
                else:
                    if math.hypot(node.cur_x - goal_x, node.cur_y - goal_y) < 1.3:
                        goal_x, goal_y = goal_pool[goal_ptr % len(goal_pool)]
                        goal_ptr += 1

                    if node.sim_now() - anchor_t > 3.5:
                        if math.hypot(node.cur_x - anchor_x, node.cur_y - anchor_y) < 0.12:
                            turn_d = 0.60 if node.scan_left_min >= node.scan_right_min else -0.60
                            esc_t0 = node.sim_now()
                            while node.sim_now() - esc_t0 < 1.0:
                                node.drive_raw(-0.10, turn_d)
                                rclpy.spin_once(node, timeout_sec=0.01)
                            goal_x, goal_y = goal_pool[goal_ptr % len(goal_pool)]
                            goal_ptr += 1
                        anchor_x, anchor_y = node.cur_x, node.cur_y
                        anchor_t = node.sim_now()

                    v_cmd, w_cmd = node.choose_open_gap_cmd(goal_x, goal_y)

                node.drive_raw(v_cmd, w_cmd)

            node.stop_drive()
            node.sim_sleep(0.3)

            # Phase 2: Spawn Identical Debris While Heartbeat Remains Active, Then Trigger Blackout
            if save_debris_into_specs or spec["debris_positions"] is None:
                kept: List[Tuple[float, float]] = []
                for attempt in range(4):
                    cand_rng = random.Random(spec["seed"] + 7919 * attempt) if attempt > 0 else rng
                    cand = select_random_debris_positions(
                        outbound_path,
                        min_count=cfg.debris_min,
                        max_count=cfg.debris_max,
                        min_spacing=cfg.min_spacing,
                        jitter_std=0.14 if cfg.use_guided_outbound else 0.22,
                        rng=cand_rng,
                    )
                    if cfg.doorway_keepouts:
                        cand = [
                            (px, py) for px, py in cand
                            if all(math.hypot(px - kx, py - ky) >= kr for kx, ky, kr in cfg.doorway_keepouts)
                        ]
                    for p in cand:
                        if all(math.hypot(p[0] - q[0], p[1] - q[1]) >= cfg.min_spacing for q in kept):
                            kept.append(p)
                    if len(kept) >= cfg.debris_min:
                        break
                debris_positions = kept[: cfg.debris_max]
                spec["debris_positions"] = [[round(px, 3), round(py, 3)] for px, py in debris_positions]
            else:
                debris_positions = [tuple(p) for p in spec["debris_positions"]]

            if debris_positions:
                def _spawn_hb_tick():
                    node.drive_raw(0.0, 0.0)
                    rclpy.spin_once(node, timeout_sec=0.01)

                spawn_random_debris_cluster(
                    debris_positions,
                    wait=True,
                    tick_fn=_spawn_hb_tick,
                )
            node.stop_drive()
            node.sim_sleep(0.2)

            turnoff_x, turnoff_y = node.cur_x, node.cur_y
            dist_from_base = math.hypot(turnoff_x - base_x, turnoff_y - base_y)
            node.set_heartbeat(False)
            t_blackout = node.sim_now()

            # Phase 3: Autonomous Recovery Monitoring (sim-time)
            max_return_wait = max(cfg.return_timeout_base, explored_m * 16.0)
            saw_recovery = False
            success = False
            return_trajectory: List[List[float]] = [[round(turnoff_x, 3), round(turnoff_y, 3)]]
            states_visited = set()

            while node.sim_now() - t_blackout < max_return_wait:
                t_step = node.sim_now()
                while node.sim_now() - t_step < 0.10:
                    rclpy.spin_once(node, timeout_sec=0.01)

                states_visited.add(node.ctrl_state)
                if node.ctrl_state in (
                    "BACKTRACKING_ROTATE",
                    "BACKTRACKING_TRANSLATE",
                    "BACKTRACKING_HYBRID",
                    "LOCAL_TD3_AVOIDANCE",
                    "OBSTACLE_HOLD",
                    "HANDOVER_TO_ARUCO",
                    "MISSION_COMPLETE",
                ):
                    saw_recovery = True
                elif (
                    "EMERGENCY_STOP" in states_visited
                    and node.ctrl_state == "NORMAL_TELEOP"
                    and math.hypot(node.cur_x - base_x, node.cur_y - base_y) < 0.85
                ):
                    saw_recovery = True

                if math.hypot(node.cur_x - return_trajectory[-1][0], node.cur_y - return_trajectory[-1][1]) >= 0.12:
                    return_trajectory.append([round(node.cur_x, 3), round(node.cur_y, 3)])

                if saw_recovery and node.ctrl_state in ("MISSION_COMPLETE", "NORMAL_TELEOP", "ARUCO_DOCKING"):
                    success = True
                    break

            return_duration = node.sim_now() - t_blackout
            true_pose = get_gazebo_true_pose("burger")
            if true_pose is not None:
                final_x, final_y, final_th = true_pose
            else:
                final_x, final_y, final_th = node.cur_x, node.cur_y, node.cur_th

            err_m = math.hypot(final_x - base_x, final_y - base_y)
            if success and err_m > 0.85:
                # Physical Gazebo pose missed base camp despite odometry arrival claim
                success = False
            elif not success and err_m < 0.60 and (saw_recovery or "EMERGENCY_STOP" in states_visited):
                success = True

            # Return path travel distance
            ret_path_len = 0.0
            for k in range(1, len(return_trajectory)):
                ret_path_len += math.hypot(
                    return_trajectory[k][0] - return_trajectory[k - 1][0],
                    return_trajectory[k][1] - return_trajectory[k - 1][1],
                )

            rec = {
                "trial": t_idx,
                "seed": spec["seed"],
                "strategy": strategy,
                "scenario": cfg.name,
                "target_depth_m": target_depth,
                "explored_path_m": round(explored_m, 3),
                "dist_from_base_m": round(dist_from_base, 3),
                "turnoff_pose": [round(turnoff_x, 3), round(turnoff_y, 3)],
                "debris_count": len(debris_positions),
                "debris_positions": spec["debris_positions"],
                "success": success,
                "return_duration_s": round(return_duration, 2),
                "terminal_error_cm": round(err_m * 100.0, 2),
                "final_pose": [round(final_x, 3), round(final_y, 3), round(final_th, 3)],
                "return_path_len_m": round(ret_path_len, 3),
                "final_ctrl_state": node.ctrl_state,
                "states_visited": sorted(states_visited),
                "outbound_path": outbound_path,
                "return_path": return_trajectory,
            }
            results.append(rec)
            if on_trial_done is not None:
                on_trial_done(rec, results)
            tag = "PASS" if success else "FAIL"
            wall_dt = max(0.1, time.time() - t_wall_trial0)
            sim_dt = max(0.0, node.sim_now() - t_sim_trial0)
            print(
                f"  [{strategy.upper():8s} | {cfg.name:15s} #{t_idx:02d}/{len(scenario_specs)}] "
                f"{tag} | depth={explored_m:5.2f}m | rocks={len(debris_positions)} | "
                f"err={err_m*100:6.1f}cm | time={return_duration:6.1f}s | end={node.ctrl_state} | RTF={sim_dt/wall_dt:.1f}x"
            )
    finally:
        delete_debris_entities(max_count=12)
        node.destroy_node()
        rclpy.shutdown()
        stop_stack(procs, logs, gz_port=gz_port)

    return _summarize_results(cfg, strategy, results)


def main():
    parser = argparse.ArgumentParser(description="Fixed-Variable A/B Disaster Recovery Benchmark")
    parser.add_argument(
        "--scenario",
        default="collapsed",
        choices=list(SCENARIOS.keys()) + ["all"],
        help="Scenario name or 'all'",
    )
    parser.add_argument("--trials", type=int, default=10, help="Trials per strategy (default 10)")
    parser.add_argument("--seed", type=int, default=2026, help="Base deterministic seed")
    parser.add_argument("--rtf", type=float, default=4.0, help="Target Gazebo RTF factor (default 4.0)")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(WS_ROOT, "benchmark_results", "ab_comparison"),
        help="Output JSON directory",
    )
    parser.add_argument(
        "--reuse-baseline",
        action="store_true",
        help="Reuse saved fixed_specs and baseline results from existing baseline-dir/out-dir JSON and run proposed only",
    )
    parser.add_argument(
        "--baseline-dir",
        default="",
        help="Optional directory to read existing fixed_specs & baseline when --out-dir differs (e.g. phase2_sim2sim)",
    )
    parser.add_argument(
        "--proposed-strategy",
        default="proposed",
        choices=["proposed", "proposed_rl"],
        help="Controller recovery_strategy to benchmark for the proposed slot (default proposed)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume unfinished trials from .partial.json checkpoint in --out-dir",
    )
    parser.add_argument(
        "--gz-port",
        type=int,
        default=11345,
        help="Unique Gazebo master TCP port for parallel runs (default 11345)",
    )
    parser.add_argument(
        "--ros-domain-id",
        type=int,
        default=None,
        help="Unique ROS_DOMAIN_ID for parallel runs (optional)",
    )
    parser.add_argument(
        "--kill-stale",
        action="store_true",
        help="Kill all global gzserver/recovery_controller before start (unsafe during parallel runs)",
    )
    args = parser.parse_args()

    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    os.environ["GAZEBO_MASTER_URI"] = f"http://127.0.0.1:{args.gz_port}"
    if args.ros_domain_id is not None:
        os.environ["ROS_DOMAIN_ID"] = str(args.ros_domain_id)

    os.makedirs(args.out_dir, exist_ok=True)
    selected = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]
    all_reports = {}

    for s_idx, s_name in enumerate(selected):
        cfg = SCENARIOS[s_name]
        out_path = os.path.join(args.out_dir, f"ab_{s_name}_{args.trials}trials.json")
        partial_path = os.path.join(args.out_dir, f"ab_{s_name}_{args.trials}trials.partial.json")

        base_trials_init: List[dict] = []
        prop_trials_init: List[dict] = []
        base_res: Optional[dict] = None

        base_src_dir = args.baseline_dir if args.baseline_dir else args.out_dir
        base_in_path = os.path.join(base_src_dir, f"ab_{s_name}_{args.trials}trials.json")

        if args.reuse_baseline and os.path.exists(base_in_path):
            with open(base_in_path, "r") as f:
                prev = json.load(f)
            specs = prev["fixed_specs"]
            base_res = prev["baseline"]
            base_trials_init = list(base_res.get("trials", []))
            print(f"\n[REUSE BASELINE] Loaded {len(specs)} fixed scenarios & baseline results from {base_in_path}")
            if args.resume and os.path.exists(partial_path):
                with open(partial_path, "r") as f:
                    part = json.load(f)
                prop_trials_init = list(part.get("proposed_trials", []))
                print(f"[RESUME PARTIAL] Loaded {len(prop_trials_init)} proposed trials from {partial_path}")
        elif args.resume and os.path.exists(partial_path):
            with open(partial_path, "r") as f:
                part = json.load(f)
            specs = part["fixed_specs"]
            base_trials_init = list(part.get("baseline_trials", []))
            prop_trials_init = list(part.get("proposed_trials", []))
            print(
                f"\n[RESUME CHECKPOINT] Loaded baseline={len(base_trials_init)}, "
                f"proposed={len(prop_trials_init)} from {partial_path}"
            )
        else:
            specs = generate_fixed_scenarios(cfg, args.trials, args.seed + s_idx * 1000)

        def _save_checkpoint(_rec: dict, _cur: List[dict], *, phase: str) -> None:
            payload = {
                "config": asdict(cfg),
                "fixed_specs": specs,
                "baseline_trials": _cur if phase == "baseline" else (
                    base_res["trials"] if base_res is not None else base_trials_init
                ),
                "proposed_trials": _cur if phase == "proposed" else prop_trials_init,
            }
            _write_json_atomic(partial_path, payload)

        if base_res is None:
            base_res = run_strategy_suite(
                cfg,
                "baseline",
                specs,
                rtf=args.rtf,
                save_debris_into_specs=True,
                initial_results=base_trials_init,
                on_trial_done=lambda rec, cur: _save_checkpoint(rec, cur, phase="baseline"),
                gz_port=args.gz_port,
                ros_domain_id=args.ros_domain_id,
                kill_stale=args.kill_stale,
            )
            base_trials_init = list(base_res["trials"])
            _save_checkpoint({}, base_trials_init, phase="baseline")

        prop_res = run_strategy_suite(
            cfg,
            args.proposed_strategy,
            specs,
            rtf=args.rtf,
            save_debris_into_specs=False,
            initial_results=prop_trials_init,
            on_trial_done=lambda rec, cur: _save_checkpoint(rec, cur, phase="proposed"),
            gz_port=args.gz_port,
            ros_domain_id=args.ros_domain_id,
            kill_stale=args.kill_stale,
        )

        # Save fixed specs and A/B comparison JSON
        scenario_payload = {
            "config": asdict(cfg),
            "fixed_specs": specs,
            "baseline": base_res,
            "proposed": prop_res,
        }
        _write_json_atomic(out_path, scenario_payload)
        if os.path.exists(partial_path):
            os.remove(partial_path)
        print(f"\n[SAVED] {out_path}")
        all_reports[s_name] = {
            "baseline": base_res["summary"],
            "proposed": prop_res["summary"],
        }

    master_path = os.path.join(args.out_dir, f"ab_summary_{args.scenario}_{args.trials}trials.json")
    _write_json_atomic(master_path, all_reports)
    print(f"\n[ALL DONE] Master summary: {master_path}")


if __name__ == "__main__":
    main()
