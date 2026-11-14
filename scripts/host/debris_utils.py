#!/usr/bin/env python3
"""
Shared utilities for dynamically spawning and cleaning up random debris/rock
obstacles in Gazebo Classic simulations.
"""

import math
import os
import random
import subprocess
import time
from typing import Callable, List, Optional, Sequence, Tuple

WS_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEBRIS_SDF_PATH = os.path.join(
    WS_ROOT, "src", "disaster_recovery", "config", "debris_obstacle.sdf"
)


def select_random_debris_positions(
    path_pts: Sequence[Sequence[float]],
    min_count: int = 3,
    max_count: int = 6,
    min_spacing: float = 1.2,
    end_margin: float = 1.2,
    jitter_std: float = 0.22,
    fallback_pose: Optional[Tuple[float, float, float]] = None,
    rng: Optional[random.Random] = None,
) -> List[Tuple[float, float]]:
    """
    Pick `N` (random in [min_count, max_count]) rock drop coordinates along a
    traversed path while avoiding the base camp and the robot's current pose.
    """
    rand = rng if rng is not None else random
    target_count = rand.randint(min_count, max_count)

    candidates: List[Tuple[float, float]] = []
    if len(path_pts) >= 2:
        start_x, start_y = float(path_pts[0][0]), float(path_pts[0][1])
        end_x, end_y = float(path_pts[-1][0]), float(path_pts[-1][1])
        for pt in path_pts:
            px, py = float(pt[0]), float(pt[1])
            if (
                math.hypot(px - start_x, py - start_y) >= end_margin
                and math.hypot(px - end_x, py - end_y) >= end_margin
            ):
                candidates.append((px, py))

    # Fallback if path is too short (e.g., manual '[O]' right after startup)
    if not candidates:
        rx, ry, ryaw = fallback_pose if fallback_pose is not None else (0.0, -4.0, 1.5708)
        for _ in range(target_count * 4):
            dist = rand.uniform(1.0, 4.5)
            ang = ryaw + rand.uniform(-0.85, 0.85)
            candidates.append((rx + dist * math.cos(ang), ry + dist * math.sin(ang)))

    shuffled = list(candidates)
    rand.shuffle(shuffled)

    selected: List[Tuple[float, float]] = []
    for spacing in (min_spacing, min_spacing * 0.55, 0.35):
        for cx, cy in shuffled:
            if len(selected) >= target_count:
                break
            jx = cx + rand.uniform(-jitter_std, jitter_std)
            jy = cy + rand.uniform(-jitter_std, jitter_std)
            if all(math.hypot(jx - sx, jy - sy) >= spacing for sx, sy in selected):
                selected.append((round(jx, 3), round(jy, 3)))
        if len(selected) >= min_count:
            break

    return selected[:target_count]


def spawn_random_debris_cluster(
    positions: Sequence[Tuple[float, float]],
    prefix: str = "disaster_debris",
    z: float = 0.225,
    wait: bool = True,
    sdf_path: str = DEBRIS_SDF_PATH,
    tick_fn: Optional[Callable[[], None]] = None,
) -> List[str]:
    """Spawn multiple rock/debris entities in Gazebo in parallel."""
    procs = []
    names: List[str] = []
    for idx, (x, y) in enumerate(positions, start=1):
        name = f"{prefix}_{idx}"
        names.append(name)
        cmd = (
            "export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=0 && "
            "source /opt/ros/humble/setup.bash && "
            f"ros2 run gazebo_ros spawn_entity.py -entity {name} "
            f"-file {sdf_path} -x {x:.3f} -y {y:.3f} -z {z:.3f} -timeout 6.0"
        )
        p = subprocess.Popen(
            f"bash -c '{cmd}'",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        procs.append(p)

    if wait:
        if tick_fn is not None:
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and any(p.poll() is None for p in procs):
                tick_fn()
                time.sleep(0.02)
            for p in procs:
                if p.poll() is None:
                    p.kill()
        else:
            for p in procs:
                try:
                    p.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    p.kill()

    return names


def delete_debris_entities(
    entity_names: Optional[Sequence[str]] = None,
    prefix: str = "disaster_debris",
    max_count: int = 8,
    wait: bool = True,
) -> None:
    """Delete spawned rock/debris entities from Gazebo."""
    if entity_names is not None:
        targets = list(entity_names)
    else:
        targets = ["disaster_debris"] + [f"{prefix}_{i}" for i in range(1, max_count + 1)]

    procs = []
    for name in targets:
        cmd = (
            "export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=0 && "
            "source /opt/ros/humble/setup.bash && "
            f"ros2 service call /delete_entity gazebo_msgs/srv/DeleteEntity \"{{name: '{name}'}}\""
        )
        p = subprocess.Popen(
            f"bash -c '{cmd}'",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        procs.append(p)

    if wait:
        for p in procs:
            try:
                p.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                p.kill()
