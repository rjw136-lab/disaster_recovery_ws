#!/usr/bin/env python3
"""
Unit tests for Proposed Layer-2 Planner:
- RollingLocalGrid ray-carving, recentering, and C-space inflation
- 8-connected 2D A* obstacle-detour planning around concave walls
- Unstuck reverse-and-pivot escape trigger
- Topological breadcrumb shortcutting & safe arrival handoff
"""

import collections
import math
import numpy as np

from disaster_recovery.local_grid_planner import (
    RollingLocalGrid,
    ProposedRecoveryPlanner,
    astar_2d,
)
from disaster_recovery.recovery_controller import Waypoint


def test_rolling_grid_recenter_and_ray_carving():
    grid = RollingLocalGrid(size_m=6.0, resolution_m=0.10, inflation_radius_m=0.15)
    pts = np.column_stack((np.full(20, 1.25), np.linspace(-0.25, 0.25, 20))).astype(np.float32)
    grid.update_scan(0.0, 0.0, 0.0, pts)

    ix, iy = grid.world_to_grid(1.25, 0.05)
    assert grid.in_bounds(ix, iy)
    assert grid.occ[ix, iy] > 0.8

    # Move robot 1.5m along +y and ensure recentering preserves obstacle location
    grid.update_scan(0.0, 1.5, 0.0, None)
    ix2, iy2 = grid.world_to_grid(1.25, 0.05)
    assert grid.in_bounds(ix2, iy2)
    assert grid.occ[ix2, iy2] > 0.6


def test_astar_2d_detours_u_shaped_barrier():
    grid = RollingLocalGrid(size_m=6.0, resolution_m=0.10, inflation_radius_m=0.15)
    grid._recenter(0.0, 0.0)

    # Build a concave barrier blocking direct path from (0, 0) to (1.8, 0)
    barrier_x = np.concatenate([
        np.full(15, 0.9),
        np.linspace(0.3, 0.9, 10),
        np.linspace(0.3, 0.9, 10),
    ])
    barrier_y = np.concatenate([
        np.linspace(-0.7, 0.7, 15),
        np.full(10, 0.7),
        np.full(10, -0.7),
    ])
    pts = np.column_stack((barrier_x, barrier_y)).astype(np.float32)
    grid.update_scan(0.0, 0.0, 0.0, pts)

    hard_blocked, soft_cost = grid.build_cost_maps(0.0, 0.0)
    s_ij = grid.world_to_grid(0.0, 0.0)
    g_ij = grid.world_to_grid(1.8, 0.0)

    path, cost = astar_2d(hard_blocked, soft_cost, s_ij, g_ij)
    assert path is not None and cost < float("inf")
    assert path[0] == s_ij and path[-1] == g_ij
    # Verify path physically routes outside y = +/-0.7 barrier
    world_ys = [grid.grid_to_world(ix, iy)[1] for ix, iy in path]
    assert max(abs(y) for y in world_ys) > 0.75


def test_planner_blocks_forward_ramming_and_triggers_unstuck_escape():
    planner = ProposedRecoveryPlanner(unstuck_window_sec=2.0, unstuck_min_disp_m=0.06)
    stack = collections.deque([Waypoint(0.0, 0.0, 0.0), Waypoint(2.0, 0.0, 0.0)])
    start_pose = stack[0]

    # Obstacle wall dead ahead at 0.16m (left side slightly wider than right)
    wall_y = np.linspace(-0.35, 0.10, 15)
    wall_pts = np.column_stack((np.full_like(wall_y, 0.16), wall_y)).astype(np.float32)

    tag, v, w, _ = planner.step(
        rx=1.5, ry=0.0, rth=0.0,
        waypoint_stack=stack, start_pose=start_pose,
        scan_pts=wall_pts, v_max=0.20, w_max=0.65,
        now_sec=10.0, wp_reach_radius=0.10, docking_dist_th=0.45,
    )
    assert v <= 0.0  # Must NEVER command positive velocity into 0.16m obstacle

    # Simulate stalling at the same pose across unstuck window -> triggers ESCAPE
    tag2, v2, w2, _ = planner.step(
        rx=1.51, ry=0.0, rth=0.0,
        waypoint_stack=stack, start_pose=start_pose,
        scan_pts=wall_pts, v_max=0.20, w_max=0.65,
        now_sec=12.5, wp_reach_radius=0.10, docking_dist_th=0.45,
    )
    assert tag2 == "ESCAPE"
    assert v2 < 0.0  # Clear rear -> backs out smoothly


def test_planner_shortcut_and_arrival_handoff():
    planner = ProposedRecoveryPlanner()
    # Straight open hallway breadcrumbs from (0,0) to (2.5,0)
    stack = collections.deque([
        Waypoint(0.0, 0.0, 0.0),
        Waypoint(0.8, 0.0, 0.0),
        Waypoint(1.6, 0.0, 0.0),
        Waypoint(2.4, 0.0, 0.0),
    ])
    start_pose = stack[0]

    # Clear hallway walls on sides at y=+/-1.2m so rays carve observed free space
    side_x = np.linspace(-2.5, 1.5, 25)
    scan_pts = np.vstack([
        np.column_stack((side_x, np.full_like(side_x, 1.2))),
        np.column_stack((side_x, np.full_like(side_x, -1.2))),
        np.column_stack((np.full(10, 3.0), np.linspace(-1.0, 1.0, 10))),
    ]).astype(np.float32)

    # Robot facing -pi (toward base camp (0,0)) from (2.2, 0.0)
    tag, v, w, subgoal = planner.step(
        rx=2.2, ry=0.0, rth=math.pi,
        waypoint_stack=stack, start_pose=start_pose,
        scan_pts=scan_pts, v_max=0.20, w_max=0.65,
        now_sec=1.0, wp_reach_radius=0.10, docking_dist_th=0.45,
    )
    assert tag == "ASTAR_TRACK"
    assert v > 0.15
    assert subgoal is not None and subgoal[0] < 2.2

    # Near base camp -> ARRIVED
    stack_home = collections.deque([start_pose])
    tag_end, v_end, _, _ = planner.step(
        rx=0.25, ry=0.0, rth=math.pi,
        waypoint_stack=stack_home, start_pose=start_pose,
        scan_pts=scan_pts, v_max=0.20, w_max=0.65,
        now_sec=5.0, wp_reach_radius=0.10, docking_dist_th=0.45,
    )
    assert tag_end == "ARRIVED"
    assert v_end == 0.0


def test_narrow_indoor_gap_beside_rock_and_cross_wall_guard():
    planner = ProposedRecoveryPlanner()  # defaults: 6.6m, 0.05m res, 0.10m inflation

    # 0.85m wide corridor (y=-0.10..0.75) with a 0.18m-radius rock at (0.0, 0.42)
    # Bottom gap = (0.42-0.18) - (-0.10) = 0.34m; test even tighter 0.28m gap (y=-0.04..0.75, bottom=0.28m)
    xs = np.linspace(-1.5, 1.5, 80)
    angs = np.linspace(-math.pi, math.pi, 48, endpoint=False)
    pts_world = np.vstack([
        np.column_stack((xs, np.full_like(xs, -0.04))),
        np.column_stack((xs, np.full_like(xs, 0.75))),
        np.column_stack((0.18 * np.cos(angs), 0.42 + 0.18 * np.sin(angs))),
    ])
    # Convert to robot frame at (-0.6, 0.12, yaw=0)
    pts_robot = (pts_world - np.array([-0.6, 0.12])).astype(np.float32)
    planner.observe(-0.6, 0.12, 0.0, pts_robot)

    hard, soft = planner.grid.build_cost_maps(-0.6, 0.12)
    stack = [Waypoint(-2.0, -0.5, 0.0)] + [Waypoint(-0.8 + 0.35 * k, 0.25, 0.0) for k in range(6)]
    path, chosen_idx = planner._plan_topological_astar(-0.6, 0.12, stack, hard, soft)
    assert path is not None and chosen_idx is not None
    assert chosen_idx != 0  # Must NOT jump/fallback blindly across room wall to idx=0


def test_front_pin_emits_negative_unpin_velocity():
    planner = ProposedRecoveryPlanner()
    stack = collections.deque([Waypoint(0.0, 0.0, 0.0), Waypoint(2.0, 0.0, 0.0)])
    # Wall at 0.12m directly in front while rear is completely open
    wall_y = np.linspace(-0.20, 0.20, 15)
    scan_pts = np.column_stack((np.full_like(wall_y, 0.12), wall_y)).astype(np.float32)
    tag, v, w, _ = planner.step(
        rx=1.5, ry=0.0, rth=0.0,
        waypoint_stack=stack, start_pose=stack[0],
        scan_pts=scan_pts, v_max=0.20, w_max=0.65,
        now_sec=1.0, wp_reach_radius=0.10, docking_dist_th=0.45,
    )
    assert tag in ("ROTATE_ALIGN", "ESCAPE")
    assert v < 0.0
    assert abs(w) > 0.1


def test_sub_lidar_plinth_virtual_obstacle_detour():
    planner = ProposedRecoveryPlanner()
    stack = collections.deque([Waypoint(0.0, 2.0, 0.0), Waypoint(0.0, 0.2, 0.0)])
    # Open 2D LiDAR (everything > 0.5m)
    angs = np.linspace(-math.pi, math.pi, 36, endpoint=False)
    open_scan = np.column_stack((1.8 * np.cos(angs), 1.8 * np.sin(angs))).astype(np.float32)

    # First forward tick records heading dir (facing +y, yaw=pi/2)
    tag1, v1, _, _ = planner.step(
        rx=0.0, ry=0.0, rth=math.pi / 2.0,
        waypoint_stack=stack, start_pose=stack[0],
        scan_pts=open_scan, v_max=0.20, w_max=0.65,
        now_sec=0.0, wp_reach_radius=0.10, docking_dist_th=0.45,
    )
    assert tag1 == "ASTAR_TRACK" and v1 > 0.05

    # Simulate physical stall (no displacement after unstuck_window_sec)
    tag_esc, v_esc, _, _ = planner.step(
        rx=0.01, ry=0.01, rth=math.pi / 2.0,
        waypoint_stack=stack, start_pose=stack[0],
        scan_pts=open_scan, v_max=0.20, w_max=0.65,
        now_sec=3.0, wp_reach_radius=0.10, docking_dist_th=0.45,
    )
    assert tag_esc == "ESCAPE" and v_esc < 0.0

    # After escape finishes, robot has backed up slightly and should detour laterally around virtual barrier
    tag_detour, _, _, subgoal = planner.step(
        rx=0.01, ry=-0.12, rth=math.pi / 2.0,
        waypoint_stack=stack, start_pose=stack[0],
        scan_pts=open_scan, v_max=0.20, w_max=0.65,
        now_sec=5.5, wp_reach_radius=0.10, docking_dist_th=0.45,
    )
    assert tag_detour in ("ASTAR_TRACK", "ROTATE_ALIGN")
    assert subgoal is not None and abs(subgoal[0]) > 0.10

