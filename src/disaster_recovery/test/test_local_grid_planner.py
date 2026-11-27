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


def test_narrow_gap_stall_does_not_seal_corridor_with_virtual_barrier():
    planner = ProposedRecoveryPlanner(unstuck_window_sec=2.0, unstuck_min_disp_m=0.06)
    stack = collections.deque([Waypoint(-1.5, 0.0, 0.0), Waypoint(1.5, 0.0, 0.0)])

    # 0.30m narrow pinch visible on 2D LiDAR (walls/rock at y = +/-0.15m in robot frame, open ahead)
    xs = np.linspace(-0.5, 1.5, 30)
    pinch_scan = np.vstack([
        np.column_stack((xs, np.full_like(xs, 0.15))),
        np.column_stack((xs, np.full_like(xs, -0.15))),
    ]).astype(np.float32)

    planner.step(
        rx=0.0, ry=0.0, rth=math.pi,
        waypoint_stack=stack, start_pose=stack[0],
        scan_pts=pinch_scan, v_max=0.20, w_max=0.65,
        now_sec=0.0, wp_reach_radius=0.10, docking_dist_th=0.45,
    )
    # Trigger displacement stall inside narrow gap -> should escape WITHOUT sealing the gap
    tag_esc, _, _, _ = planner.step(
        rx=0.01, ry=0.0, rth=math.pi,
        waypoint_stack=stack, start_pose=stack[0],
        scan_pts=pinch_scan, v_max=0.20, w_max=0.65,
        now_sec=2.5, wp_reach_radius=0.10, docking_dist_th=0.45,
    )
    assert tag_esc == "ESCAPE"
    assert len(planner._virtual_obstacles) == 0
    assert len(planner.last_clearances) == 4
    assert isinstance(planner.last_used_tight_pass, bool)


def test_house_corridor_tight_gap_rock_traversal():
    """Verify TB3 passes a 0.31m corridor rock gap without chord-clipping or rock-WP deadlock."""
    planner = ProposedRecoveryPlanner(size_m=6.0, resolution_m=0.04, inflation_radius_m=0.085)
    start = Waypoint(-2.0, -0.5, 0.0)
    wps = collections.deque([
        start,
        Waypoint(0.84, 0.08, 0.0),
        Waypoint(0.57, 0.22, 0.0),
        Waypoint(0.28, 0.32, 0.0),
        Waypoint(0.03, 0.39, 0.0),   # buried inside r=0.18m rock at (0.011, 0.389)
        Waypoint(-0.26, 0.34, 0.0),
        Waypoint(-0.56, 0.36, 0.0),
    ])

    def _scan_corridor(rx: float, ry: float, rth: float) -> np.ndarray:
        pts = []
        for k in range(360):
            ang = -math.pi + 2.0 * math.pi * k / 360.0
            ga = rth + ang
            ca, sa = math.cos(ga), math.sin(ga)
            best_t = 3.5
            if sa < -1e-6:
                t = (-0.10 - ry) / sa
                hx = rx + t * ca
                if 0.05 < t < best_t and -4.27 <= hx <= 1.52:
                    best_t = t
            if sa > 1e-6:
                t = (0.85 - ry) / sa
                hx = rx + t * ca
                if 0.05 < t < best_t and -0.12 <= hx <= 2.38:
                    best_t = t
            ocx, ocy = rx - 0.011, ry - 0.389
            b = ocx * ca + ocy * sa
            c = ocx * ocx + ocy * ocy - 0.18 * 0.18
            disc = b * b - c
            if disc >= 0.0:
                t_hit = -b - math.sqrt(disc)
                if 0.05 < t_hit < best_t:
                    best_t = t_hit
            if 0.12 <= best_t <= 3.5:
                pts.append([best_t * math.cos(ang), best_t * math.sin(ang)])
        return np.array(pts, dtype=np.float32)

    rx, ry, rth = -0.60, 0.36, 0.05
    passed = False
    for t_i in range(220):
        sc_pts = _scan_corridor(rx, ry, rth)
        _, v, w, _ = planner.step(rx, ry, rth, wps, start, sc_pts, 0.20, 0.65, t_i * 0.1, 0.10, 0.50)
        rth += w * 0.1
        rx += v * math.cos(rth) * 0.1
        ry += v * math.sin(rth) * 0.1
        dxr, dyr = rx - 0.011, ry - 0.389
        dr = math.hypot(dxr, dyr)
        if dr < 0.27:
            rx = 0.011 + dxr / max(dr, 1e-6) * 0.27
            ry = 0.389 + dyr / max(dr, 1e-6) * 0.27
        if ry < -0.01:
            ry = -0.01
        if rx > -0.12 and ry > 0.76:
            ry = 0.76
        if rx > 0.45:
            passed = True
            break
    assert passed, f"Failed to traverse house corridor gap; stopped at ({rx:.2f}, {ry:.2f})"


def test_axial_sweep_soft_costmap_preemptive_lateral_detour_and_narrow_gate_preserved():
    """
    Verify Step B axial sweep-band soft costmap:
      1) In a wide gauntlet-like corridor, an axial reciprocating obstacle (|d_path . u_hat| >= 0.65)
         projects its sweep band (half-width r_obs + 0.22 m) onto soft_cost ONLY, causing A* to
         proactively detour laterally outside the sweep lane while hard_blocked is bit-identical.
      2) In a narrow corridor, hard_blocked stays untouched so A* never seals the passage.
    """
    from disaster_recovery.dynamic_tracker import TrackedObstacle

    # Wide corridor: y in [-1.25, +1.25], robot at (0.0, 0.0) heading +x toward (2.5, 0.0)
    planner = ProposedRecoveryPlanner(size_m=6.0, resolution_m=0.04, inflation_radius_m=0.085)
    xs = np.linspace(-1.0, 3.0, 100)
    wide_walls = np.vstack([
        np.column_stack((xs, np.full_like(xs, -1.25))),
        np.column_stack((xs, np.full_like(xs, 1.25))),
    ]).astype(np.float32)
    planner.observe(0.0, 0.0, 0.0, wide_walls)

    wps_wide = collections.deque([
        Waypoint(2.4, 0.0, 0.0),
        Waypoint(1.6, 0.0, 0.0),
        Waypoint(0.8, 0.0, 0.0),
    ])

    # Confirmed axial harmonic track oscillating along x around (1.20, 0.0) with A=0.93 m, omega=0.45
    axial_trk = TrackedObstacle(
        track_id=1,
        x=1.20,
        y=0.0,
        vx=0.41,
        vy=0.0,
        radius=0.21,
        P_4x4=np.diag([0.002, 0.002, 0.01, 0.01]).astype(np.float64),
        last_update_stamp=10.0,
        age=25,
        hits=25,
        is_confirmed=True,
        consecutive_hits=25,
        imm_valid=True,
        imm_u_hat=(1.0, 0.0),
        imm_p_c=(1.20, 0.0),
        imm_amp=0.93,
        imm_omega=0.45,
        imm_phi=0.0,
        imm_t_ref=10.0,
        imm_mu_cv=0.02,
        imm_mu_harm=0.98,
        imm_sigma_h0=0.03,
        imm_sigma_omega=0.02,
    )

    hard_base, soft_base = planner.grid.build_cost_maps(0.0, 0.0, waypoint_list=list(wps_wide))
    hard_dyn, soft_dyn = planner.grid.build_cost_maps(
        0.0, 0.0, dynamic_obstacles=[axial_trk], waypoint_list=list(wps_wide)
    )

    # Hard costmap MUST be bit-exact identical (soft cost ONLY)
    assert np.array_equal(hard_base, hard_dyn), "Axial sweep projection must never modify hard_blocked"

    # Soft cost along centerline of sweep segment [0.27, 2.13] must be elevated
    ix_mid, iy_mid = planner.grid.world_to_grid(1.20, 0.0)
    ix_end, iy_end = planner.grid.world_to_grid(1.95, 0.0)
    assert float(soft_dyn[ix_mid, iy_mid]) > 1.5
    assert float(soft_dyn[ix_end, iy_end]) > 1.5
    # Outside sweep band (|y| > r_obs + 0.22 = 0.43 m), soft cost stays near zero
    ix_out, iy_out = planner.grid.world_to_grid(1.20, 0.60)
    assert float(soft_dyn[ix_out, iy_out]) == 0.0

    # A* path with axial sweep soft cost must detour laterally outside sweep band
    s_ij = planner.grid.world_to_grid(0.0, 0.0)
    g_ij = planner.grid.world_to_grid(2.4, 0.0)
    path_base, _ = astar_2d(hard_base, soft_base, s_ij, g_ij, max_expansions=10000)
    path_dyn, _ = astar_2d(hard_dyn, soft_dyn, s_ij, g_ij, max_expansions=10000)
    assert path_base is not None and path_dyn is not None

    ys_base = [abs(planner.grid.grid_to_world(ix, iy)[1]) for ix, iy in path_base]
    ys_dyn = [abs(planner.grid.grid_to_world(ix, iy)[1]) for ix, iy in path_dyn]
    assert max(ys_base) < 0.20
    assert max(ys_dyn) >= 0.42, f"Expected A* lateral detour >= 0.42 m outside sweep band, got {max(ys_dyn):.3f} m"
    assert max(ys_dyn) > max(ys_base) + 0.25

    # Planner step() should also emit a laterally steered subgoal
    tag, v_cmd, w_cmd, subgoal = planner.step(
        rx=0.0, ry=0.0, rth=0.0,
        waypoint_stack=wps_wide, start_pose=wps_wide[0],
        scan_pts=wide_walls, v_max=0.20, w_max=0.65,
        now_sec=10.0, wp_reach_radius=0.10, docking_dist_th=0.45,
        dynamic_obstacles=[axial_trk],
    )
    assert tag == "ASTAR_TRACK" and subgoal is not None
    assert abs(subgoal[1]) > 0.15, f"Expected pre-emptive lateral subgoal offset, got y={subgoal[1]:.3f}"

    # 2) Narrow corridor (total width 0.64 m: y in [-0.32, +0.32]) must NEVER be sealed
    planner_narrow = ProposedRecoveryPlanner(size_m=6.0, resolution_m=0.04, inflation_radius_m=0.085)
    narrow_walls = np.vstack([
        np.column_stack((xs, np.full_like(xs, -0.32))),
        np.column_stack((xs, np.full_like(xs, 0.32))),
    ]).astype(np.float32)
    planner_narrow.observe(0.0, 0.0, 0.0, narrow_walls)
    hard_n_base, _ = planner_narrow.grid.build_cost_maps(0.0, 0.0, waypoint_list=list(wps_wide))
    hard_n_dyn, soft_n_dyn = planner_narrow.grid.build_cost_maps(
        0.0, 0.0, dynamic_obstacles=[axial_trk], waypoint_list=list(wps_wide)
    )
    assert np.array_equal(hard_n_base, hard_n_dyn)
    sn_ij = planner_narrow.grid.world_to_grid(0.0, 0.0)
    gn_ij = planner_narrow.grid.world_to_grid(2.4, 0.0)
    path_narrow, _ = astar_2d(hard_n_dyn, soft_n_dyn, sn_ij, gn_ij, max_expansions=10000)
    assert path_narrow is not None, "Soft axial sweep band must never block a narrow corridor"


def test_stepB4_axial_sweep_band_blocks_diagonal_subgoal_penetration():
    """
    Verify Step B-4 (a): in a gauntlet-like wide corridor with an axial reciprocating cylinder
    (r=0.18 m -> band_half_w=0.52 m, w_band=3.8 > max_soft=3.6), hard_blocked stays bit-identical
    while _extract_safe_subgoal refuses diagonal chords penetrating the rail and maintains >= 0.48 m
    lateral clearance across both curved bypass checks and closed-loop traversal.
    """
    from disaster_recovery.dynamic_tracker import TrackedObstacle

    def _chord_min_dist_to_rail(
        x0: float, y0: float, x1: float, y1: float, ax: float, ay: float, bx: float, by: float
    ) -> float:
        abx, aby = bx - ax, by - ay
        ab2 = max(abx * abx + aby * aby, 1e-6)
        best = float("inf")
        for a in np.linspace(0.0, 1.0, 101):
            cx = x0 + float(a) * (x1 - x0)
            cy = y0 + float(a) * (y1 - y0)
            t = min(1.0, max(0.0, ((cx - ax) * abx + (cy - ay) * aby) / ab2))
            best = min(best, math.hypot(cx - (ax + t * abx), cy - (ay + t * aby)))
        return best

    # Gauntlet top corridor: baffle_ne at y=1.50, outer north wall at y=4.30;
    # dyn_obs_1 sweeps x in [0.567, 2.433] along y=2.65 with r=0.18 m.
    xs = np.linspace(-1.0, 4.0, 125)
    world_walls = np.vstack([
        np.column_stack((xs, np.full_like(xs, 1.50))),
        np.column_stack((xs, np.full_like(xs, 4.30))),
    ]).astype(np.float32)

    def _scan_from(rx: float, ry: float, rth: float) -> np.ndarray:
        dx = world_walls[:, 0] - rx
        dy = world_walls[:, 1] - ry
        d = np.hypot(dx, dy)
        m = (d >= 0.13) & (d <= 3.5)
        ca, sa = math.cos(-rth), math.sin(-rth)
        lx = dx[m] * ca - dy[m] * sa
        ly = dx[m] * sa + dy[m] * ca
        return np.column_stack((lx, ly)).astype(np.float32)

    ax, ay, bx, by = 1.50 - 0.933, 2.65, 1.50 + 0.933, 2.65
    gauntlet_trk = TrackedObstacle(
        track_id=1,
        x=1.50,
        y=2.65,
        vx=0.42,
        vy=0.0,
        radius=0.18,
        P_4x4=np.diag([0.002, 0.002, 0.01, 0.01]).astype(np.float64),
        last_update_stamp=10.0,
        age=25,
        hits=25,
        is_confirmed=True,
        consecutive_hits=25,
        imm_valid=True,
        imm_u_hat=(1.0, 0.0),
        imm_p_c=(1.50, 2.65),
        imm_amp=0.933,
        imm_omega=0.45,
        imm_phi=0.0,
        imm_t_ref=10.0,
        imm_mu_cv=0.01,
        imm_mu_harm=0.99,
        imm_sigma_h0=0.03,
        imm_sigma_omega=0.02,
    )
    wps = [
        Waypoint(-1.0, 2.65, math.pi),
        Waypoint(0.0, 2.65, math.pi),
        Waypoint(1.2, 2.65, math.pi),
        Waypoint(2.4, 2.65, math.pi),
    ]

    # 1) Costmap checks: hard_blocked bit-identical, centerline w_band=3.8 > 3.6, core > 3.6, outside == 0
    planner = ProposedRecoveryPlanner(size_m=6.0, resolution_m=0.04, inflation_radius_m=0.085)
    rx0, ry0, rth0 = 3.05, 2.65, math.pi
    planner.observe(rx0, ry0, rth0, _scan_from(rx0, ry0, rth0))
    hard_base, _ = planner.grid.build_cost_maps(rx0, ry0, waypoint_list=wps)
    hard_dyn, soft_dyn = planner.grid.build_cost_maps(
        rx0, ry0, dynamic_obstacles=[gauntlet_trk], waypoint_list=wps
    )
    assert np.array_equal(hard_base, hard_dyn), "Axial sweep band must NEVER modify hard_blocked"
    c_ix, c_iy = planner.grid.world_to_grid(1.50, 2.65)
    assert float(soft_dyn[c_ix, c_iy]) > 3.70, f"Expected w_band ~3.8 at rail center, got {soft_dyn[c_ix, c_iy]:.3f}"
    k_ix, k_iy = planner.grid.world_to_grid(1.50, 2.65 + 0.44)
    assert float(soft_dyn[k_ix, k_iy]) > 0.80, f"Expected elevated axial band soft_cost at 0.44 m, got {soft_dyn[k_ix, k_iy]:.3f}"
    assert planner.grid._last_axial_core_mask is not None and bool(planner.grid._last_axial_core_mask[k_ix, k_iy])
    o_ix, o_iy = planner.grid.world_to_grid(1.50, 2.65 + 0.56)
    assert float(soft_dyn[o_ix, o_iy]) == 0.0, f"Expected 0 soft_cost outside 0.52 m band, got {soft_dyn[o_ix, o_iy]:.3f}"

    def _chord_min_lat_along_rail(
        x0: float, y0: float, x1: float, y1: float, ax: float, ay: float, bx: float
    ) -> float:
        best = float("inf")
        for a in np.linspace(0.0, 1.0, 101):
            cx = x0 + float(a) * (x1 - x0)
            cy = y0 + float(a) * (y1 - y0)
            if ax <= cx <= bx:
                best = min(best, abs(cy - ay))
        return best

    # 2) Direct _extract_safe_subgoal check on a curved path bending around the sweep rail body:
    #    From (2.52, 2.73) curving out to (2.52, 3.17) and west along (2.04, 3.17), a naive chord
    #    to (2.04, 3.17) slices across the rail body x in [2.04, 2.433] at dy ~ 0.16 m;
    #    _extract_safe_subgoal must reject that chord and stay >= 0.48 m clear along the rail body.
    curved_pts_w = [
        (2.52, 2.73),
        (2.52, 2.81),
        (2.52, 2.89),
        (2.52, 2.97),
        (2.52, 3.05),
        (2.52, 3.13),
        (2.48, 3.17),
        (2.40, 3.17),
        (2.32, 3.17),
        (2.24, 3.17),
        (2.16, 3.17),
        (2.08, 3.17),
        (2.00, 3.17),
    ]
    curved_ij = [planner.grid.world_to_grid(wx, wy) for wx, wy in curved_pts_w]
    sub_curved = planner._extract_safe_subgoal(curved_ij, hard_dyn, soft_dyn)
    chord_lat_curved = _chord_min_lat_along_rail(
        curved_pts_w[0][0], curved_pts_w[0][1], sub_curved[0], sub_curved[1], ax, ay, bx
    )
    assert chord_lat_curved >= 0.48, (
        f"_extract_safe_subgoal diagonally penetrated rail body: chord_lat={chord_lat_curved:.3f} m < 0.48 m, "
        f"sub={sub_curved}"
    )

    # 3) Full closed-loop rollout approaching from outside the band (3.10, 2.65) -> (0.0, 2.65):
    #    Robot path along the sweep segment x in [ax, bx] AND every outside-band subgoal chord
    #    crossing x in [ax, bx] must maintain >= 0.48 m lateral clearance without stalling.
    planner_cl = ProposedRecoveryPlanner(size_m=6.0, resolution_m=0.04, inflation_radius_m=0.085)
    rx, ry, rth = 3.10, 2.65, math.pi
    wps_cl = collections.deque([
        Waypoint(0.0, 2.65, math.pi),
        Waypoint(1.2, 2.65, math.pi),
        Waypoint(2.4, 2.65, math.pi),
    ])
    min_lateral_along_rail = float("inf")
    min_chord_along_rail = float("inf")
    reached = False
    for step_i in range(220):
        sc = _scan_from(rx, ry, rth)
        tag, v_cmd, w_cmd, sub = planner_cl.step(
            rx=rx,
            ry=ry,
            rth=rth,
            waypoint_stack=wps_cl,
            start_pose=wps_cl[0],
            scan_pts=sc,
            v_max=0.20,
            w_max=0.65,
            now_sec=10.0 + step_i * 0.10,
            wp_reach_radius=0.15,
            docking_dist_th=0.35,
            dynamic_obstacles=[gauntlet_trk],
        )
        assert tag != "ESCAPE", f"Unexpected ESCAPE at step {step_i} ({rx:.2f}, {ry:.2f})"
        if tag == "ARRIVED":
            reached = True
            break
        assert sub is not None
        if abs(ry - 2.65) >= 0.50 or rx > bx:
            cl = _chord_min_lat_along_rail(rx, ry, sub[0], sub[1], ax, ay, bx)
            if math.isfinite(cl):
                min_chord_along_rail = min(min_chord_along_rail, cl)
        if ax <= rx <= bx:
            min_lateral_along_rail = min(min_lateral_along_rail, abs(ry - 2.65))
        rth += w_cmd * 0.10
        rx += v_cmd * math.cos(rth) * 0.10
        ry += v_cmd * math.sin(rth) * 0.10
        if math.hypot(rx - 0.0, ry - 2.65) < 0.28:
            reached = True
            break

    assert reached, f"Failed to reach goal around gauntlet axial sweep; ended at ({rx:.2f}, {ry:.2f})"
    assert min_lateral_along_rail >= 0.48, (
        f"Closed-loop trajectory along rail violated >= 0.48 m lateral clearance: {min_lateral_along_rail:.3f} m"
    )
    assert min_chord_along_rail >= 0.48, (
        f"Subgoal chord along rail penetrated within 0.48 m: {min_chord_along_rail:.3f} m"
    )


def test_stepB4_inside_axial_band_clean_lateral_escape_and_narrow_corridor_unblocked():
    """
    Verify Step B-4 (b):
      1) When the robot is ALREADY inside the 0.52 m axial sweep band at detection time
         (dead center, laterally offset, or pinned against a wall on one side), _extract_safe_subgoal
         and step() never return an empty subgoal or stall/deadlock in place; the robot smoothly
         exits laterally into the clear side lane.
      2) In a 0.64 m narrow corridor, hard_blocked is untouched and both A* and closed-loop step()
         traverse cleanly without budget exhaustion or corridor sealing.
    """
    from disaster_recovery.dynamic_tracker import TrackedObstacle

    xs = np.linspace(-1.0, 4.0, 125)
    wide_walls = np.vstack([
        np.column_stack((xs, np.full_like(xs, 1.50))),
        np.column_stack((xs, np.full_like(xs, 4.30))),
    ]).astype(np.float32)

    def _scan_walls(walls_arr: np.ndarray, rx: float, ry: float, rth: float) -> np.ndarray:
        dx = walls_arr[:, 0] - rx
        dy = walls_arr[:, 1] - ry
        d = np.hypot(dx, dy)
        m = (d >= 0.13) & (d <= 3.5)
        ca, sa = math.cos(-rth), math.sin(-rth)
        return np.column_stack((dx[m] * ca - dy[m] * sa, dx[m] * sa + dy[m] * ca)).astype(np.float32)

    gauntlet_trk = TrackedObstacle(
        track_id=2,
        x=1.00,
        y=2.65,
        vx=0.40,
        vy=0.0,
        radius=0.18,
        P_4x4=np.diag([0.002, 0.002, 0.01, 0.01]).astype(np.float64),
        last_update_stamp=10.0,
        age=25,
        hits=25,
        is_confirmed=True,
        consecutive_hits=25,
        imm_valid=True,
        imm_u_hat=(1.0, 0.0),
        imm_p_c=(1.50, 2.65),
        imm_amp=0.933,
        imm_omega=0.45,
        imm_phi=0.0,
        imm_t_ref=10.0,
        imm_mu_cv=0.01,
        imm_mu_harm=0.99,
        imm_sigma_h0=0.03,
        imm_sigma_omega=0.02,
    )

    # 1) Inside-band start poses: center (dy=0.0), +0.15 m, +0.30 m, and -0.10 m with south wall at y=2.25
    asym_walls = np.vstack([
        np.column_stack((xs, np.full_like(xs, 2.25))),
        np.column_stack((xs, np.full_like(xs, 4.30))),
    ]).astype(np.float32)
    inside_cases = [
        ("center", wide_walls, 1.90, 2.65),
        ("offset_pos_015", wide_walls, 1.90, 2.80),
        ("offset_pos_030", wide_walls, 1.90, 2.95),
        ("south_wall_cross_center", asym_walls, 1.90, 2.55),
    ]
    for case_name, walls_arr, sx0, sy0 in inside_cases:
        planner = ProposedRecoveryPlanner(size_m=6.0, resolution_m=0.04, inflation_radius_m=0.085)
        rx, ry, rth = sx0, sy0, math.pi
        wps = collections.deque([
            Waypoint(0.0, 2.65, math.pi),
            Waypoint(1.0, 2.65, math.pi),
        ])
        exited_band = False
        for step_i in range(120):
            sc = _scan_walls(walls_arr, rx, ry, rth)
            tag, v_cmd, w_cmd, sub = planner.step(
                rx=rx,
                ry=ry,
                rth=rth,
                waypoint_stack=wps,
                start_pose=wps[0],
                scan_pts=sc,
                v_max=0.20,
                w_max=0.65,
                now_sec=10.0 + step_i * 0.10,
                wp_reach_radius=0.15,
                docking_dist_th=0.35,
                dynamic_obstacles=[gauntlet_trk],
            )
            assert tag in ("ASTAR_TRACK", "ROTATE_ALIGN", "ARRIVED"), (
                f"[{case_name}] Stalled/escaped inside axial band at step {step_i}: tag={tag}"
            )
            assert sub is not None, f"[{case_name}] Empty subgoal inside axial band at step {step_i}"
            assert math.hypot(sub[0] - rx, sub[1] - ry) >= 0.06, (
                f"[{case_name}] Subgoal stuck at robot position ({rx:.2f}, {ry:.2f}) -> {sub}"
            )
            if step_i == 0:
                # First subgoal must already point away from the rail centerline
                assert abs(sub[1] - 2.65) > abs(ry - 2.65) + 0.08, (
                    f"[{case_name}] Initial escape subgoal did not move laterally outward: ry={ry:.2f}, sub={sub}"
                )
            rth += w_cmd * 0.10
            rx += v_cmd * math.cos(rth) * 0.10
            ry += v_cmd * math.sin(rth) * 0.10
            if abs(ry - 2.65) >= 0.48:
                exited_band = True
                break
        assert exited_band, f"[{case_name}] Failed to exit 0.52 m axial band cleanly; ended at ({rx:.2f}, {ry:.2f})"

    # 2) Narrow corridor (0.64 m wide: y in [2.33, 2.97]) must NEVER be sealed by w_band or band_half_w
    narrow_walls = np.vstack([
        np.column_stack((xs, np.full_like(xs, 2.33))),
        np.column_stack((xs, np.full_like(xs, 2.97))),
    ]).astype(np.float32)
    planner_n = ProposedRecoveryPlanner(size_m=6.0, resolution_m=0.04, inflation_radius_m=0.085)
    rx, ry, rth = 2.80, 2.65, math.pi
    wps_n = collections.deque([
        Waypoint(0.20, 2.65, math.pi),
        Waypoint(1.40, 2.65, math.pi),
    ])
    planner_n.observe(rx, ry, rth, _scan_walls(narrow_walls, rx, ry, rth))
    h_base, _ = planner_n.grid.build_cost_maps(rx, ry, waypoint_list=list(wps_n))
    h_dyn, s_dyn = planner_n.grid.build_cost_maps(
        rx, ry, dynamic_obstacles=[gauntlet_trk], waypoint_list=list(wps_n)
    )
    assert np.array_equal(h_base, h_dyn), "Narrow corridor hard_blocked must stay bit-identical"
    path_n, _ = astar_2d(
        h_dyn,
        s_dyn,
        planner_n.grid.world_to_grid(rx, ry),
        planner_n.grid.world_to_grid(0.40, 2.65),
        max_expansions=10000,
    )
    assert path_n is not None and len(path_n) > 0, "A* must traverse 0.64 m narrow corridor within budget"

    traversed_narrow = False
    for step_i in range(180):
        sc = _scan_walls(narrow_walls, rx, ry, rth)
        tag, v_cmd, w_cmd, sub = planner_n.step(
            rx=rx,
            ry=ry,
            rth=rth,
            waypoint_stack=wps_n,
            start_pose=wps_n[0],
            scan_pts=sc,
            v_max=0.20,
            w_max=0.65,
            now_sec=10.0 + step_i * 0.10,
            wp_reach_radius=0.15,
            docking_dist_th=0.35,
            dynamic_obstacles=[gauntlet_trk],
        )
        assert tag != "ESCAPE", f"Narrow corridor sealed/escaped at step {step_i} ({rx:.2f}, {ry:.2f})"
        if tag == "ARRIVED" or rx < 0.55:
            traversed_narrow = True
            break
        rth += w_cmd * 0.10
        rx += v_cmd * math.cos(rth) * 0.10
        ry += v_cmd * math.sin(rth) * 0.10
    assert traversed_narrow, f"Failed to traverse narrow corridor; ended at ({rx:.2f}, {ry:.2f})"


def test_empty_semantic_lists_identical_to_none_and_zero_overhead():
    """Verify semantic_obstacles=[] and semantic_hazards=[] match None bit-for-bit."""
    pts = np.column_stack((np.full(15, 1.15), np.linspace(-0.20, 0.20, 15))).astype(np.float32)

    grid_none = RollingLocalGrid(size_m=6.0, resolution_m=0.10, inflation_radius_m=0.15)
    grid_empty = RollingLocalGrid(size_m=6.0, resolution_m=0.10, inflation_radius_m=0.15)
    grid_none.update_scan(0.0, 0.0, 0.0, pts)
    grid_empty.update_scan(0.0, 0.0, 0.0, pts)

    hb_none, sc_none = grid_none.build_cost_maps(
        0.0, 0.0, semantic_obstacles=None, semantic_hazards=None
    )
    hb_empty, sc_empty = grid_empty.build_cost_maps(
        0.0, 0.0, semantic_obstacles=[], semantic_hazards=[]
    )
    assert np.array_equal(hb_none, hb_empty)
    assert np.array_equal(sc_none, sc_empty)

    planner_none = ProposedRecoveryPlanner()
    planner_empty = ProposedRecoveryPlanner()
    planner_none.replan_period_sec = 0.0
    planner_none.replan_displacement_m = 0.0
    planner_empty.replan_period_sec = 0.0
    planner_empty.replan_displacement_m = 0.0
    wps_none = collections.deque([Waypoint(0.0, 0.0, 0.0), Waypoint(1.2, 0.0, 0.0), Waypoint(2.2, 0.0, 0.0)])
    wps_empty = collections.deque([Waypoint(0.0, 0.0, 0.0), Waypoint(1.2, 0.0, 0.0), Waypoint(2.2, 0.0, 0.0)])

    res_none = planner_none.step(
        rx=2.0, ry=0.0, rth=math.pi,
        waypoint_stack=wps_none, start_pose=wps_none[0], scan_pts=pts,
        v_max=0.20, w_max=0.65, now_sec=5.0, wp_reach_radius=0.15, docking_dist_th=0.35,
        semantic_obstacles=None, semantic_hazards=None,
    )
    res_empty = planner_empty.step(
        rx=2.0, ry=0.0, rth=math.pi,
        waypoint_stack=wps_empty, start_pose=wps_empty[0], scan_pts=pts,
        v_max=0.20, w_max=0.65, now_sec=5.0, wp_reach_radius=0.15, docking_dist_th=0.35,
        semantic_obstacles=[], semantic_hazards=[],
    )
    assert res_none == res_empty
    assert planner_none.last_path_ij == planner_empty.last_path_ij
    assert np.array_equal(planner_none._astar_cache_maps[0], planner_empty._astar_cache_maps[0])
    assert np.array_equal(planner_none._astar_cache_maps[1], planner_empty._astar_cache_maps[1])




