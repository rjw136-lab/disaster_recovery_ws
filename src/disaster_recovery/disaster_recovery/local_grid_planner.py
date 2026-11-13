#!/usr/bin/env python3
"""
Rolling Local Occupancy Grid + 2D A* Subgoal Planner + Topological Breadcrumb
Shortcut + Progress Watchdog Unstuck Escape for Disaster Autonomous Recovery.

Designed as an upgraded Layer-2 planner for TurtleBot3 Burger:
- Replaces blind reactive backtracking + deadlocked hybrid clamping with
  short-horizon metric pathfinding (A* on an 8.0m x 8.0m rolling 2D grid).
- Dynamically clears transient/moving obstacles via LiDAR ray-carving + decay.
- Jumps ahead along the breadcrumb LIFO stack whenever an earlier (closer-to-home)
  waypoint is demonstrably reachable in observed free space.
"""

import heapq
import math
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _build_disk_offsets(radius_cells: int) -> List[Tuple[int, int, float]]:
    """Precompute relative (di, dj, dist_cells) inside Euclidean radius."""
    offsets: List[Tuple[int, int, float]] = []
    r2 = radius_cells * radius_cells
    for di in range(-radius_cells, radius_cells + 1):
        for dj in range(-radius_cells, radius_cells + 1):
            d2 = di * di + dj * dj
            if d2 <= r2:
                offsets.append((di, dj, math.sqrt(d2)))
    return offsets


class RollingLocalGrid:
    """World-aligned rolling 2D occupancy grid centered near the robot."""

    def __init__(
        self,
        size_m: float = 8.0,
        resolution_m: float = 0.10,
        inflation_radius_m: float = 0.16,
        soft_radius_m: float = 0.28,
    ):
        self.size_m = float(size_m)
        self.res = float(resolution_m)
        self.n = int(round(self.size_m / self.res))
        self.inflation_radius_m = float(inflation_radius_m)
        self.soft_radius_m = float(soft_radius_m)

        self.occ = np.zeros((self.n, self.n), dtype=np.float32)
        self.observed_free = np.zeros((self.n, self.n), dtype=np.float32)
        self.initialized = False
        self.origin_ix = 0
        self.origin_iy = 0

        hard_cells = max(1, int(math.ceil(self.inflation_radius_m / self.res)))
        soft_cells = max(hard_cells + 1, int(math.ceil(self.soft_radius_m / self.res)))
        self._hard_offsets = [
            (di, dj, d_cell)
            for (di, dj, d_cell) in _build_disk_offsets(hard_cells)
            if (d_cell * self.res) <= self.inflation_radius_m + 1e-6
        ] or [(0, 0, 0.0)]
        self._soft_offsets = [
            (di, dj, d_cell)
            for (di, dj, d_cell) in _build_disk_offsets(soft_cells)
            if (d_cell * self.res) <= self.soft_radius_m + 1e-6
        ]
        self._soft_cells = max(1.0, self.soft_radius_m / self.res)

    def reset(self) -> None:
        self.occ.fill(0.0)
        self.observed_free.fill(0.0)
        self.initialized = False
        self.origin_ix = 0
        self.origin_iy = 0

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        ix = int(math.floor(wx / self.res)) - self.origin_ix
        iy = int(math.floor(wy / self.res)) - self.origin_iy
        return ix, iy

    def grid_to_world(self, ix: int, iy: int) -> Tuple[float, float]:
        wx = (self.origin_ix + ix + 0.5) * self.res
        wy = (self.origin_iy + iy + 0.5) * self.res
        return wx, wy

    def in_bounds(self, ix: int, iy: int, margin: int = 0) -> bool:
        return margin <= ix < (self.n - margin) and margin <= iy < (self.n - margin)

    def _recenter(self, rx: float, ry: float) -> None:
        desired_ox = int(math.floor(rx / self.res)) - self.n // 2
        desired_oy = int(math.floor(ry / self.res)) - self.n // 2
        if not self.initialized:
            self.origin_ix = desired_ox
            self.origin_iy = desired_oy
            self.initialized = True
            return

        dx = desired_ox - self.origin_ix
        dy = desired_oy - self.origin_iy
        if abs(dx) < 2 and abs(dy) < 2:
            return

        if abs(dx) >= self.n or abs(dy) >= self.n:
            self.occ.fill(0.0)
            self.observed_free.fill(0.0)
        else:
            new_occ = np.zeros_like(self.occ)
            new_free = np.zeros_like(self.observed_free)

            src_x0 = max(0, dx)
            src_x1 = min(self.n, self.n + dx)
            src_y0 = max(0, dy)
            src_y1 = min(self.n, self.n + dy)

            dst_x0 = src_x0 - dx
            dst_x1 = src_x1 - dx
            dst_y0 = src_y0 - dy
            dst_y1 = src_y1 - dy

            new_occ[dst_x0:dst_x1, dst_y0:dst_y1] = self.occ[src_x0:src_x1, src_y0:src_y1]
            new_free[dst_x0:dst_x1, dst_y0:dst_y1] = self.observed_free[src_x0:src_x1, src_y0:src_y1]
            self.occ = new_occ
            self.observed_free = new_free

        self.origin_ix = desired_ox
        self.origin_iy = desired_oy

    def update_scan(
        self,
        rx: float,
        ry: float,
        rth: float,
        scan_pts_robot: Optional[np.ndarray],
        raw_ranges: Optional[Sequence[float]] = None,
    ) -> None:
        """Update occupancy & observed free-space from robot-frame 2D scan points."""
        self._recenter(rx, ry)

        # Slow temporal decay keeps mapped indoor walls across rotations, while
        # ray-carving below clears transient/moving obstacles within 1-2 ticks.
        self.occ *= 0.965
        self.observed_free *= 0.985

        if scan_pts_robot is None or len(scan_pts_robot) == 0:
            return

        c = math.cos(rth)
        s = math.sin(rth)

        # Transform robot-frame scan hits to world frame
        pts = np.asarray(scan_pts_robot, dtype=np.float32)
        wx = rx + pts[:, 0] * c - pts[:, 1] * s
        wy = ry + pts[:, 0] * s + pts[:, 1] * c

        # Ray-carve free space between robot center and each hit (~88% of range)
        dists = np.hypot(pts[:, 0], pts[:, 1])
        valid_ray = dists > 0.16
        if np.any(valid_ray):
            wx_r = wx[valid_ray]
            wy_r = wy[valid_ray]
            d_r = dists[valid_ray]
            max_steps = int(math.ceil(min(3.5, float(np.max(d_r))) / (self.res * 0.9)))
            max_steps = max(2, min(max_steps, 40))
            fracs = np.linspace(0.08, 0.88, max_steps, dtype=np.float32)[:, None]  # (K, 1)

            ray_x = rx + fracs * (wx_r[None, :] - rx)
            ray_y = ry + fracs * (wy_r[None, :] - ry)
            ix_f = np.floor(ray_x / self.res).astype(np.int32) - self.origin_ix
            iy_f = np.floor(ray_y / self.res).astype(np.int32) - self.origin_iy

            mask = (ix_f >= 0) & (ix_f < self.n) & (iy_f >= 0) & (iy_f < self.n)
            ix_v = ix_f[mask]
            iy_v = iy_f[mask]
            self.observed_free[ix_v, iy_v] = 1.0
            self.occ[ix_v, iy_v] *= 0.40

        # Mark robot center cell as observed free
        sx, sy = self.world_to_grid(rx, ry)
        if self.in_bounds(sx, sy):
            self.observed_free[sx, sy] = 1.0
            self.occ[sx, sy] *= 0.25

        # Mark hit endpoints as occupied (never mark solid walls/obstacles as free)
        hit_mask = dists >= 0.13
        if np.any(hit_mask):
            ix_h = np.floor(wx[hit_mask] / self.res).astype(np.int32) - self.origin_ix
            iy_h = np.floor(wy[hit_mask] / self.res).astype(np.int32) - self.origin_iy
            inb = (ix_h >= 0) & (ix_h < self.n) & (iy_h >= 0) & (iy_h < self.n)
            self.occ[ix_h[inb], iy_h[inb]] = 1.0
            self.observed_free[ix_h[inb], iy_h[inb]] = 0.0

    def build_cost_maps(
        self,
        rx: float,
        ry: float,
        extra_world_pts: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (hard_blocked bool[N,N], soft_cost float32[N,N])."""
        raw_occ = self.occ >= 0.45
        if extra_world_pts:
            raw_occ = raw_occ.copy()
            for vx, vy in extra_world_pts:
                vix, viy = self.world_to_grid(vx, vy)
                if self.in_bounds(vix, viy):
                    raw_occ[vix, viy] = True
        hard_blocked = np.zeros((self.n, self.n), dtype=bool)
        soft_cost = np.zeros((self.n, self.n), dtype=np.float32)

        occ_idx = np.argwhere(raw_occ)
        if len(occ_idx) > 0:
            ox = occ_idx[:, 0]
            oy = occ_idx[:, 1]
            for di, dj, d_cell in self._soft_offsets:
                nx = ox + di
                ny = oy + dj
                valid = (nx >= 0) & (nx < self.n) & (ny >= 0) & (ny < self.n)
                if not np.any(valid):
                    continue
                nxv = nx[valid]
                nyv = ny[valid]
                penalty = max(0.3, 5.5 * (1.0 - d_cell / max(1.0, float(self._soft_cells))))
                soft_cost[nxv, nyv] = np.maximum(soft_cost[nxv, nyv], penalty)

            for di, dj, _ in self._hard_offsets:
                nx = ox + di
                ny = oy + dj
                valid = (nx >= 0) & (nx < self.n) & (ny >= 0) & (ny < self.n)
                if np.any(valid):
                    hard_blocked[nx[valid], ny[valid]] = True

        # Safeguard: if the robot itself slightly brushes an inflated boundary,
        # downgrade inflated cells within 0.22m of robot center (keeping raw hits blocked)
        # so A* can always escape outward into clear space.
        sx, sy = self.world_to_grid(rx, ry)
        if self.in_bounds(sx, sy):
            guard_r = max(1, int(math.ceil(min(0.22, self.inflation_radius_m + 0.08) / self.res)))
            for di in range(-guard_r, guard_r + 1):
                for dj in range(-guard_r, guard_r + 1):
                    gx, gy = sx + di, sy + dj
                    if self.in_bounds(gx, gy) and not raw_occ[gx, gy]:
                        if di * di + dj * dj <= guard_r * guard_r:
                            hard_blocked[gx, gy] = False
                            soft_cost[gx, gy] = max(float(soft_cost[gx, gy]), 5.0)
            hard_blocked[sx, sy] = False

        return hard_blocked, soft_cost


_NEIGHBORS_8 = (
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, 1.41421356),
    (1, -1, 1.41421356),
    (-1, 1, 1.41421356),
    (-1, -1, 1.41421356),
)


def astar_2d(
    hard_blocked: np.ndarray,
    soft_cost: np.ndarray,
    start_ij: Tuple[int, int],
    goal_ij: Tuple[int, int],
    max_expansions: int = 3500,
) -> Tuple[Optional[List[Tuple[int, int]]], float]:
    """8-connected 2D A* planner avoiding corners and soft obstacle halos."""
    n_x, n_y = hard_blocked.shape
    sx, sy = start_ij
    gx, gy = goal_ij

    if not (0 <= sx < n_x and 0 <= sy < n_y and 0 <= gx < n_x and 0 <= gy < n_y):
        return None, float("inf")
    if hard_blocked[gx, gy]:
        return None, float("inf")
    if (sx, sy) == (gx, gy):
        return [(sx, sy)], 0.0

    def h(ix: int, iy: int) -> float:
        dx = abs(ix - gx)
        dy = abs(iy - gy)
        return max(dx, dy) + 0.41421356 * min(dx, dy)

    g_score: Dict[Tuple[int, int], float] = {(sx, sy): 0.0}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    heap: List[Tuple[float, float, int, int]] = [(h(sx, sy), 0.0, sx, sy)]
    closed = set()
    expansions = 0

    while heap and expansions < max_expansions:
        _, cur_g, cx, cy = heapq.heappop(heap)
        if (cx, cy) in closed:
            continue
        if (cx, cy) == (gx, gy):
            path = [(gx, gy)]
            cur = (gx, gy)
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path, cur_g

        closed.add((cx, cy))
        expansions += 1

        for di, dj, step_cost in _NEIGHBORS_8:
            nx, ny = cx + di, cy + dj
            if not (0 <= nx < n_x and 0 <= ny < n_y):
                continue
            if (nx, ny) in closed or hard_blocked[nx, ny]:
                continue
            # Prevent diagonal corner clipping across blocked edges
            if di != 0 and dj != 0:
                if hard_blocked[cx + di, cy] or hard_blocked[cx, cy + dj]:
                    continue

            tentative_g = cur_g + step_cost + float(soft_cost[nx, ny])
            prev_best = g_score.get((nx, ny))
            if prev_best is None or tentative_g < prev_best:
                g_score[(nx, ny)] = tentative_g
                came_from[(nx, ny)] = (cx, cy)
                f_val = tentative_g + h(nx, ny)
                heapq.heappush(heap, (f_val, tentative_g, nx, ny))

    return None, float("inf")


class ProposedRecoveryPlanner:
    """
    Integrated Layer-2 planner:
      1. Rolling Local Occupancy Grid + 2D A* detour planning
      2. Topological Breadcrumb Shortcut selection
      3. Progress-watchdog unstuck backup + widest-gap pivot escape
      4. Clearance-aware subgoal velocity regulation
    """

    def __init__(
        self,
        size_m: float = 6.0,
        resolution_m: float = 0.04,
        inflation_radius_m: float = 0.085,
        lookahead_m: float = 0.50,
        shortcut_rejoin_dist_m: float = 3.2,
        unstuck_window_sec: float = 2.8,
        unstuck_min_disp_m: float = 0.06,
    ):
        self.grid = RollingLocalGrid(
            size_m=size_m,
            resolution_m=resolution_m,
            inflation_radius_m=inflation_radius_m,
        )
        self.lookahead_m = float(lookahead_m)
        self.shortcut_rejoin_dist_m = float(shortcut_rejoin_dist_m)
        self.unstuck_window_sec = float(unstuck_window_sec)
        self.unstuck_min_disp_m = float(unstuck_min_disp_m)

        # Unstuck & shortcut cooldown state
        self._anchor_time: Optional[float] = None
        self._anchor_x: float = 0.0
        self._anchor_y: float = 0.0
        self._escape_until: float = 0.0
        self._escape_start: float = 0.0
        self._escape_reverse_dur: float = 0.90
        self._escape_turn_dir: float = 1.0
        self._front_block_since: Optional[float] = None
        self._last_chosen_idx: Optional[int] = None
        self._no_shortcut_until: float = 0.0
        self._virtual_obstacles: List[Tuple[float, float, float]] = []
        self._last_fwd_pose_dir: Optional[float] = None
        self._last_escape_xy: Optional[Tuple[float, float]] = None
        self._repeat_escape_count: int = 0
        self.escape_count: int = 0
        self.shortcut_count: int = 0
        self.last_subgoal: Optional[Tuple[float, float]] = None

    def observe(
        self,
        rx: float,
        ry: float,
        rth: float,
        scan_pts: Optional[np.ndarray],
    ) -> None:
        """Continuously map nearby walls/free-space during outbound exploration."""
        self.grid.update_scan(rx, ry, rth, scan_pts)

    def reset(self) -> None:
        self.grid.reset()
        self._anchor_time = None
        self._anchor_x = 0.0
        self._anchor_y = 0.0
        self._escape_until = 0.0
        self._escape_start = 0.0
        self._escape_reverse_dur = 0.90
        self._escape_turn_dir = 1.0
        self._front_block_since = None
        self._last_chosen_idx = None
        self._no_shortcut_until = 0.0
        self._virtual_obstacles.clear()
        self._last_fwd_pose_dir = None
        self._last_escape_xy = None
        self._repeat_escape_count = 0
        self.escape_count = 0
        self.shortcut_count = 0
        self.last_subgoal = None

    @staticmethod
    def _scan_sector_clearances(scan_pts: Optional[np.ndarray]) -> Tuple[float, float, float, float]:
        """Return (front_corridor_min, left_min, right_min, rear_min) in meters."""
        if scan_pts is None or len(scan_pts) == 0:
            return 3.5, 3.5, 3.5, 3.5

        pts = np.asarray(scan_pts, dtype=np.float32)
        x = pts[:, 0]
        y = pts[:, 1]
        r = np.hypot(x, y)
        valid = (r >= 0.10) & (r <= 3.5)
        if not np.any(valid):
            return 3.5, 3.5, 3.5, 3.5

        xv, yv, rv = x[valid], y[valid], r[valid]
        ang = np.arctan2(yv, xv)

        # Front corridor tightly matching TB3 half-width (0.089m + margin) + immediate front cone
        corridor = (xv > 0.0) & (xv < 1.2) & (np.abs(yv) <= 0.12)
        front_cone = np.abs(ang) <= math.radians(18.0)
        left = (ang > math.radians(24.0)) & (ang <= math.radians(95.0))
        right = (ang < -math.radians(24.0)) & (ang >= -math.radians(95.0))
        rear = np.abs(ang) >= math.radians(145.0)

        f_corr = float(np.min(xv[corridor])) if np.any(corridor) else 3.5
        f_cone = float(np.min(rv[front_cone])) if np.any(front_cone) else 3.5
        front_min = min(f_corr, f_cone)
        left_min = float(np.min(rv[left])) if np.any(left) else 3.5
        right_min = float(np.min(rv[right])) if np.any(right) else 3.5
        rear_min = float(np.min(rv[rear])) if np.any(rear) else 3.5
        return front_min, left_min, right_min, rear_min

    def _nearest_free_cell(
        self, hard_blocked: np.ndarray, target_ij: Tuple[int, int], max_radius_cells: int = 8
    ) -> Optional[Tuple[int, int]]:
        """Find nearest non-blocked cell to target_ij inside the grid."""
        gx = max(1, min(self.grid.n - 2, target_ij[0]))
        gy = max(1, min(self.grid.n - 2, target_ij[1]))
        if not hard_blocked[gx, gy]:
            return (gx, gy)

        best_ij = None
        best_d2 = 1e9
        for r in range(1, max_radius_cells + 1):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    nx, ny = gx + di, gy + dj
                    if self.grid.in_bounds(nx, ny, margin=1) and not hard_blocked[nx, ny]:
                        d2 = di * di + dj * dj
                        if d2 < best_d2:
                            best_d2 = d2
                            best_ij = (nx, ny)
            if best_ij is not None:
                return best_ij
        return None

    def _project_into_grid(self, rx: float, ry: float, wx: float, wy: float) -> Tuple[int, int]:
        """Clip a distant waypoint onto the interior boundary of the rolling grid."""
        half = 0.5 * self.grid.size_m - 0.35
        dx = wx - rx
        dy = wy - ry
        scale = 1.0
        if abs(dx) > half:
            scale = min(scale, half / abs(dx))
        if abs(dy) > half:
            scale = min(scale, half / abs(dy))
        px = rx + dx * scale
        py = ry + dy * scale
        ix, iy = self.grid.world_to_grid(px, py)
        ix = max(1, min(self.grid.n - 2, ix))
        iy = max(1, min(self.grid.n - 2, iy))
        return ix, iy

    def _plan_topological_astar(
        self,
        rx: float,
        ry: float,
        waypoint_list: list,
        hard_blocked: np.ndarray,
        soft_cost: np.ndarray,
        allow_shortcuts: bool = True,
    ) -> Tuple[Optional[List[Tuple[int, int]]], Optional[int]]:
        """
        Evaluate breadcrumbs from earliest (closest to base camp, idx 0) up to latest,
        selecting the earliest safe reachable breadcrumb inside the local grid.
        Returns (grid_path, selected_stack_index) or (None, None).
        """
        if not waypoint_list:
            return None, None

        start_ij = self.grid.world_to_grid(rx, ry)
        m = len(waypoint_list)

        if not allow_shortcuts:
            # Cooldown active: strictly follow immediate trailing breadcrumbs in reverse
            candidates = sorted({max(0, m - 1 - k) for k in range(min(4, m))}, reverse=True)
        else:
            # Sample candidate indices preferring earliest breadcrumbs in local box
            in_box_indices: List[int] = []
            for idx, wp in enumerate(waypoint_list):
                if wp.distance_to(rx, ry) <= self.shortcut_rejoin_dist_m:
                    ix, iy = self.grid.world_to_grid(wp.x, wp.y)
                    if self.grid.in_bounds(ix, iy, margin=2):
                        in_box_indices.append(idx)

            # Evaluate at most ~8 spread candidates starting from earliest index
            if len(in_box_indices) > 8:
                sampled = [in_box_indices[0]]
                step = max(1, len(in_box_indices) // 6)
                sampled.extend(in_box_indices[1:-1:step])
                sampled.append(in_box_indices[-1])
                candidates = sorted(set(sampled))
            else:
                candidates = in_box_indices

        astar_budget = max(4000, int(0.8 * self.grid.n * self.grid.n))
        free_search_r = max(10, int(round(0.80 / self.grid.res)))

        for idx in candidates:
            wp = waypoint_list[idx]
            goal_ij = self.grid.world_to_grid(wp.x, wp.y)
            if hard_blocked[goal_ij[0], goal_ij[1]]:
                # Do not blindly shortcut onto an obstructed waypoint; allow nearest free only for top waypoint
                if idx < m - 1:
                    continue
                goal_ij = self._nearest_free_cell(hard_blocked, goal_ij, max_radius_cells=max(6, int(round(0.60 / self.grid.res))))
                if goal_ij is None:
                    continue

            path, cost = astar_2d(hard_blocked, soft_cost, start_ij, goal_ij, max_expansions=astar_budget)
            if path is None or len(path) == 0:
                continue

            # Guard against cutting across unscanned or walled indoor rooms
            dist_euclid = wp.distance_to(rx, ry)
            if idx < m - 2 and dist_euclid > 0.70:
                observed_hits = sum(1 for (px, py) in path if self.grid.observed_free[px, py] > 0.25)
                obs_ratio = observed_hits / float(len(path))
                path_len_m = len(path) * self.grid.res
                is_deep_jump = (m - 1 - idx) > 5
                min_obs = 0.80 if is_deep_jump else 0.65
                max_stretch = (1.40 * dist_euclid + 0.45) if is_deep_jump else (1.75 * dist_euclid + 0.6)
                if obs_ratio < min_obs or path_len_m > max_stretch:
                    continue

            return path, idx

        # Fallback: project strictly along recent trailing breadcrumbs (never blindly jump to idx 0 across walls)
        fallback_order = []
        for fi in (m - 1, max(0, m - 2), max(0, m - 3), max(0, m - 5)):
            if fi not in fallback_order:
                fallback_order.append(fi)
        for fallback_idx in fallback_order:
            wp = waypoint_list[fallback_idx]
            proj_ij = self._project_into_grid(rx, ry, wp.x, wp.y)
            free_goal = self._nearest_free_cell(hard_blocked, proj_ij, max_radius_cells=free_search_r)
            if free_goal is None:
                continue
            path, _ = astar_2d(hard_blocked, soft_cost, start_ij, free_goal, max_expansions=astar_budget)
            if path is not None and len(path) > 0:
                return path, fallback_idx

        return None, None

    def step(
        self,
        rx: float,
        ry: float,
        rth: float,
        waypoint_stack,
        start_pose,
        scan_pts: Optional[np.ndarray],
        v_max: float,
        w_max: float,
        now_sec: float,
        wp_reach_radius: float,
        docking_dist_th: float,
    ) -> Tuple[str, float, float, Optional[Tuple[float, float]]]:
        """
        Run one control tick for Proposed Layer-2 backtracking.

        Returns (status_tag, v_cmd, w_cmd, subgoal_xy):
          - status_tag == "ARRIVED": ready for handover/mission complete logic.
          - status_tag == "ESCAPE": executing reverse/pivot unstuck escape.
          - status_tag == "ASTAR_TRACK": tracking A* subgoal smoothly.
          - status_tag == "ROTATE_ALIGN": large heading alignment turn in place.
        """
        self.grid.update_scan(rx, ry, rth, scan_pts)
        front_min, left_min, right_min, rear_min = self._scan_sector_clearances(scan_pts)

        # Pop already reached breadcrumbs or earlier loop-closure breadcrumbs within 0.55m
        # (non-destructive for distant shortcuts: only prune once physically reached)
        reach_prune = max(0.35, wp_reach_radius)
        if len(waypoint_stack) > 1:
            # Check if we physically arrived within 0.55m of any earlier waypoint
            best_reached_idx = None
            for idx in range(len(waypoint_stack) - 1):
                if waypoint_stack[idx].distance_to(rx, ry) < 0.55:
                    best_reached_idx = idx
                    break
            if best_reached_idx is not None:
                while len(waypoint_stack) - 1 > best_reached_idx:
                    waypoint_stack.pop()
                self.shortcut_count += 1
        while len(waypoint_stack) > 1 and waypoint_stack[-1].distance_to(rx, ry) < reach_prune:
            waypoint_stack.pop()

        if start_pose is not None:
            dist_home = start_pose.distance_to(rx, ry)
            if (len(waypoint_stack) <= 2 and dist_home <= max(docking_dist_th, wp_reach_radius)) or dist_home <= 0.28:
                return "ARRIVED", 0.0, 0.0, (start_pose.x, start_pose.y)

        if len(waypoint_stack) == 0:
            if start_pose is not None:
                waypoint_stack.append(start_pose)
            else:
                return "ARRIVED", 0.0, 0.0, None

        # 1. Watchdog & Near-Collision Unstuck Detector
        if self._anchor_time is None:
            self._anchor_time = now_sec
            self._anchor_x = rx
            self._anchor_y = ry

        if front_min < 0.165:
            if self._front_block_since is None:
                self._front_block_since = now_sec
        else:
            self._front_block_since = None

        stuck_by_disp = False
        if (now_sec - self._anchor_time) >= self.unstuck_window_sec:
            if math.hypot(rx - self._anchor_x, ry - self._anchor_y) < self.unstuck_min_disp_m:
                stuck_by_disp = True
            self._anchor_time = now_sec
            self._anchor_x = rx
            self._anchor_y = ry

        stuck_by_contact = (
            self._front_block_since is not None
            and (now_sec - self._front_block_since) >= 0.45
        )

        if now_sec >= self._escape_until and (stuck_by_disp or stuck_by_contact):
            self._virtual_obstacles = [vo for vo in self._virtual_obstacles if vo[2] > now_sec]
            if self._last_escape_xy is not None and math.hypot(rx - self._last_escape_xy[0], ry - self._last_escape_xy[1]) < 0.35:
                self._repeat_escape_count += 1
            else:
                self._repeat_escape_count = 0
            self._last_escape_xy = (rx, ry)

            # Infer a temporary virtual barrier ahead when stalled despite clear 2D LiDAR (e.g. sub-LiDAR plinth)
            if stuck_by_disp and front_min >= 0.165:
                block_dir = self._last_fwd_pose_dir if self._last_fwd_pose_dir is not None else rth
                cos_d = math.cos(block_dir)
                sin_d = math.sin(block_dir)
                px_n = -sin_d
                py_n = cos_d
                ttl = now_sec + 22.0
                for lat in (-0.08, 0.0, 0.08):
                    self._virtual_obstacles.append(
                        (rx + 0.19 * cos_d + lat * px_n, ry + 0.19 * sin_d + lat * py_n, ttl)
                    )
                if len(self._virtual_obstacles) > 24:
                    self._virtual_obstacles = self._virtual_obstacles[-24:]

            base_turn = 1.0 if left_min >= right_min else -1.0
            if self._repeat_escape_count > 0 and (self._repeat_escape_count % 2 == 1):
                base_turn = -base_turn

            self._escape_start = now_sec
            if self._repeat_escape_count >= 1:
                self._escape_reverse_dur = 1.05
                self._escape_until = now_sec + 2.25
            else:
                self._escape_reverse_dur = 0.90
                self._escape_until = now_sec + 1.85
            self._escape_turn_dir = base_turn
            self._front_block_since = None
            if (
                self._last_chosen_idx is not None
                and self._last_chosen_idx < max(0, len(waypoint_stack) - 3)
            ):
                # Ban greedy shortcuts temporarily if one led into a dead-end wall
                self._no_shortcut_until = now_sec + 10.0
            self.escape_count += 1

        if now_sec < self._escape_until:
            elapsed = now_sec - self._escape_start
            if elapsed < self._escape_reverse_dur:
                v_esc = -0.14 if rear_min > 0.22 else 0.0
                w_esc = 0.20 * self._escape_turn_dir
            else:
                v_esc = 0.0
                turn_rate = 0.92 if self._repeat_escape_count >= 1 else 0.78
                w_esc = max(-w_max, min(w_max, turn_rate * self._escape_turn_dir))
            return "ESCAPE", float(v_esc), float(w_esc), None

        # 2. Rolling 2D A* + Topological Breadcrumb Shortcut
        self._virtual_obstacles = [vo for vo in self._virtual_obstacles if vo[2] > now_sec]
        extra_pts = [(vo[0], vo[1]) for vo in self._virtual_obstacles] if self._virtual_obstacles else None
        hard_blocked, soft_cost = self.grid.build_cost_maps(rx, ry, extra_world_pts=extra_pts)
        wp_list = list(waypoint_stack)
        allow_shortcuts = now_sec >= self._no_shortcut_until
        path_ij, chosen_idx = self._plan_topological_astar(
            rx, ry, wp_list, hard_blocked, soft_cost, allow_shortcuts=allow_shortcuts
        )
        self._last_chosen_idx = chosen_idx

        # Only permanently prune 1-2 immediate rock-buried waypoints nearby; keep distant stack intact until reached
        if (
            chosen_idx is not None
            and chosen_idx < len(waypoint_stack) - 1
            and (len(waypoint_stack) - 1 - chosen_idx) <= 3
            and waypoint_stack[chosen_idx].distance_to(rx, ry) < 1.35
        ):
            skipped = (len(waypoint_stack) - 1) - chosen_idx
            for _ in range(skipped):
                if len(waypoint_stack) > 1:
                    waypoint_stack.pop()
            self.shortcut_count += 1

        # Extract smooth lookahead subgoal
        if path_ij is not None and len(path_ij) > 0:
            look_cells = max(2, int(round(self.lookahead_m / self.grid.res)))
            sub_ij = path_ij[min(look_cells, len(path_ij) - 1)]
            sub_x, sub_y = self.grid.grid_to_world(sub_ij[0], sub_ij[1])
        else:
            # No A* path even to projected frontier: backup/pivot safely instead of ramming
            turn_dir = 1.0 if left_min >= right_min else -1.0
            v_fb = -0.09 if (front_min < 0.25 and rear_min > 0.22) else 0.0
            w_fb = max(-w_max, min(w_max, 0.55 * turn_dir))
            return "ESCAPE", float(v_fb), float(w_fb), None

        self.last_subgoal = (sub_x, sub_y)

        # 3. Safe Subgoal Tracking Control (Never forces v >= 0.08 into obstacles)
        alpha = _normalize_angle(math.atan2(sub_y - ry, sub_x - rx) - rth)
        if abs(alpha) > 1.15:
            v_rot = -0.06 if (front_min < 0.14 and rear_min > 0.24) else 0.0
            w_cmd = max(-w_max, min(w_max, 1.55 * alpha))
            return "ROTATE_ALIGN", float(v_rot), float(w_cmd), (sub_x, sub_y)

        # Clearance-scaled forward velocity
        ang_scale = max(0.15, math.cos(alpha))
        if front_min < 0.165:
            # Pivot toward A* subgoal or wider side gap without advancing (unpin if touching)
            w_dir = 1.0 if alpha >= 0.0 else -1.0
            if abs(alpha) < 0.15:
                w_dir = 1.0 if left_min >= right_min else -1.0
            v_unpin = -0.06 if (front_min < 0.14 and rear_min > 0.24) else 0.0
            return "ROTATE_ALIGN", float(v_unpin), float(max(-w_max, min(w_max, 0.55 * w_dir))), (sub_x, sub_y)

        if front_min < 0.36:
            prox_scale = max(0.25, (front_min - 0.18) / (0.36 - 0.18))
            v_cmd = min(v_max, 0.14 * prox_scale * ang_scale)
        else:
            v_cmd = min(v_max, v_max * ang_scale)

        if v_cmd > 0.02:
            self._last_fwd_pose_dir = rth
        w_cmd = max(-w_max, min(w_max, 1.65 * alpha))
        return "ASTAR_TRACK", float(v_cmd), float(w_cmd), (sub_x, sub_y)
