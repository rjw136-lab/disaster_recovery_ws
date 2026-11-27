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
import time
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from disaster_recovery.dynamic_tracker import predict_track_trajectory


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
        phase2b_mode: bool = False,
        enable_spatiotemporal_dynamic_control: bool = True,
    ):
        self.size_m = float(size_m)
        self.res = float(resolution_m)
        self.n = int(round(self.size_m / self.res))
        self.inflation_radius_m = float(inflation_radius_m)
        self.soft_radius_m = float(soft_radius_m)
        self.phase2b_mode = bool(phase2b_mode)
        self.enable_spatiotemporal_dynamic_control = bool(enable_spatiotemporal_dynamic_control)

        self.occ = np.zeros((self.n, self.n), dtype=np.float32)
        self.observed_free = np.zeros((self.n, self.n), dtype=np.float32)
        self.initialized = False
        self.origin_ix = 0
        self.origin_iy = 0
        self._last_axial_band_active = False
        self._last_axial_core_mask: Optional[np.ndarray] = None

        hard_cells = max(1, int(math.ceil(self.inflation_radius_m / self.res)))
        soft_cells = max(hard_cells + 1, int(math.ceil(self.soft_radius_m / self.res)))
        self._hard_offsets = [
            (di, dj, d_cell)
            for (di, dj, d_cell) in _build_disk_offsets(hard_cells)
            if (d_cell * self.res) <= self.inflation_radius_m + 1e-6
        ] or [(0, 0, 0.0)]
        tight_radius_m = min(self.inflation_radius_m, 0.058)
        tight_cells = max(1, int(math.ceil(tight_radius_m / self.res)))
        self._tight_offsets = [
            (di, dj, d_cell)
            for (di, dj, d_cell) in _build_disk_offsets(tight_cells)
            if (d_cell * self.res) <= tight_radius_m + 1e-6
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
        self._last_axial_band_active = False
        self._last_axial_core_mask: Optional[np.ndarray] = None

    def sync_icp_correction(
        self,
        old_corr: Tuple[float, float, float],
        new_corr: Tuple[float, float, float],
    ) -> None:
        """Inverse-warp occupancy & observed_free grids across a waypoint-ICP correction jump delta_T = T_new * T_old^-1."""
        if not self.initialized:
            return
        ox, oy, oth = float(old_corr[0]), float(old_corr[1]), float(old_corr[2])
        nx, ny, nth = float(new_corr[0]), float(new_corr[1]), float(new_corr[2])
        d_th = math.atan2(math.sin(nth - oth), math.cos(nth - oth))
        c_d = math.cos(d_th)
        s_d = math.sin(d_th)
        d_tx = nx - (c_d * ox - s_d * oy)
        d_ty = ny - (s_d * ox + c_d * oy)
        if abs(d_tx) <= 1e-4 and abs(d_ty) <= 1e-4 and abs(d_th) <= 1e-4:
            return

        grid_i = np.arange(self.n, dtype=np.float32) + np.float32(self.origin_ix + 0.5)
        grid_j = np.arange(self.n, dtype=np.float32) + np.float32(self.origin_iy + 0.5)
        wx_new = grid_i[:, None] * np.float32(self.res)
        wy_new = grid_j[None, :] * np.float32(self.res)
        dx = wx_new - np.float32(d_tx)
        dy = wy_new - np.float32(d_ty)
        wx_old = np.float32(c_d) * dx + np.float32(s_d) * dy
        wy_old = -np.float32(s_d) * dx + np.float32(c_d) * dy
        old_ix = np.floor(wx_old / np.float32(self.res)).astype(np.int32) - self.origin_ix
        old_iy = np.floor(wy_old / np.float32(self.res)).astype(np.int32) - self.origin_iy
        inb = (old_ix >= 0) & (old_ix < self.n) & (old_iy >= 0) & (old_iy < self.n)
        ix_c = np.clip(old_ix, 0, self.n - 1)
        iy_c = np.clip(old_iy, 0, self.n - 1)
        self.occ = np.where(inb, self.occ[ix_c, iy_c], np.float32(0.0))
        self.observed_free = np.where(inb, self.observed_free[ix_c, iy_c], np.float32(0.0))

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

        # Phase 2-B uses fast dynamic trail decay & single-tick free-space ray-carving,
        # whereas static-grid A* (Phase 1 / 2-A) uses static occupancy persistence.
        if self.phase2b_mode:
            self.occ *= 0.955
            self.observed_free *= 0.990
            carve_decay = 0.25
        else:
            self.occ *= 0.965
            self.observed_free *= 0.985
            carve_decay = 0.40

        c = math.cos(rth)
        s = math.sin(rth)

        # In Phase 2-B, also ray-carve occupancy along open max-range rays (without marking unscanned space as observed_free)
        if self.phase2b_mode and raw_ranges is not None and len(raw_ranges) >= 36:
            rr = np.asarray(raw_ranges[::2], dtype=np.float32)
            n_sub = len(rr)
            ang_sub = np.linspace(0.0, 2.0 * math.pi, n_sub, endpoint=False, dtype=np.float32)
            open_idx = (~np.isfinite(rr)) | (rr > 3.45)
            if np.any(open_idx):
                ang_open = ang_sub[open_idx] + rth
                wx_open = rx + 3.40 * np.cos(ang_open)
                wy_open = ry + 3.40 * np.sin(ang_open)
                fracs_o = np.linspace(0.05, 0.92, 96, dtype=np.float32)[:, None]
                rx_o = rx + fracs_o * (wx_open[None, :] - rx)
                ry_o = ry + fracs_o * (wy_open[None, :] - ry)
                ix_o = np.floor(rx_o / self.res).astype(np.int32) - self.origin_ix
                iy_o = np.floor(ry_o / self.res).astype(np.int32) - self.origin_iy
                m_o = (ix_o >= 0) & (ix_o < self.n) & (iy_o >= 0) & (iy_o < self.n)
                self.occ[ix_o[m_o], iy_o[m_o]] *= carve_decay

        if scan_pts_robot is None or len(scan_pts_robot) == 0:
            return

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
            step_cap = 110 if self.phase2b_mode else 40
            max_steps = max(2, min(max_steps, step_cap))
            fracs = np.linspace(0.06, 0.90 if self.phase2b_mode else 0.88, max_steps, dtype=np.float32)[:, None]  # (K, 1)

            ray_x = rx + fracs * (wx_r[None, :] - rx)
            ray_y = ry + fracs * (wy_r[None, :] - ry)
            ix_f = np.floor(ray_x / self.res).astype(np.int32) - self.origin_ix
            iy_f = np.floor(ray_y / self.res).astype(np.int32) - self.origin_iy

            mask = (ix_f >= 0) & (ix_f < self.n) & (iy_f >= 0) & (iy_f < self.n)
            ix_v = ix_f[mask]
            iy_v = iy_f[mask]
            self.observed_free[ix_v, iy_v] = 1.0
            self.occ[ix_v, iy_v] *= carve_decay

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
        tight: bool = False,
        semantic_obstacles: Optional[Sequence[Tuple[float, float, float]]] = None,
        semantic_hazards: Optional[Sequence[Tuple[float, float, float]]] = None,
        dynamic_obstacles: Optional[Sequence[object]] = None,
        reuse_soft_cost: Optional[np.ndarray] = None,
        path_dir: Optional[Tuple[float, float]] = None,
        waypoint_list: Optional[Sequence[object]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (hard_blocked bool[N,N], soft_cost float32[N,N]) with optional Ch1/Ch2/dynamic A* costs."""
        raw_occ = self.occ >= 0.45
        if extra_world_pts:
            raw_occ = raw_occ.copy()
            for vx, vy in extra_world_pts:
                vix, viy = self.world_to_grid(vx, vy)
                if self.in_bounds(vix, viy):
                    raw_occ[vix, viy] = True
        hard_blocked = np.zeros((self.n, self.n), dtype=bool)
        compute_soft = reuse_soft_cost is None
        soft_cost = np.zeros((self.n, self.n), dtype=np.float32) if compute_soft else reuse_soft_cost

        offsets = self._tight_offsets if tight else self._hard_offsets
        occ_idx = np.argwhere(raw_occ)
        if len(occ_idx) > 0:
            ox = occ_idx[:, 0]
            oy = occ_idx[:, 1]
            if compute_soft:
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

            for di, dj, _ in offsets:
                nx = ox + di
                ny = oy + dj
                valid = (nx >= 0) & (nx < self.n) & (ny >= 0) & (ny < self.n)
                if np.any(valid):
                    hard_blocked[nx[valid], ny[valid]] = True

        # Stage 0.1 optional semantic Ch1 (low obstacles), Ch2 (gas/survivor), and dynamic track A* costs
        if semantic_obstacles:
            infl_m = 0.055 if tight else self.inflation_radius_m
            for lx, ly, lrad in semantic_obstacles:
                cix, ciy = self.world_to_grid(float(lx), float(ly))
                hard_r_m = max(0.05, float(lrad)) + infl_m
                soft_r_m = hard_r_m + 0.16 if compute_soft else hard_r_m
                r_cells = int(math.ceil(soft_r_m / self.res))
                x0 = max(0, cix - r_cells)
                x1 = min(self.n, cix + r_cells + 1)
                y0 = max(0, ciy - r_cells)
                y1 = min(self.n, ciy + r_cells + 1)
                if x0 >= x1 or y0 >= y1:
                    continue
                gx = (self.origin_ix + np.arange(x0, x1, dtype=np.float32) + 0.5) * self.res
                gy = (self.origin_iy + np.arange(y0, y1, dtype=np.float32) + 0.5) * self.res
                d_grid = np.hypot(gx[:, None] - float(lx), gy[None, :] - float(ly))
                hard_blocked[x0:x1, y0:y1] |= (d_grid <= hard_r_m)
                if compute_soft:
                    in_soft = d_grid <= soft_r_m
                    if np.any(in_soft):
                        pen = np.clip(10.0 * (1.0 - np.maximum(d_grid - hard_r_m, 0.0) / 0.16), 0.0, 10.0).astype(np.float32)
                        soft_cost[x0:x1, y0:y1] = np.maximum(soft_cost[x0:x1, y0:y1], np.where(in_soft, pen, 0.0))

        if compute_soft and semantic_hazards:
            for hx, hy, hrad in semantic_hazards:
                cix, ciy = self.world_to_grid(float(hx), float(hy))
                outer_m = max(0.15, float(hrad)) + 0.22
                r_cells = int(math.ceil(outer_m / self.res))
                x0 = max(0, cix - r_cells)
                x1 = min(self.n, cix + r_cells + 1)
                y0 = max(0, ciy - r_cells)
                y1 = min(self.n, ciy + r_cells + 1)
                if x0 >= x1 or y0 >= y1:
                    continue
                gx = (self.origin_ix + np.arange(x0, x1, dtype=np.float32) + 0.5) * self.res
                gy = (self.origin_iy + np.arange(y0, y1, dtype=np.float32) + 0.5) * self.res
                d_grid = np.hypot(gx[:, None] - float(hx), gy[None, :] - float(hy))
                in_haz = d_grid <= outer_m
                if np.any(in_haz):
                    pen = np.clip(18.0 * (1.0 - d_grid / max(outer_m, 1e-4)), 0.0, 18.0).astype(np.float32)
                    soft_cost[x0:x1, y0:y1] = np.maximum(soft_cost[x0:x1, y0:y1], np.where(in_haz, pen, 0.0))

        axial_band_applied = False
        axial_core_mask: Optional[np.ndarray] = None
        if compute_soft and dynamic_obstacles:
            # Resolve optional externally supplied path direction unit vector
            ext_dir: Optional[Tuple[float, float]] = None
            if path_dir is not None:
                dn = math.hypot(float(path_dir[0]), float(path_dir[1]))
                if dn > 1e-6:
                    ext_dir = (float(path_dir[0]) / dn, float(path_dir[1]) / dn)

            for trk in dynamic_obstacles:
                if not getattr(trk, "is_confirmed", True):
                    continue
                trad = max(0.10, float(getattr(trk, "radius", 0.15)))
                pred = predict_track_trajectory(
                    trk,
                    (0.5, 1.0),
                    enable_harmonic=self.enable_spatiotemporal_dynamic_control,
                )

                # 1) IMM-predicted future occupancy soft halos (soft cost only)
                for idx_h, w_t in ((0, 3.0), (1, 2.0)):
                    px_t = float(pred.centers[idx_h, 0])
                    py_t = float(pred.centers[idx_h, 1])
                    cix, ciy = self.world_to_grid(px_t, py_t)
                    outer_m = trad + 0.25
                    r_cells = int(math.ceil(outer_m / self.res))
                    x0 = max(0, cix - r_cells)
                    x1 = min(self.n, cix + r_cells + 1)
                    y0 = max(0, ciy - r_cells)
                    y1 = min(self.n, ciy + r_cells + 1)
                    if x0 >= x1 or y0 >= y1:
                        continue
                    gx = (self.origin_ix + np.arange(x0, x1, dtype=np.float32) + 0.5) * self.res
                    gy = (self.origin_iy + np.arange(y0, y1, dtype=np.float32) + 0.5) * self.res
                    dx_h = gx[:, None] - np.float32(px_t)
                    dy_h = gy[None, :] - np.float32(py_t)
                    d2_h = dx_h * dx_h + dy_h * dy_h
                    outer_f = np.float32(outer_m)
                    in_dyn = d2_h <= (outer_f * outer_f)
                    if np.any(in_dyn):
                        sub_s = soft_cost[x0:x1, y0:y1]
                        d_in = np.sqrt(d2_h[in_dyn])
                        pen_in = np.clip(
                            np.float32(w_t) * (np.float32(1.0) - d_in / np.float32(max(outer_m, 1e-4))),
                            np.float32(0.0),
                            np.float32(w_t),
                        )
                        sub_s[in_dyn] = np.maximum(sub_s[in_dyn], pen_in)

                # 2) Axial reciprocating sweep-band soft cost (|d_path . u_hat| >= 0.65):
                # Guides A* proactively into the clear side lane in wide corridors while
                # NEVER modifying hard_blocked so narrow gates remain passable.
                ux, uy = pred.u_hat
                pcx, pcy = pred.p_c
                if ext_dir is not None:
                    dpx, dpy = ext_dir
                elif waypoint_list and len(waypoint_list) >= 2:
                    m_wp = len(waypoint_list)
                    nearest_i = min(
                        range(m_wp),
                        key=lambda i: math.hypot(
                            float(waypoint_list[i].x) - pcx,
                            float(waypoint_list[i].y) - pcy,
                        ),
                    )
                    i_lo = max(0, nearest_i - 1)
                    i_hi = min(m_wp - 1, nearest_i + 1)
                    tx_d = float(waypoint_list[i_lo].x) - float(waypoint_list[i_hi].x)
                    ty_d = float(waypoint_list[i_lo].y) - float(waypoint_list[i_hi].y)
                    tn = math.hypot(tx_d, ty_d)
                    if tn > 1e-4:
                        dpx, dpy = tx_d / tn, ty_d / tn
                    else:
                        tx_d = float(waypoint_list[0].x) - float(rx)
                        ty_d = float(waypoint_list[0].y) - float(ry)
                        tn = math.hypot(tx_d, ty_d)
                        dpx, dpy = (tx_d / tn, ty_d / tn) if tn > 1e-4 else (1.0, 0.0)
                elif waypoint_list and len(waypoint_list) == 1:
                    tx_d = float(waypoint_list[0].x) - float(rx)
                    ty_d = float(waypoint_list[0].y) - float(ry)
                    tn = math.hypot(tx_d, ty_d)
                    dpx, dpy = (tx_d / tn, ty_d / tn) if tn > 1e-4 else (1.0, 0.0)
                else:
                    tx_d = pcx - float(rx)
                    ty_d = pcy - float(ry)
                    tn = math.hypot(tx_d, ty_d)
                    dpx, dpy = (tx_d / tn, ty_d / tn) if tn > 1e-4 else (ux, uy)

                if (
                    self.enable_spatiotemporal_dynamic_control
                    and abs(dpx * ux + dpy * uy) >= 0.65
                    and pred.amplitude >= 0.06
                ):
                    ax, ay = pred.sweep_start
                    bx, by = pred.sweep_end
                    band_half_w = trad + 0.34
                    min_x_w = min(ax, bx) - band_half_w
                    max_x_w = max(ax, bx) + band_half_w
                    min_y_w = min(ay, by) - band_half_w
                    max_y_w = max(ay, by) + band_half_w
                    x0_i, y0_i = self.world_to_grid(min_x_w, min_y_w)
                    x1_i, y1_i = self.world_to_grid(max_x_w, max_y_w)
                    x0 = max(0, x0_i)
                    x1 = min(self.n, x1_i + 1)
                    y0 = max(0, y0_i)
                    y1 = min(self.n, y1_i + 1)
                    if x0 < x1 and y0 < y1:
                        # Check whether at least one lateral bypass lane outside band_half_w
                        # is free of static hard_blocked walls so narrow corridors never blow
                        # A* expansion budgets while wide corridors enforce w_band=3.8 (>3.6).
                        nx_u, ny_u = -uy, ux
                        d0_p = max(0.24, trad + 0.06)
                        d1_p = band_half_w + self.res
                        step_p = (d1_p - d0_p) / 5.0
                        probe_dists = (
                            d0_p,
                            d0_p + step_p,
                            d0_p + 2.0 * step_p,
                            d0_p + 3.0 * step_p,
                            d0_p + 4.0 * step_p,
                            d1_p,
                        )
                        has_clear_bypass = False
                        for side_s in (1.0, -1.0):
                            valid_fracs = 0
                            clear_fracs = 0
                            for frac in (0.15, 0.50, 0.85):
                                spx = ax + frac * (bx - ax)
                                spy = ay + frac * (by - ay)
                                cix_s, ciy_s = self.world_to_grid(spx, spy)
                                if not self.in_bounds(cix_s, ciy_s):
                                    continue
                                valid_fracs += 1
                                frac_ok = True
                                for d_p in probe_dists:
                                    pix, piy = self.world_to_grid(
                                        spx + side_s * d_p * nx_u,
                                        spy + side_s * d_p * ny_u,
                                    )
                                    if (not self.in_bounds(pix, piy)) or hard_blocked[pix, piy]:
                                        frac_ok = False
                                        break
                                if frac_ok:
                                    clear_fracs += 1
                            if valid_fracs > 0 and clear_fracs >= max(1, valid_fracs - 1):
                                has_clear_bypass = True
                                break

                        gx = (self.origin_ix + np.arange(x0, x1, dtype=np.float32) + 0.5) * self.res
                        gy = (self.origin_iy + np.arange(y0, y1, dtype=np.float32) + 0.5) * self.res
                        abx = float(bx - ax)
                        aby = float(by - ay)
                        ab_len = max(math.hypot(abx, aby), 1e-6)
                        ux_s = np.float32(abx / ab_len)
                        uy_s = np.float32(aby / ab_len)
                        dx_g = gx[:, None] - np.float32(ax)
                        dy_g = gy[None, :] - np.float32(ay)
                        s_along = dx_g * ux_s + dy_g * uy_s
                        d_long = np.maximum(
                            np.float32(0.0),
                            np.maximum(-s_along, s_along - np.float32(ab_len)),
                        )
                        d_perp = np.abs(-dx_g * uy_s + dy_g * ux_s)
                        d2_seg = d_long * d_long + d_perp * d_perp
                        band_w_f = np.float32(band_half_w)
                        in_band = d2_seg <= (band_w_f * band_w_f)
                        if np.any(in_band):
                            sub_soft = soft_cost[x0:x1, y0:y1]
                            d_in = np.sqrt(d2_seg[in_band])
                            if has_clear_bypass:
                                w_band = np.float32(3.8)
                                u_p = d_perp[in_band] / np.float32(max(band_half_w, 1e-4))
                                pen_body = w_band * (np.float32(1.0) - u_p * u_p)
                                # Gentle linear funnel on semicircular endcaps avoids 90-deg
                                # ROTATE_ALIGN spin-in-place when approaching head-on while the
                                # rail body strictly blocks diagonal chords via axial_core_mask.
                                pen_cap = np.float32(1.5) * (
                                    np.float32(1.0) - d_in / np.float32(max(band_half_w, 1e-4))
                                )
                                pen_in = np.clip(
                                    np.where(d_long[in_band] <= np.float32(0.04), pen_body, pen_cap),
                                    np.float32(0.0),
                                    w_band,
                                )
                                if axial_core_mask is None:
                                    axial_core_mask = np.zeros((self.n, self.n), dtype=bool)
                                core_w = np.float32(max(0.05, band_half_w - 0.02))
                                in_core = (d_long <= np.float32(0.04)) & (d_perp <= core_w)
                                axial_core_mask[x0:x1, y0:y1] |= in_core
                                axial_band_applied = True
                            else:
                                w_band = np.float32(1.8)
                                pen_in = np.clip(
                                    w_band * (np.float32(1.0) - d_in / np.float32(max(band_half_w, 1e-4))),
                                    np.float32(0.0),
                                    w_band,
                                )
                            sub_soft[in_band] = np.maximum(sub_soft[in_band], pen_in)

        if compute_soft:
            self._last_axial_band_active = bool(axial_band_applied)
            self._last_axial_core_mask = axial_core_mask

        # Safeguard: if the robot itself slightly brushes an inflated boundary,
        # downgrade inflated cells within 0.22m of robot center (keeping raw hits blocked)
        # so A* can always escape outward into clear space.
        sx, sy = self.world_to_grid(rx, ry)
        if self.in_bounds(sx, sy):
            guard_rad_m = (
                min(0.14, 0.06 + 0.04)
                if tight
                else min(0.22, self.inflation_radius_m + 0.08)
            )
            guard_r = max(1, int(math.ceil(guard_rad_m / self.res)))
            for di in range(-guard_r, guard_r + 1):
                for dj in range(-guard_r, guard_r + 1):
                    gx, gy = sx + di, sy + dj
                    if self.in_bounds(gx, gy) and not raw_occ[gx, gy]:
                        if di * di + dj * dj <= guard_r * guard_r:
                            was_blocked = bool(hard_blocked[gx, gy])
                            hard_blocked[gx, gy] = False
                            if compute_soft and (was_blocked or not axial_band_applied):
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
    deadline: Optional[float] = None,
    heuristic_weight: float = 1.0,
) -> Tuple[Optional[List[Tuple[int, int]]], float]:
    """8-connected 2D A* planner avoiding corners and soft obstacle halos."""
    n_x, n_y = hard_blocked.shape
    sx, sy = start_ij
    gx, gy = goal_ij

    if not (0 <= sx < n_x and 0 <= sy < n_y and 0 <= gx < n_x and 0 <= gy < n_y):
        return None, float("inf")
    if hard_blocked[gx, gy]:
        return None, float("inf")
    if sx == gx and sy == gy:
        return [(sx, sy)], 0.0

    hard_flat = hard_blocked.ravel().tolist()
    soft_flat = soft_cost.ravel().tolist()
    total = n_x * n_y
    inf_val = float("inf")
    g_score = [inf_val] * total
    closed = bytearray(total)
    parent = [-1] * total

    s_idx = sx * n_y + sy
    g_idx = gx * n_y + gy
    g_score[s_idx] = 0.0
    dx0 = abs(sx - gx)
    dy0 = abs(sy - gy)
    w_h = float(heuristic_weight)
    h0 = w_h * ((dx0 if dx0 >= dy0 else dy0) + 0.41421356 * (dy0 if dx0 >= dy0 else dx0))
    heap: List[Tuple[float, float, int, int, int]] = [(h0, 0.0, sx, sy, s_idx)]
    heappop = heapq.heappop
    heappush = heapq.heappush
    perf_counter = time.perf_counter
    expansions = 0
    pop_count = 0

    while heap and expansions < max_expansions:
        if deadline is not None and (pop_count & 63) == 0 and perf_counter() > deadline:
            break
        pop_count += 1
        _, cur_g, cx, cy, c_idx = heappop(heap)
        if closed[c_idx]:
            continue
        if c_idx == g_idx:
            path: List[Tuple[int, int]] = []
            cur = g_idx
            while cur != -1:
                path.append((cur // n_y, cur % n_y))
                cur = parent[cur]
            path.reverse()
            return path, cur_g

        closed[c_idx] = 1
        expansions += 1

        for di, dj, step_cost in _NEIGHBORS_8:
            nx = cx + di
            ny = cy + dj
            if nx < 0 or nx >= n_x or ny < 0 or ny >= n_y:
                continue
            n_idx = nx * n_y + ny
            if closed[n_idx] or hard_flat[n_idx]:
                continue
            # Prevent diagonal corner clipping across blocked edges
            if di != 0 and dj != 0:
                if hard_flat[(cx + di) * n_y + cy] or hard_flat[cx * n_y + (cy + dj)]:
                    continue

            tentative_g = cur_g + step_cost + soft_flat[n_idx]
            if tentative_g < g_score[n_idx]:
                g_score[n_idx] = tentative_g
                parent[n_idx] = c_idx
                dx = gx - nx
                if dx < 0:
                    dx = -dx
                dy = gy - ny
                if dy < 0:
                    dy = -dy
                h_val = w_h * ((dx if dx >= dy else dy) + 0.41421356 * (dy if dx >= dy else dx))
                heappush(heap, (tentative_g + h_val, tentative_g, nx, ny, n_idx))

    return None, inf_val


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
        phase2b_mode: bool = False,
        enable_spatiotemporal_dynamic_control: bool = True,
    ):
        self.phase2b_mode = bool(phase2b_mode)
        self.enable_spatiotemporal_dynamic_control = bool(enable_spatiotemporal_dynamic_control)
        self.grid = RollingLocalGrid(
            size_m=size_m,
            resolution_m=resolution_m,
            inflation_radius_m=inflation_radius_m,
            phase2b_mode=self.phase2b_mode,
            enable_spatiotemporal_dynamic_control=self.enable_spatiotemporal_dynamic_control,
        )
        self.lookahead_m = float(lookahead_m)
        self.shortcut_rejoin_dist_m = float(shortcut_rejoin_dist_m)
        self.unstuck_window_sec = float(unstuck_window_sec)
        self.unstuck_min_disp_m = float(unstuck_min_disp_m)

        # A* replan gate: full A* only on period/displacement/waypoint change
        self.replan_period_sec: float = 0.25
        self.replan_displacement_m: float = 0.12
        self._astar_cache: Optional[Tuple] = None
        self._astar_cache_maps: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._astar_cache_tight: bool = False
        self._astar_cache_axial_band: bool = False
        self._astar_cache_axial_core: Optional[np.ndarray] = None
        self._astar_plan_time: float = -1.0
        self._astar_plan_x: float = 0.0
        self._astar_plan_y: float = 0.0
        self._astar_plan_wp_len: int = -1
        self._astar_plan_shortcuts: bool = True
        self._astar_plan_has_dyn: bool = False
        self.last_path_ij: Optional[List[Tuple[int, int]]] = None

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
        self.last_clearances: Tuple[float, float, float, float] = (3.5, 3.5, 3.5, 3.5)
        self.last_used_tight_pass: bool = False

        # A* replan cache
        self._astar_cache = None
        self._astar_cache_maps = None
        self._astar_cache_tight = False
        self._astar_plan_time = -1.0
        self._astar_plan_wp_len = -1
        self._astar_plan_has_dyn = False
        self.last_path_ij = None

    def get_last_path_world(self) -> List[Tuple[float, float]]:
        """Convert the most recent A* grid path to world (x, y) coordinates."""
        if not self.last_path_ij:
            return []
        return [self.grid.grid_to_world(ix, iy) for ix, iy in self.last_path_ij]

    def sync_icp_correction(
        self,
        old_corr: Tuple[float, float, float],
        new_corr: Tuple[float, float, float],
    ) -> None:
        """Transform local occupancy grid and world-frame planner anchors by delta_T = T_new * T_old^-1."""
        self.grid.sync_icp_correction(old_corr, new_corr)
        self._astar_cache = None
        self._astar_cache_maps = None
        self.last_path_ij = None
        ox, oy, oth = float(old_corr[0]), float(old_corr[1]), float(old_corr[2])
        nx, ny, nth = float(new_corr[0]), float(new_corr[1]), float(new_corr[2])
        d_th = math.atan2(math.sin(nth - oth), math.cos(nth - oth))
        c_d = math.cos(d_th)
        s_d = math.sin(d_th)
        d_tx = nx - (c_d * ox - s_d * oy)
        d_ty = ny - (s_d * ox + c_d * oy)
        if abs(d_tx) <= 1e-4 and abs(d_ty) <= 1e-4 and abs(d_th) <= 1e-4:
            return

        if self._anchor_time is not None:
            ax, ay = self._anchor_x, self._anchor_y
            self._anchor_x = d_tx + c_d * ax - s_d * ay
            self._anchor_y = d_ty + s_d * ax + c_d * ay
        if self._virtual_obstacles:
            self._virtual_obstacles = [
                (d_tx + c_d * vx - s_d * vy, d_ty + s_d * vx + c_d * vy, exp_t)
                for (vx, vy, exp_t) in self._virtual_obstacles
            ]
        if self._last_escape_xy is not None:
            ex, ey = self._last_escape_xy
            self._last_escape_xy = (d_tx + c_d * ex - s_d * ey, d_ty + s_d * ex + c_d * ey)
        if self.last_subgoal is not None:
            sx, sy = self.last_subgoal
            self.last_subgoal = (d_tx + c_d * sx - s_d * sy, d_ty + s_d * sx + c_d * sy)
        if self._last_fwd_pose_dir is not None:
            self._last_fwd_pose_dir = math.atan2(
                math.sin(self._last_fwd_pose_dir + d_th), math.cos(self._last_fwd_pose_dir + d_th)
            )

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

        # Front corridor closely matching TB3 half-width (0.089m + 1.6cm margin) + immediate front cone
        corridor = (xv > 0.0) & (xv < 0.85) & (np.abs(yv) <= 0.105)
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

    def _nearest_outside_axial_core(
        self,
        hard_blocked: np.ndarray,
        axial_core_mask: np.ndarray,
        target_ij: Tuple[int, int],
        start_ij: Tuple[int, int],
        max_radius_cells: int = 16,
    ) -> Optional[Tuple[int, int]]:
        """Project an intermediate breadcrumb buried inside an axial sweep core onto the nearest clear side lane."""
        tx, ty = target_ij
        sx, sy = start_ij
        if (
            self.grid.in_bounds(tx, ty, margin=1)
            and (not hard_blocked[tx, ty])
            and (not bool(axial_core_mask[tx, ty]))
        ):
            return (tx, ty)
        x0 = max(1, tx - max_radius_cells)
        x1 = min(self.grid.n - 1, tx + max_radius_cells + 1)
        y0 = max(1, ty - max_radius_cells)
        y1 = min(self.grid.n - 1, ty + max_radius_cells + 1)
        if x0 >= x1 or y0 >= y1:
            return None
        free_sub = (~hard_blocked[x0:x1, y0:y1]) & (~axial_core_mask[x0:x1, y0:y1])
        if not np.any(free_sub):
            return None
        fx_i, fy_i = np.nonzero(free_sub)
        nx = fx_i + x0
        ny = fy_i + y0
        d2_goal = (nx - tx) ** 2 + (ny - ty) ** 2
        min_d2 = int(np.min(d2_goal))
        ties = np.nonzero(d2_goal == min_d2)[0]
        if len(ties) == 1:
            best_t = int(ties[0])
        else:
            d2_start = (nx[ties] - sx) ** 2 + (ny[ties] - sy) ** 2
            best_t = int(ties[int(np.argmin(d2_start))])
        return int(nx[best_t]), int(ny[best_t])

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
        deadline: Optional[float] = None,
        fallback_budget_sec: float = 0.004,
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

        astar_budget = 10000
        axial_band_active = bool(
            self.enable_spatiotemporal_dynamic_control
            and getattr(self.grid, "_last_axial_band_active", False)
        )
        axial_core_mask = (
            getattr(self.grid, "_last_axial_core_mask", None) if axial_band_active else None
        )
        astar_hw = 1.25 if axial_band_active else 1.0
        free_search_r = max(10, int(round(0.80 / self.grid.res)))
        t_start = time.perf_counter()
        plan_deadline = deadline if deadline is not None else (t_start + 0.035)
        hard_deadline = max(plan_deadline, t_start + float(fallback_budget_sec))

        # Pass A: Check candidates WITHOUT relaxing blocked goal cells first so a rock-buried
        # stack-top breadcrumb (m-1) does not lure the robot against the front face of debris
        # when an earlier trailing breadcrumb (m-2, m-3, ...) is clearly reachable.
        blocked_top_candidates: List[int] = []
        n_cands = len(candidates)
        for c_i, idx in enumerate(candidates):
            t_now = time.perf_counter()
            if t_now > plan_deadline:
                break
            wp = waypoint_list[idx]
            goal_ij = self.grid.world_to_grid(wp.x, wp.y)
            if hard_blocked[goal_ij[0], goal_ij[1]]:
                if idx == m - 1:
                    blocked_top_candidates.append(idx)
                continue
            # If an intermediate breadcrumb (idx > 0) sits directly inside an active axial
            # sweep core, project it laterally to the nearest clear side-lane cell so A*
            # stays fast (<3 ms) and guides smoothly parallel to the rail without diving in.
            if (
                axial_core_mask is not None
                and idx > 0
                and bool(axial_core_mask[goal_ij[0], goal_ij[1]])
            ):
                side_goal = self._nearest_outside_axial_core(
                    hard_blocked, axial_core_mask, goal_ij, start_ij, max_radius_cells=16
                )
                if side_goal is not None:
                    goal_ij = side_goal

            rem_cands = max(1, n_cands - c_i)
            cand_slice = max(0.004, (plan_deadline - t_now) / float(rem_cands))
            cand_deadline = min(plan_deadline, t_now + cand_slice)
            path, _cost = astar_2d(
                hard_blocked, soft_cost, start_ij, goal_ij,
                max_expansions=astar_budget, deadline=cand_deadline,
                heuristic_weight=astar_hw,
            )
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

        # Pass B: Only if no unobstructed candidate succeeded, project nearest-free around stack-top
        for idx in blocked_top_candidates:
            if time.perf_counter() > plan_deadline:
                break
            wp = waypoint_list[idx]
            goal_ij = self.grid.world_to_grid(wp.x, wp.y)
            goal_ij = self._nearest_free_cell(
                hard_blocked, goal_ij, max_radius_cells=max(6, int(round(0.60 / self.grid.res)))
            )
            if goal_ij is None:
                continue
            path, _cost = astar_2d(
                hard_blocked, soft_cost, start_ij, goal_ij,
                max_expansions=astar_budget, deadline=plan_deadline,
                heuristic_weight=astar_hw,
            )
            if path is not None and len(path) > 0:
                return path, idx

        # Fallback: project strictly along recent trailing breadcrumbs (never blindly jump to idx 0 across walls)
        fallback_order = []
        for fi in (m - 2, m - 3, m - 5, m - 1):
            fi_clamped = max(0, fi)
            if fi_clamped not in fallback_order:
                fallback_order.append(fi_clamped)
        fb_deadline = min(hard_deadline, time.perf_counter() + float(fallback_budget_sec))
        for fallback_idx in fallback_order:
            if time.perf_counter() > fb_deadline:
                break
            wp = waypoint_list[fallback_idx]
            proj_ij = self._project_into_grid(rx, ry, wp.x, wp.y)
            free_goal = self._nearest_free_cell(hard_blocked, proj_ij, max_radius_cells=free_search_r)
            if free_goal is None:
                continue
            path, _ = astar_2d(
                hard_blocked, soft_cost, start_ij, free_goal,
                max_expansions=astar_budget, deadline=fb_deadline,
                heuristic_weight=astar_hw,
            )
            if path is not None and len(path) > 0:
                return path, fallback_idx

        return None, None

    def _segment_is_clear(
        self,
        p0: Tuple[int, int],
        p1: Tuple[int, int],
        hard_blocked: np.ndarray,
        soft_cost: Optional[np.ndarray] = None,
        max_soft: float = 1e9,
        allow_soft_descent: bool = False,
        axial_core_mask: Optional[np.ndarray] = None,
    ) -> bool:
        """Check whether straight line between two grid cells avoids hard_blocked (and soft halos)."""
        x0, y0 = p0
        x1, y1 = p1
        steps = max(2, int(math.ceil(2.0 * max(abs(x1 - x0), abs(y1 - y0)))))
        in_high_soft = False
        prev_soft = max_soft
        if allow_soft_descent and soft_cost is not None and self.grid.in_bounds(x0, y0):
            s0 = float(soft_cost[x0, y0])
            if s0 > max_soft:
                in_high_soft = True
                prev_soft = s0
        for s in range(1, steps + 1):
            t = s / float(steps)
            ix = int(round(x0 + t * (x1 - x0)))
            iy = int(round(y0 + t * (y1 - y0)))
            if not self.grid.in_bounds(ix, iy):
                return False
            if hard_blocked[ix, iy]:
                return False
            if s >= 2:
                if axial_core_mask is not None and bool(axial_core_mask[ix, iy]):
                    return False
                if soft_cost is not None:
                    sv = float(soft_cost[ix, iy])
                    if in_high_soft:
                        if sv > max(max_soft, prev_soft + 1e-3):
                            return False
                        prev_soft = min(prev_soft, sv)
                        if sv <= max_soft:
                            in_high_soft = False
                    elif sv > max_soft:
                        return False
        return True

    def _extract_safe_subgoal(
        self,
        path_ij: List[Tuple[int, int]],
        hard_blocked: np.ndarray,
        soft_cost: np.ndarray,
    ) -> Tuple[float, float]:
        """Pick farthest line-of-sight lookahead cell on A* path so chords never clip obstacle bends."""
        look_cells = max(2, int(round(self.lookahead_m / self.grid.res)))
        max_k = min(look_cells, len(path_ij) - 1)
        start_ij = path_ij[0]
        use_soft_descent = bool(getattr(self.grid, "_last_axial_band_active", False))
        axial_core_mask = (
            getattr(self.grid, "_last_axial_core_mask", None) if use_soft_descent else None
        )

        # Inside axial sweep band (or when the immediate next A* step grazes the core edge):
        # aim directly at the earliest clear cell outside the sweep core so the robot
        # exits laterally instead of falling through to Loop 2 (hard_blocked only) and
        # cutting a long diagonal chord across the sweep rail.
        if (
            axial_core_mask is not None
            and self.grid.in_bounds(start_ij[0], start_ij[1])
            and (
                bool(axial_core_mask[start_ij[0], start_ij[1]])
                or (
                    len(path_ij) > 1
                    and self.grid.in_bounds(path_ij[1][0], path_ij[1][1])
                    and bool(axial_core_mask[path_ij[1][0], path_ij[1][1]])
                )
            )
        ):
            for k in range(2, max_k + 1):
                px_k, py_k = path_ij[k]
                if (
                    (not bool(axial_core_mask[px_k, py_k]))
                    and self._segment_is_clear(start_ij, path_ij[k], hard_blocked)
                ):
                    return self.grid.grid_to_world(px_k, py_k)

        # Prefer farthest chord that stays clear of both hard_blocked and close soft shoulders
        for k in range(max_k, 1, -1):
            if self._segment_is_clear(
                start_ij,
                path_ij[k],
                hard_blocked,
                soft_cost=soft_cost,
                max_soft=3.6,
                allow_soft_descent=use_soft_descent,
                axial_core_mask=axial_core_mask,
            ):
                return self.grid.grid_to_world(path_ij[k][0], path_ij[k][1])

        # Tight pinch: accept farthest straight segment free of hard_blocked
        for k in range(max_k, 0, -1):
            if self._segment_is_clear(start_ij, path_ij[k], hard_blocked):
                return self.grid.grid_to_world(path_ij[k][0], path_ij[k][1])

        sub_ij = path_ij[min(2, len(path_ij) - 1)]
        return self.grid.grid_to_world(sub_ij[0], sub_ij[1])

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
        raw_ranges: Optional[Sequence[float]] = None,
        semantic_obstacles: Optional[Sequence[Tuple[float, float, float]]] = None,
        semantic_hazards: Optional[Sequence[Tuple[float, float, float]]] = None,
        dynamic_obstacles: Optional[Sequence[object]] = None,
    ) -> Tuple[str, float, float, Optional[Tuple[float, float]]]:
        """
        Run one control tick for Proposed Layer-2 backtracking.

        Returns (status_tag, v_cmd, w_cmd, subgoal_xy):
          - status_tag == "ARRIVED": ready for handover/mission complete logic.
          - status_tag == "ESCAPE": executing reverse/pivot unstuck escape.
          - status_tag == "ASTAR_TRACK": tracking A* subgoal smoothly.
          - status_tag == "ROTATE_ALIGN": large heading alignment turn in place.
        """
        self.grid.update_scan(rx, ry, rth, scan_pts, raw_ranges=raw_ranges)
        front_min, left_min, right_min, rear_min = self._scan_sector_clearances(scan_pts)
        self.last_clearances = (front_min, left_min, right_min, rear_min)
        self.last_used_tight_pass = False

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

            # Infer a temporary virtual barrier ahead only when 2D LiDAR is genuinely open
            # (e.g. sub-LiDAR plinth), never inside a narrow visible rock/wall gap (<0.50m total width).
            is_narrow_pinch = (left_min + right_min) < 0.50 or min(left_min, right_min) < 0.20
            if stuck_by_disp and front_min >= 0.26 and not is_narrow_pinch:
                block_dir = self._last_fwd_pose_dir if self._last_fwd_pose_dir is not None else rth
                cos_d = math.cos(block_dir)
                sin_d = math.sin(block_dir)
                ttl = now_sec + 9.0
                self._virtual_obstacles.append(
                    (rx + 0.19 * cos_d, ry + 0.19 * sin_d, ttl)
                )
                if len(self._virtual_obstacles) > 8:
                    self._virtual_obstacles = self._virtual_obstacles[-8:]

            base_turn = 1.0 if left_min >= right_min else -1.0
            if self._repeat_escape_count > 0 and (self._repeat_escape_count % 2 == 1):
                base_turn = -base_turn

            self._escape_start = now_sec
            if is_narrow_pinch and self._repeat_escape_count == 0:
                self._escape_reverse_dur = 0.65
                self._escape_until = now_sec + 1.05
            elif self._repeat_escape_count >= 1:
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
                if self.phase2b_mode and self._repeat_escape_count >= 2:
                    v_esc = -0.09 if rear_min > 0.14 else 0.0
                else:
                    v_esc = -0.14 if rear_min > 0.22 else 0.0
                w_esc = 0.20 * self._escape_turn_dir
            else:
                v_esc = 0.0
                if self.phase2b_mode and self._repeat_escape_count >= 2:
                    turn_rate = w_max
                else:
                    turn_rate = 0.92 if self._repeat_escape_count >= 1 else 0.78
                w_esc = max(-w_max, min(w_max, turn_rate * self._escape_turn_dir))
            return "ESCAPE", float(v_esc), float(w_esc), None

        # 2. Rolling 2D A* + Topological Breadcrumb Shortcut (2-pass adaptive inflation)
        self._virtual_obstacles = [vo for vo in self._virtual_obstacles if vo[2] > now_sec]
        extra_pts = [(vo[0], vo[1]) for vo in self._virtual_obstacles] if self._virtual_obstacles else None
        wp_list = list(waypoint_stack)
        allow_shortcuts = now_sec >= self._no_shortcut_until
        cur_origin = (self.grid.origin_ix, self.grid.origin_iy)

        has_confirmed_dyn = bool(
            dynamic_obstacles
            and any(getattr(t, "is_confirmed", True) for t in dynamic_obstacles)
        )

        # Replan gate: reuse cached A* path & cost maps within period/displacement/stack/origin window
        # so [ICP/TRK/A*] cannot stall the 10 Hz control loop with 200-500 ms spikes.
        reuse_astar = (
            self._astar_cache is not None
            and self._astar_cache[0] is not None
            and self._astar_cache_maps is not None
            and getattr(self, "_astar_plan_origin", None) == cur_origin
            and (now_sec - self._astar_plan_time) < self.replan_period_sec
            and math.hypot(rx - self._astar_plan_x, ry - self._astar_plan_y) < self.replan_displacement_m
            and len(waypoint_stack) == self._astar_plan_wp_len
            and allow_shortcuts == self._astar_plan_shortcuts
            and has_confirmed_dyn == self._astar_plan_has_dyn
        )
        if reuse_astar:
            path_ij, chosen_idx = self._astar_cache
            hard_blocked, soft_cost = self._astar_cache_maps
            self.last_used_tight_pass = self._astar_cache_tight
            self.grid._last_axial_band_active = self._astar_cache_axial_band
            self.grid._last_axial_core_mask = self._astar_cache_axial_core
        else:
            self.last_used_tight_pass = False
            total_deadline = time.perf_counter() + 0.065
            hard_blocked, soft_cost = self.grid.build_cost_maps(
                rx,
                ry,
                extra_world_pts=extra_pts,
                tight=False,
                semantic_obstacles=semantic_obstacles,
                semantic_hazards=semantic_hazards,
                dynamic_obstacles=dynamic_obstacles,
                waypoint_list=(wp_list if has_confirmed_dyn else None),
            )
            pass1_deadline = min(total_deadline, time.perf_counter() + 0.035)
            path_ij, chosen_idx = self._plan_topological_astar(
                rx,
                ry,
                wp_list,
                hard_blocked,
                soft_cost,
                allow_shortcuts=allow_shortcuts,
                deadline=pass1_deadline,
                fallback_budget_sec=0.005,
            )
            if (path_ij is None or len(path_ij) == 0) and time.perf_counter() < total_deadline:
                # Pass 2: narrow indoor corridor rock-bypass (~0.28-0.31m gap) with tight C-space margin
                hard_tight, soft_tight = self.grid.build_cost_maps(
                    rx,
                    ry,
                    extra_world_pts=extra_pts,
                    tight=True,
                    semantic_obstacles=semantic_obstacles,
                    semantic_hazards=semantic_hazards,
                    dynamic_obstacles=dynamic_obstacles,
                    reuse_soft_cost=soft_cost,
                    waypoint_list=(wp_list if has_confirmed_dyn else None),
                )
                pass2_deadline = min(total_deadline, time.perf_counter() + 0.025)
                path_t, chosen_t = self._plan_topological_astar(
                    rx,
                    ry,
                    wp_list,
                    hard_tight,
                    soft_tight,
                    allow_shortcuts=allow_shortcuts,
                    deadline=pass2_deadline,
                    fallback_budget_sec=0.005,
                )
                if path_t is not None and len(path_t) > 0:
                    path_ij, chosen_idx = path_t, chosen_t
                    hard_blocked, soft_cost = hard_tight, soft_tight
                    self.last_used_tight_pass = True
            self._astar_cache = (path_ij, chosen_idx)
            self._astar_cache_maps = (hard_blocked, soft_cost)
            self._astar_cache_tight = self.last_used_tight_pass
            self._astar_cache_axial_band = bool(getattr(self.grid, "_last_axial_band_active", False))
            self._astar_cache_axial_core = getattr(self.grid, "_last_axial_core_mask", None)
            self._astar_plan_time = now_sec
            self._astar_plan_x = rx
            self._astar_plan_y = ry
            self._astar_plan_wp_len = len(waypoint_stack)
            self._astar_plan_shortcuts = allow_shortcuts
            self._astar_plan_has_dyn = has_confirmed_dyn
            self._astar_plan_origin = cur_origin
        self._last_chosen_idx = chosen_idx
        self.last_path_ij = path_ij

        # Prune immediate rock-buried breadcrumbs when a downstream WP is already safely targeted
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
        elif len(waypoint_stack) > 1 and waypoint_stack[-1].distance_to(rx, ry) < 0.70:
            tix, tiy = self.grid.world_to_grid(waypoint_stack[-1].x, waypoint_stack[-1].y)
            if self.grid.in_bounds(tix, tiy) and hard_blocked[tix, tiy]:
                waypoint_stack.pop()

        # Extract line-of-sight safe lookahead subgoal (prevents corner-cutting into rock shoulders)
        if path_ij is not None and len(path_ij) > 0:
            sub_x, sub_y = self._extract_safe_subgoal(path_ij, hard_blocked, soft_cost)
        else:
            # No A* path even under tight-gap relaxation: backup/pivot safely instead of ramming
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
            # Pivot toward A* subgoal when clearly offset, else toward wider side gap (no +/-0.15 chatter)
            if abs(alpha) >= 0.25:
                w_dir = 1.0 if alpha >= 0.0 else -1.0
            else:
                w_dir = 1.0 if left_min >= right_min else -1.0
            v_unpin = -0.06 if (front_min < 0.14 and rear_min > 0.24) else 0.0
            return "ROTATE_ALIGN", float(v_unpin), float(max(-w_max, min(w_max, 0.55 * w_dir))), (sub_x, sub_y)

        if front_min < 0.32:
            prox_v = 0.075 + (v_max - 0.075) * max(0.0, (front_min - 0.165) / (0.32 - 0.165))
            v_cmd = min(v_max, prox_v * ang_scale)
        else:
            v_cmd = min(v_max, v_max * ang_scale)

        if v_cmd > 0.02:
            self._last_fwd_pose_dir = rth
        w_cmd = max(-w_max, min(w_max, 1.75 * alpha))
        return "ASTAR_TRACK", float(v_cmd), float(w_cmd), (sub_x, sub_y)
