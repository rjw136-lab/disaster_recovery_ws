#!/usr/bin/env python3
"""
Stage 1 (Phase 4-B): Static-Background-Subtraction Dynamic Obstacle Tracker
and Damped-CV Uncertainty Predictor for Disaster Autonomous Recovery.

Design constraints & mathematical pipeline:
  - Pure Python + NumPy only (no rclpy, SciPy, PyTorch, or OpenCV).
  - Dual-trap static/dynamic separation combining ray-casted `observed_free`
    verification with dynamic-masked `static_occ` persistence (decay 0.992).
  - O(N) 1D angular sequential clustering with 1-2 beam dropout gap tolerance,
    0 deg / 359 deg wrap-around merge, no upper ray-count cap (supports close
    d <= 0.45 m cylinders), Euclidean chord-length [0.08, 0.70] m and physical
    radius [0.05, 0.45] m filtering.
  - Taubin least-squares circle fit for surface-arc centroid bias removal with
    outward ray-direction fallback.
  - Single-model continuous-damped CV Kalman filter (gamma_v = exp(-beta * dt),
    beta = 0.35 s^-1), unified 4x4 covariance propagation P(tau) = F P F^T + Q,
    3x adaptive q_acc expansion on velocity reversal or high innovation,
    early confirmation (2 hits & ||v|| >= 0.08 m/s), 2-tick coasting.
  - 5 Hz scan.header.stamp gating, non-mutating mid-tick KF extrapolation,
    SE(2) waypoint-ICP correction jump synchronization, and 12x12 Ch3 ego grid
    projection with corrected denominator max(r_eff - r_c, eps).
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PATCH_SIZE = 12
PATCH_SPAN_M = 1.8


def _build_ego_grid_xy() -> Tuple[np.ndarray, np.ndarray]:
    """Precompute 12x12 ego patch cell centers (+x forward, +y left)."""
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


@dataclass
class TrackedObstacle:
    """Tracked dynamic obstacle state in world (odom/corrected) frame."""

    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    P_4x4: np.ndarray
    last_update_stamp: float
    age: int
    hits: int
    is_confirmed: bool
    consecutive_hits: int = 1
    missed_ticks: int = 0
    q_acc_scale: float = 1.0
    vel_history: Tuple[Tuple[float, float], ...] = ()

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)


def damped_cv_matrices(
    dt: float,
    beta: float = 0.35,
    q_acc: float = 0.30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exact continuous-time damped-CV transition F(dt) and process noise Q(dt)."""
    tau = max(0.0, float(dt))
    b = max(1e-6, float(beta))
    u = b * tau

    if u < 1e-4:
        e1 = 1.0 - u + 0.5 * u * u
        phi = tau * (1.0 - 0.5 * u + (u * u) / 6.0)
    else:
        e1 = math.exp(-u)
        phi = (1.0 - e1) / b

    F = np.array(
        [
            [1.0, 0.0, phi, 0.0],
            [0.0, 1.0, 0.0, phi],
            [0.0, 0.0, e1, 0.0],
            [0.0, 0.0, 0.0, e1],
        ],
        dtype=np.float64,
    )

    if u < 1e-3:
        q_vv = tau
        q_pv = 0.5 * tau * tau
        q_pp = (tau * tau * tau) / 3.0
    else:
        e2 = math.exp(-2.0 * u)
        q_vv = (1.0 - e2) / (2.0 * b)
        q_pv = ((1.0 - e1) ** 2) / (2.0 * b * b)
        q_pp = (2.0 * u - 3.0 + 4.0 * e1 - e2) / (2.0 * (b ** 3))

    q = max(0.0, float(q_acc))
    Q = q * np.array(
        [
            [q_pp, 0.0, q_pv, 0.0],
            [0.0, q_pp, 0.0, q_pv],
            [q_pv, 0.0, q_vv, 0.0],
            [0.0, q_pv, 0.0, q_vv],
        ],
        dtype=np.float64,
    )
    return F, Q


def propagate_state_cov(
    state: np.ndarray,
    cov_4x4: np.ndarray,
    tau: float,
    beta: float = 0.35,
    q_acc: float = 0.30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unified 4x4 state & covariance propagation: s(tau) = F s, P(tau) = F P F^T + Q."""
    F, Q = damped_cv_matrices(tau, beta=beta, q_acc=q_acc)
    s_next = F @ state
    P_next = F @ cov_4x4 @ F.T + Q
    P_next = 0.5 * (P_next + P_next.T)
    return s_next, P_next


def max_eigenvalue_2x2(cov_2x2: np.ndarray) -> float:
    """Closed-form largest eigenvalue of a symmetric 2x2 covariance matrix."""
    a = float(cov_2x2[0, 0])
    b = float(cov_2x2[0, 1])
    c = float(cov_2x2[1, 1])
    disc = math.sqrt(max(0.0, (a - c) * (a - c) + 4.0 * b * b))
    return max(0.0, 0.5 * (a + c + disc))


def fit_circle_taubin(
    pts: np.ndarray,
) -> Optional[Tuple[float, float, float, float, float]]:
    """
    Lightweight Taubin algebraic circle fit on Nx2 points (N >= 3).

    Returns (cx, cy, radius, spread_ratio, rmse) or None if degenerate.
    `spread_ratio = lam_min / lam_max` of the 2D point covariance measures
    conditioning against nearly collinear wall fragments.
    """
    n = pts.shape[0]
    if n < 3:
        return None

    m = pts.mean(axis=0)
    x = pts[:, 0].astype(np.float64) - float(m[0])
    y = pts[:, 1].astype(np.float64) - float(m[1])
    z = x * x + y * y

    Mxx = float(np.mean(x * x))
    Myy = float(np.mean(y * y))
    Mxy = float(np.mean(x * y))
    tr = Mxx + Myy
    if tr < 1e-10:
        return None
    disc_cov = math.sqrt(max(0.0, (Mxx - Myy) ** 2 + 4.0 * Mxy * Mxy))
    lam_max = 0.5 * (tr + disc_cov)
    lam_min = max(0.0, 0.5 * (tr - disc_cov))
    spread_ratio = lam_min / max(lam_max, 1e-12)

    Mxz = float(np.mean(x * z))
    Myz = float(np.mean(y * z))
    Mzz = float(np.mean(z * z))
    Mz = tr
    Cov_xy = Mxx * Myy - Mxy * Mxy
    Var_z = Mzz - Mz * Mz

    A3 = 4.0 * Mz
    A2 = -3.0 * Mz * Mz - Mzz
    A1 = Var_z * Mz + 4.0 * Cov_xy * Mz - Mxz * Mxz - Myz * Myz
    A0 = (
        Mxz * (Mxz * Myy - Myz * Mxy)
        + Myz * (Myz * Mxx - Mxz * Mxy)
        - Var_z * Cov_xy
    )
    A22 = A2 + A2
    A33 = 3.0 * A3

    x_root = 0.0
    y_val = A0
    for _ in range(12):
        Dy = A1 + x_root * (A22 + x_root * A33)
        if abs(Dy) < 1e-15:
            break
        x_new = x_root - y_val / Dy
        y_new = A0 + x_new * (A1 + x_new * (A2 + x_new * A3))
        if abs(y_new) >= abs(y_val):
            break
        x_root, y_val = x_new, y_new
        if abs(y_new) < 1e-12:
            break

    det = x_root * x_root - x_root * Mz + Cov_xy
    if abs(det) < 1e-12:
        return None

    dx = (Mxz * (Myy - x_root) - Myz * Mxy) / (2.0 * det)
    dy = (Myz * (Mxx - x_root) - Mxz * Mxy) / (2.0 * det)
    r2 = dx * dx + dy * dy + Mz + 2.0 * x_root
    if r2 <= 0.0 or not math.isfinite(r2):
        return None

    radius = math.sqrt(r2)
    cx = float(m[0]) + dx
    cy = float(m[1]) + dy
    if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(radius)):
        return None

    dists = np.hypot(pts[:, 0].astype(np.float64) - cx, pts[:, 1].astype(np.float64) - cy)
    rmse = float(np.sqrt(np.mean((dists - radius) ** 2)))
    return cx, cy, radius, spread_ratio, rmse


def correct_cluster_center_and_radius(
    pts: np.ndarray,
    robot_xy: Tuple[float, float],
    min_radius_m: float = 0.05,
    max_radius_m: float = 0.45,
    kappa_arc: float = 0.65,
    min_spread_ratio: float = 0.012,
    max_fit_rmse_m: float = 0.06,
) -> Optional[Tuple[float, float, float]]:
    """
    Estimate true obstacle center (cx, cy) and physical radius r_c from a 2D surface arc.

    Runs Taubin circle fitting first; when conditioning is poor or fitted radius lies
    outside [min_radius_m, max_radius_m], falls back to physical bounding/half-chord
    radius + outward ray-direction surface-arc bias compensation:
        p_center = p_mean + kappa_arc * r_hat * e_r
    """
    if pts.shape[0] < 3:
        return None

    mean_x = float(np.mean(pts[:, 0]))
    mean_y = float(np.mean(pts[:, 1]))
    vx = mean_x - float(robot_xy[0])
    vy = mean_y - float(robot_xy[1])
    ray_len = math.hypot(vx, vy)
    if ray_len < 1e-6:
        return None
    er_x = vx / ray_len
    er_y = vy / ray_len

    fit = fit_circle_taubin(pts)
    if fit is not None:
        cx, cy, r_fit, spread_ratio, rmse = fit
        # Verify center lies behind (away from robot wrt) the observed front arc
        proj_along_ray = (cx - mean_x) * er_x + (cy - mean_y) * er_y
        if (
            spread_ratio >= min_spread_ratio
            and min_radius_m <= r_fit <= max_radius_m
            and rmse <= max_fit_rmse_m
            and proj_along_ray >= -0.04
        ):
            return cx, cy, r_fit

    # Ray-direction centroid bias fallback (partial arc / flat moving box)
    chord_m = float(math.hypot(pts[-1, 0] - pts[0, 0], pts[-1, 1] - pts[0, 1]))
    max_mean_dist = float(np.max(np.hypot(pts[:, 0] - mean_x, pts[:, 1] - mean_y)))
    r_hat = max(0.5 * chord_m, max_mean_dist)
    if not (min_radius_m <= r_hat <= max_radius_m):
        return None

    shift = float(kappa_arc) * r_hat
    return mean_x + shift * er_x, mean_y + shift * er_y, r_hat


def _hungarian_min_cost(cost: np.ndarray) -> List[Tuple[int, int]]:
    """
    Exact O(n^3) Hungarian linear sum assignment for small rectangular cost matrices.
    Pure Python/NumPy implementation (zero SciPy dependency).
    """
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return []

    transposed = False
    if n_rows > n_cols:
        cost = cost.T
        n_rows, n_cols = cost.shape
        transposed = True

    u = np.zeros(n_rows + 1, dtype=np.float64)
    v = np.zeros(n_cols + 1, dtype=np.float64)
    p = np.zeros(n_cols + 1, dtype=np.int64)
    way = np.zeros(n_cols + 1, dtype=np.int64)

    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n_cols + 1, math.inf, dtype=np.float64)
        used = np.zeros(n_cols + 1, dtype=bool)
        row_i = cost[i - 1]
        while True:
            used[j0] = True
            i0 = int(p[j0])
            delta = math.inf
            j1 = 0
            row_i0 = cost[i0 - 1] if i0 > 0 else row_i
            for j in range(1, n_cols + 1):
                if not used[j]:
                    cur = float(row_i0[j - 1]) - float(u[i0]) - float(v[j])
                    if cur < float(minv[j]):
                        minv[j] = cur
                        way[j] = j0
                    if float(minv[j]) < delta:
                        delta = float(minv[j])
                        j1 = j
            for j in range(n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs: List[Tuple[int, int]] = []
    for j in range(1, n_cols + 1):
        if p[j] > 0:
            r_idx = int(p[j]) - 1
            c_idx = j - 1
            pairs.append((c_idx, r_idx) if transposed else (r_idx, c_idx))
    return pairs


def _build_dilation_offsets(radius_cells: int) -> Tuple[np.ndarray, np.ndarray]:
    dis: List[int] = []
    djs: List[int] = []
    for di in range(-radius_cells, radius_cells + 1):
        for dj in range(-radius_cells, radius_cells + 1):
            dis.append(di)
            djs.append(dj)
    return np.asarray(dis, dtype=np.int32), np.asarray(djs, dtype=np.int32)


class DynamicObstacleTracker:
    """
    Stage 1 dynamic obstacle detector, damped-CV Kalman tracker, and Ch3 predictor.

    Maintains a world-aligned rolling background map (`static_occ` + ray-carved
    `observed_free`), tracks dynamic cylindrical/compact obstacles across 5 Hz scan
    updates, supports non-mutating mid-tick KF extrapolation, synchronizes tracks and
    maps under SE(2) waypoint-ICP correction jumps, and generates the 12x12 Ch3 ego grid.
    """

    def __init__(
        self,
        grid_size_m: float = 6.0,
        grid_res_m: float = 0.04,
        static_occ_decay: float = 0.992,
        static_occ_thresh: float = 0.75,
        observed_free_decay: float = 0.985,
        observed_free_thresh: float = 0.50,
        omega_rot_guard_rad_s: float = 0.45,
        dynamic_mask_margin_m: float = 0.08,
        cluster_dist_thresh_m: float = 0.18,
        cluster_max_ray_gap: int = 2,
        min_cluster_points: int = 3,
        min_chord_m: float = 0.08,
        max_chord_m: float = 0.70,
        min_radius_m: float = 0.05,
        max_radius_m: float = 0.45,
        kappa_arc: float = 0.65,
        beta_damping: float = 0.35,
        q_acc_base: float = 0.30,
        q_acc_adapt_mult: float = 3.0,
        meas_noise_std_m: float = 0.030,
        init_vel_std_mps: float = 0.60,
        gating_dist_m: float = 0.45,
        confirm_min_hits: int = 2,
        confirm_min_speed_mps: float = 0.08,
        max_coast_ticks: int = 2,
        lidar_min_range_m: float = 0.12,
        lidar_max_range_m: float = 3.50,
        ch3_horizons: Sequence[float] = (0.5, 1.0, 1.5),
        ch3_weights: Sequence[float] = (1.0, 0.75, 0.50),
        ch3_eps: float = 1e-3,
    ) -> None:
        self.grid_size_m = float(grid_size_m)
        self.res = float(grid_res_m)
        self.n = int(round(self.grid_size_m / self.res))
        self.static_occ_decay = float(static_occ_decay)
        self.static_occ_thresh = float(static_occ_thresh)
        self.observed_free_decay = float(observed_free_decay)
        self.observed_free_thresh = float(observed_free_thresh)
        self.omega_rot_guard_rad_s = float(omega_rot_guard_rad_s)
        self.dynamic_mask_margin_m = float(dynamic_mask_margin_m)
        self.cluster_dist_thresh_m = float(cluster_dist_thresh_m)
        self.cluster_max_ray_gap = int(cluster_max_ray_gap)
        self.min_cluster_points = int(min_cluster_points)
        self.min_chord_m = float(min_chord_m)
        self.max_chord_m = float(max_chord_m)
        self.min_radius_m = float(min_radius_m)
        self.max_radius_m = float(max_radius_m)
        self.kappa_arc = float(kappa_arc)
        self.beta = float(beta_damping)
        self.q_acc_base = float(q_acc_base)
        self.q_acc_adapt_mult = float(q_acc_adapt_mult)
        self.meas_noise_std_m = float(meas_noise_std_m)
        self.init_vel_std_mps = float(init_vel_std_mps)
        self.gating_dist_m = float(gating_dist_m)
        self.confirm_min_hits = int(confirm_min_hits)
        self.confirm_min_speed_mps = float(confirm_min_speed_mps)
        self.max_coast_ticks = int(max_coast_ticks)
        self.lidar_min_range_m = float(lidar_min_range_m)
        self.lidar_max_range_m = float(lidar_max_range_m)
        self.ch3_horizons = tuple(float(h) for h in ch3_horizons)
        self.ch3_weights = tuple(float(w) for w in ch3_weights)
        self.ch3_eps = float(ch3_eps)

        self.static_occ = np.zeros((self.n, self.n), dtype=np.float32)
        self.observed_free = np.zeros((self.n, self.n), dtype=np.float32)
        self.grid_initialized = False
        self.origin_ix = 0
        self.origin_iy = 0

        self._dilate_mask_buf = np.zeros((self.n, self.n), dtype=bool)
        self._carve_mask_buf = np.zeros((self.n, self.n), dtype=bool)
        self._ray_fracs = np.linspace(0.06, 0.90, 28, dtype=np.float32)[:, None]

        self._R = np.eye(2, dtype=np.float64) * (self.meas_noise_std_m ** 2)
        self._H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        self._I4 = np.eye(4, dtype=np.float64)
        self._ch3_fq_cache: Dict[float, Tuple[np.ndarray, np.ndarray]] = {
            h: damped_cv_matrices(h, beta=self.beta, q_acc=self.q_acc_base)
            for h in self.ch3_horizons
        }

        self.tracks: Dict[int, TrackedObstacle] = {}
        self._next_track_id = 1
        self._last_scan_stamp: Optional[float] = None
        self._cached_ray_count = -1
        self._cached_angle_min = 0.0
        self._cached_angle_inc = 0.0
        self._cached_cos = np.empty(0, dtype=np.float32)
        self._cached_sin = np.empty(0, dtype=np.float32)

        self.last_did_update: bool = False
        self.scan_update_count: int = 0
        self.extrap_tick_count: int = 0

    def reset(self) -> None:
        """Clear all tracks, background occupancy maps, and timestamp state."""
        self.static_occ.fill(0.0)
        self.observed_free.fill(0.0)
        self.grid_initialized = False
        self.origin_ix = 0
        self.origin_iy = 0
        self.tracks.clear()
        self._next_track_id = 1
        self._last_scan_stamp = None
        self.last_did_update = False
        self.scan_update_count = 0
        self.extrap_tick_count = 0

    def _get_trig_lut(
        self, n_rays: int, angle_min: float, angle_increment: Optional[float]
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        inc = (2.0 * math.pi / n_rays) if angle_increment is None else float(angle_increment)
        if (
            n_rays != self._cached_ray_count
            or abs(angle_min - self._cached_angle_min) > 1e-7
            or abs(inc - self._cached_angle_inc) > 1e-7
        ):
            angs = angle_min + np.arange(n_rays, dtype=np.float32) * np.float32(inc)
            self._cached_cos = np.cos(angs).astype(np.float32)
            self._cached_sin = np.sin(angs).astype(np.float32)
            self._cached_ray_count = n_rays
            self._cached_angle_min = float(angle_min)
            self._cached_angle_inc = float(inc)
        return self._cached_cos, self._cached_sin, inc

    def _recenter_grid(self, rx: float, ry: float) -> None:
        desired_ox = int(math.floor(rx / self.res)) - self.n // 2
        desired_oy = int(math.floor(ry / self.res)) - self.n // 2
        if not self.grid_initialized:
            self.origin_ix = desired_ox
            self.origin_iy = desired_oy
            self.grid_initialized = True
            return

        dx = desired_ox - self.origin_ix
        dy = desired_oy - self.origin_iy
        if abs(dx) < 2 and abs(dy) < 2:
            return

        if abs(dx) >= self.n or abs(dy) >= self.n:
            self.static_occ.fill(0.0)
            self.observed_free.fill(0.0)
        else:
            new_s = np.zeros_like(self.static_occ)
            new_f = np.zeros_like(self.observed_free)
            sx0 = max(0, dx)
            sx1 = min(self.n, self.n + dx)
            sy0 = max(0, dy)
            sy1 = min(self.n, self.n + dy)
            dx0 = sx0 - dx
            dx1 = sx1 - dx
            dy0 = sy0 - dy
            dy1 = sy1 - dy
            new_s[dx0:dx1, dy0:dy1] = self.static_occ[sx0:sx1, sy0:sy1]
            new_f[dx0:dx1, dy0:dy1] = self.observed_free[sx0:sx1, sy0:sy1]
            self.static_occ = new_s
            self.observed_free = new_f

        self.origin_ix = desired_ox
        self.origin_iy = desired_oy

    def _check_static_mask_at_cells(
        self, ix: np.ndarray, iy: np.ndarray, omega: float
    ) -> np.ndarray:
        """
        Evaluate dilated static_occ >= 0.75 mask at hit cell indices via fast slice dilation.
        Uses 3x3 dilation normally and 5x5 (+1 cell = 0.04 m) when |omega| > 0.45 rad/s.
        """
        s = self.static_occ >= np.float32(self.static_occ_thresh)
        m = self._dilate_mask_buf
        np.copyto(m, s)
        m[1:, :] |= s[:-1, :]
        m[:-1, :] |= s[1:, :]
        mh = m.copy()
        mh[:, 1:] |= m[:, :-1]
        mh[:, :-1] |= m[:, 1:]
        if abs(float(omega)) > self.omega_rot_guard_rad_s:
            m2 = mh.copy()
            m2[1:, :] |= mh[:-1, :]
            m2[:-1, :] |= mh[1:, :]
            mh = m2.copy()
            mh[:, 1:] |= m2[:, :-1]
            mh[:, :-1] |= m2[:, 1:]

        inb = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
        ix_c = np.clip(ix, 0, self.n - 1)
        iy_c = np.clip(iy, 0, self.n - 1)
        return inb & mh[ix_c, iy_c]

    def _sample_observed_free(
        self,
        wx: np.ndarray,
        wy: np.ndarray,
        ix: np.ndarray,
        iy: np.ndarray,
        external_observed_free: Optional[np.ndarray],
        ext_origin_ix: int,
        ext_origin_iy: int,
        ext_res: float,
    ) -> np.ndarray:
        inb = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
        ix_c = np.clip(ix, 0, self.n - 1)
        iy_c = np.clip(iy, 0, self.n - 1)
        obs = np.where(inb, self.observed_free[ix_c, iy_c], 0.0)

        if external_observed_free is not None and ext_res > 0.0:
            enx, eny = external_observed_free.shape
            eix = np.floor(wx / ext_res).astype(np.int32) - int(ext_origin_ix)
            eiy = np.floor(wy / ext_res).astype(np.int32) - int(ext_origin_iy)
            einb = (eix >= 0) & (eix < enx) & (eiy >= 0) & (eiy < eny)
            eix_c = np.clip(eix, 0, enx - 1)
            eiy_c = np.clip(eiy, 0, eny - 1)
            obs_ext = np.where(einb, external_observed_free[eix_c, eiy_c], 0.0)
            obs = np.maximum(obs, obs_ext)
        return obs

    def _point_near_active_track(self, wx: np.ndarray, wy: np.ndarray, gate_m: float) -> np.ndarray:
        if not self.tracks:
            return np.zeros(len(wx), dtype=bool)
        g2 = gate_m * gate_m
        near = np.zeros(len(wx), dtype=bool)
        for trk in self.tracks.values():
            r_allow = gate_m + trk.radius
            d2 = (wx - trk.x) ** 2 + (wy - trk.y) ** 2
            near |= d2 <= (r_allow * r_allow if r_allow * r_allow > g2 else g2)
        return near

    def _extract_clusters(
        self,
        cand_idx: np.ndarray,
        cand_pts_w: np.ndarray,
        n_rays_total: int,
        is_full_360: bool,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        O(N) 1D angular sequential clustering with 1-2 ray dropout gap allowance
        and 0 deg / 359 deg wrap-around merge. Returns list of (world_pts, member_positions).
        """
        m = len(cand_idx)
        if m == 0:
            return []

        max_idx_diff = self.cluster_max_ray_gap + 1
        dthresh2 = self.cluster_dist_thresh_m * self.cluster_dist_thresh_m

        if m == 1:
            split_indices: List[int] = []
        else:
            gaps = cand_idx[1:] - cand_idx[:-1]
            diff = cand_pts_w[1:] - cand_pts_w[:-1]
            d2 = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]
            breaks = np.flatnonzero((gaps > max_idx_diff) | (d2 >= dthresh2))
            split_indices = (breaks + 1).tolist()

        clusters: List[np.ndarray] = np.split(np.arange(m), split_indices) if split_indices else [np.arange(m)]

        # 0 deg / 359 deg wrap-around merge preserving arc endpoint ordering
        if is_full_360 and len(clusters) >= 2:
            first_c = clusters[0]
            last_c = clusters[-1]
            i_first = int(cand_idx[first_c[0]])
            i_last = int(cand_idx[last_c[-1]])
            wrap_gap = (i_first + n_rays_total) - i_last
            if 1 <= wrap_gap <= max_idx_diff:
                dx = float(cand_pts_w[first_c[0], 0] - cand_pts_w[last_c[-1], 0])
                dy = float(cand_pts_w[first_c[0], 1] - cand_pts_w[last_c[-1], 1])
                if (dx * dx + dy * dy) < dthresh2:
                    merged = np.concatenate([last_c, first_c])
                    clusters = [merged] + clusters[1:-1]

        return [(cand_pts_w[c_ids], c_ids) for c_ids in clusters]

    def _update_background_grids(
        self,
        rx: float,
        ry: float,
        rth: float,
        ranges: np.ndarray,
        cos_lut: np.ndarray,
        sin_lut: np.ndarray,
        wx_valid: np.ndarray,
        wy_valid: np.ndarray,
        ix_valid: np.ndarray,
        iy_valid: np.ndarray,
        pre_obs_free_valid: np.ndarray,
    ) -> None:
        """
        Update observed_free and static_occ AFTER current scan dynamic extraction.
        Confirmed tracks mask out static_occ updates within (r_c + 0.08 m) to prevent
        harmonic oscillation turnaround absorption into the static background.
        """
        self.static_occ *= np.float32(self.static_occ_decay)
        self.observed_free *= np.float32(self.observed_free_decay)

        inv_res = np.float32(1.0 / self.res)
        c = math.cos(rth)
        s = math.sin(rth)
        rx_f = np.float32(rx)
        ry_f = np.float32(ry)

        # Combine open max-range rays and clipped hit rays into ONE vectorized ray-carve pass
        end_x_list: List[np.ndarray] = []
        end_y_list: List[np.ndarray] = []

        open_mask = (~np.isfinite(ranges)) | (ranges > np.float32(self.lidar_max_range_m - 0.05))
        if np.any(open_mask):
            max_r_o = np.float32(self.lidar_max_range_m - 0.10)
            lx_o = max_r_o * cos_lut[open_mask]
            ly_o = max_r_o * sin_lut[open_mask]
            end_x_list.append(rx_f + np.float32(c) * lx_o - np.float32(s) * ly_o)
            end_y_list.append(ry_f + np.float32(s) * lx_o + np.float32(c) * ly_o)

        if len(wx_valid) > 0:
            dx_w = wx_valid - rx_f
            dy_w = wy_valid - ry_f
            d_w = np.hypot(dx_w, dy_w)
            carve_mask = d_w > np.float32(0.16)
            if np.any(carve_mask):
                dw_c = d_w[carve_mask]
                scale = np.minimum(np.float32(0.90), np.maximum(np.float32(0.0), dw_c - np.float32(0.06)) / dw_c)
                end_x_list.append(rx_f + dx_w[carve_mask] * scale)
                end_y_list.append(ry_f + dy_w[carve_mask] * scale)

        if end_x_list:
            ex_all = np.concatenate(end_x_list) if len(end_x_list) > 1 else end_x_list[0]
            ey_all = np.concatenate(end_y_list) if len(end_y_list) > 1 else end_y_list[0]
            ray_x = rx_f + self._ray_fracs * (ex_all[None, :] - rx_f)
            ray_y = ry_f + self._ray_fracs * (ey_all[None, :] - ry_f)
            ix_f = np.floor(ray_x * inv_res).astype(np.int32) - self.origin_ix
            iy_f = np.floor(ray_y * inv_res).astype(np.int32) - self.origin_iy
            mf = (ix_f >= 0) & (ix_f < self.n) & (iy_f >= 0) & (iy_f < self.n)
            flat_idx = ix_f[mf] * self.n + iy_f[mf]
            cm = self._carve_mask_buf
            cm.fill(False)
            cm.ravel()[flat_idx] = True
            self.observed_free[cm] = 1.0
            self.static_occ[cm] *= np.float32(0.85)

        if len(wx_valid) > 0:
            # Dynamic masking: exclude hit cells within confirmed track radius + 0.08 m
            masked_by_dyn = np.zeros(len(wx_valid), dtype=bool)
            for trk in self.tracks.values():
                if trk.is_confirmed:
                    rad = trk.radius + self.dynamic_mask_margin_m
                    masked_by_dyn |= ((wx_valid - trk.x) ** 2 + (wy_valid - trk.y) ** 2) <= (rad * rad)

            update_mask = (~masked_by_dyn) & (ix_valid >= 0) & (ix_valid < self.n) & (iy_valid >= 0) & (iy_valid < self.n)
            if np.any(update_mask):
                ix_u = ix_valid[update_mask]
                iy_u = iy_valid[update_mask]
                was_free = pre_obs_free_valid[update_mask] >= np.float32(self.observed_free_thresh)

                # Unseen / non-free hits immediately establish static_occ >= 0.75;
                # previously-free hits build slowly (+0.34/tick) so moving objects can confirm
                # within 2 ticks before being absorbed, whereas new static drops solidify by tick 3.
                never_free = ~was_free
                if np.any(never_free):
                    self.static_occ[ix_u[never_free], iy_u[never_free]] = 1.0
                if np.any(was_free):
                    ix_wf = ix_u[was_free]
                    iy_wf = iy_u[was_free]
                    self.static_occ[ix_wf, iy_wf] = np.minimum(
                        np.float32(1.0), self.static_occ[ix_wf, iy_wf] + np.float32(0.34)
                    )

                self.observed_free[ix_u, iy_u] = 0.0

        sx = int(math.floor(rx / self.res)) - self.origin_ix
        sy = int(math.floor(ry / self.res)) - self.origin_iy
        if 0 <= sx < self.n and 0 <= sy < self.n:
            self.observed_free[sx, sy] = 1.0
            self.static_occ[sx, sy] *= np.float32(0.25)

    def _kf_step_and_associate(
        self,
        detections: List[Tuple[float, float, float]],
        dt: float,
        stamp_sec: float,
        freeze_velocity: bool = False,
    ) -> None:
        track_ids = list(self.tracks.keys())
        n_trk = len(track_ids)
        n_det = len(detections)

        # 1. Predict all existing tracks using exact damped-CV F(dt) and adaptive q_acc Q(dt)
        pred_states: Dict[int, np.ndarray] = {}
        pred_covs: Dict[int, np.ndarray] = {}
        for tid in track_ids:
            trk = self.tracks[tid]
            s0 = np.array([trk.x, trk.y, trk.vx, trk.vy], dtype=np.float64)
            q_eff = self.q_acc_base * trk.q_acc_scale
            s_p, P_p = propagate_state_cov(s0, trk.P_4x4, dt, beta=self.beta, q_acc=q_eff)
            if freeze_velocity:
                s_p[2] = trk.vx
                s_p[3] = trk.vy
            pred_states[tid] = s_p
            pred_covs[tid] = P_p

        matched_trk = set()
        matched_det = set()

        # 2. Gated minimum-cost Mahalanobis bipartite matching
        if n_trk > 0 and n_det > 0:
            big_cost = 1e6
            cost_mat = np.full((n_trk, n_det), big_cost, dtype=np.float64)
            gate_d2 = self.gating_dist_m * self.gating_dist_m
            for r_i, tid in enumerate(track_ids):
                s_p = pred_states[tid]
                S = pred_covs[tid][:2, :2] + self._R
                det_S = float(S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0])
                if det_S <= 1e-12:
                    continue
                inv_S = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]], dtype=np.float64) / det_S
                for c_j, (zx, zy, z_r) in enumerate(detections):
                    yx = zx - float(s_p[0])
                    yy = zy - float(s_p[1])
                    d2 = yx * yx + yy * yy
                    if d2 > gate_d2:
                        continue
                    maha2 = yx * (inv_S[0, 0] * yx + inv_S[0, 1] * yy) + yy * (inv_S[1, 0] * yx + inv_S[1, 1] * yy)
                    if maha2 <= 25.0:
                        cost_mat[r_i, c_j] = maha2 + 0.5 * d2

            for r_i, c_j in _hungarian_min_cost(cost_mat):
                if cost_mat[r_i, c_j] < 100.0:
                    tid = track_ids[r_i]
                    matched_trk.add(tid)
                    matched_det.add(c_j)
                    zx, zy, z_r = detections[c_j]
                    trk = self.tracks[tid]
                    s_p = pred_states[tid]
                    P_p = pred_covs[tid]

                    z = np.array([zx, zy], dtype=np.float64)
                    y = z - s_p[:2]
                    S = P_p[:2, :2] + self._R
                    det_S = float(S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0])
                    inv_S = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]], dtype=np.float64) / max(det_S, 1e-12)
                    nis = float(y @ inv_S @ y)

                    # Innovation-triggered 3x q_acc expansion if turnaround jump exceeds 95% chi2 threshold
                    if nis > 6.0 and trk.q_acc_scale < self.q_acc_adapt_mult:
                        s0 = np.array([trk.x, trk.y, trk.vx, trk.vy], dtype=np.float64)
                        _, P_p = propagate_state_cov(
                            s0,
                            trk.P_4x4,
                            dt,
                            beta=self.beta,
                            q_acc=self.q_acc_base * self.q_acc_adapt_mult,
                        )
                        S = P_p[:2, :2] + self._R
                        det_S = float(S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0])
                        inv_S = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]], dtype=np.float64) / max(det_S, 1e-12)

                    K = P_p[:, :2] @ inv_S
                    if freeze_velocity:
                        K_use = K.copy()
                        K_use[2:, :] = 0.0
                        s_new = s_p + K_use @ y
                        s_new[2] = trk.vx
                        s_new[3] = trk.vy
                        P_new = P_p
                    else:
                        s_new = s_p + K @ y
                        I_KH = self._I4 - K @ self._H
                        P_new = I_KH @ P_p @ I_KH.T + K @ self._R @ K.T
                        P_new = 0.5 * (P_new + P_new.T)

                    vx_new, vy_new = float(s_new[2]), float(s_new[3])
                    new_hist = (trk.vel_history + ((vx_new, vy_new),))[-3:]
                    # Reversal check across last 3 velocity estimates
                    has_reversal = False
                    if len(new_hist) >= 2:
                        for idx_h in range(len(new_hist) - 1):
                            dot_v = (
                                new_hist[idx_h][0] * new_hist[idx_h + 1][0]
                                + new_hist[idx_h][1] * new_hist[idx_h + 1][1]
                            )
                            if dot_v < 0.0:
                                has_reversal = True
                                break
                    next_scale = self.q_acc_adapt_mult if (has_reversal or nis > 6.0) else 1.0

                    trk.x = float(s_new[0])
                    trk.y = float(s_new[1])
                    trk.vx = vx_new
                    trk.vy = vy_new
                    trk.radius = 0.7 * trk.radius + 0.3 * float(z_r)
                    trk.P_4x4 = P_new
                    trk.last_update_stamp = float(stamp_sec)
                    trk.age += 1
                    trk.hits += 1
                    trk.consecutive_hits += 1
                    trk.missed_ticks = 0
                    trk.q_acc_scale = next_scale
                    trk.vel_history = new_hist
                    if (
                        not trk.is_confirmed
                        and trk.consecutive_hits >= self.confirm_min_hits
                        and trk.speed >= self.confirm_min_speed_mps
                    ):
                        trk.is_confirmed = True

        # 3. Coast unmatched tracks up to max_coast_ticks (2 ticks)
        to_remove: List[int] = []
        for tid in track_ids:
            if tid in matched_trk:
                continue
            trk = self.tracks[tid]
            trk.age += 1
            trk.missed_ticks += 1
            trk.consecutive_hits = 0
            limit = self.max_coast_ticks if trk.is_confirmed else 1
            if trk.missed_ticks > limit:
                to_remove.append(tid)
                continue
            s_p = pred_states[tid]
            trk.x = float(s_p[0])
            trk.y = float(s_p[1])
            trk.vx = float(s_p[2])
            trk.vy = float(s_p[3])
            trk.P_4x4 = pred_covs[tid]
            trk.last_update_stamp = float(stamp_sec)

        for tid in to_remove:
            del self.tracks[tid]

        # 4. Initialize tentative tracks for unmatched detections
        init_v2 = self.init_vel_std_mps * self.init_vel_std_mps
        sig_p2 = self.meas_noise_std_m * self.meas_noise_std_m
        for c_j, (zx, zy, z_r) in enumerate(detections):
            if c_j in matched_det:
                continue
            tid = self._next_track_id
            self._next_track_id += 1
            self.tracks[tid] = TrackedObstacle(
                track_id=tid,
                x=float(zx),
                y=float(zy),
                vx=0.0,
                vy=0.0,
                radius=float(z_r),
                P_4x4=np.diag([sig_p2, sig_p2, init_v2, init_v2]).astype(np.float64),
                last_update_stamp=float(stamp_sec),
                age=1,
                hits=1,
                is_confirmed=False,
                consecutive_hits=1,
                missed_ticks=0,
                q_acc_scale=1.0,
                vel_history=(),
            )

    def update(
        self,
        rx: float,
        ry: float,
        rth: float,
        raw_ranges: Optional[Sequence[float]],
        scan_stamp_sec: float,
        now_sec: Optional[float] = None,
        omega: float = 0.0,
        angle_min: float = 0.0,
        angle_increment: Optional[float] = None,
        external_observed_free: Optional[np.ndarray] = None,
        ext_origin_ix: int = 0,
        ext_origin_iy: int = 0,
        ext_res: float = 0.04,
        freeze_velocity: bool = False,
    ) -> List[TrackedObstacle]:
        """
        Process a control tick.

        Performs full background subtraction, clustering, circle fitting, and KF update
        ONLY when `scan_stamp_sec` advances (5 Hz); on intermediate ticks (10 Hz / 20 Hz),
        skips observation update and returns non-mutating damped-KF extrapolated tracks.
        """
        t_eval = float(scan_stamp_sec if now_sec is None else now_sec)
        stamp_f = float(scan_stamp_sec)

        # Mid-tick 5 Hz gating check
        if self._last_scan_stamp is not None and abs(stamp_f - self._last_scan_stamp) <= 1e-9:
            self.last_did_update = False
            self.extrap_tick_count += 1
            return self.get_extrapolated_tracks(now_sec=t_eval, confirmed_only=False)

        if self._last_scan_stamp is not None and stamp_f < self._last_scan_stamp - 1e-3:
            # Backward clock jump (bag restart / reset): re-anchor smoothly without negative dt
            self._last_scan_stamp = None
            self.tracks.clear()

        dt = (stamp_f - self._last_scan_stamp) if self._last_scan_stamp is not None else 0.20
        dt = max(1e-3, min(1.5, float(dt)))
        self._last_scan_stamp = stamp_f
        self.last_did_update = True
        self.scan_update_count += 1

        self._recenter_grid(rx, ry)

        if raw_ranges is None or len(raw_ranges) < 3:
            self._kf_step_and_associate([], dt, stamp_f, freeze_velocity=freeze_velocity)
            return list(self.tracks.values())

        ranges = np.asarray(raw_ranges, dtype=np.float32)
        n_rays = len(ranges)
        cos_lut, sin_lut, inc = self._get_trig_lut(n_rays, angle_min, angle_increment)
        is_full_360 = abs(n_rays * abs(inc) - 2.0 * math.pi) <= 2.5 * abs(inc)

        valid_mask = (
            np.isfinite(ranges)
            & (ranges >= np.float32(self.lidar_min_range_m))
            & (ranges <= np.float32(self.lidar_max_range_m))
        )
        valid_idx = np.flatnonzero(valid_mask)
        if len(valid_idx) == 0:
            self._update_background_grids(
                rx,
                ry,
                rth,
                ranges,
                cos_lut,
                sin_lut,
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
            )
            self._kf_step_and_associate([], dt, stamp_f, freeze_velocity=freeze_velocity)
            return list(self.tracks.values())

        r_val = ranges[valid_idx]
        lx = r_val * cos_lut[valid_idx]
        ly = r_val * sin_lut[valid_idx]
        c = np.float32(math.cos(rth))
        s = np.float32(math.sin(rth))
        wx = np.float32(rx) + c * lx - s * ly
        wy = np.float32(ry) + s * lx + c * ly

        inv_res = np.float32(1.0 / self.res)
        ix = np.floor(wx * inv_res).astype(np.int32) - self.origin_ix
        iy = np.floor(wy * inv_res).astype(np.int32) - self.origin_iy

        # Snapshot pre-update observed_free BEFORE ray-carving updates current hits
        pre_obs_free = self._sample_observed_free(
            wx, wy, ix, iy, external_observed_free, ext_origin_ix, ext_origin_iy, ext_res
        )

        # Condition (1): Exclude points falling inside dilated static_occ >= 0.75 mask
        in_static = self._check_static_mask_at_cells(ix, iy, omega=omega)
        non_static_sel = np.flatnonzero(~in_static)

        detections: List[Tuple[float, float, float]] = []
        if len(non_static_sel) >= self.min_cluster_points:
            cand_ray_idx = valid_idx[non_static_sel]
            cand_pts_w = np.column_stack((wx[non_static_sel], wy[non_static_sel])).astype(np.float64)
            cand_obs_free = pre_obs_free[non_static_sel]
            cand_near_trk = self._point_near_active_track(
                wx[non_static_sel], wy[non_static_sel], gate_m=self.gating_dist_m
            )

            raw_clusters = self._extract_clusters(
                cand_ray_idx, cand_pts_w, n_rays_total=n_rays, is_full_360=is_full_360
            )
            for cl_pts, cl_member_ids in raw_clusters:
                # NOTE: Never impose n_c <= 35; only lower bound n_c >= 3
                if cl_pts.shape[0] < self.min_cluster_points:
                    continue

                # Condition (2): New emergence in verified free-space OR near confirmed dynamic track
                cl_free = cand_obs_free[cl_member_ids]
                frac_free = float(np.mean(cl_free >= self.observed_free_thresh))
                matches_confirmed = bool(np.any(cand_near_trk[cl_member_ids]))
                if frac_free < 0.50 and not matches_confirmed:
                    continue

                chord_len = math.hypot(
                    float(cl_pts[-1, 0] - cl_pts[0, 0]),
                    float(cl_pts[-1, 1] - cl_pts[0, 1]),
                )
                if not (self.min_chord_m <= chord_len <= self.max_chord_m):
                    continue

                est = correct_cluster_center_and_radius(
                    cl_pts,
                    robot_xy=(rx, ry),
                    min_radius_m=self.min_radius_m,
                    max_radius_m=self.max_radius_m,
                    kappa_arc=self.kappa_arc,
                )
                if est is not None:
                    detections.append(est)

        # KF association & update runs BEFORE updating static_occ so newly confirmed
        # tracks immediately mask out their dynamic footprint from static_occ.
        self._kf_step_and_associate(detections, dt, stamp_f, freeze_velocity=freeze_velocity)

        self._update_background_grids(
            rx, ry, rth, ranges, cos_lut, sin_lut, wx, wy, ix, iy, pre_obs_free
        )

        return self.get_extrapolated_tracks(now_sec=t_eval, confirmed_only=False)

    # Convenience alias matching planner conventions
    step = update

    def get_extrapolated_tracks(
        self, now_sec: Optional[float] = None, confirmed_only: bool = True
    ) -> List[TrackedObstacle]:
        """
        Return non-mutating damped-KF extrapolated track states at `now_sec`.
        When `now_sec == _last_scan_stamp` (dt_extrap == 0), returns exact filter states.
        """
        if not self.tracks:
            return []

        if now_sec is None or self._last_scan_stamp is None:
            dt_extrap = 0.0
            stamp_out = 0.0 if self._last_scan_stamp is None else self._last_scan_stamp
        else:
            dt_extrap = max(0.0, float(now_sec) - float(self._last_scan_stamp))
            stamp_out = float(now_sec)

        out: List[TrackedObstacle] = []
        for trk in self.tracks.values():
            if confirmed_only and not trk.is_confirmed:
                continue
            if dt_extrap <= 1e-6:
                out.append(
                    TrackedObstacle(
                        track_id=trk.track_id,
                        x=trk.x,
                        y=trk.y,
                        vx=trk.vx,
                        vy=trk.vy,
                        radius=trk.radius,
                        P_4x4=trk.P_4x4.copy(),
                        last_update_stamp=trk.last_update_stamp,
                        age=trk.age,
                        hits=trk.hits,
                        is_confirmed=trk.is_confirmed,
                        consecutive_hits=trk.consecutive_hits,
                        missed_ticks=trk.missed_ticks,
                        q_acc_scale=trk.q_acc_scale,
                        vel_history=trk.vel_history,
                    )
                )
            else:
                s0 = np.array([trk.x, trk.y, trk.vx, trk.vy], dtype=np.float64)
                s_e, P_e = propagate_state_cov(
                    s0,
                    trk.P_4x4,
                    dt_extrap,
                    beta=self.beta,
                    q_acc=self.q_acc_base * trk.q_acc_scale,
                )
                out.append(
                    TrackedObstacle(
                        track_id=trk.track_id,
                        x=float(s_e[0]),
                        y=float(s_e[1]),
                        vx=float(s_e[2]),
                        vy=float(s_e[3]),
                        radius=trk.radius,
                        P_4x4=P_e,
                        last_update_stamp=stamp_out,
                        age=trk.age,
                        hits=trk.hits,
                        is_confirmed=trk.is_confirmed,
                        consecutive_hits=trk.consecutive_hits,
                        missed_ticks=trk.missed_ticks,
                        q_acc_scale=trk.q_acc_scale,
                        vel_history=trk.vel_history,
                    )
                )
        return out

    def apply_se2_increment(self, d_tx: float, d_ty: float, d_th: float) -> None:
        """
        Apply an SE(2) world-frame transform increment delta_T = (d_tx, d_ty, d_th)
        to all tracked obstacles, velocity histories, covariances, and background grids.
        """
        c = math.cos(d_th)
        s = math.sin(d_th)
        R2 = np.array([[c, -s], [s, c]], dtype=np.float64)
        M4 = np.zeros((4, 4), dtype=np.float64)
        M4[:2, :2] = R2
        M4[2:, 2:] = R2

        for trk in self.tracks.values():
            nx = d_tx + c * trk.x - s * trk.y
            ny = d_ty + s * trk.x + c * trk.y
            nvx = c * trk.vx - s * trk.vy
            nvy = s * trk.vx + c * trk.vy
            trk.x = float(nx)
            trk.y = float(ny)
            trk.vx = float(nvx)
            trk.vy = float(nvy)
            P_rot = M4 @ trk.P_4x4 @ M4.T
            trk.P_4x4 = 0.5 * (P_rot + P_rot.T)
            if trk.vel_history:
                trk.vel_history = tuple(
                    (c * hvx - s * hvy, s * hvx + c * hvy) for hvx, hvy in trk.vel_history
                )

        if self.grid_initialized and (abs(d_tx) > 1e-4 or abs(d_ty) > 1e-4 or abs(d_th) > 1e-4):
            # Nearest-neighbor inverse warp of static_occ and observed_free to transformed world coords
            grid_i = np.arange(self.n, dtype=np.float32) + np.float32(self.origin_ix + 0.5)
            grid_j = np.arange(self.n, dtype=np.float32) + np.float32(self.origin_iy + 0.5)
            wx_new = grid_i[:, None] * np.float32(self.res)
            wy_new = grid_j[None, :] * np.float32(self.res)
            dx = wx_new - np.float32(d_tx)
            dy = wy_new - np.float32(d_ty)
            wx_old = np.float32(c) * dx + np.float32(s) * dy
            wy_old = -np.float32(s) * dx + np.float32(c) * dy
            old_ix = np.floor(wx_old / np.float32(self.res)).astype(np.int32) - self.origin_ix
            old_iy = np.floor(wy_old / np.float32(self.res)).astype(np.int32) - self.origin_iy
            inb = (old_ix >= 0) & (old_ix < self.n) & (old_iy >= 0) & (old_iy < self.n)
            ix_c = np.clip(old_ix, 0, self.n - 1)
            iy_c = np.clip(old_iy, 0, self.n - 1)
            self.static_occ = np.where(inb, self.static_occ[ix_c, iy_c], np.float32(0.0))
            self.observed_free = np.where(inb, self.observed_free[ix_c, iy_c], np.float32(0.0))

    def sync_icp_correction(
        self,
        old_corr: Tuple[float, float, float],
        new_corr: Tuple[float, float, float],
    ) -> None:
        """
        Synchronize tracks and background grids across a persistent waypoint-ICP
        correction jump from `old_corr` to `new_corr`:
            delta_T = T_new * T_old^-1
        """
        ox, oy, oth = float(old_corr[0]), float(old_corr[1]), float(old_corr[2])
        nx, ny, nth = float(new_corr[0]), float(new_corr[1]), float(new_corr[2])
        d_th = math.atan2(math.sin(nth - oth), math.cos(nth - oth))
        c_d = math.cos(d_th)
        s_d = math.sin(d_th)
        d_tx = nx - (c_d * ox - s_d * oy)
        d_ty = ny - (s_d * ox + c_d * oy)
        self.apply_se2_increment(d_tx, d_ty, d_th)

    def predict_future(
        self, track: TrackedObstacle, tau: float
    ) -> Tuple[float, float, float, np.ndarray]:
        """
        Predict future position (px, py), uncertainty-inflated radius r_eff(tau),
        and 2x2 position covariance Sigma_p(tau) at horizon tau >= 0.
        """
        s0 = np.array([track.x, track.y, track.vx, track.vy], dtype=np.float64)
        cached = self._ch3_fq_cache.get(float(tau))
        if cached is not None:
            F, Q_base = cached
            s_t = F @ s0
            P_t = F @ track.P_4x4 @ F.T + track.q_acc_scale * Q_base
            P_t = 0.5 * (P_t + P_t.T)
        else:
            s_t, P_t = propagate_state_cov(
                s0,
                track.P_4x4,
                tau,
                beta=self.beta,
                q_acc=self.q_acc_base * track.q_acc_scale,
            )
        cov_p = P_t[:2, :2]
        lam_max = max_eigenvalue_2x2(cov_p)
        r_eff = float(track.radius) + 1.5 * math.sqrt(lam_max)
        return float(s_t[0]), float(s_t[1]), r_eff, cov_p

    def project_ch3(
        self,
        rx: float,
        ry: float,
        rth: float,
        now_sec: Optional[float] = None,
        horizons: Optional[Sequence[float]] = None,
        weights: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """
        Project confirmed dynamic obstacle future occupancies onto the 12x12 ego grid (Ch3).

        Formula (Stage 1 §1.2.5):
            ch3[i, j] = max_{k, tau} w_tau * clip(1 - max(d - r_c, 0) / max(r_eff(tau) - r_c, eps), 0, 1)
        """
        ch3 = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        if not self.tracks:
            return ch3
        confirmed = self.get_extrapolated_tracks(now_sec=now_sec, confirmed_only=True)
        if not confirmed:
            return ch3

        hs = self.ch3_horizons if horizons is None else horizons
        ws = self.ch3_weights if weights is None else weights
        c = math.cos(rth)
        s = math.sin(rth)
        eps = float(self.ch3_eps)

        for trk in confirmed:
            r_c = float(trk.radius)
            for tau, w_tau in zip(hs, ws):
                px_w, py_w, r_eff, _ = self.predict_future(trk, float(tau))
                dx = px_w - float(rx)
                dy = py_w - float(ry)
                ego_x = np.float32(c * dx + s * dy)
                ego_y = np.float32(-s * dx + c * dy)
                dist_grid = np.hypot(_EGO_GRID_X - ego_x, _EGO_GRID_Y - ego_y)
                denom = np.float32(max(r_eff - r_c, eps))
                val = np.float32(w_tau) * np.clip(
                    np.float32(1.0) - np.maximum(dist_grid - np.float32(r_c), np.float32(0.0)) / denom,
                    np.float32(0.0),
                    np.float32(1.0),
                )
                ch3 = np.maximum(ch3, val)

        return np.clip(ch3, np.float32(0.0), np.float32(1.0)).astype(np.float32)

    def get_predicted_risk_circles(
        self,
        now_sec: Optional[float] = None,
        horizons: Optional[Sequence[float]] = None,
        weights: Optional[Sequence[float]] = None,
    ) -> List[Tuple[float, float, float, float, float]]:
        """
        Return list of (px, py, r_c, r_eff, weight) across confirmed tracks and horizons.
        """
        confirmed = self.get_extrapolated_tracks(now_sec=now_sec, confirmed_only=True)
        if not confirmed:
            return []
        hs = self.ch3_horizons if horizons is None else horizons
        ws = self.ch3_weights if weights is None else weights
        out: List[Tuple[float, float, float, float, float]] = []
        for trk in confirmed:
            for tau, w_tau in zip(hs, ws):
                px, py, r_eff, _ = self.predict_future(trk, float(tau))
                out.append((px, py, float(trk.radius), r_eff, float(w_tau)))
        return out
