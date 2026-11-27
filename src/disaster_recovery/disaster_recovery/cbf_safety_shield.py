#!/usr/bin/env python3
"""
Stage 2 (Phase 4-D): Front/Rear 2-Circle Covering Soft-CBF-QP Safety Shield
with Pure-NumPy 2x2 PDAS (Semismooth Newton) + Armijo 1D Line Search.

Design constraints & mathematical specification:
  - Pure Python + NumPy ONLY (no rclpy, cvxpy, osqp, scipy, or torch).
  - Exact model.sdf TurtleBot3 Burger 2-Circle Covering:
      l_f = +0.020 m, l_r = 0.060 m (s_r = -0.060 m), r_circle = 0.098 m
      Static LiDAR point: r_o = 0.015 m, m_pad = 0.020 m => R_safe,static = 0.133 m
      (2 * R_safe,static = 0.266 m < 0.280 m narrow doorway).
  - Static TF base_scan -> base_footprint (model.sdf L174/L406: x_b = x_s - 0.032, y_b = y_s).
  - Spatial 0.10 m voxel downsampling keeping the nearest 16 distinct voxels to the 2-Circle
    within 1.20 m (replaces coarse 8-sector compression).
  - Rate-invariant continuous decay rates satisfying strict conservatism order:
      gamma_dyn = 3.0 < gamma_static = 6.0 <= gamma_gas = 8.0 [s^-1],
      alpha_i(dt) = 1 - exp(-gamma_i * dt).
  - Explicit inter-step nonholonomic curvature & obstacle acceleration margin:
      eps_inter,m = ||p_k - p_{o,i}|| * (|v_max * w_max| + |s_k| * w_max^2 + |a_{o,max,i}|) * dt,
      eps_dt_slip,m = 2 * ||p_k - p_{o,i}|| * dv_slip + eps_inter,m.
  - Asymmetric Q = diag(w_v, w_w) weighting (default w_v=1.0, w_w=0.05 => 20:1 >= 10:1)
    to compensate the 50:1 front-circle (l_f = 0.020 m) steering-authority imbalance.
  - Onboard MQ-2 Ch2 inside (h < 0) LIFO safe-waypoint escape exception branch:
    temporarily deactivates singular circular repulsion for violated Ch2 zone(s) and
    overrides nominal control monotonically toward the newest uncontaminated LIFO waypoint
    while keeping all LiDAR, Ch1, and dynamic CBF constraints active.
  - Analytic slack elimination delta_m*(u) = max(0, a_m^T u - b_m) solved via reduced-space
    2x2 Primal-Dual Active Set (Semismooth Newton) + Armijo 1D line search on box U.
  - Parameterized v_min (e.g. -0.10 vs 0.0 m/s) with built-in velocity sign-reversal
    (chattering) and intervention telemetry.
"""

from dataclasses import dataclass
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Measured TurtleBot3 Burger geometry from src/disaster_recovery/models/turtlebot3_burger/model.sdf
DEFAULT_L_F = 0.020
DEFAULT_L_R = 0.060
DEFAULT_R_CIRCLE = 0.098
DEFAULT_STATIC_PT_RADIUS = 0.015
DEFAULT_STATIC_MARGIN = 0.020
DEFAULT_R_SAFE_STATIC = DEFAULT_R_CIRCLE + DEFAULT_STATIC_PT_RADIUS + DEFAULT_STATIC_MARGIN  # 0.133 m

DEFAULT_LIDAR_OFFSET_X = -0.032  # base_scan -> base_footprint x shift (model.sdf L174, L406)
DEFAULT_LIDAR_OFFSET_Y = 0.0

DEFAULT_GAMMA_DYNAMIC = 3.0
DEFAULT_GAMMA_STATIC = 6.0
DEFAULT_GAMMA_GAS = 8.0

DEFAULT_P_DELTA_DYNAMIC = 5000.0
DEFAULT_P_DELTA_STATIC = 2000.0
DEFAULT_P_DELTA_GAS = 400.0

# Asymmetric Q = diag(w_v, w_w) with 20:1 ratio (>= 10:1) to counter l_f = 0.020 m 50:1 leverage gap
DEFAULT_W_V = 1.0
DEFAULT_W_W = 0.05

DEFAULT_SLIP_MARGIN_MPS = 0.005
DEFAULT_A_O_MAX_DYNAMIC = 0.40

DEFAULT_LIDAR_VOXEL_M = 0.10
DEFAULT_LIDAR_MAX_POINTS = 16
DEFAULT_LIDAR_MIN_RANGE_M = 0.12
DEFAULT_LIDAR_MAX_RANGE_M = 1.20


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def compute_alpha(gamma: float, dt: float) -> float:
    """Rate-invariant discrete CBF decay factor alpha(dt) = 1 - exp(-gamma * dt)."""
    g = max(0.0, float(gamma))
    dt_c = max(1e-6, float(dt))
    return 1.0 - math.exp(-g * dt_c)


def max_eigenvalue_2x2(cov_2x2: np.ndarray) -> float:
    """Closed-form largest eigenvalue of a symmetric 2x2 covariance matrix."""
    a = float(cov_2x2[0, 0])
    b = 0.5 * (float(cov_2x2[0, 1]) + float(cov_2x2[1, 0]))
    c = float(cov_2x2[1, 1])
    disc = math.sqrt(max(0.0, (a - c) * (a - c) + 4.0 * b * b))
    return max(0.0, 0.5 * (a + c + disc))


def transform_scan_to_base_footprint(
    scan_ranges: Optional[Sequence[float]] = None,
    angle_min: float = 0.0,
    angle_increment: Optional[float] = None,
    scan_pts_sensor: Optional[np.ndarray] = None,
    lidar_offset_x: float = DEFAULT_LIDAR_OFFSET_X,
    lidar_offset_y: float = DEFAULT_LIDAR_OFFSET_Y,
    min_range: float = DEFAULT_LIDAR_MIN_RANGE_M,
    max_range: float = DEFAULT_LIDAR_MAX_RANGE_M,
    cos_lut: Optional[np.ndarray] = None,
    sin_lut: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Transform 2D LiDAR observations from `base_scan` frame into `base_footprint` axle frame
    via static TF (x_b = x_s + lidar_offset_x, y_b = y_s + lidar_offset_y; default x_s - 0.032).

    Returns float64 array of shape (K, 2) containing valid points within [min_range, max_range].
    """
    if scan_pts_sensor is not None:
        pts = np.asarray(scan_pts_sensor, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
            return np.empty((0, 2), dtype=np.float64)
        xs = pts[:, 0]
        ys = pts[:, 1]
        finite = np.isfinite(xs) & np.isfinite(ys)
        r2 = xs * xs + ys * ys
        r_min_f = float(min_range)
        r_max_f = float(max_range)
        valid = finite & (r2 > 1e-12) & (r2 <= r_max_f * r_max_f)
        if not np.any(valid):
            return np.empty((0, 2), dtype=np.float64)
        xv = xs[valid]
        yv = ys[valid]
        r2v = r2[valid]
        ultra = r2v < (r_min_f * r_min_f)
        if np.any(ultra):
            scale = r_min_f / np.sqrt(r2v[ultra])
            xv = xv.copy()
            yv = yv.copy()
            xv[ultra] *= scale
            yv[ultra] *= scale
        out = np.empty((xv.shape[0], 2), dtype=np.float64)
        out[:, 0] = xv + float(lidar_offset_x)
        out[:, 1] = yv + float(lidar_offset_y)
        return out

    if scan_ranges is None:
        return np.empty((0, 2), dtype=np.float64)
    r = np.asarray(scan_ranges, dtype=np.float64)
    n = r.shape[0]
    if n == 0:
        return np.empty((0, 2), dtype=np.float64)

    r_min_f = float(min_range)
    r_max_f = float(max_range)
    valid = np.isfinite(r) & (r > 0.0) & (r <= r_max_f)
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float64)

    if cos_lut is not None and sin_lut is not None and cos_lut.shape[0] == n:
        c_arr = cos_lut[valid]
        s_arr = sin_lut[valid]
    else:
        inc = float(angle_increment) if (angle_increment is not None and angle_increment != 0.0) else (2.0 * math.pi / float(n))
        idx = np.flatnonzero(valid)
        angs = float(angle_min) + idx.astype(np.float64) * inc
        c_arr = np.cos(angs)
        s_arr = np.sin(angs)

    rv = np.maximum(r[valid], r_min_f)
    out = np.empty((rv.shape[0], 2), dtype=np.float64)
    out[:, 0] = rv * c_arr + float(lidar_offset_x)
    out[:, 1] = rv * s_arr + float(lidar_offset_y)
    return out


def voxel_downsample_nearest_2circle(
    pts_body: np.ndarray,
    l_f: float = DEFAULT_L_F,
    l_r: float = DEFAULT_L_R,
    voxel_size: float = DEFAULT_LIDAR_VOXEL_M,
    max_points: int = DEFAULT_LIDAR_MAX_POINTS,
    max_dist_m: float = DEFAULT_LIDAR_MAX_RANGE_M,
) -> np.ndarray:
    """
    Apply spatial voxel downsampling (default 0.10 m) in `base_footprint` coordinates
    and retain up to `max_points` (default 16) nearest distinct voxel representatives
    measured against the Front/Rear 2-Circle pair (p_f = (+l_f, 0), p_r = (-l_r, 0)).

    Within each voxel, the strictly closest point to the 2-Circle is retained so narrow
    doorway jambs and multiple pillars inside the same 45-deg sector are never dropped.
    """
    if pts_body is None or pts_body.shape[0] == 0 or max_points <= 0:
        return np.empty((0, 2), dtype=np.float64)

    xb = pts_body[:, 0]
    yb = pts_body[:, 1]
    yy = yb * yb
    dx_f = xb - float(l_f)
    dx_r = xb + float(l_r)
    d2_min = np.minimum(dx_f * dx_f + yy, dx_r * dx_r + yy)

    max_d2 = float(max_dist_m) * float(max_dist_m)
    in_rad = d2_min <= max_d2
    n_in = int(np.count_nonzero(in_rad))
    if n_in == 0:
        return np.empty((0, 2), dtype=np.float64)

    if n_in < pts_body.shape[0]:
        xb_u = xb[in_rad]
        yb_u = yb[in_rad]
        d2_use = d2_min[in_rad]
    else:
        xb_u = xb
        yb_u = yb
        d2_use = d2_min

    inv_vox = 1.0 / max(1e-4, float(voxel_size))
    order = np.argsort(d2_use)
    vx = np.floor(xb_u * inv_vox).astype(np.int32)
    vy = np.floor(yb_u * inv_vox).astype(np.int32)
    keys = (vx + 16384) * 32768 + (vy + 16384)

    selected_idx: List[int] = []
    seen = set()
    n_tot = order.shape[0]
    chunk = min(n_tot, max(48, max_points * 4))
    start = 0
    while start < n_tot and len(selected_idx) < max_points:
        end = min(n_tot, start + chunk)
        ord_slice = order[start:end]
        for raw_i, k in zip(ord_slice.tolist(), keys[ord_slice].tolist()):
            if k not in seen:
                seen.add(k)
                selected_idx.append(raw_i)
                if len(selected_idx) >= max_points:
                    break
        start = end
        chunk = n_tot - start

    if not selected_idx:
        return np.empty((0, 2), dtype=np.float64)
    out = np.empty((len(selected_idx), 2), dtype=np.float64)
    out[:, 0] = xb_u[selected_idx]
    out[:, 1] = yb_u[selected_idx]
    return out


def compute_kkt_residual(
    u: np.ndarray,
    u_nom: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    p_delta: np.ndarray,
    w_v: float,
    w_w: float,
    u_lo: np.ndarray,
    u_hi: np.ndarray,
) -> float:
    """
    Compute exact projected-tangent-cone KKT stationarity residual ||proj_{T_U(u)}(-grad Phi(u))||_2
    for the slack-eliminated piecewise-quadratic Soft-CBF-QP on box U = [u_lo, u_hi].
    """
    u0 = float(u[0])
    u1 = float(u[1])
    g0 = float(w_v) * (u0 - float(u_nom[0]))
    g1 = float(w_w) * (u1 - float(u_nom[1]))

    if A.shape[0] > 0:
        a0 = A[:, 0]
        a1 = A[:, 1]
        viol = a0 * u0 + a1 * u1 - b
        act = viol > 0.0
        if np.any(act):
            pw = p_delta[act] * viol[act]
            g0 += float(np.dot(pw, a0[act]))
            g1 += float(np.dot(pw, a1[act]))

    lo0, lo1 = float(u_lo[0]), float(u_lo[1])
    hi0, hi1 = float(u_hi[0]), float(u_hi[1])

    if u0 <= lo0 + 1e-13:
        rg0 = max(0.0, -g0)
    elif u0 >= hi0 - 1e-13:
        rg0 = max(0.0, g0)
    else:
        rg0 = abs(g0)

    if u1 <= lo1 + 1e-13:
        rg1 = max(0.0, -g1)
    elif u1 >= hi1 - 1e-13:
        rg1 = max(0.0, g1)
    else:
        rg1 = abs(g1)

    return math.hypot(rg0, rg1)


def solve_soft_cbf_qp_pdas(
    u_nom: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    p_delta: np.ndarray,
    w_v: float,
    w_w: float,
    u_lo: np.ndarray,
    u_hi: np.ndarray,
    max_iters: int = 25,
    kkt_tol: float = 1e-9,
) -> Tuple[np.ndarray, int, float, float, float, int]:
    """
    Pure-NumPy 2x2 Primal-Dual Active Set (Semismooth Newton) + Armijo 1D Line Search
    solving the exact slack-eliminated Soft-CBF-QP:

        min_{u in [u_lo, u_hi]}  0.5 * (u - u_nom)^T Q (u - u_nom)
                               + 0.5 * sum_m p_m * max(0, a_m^T u - b_m)^2

    where delta_m*(u) = max(0, a_m^T u - b_m) and Q = diag(w_v, w_w) > 0.

    Returns:
        (u_opt, iters, kkt_residual, max_slack, sum_slack_sq, num_active)
    """
    wv = float(w_v)
    ww = float(w_w)
    un0 = float(u_nom[0])
    un1 = float(u_nom[1])
    lo0, lo1 = float(u_lo[0]), float(u_lo[1])
    hi0, hi1 = float(u_hi[0]), float(u_hi[1])

    # Project nominal command onto box U
    u0 = min(hi0, max(lo0, un0))
    u1 = min(hi1, max(lo1, un1))

    m = A.shape[0]
    if m == 0:
        return np.array([u0, u1], dtype=np.float64), 0, 0.0, 0.0, 0.0, 0

    # Zero-overhead scalar PDAS + Armijo 1D line search for typical small constraint sets (m <= 128)
    a0l = A[:, 0].tolist()
    a1l = A[:, 1].tolist()
    bl = b.tolist()
    pl = p_delta.tolist()

    act_idx = [i for i in range(m) if a0l[i] * u0 + a1l[i] * u1 > bl[i]]
    if not act_idx:
        # Fast path: clipped nominal command satisfies all CBF inequalities (0 iterations, slack == 0)
        return np.array([u0, u1], dtype=np.float64), 0, 0.0, 0.0, 0.0, 0

    def _eval_obj_only(x0: float, x1: float) -> float:
        dv0 = x0 - un0
        dv1 = x1 - un1
        val = 0.5 * (wv * dv0 * dv0 + ww * dv1 * dv1)
        for i in range(m):
            v_i = a0l[i] * x0 + a1l[i] * x1 - bl[i]
            if v_i > 0.0:
                val += 0.5 * pl[i] * v_i * v_i
        return val

    iters_done = 0
    kkt_res = math.inf

    for it in range(max_iters):
        dv0 = u0 - un0
        dv1 = u1 - un1
        phi = 0.5 * (wv * dv0 * dv0 + ww * dv1 * dv1)
        g0 = wv * dv0
        g1 = ww * dv1
        h00 = wv
        h01 = 0.0
        h11 = ww

        for i in act_idx:
            ai0 = a0l[i]
            ai1 = a1l[i]
            pi = pl[i]
            vi = ai0 * u0 + ai1 * u1 - bl[i]
            pv = pi * vi
            phi += 0.5 * pv * vi
            g0 += pv * ai0
            g1 += pv * ai1
            pa0 = pi * ai0
            h00 += pa0 * ai0
            h01 += pa0 * ai1
            h11 += pi * ai1 * ai1

        # Exact projected-tangent-cone KKT stationarity residual
        if u0 <= lo0 + 1e-13:
            rg0 = max(0.0, -g0)
        elif u0 >= hi0 - 1e-13:
            rg0 = max(0.0, g0)
        else:
            rg0 = abs(g0)

        if u1 <= lo1 + 1e-13:
            rg1 = max(0.0, -g1)
        elif u1 >= hi1 - 1e-13:
            rg1 = max(0.0, g1)
        else:
            rg1 = abs(g1)

        kkt_res = math.hypot(rg0, rg1)
        if kkt_res <= kkt_tol:
            break

        # Reduced-space box active set (binding box coordinates)
        bind0 = (u0 <= lo0 + 1e-12 and g0 > 0.0) or (u0 >= hi0 - 1e-12 and g0 < 0.0)
        bind1 = (u1 <= lo1 + 1e-12 and g1 > 0.0) or (u1 >= hi1 - 1e-12 and g1 < 0.0)

        if bind0 and bind1:
            break
        elif bind0:
            du0 = 0.0
            du1 = -g1 / h11
        elif bind1:
            du0 = -g0 / h00
            du1 = 0.0
        else:
            # Closed-form 2x2 Cramer's rule inverse of SPD generalized Hessian H_A
            det = h00 * h11 - h01 * h01
            if det > 1e-18 * (h00 * h11):
                inv_det = 1.0 / det
                du0 = -(h11 * g0 - h01 * g1) * inv_det
                du1 = -(-h01 * g0 + h00 * g1) * inv_det
            else:
                du0 = -g0 / h00
                du1 = -g1 / h11

        # Compute exact first box-boundary hit step alpha_box in (0, 1) if full step exits U
        alpha_box = 1.0
        if du0 > 1e-15 and u0 + du0 > hi0:
            alpha_box = min(alpha_box, (hi0 - u0) / du0)
        elif du0 < -1e-15 and u0 + du0 < lo0:
            alpha_box = min(alpha_box, (lo0 - u0) / du0)
        if du1 > 1e-15 and u1 + du1 > hi1:
            alpha_box = min(alpha_box, (hi1 - u1) / du1)
        elif du1 < -1e-15 and u1 + du1 < lo1:
            alpha_box = min(alpha_box, (lo1 - u1) / du1)

        # Projected Armijo 1D line search: try full Newton step (1.0), exact box face hit, then halve
        c1 = 1e-4
        accepted = False
        best_u0 = u0
        best_u1 = u1
        best_phi = phi

        # Test alpha = 1.0 first (common case: 1-step exact convergence!)
        c0 = min(hi0, max(lo0, u0 + du0))
        c1_u = min(hi1, max(lo1, u1 + du1))
        slope = g0 * (c0 - u0) + g1 * (c1_u - u1)
        if slope < 0.0:
            phi_c = _eval_obj_only(c0, c1_u)
            if phi_c < best_phi:
                best_phi = phi_c
                best_u0 = c0
                best_u1 = c1_u
            if phi_c <= phi + c1 * slope:
                u0, u1 = c0, c1_u
                accepted = True

        if not accepted and 1e-12 < alpha_box < 1.0 - 1e-12:
            c0 = min(hi0, max(lo0, u0 + alpha_box * du0))
            c1_u = min(hi1, max(lo1, u1 + alpha_box * du1))
            slope = g0 * (c0 - u0) + g1 * (c1_u - u1)
            if slope < 0.0:
                phi_c = _eval_obj_only(c0, c1_u)
                if phi_c < best_phi:
                    best_phi = phi_c
                    best_u0 = c0
                    best_u1 = c1_u
                if phi_c <= phi + c1 * slope:
                    u0, u1 = c0, c1_u
                    accepted = True

        if not accepted:
            alpha = 0.5 * (alpha_box if (1e-12 < alpha_box < 1.0) else 1.0)
            for _ in range(32):
                c0 = min(hi0, max(lo0, u0 + alpha * du0))
                c1_u = min(hi1, max(lo1, u1 + alpha * du1))
                slope = g0 * (c0 - u0) + g1 * (c1_u - u1)
                if slope < 0.0:
                    phi_c = _eval_obj_only(c0, c1_u)
                    if phi_c < best_phi:
                        best_phi = phi_c
                        best_u0 = c0
                        best_u1 = c1_u
                    if phi_c <= phi + c1 * slope or (phi_c < phi and abs(slope) < 1e-14):
                        u0, u1 = c0, c1_u
                        accepted = True
                        break
                alpha *= 0.5
                if alpha < 1e-14:
                    break

        if not accepted:
            if best_phi < phi:
                u0, u1 = best_u0, best_u1
            else:
                break

        iters_done = it + 1
        act_idx = [i for i in range(m) if a0l[i] * u0 + a1l[i] * u1 > bl[i]]

    # Final KKT & slack diagnostics
    g0 = wv * (u0 - un0)
    g1 = ww * (u1 - un1)
    max_slack = 0.0
    sum_slack_sq = 0.0
    num_active = len(act_idx)
    for i in act_idx:
        vi = a0l[i] * u0 + a1l[i] * u1 - bl[i]
        if vi > max_slack:
            max_slack = vi
        sum_slack_sq += vi * vi
        pv = pl[i] * vi
        g0 += pv * a0l[i]
        g1 += pv * a1l[i]

    if u0 <= lo0 + 1e-13:
        rg0 = max(0.0, -g0)
    elif u0 >= hi0 - 1e-13:
        rg0 = max(0.0, g0)
    else:
        rg0 = abs(g0)

    if u1 <= lo1 + 1e-13:
        rg1 = max(0.0, -g1)
    elif u1 >= hi1 - 1e-13:
        rg1 = max(0.0, g1)
    else:
        rg1 = abs(g1)

    kkt_res = math.hypot(rg0, rg1)
    return np.array([u0, u1], dtype=np.float64), iters_done, kkt_res, max_slack, sum_slack_sq, num_active


@dataclass
class CBFResult:
    """Output and diagnostic telemetry from one Soft-CBF-QP filter step."""

    v: float
    w: float
    v_nom: float
    w_nom: float
    intervened: bool
    intervention_norm: float
    num_constraints: int
    num_active_constraints: int
    max_slack: float
    sum_slack_sq: float
    min_h_static: float
    min_h_dynamic: float
    min_h_gas: float
    pdas_iters: int
    kkt_residual: float
    mq2_escape_active: bool
    mq2_escape_target: Optional[Tuple[float, float]]
    solve_time_ms: float


class DiscreteSoftCBFShield:
    """
    Stage 2 Front/Rear 2-Circle Covering Soft-CBF-QP Safety Shield.

    Unifies voxelized static LiDAR points (base_scan -> base_footprint TF), Ch1 low-lying
    obstacles, Ch2 gas/survivor zones (with MQ-2 inside LIFO escape branch), and Stage 1
    confirmed dynamic obstacle tracks into a single 2x2 PDAS Soft-CBF-QP filter.
    """

    def __init__(
        self,
        l_f: float = DEFAULT_L_F,
        l_r: float = DEFAULT_L_R,
        r_circle: float = DEFAULT_R_CIRCLE,
        static_point_radius: float = DEFAULT_STATIC_PT_RADIUS,
        static_margin: float = DEFAULT_STATIC_MARGIN,
        dynamic_margin_base: float = 0.04,
        dynamic_cov_sigma_mult: float = 1.2,
        semantic_margin: float = 0.05,
        lidar_offset_x: float = DEFAULT_LIDAR_OFFSET_X,
        lidar_offset_y: float = DEFAULT_LIDAR_OFFSET_Y,
        v_min: float = -0.10,
        v_max: float = 0.20,
        w_max: float = 0.65,
        gamma_dynamic: float = DEFAULT_GAMMA_DYNAMIC,
        gamma_static: float = DEFAULT_GAMMA_STATIC,
        gamma_gas: float = DEFAULT_GAMMA_GAS,
        p_delta_dynamic: float = DEFAULT_P_DELTA_DYNAMIC,
        p_delta_static: float = DEFAULT_P_DELTA_STATIC,
        p_delta_gas: float = DEFAULT_P_DELTA_GAS,
        front_circle_priority: float = 2.0,
        w_v: float = DEFAULT_W_V,
        w_w: float = DEFAULT_W_W,
        slip_margin_mps: float = DEFAULT_SLIP_MARGIN_MPS,
        a_o_max_dynamic: float = DEFAULT_A_O_MAX_DYNAMIC,
        dynamic_graze_buffer_m: float = 0.06,
        lidar_voxel_m: float = DEFAULT_LIDAR_VOXEL_M,
        lidar_max_points: int = DEFAULT_LIDAR_MAX_POINTS,
        lidar_min_range_m: float = DEFAULT_LIDAR_MIN_RANGE_M,
        lidar_max_range_m: float = DEFAULT_LIDAR_MAX_RANGE_M,
        enable_mq2_lifo_escape: bool = True,
        dt_default: float = 0.10,
        max_pdas_iters: int = 25,
        kkt_tol: float = 1e-9,
        chatter_deadband_mps: float = 0.01,
        rate_penalty_v: float = 0.25,
        rate_penalty_w: float = 0.01,
        tangential_steer_gain: float = 0.55,
        reverse_hysteresis_ticks: int = 1,
        rear_reverse_block_m: float = 0.165,
    ) -> None:
        self.l_f = float(l_f)
        self.l_r = float(l_r)
        self.r_circle = float(r_circle)
        self.static_point_radius = float(static_point_radius)
        self.static_margin = float(static_margin)
        self.r_safe_static = self.r_circle + self.static_point_radius + self.static_margin
        self.dynamic_margin_base = float(dynamic_margin_base)
        self.dynamic_cov_sigma_mult = float(dynamic_cov_sigma_mult)
        self.semantic_margin = float(semantic_margin)

        self.lidar_offset_x = float(lidar_offset_x)
        self.lidar_offset_y = float(lidar_offset_y)

        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self._u_lo = np.array([self.v_min, -self.w_max], dtype=np.float64)
        self._u_hi = np.array([self.v_max, self.w_max], dtype=np.float64)
        self._u_lo_buf = self._u_lo.copy()
        self._u_hi_buf = self._u_hi.copy()
        self._u_lo_zero = np.array([0.0, -self.w_max], dtype=np.float64)
        self._u_nom_buf = np.zeros(2, dtype=np.float64)

        self.gamma_dynamic = float(gamma_dynamic)
        self.gamma_static = float(gamma_static)
        self.gamma_gas = float(gamma_gas)

        self.p_delta_dynamic = float(p_delta_dynamic)
        self.p_delta_static = float(p_delta_static)
        self.p_delta_gas = float(p_delta_gas)
        # Compensates front (l_f=0.020) vs rear (l_r=0.060) 3:1 steering-leverage asymmetry
        # so equal side-wall grazing never lets rear swing steer the nose into a narrow doorway wall
        self.front_circle_priority = max(1.0, float(front_circle_priority))

        self.w_v = float(w_v)
        self.w_w = float(w_w)
        self.slip_margin_mps = float(slip_margin_mps)
        self.a_o_max_dynamic = float(a_o_max_dynamic)
        self.dynamic_graze_buffer_m = float(dynamic_graze_buffer_m)

        self.lidar_voxel_m = float(lidar_voxel_m)
        self.lidar_max_points = int(lidar_max_points)
        self.lidar_min_range_m = float(lidar_min_range_m)
        self.lidar_max_range_m = float(lidar_max_range_m)
        self.enable_mq2_lifo_escape = bool(enable_mq2_lifo_escape)
        self.dt_default = float(dt_default)
        self.max_pdas_iters = int(max_pdas_iters)
        self.kkt_tol = float(kkt_tol)
        self.chatter_deadband_mps = float(chatter_deadband_mps)
        self.rate_penalty_v = max(0.0, float(rate_penalty_v))
        self.rate_penalty_w = max(0.0, float(rate_penalty_w))
        self.tangential_steer_gain = max(0.0, float(tangential_steer_gain))
        self.reverse_hysteresis_ticks = max(0, int(reverse_hysteresis_ticks))
        self.rear_reverse_block_m = max(0.0, float(rear_reverse_block_m))

        # Cached trig LUT and LDS-01 (<0.12 m -> inf/0/ultra-close) near-ray latch buffer
        self._lut_key: Optional[Tuple[int, float, float]] = None
        self._cos_lut: Optional[np.ndarray] = None
        self._sin_lut: Optional[np.ndarray] = None
        self._near_ray_ttl: Optional[np.ndarray] = None
        self._near_ray_arm_x: Optional[np.ndarray] = None
        self._near_ray_arm_y: Optional[np.ndarray] = None
        self._near_ray_arm_th: Optional[np.ndarray] = None

        # Episode telemetry counters (chattering, intervention, timing)
        self.reset()

    def reset(self) -> None:
        """Clear episode-level chattering and intervention statistics."""
        self.total_steps = 0
        self.intervention_steps = 0
        self.slack_steps = 0
        self.mq2_escape_steps = 0
        self.v_sign_reversals = 0
        self.v_stall_toggles = 0
        self.reverse_steps = 0
        self._prev_v: Optional[float] = None
        self._prev_w: Optional[float] = None
        self._prev_nonzero_v_sign: int = 0
        self._prev_moving_fwd: Optional[bool] = None
        self._rev_hyst_ticks: int = 0
        self._last_tan_sign: float = 0.0
        self._last_rear_min_m: float = math.inf
        self._last_dyn_threat_ycpa: Optional[float] = None
        self._last_has_dyn: bool = False
        self._last_dyn_block_reverse: bool = False
        self._near_ray_ttl = None
        self._near_ray_arm_x = None
        self._near_ray_arm_y = None
        self._near_ray_arm_th = None
        self.sum_abs_dv = 0.0
        self.sum_abs_dw = 0.0
        self.sum_intervention_norm = 0.0
        self.max_slack_seen = 0.0
        self.total_solve_ms = 0.0
        self.max_solve_ms = 0.0

    def get_stats(self) -> Dict[str, float]:
        """Return summary metrics including velocity sign-reversal (chattering) rates."""
        n = self.total_steps
        denom = max(1, n)
        denom_trans = max(1, n - 1)
        return {
            "v_min": self.v_min,
            "total_steps": float(n),
            "intervention_steps": float(self.intervention_steps),
            "intervention_rate_pct": 100.0 * self.intervention_steps / denom,
            "mean_intervention_norm": self.sum_intervention_norm / denom,
            "slack_steps": float(self.slack_steps),
            "max_slack_seen": self.max_slack_seen,
            "mq2_escape_steps": float(self.mq2_escape_steps),
            "v_sign_reversals": float(self.v_sign_reversals),
            "v_sign_reversal_rate_pct": 100.0 * self.v_sign_reversals / denom_trans,
            "v_stall_toggles": float(self.v_stall_toggles),
            "v_stall_toggle_rate_pct": 100.0 * self.v_stall_toggles / denom_trans,
            "reverse_steps": float(self.reverse_steps),
            "reverse_rate_pct": 100.0 * self.reverse_steps / denom,
            "mean_abs_dv": self.sum_abs_dv / denom_trans,
            "mean_abs_dw": self.sum_abs_dw / denom_trans,
            "mean_solve_ms": self.total_solve_ms / denom,
            "max_solve_ms": self.max_solve_ms,
        }

    def _get_trig_lut(
        self, n_rays: int, angle_min: float, angle_increment: Optional[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        inc = float(angle_increment) if (angle_increment is not None and angle_increment != 0.0) else (2.0 * math.pi / float(n_rays))
        key = (int(n_rays), round(float(angle_min), 6), round(inc, 8))
        if self._lut_key != key or self._cos_lut is None or self._sin_lut is None:
            angs = float(angle_min) + np.arange(n_rays, dtype=np.float64) * inc
            self._cos_lut = np.cos(angs)
            self._sin_lut = np.sin(angs)
            self._lut_key = key
        return self._cos_lut, self._sin_lut

    def _select_safe_lifo_waypoint(
        self,
        rx: float,
        ry: float,
        violated_ch2_world: Sequence[Tuple[float, float, float]],
        lifo_waypoints: Optional[Sequence[Any]],
        safe_escape_wp: Optional[Tuple[float, float]],
    ) -> Optional[Tuple[float, float]]:
        """
        Select the most recent LIFO waypoint outside all violated Ch2 gas zones whose
        robot->waypoint segment neither heads into nor cuts a chord through any violated
        gas zone. Falls back to a radial outward escape target (never a chord-cutting WP)
        if no safe non-chord LIFO waypoint exists.
        """
        if safe_escape_wp is not None:
            return float(safe_escape_wp[0]), float(safe_escape_wp[1])

        def _is_segment_safe(wx: float, wy: float) -> bool:
            vx = wx - rx
            vy = wy - ry
            seg_len = math.hypot(vx, vy)
            if seg_len <= 0.05:
                return False
            ux = vx / seg_len
            uy = vy / seg_len
            for hx, hy, hr in violated_ch2_world:
                dx_h = hx - rx
                dy_h = hy - ry
                dist_r = math.hypot(dx_h, dy_h)
                if dist_r <= 0.12:
                    # Robot is essentially at gas center (local MQ-2 trigger): any direction moves outward
                    continue
                t_proj = dx_h * ux + dy_h * uy
                cos_to_center = t_proj / dist_r
                # Reject waypoints directed toward the gas center
                if cos_to_center > 0.20:
                    return False
                if t_proj > 0.0:
                    t_clamped = min(seg_len, t_proj)
                    d_seg_min = math.hypot((rx + t_clamped * ux) - hx, (ry + t_clamped * uy) - hy)
                    if d_seg_min < min(dist_r - 0.02, hr):
                        return False
            return True

        best_safe_wp: Optional[Tuple[float, float]] = None
        best_safe_clr = -math.inf
        best_any_wp: Optional[Tuple[float, float]] = None
        best_any_clr = -math.inf

        if lifo_waypoints:
            for wp in reversed(lifo_waypoints):
                if hasattr(wp, "x") and hasattr(wp, "y"):
                    wx, wy = float(wp.x), float(wp.y)
                else:
                    wx, wy = float(wp[0]), float(wp[1])

                dist_wp = math.hypot(wx - rx, wy - ry)
                min_clr = math.inf
                for hx, hy, hr in violated_ch2_world:
                    r_safe_h = self.r_circle + hr + self.semantic_margin
                    clr = math.hypot(wx - hx, wy - hy) - r_safe_h
                    if clr < min_clr:
                        min_clr = clr

                seg_ok = _is_segment_safe(wx, wy)
                if min_clr > 0.0 and dist_wp > 0.05 and seg_ok:
                    return wx, wy
                if seg_ok and dist_wp > 0.05 and min_clr > best_safe_clr:
                    best_safe_clr = min_clr
                    best_safe_wp = (wx, wy)
                if min_clr > best_any_clr:
                    best_any_clr = min_clr
                    best_any_wp = (wx, wy)

        if best_safe_wp is not None and best_safe_clr > -0.10:
            return best_safe_wp

        # Radial outward fallback away from violated gas zone center(s) (never cuts a chord)
        out_x = 0.0
        out_y = 0.0
        max_r_safe = 0.0
        for hx, hy, hr in violated_ch2_world:
            dx_o = rx - hx
            dy_o = ry - hy
            d_o = math.hypot(dx_o, dy_o)
            r_safe_h = self.r_circle + hr + self.semantic_margin
            if r_safe_h > max_r_safe:
                max_r_safe = r_safe_h
            if d_o > 1e-4:
                out_x += dx_o / d_o
                out_y += dy_o / d_o
        out_norm = math.hypot(out_x, out_y)
        if out_norm > 1e-4:
            step_d = max(0.35, max_r_safe + 0.25)
            return (rx + (out_x / out_norm) * step_d, ry + (out_y / out_norm) * step_d)

        return best_any_wp

    def _compute_mq2_escape_nominal(
        self,
        rx: float,
        ry: float,
        rth: float,
        target_xy: Tuple[float, float],
    ) -> Tuple[float, float]:
        """Monotonic approach nominal command toward the selected safe LIFO waypoint."""
        dx = float(target_xy[0]) - rx
        dy = float(target_xy[1]) - ry
        dist = math.hypot(dx, dy)
        if dist < 1e-4:
            return 0.0, 0.0
        alpha = _wrap_angle(math.atan2(dy, dx) - rth)
        w_esc = max(-self.w_max, min(self.w_max, 1.60 * alpha))
        if abs(alpha) > 1.15:
            v_esc = 0.0
        else:
            v_esc = max(0.0, min(self.v_max, self.v_max * max(0.20, math.cos(alpha))))
        return float(v_esc), float(w_esc)

    @staticmethod
    def _unpack_dynamic_obstacle(obs: Any) -> Tuple[float, float, float, float, float, float]:
        """
        Unpack a TrackedObstacle instance or tuple into (x, y, vx, vy, radius, lam_max_pos_cov).
        """
        if hasattr(obs, "x") and hasattr(obs, "y") and hasattr(obs, "vx") and hasattr(obs, "vy"):
            ox = float(obs.x)
            oy = float(obs.y)
            vx = float(obs.vx)
            vy = float(obs.vy)
            rad = float(getattr(obs, "radius", 0.18))
            P = getattr(obs, "P_4x4", None)
            if P is not None:
                lam_max = max_eigenvalue_2x2(np.asarray(P, dtype=np.float64)[:2, :2])
            else:
                lam_max = float(getattr(obs, "lam_max", 0.0))
            return ox, oy, vx, vy, rad, lam_max

        # Tuple form: (x, y, vx, vy, radius) or (x, y, vx, vy, radius, lam_max_or_cov2x2)
        ox, oy, vx, vy, rad = (
            float(obs[0]),
            float(obs[1]),
            float(obs[2]),
            float(obs[3]),
            float(obs[4]),
        )
        lam_max = 0.0
        if len(obs) >= 6 and obs[5] is not None:
            extra = obs[5]
            if isinstance(extra, np.ndarray) and extra.ndim == 2 and extra.shape[0] >= 2:
                lam_max = max_eigenvalue_2x2(extra[:2, :2])
            else:
                lam_max = max(0.0, float(extra))
        return ox, oy, vx, vy, rad, lam_max

    def _update_chatter_stats(self, result: CBFResult) -> None:
        self.total_steps += 1
        if result.intervened:
            self.intervention_steps += 1
        self.sum_intervention_norm += result.intervention_norm
        if result.max_slack > 1e-4:
            self.slack_steps += 1
        if result.max_slack > self.max_slack_seen:
            self.max_slack_seen = result.max_slack
        if result.mq2_escape_active:
            self.mq2_escape_steps += 1
        self.total_solve_ms += result.solve_time_ms
        if result.solve_time_ms > self.max_solve_ms:
            self.max_solve_ms = result.solve_time_ms

        v = result.v
        w = result.w
        if self._prev_v is not None:
            self.sum_abs_dv += abs(v - self._prev_v)
            self.sum_abs_dw += abs(w - self._prev_w)
        self._prev_v = v
        self._prev_w = w

        db = self.chatter_deadband_mps
        if v < -db:
            self.reverse_steps += 1
            cur_sign = -1
        elif v > db:
            cur_sign = 1
        else:
            cur_sign = 0

        if cur_sign != 0:
            if self._prev_nonzero_v_sign != 0 and cur_sign != self._prev_nonzero_v_sign:
                self.v_sign_reversals += 1
            self._prev_nonzero_v_sign = cur_sign

        moving_fwd = v > 0.02
        stalled = v <= 0.005
        if self._prev_moving_fwd is None:
            if moving_fwd:
                self._prev_moving_fwd = True
            elif stalled:
                self._prev_moving_fwd = False
        else:
            if self._prev_moving_fwd and stalled:
                self.v_stall_toggles += 1
                self._prev_moving_fwd = False
            elif (not self._prev_moving_fwd) and moving_fwd:
                self.v_stall_toggles += 1
                self._prev_moving_fwd = True

    def _apply_near_ray_latch(
        self,
        ranges: np.ndarray,
        rx: float = 0.0,
        ry: float = 0.0,
        rth: float = 0.0,
    ) -> np.ndarray:
        """
        Guard against LDS-01 minimum-range blind spot (< 0.12 m returning 0/inf/nan or
        positive ultra-close values 0 < r < min_range):
          - Positive ultra-close returns (0 < r < min_range) are distinguished from 0/inf/nan
            dropouts, clamped to lidar_min_range_m, and arm/refresh the latch TTL.
          - Close valid hits (min_range <= r < 0.185 m) arm/refresh TTL and store armed pose.
          - Dropouts (<= 0 / inf / nan) on latched rays stay clamped at lidar_min_range_m;
            TTL only decrements once the robot moves/rotates away from the armed pose
            (>= 0.03 m or >= 0.08 rad) so prolonged wall-pressed stalls never lose the barrier.
        """
        n = ranges.shape[0]
        if self._near_ray_ttl is None or self._near_ray_ttl.shape[0] != n:
            self._near_ray_ttl = np.zeros(n, dtype=np.int16)
            self._near_ray_arm_x = np.zeros(n, dtype=np.float64)
            self._near_ray_arm_y = np.zeros(n, dtype=np.float64)
            self._near_ray_arm_th = np.zeros(n, dtype=np.float64)
        ttl = self._near_ray_ttl
        arm_x = self._near_ray_arm_x
        arm_y = self._near_ray_arm_y
        arm_th = self._near_ray_arm_th

        r_min = self.lidar_min_range_m
        rx_f = float(rx)
        ry_f = float(ry)
        rth_f = float(rth)

        finite_pos = np.isfinite(ranges) & (ranges > 0.0)
        ultra_close = finite_pos & (ranges < r_min)
        close_hit = finite_pos & (ranges >= r_min) & (ranges < 0.185)
        clear_hit = finite_pos & (ranges >= 0.185)
        arm_mask = ultra_close | close_hit
        dropped = (~finite_pos) & (ttl > 0)

        if np.any(arm_mask):
            ttl[arm_mask] = 6
            arm_x[arm_mask] = rx_f
            arm_y[arm_mask] = ry_f
            arm_th[arm_mask] = rth_f
        ttl[clear_hit] = 0

        has_ultra = bool(np.any(ultra_close))
        has_drop = bool(np.any(dropped))
        if has_ultra or has_drop:
            out = ranges.copy()
            if has_ultra:
                out[ultra_close] = r_min
            if has_drop:
                out[dropped] = r_min
                drop_idx = np.flatnonzero(dropped)
                dx = rx_f - arm_x[drop_idx]
                dy = ry_f - arm_y[drop_idx]
                d2 = dx * dx + dy * dy
                dth = np.abs(((rth_f - arm_th[drop_idx] + math.pi) % (2.0 * math.pi)) - math.pi)
                moved_away = (d2 >= 0.03 * 0.03) | (dth >= 0.08)
                if np.any(moved_away):
                    ttl[drop_idx[moved_away]] -= 1
            return out
        return ranges

    def build_constraints(
        self,
        rx: float = 0.0,
        ry: float = 0.0,
        rth: float = 0.0,
        dt: Optional[float] = None,
        scan_ranges: Optional[Sequence[float]] = None,
        angle_min: float = 0.0,
        angle_increment: Optional[float] = None,
        scan_pts_sensor: Optional[np.ndarray] = None,
        scan_pose: Optional[Tuple[float, float, float]] = None,
        dynamic_obstacles: Optional[Sequence[Any]] = None,
        ch1_obstacles: Optional[Sequence[Tuple[float, float, float]]] = None,
        ch2_hazards: Optional[Sequence[Tuple[float, float, float]]] = None,
        lifo_waypoints: Optional[Sequence[Any]] = None,
        safe_escape_wp: Optional[Tuple[float, float]] = None,
        use_near_latch: bool = False,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        float,
        float,
        bool,
        Optional[Tuple[float, float]],
    ]:
        """
        Construct the unified 2-Circle linear CBF constraint matrices (A, b, p_delta)
        in robot `base_footprint` coordinates, plus min barrier values and MQ-2 escape state.
        """
        dt_use = max(1e-4, float(self.dt_default if dt is None else dt))
        cos_th = math.cos(float(rth))
        sin_th = math.sin(float(rth))
        rx_f = float(rx)
        ry_f = float(ry)

        # 1. LiDAR points: base_scan -> base_footprint TF + optional near-ray latch +
        #    5 Hz scan-pose ego-motion compensation + 0.10 m voxel nearest-16 downsampling
        cos_lut = None
        sin_lut = None
        ranges_use = scan_ranges
        if scan_pts_sensor is None and scan_ranges is not None and len(scan_ranges) > 0:
            cos_lut, sin_lut = self._get_trig_lut(len(scan_ranges), angle_min, angle_increment)
            if use_near_latch:
                r_arr = scan_ranges if (isinstance(scan_ranges, np.ndarray) and scan_ranges.dtype == np.float64) else np.asarray(scan_ranges, dtype=np.float64)
                ranges_use = self._apply_near_ray_latch(r_arr, rx=rx_f, ry=ry_f, rth=float(rth))

        pts_body = transform_scan_to_base_footprint(
            scan_ranges=ranges_use,
            angle_min=angle_min,
            angle_increment=angle_increment,
            scan_pts_sensor=scan_pts_sensor,
            lidar_offset_x=self.lidar_offset_x,
            lidar_offset_y=self.lidar_offset_y,
            min_range=self.lidar_min_range_m,
            max_range=self.lidar_max_range_m,
            cos_lut=cos_lut,
            sin_lut=sin_lut,
        )

        # Compensate 5 Hz scan staleness by transforming points from scan_pose into current pose
        if scan_pose is not None and pts_body.shape[0] > 0:
            sx, sy, sth = float(scan_pose[0]), float(scan_pose[1]), float(scan_pose[2])
            dx_w = sx - rx_f
            dy_w = sy - ry_f
            dth = _wrap_angle(sth - float(rth))
            if (abs(dx_w) + abs(dy_w) + abs(dth) > 1e-4) and (dx_w * dx_w + dy_w * dy_w < 0.25):
                cd = math.cos(dth)
                sd = math.sin(dth)
                tx = cos_th * dx_w + sin_th * dy_w
                ty = -sin_th * dx_w + cos_th * dy_w
                xb_s = pts_body[:, 0]
                yb_s = pts_body[:, 1]
                xb_new = cd * xb_s - sd * yb_s + tx
                yb_new = sd * xb_s + cd * yb_s + ty
                pts_body[:, 0] = xb_new
                pts_body[:, 1] = yb_new

        static_pts = voxel_downsample_nearest_2circle(
            pts_body,
            l_f=self.l_f,
            l_r=self.l_r,
            voxel_size=self.lidar_voxel_m,
            max_points=self.lidar_max_points,
            max_dist_m=self.lidar_max_range_m,
        )
        self._last_rear_min_m = math.inf

        # Collect non-LiDAR extra obstacles (typically 0..6 items) into compact tuples first
        aodt_static = compute_alpha(self.gamma_static, dt_use) / dt_use
        aodt_dyn = compute_alpha(self.gamma_dynamic, dt_use) / dt_use
        aodt_gas = compute_alpha(self.gamma_gas, dt_use) / dt_use

        # Extra items: (xb, yb, vxb, vyb, r_safe, aodt, p_delta, a_max, cat)
        extra_obs: List[Tuple[float, float, float, float, float, float, float, float, int]] = []

        # 2. Ch1 low-lying semantic obstacles (world -> body frame)
        if ch1_obstacles:
            for ox, oy, rad in ch1_obstacles:
                dxw = float(ox) - rx_f
                dyw = float(oy) - ry_f
                xb = cos_th * dxw + sin_th * dyw
                yb = -sin_th * dxw + cos_th * dyw
                r_safe = self.r_circle + float(rad) + self.semantic_margin
                if math.hypot(xb, yb) - r_safe <= 1.50:
                    extra_obs.append((xb, yb, 0.0, 0.0, r_safe, aodt_static, self.p_delta_static, 0.0, 0))

        # 3. Ch2 gas/survivor zones + MQ-2 inside (deep interior / local MQ-2) LIFO escape branch
        mq2_escape_active = False
        mq2_escape_target: Optional[Tuple[float, float]] = None
        min_h_gas = math.inf
        violated_ch2_world: List[Tuple[float, float, float]] = []

        if ch2_hazards:
            for item in ch2_hazards:
                hx_f, hy_f, hr_f = float(item[0]), float(item[1]), float(item[2])
                allow_mq2 = bool(item[3]) if len(item) >= 4 else True
                is_local_mq2 = bool(item[4]) if len(item) >= 5 else False
                dxw = hx_f - rx_f
                dyw = hy_f - ry_f
                xb = cos_th * dxw + sin_th * dyw
                yb = -sin_th * dxw + cos_th * dyw
                r_safe = self.r_circle + hr_f + self.semantic_margin
                d2_f = (self.l_f - xb) ** 2 + yb * yb
                d2_r = (-self.l_r - xb) ** 2 + yb * yb
                h_min_i = min(d2_f, d2_r) - r_safe * r_safe
                if h_min_i < min_h_gas:
                    min_h_gas = h_min_i

                d_center = math.hypot(xb, yb)
                # Trigger MQ-2 LIFO escape ONLY for GAS_HAZARD entries when either:
                #   (a) locally triggered by onboard MQ-2 while robot center is inside hr (d < hr), OR
                #   (b) robot center is deep inside the actual gas radius hr (singularity risk zone).
                # Outer padding grazing (hr <= d < r_safe) and non-gas classes (person/AUDIO_DISTRESS)
                # MUST retain standard Ch2 CBF repulsion and never invoke mq2_escape.
                deep_gas_r = min(0.75 * hr_f, max(0.50 * hr_f, hr_f - self.r_circle - self.semantic_margin))
                in_mq2_escape_zone = (
                    self.enable_mq2_lifo_escape
                    and allow_mq2
                    and (h_min_i < 0.0)
                    and ((is_local_mq2 and d_center < hr_f) or (d_center < deep_gas_r))
                )
                if in_mq2_escape_zone:
                    violated_ch2_world.append((hx_f, hy_f, hr_f))
                elif d_center - r_safe <= 1.50:
                    extra_obs.append((xb, yb, 0.0, 0.0, r_safe, aodt_gas, self.p_delta_gas, 0.0, 2))

            if violated_ch2_world:
                mq2_escape_active = True
                mq2_escape_target = self._select_safe_lifo_waypoint(
                    rx=rx_f,
                    ry=ry_f,
                    violated_ch2_world=violated_ch2_world,
                    lifo_waypoints=lifo_waypoints,
                    safe_escape_wp=safe_escape_wp,
                )

        # 4. Stage 1 confirmed dynamic obstacles (world -> body frame)
        has_dyn = False
        best_dyn_d = 1.25
        best_dyn_ycpa: Optional[float] = None
        dyn_block_reverse = False
        dyn_rear_aw_zero_flags: List[bool] = []
        dyn_cpa_av_override: List[Optional[Tuple[float, float]]] = []
        if dynamic_obstacles:
            l_f_val = self.l_f
            for dob in dynamic_obstacles:
                if hasattr(dob, "is_confirmed") and not bool(dob.is_confirmed):
                    continue
                ox, oy, vx, vy, rad, lam_max = self._unpack_dynamic_obstacle(dob)
                dxw = ox - rx_f
                dyw = oy - ry_f
                xb = cos_th * dxw + sin_th * dyw
                yb = -sin_th * dxw + cos_th * dyw
                vxb = cos_th * vx + sin_th * vy
                vyb = -sin_th * vx + cos_th * vy
                spd_v = math.hypot(vxb, vyb)
                if spd_v > 1.0:
                    sc_v = 1.0 / spd_v
                    vxb *= sc_v
                    vyb *= sc_v
                    spd_v = 1.0
                sigma_pad = min(0.14, self.dynamic_cov_sigma_mult * math.sqrt(max(0.0, lam_max)))
                m_pad = self.dynamic_margin_base + sigma_pad
                r_safe = self.r_circle + rad + m_pad
                d_o = math.hypot(xb, yb)
                if d_o - r_safe <= 2.20:
                    extra_obs.append((xb, yb, vxb, vyb, r_safe, aodt_dyn, self.p_delta_dynamic, self.a_o_max_dynamic, 1))
                    has_dyn = True
                    front_app_i = (xb > l_f_val) and (((xb - l_f_val) * vxb + yb * vyb) < 0.0)
                    dot_app = xb * vxb + yb * vyb
                    lat_exit_i = False
                    cpa_ov_i: Optional[Tuple[float, float]] = None
                    if (d_o <= r_safe + 1.20) and (spd_v > 0.05) and (dot_app < 0.0):
                        inv_spd = 1.0 / spd_v
                        ux_d = vxb * inv_spd
                        uy_d = vyb * inv_spd
                        d_lat_d = xb * uy_d - yb * ux_d
                        abs_d_lat_d = abs(d_lat_d)
                        if (abs_d_lat_d > 0.02) and (abs_d_lat_d < r_safe + 0.25):
                            exit_align_i = (d_lat_d * (-uy_d)) / abs_d_lat_d
                            if exit_align_i > 0.20:
                                lat_exit_i = True
                                dyn_block_reverse = True
                                if exit_align_i > 0.25:
                                    t_cpa0 = min(0.6, max(0.0, -dot_app / max(spd_v * spd_v, 1e-4)))
                                    av_cpa_i = -2.0 * t_cpa0 * d_lat_d * (-uy_d)
                                    cpa_ov_i = (av_cpa_i, r_safe - 0.03)
                    dyn_rear_aw_zero_flags.append(front_app_i or lat_exit_i)
                    dyn_cpa_av_override.append(cpa_ov_i)
                    if d_o < best_dyn_d and (dot_app < 0.0 or (d_o - r_safe) < 0.20):
                        rvx = vxb - 0.12
                        t_cpa = max(0.0, min(1.5, -(xb * rvx + yb * vyb) / max(rvx * rvx + vyb * vyb, 1e-6)))
                        best_dyn_d = d_o
                        best_dyn_ycpa = yb + vyb * t_cpa
        self._last_dyn_threat_ycpa = best_dyn_ycpa
        self._last_has_dyn = has_dyn
        self._last_dyn_block_reverse = dyn_block_reverse

        n_stat = static_pts.shape[0]
        n_ext = len(extra_obs)
        n_obs = n_stat + n_ext
        if n_obs == 0:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                math.inf,
                math.inf,
                min_h_gas,
                mq2_escape_active,
                mq2_escape_target,
            )

        xb_arr = np.empty(n_obs, dtype=np.float64)
        yb_arr = np.empty(n_obs, dtype=np.float64)
        rsafe_arr = np.empty(n_obs, dtype=np.float64)
        aodt_arr = np.empty(n_obs, dtype=np.float64)
        pdel_arr = np.empty(n_obs, dtype=np.float64)
        if has_dyn:
            vxb_arr = np.zeros(n_obs, dtype=np.float64)
            vyb_arr = np.zeros(n_obs, dtype=np.float64)
            amax_arr = np.zeros(n_obs, dtype=np.float64)

        if n_stat > 0:
            xb_arr[:n_stat] = static_pts[:, 0]
            yb_arr[:n_stat] = static_pts[:, 1]
            rsafe_arr[:n_stat] = self.r_safe_static
            aodt_arr[:n_stat] = aodt_static
            pdel_arr[:n_stat] = self.p_delta_static

        n_ch1_added = 0
        dyn_start = n_obs
        for idx_e, item in enumerate(extra_obs):
            j = n_stat + idx_e
            xb_arr[j] = item[0]
            yb_arr[j] = item[1]
            rsafe_arr[j] = item[4]
            aodt_arr[j] = item[5]
            pdel_arr[j] = item[6]
            if has_dyn:
                vxb_arr[j] = item[2]
                vyb_arr[j] = item[3]
                amax_arr[j] = item[7]
            if item[8] == 0:
                n_ch1_added += 1
            elif item[8] == 1 and dyn_start == n_obs:
                dyn_start = j

        # Build 2 * n_obs constraints (front circle s_f = +l_f, rear circle s_r = -l_r)
        A = np.empty((2 * n_obs, 2), dtype=np.float64)
        b_vec = np.empty((2 * n_obs,), dtype=np.float64)
        p_vec = np.empty((2 * n_obs,), dtype=np.float64)

        vw_term = abs(self.v_max * self.w_max)
        w2_term = self.w_max * self.w_max
        rsafe2 = rsafe_arr * rsafe_arr
        dv_slip_2 = 2.0 * self.slip_margin_mps
        neg_yb = -yb_arr
        yy = neg_yb * neg_yb

        stat_end = n_stat + n_ch1_added
        min_h_static = math.inf
        min_h_dynamic = math.inf

        p_front = pdel_arr * self.front_circle_priority if self.front_circle_priority != 1.0 else pdel_arr

        for k_idx, s_k in enumerate((self.l_f, -self.l_r)):
            sl = slice(k_idx * n_obs, (k_idx + 1) * n_obs)
            dx = s_k - xb_arr
            d2 = dx * dx + yy
            d = np.sqrt(d2)
            h = d2 - rsafe2

            base_rate = dv_slip_2 + (vw_term + abs(s_k) * w2_term) * dt_use
            if has_dyn:
                eps_dt_slip = d * (base_rate + amax_arr * dt_use)
                # Grazing guard: d*(...) vanishes as d -> 0, so enforce an absolute floor
                # on dynamic rows only (static rows and QP dimensions are unchanged).
                if dyn_start < n_obs:
                    v_rel_cap = (
                        self.v_max + abs(s_k) * self.w_max + self.a_o_max_dynamic * dt_use
                    )
                    eps_dt_slip[dyn_start:] = np.maximum(
                        eps_dt_slip[dyn_start:],
                        v_rel_cap * dt_use + self.dynamic_graze_buffer_m,
                    )
                b_vec[sl] = aodt_arr * h - 2.0 * (dx * vxb_arr + neg_yb * vyb_arr) - eps_dt_slip
            else:
                b_vec[sl] = aodt_arr * h - d * base_rate
            A[sl, 0] = -2.0 * dx
            A[sl, 1] = (-2.0 * s_k) * neg_yb
            if has_dyn and dyn_start < n_obs:
                row_base = k_idx * n_obs + dyn_start
                for i_d, cpa_ov in enumerate(dyn_cpa_av_override):
                    if cpa_ov is not None and d[dyn_start + i_d] > cpa_ov[1]:
                        A[row_base + i_d, 0] = cpa_ov[0]
                if k_idx == 1:
                    # Front-approaching or laterally-exited dynamic obstacles:
                    # project rear-circle a_w -> 0 so rear leverage (-l_r) never counter-steers
                    # the nose into the obstacle against the front circle (+l_f).
                    for i_d, zero_aw in enumerate(dyn_rear_aw_zero_flags):
                        if zero_aw:
                            A[row_base + i_d, 1] = 0.0
            p_vec[sl] = p_front if k_idx == 0 else pdel_arr

            if stat_end > 0:
                hs_min = float(h[:stat_end].min())
                if hs_min < min_h_static:
                    min_h_static = hs_min
                if use_near_latch and k_idx == 1 and n_stat > 0 and self.v_min < 0.0:
                    # Rear static points directly behind chassis rear circle (-l_r): dx > 0 and |y| <= 0.105m
                    dx_s = dx[:n_stat]
                    rear_m = (dx_s > 0.0) & (np.abs(yb_arr[:n_stat]) <= 0.105)
                    if np.any(rear_m):
                        self._last_rear_min_m = float(d[:n_stat][rear_m].min()) + self.l_r + self.lidar_offset_x
            if has_dyn and dyn_start < n_obs:
                hd_min = float(h[dyn_start:].min())
                if hd_min < min_h_dynamic:
                    min_h_dynamic = hd_min

        return (
            A,
            b_vec,
            p_vec,
            min_h_static,
            min_h_dynamic,
            min_h_gas,
            mq2_escape_active,
            mq2_escape_target,
        )

    def filter(
        self,
        v_nom: float,
        w_nom: float,
        rx: float = 0.0,
        ry: float = 0.0,
        rth: float = 0.0,
        dt: Optional[float] = None,
        scan_ranges: Optional[Sequence[float]] = None,
        angle_min: float = 0.0,
        angle_increment: Optional[float] = None,
        scan_pts_sensor: Optional[np.ndarray] = None,
        scan_pose: Optional[Tuple[float, float, float]] = None,
        dynamic_obstacles: Optional[Sequence[Any]] = None,
        ch1_obstacles: Optional[Sequence[Tuple[float, float, float]]] = None,
        ch2_hazards: Optional[Sequence[Tuple[float, float, float]]] = None,
        lifo_waypoints: Optional[Sequence[Any]] = None,
        safe_escape_wp: Optional[Tuple[float, float]] = None,
        update_stats: bool = True,
        prefer_no_reverse: bool = False,
        suppress_tangential_steer: bool = False,
    ) -> CBFResult:
        """
        Filter nominal command (v_nom, w_nom) through the 2-Circle Soft-CBF-QP safety shield.
        """
        t0 = time.perf_counter()

        (
            A,
            b_vec,
            p_vec,
            min_h_static,
            min_h_dynamic,
            min_h_gas,
            mq2_escape_active,
            mq2_escape_target,
        ) = self.build_constraints(
            rx=rx,
            ry=ry,
            rth=rth,
            dt=dt,
            scan_ranges=scan_ranges,
            angle_min=angle_min,
            angle_increment=angle_increment,
            scan_pts_sensor=scan_pts_sensor,
            scan_pose=scan_pose,
            dynamic_obstacles=dynamic_obstacles,
            ch1_obstacles=ch1_obstacles,
            ch2_hazards=ch2_hazards,
            lifo_waypoints=lifo_waypoints,
            safe_escape_wp=safe_escape_wp,
            use_near_latch=update_stats,
        )

        v_eff = float(v_nom)
        w_eff = float(w_nom)
        if mq2_escape_active and mq2_escape_target is not None:
            v_eff, w_eff = self._compute_mq2_escape_nominal(
                rx=float(rx),
                ry=float(ry),
                rth=float(rth),
                target_xy=mq2_escape_target,
            )

        u_nom_buf = self._u_nom_buf
        u_nom_buf[0] = v_eff
        u_nom_buf[1] = w_eff

        # Block unnecessary reverse during verified crossing HOLD/DASH when barriers are not breached
        # (full reverse remains available whenever min_h_dynamic < 0 or min_h_static < -0.002).
        block_rev_step = (
            self._last_dyn_block_reverse
            or (
                bool(prefer_no_reverse)
                and (min_h_dynamic >= 0.0)
                and (min_h_static >= -0.002)
            )
        )
        u_lo_step = self._u_lo_zero if (block_rev_step and self.v_min < 0.0) else self._u_lo
        v_min_step = 0.0 if (block_rev_step and self.v_min < 0.0) else self.v_min

        if not update_stats:
            u_opt, iters, kkt_res, max_slack, sum_slack_sq, n_act = solve_soft_cbf_qp_pdas(
                u_nom=u_nom_buf,
                A=A,
                b=b_vec,
                p_delta=p_vec,
                w_v=self.w_v,
                w_w=self.w_w,
                u_lo=u_lo_step,
                u_hi=self._u_hi,
                max_iters=self.max_pdas_iters,
                kkt_tol=self.kkt_tol,
            )
            solve_ms = (time.perf_counter() - t0) * 1000.0
            v_out = float(u_opt[0])
            w_out = float(u_opt[1])
            interv_norm = math.hypot(v_out - float(v_nom), w_out - float(w_nom))
            intervened = (interv_norm > 1e-4) or mq2_escape_active
            return CBFResult(
                v=v_out,
                w=w_out,
                v_nom=v_eff,
                w_nom=w_eff,
                intervened=intervened,
                intervention_norm=interv_norm,
                num_constraints=int(A.shape[0]),
                num_active_constraints=n_act,
                max_slack=max_slack,
                sum_slack_sq=sum_slack_sq,
                min_h_static=min_h_static,
                min_h_dynamic=min_h_dynamic,
                min_h_gas=min_h_gas,
                pdas_iters=iters,
                kkt_residual=kkt_res,
                mq2_escape_active=mq2_escape_active,
                mq2_escape_target=mq2_escape_target,
                solve_time_ms=solve_ms,
            )

        u_lo_use = u_lo_step
        u_hi_use = self._u_hi
        wv_use = self.w_v
        ww_use = self.w_w

        # Fast-path check: keep exact zero intervention inside safe interior when not in hysteresis
        has_nom_viol = False
        if A.shape[0] > 0:
            v_clip = self.v_max if v_eff > self.v_max else (v_min_step if v_eff < v_min_step else v_eff)
            w_clip = self.w_max if w_eff > self.w_max else (-self.w_max if w_eff < -self.w_max else w_eff)
            u_nom_buf[0] = v_clip
            u_nom_buf[1] = w_clip
            has_nom_viol = bool((A @ u_nom_buf > b_vec).any())
            u_nom_buf[0] = v_eff
            u_nom_buf[1] = w_eff

        hyst_active = False
        if self._rev_hyst_ticks > 0:
            if min_h_dynamic >= 0.015 and min_h_static >= 0.005:
                self._rev_hyst_ticks = 0
            else:
                hyst_active = True

        if has_nom_viol or hyst_active:
            v_lo_eff = v_min_step
            v_hi_eff = self.v_max

            # 1) Block backing into a close rear static wall/corner (cached in build_constraints)
            if v_lo_eff < 0.0 and self.rear_reverse_block_m > 0.0 and self._last_rear_min_m < self.rear_reverse_block_m:
                v_lo_eff = 0.0

            # 2) Tangential steering preference for nearby/approaching dynamic obstacles (cached in build_constraints)
            best_ycpa = self._last_dyn_threat_ycpa
            if (
                (not suppress_tangential_steer)
                and self.tangential_steer_gain > 0.0
                and best_ycpa is not None
                and not mq2_escape_active
            ):
                if abs(best_ycpa) < 0.06 and self._last_tan_sign != 0.0:
                    tan_sign = self._last_tan_sign
                else:
                    tan_sign = -1.0 if best_ycpa >= 0.0 else 1.0
                    self._last_tan_sign = tan_sign
                w_tan = tan_sign * min(self.w_max, self.tangential_steer_gain)
                w_eff = max(-self.w_max, min(self.w_max, 0.35 * w_eff + 0.65 * w_tan))

            # 3) Forward-reverse hysteresis: hold v <= 0 for 1 tick while barrier is still deeply active
            if hyst_active and v_hi_eff > 0.0:
                v_hi_eff = 0.0

            # 4) Lightweight analytic velocity-rate penalty (only when dynamic obstacles are active;
            #    keep static wall braking/reversing crisp in static scenarios)
            if self._last_has_dyn and self._prev_v is not None and self._prev_w is not None:
                lv = self.rate_penalty_v
                lw = self.rate_penalty_w
                if lv > 0.0 or lw > 0.0:
                    wv_use = self.w_v + lv
                    ww_use = self.w_w + lw
                    u_nom_buf[0] = (self.w_v * v_eff + lv * self._prev_v) / wv_use
                    u_nom_buf[1] = (self.w_w * w_eff + lw * self._prev_w) / ww_use
                else:
                    u_nom_buf[0] = v_eff
                    u_nom_buf[1] = w_eff
            else:
                u_nom_buf[0] = v_eff
                u_nom_buf[1] = w_eff

            u_lo_use = self._u_lo_buf
            u_lo_use[0] = v_lo_eff
            u_lo_use[1] = -self.w_max
            u_hi_use = self._u_hi_buf
            u_hi_use[0] = v_hi_eff
            u_hi_use[1] = self.w_max

            u_opt, iters, kkt_res, max_slack, sum_slack_sq, n_act = solve_soft_cbf_qp_pdas(
                u_nom=u_nom_buf,
                A=A,
                b=b_vec,
                p_delta=p_vec,
                w_v=wv_use,
                w_w=ww_use,
                u_lo=u_lo_use,
                u_hi=u_hi_use,
                max_iters=self.max_pdas_iters,
                kkt_tol=self.kkt_tol,
            )
        else:
            u_opt, iters, kkt_res, max_slack, sum_slack_sq, n_act = solve_soft_cbf_qp_pdas(
                u_nom=u_nom_buf,
                A=A,
                b=b_vec,
                p_delta=p_vec,
                w_v=self.w_v,
                w_w=self.w_w,
                u_lo=u_lo_step,
                u_hi=self._u_hi,
                max_iters=self.max_pdas_iters,
                kkt_tol=self.kkt_tol,
            )

        solve_ms = (time.perf_counter() - t0) * 1000.0
        v_out = float(u_opt[0])
        w_out = float(u_opt[1])
        if update_stats:
            if v_out < -self.chatter_deadband_mps:
                self._rev_hyst_ticks = self.reverse_hysteresis_ticks
            elif self._rev_hyst_ticks > 0:
                self._rev_hyst_ticks -= 1
            if not has_nom_viol and min_h_dynamic >= 0.15:
                self._last_tan_sign = 0.0
        interv_norm = math.hypot(v_out - float(v_nom), w_out - float(w_nom))
        intervened = (interv_norm > 1e-4) or mq2_escape_active

        res = CBFResult(
            v=v_out,
            w=w_out,
            v_nom=v_eff,
            w_nom=w_eff,
            intervened=intervened,
            intervention_norm=interv_norm,
            num_constraints=int(A.shape[0]),
            num_active_constraints=n_act,
            max_slack=max_slack,
            sum_slack_sq=sum_slack_sq,
            min_h_static=min_h_static,
            min_h_dynamic=min_h_dynamic,
            min_h_gas=min_h_gas,
            pdas_iters=iters,
            kkt_residual=kkt_res,
            mq2_escape_active=mq2_escape_active,
            mq2_escape_target=mq2_escape_target,
            solve_time_ms=solve_ms,
        )

        if update_stats:
            self._update_chatter_stats(res)
        return res
