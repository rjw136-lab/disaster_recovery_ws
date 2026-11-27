#!/usr/bin/env python3
"""
Unit & Stress Tests for Stage 2 (Phase 4-D): Front/Rear 2-Circle Covering Soft-CBF-QP
Safety Shield (Pure-NumPy 2x2 PDAS + Armijo 1D Line Search).

Covers:
  1. Pure-NumPy dependency isolation (zero rclpy, cvxpy, osqp, scipy, torch).
  2. model.sdf 2-Circle Covering geometry (l_f=0.020, l_r=0.060, r_circle=0.098,
     R_safe,static=0.133) + zero rear protrusion + zero interior intervention.
  3. base_scan -> base_footprint static TF (x_b = x_s - 0.032, y_b = y_s) accuracy.
  4. Asymmetric Q = diag(w_v, w_w) (w_v : w_w >= 10:1) steering-authority compensation.
  5. Mandatory PDAS stress gate: kappa = 10^4 (KKT residual <= 1e-6, ref error <= 1e-5).
  6. 0.28 m narrow doorway |phi| <= 10 deg zero-slack feasibility + eccentric soft recovery.
  7. 0.10 m spatial voxel downsampling keeping nearest 16 points to 2-Circle.
  8. Continuous decay order gamma_dyn=3.0 < gamma_static=6.0 <= gamma_gas=8.0 and
     10 Hz / 20 Hz rate-invariant consistency + eps_inter & dynamic velocity braking.
  9. Onboard MQ-2 Ch2 inside (h < 0) LIFO safe-waypoint escape exception branch.
 10. v_min = -0.10 vs 0.0 parameterization and velocity sign-reversal (chattering) telemetry.
 11. QP + full shield latency < 0.25 ms.
 12. recovery_controller.py enable_cbf_safety_shield default-False & guard gating integration.
"""

import ast
import math
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import pytest

PKG_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, PKG_SRC_DIR)

from disaster_recovery.cbf_safety_shield import (  # noqa: E402
    DEFAULT_L_F,
    DEFAULT_L_R,
    DEFAULT_LIDAR_OFFSET_X,
    DEFAULT_R_CIRCLE,
    DEFAULT_R_SAFE_STATIC,
    DiscreteSoftCBFShield,
    compute_alpha,
    compute_kkt_residual,
    solve_soft_cbf_qp_pdas,
    transform_scan_to_base_footprint,
    voxel_downsample_nearest_2circle,
)
from disaster_recovery.dynamic_tracker import TrackedObstacle  # noqa: E402


def _make_open_scan(n_rays: int = 360, fill: float = 3.5) -> np.ndarray:
    return np.full(n_rays, fill, dtype=np.float64)


def _solve_reference_convex_2d(
    u_nom: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    p_delta: np.ndarray,
    w_v: float,
    w_w: float,
    u_lo: np.ndarray,
    u_hi: np.ndarray,
) -> np.ndarray:
    """
    Independent exact reference solver for small 2D box-constrained piecewise-quadratic QP:
    enumerates candidate active-set regions via damped active-set + coordinate refinement
    to 1e-13 precision, completely independent of solve_soft_cbf_qp_pdas.
    """
    wv = float(w_v)
    ww = float(w_w)
    un = np.asarray(u_nom, dtype=np.float64)
    lo = np.asarray(u_lo, dtype=np.float64)
    hi = np.asarray(u_hi, dtype=np.float64)

    def obj(u: np.ndarray) -> float:
        d = u - un
        v = 0.5 * (wv * d[0] * d[0] + ww * d[1] * d[1])
        if A.shape[0] > 0:
            r = np.maximum(0.0, A @ u - b)
            v += 0.5 * float(np.dot(p_delta * r, r))
        return v

    # Exact 1D golden-section + coordinate minimization (globally convergent on smooth strongly convex 2D box)
    u = np.clip(un.copy(), lo, hi)
    # Warm start via exact 2D active-set enumeration over small M when M <= 10
    m = A.shape[0]
    best_u = u.copy()
    best_val = obj(best_u)

    if m <= 10:
        Q = np.diag([wv, ww])
        # Enumerate all 2^m constraint active subsets and all 9 box-face combinations
        # (interior, lo0, hi0) x (interior, lo1, hi1)
        box_modes = [
            (None, None),
            (lo[0], None),
            (hi[0], None),
            (None, lo[1]),
            (None, hi[1]),
            (lo[0], lo[1]),
            (lo[0], hi[1]),
            (hi[0], lo[1]),
            (hi[0], hi[1]),
        ]
        for mask in range(1 << m):
            act = [i for i in range(m) if (mask >> i) & 1]
            H = Q.copy()
            rhs = Q @ un
            for i in act:
                ai = A[i]
                pi = p_delta[i]
                H += pi * np.outer(ai, ai)
                rhs += pi * b[i] * ai
            for f0, f1 in box_modes:
                cand = np.zeros(2, dtype=np.float64)
                if f0 is None and f1 is None:
                    cand = np.linalg.solve(H, rhs)
                elif f0 is not None and f1 is None:
                    cand[0] = f0
                    cand[1] = (rhs[1] - H[1, 0] * f0) / H[1, 1]
                elif f0 is None and f1 is not None:
                    cand[1] = f1
                    cand[0] = (rhs[0] - H[0, 1] * f1) / H[0, 0]
                else:
                    cand[0] = f0
                    cand[1] = f1
                if (
                    lo[0] - 1e-12 <= cand[0] <= hi[0] + 1e-12
                    and lo[1] - 1e-12 <= cand[1] <= hi[1] + 1e-12
                ):
                    cand = np.clip(cand, lo, hi)
                    val = obj(cand)
                    if val < best_val:
                        best_val = val
                        best_u = cand.copy()
        return best_u

    # For larger M, run dense multistart Nelder-Mead-free projected L-BFGS / exact 1D golden-section sweeps
    u = best_u.copy()
    for _ in range(400):
        prev = u.copy()
        for coord in (0, 1):
            # Exact 1D piecewise-quadratic minimum along coordinate `coord`
            other = 1 - coord
            u_other = u[other]
            # Breakpoints where a_m0*u0 + a_m1*u1 = b_m
            bps = [lo[coord], hi[coord]]
            for i in range(m):
                ac = A[i, coord]
                if abs(ac) > 1e-14:
                    bp = (b[i] - A[i, other] * u_other) / ac
                    if lo[coord] < bp < hi[coord]:
                        bps.append(bp)
            bps.sort()
            best_c = u[coord]
            best_v = obj(u)
            for k in range(len(bps) - 1):
                mid = 0.5 * (bps[k] + bps[k + 1])
                if bps[k + 1] - bps[k] < 1e-14:
                    continue
                u_mid = u.copy()
                u_mid[coord] = mid
                act = (A @ u_mid - b) > 0.0
                qc = wv if coord == 0 else ww
                num = qc * un[coord]
                den = qc
                if np.any(act):
                    ac = A[act, coord]
                    ao = A[act, other]
                    pa = p_delta[act]
                    num += float(np.dot(pa * (b[act] - ao * u_other), ac))
                    den += float(np.dot(pa * ac, ac))
                x_star = min(bps[k + 1], max(bps[k], num / den))
                u_try = u.copy()
                u_try[coord] = x_star
                v_try = obj(u_try)
                if v_try < best_v:
                    best_v = v_try
                    best_c = x_star
            u[coord] = best_c
        if np.linalg.norm(u - prev) < 1e-14:
            break
    return u


def _has_zero_slack_feasible_control(
    A: np.ndarray, b: np.ndarray, u_lo: np.ndarray, u_hi: np.ndarray
) -> Tuple[bool, np.ndarray]:
    """
    Exact 2D half-plane intersection check for whether {u in [u_lo, u_hi] : A u <= b} is non-empty.
    Tests box corners, box-edge/line intersections, and pairwise constraint intersections.
    """
    # Append 4 box inequalities: -u0 <= -lo0, u0 <= hi0, -u1 <= -lo1, u1 <= hi1
    A_box = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]], dtype=np.float64)
    b_box = np.array([-u_lo[0], u_hi[0], -u_lo[1], u_hi[1]], dtype=np.float64)
    A_all = np.vstack([A, A_box]) if A.shape[0] > 0 else A_box
    b_all = np.concatenate([b, b_box]) if A.shape[0] > 0 else b_box
    n = A_all.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            det = A_all[i, 0] * A_all[j, 1] - A_all[i, 1] * A_all[j, 0]
            if abs(det) < 1e-14:
                continue
            u0 = (b_all[i] * A_all[j, 1] - A_all[i, 1] * b_all[j]) / det
            u1 = (A_all[i, 0] * b_all[j] - b_all[i] * A_all[j, 0]) / det
            cand = np.array([u0, u1], dtype=np.float64)
            if np.all(A_all @ cand - b_all <= 1e-9):
                return True, np.clip(cand, u_lo, u_hi)
    return False, np.zeros(2, dtype=np.float64)


def test_pure_numpy_no_forbidden_imports():
    """Verify cbf_safety_shield.py imports ONLY Python stdlib + NumPy (no rclpy/cvxpy/osqp/scipy/torch)."""
    mod_path = os.path.join(PKG_SRC_DIR, "disaster_recovery", "cbf_safety_shield.py")
    with open(mod_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=mod_path)

    forbidden = {"rclpy", "cvxpy", "osqp", "scipy", "torch", "daqp", "cv2", "onnxruntime"}
    found_imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found_imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            found_imports.add(node.module.split(".")[0])

    assert found_imports.isdisjoint(forbidden), f"Forbidden imports detected: {found_imports & forbidden}"
    assert "numpy" in found_imports


def test_model_sdf_2circle_coverage_and_zero_interior_intervention():
    """
    Verify model.sdf TurtleBot3 Burger 2-Circle Covering (l_f=0.020, l_r=0.060, r_circle=0.098,
    R_safe,static=0.133) encloses 100% of chassis/wheel/lidar/caster collision geometries with
    0.000 m rear protrusion, and produces exact zero intervention inside the safe set.
    """
    assert math.isclose(DEFAULT_L_F, 0.020, abs_tol=1e-9)
    assert math.isclose(DEFAULT_L_R, 0.060, abs_tol=1e-9)
    assert math.isclose(DEFAULT_R_CIRCLE, 0.098, abs_tol=1e-9)
    assert math.isclose(DEFAULT_R_SAFE_STATIC, 0.133, abs_tol=1e-9)

    pf = np.array([DEFAULT_L_F, 0.0])
    pr = np.array([-DEFAULT_L_R, 0.0])

    # Dense boundary sampling of all model.sdf collision primitives in base_footprint frame:
    # 1) base_collision box: x in [-0.102, +0.038], y in [-0.070, +0.070]
    xs_box = np.linspace(-0.102, 0.038, 141)
    ys_box = np.linspace(-0.070, 0.070, 141)
    box_pts = np.vstack(
        [
            np.column_stack([xs_box, np.full_like(xs_box, 0.070)]),
            np.column_stack([xs_box, np.full_like(xs_box, -0.070)]),
            np.column_stack([np.full_like(ys_box, 0.038), ys_box]),
            np.column_stack([np.full_like(ys_box, -0.102), ys_box]),
        ]
    )
    # 2) wheel_left / wheel_right cylinders: x in [-0.033, +0.033], y in [+/-0.071, +/-0.089]
    xs_wh = np.linspace(-0.033, 0.033, 67)
    ys_wh = np.linspace(0.071, 0.089, 19)
    wheel_pts = np.vstack(
        [
            np.column_stack([xs_wh, np.full_like(xs_wh, 0.089)]),
            np.column_stack([xs_wh, np.full_like(xs_wh, -0.089)]),
            np.column_stack([np.full_like(ys_wh, 0.033), ys_wh]),
            np.column_stack([np.full_like(ys_wh, -0.033), ys_wh]),
            np.column_stack([np.full_like(ys_wh, 0.033), -ys_wh]),
            np.column_stack([np.full_like(ys_wh, -0.033), -ys_wh]),
        ]
    )
    # 3) lidar_sensor_collision: center (-0.020, 0), radius 0.0508
    ang = np.linspace(0.0, 2.0 * math.pi, 180, endpoint=False)
    lidar_pts = np.column_stack([-0.020 + 0.0508 * np.cos(ang), 0.0508 * np.sin(ang)])
    # 4) caster_back_link: center (-0.081, 0), radius 0.005
    caster_pts = np.column_stack([-0.081 + 0.005 * np.cos(ang), 0.005 * np.sin(ang)])

    all_pts = np.vstack([box_pts, wheel_pts, lidar_pts, caster_pts])
    d_f = np.linalg.norm(all_pts - pf[None, :], axis=1)
    d_r = np.linalg.norm(all_pts - pr[None, :], axis=1)
    max_min_dist = float(np.max(np.minimum(d_f, d_r)))

    # Worst boundary point is wheel outer midpoint (-0.020, +/-0.089) at sqrt(0.040^2 + 0.089^2) = 0.097576 m
    assert max_min_dist <= 0.09760 < DEFAULT_R_CIRCLE

    # Safe-set interior: open environment (nearest wall 1.1 m away) => exact zero intervention
    shield = DiscreteSoftCBFShield()
    scan = _make_open_scan(360, fill=1.10)
    res = shield.filter(v_nom=0.18, w_nom=0.25, scan_ranges=scan)
    assert res.pdas_iters == 0
    assert res.max_slack == 0.0
    assert res.intervened is False
    assert math.hypot(res.v - 0.18, res.w - 0.25) == 0.0


def test_base_scan_to_base_footprint_static_tf():
    """
    Verify /scan coordinates are transformed via base_scan -> base_footprint static TF
    (x_b = x_s - 0.032, y_b = y_s) BEFORE 2-Circle distance & CBF constraint computation.
    """
    assert math.isclose(DEFAULT_LIDAR_OFFSET_X, -0.032, abs_tol=1e-9)

    # 1. Direct coordinate transform check for front (0 deg), left (90 deg), rear (180 deg) rays
    scan = _make_open_scan(360, fill=3.5)
    scan[0] = 0.500    # base_scan: (+0.500, 0.0) -> base_footprint: (+0.468, 0.0)
    scan[90] = 0.400   # base_scan: (0.0, +0.400) -> base_footprint: (-0.032, +0.400)
    scan[180] = 0.300  # base_scan: (-0.300, 0.0) -> base_footprint: (-0.332, 0.0)

    pts_b = transform_scan_to_base_footprint(
        scan_ranges=scan, angle_min=0.0, angle_increment=2.0 * math.pi / 360.0
    )
    assert pts_b.shape == (3, 2)
    assert np.allclose(pts_b[0], [0.468, 0.0], atol=1e-6)
    assert np.allclose(pts_b[1], [-0.032, 0.400], atol=1e-6)
    assert np.allclose(pts_b[2], [-0.332, 0.0], atol=1e-6)

    # 2. Sensor-frame Cartesian input also receives exact x - 0.032 shift
    pts_s = np.array([[0.200, 0.0], [-0.200, 0.0]], dtype=np.float64)
    pts_b2 = transform_scan_to_base_footprint(scan_pts_sensor=pts_s)
    assert np.allclose(pts_b2, [[0.168, 0.0], [-0.232, 0.0]], atol=1e-6)

    # 3. Safety-critical CBF consequence:
    # Obstacle at x_s = +0.170 m in base_scan:
    #   - True base_footprint x_b = 0.170 - 0.032 = +0.138 m
    #   - Distance to front circle p_f (+0.020) = 0.118 m < R_safe,static (0.133 m) => h < 0!
    #   - Without TF shift it would erroneously look like d = 0.150 m > 0.133 m (h > 0).
    scan_front = _make_open_scan(360, fill=3.5)
    scan_front[0] = 0.170
    shield_tf = DiscreteSoftCBFShield(lidar_offset_x=-0.032)
    shield_no_tf = DiscreteSoftCBFShield(lidar_offset_x=0.0)

    res_tf = shield_tf.filter(v_nom=0.18, w_nom=0.0, scan_ranges=scan_front, update_stats=False)
    res_no_tf = shield_no_tf.filter(v_nom=0.18, w_nom=0.0, scan_ranges=scan_front, update_stats=False)

    expected_h_tf = (0.138 - 0.020) ** 2 - 0.133 ** 2
    expected_h_no_tf = (0.170 - 0.020) ** 2 - 0.133 ** 2
    assert math.isclose(res_tf.min_h_static, expected_h_tf, abs_tol=1e-6)
    assert math.isclose(res_no_tf.min_h_static, expected_h_no_tf, abs_tol=1e-6)
    assert res_tf.min_h_static < 0.0 < res_no_tf.min_h_static
    assert res_tf.v < 0.0  # Actively brakes/backs out instead of colliding
    assert res_tf.v < res_no_tf.v - 0.10


def test_asymmetric_q_compensates_front_circle_steering_authority():
    """
    Verify Q = diag(w_v, w_w) with w_v : w_w >= 10:1 (default 20:1) significantly boosts
    angular steering response |dw| on a front-diagonal obstacle compared to isotropic Q = I.
    """
    shield_asym = DiscreteSoftCBFShield()
    assert shield_asym.w_v / shield_asym.w_w >= 10.0

    shield_iso = DiscreteSoftCBFShield(w_v=1.0, w_w=1.0)

    # Single front-left diagonal obstacle (base_footprint: (+0.145, +0.110)) grazing front circle p_f=(0.02, 0)
    # In base_scan: x_s = 0.145 + 0.032 = 0.177, y_s = +0.110
    pts_sensor = np.array([[0.177, 0.110]], dtype=np.float64)

    res_asym = shield_asym.filter(v_nom=0.20, w_nom=0.0, scan_pts_sensor=pts_sensor, update_stats=False)
    res_iso = shield_iso.filter(v_nom=0.20, w_nom=0.0, scan_pts_sensor=pts_sensor, update_stats=False)

    # Obstacle is on front-left (+y): shield must steer RIGHT (w < 0)
    assert res_asym.w < 0.0
    assert abs(res_asym.w) > 5.0 * abs(res_iso.w)


def test_pdas_solver_kappa_1e4_stress_gate():
    """
    MANDATORY GATE: Pure-NumPy 2x2 PDAS + Armijo 1D line search at condition number
    kappa = p_delta / min(w_v, w_w) = 10^4 (and 10^5) must achieve:
      - KKT residual <= 1e-6
      - Reference convex solution error ||u_pdas - u_ref||_2 <= 1e-5
    across interior, active-ridge, opposing-wall, box-edge, and box-corner stress cases.
    """
    rng = np.random.default_rng(20260522)
    u_lo = np.array([-0.10, -0.65], dtype=np.float64)
    u_hi = np.array([0.20, 0.65], dtype=np.float64)

    # Explicit hand-crafted high-kappa stress cases + 60 randomized kappa = 1e4 cases
    cases: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]] = [
        # Case 1: Sharp opposing walls at kappa = 1e4 (w_v=1.0, w_w=0.5 -> min=0.5, p=5000 => kappa=1e4)
        (
            "opposing_walls_k1e4",
            np.array([0.20, 0.30]),
            np.array([[0.35, 0.012], [0.35, -0.012], [-0.20, 0.04], [0.10, -0.05]]),
            np.array([0.010, 0.008, 0.005, 0.002]),
            np.array([5000.0, 5000.0, 5000.0, 5000.0]),
            1.0,
            0.5,
        ),
        # Case 2: Box-face binding + off-diagonal high-kappa ridge (w_v=1.0, w_w=0.05, p=5000 => kappa=1e5)
        (
            "box_face_ridge_k1e5",
            np.array([0.20, 0.65]),
            np.array([[0.50, 0.025], [0.48, -0.018], [-0.30, 0.045]]),
            np.array([-0.030, -0.020, -0.010]),
            np.array([5000.0, 5000.0, 2000.0]),
            1.0,
            0.05,
        ),
        # Case 3: Exact kappa = 1e4 with w_v = w_w = 1.0, p = 1e4, conflicting infeasible constraints
        (
            "conflicting_infeasible_k1e4",
            np.array([0.15, -0.20]),
            np.array([[1.0, 0.02], [-1.0, 0.02], [0.05, 0.10], [0.05, -0.10]]),
            np.array([-0.05, -0.04, -0.01, -0.01]),
            np.array([10000.0, 10000.0, 10000.0, 10000.0]),
            1.0,
            1.0,
        ),
    ]

    for idx in range(60):
        m = int(rng.integers(3, 10))
        u_nom = np.array([rng.uniform(-0.10, 0.20), rng.uniform(-0.65, 0.65)])
        # Realistic CBF row scales: a_v ~ O(0.1..1.0), a_w ~ O(0.005..0.06)
        a0 = rng.uniform(-1.0, 1.0, size=m)
        a1 = rng.uniform(-0.08, 0.08, size=m)
        A = np.column_stack([a0, a1])
        b = rng.uniform(-0.08, 0.05, size=m)
        w_v = 1.0
        w_w = float(rng.choice([0.05, 0.10, 0.50, 1.0]))
        p_scale = 1e4 * min(w_v, w_w)
        p_delta = rng.uniform(0.5 * p_scale, 1.0 * p_scale, size=m)
        cases.append((f"rand_k1e4_{idx}", u_nom, A, b, p_delta, w_v, w_w))

    max_kkt = 0.0
    max_err = 0.0
    max_iters = 0

    for name, u_nom, A, b, p_delta, w_v, w_w in cases:
        u_opt, iters, kkt_res, _, _, _ = solve_soft_cbf_qp_pdas(
            u_nom=u_nom,
            A=A,
            b=b,
            p_delta=p_delta,
            w_v=w_v,
            w_w=w_w,
            u_lo=u_lo,
            u_hi=u_hi,
            max_iters=25,
            kkt_tol=1e-10,
        )
        u_ref = _solve_reference_convex_2d(u_nom, A, b, p_delta, w_v, w_w, u_lo, u_hi)
        kkt_chk = compute_kkt_residual(u_opt, u_nom, A, b, p_delta, w_v, w_w, u_lo, u_hi)
        err = float(np.linalg.norm(u_opt - u_ref))

        max_kkt = max(max_kkt, kkt_res, kkt_chk)
        max_err = max(max_err, err)
        max_iters = max(max_iters, iters)

        assert kkt_res <= 1e-6, f"[{name}] KKT residual {kkt_res:.3e} > 1e-6"
        assert kkt_chk <= 1e-6, f"[{name}] KKT check {kkt_chk:.3e} > 1e-6"
        assert err <= 1e-5, f"[{name}] Ref error {err:.3e} > 1e-5 (pdas={u_opt}, ref={u_ref})"

    assert max_kkt <= 1e-6
    assert max_err <= 1e-5
    assert max_iters <= 20


def test_narrow_doorway_028m_feasible_zero_slack_and_eccentric_soft_recovery():
    """
    Verify 0.28 m narrow doorway (walls at y = +/-0.140 m):
      1. Centered phi = 0 deg passes at full speed with exact slack == 0 and zero intervention.
      2. For |phi| <= 10 deg (2-circle midpoint centered), strict zero-slack feasible control exists
         (A u <= b, u in U non-empty), and closed-loop traversal passes cleanly without collision.
      3. Under eccentric entry (y = +0.022 m, phi = +14 deg) where walls conflict, soft slack > 0
         prevents infeasibility/deadlock and steers smoothly back toward doorway centerline.
    """
    shield = DiscreteSoftCBFShield()

    # World doorway walls: y = +0.140 and y = -0.140 along x in [-0.30, +0.60]
    wall_xs = np.linspace(-0.30, 0.60, 19)
    world_wall_pts = np.vstack(
        [
            np.column_stack([wall_xs, np.full_like(wall_xs, 0.140)]),
            np.column_stack([wall_xs, np.full_like(wall_xs, -0.140)]),
        ]
    )

    def world_to_sensor_pts(rx: float, ry: float, rth: float) -> np.ndarray:
        c, s = math.cos(rth), math.sin(rth)
        dx = world_wall_pts[:, 0] - rx
        dy = world_wall_pts[:, 1] - ry
        xb = c * dx + s * dy
        yb = -s * dx + c * dy
        # Inverse static TF: x_s = x_b - lidar_offset_x = x_b + 0.032
        return np.column_stack([xb - DEFAULT_LIDAR_OFFSET_X, yb])

    # 1. Centered phi = 0 deg (and |phi| <= 5 deg): exact slack == 0 and zero intervention at doorway transit speed
    pts_s0 = world_to_sensor_pts(0.0, 0.0, 0.0)
    res0 = shield.filter(v_nom=0.15, w_nom=0.0, scan_pts_sensor=pts_s0, update_stats=False)
    assert res0.max_slack == 0.0
    assert res0.pdas_iters == 0
    assert res0.intervened is False
    assert math.isclose(res0.v, 0.15, abs_tol=1e-9)

    # 2. |phi| <= 10 deg with 2-circle midpoint centered at y = 0:
    #    midpoint is at (-0.020, 0) in body frame => axle sits at y = +0.020 * sin(phi)
    for phi_deg in (-10.0, -5.0, 0.0, 5.0, 10.0):
        phi = math.radians(phi_deg)
        rx = 0.0
        ry = 0.020 * math.sin(phi)
        pts_s = world_to_sensor_pts(rx, ry, phi)
        A, b, _, _, _, _, _, _ = shield.build_constraints(scan_pts_sensor=pts_s)
        feasible, u_feas = _has_zero_slack_feasible_control(A, b, shield._u_lo, shield._u_hi)
        assert feasible, f"Expected zero-slack feasible control in 0.28 m doorway at phi={phi_deg} deg"
        # Feeding feasible nominal yields exact zero slack
        res_f = shield.filter(
            v_nom=float(u_feas[0]),
            w_nom=float(u_feas[1]),
            scan_pts_sensor=pts_s,
            update_stats=False,
        )
        assert res_f.max_slack <= 1e-7, f"phi={phi_deg} deg feasible slack={res_f.max_slack}"

    # Closed-loop traversal starting at phi = +10 deg through the 0.28 m doorway
    rx, ry, rth = -0.15, 0.020 * math.sin(math.radians(10.0)), math.radians(10.0)
    dt = 0.10
    min_phys_dist = math.inf
    aligned_slacks: List[float] = []
    for _ in range(95):
        pts_s = world_to_sensor_pts(rx, ry, rth)
        # Pure-pursuit-like nominal steering toward doorway exit (0.8, 0.0)
        alpha_g = math.atan2(0.0 - ry, 0.80 - rx) - rth
        w_nom = max(-0.65, min(0.65, 1.8 * alpha_g))
        v_nom = 0.15
        res = shield.filter(v_nom=v_nom, w_nom=w_nom, dt=dt, scan_pts_sensor=pts_s, update_stats=False)
        rx += res.v * math.cos(rth) * dt
        ry += res.v * math.sin(rth) * dt
        rth += res.w * dt
        pf = np.array([rx + DEFAULT_L_F * math.cos(rth), ry + DEFAULT_L_F * math.sin(rth)])
        pr = np.array([rx - DEFAULT_L_R * math.cos(rth), ry - DEFAULT_L_R * math.sin(rth)])
        d_now = min(
            float(np.min(np.linalg.norm(world_wall_pts - pf[None, :], axis=1))),
            float(np.min(np.linalg.norm(world_wall_pts - pr[None, :], axis=1))),
        )
        min_phys_dist = min(min_phys_dist, d_now)
        if abs(rth) < math.radians(3.0) and abs(ry) < 0.005:
            aligned_slacks.append(res.max_slack)
        if rx > 0.55:
            break

    assert rx > 0.50, f"Robot failed to traverse 0.28 m doorway (stopped at rx={rx:.3f})"
    assert min_phys_dist > DEFAULT_R_CIRCLE, f"Physical collision in doorway: d_min={min_phys_dist:.4f} <= 0.098"
    assert aligned_slacks and min(aligned_slacks) <= 1e-4

    # 3. Eccentric entry (y = +0.022, phi = +14 deg): soft slack activates gracefully & steers back right (w < 0)
    pts_ecc = world_to_sensor_pts(0.0, 0.022, math.radians(14.0))
    res_ecc = shield.filter(v_nom=0.18, w_nom=0.15, scan_pts_sensor=pts_ecc, update_stats=False)
    assert math.isfinite(res_ecc.v) and math.isfinite(res_ecc.w)
    assert res_ecc.kkt_residual <= 1e-6
    assert res_ecc.max_slack > 0.0  # Soft relaxation prevents infeasibility deadlock
    assert res_ecc.w < 0.0  # Steers right back toward doorway centerline


def test_voxel_downsampling_010m_nearest_16_points():
    """
    Verify 0.10 m spatial voxel downsampling retains at most 16 nearest distinct voxels to
    the 2-Circle and preserves multiple pillars inside the same 45-deg sector (unlike 8-sector).
    """
    # Construct dense points: 2 distinct pillars inside the SAME +10..+25 deg sector,
    # plus 2 narrow doorway jambs and 40 distant wall points
    pts_list = []
    # Pillar A around (0.35, 0.08), dense cluster of 15 rays
    for dx in np.linspace(-0.02, 0.02, 5):
        for dy in np.linspace(-0.02, 0.02, 3):
            pts_list.append([0.35 + dx, 0.08 + dy])
    # Pillar B around (0.48, 0.22) in SAME 45-deg sector, dense cluster of 15 rays
    for dx in np.linspace(-0.02, 0.02, 5):
        for dy in np.linspace(-0.02, 0.02, 3):
            pts_list.append([0.48 + dx, 0.22 + dy])
    # Doorway left/right jambs at (0.20, +0.14) and (0.20, -0.14)
    pts_list.append([0.20, 0.14])
    pts_list.append([0.20, -0.14])
    # 30 background ring points at r = 0.95 m
    for ang in np.linspace(0.0, 2.0 * math.pi, 30, endpoint=False):
        pts_list.append([0.95 * math.cos(ang), 0.95 * math.sin(ang)])

    pts_body = np.asarray(pts_list, dtype=np.float64)
    ds = voxel_downsample_nearest_2circle(pts_body, voxel_size=0.10, max_points=16, max_dist_m=1.20)

    assert ds.shape[0] <= 16
    assert ds.shape[0] >= 4
    # Verify all returned points live in distinct 0.10 m voxels
    vox_keys = set(zip(np.floor(ds[:, 0] / 0.10).astype(int), np.floor(ds[:, 1] / 0.10).astype(int)))
    assert len(vox_keys) == ds.shape[0]

    # Verify BOTH doorway jambs AND BOTH same-sector pillars survived!
    has_jamb_l = np.any(np.hypot(ds[:, 0] - 0.20, ds[:, 1] - 0.14) < 0.02)
    has_jamb_r = np.any(np.hypot(ds[:, 0] - 0.20, ds[:, 1] + 0.14) < 0.02)
    has_pillar_a = np.any(np.hypot(ds[:, 0] - 0.35, ds[:, 1] - 0.08) < 0.04)
    has_pillar_b = np.any(np.hypot(ds[:, 0] - 0.48, ds[:, 1] - 0.22) < 0.04)
    assert has_jamb_l and has_jamb_r and has_pillar_a and has_pillar_b


def test_decay_conservatism_order_and_multirate_invariance():
    """
    Verify gamma_dyn = 3.0 < gamma_static = 6.0 <= gamma_gas = 8.0 s^-1 yields strict
    alpha_dyn < alpha_static <= alpha_gas at both 10 Hz and 20 Hz, exact semigroup
    invariance (1 - alpha(0.05))^2 == 1 - alpha(0.10), and earlier braking on dynamic obstacles.
    """
    shield = DiscreteSoftCBFShield()
    assert shield.gamma_dynamic < shield.gamma_static <= shield.gamma_gas

    for dt in (0.10, 0.05):
        a_dyn = compute_alpha(shield.gamma_dynamic, dt)
        a_stat = compute_alpha(shield.gamma_static, dt)
        a_gas = compute_alpha(shield.gamma_gas, dt)
        assert 0.0 < a_dyn < a_stat <= a_gas < 1.0

    # Exact continuous-time multi-rate semigroup consistency (20 Hz x 2 ticks == 10 Hz x 1 tick)
    for g in (shield.gamma_dynamic, shield.gamma_static, shield.gamma_gas):
        lhs = (1.0 - compute_alpha(g, 0.05)) ** 2
        rhs = 1.0 - compute_alpha(g, 0.10)
        assert math.isclose(lhs, rhs, abs_tol=1e-12)

    # Dynamic approaching obstacle vs static obstacle at identical center/radius -> stricter braking for dynamic
    # Obstacle at (0.55, 0.0), radius 0.18 m
    trk_approaching = TrackedObstacle(
        track_id=1,
        x=0.55,
        y=0.0,
        vx=-0.36,
        vy=0.0,
        radius=0.18,
        P_4x4=np.diag([0.001, 0.001, 0.01, 0.01]),
        last_update_stamp=1.0,
        age=5,
        hits=5,
        is_confirmed=True,
    )
    res_dyn = shield.filter(
        v_nom=0.20, w_nom=0.0, rx=0.0, ry=0.0, rth=0.0,
        dynamic_obstacles=[trk_approaching], update_stats=False,
    )
    res_ch1 = shield.filter(
        v_nom=0.20, w_nom=0.0, rx=0.0, ry=0.0, rth=0.0,
        ch1_obstacles=[(0.55, 0.0, 0.18)], update_stats=False,
    )
    res_gas = shield.filter(
        v_nom=0.20, w_nom=0.0, rx=0.0, ry=0.0, rth=0.0,
        ch2_hazards=[(0.55, 0.0, 0.18)], update_stats=False,
    )
    assert res_dyn.v < res_ch1.v <= res_gas.v


def test_mq2_ch2_inside_lifo_escape_exception_branch():
    """
    Verify onboard MQ-2 Ch2 inside (h < 0, robot right at gas center p_k - p_o ~ 0) triggers
    LIFO safe-waypoint escape without singular repulsion/runaway while strictly enforcing
    LiDAR obstacle CBF constraints during escape.
    """
    shield = DiscreteSoftCBFShield(enable_mq2_lifo_escape=True)

    # Robot at (2.0, 0.0, yaw=0.0) facing +x (deeper into gas), while LIFO stack has safe WP behind at (1.0, 0.0)
    lifo_wps = [(0.0, 0.0), (1.0, 0.0), (1.95, 0.0)]
    ch2_gas = [(2.0, 0.0, 0.55)]  # Onboard MQ-2 registered right at robot's current pose (h < 0!)

    # Case 1: Facing away from w_safe (1.0, 0.0) -> must NOT run forward; pivots toward w_safe
    res_away = shield.filter(
        v_nom=0.20, w_nom=0.0, rx=2.0, ry=0.0, rth=0.0,
        ch2_hazards=ch2_gas, lifo_waypoints=lifo_wps, update_stats=False,
    )
    assert res_away.mq2_escape_active is True
    assert res_away.mq2_escape_target == (1.0, 0.0)
    assert res_away.min_h_gas < 0.0
    assert abs(res_away.v) <= 1e-6  # No forward runaway deeper into gas!
    assert abs(res_away.w) >= 0.50  # Actively pivots toward w_safe

    # Case 2: Once facing toward w_safe (yaw = pi), drives monotonically out toward (1.0, 0.0)
    res_face = shield.filter(
        v_nom=0.0, w_nom=0.0, rx=2.0, ry=0.0, rth=math.pi,
        ch2_hazards=ch2_gas, lifo_waypoints=lifo_wps, update_stats=False,
    )
    assert res_face.mq2_escape_active is True
    assert res_face.mq2_escape_target == (1.0, 0.0)
    assert res_face.v >= 0.15  # Drives cleanly out toward safe WP

    # Case 3: LiDAR obstacle directly blocking the escape path is STILL strictly enforced!
    scan_blocked = _make_open_scan(360, fill=3.5)
    scan_blocked[0] = 0.175  # 0.143 m in base_footprint -> 0.123 m from front circle < 0.133 m!
    res_blocked = shield.filter(
        v_nom=0.0, w_nom=0.0, rx=2.0, ry=0.0, rth=math.pi,
        scan_ranges=scan_blocked, ch2_hazards=ch2_gas, lifo_waypoints=lifo_wps, update_stats=False,
    )
    assert res_blocked.mq2_escape_active is True
    assert res_blocked.v < 0.05  # LiDAR CBF halts/backs off despite MQ-2 escape override!
    assert res_blocked.num_constraints >= 2


def test_v_min_parameterization_and_chattering_instrumentation():
    """
    Verify v_min = -0.10 vs 0.0 parameterization strictly enforces v >= v_min and
    measures velocity sign-reversal (chattering) frequency via get_stats().
    """
    shield_neg = DiscreteSoftCBFShield(v_min=-0.10)
    shield_zero = DiscreteSoftCBFShield(v_min=0.0)

    # Sequence of alternating approaching dynamic obstacle & open corridor ticks
    for step in range(30):
        approaching = (step % 6) in (2, 3)
        if approaching:
            trk = TrackedObstacle(
                track_id=1, x=0.38, y=0.0, vx=-0.40, vy=0.0, radius=0.18,
                P_4x4=np.diag([0.001, 0.001, 0.01, 0.01]),
                last_update_stamp=step * 0.1, age=4, hits=4, is_confirmed=True,
            )
            dyn = [trk]
        else:
            dyn = None

        r_neg = shield_neg.filter(v_nom=0.18, w_nom=0.0, dynamic_obstacles=dyn)
        r_zero = shield_zero.filter(v_nom=0.18, w_nom=0.0, dynamic_obstacles=dyn)
        assert -0.10 - 1e-9 <= r_neg.v <= 0.20 + 1e-9
        assert 0.0 - 1e-9 <= r_zero.v <= 0.20 + 1e-9

    st_neg = shield_neg.get_stats()
    st_zero = shield_zero.get_stats()

    assert st_neg["v_min"] == -0.10
    assert st_zero["v_min"] == 0.0
    assert st_neg["reverse_steps"] > 0
    assert st_neg["v_sign_reversals"] > 0
    assert st_zero["reverse_steps"] == 0
    assert st_zero["v_sign_reversals"] == 0
    assert st_zero["v_stall_toggles"] > 0


def test_qp_latency_under_025ms():
    """Verify full Soft-CBF-QP pipeline (360-ray TF + 0.10 m voxel 16 pts + 2 dyn + PDAS) < 0.25 ms."""
    shield = DiscreteSoftCBFShield()
    scan = _make_open_scan(360, fill=0.65)
    # Tight corridor walls + close front-left obstacle forcing active PDAS iterations
    for deg in range(360):
        rad = math.radians(deg)
        c, s = math.cos(rad), math.sin(rad)
        r_lat = 0.22 / max(abs(s), 1e-4)
        scan[deg] = min(3.5, r_lat)
    scan[10] = 0.22
    scan[350] = 0.24

    dyns = [
        TrackedObstacle(
            track_id=1, x=0.55, y=0.12, vx=-0.32, vy=-0.10, radius=0.20,
            P_4x4=np.diag([0.002, 0.002, 0.02, 0.02]),
            last_update_stamp=1.0, age=6, hits=6, is_confirmed=True,
        ),
        TrackedObstacle(
            track_id=2, x=0.85, y=-0.20, vx=-0.25, vy=0.15, radius=0.18,
            P_4x4=np.diag([0.002, 0.002, 0.02, 0.02]),
            last_update_stamp=1.0, age=6, hits=6, is_confirmed=True,
        ),
    ]
    ch1 = [(0.70, 0.25, 0.18)]
    ch2 = [(0.90, -0.35, 0.45)]

    # Warm-up LUT & JIT/cache
    for _ in range(20):
        shield.filter(
            v_nom=0.20, w_nom=0.10, rx=0.0, ry=0.0, rth=0.0,
            scan_ranges=scan, dynamic_obstacles=dyns, ch1_obstacles=ch1, ch2_hazards=ch2,
            update_stats=False,
        )

    n_iters = 500
    t0 = time.perf_counter()
    for _ in range(n_iters):
        res = shield.filter(
            v_nom=0.20, w_nom=0.10, rx=0.0, ry=0.0, rth=0.0,
            scan_ranges=scan, dynamic_obstacles=dyns, ch1_obstacles=ch1, ch2_hazards=ch2,
            update_stats=False,
        )
    mean_ms = (time.perf_counter() - t0) * 1000.0 / n_iters

    assert res.num_constraints >= 32
    assert res.kkt_residual <= 1e-6
    assert mean_ms < 0.25, f"Mean Soft-CBF-QP latency {mean_ms:.4f} ms >= 0.25 ms budget"


def test_recovery_controller_cbf_flag_and_guard_gating():
    """
    Verify recovery_controller.py declares enable_cbf_safety_shield defaulting to False,
    wires _apply_cbf_safety_shield, and disables external heuristic guards only when True.
    """
    ctrl_path = os.path.join(PKG_SRC_DIR, "disaster_recovery", "recovery_controller.py")
    with open(ctrl_path, "r", encoding="utf-8") as f:
        src = f.read()

    assert "self.declare_parameter('enable_cbf_safety_shield', False)" in src
    assert "self.declare_parameter('cbf_v_min', -0.10)" in src
    assert "def _apply_cbf_safety_shield(" in src
    assert "use_legacy_guards = self.is_phase2b and (not self.enable_cbf_safety_shield)" in src
    assert "if self.cbf_shield is not None:" in src


def test_near_ray_latch_and_scan_pose_egomotion_compensation():
    """
    Verify (1) LDS-01 < 0.12 m -> inf near-ray latch prevents close obstacles from vanishing
    and (2) 5 Hz scan_pose ego-motion compensation shifts stale scan points into current body frame.
    """
    shield = DiscreteSoftCBFShield()
    scan_close = _make_open_scan(360, fill=3.5)
    scan_close[0] = 0.165  # close front hit (< 0.185 m) arms latch on beam 0
    r1 = shield.filter(v_nom=0.18, w_nom=0.0, scan_ranges=scan_close, update_stats=True)
    assert r1.v < 0.05

    # Next tick: robot grazed closer (< 0.12 m), so LDS-01 returns inf on beam 0
    scan_inf = _make_open_scan(360, fill=3.5)
    scan_inf[0] = math.inf
    r2 = shield.filter(v_nom=0.18, w_nom=0.0, scan_ranges=scan_inf, update_stats=True)
    # Latch holds beam 0 at 0.12 m (x_b = 0.088 m -> 0.068 m from p_f < 0.133 m), braking hard!
    assert r2.min_h_static < 0.0
    assert r2.v < 0.0

    # Ego-motion compensation: scan taken at (0.0, 0.0, 0.0) saw front obstacle at r = 0.200 m
    # (x_b = 0.168 m); robot moved forward +0.030 m to (0.030, 0.0, 0.0) before next scan.
    shield.reset()
    scan_stale = _make_open_scan(360, fill=3.5)
    scan_stale[0] = 0.200
    r_stale = shield.filter(
        v_nom=0.20, w_nom=0.0, rx=0.030, ry=0.0, rth=0.0,
        scan_ranges=scan_stale, scan_pose=(0.0, 0.0, 0.0), update_stats=False,
    )
    # Effective x_b = 0.168 - 0.030 = 0.138 m => d_f = 0.118 m < 0.133 m => h < 0
    expected_h = (0.138 - DEFAULT_L_F) ** 2 - DEFAULT_R_SAFE_STATIC ** 2
    assert math.isclose(r_stale.min_h_static, expected_h, abs_tol=1e-6)
    assert r_stale.v < 0.0


def test_ch2_outer_grazing_preserves_cbf_without_mq2_escape():
    """
    (a) Pre-cached external GAS_HAZARD outer padding grazing (hr=0.55 <= d=0.69 < r_safe=0.698)
    must NOT trigger mq2_escape and must retain active Ch2 CBF repulsion.
    """
    shield = DiscreteSoftCBFShield(
        v_min=-0.06,
        v_max=0.20,
        w_max=0.65,
        slip_margin_mps=0.015,
        enable_mq2_lifo_escape=True,
    )
    # Gas hazard centered at (2.80, 0.20), hr=0.55 => r_safe=0.698 m
    # Robot grazing north outer halo at (2.80, 0.89) -> d_center = 0.69 m (hr <= d < r_safe), heading -x (pi)
    ch2_ext_gas = [(2.80, 0.20, 0.55, True, False)]
    lifo_wps = [(0.0, 0.0), (1.5, 0.0), (2.7, 0.0)]  # chord-cutting WP across gas center
    res = shield.filter(
        v_nom=0.18,
        w_nom=0.20,  # nominal trying to turn south into gas
        rx=2.80,
        ry=0.89,
        rth=math.pi,
        ch2_hazards=ch2_ext_gas,
        lifo_waypoints=lifo_wps,
        update_stats=True,
    )
    assert res.min_h_gas < 0.0
    assert res.mq2_escape_active is False
    assert res.num_constraints >= 2  # Ch2 CBF rows remained active!
    assert shield.get_stats()["mq2_escape_steps"] == 0.0


def test_deep_interior_local_mq2_escape_and_no_chord_cut():
    """
    (b) Deep interior / local MQ-2 alarm triggers escape cleanly and _select_safe_lifo_waypoint
    never picks a LIFO waypoint whose robot->target segment cuts a chord through the gas center.
    """
    shield = DiscreteSoftCBFShield(
        v_min=-0.06,
        v_max=0.20,
        w_max=0.65,
        slip_margin_mps=0.015,
        enable_mq2_lifo_escape=True,
    )
    # Robot offset inside gas zone at (3.20, 0.40), gas at (2.90, 0.20, hr=0.55), d_center=0.3606 < 0.402
    # Newest LIFO WP (2.10, -0.10) is outside r_safe but cuts directly through gas center (chord)!
    # Earlier LIFO WP (3.85, 0.55) is safely outward without crossing gas interior.
    ch2_local = [(2.90, 0.20, 0.55, True, True)]
    lifo_with_safe = [(0.0, 0.0), (3.85, 0.55), (2.10, -0.10)]
    res = shield.filter(
        v_nom=0.18,
        w_nom=0.0,
        rx=3.20,
        ry=0.40,
        rth=0.0,
        ch2_hazards=ch2_local,
        lifo_waypoints=lifo_with_safe,
        update_stats=True,
    )
    assert res.mq2_escape_active is True
    assert res.mq2_escape_target == (3.85, 0.55), f"Must skip chord-cutting WP, got {res.mq2_escape_target}"
    assert shield.get_stats()["mq2_escape_steps"] == 1.0

    # When ONLY chord-cutting LIFO WPs exist, must fall back to radial outward escape (never chord!)
    lifo_chord_only = [(2.10, -0.10)]
    res_rad = shield.filter(
        v_nom=0.18,
        w_nom=0.0,
        rx=3.20,
        ry=0.40,
        rth=0.0,
        ch2_hazards=ch2_local,
        lifo_waypoints=lifo_chord_only,
        update_stats=False,
    )
    assert res_rad.mq2_escape_active is True
    assert res_rad.mq2_escape_target is not None
    # Radial outward from (2.90, 0.20) through (3.20, 0.40) must point in (+x, +y) quadrant
    assert res_rad.mq2_escape_target[0] > 3.20 and res_rad.mq2_escape_target[1] > 0.40


def test_person_no_mq2_escape_cache_group_separation_and_rear_sem_prox():
    """
    (c) Verify:
      1. SemanticObjectCache separates dedup by Ch2 class group (GAS_HAZARD, person, AUDIO_DISTRESS
         within dedup_radius_m=0.45m never overwrite each other).
      2. person & AUDIO_DISTRESS pass allow_mq2_escape=False so CBF never invokes mq2_escape on them.
      3. Rear survivor at (5.85, 0.45) behind robot returning west from (4.74, 0.21) yields sem_prox == 0.0,
         whereas forward hazard yields high sem_prox.
    """
    from disaster_recovery.recovery_controller import RecoveryController, SemanticObjectCache

    cache = SemanticObjectCache(dedup_radius_m=0.45, default_ch2_radius_m=0.55)
    cache.upsert_object("person", 3.00, 0.20, radius=0.45, channel=2, marker_id=11)
    cache.upsert_object("GAS_HAZARD", 3.10, 0.25, radius=0.55, channel=2, marker_id=10)
    cache.upsert_object("AUDIO_DISTRESS", 3.05, 0.22, radius=0.40, channel=2, marker_id=12)
    assert len(cache) == 3, "GAS_HAZARD, person, and AUDIO_DISTRESS within 0.45m must not overwrite each other"

    _, ch2_cbf = cache.get_cbf_channel_tuples()
    assert len(ch2_cbf) == 3
    gas_flags = [item[3] for item in ch2_cbf]
    assert gas_flags.count(True) == 1 and gas_flags.count(False) == 2

    # Even deep proximity to person/AUDIO_DISTRESS must NEVER trigger mq2_escape
    shield = DiscreteSoftCBFShield(v_min=-0.06, slip_margin_mps=0.015, enable_mq2_lifo_escape=True)
    non_gas_only = [item for item in ch2_cbf if not item[3]]
    res_p = shield.filter(
        v_nom=0.15,
        w_nom=0.0,
        rx=3.02,
        ry=0.20,
        rth=0.0,
        ch2_hazards=non_gas_only,
        lifo_waypoints=[(0.0, 0.0), (1.0, 0.0)],
        update_stats=True,
    )
    assert res_p.mq2_escape_active is False
    assert shield.get_stats()["mq2_escape_steps"] == 0.0
    assert res_p.num_constraints >= 4

    # Rear survivor sem_prox attenuation vs forward hazard
    ctrl = RecoveryController.__new__(RecoveryController)
    ctrl.current_x = 4.74
    ctrl.current_y = 0.21
    ctrl.current_theta = math.pi  # Facing west toward base camp (0, 0)
    rear_survivor = [(5.85, 0.45, 0.45)]
    prox_rear = ctrl._compute_semantic_proximity([], rear_survivor, target_xy=(4.20, 0.10))
    assert prox_rear < 0.05, f"Rear survivor must not saturate sem_prox, got {prox_rear:.4f}"

    # Forward gas hazard ahead in -x direction still triggers full sem_prox
    fwd_gas = [(3.80, 0.21, 0.55)]  # d=0.94, clearance=0.39 -> base_prox=1.0, cos=+1.0
    prox_fwd = ctrl._compute_semantic_proximity([], fwd_gas, target_xy=(3.50, 0.20))
    assert prox_fwd > 0.95, f"Forward hazard must retain full sem_prox, got {prox_fwd:.4f}"


def test_ultra_close_positive_ray_clamp_stalled_latch_and_runtime_slip_margin():
    """
    (d) Verify:
      1. Positive ultra-close rays (0 < r < 0.120m) are distinguished from 0.0/inf/nan dropouts
         and clamped to min_range=0.120m in both transform_scan_to_base_footprint and _apply_near_ray_latch.
      2. Prolonged obstacle-pressed stall (> 6 ticks of inf/0 dropouts at same pose) keeps barrier latched
         in CBF without vanishing, while moving away or seeing clear hit (>= 0.185m) releases it.
      3. Operates cleanly with production cbf_slip_margin_mps=0.015 and v_min=-0.06.
    """
    shield = DiscreteSoftCBFShield(
        v_min=-0.06,
        v_max=0.20,
        w_max=0.65,
        slip_margin_mps=0.015,
    )
    # Positive ultra-close 0.075 m straight ahead (even without prior arming)
    scan_ultra = _make_open_scan(360, fill=3.5)
    scan_ultra[0] = 0.075
    scan_ultra[90] = 0.0   # dropout -> must NOT create obstacle on left
    scan_ultra[270] = math.nan  # dropout -> must NOT create obstacle on right
    r_ultra = shield.filter(
        v_nom=0.18, w_nom=0.0, rx=1.0, ry=0.0, rth=0.0, scan_ranges=scan_ultra, update_stats=True
    )
    expected_h_val = (0.088 - DEFAULT_L_F) ** 2 - DEFAULT_R_SAFE_STATIC ** 2
    assert math.isclose(r_ultra.min_h_static, expected_h_val, abs_tol=1e-6)
    assert r_ultra.min_h_static < 0.0
    assert r_ultra.v < 0.0  # Actively brakes/backs off under cbf_slip_margin_mps=0.015

    # Prolonged wall-pressed stall: 12 consecutive ticks (> ttl=6) of inf dropout at SAME pose (1.0, 0.0, 0.0)
    scan_inf = _make_open_scan(360, fill=3.5)
    scan_inf[0] = math.inf
    for _ in range(12):
        r_stall = shield.filter(
            v_nom=0.18, w_nom=0.0, rx=1.0, ry=0.0, rth=0.0, scan_ranges=scan_inf, update_stats=True
        )
        assert r_stall.min_h_static < 0.0, "Barrier vanished during prolonged wall-pressed stall!"
        assert r_stall.v < 0.0

    # Once robot backs away >= 0.03 m, TTL decays over 6 ticks so no permanent ghost barrier remains
    for _ in range(7):
        r_away = shield.filter(
            v_nom=0.18, w_nom=0.0, rx=0.95, ry=0.0, rth=0.0, scan_ranges=scan_inf, update_stats=True
        )
    assert r_away.min_h_static > 0.0
    assert r_away.v > 0.15


def test_episode_reset_completeness_fresh_odom_scan_binding_and_status_telemetry():
    """
    Step 2.1 unit test:
      1. cb_reset_episode() calls stop_robot(), refreshes last_heartbeat_time, clears residual
         variables (previous_state, aruco_start_time, td3_avoid_step, _start_icp_diag_count,
         last_icp_stats), and sets odom_initialized=False so stale pre-reset raw_odom is NEVER
         bound as start_pose and start_scan_pts is NEVER left stuck as None.
      2. Fresh /odom + /scan arriving in either order (odom->scan or scan->odom) bind start_pose
         and start_scan_pts cleanly at (0, 0), and control_loop transitions INIT -> NORMAL_TELEOP
         without spurious HEARTBEAT_TIMEOUT while publishing mq2_escape_steps, DYN, and CROSS.
    """
    from collections import deque
    import types
    from disaster_recovery.recovery_controller import (
        CrossingLaneScheduler,
        RecoveryController,
        SemanticObjectCache,
        SystemState,
        Waypoint,
    )

    class _FakeTime:
        def __init__(self, ns: int):
            self.nanoseconds = int(ns)

        def __sub__(self, other):
            return _FakeTime(self.nanoseconds - other.nanoseconds)

    def _make_ctrl():
        c = RecoveryController.__new__(RecoveryController)
        c.state = SystemState.BACKTRACKING_HYBRID
        c.previous_state = SystemState.LOCAL_TD3_AVOIDANCE
        c.aruco_start_time = _FakeTime(123)
        c.td3_avoid_step = 42
        c._start_icp_diag_count = 145
        c.last_icp_stats = {"ok": False, "inlier_ratio": 0.10}
        c.odom_initialized = True
        c.heartbeat_received_once = True
        c.heartbeat_timeout = 3.0
        c.gas_threshold = 0.40
        c.gas_stale_timeout = 2.0
        c.gas_stale_triggers = False
        c.current_gas_level = 0.0
        c.last_gas_time = None
        c.raw_odom_x = 3.80
        c.raw_odom_y = 0.20
        c.raw_odom_theta = 1.57
        c.current_x = 3.80
        c.current_y = 0.20
        c.current_theta = 1.57
        c.corr_x = 0.12
        c.corr_y = -0.05
        c.corr_th = 0.03
        c.scan_match_count = 17
        c.log_dist_th = 0.30
        c.log_angle_th = math.radians(25.0)
        c.last_logged_x = 3.80
        c.last_logged_y = 0.20
        c.last_logged_theta = 1.57
        c.waypoint_stack = deque([Waypoint(0.0, 0.0, 0.0), Waypoint(3.80, 0.20, 1.57)])
        c.start_pose = c.waypoint_stack[0]
        c.start_scan_pts = np.ones((40, 2), dtype=np.float32)
        c.latest_scan_pts = np.ones((40, 2), dtype=np.float32)
        c.latest_raw_ranges = [2.0] * 360
        c.latest_scan_spike_ratio = 0.05
        c.latest_scan_pose = (3.80, 0.20, 1.57)
        c.latest_scan_stamp_sec = 100.0
        c.latest_scan_angle_min = 0.0
        c.latest_scan_angle_increment = 2.0 * math.pi / 360.0
        c.latest_dynamic_tracks = []
        c.latest_dynamic_ch3 = None
        c.semantic_cache = SemanticObjectCache()
        c.semantic_cache.upsert_object("chair", 2.0, 0.1, channel=1)
        c.local_planner = None
        c.context_rl_policy = None
        c.dynamic_tracker = None
        c.cbf_shield = DiscreteSoftCBFShield(v_min=-0.06, slip_margin_mps=0.015)
        c.cbf_shield.mq2_escape_steps = 7
        c.crossing_scheduler = CrossingLaneScheduler()
        c.recovery_strategy = "proposed_rl"
        c.enable_lidar_despike = False
        c.smoke_lidar_noise_prob = 0.0
        c._scan_tick_seq = 0
        c.lidar_fov = math.radians(20.0)
        c.lidar_min_dist = 0.12
        c.lidar_danger_dist = 0.25
        c.td3_clearance_dist = 0.42
        c.enable_lidar_slip_residual = False
        c._prev_slip_scan_pts = None
        c._prev_slip_scan_stamp = None
        c.enable_teleop_rl_warmup = False
        c.clear_stack_on_link_restore = False
        c.reset_corr_on_episode = True
        c.docking_dist_th = 0.50
        c.last_cmd_v = 0.18
        c.last_cmd_w = 0.40
        c.actual_v = 0.18
        c.actual_w = 0.40
        c.last_heartbeat_time = _FakeTime(int(10.0 * 1e9))
        c._clock_now_ns = int(150.0 * 1e9)  # 140 s stale heartbeat before reset!
        c.get_clock = lambda: types.SimpleNamespace(now=lambda: _FakeTime(c._clock_now_ns))
        c.stop_calls = 0
        orig_stop = c.stop_robot

        def _spy_stop():
            c.stop_calls += 1
            orig_stop()

        c.stop_robot = _spy_stop
        c.statuses = []
        c.pub_system_status = types.SimpleNamespace(publish=lambda m: c.statuses.append(m.data))
        c.pub_cmd_vel = types.SimpleNamespace(publish=lambda m: None)
        c.get_logger = lambda: types.SimpleNamespace(
            info=lambda *a, **k: None, warn=lambda *a, **k: None, error=lambda *a, **k: None
        )
        return c

    odom_origin = types.SimpleNamespace(
        pose=types.SimpleNamespace(
            pose=types.SimpleNamespace(
                position=types.SimpleNamespace(x=0.0, y=0.0),
                orientation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        ),
        twist=types.SimpleNamespace(
            twist=types.SimpleNamespace(
                linear=types.SimpleNamespace(x=0.0),
                angular=types.SimpleNamespace(z=0.0),
            )
        ),
    )
    scan_origin = types.SimpleNamespace(
        ranges=[1.8] * 360,
        angle_min=0.0,
        angle_increment=2.0 * math.pi / 360.0,
        header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=150, nanosec=200000000)),
    )

    # Order 1: reset -> fresh /odom -> fresh /scan -> control_loop
    c1 = _make_ctrl()
    c1.cb_reset_episode(types.SimpleNamespace(data=True))
    assert c1.stop_calls == 1
    assert c1.last_cmd_v == 0.0 and c1.last_cmd_w == 0.0
    assert c1.last_heartbeat_time.nanoseconds == int(150.0 * 1e9)
    assert c1.previous_state == SystemState.INIT
    assert c1.aruco_start_time is None
    assert c1.td3_avoid_step == 0
    assert c1._start_icp_diag_count == 0
    assert c1.last_icp_stats is None
    assert c1.odom_initialized is False
    assert c1.start_pose is None and c1.start_scan_pts is None
    assert len(c1.waypoint_stack) == 0
    assert c1.state == SystemState.INIT
    assert c1.cbf_shield.mq2_escape_steps == 0

    c1.cb_odometry(odom_origin)
    assert c1.odom_initialized is True
    assert c1.start_pose is not None
    assert math.hypot(c1.start_pose.x, c1.start_pose.y) < 1e-6
    assert len(c1.waypoint_stack) == 1
    c1.cb_scan(scan_origin)
    assert c1.start_scan_pts is not None and c1.start_pose.scan_pts is not None
    c1._clock_now_ns = int(150.2 * 1e9)
    c1.control_loop()  # INIT -> NORMAL_TELEOP
    assert c1.state == SystemState.NORMAL_TELEOP
    c1._clock_now_ns = int(150.3 * 1e9)
    c1.control_loop()  # Must NOT trigger stale 140s HEARTBEAT_TIMEOUT
    assert c1.state == SystemState.NORMAL_TELEOP
    assert "|DYN:0|CROSS:INACTIVE" in c1.statuses[-1]
    cbf_part = [p for p in c1.statuses[-1].split("|") if p.startswith("CBF:")][0]
    assert len(cbf_part.split(":", 1)[1].split(",")) == 9

    # Order 2: reset -> fresh /scan -> fresh /odom -> start_pose & start_scan_pts both bound!
    c2 = _make_ctrl()
    c2.cb_reset_episode(types.SimpleNamespace(data=True))
    c2.cb_scan(scan_origin)
    assert c2.start_pose is None and c2.start_scan_pts is None
    c2.cb_odometry(odom_origin)
    assert c2.odom_initialized is True
    assert c2.start_pose is not None and math.hypot(c2.start_pose.x, c2.start_pose.y) < 1e-6
    assert c2.start_scan_pts is not None and c2.start_pose.scan_pts is not None
    assert len(c2.waypoint_stack) == 1


def test_semantic_world_min_wall_geom_exact_openings_pillars_and_survivor():
    """Verify Step 2.2 exact geometric wall distance in disaster_semantic_rescue_world.world."""
    scripts_host = os.path.abspath(os.path.join(PKG_SRC_DIR, "..", "..", "scripts", "host"))
    if scripts_host not in sys.path:
        sys.path.insert(0, scripts_host)
    from run_phase3_semantic_suite import compute_semantic_world_min_wall_geom

    # Partition 1 doorway exact opening: p1_n y>=+0.945, p1_s y<=-0.745 at x=1.45 (thick 0.16)
    assert math.isclose(compute_semantic_world_min_wall_geom([(1.45, 0.845)]), 0.100, abs_tol=1e-6)
    assert math.isclose(compute_semantic_world_min_wall_geom([(1.45, -0.645)]), 0.100, abs_tol=1e-6)
    # Partition 2 doorway exact opening: p2_n y>=+0.745, p2_s y<=-0.945 at x=4.35
    assert math.isclose(compute_semantic_world_min_wall_geom([(4.35, 0.645)]), 0.100, abs_tol=1e-6)
    assert math.isclose(compute_semantic_world_min_wall_geom([(4.35, -0.845)]), 0.100, abs_tol=1e-6)
    # Internal pillars pil_1(0.40,1.65,0.36), pil_2(3.00,-1.75,0.40), pil_3(6.60,-1.55,0.45)
    assert math.isclose(compute_semantic_world_min_wall_geom([(0.40, 1.37)]), 0.100, abs_tol=1e-6)
    assert math.isclose(compute_semantic_world_min_wall_geom([(3.00, -1.45)]), 0.100, abs_tol=1e-6)
    assert math.isclose(compute_semantic_world_min_wall_geom([(6.60, -1.225)]), 0.100, abs_tol=1e-6)
    # Standing survivor_person(5.85, 0.45, r=0.22)
    assert math.isclose(compute_semantic_world_min_wall_geom([(5.53, 0.45)]), 0.100, abs_tol=1e-6)



