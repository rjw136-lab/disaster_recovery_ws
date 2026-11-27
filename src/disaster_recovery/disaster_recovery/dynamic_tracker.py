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
CH3_MAX_R_INFL = 0.28


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
_EGO_FLAT_X = _EGO_GRID_X.ravel()
_EGO_FLAT_Y = _EGO_GRID_Y.ravel()


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
    birth_x: float = 0.0
    birth_y: float = 0.0
    max_speed_seen: float = 0.0
    max_disp_seen: float = 0.0
    pos_history: Tuple[Tuple[float, float, float], ...] = ()
    imm_valid: bool = False
    imm_u_hat: Tuple[float, float] = (1.0, 0.0)
    imm_p_c: Tuple[float, float] = (0.0, 0.0)
    imm_amp: float = 0.0
    imm_omega: float = 0.0
    imm_phi: float = 0.0
    imm_t_ref: float = 0.0
    imm_mu_cv: float = 1.0
    imm_mu_harm: float = 0.0
    imm_sigma_h0: float = 0.03
    imm_sigma_omega: float = 0.25
    imm_harm_rmse: float = 0.0
    imm_cv_rmse: float = 0.0

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)


@dataclass
class TrajectoryPrediction:
    """IMM trajectory rollout result over horizon grid `taus`."""

    taus: np.ndarray
    centers: np.ndarray
    sigmas: np.ndarray
    u_hat: Tuple[float, float]
    p_c: Tuple[float, float]
    amplitude: float
    omega: float
    phi: float
    sweep_start: Tuple[float, float]
    sweep_end: Tuple[float, float]
    mu_cv: float
    mu_harm: float
    is_harmonic: bool


_IMM_W_COARSE = np.linspace(0.25, 2.20, 16, dtype=np.float64)
_IMM_FINE_OFFSETS = np.linspace(-1.0, 1.0, 7, dtype=np.float64)
_IMM_WARM_OFFSETS = np.linspace(-0.08, 0.08, 7, dtype=np.float64)


def _solve_sym3x3_batch(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    e: np.ndarray,
    f: np.ndarray,
    r0: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form vectorized solver for symmetric 3x3 linear systems G x = r."""
    c00 = d * f - e * e
    c01 = c * e - b * f
    c02 = b * e - c * d
    c11 = a * f - c * c
    c12 = b * c - a * e
    c22 = a * d - b * b
    det = a * c00 + b * c01 + c * c02
    safe_det = np.where(np.abs(det) > 1e-12, det, 1.0)
    inv_det = 1.0 / safe_det
    x0 = (c00 * r0 + c01 * r1 + c02 * r2) * inv_det
    x1 = (c01 * r0 + c11 * r1 + c12 * r2) * inv_det
    x2 = (c02 * r0 + c12 * r1 + c22 * r2) * inv_det
    return x0, x1, x2, det


def _eval_harmonic_grid_1d(
    dm: np.ndarray,
    s: np.ndarray,
    w_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate variable-projection 1D harmonic OLS `s(dm) = s0 + a cos(w dm) + b sin(w dm)`
    across all candidate frequencies in `w_grid` simultaneously.
    Returns (s0, a_c, b_s, sse, det).
    """
    n_f = float(dm.shape[0])
    ang = w_grid[:, None] * dm[None, :]
    cs = np.cos(ang)
    sn = np.sin(ang)

    g00 = np.full(w_grid.shape[0], n_f + 1e-9, dtype=np.float64)
    g01 = cs.sum(axis=1)
    g02 = sn.sum(axis=1)
    g11 = (cs * cs).sum(axis=1) + 1e-6
    g12 = (cs * sn).sum(axis=1)
    g22 = (sn * sn).sum(axis=1) + 1e-6

    r0 = np.full(w_grid.shape[0], float(s.sum()), dtype=np.float64)
    r1 = cs @ s
    r2 = sn @ s

    s0, a_c, b_s, det = _solve_sym3x3_batch(g00, g01, g02, g11, g12, g22, r0, r1, r2)
    pred = s0[:, None] + a_c[:, None] * cs + b_s[:, None] * sn
    err = s[None, :] - pred
    sse = (err * err).sum(axis=1)
    return s0, a_c, b_s, sse, det


def fit_pca_and_harmonic_imm(
    pos_history: Sequence[Tuple[float, float, float]],
    fallback_vx: float = 0.0,
    fallback_vy: float = 0.0,
    meas_noise_std: float = 0.030,
    min_samples: int = 10,
    min_span_sec: float = 1.8,
    w_min: float = 0.25,
    w_max: float = 2.20,
    max_amplitude_m: float = 2.20,
    prev_omega: float = 0.0,
) -> Dict[str, object]:
    """
    Estimate principal 1D oscillation axis `u_hat` via 2x2 PCA on recent position history
    and fit parallel (a) 1D CV and (b) 1D Harmonic models with BIC-adjusted residual
    likelihood IMM weights (`mu_cv`, `mu_harm`).
    """
    n = len(pos_history)
    if n == 0:
        spd = math.hypot(fallback_vx, fallback_vy)
        u_hat = (fallback_vx / spd, fallback_vy / spd) if spd > 1e-6 else (1.0, 0.0)
        return {
            "valid": False,
            "u_hat": u_hat,
            "p_c": (0.0, 0.0),
            "amp": 0.0,
            "omega": 0.0,
            "phi": 0.0,
            "t_ref": 0.0,
            "mu_cv": 1.0,
            "mu_harm": 0.0,
            "sigma_h0": max(0.03, float(meas_noise_std)),
            "sigma_omega": 0.35,
            "harm_rmse": 0.0,
            "cv_rmse": 0.0,
        }

    arr = np.asarray(pos_history, dtype=np.float64)
    ts = arr[:, 0]
    xs = arr[:, 1]
    ys = arr[:, 2]
    t_ref = float(ts[-1])
    t_span = float(ts[-1] - ts[0])

    mx = float(xs.mean())
    my = float(ys.mean())
    dx = xs - mx
    dy = ys - my

    # 2x2 PCA for principal reciprocation axis u_hat
    inv_n = 1.0 / float(n)
    cxx = float((dx * dx).sum()) * inv_n
    cyy = float((dy * dy).sum()) * inv_n
    cxy = float((dx * dy).sum()) * inv_n
    tr = cxx + cyy
    disc = math.sqrt(max(0.0, (cxx - cyy) ** 2 + 4.0 * cxy * cxy))
    lam1 = 0.5 * (tr + disc)
    lam2 = max(0.0, 0.5 * (tr - disc))

    if lam1 > 1e-6:
        if abs(cxy) > 1e-12:
            vx1, vy1 = lam1 - cyy, cxy
            vx2, vy2 = cxy, lam1 - cxx
            if (vx1 * vx1 + vy1 * vy1) >= (vx2 * vx2 + vy2 * vy2):
                ux, uy = vx1, vy1
            else:
                ux, uy = vx2, vy2
            un = math.hypot(ux, uy)
            ux, uy = ux / un, uy / un
        else:
            ux, uy = (1.0, 0.0) if cxx >= cyy else (0.0, 1.0)
    else:
        spd = math.hypot(fallback_vx, fallback_vy)
        ux, uy = (fallback_vx / spd, fallback_vy / spd) if spd > 1e-6 else (1.0, 0.0)

    # Canonical sign for deterministic u_hat
    if ux < -1e-6 or (abs(ux) <= 1e-6 and uy < 0.0):
        ux, uy = -ux, -uy

    s = dx * ux + dy * uy
    s_min = float(s.min())
    s_max = float(s.max())
    s_mid_obs = 0.5 * (s_min + s_max)
    r_obs = 0.5 * (s_max - s_min)
    p_c_obs = (mx + s_mid_obs * ux, my + s_mid_obs * uy)
    sigma_perp = math.sqrt(lam2)

    if n < min_samples or t_span < min_span_sec or lam1 < 1e-4:
        return {
            "valid": False,
            "u_hat": (float(ux), float(uy)),
            "p_c": p_c_obs,
            "amp": float(r_obs),
            "omega": 0.0,
            "phi": 0.0,
            "t_ref": t_ref,
            "mu_cv": 1.0,
            "mu_harm": 0.0,
            "sigma_h0": max(0.03, float(meas_noise_std), sigma_perp),
            "sigma_omega": 0.35,
            "harm_rmse": 0.0,
            "cv_rmse": 0.0,
        }

    # Center time at midpoint for orthogonal cos/sin conditioning
    t_mid = 0.5 * (float(ts[0]) + t_ref)
    dm = ts - t_mid

    # (a) 1D CV model along u_hat: s(dm) = c0 + c1 * dm
    dm_mean = float(dm.mean())
    s_mean = float(s.mean())
    dm_c = dm - dm_mean
    s_c = s - s_mean
    denom_cv = float((dm_c * dm_c).sum())
    c1 = float((dm_c * s_c).sum()) / max(denom_cv, 1e-12)
    c0 = s_mean - c1 * dm_mean
    err_cv = s - (c0 + c1 * dm)
    sse_cv = float((err_cv * err_cv).sum())
    cv_rmse = math.sqrt(sse_cv * inv_n)

    # (b) 1D Harmonic model via warm-started / coarse-to-fine variable projection
    def _pick_best_with_curv(
        w_arr: np.ndarray,
    ) -> Optional[Tuple[float, float, float, float, float, float]]:
        s0_v, a_v, b_v, sse_v, det_v = _eval_harmonic_grid_1d(dm, s, w_arr)
        amp_v = np.hypot(a_v, b_v)
        valid_cand = (
            (np.abs(det_v) > 1e-9)
            & (amp_v >= 0.06)
            & (amp_v <= max_amplitude_m)
            & (np.abs(s0_v) <= max_amplitude_m)
            & ((w_arr * t_span) >= 0.85)
        )
        if not np.any(valid_cand):
            return None
        masked_sse = np.where(valid_cand, sse_v, np.inf)
        bi = int(np.argmin(masked_sse))
        if not math.isfinite(float(masked_sse[bi])):
            return None
        d2_curv = 0.0
        if 0 < bi < (len(w_arr) - 1):
            sl = float(w_arr[bi] - w_arr[bi - 1])
            sr = float(w_arr[bi + 1] - w_arr[bi])
            if sl > 1e-4 and sr > 1e-4:
                d2_curv = 2.0 * (
                    (float(sse_v[bi + 1]) - float(sse_v[bi])) / sr
                    - (float(sse_v[bi]) - float(sse_v[bi - 1])) / sl
                ) / (sl + sr)
        return (
            float(w_arr[bi]),
            float(s0_v[bi]),
            float(a_v[bi]),
            float(b_v[bi]),
            float(sse_v[bi]),
            float(d2_curv),
        )

    best = None
    # Warm-start single-pass local search when previous omega is already locked cleanly
    if w_min <= prev_omega <= w_max:
        w_warm = np.clip(prev_omega + _IMM_WARM_OFFSETS, w_min, w_max)
        cand_warm = _pick_best_with_curv(w_warm)
        if cand_warm is not None:
            warm_rmse = math.sqrt(max(0.0, cand_warm[4]) * inv_n)
            # Accept warm start if interior minimum and clean residual
            if warm_rmse <= 0.055 and abs(cand_warm[0] - prev_omega) < 0.068:
                best = cand_warm

    if best is None:
        best_c = _pick_best_with_curv(_IMM_W_COARSE)
        if best_c is None:
            return {
                "valid": False,
                "u_hat": (float(ux), float(uy)),
                "p_c": p_c_obs,
                "amp": float(r_obs),
                "omega": 0.0,
                "phi": 0.0,
                "t_ref": t_ref,
                "mu_cv": 1.0,
                "mu_harm": 0.0,
                "sigma_h0": max(0.03, float(meas_noise_std), sigma_perp),
                "sigma_omega": 0.35,
                "harm_rmse": cv_rmse,
                "cv_rmse": cv_rmse,
            }
        dw_coarse = float(_IMM_W_COARSE[1] - _IMM_W_COARSE[0])
        w_fine = np.clip(best_c[0] + _IMM_FINE_OFFSETS * dw_coarse, w_min, w_max)
        cand_f = _pick_best_with_curv(w_fine)
        best = cand_f if (cand_f is not None and cand_f[4] <= best_c[4]) else best_c

    w_opt, s0_opt, a_opt, b_opt, sse_harm, d2_sse = best
    amp_opt = math.hypot(a_opt, b_opt)
    phi_mid = math.atan2(a_opt, b_opt)
    phi_ref = math.atan2(
        math.sin(phi_mid + w_opt * (t_ref - t_mid)),
        math.cos(phi_mid + w_opt * (t_ref - t_mid)),
    )
    harm_rmse = math.sqrt(max(0.0, sse_harm) * inv_n)
    pc_x = mx + s0_opt * ux
    pc_y = my + s0_opt * uy

    # Estimate frequency uncertainty sigma_omega from SSE(omega) local curvature (Laplace / Fisher info)
    noise_var = max(harm_rmse * harm_rmse, float(meas_noise_std) ** 2, 4e-4)
    if d2_sse > 1e-6:
        sigma_omega = math.sqrt(2.0 * noise_var / d2_sse)
    else:
        sigma_omega = 0.35
    sigma_omega = max(0.015, min(0.45, sigma_omega))

    # Reject non-1D (e.g. circular) or poorly-fitting harmonic hypotheses
    anisotropy_ok = (lam2 / max(lam1, 1e-9)) < 0.28
    fit_ok = harm_rmse <= 0.11
    if anisotropy_ok and fit_ok:
        sig0_sq = max(0.032 * 0.032, float(meas_noise_std) ** 2)
        # BIC-penalized log-likelihood difference: Harmonic (4 params) vs CV (2 params)
        delta_ell = (sse_cv - sse_harm) / (2.0 * sig0_sq) - math.log(float(n))
        delta_ell_c = max(-20.0, min(20.0, delta_ell))
        mu_harm = 1.0 / (1.0 + math.exp(-delta_ell_c))
        mu_cv = 1.0 - mu_harm
    else:
        mu_harm = 0.0
        mu_cv = 1.0

    sigma_h0 = math.sqrt(max(harm_rmse * harm_rmse, float(meas_noise_std) ** 2) + lam2)
    return {
        "valid": True,
        "u_hat": (float(ux), float(uy)),
        "p_c": (float(pc_x), float(pc_y)),
        "amp": float(amp_opt),
        "omega": float(w_opt),
        "phi": float(phi_ref),
        "t_ref": t_ref,
        "mu_cv": float(mu_cv),
        "mu_harm": float(mu_harm),
        "sigma_h0": float(sigma_h0),
        "sigma_omega": float(sigma_omega),
        "harm_rmse": float(harm_rmse),
        "cv_rmse": float(cv_rmse),
    }


def predict_track_trajectory(
    track: TrackedObstacle,
    taus: Sequence[float],
    beta: float = 0.35,
    q_acc_base: float = 0.30,
    enable_harmonic: bool = True,
) -> TrajectoryPrediction:
    """
    Lightweight 2-model (CV + 1D Harmonic) IMM trajectory rollout for `tau in [0.0, 4.5+] s`.
    Vectorized pure-NumPy implementation (< 0.05 ms for 30+ horizons).
    """
    taus_arr = np.maximum(0.0, np.asarray(taus, dtype=np.float64))
    k_len = int(taus_arr.shape[0])

    # Extract basic kinematic state (duck-type safe for lightweight mocks)
    tx = float(getattr(track, "x", 0.0))
    ty = float(getattr(track, "y", 0.0))
    tvx = float(getattr(track, "vx", 0.0))
    tvy = float(getattr(track, "vy", 0.0))
    q_scale = float(getattr(track, "q_acc_scale", 1.0))
    P = getattr(track, "P_4x4", None)
    if P is None:
        P = np.diag([0.0025, 0.0025, 0.04, 0.04]).astype(np.float64)
    else:
        P = np.asarray(P, dtype=np.float64)

    # (a) Vectorized damped-CV state and covariance propagation
    b = max(1e-6, float(beta))
    u = b * taus_arr
    e1 = np.exp(-u)
    phi_cv = np.where(
        u < 1e-4,
        taus_arr * (1.0 - 0.5 * u + (u * u) / 6.0),
        (1.0 - e1) / b,
    )
    e2 = np.exp(-2.0 * u)
    q_pp = np.where(
        u < 1e-3,
        (taus_arr ** 3) / 3.0,
        (2.0 * u - 3.0 + 4.0 * e1 - e2) / (2.0 * (b ** 3)),
    )
    q_eff = max(0.0, float(q_acc_base) * q_scale)

    px_cv = tx + phi_cv * tvx
    py_cv = ty + phi_cv * tvy

    p00 = float(P[0, 0]) + 2.0 * phi_cv * float(P[0, 2]) + (phi_cv * phi_cv) * float(P[2, 2]) + q_eff * q_pp
    p11 = float(P[1, 1]) + 2.0 * phi_cv * float(P[1, 3]) + (phi_cv * phi_cv) * float(P[3, 3]) + q_eff * q_pp
    p01 = (
        float(P[0, 1])
        + phi_cv * (float(P[0, 3]) + float(P[1, 2]))
        + (phi_cv * phi_cv) * float(P[2, 3])
    )
    disc_cv = np.sqrt(np.maximum(0.0, (p00 - p11) ** 2 + 4.0 * p01 * p01))
    var_cv = np.maximum(0.0, 0.5 * (p00 + p11 + disc_cv))

    # (b) IMM harmonic state & sweep parameters
    imm_valid = bool(enable_harmonic) and bool(getattr(track, "imm_valid", False))
    mu_harm = float(getattr(track, "imm_mu_harm", 0.0)) if imm_valid else 0.0
    mu_harm = max(0.0, min(1.0, mu_harm))
    mu_cv = 1.0 - mu_harm

    raw_u = getattr(track, "imm_u_hat", None)
    if raw_u is not None and (imm_valid or len(getattr(track, "pos_history", ())) >= 3):
        ux, uy = float(raw_u[0]), float(raw_u[1])
        un = math.hypot(ux, uy)
        if un > 1e-6:
            ux, uy = ux / un, uy / un
        else:
            spd = math.hypot(tvx, tvy)
            ux, uy = (tvx / spd, tvy / spd) if spd > 1e-6 else (1.0, 0.0)
    else:
        spd = math.hypot(tvx, tvy)
        ux, uy = (tvx / spd, tvy / spd) if spd > 1e-6 else (1.0, 0.0)

    if imm_valid and mu_harm > 1e-4:
        pc_x, pc_y = float(track.imm_p_c[0]), float(track.imm_p_c[1])
        amp = float(track.imm_amp)
        omega = float(track.imm_omega)
        phi_ref = float(track.imm_phi)
        dt_anchor = max(0.0, float(getattr(track, "last_update_stamp", 0.0)) - float(track.imm_t_ref))
        dt_total = dt_anchor + taus_arr
        s_h = amp * np.sin(omega * dt_total + phi_ref)
        px_h = pc_x + ux * s_h
        py_h = pc_y + uy * s_h

        sig_h0 = max(0.025, float(getattr(track, "imm_sigma_h0", 0.03)))
        sig_w = max(0.015, float(getattr(track, "imm_sigma_omega", 0.20)))
        var_h = sig_h0 * sig_h0 + (amp * sig_w * dt_total) ** 2

        px_mix = mu_cv * px_cv + mu_harm * px_h
        py_mix = mu_cv * py_cv + mu_harm * py_h
        d2_cv = (px_cv - px_mix) ** 2 + (py_cv - py_mix) ** 2
        d2_h = (px_h - px_mix) ** 2 + (py_h - py_mix) ** 2
        var_mix = mu_cv * (var_cv + d2_cv) + mu_harm * (var_h + d2_h)
        sigmas = np.sqrt(np.maximum(0.0, var_mix))
    else:
        px_mix = px_cv
        py_mix = py_cv
        sigmas = np.sqrt(var_cv)
        omega = float(getattr(track, "imm_omega", 0.0)) if imm_valid else 0.0
        phi_ref = float(getattr(track, "imm_phi", 0.0)) if imm_valid else 0.0
        # Sweep fallback for CV-dominant tracks: union of observed span + short 1.8s CV lookahead
        hist = getattr(track, "pos_history", ())
        v_along = tvx * ux + tvy * uy
        proj_end = v_along * 1.8
        if len(hist) >= 2:
            s_vals = [(hx - tx) * ux + (hy - ty) * uy for (_, hx, hy) in hist]
            s_lo = min(min(s_vals), 0.0, proj_end)
            s_hi = max(max(s_vals), 0.0, proj_end)
        else:
            s_lo = min(0.0, proj_end)
            s_hi = max(0.0, proj_end)
        s_mid = 0.5 * (s_lo + s_hi)
        pc_x = tx + s_mid * ux
        pc_y = ty + s_mid * uy
        amp = 0.5 * (s_hi - s_lo)

    if imm_valid and mu_harm < 0.5:
        # When harmonic is not yet dominant, keep conservative sweep span combining observed + short CV
        hist = getattr(track, "pos_history", ())
        v_along = tvx * ux + tvy * uy
        proj_end = v_along * 1.8
        if len(hist) >= 2:
            s_vals = [(hx - tx) * ux + (hy - ty) * uy for (_, hx, hy) in hist]
            s_lo = min(min(s_vals), 0.0, proj_end)
            s_hi = max(max(s_vals), 0.0, proj_end)
        else:
            s_lo = min(0.0, proj_end)
            s_hi = max(0.0, proj_end)
        s_mid = 0.5 * (s_lo + s_hi)
        pc_x = tx + s_mid * ux
        pc_y = ty + s_mid * uy
        amp = 0.5 * (s_hi - s_lo)

    centers = np.empty((k_len, 2), dtype=np.float64)
    centers[:, 0] = px_mix
    centers[:, 1] = py_mix
    sweep_start = (pc_x - amp * ux, pc_y - amp * uy)
    sweep_end = (pc_x + amp * ux, pc_y + amp * uy)

    return TrajectoryPrediction(
        taus=taus_arr,
        centers=centers,
        sigmas=sigmas,
        u_hat=(ux, uy),
        p_c=(pc_x, pc_y),
        amplitude=float(amp),
        omega=float(omega),
        phi=float(phi_ref),
        sweep_start=sweep_start,
        sweep_end=sweep_end,
        mu_cv=float(mu_cv),
        mu_harm=float(mu_harm),
        is_harmonic=bool(imm_valid and mu_harm >= 0.5),
    )


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

    pts64 = np.asarray(pts, dtype=np.float64)
    inv_n = 1.0 / float(n)
    m0 = float(pts64[:, 0].sum()) * inv_n
    m1 = float(pts64[:, 1].sum()) * inv_n
    x = pts64[:, 0] - m0
    y = pts64[:, 1] - m1
    z = x * x + y * y

    Mxx = float((x * x).sum()) * inv_n
    Myy = float((y * y).sum()) * inv_n
    Mxy = float((x * y).sum()) * inv_n
    tr = Mxx + Myy
    if tr < 1e-10:
        return None
    disc_cov = math.sqrt(max(0.0, (Mxx - Myy) ** 2 + 4.0 * Mxy * Mxy))
    lam_max = 0.5 * (tr + disc_cov)
    lam_min = max(0.0, 0.5 * (tr - disc_cov))
    spread_ratio = lam_min / max(lam_max, 1e-12)

    Mxz = float((x * z).sum()) * inv_n
    Myz = float((y * z).sum()) * inv_n
    Mzz = float((z * z).sum()) * inv_n
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
    cx = m0 + dx
    cy = m1 + dy
    if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(radius)):
        return None

    dists = np.hypot(x - dx, y - dy)
    diff_r = dists - radius
    rmse = math.sqrt(float((diff_r * diff_r).sum()) * inv_n)
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
    n_pts = pts.shape[0]
    if n_pts < 3:
        return None

    inv_n = 1.0 / float(n_pts)
    mean_x = float(pts[:, 0].sum()) * inv_n
    mean_y = float(pts[:, 1].sum()) * inv_n
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
    if n_rows == 1 and n_cols == 1:
        return [(0, 0)]
    if n_rows == 2 and n_cols == 2:
        if (cost[0, 0] + cost[1, 1]) <= (cost[0, 1] + cost[1, 0]):
            return [(0, 0), (1, 1)]
        return [(0, 1), (1, 0)]

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
        confirm_min_hits: int = 5,
        confirm_min_speed_mps: float = 0.12,
        confirm_min_diameter_m: float = 0.20,
        confirm_max_diameter_m: float = 0.55,
        max_confirm_spike_ratio: float = 0.03,
        imm_history_window_sec: float = 6.0,
        imm_min_samples: int = 10,
        imm_min_span_sec: float = 1.8,
        max_coast_ticks: int = 2,
        lidar_min_range_m: float = 0.12,
        lidar_max_range_m: float = 3.50,
        lidar_offset_x_m: float = 0.0,
        confirm_min_disp_m: float = 0.08,
        ch3_horizons: Sequence[float] = (0.5, 1.0, 1.5),
        ch3_weights: Sequence[float] = (1.0, 0.75, 0.50),
        ch3_eps: float = 1e-3,
        enable_spatiotemporal_dynamic_control: bool = True,
    ) -> None:
        self.enable_spatiotemporal_dynamic_control = bool(enable_spatiotemporal_dynamic_control)
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
        self.confirm_min_diameter_m = float(confirm_min_diameter_m)
        self.confirm_max_diameter_m = float(confirm_max_diameter_m)
        self.max_confirm_spike_ratio = float(max_confirm_spike_ratio)
        self.imm_history_window_sec = float(imm_history_window_sec)
        self.imm_min_samples = int(imm_min_samples)
        self.imm_min_span_sec = float(imm_min_span_sec)
        self.max_coast_ticks = int(max_coast_ticks)
        self.lidar_min_range_m = float(lidar_min_range_m)
        self.lidar_max_range_m = float(lidar_max_range_m)
        self.lidar_offset_x_m = float(lidar_offset_x_m)
        self.confirm_min_disp_m = float(confirm_min_disp_m)
        self.ch3_horizons = tuple(float(h) for h in ch3_horizons)
        self.ch3_weights = tuple(float(w) for w in ch3_weights)
        self.ch3_eps = float(ch3_eps)

        self.static_occ = np.zeros((self.n, self.n), dtype=np.float32)
        self.observed_free = np.zeros((self.n, self.n), dtype=np.float32)
        self.grid_initialized = False
        self.origin_ix = 0
        self.origin_iy = 0

        self._dilate_mask_buf = np.zeros((self.n, self.n), dtype=bool)
        self._dilate_mh_buf = np.zeros((self.n, self.n), dtype=bool)
        df_init = (0.95 - 0.03) / 63.0
        self._ray_fracs_init = (
            np.linspace(0.03, 0.95, 64, dtype=np.float32)[:, None],
            np.linspace(0.03 + 0.5 * df_init, 0.95 + 0.5 * df_init, 64, dtype=np.float32)[:, None],
        )
        df_steady = (0.95 - 0.03) / 31.0
        self._ray_fracs_steady = tuple(
            np.linspace(0.03 + k * df_steady * 0.25, 0.95 + k * df_steady * 0.25, 32, dtype=np.float32)[:, None]
            for k in range(4)
        )
        self._extrap_cache_stamp: Optional[float] = None
        self._extrap_cache_states: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        self._R = np.eye(2, dtype=np.float64) * (self.meas_noise_std_m ** 2)
        self._H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        self._I4 = np.eye(4, dtype=np.float64)
        self._ch3_fq_cache: Dict[float, Tuple[np.ndarray, np.ndarray]] = {
            h: damped_cv_matrices(h, beta=self.beta, q_acc=self.q_acc_base)
            for h in self.ch3_horizons
        }
        f_list, ft_list, q_list = [], [], []
        for h in self.ch3_horizons:
            Fh, Qh = self._ch3_fq_cache[h]
            f_list.append(Fh)
            ft_list.append(Fh.T)
            q_list.append(Qh)
        self._ch3_F_batch = np.stack(f_list, axis=0) if f_list else np.empty((0, 4, 4))
        self._ch3_FT_batch = np.stack(ft_list, axis=0) if ft_list else np.empty((0, 4, 4))
        self._ch3_Q_batch = np.stack(q_list, axis=0) if q_list else np.empty((0, 4, 4))
        self._ch3_w_arr = np.asarray(self.ch3_weights, dtype=np.float32)
        self._step_fq_cache: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
        for h_pre in (0.05, 0.10, 0.15, 0.20):
            self._step_fq_cache[round(h_pre, 4)] = damped_cv_matrices(h_pre, beta=self.beta, q_acc=self.q_acc_base)

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
        self.last_spike_ratio: float = 0.0
        self.last_scan_corrupted: bool = False

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
        self._extrap_cache_stamp = None
        self._extrap_cache_states.clear()
        self.last_did_update = False
        self.scan_update_count = 0
        self.extrap_tick_count = 0
        self.last_spike_ratio = 0.0
        self.last_scan_corrupted = False

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
        mh = self._dilate_mh_buf
        np.copyto(m, s)
        m[1:, :] |= s[:-1, :]
        m[:-1, :] |= s[1:, :]
        np.copyto(mh, m)
        mh[:, 1:] |= m[:, :-1]
        mh[:, :-1] |= m[:, 1:]
        if abs(float(omega)) > self.omega_rot_guard_rad_s:
            np.copyto(m, mh)
            m[1:, :] |= mh[:-1, :]
            m[:-1, :] |= mh[1:, :]
            np.copyto(mh, m)
            mh[:, 1:] |= m[:, :-1]
            mh[:, :-1] |= m[:, 1:]

        inb = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
        ix_c = ix.clip(0, self.n - 1)
        iy_c = iy.clip(0, self.n - 1)
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

    def _point_near_confirmed_track(self, wx: np.ndarray, wy: np.ndarray, gate_m: float) -> np.ndarray:
        if not self.tracks:
            return np.zeros(len(wx), dtype=bool)
        g2 = gate_m * gate_m
        near = np.zeros(len(wx), dtype=bool)
        for trk in self.tracks.values():
            if not trk.is_confirmed:
                continue
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
    ) -> List[Tuple[np.ndarray, Optional[Tuple[int, int]], Optional[Tuple[Tuple[int, int], Tuple[int, int]]]]]:
        """
        O(N) 1D angular sequential clustering with 1-2 ray dropout gap allowance
        and 0 deg / 359 deg wrap-around merge. Returns zero-copy slice descriptors.
        """
        m = len(cand_idx)
        if m == 0:
            return []

        max_idx_diff = self.cluster_max_ray_gap + 1
        dthresh2 = self.cluster_dist_thresh_m * self.cluster_dist_thresh_m

        if m == 1:
            ranges_se: List[Tuple[int, int]] = [(0, 1)]
        else:
            gaps = cand_idx[1:] - cand_idx[:-1]
            diff = cand_pts_w[1:] - cand_pts_w[:-1]
            d2 = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]
            breaks = ((gaps > max_idx_diff) | (d2 >= dthresh2)).nonzero()[0]
            if len(breaks) == 0:
                ranges_se = [(0, m)]
            else:
                b_list = breaks.tolist()
                ranges_se = []
                prev = 0
                for b in b_list:
                    ranges_se.append((prev, b + 1))
                    prev = b + 1
                ranges_se.append((prev, m))

        # 0 deg / 359 deg wrap-around merge preserving arc endpoint ordering
        merged_wrap = False
        if is_full_360 and len(ranges_se) >= 2:
            s0, e0 = ranges_se[0]
            sl, el = ranges_se[-1]
            wrap_gap = (int(cand_idx[s0]) + n_rays_total) - int(cand_idx[el - 1])
            if 1 <= wrap_gap <= max_idx_diff:
                dx = float(cand_pts_w[s0, 0] - cand_pts_w[el - 1, 0])
                dy = float(cand_pts_w[s0, 1] - cand_pts_w[el - 1, 1])
                if (dx * dx + dy * dy) < dthresh2:
                    merged_wrap = True

        out: List[Tuple[np.ndarray, Optional[Tuple[int, int]], Optional[Tuple[Tuple[int, int], Tuple[int, int]]]]] = []
        if merged_wrap:
            s0, e0 = ranges_se[0]
            sl, el = ranges_se[-1]
            pts_w = np.concatenate((cand_pts_w[sl:el], cand_pts_w[s0:e0]), axis=0)
            out.append((pts_w, None, ((sl, el), (s0, e0))))
            for s_i, e_i in ranges_se[1:-1]:
                out.append((cand_pts_w[s_i:e_i], (s_i, e_i), None))
        else:
            for s_i, e_i in ranges_se:
                out.append((cand_pts_w[s_i:e_i], (s_i, e_i), None))
        return out

    def _is_static_wall_fragment(
        self,
        cl_pts: np.ndarray,
        start_v_pos: int,
        end_v_pos: int,
        valid_idx: np.ndarray,
        wx: np.ndarray,
        wy: np.ndarray,
        in_static: np.ndarray,
        n_rays_total: int,
        is_full_360: bool,
    ) -> bool:
        """
        Reject non-static clusters carved out of a continuous static wall by grazing-ray
        free-space carving or partial static_occ masking.

        Checks whether immediate angular neighbor rays in the full valid scan belong to
        `in_static` within `cluster_dist_thresh_m`:
          - Both ends continuously attached to `in_static` -> static wall window.
          - One end continuously attached to `in_static` AND collinear flat line
            (spread_ratio < 0.008, neighbor perp_dist < 0.035 m) or extended static
            segment chord > max_chord_m -> static wall edge fragment.
        Isolated moving cylinders (both ends unattached) short-circuit in O(1).
        """
        n_valid = len(valid_idx)
        if n_valid < 2:
            return False

        max_idx_diff = self.cluster_max_ray_gap + 1
        dthresh2 = self.cluster_dist_thresh_m * self.cluster_dist_thresh_m

        adj_left = False
        v_prev = -1
        if is_full_360 or start_v_pos > 0:
            v_prev = (start_v_pos - 1) % n_valid
            if v_prev != end_v_pos and bool(in_static[v_prev]):
                r_s = int(valid_idx[start_v_pos])
                r_p = int(valid_idx[v_prev])
                gap_l = (r_s - r_p) % n_rays_total if is_full_360 else (r_s - r_p)
                if 1 <= gap_l <= max_idx_diff:
                    dx_l = float(wx[v_prev]) - float(cl_pts[0, 0])
                    dy_l = float(wy[v_prev]) - float(cl_pts[0, 1])
                    if (dx_l * dx_l + dy_l * dy_l) < dthresh2:
                        adj_left = True

        adj_right = False
        v_next = -1
        if is_full_360 or (end_v_pos + 1) < n_valid:
            v_next = (end_v_pos + 1) % n_valid
            if v_next != start_v_pos and bool(in_static[v_next]):
                r_e = int(valid_idx[end_v_pos])
                r_n = int(valid_idx[v_next])
                gap_r = (r_n - r_e) % n_rays_total if is_full_360 else (r_n - r_e)
                if 1 <= gap_r <= max_idx_diff:
                    dx_r = float(wx[v_next]) - float(cl_pts[-1, 0])
                    dy_r = float(wy[v_next]) - float(cl_pts[-1, 1])
                    if (dx_r * dx_r + dy_r * dy_r) < dthresh2:
                        adj_right = True

        if not adj_left and not adj_right:
            return False

        # Compute cluster 2D covariance spread_ratio and chord line for collinearity verification
        n_cl = cl_pts.shape[0]
        inv_n = 1.0 / float(n_cl)
        mx = float(cl_pts[:, 0].sum()) * inv_n
        my = float(cl_pts[:, 1].sum()) * inv_n
        dx_c = cl_pts[:, 0] - mx
        dy_c = cl_pts[:, 1] - my
        cxx = float((dx_c * dx_c).sum()) * inv_n
        cyy = float((dy_c * dy_c).sum()) * inv_n
        cxy = float((dx_c * dy_c).sum()) * inv_n
        tr = cxx + cyy
        if tr > 1e-10:
            disc = math.sqrt(max(0.0, (cxx - cyy) ** 2 + 4.0 * cxy * cxy))
            lam_max = 0.5 * (tr + disc)
            lam_min = max(0.0, 0.5 * (tr - disc))
            spread_ratio = lam_min / max(lam_max, 1e-12)
        else:
            spread_ratio = 0.0

        lx = float(cl_pts[-1, 0] - cl_pts[0, 0])
        ly = float(cl_pts[-1, 1] - cl_pts[0, 1])
        l_len = math.hypot(lx, ly)
        x0_c = float(cl_pts[0, 0])
        y0_c = float(cl_pts[0, 1])

        def _perp_dist(px: float, py: float) -> float:
            if l_len <= 1e-6:
                return 0.0
            return abs((px - x0_c) * ly - (py - y0_c) * lx) / l_len

        # Both ends flush & collinear with continuous static wall rays -> carved wall interior window
        if adj_left and adj_right:
            perp_l = _perp_dist(float(wx[v_prev]), float(wy[v_prev]))
            perp_r = _perp_dist(float(wx[v_next]), float(wy[v_next]))
            if spread_ratio < 0.015 and perp_l < 0.045 and perp_r < 0.045:
                return True
            return False

        # Single-end static attachment: require collinearity with the static neighbor and walked wall
        v_adj = v_prev if adj_left else v_next
        ax = float(wx[v_adj])
        ay = float(wy[v_adj])
        perp_adj = _perp_dist(ax, ay)
        if perp_adj >= 0.035:
            return False
        if spread_ratio < 0.008:
            return True

        # Walk outward along contiguous rays on the attached static side to verify collinear wall chord
        step_dir = -1 if adj_left else 1
        cur_v = v_adj
        cur_x, cur_y = ax, ay
        n_static_walked = 1 if bool(in_static[cur_v]) else 0
        for _ in range(32):
            nxt_v = (cur_v + step_dir) % n_valid if is_full_360 else (cur_v + step_dir)
            if nxt_v < 0 or nxt_v >= n_valid or nxt_v == start_v_pos or nxt_v == end_v_pos:
                break
            r_cur = int(valid_idx[cur_v])
            r_nxt = int(valid_idx[nxt_v])
            gap_w = (
                ((r_cur - r_nxt) % n_rays_total if is_full_360 else (r_cur - r_nxt))
                if step_dir < 0
                else ((r_nxt - r_cur) % n_rays_total if is_full_360 else (r_nxt - r_cur))
            )
            if not (1 <= gap_w <= max_idx_diff):
                break
            nx_w = float(wx[nxt_v])
            ny_w = float(wy[nxt_v])
            if ((nx_w - cur_x) ** 2 + (ny_w - cur_y) ** 2) >= dthresh2:
                break
            if _perp_dist(nx_w, ny_w) >= 0.06:
                break
            cur_v = nxt_v
            cur_x, cur_y = nx_w, ny_w
            if bool(in_static[cur_v]):
                n_static_walked += 1

        far_x = float(cl_pts[-1, 0]) if adj_left else float(cl_pts[0, 0])
        far_y = float(cl_pts[-1, 1]) if adj_left else float(cl_pts[0, 1])
        ext_chord = math.hypot(cur_x - far_x, cur_y - far_y)
        if ext_chord > self.max_chord_m and n_static_walked >= 2 and spread_ratio < 0.015:
            return True
        return False

    def _update_background_grids(
        self,
        rx: float,
        ry: float,
        rth: float,
        ranges: np.ndarray,
        cos_lut: np.ndarray,
        sin_lut: np.ndarray,
        r_valid: np.ndarray,
        wx_valid: np.ndarray,
        wy_valid: np.ndarray,
        ix_valid: np.ndarray,
        iy_valid: np.ndarray,
        pre_obs_free_valid: np.ndarray,
        rot_guard_active: bool = False,
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

        # Unified strided ray-carving: stride=2 (64 steps) on initial 2 scans for instant 100%
        # corridor coverage, then stride=4 (90 rays, 32 staggered steps) per steady-state scan.
        if self.scan_update_count <= 2:
            stride = 2
            parity = self.scan_update_count % 2
            fracs = self._ray_fracs_init[parity]
        else:
            stride = 4
            parity = self.scan_update_count % 4
            fracs = self._ray_fracs_steady[parity]

        r_sub = ranges[parity::stride]
        c_sub = cos_lut[parity::stride]
        s_sub = sin_lut[parity::stride]
        valid_r = np.isfinite(r_sub) & (r_sub >= np.float32(self.lidar_min_range_m))
        max_r_o = np.float32(min(0.5 * self.grid_size_m - self.res, self.lidar_max_range_m - 0.10))
        r_max_th = np.float32(self.lidar_max_range_m - 0.05)
        r_carve = np.where(
            valid_r & (r_sub <= r_max_th),
            np.minimum(r_sub * np.float32(0.90), np.maximum(np.float32(0.0), r_sub - np.float32(0.06))),
            np.where(np.isposinf(r_sub) | (valid_r & (r_sub > r_max_th)), max_r_o, np.float32(0.0)),
        )
        active_rays = r_carve > np.float32(0.16)
        if active_rays.any():
            rc = r_carve[active_rays]
            off_x = np.float32(self.lidar_offset_x_m)
            has_off = abs(self.lidar_offset_x_m) > 1e-6
            lx = rc * c_sub[active_rays] + off_x if has_off else rc * c_sub[active_rays]
            ly = rc * s_sub[active_rays]
            if abs(rth) <= 1e-6:
                sx_ray = rx_f + off_x if has_off else rx_f
                sy_ray = ry_f
                ex_all = rx_f + lx
                ey_all = ry_f + ly
            else:
                c_f = np.float32(c)
                s_f = np.float32(s)
                sx_ray = rx_f + c_f * off_x if has_off else rx_f
                sy_ray = ry_f + s_f * off_x if has_off else ry_f
                ex_all = rx_f + c_f * lx - s_f * ly
                ey_all = ry_f + s_f * lx + c_f * ly
            min_gx = np.float32(self.origin_ix * self.res + 0.001)
            max_gx = np.float32((self.origin_ix + self.n) * self.res - 0.001)
            min_gy = np.float32(self.origin_iy * self.res + 0.001)
            max_gy = np.float32((self.origin_iy + self.n) * self.res - 0.001)

            origin_in_box = (min_gx <= sx_ray <= max_gx) and (min_gy <= sy_ray <= max_gy)
            # Ray-box proportional clipping from sensor origin (sx_ray, sy_ray) preserves true ray angle.
            # Fast path skips division allocations when origin and all ray endpoints lie strictly inside the grid.
            ex_clip = ex_all
            ey_clip = ey_all
            has_zero_t = False
            if (
                not origin_in_box
                or float(ex_all.min()) < float(min_gx)
                or float(ex_all.max()) > float(max_gx)
                or float(ey_all.min()) < float(min_gy)
                or float(ey_all.max()) > float(max_gy)
            ):
                dx_r = ex_all - sx_ray
                dy_r = ey_all - sy_ray
                t_max = np.ones_like(dx_r, dtype=np.float32)
                pos_dx = dx_r > np.float32(1e-6)
                if pos_dx.any():
                    t_max[pos_dx] = np.minimum(t_max[pos_dx], (max_gx - sx_ray) / dx_r[pos_dx])
                neg_dx = dx_r < np.float32(-1e-6)
                if neg_dx.any():
                    t_max[neg_dx] = np.minimum(t_max[neg_dx], (min_gx - sx_ray) / dx_r[neg_dx])
                pos_dy = dy_r > np.float32(1e-6)
                if pos_dy.any():
                    t_max[pos_dy] = np.minimum(t_max[pos_dy], (max_gy - sy_ray) / dy_r[pos_dy])
                neg_dy = dy_r < np.float32(-1e-6)
                if neg_dy.any():
                    t_max[neg_dy] = np.minimum(t_max[neg_dy], (min_gy - sy_ray) / dy_r[neg_dy])
                t_max = np.clip(t_max, np.float32(0.0), np.float32(1.0))
                has_zero_t = bool((t_max <= np.float32(0.0)).any())
                ex_clip = sx_ray + t_max * dx_r
                ey_clip = sy_ray + t_max * dy_r

            ray_x = sx_ray + fracs * (ex_clip - sx_ray)[None, :]
            ray_y = sy_ray + fracs * (ey_clip - sy_ray)[None, :]
            shift_off = 1024
            ix_f = (ray_x * inv_res - np.float32(self.origin_ix - shift_off)).astype(np.int32) - shift_off
            iy_f = (ray_y * inv_res - np.float32(self.origin_iy - shift_off)).astype(np.int32) - shift_off
            if origin_in_box and not has_zero_t:
                flat_idx = (ix_f * self.n + iy_f).ravel()
                self.observed_free.ravel()[flat_idx] = 1.0
                self.static_occ.ravel()[flat_idx] = 0.0
            else:
                in_bounds = (ix_f >= 0) & (ix_f < self.n) & (iy_f >= 0) & (iy_f < self.n)
                if has_zero_t:
                    in_bounds &= (t_max[None, :] > np.float32(0.0))
                if in_bounds.any():
                    flat_idx = ix_f[in_bounds] * self.n + iy_f[in_bounds]
                    self.observed_free.ravel()[flat_idx] = 1.0
                    self.static_occ.ravel()[flat_idx] = 0.0

        if len(wx_valid) > 0:
            # Dynamic masking: exclude hit cells within confirmed track radius + 0.08 m,
            # and also protect clean-scan actively moving tentative tracks (hits >= 2, speed >= 0.10 m/s)
            # so overlapping cylinders are not prematurely solidified into static_occ before hit 5.
            masked_by_dyn = np.zeros(len(wx_valid), dtype=bool)
            for trk in self.tracks.values():
                if trk.is_confirmed:
                    rad = trk.radius + self.dynamic_mask_margin_m
                    masked_by_dyn |= ((wx_valid - trk.x) ** 2 + (wy_valid - trk.y) ** 2) <= (rad * rad)
                elif (
                    (not self.last_scan_corrupted)
                    and (not rot_guard_active)
                    and trk.consecutive_hits >= 2
                    and trk.speed >= 0.10
                ):
                    rad = trk.radius + 0.04
                    masked_by_dyn |= ((wx_valid - trk.x) ** 2 + (wy_valid - trk.y) ** 2) <= (rad * rad)

            update_mask = (~masked_by_dyn) & (ix_valid >= 0) & (ix_valid < self.n) & (iy_valid >= 0) & (iy_valid < self.n)
            if np.any(update_mask):
                ix_u = ix_valid[update_mask]
                iy_u = iy_valid[update_mask]
                was_free = pre_obs_free_valid[update_mask] >= np.float32(self.observed_free_thresh)

                # Never-free hits establish static_occ = 1.0 & clear observed_free immediately;
                # previously-free hits build static_occ slowly (+0.34/tick) and gently decay
                # observed_free (*0.80) so overlapping dynamic cylinders confirm on tick 2
                # while stationary objects still solidify into static_occ >= 0.75 by tick 3.
                never_free = ~was_free
                if np.any(never_free):
                    ix_nf = ix_u[never_free]
                    iy_nf = iy_u[never_free]
                    self.static_occ[ix_nf, iy_nf] = 1.0
                    self.observed_free[ix_nf, iy_nf] = 0.0
                if np.any(was_free):
                    ix_wf = ix_u[was_free]
                    iy_wf = iy_u[was_free]
                    self.static_occ[ix_wf, iy_wf] = np.minimum(
                        np.float32(1.0), self.static_occ[ix_wf, iy_wf] + np.float32(0.34)
                    )
                    self.observed_free[ix_wf, iy_wf] *= np.float32(0.80)

        sx = int(math.floor(rx / self.res)) - self.origin_ix
        sy = int(math.floor(ry / self.res)) - self.origin_iy
        if 0 <= sx < self.n and 0 <= sy < self.n:
            self.observed_free[sx, sy] = 1.0
            self.static_occ[sx, sy] *= np.float32(0.25)

    def _update_track_imm(self, trk: TrackedObstacle) -> None:
        """Refit 2x2 PCA + CV/Harmonic IMM parameters on confirmed track position history."""
        if not self.enable_spatiotemporal_dynamic_control:
            trk.imm_valid = False
            trk.imm_mu_cv = 1.0
            trk.imm_mu_harm = 0.0
            return
        fit = fit_pca_and_harmonic_imm(
            trk.pos_history,
            fallback_vx=trk.vx,
            fallback_vy=trk.vy,
            meas_noise_std=self.meas_noise_std_m,
            min_samples=self.imm_min_samples,
            min_span_sec=self.imm_min_span_sec,
            prev_omega=(trk.imm_omega if (trk.imm_valid and trk.imm_mu_harm >= 0.80) else 0.0),
        )
        trk.imm_valid = bool(fit["valid"])
        trk.imm_u_hat = fit["u_hat"]  # type: ignore[assignment]
        trk.imm_p_c = fit["p_c"]  # type: ignore[assignment]
        trk.imm_amp = float(fit["amp"])
        trk.imm_omega = float(fit["omega"])
        trk.imm_phi = float(fit["phi"])
        trk.imm_t_ref = float(fit["t_ref"])
        trk.imm_mu_cv = float(fit["mu_cv"])
        trk.imm_mu_harm = float(fit["mu_harm"])
        trk.imm_sigma_h0 = float(fit["sigma_h0"])
        trk.imm_sigma_omega = float(fit["sigma_omega"])
        trk.imm_harm_rmse = float(fit["harm_rmse"])
        trk.imm_cv_rmse = float(fit["cv_rmse"])

    def _kf_step_and_associate(
        self,
        detections: List[Tuple[float, float, float]],
        dt: float,
        stamp_sec: float,
        freeze_velocity: bool = False,
        allow_confirm: bool = True,
        rot_guard_active: bool = False,
    ) -> None:
        track_ids = list(self.tracks.keys())
        n_trk = len(track_ids)
        n_det = len(detections)

        # 1. Predict all existing tracks using exact damped-CV F(dt) and adaptive q_acc Q(dt)
        pred_states: Dict[int, np.ndarray] = {}
        pred_covs: Dict[int, np.ndarray] = {}
        dt_key = round(dt, 4)
        cached_step = self._step_fq_cache.get(dt_key)
        if cached_step is None:
            cached_step = damped_cv_matrices(dt, beta=self.beta, q_acc=self.q_acc_base)
            if len(self._step_fq_cache) < 32:
                self._step_fq_cache[dt_key] = cached_step
        F_dt, Q_dt_base = cached_step
        for tid in track_ids:
            trk = self.tracks[tid]
            s0 = np.array([trk.x, trk.y, trk.vx, trk.vy], dtype=np.float64)
            s_p = F_dt @ s0
            P_p = F_dt @ trk.P_4x4 @ F_dt.T + trk.q_acc_scale * Q_dt_base
            P_p = 0.5 * (P_p + P_p.T)
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
            inv_S_list: List[Optional[np.ndarray]] = [None] * n_trk
            r_var = self.meas_noise_std_m * self.meas_noise_std_m
            for r_i, tid in enumerate(track_ids):
                s_p = pred_states[tid]
                P_p = pred_covs[tid]
                s00 = float(P_p[0, 0]) + r_var
                s01 = float(P_p[0, 1])
                s11 = float(P_p[1, 1]) + r_var
                det_S = s00 * s11 - s01 * s01
                if det_S <= 1e-12:
                    continue
                inv_det = 1.0 / det_S
                i00 = s11 * inv_det
                i01 = -s01 * inv_det
                i11 = s00 * inv_det
                inv_S_list[r_i] = np.array([[i00, i01], [i01, i11]], dtype=np.float64)
                px_p = float(s_p[0])
                py_p = float(s_p[1])
                for c_j, (zx, zy, z_r) in enumerate(detections):
                    yx = zx - px_p
                    yy = zy - py_p
                    d2 = yx * yx + yy * yy
                    if d2 > gate_d2:
                        continue
                    maha2 = yx * (i00 * yx + i01 * yy) + yy * (i01 * yx + i11 * yy)
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

                    y = np.array([zx - float(s_p[0]), zy - float(s_p[1])], dtype=np.float64)
                    inv_S = inv_S_list[r_i]
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
                        s00 = float(P_p[0, 0]) + r_var
                        s01 = float(P_p[0, 1])
                        s11 = float(P_p[1, 1]) + r_var
                        inv_det = 1.0 / max(s00 * s11 - s01 * s01, 1e-12)
                        inv_S = np.array([[s11 * inv_det, -s01 * inv_det], [-s01 * inv_det, s00 * inv_det]], dtype=np.float64)

                    K = P_p[:, :2] @ inv_S
                    if freeze_velocity:
                        K_use = K.copy()
                        K_use[2:, :] = 0.0
                        s_new = s_p + K_use @ y
                        s_new[2] = trk.vx
                        s_new[3] = trk.vy
                        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
                        I_KH = np.eye(4, dtype=np.float64) - K_use @ H
                        R_mat = np.diag([r_var, r_var])
                        P_new = I_KH @ P_p @ I_KH.T + K_use @ R_mat @ K_use.T
                        P_new = 0.5 * (P_new + P_new.T)
                    else:
                        s_new = s_p + K @ y
                        P_new = P_p - K @ (P_p[:2, :])
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
                    if allow_confirm or trk.is_confirmed:
                        trk.consecutive_hits += 1
                    else:
                        trk.consecutive_hits = 0
                    trk.missed_ticks = 0
                    trk.q_acc_scale = next_scale
                    trk.vel_history = new_hist
                    t_cutoff = float(stamp_sec) - self.imm_history_window_sec
                    trimmed_hist = tuple(
                        item for item in trk.pos_history if item[0] >= t_cutoff
                    )
                    trk.pos_history = (trimmed_hist + ((float(stamp_sec), float(zx), float(zy)),))[-40:]
                    cur_spd = trk.speed
                    if cur_spd > trk.max_speed_seen:
                        trk.max_speed_seen = cur_spd
                    net_disp = math.hypot(trk.x - trk.birth_x, trk.y - trk.birth_y)
                    if net_disp > trk.max_disp_seen:
                        trk.max_disp_seen = net_disp
                    diam_m = 2.0 * float(trk.radius)
                    req_disp_m = (
                        max(self.confirm_min_disp_m, 0.25)
                        if rot_guard_active
                        else self.confirm_min_disp_m
                    )
                    if (
                        allow_confirm
                        and not trk.is_confirmed
                        and trk.consecutive_hits >= self.confirm_min_hits
                        and cur_spd >= self.confirm_min_speed_mps
                        and net_disp >= req_disp_m
                        and self.confirm_min_diameter_m <= diam_m <= self.confirm_max_diameter_m
                    ):
                        trk.is_confirmed = True
                    elif (
                        trk.is_confirmed
                        and trk.hits >= 6
                        and trk.max_disp_seen < 0.08
                        and trk.max_speed_seen < 0.16
                    ):
                        # Demote stationary debris/furniture mis-confirmed by transient sensor noise
                        trk.is_confirmed = False

                    if trk.is_confirmed and len(trk.pos_history) >= 3:
                        if (
                            (not trk.imm_valid)
                            or trk.imm_mu_harm < 0.80
                            or len(trk.pos_history) <= 14
                            or (trk.hits % 2 == 0)
                        ):
                            self._update_track_imm(trk)

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
                consecutive_hits=(1 if allow_confirm else 0),
                missed_ticks=0,
                q_acc_scale=1.0,
                vel_history=(),
                birth_x=float(zx),
                birth_y=float(zy),
                max_speed_seen=0.0,
                max_disp_seen=0.0,
                pos_history=((float(stamp_sec), float(zx), float(zy)),),
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
        spike_ratio: float = 0.0,
        scan_corrupted: bool = False,
    ) -> List[TrackedObstacle]:
        """
        Process a control tick.

        Performs full background subtraction, clustering, circle fitting, and KF update
        ONLY when `scan_stamp_sec` advances (5 Hz); on intermediate ticks (10 Hz / 20 Hz),
        skips observation update and returns non-mutating damped-KF extrapolated tracks.
        """
        t_eval = float(scan_stamp_sec if now_sec is None else now_sec)
        stamp_f = float(scan_stamp_sec)
        spk_f = float(spike_ratio)
        is_corrupt = bool(scan_corrupted) or (spk_f >= self.max_confirm_spike_ratio)
        rot_guard_active = abs(float(omega)) > self.omega_rot_guard_rad_s
        self.last_spike_ratio = spk_f
        self.last_scan_corrupted = is_corrupt
        allow_confirm = not is_corrupt

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
        self._extrap_cache_stamp = None
        self._extrap_cache_states.clear()
        self.last_did_update = True
        self.scan_update_count += 1

        self._recenter_grid(rx, ry)

        if raw_ranges is None or len(raw_ranges) < 3:
            self._kf_step_and_associate(
                [], dt, stamp_f, freeze_velocity=freeze_velocity, allow_confirm=allow_confirm
            )
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
        valid_idx = valid_mask.nonzero()[0]
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
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
            )
            self._kf_step_and_associate(
                [], dt, stamp_f, freeze_velocity=freeze_velocity, allow_confirm=allow_confirm
            )
            return list(self.tracks.values())

        r_val = ranges[valid_idx]
        lx = r_val * cos_lut[valid_idx] + np.float32(self.lidar_offset_x_m)
        ly = r_val * sin_lut[valid_idx]
        c = np.float32(math.cos(rth))
        s = np.float32(math.sin(rth))
        wx = np.float32(rx) + c * lx - s * ly
        wy = np.float32(ry) + s * lx + c * ly

        inv_res = np.float32(1.0 / self.res)
        shift_off = 1024
        ix = (wx * inv_res - np.float32(self.origin_ix - shift_off)).astype(np.int32) - shift_off
        iy = (wy * inv_res - np.float32(self.origin_iy - shift_off)).astype(np.int32) - shift_off

        # Snapshot pre-update observed_free BEFORE ray-carving updates current hits
        pre_obs_free = self._sample_observed_free(
            wx, wy, ix, iy, external_observed_free, ext_origin_ix, ext_origin_iy, ext_res
        )

        # Condition (1): Exclude points falling inside dilated static_occ >= 0.75 mask
        in_static = self._check_static_mask_at_cells(ix, iy, omega=omega)
        non_static_sel = (~in_static).nonzero()[0]

        detections: List[Tuple[float, float, float]] = []
        if len(non_static_sel) >= self.min_cluster_points:
            cand_ray_idx = valid_idx[non_static_sel]
            cand_pts_w = np.empty((len(non_static_sel), 2), dtype=np.float64)
            cand_pts_w[:, 0] = wx[non_static_sel]
            cand_pts_w[:, 1] = wy[non_static_sel]
            cand_obs_free = pre_obs_free[non_static_sel]
            cand_near_trk = self._point_near_confirmed_track(
                wx[non_static_sel], wy[non_static_sel], gate_m=self.gating_dist_m
            )

            raw_clusters = self._extract_clusters(
                cand_ray_idx, cand_pts_w, n_rays_total=n_rays, is_full_360=is_full_360
            )
            free_mask_all = cand_obs_free >= np.float32(self.observed_free_thresh)
            for cl_pts, seg_se, wrap_se in raw_clusters:
                # NOTE: Never impose n_c <= 35; only lower bound n_c >= 3
                n_cl = cl_pts.shape[0]
                if n_cl < self.min_cluster_points:
                    continue

                # Condition (2): New emergence in verified free-space OR near confirmed dynamic track
                if seg_se is not None:
                    s_i, e_i = seg_se
                    n_free = int(free_mask_all[s_i:e_i].sum())
                    matches_confirmed = bool(cand_near_trk[s_i:e_i].any())
                else:
                    (sl, el), (s0, e0) = wrap_se  # type: ignore[misc]
                    n_free = int(free_mask_all[sl:el].sum()) + int(free_mask_all[s0:e0].sum())
                    matches_confirmed = bool(cand_near_trk[sl:el].any() or cand_near_trk[s0:e0].any())
                if (n_free / float(n_cl)) < 0.50 and not matches_confirmed:
                    continue

                chord_len = math.hypot(
                    float(cl_pts[-1, 0] - cl_pts[0, 0]),
                    float(cl_pts[-1, 1] - cl_pts[0, 1]),
                )
                if not (self.min_chord_m <= chord_len <= self.max_chord_m):
                    continue

                if seg_se is not None:
                    start_v_pos = int(non_static_sel[seg_se[0]])
                    end_v_pos = int(non_static_sel[seg_se[1] - 1])
                else:
                    (sl, el), (s0, e0) = wrap_se  # type: ignore[misc]
                    start_v_pos = int(non_static_sel[sl])
                    end_v_pos = int(non_static_sel[e0 - 1])
                if self._is_static_wall_fragment(
                    cl_pts,
                    start_v_pos,
                    end_v_pos,
                    valid_idx,
                    wx,
                    wy,
                    in_static,
                    n_rays,
                    is_full_360,
                ):
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
        self._kf_step_and_associate(
            detections,
            dt,
            stamp_f,
            freeze_velocity=freeze_velocity,
            allow_confirm=allow_confirm,
            rot_guard_active=rot_guard_active,
        )

        self._update_background_grids(
            rx,
            ry,
            rth,
            ranges,
            cos_lut,
            sin_lut,
            r_val,
            wx,
            wy,
            ix,
            iy,
            pre_obs_free,
            rot_guard_active=rot_guard_active,
        )

        if abs(t_eval - stamp_f) <= 1e-6:
            return list(self.tracks.values())
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
        if dt_extrap <= 1e-6:
            for trk in self.tracks.values():
                if confirmed_only and not trk.is_confirmed:
                    continue
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
                        birth_x=trk.birth_x,
                        birth_y=trk.birth_y,
                        max_speed_seen=trk.max_speed_seen,
                        max_disp_seen=trk.max_disp_seen,
                        pos_history=trk.pos_history,
                        imm_valid=trk.imm_valid,
                        imm_u_hat=trk.imm_u_hat,
                        imm_p_c=trk.imm_p_c,
                        imm_amp=trk.imm_amp,
                        imm_omega=trk.imm_omega,
                        imm_phi=trk.imm_phi,
                        imm_t_ref=trk.imm_t_ref,
                        imm_mu_cv=trk.imm_mu_cv,
                        imm_mu_harm=trk.imm_mu_harm,
                        imm_sigma_h0=trk.imm_sigma_h0,
                        imm_sigma_omega=trk.imm_sigma_omega,
                        imm_harm_rmse=trk.imm_harm_rmse,
                        imm_cv_rmse=trk.imm_cv_rmse,
                    )
                )
        else:
            dt_k = round(dt_extrap, 4)
            cached_ext = self._step_fq_cache.get(dt_k)
            if cached_ext is None:
                cached_ext = damped_cv_matrices(dt_extrap, beta=self.beta, q_acc=self.q_acc_base)
                if len(self._step_fq_cache) < 32:
                    self._step_fq_cache[dt_k] = cached_ext
            F_e, Q_e_base = cached_ext
            self._extrap_cache_states.clear()
            self._extrap_cache_stamp = stamp_out
            for tid, trk in self.tracks.items():
                if confirmed_only and not trk.is_confirmed:
                    continue
                s0 = np.array([trk.x, trk.y, trk.vx, trk.vy], dtype=np.float64)
                s_e = F_e @ s0
                P_e = F_e @ trk.P_4x4 @ F_e.T + trk.q_acc_scale * Q_e_base
                P_e = 0.5 * (P_e + P_e.T)
                self._extrap_cache_states[tid] = (s_e, P_e)
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
                        birth_x=trk.birth_x,
                        birth_y=trk.birth_y,
                        max_speed_seen=trk.max_speed_seen,
                        max_disp_seen=trk.max_disp_seen,
                        pos_history=trk.pos_history,
                        imm_valid=trk.imm_valid,
                        imm_u_hat=trk.imm_u_hat,
                        imm_p_c=trk.imm_p_c,
                        imm_amp=trk.imm_amp,
                        imm_omega=trk.imm_omega,
                        imm_phi=trk.imm_phi,
                        imm_t_ref=trk.imm_t_ref,
                        imm_mu_cv=trk.imm_mu_cv,
                        imm_mu_harm=trk.imm_mu_harm,
                        imm_sigma_h0=trk.imm_sigma_h0,
                        imm_sigma_omega=trk.imm_sigma_omega,
                        imm_harm_rmse=trk.imm_harm_rmse,
                        imm_cv_rmse=trk.imm_cv_rmse,
                    )
                )
        return out

    def apply_se2_increment(self, d_tx: float, d_ty: float, d_th: float) -> None:
        """
        Apply an SE(2) world-frame transform increment delta_T = (d_tx, d_ty, d_th)
        to all tracked obstacles, velocity histories, covariances, and background grids.
        """
        self._extrap_cache_stamp = None
        self._extrap_cache_states.clear()
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
            nbx = d_tx + c * trk.birth_x - s * trk.birth_y
            nby = d_ty + s * trk.birth_x + c * trk.birth_y
            trk.birth_x = float(nbx)
            trk.birth_y = float(nby)
            P_rot = M4 @ trk.P_4x4 @ M4.T
            trk.P_4x4 = 0.5 * (P_rot + P_rot.T)
            if trk.vel_history:
                trk.vel_history = tuple(
                    (c * hvx - s * hvy, s * hvx + c * hvy) for hvx, hvy in trk.vel_history
                )
            if trk.pos_history:
                trk.pos_history = tuple(
                    (ht, float(d_tx + c * hx - s * hy), float(d_ty + s * hx + c * hy))
                    for ht, hx, hy in trk.pos_history
                )
            ux_o, uy_o = trk.imm_u_hat
            trk.imm_u_hat = (float(c * ux_o - s * uy_o), float(s * ux_o + c * uy_o))
            pcx_o, pcy_o = trk.imm_p_c
            trk.imm_p_c = (
                float(d_tx + c * pcx_o - s * pcy_o),
                float(d_ty + s * pcx_o + c * pcy_o),
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

    def predict_trajectory(
        self,
        track: TrackedObstacle,
        taus: Sequence[float],
    ) -> TrajectoryPrediction:
        """
        Predict future centers p_hat(t + tau), uncertainty radii sigma(tau),
        estimated sweep segment [p_c - A*u_hat, p_c + A*u_hat], and principal axis u_hat
        for tau in [0.0, 4.5] s using the 2-model CV + Harmonic IMM predictor.
        """
        return predict_track_trajectory(
            track,
            taus=taus,
            beta=self.beta,
            q_acc_base=self.q_acc_base,
            enable_harmonic=self.enable_spatiotemporal_dynamic_control,
        )

    def predict_future(
        self, track: TrackedObstacle, tau: float
    ) -> Tuple[float, float, float, np.ndarray]:
        """
        Predict future position (px, py), uncertainty-inflated radius r_eff(tau),
        and 2x2 position covariance Sigma_p(tau) at horizon tau >= 0.
        Uses CV+Harmonic IMM blend when harmonic model has non-trivial probability.
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
        if (
            self.enable_spatiotemporal_dynamic_control
            and getattr(track, "imm_valid", False)
            and float(getattr(track, "imm_mu_harm", 0.0)) > 0.01
        ):
            pred = self.predict_trajectory(track, (float(tau),))
            px = float(pred.centers[0, 0])
            py = float(pred.centers[0, 1])
            sig = float(pred.sigmas[0])
            cov_imm = np.diag([sig * sig, sig * sig]).astype(np.float64)
            r_eff = float(track.radius) + min(1.5 * sig, CH3_MAX_R_INFL)
            return px, py, r_eff, cov_imm
        lam_max = max_eigenvalue_2x2(cov_p)
        r_eff = float(track.radius) + min(1.5 * math.sqrt(lam_max), CH3_MAX_R_INFL)
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

        use_default_h = horizons is None and weights is None
        eps = float(self.ch3_eps)
        c = math.cos(rth)
        s = math.sin(rth)
        rx_f = float(rx)
        ry_f = float(ry)

        dt_extrap = 0.0
        cache_hit = (
            now_sec is not None
            and self._extrap_cache_stamp is not None
            and abs(float(now_sec) - self._extrap_cache_stamp) <= 1e-9
        )
        if not cache_hit:
            F_ext = None
            Q_ext_base = None
            if now_sec is not None and self._last_scan_stamp is not None:
                dt_extrap = max(0.0, float(now_sec) - float(self._last_scan_stamp))
                if dt_extrap > 1e-6:
                    dt_k = round(dt_extrap, 4)
                    cached_ext = self._step_fq_cache.get(dt_k)
                    if cached_ext is None:
                        cached_ext = damped_cv_matrices(dt_extrap, beta=self.beta, q_acc=self.q_acc_base)
                        if len(self._step_fq_cache) < 32:
                            self._step_fq_cache[dt_k] = cached_ext
                    F_ext, Q_ext_base = cached_ext
            self._extrap_cache_states.clear()
            if F_ext is not None and now_sec is not None:
                self._extrap_cache_stamp = float(now_sec)
            else:
                self._extrap_cache_stamp = None
        else:
            F_ext = None
            Q_ext_base = None

        if use_default_h:
            F_b = self._ch3_F_batch
            FT_b = self._ch3_FT_batch
            Q_b = self._ch3_Q_batch
            w_b = self._ch3_w_arr
            n_h = len(self.ch3_horizons)
            if n_h == 0:
                return ch3

            max_flat = np.zeros(PATCH_SIZE * PATCH_SIZE, dtype=np.float32)
            has_confirmed = False
            for tid, trk in self.tracks.items():
                if not trk.is_confirmed:
                    continue
                has_confirmed = True
                r_c = float(trk.radius)
                if cache_hit and tid in self._extrap_cache_states:
                    s_cur, P_cur = self._extrap_cache_states[tid]
                else:
                    s_cur = np.array([trk.x, trk.y, trk.vx, trk.vy], dtype=np.float64)
                    P_cur = trk.P_4x4
                    if F_ext is not None:
                        s_cur = F_ext @ s_cur
                        P_cur = F_ext @ P_cur @ F_ext.T + trk.q_acc_scale * Q_ext_base
                        P_cur = 0.5 * (P_cur + P_cur.T)
                        if now_sec is not None:
                            self._extrap_cache_states[tid] = (s_cur, P_cur)

                s_h = F_b @ s_cur  # (H, 4)
                P_h = F_b @ P_cur @ FT_b + trk.q_acc_scale * Q_b  # (H, 4, 4)
                dx_w = s_h[:, 0] - rx_f
                dy_w = s_h[:, 1] - ry_f
                ex = (c * dx_w + s * dy_w).astype(np.float32)[:, None]
                ey = (-s * dx_w + c * dy_w).astype(np.float32)[:, None]

                a = P_h[:, 0, 0]
                b = 0.5 * (P_h[:, 0, 1] + P_h[:, 1, 0])
                cp = P_h[:, 1, 1]
                disc = np.sqrt(np.maximum(0.0, (a - cp) ** 2 + 4.0 * b * b))
                lam_max = np.maximum(0.0, 0.5 * (a + cp + disc))
                r_eff = r_c + np.minimum(1.5 * np.sqrt(lam_max), CH3_MAX_R_INFL)
                denom = np.maximum(r_eff - r_c, eps).astype(np.float32)[:, None]

                gx = _EGO_FLAT_X[None, :] - ex
                gy = _EGO_FLAT_Y[None, :] - ey
                dist = np.sqrt(gx * gx + gy * gy)
                ratio = np.maximum(np.float32(0.0), dist - np.float32(r_c)) / denom
                val = w_b[:, None] * np.maximum(np.float32(0.0), np.minimum(np.float32(1.0), np.float32(1.0) - ratio))
                max_flat = np.maximum(max_flat, val.max(axis=0))

            if not has_confirmed:
                return ch3
            return np.minimum(np.float32(1.0), np.maximum(np.float32(0.0), max_flat)).reshape(PATCH_SIZE, PATCH_SIZE)

        hs = self.ch3_horizons if horizons is None else horizons
        ws = self.ch3_weights if weights is None else weights
        ego_xs: List[float] = []
        ego_ys: List[float] = []
        rcs: List[float] = []
        denoms: List[float] = []
        wvals: List[float] = []

        for tid, trk in self.tracks.items():
            if not trk.is_confirmed:
                continue
            r_c = float(trk.radius)
            if tid in self._extrap_cache_states and self._extrap_cache_stamp is not None and now_sec is not None and abs(float(now_sec) - self._extrap_cache_stamp) <= 1e-9:
                s_cur, P_cur = self._extrap_cache_states[tid]
            else:
                s_cur = np.array([trk.x, trk.y, trk.vx, trk.vy], dtype=np.float64)
                P_cur = trk.P_4x4
                if F_ext is not None:
                    s_cur = F_ext @ s_cur
                    P_cur = F_ext @ P_cur @ F_ext.T + trk.q_acc_scale * Q_ext_base
                    P_cur = 0.5 * (P_cur + P_cur.T)
                    if now_sec is not None:
                        self._extrap_cache_states[tid] = (s_cur, P_cur)

            for tau, w_tau in zip(hs, ws):
                cached = self._ch3_fq_cache.get(float(tau))
                if cached is not None:
                    F_h, Q_h = cached
                    s_t = F_h @ s_cur
                    P_t = F_h @ P_cur @ F_h.T + trk.q_acc_scale * Q_h
                else:
                    s_t, P_t = propagate_state_cov(
                        s_cur,
                        P_cur,
                        float(tau),
                        beta=self.beta,
                        q_acc=self.q_acc_base * trk.q_acc_scale,
                    )
                lam_max = max_eigenvalue_2x2(P_t[:2, :2])
                r_eff = r_c + min(1.5 * math.sqrt(lam_max), CH3_MAX_R_INFL)
                dx = float(s_t[0]) - rx_f
                dy = float(s_t[1]) - ry_f
                ego_xs.append(c * dx + s * dy)
                ego_ys.append(-s * dx + c * dy)
                rcs.append(r_c)
                denoms.append(max(r_eff - r_c, eps))
                wvals.append(float(w_tau))

        if not ego_xs:
            return ch3

        ex = np.asarray(ego_xs, dtype=np.float32)[:, None, None]
        ey = np.asarray(ego_ys, dtype=np.float32)[:, None, None]
        rc = np.asarray(rcs, dtype=np.float32)[:, None, None]
        dn = np.asarray(denoms, dtype=np.float32)[:, None, None]
        wv = np.asarray(wvals, dtype=np.float32)[:, None, None]

        dist_grid = np.hypot(_EGO_GRID_X[None, :, :] - ex, _EGO_GRID_Y[None, :, :] - ey)
        val = wv * np.clip(
            np.float32(1.0) - np.maximum(dist_grid - rc, np.float32(0.0)) / dn,
            np.float32(0.0),
            np.float32(1.0),
        )
        ch3 = val.max(axis=0)
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
