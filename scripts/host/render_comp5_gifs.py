#!/usr/bin/env python3
"""
Render 5-Stage Autonomous Return Comparison GIFs (comp5_*.gif):
  Baseline -> Phase 1 -> Phase 2-A -> Phase 2-B -> Phase 3

Supports:
  1. --mode classic : Render 5-way comparison GIFs for smoke_lidar_corruption,
                      dynamic_crossfire_alley, and u_pocket_trap using matched seeds.
  2. --mode semantic: Render 5-way Semantic Rescue Arena comparison GIF on top of
                      the real slam_toolbox /map OccupancyGrid with YOLO26n survivor
                      & RViz2 semantic map coordinate stamping visualization.
"""

import argparse
import json
import math
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from plot_scenario_routes_15_atlas import ensure_reference_model_cache  # noqa: E402
from world_map_geometry import extract_world_footprint  # noqa: E402

WS_WORLDS = os.path.join(WS_ROOT, "src", "disaster_recovery", "worlds")
BM_ROOT = os.path.join(WS_ROOT, "benchmark_results")
HDD_EXP = "/hdd/rl_v2_exp" if os.path.exists("/hdd/rl_v2_exp") else "/home/jwroh/hdd/rl_v2_exp"
OUT_EVIDENCE = os.path.join(WS_ROOT, "docs", "images", "evidence")


def _select_trial(records: List[dict], trial_num: int) -> dict:
    for rec in records:
        if int(rec.get("trial", -1)) == int(trial_num):
            return rec
    raise RuntimeError(f"Trial #{trial_num} not found in JSON records")


def _resample_path(pts: np.ndarray, n_steps: int) -> np.ndarray:
    """Resample 2D polyline to n_steps evenly spaced arc-length samples."""
    if pts.ndim != 2 or len(pts) == 0:
        return np.zeros((n_steps, 2), dtype=float)
    if len(pts) == 1:
        return np.repeat(pts[:1, :2], n_steps, axis=0)
    xy = pts[:, :2].astype(float)
    diffs = np.diff(xy, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cum[-1])
    if total < 1e-6:
        return np.repeat(xy[:1], n_steps, axis=0)
    target = np.linspace(0.0, total, n_steps)
    rx = np.interp(target, cum, xy[:, 0])
    ry = np.interp(target, cum, xy[:, 1])
    return np.column_stack([rx, ry])


def _encode_fig_stream_to_gif(
    frames_cb,
    n_anim_frames: int,
    n_hold_frames: int,
    w_px: int,
    h_px: int,
    fps: int,
    out_gif: str,
) -> None:
    """Render frames directly into an MP4 buffer via ffmpeg and convert to crisp palette GIF."""
    os.makedirs(os.path.dirname(os.path.abspath(out_gif)), exist_ok=True)
    tmp_mp4 = out_gif + ".tmp.mp4"
    tmp_pal = out_gif + ".pal.png"
    dpi = 100
    fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi, facecolor="#0b1220")
    ff_cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w_px}x{h_px}", "-pix_fmt", "rgba", "-r", str(fps),
        "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
        tmp_mp4,
    ]
    proc = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        seq = list(range(n_anim_frames)) + [n_anim_frames - 1] * n_hold_frames
        for seq_idx, anim_idx in enumerate(seq):
            fig.clf()
            frames_cb(fig, anim_idx, n_anim_frames, seq_idx)
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
            if proc.stdin is not None:
                proc.stdin.write(buf.tobytes())
    finally:
        plt.close(fig)
        if proc.stdin is not None:
            proc.stdin.close()
        proc.wait(timeout=30.0)

    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", tmp_mp4, "-vf", f"fps={fps},palettegen=max_colors=128:stats_mode=diff", tmp_pal],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", tmp_mp4, "-i", tmp_pal, "-lavfi", f"fps={fps}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3", out_gif],
        check=True,
    )
    for p in (tmp_mp4, tmp_pal):
        if os.path.exists(p):
            os.remove(p)
    print(f"[SAVED GIF] {out_gif} ({os.path.getsize(out_gif)} B)", flush=True)


def _format_outcome_badge(rec: dict, fallback_key: str) -> Tuple[bool, str]:
    suc = bool(rec.get("strict_hazard_free_success", rec.get("success", False)))
    dur = float(rec.get("return_duration_s", 0.0))
    err = float(rec.get("terminal_error_cm", 0.0))
    if "low_obstacle_collision" in rec or "gas_zone_violated" in rec:
        low_hit = bool(rec.get("low_obstacle_collision", False))
        gas_hit = bool(rec.get("gas_zone_violated", False))
        if suc:
            tag = "PASS"
        elif low_hit and gas_hit:
            tag = "LOW-OBS+GAS HIT"
        elif low_hit:
            tag = "LOW-OBS HIT"
        elif gas_hit:
            tag = "GAS VIOLATED"
        elif not bool(rec.get("arrival_success", rec.get("success", False))):
            tag = "FAIL"
        else:
            tag = "COLLISION"
        return suc, f"[{tag} | {dur:.1f}s | err={err:.1f}cm]"

    tag = "PASS" if suc else "FAIL"
    return suc, f"[{tag} | {dur:.1f}s | err={err:.1f}cm]"


def render_classic_comp5(model_roots: str) -> None:
    specs = [
        (
            "smoke_lidar_corruption",
            1,
            "#15 Smoke LiDAR Hazard (Trial #01 | Seed 2127) — 5-Stage Autonomous Return Comparison",
            "disaster_smoke_hazard_world.world",
            (-1.5, 11.2, -2.5, 2.5),
            "comp5_smoke_lidar_corruption.gif",
        ),
        (
            "dynamic_crossfire_alley",
            1,
            "#14 Dynamic Crossfire Alley (Trial #01 | Seed 2127) — 5-Stage Return vs Moving Gate Pillars",
            "disaster_dynamic_crossfire_world.world",
            (-1.5, 11.2, -2.5, 2.5),
            "comp5_dynamic_crossfire_alley.gif",
        ),
        (
            "u_pocket_trap",
            10,
            "#11 U-Pocket Trap (Trial #10 | Seed 3036) — 5-Stage Autonomous Return Comparison",
            "disaster_u_pocket_world.world",
            (-1.3, 11.4, -3.5, 3.5),
            "comp5_u_pocket_trap.gif",
        ),
    ]

    for sc, tid, title, wfile, roi, out_name in specs:
        if sc == "u_pocket_trap":
            p1_path = os.path.join(HDD_EXP, "bench_hard8_10trials", "phase1", f"ab_{sc}_10trials.json")
            p2a_path = os.path.join(HDD_EXP, "bench_hard8_10trials", "phase2a", f"ab_{sc}_10trials.json")
        else:
            p1_path = os.path.join(HDD_EXP, "bench_hard8_phase1", f"ab_{sc}_10trials.json")
            p2a_path = os.path.join(HDD_EXP, "bench_hard8_phase2a", f"ab_{sc}_10trials.json")
        p2b_path = os.path.join(BM_ROOT, "phase2b_strict", f"ab_{sc}_10trials.json")
        p3_path = os.path.join(BM_ROOT, "phase3_semantic_v3", f"ab_{sc}_10trials.json")

        j_p1 = json.load(open(p1_path, "r", encoding="utf-8"))
        j_p2a = json.load(open(p2a_path, "r", encoding="utf-8"))
        j_p2b = json.load(open(p2b_path, "r", encoding="utf-8"))
        j_p3 = json.load(open(p3_path, "r", encoding="utf-8"))

        stages = [
            ("Baseline\n(4-Tier Reactive)", _select_trial(j_p2b["baseline"]["trials"], tid), "#ef4444"),
            ("Phase 1\n(Static Grid A*)", _select_trial(j_p1["proposed"]["trials"], tid), "#3b82f6"),
            ("Phase 2-A\n(Naive RL Blend)", _select_trial(j_p2a["proposed"]["trials"], tid), "#f97316"),
            ("Phase 2-B\n(Safety + 1ch RL)", _select_trial(j_p2b["proposed"]["trials"], tid), "#10b981"),
            ("Phase 3\n(3ch Semantic RL)", _select_trial(j_p3["proposed"]["trials"], tid), "#22c55e"),
        ]

        fp = extract_world_footprint(os.path.join(WS_WORLDS, wfile), models_root=model_roots, z_slice=0.17)
        polys = [p for _, p in fp.polygons if len(p) >= 3]
        segs = [s for _, s in fp.segments]

        max_dur = max(float(s[1].get("return_duration_s", 60.0)) for s in stages)
        n_frames = 54 if sc == "dynamic_crossfire_alley" else 42
        resampled = []
        for _, rec, _ in stages:
            pts = np.asarray(rec.get("return_path", []), dtype=float)
            # Proportionally pace trajectories relative to max_dur so faster policies finish earlier!
            dur = max(5.0, float(rec.get("return_duration_s", max_dur)))
            active_frames = max(6, int(round(n_frames * min(1.0, dur / max_dur))))
            path_active = _resample_path(pts, active_frames)
            if active_frames < n_frames:
                pad = np.repeat(path_active[-1:], n_frames - active_frames, axis=0)
                path_full = np.vstack([path_active, pad])
            else:
                path_full = path_active
            resampled.append(path_full)

        debris = stages[4][1].get("debris_positions", []) or stages[3][1].get("debris_positions", [])
        out_path = np.asarray(stages[4][1].get("outbound_path", []), dtype=float)
        turnoff = stages[4][1].get("turnoff_pose") or (out_path[-1].tolist() if len(out_path) else [8.0, 0.0])

        def _draw_frame(fig, f_idx: int, total_f: int, _seq: int):
            axes = fig.subplots(1, 5)
            t_sim = (f_idx / max(1, total_f - 1)) * max_dur
            for i, (ax, (st_label, rec, col), r_pts) in enumerate(zip(axes, stages, resampled)):
                ax.set_facecolor("#162032")
                if polys:
                    ax.add_collection(PolyCollection(polys, facecolor="#475569", edgecolor="#94a3b8", linewidths=0.85, alpha=0.92, zorder=1))
                if segs:
                    ax.add_collection(LineCollection(segs, colors="#94a3b8", linewidths=0.8, alpha=0.85, zorder=1))

                if sc == "dynamic_crossfire_alley":
                    # Animated oscillating pillars matching Gazebo crossfire world timing
                    amp = 0.36 / 0.95
                    y1 = 0.00 + amp * math.sin(0.95 * t_sim)
                    y2 = 0.725 + amp * math.sin(0.95 * t_sim + 1.57)
                    ax.plot([2.15, 2.15], [-amp, amp], color="#ef4444", ls=":", lw=1.2, alpha=0.7, zorder=2)
                    ax.plot([5.05, 5.05], [0.725 - amp, 0.725 + amp], color="#f97316", ls=":", lw=1.2, alpha=0.7, zorder=2)
                    ax.add_patch(mpatches.Circle((2.15, y1), 0.21, facecolor="#ef4444", edgecolor="#fecaca", lw=1.2, alpha=0.92, zorder=4))
                    ax.add_patch(mpatches.Circle((5.05, y2), 0.21, facecolor="#f97316", edgecolor="#fed7aa", lw=1.2, alpha=0.92, zorder=4))

                for dx, dy in debris:
                    ax.add_patch(mpatches.Circle((dx, dy), 0.22, facecolor="#9333ea", edgecolor="#e9d5ff", lw=0.9, alpha=0.85, zorder=3))

                if len(out_path) >= 2:
                    ax.plot(out_path[:, 0], out_path[:, 1], color="#94a3b8", ls="--", lw=1.0, alpha=0.55, zorder=3)
                ax.scatter([0.0], [0.0], marker="*", s=150, color="#38bdf8", edgecolor="#ffffff", lw=0.8, zorder=8)
                ax.scatter([turnoff[0]], [turnoff[1]], marker="X", s=65, color="#f59e0b", edgecolor="#0b1220", lw=0.6, zorder=8)

                cur_traj = r_pts[: f_idx + 1]
                ax.plot(cur_traj[:, 0], cur_traj[:, 1], color=col, lw=2.3, alpha=0.95, zorder=6)
                ax.scatter([cur_traj[-1, 0]], [cur_traj[-1, 1]], s=65, color=col, edgecolor="#ffffff", lw=1.3, zorder=9)

                is_ok, badge = _format_outcome_badge(rec, st_label)
                t_col = "#4ade80" if is_ok else "#f87171"
                ax.set_title(f"{st_label}\n{badge}", color=t_col, fontsize=8.8, fontweight="bold", pad=6)
                ax.set_xlim(roi[0], roi[1])
                ax.set_ylim(roi[2], roi[3])
                ax.set_aspect("equal", adjustable="box")
                ax.grid(True, ls=":", color="#334155", alpha=0.55)
                ax.tick_params(colors="#94a3b8", labelsize=7.0)

            legend_handles = [
                mlines.Line2D([], [], color="#38bdf8", marker="*", markersize=9, ls="None", label="Base Camp"),
                mlines.Line2D([], [], color="#f59e0b", marker="X", markersize=7, ls="None", label="Blackout"),
                mlines.Line2D([], [], color="#94a3b8", ls="--", lw=1.2, label="Outbound"),
                mlines.Line2D([], [], color="#9333ea", marker="o", markersize=7, ls="None", label="Debris"),
            ]
            if sc == "dynamic_crossfire_alley":
                legend_handles.append(mlines.Line2D([], [], color="#ef4444", marker="o", markersize=7, ls=":", label="Moving Gate Pillars"))
            fig.legend(
                handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.905),
                ncol=len(legend_handles), fontsize=8.2, frameon=True,
                facecolor="#162032", edgecolor="#334155", labelcolor="#e2e8f0",
            )
            fig.suptitle(f"{title} (t={t_sim:.1f}s)", color="#f8fafc", fontsize=12.5, fontweight="bold", y=0.975)
            fig.tight_layout(rect=[0.005, 0.01, 0.995, 0.84])

        _encode_fig_stream_to_gif(
            _draw_frame,
            n_anim_frames=n_frames,
            n_hold_frames=14,
            w_px=1900,
            h_px=470,
            fps=12,
            out_gif=os.path.join(OUT_EVIDENCE, out_name),
        )


def render_semantic_comp5(json_path: str, out_gif: str, model_roots: str) -> None:
    """Render 5-stage Semantic Rescue Arena GIF over real slam_toolbox /map + RViz2 semantic coord stamps."""
    data = json.load(open(json_path, "r", encoding="utf-8"))
    occ = data["slam_map"]
    w = int(occ["width"])
    h = int(occ["height"])
    res = float(occ["resolution"])
    ox = float(occ["origin_x"])
    oy = float(occ["origin_y"])
    grid = np.asarray(occ["data"], dtype=np.int16).reshape((h, w))

    # Dark-mode SLAM OccupancyGrid RGB overlay so it looks cohesive with comp4/comp5 style
    rgb = np.full((h, w, 3), [17, 24, 39], dtype=np.uint8)       # Unknown (-1): deep navy
    rgb[grid == 0] = [30, 41, 59]                                 # Free space (0): dark slate
    rgb[grid > 50] = [203, 213, 225]                              # Occupied walls (>50): bright light-slate
    extent = [ox, ox + w * res, oy, oy + h * res]

    fp = extract_world_footprint(os.path.join(WS_WORLDS, "disaster_semantic_rescue_world.world"), models_root=model_roots, z_slice=0.17)
    polys = [p for _, p in fp.polygons if len(p) >= 3]

    stage_keys = [
        ("baseline", "Baseline\n(4-Tier Reactive)", "#ef4444", False),
        ("phase1", "Phase 1\n(Static Grid A*)", "#3b82f6", False),
        ("phase2a", "Phase 2-A\n(Naive RL Blend)", "#f97316", False),
        ("phase2b", "Phase 2-B\n(Safety + 1ch RL)", "#10b981", False),
        ("phase3", "Phase 3\n(Semantic SLAM + 3ch)", "#22c55e", True),
    ]

    stages = []
    for key, label, col, sem_on in stage_keys:
        rec = data["strategies"][key]
        stages.append((key, label, col, sem_on, rec))

    out_pts_ref = np.asarray(data["outbound_path"], dtype=float)
    low_list = data["low_obstacles"]
    box_list = data.get("box_obstacles", [])
    gas_xy = data["gas_zone"]
    survivor_xy = data.get("survivor_pose", [5.85, 0.45])
    stamped = data.get("stamped_map_objects", [])
    turnoff = data.get("turnoff_pose", out_pts_ref[-1].tolist() if len(out_pts_ref) else [4.95, 0.35])

    max_dur = max(float(s[4].get("return_duration_s", 40.0)) for s in stages)
    n_out_frames = 14
    n_ret_frames = 44
    total_frames = n_out_frames + n_ret_frames

    out_anim = _resample_path(out_pts_ref, n_out_frames)
    ret_anims = []
    for _, _, _, _, rec in stages:
        r_pts = np.asarray(rec.get("return_path", []), dtype=float)
        dur = max(4.0, float(rec.get("return_duration_s", max_dur)))
        act_f = max(6, int(round(n_ret_frames * min(1.0, dur / max_dur))))
        p_act = _resample_path(r_pts, act_f)
        if act_f < n_ret_frames:
            p_act = np.vstack([p_act, np.repeat(p_act[-1:], n_ret_frames - act_f, axis=0)])
        ret_anims.append(p_act)

    roi = (-1.35, 7.75, -2.55, 2.55)

    def _draw_semantic_frame(fig, f_idx: int, _total_f: int, _seq: int):
        axes = fig.subplots(1, 5)
        is_ret = f_idx >= n_out_frames
        r_idx = f_idx - n_out_frames
        progress_out = min(1.0, (f_idx + 1) / float(n_out_frames))
        phase_str = f"Return t={max(0.0, (r_idx / max(1, n_ret_frames - 1)) * max_dur):.1f}s" if is_ret else "Outbound Mapping + YOLO26n Stamp"

        for ax, (key, label, col, sem_on, rec), r_anim in zip(axes, stages, ret_anims):
            ax.set_facecolor("#0f172a")
            # 1. Real slam_toolbox /map OccupancyGrid layer + Gazebo perimeter walls
            ax.imshow(rgb, origin="lower", extent=extent, interpolation="nearest", alpha=0.88, zorder=0)
            if polys:
                ax.add_collection(PolyCollection(polys, facecolor="none", edgecolor="#94a3b8", linewidths=0.75, alpha=0.65, zorder=1))

            # 2. Ground-truth obstacles in scene (3D box spawns upon secondary collapse at blackout)
            if is_ret:
                for bx, by in box_list:
                    ax.add_patch(mpatches.Rectangle((bx - 0.21, by - 0.21), 0.42, 0.42, facecolor="#d97706", edgecolor="#fef3c7", lw=1.0, alpha=0.95, zorder=5))

            for idx_l, (lx, ly) in enumerate(low_list, start=1):
                if sem_on:
                    ax.add_patch(mpatches.Circle((lx, ly), 0.22, facecolor="#ec4899", edgecolor="#fbcfe8", lw=1.2, alpha=0.88, zorder=4))
                else:
                    # Blind models cannot see sub-LiDAR low obstacles on /scan
                    ax.add_patch(mpatches.Circle((lx, ly), 0.22, facecolor="#ec4899", edgecolor="#fbcfe8", ls=":", lw=1.0, alpha=0.28, zorder=3))

            if sem_on:
                ax.add_patch(mpatches.Circle(gas_xy, 0.55, facecolor="#ea580c", edgecolor="#fdba74", ls="--", lw=1.3, alpha=0.25, zorder=3))
                ax.scatter([gas_xy[0]], [gas_xy[1]], marker="D", s=55, color="#f97316", edgecolor="#ffffff", lw=0.9, zorder=5)
            else:
                ax.add_patch(mpatches.Circle(gas_xy, 0.55, facecolor="#ea580c", edgecolor="#fdba74", ls=":", lw=1.0, alpha=0.12, zorder=2))
                ax.scatter([gas_xy[0]], [gas_xy[1]], marker="D", s=45, color="#f97316", alpha=0.40, zorder=4)

            # Survivor person marker
            ax.scatter([survivor_xy[0]], [survivor_xy[1]], marker="o", s=95, color="#ef4444", edgecolor="#ffffff", lw=1.1, zorder=6)
            ax.add_patch(mpatches.Circle(survivor_xy, 0.45, facecolor="#ef4444", edgecolor="#fca5a5", ls=":" if not sem_on else "--", lw=1.0, alpha=0.14 if sem_on else 0.06, zorder=2))

            # 3. RViz2 /map Stamped Coordinates HUD (Phase 3 stamps coordinates during Outbound and persists them across blackout)
            if sem_on:
                # Filter out low_confidence duplicates and keep crisp single label per object/cluster
                clean_stamps = [o for o in stamped if not o.get("low_confidence", False)]
                low_idx = 0
                for st_obj in clean_stamps:
                    cname = str(st_obj.get("class_name", "obj"))
                    sx, sy = float(st_obj["x"]), float(st_obj["y"])
                    appear_prog = 0.85 if cname == "person" else (0.55 if "GAS" in cname else 0.40)
                    if progress_out >= appear_prog:
                        if cname == "person":
                            txt_c, dx_t, dy_t = "#fca5a5", 0.0, -0.95
                            lbl = f"person ({sx:.2f},{sy:.2f})"
                        elif "GAS" in cname:
                            txt_c, dx_t, dy_t = "#fdba74", 0.0, -1.05
                            lbl = f"GAS ({sx:.2f},{sy:.2f})"
                        else:
                            txt_c = "#f9a8d4"
                            dx_t = -0.30 if low_idx == 0 else 0.40
                            dy_t = 0.85 if low_idx == 0 else 0.65
                            low_idx += 1
                            lbl = f"low ({sx:.2f},{sy:.2f})"
                        ax.text(
                            sx + dx_t, sy + dy_t, lbl, color="#f8fafc", fontsize=5.7, fontweight="bold", ha="center",
                            bbox=dict(boxstyle="round,pad=0.16", fc="#0b1220", ec=txt_c, lw=0.8, alpha=0.92),
                            zorder=10,
                        )
                mode_tag = "RViz2 /map Stamped -> Cached" if is_ret else "YOLO26n + LiDAR Stamping..."
                mode_col = "#38bdf8"
            else:
                mode_tag = "Blind: No Semantic Map"
                mode_col = "#94a3b8"

            ax.text(
                0.02, 0.03, mode_tag, transform=ax.transAxes, color=mode_col, fontsize=6.6, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", fc="#0b1220", ec=mode_col, lw=0.7, alpha=0.85),
                zorder=11,
            )

            # 4. Outbound + Blackout + Return Trajectory
            ax.scatter([0.0], [0.0], marker="*", s=135, color="#38bdf8", edgecolor="#ffffff", lw=0.7, zorder=8)
            if not is_ret:
                seg_out = out_anim[: f_idx + 1]
                ax.plot(seg_out[:, 0], seg_out[:, 1], color="#38bdf8", ls="--", lw=1.3, alpha=0.85, zorder=5)
                rx, ry = seg_out[-1, 0], seg_out[-1, 1]
                ax.scatter([rx], [ry], s=60, color="#38bdf8", edgecolor="#ffffff", lw=1.2, zorder=9)
            else:
                ax.plot(out_pts_ref[:, 0], out_pts_ref[:, 1], color="#64748b", ls="--", lw=1.0, alpha=0.55, zorder=4)
                ax.scatter([turnoff[0]], [turnoff[1]], marker="X", s=60, color="#f59e0b", edgecolor="#0b1220", lw=0.5, zorder=8)
                seg_ret = r_anim[: r_idx + 1]
                ax.plot(seg_ret[:, 0], seg_ret[:, 1], color=col, lw=2.3, alpha=0.95, zorder=7)
                rx, ry = seg_ret[-1, 0], seg_ret[-1, 1]
                ax.scatter([rx], [ry], s=65, color=col, edgecolor="#ffffff", lw=1.3, zorder=9)

                # Highlight real-time violation if Blind robot runs over low obstacle or gas zone
                hit_low = any(math.hypot(px - lx, py - ly) < 0.255 for px, py in seg_ret for lx, ly in low_list)
                hit_gas = any(math.hypot(px - gas_xy[0], py - gas_xy[1]) < 0.55 for px, py in seg_ret)
                alerts = []
                if hit_low:
                    alerts.append("LOW-OBS HIT!")
                if hit_gas:
                    alerts.append("GAS HIT!")
                if alerts:
                    ax.text(
                        0.50, 0.91, " + ".join(alerts), transform=ax.transAxes,
                        ha="center", color="#fef2f2", fontsize=7.0, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.20", fc="#dc2626", ec="#fecaca", lw=0.9, alpha=0.92),
                        zorder=12,
                    )

            is_ok, badge = _format_outcome_badge(rec, key)
            t_col = "#4ade80" if is_ok else "#f87171"
            ax.set_title(f"{label}\n{badge}", color=t_col, fontsize=8.6, fontweight="bold", pad=6)
            ax.set_xlim(roi[0], roi[1])
            ax.set_ylim(roi[2], roi[3])
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, ls=":", color="#334155", alpha=0.45)
            ax.tick_params(colors="#94a3b8", labelsize=6.8)

        legend_handles = [
            mlines.Line2D([], [], color="#38bdf8", marker="*", markersize=9, ls="None", label="Base Camp (0,0)"),
            mlines.Line2D([], [], color="#f59e0b", marker="X", markersize=7, ls="None", label="Blackout"),
            mlines.Line2D([], [], color="#ef4444", marker="o", markersize=8, ls="None", label="Survivor (person)"),
            mlines.Line2D([], [], color="#ec4899", marker="o", markersize=7, ls="None", label="Low-Obs (h=0.10m, LiDAR Blind)"),
            mlines.Line2D([], [], color="#f97316", marker="D", markersize=7, ls="None", label="Gas Hazard (r=0.55m)"),
            mlines.Line2D([], [], color="#d97706", marker="s", markersize=7, ls="None", label="3D Box (LiDAR Visible)"),
            mlines.Line2D([], [], color="#cbd5e1", marker="s", markersize=7, ls="None", label="SLAM /map Walls"),
        ]
        fig.legend(
            handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.905),
            ncol=7, fontsize=7.9, frameon=True,
            facecolor="#162032", edgecolor="#334155", labelcolor="#e2e8f0",
        )
        fig.suptitle(
            f"Semantic Rescue Arena (/map Frame + Survivor & Hazard Coords) — 5-Stage Return Comparison ({phase_str})",
            color="#f8fafc", fontsize=12.0, fontweight="bold", y=0.975,
        )
        fig.tight_layout(rect=[0.005, 0.01, 0.995, 0.84])

    _encode_fig_stream_to_gif(
        _draw_semantic_frame,
        n_anim_frames=total_frames,
        n_hold_frames=16,
        w_px=1950,
        h_px=490,
        fps=12,
        out_gif=out_gif,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Render 5-Stage comparison GIFs (comp5_*.gif)")
    parser.add_argument("--mode", choices=["classic", "semantic"], required=True)
    parser.add_argument("--semantic-json", default=os.path.join(BM_ROOT, "phase3_semantic_v3", "comp5_semantic_rescue.json"))
    parser.add_argument("--out-gif", default=os.path.join(OUT_EVIDENCE, "comp5_semantic_rescue_map.gif"))
    args = parser.parse_args()

    model_roots = ensure_reference_model_cache(WS_ROOT)
    if args.mode == "classic":
        render_classic_comp5(model_roots)
    else:
        render_semantic_comp5(args.semantic_json, args.out_gif, model_roots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
