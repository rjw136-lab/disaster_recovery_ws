#!/usr/bin/env python3
"""
Phase 4 Real (4-Ch RL + 2-Circle Soft-CBF-QP + Semantic A*) Full E2E 3D Recording.

Records the complete Trial #10 (seed=3336) scenario in Gazebo 3D + RViz2 (/map):
  - low_obstacles = [(2.09, -0.04), (3.58, 0.0)]
  - gas_zone      = (2.65, 0.32)
  - improved north outbound safe path:
      [(0.9, 0.0), (1.55, 0.55), (2.7, 0.60), (3.95, 0.55), (4.45, 0.30), (4.95, 0.25)]
  - recovery_controller params:
      enable_dynamic_tracker:=true, enable_cbf_safety_shield:=true, cbf_v_min:=-0.08,
      enable_semantic_astar_cost:=true, phase4_rl_onnx_path:=real_tb3_policy_student_phase4.onnx

Outputs MP4, GIF, representative keyframe PNGs, 2D trajectory overlay PNG, and metrics JSON.
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rclpy
from std_msgs.msg import String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ab_disaster_benchmark import (  # noqa: E402
    _kill_matching_gazebo_procs,
    get_gazebo_true_pose,
    run_cmd_bounded,
)
from record_3d_evidence import (  # noqa: E402
    _build_base_camp_sdf,
    _build_ribbon_sdf,
    _ensure_gazebo_gui_ini,
    _spawn_gazebo_visual,
    _subsampled_pts,
)
from run_phase3_semantic_suite import (  # noqa: E402
    SEMANTIC_WORLD,
    SemanticSuiteNode,
    _annotate_mp4_and_gif,
    _arrange_3pane_windows,
    _gas_disc_sdf,
    _low_obstacle_sdf,
    _make_rviz_semantic_config,
    _spawn_model_at,
    _stop_procs,
    compute_semantic_world_min_wall_geom,
)
from run_phase4_semantic_benchmark import DEFAULT_P4_REAL_ONNX  # noqa: E402


TRIAL10_LOW_OBS: List[Tuple[float, float]] = [(2.09, -0.04), (3.58, 0.0)]
TRIAL10_GAS_XY: Tuple[float, float] = (2.65, 0.32)
IMPROVED_OUT_WPS: List[Tuple[float, float]] = [
    (0.9, 0.0),
    (1.55, 0.55),
    (2.7, 0.60),
    (3.95, 0.55),
    (4.45, 0.30),
    (4.95, 0.25),
]


def _render_phase4_trial10_overlay(
    out_png: str,
    outbound_pts: Sequence[Tuple[float, float]],
    return_pts: Sequence[Tuple[float, float]],
    low_list: Sequence[Tuple[float, float]],
    gas_xy: Tuple[float, float],
    metrics: dict,
) -> None:
    """Render a crisp 2D top-down trajectory & safety-clearance overlay for Phase 4 Trial #10."""
    fig, ax = plt.subplots(figsize=(12.2, 6.2), dpi=160)

    # Outer walls [-1.2, 7.6] x [-2.4, 2.4] and Partition 1/2 + Pillars
    walls = [
        ([-1.2, 7.6], [2.4, 2.4]),
        ([-1.2, 7.6], [-2.4, -2.4]),
        ([-1.2, -1.2], [-2.4, 2.4]),
        ([7.6, 7.6], [-2.4, 2.4]),
        ([1.45, 1.45], [0.945, 2.4]),
        ([1.45, 1.45], [-2.4, -0.745]),
        ([4.35, 4.35], [0.745, 2.4]),
        ([4.35, 4.35], [-2.4, -0.945]),
    ]
    for wx, wy in walls:
        ax.plot(wx, wy, color="#1e293b", linewidth=5.0, solid_capstyle="butt", zorder=3)

    for px, py in [(0.4, 1.65), (3.0, -1.75), (6.6, -1.55)]:
        ax.add_patch(plt.Rectangle((px - 0.14, py - 0.14), 0.28, 0.28, color="#334155", zorder=4))

    for i_l, (lx, ly) in enumerate(low_list):
        lbl = "Low Obstacle (Ch1, r=0.255m)" if i_l == 0 else None
        ax.add_patch(plt.Rectangle((lx - 0.19, ly - 0.13), 0.38, 0.26, fc="#db2777", ec="#831843", lw=1.2, alpha=0.88, zorder=5, label=lbl))
        ax.add_patch(plt.Circle((lx, ly), 0.255, ec="#db2777", fc="none", ls="--", lw=1.1, alpha=0.8, zorder=4))

    gx, gy = gas_xy
    ax.add_patch(plt.Circle((gx, gy), 0.55, fc="#ea580c", ec="#9a3412", alpha=0.26, lw=2.0, zorder=4, label="Gas Hazard Zone (hr=0.55m)"))
    ax.add_patch(plt.Circle((gx, gy), 0.72, fc="none", ec="#f97316", ls=":", lw=1.6, zorder=4, label="Ch2 Soft-CBF Halo (r_safe~0.72m)"))
    ax.scatter([gx], [gy], marker="D", s=110, color="#ea580c", ec="#ffffff", lw=1.2, zorder=6)

    ax.scatter([5.85], [0.45], marker="o", s=170, color="#dc2626", ec="#ffffff", lw=1.4, zorder=6, label="Survivor (5.85, 0.45)")
    ax.scatter([0.0], [0.0], marker="*", s=280, color="#22c55e", ec="#0f172a", lw=1.2, zorder=7, label="Base Camp (0.00, 0.00)")

    if len(outbound_pts) >= 2:
        ox, oy = zip(*outbound_pts)
        ax.plot(ox, oy, "--", color="#0284c7", lw=2.0, label="Outbound North Safe Path", zorder=6)
    if len(return_pts) >= 2:
        rx, ry = zip(*return_pts)
        ax.plot(rx, ry, "-", color="#16a34a", lw=3.0, label="phase4_real Autonomous Return", zorder=7)

    info = (
        "phase4_real (Trial #10, seed=3336)\n"
        f"ONNX: real_tb3_policy_student_phase4.onnx\n"
        f"CBF v_min={metrics['cbf_v_min']} | SemA*={metrics['enable_semantic_astar_cost']}\n"
        f"min_low={metrics['min_low_obstacle_dist_m']:.3f}m (>=0.255m)\n"
        f"min_gas={metrics['min_gas_zone_dist_m']:.3f}m (>=0.550m)\n"
        f"min_scan={metrics['min_scan_m']:.3f}m | wall_geom={metrics['min_wall_geom_m']:.3f}m\n"
        f"return_time={metrics['return_duration_s']:.2f}s | err={metrics['terminal_error_cm']:.2f}cm\n"
        f"mq2_escape={metrics['cbf_mq2_escape_steps']} | max_dyn={metrics['max_confirmed_dyn_tracks']}"
    )
    ax.text(
        0.015, 0.03, info,
        transform=ax.transAxes,
        fontsize=8.7,
        family="monospace",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.45", fc="#f8fafc", ec="#0f766e", lw=1.5, alpha=0.95),
        zorder=9,
    )

    ax.set_xlim(-1.45, 7.85)
    ax.set_ylim(-2.6, 2.6)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.set_title(
        "Phase 4 Real (4-Ch RL + 2-Circle Soft-CBF-QP + Semantic A*) — Trial #10 Full E2E Trajectory",
        fontsize=12.5, fontweight="bold", pad=10,
    )
    ax.legend(loc="upper right", fontsize=8.4, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_record_phase4(args) -> int:
    """Record full outbound + blackout return of phase4_real on Trial #10 in 3D Gazebo + RViz2."""
    gz_port, dom_id, disp = args.gz_port, args.ros_domain_id, args.display
    gz_uri = f"http://127.0.0.1:{gz_port}"
    _kill_matching_gazebo_procs(gz_uri)
    _ensure_gazebo_gui_ini()

    rviz_tmp = "/tmp/disaster_recovery/p4_real_t10.rviz"
    src_rviz = os.path.join(WS_ROOT, "src", "disaster_recovery", "config", "disaster_station.rviz")
    _make_rviz_semantic_config(src_rviz, rviz_tmp, win_w=700, win_h=1080, win_x=1220, win_y=0)
    with open(rviz_tmp, "r", encoding="utf-8") as f:
        rtxt = f.read()
    rtxt = re.sub(
        r"(- Class: rviz_default_plugins/Image\s+Enabled:\s*)true(.*?Name:\s*Virtual Camera View.*?Value:\s*)true",
        r"\1false\2false",
        rtxt,
        flags=re.DOTALL,
    )
    rtxt = rtxt.replace("Target Frame: base_footprint", "Target Frame: <Fixed Frame>")
    rtxt = rtxt.replace("Distance: 8.2", "Distance: 9.2")
    rtxt = rtxt.replace("Yaw: 0.785398006439209", "Yaw: 1.5708")
    rtxt = rtxt.replace("Pitch: 1.48", "Pitch: 1.56")
    rtxt = re.sub(
        r"(Focal Point:\s*\n\s*X:\s*)[^\n]+\n\s*Y:\s*[^\n]+\n\s*Z:\s*[^\n]+",
        r"\1 2.8\n        Y: 0.1\n        Z: 0.0",
        rtxt,
    )
    with open(rviz_tmp, "w", encoding="utf-8") as f:
        f.write(rtxt)

    os.environ["DISPLAY"] = disp
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    os.environ["ROS_DOMAIN_ID"] = str(dom_id)
    os.environ["GAZEBO_MASTER_URI"] = gz_uri
    os.environ["GAZEBO_MODEL_DATABASE_URI"] = ""
    use_vgl = bool(shutil.which("vglrun")) and os.path.exists("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    vgl_prefix = "vglrun -d egl " if use_vgl else ""
    sw_env = "" if use_vgl else "LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=16 "
    env_prefix = (
        f"unset QT_QPA_PLATFORM_PLUGIN_PATH QT_PLUGIN_PATH && "
        f"source /opt/ros/humble/setup.bash && source {WS_ROOT}/install/setup.bash && "
        f"export DISPLAY={disp} {sw_env}QT_X11_NO_MITSHM=1 "
        f"ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID={dom_id} GAZEBO_MASTER_URI={gz_uri} "
        f"GAZEBO_MODEL_PATH=/opt/ros/humble/share/turtlebot3_gazebo/models:${{GAZEBO_MODEL_PATH:-}} "
        f'GAZEBO_MODEL_DATABASE_URI="" && '
    )
    params_file = os.path.join(WS_ROOT, "src", "disaster_recovery", "config", "recovery_params.yaml")
    onnx_path = DEFAULT_P4_REAL_ONNX

    hdd_dir = "/hdd/evidence/phase4_semantic"
    doc_dir = os.path.join(WS_ROOT, "docs", "images", "evidence_v4")
    os.makedirs(hdd_dir, exist_ok=True)
    os.makedirs(doc_dir, exist_ok=True)
    os.makedirs("/tmp/disaster_recovery", exist_ok=True)

    out_mp4 = os.path.join(hdd_dir, "phase4_real_trial10_e2e_3d.mp4")
    raw_mp4 = out_mp4 + ".raw.mp4"
    out_gif = os.path.join(doc_dir, "phase4_real_trial10_e2e_3d.gif")
    snap_out_png = os.path.join(doc_dir, "phase4_real_trial10_outbound.png")
    snap_cbf_png = os.path.join(doc_dir, "phase4_real_trial10_cbf_avoidance.png")
    snap_done_png = os.path.join(doc_dir, "phase4_real_trial10_complete.png")
    overlay_png = os.path.join(doc_dir, "phase4_real_trial10_trajectory_overlay.png")
    metrics_json = os.path.join(hdd_dir, "phase4_real_trial10_e2e_metrics.json")

    procs = []
    p_xvfb = subprocess.Popen(
        ["Xvfb", disp, "-screen", "0", "1920x1080x24", "-ac", "+extension", "GLX", "+render", "-noreset"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid,
    )
    procs.append(p_xvfb)
    time.sleep(1.5)

    low_list = list(TRIAL10_LOW_OBS)
    gas_xy = TRIAL10_GAS_XY
    out_wps = list(IMPROVED_OUT_WPS)

    try:
        p_gz = subprocess.Popen(
            ["bash", "-c", env_prefix + f"ros2 launch disaster_recovery disaster_world.launch.py x_pose:=0.0 y_pose:=0.0 yaw_pose:=0.0 world:={SEMANTIC_WORLD} gui:=false"],
            stdout=open("/tmp/disaster_recovery/p4_rec_gz.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid,
        )
        procs.append(p_gz)
        time.sleep(5.0)

        p_slam = subprocess.Popen(
            ["bash", "-c", env_prefix + "ros2 launch slam_toolbox online_async_launch.py use_sim_time:=true"],
            stdout=open("/tmp/disaster_recovery/p4_rec_slam.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid,
        )
        procs.append(p_slam)

        p_ctrl = subprocess.Popen(
            [
                "bash", "-c",
                env_prefix + f"ros2 run disaster_recovery recovery_controller --ros-args --params-file {params_file} "
                f"-p use_sim_time:=true -p use_aruco_docking:=false -p recovery_strategy:=proposed_rl "
                f"-p use_semantic_rl_policy:=true "
                f"-p enable_semantic_astar_cost:=true "
                f"-p enable_dynamic_tracker:=true "
                f"-p enable_cbf_safety_shield:=true "
                f"-p cbf_v_min:=-0.08 "
                f"-p context_rl_onnx_path:={onnx_path} "
                f"-p semantic_rl_onnx_path:={onnx_path} "
                f"-p phase4_rl_onnx_path:={onnx_path} "
                f"-p reset_corr_on_episode:=true",
            ],
            stdout=open("/tmp/disaster_recovery/p4_rec_ctrl.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid,
        )
        procs.append(p_ctrl)

        p_rviz = subprocess.Popen(
            ["bash", "-c", env_prefix + f"{vgl_prefix}rviz2 -d {rviz_tmp} --ros-args -p use_sim_time:=true"],
            stdout=open("/tmp/disaster_recovery/p4_rec_rviz.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid,
        )
        procs.append(p_rviz)

        rclpy.init()
        node = SemanticSuiteNode()
        t0 = time.time()
        while time.time() - t0 < 60.0:
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.odom_ready and node.scan_ready and node.sim_now() > 0.2:
                break

        p_gzc = subprocess.Popen(
            ["bash", "-c", env_prefix + f"{vgl_prefix}gzclient"],
            stdout=open("/tmp/disaster_recovery/p4_rec_gzc.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid,
        )
        procs.append(p_gzc)
        time.sleep(14.0)
        upd_hz = int(round(max(1.0, min(3.5, float(args.rtf))) * 1000.0))
        run_cmd_bounded(f"GAZEBO_MASTER_URI={gz_uri} timeout -k 1 2 gz physics -u {upd_hz} >/dev/null 2>&1", timeout_s=4.0)

        for i_l, (lx, ly) in enumerate(low_list, start=1):
            _spawn_model_at(node, f"low_obs_{i_l}", _low_obstacle_sdf(f"low_obs_{i_l}"), lx, ly, 0.0)
        _spawn_model_at(node, "gas_zone_1", _gas_disc_sdf("gas_zone_1"), gas_xy[0], gas_xy[1], 0.0)
        _spawn_gazebo_visual(node.cli_spawn, node, "gz_base_camp", _build_base_camp_sdf(0.0, 0.0), wait=True)

        node.reset_trial_telemetry()
        _arrange_3pane_windows(disp, rviz_h=1080)
        node.sim_sleep(0.8)

        out_pts: List[Tuple[float, float]] = [(0.0, 0.0)]
        ret_pts: List[Tuple[float, float]] = []
        last_marker_pub = [0.0]

        def _tick() -> None:
            # Keep teleop fast (pure in-memory odom; never spawn gz subprocesses inside on_step)
            cx, cy = node.cur_x, node.cur_y
            if not ret_pts and math.hypot(cx - out_pts[-1][0], cy - out_pts[-1][1]) >= 0.08:
                out_pts.append((cx, cy))
            now_w = time.time()
            if now_w - last_marker_pub[0] >= 0.25:
                last_marker_pub[0] = now_w
                node.publish_overlay_markers(0.0, 0.0, out_pts, ret_pts, low_list, gas_xy)

        _tick()
        p_ff = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "x11grab", "-video_size", "1920x1080", "-framerate", "20", "-i", disp, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-pix_fmt", "yuv420p", raw_mp4],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid,
        )
        procs.append(p_ff)
        time.sleep(0.5)
        t_rec_start = time.time()

        # Full outbound north safe path recording
        outbound_path = node.drive_to_waypoints(out_wps, max_sim_sec=75.0, on_step=_tick)
        if len(outbound_path) > len(out_pts):
            out_pts = list(outbound_path)

        tp_out_end = get_gazebo_true_pose("burger")
        ox_e, oy_e = (float(tp_out_end[0]), float(tp_out_end[1])) if tp_out_end is not None else (node.cur_x, node.cur_y)
        outbound_end_err_m = math.hypot(ox_e - out_wps[-1][0], oy_e - out_wps[-1][1])
        outbound_reached = bool(outbound_end_err_m <= 0.40)

        _spawn_gazebo_visual(
            node.cli_spawn, node, "gz_outbound_path",
            _build_ribbon_sdf("gz_outbound_path", _subsampled_pts(out_pts, 0.18), 0.042, 0.003, 0.0025, (0.05, 0.78, 1.0, 1.0), (0.04, 0.62, 0.92, 1.0)),
            wait=True,
        )

        # Register semantic cache (Ch1 low obstacles + Ch2 gas hazard + survivor)
        cache_objs = [
            {"marker_id": 1, "class_id": 56, "class_name": "chair", "x": low_list[0][0], "y": low_list[0][1], "z": 0.05, "radius": 0.24, "channel": 1, "conf": 0.92, "hits": 6, "low_confidence": False},
            {"marker_id": 2, "class_id": 56, "class_name": "chair", "x": low_list[1][0], "y": low_list[1][1], "z": 0.05, "radius": 0.24, "channel": 1, "conf": 0.90, "hits": 6, "low_confidence": False},
            {"marker_id": 10, "class_id": 1001, "class_name": "GAS_HAZARD", "x": gas_xy[0], "y": gas_xy[1], "z": 0.0, "radius": 0.55, "channel": 2, "conf": 0.85, "hits": 10, "low_confidence": False},
            {"marker_id": 11, "class_id": 0, "class_name": "person", "x": 5.85, "y": 0.45, "z": 0.4, "radius": 0.45, "channel": 2, "conf": 0.94, "hits": 8, "low_confidence": False},
        ]
        node.pub_cached.publish(String(data=json.dumps({"frame_id": "odom", "objects": cache_objs})))
        node.sim_sleep(0.4)
        snap_out_ts = max(1.0, (time.time() - t_rec_start) - 0.3)

        # Blackout trigger
        t_cut_wall = max(0.5, time.time() - t_rec_start)
        t_rec_wall: Optional[float] = None
        t_done_wall: Optional[float] = None
        snap_cbf_ts: Optional[float] = None
        node.hb_enabled = False
        t_blk = node.sim_now()
        w_blk = time.time()
        saw_rec = False
        mission_complete = False
        px0, py0 = ox_e, oy_e
        ret_pts = [(px0, py0)]
        min_low_d = 99.0
        min_gas_d = 99.0
        min_scan_m = 10.0

        latest_tp: List[Optional[Tuple[float, float, float]]] = [tp_out_end]
        stop_tp_poll = threading.Event()

        def _poll_true_pose() -> None:
            while not stop_tp_poll.is_set():
                p = get_gazebo_true_pose("burger")
                if p is not None and math.isfinite(p[0]) and math.isfinite(p[1]):
                    latest_tp[0] = p
                time.sleep(0.05)

        tp_thread = threading.Thread(target=_poll_true_pose, daemon=True)
        tp_thread.start()

        try:
            while (node.sim_now() - t_blk < 130.0) and (time.time() - w_blk < 300.0):
                for _ in range(15):
                    rclpy.spin_once(node, timeout_sec=0.002)
                    if node.ctrl_state == "MISSION_COMPLETE":
                        break
                tp = latest_tp[0]
                if tp is not None and math.isfinite(tp[0]) and math.isfinite(tp[1]):
                    px, py = float(tp[0]), float(tp[1])
                else:
                    px, py = node.cur_x, node.cur_y

                if (
                    node.ctrl_state in ("EMERGENCY_STOP", "BACKTRACKING_ROTATE", "BACKTRACKING_HYBRID", "BACKTRACKING_TRANSLATE", "LOCAL_TD3_AVOIDANCE", "MISSION_COMPLETE")
                    or math.hypot(px - px0, py - py0) >= 0.15
                    or math.hypot(node.cur_x - px0, node.cur_y - py0) >= 0.15
                ):
                    if not saw_rec:
                        saw_rec = True
                        t_rec_wall = max(t_cut_wall + 0.1, time.time() - t_rec_start)

                if saw_rec:
                    d_l = min(math.hypot(px - lx, py - ly) for lx, ly in low_list)
                    d_g = math.hypot(px - gas_xy[0], py - gas_xy[1])
                    min_low_d = min(min_low_d, d_l)
                    min_gas_d = min(min_gas_d, d_g)
                    min_scan_m = min(min_scan_m, node.scan_all_min)
                    if snap_cbf_ts is None and abs(px - gas_xy[0]) <= 0.35:
                        snap_cbf_ts = max(1.0, time.time() - t_rec_start)

                if math.hypot(px - ret_pts[-1][0], py - ret_pts[-1][1]) >= 0.08:
                    ret_pts.append((px, py))
                now_w = time.time()
                if now_w - last_marker_pub[0] >= 0.20:
                    last_marker_pub[0] = now_w
                    node.publish_overlay_markers(0.0, 0.0, out_pts, ret_pts, low_list, gas_xy)

                if saw_rec and node.ctrl_state == "MISSION_COMPLETE":
                    mission_complete = True
                    t_done_wall = max((t_rec_wall or t_cut_wall) + 0.1, time.time() - t_rec_start)
                    break
                t_s = node.sim_now()
                w_s = time.time()
                while (node.sim_now() - t_s < 0.04) and (time.time() - w_s < 1.0):
                    rclpy.spin_once(node, timeout_sec=0.002)
        finally:
            stop_tp_poll.set()
            tp_thread.join(timeout=1.5)

        ret_dur = node.sim_now() - t_blk
        tp_end = get_gazebo_true_pose("burger")
        fx, fy = (float(tp_end[0]), float(tp_end[1])) if tp_end is not None else (node.cur_x, node.cur_y)
        if saw_rec:
            min_low_d = min(min_low_d, min(math.hypot(fx - lx, fy - ly) for lx, ly in low_list))
            min_gas_d = min(min_gas_d, math.hypot(fx - gas_xy[0], fy - gas_xy[1]))
            ret_pts.append((fx, fy))

        err_m = math.hypot(fx, fy)
        arrived = bool(mission_complete and err_m <= 0.85)
        min_wall_geom = compute_semantic_world_min_wall_geom(ret_pts)

        _spawn_gazebo_visual(
            node.cli_spawn, node, "gz_return_path",
            _build_ribbon_sdf("gz_return_path", _subsampled_pts(ret_pts, 0.14), 0.048, 0.004, 0.003, (0.09, 0.84, 0.36, 1.0), (0.05, 0.68, 0.28, 1.0)),
            wait=True,
        )
        node.publish_overlay_markers(0.0, 0.0, out_pts, ret_pts, low_list, gas_xy)
        time.sleep(1.8)
        snap_done_ts = max(1.0, (time.time() - t_rec_start) - 0.2)

        if p_ff.stdin:
            p_ff.stdin.write(b"q")
            p_ff.stdin.flush()
        p_ff.wait(timeout=15.0)

        if snap_cbf_ts is None:
            snap_cbf_ts = max(1.0, (t_rec_wall or t_cut_wall) + 0.5 * ((t_done_wall or snap_done_ts) - (t_rec_wall or t_cut_wall)))

        err_cm = err_m * 100.0
        low_col = bool(min_low_d < 0.255)
        gas_viol = bool(min_gas_d < 0.55)
        wall_col = bool(min_wall_geom < 0.105 or min_scan_m < 0.125)
        strict_pass = bool(arrived and (not low_col) and (not wall_col) and (not gas_viol))

        tag = "phase4_real (4-Ch RL + 2-Circle Soft-CBF-QP + Semantic A*)"
        segs = [
            (0.0, t_cut_wall, f"[{tag} | DRIVING] COMM: ON | Outbound North Safe Path", (30, 64, 175, 238)),
        ]
        if t_rec_wall is not None:
            segs.append((t_cut_wall, t_rec_wall, f"[{tag} | COMM LOST!] Watchdog Timeout (3.0s)...", (180, 83, 9, 242)))
            rec_seg_txt = f"[{tag} | RECOVERY] North Gas Outer CBF-QP + Semantic A* Detour"
            if t_done_wall is not None:
                segs.append((t_rec_wall, t_done_wall, rec_seg_txt, (15, 76, 129, 238)))
                done_txt = (
                    f"[phase4_real | COMPLETE] Ret={ret_dur:.1f}s Err={err_cm:.1f}cm "
                    f"Low={min_low_d:.2f}m Gas={min_gas_d:.2f}m Scan={min_scan_m:.2f}m MQ2={node.cbf_mq2_escape_steps}"
                )
                done_col = (21, 128, 61, 242) if strict_pass else (153, 27, 27, 242)
                segs.append((t_done_wall, None, done_txt, done_col))
            else:
                segs.append((t_rec_wall, None, rec_seg_txt, (15, 76, 129, 238)))

        _annotate_mp4_and_gif(raw_mp4, out_mp4, out_gif, tag, gif_fps=8, gif_width=840, phase_segments=segs)
        if os.path.exists(raw_mp4):
            os.remove(raw_mp4)

        for ts, dst in [(snap_out_ts, snap_out_png), (snap_cbf_ts, snap_cbf_png), (snap_done_ts, snap_done_png)]:
            subprocess.run(
                ["ffmpeg", "-y", "-ss", f"{ts:.2f}", "-i", out_mp4, "-vframes", "1", dst],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

        metrics = {
            "mode": "phase4_real",
            "trial": 10,
            "seed": 3336,
            "onnx": os.path.basename(onnx_path),
            "cbf_v_min": -0.08,
            "enable_semantic_astar_cost": True,
            "enable_dynamic_tracker": True,
            "enable_cbf_safety_shield": True,
            "outbound_reached": outbound_reached,
            "outbound_end_err_m": round(outbound_end_err_m, 4),
            "max_roll_pitch_deg": round(node.max_roll_pitch_deg, 2),
            "arrival_success": arrived,
            "strict_hazard_free_success": strict_pass,
            "low_obstacle_collision": low_col,
            "wall_collision": wall_col,
            "gas_zone_violated": gas_viol,
            "min_low_obstacle_dist_m": round(min_low_d, 4),
            "min_gas_zone_dist_m": round(min_gas_d, 4),
            "min_wall_geom_m": round(min_wall_geom, 4),
            "min_scan_m": round(min_scan_m, 4),
            "return_duration_s": round(ret_dur, 2),
            "terminal_error_cm": round(err_cm, 2),
            "cbf_mq2_escape_steps": int(node.cbf_mq2_escape_steps),
            "max_confirmed_dyn_tracks": int(node.max_confirmed_dyn_tracks),
            "crossing_scheduler_modes": sorted(node.crossing_scheduler_modes),
            "low_obstacles": low_list,
            "gas_zone": list(gas_xy),
            "out_wps": out_wps,
            "artifacts": {
                "mp4": out_mp4,
                "gif": out_gif,
                "keyframe_outbound_png": snap_out_png,
                "keyframe_cbf_avoidance_png": snap_cbf_png,
                "keyframe_complete_png": snap_done_png,
                "trajectory_overlay_png": overlay_png,
            },
        }
        _render_phase4_trial10_overlay(overlay_png, out_pts, ret_pts, low_list, gas_xy, metrics)
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        for p_src in (out_gif, snap_out_png, snap_cbf_png, snap_done_png, overlay_png):
            if os.path.exists(p_src):
                shutil.copy2(p_src, os.path.join(hdd_dir, os.path.basename(p_src)))

        node.destroy_node()
        rclpy.shutdown()
        print("\n=== [PHASE4_REAL TRIAL #10 3D RECORDING DONE] ===", flush=True)
        print(json.dumps(metrics, indent=2), flush=True)
        return 0 if strict_pass else 1
    finally:
        _stop_procs(procs, gz_port)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gz-port", type=int, default=11368)
    parser.add_argument("--ros-domain-id", type=int, default=68)
    parser.add_argument("--display", default=":98")
    parser.add_argument("--rtf", type=float, default=2.0)
    return run_record_phase4(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
