#!/usr/bin/env python3
"""
3D Visual Evidence Recorder (Gazebo Classic Follow-Cam + RViz2 3D Dual-Pane).

Records reproducible MP4 clips and keyframe PNG snapshots from specific
ab_disaster_benchmark.py scenarios/trials inside headless Docker containers
using Xvfb + Mesa software GLX (LIBGL_ALWAYS_SOFTWARE=1) + ffmpeg x11grab.
"""

import argparse
import ctypes
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Optional

import rclpy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ab_disaster_benchmark import (  # noqa: E402
    SCENARIOS,
    SimHarnessNode,
    _kill_matching_gazebo_procs,
    get_gazebo_true_pose,
    run_cmd_bounded,
)
from debris_utils import delete_debris_entities, spawn_random_debris_cluster  # noqa: E402


def _make_rviz_config(src_rviz: str, dst_rviz: str) -> None:
    """Create dual-pane headless-friendly RViz2 config (right 700x1080 pure 3D canvas tracking burger)."""
    with open(src_rviz, "r", encoding="utf-8") as f:
        txt = f.read()

    # Keep world-locked odom grid while camera orbits around base_footprint!
    txt = txt.replace("Fixed Frame: base_link", "Fixed Frame: odom")
    txt = txt.replace("Target Frame: <Fixed Frame>", "Target Frame: base_footprint")
    txt = txt.replace("Size (m): 0.03", "Size (m): 0.07")
    txt = txt.replace("Distance: 2.733222484588623", "Distance: 5.5")
    txt = txt.replace("Pitch: 0.785398006439209", "Pitch: 0.95")
    txt = txt.replace("Hide Left Dock: false", "Hide Left Dock: true")
    txt = txt.replace("Hide Right Dock: false", "Hide Right Dock: true")
    # Disable docked Image pane so full 700x1080 right window renders 3D RViz2 LiDAR/Robot/TF
    txt = re.sub(
        r"(- Class: rviz_default_plugins/Image\s+Enabled:\s*)true(.*?Value:\s*/camera/image_raw\s+Value:\s*)true",
        r"\1false\2false",
        txt,
        flags=re.DOTALL,
    )
    # Remove Panels block & Qt QMainWindow binary dock layout so right 700x1080 pane is 100% 3D viewport
    txt = re.sub(r"^Panels:.*?(?=^Visualization Manager:)", "Panels: []\n", txt, flags=re.DOTALL | re.MULTILINE)
    txt = re.sub(r"^\s*QMainWindow State:.*\n", "", txt, flags=re.MULTILINE)
    # Match strictly 2-space indented Window Geometry fields (never 8-space Focal Point X/Y!)
    txt = re.sub(r"^  Height:\s*\d+$", "  Height: 1080", txt, flags=re.MULTILINE)
    txt = re.sub(r"^  Width:\s*\d+$", "  Width: 700", txt, flags=re.MULTILINE)
    txt = re.sub(r"^  X:\s*\d+$", "  X: 1220", txt, flags=re.MULTILINE)
    txt = re.sub(r"^  Y:\s*\d+$", "  Y: 0", txt, flags=re.MULTILINE)

    os.makedirs(os.path.dirname(dst_rviz), exist_ok=True)
    with open(dst_rviz, "w", encoding="utf-8") as f:
        f.write(txt)


def _ensure_gazebo_gui_ini() -> None:
    """Configure gzclient default window size so it occupies left 1220x1080 cleanly."""
    gz_dir = os.path.expanduser("~/.gazebo")
    os.makedirs(gz_dir, exist_ok=True)
    ini_path = os.path.join(gz_dir, "gui.ini")
    with open(ini_path, "w", encoding="utf-8") as f:
        f.write("[geometry]\nx=0\ny=0\nwidth=1220\nheight=1080\n")


def _arrange_x11_windows(display_str: str) -> None:
    """Best-effort snap gzclient (left 1220x1080) and rviz2 (right 700x1080) via libX11."""
    try:
        x11 = ctypes.CDLL("libX11.so.6")
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
        x11.XDefaultRootWindow.restype = ctypes.c_ulong
        x11.XQueryTree.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),
            ctypes.POINTER(ctypes.c_uint),
        ]
        x11.XFetchName.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_char_p)]
        x11.XMoveResizeWindow.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint
        ]
        x11.XFlush.argtypes = [ctypes.c_void_p]
        x11.XCloseDisplay.argtypes = [ctypes.c_void_p]

        dpy = x11.XOpenDisplay(display_str.encode("utf-8"))
        if not dpy:
            return
        root = x11.XDefaultRootWindow(dpy)
        r_ret = ctypes.c_ulong()
        p_ret = ctypes.c_ulong()
        children = ctypes.POINTER(ctypes.c_ulong)()
        n_child = ctypes.c_uint()
        if x11.XQueryTree(dpy, root, ctypes.byref(r_ret), ctypes.byref(p_ret), ctypes.byref(children), ctypes.byref(n_child)):
            for idx in range(int(n_child.value)):
                win = children[idx]
                name_ptr = ctypes.c_char_p()
                if x11.XFetchName(dpy, win, ctypes.byref(name_ptr)) and name_ptr.value:
                    title = name_ptr.value.decode("utf-8", errors="ignore").lower()
                    if "gazebo" in title:
                        x11.XMoveResizeWindow(dpy, win, 0, 0, 1220, 1080)
                    elif "rviz" in title:
                        x11.XMoveResizeWindow(dpy, win, 1220, 0, 700, 1080)
        x11.XFlush(dpy)
        x11.XCloseDisplay(dpy)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Record 3D Gazebo + RViz2 A/B evidence MP4 and PNG snapshots")
    parser.add_argument("--scenario", required=True, choices=list(SCENARIOS.keys()))
    parser.add_argument("--strategy", required=True, choices=["baseline", "proposed", "proposed_rl"])
    parser.add_argument("--onnx-path", default="", help="Optional ContextRL ONNX path for proposed_rl")
    parser.add_argument("--trial", type=int, default=1, help="1-indexed trial from existing A/B JSON")
    parser.add_argument("--ab-dir", default=os.path.join(WS_ROOT, "benchmark_results", "ab_comparison"))
    parser.add_argument("--out-mp4", required=True)
    parser.add_argument("--snapshot-dir", default=os.path.join(WS_ROOT, "docs", "images", "evidence"))
    parser.add_argument("--tag", default="", help="Optional prefix for extracted PNG snapshots")
    parser.add_argument("--display", default=":99")
    parser.add_argument("--gz-port", type=int, default=11355)
    parser.add_argument("--ros-domain-id", type=int, default=42)
    parser.add_argument("--rtf", type=float, default=2.0)
    parser.add_argument("--max-sim-sec", type=float, default=145.0, help="Max sim seconds after blackout")
    args = parser.parse_args()

    cfg = SCENARIOS[args.scenario]
    json_path = os.path.join(args.ab_dir, f"ab_{args.scenario}_10trials.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing A/B benchmark JSON: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        ab_data = json.load(f)
    specs = ab_data["fixed_specs"]
    if args.trial < 1 or args.trial > len(specs):
        raise ValueError(f"Invalid --trial {args.trial}; available 1..{len(specs)}")
    spec = specs[args.trial - 1]
    debris_positions = [tuple(p) for p in spec["debris_positions"]]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_mp4)), exist_ok=True)
    os.makedirs(args.snapshot_dir, exist_ok=True)

    _ensure_gazebo_gui_ini()
    rviz_tmp = f"/tmp/disaster_recovery/record_{args.scenario}_{args.strategy}.rviz"
    src_rviz = os.path.join(WS_ROOT, "src", "disaster_recovery", "config", "disaster_station.rviz")
    _make_rviz_config(src_rviz, rviz_tmp)

    gz_master_uri = f"http://127.0.0.1:{args.gz_port}"
    os.environ["DISPLAY"] = args.display
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    os.environ["QT_X11_NO_MITSHM"] = "1"
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    os.environ["ROS_DOMAIN_ID"] = str(args.ros_domain_id)
    os.environ["GAZEBO_MASTER_URI"] = gz_master_uri
    os.environ["GAZEBO_MODEL_DATABASE_URI"] = ""

    # Clean matching port procs first
    _kill_matching_gazebo_procs(gz_master_uri)

    p_xvfb = subprocess.Popen(
        ["Xvfb", args.display, "-screen", "0", "1920x1080x24", "-ac", "+extension", "GLX", "+render", "-noreset"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    time.sleep(1.5)

    env_prefix = (
        f"source /opt/ros/humble/setup.bash && source {WS_ROOT}/install/setup.bash && "
        f"export DISPLAY={args.display} LIBGL_ALWAYS_SOFTWARE=1 QT_X11_NO_MITSHM=1 "
        f"ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID={args.ros_domain_id} GAZEBO_MASTER_URI={gz_master_uri} "
        f'GAZEBO_MODEL_DATABASE_URI="" && '
    )

    rx, ry, ryaw = cfg.reset_pose
    p_gz = subprocess.Popen(
        [
            "bash", "-c",
            env_prefix + f"ros2 launch disaster_recovery disaster_world.launch.py x_pose:={rx} y_pose:={ry} yaw_pose:={ryaw} world:={cfg.world_file} gui:=false",
        ],
        stdout=open(f"/tmp/disaster_recovery/rec_gz_{args.scenario}_{args.strategy}.log", "w"),
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    time.sleep(5.0)

    params_file = os.path.join(WS_ROOT, "src", "disaster_recovery", "config", "recovery_params.yaml")
    onnx_override = f" -p context_rl_onnx_path:={args.onnx_path}" if args.onnx_path else ""
    p_ctrl = subprocess.Popen(
        [
            "bash", "-c",
            env_prefix + f"ros2 run disaster_recovery recovery_controller --ros-args --params-file {params_file} -p use_sim_time:=true -p use_aruco_docking:=false -p recovery_strategy:={args.strategy}{onnx_override}",
        ],
        stdout=open(f"/tmp/disaster_recovery/rec_ctrl_{args.scenario}_{args.strategy}.log", "w"),
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    p_rviz = subprocess.Popen(
        ["bash", "-c", env_prefix + f"rviz2 -d {rviz_tmp} --ros-args -p use_sim_time:=true"],
        stdout=open(f"/tmp/disaster_recovery/rec_rviz_{args.scenario}_{args.strategy}.log", "w"),
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    time.sleep(4.5)

    p_ffmpeg: Optional[subprocess.Popen] = None
    p_gzclient: Optional[subprocess.Popen] = None
    rclpy.init()
    node = SimHarnessNode()

    try:
        t0 = time.time()
        while time.time() - t0 < 120.0:
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.odom_ready and node.scan_ready and node.sim_now() > 0.2:
                break

        # Launch gzclient ONLY after gzserver & burger have finished spawning (prevents 100s house world load timeout)
        p_gzclient = subprocess.Popen(
            ["bash", "-c", env_prefix + "gzclient"],
            stdout=open(f"/tmp/disaster_recovery/rec_gzclient_{args.scenario}_{args.strategy}.log", "w"),
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        time.sleep(16.0)

        u_rate = int(round(args.rtf / cfg.max_step_size))
        run_cmd_bounded(f"GAZEBO_MASTER_URI={gz_master_uri} timeout -k 1 2 gz physics -u {u_rate} >/dev/null 2>&1", timeout_s=4.0)
        run_cmd_bounded(f"GAZEBO_MASTER_URI={gz_master_uri} timeout -k 1 3 gz camera -c gzclient_camera -f burger >/dev/null 2>&1", timeout_s=4.0)
        _arrange_x11_windows(args.display)

        # Outbound exploration phase
        node.set_heartbeat(True)
        node.trigger_episode_reset()
        node.sim_sleep(0.3)

        target_depth = spec["target_depth"]
        goal_pool = spec["goal_pool"]
        goal_ptr = 0
        explored_m = 0.0
        prev_ox, prev_oy = node.cur_x, node.cur_y
        outbound_start = node.sim_now()
        last_arrange = time.time()

        while node.sim_now() - outbound_start < cfg.outbound_timeout_sim:
            rclpy.spin_once(node, timeout_sec=0.02)
            step_d = math.hypot(node.cur_x - prev_ox, node.cur_y - prev_oy)
            if step_d >= 0.08:
                explored_m += step_d
                prev_ox, prev_oy = node.cur_x, node.cur_y
            dist_from_base = math.hypot(node.cur_x - rx, node.cur_y - ry)
            if explored_m >= target_depth and dist_from_base >= cfg.min_blackout_dist:
                break
            gx, gy = goal_pool[goal_ptr % len(goal_pool)]
            if math.hypot(gx - node.cur_x, gy - node.cur_y) < 0.70 and goal_ptr + 1 < len(goal_pool):
                goal_ptr += 1
            v_cmd, w_cmd = node.choose_open_gap_cmd(gx, gy)
            node.drive_raw(v_cmd, w_cmd)
            if time.time() - last_arrange > 4.0:
                _arrange_x11_windows(args.display)
                last_arrange = time.time()

        node.stop_drive()
        node.sim_sleep(0.2)

        # Start ffmpeg 3D recording BEFORE spawning debris & blackout!
        _arrange_x11_windows(args.display)
        run_cmd_bounded(f"GAZEBO_MASTER_URI={gz_master_uri} timeout -k 1 3 gz camera -c gzclient_camera -f burger >/dev/null 2>&1", timeout_s=4.0)
        time.sleep(2.0)
        p_ffmpeg = subprocess.Popen(
            [
                "ffmpeg", "-y", "-f", "x11grab",
                "-video_size", "1920x1080", "-framerate", "15", "-i", args.display,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                args.out_mp4,
            ],
            stdin=subprocess.PIPE,
            stdout=open("/tmp/disaster_recovery/rec_ffmpeg.log", "w"),
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        time.sleep(0.8)

        if debris_positions:
            def _hb_tick():
                node.drive_raw(0.0, 0.0)
                rclpy.spin_once(node, timeout_sec=0.01)
            spawn_random_debris_cluster(debris_positions, wait=True, tick_fn=_hb_tick)
        node.stop_drive()
        node.sim_sleep(0.3)

        # Blackout trigger -> record Autonomous Recovery!
        node.set_heartbeat(False)
        t_blackout = node.sim_now()
        saw_recovery = False
        success = False
        while node.sim_now() - t_blackout < args.max_sim_sec:
            t_s = node.sim_now()
            while node.sim_now() - t_s < 0.10:
                rclpy.spin_once(node, timeout_sec=0.01)
            if node.ctrl_state in ("BACKTRACKING_ROTATE", "BACKTRACKING_TRANSLATE", "BACKTRACKING_HYBRID", "LOCAL_TD3_AVOIDANCE", "OBSTACLE_HOLD", "MISSION_COMPLETE"):
                saw_recovery = True
            if saw_recovery and node.ctrl_state in ("MISSION_COMPLETE", "NORMAL_TELEOP"):
                success = True
                node.sim_sleep(1.0)
                break

        true_p = get_gazebo_true_pose("burger")
        fx, fy = (true_p[0], true_p[1]) if true_p else (node.cur_x, node.cur_y)
        err_cm = math.hypot(fx - rx, fy - ry) * 100.0
        print(f"[RECORDED] {args.scenario}/{args.strategy} trial#{args.trial} | success={success} err={err_cm:.1f}cm sim_dur={node.sim_now()-t_blackout:.1f}s -> {args.out_mp4}")

    finally:
        if p_ffmpeg is not None and p_ffmpeg.poll() is None:
            try:
                p_ffmpeg.communicate(input=b"q", timeout=6.0)
            except Exception:
                try:
                    os.killpg(os.getpgid(p_ffmpeg.pid), signal.SIGINT)
                    p_ffmpeg.wait(timeout=4.0)
                except Exception:
                    pass
        try:
            delete_debris_entities(max_count=12)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()
        for p in (p_rviz, p_gzclient, p_ctrl, p_gz, p_xvfb):
            if p is None:
                continue
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass
        time.sleep(0.4)
        for p in (p_rviz, p_gzclient, p_ctrl, p_gz, p_xvfb):
            if p is None:
                continue
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                pass
        _kill_matching_gazebo_procs(gz_master_uri)

    # Integrity verification (ffprobe) & keyframe PNG extraction
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames,duration", "-of", "json", args.out_mp4],
        capture_output=True, text=True,
    )
    print("[FFPROBE]", probe.stdout.strip())
    dur = 10.0
    try:
        info = json.loads(probe.stdout)
        dur = float(info["streams"][0].get("duration", 10.0))
    except Exception:
        pass

    prefix = args.tag if args.tag else f"{args.scenario}_{args.strategy}_t{args.trial:02d}"
    for label, frac in [("encounter", 0.55), ("outcome", 0.85)]:
        ts = max(0.5, min(dur - 0.5, dur * frac))
        out_png = os.path.join(args.snapshot_dir, f"{prefix}_{label}.png")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{ts:.2f}", "-i", args.out_mp4, "-vframes", "1", out_png],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"[SNAPSHOT] {out_png} ({os.path.getsize(out_png) if os.path.exists(out_png) else 0} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
