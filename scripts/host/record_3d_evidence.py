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
from typing import List, Optional, Sequence, Tuple

import rclpy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

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

    # Keep world-locked map frame (from slam_toolbox) while camera tracks base_footprint!
    txt = txt.replace("Fixed Frame: base_link", "Fixed Frame: map")
    txt = txt.replace("Target Frame: <Fixed Frame>", "Target Frame: base_footprint")
    txt = txt.replace("Size (m): 0.03", "Size (m): 0.07")
    txt = txt.replace("Distance: 2.733222484588623", "Distance: 8.5")
    txt = txt.replace("Pitch: 0.785398006439209", "Pitch: 1.45")
    txt = txt.replace("Target Frame Rate: 15", "Target Frame Rate: 60")
    txt = txt.replace("Frame Rate: 30", "Frame Rate: 60")
    txt = txt.replace("Hide Left Dock: false", "Hide Left Dock: true")
    txt = txt.replace("Hide Right Dock: false", "Hide Right Dock: true")
    # Inject MarkerArray display for /debris_markers so RViz2 shows 3D Debris & Base Camp markers + text
    marker_display = (
        "    - Class: rviz_default_plugins/MarkerArray\n"
        "      Enabled: true\n"
        "      Name: DebrisMarkers\n"
        "      Namespaces:\n"
        "        {}\n"
        "      Topic:\n"
        "        Depth: 5\n"
        "        Durability Policy: Volatile\n"
        "        History Policy: Keep Last\n"
        "        Reliability Policy: Reliable\n"
        "        Value: /debris_markers\n"
        "      Value: true\n"
    )
    txt = txt.replace("  Enabled: true\n  Global Options:", marker_display + "  Enabled: true\n  Global Options:")
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
    """Configure gzclient default window size so left 228px GUI tree is off-screen and 3D viewport fills 0..1220."""
    gz_dir = os.path.expanduser("~/.gazebo")
    os.makedirs(gz_dir, exist_ok=True)
    ini_path = os.path.join(gz_dir, "gui.ini")
    with open(ini_path, "w", encoding="utf-8") as f:
        f.write("[geometry]\nx=-228\ny=0\nwidth=1448\nheight=1080\n")


def _arrange_x11_windows(display_str: str) -> None:
    """Best-effort snap gzclient (x=-228..1220, cropping left GUI panel) and rviz2 (right 700x1080) via libX11."""
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
                        x11.XMoveResizeWindow(dpy, win, -228, 0, 1448, 1080)
                    elif "rviz" in title:
                        x11.XMoveResizeWindow(dpy, win, 1220, 0, 700, 1080)
        x11.XFlush(dpy)
        x11.XCloseDisplay(dpy)
    except Exception:
        pass


def _publish_rviz_markers(
    pub,
    node: SimHarnessNode,
    rx: float,
    ry: float,
    debris_positions: Sequence[Tuple[float, float]],
    outbound_pts: Sequence[Tuple[float, float]],
    return_pts: Sequence[Tuple[float, float]] = (),
    show_debris: bool = True,
) -> None:
    """Publish 3D START (BASE CAMP), Outbound Breadcrumb Trail, Green Return Trajectory, and Debris markers to RViz2."""
    ma = MarkerArray()
    stamp = node.get_clock().now().to_msg()

    # 1. Base camp disk + text
    m_base = Marker()
    m_base.header.frame_id = "odom"
    m_base.header.stamp = stamp
    m_base.ns = "base_camp"
    m_base.id = 0
    m_base.type = Marker.CYLINDER
    m_base.action = Marker.ADD
    m_base.pose.position.x = float(rx)
    m_base.pose.position.y = float(ry)
    m_base.pose.position.z = 0.04
    m_base.pose.orientation.w = 1.0
    m_base.scale.x = 0.72
    m_base.scale.y = 0.72
    m_base.scale.z = 0.06
    m_base.color.r = 0.10
    m_base.color.g = 0.95
    m_base.color.b = 0.30
    m_base.color.a = 0.88
    ma.markers.append(m_base)

    m_base_txt = Marker()
    m_base_txt.header.frame_id = "odom"
    m_base_txt.header.stamp = stamp
    m_base_txt.ns = "base_camp"
    m_base_txt.id = 1
    m_base_txt.type = Marker.TEXT_VIEW_FACING
    m_base_txt.action = Marker.ADD
    m_base_txt.pose.position.x = float(rx)
    m_base_txt.pose.position.y = float(ry)
    m_base_txt.pose.position.z = 0.58
    m_base_txt.pose.orientation.w = 1.0
    m_base_txt.scale.z = 0.34
    m_base_txt.color.r = 0.25
    m_base_txt.color.g = 1.0
    m_base_txt.color.b = 0.45
    m_base_txt.color.a = 1.0
    m_base_txt.text = f"START (BASE CAMP) ({rx:.1f},{ry:.1f})"
    ma.markers.append(m_base_txt)

    # 2. Outbound breadcrumb trail (cyan)
    if len(outbound_pts) >= 2:
        m_path = Marker()
        m_path.header.frame_id = "odom"
        m_path.header.stamp = stamp
        m_path.ns = "breadcrumbs"
        m_path.id = 2
        m_path.type = Marker.LINE_STRIP
        m_path.action = Marker.ADD
        m_path.pose.orientation.w = 1.0
        m_path.scale.x = 0.05
        m_path.color.r = 0.15
        m_path.color.g = 0.75
        m_path.color.b = 1.0
        m_path.color.a = 0.55
        for px, py in outbound_pts:
            pt = Point()
            pt.x = float(px)
            pt.y = float(py)
            pt.z = 0.03
            m_path.points.append(pt)
        ma.markers.append(m_path)

    # 2b. Active Return Robot Trajectory (bright green LINE_STRIP)
    if len(return_pts) >= 2:
        m_ret = Marker()
        m_ret.header.frame_id = "odom"
        m_ret.header.stamp = stamp
        m_ret.ns = "return_trajectory"
        m_ret.id = 3
        m_ret.type = Marker.LINE_STRIP
        m_ret.action = Marker.ADD
        m_ret.pose.orientation.w = 1.0
        m_ret.scale.x = 0.09
        m_ret.color.r = 0.15
        m_ret.color.g = 1.0
        m_ret.color.b = 0.35
        m_ret.color.a = 0.96
        for px, py in return_pts:
            pt = Point()
            pt.x = float(px)
            pt.y = float(py)
            pt.z = 0.06
            m_ret.points.append(pt)
        ma.markers.append(m_ret)

    # 3. Spawned Debris positions (bright yellow/orange 3D cylinders + floating text labels)
    if show_debris:
        for idx, (dx, dy) in enumerate(debris_positions, start=1):
            m_deb = Marker()
            m_deb.header.frame_id = "odom"
            m_deb.header.stamp = stamp
            m_deb.ns = "debris"
            m_deb.id = 10 + idx * 2
            m_deb.type = Marker.CYLINDER
            m_deb.action = Marker.ADD
            m_deb.pose.position.x = float(dx)
            m_deb.pose.position.y = float(dy)
            m_deb.pose.position.z = 0.24
            m_deb.pose.orientation.w = 1.0
            m_deb.scale.x = 0.56
            m_deb.scale.y = 0.56
            m_deb.scale.z = 0.48
            m_deb.color.r = 1.0
            m_deb.color.g = 0.78
            m_deb.color.b = 0.0
            m_deb.color.a = 0.88
            ma.markers.append(m_deb)

            m_txt = Marker()
            m_txt.header.frame_id = "odom"
            m_txt.header.stamp = stamp
            m_txt.ns = "debris"
            m_txt.id = 11 + idx * 2
            m_txt.type = Marker.TEXT_VIEW_FACING
            m_txt.action = Marker.ADD
            m_txt.pose.position.x = float(dx)
            m_txt.pose.position.y = float(dy)
            m_txt.pose.position.z = 0.75
            m_txt.pose.orientation.w = 1.0
            m_txt.scale.z = 0.30
            m_txt.color.r = 1.0
            m_txt.color.g = 0.95
            m_txt.color.b = 0.10
            m_txt.color.a = 1.0
            m_txt.text = f"DEBRIS #{idx} ({dx:.1f},{dy:.1f})"
            ma.markers.append(m_txt)

    pub.publish(ma)


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
    parser.add_argument("--rtf", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=30, help="Recording FPS (30 or 60)")
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

    use_vgl = bool(shutil.which("vglrun")) and os.path.exists("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    vgl_prefix = "vglrun -d egl " if use_vgl else ""
    gz_master_uri = f"http://127.0.0.1:{args.gz_port}"
    os.environ["DISPLAY"] = args.display
    if use_vgl:
        os.environ.pop("LIBGL_ALWAYS_SOFTWARE", None)
    else:
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        os.environ["LP_NUM_THREADS"] = "16"
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

    sw_env = "" if use_vgl else "LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=16 "
    env_prefix = (
        f"source /opt/ros/humble/setup.bash && source {WS_ROOT}/install/setup.bash && "
        f"export DISPLAY={args.display} {sw_env}QT_X11_NO_MITSHM=1 "
        f"ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID={args.ros_domain_id} GAZEBO_MASTER_URI={gz_master_uri} "
        f'GAZEBO_MODEL_DATABASE_URI="" && '
    )
    print(f"[GL] VirtualGL={'ENABLED (Tesla V100 EGL)' if use_vgl else 'DISABLED (Mesa llvmpipe CPU)'}", flush=True)

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

    p_slam = subprocess.Popen(
        [
            "bash", "-c",
            env_prefix + "ros2 launch slam_toolbox online_async_launch.py use_sim_time:=true",
        ],
        stdout=open(f"/tmp/disaster_recovery/rec_slam_{args.scenario}_{args.strategy}.log", "w"),
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    p_rviz = subprocess.Popen(
        ["bash", "-c", env_prefix + f"{vgl_prefix}rviz2 -d {rviz_tmp} --ros-args -p use_sim_time:=true"],
        stdout=open(f"/tmp/disaster_recovery/rec_rviz_{args.scenario}_{args.strategy}.log", "w"),
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    time.sleep(4.5)

    p_ffmpeg: Optional[subprocess.Popen] = None
    p_gzclient: Optional[subprocess.Popen] = None
    rclpy.init()
    node = SimHarnessNode()
    pub_markers = node.create_publisher(MarkerArray, "/debris_markers", 10)
    outbound_pts: List[Tuple[float, float]] = [(rx, ry)]

    try:
        t0 = time.time()
        while time.time() - t0 < 120.0:
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.odom_ready and node.scan_ready and node.sim_now() > 0.2:
                break

        # Launch gzclient ONLY after gzserver & burger have finished spawning (prevents 100s house world load timeout)
        p_gzclient = subprocess.Popen(
            ["bash", "-c", env_prefix + f"{vgl_prefix}gzclient"],
            stdout=open(f"/tmp/disaster_recovery/rec_gzclient_{args.scenario}_{args.strategy}.log", "w"),
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        time.sleep(16.0)

        u_rate = int(round(args.rtf / cfg.max_step_size))
        run_cmd_bounded(f"GAZEBO_MASTER_URI={gz_master_uri} timeout -k 1 2 gz physics -u {u_rate} >/dev/null 2>&1", timeout_s=4.0)
        if args.scenario == "tunnel":
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
        last_marker_pub = 0.0

        while node.sim_now() - outbound_start < cfg.outbound_timeout_sim:
            rclpy.spin_once(node, timeout_sec=0.02)
            step_d = math.hypot(node.cur_x - prev_ox, node.cur_y - prev_oy)
            if step_d >= 0.08:
                explored_m += step_d
                prev_ox, prev_oy = node.cur_x, node.cur_y
                outbound_pts.append((node.cur_x, node.cur_y))
            if time.time() - last_marker_pub > 0.4:
                _publish_rviz_markers(pub_markers, node, rx, ry, debris_positions, outbound_pts, show_debris=False)
                last_marker_pub = time.time()
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
        if args.scenario == "tunnel":
            run_cmd_bounded(f"GAZEBO_MASTER_URI={gz_master_uri} timeout -k 1 3 gz camera -c gzclient_camera -f burger >/dev/null 2>&1", timeout_s=4.0)
        return_pts: List[Tuple[float, float]] = [(node.cur_x, node.cur_y)]
        _publish_rviz_markers(pub_markers, node, rx, ry, debris_positions, outbound_pts, return_pts=return_pts, show_debris=True)
        time.sleep(2.0)
        raw_mp4 = args.out_mp4 + ".raw.mp4"
        t_rec_wall0 = time.time()
        telemetry_log: List[Tuple[float, float, str, float]] = []
        p_ffmpeg = subprocess.Popen(
            [
                "ffmpeg", "-y", "-f", "x11grab",
                "-video_size", "1920x1080", "-framerate", str(args.fps), "-i", args.display,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-pix_fmt", "yuv420p",
                raw_mp4,
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
        _publish_rviz_markers(pub_markers, node, rx, ry, debris_positions, outbound_pts, return_pts=return_pts, show_debris=True)
        node.stop_drive()
        node.sim_sleep(0.3)

        # Blackout trigger -> record Autonomous Recovery!
        node.set_heartbeat(False)
        t_blackout = node.sim_now()
        saw_recovery = False
        success = False
        prev_rx, prev_ry = node.cur_x, node.cur_y
        while node.sim_now() - t_blackout < args.max_sim_sec:
            t_s = node.sim_now()
            while node.sim_now() - t_s < 0.033:
                rclpy.spin_once(node, timeout_sec=0.005)
            if math.hypot(node.cur_x - prev_rx, node.cur_y - prev_ry) >= 0.03:
                prev_rx, prev_ry = node.cur_x, node.cur_y
                return_pts.append((node.cur_x, node.cur_y))
            sim_el = max(0.0, node.sim_now() - t_blackout)
            dist_rem_now = math.hypot(node.cur_x - rx, node.cur_y - ry)
            telemetry_log.append((time.time() - t_rec_wall0, sim_el, str(node.ctrl_state), dist_rem_now))
            if time.time() - last_marker_pub > 0.033:
                _publish_rviz_markers(pub_markers, node, rx, ry, debris_positions, outbound_pts, return_pts=return_pts, show_debris=True)
                last_marker_pub = time.time()
            if node.ctrl_state in ("BACKTRACKING_ROTATE", "BACKTRACKING_TRANSLATE", "BACKTRACKING_HYBRID", "LOCAL_TD3_AVOIDANCE", "OBSTACLE_HOLD", "MISSION_COMPLETE"):
                saw_recovery = True
            if saw_recovery and node.ctrl_state in ("MISSION_COMPLETE", "NORMAL_TELEOP"):
                success = True
                return_pts.append((node.cur_x, node.cur_y))
                _publish_rviz_markers(pub_markers, node, rx, ry, debris_positions, outbound_pts, return_pts=return_pts, show_debris=True)
                for _ in range(10):
                    node.sim_sleep(0.12)
                    telemetry_log.append((time.time() - t_rec_wall0, max(0.0, node.sim_now() - t_blackout), "MISSION_COMPLETE", math.hypot(node.cur_x - rx, node.cur_y - ry)))
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
        for p in (p_rviz, p_slam, p_gzclient, p_ctrl, p_gz, p_xvfb):
            if p is None:
                continue
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass
        time.sleep(0.4)
        for p in (p_rviz, p_slam, p_gzclient, p_ctrl, p_gz, p_xvfb):
            if p is None:
                continue
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                pass
        _kill_matching_gazebo_procs(gz_master_uri)

    # Apply Top/Bottom HUD Banner Overlay onto args.out_mp4
    _apply_hud_overlay(
        raw_mp4=raw_mp4,
        out_mp4=args.out_mp4,
        scenario=args.scenario,
        strategy=args.strategy,
        trial=args.trial,
        fps=args.fps,
        telemetry_log=telemetry_log,
        success=success,
        err_cm=err_cm,
    )

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
    snap_points = [
        ("encounter", max(0.5, min(dur - 0.5, dur * 0.55))),
        ("outcome", max(0.5, dur - 0.6) if success else max(0.5, min(dur - 0.4, dur * 0.85))),
    ]
    for label, ts in snap_points:
        out_png = os.path.join(args.snapshot_dir, f"{prefix}_{label}.png")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{ts:.2f}", "-i", args.out_mp4, "-vframes", "1", out_png],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"[SNAPSHOT] {out_png} ({os.path.getsize(out_png) if os.path.exists(out_png) else 0} B)")
    return 0


def _apply_hud_overlay(
    raw_mp4: str,
    out_mp4: str,
    scenario: str,
    strategy: str,
    trial: int,
    fps: int,
    telemetry_log: Sequence[Tuple[float, float, str, float]],
    success: bool,
    err_cm: float,
) -> None:
    """Stamp top telemetry HUD bar and bottom event banner onto every frame of raw_mp4 -> out_mp4."""
    if not os.path.exists(raw_mp4):
        return
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
    except Exception:
        shutil.move(raw_mp4, out_mp4)
        return

    font_paths = [
        "/tmp/NanumGothicBold.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    font_top = ImageFont.load_default()
    font_bot = ImageFont.load_default()
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font_top = ImageFont.truetype(fp, 22)
                font_bot = ImageFont.truetype(fp, 24)
                break
            except Exception:
                continue

    strat_label = {
        "baseline": "Baseline (단순 역추적)",
        "proposed": "Phase 1 (proposed A*)",
        "proposed_rl": "Phase 2 (proposed_rl)",
    }.get(strategy, strategy)

    has_nvenc = False
    try:
        enc_chk = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=4.0)
        has_nvenc = "h264_nvenc" in enc_chk.stdout
    except Exception:
        has_nvenc = False

    vcodec_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20"] if has_nvenc else ["-c:v", "libx264", "-preset", "fast", "-crf", "20"]

    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", raw_mp4, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
    )
    enc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", "1920x1080", "-pix_fmt", "rgb24", "-r", str(fps), "-i", "-",
            *vcodec_args, "-pix_fmt", "yuv420p",
            out_mp4,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    frame_bytes = 1920 * 1080 * 3
    frame_idx = 0
    total_wall = telemetry_log[-1][0] if telemetry_log else 10.0

    while True:
        buf = dec.stdout.read(frame_bytes) if dec.stdout else b""
        if len(buf) < frame_bytes:
            break
        t_wall = frame_idx / float(max(1, fps))
        frame_idx += 1

        # Lookup closest telemetry sample
        sim_el, st_name, dist_rem = 0.0, "BLACKOUT_DETECTED", 0.0
        if telemetry_log:
            best_row = min(telemetry_log, key=lambda r: abs(r[0] - t_wall))
            sim_el, st_name, dist_rem = best_row[1], best_row[2], best_row[3]

        img = Image.frombytes("RGB", (1920, 1080), buf)
        draw = ImageDraw.Draw(img, "RGBA")

        # Top HUD Bar
        draw.rectangle([(0, 0), (1920, 48)], fill=(15, 23, 42, 220))
        draw.line([(0, 48), (1920, 48)], fill=(56, 189, 248, 255), width=2)
        top_txt = (
            f"  {strat_label}  |  {scenario} Trial #{trial:02d}  |  "
            f"상태: {st_name}  |  경과: {sim_el:5.1f}s  |  잔여: {dist_rem:4.2f}m"
        )
        draw.text((18, 10), top_txt, font=font_top, fill=(248, 250, 252, 255))
        draw.text((1232, 10), "[RViz2: START(BASE CAMP) & 초록 복귀 궤적 & 노란 낙석 마커]", font=font_top, fill=(134, 239, 172, 255))

        # Bottom Event Banner
        if sim_el <= 4.5:
            ev_txt = "[통신 단절(BLACKOUT) -> 낙석 발생 -> 자율복귀 시작]"
            bg_col = (180, 83, 9, 230)
            bd_col = (251, 191, 36, 255)
        elif success and (st_name in ("MISSION_COMPLETE", "NORMAL_TELEOP") or dist_rem <= 0.45 or t_wall >= total_wall - 1.5):
            ev_txt = f"[BASE CAMP 복귀 성공 (최종 오차 {err_cm:.1f}cm)]"
            bg_col = (21, 128, 61, 235)
            bd_col = (74, 222, 128, 255)
        elif (not success) and t_wall >= total_wall * 0.72:
            ev_txt = f"[낙석 구간 교착/충돌 — 복귀 실패 (잔여 오차 {err_cm:.1f}cm)]"
            bg_col = (185, 28, 28, 235)
            bd_col = (248, 113, 113, 255)
        else:
            ev_txt = f"[자율복귀 주행 중 — 낙석 우회 및 기지 복귀 (잔여 {dist_rem:.2f}m)]"
            bg_col = (30, 41, 59, 215)
            bd_col = (148, 163, 184, 255)

        draw.rounded_rectangle([(340, 1018), (1580, 1070)], radius=10, fill=bg_col, outline=bd_col, width=2)
        draw.text((380, 1029), ev_txt, font=font_bot, fill=(255, 255, 255, 255))

        if enc.stdin:
            enc.stdin.write(img.tobytes())

    if dec.stdout:
        dec.stdout.close()
    dec.wait()
    if enc.stdin:
        enc.stdin.close()
    enc.wait()
    try:
        os.remove(raw_mp4)
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
