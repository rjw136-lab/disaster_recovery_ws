#!/usr/bin/env python3
"""
3D Visual Evidence Recorder (Gazebo Classic + RViz2 LiDAR + Robot Front Camera 3-Pane).

Synchronized with ab_disaster_benchmark.py's verified reset, reference-outbound replay,
heartbeat-safe debris spawning, Phase 4 / Phase 4 Real controller launch, and strict
collision-free evaluation (lidar_collision_steps == 0 and debris_collision == False).
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
import threading
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Point
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image as RosImage
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
    normalize_angle,
    run_cmd_bounded,
)
from debris_utils import delete_debris_entities, spawn_random_debris_cluster  # noqa: E402


def _make_rviz_config(src_rviz: str, dst_rviz: str, rviz_height: int = 640) -> None:
    """Create headless-friendly RViz2 config (top-right 700x640 when camera enabled, else 700x1080)."""
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
    txt = re.sub(r"^  Height:\s*\d+$", f"  Height: {int(rviz_height)}", txt, flags=re.MULTILINE)
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


def _arrange_x11_windows(display_str: str, enable_camera: bool = True, rviz_height: int = 640) -> None:
    """Best-effort snap gzclient (left 1220x1080), rviz2 (top-right 700x640 or 700x1080), and front camera (bottom-right 700x440) via libX11."""
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
        rh = int(rviz_height) if enable_camera else 1080
        if x11.XQueryTree(dpy, root, ctypes.byref(r_ret), ctypes.byref(p_ret), ctypes.byref(children), ctypes.byref(n_child)):
            for idx in range(int(n_child.value)):
                win = children[idx]
                name_ptr = ctypes.c_char_p()
                if x11.XFetchName(dpy, win, ctypes.byref(name_ptr)) and name_ptr.value:
                    title = name_ptr.value.decode("utf-8", errors="ignore").lower()
                    if "gazebo" in title:
                        x11.XMoveResizeWindow(dpy, win, -228, 0, 1448, 1080)
                    elif "rviz" in title:
                        x11.XMoveResizeWindow(dpy, win, 1220, 0, 700, rh)
                    elif enable_camera and ("camera" in title or "front" in title):
                        x11.XMoveResizeWindow(dpy, win, 1220, rh, 700, 1080 - rh)
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


def _subsampled_pts(pts: Sequence[Tuple[float, float]], min_step: float = 0.18) -> List[Tuple[float, float]]:
    """Subsample a 2D polyline while preserving start and end points."""
    if len(pts) <= 2:
        return [(float(p[0]), float(p[1])) for p in pts]
    out: List[Tuple[float, float]] = [(float(pts[0][0]), float(pts[0][1]))]
    for px, py in pts[1:-1]:
        if math.hypot(float(px) - out[-1][0], float(py) - out[-1][1]) >= min_step:
            out.append((float(px), float(py)))
    ex, ey = float(pts[-1][0]), float(pts[-1][1])
    if math.hypot(ex - out[-1][0], ey - out[-1][1]) >= 0.02:
        out.append((ex, ey))
    return out


def _build_ribbon_sdf(
    model_name: str,
    pts: Sequence[Tuple[float, float]],
    width: float,
    height: float,
    z: float,
    rgba: Tuple[float, float, float, float],
    emissive_rgba: Tuple[float, float, float, float],
) -> str:
    """Build a static collisionless SDF floor ribbon connecting pts."""
    r, g, b, a = rgba
    er, eg, eb, ea = emissive_rgba
    mat_xml = (
        f"<material><ambient>{r:.3f} {g:.3f} {b:.3f} {a:.3f}</ambient>"
        f"<diffuse>{r:.3f} {g:.3f} {b:.3f} {a:.3f}</diffuse>"
        f"<emissive>{er:.3f} {eg:.3f} {eb:.3f} {ea:.3f}</emissive></material>"
    )
    visuals: List[str] = []
    for idx in range(len(pts) - 1):
        x0, y0 = pts[idx]
        x1, y1 = pts[idx + 1]
        seg_len = math.hypot(x1 - x0, y1 - y0)
        if seg_len < 0.015:
            continue
        mx = 0.5 * (x0 + x1)
        my = 0.5 * (y0 + y1)
        yaw = math.atan2(y1 - y0, x1 - x0)
        box_len = seg_len + width * 0.45
        visuals.append(
            f'<visual name="seg_{idx}"><cast_shadows>false</cast_shadows>'
            f'<pose>{mx:.4f} {my:.4f} {z:.4f} 0 0 {yaw:.4f}</pose>'
            f'<geometry><box><size>{box_len:.4f} {width:.4f} {height:.4f}</size></box></geometry>'
            f'{mat_xml}</visual>'
        )
    if not visuals:
        return ""
    return (
        f'<?xml version="1.0"?><sdf version="1.6"><model name="{model_name}">'
        f'<static>true</static><link name="link">{"".join(visuals)}</link></model></sdf>'
    )


CAMERA_OVERRIDES_3D = {
    "u_pocket_trap": "4.9 -3.35 9.8 0 1.24 1.5708",
    "slip_chicane": "4.8 -3.20 9.4 0 1.24 1.5708",
    "dynamic_crossfire_alley": "4.8 -3.20 9.4 0 1.24 1.5708",
    "smoke_lidar_corruption": "4.8 -3.20 9.4 0 1.24 1.5708",
    "dynamic_gauntlet": "0.0 -3.25 12.6 0 1.32 1.5708",
    "house_hard": "-0.3 -4.25 11.8 0 1.24 1.5708",
}


def _make_camera_world(src_world: str, dst_world: str, scenario: str) -> str:
    """Create a temporary .world copy with an angled 3D perspective overview camera."""
    pose_str = CAMERA_OVERRIDES_3D.get(scenario)
    if not pose_str or not os.path.exists(src_world):
        return src_world
    with open(src_world, "r", encoding="utf-8") as f:
        txt = f.read()
    new_txt, n = re.subn(
        r"(<camera\b[^>]*>\s*<pose(?:\s+frame='')?>)([^<]+)(</pose>)",
        rf"\g<1>{pose_str}\g<3>",
        txt,
        count=1,
    )
    if n == 0:
        return src_world
    os.makedirs(os.path.dirname(os.path.abspath(dst_world)), exist_ok=True)
    with open(dst_world, "w", encoding="utf-8") as f:
        f.write(new_txt)
    return dst_world


def _build_base_camp_sdf(rx: float, ry: float, radius: float = 0.36) -> str:
    """Build a hollow emerald-green ring on the Gazebo floor marking Base Camp without covering the robot."""
    n_seg = 16
    ring_pts: List[Tuple[float, float]] = []
    for k in range(n_seg + 1):
        ang = 2.0 * math.pi * float(k) / float(n_seg)
        ring_pts.append((rx + radius * math.cos(ang), ry + radius * math.sin(ang)))
    return _build_ribbon_sdf(
        "gz_base_camp",
        ring_pts,
        width=0.055,
        height=0.004,
        z=0.0025,
        rgba=(0.10, 0.95, 0.35, 1.0),
        emissive_rgba=(0.08, 0.80, 0.25, 1.0),
    )


def _spawn_gazebo_visual(spawn_cli, node, name: str, sdf_xml: str, wait: bool = False, keep_hb: bool = False) -> None:
    """Spawn a collisionless visual SDF in Gazebo via /spawn_entity (optionally pumping heartbeat)."""
    if not sdf_xml or spawn_cli is None:
        return
    if wait:
        spawn_cli.wait_for_service(timeout_sec=3.0)
    if not spawn_cli.service_is_ready():
        return
    req = SpawnEntity.Request()
    req.name = name
    req.xml = sdf_xml
    req.robot_namespace = ""
    req.reference_frame = "world"
    fut = spawn_cli.call_async(req)
    if wait:
        t0 = time.time()
        while not fut.done() and (time.time() - t0) < 4.0:
            if keep_hb:
                node.drive_raw(0.0, 0.0)
            rclpy.spin_once(node, timeout_sec=0.02)


def main() -> int:
    parser = argparse.ArgumentParser(description="Record 3D Gazebo + RViz2 + Front Camera evidence MP4 and PNG snapshots")
    parser.add_argument("--scenario", required=True, choices=list(SCENARIOS.keys()))
    parser.add_argument("--strategy", required=True, choices=["baseline", "proposed", "proposed_rl", "phase4", "phase4_real"])
    parser.add_argument("--onnx-path", default="", help="Optional ContextRL/Phase4 ONNX path")
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
    parser.add_argument("--out-trace-json", default="", help="Optional JSON path to save outbound/return trajectory coordinates")
    parser.add_argument("--no-camera", action="store_true", help="Disable bottom-right Robot Front Camera pane (2-pane mode)")
    parser.add_argument("--no-yolo", action="store_true", help="Disable automatic YOLO26n overlay for Phase 3/4 camera pane")
    parser.add_argument("--yolo-weights", default="/ws/yolo26n.pt", help="YOLO weights path for Phase 3/4 host_ai_pipeline")
    parser.add_argument("--strict-collision-free", action="store_true", help="Fail with exit code 2 unless arrival passes with 0 lidar/debris collisions")
    parser.add_argument("--extra-ctrl-param", action="append", default=[], help="Extra ROS2 controller param (key:=value)")
    args = parser.parse_args()

    enable_camera = not args.no_camera
    rviz_height = 640 if enable_camera else 1080
    cfg = SCENARIOS[args.scenario]

    # Locate matching A/B JSON (10trials default, fallback to any ab_<scenario>_*trials.json)
    json_path = os.path.join(args.ab_dir, f"ab_{args.scenario}_10trials.json")
    if not os.path.exists(json_path):
        cands = sorted(
            f for f in os.listdir(args.ab_dir)
            if f.startswith(f"ab_{args.scenario}_") and f.endswith("trials.json")
        ) if os.path.isdir(args.ab_dir) else []
        if cands:
            json_path = os.path.join(args.ab_dir, cands[0])
        else:
            raise FileNotFoundError(f"Missing A/B benchmark JSON: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        ab_data = json.load(f)
    specs = ab_data["fixed_specs"]
    if args.trial < 1 or args.trial > len(specs):
        raise ValueError(f"Invalid --trial {args.trial}; available 1..{len(specs)}")
    spec = specs[args.trial - 1]
    raw_deb = spec.get("debris_positions") or ab_data.get("proposed", {}).get("trials", [{}] * args.trial)[args.trial - 1].get("debris_positions") or []
    debris_positions = [tuple(p) for p in raw_deb]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_mp4)), exist_ok=True)
    os.makedirs(args.snapshot_dir, exist_ok=True)

    _ensure_gazebo_gui_ini()
    rviz_tmp = f"/tmp/disaster_recovery/record_{args.scenario}_{args.strategy}.rviz"
    src_rviz = os.path.join(WS_ROOT, "src", "disaster_recovery", "config", "disaster_station.rviz")
    _make_rviz_config(src_rviz, rviz_tmp, rviz_height=rviz_height)

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
    gazebo_models = os.path.expanduser("~/.gazebo/models")
    pkg_models = os.path.join(WS_ROOT, "src", "disaster_recovery", "models")
    env_prefix = (
        "unset QT_QPA_PLATFORM_PLUGIN_PATH QT_PLUGIN_PATH && "
        f"source /opt/ros/humble/setup.bash && source {WS_ROOT}/install/setup.bash && "
        f"export DISPLAY={args.display} {sw_env}QT_X11_NO_MITSHM=1 "
        f"ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID={args.ros_domain_id} "
        f"GAZEBO_IP=127.0.0.1 GAZEBO_MASTER_URI={gz_master_uri} "
        f"GAZEBO_MODEL_PATH='{gazebo_models}:{pkg_models}:/opt/ros/humble/share/turtlebot3_gazebo/models' "
        "TURTLEBOT3_MODEL=burger "
        'GAZEBO_MODEL_DATABASE_URI="" && '
    )
    print(f"[GL] VirtualGL={'ENABLED (Tesla V100 EGL)' if use_vgl else 'DISABLED (Mesa llvmpipe CPU)'}", flush=True)

    rx, ry, ryaw = cfg.reset_pose
    rec_world = _make_camera_world(cfg.world_file, f"/tmp/disaster_recovery/rec_world_{args.scenario}.world", args.scenario)
    p_gz = subprocess.Popen(
        [
            "bash", "-c",
            env_prefix + f"ros2 launch disaster_recovery disaster_world.launch.py x_pose:={rx} y_pose:={ry} yaw_pose:={ryaw} world:={rec_world} gui:=false",
        ],
        stdout=open(f"/tmp/disaster_recovery/rec_gz_{args.scenario}_{args.strategy}.log", "w"),
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    time.sleep(5.5)

    # Exact ab_disaster_benchmark.py launch_stack() controller parameter parity
    params_file = os.path.join(WS_ROOT, "src", "disaster_recovery", "config", "recovery_params.yaml")
    if args.strategy in ("phase4", "phase4_real"):
        actual_strategy = "proposed_rl"
        default_p4_name = (
            "real_tb3_policy_student_phase4.onnx"
            if args.strategy == "phase4_real"
            else "sim_policy_student_phase4.onnx"
        )
        default_cbf_v_min = -0.08 if args.strategy == "phase4_real" else -0.06
        eff_onnx = args.onnx_path or os.path.join(
            WS_ROOT, "src", "disaster_recovery", "config", default_p4_name
        )
        onnx_override = (
            " -p use_semantic_rl_policy:=true"
            " -p enable_dynamic_tracker:=true"
            " -p enable_cbf_safety_shield:=true"
            f" -p cbf_v_min:={default_cbf_v_min}"
            f" -p context_rl_onnx_path:={eff_onnx}"
            f" -p semantic_rl_onnx_path:={eff_onnx}"
            f" -p phase4_rl_onnx_path:={eff_onnx}"
        )
    else:
        actual_strategy = args.strategy
        eff_onnx = args.onnx_path
        if args.onnx_path and "semantic_v3" in args.onnx_path:
            onnx_override = (
                " -p use_semantic_rl_policy:=true"
                f" -p context_rl_onnx_path:={args.onnx_path}"
                f" -p semantic_rl_onnx_path:={args.onnx_path}"
            )
        else:
            onnx_override = f" -p context_rl_onnx_path:={args.onnx_path}" if args.onnx_path else ""
    smoke_override = f" -p smoke_lidar_noise_prob:={cfg.smoke_noise_prob}" if cfg.smoke_noise_prob > 0.0 else ""
    extra_override = "".join(f" -p {p}" for p in (args.extra_ctrl_param or []))
    p_ctrl = subprocess.Popen(
        [
            "bash", "-c",
            env_prefix
            + f"ros2 run disaster_recovery recovery_controller --ros-args --params-file {params_file} "
            f"-p use_sim_time:=true -p use_aruco_docking:=false -p recovery_strategy:={actual_strategy}"
            f"{onnx_override}{smoke_override}{extra_override}",
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

    is_p3_or_p4 = args.strategy in ("phase4", "phase4_real") or ("semantic_v3" in (eff_onnx or ""))
    use_yolo = bool(enable_camera and (not args.no_yolo) and is_p3_or_p4 and os.path.exists(args.yolo_weights))
    p_ai: Optional[subprocess.Popen] = None
    if use_yolo:
        p_ai = subprocess.Popen(
            [
                "bash", "-c",
                env_prefix
                + "python3 -c \"import disaster_recovery.host_ai_pipeline as hap; "
                "hap._preload_nvidia_cuda_libs = lambda: None; hap.main()\" --ros-args "
                f"-p enable_heartbeat:=false -p yolo_model_path:={args.yolo_weights} "
                f"-p image_topic:=/camera/image_raw -p use_compressed:=false -p rotation_angle:=0 "
                f"-p show_window:=false -p only_person_detection:=false "
                f"-p confidence_threshold:=0.25 -p person_conf_threshold:=0.30 -p whisper_device:=cuda",
            ],
            stdout=open(f"/tmp/disaster_recovery/rec_ai_{args.scenario}_{args.strategy}.log", "w"),
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        print(f"[YOLO] host_ai_pipeline ENABLED ({args.yolo_weights} -> /detection/image)", flush=True)
    time.sleep(4.5)

    p_ffmpeg: Optional[subprocess.Popen] = None
    p_gzclient: Optional[subprocess.Popen] = None
    stop_cam = threading.Event()
    cam_th: Optional[threading.Thread] = None
    rclpy.init()
    node = SimHarnessNode()
    node.crossfire_dynamic = bool(cfg.crossfire_dynamic)
    pub_markers = node.create_publisher(MarkerArray, "/debris_markers", 10)
    spawn_cli = node.create_client(SpawnEntity, "/spawn_entity")
    outbound_pts: List[Tuple[float, float]] = [(rx, ry)]

    if enable_camera:
        cam_lock = threading.Lock()
        cam_holder = {"raw": None, "det": None, "count": 0}
        cam_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
        det_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)

        def _decode_ros_img(msg: RosImage) -> Optional[np.ndarray]:
            try:
                enc = (msg.encoding or "rgb8").lower()
                arr = np.frombuffer(msg.data, dtype=np.uint8)
                if "rgb8" in enc:
                    return arr.reshape((msg.height, msg.width, 3))[:, :, ::-1].copy()
                if "bgr8" in enc:
                    return arr.reshape((msg.height, msg.width, 3)).copy()
            except Exception:
                return None
            return None

        def _cb_cam(msg: RosImage) -> None:
            bgr = _decode_ros_img(msg)
            if bgr is not None:
                with cam_lock:
                    cam_holder["raw"] = bgr
                    cam_holder["count"] += 1

        def _cb_det(msg: RosImage) -> None:
            bgr = _decode_ros_img(msg)
            if bgr is not None:
                with cam_lock:
                    cam_holder["det"] = bgr

        node.create_subscription(RosImage, "/camera/image_raw", _cb_cam, cam_qos)
        if use_yolo:
            node.create_subscription(RosImage, "/detection/image", _cb_det, det_qos)

        def _cam_viewer() -> None:
            import cv2  # Import lazily AFTER launching Gazebo/RViz2 so cv2 never pollutes their Qt env

            win = "Robot Front Camera"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow(win, 700, 1080 - rviz_height)
            cv2.moveWindow(win, 1220, rviz_height)
            pane_h = 1080 - rviz_height
            while not stop_cam.is_set():
                with cam_lock:
                    det_fr = None if cam_holder["det"] is None else cam_holder["det"].copy()
                    raw_fr = None if cam_holder["raw"] is None else cam_holder["raw"].copy()
                fr = det_fr if (use_yolo and det_fr is not None) else raw_fr
                using_det = bool(use_yolo and det_fr is not None)
                if fr is None:
                    canvas = np.full((pane_h, 700, 3), 24, dtype=np.uint8)
                    cv2.putText(canvas, "ROBOT FRONT CAMERA (/camera/image_raw)", (24, pane_h // 2 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (200, 220, 240), 2, cv2.LINE_AA)
                    cv2.putText(canvas, "Waiting for camera stream...", (180, pane_h // 2 + 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (130, 160, 190), 1, cv2.LINE_AA)
                else:
                    canvas = cv2.resize(fr, (700, pane_h), interpolation=cv2.INTER_LINEAR)
                cv2.rectangle(canvas, (0, 0), (700, 34), (15, 23, 42), -1)
                cv2.line(canvas, (0, 34), (700, 34), (56, 189, 248) if not using_det else (74, 222, 128), 2)
                st_txt = str(getattr(node, "ctrl_state", "INIT"))
                hdr_lbl = (
                    f"YOLO26n CAMERA (/detection/image) | {st_txt}"
                    if using_det
                    else f"ROBOT FRONT CAMERA (/camera/image_raw) | {st_txt}"
                )
                cv2.putText(canvas, hdr_lbl,
                            (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 248, 255), 1, cv2.LINE_AA)
                cv2.imshow(win, canvas)
                cv2.waitKey(30)
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            except Exception:
                pass

    success = False
    strict_pass = False
    arrival_success = False
    debris_collision = False
    lidar_collision = False
    lidar_collision_steps = 0
    min_scan_m = 10.0
    min_debris_dist_m = 99.0
    err_cm = 999.0
    telemetry_log: List[Tuple[float, float, str, str, bool, float, float, float, float, float, int]] = []
    raw_mp4 = args.out_mp4 + ".raw.mp4"

    odom_vel = [0.0, 0.0]
    try:
        from nav_msgs.msg import Odometry as _OdomMsg
        def _cb_rec_odom(msg: "_OdomMsg") -> None:
            odom_vel[0] = float(msg.twist.twist.linear.x)
            odom_vel[1] = float(msg.twist.twist.angular.z)
        node.create_subscription(_OdomMsg, "/odom", _cb_rec_odom, 10)
    except Exception:
        pass

    t_rec_wall0 = time.time()
    t_rec_sim0 = 0.0

    def _stop_ffmpeg_rec() -> None:
        nonlocal p_ffmpeg
        if p_ffmpeg is not None and p_ffmpeg.poll() is None:
            try:
                p_ffmpeg.communicate(input=b"q", timeout=6.0)
            except Exception:
                try:
                    os.killpg(os.getpgid(p_ffmpeg.pid), signal.SIGINT)
                    p_ffmpeg.wait(timeout=4.0)
                except Exception:
                    pass
        p_ffmpeg = None

    def _start_ffmpeg_rec() -> None:
        nonlocal p_ffmpeg, t_rec_wall0, t_rec_sim0
        _stop_ffmpeg_rec()
        telemetry_log.clear()
        t_rec_wall0 = time.time()
        t_rec_sim0 = node.sim_now()
        p_ffmpeg = subprocess.Popen(
            [
                "ffmpeg", "-y", "-f", "x11grab",
                "-video_size", "1920x1080", "-framerate", str(args.fps), "-i", args.display,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p",
                raw_mp4,
            ],
            stdin=subprocess.PIPE,
            stdout=open("/tmp/disaster_recovery/rec_ffmpeg.log", "w"),
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        time.sleep(0.3)

    try:
        t0 = time.time()
        while time.time() - t0 < 120.0:
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.odom_ready and node.scan_ready and node.sim_now() > 0.2:
                break
        if not (node.odom_ready and node.scan_ready):
            raise RuntimeError("Sensors failed to become ready in record_3d_evidence.py")

        # Launch gzclient AFTER gzserver & burger have spawned (prevents house world timeout)
        p_gzclient = subprocess.Popen(
            ["bash", "-c", env_prefix + f"{vgl_prefix}gzclient"],
            stdout=open(f"/tmp/disaster_recovery/rec_gzclient_{args.scenario}_{args.strategy}.log", "w"),
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        time.sleep(14.0)

        if enable_camera:
            cam_th = threading.Thread(target=_cam_viewer, daemon=True)
            cam_th.start()
            time.sleep(0.5)

        u_rate = int(round(args.rtf / cfg.max_step_size))
        run_cmd_bounded(f"GAZEBO_MASTER_URI={gz_master_uri} timeout -k 1 2 gz physics -u {u_rate} >/dev/null 2>&1", timeout_s=4.0)
        if args.scenario == "tunnel":
            run_cmd_bounded(f"GAZEBO_MASTER_URI={gz_master_uri} timeout -k 1 3 gz camera -c gzclient_camera -f burger >/dev/null 2>&1", timeout_s=4.0)
        _arrange_x11_windows(args.display, enable_camera=enable_camera, rviz_height=rviz_height)

        base_x, base_y, base_yaw = rx, ry, ryaw
        debris_cleanup_count = max(16, cfg.debris_max + 2)

        # Verified reset mirroring ab_disaster_benchmark.py::_perform_verified_reset()
        def _perform_verified_reset() -> None:
            reset_ok = False
            for rst_attempt in range(1, 5):
                node.set_heartbeat(True)
                node.stop_drive()
                node.recording_return_path = False
                node.trigger_episode_reset()
                node.sim_sleep(0.2)
                delete_debris_entities(max_count=debris_cleanup_count)
                node.reset_gazebo_world(timeout_s=3.0)
                rc_pose = run_cmd_bounded(
                    f"GAZEBO_MASTER_URI={gz_master_uri} timeout -k 1 2 gz model -m burger "
                    f"-x {base_x} -y {base_y} -z 0.01 -R 0 -P 0 -Y {base_yaw} >/dev/null 2>&1",
                    timeout_s=3.0,
                )
                run_cmd_bounded(
                    f"GAZEBO_MASTER_URI={gz_master_uri} timeout -k 1 2 gz physics -u {u_rate} >/dev/null 2>&1",
                    timeout_s=4.0,
                )
                node.sim_sleep(0.6)
                node.trigger_episode_reset()
                node.sim_sleep(0.6)

                wait_t0 = node.sim_now()
                wait_w0 = time.time()
                prev_clk = wait_t0
                mono_ticks = 0
                while (time.time() - wait_w0) < 12.0:
                    rclpy.spin_once(node, timeout_sec=0.02)
                    now_clk = node.sim_now()
                    if now_clk < prev_clk - 1e-3:
                        mono_ticks = 0
                        wait_t0 = now_clk
                    elif now_clk > prev_clk + 1e-3:
                        mono_ticks += 1
                    prev_clk = now_clk
                    pose_err = math.hypot(node.cur_x - base_x, node.cur_y - base_y)
                    if rc_pose == 0 and node.ctrl_state == "NORMAL_TELEOP" and pose_err < 0.10 and mono_ticks >= 5:
                        reset_ok = True
                        break
                    if (now_clk - wait_t0) >= 6.0 and mono_ticks >= 5:
                        break
                if reset_ok:
                    break
                print(
                    f"  [RESET RETRY {rst_attempt}/4] {cfg.name} #{args.trial:02d}: "
                    f"state={node.ctrl_state}, dist={math.hypot(node.cur_x - base_x, node.cur_y - base_y):.3f}m",
                    flush=True,
                )
            node.trigger_episode_reset()
            node.reset_outbound_grid()
            node.sim_sleep(0.4)

        # Phase 1: Outbound Exploration — exact ab_disaster_benchmark.py Reference Replay + fallback
        ref_out_path = spec.get("reference_outbound_path")
        ref_turnoff = spec.get("reference_turnoff_pose")
        ref_explored = spec.get("reference_explored_m")
        target_depth = spec["target_depth"]
        goal_pool = spec["goal_pool"]

        def _log_telem(phase_name: str, hb_on: bool, px_v: float, py_v: float, exp_m: float, state_override: Optional[str] = None) -> None:
            st_val = state_override if state_override is not None else str(node.ctrl_state)
            sim_el_val = max(0.0, node.sim_now() - t_rec_sim0)
            dist_b = math.hypot(px_v - rx, py_v - ry)
            scan_v = float(getattr(node, "scan_all_min", 10.0))
            telemetry_log.append((
                time.time() - t_rec_wall0,
                sim_el_val,
                phase_name,
                st_val,
                bool(hb_on),
                dist_b,
                float(exp_m),
                scan_v,
                float(odom_vel[0]),
                float(odom_vel[1]),
                int(lidar_collision_steps),
            ))

        explored_m = 0.0
        for out_attempt in range(2):
            _perform_verified_reset()
            _spawn_gazebo_visual(spawn_cli, node, "gz_base_camp", _build_base_camp_sdf(rx, ry), wait=True, keep_hb=True)
            _arrange_x11_windows(args.display, enable_camera=enable_camera, rviz_height=rviz_height)
            if args.scenario == "tunnel":
                run_cmd_bounded(f"GAZEBO_MASTER_URI={gz_master_uri} timeout -k 1 3 gz camera -c gzclient_camera -f burger >/dev/null 2>&1", timeout_s=4.0)

            # Start full-episode 30fps recording right from Base Camp departure (Phase 1 -> Phase 2 -> Phase 3 -> Complete)
            _start_ffmpeg_rec()

            use_ref_replay = bool(ref_out_path and len(ref_out_path) >= 2)
            active_pool = [tuple(p) for p in ref_out_path[1:]] if use_ref_replay else list(goal_pool)
            goal_ptr = 0
            goal_x, goal_y = active_pool[goal_ptr]
            goal_ptr += 1
            checkpoints_done = 0
            cp_start_t = node.sim_now()
            outbound_pts = [(node.cur_x, node.cur_y)]
            explored_m = 0.0
            prev_ox, prev_oy = node.cur_x, node.cur_y
            prev_gz_ox, prev_gz_oy = node.cur_x, node.cur_y
            out_gz_idx = 0
            anchor_x, anchor_y = node.cur_x, node.cur_y
            anchor_t = node.sim_now()
            t_exp_start = node.sim_now()
            target_exp_cap = float(ref_explored) if (use_ref_replay and ref_explored) else float(target_depth)
            last_arrange = time.time()
            last_marker_pub = 0.0
            _publish_rviz_markers(pub_markers, node, rx, ry, debris_positions, outbound_pts, show_debris=False)
            _log_telem("OUTBOUND", True, node.cur_x, node.cur_y, explored_m)

            def _phase1_done() -> bool:
                if (node.sim_now() - t_exp_start) >= cfg.outbound_timeout_sim:
                    return True
                if use_ref_replay:
                    if goal_ptr >= len(active_pool) and math.hypot(node.cur_x - goal_x, node.cur_y - goal_y) < 0.15:
                        return True
                    if explored_m >= target_exp_cap and ref_turnoff and math.hypot(node.cur_x - ref_turnoff[0], node.cur_y - ref_turnoff[1]) < 0.25:
                        return True
                    if explored_m >= target_exp_cap * 1.04:
                        return True
                    return False
                if explored_m < target_depth:
                    return False
                if cfg.min_blackout_dist > 0.0 and math.hypot(node.cur_x - base_x, node.cur_y - base_y) < cfg.min_blackout_dist:
                    return False
                if cfg.min_checkpoints > 0 and checkpoints_done < cfg.min_checkpoints:
                    return False
                return True

            t_wall_out0 = time.time()
            max_wall_out = max(120.0, cfg.outbound_timeout_sim * 1.5)
            while not _phase1_done():
                if time.time() - t_wall_out0 > max_wall_out:
                    break
                t_tick = node.sim_now()
                t_w_tick = time.time()
                while node.sim_now() - t_tick < 0.05 and (time.time() - t_w_tick) < 0.6:
                    rclpy.spin_once(node, timeout_sec=0.01)

                step_d = math.hypot(node.cur_x - prev_ox, node.cur_y - prev_oy)
                explored_m += step_d
                prev_ox, prev_oy = node.cur_x, node.cur_y
                if math.hypot(node.cur_x - outbound_pts[-1][0], node.cur_y - outbound_pts[-1][1]) >= 0.08:
                    outbound_pts.append((node.cur_x, node.cur_y))
                if math.hypot(node.cur_x - prev_gz_ox, node.cur_y - prev_gz_oy) >= 0.18:
                    seg_out_sdf = _build_ribbon_sdf(
                        f"gz_out_{out_gz_idx}",
                        [(prev_gz_ox, prev_gz_oy), (node.cur_x, node.cur_y)],
                        width=0.042,
                        height=0.003,
                        z=0.0025,
                        rgba=(0.05, 0.78, 1.0, 1.0),
                        emissive_rgba=(0.04, 0.62, 0.92, 1.0),
                    )
                    _spawn_gazebo_visual(spawn_cli, node, f"gz_out_{out_gz_idx}", seg_out_sdf, wait=False)
                    out_gz_idx += 1
                    prev_gz_ox, prev_gz_oy = node.cur_x, node.cur_y

                _log_telem("OUTBOUND", True, node.cur_x, node.cur_y, explored_m)

                if time.time() - last_marker_pub > 0.10:
                    _publish_rviz_markers(pub_markers, node, rx, ry, debris_positions, outbound_pts, show_debris=False)
                    last_marker_pub = time.time()

                if use_ref_replay:
                    while goal_ptr < len(active_pool) and math.hypot(node.cur_x - goal_x, node.cur_y - goal_y) < 0.16:
                        goal_x, goal_y = active_pool[goal_ptr]
                        goal_ptr += 1
                        checkpoints_done += 1
                        cp_start_t = node.sim_now()
                    v_cmd, w_cmd = node.choose_guided_astar_cmd(goal_x, goal_y)
                elif cfg.funnel_collapsed and node.cur_y < 1.2:
                    target_h = max(1.10, min(2.04, 1.57 + 0.45 * (node.cur_x - (-0.75))))
                    h_diff = normalize_angle(target_h - node.cur_th)
                    w_cmd = max(-0.35, min(0.35, 1.2 * h_diff + 0.25 * (node.scan_right_min - node.scan_left_min)))
                    v_cmd = 0.20
                    if node.scan_front_min < 0.45:
                        v_cmd = 0.08
                        w_cmd = 0.45 if node.scan_left_min > node.scan_right_min else -0.45
                elif cfg.use_guided_outbound:
                    if math.hypot(node.cur_x - goal_x, node.cur_y - goal_y) < 0.55:
                        goal_x, goal_y = goal_pool[goal_ptr % len(goal_pool)]
                        goal_ptr += 1
                        checkpoints_done += 1
                        cp_start_t = node.sim_now()
                    elif node.sim_now() - cp_start_t > 15.0:
                        goal_x, goal_y = goal_pool[goal_ptr % len(goal_pool)]
                        goal_ptr += 1
                        cp_start_t = node.sim_now()
                    if node.sim_now() - anchor_t > 3.5:
                        if math.hypot(node.cur_x - anchor_x, node.cur_y - anchor_y) < 0.12:
                            turn_d = 0.60 if node.scan_left_min >= node.scan_right_min else -0.60
                            esc_t0 = node.sim_now()
                            esc_w0 = time.time()
                            while node.sim_now() - esc_t0 < 0.9 and (time.time() - esc_w0) < 3.0:
                                node.drive_raw(-0.08, turn_d)
                                rclpy.spin_once(node, timeout_sec=0.01)
                        anchor_x, anchor_y = node.cur_x, node.cur_y
                        anchor_t = node.sim_now()
                    v_cmd, w_cmd = node.choose_guided_astar_cmd(goal_x, goal_y)
                else:
                    if math.hypot(node.cur_x - goal_x, node.cur_y - goal_y) < 1.3:
                        goal_x, goal_y = goal_pool[goal_ptr % len(goal_pool)]
                        goal_ptr += 1
                    if node.sim_now() - anchor_t > 3.5:
                        if math.hypot(node.cur_x - anchor_x, node.cur_y - anchor_y) < 0.12:
                            turn_d = 0.60 if node.scan_left_min >= node.scan_right_min else -0.60
                            esc_t0 = node.sim_now()
                            esc_w0 = time.time()
                            while node.sim_now() - esc_t0 < 1.0 and (time.time() - esc_w0) < 3.0:
                                node.drive_raw(-0.10, turn_d)
                                rclpy.spin_once(node, timeout_sec=0.01)
                            goal_x, goal_y = goal_pool[goal_ptr % len(goal_pool)]
                            goal_ptr += 1
                        anchor_x, anchor_y = node.cur_x, node.cur_y
                        anchor_t = node.sim_now()
                    v_cmd, w_cmd = node.choose_open_gap_cmd(goal_x, goal_y)

                node.drive_raw(v_cmd, w_cmd)
                if time.time() - last_arrange > 4.0:
                    _arrange_x11_windows(args.display, enable_camera=enable_camera, rviz_height=rviz_height)
                    last_arrange = time.time()

            node.stop_drive()
            if math.hypot(node.cur_x - prev_gz_ox, node.cur_y - prev_gz_oy) >= 0.03:
                seg_out_sdf = _build_ribbon_sdf(
                    f"gz_out_{out_gz_idx}",
                    [(prev_gz_ox, prev_gz_oy), (node.cur_x, node.cur_y)],
                    width=0.042,
                    height=0.003,
                    z=0.0025,
                    rgba=(0.05, 0.78, 1.0, 1.0),
                    emissive_rgba=(0.04, 0.62, 0.92, 1.0),
                )
                _spawn_gazebo_visual(spawn_cli, node, f"gz_out_{out_gz_idx}", seg_out_sdf, wait=False)
                out_gz_idx += 1
            node.sim_sleep(0.3)

            if ref_turnoff is not None and ref_explored is not None:
                d_exp = abs(explored_m - float(ref_explored))
                d_turn = math.hypot(node.cur_x - float(ref_turnoff[0]), node.cur_y - float(ref_turnoff[1]))
                if (d_exp >= 1.0 or d_turn >= 1.5) and out_attempt == 0:
                    print(
                        f"  [OUTBOUND ASYMMETRY RETRY] {cfg.name} #{args.trial:02d}: "
                        f"|d_exp|={d_exp:.2f}m, d_turn={d_turn:.2f}m -> retrying outbound (restarting clean recording)",
                        flush=True,
                    )
                    _stop_ffmpeg_rec()
                    continue
                elif d_exp >= 1.0 or d_turn >= 1.5:
                    print(
                        f"  [OUTBOUND ASYMMETRY WARN] {cfg.name} #{args.trial:02d}: "
                        f"|d_exp|={d_exp:.2f}m, d_turn={d_turn:.2f}m",
                        flush=True,
                    )
            break

        # Phase 2: Spawn Debris & Activate Crossfire LIVE during recording (keeps heartbeat alive!)
        return_pts: List[Tuple[float, float]] = [(node.cur_x, node.cur_y)]
        _log_telem("DEBRIS_SPAWN", True, node.cur_x, node.cur_y, explored_m)
        if debris_positions:
            def _hb_tick() -> None:
                node.drive_raw(0.0, 0.0)
                rclpy.spin_once(node, timeout_sec=0.01)
                _log_telem("DEBRIS_SPAWN", True, node.cur_x, node.cur_y, explored_m)

            spawn_random_debris_cluster(debris_positions, wait=True, tick_fn=_hb_tick)
        _publish_rviz_markers(pub_markers, node, rx, ry, debris_positions, outbound_pts, return_pts=return_pts, show_debris=True)
        _log_telem("DEBRIS_SPAWN", True, node.cur_x, node.cur_y, explored_m)
        node.stop_drive()
        node.sim_sleep(0.2)

        turnoff_x, turnoff_y = node.cur_x, node.cur_y
        if cfg.crossfire_dynamic:
            # Keep heartbeat alive during gz model placement
            node.drive_raw(0.0, 0.0)
            rclpy.spin_once(node, timeout_sec=0.01)
            run_cmd_bounded(
                f"GAZEBO_MASTER_URI={gz_master_uri} timeout -k 1 2 gz model -m dyn_obs_1 -x 2.15 -y 0.00 -z 0.25 >/dev/null 2>&1 && "
                f"GAZEBO_MASTER_URI={gz_master_uri} timeout -k 1 2 gz model -m dyn_obs_2 -x 5.05 -y 0.725 -z 0.25 >/dev/null 2>&1",
                timeout_s=4.0,
            )
            node.drive_raw(0.0, 0.0)
            rclpy.spin_once(node, timeout_sec=0.01)

        # Phase 3: Trigger Blackout & Monitor Autonomous Recovery (exact ab_disaster_benchmark.py logic)
        node.crossfire_t0 = node.sim_now()
        node.set_heartbeat(False)
        t_blackout = node.sim_now()
        _log_telem("BLACKOUT", False, node.cur_x, node.cur_y, explored_m)
        prev_sim_clock = t_blackout
        sim_elapsed_s = 0.0
        max_return_wait = min(float(args.max_sim_sec), max(cfg.return_timeout_base, explored_m * 16.0))
        max_arrival_error_m = 0.45
        states_visited = set()
        saw_recovery = False
        prev_rx, prev_ry = node.cur_x, node.cur_y
        prev_gz_rx, prev_gz_ry = node.cur_x, node.cur_y
        ret_gz_idx = 0
        last_marker_pub = 0.0

        latest_tp: List[Optional[Tuple[float, float, float]]] = [get_gazebo_true_pose("burger")]
        stop_tp_poll = threading.Event()

        def _poll_true_pose() -> None:
            while not stop_tp_poll.is_set():
                tp_val = get_gazebo_true_pose("burger")
                if tp_val is not None:
                    latest_tp[0] = tp_val
                stop_tp_poll.wait(0.04)

        tp_thread = threading.Thread(target=_poll_true_pose, daemon=True)
        tp_thread.start()
        try:
            t_wall_ret0 = time.time()
            max_wall_ret = max(180.0, max_return_wait * 1.5)
            while sim_elapsed_s < max_return_wait and (node.sim_now() - t_blackout) < max_return_wait:
                if time.time() - t_wall_ret0 > max_wall_ret:
                    break
                t_s = node.sim_now()
                t_w_step = time.time()
                while (node.sim_now() - t_s) < 0.05 and (time.time() - t_w_step) < 0.6:
                    rclpy.spin_once(node, timeout_sec=0.008)
                    now_clk = node.sim_now()
                    dt_clk = now_clk - prev_sim_clock
                    if 0.0 < dt_clk < 5.0:
                        sim_elapsed_s += dt_clk
                    elif dt_clk < -1e-3:
                        t_s = now_clk
                    prev_sim_clock = now_clk

                tp_now = (node.true_x, node.true_y, node.true_th) if node.true_pose_ready else latest_tp[0]
                px_chk, py_chk = (tp_now[0], tp_now[1]) if tp_now is not None else (node.cur_x, node.cur_y)

                if math.hypot(px_chk - prev_rx, py_chk - prev_ry) >= 0.03:
                    prev_rx, prev_ry = px_chk, py_chk
                    return_pts.append((px_chk, py_chk))
                if math.hypot(px_chk - prev_gz_rx, py_chk - prev_gz_ry) >= 0.18:
                    seg_sdf = _build_ribbon_sdf(
                        f"gz_ret_{ret_gz_idx}",
                        [(prev_gz_rx, prev_gz_ry), (px_chk, py_chk)],
                        width=0.048,
                        height=0.004,
                        z=0.0045,
                        rgba=(0.10, 1.0, 0.28, 1.0),
                        emissive_rgba=(0.08, 0.92, 0.22, 1.0),
                    )
                    _spawn_gazebo_visual(spawn_cli, node, f"gz_ret_{ret_gz_idx}", seg_sdf, wait=False)
                    ret_gz_idx += 1
                    prev_gz_rx, prev_gz_ry = px_chk, py_chk

                cur_phase = "BLACKOUT" if (not saw_recovery or (node.sim_now() - t_blackout) <= 2.5) else "RECOVERY"
                _log_telem(cur_phase, False, px_chk, py_chk, explored_m)
                if time.time() - last_marker_pub > 0.04:
                    _publish_rviz_markers(pub_markers, node, rx, ry, debris_positions, outbound_pts, return_pts=return_pts, show_debris=True)
                    last_marker_pub = time.time()

                states_visited.add(node.ctrl_state)
                if node.ctrl_state in (
                    "BACKTRACKING_ROTATE",
                    "BACKTRACKING_TRANSLATE",
                    "BACKTRACKING_HYBRID",
                    "LOCAL_TD3_AVOIDANCE",
                    "OBSTACLE_HOLD",
                    "HANDOVER_TO_ARUCO",
                    "MISSION_COMPLETE",
                ):
                    saw_recovery = True
                elif node.ctrl_state != "NORMAL_TELEOP" and math.hypot(px_chk - turnoff_x, py_chk - turnoff_y) >= 0.15:
                    saw_recovery = True

                if saw_recovery:
                    if node.scan_all_min < min_scan_m:
                        min_scan_m = node.scan_all_min
                    if node.scan_all_min <= 0.121:
                        lidar_collision_steps += 1
                    if debris_positions:
                        cur_dd = min(math.hypot(px_chk - dx, py_chk - dy) for dx, dy in debris_positions)
                        min_debris_dist_m = min(min_debris_dist_m, cur_dd)

                if node.ctrl_state == "MISSION_COMPLETE" or (saw_recovery and node.ctrl_state in ("NORMAL_TELEOP", "ARUCO_DOCKING")):
                    saw_recovery = True
                    success = True
                    if math.hypot(px_chk - prev_gz_rx, py_chk - prev_gz_ry) >= 0.03:
                        seg_sdf = _build_ribbon_sdf(
                            f"gz_ret_{ret_gz_idx}",
                            [(prev_gz_rx, prev_gz_ry), (px_chk, py_chk)],
                            width=0.048,
                            height=0.004,
                            z=0.0045,
                            rgba=(0.10, 1.0, 0.28, 1.0),
                            emissive_rgba=(0.08, 0.92, 0.22, 1.0),
                        )
                        _spawn_gazebo_visual(spawn_cli, node, f"gz_ret_{ret_gz_idx}", seg_sdf, wait=False)
                        ret_gz_idx += 1
                    return_pts.append((px_chk, py_chk))
                    _publish_rviz_markers(pub_markers, node, rx, ry, debris_positions, outbound_pts, return_pts=return_pts, show_debris=True)
                    for _ in range(10):
                        node.sim_sleep(0.12)
                        _log_telem("COMPLETE", False, px_chk, py_chk, explored_m, state_override="MISSION_COMPLETE")
                    break
        finally:
            stop_tp_poll.set()
            tp_thread.join(timeout=1.0)

        true_p = (node.true_x, node.true_y, node.true_th) if node.true_pose_ready else (get_gazebo_true_pose("burger") or latest_tp[0])
        fx, fy = (true_p[0], true_p[1]) if true_p else (node.cur_x, node.cur_y)
        err_m = math.hypot(fx - rx, fy - ry)
        err_cm = err_m * 100.0
        if success and err_m > max_arrival_error_m:
            success = False
        arrival_success = bool(success)
        debris_collision = bool(debris_positions and min_debris_dist_m < 0.265)
        lidar_collision = bool(lidar_collision_steps >= 4)
        strict_pass = bool(arrival_success and lidar_collision_steps == 0 and (not debris_collision))
        if args.strict_collision_free:
            success = strict_pass

        print(
            f"[RECORDED] {args.scenario}/{args.strategy} trial#{args.trial} | "
            f"success={success} strict_pass={strict_pass} lidar_col={lidar_collision_steps} "
            f"min_scan={min_scan_m:.3f}m dcol={debris_collision} min_deb={min_debris_dist_m:.3f}m "
            f"err={err_cm:.1f}cm sim_dur={node.sim_now()-t_blackout:.1f}s -> {args.out_mp4}",
            flush=True,
        )
        if args.out_trace_json:
            os.makedirs(os.path.dirname(os.path.abspath(args.out_trace_json)), exist_ok=True)
            trace_payload = {
                "scenario": args.scenario,
                "strategy": args.strategy,
                "trial": int(args.trial),
                "seed": spec.get("seed"),
                "target_depth_m": spec.get("target_depth"),
                "success": bool(success),
                "strict_pass": bool(strict_pass),
                "lidar_collision_steps": int(lidar_collision_steps),
                "min_scan_m": round(float(min_scan_m), 4),
                "debris_collision": bool(debris_collision),
                "min_debris_dist_m": round(float(min_debris_dist_m), 4),
                "error_cm": round(float(err_cm), 2),
                "return_duration_s": round(float(node.sim_now() - t_blackout), 2),
                "base_pose": [round(float(rx), 4), round(float(ry), 4)],
                "turnoff_pose": [round(float(turnoff_x), 4), round(float(turnoff_y), 4)],
                "debris_positions": [[round(float(dx), 4), round(float(dy), 4)] for dx, dy in debris_positions],
                "outbound_path": [[round(float(x), 4), round(float(y), 4)] for x, y in outbound_pts],
                "return_path": [[round(float(x), 4), round(float(y), 4)] for x, y in return_pts],
            }
            with open(args.out_trace_json, "w", encoding="utf-8") as tf:
                json.dump(trace_payload, tf, indent=2, ensure_ascii=False)
            print(f"[TRACE SAVED] {args.out_trace_json} (out={len(outbound_pts)} pts, ret={len(return_pts)} pts)", flush=True)

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
        stop_cam.set()
        if cam_th is not None:
            try:
                cam_th.join(timeout=2.5)
            except Exception:
                pass
        try:
            delete_debris_entities(max_count=16)
        except Exception:
            pass
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
        for p in (p_ai, p_rviz, p_slam, p_gzclient, p_ctrl, p_gz, p_xvfb):
            if p is None:
                continue
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass
        time.sleep(0.4)
        for p in (p_ai, p_rviz, p_slam, p_gzclient, p_ctrl, p_gz, p_xvfb):
            if p is None:
                continue
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                pass
        _kill_matching_gazebo_procs(gz_master_uri)

    if args.strict_collision_free and not strict_pass:
        try:
            if os.path.exists(raw_mp4):
                os.remove(raw_mp4)
        except Exception:
            pass
        print(
            f"[STRICT FAIL] Discarding recording for {args.scenario}/{args.strategy} trial#{args.trial} "
            f"(lidar_col={lidar_collision_steps}, dcol={debris_collision}, err={err_cm:.1f}cm)",
            flush=True,
        )
        return 2

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
        onnx_path=eff_onnx,
        enable_camera=enable_camera,
        lidar_collision_steps=lidar_collision_steps,
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
    telemetry_log: Sequence[Tuple[float, float, str, str, bool, float, float, float, float, float, int]],
    success: bool,
    err_cm: float,
    onnx_path: str = "",
    enable_camera: bool = True,
    lidar_collision_steps: int = 0,
) -> None:
    """Stamp top telemetry bar and live 2-row bottom situational HUD onto every frame of raw_mp4 -> out_mp4."""
    if not os.path.exists(raw_mp4):
        return
    try:
        from PIL import Image, ImageDraw, ImageFont
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
    font_sub = ImageFont.load_default()
    font_pill = ImageFont.load_default()
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font_top = ImageFont.truetype(fp, 19)
                font_bot = ImageFont.truetype(fp, 21)
                font_sub = ImageFont.truetype(fp, 17)
                font_pill = ImageFont.truetype(fp, 15)
                break
            except Exception:
                continue

    is_sem_v3 = "semantic_v3" in (onnx_path or "")
    strat_label = {
        "baseline": "Baseline (단순 역추적)",
        "proposed": "Phase 1 (proposed A*)",
        "proposed_rl": (
            "Phase 3 (3-Ch Semantic Context-RL)"
            if is_sem_v3
            else "Phase 2-B (Adaptive Safety + RL)"
        ),
        "phase4": "Phase 4 Step B-4 (Sim v4.4 · LiDAR+Cam+CBF)",
        "phase4_real": "Phase 4 Step B-4 (Real TB3 v3 · LiDAR+Cam+CBF)",
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
            *vcodec_args, "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            out_mp4,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    frame_bytes = 1920 * 1080 * 3
    frame_idx = 0
    total_wall = telemetry_log[-1][0] if telemetry_log else 10.0

    # Pre-index telemetry timestamps for O(1) advancing lookup
    tlog = list(telemetry_log)
    t_ptr = 0

    while True:
        buf = dec.stdout.read(frame_bytes) if dec.stdout else b""
        if len(buf) < frame_bytes:
            break
        t_wall = frame_idx / float(max(1, fps))
        frame_idx += 1

        # Default telemetry values
        sim_el, phase, st_name, hb_on = 0.0, "OUTBOUND", "NORMAL_TELEOP", True
        dist_base, explored_m_now, scan_min, lin_v, ang_v, col_now = 0.0, 0.0, 10.0, 0.0, 0.0, 0
        if tlog:
            while t_ptr + 1 < len(tlog) and abs(tlog[t_ptr + 1][0] - t_wall) <= abs(tlog[t_ptr][0] - t_wall):
                t_ptr += 1
            (
                _,
                sim_el,
                phase,
                st_name,
                hb_on,
                dist_base,
                explored_m_now,
                scan_min,
                lin_v,
                ang_v,
                col_now,
            ) = tlog[t_ptr]

        img = Image.frombytes("RGB", (1920, 1080), buf)
        draw = ImageDraw.Draw(img, "RGBA")

        # Top HUD Bar
        draw.rectangle([(0, 0), (1920, 48)], fill=(15, 23, 42, 225))
        draw.line([(0, 48), (1920, 48)], fill=(56, 189, 248, 255), width=2)
        top_txt = (
            f"{strat_label} | {scenario} #{trial:02d} | "
            f"{st_name} | SIM {sim_el:5.1f}s | 기지거리 {dist_base:4.2f}m | COL:{col_now}"
        )
        draw.text((16, 12), top_txt, font=font_top, fill=(248, 250, 252, 255))
        legend_txt = (
            "[Gazebo+RViz2+Cam: 하늘=진입 · 초록=복귀 · 노랑=낙석]"
            if enable_camera
            else "[Gazebo+RViz2: 하늘=진입 · 초록=복귀 · 노랑=낙석]"
        )
        draw.text((1240, 12), legend_txt, font=font_top, fill=(134, 239, 172, 255))

        # Determine live stage index, event description, and accent colors
        if phase == "OUTBOUND":
            stage_idx = 0
            ev_txt = f"원격 진입 주행 중 — Base Camp 출발, 재난 구역 탐색 (탐색 {explored_m_now:.2f}m)"
            bd_col = (56, 189, 248, 255)
            hdr_bg = (12, 74, 110, 220)
        elif phase == "DEBRIS_SPAWN":
            stage_idx = 1
            ev_txt = "2차 붕괴 낙석 발생 — 진입로 차단 감지 (통신 단절 임박)"
            bd_col = (251, 146, 60, 255)
            hdr_bg = (154, 52, 18, 230)
        elif phase == "BLACKOUT":
            stage_idx = 1
            ev_txt = "통신 단절(BLACKOUT) 발생 — 제어 신호 두절 -> 자율 복귀 모드 전환"
            bd_col = (248, 113, 113, 255)
            hdr_bg = (153, 27, 27, 235)
        elif phase == "COMPLETE" or (success and st_name == "MISSION_COMPLETE"):
            stage_idx = 3
            ev_txt = f"BASE CAMP 자율 복귀 성공 — 최종 오차 {err_cm:.1f}cm | 충돌 {lidar_collision_steps}회"
            bd_col = (74, 222, 128, 255)
            hdr_bg = (21, 128, 61, 235)
        elif (not success) and t_wall >= total_wall - 2.0:
            stage_idx = 2
            ev_txt = f"낙석 구간 교착/충돌 — 복귀 실패 (잔여 오차 {err_cm:.1f}cm)"
            bd_col = (248, 113, 113, 255)
            hdr_bg = (153, 27, 27, 235)
        else:
            stage_idx = 2
            if st_name == "BACKTRACKING_ROTATE":
                ev_txt = f"자율 복귀 — 귀환 방위 정렬 중 (잔여 {dist_base:.2f}m)"
                bd_col = (56, 189, 248, 255)
                hdr_bg = (30, 58, 138, 225)
            elif st_name == "LOCAL_TD3_AVOIDANCE":
                ev_txt = f"자율 복귀 · RL/CBF 회피 — 장애물 근접({scan_min:.2f}m), 실시간 회피 기동"
                bd_col = (250, 204, 21, 255)
                hdr_bg = (133, 77, 14, 230)
            elif st_name == "OBSTACLE_HOLD":
                ev_txt = f"자율 복귀 · 안전 정지 — 전방 장애물({scan_min:.2f}m) 안전 대기"
                bd_col = (251, 146, 60, 255)
                hdr_bg = (154, 52, 18, 230)
            else:
                ev_txt = f"자율 복귀 주행 중 — 낙석·동적 장애물 우회 및 기지 귀환 (잔여 {dist_base:.2f}m)"
                bd_col = (45, 212, 191, 255)
                hdr_bg = (15, 118, 110, 225)

        # Bottom Live Situational HUD Panel (left Gazebo pane width: x=12..1208, never overlaps right Camera pane!)
        hud_x0, hud_y0, hud_x1, hud_y1 = 12, 974, 1208, 1074
        draw.rounded_rectangle([(hud_x0, hud_y0), (hud_x1, hud_y1)], radius=10, fill=(10, 16, 30, 236), outline=bd_col, width=2)
        draw.rounded_rectangle([(hud_x0 + 2, hud_y0 + 2), (hud_x1 - 2, hud_y0 + 48)], radius=8, fill=hdr_bg)
        draw.line([(hud_x0 + 10, hud_y0 + 50), (hud_x1 - 10, hud_y0 + 50)], fill=(100, 116, 139, 160), width=1)

        # Row 1 Left: 4-Step Mission Progress Pills
        steps_labels = ["1.진입", "2.낙석/단절", "3.자율복귀", "4.완료"]
        px_cur = hud_x0 + 12
        for idx_s, s_lbl in enumerate(steps_labels):
            pw = 74 if idx_s in (0, 3) else (106 if idx_s == 1 else 96)
            if idx_s == stage_idx:
                p_fill = (248, 250, 252, 240)
                p_txt = (15, 23, 42, 255)
            elif idx_s < stage_idx:
                p_fill = (22, 101, 52, 210)
                p_txt = (187, 247, 208, 255)
            else:
                p_fill = (30, 41, 59, 180)
                p_txt = (148, 163, 184, 255)
            draw.rounded_rectangle([(px_cur, hud_y0 + 10), (px_cur + pw, hud_y0 + 38)], radius=6, fill=p_fill)
            draw.text((px_cur + 9, hud_y0 + 15), s_lbl, font=font_pill, fill=p_txt)
            px_cur += pw + 8

        # Row 1 Right: Live Situational Event Brief
        draw.text((px_cur + 8, hud_y0 + 13), ev_txt, font=font_bot, fill=(255, 255, 255, 255))

        # Row 2: Live Telemetry Readout Bar
        y_row2 = hud_y0 + 62
        link_txt = "통신: 정상(ONLINE)" if hb_on else "통신: 단절(BLACKOUT)"
        link_col = (74, 222, 128, 255) if hb_on else (248, 113, 113, 255)
        draw.text((hud_x0 + 16, y_row2), link_txt, font=font_sub, fill=link_col)

        draw.text((hud_x0 + 218, y_row2), f"상태: {st_name}", font=font_sub, fill=(226, 232, 240, 255))

        vel_txt = f"속도: v={lin_v:+.2f}m/s  w={ang_v:+.2f}rad/s"
        draw.text((hud_x0 + 492, y_row2), vel_txt, font=font_sub, fill=(186, 230, 253, 255))

        scan_col = (248, 113, 113, 255) if scan_min < 0.16 else ((251, 191, 36, 255) if scan_min < 0.28 else (134, 239, 172, 255))
        scan_disp = f"{scan_min:.2f}m" if scan_min < 9.5 else ">3.5m"
        draw.text((hud_x0 + 798, y_row2), f"LiDAR최소: {scan_disp}", font=font_sub, fill=scan_col)

        draw.text((hud_x0 + 972, y_row2), f"기지: {dist_base:.2f}m", font=font_sub, fill=(226, 232, 240, 255))

        col_col = (74, 222, 128, 255) if col_now == 0 else (248, 113, 113, 255)
        draw.text((hud_x0 + 1112, y_row2), f"COL: {col_now}", font=font_sub, fill=col_col)

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
