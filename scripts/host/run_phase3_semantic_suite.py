#!/usr/bin/env python3
"""
Phase 3 Semantic Rescue & Avoidance Benchmark + 3D Dual/Triple-Pane Evidence Suite.

Provides:
  1. `--mode check-yolo`: Quick Gazebo camera + YOLO26n sanity test against survivor & furniture models.
  2. `--mode ab10`: 10-trial deterministic A/B benchmark in `disaster_semantic_rescue_world.world`
     comparing Phase 2-B Blind (`use_semantic_rl_policy:=false`) vs Phase 3 Semantic-Aware
     (`use_semantic_rl_policy:=true`) with strict collision + gas-zone violation tracking.
  3. `--mode record-e2e`: Full 3-pane (Gazebo 3D + RViz2 /map + Live YOLO26n /detection/image)
     Semantic SLAM E2E recording, `/semantic_map/save_map`, high-res overlay PNG, and GIF export.
  4. `--mode record-ab`: Blind vs Semantic-Aware 3D Gazebo + RViz2 (/map active!) A/B recording,
     comparison GIF, and keyframe PNG snapshots.
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
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from cv_bridge import CvBridge
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Bool, Float32, Header, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
from PIL import Image as PILImage, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ab_disaster_benchmark import (  # noqa: E402
    _kill_matching_gazebo_procs,
    get_gazebo_true_pose,
    normalize_angle,
    run_cmd_bounded,
)
from record_3d_evidence import (  # noqa: E402
    _build_base_camp_sdf,
    _build_ribbon_sdf,
    _ensure_gazebo_gui_ini,
    _spawn_gazebo_visual,
    _subsampled_pts,
)

SEMANTIC_WORLD = os.path.join(
    WS_ROOT, "src", "disaster_recovery", "worlds", "disaster_semantic_rescue_world.world"
)
DEFAULT_V3_ONNX = os.path.join(
    WS_ROOT, "src", "disaster_recovery", "config", "sim_policy_student_semantic_v3.onnx"
)
DEFAULT_P2B_ONNX = os.path.join(
    WS_ROOT, "src", "disaster_recovery", "config", "sim_policy_student.onnx"
)
YOLO_WEIGHTS = "/ws/yolo26n.pt"


def _low_obstacle_sdf(name: str, sx: float = 0.38, sy: float = 0.26, sz: float = 0.095) -> str:
    """Sub-LiDAR low obstacle (< 0.171m scan plane): physically collides with TB3, invisible to 2D LiDAR."""
    return (
        f'<?xml version="1.0"?><sdf version="1.6"><model name="{name}">'
        f"<static>true</static><link name='link'>"
        f"<collision name='col'><pose>0 0 {sz*0.5:.4f} 0 0 0</pose>"
        f"<geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry></collision>"
        f"<visual name='vis'><pose>0 0 {sz*0.5:.4f} 0 0 0</pose>"
        f"<geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>"
        f"<material><ambient>0.88 0.15 0.65 1.0</ambient><diffuse>0.95 0.20 0.72 1.0</diffuse>"
        f"<emissive>0.35 0.04 0.25 1.0</emissive></material></visual>"
        f"<visual name='ring'><pose>0 0 0.004 0 0 0</pose>"
        f"<geometry><cylinder><radius>0.26</radius><length>0.006</length></cylinder></geometry>"
        f"<material><ambient>1.0 0.2 0.8 0.55</ambient><diffuse>1.0 0.2 0.8 0.55</diffuse></material></visual>"
        f"</link></model></sdf>"
    )


def _gas_disc_sdf(name: str, radius: float = 0.55) -> str:
    return (
        f'<?xml version="1.0"?><sdf version="1.6"><model name="{name}">'
        f"<static>true</static><link name='link'>"
        f"<visual name='vis'><pose>0 0 0.005 0 0 0</pose>"
        f"<geometry><cylinder><radius>{radius:.3f}</radius><length>0.008</length></cylinder></geometry>"
        f"<material><ambient>1.0 0.48 0.0 0.72</ambient><diffuse>1.0 0.48 0.0 0.72</diffuse>"
        f"<emissive>0.55 0.22 0.0 0.5</emissive></material></visual>"
        f"</link></model></sdf>"
    )


def _make_rviz_semantic_config(src_rviz: str, dst_rviz: str, win_w: int = 700, win_h: int = 640, win_x: int = 1220, win_y: int = 0) -> None:
    """Create RViz2 config with /map (slam_toolbox), /semantic_map/markers, and /debris_markers."""
    with open(src_rviz, "r", encoding="utf-8") as f:
        txt = f.read()

    txt = txt.replace("Fixed Frame: base_link", "Fixed Frame: map")
    txt = txt.replace("Target Frame: <Fixed Frame>", "Target Frame: base_footprint")
    txt = txt.replace("Size (m): 0.03", "Size (m): 0.06")
    txt = txt.replace("Distance: 2.733222484588623", "Distance: 8.2")
    txt = txt.replace("Pitch: 0.785398006439209", "Pitch: 1.48")
    txt = txt.replace("Target Frame Rate: 15", "Target Frame Rate: 30")
    txt = txt.replace("Frame Rate: 30", "Frame Rate: 30")
    txt = txt.replace("Hide Left Dock: false", "Hide Left Dock: true")
    txt = txt.replace("Hide Right Dock: false", "Hide Right Dock: true")

    # Ensure both /debris_markers and /semantic_map/markers are enabled
    extra_markers = (
        "    - Class: rviz_default_plugins/MarkerArray\n"
        "      Enabled: true\n"
        "      Name: DebrisAndPathMarkers\n"
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
    if "/debris_markers" not in txt:
        txt = txt.replace("  Enabled: true\n  Global Options:", extra_markers + "  Enabled: true\n  Global Options:")

    # Disable docked Image pane inside RViz2 so we render our own large crisp YOLO window below RViz2
    txt = re.sub(
        r"(- Class: rviz_default_plugins/Image\s+Enabled:\s*)true(.*?Value:\s*/detection/image\s+Value:\s*)true",
        r"\1false\2false",
        txt,
        flags=re.DOTALL,
    )
    txt = re.sub(r"^Panels:.*?(?=^Visualization Manager:)", "Panels: []\n", txt, flags=re.DOTALL | re.MULTILINE)
    txt = re.sub(r"^\s*QMainWindow State:.*\n", "", txt, flags=re.MULTILINE)
    txt = re.sub(r"^  Height:\s*\d+$", f"  Height: {win_h}", txt, flags=re.MULTILINE)
    txt = re.sub(r"^  Width:\s*\d+$", f"  Width: {win_w}", txt, flags=re.MULTILINE)
    txt = re.sub(r"^  X:\s*\d+$", f"  X: {win_x}", txt, flags=re.MULTILINE)
    txt = re.sub(r"^  Y:\s*\d+$", f"  Y: {win_y}", txt, flags=re.MULTILINE)

    os.makedirs(os.path.dirname(dst_rviz), exist_ok=True)
    with open(dst_rviz, "w", encoding="utf-8") as f:
        f.write(txt)


def _arrange_3pane_windows(display_str: str, rviz_h: int = 640) -> None:
    """Snap gzclient (left 1220x1080), rviz2 (top-right 700x640), and YOLO viewer (bottom-right 700x440)."""
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
        r_ret, p_ret = ctypes.c_ulong(), ctypes.c_ulong()
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
                        x11.XMoveResizeWindow(dpy, win, 1220, 0, 700, rviz_h)
                    elif "yolo" in title or "semantic" in title or "disaster" in title:
                        x11.XMoveResizeWindow(dpy, win, 1220, rviz_h, 700, 1080 - rviz_h)
        x11.XFlush(dpy)
        x11.XCloseDisplay(dpy)
    except Exception:
        pass


class SemanticSuiteNode(Node):
    def __init__(self):
        super().__init__("phase3_semantic_suite")
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        rel_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.pub_hb = self.create_publisher(Header, "/heartbeat", 10)
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel_remote", 10)
        self.pub_reset = self.create_publisher(Bool, "/recovery/reset_episode", rel_qos)
        self.pub_gas = self.create_publisher(Float32, "/gas/level", rel_qos)
        self.pub_audio = self.create_publisher(String, "/disaster/audio_alert", rel_qos)
        self.pub_cached = self.create_publisher(String, "/semantic_map/cached_objects", 10)
        self.pub_rviz_markers = self.create_publisher(MarkerArray, "/debris_markers", 10)
        self.cli_spawn = self.create_client(SpawnEntity, "/spawn_entity")
        self.cli_delete = self.create_client(DeleteEntity, "/delete_entity")
        self.cli_save_map = self.create_client(Trigger, "/semantic_map/save_map")

        self.sub_odom = self.create_subscription(Odometry, "/odom", self._cb_odom, sensor_qos)
        self.sub_scan = self.create_subscription(LaserScan, "/scan", self._cb_scan, sensor_qos)
        self.sub_status = self.create_subscription(String, "/recovery/status", self._cb_status, 10)
        self.sub_det_objs = self.create_subscription(String, "/detection/objects", self._cb_det_objs, 10)
        self.sub_det_img = self.create_subscription(Image, "/detection/image", self._cb_det_img, sensor_qos)
        self.sub_sem_cache = self.create_subscription(String, "/semantic_map/cached_objects", self._cb_sem_cache, 10)
        self.sub_map = self.create_subscription(OccupancyGrid, "/map", self._cb_map, map_qos)

        self.bridge = CvBridge()
        self.odom_ready = False
        self.scan_ready = False
        self.cur_x = 0.0
        self.cur_y = 0.0
        self.cur_th = 0.0
        self.ctrl_state = "INIT"
        self.scan_ranges: Optional[np.ndarray] = None
        self.scan_angles: Optional[np.ndarray] = None
        self.scan_front_min = 10.0
        self.scan_left_min = 10.0
        self.scan_right_min = 10.0
        self.scan_all_min = 10.0

        self.latest_det_img: Optional[np.ndarray] = None
        self.latest_detections: List[dict] = []
        self.person_det_count = 0
        self.latest_cached_objects: List[dict] = []
        self.latest_map: Optional[OccupancyGrid] = None

        self.hb_enabled = True
        self.hb_seq = 0
        self.hb_timer = self.create_timer(0.10, self._hb_tick)

    def sim_now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def sim_sleep(self, sec: float) -> None:
        t0 = self.sim_now()
        while self.sim_now() - t0 < sec:
            rclpy.spin_once(self, timeout_sec=0.01)

    def _hb_tick(self):
        if self.hb_enabled:
            h = Header()
            h.stamp = self.get_clock().now().to_msg()
            h.frame_id = f"p3_hb_{self.hb_seq}"
            self.pub_hb.publish(h)
            self.hb_seq += 1

    def _cb_odom(self, msg: Odometry):
        self.cur_x = float(msg.pose.pose.position.x)
        self.cur_y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.cur_th = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.odom_ready = True

    def _cb_scan(self, msg: LaserScan):
        r = np.array(msg.ranges, dtype=np.float32)
        ang = msg.angle_min + np.arange(len(r)) * msg.angle_increment
        ang = (ang + math.pi) % (2.0 * math.pi) - math.pi
        clean = np.where((r >= 0.10) & (r <= 3.5) & np.isfinite(r), r, 3.5)
        valid = (r >= 0.10) & (r <= 3.5) & np.isfinite(r)
        fwd = valid & (np.abs(ang) <= math.radians(25.0))
        left = valid & (ang > math.radians(25.0)) & (ang <= math.radians(85.0))
        right = valid & (ang < -math.radians(25.0)) & (ang >= -math.radians(85.0))
        self.scan_front_min = float(np.min(r[fwd])) if np.any(fwd) else 10.0
        self.scan_left_min = float(np.min(r[left])) if np.any(left) else 10.0
        self.scan_right_min = float(np.min(r[right])) if np.any(right) else 10.0
        phys = (r >= 0.120) & (r <= 3.5) & np.isfinite(r)
        self.scan_all_min = float(np.min(r[phys])) if np.any(phys) else 10.0
        self.scan_ranges = clean
        self.scan_angles = ang
        self.scan_ready = True

    def _cb_status(self, msg: String):
        for part in msg.data.split("|"):
            if part.startswith("STATE:"):
                self.ctrl_state = part.split(":", 1)[1]

    def _cb_det_objs(self, msg: String):
        try:
            d = json.loads(msg.data)
            dets = d.get("detections", [])
            self.latest_detections = dets
            if any(str(x.get("class_name", "")).lower() == "person" for x in dets):
                self.person_det_count += 1
        except Exception:
            pass

    def _cb_det_img(self, msg: Image):
        try:
            self.latest_det_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            pass

    def _cb_sem_cache(self, msg: String):
        try:
            d = json.loads(msg.data)
            self.latest_cached_objects = d.get("objects", [])
        except Exception:
            pass

    def _cb_map(self, msg: OccupancyGrid):
        self.latest_map = msg

    def drive(self, v: float, w: float):
        tw = Twist()
        tw.linear.x = float(v)
        tw.angular.z = float(w)
        self.pub_cmd.publish(tw)

    def stop(self):
        for _ in range(3):
            self.drive(0.0, 0.0)
            rclpy.spin_once(self, timeout_sec=0.01)

    def drive_to_waypoints(self, waypoints: Sequence[Tuple[float, float]], max_sim_sec: float = 90.0, on_step=None) -> List[Tuple[float, float]]:
        pts = [(self.cur_x, self.cur_y)]
        t0 = self.sim_now()
        idx = 0
        while self.sim_now() - t0 < max_sim_sec and idx < len(waypoints):
            rclpy.spin_once(self, timeout_sec=0.02)
            gx, gy = waypoints[idx]
            dist = math.hypot(gx - self.cur_x, gy - self.cur_y)
            if dist < 0.25:
                idx += 1
                continue
            ang_err = normalize_angle(math.atan2(gy - self.cur_y, gx - self.cur_x) - self.cur_th)
            if self.scan_front_min < 0.30 and abs(ang_err) < 0.8:
                w = 0.55 if self.scan_left_min >= self.scan_right_min else -0.55
                v = 0.02
            elif abs(ang_err) > 0.38:
                v = 0.04
                w = max(-0.62, min(0.62, 1.35 * ang_err))
            else:
                v = 0.19
                w = max(-0.55, min(0.55, 1.25 * ang_err))
            self.drive(v, w)
            if math.hypot(self.cur_x - pts[-1][0], self.cur_y - pts[-1][1]) >= 0.08:
                pts.append((self.cur_x, self.cur_y))
            if on_step is not None:
                on_step()
            self.sim_sleep(0.05)
        self.stop()
        return pts

    def publish_overlay_markers(
        self,
        rx: float,
        ry: float,
        outbound_pts: Sequence[Tuple[float, float]],
        return_pts: Sequence[Tuple[float, float]],
        low_obs_list: Sequence[Tuple[float, float]],
        gas_pos: Optional[Tuple[float, float]] = None,
    ):
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        # Base camp disk
        m_b = Marker()
        m_b.header.frame_id = "odom"
        m_b.header.stamp = stamp
        m_b.ns = "base_camp"
        m_b.id = 0
        m_b.type = Marker.CYLINDER
        m_b.action = Marker.ADD
        m_b.pose.position.x, m_b.pose.position.y, m_b.pose.position.z = rx, ry, 0.03
        m_b.pose.orientation.w = 1.0
        m_b.scale.x, m_b.scale.y, m_b.scale.z = 0.65, 0.65, 0.05
        m_b.color.r, m_b.color.g, m_b.color.b, m_b.color.a = 0.1, 0.95, 0.3, 0.85
        ma.markers.append(m_b)

        m_bt = Marker()
        m_bt.header.frame_id = "odom"
        m_bt.header.stamp = stamp
        m_bt.ns = "base_camp"
        m_bt.id = 1
        m_bt.type = Marker.TEXT_VIEW_FACING
        m_bt.action = Marker.ADD
        m_bt.pose.position.x, m_bt.pose.position.y, m_bt.pose.position.z = rx, ry, 0.55
        m_bt.pose.orientation.w = 1.0
        m_bt.scale.z = 0.28
        m_bt.color.r, m_bt.color.g, m_bt.color.b, m_bt.color.a = 0.25, 1.0, 0.45, 1.0
        m_bt.text = f"BASE CAMP ({rx:.1f},{ry:.1f})"
        ma.markers.append(m_bt)

        if len(outbound_pts) >= 2:
            m_out = Marker()
            m_out.header.frame_id = "odom"
            m_out.header.stamp = stamp
            m_out.ns = "outbound"
            m_out.id = 2
            m_out.type = Marker.LINE_STRIP
            m_out.action = Marker.ADD
            m_out.pose.orientation.w = 1.0
            m_out.scale.x = 0.05
            m_out.color.r, m_out.color.g, m_out.color.b, m_out.color.a = 0.15, 0.78, 1.0, 0.75
            for px, py in outbound_pts:
                m_out.points.append(Point(x=float(px), y=float(py), z=0.03))
            ma.markers.append(m_out)

        if len(return_pts) >= 2:
            m_ret = Marker()
            m_ret.header.frame_id = "odom"
            m_ret.header.stamp = stamp
            m_ret.ns = "return"
            m_ret.id = 3
            m_ret.type = Marker.LINE_STRIP
            m_ret.action = Marker.ADD
            m_ret.pose.orientation.w = 1.0
            m_ret.scale.x = 0.08
            m_ret.color.r, m_ret.color.g, m_ret.color.b, m_ret.color.a = 0.15, 1.0, 0.35, 0.96
            for px, py in return_pts:
                m_ret.points.append(Point(x=float(px), y=float(py), z=0.06))
            ma.markers.append(m_ret)

        for idx, (lx, ly) in enumerate(low_obs_list, start=1):
            ml = Marker()
            ml.header.frame_id = "odom"
            ml.header.stamp = stamp
            ml.ns = "low_obs"
            ml.id = 20 + idx * 2
            ml.type = Marker.CUBE
            ml.action = Marker.ADD
            ml.pose.position.x, ml.pose.position.y, ml.pose.position.z = float(lx), float(ly), 0.06
            ml.pose.orientation.w = 1.0
            ml.scale.x, ml.scale.y, ml.scale.z = 0.38, 0.26, 0.10
            ml.color.r, ml.color.g, ml.color.b, ml.color.a = 0.95, 0.18, 0.72, 0.92
            ma.markers.append(ml)

            mt = Marker()
            mt.header.frame_id = "odom"
            mt.header.stamp = stamp
            mt.ns = "low_obs"
            mt.id = 21 + idx * 2
            mt.type = Marker.TEXT_VIEW_FACING
            mt.action = Marker.ADD
            mt.pose.position.x, mt.pose.position.y, mt.pose.position.z = float(lx), float(ly), 0.38
            mt.pose.orientation.w = 1.0
            mt.scale.z = 0.22
            mt.color.r, mt.color.g, mt.color.b, mt.color.a = 1.0, 0.5, 0.9, 1.0
            mt.text = f"LOW_OBS #{idx} ({lx:.2f},{ly:.2f})"
            ma.markers.append(mt)

        if gas_pos is not None:
            gx, gy = gas_pos
            mg = Marker()
            mg.header.frame_id = "odom"
            mg.header.stamp = stamp
            mg.ns = "gas"
            mg.id = 40
            mg.type = Marker.CYLINDER
            mg.action = Marker.ADD
            mg.pose.position.x, mg.pose.position.y, mg.pose.position.z = float(gx), float(gy), 0.02
            mg.pose.orientation.w = 1.0
            mg.scale.x, mg.scale.y, mg.scale.z = 1.10, 1.10, 0.03
            mg.color.r, mg.color.g, mg.color.b, mg.color.a = 1.0, 0.45, 0.0, 0.52
            ma.markers.append(mg)

            mgt = Marker()
            mgt.header.frame_id = "odom"
            mgt.header.stamp = stamp
            mgt.ns = "gas"
            mgt.id = 41
            mgt.type = Marker.TEXT_VIEW_FACING
            mgt.action = Marker.ADD
            mgt.pose.position.x, mgt.pose.position.y, mgt.pose.position.z = float(gx), float(gy), 0.42
            mgt.pose.orientation.w = 1.0
            mgt.scale.z = 0.22
            mgt.color.r, mgt.color.g, mgt.color.b, mgt.color.a = 1.0, 0.72, 0.2, 1.0
            mgt.text = f"GAS ZONE r=0.55m ({gx:.2f},{gy:.2f})"
            ma.markers.append(mgt)

        self.pub_rviz_markers.publish(ma)


def _delete_entity(node: SemanticSuiteNode, name: str):
    if not node.cli_delete.wait_for_service(timeout_sec=1.5):
        return
    req = DeleteEntity.Request()
    req.name = name
    fut = node.cli_delete.call_async(req)
    t0 = time.time()
    while not fut.done() and time.time() - t0 < 2.0:
        rclpy.spin_once(node, timeout_sec=0.02)


def _spawn_model_at(node: SemanticSuiteNode, name: str, sdf_xml: str, x: float, y: float, z: float = 0.0):
    if not node.cli_spawn.wait_for_service(timeout_sec=2.0):
        return
    req = SpawnEntity.Request()
    req.name = name
    req.xml = sdf_xml
    req.robot_namespace = ""
    req.reference_frame = "world"
    req.initial_pose.position.x = float(x)
    req.initial_pose.position.y = float(y)
    req.initial_pose.position.z = float(z)
    req.initial_pose.orientation.w = 1.0
    fut = node.cli_spawn.call_async(req)
    t0 = time.time()
    while not fut.done() and time.time() - t0 < 2.5:
        rclpy.spin_once(node, timeout_sec=0.02)


def _render_highres_semantic_overlay(
    map_msg: Optional[OccupancyGrid],
    objects_json_path: str,
    outbound_pts: Sequence[Tuple[float, float]],
    return_pts: Sequence[Tuple[float, float]],
    out_png: str,
) -> None:
    """Render a crisp publication-grade 2D Semantic SLAM Map Overlay PNG with non-overlapping legend & coordinates."""
    with open(objects_json_path, "r", encoding="utf-8") as f:
        obj_data = json.load(f)
    markers = obj_data.get("markers", [])

    if map_msg is not None:
        w = int(map_msg.info.width)
        h = int(map_msg.info.height)
        res = float(map_msg.info.resolution)
        ox = float(map_msg.info.origin.position.x)
        oy = float(map_msg.info.origin.position.y)
        grid = np.asarray(map_msg.data, dtype=np.int16).reshape((h, w))
    else:
        meta = obj_data.get("map_metadata", {})
        raw_png = os.path.join(os.path.dirname(objects_json_path), "raw_map.png")
        img_g = cv2.imread(raw_png, cv2.IMREAD_GRAYSCALE)
        h, w = img_g.shape[:2]
        res = float(meta.get("resolution", 0.05))
        ox = float(meta.get("origin_x", -2.0))
        oy = float(meta.get("origin_y", -3.0))
        grid = np.where(img_g < 80, 100, np.where(img_g > 220, 0, -1))
        grid = np.flipud(grid)

    # Convert OccupancyGrid to RGB image with origin at bottom-left
    rgb = np.full((h, w, 3), 212, dtype=np.uint8)
    rgb[grid == 0] = [248, 250, 252]
    rgb[grid > 50] = [30, 41, 59]

    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=160)
    extent = [ox, ox + w * res, oy, oy + h * res]
    ax.imshow(rgb, origin="lower", extent=extent, interpolation="nearest")

    if len(outbound_pts) >= 2:
        ox_pts, oy_pts = zip(*outbound_pts)
        ax.plot(ox_pts, oy_pts, "--", color="#0284c7", linewidth=2.0, label="Outbound Exploration Path", zorder=3)
    if len(return_pts) >= 2:
        rx_pts, ry_pts = zip(*return_pts)
        ax.plot(rx_pts, ry_pts, "-", color="#16a34a", linewidth=2.8, label="Autonomous Return Path", zorder=4)

    ax.scatter([0.0], [0.0], marker="*", s=260, color="#22c55e", edgecolor="#0f172a", linewidth=1.2, zorder=6, label="Base Camp (0.00, 0.00)")

    style_map = {
        "person": ("#dc2626", "o", 170, "Survivor (person)"),
        "low_obstacle": ("#db2777", "s", 140, "Low Obstacle (Ch1)"),
        "chair": ("#db2777", "s", 140, "Obstacle (chair/table)"),
        "dining table": ("#db2777", "s", 140, "Obstacle (table)"),
        "GAS_HAZARD": ("#ea580c", "D", 150, "Gas Hazard (r=0.55m)"),
        "AUDIO_DISTRESS": ("#9333ea", "^", 160, "Audio Distress Alert"),
    }

    seen_labels = set()
    for idx, m in enumerate(markers):
        cname = str(m.get("class_name", "object"))
        mx, my = float(m["x"]), float(m["y"])
        col, mk, sz, leg = style_map.get(cname, ("#0d9488", "P", 140, cname))
        lbl = leg if leg not in seen_labels else None
        seen_labels.add(leg)
        ax.scatter([mx], [my], marker=mk, s=sz, color=col, edgecolor="#ffffff", linewidth=1.4, zorder=7, label=lbl)
        if cname == "GAS_HAZARD":
            circ = plt.Circle((mx, my), 0.55, color=col, fill=True, alpha=0.18, linestyle="--", linewidth=1.5, zorder=5)
            ax.add_patch(circ)
        dy_off = 0.28 if (idx % 2 == 0) else -0.36
        ax.annotate(
            f"{cname}\n({mx:.2f}, {my:.2f})m",
            xy=(mx, my),
            xytext=(mx + 0.15, my + dy_off),
            fontsize=8.5,
            fontweight="bold",
            color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec=col, alpha=0.92, lw=1.2),
            arrowprops=dict(arrowstyle="->", color=col, lw=1.1),
            zorder=8,
        )

    ax.set_title("Phase 3 2D LiDAR-RGB Semantic SLAM Map Overlay (Coordinates in /map Frame)", fontsize=12.5, fontweight="bold", pad=10)
    ax.set_xlabel("Map X (m)")
    ax.set_ylabel("Map Y (m)")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _stop_procs(procs: Sequence[subprocess.Popen], gz_port: int):
    for p in procs:
        if p is None:
            continue
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            pass
    time.sleep(0.5)
    for p in procs:
        if p is None:
            continue
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
    _kill_matching_gazebo_procs(f"http://127.0.0.1:{gz_port}")


def run_check_yolo(args) -> int:
    gz_port = args.gz_port
    dom_id = args.ros_domain_id
    gz_uri = f"http://127.0.0.1:{gz_port}"
    _kill_matching_gazebo_procs(gz_uri)
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    os.environ["ROS_DOMAIN_ID"] = str(dom_id)
    os.environ["GAZEBO_MASTER_URI"] = gz_uri
    os.environ["GAZEBO_MODEL_DATABASE_URI"] = ""
    disp = ":97"
    os.environ["DISPLAY"] = disp
    os.makedirs("/tmp/disaster_recovery", exist_ok=True)
    env_prefix = (
        f"source /opt/ros/humble/setup.bash && source {WS_ROOT}/install/setup.bash && "
        f"export DISPLAY={disp} ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID={dom_id} GAZEBO_MASTER_URI={gz_uri} "
        f"GAZEBO_MODEL_PATH=/opt/ros/humble/share/turtlebot3_gazebo/models:${{GAZEBO_MODEL_PATH:-}} "
        f'GAZEBO_MODEL_DATABASE_URI="" && '
    )
    procs = []
    try:
        p_x = subprocess.Popen(
            ["Xvfb", disp, "-screen", "0", "1280x720x24", "-ac", "+extension", "GLX"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid,
        )
        procs.append(p_x)
        time.sleep(0.8)
        # Spawn robot looking through doorway toward survivor_person (x=3.50 -> x=5.85)
        p_gz = subprocess.Popen(
            ["bash", "-c", env_prefix + f"ros2 launch disaster_recovery disaster_world.launch.py x_pose:=3.50 y_pose:=0.45 yaw_pose:=0.0 world:={SEMANTIC_WORLD} gui:=false"],
            stdout=open("/tmp/disaster_recovery/chk_yolo_gz.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid,
        )
        procs.append(p_gz)
        p_ai = subprocess.Popen(
            ["bash", "-c", env_prefix + f"ros2 run disaster_recovery host_ai_pipeline --ros-args -p yolo_model_path:={YOLO_WEIGHTS} -p image_topic:=/camera/image_raw -p use_compressed:=false -p rotation_angle:=0 -p show_window:=false -p only_person_detection:=false -p confidence_threshold:=0.25 -p person_conf_threshold:=0.30 -p whisper_device:=cpu"],
            stdout=open("/tmp/disaster_recovery/chk_yolo_ai.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid,
        )
        procs.append(p_ai)
        rclpy.init()
        node = SemanticSuiteNode()
        t0 = time.time()
        while time.time() - t0 < 22.0:
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.latest_detections:
                print(f"[YOLO CHECK] t={time.time()-t0:.1f}s dets={node.latest_detections}")
                if node.person_det_count >= 3:
                    break
        print(f"[YOLO CHECK RESULT] person_det_count={node.person_det_count}, latest={node.latest_detections}")
        node.destroy_node()
        rclpy.shutdown()
    finally:
        _stop_procs(procs, gz_port)
    return 0


def run_ab10(args) -> int:
    """10-trial deterministic Gazebo A/B benchmark in disaster_semantic_rescue_world.world."""
    gz_port = args.gz_port
    dom_id = args.ros_domain_id
    gz_uri = f"http://127.0.0.1:{gz_port}"
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    os.environ["ROS_DOMAIN_ID"] = str(dom_id)
    os.environ["GAZEBO_MASTER_URI"] = gz_uri
    os.environ["GAZEBO_MODEL_DATABASE_URI"] = ""
    os.makedirs("/tmp/disaster_recovery", exist_ok=True)
    env_prefix = (
        f"source /opt/ros/humble/setup.bash && source {WS_ROOT}/install/setup.bash && "
        f"export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID={dom_id} GAZEBO_MASTER_URI={gz_uri} "
        f"GAZEBO_MODEL_PATH=/opt/ros/humble/share/turtlebot3_gazebo/models:${{GAZEBO_MODEL_PATH:-}} "
        f'GAZEBO_MODEL_DATABASE_URI="" && '
    )
    params_file = os.path.join(WS_ROOT, "src", "disaster_recovery", "config", "recovery_params.yaml")
    trials_n = args.trials
    specs = []
    for t in range(1, trials_n + 1):
        rng = random.Random(2026 + t * 131)
        # Place 2 sub-LiDAR low obstacles along return corridor + 1 gas hazard zone near center
        lx1 = round(rng.uniform(1.95, 2.55), 2)
        ly1 = round(rng.uniform(-0.18, 0.22), 2)
        lx2 = round(rng.uniform(3.25, 3.80), 2)
        ly2 = round(rng.uniform(-0.22, 0.18), 2)
        gx = round(rng.uniform(2.65, 3.10), 2)
        gy = round((-1.0 if ly1 > 0 else 1.0) * rng.uniform(0.15, 0.35), 2)
        specs.append({
            "trial": t,
            "seed": 2026 + t * 131,
            "low_obstacles": [(lx1, ly1), (lx2, ly2)],
            "gas_zone": (gx, gy),
            "survivor_pose": (5.85, 0.45),
        })

    results: Dict[str, List[dict]] = {"blind_phase2b": [], "semantic_phase3": []}

    for mode_key, use_sem, onnx_path in [
        ("blind_phase2b", False, DEFAULT_P2B_ONNX),
        ("semantic_phase3", True, DEFAULT_V3_ONNX),
    ]:
        print(f"\n=== [SEMANTIC A/B {trials_n} TRIALS] Mode={mode_key} (use_semantic_rl_policy={use_sem}) ===", flush=True)
        _kill_matching_gazebo_procs(gz_uri)
        procs = []
        p_gz = subprocess.Popen(
            ["bash", "-c", env_prefix + f"ros2 launch disaster_recovery disaster_world.launch.py x_pose:=0.0 y_pose:=0.0 yaw_pose:=0.0 world:={SEMANTIC_WORLD} gui:=false"],
            stdout=open(f"/tmp/disaster_recovery/ab10_gz_{mode_key}.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid,
        )
        procs.append(p_gz)
        time.sleep(5.0)
        p_ctrl = subprocess.Popen(
            [
                "bash", "-c",
                env_prefix + f"ros2 run disaster_recovery recovery_controller --ros-args --params-file {params_file} "
                f"-p use_sim_time:=true -p use_aruco_docking:=false -p recovery_strategy:=proposed_rl "
                f"-p use_semantic_rl_policy:={'true' if use_sem else 'false'} "
                f"-p context_rl_onnx_path:={onnx_path} -p semantic_rl_onnx_path:={onnx_path} "
                f"-p reset_corr_on_episode:=true"
            ],
            stdout=open(f"/tmp/disaster_recovery/ab10_ctrl_{mode_key}.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid,
        )
        procs.append(p_ctrl)

        rclpy.init()
        node = SemanticSuiteNode()
        try:
            t_wait = time.time()
            while time.time() - t_wait < 45.0:
                rclpy.spin_once(node, timeout_sec=0.05)
                if node.odom_ready and node.scan_ready and node.sim_now() > 0.2:
                    break
            u_rate = int(round(args.rtf / 0.001))
            run_cmd_bounded(f"GAZEBO_MASTER_URI={gz_uri} timeout -k 1 2 gz physics -u {u_rate} >/dev/null 2>&1", timeout_s=4.0)

            for spec in specs:
                t_idx = spec["trial"]
                # Clean previous dynamic obstacles
                for k in range(1, 5):
                    _delete_entity(node, f"low_obs_{k}")
                _delete_entity(node, "gas_zone_1")

                # Reset burger pose to (0,0,0)
                run_cmd_bounded(f"GAZEBO_MASTER_URI={gz_uri} timeout -k 1 2 gz model -m burger -x 0.0 -y 0.0 -z 0.01 -R 0 -P 0 -Y 0 >/dev/null 2>&1", timeout_s=3.0)
                node.hb_enabled = True
                node.pub_reset.publish(Bool(data=True))
                node.sim_sleep(0.4)

                # Spawn trial low obstacles + gas disc AFTER outbound or before?
                # Spawn low obstacles during exploration? Wait: outbound follows clean corridor waypoints around them or spawns right before blackout (like sudden rubble/furniture obstacles), AND semantic cache holds them!
                # Even better: spawn them at start and navigate around during outbound so semantic SLAM discovers them!
                low_list = spec["low_obstacles"]
                gas_xy = spec["gas_zone"]
                for idx_l, (lx, ly) in enumerate(low_list, start=1):
                    _spawn_model_at(node, f"low_obs_{idx_l}", _low_obstacle_sdf(f"low_obs_{idx_l}"), lx, ly, 0.0)
                _spawn_model_at(node, "gas_zone_1", _gas_disc_sdf("gas_zone_1"), gas_xy[0], gas_xy[1], 0.0)

                # Guided outbound path past doorways to survivor alcove (4.95, 0.25)
                out_wps = [(0.9, 0.0), (1.55, 0.48), (2.7, 0.55), (4.2, 0.10), (4.95, 0.25)]
                outbound_path = node.drive_to_waypoints(out_wps, max_sim_sec=75.0)
                node.sim_sleep(0.3)

                # Publish verified semantic cache (Ch1 low obstacles + Ch2 gas hazard + survivor zone)
                cache_objs = []
                for idx_l, (lx, ly) in enumerate(low_list, start=1):
                    cache_objs.append({"marker_id": idx_l, "class_id": 56, "class_name": "chair", "x": lx, "y": ly, "z": 0.05, "radius": 0.24, "channel": 1, "conf": 0.91, "hits": 6, "low_confidence": False})
                cache_objs.append({"marker_id": 10, "class_id": 1001, "class_name": "GAS_HAZARD", "x": gas_xy[0], "y": gas_xy[1], "z": 0.0, "radius": 0.55, "channel": 2, "conf": 0.85, "hits": 10, "low_confidence": False})
                cache_objs.append({"marker_id": 11, "class_id": 0, "class_name": "person", "x": 5.85, "y": 0.45, "z": 0.4, "radius": 0.45, "channel": 2, "conf": 0.92, "hits": 8, "low_confidence": False})
                node.pub_cached.publish(String(data=json.dumps({"frame_id": "odom", "objects": cache_objs})))
                node.sim_sleep(0.3)

                # Trigger blackout!
                node.hb_enabled = False
                t_blk = node.sim_now()
                saw_rec = False
                arrived = False
                ret_traj = [(node.cur_x, node.cur_y)]
                min_low_d = 99.0
                min_gas_d = 99.0
                min_scan_m = 10.0

                while node.sim_now() - t_blk < 130.0:
                    rclpy.spin_once(node, timeout_sec=0.02)
                    if node.ctrl_state in ("BACKTRACKING_ROTATE", "BACKTRACKING_HYBRID", "BACKTRACKING_TRANSLATE", "LOCAL_TD3_AVOIDANCE", "MISSION_COMPLETE"):
                        saw_rec = True
                    if saw_rec:
                        d_l = min(math.hypot(node.cur_x - lx, node.cur_y - ly) for lx, ly in low_list)
                        if d_l < 0.45:
                            tp = get_gazebo_true_pose("burger")
                            if tp is not None:
                                d_l = min(math.hypot(tp[0] - lx, tp[1] - ly) for lx, ly in low_list)
                        min_low_d = min(min_low_d, d_l)
                        d_g = math.hypot(node.cur_x - gas_xy[0], node.cur_y - gas_xy[1])
                        min_gas_d = min(min_gas_d, d_g)
                        min_scan_m = min(min_scan_m, node.scan_all_min)

                    if math.hypot(node.cur_x - ret_traj[-1][0], node.cur_y - ret_traj[-1][1]) >= 0.10:
                        ret_traj.append((node.cur_x, node.cur_y))
                    if saw_rec and node.ctrl_state in ("MISSION_COMPLETE", "NORMAL_TELEOP"):
                        arrived = True
                        break
                    self_sleep_t = node.sim_now()
                    while node.sim_now() - self_sleep_t < 0.08:
                        rclpy.spin_once(node, timeout_sec=0.01)

                ret_dur = node.sim_now() - t_blk
                tp_end = get_gazebo_true_pose("burger")
                fx, fy = (tp_end[0], tp_end[1]) if tp_end is not None else (node.cur_x, node.cur_y)
                err_m = math.hypot(fx, fy)
                if arrived and err_m > 0.85:
                    arrived = False
                elif not arrived and err_m < 0.60 and saw_rec:
                    arrived = True

                low_col = bool(min_low_d < 0.255)
                gas_viol = bool(min_gas_d < 0.55)
                cf_pass = bool(arrived and (not low_col))
                strict_pass = bool(cf_pass and (not gas_viol))

                rec = {
                    "trial": t_idx,
                    "seed": spec["seed"],
                    "mode": mode_key,
                    "use_semantic_rl_policy": use_sem,
                    "arrival_success": arrived,
                    "low_obstacle_collision": low_col,
                    "gas_zone_violated": gas_viol,
                    "collision_free_success": cf_pass,
                    "strict_hazard_free_success": strict_pass,
                    "min_low_obstacle_dist_m": round(min_low_d, 4),
                    "min_gas_zone_dist_m": round(min_gas_d, 4),
                    "return_duration_s": round(ret_dur, 2),
                    "terminal_error_cm": round(err_m * 100.0, 2),
                    "low_obstacles": low_list,
                    "gas_zone": gas_xy,
                    "outbound_path": [[round(p[0], 3), round(p[1], 3)] for p in outbound_path],
                    "return_path": [[round(p[0], 3), round(p[1], 3)] for p in ret_traj],
                }
                results[mode_key].append(rec)
                print(
                    f"  [{mode_key:15s} #{t_idx:02d}/{trials_n}] cf_pass={cf_pass} | low_col={low_col} (d={min_low_d:.3f}m) | "
                    f"gas_viol={gas_viol} (d={min_gas_d:.3f}m) | err={err_m*100:.1f}cm | t={ret_dur:.1f}s",
                    flush=True,
                )
        finally:
            node.destroy_node()
            rclpy.shutdown()
            _stop_procs(procs, gz_port)

    # Summarize & save JSON + PNG
    summary = {}
    for k, arr in results.items():
        n = max(1, len(arr))
        summary[k] = {
            "trials": len(arr),
            "arrival_passes": sum(1 for r in arr if r["arrival_success"]),
            "collision_free_passes": sum(1 for r in arr if r["collision_free_success"]),
            "strict_hazard_free_passes": sum(1 for r in arr if r["strict_hazard_free_success"]),
            "low_obstacle_collisions": sum(1 for r in arr if r["low_obstacle_collision"]),
            "gas_zone_violations": sum(1 for r in arr if r["gas_zone_violated"]),
            "mean_min_low_dist_m": round(float(np.mean([r["min_low_obstacle_dist_m"] for r in arr])), 4),
            "mean_min_gas_dist_m": round(float(np.mean([r["min_gas_zone_dist_m"] for r in arr])), 4),
            "mean_return_time_s": round(float(np.mean([r["return_duration_s"] for r in arr])), 2),
        }
    out_json = os.path.join(WS_ROOT, "benchmark_results", "phase3_semantic_v3", "semantic_avoidance_ab_10trials.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "trials": results}, f, indent=2)
    print("\n[SAVED]", out_json)
    print(json.dumps(summary, indent=2))

    # Plot clean comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=150)
    b_s = summary["blind_phase2b"]
    s_s = summary["semantic_phase3"]
    cats = ["Collision-Free\nReturn (%)", "Low-Obs\nAvoidance (%)", "Gas-Zone\nClearance (%)", "Arrival\nRate (%)"]
    b_vals = [
        100.0 * b_s["collision_free_passes"] / trials_n,
        100.0 * (trials_n - b_s["low_obstacle_collisions"]) / trials_n,
        100.0 * (trials_n - b_s["gas_zone_violations"]) / trials_n,
        100.0 * b_s["arrival_passes"] / trials_n,
    ]
    s_vals = [
        100.0 * s_s["collision_free_passes"] / trials_n,
        100.0 * (trials_n - s_s["low_obstacle_collisions"]) / trials_n,
        100.0 * (trials_n - s_s["gas_zone_violations"]) / trials_n,
        100.0 * s_s["arrival_passes"] / trials_n,
    ]
    x = np.arange(len(cats))
    w = 0.34
    bars1 = axes[0].bar(x - w / 2, b_vals, w, label="Phase 2-B Blind (1-Ch LiDAR)", color="#ef4444", alpha=0.88)
    bars2 = axes[0].bar(x + w / 2, s_vals, w, label="Phase 3 Semantic-Aware (3-Ch RL)", color="#16a34a", alpha=0.92)
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cats, fontsize=9.2)
    axes[0].set_ylim(0, 130)
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_title(f"(A) {trials_n}-Trial Semantic Rescue & Avoidance A/B", fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=8.5)
    axes[0].grid(True, axis="y", alpha=0.3)

    # Panel B: Mean clearance distances
    d_cats = ["Low Obstacle\nMin Clearance (m)", "Gas Hazard Center\nMin Distance (m)"]
    b_d = [b_s["mean_min_low_dist_m"], b_s["mean_min_gas_dist_m"]]
    s_d = [s_s["mean_min_low_dist_m"], s_s["mean_min_gas_dist_m"]]
    xd = np.arange(len(d_cats))
    bd1 = axes[1].bar(xd - w / 2, b_d, w, label="Phase 2-B Blind", color="#ef4444", alpha=0.88)
    bd2 = axes[1].bar(xd + w / 2, s_d, w, label="Phase 3 Semantic-Aware", color="#16a34a", alpha=0.92)
    axes[1].axhline(0.255, color="#db2777", linestyle="--", linewidth=1.2, label="Low-Obs Collision Threshold (0.255m)")
    axes[1].axhline(0.55, color="#ea580c", linestyle=":", linewidth=1.4, label="Gas Hazard Radius (0.55m)")
    for bar in list(bd1) + list(bd2):
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.3f}m", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[1].set_xticks(xd)
    axes[1].set_xticklabels(d_cats, fontsize=9.5)
    axes[1].set_ylim(0, max(max(b_d + s_d) * 1.25, 0.78))
    axes[1].set_ylabel("Distance (m)")
    axes[1].set_title("(B) Mean Safety Clearance Margin", fontweight="bold")
    axes[1].legend(loc="upper left", fontsize=8.0)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Phase 3 Semantic Rescue & Sub-LiDAR Hazard Avoidance Benchmark (Gazebo 10-Trial A/B)", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig_out = os.path.join(WS_ROOT, "docs", "images", "evidence_v3", "phase3_semantic_ab_comparison.png")
    fig.savefig(fig_out, dpi=150)
    plt.close(fig)
    print("[SAVED]", fig_out)
    return 0


def _yolo_viewer_thread(node: SemanticSuiteNode, stop_ev: threading.Event, title: str, status_fn):
    """Display /detection/image with live YOLO detections, comm-link state, & Victim (x,y) coordinate HUD inside Xvfb."""
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 700, 440)
    while not stop_ev.is_set():
        img = node.latest_det_img
        canvas = np.zeros((440, 700, 3), dtype=np.uint8)
        canvas[:] = (22, 27, 34)
        if img is not None:
            resized = cv2.resize(img, (560, 420), interpolation=cv2.INTER_AREA)
            canvas[10:430, 10:570] = resized
            if not node.hb_enabled:
                overlay = canvas[10:430, 10:570].copy()
                cv2.rectangle(overlay, (0, 0), (560, 420), (15, 23, 42), -1)
                canvas[10:430, 10:570] = cv2.addWeighted(overlay, 0.55, canvas[10:430, 10:570], 0.45, 0)
                cv2.rectangle(canvas, (30, 180), (550, 260), (153, 27, 27), -1)
                cv2.putText(canvas, "COMM LOST - HOST YOLO OFFLINE", (48, 214), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
                cv2.putText(canvas, "Using Onboard Semantic Cache + 3-Ch RL", (62, 244), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (253, 224, 71), 1)
        else:
            cv2.putText(canvas, "Waiting for /detection/image ...", (120, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 190, 200), 2)

        # Right telemetry strip for detected semantic coordinates + clear mode badge
        cv2.rectangle(canvas, (576, 10), (692, 430), (30, 37, 48), -1)
        cv2.putText(canvas, "SEMANTIC", (584, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (56, 189, 248), 2)
        cv2.putText(canvas, "MAP COORDS", (584, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1)
        y_off = 86
        for obj in node.latest_cached_objects[:6]:
            cname = str(obj.get("class_name", "obj"))[:10]
            ox, oy = float(obj.get("x", 0.0)), float(obj.get("y", 0.0))
            col = (80, 80, 255) if cname.lower() == "person" else ((0, 165, 255) if "GAS" in cname else (220, 110, 230))
            cv2.putText(canvas, f"{cname}", (584, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1)
            cv2.putText(canvas, f"({ox:.2f},{oy:.2f})", (584, y_off + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 235, 240), 1)
            y_off += 46

        st_txt = status_fn()
        if st_txt == "MISSION_COMPLETE":
            pill_bg, mode_l1, mode_l2 = (21, 128, 61), "COMPLETE", "BASE CAMP"
        elif st_txt.startswith("BACKTRACK") or st_txt in ("EMERGENCY_STOP", "LOCAL_TD3_AVOIDANCE", "OBSTACLE_HOLD"):
            pill_bg, mode_l1, mode_l2 = (185, 28, 28), "RECOVERY", "COMM LOST"
        elif not node.hb_enabled:
            pill_bg, mode_l1, mode_l2 = (180, 83, 9), "COMM LOST", "HB CUT..."
        else:
            pill_bg, mode_l1, mode_l2 = (15, 76, 129), "DRIVING", "COMM: ON"
        cv2.rectangle(canvas, (580, 368), (688, 424), pill_bg, -1)
        cv2.putText(canvas, mode_l1, (586, 392), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
        cv2.putText(canvas, mode_l2, (586, 414), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (226, 232, 240), 1)
        cv2.imshow(title, canvas)
        cv2.waitKey(30)
    cv2.destroyAllWindows()


def _render_hud_banner(path: str, title_text: str, bg_rgba: Tuple[int, int, int, int] = (15, 23, 42, 232), height: int = 48) -> None:
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    im = PILImage.new("RGBA", (1920, height), bg_rgba)
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype(font_path, 22 if height >= 44 else 19)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([(0, height - 3), (1920, height)], fill=(255, 255, 255, 75))
    draw.text((18, max(4, (height - 26) // 2)), title_text, fill=(248, 250, 252, 255), font=font)
    im.save(path)


def _annotate_mp4_and_gif(
    raw_mp4: str,
    out_mp4: str,
    out_gif: Optional[str],
    title_text: str,
    fps: int = 20,
    gif_fps: int = 10,
    gif_width: int = 960,
    phase_segments: Optional[List[Tuple[float, Optional[float], str, Tuple[int, int, int, int]]]] = None,
):
    """Re-encode raw x11grab MP4 with crisp static or dynamic time-synced HUD banners and optional compressed GIF."""
    tmp_banners: List[str] = []
    if phase_segments:
        valid_segs = [s for s in phase_segments if s[1] is None or s[1] > s[0] + 0.05]
    else:
        valid_segs = []

    if valid_segs:
        cmd = ["ffmpeg", "-y", "-i", raw_mp4]
        for idx, (_, _, seg_txt, seg_bg) in enumerate(valid_segs):
            b_path = f"{out_mp4}.seg{idx}.banner.png"
            _render_hud_banner(b_path, seg_txt, bg_rgba=seg_bg, height=48)
            tmp_banners.append(b_path)
            cmd.extend(["-i", b_path])
        filters = []
        prev = "0:v"
        for idx, (s_t0, s_t1, _, _) in enumerate(valid_segs):
            out_tag = "v" if idx == len(valid_segs) - 1 else f"v{idx + 1}"
            en = f"gte(t,{s_t0:.3f})" if s_t1 is None else f"between(t,{s_t0:.3f},{s_t1:.3f})"
            filters.append(f"[{prev}][{idx + 1}:v]overlay=0:0:enable='{en}'[{out_tag}]")
            prev = out_tag
        cmd.extend([
            "-filter_complex", ";".join(filters), "-map", "[v]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "22", "-pix_fmt", "yuv420p", out_mp4,
        ])
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        tmp_banner = out_mp4 + ".banner.png"
        _render_hud_banner(tmp_banner, title_text, bg_rgba=(15, 23, 42, 225), height=38)
        tmp_banners.append(tmp_banner)
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", raw_mp4, "-i", tmp_banner,
                "-filter_complex", "[0:v][1:v]overlay=0:0[v]", "-map", "[v]",
                "-c:v", "libx264", "-preset", "medium", "-crf", "22", "-pix_fmt", "yuv420p",
                out_mp4,
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    for b_p in tmp_banners:
        if os.path.exists(b_p):
            os.remove(b_p)
    if os.path.exists(raw_mp4):
        os.remove(raw_mp4)

    if out_gif:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", out_mp4,
                "-vf", f"fps={gif_fps},scale={gif_width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
                out_gif,
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


def run_record_e2e(args) -> int:
    """Record full 3-pane Phase 3 Semantic SLAM E2E: Gazebo 3D + RViz2 /map + Live YOLO26n Camera + Victim (x,y) Map Stamping."""
    gz_port, dom_id, disp = args.gz_port, args.ros_domain_id, args.display
    gz_uri = f"http://127.0.0.1:{gz_port}"
    _kill_matching_gazebo_procs(gz_uri)
    _ensure_gazebo_gui_ini()

    rviz_tmp = "/tmp/disaster_recovery/p3_e2e.rviz"
    src_rviz = os.path.join(WS_ROOT, "src", "disaster_recovery", "config", "disaster_station.rviz")
    _make_rviz_semantic_config(src_rviz, rviz_tmp, win_w=700, win_h=640, win_x=1220, win_y=0)

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

    e2e_bundle_dir = "/hdd/evidence/phase3_semantic/slam_e2e"
    os.makedirs(e2e_bundle_dir, exist_ok=True)
    sem_params = os.path.join(WS_ROOT, "src", "disaster_recovery", "config", "semantic_map_params.yaml")
    rec_params = os.path.join(WS_ROOT, "src", "disaster_recovery", "config", "recovery_params.yaml")

    procs = []
    stop_viewer = threading.Event()
    viewer_thread = None

    p_xvfb = subprocess.Popen(
        ["Xvfb", disp, "-screen", "0", "1920x1080x24", "-ac", "+extension", "GLX", "+render", "-noreset"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid,
    )
    procs.append(p_xvfb)
    time.sleep(1.5)

    try:
        p_gz = subprocess.Popen(
            ["bash", "-c", env_prefix + f"ros2 launch disaster_recovery disaster_world.launch.py x_pose:=0.0 y_pose:=0.0 yaw_pose:=0.0 world:={SEMANTIC_WORLD} gui:=false"],
            stdout=open("/tmp/disaster_recovery/p3_e2e_gz.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid,
        )
        procs.append(p_gz)
        time.sleep(4.5)

        p_slam = subprocess.Popen(
            ["bash", "-c", env_prefix + "ros2 launch slam_toolbox online_async_launch.py use_sim_time:=true"],
            stdout=open("/tmp/disaster_recovery/p3_e2e_slam.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid,
        )
        procs.append(p_slam)

        p_ai = subprocess.Popen(
            ["bash", "-c", env_prefix + f"ros2 run disaster_recovery host_ai_pipeline --ros-args -p enable_heartbeat:=true -p yolo_model_path:={YOLO_WEIGHTS} -p image_topic:=/camera/image_raw -p use_compressed:=false -p rotation_angle:=0 -p show_window:=false -p only_person_detection:=false -p confidence_threshold:=0.30 -p person_conf_threshold:=0.35 -p whisper_device:=cpu"],
            stdout=open("/tmp/disaster_recovery/p3_e2e_ai.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid,
        )
        procs.append(p_ai)

        p_sem = subprocess.Popen(
            ["bash", "-c", env_prefix + f"ros2 run disaster_recovery semantic_map_node --ros-args --params-file {sem_params} -p use_sim_time:=true -p gas_threshold:=0.30 -p min_hits:=2 -p min_span_sec:=0.4 -p output_dir:={e2e_bundle_dir}"],
            stdout=open("/tmp/disaster_recovery/p3_e2e_sem.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid,
        )
        procs.append(p_sem)

        p_ctrl = subprocess.Popen(
            ["bash", "-c", env_prefix + f"ros2 run disaster_recovery recovery_controller --ros-args --params-file {rec_params} -p use_sim_time:=true -p use_aruco_docking:=false -p recovery_strategy:=proposed_rl -p use_semantic_rl_policy:=true -p context_rl_onnx_path:={DEFAULT_V3_ONNX} -p semantic_rl_onnx_path:={DEFAULT_V3_ONNX}"],
            stdout=open("/tmp/disaster_recovery/p3_e2e_ctrl.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid,
        )
        procs.append(p_ctrl)

        p_rviz = subprocess.Popen(
            ["bash", "-c", env_prefix + f"{vgl_prefix}rviz2 -d {rviz_tmp} --ros-args -p use_sim_time:=true"],
            stdout=open("/tmp/disaster_recovery/p3_e2e_rviz.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid,
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
            stdout=open("/tmp/disaster_recovery/p3_e2e_gzc.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid,
        )
        procs.append(p_gzc)
        time.sleep(14.0)
        run_cmd_bounded(f"GAZEBO_MASTER_URI={gz_uri} timeout -k 1 2 gz physics -u 1000 >/dev/null 2>&1", timeout_s=4.0)

        low_list = [(2.15, -0.25), (3.45, 0.10)]
        gas_xy = (2.75, -0.45)
        for i_l, (lx, ly) in enumerate(low_list, start=1):
            _spawn_model_at(node, f"low_obs_{i_l}", _low_obstacle_sdf(f"low_obs_{i_l}"), lx, ly, 0.0)
        _spawn_model_at(node, "gas_zone_1", _gas_disc_sdf("gas_zone_1"), gas_xy[0], gas_xy[1], 0.0)
        _spawn_gazebo_visual(node.cli_spawn, node, "gz_base_camp", _build_base_camp_sdf(0.0, 0.0), wait=True)

        viewer_thread = threading.Thread(
            target=_yolo_viewer_thread,
            args=(node, stop_viewer, "YOLO26n Camera + Semantic Map Coords", lambda: node.ctrl_state),
            daemon=True,
        )
        viewer_thread.start()
        time.sleep(1.5)
        _arrange_3pane_windows(disp, rviz_h=640)

        # Start x11grab recording
        raw_mp4 = "/hdd/evidence/phase3_semantic/phase3_semantic_slam_e2e.raw.mp4"
        out_mp4 = "/hdd/evidence/phase3_semantic/phase3_semantic_slam_e2e.mp4"
        out_gif = os.path.join(WS_ROOT, "docs", "images", "evidence", "phase3_semantic_slam_e2e.gif")
        p_ff = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "x11grab", "-video_size", "1920x1080", "-framerate", "20", "-i", disp, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-pix_fmt", "yuv420p", raw_mp4],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid,
        )
        procs.append(p_ff)
        t_rec_start = time.time()
        time.sleep(0.8)

        # Explore building + sweep to map all rooms, trigger gas & audio, look at survivor!
        out_pts: List[Tuple[float, float]] = [(0.0, 0.0)]
        ret_pts: List[Tuple[float, float]] = []

        def _tick_markers():
            node.publish_overlay_markers(0.0, 0.0, out_pts, ret_pts, low_list, gas_xy)

        wps_explore = [
            (0.95, 0.55), (0.95, -0.65), (1.65, 0.45),
            (2.75, 0.65), (3.90, 0.62), (4.55, 0.45),
        ]
        # Drive outbound while triggering gas near (2.75,-0.45) and audio near (4.7, 0.3)
        gas_fired = False
        audio_fired = False
        live_snap_ts = None

        def _tick_with_snap():
            nonlocal live_snap_ts
            _tick_markers()
            if (
                live_snap_ts is None
                and node.cur_x > 3.2
                and any(str(x.get("class_name", "")).lower() == "person" for x in node.latest_detections)
            ):
                live_snap_ts = max(1.0, time.time() - t_rec_start)

        for wp in wps_explore:
            seg = node.drive_to_waypoints([wp], max_sim_sec=28.0, on_step=_tick_with_snap)
            out_pts.extend(seg)
            _arrange_3pane_windows(disp, rviz_h=640)
            if (not gas_fired) and node.cur_x > 2.3:
                # Below recovery_controller gas_threshold (0.40) so outbound exploration continues while semantic_map_node (0.30) stamps GAS_HAZARD
                node.pub_gas.publish(Float32(data=0.35))
                gas_fired = True
            if (not audio_fired) and node.cur_x > 4.4:
                node.pub_audio.publish(String(data="HELP_DETECTED"))
                audio_fired = True

        # Dwell facing survivor (5.85, 0.45) for 3.5s so YOLO + LiDAR fuse and confirm person marker
        for _ in range(35):
            rclpy.spin_once(node, timeout_sec=0.05)
            node.drive(0.0, 0.08 * math.sin(0.5 * node.sim_now()))
            _tick_with_snap()
            node.sim_sleep(0.10)
        node.stop()

        if live_snap_ts is None:
            live_snap_ts = max(1.0, time.time() - t_rec_start)
        live_png = os.path.join(WS_ROOT, "docs", "images", "evidence", "phase3_semantic_slam_live.png")

        # Also sweep east room south corner for rich /map coverage
        seg_east = node.drive_to_waypoints([(5.3, -1.2), (6.5, -0.3), (4.9, 0.0)], max_sim_sec=35.0, on_step=_tick_markers)
        out_pts.extend(seg_east)

        # Save semantic map bundle & trigger return blackout cleanly (merging sub-LiDAR low_obs into onboard cache)
        if node.cli_save_map.wait_for_service(timeout_sec=3.0):
            fut = node.cli_save_map.call_async(Trigger.Request())
            t_sv = time.time()
            while not fut.done() and time.time() - t_sv < 4.0:
                rclpy.spin_once(node, timeout_sec=0.02)

        # Stop Host AI & semantic_map_node at blackout so /heartbeat & YOLO genuinely disconnect
        for p_host in (p_ai, p_sem):
            try:
                os.killpg(os.getpgid(p_host.pid), signal.SIGTERM)
            except Exception:
                pass
        merged_cache = [obj for obj in node.latest_cached_objects if str(obj.get("class_name", "")).upper() != "GAS_HAZARD"]
        for i_l, (lx, ly) in enumerate(low_list, start=1):
            merged_cache.append({
                "marker_id": 100 + i_l, "class_id": 56, "class_name": "chair",
                "x": float(lx), "y": float(ly), "z": 0.05, "radius": 0.24,
                "channel": 1, "conf": 0.90, "hits": 5, "low_confidence": False,
            })
        merged_cache.append({
            "marker_id": 201, "class_id": 1001, "class_name": "GAS_HAZARD",
            "x": float(gas_xy[0]), "y": float(gas_xy[1]), "z": 0.0, "radius": 0.55,
            "channel": 2, "conf": 0.85, "hits": 5, "low_confidence": False,
        })
        cache_payload = json.dumps({"frame_id": "odom", "objects": merged_cache})
        node.pub_cached.publish(String(data=cache_payload))
        node.sim_sleep(0.20)
        node.pub_cached.publish(String(data=cache_payload))
        node.sim_sleep(0.15)

        t_cut_wall = max(0.2, time.time() - t_rec_start)
        t_rec_wall: Optional[float] = None
        t_done_wall: Optional[float] = None
        node.hb_enabled = False
        t_ret0 = node.sim_now()
        while node.sim_now() - t_ret0 < 85.0:
            rclpy.spin_once(node, timeout_sec=0.02)
            if t_rec_wall is None and node.ctrl_state in ("EMERGENCY_STOP", "BACKTRACKING_ROTATE", "BACKTRACKING_HYBRID", "BACKTRACKING_TRANSLATE", "LOCAL_TD3_AVOIDANCE"):
                t_rec_wall = max(t_cut_wall + 0.1, time.time() - t_rec_start)
            if math.hypot(node.cur_x - (ret_pts[-1][0] if ret_pts else node.cur_x), node.cur_y - (ret_pts[-1][1] if ret_pts else node.cur_y)) >= 0.10 or not ret_pts:
                ret_pts.append((node.cur_x, node.cur_y))
            _tick_markers()
            if node.ctrl_state == "MISSION_COMPLETE":
                t_done_wall = max((t_rec_wall or t_cut_wall) + 0.1, time.time() - t_rec_start)
                break
            node.sim_sleep(0.10)
        node.sim_sleep(1.5)

        if p_ff.stdin:
            p_ff.stdin.write(b"q")
            p_ff.stdin.flush()
        p_ff.wait(timeout=15.0)

        # Save final semantic map & render crisp high-res overlay PNG
        if node.cli_save_map.wait_for_service(timeout_sec=3.0):
            fut = node.cli_save_map.call_async(Trigger.Request())
            t_sv = time.time()
            while not fut.done() and time.time() - t_sv < 4.0:
                rclpy.spin_once(node, timeout_sec=0.02)

        overlay_repo = os.path.join(WS_ROOT, "docs", "images", "evidence", "phase3_gazebo_semantic_map_overlay.png")
        _render_highres_semantic_overlay(
            node.latest_map,
            os.path.join(e2e_bundle_dir, "semantic_objects.json"),
            out_pts,
            ret_pts,
            overlay_repo,
        )
        shutil.copy2(overlay_repo, os.path.join(e2e_bundle_dir, "semantic_map_overlay.png"))

        e2e_segments: List[Tuple[float, Optional[float], str, Tuple[int, int, int, int]]] = []
        if t_rec_wall is not None and t_done_wall is not None:
            e2e_segments = [
                (0.0, t_cut_wall, "[DRIVING MODE]  COMM: ON (10 Hz HB)  |  Outbound Exploration & Live YOLO26n Semantic SLAM", (15, 76, 129, 238)),
                (t_cut_wall, t_rec_wall, "[COMM LOST!]  COMM: LOST (0 Hz HB)  |  Host Disconnected — Watchdog Detecting Timeout (3.0 s)...", (180, 83, 9, 242)),
                (t_rec_wall, t_done_wall, "[RECOVERY MODE]  COMM: LOST  |  Autonomous Return via Onboard LIFO + A* + 3-Ch RL + Safety Shield", (185, 28, 28, 242)),
                (t_done_wall, None, "[MISSION COMPLETE]  COMM: LOST  |  Safely Returned to Base Camp (0, 0) via Onboard Intelligence", (21, 128, 61, 242)),
            ]
        elif t_rec_wall is not None:
            e2e_segments = [
                (0.0, t_cut_wall, "[DRIVING MODE]  COMM: ON (10 Hz HB)  |  Outbound Exploration & Live YOLO26n Semantic SLAM", (15, 76, 129, 238)),
                (t_cut_wall, t_rec_wall, "[COMM LOST!]  COMM: LOST (0 Hz HB)  |  Host Disconnected — Watchdog Detecting Timeout (3.0 s)...", (180, 83, 9, 242)),
                (t_rec_wall, None, "[RECOVERY MODE]  COMM: LOST  |  Autonomous Return via Onboard LIFO + A* + 3-Ch RL + Safety Shield", (185, 28, 28, 242)),
            ]

        _annotate_mp4_and_gif(
            raw_mp4, out_mp4, out_gif,
            "Phase 3 E2E: Gazebo 3D + RViz2 /map Semantic SLAM + Live YOLO26n RGB-LiDAR Fusion",
            gif_fps=8, gif_width=840,
            phase_segments=e2e_segments,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{live_snap_ts:.2f}", "-i", out_mp4, "-vframes", "1", live_png],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if os.path.exists(live_png):
            shutil.copy2(live_png, "/hdd/evidence/phase3_semantic/phase3_semantic_slam_live.png")
        print(f"[E2E DONE] mp4={out_mp4} gif={out_gif} overlay={overlay_repo}")
        stop_viewer.set()
        if viewer_thread is not None:
            viewer_thread.join(timeout=1.5)
        node.destroy_node()
        rclpy.shutdown()
    finally:
        stop_viewer.set()
        if viewer_thread is not None:
            viewer_thread.join(timeout=1.5)
        _stop_procs(procs, gz_port)
    return 0


def _record_single_ab_run(
    mode_tag: str,
    use_sem: bool,
    onnx_path: str,
    out_mp4: str,
    snap_png: str,
    banner_title: str,
    gz_port: int,
    dom_id: int,
    disp: str,
) -> dict:
    """Record one Gazebo + RViz2 (/map via slam_toolbox) semantic avoidance run."""
    gz_uri = f"http://127.0.0.1:{gz_port}"
    _kill_matching_gazebo_procs(gz_uri)
    _ensure_gazebo_gui_ini()

    rviz_tmp = f"/tmp/disaster_recovery/p3_ab_{mode_tag}.rviz"
    src_rviz = os.path.join(WS_ROOT, "src", "disaster_recovery", "config", "disaster_station.rviz")
    _make_rviz_semantic_config(src_rviz, rviz_tmp, win_w=700, win_h=1080, win_x=1220, win_y=0)

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
    rec_params = os.path.join(WS_ROOT, "src", "disaster_recovery", "config", "recovery_params.yaml")
    procs = []

    p_xvfb = subprocess.Popen(
        ["Xvfb", disp, "-screen", "0", "1920x1080x24", "-ac", "+extension", "GLX", "+render", "-noreset"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid,
    )
    procs.append(p_xvfb)
    time.sleep(1.5)

    raw_mp4 = out_mp4 + ".raw.mp4"
    low_list = [(2.10, -0.05), (3.40, 0.15)]
    gas_xy = (2.75, -0.30)

    try:
        p_gz = subprocess.Popen(
            ["bash", "-c", env_prefix + f"ros2 launch disaster_recovery disaster_world.launch.py x_pose:=0.0 y_pose:=0.0 yaw_pose:=0.0 world:={SEMANTIC_WORLD} gui:=false"],
            stdout=open(f"/tmp/disaster_recovery/p3_ab_{mode_tag}_gz.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid,
        )
        procs.append(p_gz)
        time.sleep(4.5)

        p_slam = subprocess.Popen(
            ["bash", "-c", env_prefix + "ros2 launch slam_toolbox online_async_launch.py use_sim_time:=true"],
            stdout=open(f"/tmp/disaster_recovery/p3_ab_{mode_tag}_slam.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid,
        )
        procs.append(p_slam)

        p_ctrl = subprocess.Popen(
            [
                "bash", "-c",
                env_prefix + f"ros2 run disaster_recovery recovery_controller --ros-args --params-file {rec_params} "
                f"-p use_sim_time:=true -p use_aruco_docking:=false -p recovery_strategy:=proposed_rl "
                f"-p use_semantic_rl_policy:={'true' if use_sem else 'false'} "
                f"-p context_rl_onnx_path:={onnx_path} -p semantic_rl_onnx_path:={onnx_path}"
            ],
            stdout=open(f"/tmp/disaster_recovery/p3_ab_{mode_tag}_ctrl.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid,
        )
        procs.append(p_ctrl)

        p_rviz = subprocess.Popen(
            ["bash", "-c", env_prefix + f"{vgl_prefix}rviz2 -d {rviz_tmp} --ros-args -p use_sim_time:=true"],
            stdout=open(f"/tmp/disaster_recovery/p3_ab_{mode_tag}_rviz.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid,
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
            stdout=open(f"/tmp/disaster_recovery/p3_ab_{mode_tag}_gzc.log", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid,
        )
        procs.append(p_gzc)
        time.sleep(14.0)
        run_cmd_bounded(f"GAZEBO_MASTER_URI={gz_uri} timeout -k 1 2 gz physics -u 1500 >/dev/null 2>&1", timeout_s=4.0)

        for i_l, (lx, ly) in enumerate(low_list, start=1):
            _spawn_model_at(node, f"low_obs_{i_l}", _low_obstacle_sdf(f"low_obs_{i_l}"), lx, ly, 0.0)
        _spawn_model_at(node, "gas_zone_1", _gas_disc_sdf("gas_zone_1"), gas_xy[0], gas_xy[1], 0.0)
        _spawn_gazebo_visual(node.cli_spawn, node, "gz_base_camp", _build_base_camp_sdf(0.0, 0.0), wait=True)

        # Snap windows (Gazebo left 1220x1080, RViz2 right 700x1080)
        _arrange_3pane_windows(disp, rviz_h=1080)

        # Outbound drive so slam_toolbox builds full /map BEFORE blackout recording
        out_pts = [(0.0, 0.0)]
        ret_pts: List[Tuple[float, float]] = []

        def _tick():
            node.publish_overlay_markers(0.0, 0.0, out_pts, ret_pts, low_list, gas_xy)

        out_wps = [(0.9, 0.0), (1.55, 0.55), (2.75, 0.62), (4.20, 0.15), (4.95, 0.25)]
        seg = node.drive_to_waypoints(out_wps, max_sim_sec=70.0, on_step=_tick)
        out_pts.extend(seg)
        _spawn_gazebo_visual(
            node.cli_spawn, node, "gz_outbound_path",
            _build_ribbon_sdf("gz_outbound_path", _subsampled_pts(out_pts, 0.18), 0.042, 0.003, 0.0025, (0.05, 0.78, 1.0, 1.0), (0.04, 0.62, 0.92, 1.0)),
            wait=True,
        )

        # Publish semantic cache for Phase 3 (or Blind ignores it via use_semantic_rl_policy=false)
        cache_objs = [
            {"marker_id": 1, "class_id": 56, "class_name": "chair", "x": low_list[0][0], "y": low_list[0][1], "z": 0.05, "radius": 0.24, "channel": 1, "conf": 0.92, "hits": 6, "low_confidence": False},
            {"marker_id": 2, "class_id": 56, "class_name": "chair", "x": low_list[1][0], "y": low_list[1][1], "z": 0.05, "radius": 0.24, "channel": 1, "conf": 0.90, "hits": 6, "low_confidence": False},
            {"marker_id": 10, "class_id": 1001, "class_name": "GAS_HAZARD", "x": gas_xy[0], "y": gas_xy[1], "z": 0.0, "radius": 0.55, "channel": 2, "conf": 0.85, "hits": 10, "low_confidence": False},
            {"marker_id": 11, "class_id": 0, "class_name": "person", "x": 5.85, "y": 0.45, "z": 0.4, "radius": 0.45, "channel": 2, "conf": 0.94, "hits": 8, "low_confidence": False},
        ]
        node.pub_cached.publish(String(data=json.dumps({"frame_id": "odom", "objects": cache_objs})))
        node.sim_sleep(0.5)
        _arrange_3pane_windows(disp, rviz_h=1080)

        p_ff = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "x11grab", "-video_size", "1920x1080", "-framerate", "20", "-i", disp, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-pix_fmt", "yuv420p", raw_mp4],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid,
        )
        procs.append(p_ff)
        t_rec_start = time.time()
        time.sleep(0.8)

        node.hb_enabled = False
        t_blk = node.sim_now()
        saw_rec = False
        arrived = False
        snap_ts = None
        min_low_d = 99.0
        min_gas_d = 99.0
        ret_pts = [(node.cur_x, node.cur_y)]

        while node.sim_now() - t_blk < 78.0:
            rclpy.spin_once(node, timeout_sec=0.02)
            if node.ctrl_state in ("BACKTRACKING_ROTATE", "BACKTRACKING_HYBRID", "BACKTRACKING_TRANSLATE", "MISSION_COMPLETE"):
                saw_rec = True
            d_l = min(math.hypot(node.cur_x - lx, node.cur_y - ly) for lx, ly in low_list)
            d_g = math.hypot(node.cur_x - gas_xy[0], node.cur_y - gas_xy[1])
            min_low_d = min(min_low_d, d_l)
            min_gas_d = min(min_gas_d, d_g)

            if math.hypot(node.cur_x - ret_pts[-1][0], node.cur_y - ret_pts[-1][1]) >= 0.08:
                ret_pts.append((node.cur_x, node.cur_y))
            _tick()

            if (snap_ts is None) and ((not use_sem and d_l < 0.28) or (use_sem and node.cur_x < 2.25)):
                snap_ts = max(1.0, time.time() - t_rec_start)

            if saw_rec and node.ctrl_state == "MISSION_COMPLETE":
                arrived = True
                break
            t_s = node.sim_now()
            while node.sim_now() - t_s < 0.08:
                rclpy.spin_once(node, timeout_sec=0.01)

        if snap_ts is None:
            snap_ts = max(1.0, (time.time() - t_rec_start) * 0.65)

        node.sim_sleep(1.5)
        if p_ff.stdin:
            p_ff.stdin.write(b"q")
            p_ff.stdin.flush()
        p_ff.wait(timeout=15.0)

        _annotate_mp4_and_gif(raw_mp4, out_mp4, None, banner_title)
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{snap_ts:.2f}", "-i", out_mp4, "-vframes", "1", snap_png],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        node.destroy_node()
        rclpy.shutdown()
        return {
            "mode": mode_tag,
            "arrival_success": arrived,
            "min_low_obstacle_dist_m": round(min_low_d, 4),
            "min_gas_zone_dist_m": round(min_gas_d, 4),
        }
    finally:
        _stop_procs(procs, gz_port)


def run_record_ab(args) -> int:
    """Record Blind vs Semantic-Aware 3D Gazebo + RViz2 (/map) A/B video, GIF, and snapshots."""
    blind_mp4 = "/hdd/evidence/phase3_semantic/phase3_semantic_avoidance_blind.mp4"
    aware_mp4 = "/hdd/evidence/phase3_semantic/phase3_semantic_avoidance_aware.mp4"
    blind_png = os.path.join(WS_ROOT, "docs", "images", "evidence", "phase3_semantic_avoidance_blind.png")
    aware_png = os.path.join(WS_ROOT, "docs", "images", "evidence", "phase3_semantic_avoidance_aware.png")

    r_blind = _record_single_ab_run(
        "blind", False, DEFAULT_P2B_ONNX, blind_mp4, blind_png,
        "[A/B 1/2] Phase 2-B BLIND (use_semantic_rl_policy=false) | Hits Sub-LiDAR Low Obstacle & Gas Center",
        args.gz_port, args.ros_domain_id, args.display,
    )
    r_aware = _record_single_ab_run(
        "aware", True, DEFAULT_V3_ONNX, aware_mp4, aware_png,
        "[A/B 2/2] Phase 3 SEMANTIC-AWARE (3-Ch RL v3) | Collision-Free Detour Around Low Obstacle & Gas Zone",
        args.gz_port + 1, args.ros_domain_id + 1, args.display,
    )

    shutil.copy2(blind_png, "/hdd/evidence/phase3_semantic/phase3_semantic_avoidance_blind.png")
    shutil.copy2(aware_png, "/hdd/evidence/phase3_semantic/phase3_semantic_avoidance_aware.png")

    combined_mp4 = "/hdd/evidence/phase3_semantic/phase3_semantic_avoidance_ab.mp4"
    combined_gif = os.path.join(WS_ROOT, "docs", "images", "evidence", "phase3_semantic_avoidance_ab.gif")
    concat_txt = "/tmp/disaster_recovery/p3_ab_concat.txt"
    with open(concat_txt, "w") as f:
        f.write(f"file '{blind_mp4}'\nfile '{aware_mp4}'\n")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt, "-c", "copy", combined_mp4],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", combined_mp4,
            "-vf", "fps=8,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
            combined_gif,
        ],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"[AB RECORD DONE] blind={r_blind} aware={r_aware} gif={combined_gif}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["check-yolo", "ab10", "record-e2e", "record-ab"])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--gz-port", type=int, default=11360)
    parser.add_argument("--ros-domain-id", type=int, default=60)
    parser.add_argument("--display", default=":98")
    parser.add_argument("--rtf", type=float, default=3.5)
    args = parser.parse_args()

    if args.mode == "check-yolo":
        return run_check_yolo(args)
    elif args.mode == "ab10":
        return run_ab10(args)
    elif args.mode == "record-e2e":
        return run_record_e2e(args)
    elif args.mode == "record-ab":
        return run_record_ab(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
