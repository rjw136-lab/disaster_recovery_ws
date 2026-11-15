#!/usr/bin/env python3
"""
Capture angled 3D Gazebo overview renders for every unique disaster .world
and compose them into docs/images/scenario_worlds_15_3d.png.
"""

import os
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
WS_WORLDS = os.path.join(WS_ROOT, "src", "disaster_recovery", "worlds")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ab_disaster_benchmark import _kill_matching_gazebo_procs  # noqa: E402
from record_3d_evidence import _ensure_gazebo_gui_ini  # noqa: E402

# (key, title, world_filename, pose_xyzrpy, reset_xyyaw, cam_pose)
WORLD_CARDS: List[Tuple[str, str, str, Tuple[float, float, float], str]] = [
    ("collapsed", "#1·#2·#6·#7 Collapsed Building (20x20m)", "disaster_collapsed_building.world", (0.0, -4.0, 1.5708), "3.5 -9.5 14.5 0 0.92 1.5708"),
    ("tunnel", "#3·#8 Disaster Tunnel (SUV / Excavator)", "disaster_tunnel_world.world", (-0.5, -3.0, 1.5708), "-0.2 -8.2 10.5 0 0.72 1.5708"),
    ("house", "#4·#9 Indoor House (15x11m Multi-Room)", "disaster_house_world.world", (-2.0, -0.5, 0.0), "-0.3 -7.6 12.2 0 1.04 1.5708"),
    ("maze", "#5·#10 Dynamic Maze (5x5m + Moving Obs)", "disaster_maze_world.world", (0.0, 0.0, 0.0), "0.0 -4.8 6.4 0 0.94 1.5708"),
    ("u_pocket", "#11 U-Pocket Trap (Cul-de-sac + Walls)", "disaster_u_pocket_world.world", (0.0, 0.0, 0.0), "4.9 -5.4 9.6 0 1.06 1.5708"),
    ("slip_chicane", "#12 Slip Chicane (Ice Zone mu=0.32)", "disaster_slip_chicane_world.world", (0.0, 0.0, 0.0), "4.8 -4.8 9.2 0 1.06 1.5708"),
    ("dynamic_gauntlet", "#13 Dynamic Gauntlet (Spinner + Falling)", "disaster_dynamic_gauntlet_world.world", (-3.3, 2.8, 0.0), "0.0 -5.4 8.2 0 0.98 1.5708"),
    ("dynamic_crossfire", "#14 Dynamic Crossfire Alley (Oscillating)", "disaster_dynamic_crossfire_world.world", (0.0, 0.0, 0.0), "4.8 -4.8 9.2 0 1.06 1.5708"),
    ("smoke_lidar", "#15 Smoke LiDAR Hazard (Dust/Corruption)", "disaster_smoke_hazard_world.world", (0.0, 0.0, 0.0), "4.8 -4.8 9.2 0 1.06 1.5708"),
]


def _patch_world_camera(src_world: str, dst_world: str, cam_pose: str) -> str:
    with open(src_world, "r", encoding="utf-8") as f:
        txt = f.read()
    if "<camera" in txt:
        new_txt, _ = re.subn(
            r"(<camera\b[^>]*>\s*<pose(?:\s+frame='')?>)([^<]+)(</pose>)",
            rf"\g<1>{cam_pose}\g<3>",
            txt,
            count=1,
        )
    else:
        gui_xml = (
            f"    <gui fullscreen='0'>\n"
            f"      <camera name='user_camera'>\n"
            f"        <pose>{cam_pose}</pose>\n"
            f"        <view_controller>orbit</view_controller>\n"
            f"      </camera>\n"
            f"    </gui>\n"
        )
        new_txt = txt.replace("<world name=", gui_xml + "  <world name=", 1)
    with open(dst_world, "w", encoding="utf-8") as f:
        f.write(new_txt)
    return dst_world


def _capture_one(key: str, w_file: str, pose: Tuple[float, float, float], cam_pose: str, display: str, gz_port: int, ros_dom: int, out_png: str) -> None:
    gz_uri = f"http://127.0.0.1:{gz_port}"
    _kill_matching_gazebo_procs(gz_uri)
    use_vgl = bool(shutil.which("vglrun")) and os.path.exists("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    vgl = "vglrun -d egl " if use_vgl else ""
    sw_env = "" if use_vgl else "LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=16 "

    tmp_world = f"/tmp/disaster_recovery/atlas3d_{key}.world"
    os.makedirs("/tmp/disaster_recovery", exist_ok=True)
    _patch_world_camera(os.path.join(WS_WORLDS, w_file), tmp_world, cam_pose)

    env_prefix = (
        f"source /opt/ros/humble/setup.bash && source {WS_ROOT}/install/setup.bash && "
        f"export DISPLAY={display} {sw_env}QT_X11_NO_MITSHM=1 "
        f"ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID={ros_dom} GAZEBO_MASTER_URI={gz_uri} "
        f'GAZEBO_MODEL_DATABASE_URI="" && '
    )
    rx, ry, ryaw = pose
    p_gz = subprocess.Popen(
        ["bash", "-c", env_prefix + f"ros2 launch disaster_recovery disaster_world.launch.py x_pose:={rx} y_pose:={ry} yaw_pose:={ryaw} world:={tmp_world} gui:=false"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    p_cli = None
    try:
        time.sleep(4.5)
        p_cli = subprocess.Popen(
            ["bash", "-c", env_prefix + f"{vgl}gzclient"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        time.sleep(6.0)
        raw_shot = f"/tmp/disaster_recovery/raw3d_{key}.png"
        deadline = time.time() + 26.0
        captured_ok = False
        while time.time() < deadline:
            subprocess.run(
                ["ffmpeg", "-y", "-f", "x11grab", "-video_size", "1920x1080", "-i", display, "-frames:v", "1", raw_shot],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10.0,
            )
            if os.path.exists(raw_shot):
                im = Image.open(raw_shot).convert("RGB")
                # Check top-left viewport area (outside splash banner) to confirm Gazebo 3D scene finished loading
                corner = im.crop((260, 120, 520, 320))
                extrema = corner.getextrema()
                mean_v = sum(hi for _, hi in extrema) / 3.0
                if mean_v > 35.0:
                    time.sleep(1.0)
                    subprocess.run(
                        ["ffmpeg", "-y", "-f", "x11grab", "-video_size", "1920x1080", "-i", display, "-frames:v", "1", raw_shot],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=10.0,
                    )
                    im = Image.open(raw_shot)
                    cropped = im.crop((232, 82, 1216, 1004)).resize((960, 700), Image.Resampling.LANCZOS)
                    cropped.save(out_png)
                    captured_ok = True
                    print(f"[CAPTURED 3D] {key} -> {out_png}", flush=True)
                    break
            time.sleep(2.0)
        if not captured_ok:
            print(f"[WARN 3D TIMEOUT] {key}", flush=True)
    finally:
        for p in (p_cli, p_gz):
            if p is not None and p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    pass
        _kill_matching_gazebo_procs(gz_uri)


def main() -> int:
    _ensure_gazebo_gui_ini()
    display = ":97"
    gz_port = 11397
    ros_dom = 97
    p_xvfb = subprocess.Popen(
        ["Xvfb", display, "-screen", "0", "1920x1080x24", "-ac", "+extension", "GLX", "+render", "-noreset"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    time.sleep(1.5)
    cards = []
    try:
        for key, title, w_file, pose, cam_pose in WORLD_CARDS:
            p_png = f"/tmp/disaster_recovery/card3d_{key}.png"
            _capture_one(key, w_file, pose, cam_pose, display, gz_port, ros_dom, p_png)
            cards.append((title, p_png))
    finally:
        try:
            os.killpg(os.getpgid(p_xvfb.pid), signal.SIGKILL)
        except Exception:
            pass

    # Compose crisp 3x3 3D World Atlas
    cw, ch = 960, 700
    hdr_h = 42
    pad = 18
    top_banner = 66
    W = pad + 3 * (cw + pad)
    H = top_banner + 3 * (ch + hdr_h + pad) + pad
    canvas = Image.new("RGB", (W, H), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)

    font_title = ImageFont.load_default()
    font_card = ImageFont.load_default()
    for fp in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",):
        if os.path.exists(fp):
            font_title = ImageFont.truetype(fp, 30)
            font_card = ImageFont.truetype(fp, 21)
            break

    draw.text((pad, 16), "15 Disaster Scenarios: 3D Gazebo Simulation Environments (9 Unique Worlds)", font=font_title, fill=(15, 23, 42))

    for idx, (title, p_png) in enumerate(cards):
        r, c = divmod(idx, 3)
        x0 = pad + c * (cw + pad)
        y0 = top_banner + r * (ch + hdr_h + pad)
        draw.rectangle([(x0 - 2, y0 - 2), (x0 + cw + 2, y0 + hdr_h + ch + 2)], fill=(255, 255, 255), outline=(203, 213, 225), width=2)
        draw.rectangle([(x0, y0), (x0 + cw, y0 + hdr_h)], fill=(15, 23, 42))
        draw.text((x0 + 14, y0 + 10), title, font=font_card, fill=(248, 250, 252))
        if os.path.exists(p_png):
            im = Image.open(p_png)
            canvas.paste(im, (x0, y0 + hdr_h))

    out_atlas = os.path.join(WS_ROOT, "docs", "images", "scenario_worlds_15_3d.png")
    canvas.save(out_atlas, optimize=True)
    print(f"[SAVED 3D ATLAS] {out_atlas} ({os.path.getsize(out_atlas)} B)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
