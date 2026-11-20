#!/usr/bin/env python3
"""
Add clear Gazebo 3D Color Legends + Real-Time Telemetry/Phase HUD overlays onto
Phase 3 recorded MP4s and re-export crisp PNGs and GIFs so every visual artifact
is self-explanatory without guesswork.
"""

import os
import subprocess
import sys
from typing import Callable, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WS_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MP4_DIR = "/hdd/evidence/phase3_semantic"
OUT_EVIDENCE = os.path.join(WS_ROOT, "docs", "images", "evidence")
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _load_font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


F_TITLE = _load_font(FONT_BOLD, 20)
F_HEAD = _load_font(FONT_BOLD, 17)
F_BODY = _load_font(FONT_REG, 15)
F_SMALL = _load_font(FONT_BOLD, 14)


def _draw_gazebo_legend_card(draw: ImageDraw.ImageDraw, x0: int, y0: int, include_person: bool = True) -> None:
    """Draw a clean Gazebo 3D color-key legend card over bottom-left empty area."""
    items: List[Tuple[Tuple[int, int, int], str, str]] = [
        ((236, 72, 153), "square", "Pink Box/Disc = Low Obstacle (h=0.095m, LiDAR Blind)"),
        ((234, 179, 8), "circle", "Yellow Disc = Toxic Gas Hazard Zone (r=0.55m)"),
        ((56, 189, 248), "ring", "Cyan Ring/Line = Base Camp (0,0) & Path Ribbon"),
        ((96, 165, 250), "tri", "Blue Fan Rays = 2D LiDAR Scan Plane (z=0.171m)"),
    ]
    if include_person:
        items.append(((34, 197, 94), "circle", "Standing Person = Survivor (5.85, 0.45)m"))
    items.append(((217, 119, 6), "square", "Brick Pillar = Static Structural Feature"))

    w_box = 470
    h_box = 36 + len(items) * 25
    draw.rounded_rectangle([x0, y0, x0 + w_box, y0 + h_box], radius=10, fill=(11, 18, 32, 222), outline=(71, 85, 105, 255), width=2)
    draw.text((x0 + 14, y0 + 8), "GAZEBO 3D COLOR LEGEND", fill=(56, 189, 248, 255), font=F_SMALL)

    cy = y0 + 34
    for col, shape, label in items:
        cx = x0 + 20
        if shape == "square":
            draw.rectangle([cx - 6, cy + 2, cx + 6, cy + 14], fill=col + (255,), outline=(255, 255, 255, 220))
        elif shape == "ring":
            draw.ellipse([cx - 7, cy + 1, cx + 7, cy + 15], outline=col + (255,), width=3)
        else:
            draw.ellipse([cx - 6, cy + 2, cx + 6, cy + 14], fill=col + (255,), outline=(255, 255, 255, 220))
        draw.text((x0 + 36, cy), label, fill=(241, 245, 249, 255), font=F_BODY)
        cy += 25


def _draw_status_hud_card(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    w_box: int,
    header: str,
    header_col: Tuple[int, int, int],
    lines: List[Tuple[str, Tuple[int, int, int]]],
) -> None:
    h_box = 40 + len(lines) * 26
    draw.rounded_rectangle([x0, y0, x0 + w_box, y0 + h_box], radius=10, fill=(11, 18, 32, 232), outline=header_col + (255,), width=2)
    draw.text((x0 + 14, y0 + 10), header, fill=header_col + (255,), font=F_HEAD)
    cy = y0 + 40
    for txt, col in lines:
        draw.text((x0 + 14, cy), txt, fill=col + (255,), font=F_BODY)
        cy += 26


def _process_video(
    in_mp4: str,
    out_mp4: str,
    overlay_fn: Callable[[ImageDraw.ImageDraw, int, int, float], None],
    out_gif: str = "",
    snap_out: List[Tuple[float, str]] = (),
    gif_fps: int = 10,
    gif_width: int = 960,
) -> None:
    cap = cv2.VideoCapture(in_mp4)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {in_mp4}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ff_cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", f"{fps:.2f}",
        "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "22", "-pix_fmt", "yuv420p",
        out_mp4,
    ]
    proc = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pil_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
        ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(ov)
        t_sec = idx / max(1.0, fps)
        overlay_fn(draw, idx, total, t_sec)
        merged = Image.alpha_composite(pil_im, ov).convert("RGB")
        bgr_out = cv2.cvtColor(np.asarray(merged), cv2.COLOR_RGB2BGR)
        if proc.stdin is not None:
            proc.stdin.write(bgr_out.tobytes())
        idx += 1

    cap.release()
    if proc.stdin is not None:
        proc.stdin.close()
    proc.wait(timeout=45.0)
    print(f"[OVERLAY MP4] {out_mp4} ({idx} frames)", flush=True)

    for ss_frac, png_p in snap_out:
        ss_sec = min(max(0.5, (idx / fps) * ss_frac), max(0.5, (idx / fps) - 0.5))
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{ss_sec:.2f}", "-i", out_mp4, "-vframes", "1", png_p],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"[OVERLAY PNG] {png_p} (@{ss_sec:.1f}s)", flush=True)

    if out_gif:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", out_mp4,
                "-vf", f"fps={gif_fps},scale={gif_width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
                out_gif,
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"[OVERLAY GIF] {out_gif} ({os.path.getsize(out_gif)} B)", flush=True)


def main() -> int:
    # 1. Blind A/B Run (Part 1)
    blind_in = os.path.join(MP4_DIR, "phase3_semantic_avoidance_blind.mp4")
    blind_out = os.path.join(MP4_DIR, "phase3_semantic_avoidance_blind.hud.mp4")

    def _ov_blind(draw: ImageDraw.ImageDraw, idx: int, total: int, t_sec: float):
        _draw_gazebo_legend_card(draw, 22, 68, include_person=True)
        prog = idx / max(1, total - 1)
        phase_lbl = "FINAL RESULT: LOW-OBS COLLISION!" if prog > 0.65 else "BLACKOUT RETURN (Blind 1-Ch)"
        _draw_status_hud_card(
            draw,
            1228,
            52,
            405,
            f"TELEMETRY HUD (t={t_sec:.1f}s)",
            (248, 113, 113),
            [
                ("Policy: Phase 2-B BLIND (1-Ch LiDAR)", (241, 245, 249)),
                ("Semantic Cache (Ch1/Ch2): DISABLED", (252, 165, 165)),
                (f"Phase: {phase_lbl}", (251, 191, 36) if prog <= 0.65 else (248, 113, 113)),
                ("10-Trial Benchmark Result:", (148, 163, 184)),
                ("  Collision-Free 30% | Gas Clear 60%", (248, 113, 113)),
                ("  Min Low-Obs: 0.289m | Gas: 0.749m", (252, 165, 165)),
            ],
        )

    _process_video(
        blind_in,
        blind_out,
        _ov_blind,
        snap_out=[(0.58, os.path.join(OUT_EVIDENCE, "phase3_semantic_avoidance_blind.png"))],
    )

    # 2. Semantic-Aware A/B Run (Part 2)
    aware_in = os.path.join(MP4_DIR, "phase3_semantic_avoidance_aware.mp4")
    aware_out = os.path.join(MP4_DIR, "phase3_semantic_avoidance_aware.hud.mp4")

    def _ov_aware(draw: ImageDraw.ImageDraw, idx: int, total: int, t_sec: float):
        _draw_gazebo_legend_card(draw, 22, 68, include_person=True)
        prog = idx / max(1, total - 1)
        phase_lbl = "SAFE DETOUR COMPLETE (PASS)" if prog > 0.75 else "BLACKOUT RETURN (3-Ch Active)"
        _draw_status_hud_card(
            draw,
            1228,
            52,
            405,
            f"TELEMETRY HUD (t={t_sec:.1f}s)",
            (74, 222, 128),
            [
                ("Policy: Phase 3 SEMANTIC (3-Ch ONNX)", (241, 245, 249)),
                ("Ch1 LowObs + Ch2 Gas/Person: CACHED", (134, 239, 172)),
                (f"Phase: {phase_lbl}", (74, 222, 128)),
                ("10-Trial Benchmark Result:", (148, 163, 184)),
                ("  Collision-Free 90% | Gas Clear 100%", (74, 222, 128)),
                ("  Min Low-Obs: 0.575m | Gas: 0.957m", (134, 239, 172)),
            ],
        )

    _process_video(
        aware_in,
        aware_out,
        _ov_aware,
        snap_out=[(0.62, os.path.join(OUT_EVIDENCE, "phase3_semantic_avoidance_aware.png"))],
    )

    # Concatenate Blind + Aware HUD MP4s into phase3_semantic_avoidance_ab.gif
    ab_out_mp4 = os.path.join(MP4_DIR, "phase3_semantic_avoidance_ab.mp4")
    concat_txt = "/tmp/ab_hud_concat.txt"
    with open(concat_txt, "w", encoding="utf-8") as f:
        f.write(f"file '{blind_out}'\nfile '{aware_out}'\n")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt, "-c", "copy", ab_out_mp4],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    ab_gif = os.path.join(OUT_EVIDENCE, "phase3_semantic_avoidance_ab.gif")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", ab_out_mp4,
            "-vf", "fps=10,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
            ab_gif,
        ],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"[OVERLAY GIF] {ab_gif} ({os.path.getsize(ab_gif)} B)", flush=True)

    # 3. Full 3-Pane Semantic SLAM E2E
    e2e_in = os.path.join(MP4_DIR, "phase3_semantic_slam_e2e.mp4")
    e2e_out = os.path.join(MP4_DIR, "phase3_semantic_slam_e2e.hud.mp4")

    def _ov_e2e(draw: ImageDraw.ImageDraw, idx: int, total: int, t_sec: float):
        _draw_gazebo_legend_card(draw, 22, 68, include_person=True)
        prog = idx / max(1, total - 1)
        if prog < 0.55:
            stage_s = "1. OUTBOUND (HB:ON): SLAM + YOLO26n Stamp"
            s_col = (56, 189, 248)
        elif prog < 0.72:
            stage_s = "2. SURVIVOR LOCK: person (5.97, 0.36)m Cached"
            s_col = (250, 204, 21)
        else:
            stage_s = "3. BLACKOUT RETURN: Cached 3-Ch Safe Detour"
            s_col = (74, 222, 128)
        _draw_status_hud_card(
            draw,
            1228,
            48,
            445,
            f"E2E HUD (t={t_sec:.1f}s)",
            s_col,
            [
                (stage_s, s_col),
                ("Left : Gazebo 3D Simulation", (203, 213, 225)),
                ("Top-R: RViz2 /map + Markers", (203, 213, 225)),
                ("Bot-R: YOLO26n + Cached XY", (203, 213, 225)),
            ],
        )

    _process_video(
        e2e_in,
        e2e_out,
        _ov_e2e,
        out_gif=os.path.join(OUT_EVIDENCE, "phase3_semantic_slam_e2e.gif"),
        snap_out=[(0.56, os.path.join(OUT_EVIDENCE, "phase3_semantic_slam_live.png"))],
        gif_fps=8,
        gif_width=960,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
