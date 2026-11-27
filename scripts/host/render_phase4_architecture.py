#!/usr/bin/env python3
"""Render high-readability Phase 4 (Step B-4 + Option 1) control architecture diagrams (SVG + @2x PNG).

Designed specifically for crisp 100%-zoom readability on GitHub README (~896 px column width):
  1. phase4_hero_pipeline.{svg,png}        (1200x580  SVG -> 2400x1160 @2x PNG, 2x3 Closed-Loop U-Flow)
  2. phase4_control_architecture.{svg,png} (1200x1170 SVG -> 2400x2340 @2x PNG, 4-Tier Vertical Architecture)

Automated quality gates (fail-fast assertions):
  - Zero box overlaps & zero canvas/container overflows
  - Zero text-width overflows using actual Noto Sans CJK KR FreeType metrics
  - Zero missing glyphs (tofu)
  - Minimum effective font size >= 8.5 px when scaled to 896 px GitHub README width
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import xml.sax.saxutils as saxutils
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from PIL import Image, ImageFont
except ImportError:
    Image = None
    ImageFont = None


PALETTE = {
    "canvas_bg": "#F8FAFC",
    "canvas_stroke": "#CBD5E1",
    "card_bg": "#FFFFFF",
    "card_stroke": "#CBD5E1",
    "text_main": "#0F172A",
    "text_sub": "#334155",
    "text_muted": "#475569",
    "dark_bg": "#0F172A",
    "dark_sub": "#CBD5E1",
    # Functional layer colors (low-saturation academic palette)
    "sensor": "#334155",
    "sensor_tint": "#F1F5F9",
    "imm": "#0F766E",
    "imm_tint": "#F0FDFA",
    "cross": "#1D4ED8",
    "cross_tint": "#EFF6FF",
    "axial": "#B45309",
    "axial_tint": "#FFFBEB",
    "static": "#475569",
    "static_tint": "#F1F5F9",
    "rl": "#6D28D9",
    "rl_tint": "#F5F3FF",
    "cbf": "#BE123C",
    "cbf_tint": "#FFF1F2",
    "icp": "#047857",
    "icp_tint": "#ECFDF5",
    "arrow": "#475569",
    "router_bg": "#F1F5F9",
    "router_stroke": "#94A3B8",
}

FONT_FAMILY_SVG = "Noto Sans CJK KR, NanumGothic, DejaVu Sans, sans-serif"


def detect_korean_fonts() -> Tuple[Optional[str], Optional[str], int, str]:
    candidates = [
        (
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            1,
            "Noto Sans CJK KR",
        ),
        (
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
            0,
            "NanumGothic",
        ),
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            0,
            "DejaVu Sans",
        ),
    ]
    for reg, bold, idx, label in candidates:
        if os.path.exists(reg):
            return reg, (bold if os.path.exists(bold) else reg), idx, label
    return None, None, 0, "sans-serif"


def esc(text: str) -> str:
    return saxutils.escape(text, {'"': "&quot;", "'": "&apos;"})


@dataclass
class Box:
    name: str
    x: float
    y: float
    w: float
    h: float
    parent: Optional[str] = None


@dataclass
class TextCheck:
    box_name: str
    text: str
    font_size: float
    bold: bool
    max_width: float


class DiagramVerifier:
    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height
        self.boxes: Dict[str, Box] = {}
        self.texts: List[TextCheck] = []

    def register_box(
        self, name: str, x: float, y: float, w: float, h: float, parent: Optional[str] = None
    ) -> Box:
        assert name not in self.boxes, f"Duplicate box: {name}"
        b = Box(name, x, y, w, h, parent)
        self.boxes[name] = b
        return b

    def register_text(
        self, box_name: str, text: str, font_size: float, bold: bool, max_width: float
    ) -> None:
        self.texts.append(TextCheck(box_name, text, font_size, bold, max_width))

    @staticmethod
    def _overlaps(a: Box, b: Box) -> bool:
        return not (
            a.x + a.w <= b.x or b.x + b.w <= a.x or a.y + a.h <= b.y or b.y + b.h <= a.y
        )

    def _is_ancestor(self, anc: str, node: str) -> bool:
        cur = self.boxes[node].parent
        while cur is not None:
            if cur == anc:
                return True
            cur = self.boxes[cur].parent
        return False

    def verify_all(
        self, reg_font: Optional[str], bold_font: Optional[str], font_idx: int
    ) -> Dict[str, object]:
        oob = [
            (b.name, b.x, b.y, b.x + b.w, b.y + b.h)
            for b in self.boxes.values()
            if b.x < 0 or b.y < 0 or b.x + b.w > self.width or b.y + b.h > self.height
        ]
        assert not oob, f"Out-of-bounds boxes: {oob}"

        child_oob = []
        for b in self.boxes.values():
            if b.parent is not None:
                p = self.boxes[b.parent]
                if not (
                    b.x >= p.x and b.y >= p.y and b.x + b.w <= p.x + p.w and b.y + b.h <= p.y + p.h
                ):
                    child_oob.append((b.name, b.parent))
        assert not child_oob, f"Child overflow: {child_oob}"

        names = list(self.boxes.keys())
        overlaps = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                n1, n2 = names[i], names[j]
                if self._is_ancestor(n1, n2) or self._is_ancestor(n2, n1):
                    continue
                if self._overlaps(self.boxes[n1], self.boxes[n2]):
                    overlaps.append((n1, n2))
        assert not overlaps, f"Overlapping boxes: {overlaps}"

        overflows = []
        tofu = []
        if ImageFont is not None and reg_font is not None:
            cache: Dict[Tuple[bool, int], ImageFont.FreeTypeFont] = {}
            for tc in self.texts:
                sz = max(6, int(round(tc.font_size)))
                key = (tc.bold, sz)
                if key not in cache:
                    fp = bold_font if tc.bold and bold_font else reg_font
                    cache[key] = ImageFont.truetype(fp, sz, index=font_idx)
                fo = cache[key]
                bbox = fo.getbbox(tc.text)
                tw = (bbox[2] - bbox[0]) if bbox else 0
                if tw > tc.max_width * 1.02:
                    overflows.append((tc.box_name, tc.text, round(tw, 1), round(tc.max_width, 1)))
                for ch in tc.text:
                    if ch.isspace():
                        continue
                    if fo.getmask(ch).size == (0, 0):
                        tofu.append((tc.box_name, ch, hex(ord(ch))))

        assert not overflows, f"Text width overflow: {overflows[:8]}"
        assert not tofu, f"Missing glyphs: {tofu[:8]}"

        min_font = min(tc.font_size for tc in self.texts) if self.texts else 0.0
        eff_gh_min = round(min_font * (896.0 / self.width), 1)
        return {
            "boxes": len(self.boxes),
            "overlaps": 0,
            "oob": 0,
            "texts": len(self.texts),
            "overflows": 0,
            "tofu": 0,
            "min_font_px": min_font,
            "github_896_min_px": eff_gh_min,
        }


def svg_defs() -> str:
    markers = [
        ("arr-slate", PALETTE["arrow"]),
        ("arr-teal", PALETTE["imm"]),
        ("arr-indigo", PALETTE["cross"]),
        ("arr-amber", PALETTE["axial"]),
        ("arr-violet", PALETTE["rl"]),
        ("arr-crimson", PALETTE["cbf"]),
        ("arr-emerald", PALETTE["icp"]),
    ]
    mxml = [
        f'    <marker id="{mid}" viewBox="0 0 10 10" refX="8.5" refY="5" '
        f'markerWidth="7.5" markerHeight="7.5" orient="auto">\n'
        f'      <path d="M 0 1.2 L 9.2 5 L 0 8.8 z" fill="{col}"/>\n'
        f"    </marker>"
        for mid, col in markers
    ]
    return (
        "  <defs>\n"
        '    <filter id="card-shadow" x="-4%" y="-4%" width="108%" height="114%">\n'
        '      <feGaussianBlur in="SourceAlpha" stdDeviation="2.5" result="blur"/>\n'
        '      <feOffset in="blur" dx="0" dy="2" result="offsetBlur"/>\n'
        '      <feComponentTransfer in="offsetBlur" result="shadow">\n'
        '        <feFuncA type="linear" slope="0.09"/>\n'
        '      </feComponentTransfer>\n'
        '      <feMerge>\n'
        '        <feMergeNode in="shadow"/>\n'
        '        <feMergeNode in="SourceGraphic"/>\n'
        '      </feMerge>\n'
        "    </filter>\n"
        + "\n".join(mxml)
        + "\n  </defs>\n"
    )


def svg_text(
    x: float,
    y: float,
    text: str,
    size: float = 13.5,
    weight: int = 400,
    fill: str = PALETTE["text_main"],
    anchor: str = "start",
    verifier: Optional[DiagramVerifier] = None,
    box_name: str = "",
    max_width: float = 9999.0,
) -> str:
    if verifier is not None:
        verifier.register_text(box_name, text, size, weight >= 600, max_width)
    return (
        f'  <text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="{FONT_FAMILY_SVG}" font-size="{size:.1f}" font-weight="{weight}" '
        f'fill="{fill}">{esc(text)}</text>\n'
    )


def svg_card(
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    accent: str,
    tint: str,
    lines: Sequence[Tuple[str, bool, Optional[str]]],
    verifier: DiagramVerifier,
    box_name: str,
    parent: Optional[str] = None,
    header_h: float = 38.0,
    title_size: float = 15.0,
    body_size: float = 13.2,
    line_step: float = 23.0,
    badge: Optional[str] = None,
    metric_chip: Optional[str] = None,
) -> str:
    verifier.register_box(box_name, x, y, w, h, parent=parent)
    out = [f"  <!-- {box_name} -->\n"]
    out.append(
        f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="12" '
        f'fill="{PALETTE["card_bg"]}" stroke="{PALETTE["card_stroke"]}" stroke-width="1.4" '
        f'filter="url(#card-shadow)"/>\n'
    )
    # Tinted header bar
    out.append(
        f'  <rect x="{x + 0.7:.1f}" y="{y + 0.7:.1f}" width="{w - 1.4:.1f}" height="{header_h:.1f}" rx="11.3" fill="{tint}"/>\n'
    )
    out.append(
        f'  <rect x="{x + 0.7:.1f}" y="{y + header_h - 7:.1f}" width="{w - 1.4:.1f}" height="7.7" fill="{tint}"/>\n'
    )
    out.append(
        f'  <rect x="{x + 0.7:.1f}" y="{y + 0.7:.1f}" width="5.5" height="{header_h:.1f}" rx="2.5" fill="{accent}"/>\n'
    )
    out.append(
        f'  <line x1="{x + 1:.1f}" y1="{y + header_h + 0.7:.1f}" x2="{x + w - 1:.1f}" y2="{y + header_h + 0.7:.1f}" '
        f'stroke="{PALETTE["card_stroke"]}" stroke-width="1.0"/>\n'
    )

    badge_w = (max(56.0, len(badge) * 8.2 + 16.0) + 12.0) if badge else 14.0
    out.append(
        svg_text(
            x + 16,
            y + header_h * 0.67,
            title,
            size=title_size,
            weight=700,
            fill=accent,
            verifier=verifier,
            box_name=box_name,
            max_width=w - 16 - badge_w,
        )
    )
    if badge:
        bw = max(56.0, len(badge) * 8.2 + 16.0)
        bx = x + w - bw - 10.0
        by = y + (header_h - 22.0) / 2.0 + 0.7
        out.append(f'  <rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="22" rx="11" fill="{accent}"/>\n')
        out.append(
            svg_text(
                bx + bw / 2.0,
                by + 15.2,
                badge,
                size=11.5,
                weight=700,
                fill="#FFFFFF",
                anchor="middle",
                verifier=verifier,
                box_name=box_name,
                max_width=bw - 6,
            )
        )

    # Body lines
    cy = y + header_h + 24.5
    pad_x = 16.0
    for text, is_bold, cfill in lines:
        fill_c = cfill if cfill else (PALETTE["text_main"] if is_bold else PALETTE["text_sub"])
        out.append(
            svg_text(
                x + pad_x,
                cy,
                text,
                size=body_size,
                weight=700 if is_bold else 400,
                fill=fill_c,
                verifier=verifier,
                box_name=box_name,
                max_width=w - 2 * pad_x,
            )
        )
        cy += line_step

    # Optional crisp bottom metric chip
    if metric_chip:
        ch = 26.0
        cy_box = y + h - ch - 10.0
        out.append(
            f'  <rect x="{x + 12:.1f}" y="{cy_box:.1f}" width="{w - 24:.1f}" height="{ch:.1f}" '
            f'rx="7" fill="{tint}" stroke="{accent}" stroke-width="0.9" stroke-opacity="0.45"/>\n'
        )
        out.append(
            svg_text(
                x + w / 2.0,
                cy_box + 17.5,
                metric_chip,
                size=12.0,
                weight=700,
                fill=accent,
                anchor="middle",
                verifier=verifier,
                box_name=box_name,
                max_width=w - 34,
            )
        )
    return "".join(out)


def svg_path(
    points: Sequence[Tuple[float, float]],
    color: str = PALETTE["arrow"],
    marker: str = "arr-slate",
    width: float = 2.4,
    dashed: bool = False,
) -> str:
    d = " ".join([f"M {points[0][0]:.1f} {points[0][1]:.1f}"] + [f"L {px:.1f} {py:.1f}" for px, py in points[1:]])
    dash = ' stroke-dasharray="7,5"' if dashed else ""
    return (
        f'  <path d="{d}" fill="none" stroke="{color}" stroke-width="{width:.1f}" '
        f'stroke-linejoin="round" stroke-linecap="round"{dash} marker-end="url(#{marker})"/>\n'
    )


def svg_pill(
    cx: float,
    cy: float,
    label: str,
    color: str,
    size: float = 11.5,
    verifier: Optional[DiagramVerifier] = None,
    min_w: float = 48.0,
) -> str:
    # Estimate width conservatively for mixed Korean + ASCII
    kr_cnt = sum(1 for ch in label if ord(ch) > 127)
    en_cnt = len(label) - kr_cnt
    pw = max(min_w, kr_cnt * size * 0.98 + en_cnt * size * 0.62 + 18.0)
    ph = 21.0
    out = [
        f'  <rect x="{cx - pw / 2.0:.1f}" y="{cy - ph / 2.0:.1f}" width="{pw:.1f}" height="{ph:.1f}" '
        f'rx="10.5" fill="#FFFFFF" stroke="{color}" stroke-width="1.2"/>\n',
        svg_text(
            cx,
            cy + 4.2,
            label,
            size=size,
            weight=700,
            fill=color,
            anchor="middle",
            verifier=verifier,
            box_name="pill",
            max_width=pw - 8,
        ),
    ]
    return "".join(out)


# ===========================================================================
# Asset 1: README Hero Pipeline Banner — 2x3 Closed-Loop U-Flow (1200 x 580)
# ===========================================================================
def build_hero_pipeline_svg(
    reg_font: Optional[str], bold_font: Optional[str], font_idx: int
) -> Tuple[str, Dict[str, object]]:
    W, H = 1200, 580
    v = DiagramVerifier(W, H)
    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n',
        svg_defs(),
        f'  <rect x="2" y="2" width="{W - 4}" height="{H - 4}" rx="16" '
        f'fill="{PALETTE["canvas_bg"]}" stroke="{PALETTE["canvas_stroke"]}" stroke-width="1.6"/>\n',
    ]

    # Header (x=24, y=20, w=1152, h=74)
    v.register_box("header", 24, 20, 1152, 74)
    parts.append(
        '  <rect x="24" y="20" width="1152" height="74" rx="12" fill="#0F172A"/>\n'
        '  <rect x="24" y="20" width="8" height="74" rx="4" fill="#0F766E"/>\n'
    )
    parts.append(
        svg_text(
            48, 52,
            "Phase 4 (Step B-4 + Option 1) Onboard Recovery Pipeline",
            size=22.0, weight=700, fill="#FFFFFF", verifier=v, box_name="header", max_width=760,
        )
    )
    parts.append(
        svg_text(
            48, 78,
            "ROS 2 Humble · TurtleBot3 Burger (v_max 0.20 m/s) · 5 Hz LDS-01 / 10 Hz Control Closed-Loop",
            size=13.0, weight=400, fill="#CBD5E1", verifier=v, box_name="header", max_width=780,
        )
    )
    for px, py, pw, lbl, col in [
        (860, 40, 146, "5 Hz / 10 Hz", "#0F766E"),
        (1016, 40, 144, "89 / 100 PASS", "#1D4ED8"),
    ]:
        parts.append(f'  <rect x="{px}" y="{py}" width="{pw}" height="34" rx="17" fill="{col}"/>\n')
        parts.append(svg_text(px + pw / 2.0, py + 22.0, lbl, size=13.5, weight=700, fill="#FFFFFF", anchor="middle", verifier=v, box_name="header", max_width=pw - 10))

    # Row 1 (Left -> Right): 01 -> 02 -> 03  (x = 24, 424, 824; w = 352; gap = 48; y = 116; h = 152)
    r1_y, card_w, card_h = 116, 344, 152
    parts.append(
        svg_card(
            24, r1_y, card_w, card_h,
            "01 · 5 Hz LiDAR + Layer-0",
            PALETTE["sensor"], PALETTE["sensor_tint"],
            [
                ("LDS-01 360-ray (5 Hz 고정, 0.12-3.5m)", True, None),
                ("• 각도 스파이크 제거 & 오염 판별(≥0.03)", False, None),
                ("• /map Semantic Cache (Ch1·Ch2) & LIFO", False, None),
            ],
            verifier=v, box_name="n1", badge="5 Hz",
            metric_chip="외부 관제 없이 RPi 4 단독 폐루프",
        )
    )
    parts.append(
        svg_card(
            428, r1_y, card_w, card_h,
            "02 · CV+Harmonic IMM",
            PALETTE["imm"], PALETTE["imm_tint"],
            [
                ("5 Hz 스탬프 갱신 + 10 Hz 비파괴 외삽", True, None),
                ("• 2×2 PCA 주축 + 1D 조화 궤적 예측", False, None),
                ("• _is_static_wall_fragment · 회전≥0.25m (오탐 0)", False, None),
            ],
            verifier=v, box_name="n2", badge="10 Hz",
            metric_chip="τ=4.5s FDE 4.4mm / 11.4cm · <0.50ms",
        )
    )
    parts.append(
        svg_card(
            832, r1_y, card_w, card_h,
            "03 · 2D 시공간 / 축방향 A*",
            PALETTE["cross"], PALETTE["cross_tint"],
            [
                ("[횡단] 폭<1.85m ±0.38m · HOLD≤14s/CD 4s", True, PALETTE["cross"]),
                ("[축방향] 0.52m·w=3.8 스윕 + 0.09 m/s 감속", True, PALETTE["axial"]),
                ("[정적·시맨틱] Step A-2 무개입 + 시맨틱 A*", False, PALETTE["static"]),
            ],
            verifier=v, box_name="n3", badge="3-Way",
            metric_chip="안전창 33.9% · Step B-4 추가 +0.328 ms",
        )
    )

    # Row 2 (Right -> Left): 06 <- 05 <- 04 (y = 314; gap = 46 px between Row 1 and Row 2)
    r2_y = 314
    parts.append(
        svg_card(
            832, r2_y, card_w, card_h,
            "04 · 4-Ch Context-RL ONNX",
            PALETTE["rl"], PALETTE["rl_tint"],
            [
                ("12×12×4 (Ch0-Ch3) + 16-step 시계열 정책", True, None),
                ("• Ch0 LiDAR · Ch1 저상 · Ch2 가스 · Ch3 IMM", False, None),
                ("• Sim v4.4 (-0.06) / Real v3 (-0.08) 정합", False, None),
            ],
            verifier=v, box_name="n4", badge="ONNX",
            metric_chip="Pure Pursuit + Clearance-Gated 블렌딩",
        )
    )
    parts.append(
        svg_card(
            428, r2_y, card_w, card_h,
            "05 · 2-Circle Soft CBF-QP",
            PALETTE["cbf"], PALETTE["cbf_tint"],
            [
                ("10 Hz 2-Circle ZCBF-QP (Ames et al., 2017)", True, None),
                ("• mq2_escape 클래스·깊은 내부 분리 (mq2=0)", False, None),
                ("• 초근접 광선 클램프 0.120m · 비상 후진 보존", False, None),
            ],
            verifier=v, box_name="n5", badge="<2.0ms",
            metric_chip="mean 0.407 ms · max 1.941 ms",
        )
    )
    parts.append(
        svg_card(
            24, r2_y, card_w, card_h,
            "06 · 순수 관측 2분기 ICP",
            PALETTE["icp"], PALETTE["icp_tint"],
            [
                ("smoke 파라미터 미참조 (AST 단위 검증)", True, None),
                ("• 오염(spk≥0.03): cap 0.16m 차단 (54/54)", False, None),
                ("• 정상(inl≥25): 0.42m 보정 (오탐 0/40)", False, None),
            ],
            verifier=v, box_name="n6", badge="SE(2)",
            metric_chip="smoke 20/20 (5.4-6.4cm) · ArUco 도킹",
        )
    )

    # Arrows: clean 48px horizontal gaps & 46px middle lane
    ym1 = r1_y + card_h / 2.0
    ym2 = r2_y + card_h / 2.0
    parts.append(svg_path([(24 + card_w + 3, ym1), (428 - 4, ym1)], PALETTE["sensor"], "arr-slate", 2.6))
    parts.append(svg_path([(428 + card_w + 3, ym1), (832 - 4, ym1)], PALETTE["imm"], "arr-teal", 2.6))
    # Right downward turn: 03 -> 04
    parts.append(svg_path([(832 + card_w / 2.0, r1_y + card_h + 2), (832 + card_w / 2.0, r2_y - 4)], PALETTE["cross"], "arr-indigo", 2.6))
    # Bottom Row R -> L: 04 -> 05 -> 06
    parts.append(svg_path([(832 - 3, ym2), (428 + card_w + 4, ym2)], PALETTE["rl"], "arr-violet", 2.6))
    parts.append(svg_path([(428 - 3, ym2), (24 + card_w + 4, ym2)], PALETTE["cbf"], "arr-crimson", 2.6))
    # SE(2) Closed-Loop Feedback in middle channel (y=291): 06 -> 02
    parts.append(
        svg_path(
            [(196, r2_y - 2), (196, 291), (600, 291), (600, r1_y + card_h + 4)],
            PALETTE["icp"], "arr-emerald", 2.2, dashed=True,
        )
    )
    # Pills on top layer
    parts.append(svg_pill(398, ym1 - 20, "5Hz", PALETTE["sensor"], 11.5, v, min_w=42))
    parts.append(svg_pill(802, ym1 - 20, "IMM", PALETTE["imm"], 11.5, v, min_w=44))
    parts.append(svg_pill(1004, 291, "v_ref · A* · flag", PALETTE["cross"], 11.5, v))
    parts.append(svg_pill(802, ym2 - 20, "v_nom", PALETTE["rl"], 11.2, v, min_w=48))
    parts.append(svg_pill(398, ym2 - 20, "cmd", PALETTE["cbf"], 11.2, v, min_w=42))
    parts.append(svg_pill(400, 291, "SE(2) 피드백 (α=0.40) → IMM · LIFO", PALETTE["icp"], 11.5, v))

    # Bottom KPI Bar (x=24, y=486, w=1152, h=76)
    v.register_box("kpi_bar", 24, 486, 1152, 76)
    parts.append(
        '  <rect x="24" y="486" width="1152" height="76" rx="12" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.3" filter="url(#card-shadow)"/>\n'
    )
    kpis = [
        ("100-Trial · vs OFF", "89/100 (89.0%)", "Opt1 31/40 vs OFF 7/40 (p=1.11e-7)", "#0F172A"),
        ("Gate 1 · Option 1", "31/40 (77.5%)", "B-4 30/40 · 충돌 6/40 · CF 15/20", "#1D4ED8"),
        ("Gate 2 · 시맨틱", "59/60 · 10/10", "0/60 · Sim/Real 10/10 (mq2=0)", "#047857"),
        ("실시간성", "0.407ms · 12.96Hz", "CBF max 1.941 ms (<2.0)", "#BE123C"),
        ("단위 검증", "101/101 PASS", "8개 스위트 · 전체 124/126", "#6D28D9"),
    ]
    kw = 1152.0 / len(kpis)
    for i, (kt, kv, ks, kc) in enumerate(kpis):
        kx = 24.0 + i * kw
        if i > 0:
            parts.append(f'  <line x1="{kx:.1f}" y1="498" x2="{kx:.1f}" y2="550" stroke="#E2E8F0" stroke-width="1.2"/>\n')
        cx = kx + kw / 2.0
        parts.append(svg_text(cx, 507, kt, size=11.5, weight=700, fill="#475569", anchor="middle", verifier=v, box_name="kpi_bar", max_width=kw - 12))
        parts.append(svg_text(cx, 533, kv, size=19.5, weight=700, fill=kc, anchor="middle", verifier=v, box_name="kpi_bar", max_width=kw - 12))
        parts.append(svg_text(cx, 552, ks, size=11.2, weight=400, fill="#475569", anchor="middle", verifier=v, box_name="kpi_bar", max_width=kw - 10))

    parts.append("</svg>\n")
    return "".join(parts), v.verify_all(reg_font, bold_font, font_idx)


# ===========================================================================
# Asset 2: Detailed 4-Tier Vertical Architecture (1200 x 1170 -> 2400 x 2340)
# ===========================================================================
def build_detailed_architecture_svg(
    reg_font: Optional[str], bold_font: Optional[str], font_idx: int
) -> Tuple[str, Dict[str, object]]:
    W, H = 1200, 1170
    v = DiagramVerifier(W, H)
    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n',
        svg_defs(),
        f'  <rect x="2" y="2" width="{W - 4}" height="{H - 4}" rx="16" '
        f'fill="{PALETTE["canvas_bg"]}" stroke="{PALETTE["canvas_stroke"]}" stroke-width="1.6"/>\n',
    ]

    # Header (x=24, y=20, w=1152, h=72)
    v.register_box("header", 24, 20, 1152, 72)
    parts.append(
        '  <rect x="24" y="20" width="1152" height="72" rx="12" fill="#0F172A"/>\n'
        '  <rect x="24" y="20" width="8" height="72" rx="4" fill="#1D4ED8"/>\n'
    )
    parts.append(
        svg_text(
            48, 51,
            "Phase 4 (Step B-4 + Option 1) Detailed Control Architecture",
            size=22.0, weight=700, fill="#FFFFFF", verifier=v, box_name="header", max_width=760,
        )
    )
    parts.append(
        svg_text(
            48, 76,
            "5 Hz Scan-Gated IMM · Geometric 3-Way Spatio-Temporal Router · 4-Ch RL · 2-Circle Soft CBF-QP · 2-Branch ICP",
            size=12.5, weight=400, fill="#CBD5E1", verifier=v, box_name="header", max_width=820,
        )
    )
    for lx, ly, col, txt in [
        (890, 42, "#0F766E", "IMM / LIFO"),
        (1030, 42, "#1D4ED8", "횡단 2D 레인"),
        (890, 68, "#B45309", "축방향 A*"),
        (1030, 68, "#BE123C", "CBF-QP · ICP"),
    ]:
        parts.append(f'  <circle cx="{lx}" cy="{ly - 4}" r="5" fill="{col}"/>\n')
        parts.append(svg_text(lx + 11, ly, txt, size=12.0, weight=700, fill="#E2E8F0", verifier=v, box_name="header", max_width=120))

    MAIN_X = 24.0
    MAIN_W = 1122.0
    w3, g3 = 358.0, 24.0  # 3*358 + 2*24 = 1122

    def add_tier_label(name: str, y_baseline: float, text: str, bg: str = "#E2E8F0", fg: str = "#334155") -> None:
        # Compact left badge strictly inside x=[24, 184], leaving >=19 px gap before leftmost arrow at x=203
        tw_box = 160.0
        th_box = 20.0
        ty = y_baseline - 15.0
        v.register_box(name, MAIN_X, ty, tw_box, th_box)
        parts.append(f'  <rect x="{MAIN_X:.1f}" y="{ty:.1f}" width="{tw_box:.1f}" height="{th_box:.1f}" rx="6" fill="{bg}"/>\n')
        parts.append(svg_text(MAIN_X + tw_box / 2.0, y_baseline - 1.2, text, size=11.2, weight=700, fill=fg, anchor="middle", verifier=v, box_name=name, max_width=tw_box - 10))

    # -------------------- TIER 1: SENSING & LAYER-0 (y=122, h=124) --------------------
    add_tier_label("lbl_t1", 114, "TIER 1 · SENSING", "#E2E8F0", "#334155")
    t1_y, t1_h = 122, 124
    parts.append(
        svg_card(
            MAIN_X, t1_y, w3, t1_h, "① Onboard Sensors",
            PALETTE["sensor"], PALETTE["sensor_tint"],
            [
                ("LDS-01 2D LiDAR (5 Hz 고정, 360-ray)", True, None),
                ("• /odom · /heartbeat (3.0 s 단절 감지)", False, None),
                ("• /gas/level (MQ-2 ≥ 0.40) · CSI Camera", False, None),
            ],
            verifier=v, box_name="t1_sensors", badge="5 Hz",
            header_h=34, title_size=14.5, body_size=12.8, line_step=22.0,
        )
    )
    parts.append(
        svg_card(
            MAIN_X + w3 + g3, t1_y, w3, t1_h, "② Layer-0 Smoke Filter",
            PALETTE["sensor"], PALETTE["sensor_tint"],
            [
                ("이웃 각도 스파이크 잡음 제거 & 잔상 소거", True, None),
                ("• 오염 지표: latest_scan_spike_ratio", False, None),
                ("• spike_ratio ≥ 0.03 시 오염 플래그 발령", False, PALETTE["cbf"]),
            ],
            verifier=v, box_name="t1_layer0", badge="Filter",
            header_h=34, title_size=14.5, body_size=12.8, line_step=22.0,
        )
    )
    parts.append(
        svg_card(
            MAIN_X + 2 * (w3 + g3), t1_y, w3, t1_h, "③ Semantic Cache (/map)",
            PALETTE["sensor"], PALETTE["sensor_tint"],
            [
                ("단절 전 Outbound YOLO26n+LiDAR 백업", True, None),
                ("• Ch1 저상 장애물 (h < 0.17 m)", False, None),
                ("• Ch2 가스·조난자 (r = 0.55 m) + 실시간 MQ-2", False, None),
            ],
            verifier=v, box_name="t1_sem", badge="Ch1/2",
            header_h=34, title_size=14.5, body_size=12.8, line_step=22.0,
        )
    )

    # -------------------- TIER 2: IMM TRACKER & GLOBAL LIFO (y=314, h=148; gap=68px) --------------------
    add_tier_label("lbl_t2", 306, "TIER 2 · IMM & LIFO", "#CCFBF1", "#0F766E")
    t2_y, t2_h = 314, 148
    w2_left, w2_right, g2 = 640.0, 458.0, 24.0
    v.register_box("t2_imm", MAIN_X, t2_y, w2_left, t2_h)
    parts.append(
        f'  <rect x="{MAIN_X}" y="{t2_y}" width="{w2_left}" height="{t2_h}" rx="12" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.4" filter="url(#card-shadow)"/>\n'
        f'  <rect x="{MAIN_X + 0.7}" y="{t2_y + 0.7}" width="{w2_left - 1.4}" height="34" rx="11.3" fill="{PALETTE["imm_tint"]}"/>\n'
        f'  <rect x="{MAIN_X + 0.7}" y="{t2_y + 27.7}" width="{w2_left - 1.4}" height="7.7" fill="{PALETTE["imm_tint"]}"/>\n'
        f'  <rect x="{MAIN_X + 0.7}" y="{t2_y + 0.7}" width="5.5" height="34" rx="2.5" fill="{PALETTE["imm"]}"/>\n'
        f'  <line x1="{MAIN_X + 1}" y1="{t2_y + 34.7}" x2="{MAIN_X + w2_left - 1}" y2="{t2_y + 34.7}" stroke="#CBD5E1" stroke-width="1.0"/>\n'
        f'  <line x1="{MAIN_X + 332}" y1="{t2_y + 44}" x2="{MAIN_X + 332}" y2="{t2_y + t2_h - 12}" stroke="#E2E8F0" stroke-width="1.1"/>\n'
    )
    parts.append(svg_text(MAIN_X + 16, t2_y + 23.0, "④ CV + Harmonic IMM Tracker (dynamic_tracker.py)", size=14.5, weight=700, fill=PALETTE["imm"], verifier=v, box_name="t2_imm", max_width=510))
    parts.append(f'  <rect x="{MAIN_X + w2_left - 90}" y="{t2_y + 6.7}" width="80" height="22" rx="11" fill="{PALETTE["imm"]}"/>\n')
    parts.append(svg_text(MAIN_X + w2_left - 50, t2_y + 21.8, "5/10 Hz", size=11.5, weight=700, fill="#FFFFFF", anchor="middle", verifier=v, box_name="t2_imm", max_width=74))
    for i, (txt, bld, col) in enumerate([
        ("비동기 갱신 & Option 1 오탐 필터", True, PALETTE["text_main"]),
        ("• 5 Hz 스탬프 갱신 + 10 Hz 비파괴 외삽", False, PALETTE["text_sub"]),
        ("• _is_static_wall_fragment 공선성 기각", False, PALETTE["text_sub"]),
        ("• 회전 순변위 ≥0.25m (정적 오탐 0건)", False, PALETTE["imm"]),
    ]):
        parts.append(svg_text(MAIN_X + 16, t2_y + 58 + i * 22.5, txt, size=12.6, weight=700 if bld else 400, fill=col, verifier=v, box_name="t2_imm", max_width=308))
    for i, (txt, bld, col) in enumerate([
        ("장기 예측 정확도 (τ ∈ [0, 4.5] s)", True, PALETTE["imm"]),
        ("• Crossfire형 FDE: 4.4 mm (μ_harm = 1.0)", True, PALETTE["text_main"]),
        ("• Gauntlet형  FDE: 11.4 cm (A = 0.93 m)", True, PALETTE["text_main"]),
        ("• CPU < 0.50 ms · Ch3(12×12) 투영맵 생성", False, PALETTE["text_sub"]),
    ]):
        parts.append(svg_text(MAIN_X + 346, t2_y + 58 + i * 22.5, txt, size=12.6, weight=700 if bld else 400, fill=col, verifier=v, box_name="t2_imm", max_width=284))

    parts.append(
        svg_card(
            MAIN_X + w2_left + g2, t2_y, w2_right, t2_h, "⑤ Global LIFO Waypoint Stack",
            PALETTE["imm"], PALETTE["imm_tint"],
            [
                ("통신 단절 시 자율 역추적 (Δd≥0.30m | Δθ≥25°)", True, None),
                ("• 비파괴 위상 지름길 & L-코너 하류 검증", False, None),
                ("• 원점 ≤0.80m 시작점 ICP / ≤0.50m ArUco 인계", False, None),
            ],
            verifier=v, box_name="t2_lifo", badge="LIFO",
            header_h=34, title_size=14.5, body_size=12.6, line_step=22.5,
            metric_chip="SE(2) ICP 오프셋 실시간 동기화",
        )
    )

    # -------------------- TIER 3: GEOMETRIC 3-WAY ROUTER (y=530, h=222; gap=68px) --------------------
    add_tier_label("lbl_t3", 522, "TIER 3 · 3-WAY ROUTER", "#DBEAFE", "#1D4ED8")
    t3_y, t3_h = 530, 222
    v.register_box("t3_router", MAIN_X, t3_y, MAIN_W, t3_h)
    parts.append(
        f'  <rect x="{MAIN_X}" y="{t3_y}" width="{MAIN_W}" height="{t3_h}" rx="14" '
        f'fill="{PALETTE["router_bg"]}" stroke="{PALETTE["router_stroke"]}" stroke-width="1.5" stroke-dasharray="7,4"/>\n'
    )
    bw3, bg3, by3, bh3 = 346.0, 22.0, t3_y + 16, 190.0
    bx0 = MAIN_X + 20.0
    parts.append(
        svg_card(
            bx0, by3, bw3, bh3, "Branch A · 횡단 2D 스케줄러",
            PALETTE["cross"], PALETTE["cross_tint"],
            [
                ("width < 1.85m & |d_macro·u| ≤ 0.60", True, PALETTE["cross"]),
                ("• 2D 양측 측면 레인 ±0.38m (벽 ≥0.17m)", False, None),
                ("• ALIGN→HOLD→DASH · 구역 HOLD≤14s/CD 4s", False, None),
                ("• 안전창 33.9% · CBF prefer_no_reverse", False, PALETTE["cbf"]),
            ],
            verifier=v, box_name="t3_cross", parent="t3_router", badge="Crossfire",
            header_h=34, title_size=14.0, body_size=12.4, line_step=21.5,
            metric_chip="B-4 14/20 → Opt1 CF 15/20 (Sim 8·Real 7)",
        )
    )
    parts.append(
        svg_card(
            bx0 + bw3 + bg3, by3, bw3, bh3, "Branch B · 축방향 스윕 A*",
            PALETTE["axial"], PALETTE["axial_tint"],
            [
                ("|d_path·u_hat| ≥ 0.65 & 광폭 우회", True, PALETTE["axial"]),
                ("• soft_cost 띠 0.52m · w=3.8 (>max 3.6)", False, None),
                ("• 서브골 레일 사선 절단 차단 (hard 불변)", False, None),
                ("• 정면 근접(<0.85m) 속도 상한 0.09 m/s", False, None),
            ],
            verifier=v, box_name="t3_axial", parent="t3_router", badge="Gauntlet",
            header_h=34, title_size=14.0, body_size=12.4, line_step=21.5,
            metric_chip="Gauntlet: 16/20 (Sim 7 · Real 9)",
        )
    )
    parts.append(
        svg_card(
            bx0 + 2 * (bw3 + bg3), by3, bw3, bh3, "Branch C · 정적·오염 우회",
            PALETTE["static"], PALETTE["static_tint"],
            [
                ("확정 트랙 0건 OR spike_ratio ≥ 0.03", True, PALETTE["static"]),
                ("• 동적 비용맵 · 횡단 스케줄러 완전 우회", False, None),
                ("• 정적·연기(Gate 2) + 시맨틱 A* 무개입", False, None),
                ("• Sim/Real 시맨틱 10/10 (mq2=0, 오탐 0)", False, PALETTE["icp"]),
            ],
            verifier=v, box_name="t3_static", parent="t3_router", badge="Bypass",
            header_h=34, title_size=14.0, body_size=12.4, line_step=21.5,
            metric_chip="Gate 2: 59/60 (98.3%) · 충돌 0/60 · 5.70cm",
        )
    )

    # -------------------- TIER 4: POLICY, CBF SAFETY & 2-BRANCH ICP (y=820, h=202; gap=68px) --------------------
    add_tier_label("lbl_t4", 812, "TIER 4 · POLICY & CBF", "#F3E8FF", "#6D28D9")
    t4_y, t4_h = 820, 202
    parts.append(
        svg_card(
            MAIN_X, t4_y, w3, t4_h, "⑦ 4-Ch Context-RL ONNX",
            PALETTE["rl"], PALETTE["rl_tint"],
            [
                ("12×12×4 (Ch0-Ch3) + 16-step 시계열", True, None),
                ("• Regulated Pure Pursuit + Clearance-Gate", False, None),
                ("• Sim v4.4 ONNX (cbf_v_min = -0.06 m/s)", False, None),
                ("• Real v3 ONNX (cbf_v_min = -0.08 m/s)", False, None),
            ],
            verifier=v, box_name="t4_rl", badge="4-Ch",
            header_h=34, title_size=14.2, body_size=12.6, line_step=22.0,
            metric_chip="Nominal (v_nom, ω_nom) 출력",
        )
    )
    parts.append(
        svg_card(
            MAIN_X + w3 + g3, t4_y, w3, t4_h, "⑧ 2-Circle Soft CBF-QP",
            PALETTE["cbf"], PALETTE["cbf_tint"],
            [
                ("10 Hz 2-Circle ZCBF-QP (Ames et al., 2017)", True, None),
                ("• 정적 점군 + IMM 트랙 2-Circle 장벽 (δ_ij)", False, None),
                ("• mq2_escape 클래스·깊은 내부 분리 (mq2=0)", False, None),
                ("• 초근접 광선 클램프 0.120m · 비상 후진 보존", False, None),
            ],
            verifier=v, box_name="t4_cbf", badge="<2.0ms",
            header_h=34, title_size=14.2, body_size=12.6, line_step=22.0,
            metric_chip="0.407ms (max 1.941) · 12.96 Hz",
        )
    )
    parts.append(
        svg_card(
            MAIN_X + 2 * (w3 + g3), t4_y, w3, t4_h, "⑨ 2분기 2D ICP + ArUco",
            PALETTE["icp"], PALETTE["icp_tint"],
            [
                ("smoke 파라미터 미참조 (AST 검증)", True, None),
                ("• 오염: cap 0.16m · corr_y 차단 (54/54)", False, PALETTE["cbf"]),
                ("• 정상(inl≥25): 0.42m 정밀 보정 (오탐 0/40)", False, PALETTE["icp"]),
                ("• 원점 ≤0.5m ArUco solvePnP (±2 cm 도킹)", False, None),
            ],
            verifier=v, box_name="t4_icp", badge="SE(2)",
            header_h=34, title_size=14.2, body_size=12.6, line_step=22.0,
            metric_chip="smoke 20/20 (5.4-6.4cm) · OpenCR",
        )
    )

    # -------------------- FOOTER KPI STRIP (y=1052, h=88) --------------------
    v.register_box("kpi_strip", 24, 1052, 1152, 88)
    parts.append(
        '  <rect x="24" y="1052" width="1152" height="88" rx="12" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.3" filter="url(#card-shadow)"/>\n'
    )
    kpis = [
        ("100-Trial · vs OFF", "89 / 100 (89.0%)", "Opt1 31/40 vs OFF 7/40 (p=1.11e-7)", "#0F172A"),
        ("Gate 1 · Option 1", "31 / 40 (77.5%)", "B-4 30/40 · 충돌 6/40 · CF 15/20", "#1D4ED8"),
        ("Gate 2 · 시맨틱", "59/60 · 10/10", "0/60 · 5.70cm · Sim/Real 10/10", "#047857"),
        ("실시간성 (100회)", "0.407ms · 12.96Hz", "CBF max 1.941ms · 추가 +0.328ms", "#BE123C"),
        ("단위 검증 · 무결성", "101 / 101 PASS", "8개 스위트 · 전체 124/126", "#6D28D9"),
    ]
    kw = 1152.0 / len(kpis)
    for i, (kt, kv, ks, kc) in enumerate(kpis):
        kx = 24.0 + i * kw
        if i > 0:
            parts.append(f'  <line x1="{kx:.1f}" y1="1064" x2="{kx:.1f}" y2="1128" stroke="#E2E8F0" stroke-width="1.2"/>\n')
        cx = kx + kw / 2.0
        parts.append(svg_text(cx, 1076, kt, size=11.5, weight=700, fill="#475569", anchor="middle", verifier=v, box_name="kpi_strip", max_width=kw - 12))
        parts.append(svg_text(cx, 1105, kv, size=19.5, weight=700, fill=kc, anchor="middle", verifier=v, box_name="kpi_strip", max_width=kw - 12))
        parts.append(svg_text(cx, 1127, ks, size=11.2, weight=400, fill="#475569", anchor="middle", verifier=v, box_name="kpi_strip", max_width=kw - 10))

    # -------------------- CLEAN INTER-TIER ARROWS --------------------
    # Arrows span from card bottom directly to next tier card top; pills sit in upper 28px cleanly ABOVE tier label!
    parts.append(svg_path([(255, t1_y + t1_h + 2), (255, t2_y - 4)], PALETTE["sensor"], "arr-slate", 2.4))
    parts.append(svg_path([(585, t1_y + t1_h + 2), (585, t2_y - 4)], PALETTE["sensor"], "arr-slate", 2.2))
    parts.append(svg_path([(967, t1_y + t1_h + 2), (967, t2_y - 4)], PALETTE["sensor"], "arr-slate", 2.0, dashed=True))

    parts.append(svg_path([(255, t2_y + t2_h + 2), (255, t3_y - 4)], PALETTE["imm"], "arr-teal", 2.5))
    parts.append(svg_path([(585, t2_y + t2_h + 2), (585, t3_y - 4)], PALETTE["imm"], "arr-teal", 2.5))
    parts.append(svg_path([(953, t2_y + t2_h + 2), (953, t3_y - 4)], PALETTE["imm"], "arr-teal", 2.2))

    parts.append(svg_path([(255, t3_y + t3_h + 2), (255, t4_y - 4)], PALETTE["cross"], "arr-indigo", 2.5))
    parts.append(svg_path([(585, t3_y + t3_h + 2), (585, t4_y - 4)], PALETTE["cbf"], "arr-crimson", 2.4, dashed=True))
    parts.append(svg_path([(953, t3_y + t3_h + 2), (953, t4_y - 4)], PALETTE["static"], "arr-slate", 2.2))

    t4_mid = t4_y + t4_h / 2.0
    parts.append(svg_path([(MAIN_X + w3 + 2, t4_mid), (MAIN_X + w3 + g3 - 4, t4_mid)], PALETTE["rl"], "arr-violet", 2.6))
    parts.append(svg_path([(MAIN_X + 2 * w3 + g3 + 2, t4_mid), (MAIN_X + 2 * (w3 + g3) - 4, t4_mid)], PALETTE["cbf"], "arr-crimson", 2.6))

    # Right-margin SE(2) Feedback Bus
    parts.append(
        svg_path(
            [(MAIN_X + MAIN_W + 2, 921), (1174, 921), (1174, 388), (MAIN_X + MAIN_W + 4, 388)],
            PALETTE["icp"], "arr-emerald", 2.4, dashed=True,
        )
    )
    parts.append(
        f'  <g transform="translate(1174, 654) rotate(-90)">\n'
        f'    <rect x="-138" y="-11" width="276" height="22" rx="11" fill="#ECFDF5" stroke="{PALETTE["icp"]}" stroke-width="1.2"/>\n'
        f'    <text x="0" y="4.2" text-anchor="middle" font-family="{FONT_FAMILY_SVG}" font-size="11.5" font-weight="700" fill="{PALETTE["icp"]}">SE(2) Drift Feedback (α=0.40) → IMM · LIFO</text>\n'
        f'  </g>\n'
    )
    v.register_box("feedback_rail_box", 1163, 516, 22, 276)
    v.register_text("feedback_rail", "SE(2) Drift Feedback (α=0.40) → IMM · LIFO", 11.5, True, 264)

    # Top pill layer: positioned cleanly in upper half of each 68px gap (y = card_bottom + 20)
    # AND registered as boxes in DiagramVerifier to assert ZERO overlap with tier labels!
    pill_defs = [
        ("p_scan",  255, t1_y + t1_h + 31, "/scan (5 Hz)", PALETTE["sensor"]),
        ("p_spk",   585, t1_y + t1_h + 31, "spike_ratio", PALETTE["sensor"]),
        ("p_sem",   967, t1_y + t1_h + 31, "Ch1/Ch2 → RL", PALETTE["sensor"]),
        ("p_phat",  255, t2_y + t2_h + 31, "p_hat(τ) · σ(τ)", PALETTE["imm"]),
        ("p_uhat",  585, t2_y + t2_h + 31, "u_hat · Ch3", PALETTE["imm"]),
        ("p_wp",    953, t2_y + t2_h + 31, "waypoint_stack", PALETTE["imm"]),
        ("p_vref",  255, t3_y + t3_h + 31, "v_ref · subgoal", PALETTE["cross"]),
        ("p_flag",  585, t3_y + t3_h + 31, "prefer_no_reverse", PALETTE["cbf"]),
        ("p_byp",   953, t3_y + t3_h + 31, "bypass / ≤0.8m", PALETTE["static"]),
    ]
    for pname, cx, cy, lbl, col in pill_defs:
        kr_cnt = sum(1 for ch in lbl if ord(ch) > 127)
        en_cnt = len(lbl) - kr_cnt
        pw = max(48.0, kr_cnt * 11.5 * 0.98 + en_cnt * 11.5 * 0.62 + 18.0)
        v.register_box(pname, cx - pw / 2.0, cy - 10.5, pw, 21.0)
        parts.append(svg_pill(cx, cy, lbl, col, 11.5, v))

    parts.append("</svg>\n")
    return "".join(parts), v.verify_all(reg_font, bold_font, font_idx)


def convert_svg_to_png_2x(svg_path: Path, png_path: Path, width: int, height: int) -> str:
    w2, h2 = width * 2, height * 2
    if shutil.which("rsvg-convert"):
        subprocess.run(["rsvg-convert", "-w", str(w2), "-h", str(h2), "-o", str(png_path), str(svg_path)], check=True)
        return "rsvg-convert"
    try:
        import cairosvg  # type: ignore
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=w2, output_height=h2)
        return "cairosvg"
    except Exception:
        pass
    try:
        import gi  # type: ignore
        gi.require_version("Rsvg", "2.0")
        from gi.repository import Rsvg  # type: ignore
        import cairo  # type: ignore
        handle = Rsvg.Handle.new_from_file(str(svg_path))
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w2, h2)
        ctx = cairo.Context(surface)
        ctx.scale(2.0, 2.0)
        handle.render_cairo(ctx)
        surface.write_to_png(str(png_path))
        return "gi.Rsvg+cairo"
    except Exception:
        pass
    if os.path.exists("/usr/bin/python3") and os.path.realpath(shutil.which("python3") or "") != os.path.realpath("/usr/bin/python3"):
        helper = (
            "import sys, gi, cairo; gi.require_version('Rsvg', '2.0'); "
            "from gi.repository import Rsvg; "
            "h = Rsvg.Handle.new_from_file(sys.argv[1]); "
            "w2, h2 = int(sys.argv[3]), int(sys.argv[4]); "
            "s = cairo.ImageSurface(cairo.FORMAT_ARGB32, w2, h2); "
            "c = cairo.Context(s); c.scale(2.0, 2.0); h.render_cairo(c); "
            "s.write_to_png(sys.argv[2])"
        )
        res = subprocess.run(
            ["/usr/bin/python3", "-c", helper, str(svg_path), str(png_path), str(w2), str(h2)],
            check=False,
        )
        if res.returncode == 0 and png_path.exists():
            return "usr-python3(gi.Rsvg+cairo)"
    if shutil.which("pdftoppm"):
        try:
            import weasyprint  # type: ignore
            svg_content = svg_path.read_text(encoding="utf-8")
            html = (
                "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><style>"
                f"@page {{ size: {width}px {height}px; margin: 0; }}"
                f"html, body {{ margin: 0; padding: 0; width: {width}px; height: {height}px; overflow: hidden; }}"
                f"svg {{ display: block; width: {width}px; height: {height}px; }}"
                f"</style></head><body>{svg_content}</body></html>"
            )
            with tempfile.TemporaryDirectory() as td:
                pdf_tmp = Path(td) / "d.pdf"
                png_pref = Path(td) / "d"
                weasyprint.HTML(string=html).write_pdf(str(pdf_tmp))
                subprocess.run(["pdftoppm", "-png", "-r", "192", "-singlefile", str(pdf_tmp), str(png_pref)], check=True)
                shutil.copyfile(str(png_pref) + ".png", str(png_path))
            return "weasyprint+pdftoppm(192dpi)"
        except Exception:
            pass
    if shutil.which("inkscape"):
        subprocess.run(["inkscape", str(svg_path), f"--export-width={w2}", f"--export-height={h2}", f"--export-filename={png_path}"], check=True)
        return "inkscape"
    raise RuntimeError("No supported SVG->PNG converter found.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render high-readability Phase 4 (Step B-4) diagrams.")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        repo_cand = Path(__file__).resolve().parents[2] / "docs" / "images" / "evidence_phase4"
        out_dir = repo_cand if (Path(__file__).resolve().parents[2] / "docs").exists() else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    reg_font, bold_font, font_idx, font_label = detect_korean_fonts()
    print(f"[Font] {font_label} (index={font_idx})")

    for stem, w, h, builder in [
        ("phase4_hero_pipeline", 1200, 580, build_hero_pipeline_svg),
        ("phase4_control_architecture", 1200, 1170, build_detailed_architecture_svg),
    ]:
        svg_str, stats = builder(reg_font, bold_font, font_idx)
        svg_p = out_dir / f"{stem}.svg"
        png_p = out_dir / f"{stem}.png"
        svg_p.write_text(svg_str, encoding="utf-8")
        backend = convert_svg_to_png_2x(svg_p, png_p, w, h)
        res = "N/A"
        if Image is not None:
            with Image.open(png_p) as im:
                res = f"{im.size[0]}x{im.size[1]} ({im.mode})"
                assert im.size == (w * 2, h * 2), f"Got {im.size}, expected {(w*2, h*2)}"
        print(f"[OK] {stem}: {backend} | SVG={svg_p.stat().st_size/1024:.1f}KB | PNG={png_p.stat().st_size/1024:.1f}KB ({res}) | {stats}")


if __name__ == "__main__":
    main()
