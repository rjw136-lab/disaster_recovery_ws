#!/usr/bin/env python3
"""
Extract accurate 2D robot-height (z ~ 0.04..0.35m) map obstacles & walls from
Gazebo .world, included SDF models, COLLADA (.dae), and Wavefront (.obj) files.

Used by plot_ab_comparison.py to render:
  1. Top-down layout maps of every disaster environment (ab_environment_maps.png).
  2. Structural wall/debris overlays underneath paired trajectory plots.
"""

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np


DEFAULT_MODELS_ROOT = "/home/lab/.gazebo/models"
SKIP_MODEL_SUBSTRINGS = (
    "ground_plane",
    "asphalt_plane",
    "goal_marker",
    "spawn_marker",
    "sun",
)


@dataclass
class WorldFootprint:
    """2D geometry slices extracted at robot height for one Gazebo world."""

    polygons: List[Tuple[str, np.ndarray]] = field(default_factory=list)
    segments: List[Tuple[str, np.ndarray]] = field(default_factory=list)
    dynamic_markers: List[Tuple[str, float, float, float]] = field(default_factory=list)


def _parse_floats(text: Optional[str], default: Tuple[float, ...]) -> Tuple[float, ...]:
    if not text or not text.strip():
        return default
    vals = [float(x) for x in text.strip().split()]
    return tuple(vals) if len(vals) >= len(default) else default


def _pose_matrix(pose_text: Optional[str]) -> np.ndarray:
    x, y, z, r, p, yw = _parse_floats(pose_text, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))[:6]
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(yw), math.sin(yw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    m = np.eye(4)
    m[:3, :3] = rz @ ry @ rx
    m[:3, 3] = [x, y, z]
    return m


def _convex_hull_2d(pts: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain 2D convex hull (pure NumPy, no scipy needed)."""
    if len(pts) < 3:
        return pts
    uniq = np.unique(np.round(pts, 4), axis=0)
    if len(uniq) < 3:
        return uniq
    order = np.lexsort((uniq[:, 1], uniq[:, 0]))
    sorted_pts = uniq[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[np.ndarray] = []
    for p in sorted_pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-9:
            lower.pop()
        lower.append(p)
    upper: List[np.ndarray] = []
    for p in sorted_pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-9:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _slice_triangles_xy(tris_world: np.ndarray, z_slice: float) -> List[np.ndarray]:
    """Return list of (2, 2) line segments where 3D triangles cross z=z_slice."""
    segs = []
    for tri in tris_world:
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            za, zb = tri[a, 2], tri[b, 2]
            if (za - z_slice) * (zb - z_slice) <= 0.0 and abs(za - zb) > 1e-6:
                t = (z_slice - za) / (zb - za)
                if -1e-6 <= t <= 1.0 + 1e-6:
                    t = min(1.0, max(0.0, t))
                    pts.append(tri[a, :2] + t * (tri[b, :2] - tri[a, :2]))
        if len(pts) >= 2 and np.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1]) > 1e-4:
            segs.append(np.array([pts[0], pts[1]]))
    return segs


def _box_triangles(sx: float, sy: float, sz: float) -> np.ndarray:
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    v = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz],  [hx, -hy, hz],  [hx, hy, hz],  [-hx, hy, hz],
    ])
    idx = [
        (0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    ]
    return v[np.array(idx)]


def _cylinder_triangles(radius: float, length: float, sides: int = 16) -> np.ndarray:
    hz = length * 0.5
    ang = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    bot = np.column_stack([radius * np.cos(ang), radius * np.sin(ang), np.full(sides, -hz)])
    top = np.column_stack([radius * np.cos(ang), radius * np.sin(ang), np.full(sides, hz)])
    tris = []
    for i in range(sides):
        j = (i + 1) % sides
        tris.append([bot[i], bot[j], top[j]])
        tris.append([bot[i], top[j], top[i]])
        tris.append([[0.0, 0.0, -hz], bot[j], bot[i]])
        tris.append([[0.0, 0.0, hz], top[i], top[j]])
    return np.array(tris)


def _transform_tris(tris: np.ndarray, mat: np.ndarray, scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    flat = tris.reshape(-1, 3) * np.array(scale)
    hom = np.hstack([flat, np.ones((len(flat), 1))])
    out = (mat @ hom.T).T[:, :3]
    return out.reshape(-1, 3, 3)


def _footprint_from_tris(tris_world: np.ndarray, z_pref: float = 0.20) -> Optional[np.ndarray]:
    zs = tris_world[:, :, 2]
    z_min, z_max = float(zs.min()), float(zs.max())
    if z_max < 0.04 or z_min > 0.35:
        return None
    z_cut = min(max(z_pref, z_min + 0.01), z_max - 0.01) if z_max - z_min > 0.025 else 0.5 * (z_min + z_max)
    segs = _slice_triangles_xy(tris_world, z_cut)
    if not segs:
        # Low slab slightly touching band: project full XY convex hull
        pts = tris_world.reshape(-1, 3)[:, :2]
        return _convex_hull_2d(pts)
    pts = np.vstack(segs)
    return _convex_hull_2d(pts)


@lru_cache(maxsize=32)
def _load_dae_local_tris(dae_path: str) -> Tuple[np.ndarray, float, str]:
    """Parse COLLADA vertices and scene-node transforms into model-local triangles (meters)."""
    root = ET.parse(dae_path).getroot()
    ns = "{" + root.tag.split("}")[0].strip("{") + "}" if "}" in root.tag else ""
    unit_el = root.find(f".//{ns}unit")
    unit_m = float(unit_el.get("meter", "1.0")) if unit_el is not None else 1.0
    up_el = root.find(f".//{ns}up_axis")
    up_axis = up_el.text.strip() if up_el is not None and up_el.text else "Z_UP"

    geoms: Dict[str, np.ndarray] = {}
    for geom in root.findall(f".//{ns}geometry"):
        gid = "#" + geom.get("id", "")
        mesh = geom.find(f"{ns}mesh")
        if mesh is None:
            continue
        sources: Dict[str, np.ndarray] = {}
        for src in mesh.findall(f"{ns}source"):
            acc = src.find(f".//{ns}accessor")
            stride = int(acc.get("stride", "3")) if acc is not None else 3
            fa = src.find(f"{ns}float_array")
            if fa is not None and fa.text and stride == 3:
                arr = np.fromstring(fa.text, sep=" ")
                if len(arr) % 3 == 0:
                    sources["#" + src.get("id", "")] = arr.reshape(-1, 3)
        verts_el = mesh.find(f"{ns}vertices")
        if verts_el is not None:
            for inp in verts_el.findall(f"{ns}input"):
                if inp.get("semantic") == "POSITION":
                    sources["#" + verts_el.get("id", "")] = sources.get(inp.get("source", ""))

        tris_all = []
        for prim_tag in ("triangles", "polylist"):
            for prim in mesh.findall(f"{ns}{prim_tag}"):
                inputs = prim.findall(f"{ns}input")
                in_stride = max([int(i.get("offset", 0)) for i in inputs], default=0) + 1
                v_off, pts_arr = None, None
                for inp in inputs:
                    if inp.get("semantic") == "VERTEX":
                        v_off = int(inp.get("offset", 0))
                        pts_arr = sources.get(inp.get("source", ""))
                p_el = prim.find(f"{ns}p")
                if p_el is None or pts_arr is None or v_off is None or not p_el.text:
                    continue
                idx = np.fromstring(p_el.text, sep=" ", dtype=int)
                v_idx = idx[v_off::in_stride]
                if prim_tag == "triangles":
                    for k in range(0, len(v_idx) - 2, 3):
                        tris_all.append(pts_arr[[v_idx[k], v_idx[k + 1], v_idx[k + 2]]])
                else:
                    vc_el = prim.find(f"{ns}vcount")
                    vcounts = np.fromstring(vc_el.text, sep=" ", dtype=int) if vc_el is not None and vc_el.text else []
                    pos = 0
                    for c in vcounts:
                        poly = v_idx[pos:pos + c]
                        pos += c
                        for k in range(1, c - 1):
                            tris_all.append(pts_arr[[poly[0], poly[k], poly[k + 1]]])
        if tris_all:
            geoms[gid] = np.array(tris_all, dtype=float)

    nodes_by_id = {"#" + n.get("id", ""): n for n in root.findall(f".//{ns}node") if n.get("id")}
    collected: List[np.ndarray] = []

    def walk(node: ET.Element, m_acc: np.ndarray) -> None:
        m_loc = np.eye(4)
        for ch in node:
            tag = ch.tag.split("}")[-1]
            if tag == "matrix" and ch.text:
                m_loc = m_loc @ np.fromstring(ch.text, sep=" ").reshape(4, 4)
            elif tag == "translate" and ch.text:
                t = np.eye(4)
                t[:3, 3] = np.fromstring(ch.text, sep=" ")[:3]
                m_loc = m_loc @ t
            elif tag == "scale" and ch.text:
                s = np.eye(4)
                vals = np.fromstring(ch.text, sep=" ")[:3]
                s[0, 0], s[1, 1], s[2, 2] = vals
                m_loc = m_loc @ s
        m_cur = m_acc @ m_loc
        for ig in node.findall(f"{ns}instance_geometry"):
            url = ig.get("url", "")
            if url in geoms:
                collected.append(_transform_tris(geoms[url], m_cur))
        for ino in node.findall(f"{ns}instance_node"):
            url = ino.get("url", "")
            if url in nodes_by_id:
                walk(nodes_by_id[url], m_cur)
        for child in node.findall(f"{ns}node"):
            walk(child, m_cur)

    vs = root.find(f".//{ns}visual_scene")
    if vs is not None:
        for top in vs.findall(f"{ns}node"):
            walk(top, np.eye(4))

    if not collected:
        return np.zeros((0, 3, 3), dtype=float), unit_m, up_axis
    merged = np.vstack(collected) * unit_m
    if up_axis == "Y_UP":
        merged = merged[:, :, [0, 2, 1]] * np.array([1.0, -1.0, 1.0])
    return merged, unit_m, up_axis


@lru_cache(maxsize=16)
def _load_obj_local_tris(obj_path: str) -> np.ndarray:
    verts: List[List[float]] = []
    tris: List[List[List[float]]] = []
    with open(obj_path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()[1:4]
                verts.append([float(x) for x in parts])
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    v_i = int(tok.split("/")[0])
                    idx.append(v_i - 1 if v_i > 0 else len(verts) + v_i)
                for k in range(1, len(idx) - 1):
                    tris.append([verts[idx[0]], verts[idx[k]], verts[idx[k + 1]]])
    return np.array(tris, dtype=float) if tris else np.zeros((0, 3, 3), dtype=float)


def _resolve_model_uri(uri: str, models_root: str) -> Optional[str]:
    if not uri or not uri.startswith("model://"):
        return None
    rel = uri[len("model://"):]
    cand = os.path.join(models_root, rel)
    return cand if os.path.exists(cand) else None


def _process_geometry_elem(
    geom_el: Optional[ET.Element],
    mat_world: np.ndarray,
    label: str,
    models_root: str,
    fp: WorldFootprint,
    z_slice: float,
) -> None:
    if geom_el is None:
        return

    box_el = geom_el.find("box")
    if box_el is not None:
        sx, sy, sz = _parse_floats(box_el.findtext("size"), (0.5, 0.5, 0.5))[:3]
        if max(sx, sy) >= 15.0 and min(sx, sy) >= 15.0:
            return
        tris = _transform_tris(_box_triangles(sx, sy, sz), mat_world)
        poly = _footprint_from_tris(tris, z_slice)
        if poly is not None and len(poly) >= 3:
            fp.polygons.append((label, poly))
        return

    cyl_el = geom_el.find("cylinder")
    if cyl_el is not None:
        r = float(cyl_el.findtext("radius", "0.2"))
        l = float(cyl_el.findtext("length", "1.0"))
        tris = _transform_tris(_cylinder_triangles(r, l), mat_world)
        poly = _footprint_from_tris(tris, z_slice)
        if poly is not None and len(poly) >= 3:
            fp.polygons.append((label, poly))
        return

    mesh_el = geom_el.find("mesh")
    if mesh_el is not None:
        uri = mesh_el.findtext("uri", "")
        mesh_path = _resolve_model_uri(uri, models_root)
        if mesh_path is None:
            return
        scale = _parse_floats(mesh_el.findtext("scale"), (1.0, 1.0, 1.0))[:3]
        if mesh_path.lower().endswith(".dae"):
            local_tris, _, _ = _load_dae_local_tris(mesh_path)
        elif mesh_path.lower().endswith(".obj"):
            local_tris = _load_obj_local_tris(mesh_path)
        else:
            return
        if len(local_tris) == 0:
            return
        tris_w = _transform_tris(local_tris, mat_world, scale)
        segs = _slice_triangles_xy(tris_w, z_slice)
        for seg in segs:
            fp.segments.append((label, seg))


def _process_model_elem(
    model_el: ET.Element,
    parent_mat: np.ndarray,
    models_root: str,
    fp: WorldFootprint,
    z_slice: float,
    depth: int = 0,
) -> None:
    if depth > 4:
        return
    name = model_el.get("name", "")
    low_name = name.lower()
    if any(tok in low_name for tok in SKIP_MODEL_SUBSTRINGS):
        return

    m_model = parent_mat @ _pose_matrix(model_el.findtext("pose"))

    # Dynamic obstacle marker detection (maze)
    if "obstacle" in low_name and (model_el.find("plugin") is not None or "dqn" in low_name):
        inc = model_el.find("include")
        inc_pose = _pose_matrix(inc.findtext("pose")) if inc is not None else np.eye(4)
        m_dyn = m_model @ inc_pose
        fp.dynamic_markers.append((name, float(m_dyn[0, 3]), float(m_dyn[1, 3]), 0.12))
        return

    # Recursive nested <include> inside <model> (e.g., turtlebot3_house, turtlebot3_dqn_world)
    for inc in model_el.findall("include"):
        uri = inc.findtext("uri", "")
        model_dir = _resolve_model_uri(uri, models_root)
        if model_dir and os.path.isdir(model_dir):
            sdf_path = os.path.join(model_dir, "model.sdf")
            if os.path.exists(sdf_path):
                m_inc = m_model @ _pose_matrix(inc.findtext("pose"))
                sub_root = ET.parse(sdf_path).getroot()
                sub_model = sub_root.find("model")
                if sub_model is not None:
                    _process_model_elem(sub_model, m_inc, models_root, fp, z_slice, depth + 1)

    for link in model_el.findall("link"):
        m_link = m_model @ _pose_matrix(link.findtext("pose"))
        lbl = f"{name}/{link.get('name', '')}".strip("/")
        cols = link.findall("collision")
        # Prefer collision geometry; fall back to visual if collision omitted
        geoms = cols if cols else link.findall("visual")
        for col in geoms:
            m_col = m_link @ _pose_matrix(col.findtext("pose"))
            _process_geometry_elem(col.find("geometry"), m_col, lbl, models_root, fp, z_slice)


@lru_cache(maxsize=8)
def extract_world_footprint(
    world_file: str,
    models_root: str = DEFAULT_MODELS_ROOT,
    z_slice: float = 0.17,
) -> WorldFootprint:
    """Parse all static structural obstacles and dynamic markers in a Gazebo .world."""
    fp = WorldFootprint()
    if not os.path.exists(world_file):
        return fp
    root = ET.parse(world_file).getroot()
    world = root.find("world")
    if world is None:
        return fp

    for inc in world.findall("include"):
        uri = inc.findtext("uri", "")
        model_dir = _resolve_model_uri(uri, models_root)
        if model_dir and os.path.isdir(model_dir):
            sdf_path = os.path.join(model_dir, "model.sdf")
            if os.path.exists(sdf_path):
                m_inc = _pose_matrix(inc.findtext("pose"))
                sub_root = ET.parse(sdf_path).getroot()
                sub_model = sub_root.find("model")
                if sub_model is not None:
                    _process_model_elem(sub_model, m_inc, models_root, fp, z_slice, 1)

    for model_el in world.findall("model"):
        _process_model_elem(model_el, np.eye(4), models_root, fp, z_slice, 0)

    return fp
