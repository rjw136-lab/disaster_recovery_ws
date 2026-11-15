#!/usr/bin/env python3
"""Generate the 15-Scenario 2D Structural Map & 4-Stage Return Route Atlas."""

import json
import os
import sys
import shutil
import urllib.request
from typing import Dict, List, Tuple

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

from world_map_geometry import extract_world_footprint  # noqa: E402

WS_WORLDS = os.path.join(WS_ROOT, "src", "disaster_recovery", "worlds")
BM_ROOT = os.path.join(WS_ROOT, "benchmark_results")
HDD_ROOT = "/hdd/rl_v2_exp" if os.path.exists("/hdd/rl_v2_exp") else os.path.expanduser("~/hdd/rl_v2_exp")


# Required Gazebo models for the House and Maze maps. Keep local installations
# when available; otherwise cache the official SDF files inside the workspace.
MODEL_CACHE_ROOT = os.path.join(WS_ROOT, ".cache_atlas_models")
REFERENCE_MODEL_FILES = {
    "turtlebot3_house/model.sdf": (
        "https://raw.githubusercontent.com/ROBOTIS-GIT/turtlebot3_simulations/main/"
        "turtlebot3_gazebo/models/turtlebot3_house/model.sdf"
    ),
    "turtlebot3_dqn_world/model.sdf": (
        "https://raw.githubusercontent.com/ROBOTIS-GIT/turtlebot3_simulations/main/"
        "turtlebot3_gazebo/models/turtlebot3_dqn_world/model.sdf"
    ),
    "turtlebot3_dqn_world/inner_walls/model.sdf": (
        "https://raw.githubusercontent.com/ROBOTIS-GIT/turtlebot3_simulations/main/"
        "turtlebot3_gazebo/models/turtlebot3_dqn_world/inner_walls/model.sdf"
    ),
}


def ensure_reference_model_cache(ws_root: str) -> str:
    """Resolve House/Maze collision SDFs explicitly instead of accepting blank maps."""
    roots = []
    raw_roots = [
        os.environ.get("GAZEBO_MODEL_PATH", ""),
        os.environ.get("GAZEBO_MODELS_DIR", ""),
        os.path.expanduser("~/.gazebo/models"),
        "/opt/ros/humble/share/turtlebot3_gazebo/models",
        "/opt/ros/jazzy/share/turtlebot3_gazebo/models",
        "/usr/share/gazebo-11/models",
        os.path.join(ws_root, "models"),
        os.path.join(ws_root, "src", "disaster_recovery", "models"),
    ]
    for raw in raw_roots:
        for item in raw.split(os.pathsep):
            item = item.strip()
            if not item:
                continue
            root = os.path.abspath(os.path.expanduser(item))
            if os.path.isdir(root) and root not in roots:
                roots.append(root)

    os.makedirs(MODEL_CACHE_ROOT, exist_ok=True)
    for rel_path, url in REFERENCE_MODEL_FILES.items():
        target = os.path.join(MODEL_CACHE_ROOT, rel_path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.isfile(target) and os.path.getsize(target) > 500:
            continue

        source = None
        for root in roots:
            candidate = os.path.join(root, rel_path)
            if os.path.isfile(candidate) and os.path.getsize(candidate) > 500:
                source = candidate
                break

        if source:
            shutil.copy2(source, target)
            print("[MODEL] Reused local SDF: " + source)
        else:
            print("[MODEL] Downloading missing SDF: " + rel_path)
            try:
                urllib.request.urlretrieve(url, target)
            except Exception as exc:
                raise RuntimeError(
                    "Cannot obtain " + rel_path
                    + ". Install the official turtlebot3_simulations Gazebo models "
                    "or allow HTTPS access."
                ) from exc
            if not os.path.isfile(target) or os.path.getsize(target) < 500:
                raise RuntimeError("Downloaded SDF is empty or invalid: " + target)

    # Cache first for deterministic House/Maze resolution; retain host model roots
    # for the other environments (vehicles, debris, and dynamic obstacles).
    return os.pathsep.join([MODEL_CACHE_ROOT] + roots)



# (scenario_key, title_label, trial_1idx, world_filename, roi)
PANELS: List[Tuple[str, str, int, str, Tuple[float, float, float, float]]] = [
    ("collapsed", "#1 Collapsed (Basic) (T#03)", 3, "disaster_collapsed_building.world", (-4.5, 12.5, -5.5, 13.5)),
    ("collapsed_heavy", "#2 Collapsed Heavy (T#02)", 2, "disaster_collapsed_building.world", (-4.5, 12.5, -5.5, 13.5)),
    ("tunnel", "#3 Tunnel (Basic) (T#05)", 5, "disaster_tunnel_world.world", (-5.5, 5.5, -4.2, 18.0)),
    ("house", "#4 House (Indoor) (T#04)", 4, "disaster_house_world.world", (-7.6, 7.6, -5.6, 5.4)),
    ("maze", "#5 Maze (Dynamic) (T#07)", 7, "disaster_maze_world.world", (-2.75, 2.75, -2.75, 2.75)),
    ("collapsed_hard", "#6 Collapsed HARD (T#01)", 1, "disaster_collapsed_building.world", (-4.5, 12.5, -5.5, 13.5)),
    ("collapsed_heavy_hard", "#7 Collapsed EXTREME (T#01)", 1, "disaster_collapsed_building.world", (-4.5, 12.5, -5.5, 13.5)),
    ("tunnel_hard", "#8 Tunnel HARD (T#01)", 1, "disaster_tunnel_world.world", (-5.5, 5.5, -4.2, 19.8)),
    ("house_hard", "#9 House HARD (T#08)", 8, "disaster_house_world.world", (-7.6, 7.6, -5.6, 5.4)),
    ("maze_hard", "#10 Maze HARD (T#01)", 1, "disaster_maze_world.world", (-2.75, 2.75, -2.75, 2.75)),
    ("u_pocket_trap", "#11 U-Pocket Trap (T#10)", 10, "disaster_u_pocket_world.world", (-1.3, 11.4, -3.5, 3.5)),
    ("slip_chicane", "#12 Slip Chicane (mu=0.32) (T#01)", 1, "disaster_slip_chicane_world.world", (-1.7, 11.4, -2.6, 2.6)),
    ("dynamic_gauntlet", "#13 Dynamic Gauntlet (T#01)", 1, "disaster_dynamic_gauntlet_world.world", (-4.2, 4.2, -4.2, 4.2)),
    ("dynamic_crossfire_alley", "#14 Dynamic Crossfire Alley (T#01)", 1, "disaster_dynamic_crossfire_world.world", (-1.7, 11.4, -2.6, 2.6)),
    ("smoke_lidar_corruption", "#15 Smoke LiDAR Hazard (T#01)", 1, "disaster_smoke_hazard_world.world", (-1.7, 11.4, -2.6, 2.6)),
]


def _load_stage_trials(sc: str) -> Dict[str, dict]:
    """Load the 4 benchmark stage JSON records (baseline, p1, p2a, p2b) for scenario sc."""
    basic5 = {"collapsed", "collapsed_heavy", "tunnel", "house", "maze"}
    new2 = {"dynamic_crossfire_alley", "smoke_lidar_corruption"}
    if sc in basic5:
        p_b_p1 = os.path.join(BM_ROOT, "ab_comparison", f"ab_{sc}_10trials.json")
        p_p2a = os.path.join(BM_ROOT, "phase2a_sim2sim", f"ab_{sc}_10trials.json")
    elif sc in new2:
        p_b_p1 = os.path.join(HDD_ROOT, "bench_hard8_phase1", f"ab_{sc}_10trials.json")
        p_p2a = os.path.join(HDD_ROOT, "bench_hard8_phase2a", f"ab_{sc}_10trials.json")
    else:
        p_b_p1 = os.path.join(HDD_ROOT, "bench_hard8_10trials", "phase1", f"ab_{sc}_10trials.json")
        p_p2a = os.path.join(HDD_ROOT, "bench_hard8_10trials", "phase2a", f"ab_{sc}_10trials.json")
    p_p2b = os.path.join(HDD_ROOT, "bench_phase2b_v3_dynfix", f"ab_{sc}_10trials.json")

    j_bp1 = json.load(open(p_b_p1, "r", encoding="utf-8"))
    j_p2a = json.load(open(p_p2a, "r", encoding="utf-8"))
    j_p2b = json.load(open(p_p2b, "r", encoding="utf-8"))
    return {
        "bas": j_bp1["baseline"]["trials"],
        "p1": j_bp1["proposed"]["trials"],
        "p2a": j_p2a["proposed"]["trials"],
        "p2b": j_p2b["proposed"]["trials"],
    }


def main() -> int:
    global MODEL_ROOTS
    MODEL_ROOTS = ensure_reference_model_cache(WS_ROOT)
    print(f"[MODEL] Gazebo model lookup roots: {MODEL_ROOTS}")
    fig, axes = plt.subplots(3, 5, figsize=(26, 15.5), facecolor="#f8fafc")

    for ax, (sc, title_base, t_idx, w_file, roi) in zip(axes.flat, PANELS):
        ax.set_facecolor("#ffffff")
        stages = _load_stage_trials(sc)
        r_bas = stages["bas"][t_idx - 1]
        r_p1 = stages["p1"][t_idx - 1]
        r_p2a = stages["p2a"][t_idx - 1]
        r_p2b = stages["p2b"][t_idx - 1]

        # 1. Gazebo collision geometry at LiDAR height (0.17m).
        # Explicit model roots prevent House/Maze partitions from silently disappearing.
        fp = extract_world_footprint(
            os.path.join(WS_WORLDS, w_file),
            models_root=MODEL_ROOTS,
            z_slice=0.17,
        )
        wall_count = len(fp.polygons)
        if sc in {"house", "house_hard"} and wall_count < 24:
            raise RuntimeError(
                f"{sc}: only {wall_count} wall polygons found; expected the "
                "TurtleBot3 House partitions. Check .cache_atlas_models/turtlebot3_house/model.sdf."
            )
        if sc in {"maze", "maze_hard"} and wall_count < 11:
            raise RuntimeError(
                f"{sc}: only {wall_count} wall polygons found; expected 4 perimeter "
                "walls + 7 inner partitions. Check turtlebot3_dqn_world/inner_walls/model.sdf."
            )
        if sc in {"house", "house_hard", "maze", "maze_hard"}:
            print(f"[GEOMETRY] {sc}: {wall_count} collision polygons loaded at z=0.17 m")

        if fp.polygons:
            polys = [p for _, p in fp.polygons if len(p) >= 3]
            pc = PolyCollection(
                polys,
                facecolor="#cbd5e1",
                edgecolor="#334155",
                linewidths=0.95,
                alpha=0.92,
                antialiased=True,
                zorder=1,
            )
            ax.add_collection(pc)
        if fp.segments:
            segs = [s for _, s in fp.segments]
            lc = LineCollection(segs, colors="#475569", linewidths=0.75, alpha=0.80, zorder=1)
            ax.add_collection(lc)
        if sc == "dynamic_crossfire_alley":
            # Active return-phase sweep trajectories across Gate 1 & Gate 2
            amp = 0.36 / 0.95
            for cx, cy0, col in ((2.15, 0.00, "#dc2626"), (5.05, 0.725, "#ea580c")):
                ax.plot([cx, cx], [cy0 - amp, cy0 + amp], color=col, ls="--", lw=2.0, alpha=0.85, zorder=2)
                ax.add_patch(mpatches.Circle((cx, cy0), 0.21, facecolor="#fca5a5", edgecolor=col, ls="--", lw=1.2, alpha=0.75, zorder=2))
        else:
            for _, dx, dy, dr in fp.dynamic_markers:
                ax.add_patch(mpatches.Circle((dx, dy), max(dr, 0.16), facecolor="#fca5a5", edgecolor="#b91c1c", ls="--", lw=1.0, alpha=0.7, zorder=2))

        # 2. Spawned fallen debris rocks (purple circles)
        for deb in r_p2b.get("debris_positions", []):
            ax.add_patch(mpatches.Circle((deb[0], deb[1]), 0.22, facecolor="#a855f7", edgecolor="#581c87", lw=0.8, alpha=0.72, zorder=3))

        # 3. Outbound exploration trajectory (grey dashed)
        out_pts = np.array(r_p2b.get("outbound_path", []), dtype=float)
        if len(out_pts) >= 2:
            ax.plot(out_pts[:, 0], out_pts[:, 1], color="#64748b", ls="--", lw=1.25, alpha=0.85, zorder=4)

        # 4. 4-Stage Return trajectories
        for rec, col, ls, lw, zo in (
            (r_bas, "#dc2626", ":", 1.35, 5),
            (r_p1, "#2563eb", "-.", 1.50, 6),
            (r_p2a, "#ea580c", "--", 1.50, 7),
            (r_p2b, "#16a34a", "-", 2.35, 8),
        ):
            r_pts = np.array(rec.get("return_path", []), dtype=float)
            if len(r_pts) >= 2:
                ax.plot(r_pts[:, 0], r_pts[:, 1], color=col, ls=ls, lw=lw, alpha=0.92, zorder=zo)

        # 5. Base Camp (blue star) & Blackout point (orange X)
        if len(out_pts) >= 1:
            ax.scatter([out_pts[0, 0]], [out_pts[0, 1]], marker="*", s=160, color="#2563eb", edgecolor="#0f172a", lw=0.7, zorder=10)
        tp = r_p2b.get("turnoff_pose")
        if tp:
            ax.scatter([tp[0]], [tp[1]], marker="X", s=85, color="#f59e0b", edgecolor="#78350f", lw=0.8, zorder=10)

        wall_tag = f"WALLS:{wall_count} | " if sc in {"house", "house_hard", "maze", "maze_hard"} else ""
        tag_s = (
            f"[{wall_tag}BAS:{'O' if r_bas['success'] else 'X'} | "
            f"P1:{'O' if r_p1['success'] else 'X'} | "
            f"P2A:{'O' if r_p2a['success'] else 'X'} | "
            f"P2B:{'O' if r_p2b['success'] else 'X'}]"
        )
        ax.set_title(f"{title_base}\n{tag_s}", fontsize=9.8, fontweight="bold", pad=6, color="#0f172a")
        ax.set_xlim(roi[0], roi[1])
        ax.set_ylim(roi[2], roi[3])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, ls=":", alpha=0.45, color="#94a3b8")
        ax.tick_params(labelsize=7.5)

    handles = [
        mlines.Line2D([], [], color="#cbd5e1", marker="s", markersize=8, markeredgecolor="#475569", ls="None", label="Gazebo Fixed Walls / Obstacles"),
        mlines.Line2D([], [], color="#2563eb", marker="*", markersize=11, markeredgecolor="#0f172a", ls="None", label="Base Camp (Start Pose)"),
        mlines.Line2D([], [], color="#64748b", ls="--", lw=1.5, label="Outbound Exploration Route (Entry)"),
        mlines.Line2D([], [], color="#f59e0b", marker="X", markersize=8, markeredgecolor="#78350f", ls="None", label="Blackout Turnoff Point"),
        mlines.Line2D([], [], color="#a855f7", marker="o", markersize=8, markeredgecolor="#581c87", ls="None", label="Spawned Fallen Debris"),
        mlines.Line2D([], [], color="#fca5a5", marker="o", markersize=8, markeredgecolor="#dc2626", ls="--", lw=1.4, label="Dynamic Moving Obstacle"),
        mlines.Line2D([], [], color="#dc2626", ls=":", lw=1.6, label="Baseline Return (4-Tier Bug)"),
        mlines.Line2D([], [], color="#2563eb", ls="-.", lw=1.6, label="Phase 1 Return (Rolling Grid A*)"),
        mlines.Line2D([], [], color="#ea580c", ls="--", lw=1.6, label="Phase 2-A Return (Naive RL)"),
        mlines.Line2D([], [], color="#16a34a", ls="-", lw=2.5, label="Phase 2-B Return (Adaptive Envelope + RL)"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.962),
        ncol=5,
        fontsize=9.2,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
    )
    fig.suptitle(
        "15 Disaster Scenarios: 2D Architectural Walls, Outbound Routes, Debris & 4-Stage Return Trajectories",
        fontsize=14.5,
        fontweight="bold",
        y=0.988,
        color="#0f172a",
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    out_png = os.path.join(WS_ROOT, "docs", "images", "scenario_routes_15_atlas.png")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[SAVED] {out_png} ({os.path.getsize(out_png)} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
