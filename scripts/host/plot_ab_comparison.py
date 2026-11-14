#!/usr/bin/env python3
"""
Generate clean, uncluttered, intuitive comparison plots for the
Fixed-Variable A/B Disaster Recovery Benchmark.

Fixes every readability issue present in stacked-spaghetti trajectory plots:
  1. Small-multiples layout: EACH trial (#01..#10) gets its own dedicated subplot
     showing ONLY that trial's outbound path, numbered debris rocks, Baseline
     return, and Proposed return -- zero cross-trial clutter.
  2. Shared figure-level legend placed completely outside the data axes so
     markers (like Base Camp) are never hidden behind legend boxes.
  3. Directional arrowheads clearly distinguish Entry (Outbound) vs Return.

Outputs into /home/lab/disaster_recovery_ws/docs/images/:
  - ab_environment_maps.png
  - ab_kpi_overview.png
  - ab_collapsed_10trials_grid.png
  - ab_other_envs_case_grid.png
  - ab_trial_pass_error_matrix.png
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np

from world_map_geometry import extract_world_footprint


WS_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
WS_WORLDS = os.path.join(WS_ROOT, "src", "disaster_recovery", "worlds")
WORLD_MAP_SPECS = {
    "collapsed": {
        "world_file": f"{WS_WORLDS}/disaster_collapsed_building.world",
        "reset_pose": (0.0, -4.0, 1.5708),
        "debris_zone": (-2.0, 10.0, 2.0, 12.0),
        "roi": (-5.6, 15.6, -6.6, 14.6),
        "panel_title": "1 & 2. Collapsed Building (20m x 20m Arena)\nNormal: 3-6 Rocks (>=1.2m apart) | Heavy: 6-9 Rocks (>=0.9m)",
    },
    "collapsed_heavy": {
        "world_file": f"{WS_WORLDS}/disaster_collapsed_building.world",
        "reset_pose": (0.0, -4.0, 1.5708),
        "debris_zone": (-2.0, 10.0, 2.0, 12.0),
        "roi": (-5.6, 15.6, -6.6, 14.6),
        "panel_title": "Collapsed Heavy",
    },
    "tunnel": {
        "world_file": f"{WS_WORLDS}/disaster_tunnel_world.world",
        "reset_pose": (-0.5, -3.0, 1.5708),
        "debris_zone": (-2.5, 2.5, 2.0, 22.0),
        "roi": (-5.8, 5.8, -4.5, 23.5),
        "panel_title": "3. Disaster Tunnel (Narrow Corridor + Wreckage)\nArch Walls (X=+-5.1m) + SUV, Excavator, Beams | 3-6 Rocks",
    },
    "house": {
        "world_file": f"{WS_WORLDS}/disaster_house_world.world",
        "reset_pose": (-2.0, -0.5, 0.0),
        "debris_zone": (-6.2, 5.5, -3.2, 4.4),
        "roi": (-8.0, 8.0, -5.8, 5.8),
        "panel_title": "4. Indoor House (Multi-Room & Narrow Doorways)\n15m x 11m Floorplan with Partition Bottlenecks | 2-4 Rocks",
    },
    "maze": {
        "world_file": f"{WS_WORLDS}/disaster_maze_world.world",
        "reset_pose": (0.0, 0.0, 0.0),
        "debris_zone": (-2.0, 2.0, -2.0, 2.0),
        "roi": (-2.75, 2.75, -2.75, 2.75),
        "panel_title": "5. Dynamic Maze (5m x 5m + Moving Obstacles)\nTight Partitions + 2 Moving Obstacles | 2-4 Rocks",
    },
}


SCENARIO_ORDER = ["collapsed", "collapsed_heavy", "tunnel", "house", "maze"]
SCENARIO_SHORT = {
    "collapsed": "Collapsed\n(3-6 Rocks)",
    "collapsed_heavy": "Heavy Debris\n(6-9 Rocks)",
    "tunnel": "Tunnel\n(Corridor)",
    "house": "House\n(Multi-Room)",
    "maze": "Maze\n(Dynamic)",
}
SCENARIO_INLINE = {
    "collapsed": "Collapsed Bldg (3-6 Rocks)",
    "collapsed_heavy": "Collapsed Heavy (6-9 Rocks)",
    "tunnel": "Disaster Tunnel",
    "house": "Indoor House",
    "maze": "Dynamic Maze",
}

BASE_COLOR = "#d95f02"
PROP_COLOR = "#1b9e77"
OUT_COLOR = "#7f8c8d"
ROCK_COLOR = "#8e44ad"
FAIL_COLOR = "#c0392b"
BASE_CAMP_COLOR = "#1f618d"
DARK_TEXT = "#222222"


def load_results(in_dir: str) -> Dict[str, dict]:
    loaded = {}
    all_files = sorted(os.listdir(in_dir)) if os.path.isdir(in_dir) else []
    for name in SCENARIO_ORDER:
        pat = re.compile(rf"^ab_{re.escape(name)}_\d+trials\.json$")
        matches = [os.path.join(in_dir, fn) for fn in all_files if pat.match(fn)]
        if matches:
            with open(matches[-1], "r") as f:
                loaded[name] = json.load(f)
    return loaded


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#bbbbbb")
    ax.spines["bottom"].set_color("#bbbbbb")
    ax.tick_params(colors=DARK_TEXT, labelsize=8.5)


def _add_mid_arrow(ax, pts: np.ndarray, color: str, frac: float = 0.52, lw: float = 1.8):
    """Draw a clean directional arrowhead along the trajectory without clutter."""
    if len(pts) < 4:
        return
    idx = max(1, min(len(pts) - 2, int(len(pts) * frac)))
    p0 = pts[idx - 1]
    p1 = pts[min(len(pts) - 1, idx + 1)]
    if np.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 0.05:
        return
    ax.annotate(
        "",
        xy=(p1[0], p1[1]),
        xytext=(p0[0], p0[1]),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=11),
        zorder=6,
    )


def _draw_world_backdrop(
    ax,
    scenario_name: str,
    poly_face: str = "#cbd5e1",
    poly_edge: str = "#94a3b8",
    seg_color: str = "#64748b",
    alpha: float = 0.55,
    seg_lw: float = 1.0,
    zorder: int = 1,
):
    """Draw static Gazebo world walls & obstacles at robot height without altering axis limits."""
    spec = WORLD_MAP_SPECS.get(scenario_name)
    if not spec:
        return None
    fp = extract_world_footprint(spec["world_file"])
    if fp.polygons:
        polys = [p for _, p in fp.polygons if len(p) >= 3]
        if polys:
            pc = PolyCollection(
                polys,
                facecolors=poly_face,
                edgecolors=poly_edge,
                linewidths=0.75,
                alpha=alpha,
                zorder=zorder,
            )
            pc.set_clip_on(True)
            ax.add_collection(pc, autolim=False)
    if fp.segments:
        # Filter rock_1 far outside tunnel ROI for clean vector rendering
        segs = []
        for lbl, seg in fp.segments:
            if scenario_name == "tunnel" and "rock_1" in lbl.lower():
                continue
            segs.append(seg)
        if segs:
            lc = LineCollection(
                segs,
                colors=seg_color,
                linewidths=seg_lw,
                alpha=min(0.85, alpha + 0.15),
                zorder=zorder,
            )
            lc.set_clip_on(True)
            ax.add_collection(lc, autolim=False)
    for _, dx, dy, dr in fp.dynamic_markers:
        circ = mpatches.Circle(
            (dx, dy),
            radius=max(0.16, dr),
            facecolor="#e74c3c",
            edgecolor="#922b21",
            alpha=0.80,
            zorder=zorder + 1,
        )
        ax.add_patch(circ)
    return fp


def _draw_single_paired_trial(
    ax,
    scenario_name: str,
    bx: float,
    by: float,
    bt: dict,
    pt: dict,
    title: str,
    show_rock_numbers: bool = True,
):
    """Render exactly ONE trial (same seed, same debris) on one clean subplot."""
    _style_ax(ax)
    _draw_world_backdrop(ax, scenario_name, poly_face="#d8dee9", poly_edge="#94a3b8", seg_color="#7b8794", alpha=0.52, seg_lw=0.85, zorder=1)

    b_out = np.array(bt["outbound_path"], dtype=float)
    out_p = np.array(pt["outbound_path"], dtype=float)
    bret = np.array(bt["return_path"], dtype=float)
    pret = np.array(pt["return_path"], dtype=float)

    if len(b_out) > 1 and len(out_p) > 1:
        b_end = bt["turnoff_pose"]
        p_end = pt["turnoff_pose"]
        if np.hypot(b_end[0] - p_end[0], b_end[1] - p_end[1]) > 1.2:
            ax.plot(b_out[:, 0], b_out[:, 1], color="#e59866", linestyle=":", linewidth=1.1, alpha=0.65, zorder=2)
    if len(out_p) > 1:
        ax.plot(out_p[:, 0], out_p[:, 1], color=OUT_COLOR, linestyle="--", linewidth=1.3, alpha=0.78, zorder=2)
        _add_mid_arrow(ax, out_p, OUT_COLOR, frac=0.45, lw=1.2)

    if len(bret) > 1:
        ax.plot(bret[:, 0], bret[:, 1], color=BASE_COLOR, linewidth=2.1, alpha=0.88, zorder=3)
        _add_mid_arrow(ax, bret, BASE_COLOR, frac=0.60, lw=1.6)

    if len(pret) > 1:
        ax.plot(pret[:, 0], pret[:, 1], color=PROP_COLOR, linewidth=2.3, alpha=0.95, zorder=4)
        _add_mid_arrow(ax, pret, PROP_COLOR, frac=0.65, lw=1.7)

    # Numbered fallen rocks for THIS single trial only
    for r_i, (rx, ry) in enumerate(bt["debris_positions"], start=1):
        ax.scatter([rx], [ry], color=ROCK_COLOR, marker="^", s=72, edgecolor="white", linewidth=0.5, zorder=5)
        if show_rock_numbers:
            ax.text(rx + 0.14, ry + 0.12, f"R{r_i}", fontsize=6.8, fontweight="bold", color=ROCK_COLOR, zorder=6)

    # Base Camp & Blackout Start & End points
    ax.scatter([bx], [by], color=BASE_CAMP_COLOR, marker="*", s=180, edgecolor="white", linewidth=0.5, zorder=8)
    tx, ty = pt["turnoff_pose"]
    ax.scatter([tx], [ty], color="#f39c12", marker="s", s=46, edgecolor="black", linewidth=0.5, zorder=7)

    ax.scatter(
        [bt["final_pose"][0]],
        [bt["final_pose"][1]],
        color=FAIL_COLOR if not bt["success"] else BASE_COLOR,
        marker="X" if not bt["success"] else "o",
        s=75,
        edgecolor="white",
        linewidth=0.6,
        zorder=9,
    )
    ax.scatter(
        [pt["final_pose"][0]],
        [pt["final_pose"][1]],
        color=PROP_COLOR if pt["success"] else FAIL_COLOR,
        marker="o" if pt["success"] else "X",
        s=70,
        edgecolor="white",
        linewidth=0.6,
        zorder=9,
    )

    ax.set_title(title, fontsize=8.8, fontweight="bold", pad=5)
    ax.set_aspect("equal", adjustable="datalim")
    if scenario_name == "maze":
        ax.set_xlim(-2.85, 2.85)
        ax.set_ylim(-2.85, 2.85)
    else:
        ax.margins(0.14)
    ax.grid(True, linestyle=":", alpha=0.30)


def _shared_traj_legend(fig):
    handles = [
        mpatches.Patch(facecolor="#d8dee9", edgecolor="#94a3b8", label="Map Walls / Static Obstacles"),
        mlines.Line2D([], [], color=OUT_COLOR, linestyle="--", lw=1.6, label="Entry Path (Outbound)"),
        mlines.Line2D([], [], color=BASE_COLOR, lw=2.3, label="Baseline Return (Current)"),
        mlines.Line2D([], [], color=PROP_COLOR, lw=2.5, label="Proposed Return (Grid A* + Escape)"),
        mlines.Line2D([], [], color=ROCK_COLOR, marker="^", linestyle="None", markersize=8, label="Fallen Debris (R1..Rn)"),
        mlines.Line2D([], [], color=BASE_CAMP_COLOR, marker="*", linestyle="None", markersize=11, label="Base Camp (Goal)"),
        mlines.Line2D([], [], color="#f39c12", marker="s", linestyle="None", markersize=7, markeredgecolor="black", label="Blackout Trigger"),
        mlines.Line2D([], [], color=FAIL_COLOR, marker="X", linestyle="None", markersize=8, label="Failed / Deadlock Stop"),
        mlines.Line2D([], [], color=PROP_COLOR, marker="o", linestyle="None", markersize=7, label="Success Arrival"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.948),
        ncol=5,
        fontsize=8.4,
        frameon=True,
        facecolor="#fafafa",
        edgecolor="#dddddd",
    )


def plot_kpi_overview(data: Dict[str, dict], out_path: str) -> None:
    names = [n for n in SCENARIO_ORDER if n in data]
    if not names:
        return

    x = np.arange(len(names))
    w = 0.35

    base_sr = [data[n]["baseline"]["summary"]["success_rate_pct"] for n in names]
    prop_sr = [data[n]["proposed"]["summary"]["success_rate_pct"] for n in names]
    base_err = [data[n]["baseline"]["summary"]["mean_error_cm_all"] for n in names]
    prop_err = [data[n]["proposed"]["summary"]["mean_error_cm_all"] for n in names]
    base_t = [data[n]["baseline"]["summary"]["mean_time_s_all"] for n in names]
    prop_t = [data[n]["proposed"]["summary"]["mean_time_s_all"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.8), dpi=200)
    fig.patch.set_facecolor("white")

    # 1. Success Rate
    ax = axes[0]
    _style_ax(ax)
    b1 = ax.bar(x - w / 2, base_sr, w, label="Baseline (Current)", color=BASE_COLOR, alpha=0.88)
    b2 = ax.bar(x + w / 2, prop_sr, w, label="Proposed (Grid A* + Escape)", color=PROP_COLOR, alpha=0.92)
    ax.set_ylim(0, 122)
    ax.set_ylabel("Success Rate (%)", fontsize=10.5, fontweight="bold")
    ax.set_title("Return Success Rate (Higher = Better)", fontsize=11.5, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_SHORT[n] for n in names], fontsize=8.4)
    ax.grid(axis="y", linestyle="--", alpha=0.30)
    for bar, v in zip(b1, base_sr):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color=BASE_COLOR)
    for bar, v in zip(b2, prop_sr):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color=PROP_COLOR)

    # 2. Mean Error
    ax = axes[1]
    _style_ax(ax)
    b1 = ax.bar(x - w / 2, base_err, w, color=BASE_COLOR, alpha=0.88)
    b2 = ax.bar(x + w / 2, prop_err, w, color=PROP_COLOR, alpha=0.92)
    ymax_e = max(max(base_err), max(prop_err)) * 1.20
    ax.set_ylim(0, max(50.0, ymax_e))
    ax.set_ylabel("Mean Final Distance Error (cm)", fontsize=10.5, fontweight="bold")
    ax.set_title("Mean Final Distance Error (Lower = Better)", fontsize=11.5, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_SHORT[n] for n in names], fontsize=8.4)
    ax.grid(axis="y", linestyle="--", alpha=0.30)
    for bar, v in zip(b1, base_err):
        ax.text(bar.get_x() + bar.get_width() / 2, v + ymax_e * 0.015, f"{v:.0f}", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color=BASE_COLOR)
    for bar, v in zip(b2, prop_err):
        ax.text(bar.get_x() + bar.get_width() / 2, v + ymax_e * 0.015, f"{v:.1f}", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color=PROP_COLOR)

    # 3. Mean Return Time
    ax = axes[2]
    _style_ax(ax)
    b1 = ax.bar(x - w / 2, base_t, w, color=BASE_COLOR, alpha=0.88)
    b2 = ax.bar(x + w / 2, prop_t, w, color=PROP_COLOR, alpha=0.92)
    ymax_t = max(max(base_t), max(prop_t)) * 1.20
    ax.set_ylim(0, max(50.0, ymax_t))
    ax.set_ylabel("Mean Return Time (s)", fontsize=10.5, fontweight="bold")
    ax.set_title("Mean Return Time (Lower = Better)", fontsize=11.5, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_SHORT[n] for n in names], fontsize=8.4)
    ax.grid(axis="y", linestyle="--", alpha=0.30)
    for bar, v in zip(b1, base_t):
        ax.text(bar.get_x() + bar.get_width() / 2, v + ymax_t * 0.015, f"{v:.0f}s", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color=BASE_COLOR)
    for bar, v in zip(b2, prop_t):
        ax.text(bar.get_x() + bar.get_width() / 2, v + ymax_t * 0.015, f"{v:.0f}s", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color=PROP_COLOR)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, fontsize=10.0, frameon=True, facecolor="#fafafa", edgecolor="#dddddd")
    fig.suptitle(
        "Fixed-Variable A/B Benchmark Overview (10 Paired Trials per Environment)",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.87])
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[PLOT] Saved {out_path}")


def plot_collapsed_10trials_grid(data: Dict[str, dict], out_path: str) -> None:
    """2x5 Small-Multiples plot: Trial #01..#10 each gets its OWN clean trajectory map."""
    if "collapsed" not in data:
        return
    payload = data["collapsed"]
    bx, by, _ = payload["config"]["reset_pose"]
    base_map = {t["trial"]: t for t in payload["baseline"]["trials"]}
    prop_map = {t["trial"]: t for t in payload["proposed"]["trials"]}
    trial_ids = sorted(base_map.keys())

    fig, axes = plt.subplots(2, 5, figsize=(20.0, 9.6), dpi=180)
    fig.patch.set_facecolor("white")
    axes_flat = axes.reshape(-1)

    for i, ax in enumerate(axes_flat):
        if i >= len(trial_ids):
            ax.axis("off")
            continue
        tid = trial_ids[i]
        bt = base_map[tid]
        pt = prop_map[tid]
        b_tag = f"{'PASS' if bt['success'] else 'FAIL'} {bt['terminal_error_cm']:.0f}cm"
        p_tag = f"{'PASS' if pt['success'] else 'FAIL'} {pt['terminal_error_cm']:.0f}cm"
        title = f"Trial #{tid:02d} ({pt['explored_path_m']:.1f}m, {pt['debris_count']} rocks)\nBase: {b_tag} | Prop: {p_tag}"
        _draw_single_paired_trial(ax, "collapsed", bx, by, bt, pt, title, show_rock_numbers=True)
        if i >= 5:
            ax.set_xlabel("X (m)", fontsize=8.5)
        if i % 5 == 0:
            ax.set_ylabel("Y (m)", fontsize=8.5)

    _shared_traj_legend(fig)
    fig.suptitle(
        "Collapsed Building (Current Map) - Individual Trial-by-Trial Trajectories (#01 ~ #10)",
        fontsize=14.0,
        fontweight="bold",
        y=0.985,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[PLOT] Saved {out_path}")


def plot_other_envs_case_grid(data: Dict[str, dict], out_path: str) -> None:
    """Clean paired case grid across other environments (2 representative trials per row)."""
    others = [n for n in ["collapsed_heavy", "tunnel", "house", "maze"] if n in data]
    if not others:
        return

    fig, axes = plt.subplots(len(others), 2, figsize=(13.5, 4.6 * len(others) + 1.0), dpi=180)
    fig.patch.set_facecolor("white")
    if len(others) == 1:
        axes = np.array([axes])

    for r_i, name in enumerate(others):
        payload = data[name]
        bx, by, _ = payload["config"]["reset_pose"]
        bmap = {t["trial"]: t for t in payload["baseline"]["trials"]}
        pmap = {t["trial"]: t for t in payload["proposed"]["trials"]}

        # Pick top 2 contrasting trials: (1) worst Baseline failure rescued by Proposed, (2) 2nd most challenging
        scored: List[Tuple[float, int]] = []
        for tid, bt in bmap.items():
            pt = pmap.get(tid)
            if not pt:
                continue
            sc = (1000.0 if (not bt["success"] and pt["success"]) else 0.0) + bt["terminal_error_cm"] + 15.0 * bt["debris_count"]
            scored.append((sc, tid))
        scored.sort(reverse=True)
        chosen = [s[1] for s in scored[:2]]

        for c_i in range(2):
            ax = axes[r_i, c_i]
            if c_i >= len(chosen):
                ax.axis("off")
                continue
            tid = chosen[c_i]
            bt = bmap[tid]
            pt = pmap[tid]
            b_tag = f"{'PASS' if bt['success'] else 'FAIL'} {bt['terminal_error_cm']:.0f}cm ({bt['return_duration_s']:.0f}s)"
            p_tag = f"{'PASS' if pt['success'] else 'FAIL'} {pt['terminal_error_cm']:.1f}cm ({pt['return_duration_s']:.0f}s)"
            title = f"[{SCENARIO_INLINE[name]}] Trial #{tid:02d} | {pt['explored_path_m']:.1f}m, {pt['debris_count']} rocks\nBase: {b_tag}  vs  Prop: {p_tag}"
            _draw_single_paired_trial(ax, name, bx, by, bt, pt, title, show_rock_numbers=True)
            ax.set_xlabel("X (m)", fontsize=8.8)
            if c_i == 0:
                ax.set_ylabel("Y (m)", fontsize=8.8)

    _shared_traj_legend(fig)
    fig.suptitle(
        "Other Disaster Environments - Key Paired Trial Trajectories (Baseline vs. Proposed)",
        fontsize=13.5,
        fontweight="bold",
        y=0.988,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[PLOT] Saved {out_path}")


def plot_trial_pass_error_matrix(data: Dict[str, dict], out_path: str) -> None:
    """Clean per-trial Pass/Fail & Log-Error strip across every scenario and trial #1..#10."""
    names = [n for n in SCENARIO_ORDER if n in data]
    if not names:
        return

    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 4.8), dpi=190, sharey=True)
    fig.patch.set_facecolor("white")
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        _style_ax(ax)
        payload = data[name]
        b_trials = payload["baseline"]["trials"]
        p_trials = payload["proposed"]["trials"]
        tids = [t["trial"] for t in b_trials]
        b_errs = [max(1.0, t["terminal_error_cm"]) for t in b_trials]
        p_errs = [max(1.0, t["terminal_error_cm"]) for t in p_trials]

        ax.plot(tids, b_errs, color=BASE_COLOR, marker="o", lw=1.7, ms=5.5, alpha=0.88, label="Baseline")
        ax.plot(tids, p_errs, color=PROP_COLOR, marker="s", lw=1.9, ms=5.5, alpha=0.95, label="Proposed")
        ax.axhline(85.0, color="#555555", linestyle="--", lw=1.0, label="Pass Limit (85 cm)")
        ax.set_yscale("log")
        ax.set_ylim(1.0, 2200.0)
        ax.set_xticks(tids)
        ax.set_xlabel("Trial #", fontsize=9.2)
        sb = payload["baseline"]["summary"]["success_rate_pct"]
        sp = payload["proposed"]["summary"]["success_rate_pct"]
        ax.set_title(f"{SCENARIO_SHORT[name]}\n(Base {sb:.0f}% vs Prop {sp:.0f}%)", fontsize=9.5, fontweight="bold")
        ax.grid(True, which="both", linestyle=":", alpha=0.30)

    axes[0].set_ylabel("Terminal Return Error (cm, log scale)", fontsize=10.5, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3, fontsize=9.5, frameon=True, facecolor="#fafafa", edgecolor="#dddddd")
    fig.suptitle(
        "Per-Trial Return Error Stability (#01 ~ #10 Across All Scenarios)",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.87])
    plt.savefig(out_path, dpi=190)
    plt.close(fig)
    print(f"[PLOT] Saved {out_path}")


def plot_environment_maps(data: Dict[str, dict], out_path: str) -> None:
    """Render true-scale top-down layouts of every disaster world so readers see the actual arenas."""
    map_names = ["collapsed", "tunnel", "house", "maze"]
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 13.5), dpi=200)
    fig.patch.set_facecolor("white")

    note_bbox = dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cbd5e1", alpha=0.90)

    for ax, name in zip(axes.reshape(-1), map_names):
        _style_ax(ax)
        spec = WORLD_MAP_SPECS[name]
        bx, by, byaw = spec["reset_pose"]
        dx0, dx1, dy0, dy1 = spec["debris_zone"]
        rx0, rx1, ry0, ry1 = spec["roi"]

        # Shaded dynamic random rock drop zone
        zone = mpatches.Rectangle(
            (dx0, dy0),
            dx1 - dx0,
            dy1 - dy0,
            facecolor="#8e44ad",
            edgecolor="#8e44ad",
            linestyle="--",
            linewidth=1.3,
            alpha=0.08,
            zorder=0,
        )
        zone_border = mpatches.Rectangle(
            (dx0, dy0),
            dx1 - dx0,
            dy1 - dy0,
            facecolor="none",
            edgecolor="#8e44ad",
            linestyle="--",
            linewidth=1.3,
            alpha=0.65,
            zorder=2,
        )
        ax.add_patch(zone)
        ax.add_patch(zone_border)

        # True-scale walls & obstacles at robot LiDAR height (z=0.17m)
        _draw_world_backdrop(
            ax,
            name,
            poly_face="#475569",
            poly_edge="#1e293b",
            seg_color="#334155",
            alpha=0.82,
            seg_lw=1.1,
            zorder=3,
        )

        # Base Camp + Entry heading vector
        ax.scatter([bx], [by], color=BASE_CAMP_COLOR, marker="*", s=260, edgecolor="white", linewidth=0.7, zorder=8)
        span = max(rx1 - rx0, ry1 - ry0)
        arrow_len = 0.08 * span
        ax.annotate(
            "",
            xy=(bx + arrow_len * np.cos(byaw), by + arrow_len * np.sin(byaw)),
            xytext=(bx, by),
            arrowprops=dict(arrowstyle="-|>", color=BASE_CAMP_COLOR, lw=2.2, mutation_scale=13),
            zorder=8,
        )
        if name == "tunnel":
            ax.text(
                bx + 0.85,
                by,
                f"Base Camp\n({bx:.1f}, {by:.1f})",
                ha="left",
                va="center",
                fontsize=8.2,
                fontweight="bold",
                color=BASE_CAMP_COLOR,
                bbox=note_bbox,
                zorder=9,
            )
        else:
            ax.text(
                bx,
                by - 0.055 * (ry1 - ry0),
                f"Base Camp\n({bx:.1f}, {by:.1f})",
                ha="center",
                va="top",
                fontsize=8.2,
                fontweight="bold",
                color=BASE_CAMP_COLOR,
                bbox=note_bbox,
                zorder=9,
            )

        # Representative Trial #01 outbound trajectory if benchmark JSON exists
        if name in data and data[name].get("proposed", {}).get("trials"):
            t1 = data[name]["proposed"]["trials"][0]
            op = np.array(t1["outbound_path"], dtype=float)
            if len(op) > 1:
                ax.plot(op[:, 0], op[:, 1], color=PROP_COLOR, linestyle="--", lw=1.7, alpha=0.80, zorder=5)
                _add_mid_arrow(ax, op, PROP_COLOR, frac=0.55, lw=1.5)

        # Environment-specific clear landmark callouts
        if name == "collapsed":
            ax.text(4.0, -1.8, "Tilted Wall A\n& Pillars", fontsize=7.6, color="#334155", ha="center", bbox=note_bbox, zorder=9)
            ax.text(5.4, 5.6, "Rubble Piles", fontsize=7.6, color="#334155", ha="center", bbox=note_bbox, zorder=9)
            ax.text(4.0, 11.1, "Random Rock Drop Zone\n(3-6 Normal / 6-9 Heavy)", fontsize=7.8, color="#6c3483", fontweight="bold", ha="center", bbox=note_bbox, zorder=9)
        elif name == "tunnel":
            ax.text(0.4, 1.35, "SUV\n(Chassis Clearance 0.55m)", fontsize=7.4, color="#334155", ha="center", bbox=note_bbox, zorder=9)
            ax.text(1.5, 7.6, "Excavator\n& Cinder Blocks", fontsize=7.4, color="#334155", ha="center", bbox=note_bbox, zorder=9)
            ax.text(-5.2, 11.2, "Arch Wall", fontsize=7.2, color="#334155", rotation=90, va="center", ha="center", bbox=note_bbox, zorder=9)
            ax.text(0.0, 20.3, "Rock Drop Zone\n(3-6 Rocks)", fontsize=7.6, color="#6c3483", fontweight="bold", ha="center", bbox=note_bbox, zorder=9)
        elif name == "house":
            ax.text(-4.0, 2.3, "West Rooms\n& Narrow Doorways", fontsize=7.6, color="#334155", ha="center", bbox=note_bbox, zorder=9)
            ax.text(3.2, -1.8, "East Rooms", fontsize=7.6, color="#334155", ha="center", bbox=note_bbox, zorder=9)
            ax.text(0.0, 3.6, "Rock Drop Zone (2-4 Rocks)", fontsize=7.8, color="#6c3483", fontweight="bold", ha="center", bbox=note_bbox, zorder=9)
        elif name == "maze":
            for idx, (ox, oy) in enumerate(((2.0, 2.0), (-2.0, -2.0)), start=1):
                ring = mpatches.Circle((ox, oy), radius=0.38, facecolor="none", edgecolor="#c0392b", linestyle=":", linewidth=1.6, zorder=6)
                ax.add_patch(ring)
            ax.text(1.52, 1.88, "Dynamic Obst #1\n(Moving Plugin)", fontsize=7.5, fontweight="bold", color="#c0392b", ha="right", va="center", bbox=note_bbox, zorder=9)
            ax.text(-1.52, -1.88, "Dynamic Obst #2\n(Moving Plugin)", fontsize=7.5, fontweight="bold", color="#c0392b", ha="left", va="center", bbox=note_bbox, zorder=9)
            ax.text(0.0, 2.15, "Inner Partitions & Dynamic Rock Zone (2-4 Rocks)", fontsize=7.6, color="#6c3483", fontweight="bold", ha="center", bbox=note_bbox, zorder=9)

        ax.set_xlim(rx0, rx1)
        ax.set_ylim(ry0, ry1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(spec["panel_title"], fontsize=10.5, fontweight="bold", pad=8)
        ax.set_xlabel("X (m)", fontsize=9.0)
        ax.set_ylabel("Y (m)", fontsize=9.0)
        ax.grid(True, linestyle=":", alpha=0.35)

    handles = [
        mpatches.Patch(facecolor="#475569", edgecolor="#1e293b", alpha=0.82, label="Fixed Walls / Structural Obstacles (LiDAR Height z=0.17m)"),
        mpatches.Patch(facecolor="#8e44ad", edgecolor="#8e44ad", linestyle="--", alpha=0.22, label="Random Fallen Rock Spawn Zone"),
        mlines.Line2D([], [], color=BASE_CAMP_COLOR, marker="*", linestyle="None", markersize=12, label="Base Camp (Start & Return Goal)"),
        mlines.Line2D([], [], color=PROP_COLOR, linestyle="--", lw=1.8, label="Sample Outbound Entry Path (Trial #01)"),
        mlines.Line2D([], [], color="#e74c3c", marker="o", linestyle="None", markersize=8, markeredgecolor="#922b21", label="Dynamic Moving Obstacles (Maze Only)"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        fontsize=9.0,
        frameon=True,
        facecolor="#fafafa",
        edgecolor="#dddddd",
    )
    fig.suptitle(
        "5 Disaster Simulation Environments - Top-Down Architectural Floorplans & Hazard Zones",
        fontsize=14.0,
        fontweight="bold",
        y=0.985,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.895])
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[PLOT] Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        default=os.path.join(WS_ROOT, "benchmark_results", "ab_comparison"),
    )
    parser.add_argument(
        "--img-dir",
        default=os.path.join(WS_ROOT, "docs", "images"),
    )
    args = parser.parse_args()
    os.makedirs(args.img_dir, exist_ok=True)
    data = load_results(args.in_dir)
    if not data:
        print(f"No benchmark JSON files found in {args.in_dir}")
        return
    plot_environment_maps(data, os.path.join(args.img_dir, "ab_environment_maps.png"))
    plot_kpi_overview(data, os.path.join(args.img_dir, "ab_kpi_overview.png"))
    plot_collapsed_10trials_grid(data, os.path.join(args.img_dir, "ab_collapsed_10trials_grid.png"))
    plot_other_envs_case_grid(data, os.path.join(args.img_dir, "ab_other_envs_case_grid.png"))
    plot_trial_pass_error_matrix(data, os.path.join(args.img_dir, "ab_trial_pass_error_matrix.png"))


if __name__ == "__main__":
    main()
