#!/usr/bin/env python3
"""
Phase 4 (Sim v4.4 + Real TB3 v3) 10-trial deterministic benchmark in
`disaster_semantic_rescue_world.world`.

Uses the exact same deterministic trial seeds, sub-LiDAR low obstacle & gas hazard
placements, outbound waypoints, semantic cache payload, and strict hazard-free
evaluation criteria as `run_phase3_semantic_suite.py --mode ab10`, so results are
directly comparable against `blind_phase2b` (0/10) and `semantic_phase3` (9/10).
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from std_msgs.msg import Bool, String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ab_disaster_benchmark import (  # noqa: E402
    _kill_matching_gazebo_procs,
    get_gazebo_true_pose,
    run_cmd_bounded,
)
from run_phase3_semantic_suite import (  # noqa: E402
    SEMANTIC_WORLD,
    SemanticSuiteNode,
    _delete_entity,
    _gas_disc_sdf,
    _low_obstacle_sdf,
    _spawn_model_at,
    _stop_procs,
    compute_semantic_world_min_wall_geom,
)

DEFAULT_P4_SIM_ONNX = os.path.join(
    WS_ROOT, "src", "disaster_recovery", "config", "sim_policy_student_phase4.onnx"
)
DEFAULT_P4_REAL_ONNX = os.path.join(
    WS_ROOT, "src", "disaster_recovery", "config", "real_tb3_policy_student_phase4.onnx"
)


def build_specs(trials_n: int) -> List[dict]:
    specs = []
    for t in range(1, trials_n + 1):
        rng = random.Random(2026 + t * 131)
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
    return specs


def run_benchmark(args) -> int:
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
    specs = build_specs(args.trials)

    mode_defs = {
        "phase4_sim": (DEFAULT_P4_SIM_ONNX, -0.06),
        "phase4_real": (DEFAULT_P4_REAL_ONNX, -0.08),
    }
    selected_modes = [m.strip() for m in args.modes.split(",") if m.strip() in mode_defs]
    if not selected_modes:
        raise ValueError(f"No valid modes selected from {args.modes}")

    results: Dict[str, List[dict]] = {m: [] for m in selected_modes}

    for mode_key in selected_modes:
        onnx_path, cbf_v_min = mode_defs[mode_key]
        print(
            f"\n=== [PHASE 4 SEMANTIC {args.trials} TRIALS] Mode={mode_key} "
            f"(onnx={os.path.basename(onnx_path)}, cbf_v_min={cbf_v_min}) ===",
            flush=True,
        )
        _kill_matching_gazebo_procs(gz_uri)
        procs = []
        p_gz = subprocess.Popen(
            [
                "bash",
                "-c",
                env_prefix
                + f"ros2 launch disaster_recovery disaster_world.launch.py "
                f"x_pose:=0.0 y_pose:=0.0 yaw_pose:=0.0 world:={SEMANTIC_WORLD} gui:=false",
            ],
            stdout=open(f"/tmp/disaster_recovery/p4sem_gz_{mode_key}.log", "w"),
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        procs.append(p_gz)
        time.sleep(5.0)

        p_ctrl = subprocess.Popen(
            [
                "bash",
                "-c",
                env_prefix
                + f"ros2 run disaster_recovery recovery_controller --ros-args --params-file {params_file} "
                f"-p use_sim_time:=true -p use_aruco_docking:=false -p recovery_strategy:=proposed_rl "
                f"-p use_semantic_rl_policy:=true "
                f"-p enable_semantic_astar_cost:={'true' if args.enable_semantic_astar_cost else 'false'} "
                f"-p enable_dynamic_tracker:=true "
                f"-p enable_cbf_safety_shield:=true "
                f"-p cbf_v_min:={cbf_v_min} "
                f"-p context_rl_onnx_path:={onnx_path} "
                f"-p semantic_rl_onnx_path:={onnx_path} "
                f"-p phase4_rl_onnx_path:={onnx_path} "
                f"-p reset_corr_on_episode:=true",
            ],
            stdout=open(f"/tmp/disaster_recovery/p4sem_ctrl_{mode_key}.log", "w"),
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
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
            run_cmd_bounded(
                f"GAZEBO_MASTER_URI={gz_uri} timeout -k 1 2 gz physics -u {u_rate} >/dev/null 2>&1",
                timeout_s=4.0,
            )

            for spec in specs:
                t_idx = spec["trial"]
                for k in range(1, 5):
                    _delete_entity(node, f"low_obs_{k}")
                _delete_entity(node, "gas_zone_1")

                node.reset_episode_and_sync(gz_uri, target_x=0.0, target_y=0.0, tol_m=0.15)

                low_list = spec["low_obstacles"]
                gas_xy = spec["gas_zone"]
                for idx_l, (lx, ly) in enumerate(low_list, start=1):
                    _spawn_model_at(node, f"low_obs_{idx_l}", _low_obstacle_sdf(f"low_obs_{idx_l}"), lx, ly, 0.0)
                _spawn_model_at(node, "gas_zone_1", _gas_disc_sdf("gas_zone_1"), gas_xy[0], gas_xy[1], 0.0)

                out_wps = [(0.9, 0.0), (1.55, 0.55), (2.7, 0.60), (3.95, 0.55), (4.45, 0.30), (4.95, 0.25)]
                outbound_path = node.drive_to_waypoints(out_wps, max_sim_sec=75.0)
                node.sim_sleep(0.3)

                cache_objs = []
                for idx_l, (lx, ly) in enumerate(low_list, start=1):
                    cache_objs.append({
                        "marker_id": idx_l,
                        "class_id": 56,
                        "class_name": "chair",
                        "x": lx,
                        "y": ly,
                        "z": 0.05,
                        "radius": 0.24,
                        "channel": 1,
                        "conf": 0.91,
                        "hits": 6,
                        "low_confidence": False,
                    })
                cache_objs.append({
                    "marker_id": 10,
                    "class_id": 1001,
                    "class_name": "GAS_HAZARD",
                    "x": gas_xy[0],
                    "y": gas_xy[1],
                    "z": 0.0,
                    "radius": 0.55,
                    "channel": 2,
                    "conf": 0.85,
                    "hits": 10,
                    "low_confidence": False,
                })
                cache_objs.append({
                    "marker_id": 11,
                    "class_id": 0,
                    "class_name": "person",
                    "x": 5.85,
                    "y": 0.45,
                    "z": 0.4,
                    "radius": 0.45,
                    "channel": 2,
                    "conf": 0.92,
                    "hits": 8,
                    "low_confidence": False,
                })
                node.pub_cached.publish(String(data=json.dumps({"frame_id": "odom", "objects": cache_objs})))
                node.sim_sleep(0.3)

                node.hb_enabled = False
                t_blk = node.sim_now()
                w_blk = time.time()
                saw_rec = False
                mission_complete = False
                tp_init = get_gazebo_true_pose("burger")
                if tp_init is not None and math.isfinite(tp_init[0]) and math.isfinite(tp_init[1]):
                    px0, py0 = float(tp_init[0]), float(tp_init[1])
                else:
                    px0, py0 = node.cur_x, node.cur_y
                outbound_end_err_m = math.hypot(px0 - out_wps[-1][0], py0 - out_wps[-1][1])
                outbound_reached = bool(outbound_end_err_m <= 0.40)
                ret_traj = [(px0, py0)]
                min_low_d = 99.0
                min_gas_d = 99.0
                min_scan_m = 10.0

                latest_tp: List[Optional[Tuple[float, float, float]]] = [tp_init]
                stop_tp_poll = threading.Event()

                def _poll_true_pose() -> None:
                    while not stop_tp_poll.is_set():
                        p = get_gazebo_true_pose("burger")
                        if p is not None and math.isfinite(p[0]) and math.isfinite(p[1]):
                            latest_tp[0] = p
                        time.sleep(0.04)

                tp_thread = threading.Thread(target=_poll_true_pose, daemon=True)
                tp_thread.start()

                try:
                    while (node.sim_now() - t_blk < args.max_sim_sec) and (time.time() - w_blk < 300.0):
                        for _ in range(25):
                            rclpy.spin_once(node, timeout_sec=0.002)
                            if node.ctrl_state == "MISSION_COMPLETE":
                                break
                        tp = latest_tp[0]
                        if tp is not None and math.isfinite(tp[0]) and math.isfinite(tp[1]):
                            px, py = float(tp[0]), float(tp[1])
                        else:
                            px, py = node.cur_x, node.cur_y
                        if (
                            node.ctrl_state in (
                                "BACKTRACKING_ROTATE",
                                "BACKTRACKING_HYBRID",
                                "BACKTRACKING_TRANSLATE",
                                "LOCAL_TD3_AVOIDANCE",
                                "MISSION_COMPLETE",
                            )
                            or math.hypot(px - px0, py - py0) >= 0.15
                            or math.hypot(node.cur_x - px0, node.cur_y - py0) >= 0.15
                        ):
                            saw_rec = True
                        if saw_rec:
                            d_l = min(math.hypot(px - lx, py - ly) for lx, ly in low_list)
                            min_low_d = min(min_low_d, d_l)
                            d_g = math.hypot(px - gas_xy[0], py - gas_xy[1])
                            min_gas_d = min(min_gas_d, d_g)
                            min_scan_m = min(min_scan_m, node.scan_all_min)

                        if math.hypot(px - ret_traj[-1][0], py - ret_traj[-1][1]) >= 0.10:
                            ret_traj.append((px, py))
                        if node.ctrl_state == "MISSION_COMPLETE":
                            mission_complete = True
                            break
                        self_sleep_t = node.sim_now()
                        self_sleep_w = time.time()
                        while (node.sim_now() - self_sleep_t < 0.04) and (time.time() - self_sleep_w < 0.2):
                            rclpy.spin_once(node, timeout_sec=0.002)
                            if node.ctrl_state == "MISSION_COMPLETE":
                                mission_complete = True
                                break
                        if mission_complete:
                            break
                finally:
                    stop_tp_poll.set()
                    tp_thread.join(timeout=1.5)

                ret_dur = node.sim_now() - t_blk
                tp_end = get_gazebo_true_pose("burger")
                if tp_end is not None and math.isfinite(tp_end[0]) and math.isfinite(tp_end[1]):
                    fx, fy = float(tp_end[0]), float(tp_end[1])
                else:
                    fx, fy = node.cur_x, node.cur_y
                if saw_rec:
                    min_low_d = min(min_low_d, min(math.hypot(fx - lx, fy - ly) for lx, ly in low_list))
                    min_gas_d = min(min_gas_d, math.hypot(fx - gas_xy[0], fy - gas_xy[1]))
                    min_scan_m = min(min_scan_m, node.scan_all_min)
                err_m = math.hypot(fx, fy)
                arrived = bool(mission_complete and err_m <= 0.85)

                min_wall_geom = compute_semantic_world_min_wall_geom(ret_traj + [(fx, fy)])

                low_col = bool(min_low_d < 0.255)
                gas_viol = bool(min_gas_d < 0.55)
                wall_col = bool(min_wall_geom < 0.105 or min_scan_m < 0.125)
                cf_pass = bool(arrived and (not low_col) and (not wall_col))
                strict_pass = bool(cf_pass and (not gas_viol))

                rec = {
                    "trial": t_idx,
                    "seed": spec["seed"],
                    "mode": mode_key,
                    "onnx": os.path.basename(onnx_path),
                    "cbf_v_min": cbf_v_min,
                    "use_semantic_rl_policy": True,
                    "enable_semantic_astar_cost": bool(args.enable_semantic_astar_cost),
                    "enable_dynamic_tracker": True,
                    "enable_cbf_safety_shield": True,
                    "outbound_reached": outbound_reached,
                    "outbound_end_err_m": round(outbound_end_err_m, 4),
                    "max_roll_pitch_deg": round(node.max_roll_pitch_deg, 2),
                    "cbf_mq2_escape_steps": int(node.cbf_mq2_escape_steps),
                    "max_confirmed_dyn_tracks": int(node.max_confirmed_dyn_tracks),
                    "crossing_scheduler_modes": sorted(node.crossing_scheduler_modes),
                    "mission_complete": mission_complete,
                    "arrival_success": arrived,
                    "low_obstacle_collision": low_col,
                    "wall_collision": wall_col,
                    "gas_zone_violated": gas_viol,
                    "collision_free_success": cf_pass,
                    "strict_hazard_free_success": strict_pass,
                    "min_low_obstacle_dist_m": round(min_low_d, 4),
                    "min_gas_zone_dist_m": round(min_gas_d, 4),
                    "min_wall_geom_m": round(min_wall_geom, 4),
                    "min_scan_m": round(min_scan_m, 4),
                    "return_duration_s": round(ret_dur, 2),
                    "terminal_error_cm": round(err_m * 100.0, 2),
                    "low_obstacles": low_list,
                    "gas_zone": gas_xy,
                    "outbound_path": [[round(p[0], 3), round(p[1], 3)] for p in outbound_path],
                    "return_path": [[round(p[0], 3), round(p[1], 3)] for p in ret_traj],
                }
                results[mode_key].append(rec)
                print(
                    f"  [{mode_key:11s} #{t_idx:02d}/{args.trials}] out_ok={outbound_reached}({outbound_end_err_m:.2f}m) "
                    f"tilt={node.max_roll_pitch_deg:.1f}deg | strict={strict_pass} cf={cf_pass} | "
                    f"low={low_col}({min_low_d:.3f}) gas={gas_viol}({min_gas_d:.3f}) "
                    f"wall={wall_col}(wg={min_wall_geom:.3f},sc={min_scan_m:.3f}) | "
                    f"mq2={node.cbf_mq2_escape_steps} dyn={node.max_confirmed_dyn_tracks} | "
                    f"err={err_m*100:.1f}cm t={ret_dur:.1f}s",
                    flush=True,
                )
        finally:
            node.destroy_node()
            rclpy.shutdown()
            _stop_procs(procs, gz_port)

    summary = {}
    for k, arr in results.items():
        summary[k] = {
            "trials": len(arr),
            "outbound_reached_count": sum(1 for r in arr if r["outbound_reached"]),
            "arrival_passes": sum(1 for r in arr if r["arrival_success"]),
            "collision_free_passes": sum(1 for r in arr if r["collision_free_success"]),
            "strict_hazard_free_passes": sum(1 for r in arr if r["strict_hazard_free_success"]),
            "low_obstacle_collisions": sum(1 for r in arr if r["low_obstacle_collision"]),
            "wall_collisions": sum(1 for r in arr if r["wall_collision"]),
            "gas_zone_violations": sum(1 for r in arr if r["gas_zone_violated"]),
            "total_cbf_mq2_escape_steps": sum(int(r["cbf_mq2_escape_steps"]) for r in arr),
            "max_confirmed_dyn_tracks": max((int(r["max_confirmed_dyn_tracks"]) for r in arr), default=0),
            "mean_min_low_dist_m": round(float(np.mean([r["min_low_obstacle_dist_m"] for r in arr])), 4),
            "mean_min_gas_dist_m": round(float(np.mean([r["min_gas_zone_dist_m"] for r in arr])), 4),
            "mean_min_scan_m": round(float(np.mean([r["min_scan_m"] for r in arr])), 4),
            "mean_return_time_s": round(float(np.mean([r["return_duration_s"] for r in arr])), 2),
            "mean_terminal_error_cm": round(float(np.mean([r["terminal_error_cm"] for r in arr])), 2),
        }

    out_dir = os.path.join(WS_ROOT, "benchmark_results", "phase4_semantic")
    os.makedirs(out_dir, exist_ok=True)
    out_json = args.out_json if args.out_json else os.path.join(out_dir, "semantic_avoidance_phase4_10trials.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "world": "disaster_semantic_rescue_world.world",
                "enable_semantic_astar_cost": bool(args.enable_semantic_astar_cost),
                "summary": summary,
                "trials": results,
            },
            f,
            indent=2,
        )
    print("\n=== PHASE 4 SEMANTIC SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved: {out_json}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="phase4_sim,phase4_real")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--gz-port", type=int, default=11362)
    parser.add_argument("--ros-domain-id", type=int, default=62)
    parser.add_argument("--rtf", type=float, default=3.5)
    parser.add_argument("--max-sim-sec", type=float, default=130.0)
    parser.add_argument("--enable-semantic-astar-cost", dest="enable_semantic_astar_cost", action="store_true", default=True)
    parser.add_argument("--disable-semantic-astar-cost", dest="enable_semantic_astar_cost", action="store_false")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
