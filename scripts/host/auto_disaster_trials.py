#!/usr/bin/env python3
"""
=================================================================================================
Project: Disaster Recovery TurtleBot3 Burger System
Script : auto_disaster_trials.py (Autonomous Disaster Recovery Live Benchmark)
Author : Autonomous Systems Laboratory / Jong Woo Ro
Usage  : python3 scripts/host/auto_disaster_trials.py [--trials 10] [--collapsed | --tunnel | --house]
=================================================================================================
Automated End-to-End Mission Benchmark:
  1. AI autonomously drives TurtleBot3 Burger 12~20m into the disaster terrain (outbound drive).
  2. Randomly drops 3~6 dynamic rock/debris obstacles along the traversed return path.
  3. Drops Heartbeat (Simulating Communication Blackout '[H]').
  4. Monitors Fail-Safe Watchdog trigger & 4-Layer Autonomous Backtracking Return.
  5. Records Return Time, Terminal Pose Error (cm), and Success Rate across trials.
  6. Plots high-resolution trajectory map and benchmark graphs.
=================================================================================================
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Header, String

_HOST_DIR = os.path.dirname(os.path.abspath(__file__))
if _HOST_DIR not in sys.path:
    sys.path.insert(0, _HOST_DIR)

from debris_utils import (
    delete_debris_entities,
    select_random_debris_positions,
    spawn_random_debris_cluster,
)

# Localhost only
os.environ["ROS_LOCALHOST_ONLY"] = "1"
os.environ["ROS_DOMAIN_ID"] = "0"

WORLD_CONFIGS = {
    "collapsed": {
        "env_name": "붕괴 건물 재난 환경 (Collapsed Building World)",
        "reset_pose": (0.0, -4.0, 1.5708),
        "default_yaw": 1.5708,
        "json_path": "/tmp/collapsed_trials_result.json",
        "plot_output": "/home/lab/disaster_recovery_ws/docs/images/collapsed_disaster_benchmark_trajectories.png",
    },
    "tunnel": {
        "env_name": "터널 붕괴 험지 환경 (Gazebo-Tunnel-World)",
        "reset_pose": (-0.5, -3.0, 1.57),
        "default_yaw": 1.57,
        "json_path": "/tmp/tunnel_trials_result.json",
        "plot_output": "/home/lab/disaster_recovery_ws/docs/images/tunnel_disaster_benchmark_trajectories.png",
    },
    "house": {
        "env_name": "실내 가옥 재난 환경 (House World)",
        "reset_pose": (-2.0, -0.5, 0.0),
        "default_yaw": 0.0,
        "json_path": "/tmp/house_trials_result.json",
        "plot_output": "/home/lab/disaster_recovery_ws/docs/images/house_disaster_benchmark_trajectories.png",
    },
}


def normalize_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class AutonomousDisasterBenchmark(Node):
    def __init__(self):
        super().__init__('auto_disaster_benchmark')

        self.pub_teleop = self.create_publisher(Twist, '/cmd_vel_remote', 10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_heartbeat = self.create_publisher(Header, '/heartbeat', 10)
        self.pub_host_hb_enable = self.create_publisher(Bool, '/host/enable_heartbeat', 10)

        self.sub_odom = self.create_subscription(Odometry, '/odom', self._cb_odom, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self._cb_scan, qos_profile_sensor_data)
        self.sub_status = self.create_subscription(String, '/recovery/status', self._cb_status, 10)

        self.cur_x = 0.0
        self.cur_y = 0.0
        self.cur_th = 0.0
        self.odom_ready = False
        self.corr_x = 0.0
        self.corr_y = 0.0
        self.corr_th = 0.0
        self.corr_ready = False
        self.system_state = "UNKNOWN"
        self.heartbeat_enabled = True
        self.hb_seq = 0

        # Laser Scan clearance info
        self.scan_front_min = 10.0
        self.scan_left_min = 10.0
        self.scan_right_min = 10.0
        self.scan_ranges = None
        self.scan_angles = None
        self.scan_ready = False

        # 10Hz Heartbeat Timer
        self.timer_hb = self.create_timer(0.10, self._hb_tick)

    def set_heartbeat(self, enabled: bool):
        self.heartbeat_enabled = enabled
        msg = Bool()
        msg.data = enabled
        self.pub_host_hb_enable.publish(msg)

    def _hb_tick(self):
        if self.heartbeat_enabled:
            hdr = Header()
            hdr.stamp = self.get_clock().now().to_msg()
            hdr.frame_id = f"auto_benchmark_hb_{self.hb_seq}"
            self.pub_heartbeat.publish(hdr)
            self.hb_seq += 1

    def _cb_odom(self, msg: Odometry):
        self.cur_x = msg.pose.pose.position.x
        self.cur_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.cur_th = math.atan2(siny_cosp, cosy_cosp)
        self.odom_ready = True

    def _cb_scan(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        angles = (angles + np.pi) % (2.0 * np.pi) - np.pi

        clean_ranges = np.where(
            (ranges >= 0.10) & (ranges <= 3.5) & np.isfinite(ranges),
            ranges,
            3.5,
        )
        valid = (ranges >= 0.10) & (ranges <= 3.5) & np.isfinite(ranges)
        fwd_mask = valid & (np.abs(angles) <= math.radians(25.0))
        left_mask = valid & (angles > math.radians(25.0)) & (angles <= math.radians(85.0))
        right_mask = valid & (angles < -math.radians(25.0)) & (angles >= -math.radians(85.0))

        self.scan_front_min = float(np.min(ranges[fwd_mask])) if np.any(fwd_mask) else 10.0
        self.scan_left_min = float(np.min(ranges[left_mask])) if np.any(left_mask) else 10.0
        self.scan_right_min = float(np.min(ranges[right_mask])) if np.any(right_mask) else 10.0
        self.scan_ranges = clean_ranges
        self.scan_angles = angles
        self.scan_ready = True

    def choose_open_gap_cmd(self, goal_x: float, goal_y: float):
        """Select reactive steering toward wide LiDAR gaps biased toward an interior goal."""
        if self.scan_ranges is None or self.scan_angles is None:
            return 0.14, 0.0

        # Emergency backup/pivot if very close ahead
        if self.scan_front_min < 0.28:
            w_turn = 0.55 if self.scan_left_min >= self.scan_right_min else -0.55
            return -0.08, w_turn

        goal_rel = normalize_angle(
            math.atan2(goal_y - self.cur_y, goal_x - self.cur_x) - self.cur_th
        )
        cand_angles = np.deg2rad(np.arange(-95, 96, 5))
        half_win = math.radians(18.0)

        best_score = -1e9
        best_ang = 0.0
        for cand in cand_angles:
            mask = np.abs(((self.scan_angles - cand + np.pi) % (2.0 * np.pi)) - np.pi) <= half_win
            sector_min = float(np.min(self.scan_ranges[mask])) if np.any(mask) else 3.5
            if sector_min < 0.34:
                continue
            goal_align = math.cos(normalize_angle(float(cand) - goal_rel))
            score = min(sector_min, 2.5) + 0.70 * goal_align - 0.12 * abs(float(cand))
            if score > best_score:
                best_score = score
                best_ang = float(cand)

        if best_score < -1e8:
            w_turn = 0.55 if self.scan_left_min >= self.scan_right_min else -0.55
            return 0.0, w_turn

        w_cmd = max(-0.55, min(0.55, 1.25 * best_ang))
        v_cmd = 0.18 if self.scan_front_min >= 0.50 else 0.10
        return v_cmd, w_cmd

    def _cb_status(self, msg: String):
        parts = msg.data.split('|')
        for p in parts:
            if p.startswith("STATE:"):
                self.system_state = p.split(':')[1]
            elif p.startswith("POSE:"):
                try:
                    coords = p.split(':')[1].split(',')
                    self.corr_x = float(coords[0])
                    self.corr_y = float(coords[1])
                    self.corr_th = float(coords[2])
                    self.corr_ready = True
                except Exception:
                    pass

    def drive_raw(self, v: float, w: float):
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.pub_cmd_vel.publish(cmd)
        self.pub_teleop.publish(cmd)

    def stop_drive(self):
        self.drive_raw(0.0, 0.0)


def run_benchmark(
    trials: int = 10,
    world_name: str = "collapsed",
    target_yaw: float = 1.5708,
    deep: bool = True,
    min_depth: float = 12.0,
    max_depth: float = 20.0,
    drop_debris: bool = True,
    debris_min: int = 3,
    debris_max: int = 6,
):
    cfg = WORLD_CONFIGS.get(world_name, WORLD_CONFIGS["collapsed"])
    reset_x, reset_y, reset_yaw = cfg["reset_pose"]

    rclpy.init()
    node = AutonomousDisasterBenchmark()

    if drop_debris:
        delete_debris_entities(max_count=max(8, debris_max + 2), wait=True)

    debris_tag = f" [동적 암석 랜덤 투하 {debris_min}~{debris_max}개]" if drop_debris else ""
    deep_tag = f" [편도 {min_depth:.0f}~{max_depth:.0f}m 침투 모드]" if deep else ""
    print(f"\n{'='*95}")
    print(f"🚀 [자율 재난복귀 라이브 관제 벤치마크]{deep_tag}{debris_tag} {cfg['env_name']}")
    print(f"{'='*95}")
    print(f"• 총 시험 횟수 : {trials} 회차")
    print(f"• 탐사 모드   : LiDAR 오픈 갭 자율 탐사 + 능동 장애물 회피 (편도 {min_depth:.1f}~{max_depth:.1f}m)")
    print(f"• 낙석 모사   : 복귀로 상 실시간 랜덤 암석 {debris_min}~{debris_max}개 불시 투하")
    print(f"• 복귀 방식   : 4계층 하이브리드 자율복귀 (Breadcrumb + TD3 우회 + 2D ICP 슬립리셋 + 원점 안착)")
    print(f"• 시각 관제   : Gazebo 3D GUI 뷰포트 및 RViz2 SLAM 지도 실시간 3D 기동 가시화")
    print(f"{'='*95}\n")

    print("[대기] 센서(/odom, /scan) 및 복귀 제어기 통신 초기화 중...", end="", flush=True)
    t_start = time.time()
    while not node.odom_ready and (time.time() - t_start < 12.0):
        rclpy.spin_once(node, timeout_sec=0.1)
    if not node.odom_ready:
        print("\n[오류] /odom 수신 실패. Gazebo가 켜져 있는지 확인하세요.")
        node.destroy_node()
        rclpy.shutdown()
        return

    for _ in range(25):
        rclpy.spin_once(node, timeout_sec=0.08)
    print(" [센서 연결 완료!]\n")

    base_camp_x = node.corr_x if node.corr_ready else node.cur_x
    base_camp_y = node.corr_y if node.corr_ready else node.cur_y
    print(f"[베이스캠프 등록] 원점 좌표: ({base_camp_x:.3f}, {base_camp_y:.3f})\n")

    results = []

    for trial_idx in range(1, trials + 1):
        print(f"▶ [회차 {trial_idx:02d}/{trials:02d}] " + "-"*65)
        node.set_heartbeat(True)
        outbound_path = [[float(node.cur_x), float(node.cur_y)]]
        return_path = []

        t_wait_norm = time.time()
        while node.system_state != "NORMAL_TELEOP" and (time.time() - t_wait_norm < 8.0):
            rclpy.spin_once(node, timeout_sec=0.1)

        cur_heading = node.cur_th
        heading_err = abs(normalize_angle(target_yaw - cur_heading))
        if heading_err > math.radians(25.0):
            print(f"  [출발 정렬] 탐사 경로 방향 정렬 중 (현재: {math.degrees(cur_heading):.1f}° -> 목표: {math.degrees(target_yaw):.1f}°)...")
            t_align = time.time()
            while abs(normalize_angle(target_yaw - node.cur_th)) > math.radians(12.0) and (time.time() - t_align < 4.5):
                diff = normalize_angle(target_yaw - node.cur_th)
                w_align = 0.40 if diff > 0 else -0.40
                node.drive_raw(0.0, w_align)
                rclpy.spin_once(node, timeout_sec=0.05)
                time.sleep(0.05)
            node.stop_drive()
            time.sleep(0.3)

        trial_start_x, trial_start_y = node.cur_x, node.cur_y
        target_depth = (
            round(random.uniform(min_depth, max_depth), 2)
            if deep
            else round(0.5 + (trial_idx % 4) * 0.15, 2)
        )

        # Pick an interior exploration anchor for collapsed building diversity
        if world_name == "collapsed":
            goal_x = random.uniform(-1.5, 8.5)
            goal_y = random.uniform(8.5, 12.0)
        else:
            goal_x = trial_start_x + target_depth * math.cos(target_yaw)
            goal_y = trial_start_y + target_depth * math.sin(target_yaw)

        explored_path_m = 0.0
        prev_x, prev_y = node.cur_x, node.cur_y

        if deep:
            print(f"  [1단계: 재난 지역 심층 침투] 목표 편도 주행 거리: {target_depth:.2f} m (내부 목표 방향: ({goal_x:.1f}, {goal_y:.1f}))...")
            t_exp_start = time.time()
            max_exp_time = target_depth / 0.10 + 30.0
            stuck_anchor_x, stuck_anchor_y = node.cur_x, node.cur_y
            stuck_anchor_t = time.time()

            while time.time() - t_exp_start < max_exp_time:
                rclpy.spin_once(node, timeout_sec=0.03)
                step_d = math.hypot(node.cur_x - prev_x, node.cur_y - prev_y)
                explored_path_m += step_d
                prev_x, prev_y = node.cur_x, node.cur_y
                outbound_path.append([float(node.cur_x), float(node.cur_y)])

                if explored_path_m >= target_depth:
                    print(f"    -> 목표 편도 침투 거리({explored_path_m:.2f}m / 목표 {target_depth:.2f}m) 도달! 현재 좌표: ({node.cur_x:.2f}, {node.cur_y:.2f})")
                    break

                if world_name == "tunnel":
                    target_heading = 1.57 + 0.45 * (node.cur_x - (-0.75))
                    target_heading = max(1.10, min(2.04, target_heading))
                    heading_diff = normalize_angle(target_heading - node.cur_th)
                    w_align = 1.2 * heading_diff + 0.25 * (node.scan_right_min - node.scan_left_min)
                    w_cmd = max(-0.35, min(0.35, w_align))
                    v_cmd = 0.20
                    if node.scan_front_min < 0.45:
                        v_cmd = 0.08
                        w_cmd = 0.45 if node.scan_left_min > node.scan_right_min else -0.45
                else:
                    # Adaptive LiDAR gap-following + interior goal bias
                    if math.hypot(node.cur_x - goal_x, node.cur_y - goal_y) < 1.8:
                        goal_x = random.uniform(-2.0, 10.0)
                        goal_y = random.uniform(2.0, 12.0)

                    # Stuck recovery guard (no net progress for 3.5s)
                    if time.time() - stuck_anchor_t > 3.5:
                        if math.hypot(node.cur_x - stuck_anchor_x, node.cur_y - stuck_anchor_y) < 0.12:
                            turn_dir = 0.60 if node.scan_left_min >= node.scan_right_min else -0.60
                            for _ in range(10):
                                node.drive_raw(-0.10, turn_dir)
                                rclpy.spin_once(node, timeout_sec=0.05)
                                time.sleep(0.05)
                            goal_x = random.uniform(-2.5, 10.5)
                            goal_y = random.uniform(1.0, 12.0)
                        stuck_anchor_x, stuck_anchor_y = node.cur_x, node.cur_y
                        stuck_anchor_t = time.time()

                    v_cmd, w_cmd = node.choose_open_gap_cmd(goal_x, goal_y)

                node.drive_raw(v_cmd, w_cmd)
                time.sleep(0.05)
        else:
            explore_duration = 6.0 + (trial_idx % 4) * 1.3
            print(f"  [1단계: 단기 탐사 진입] AI 자율 탐사 중 ({explore_duration:.1f}초)...")
            t_exp_start = time.time()
            while time.time() - t_exp_start < explore_duration:
                rclpy.spin_once(node, timeout_sec=0.03)
                explored_path_m += math.hypot(node.cur_x - prev_x, node.cur_y - prev_y)
                prev_x, prev_y = node.cur_x, node.cur_y
                outbound_path.append([float(node.cur_x), float(node.cur_y)])
                v_cmd, w_cmd = node.choose_open_gap_cmd(goal_x, goal_y)
                node.drive_raw(v_cmd, w_cmd)
                time.sleep(0.05)

        node.stop_drive()
        for _ in range(8):
            rclpy.spin_once(node, timeout_sec=0.05)
            explored_path_m += math.hypot(node.cur_x - prev_x, node.cur_y - prev_y)
            prev_x, prev_y = node.cur_x, node.cur_y
            outbound_path.append([float(node.cur_x), float(node.cur_y)])

        turnoff_x, turnoff_y = node.cur_x, node.cur_y
        dist_from_base = math.hypot(turnoff_x - base_camp_x, turnoff_y - base_camp_y)
        print(f"  [진입 완료] 편도 주행 거리: {explored_path_m:.2f} m (원점 직선 이격: {dist_from_base:.2f} m) | 현재 좌표: ({turnoff_x:.2f}, {turnoff_y:.2f})")

        # Phase 2: Random 3~6 Rock Drop & Communication Blackout Trigger
        print("  [2단계: 통신 단절] ⚠️ Heartbeat 송출 중단 (통신 두절 모사)!")
        node.set_heartbeat(False)
        t_drop = time.time()

        debris_positions = []
        spawned_debris_names = []
        if drop_debris and len(outbound_path) > 10:
            debris_positions = select_random_debris_positions(
                outbound_path,
                min_count=debris_min,
                max_count=debris_max,
                min_spacing=1.2,
                end_margin=1.2,
            )
            spawned_debris_names = spawn_random_debris_cluster(debris_positions, wait=True)
            coords_str = ", ".join(f"({px:.2f}, {py:.2f})" for px, py in debris_positions)
            print(f"  [동적 암석 랜덤 투하] 🚨 복귀 경로 상에 암석 {len(debris_positions)}개 투하 완료! 위치: [{coords_str}]")

        # Wait for watchdog fail-safe trigger (3.0s timeout)
        while node.system_state == "NORMAL_TELEOP" and (time.time() - t_drop < 6.0):
            rclpy.spin_once(node, timeout_sec=0.1)
            outbound_path.append([float(node.cur_x), float(node.cur_y)])

        print(f"  [안전 워치독 발동] 제어기 상태: {node.system_state} -> 4계층 자율복귀 시작!")

        # Phase 3: Autonomous Return Tracking
        t_return_start = time.time()
        reached = False
        max_return_wait = max(240.0, target_depth * 22.0) if deep else 60.0

        while time.time() - t_return_start < max_return_wait:
            rclpy.spin_once(node, timeout_sec=0.1)
            cx = node.corr_x if node.corr_ready else node.cur_x
            cy = node.corr_y if node.corr_ready else node.cur_y
            return_path.append([float(cx), float(cy)])
            dist_to_base = math.hypot(cx - base_camp_x, cy - base_camp_y)

            if node.system_state in ["MISSION_COMPLETE", "DOCKING_SEARCH", "DOCKING_APPROACH"]:
                reached = True
                break
            elif (time.time() - t_return_start > 5.0) and dist_to_base <= 0.12:
                reached = True
                break

            time.sleep(0.1)

        t_return_elapsed = time.time() - t_return_start
        final_x = node.corr_x if node.corr_ready else node.cur_x
        final_y = node.corr_y if node.corr_ready else node.cur_y
        final_dist_err_cm = math.hypot(final_x - base_camp_x, final_y - base_camp_y) * 100.0
        success = (final_dist_err_cm <= 25.0) or reached

        status_str = "✅ 복귀 성공 (PASS)" if success else "❌ 복귀 실패 (FAIL)"
        print(f"  [3단계: 귀환 결과] {status_str} | 원점 종단오차: {final_dist_err_cm:.2f} cm | 소요 시간: {t_return_elapsed:.1f}초\n")

        results.append({
            "trial": trial_idx,
            "explore_dist_m": explored_path_m,
            "straight_dist_m": dist_from_base,
            "return_time_s": t_return_elapsed,
            "pos_err_cm": final_dist_err_cm,
            "success": success,
            "debris_count": len(debris_positions),
            "debris_positions": [[float(px), float(py)] for px, py in debris_positions],
            "debris_pos": [float(debris_positions[0][0]), float(debris_positions[0][1])] if debris_positions else None,
            "outbound_path": outbound_path,
            "return_path": return_path,
        })

        # Inter-trial Cooldown: remove dynamic debris, restore heartbeat and reset to base camp pose
        if drop_debris and spawned_debris_names:
            delete_debris_entities(spawned_debris_names, wait=True)
        node.set_heartbeat(True)
        subprocess.call(
            f"gz model -m burger -x {reset_x:.3f} -y {reset_y:.3f} -z 0.01 -Y {reset_yaw:.4f} 2>/dev/null || true; "
            f"gz model -m burger_cam -x {reset_x:.3f} -y {reset_y:.3f} -z 0.01 -Y {reset_yaw:.4f} 2>/dev/null || true",
            shell=True,
        )
        time.sleep(1.2)
        t_rst_wait = time.time()
        while node.system_state != "NORMAL_TELEOP" and (time.time() - t_rst_wait < 5.0):
            rclpy.spin_once(node, timeout_sec=0.1)
            time.sleep(0.05)
        time.sleep(1.0)

    # Summary Report
    print(f"\n{'='*95}")
    print(f"📊 [{trials}회 {cfg['env_name']} 자율복귀 벤치마크 최종 정량 통계 보고서]")
    print(f"{'='*95}")
    print(f"{'회차':<6} | {'편도 침투거리':<14} | {'투하 암석':<10} | {'복귀 소요시간':<14} | {'원점 종단오차 (cm)':<20} | {'최종 판정'}")
    print(f"{'-'*95}")

    for r in results:
        res_tag = "성공 (PASS)" if r['success'] else "실패 (FAIL)"
        print(
            f"#{r['trial']:02d}   | {r['explore_dist_m']:>6.2f} m       | {r['debris_count']:>2d} 개      | "
            f"{r['return_time_s']:>6.1f} 초       | {r['pos_err_cm']:>8.2f} cm         | {res_tag}"
        )

    print(f"{'-'*95}")
    succ_count = sum(1 for r in results if r['success'])
    succ_rate = (succ_count / trials) * 100.0
    avg_err = (sum(r['pos_err_cm'] for r in results if r['success']) / succ_count) if succ_count > 0 else 0.0
    avg_time = (sum(r['return_time_s'] for r in results if r['success']) / succ_count) if succ_count > 0 else 0.0

    print(f"• 전체 복귀 성공률 : {succ_rate:.1f}% ({succ_count}/{trials} 회 완료)")
    print(f"• 평균 종단 위치오차: {avg_err:.2f} cm")
    print(f"• 평균 복귀 소요시간: {avg_time:.1f} 초")
    print(f"{'='*95}\n")

    dump_data = {
        "world": world_name,
        "base_camp_x": float(base_camp_x),
        "base_camp_y": float(base_camp_y),
        "trials": results,
    }
    json_path = cfg["json_path"]
    with open(json_path, 'w') as f:
        json.dump(dump_data, f)
    print(f"[데이터 저장] 벤치마크 궤적 데이터 저장: {json_path}")

    plot_script = "/home/lab/disaster_recovery_ws/scripts/host/plot_tunnel_trials.py"
    cmd_plot = f"/home/lab/ros2_ai_venv/bin/python3 {plot_script} --json {json_path} --output {cfg['plot_output']}"
    try:
        subprocess.call(cmd_plot, shell=True)
    except Exception as e:
        print(f"[경고] 플롯 생성 실패: {e}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Autonomous Disaster Trials")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials (default: 10)")
    parser.add_argument("--collapsed", action="store_true", help="Collapsed Building World mode (default)")
    parser.add_argument("--tunnel", action="store_true", help="Tunnel World mode (legacy)")
    parser.add_argument("--house", action="store_true", help="House World mode")
    parser.add_argument("--deep", action="store_true", default=True, help="Deep penetration mode (default: True)")
    parser.add_argument("--quick", action="store_true", help="Quick stress test mode (0.5m ~ 0.8m per trial)")
    parser.add_argument("--min-depth", type=float, default=12.0, help="Minimum outbound penetration distance in meters (default: 12.0)")
    parser.add_argument("--max-depth", type=float, default=20.0, help="Maximum outbound penetration distance in meters (default: 20.0)")
    parser.add_argument("--debris", action="store_true", default=True, help="Enable random dynamic rock drops (default: True)")
    parser.add_argument("--no-debris", dest="debris", action="store_false", help="Disable dynamic debris drop")
    parser.add_argument("--debris-min", type=int, default=3, help="Minimum random rocks per trial (default: 3)")
    parser.add_argument("--debris-max", type=int, default=6, help="Maximum random rocks per trial (default: 6)")
    parser.add_argument("--yaw", type=float, default=None, help="Custom outbound yaw in radians")
    args = parser.parse_args()

    if args.tunnel:
        world_name = "tunnel"
    elif args.house:
        world_name = "house"
    else:
        world_name = "collapsed"

    target_yaw = args.yaw if args.yaw is not None else WORLD_CONFIGS[world_name]["default_yaw"]
    is_deep = False if args.quick else args.deep
    run_benchmark(
        trials=args.trials,
        world_name=world_name,
        target_yaw=target_yaw,
        deep=is_deep,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        drop_debris=args.debris,
        debris_min=args.debris_min,
        debris_max=args.debris_max,
    )
