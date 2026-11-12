#!/usr/bin/env python3
"""
=================================================================================================
Project: Disaster Recovery TurtleBot3 Burger System
Script : auto_disaster_trials.py (Autonomous 10-Trial Disaster Recovery Live Benchmark)
Author : Autonomous Systems Laboratory / Jong Woo Ro
Usage  : python3 scripts/host/auto_disaster_trials.py [--trials 10] [--tunnel | --house]
=================================================================================================
Automated End-to-End Mission Benchmark:
  1. AI autonomously teleoperates TurtleBot3 Burger into the disaster terrain (outbound drive).
  2. Drops Heartbeat (Simulating Communication Blackout '[H]').
  3. Monitors Fail-Safe Watchdog trigger & 4-Layer Autonomous Backtracking Return.
  4. Records Return Time, Terminal Pose Error (cm), and Success Rate across 10 trials.
  5. Plots high-resolution trajectory map and benchmark graphs directly to Windows Desktop.
=================================================================================================
"""

import os
import sys
import time
import math
import json
import subprocess
import argparse
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Header, String

# Localhost only
os.environ["ROS_LOCALHOST_ONLY"] = "1"
os.environ["ROS_DOMAIN_ID"] = "0"


def normalize_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class AutonomousDisasterBenchmark(Node):
    def __init__(self):
        super().__init__('auto_disaster_benchmark')

        self.pub_teleop = self.create_publisher(Twist, '/cmd_vel_remote', 10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_heartbeat = self.create_publisher(Header, '/heartbeat', 10)

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
        self.scan_ready = False

        # 10Hz Heartbeat Timer
        self.timer_hb = self.create_timer(0.10, self._hb_tick)

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

        valid = (ranges >= 0.10) & (ranges <= 3.5) & np.isfinite(ranges)
        fwd_mask = valid & (np.abs(angles) <= math.radians(25.0))
        left_mask = valid & (angles > math.radians(25.0)) & (angles <= math.radians(75.0))
        right_mask = valid & (angles < -math.radians(25.0)) & (angles >= -math.radians(75.0))

        self.scan_front_min = float(np.min(ranges[fwd_mask])) if np.any(fwd_mask) else 10.0
        self.scan_left_min = float(np.min(ranges[left_mask])) if np.any(left_mask) else 10.0
        self.scan_right_min = float(np.min(ranges[right_mask])) if np.any(right_mask) else 10.0
        self.scan_ready = True

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


def run_benchmark(trials: int = 5, target_yaw: float = 1.57, deep: bool = True):
    rclpy.init()
    node = AutonomousDisasterBenchmark()

    deep_tag = " [심층 침투 100m 누적 모드]" if deep else ""
    env_name = "터널 붕괴 험지 환경 (Gazebo-Tunnel-World)" if abs(target_yaw - 1.57) < 0.2 else "실내 가옥 재난 환경 (House World)"
    print(f"\n{'='*95}")
    print(f"🚀 [자율 재난복귀 라이브 관제 벤치마크]{deep_tag} {env_name}")
    print(f"{'='*95}")
    print(f"• 총 시험 횟수 : {trials} 회차")
    print(f"• 탐사 모드   : AI 자율 차선 중앙추종 + 능동 장애물 회피 (최대 깊이 32m, 누적 102m)")
    print(f"• 복귀 트리거 : Heartbeat 송출 중단 (워치독 3.0s 타임아웃)")
    print(f"• 복귀 방식   : 4계층 하이브리드 자율복귀 (Breadcrumb + TD3 + 2D ICP + 원점 안착)")
    print(f"• 시각 관제   : Gazebo 3D GUI 뷰포트에서 실시간 3D 기동 가시화")
    print(f"{'='*95}\n")

    print("[대기] 센서(/odom, /scan) 및 복귀 제어기 통신 초기화 중...", end="", flush=True)
    t_start = time.time()
    while not node.odom_ready and (time.time() - t_start < 12.0):
        rclpy.spin_once(node, timeout_sec=0.1)
    if not node.odom_ready:
        print("\n[오류] /odom 수신 실패. Gazebo가 켜져 있는지 확인하세요.")
        return

    # Wait for recovery controller to stabilize in NORMAL_TELEOP
    for _ in range(25):
        rclpy.spin_once(node, timeout_sec=0.08)
    print(" [센서 연결 완료!]\n")

    # Record Base Camp Pose
    base_camp_x = node.corr_x if node.corr_ready else node.cur_x
    base_camp_y = node.corr_y if node.corr_ready else node.cur_y
    print(f"[베이스캠프 등록] 원점 좌표: ({base_camp_x:.3f}, {base_camp_y:.3f})\n")

    results = []

    for trial_idx in range(1, trials + 1):
        print(f"▶ [회차 {trial_idx:02d}/{trials:02d}] " + "-"*65)
        node.heartbeat_enabled = True
        outbound_path = [[float(node.cur_x), float(node.cur_y)]]
        return_path = []

        # Ensure system is in NORMAL_TELEOP mode before start
        t_wait_norm = time.time()
        while node.system_state != "NORMAL_TELEOP" and (time.time() - t_wait_norm < 8.0):
            rclpy.spin_once(node, timeout_sec=0.1)

        # Orient towards target exploration corridor
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

        # Phase 1: Outbound Autonomous AI Exploration Drive
        deep_targets = [10.0, 15.0, 20.0, 25.0, 32.0]
        target_depth = deep_targets[(trial_idx - 1) % len(deep_targets)] if deep else (0.5 + (trial_idx % 4) * 0.15)
        
        if deep:
            print(f"  [1단계: 터널 심층 침투] 목표 전진 거리: {target_depth:.1f} m (원점 기준 목표 깊이: y={base_camp_y + target_depth:.1f}m)...")
            t_exp_start = time.time()
            max_exp_time = target_depth / 0.12 + 15.0

            while time.time() - t_exp_start < max_exp_time:
                rclpy.spin_once(node, timeout_sec=0.03)
                outbound_path.append([float(node.cur_x), float(node.cur_y)])

                cur_depth = node.cur_y - base_camp_y
                if cur_depth >= target_depth:
                    print(f"    -> 목표 심층 거리({target_depth:.1f}m) 도달! 현재 좌표: ({node.cur_x:.2f}, {node.cur_y:.2f})")
                    break

                # Corridor lane centering around x = -0.75
                e_x = -0.75 - node.cur_x
                w_align = 0.50 * e_x + 0.30 * (node.scan_right_min - node.scan_left_min)
                w_cmd = max(-0.35, min(0.35, w_align))
                v_cmd = 0.20

                # Front collision avoidance
                if node.scan_front_min < 0.45:
                    v_cmd = 0.08
                    w_cmd = 0.45 if node.scan_left_min > node.scan_right_min else -0.45

                node.drive_raw(v_cmd, w_cmd)
                time.sleep(0.05)
        else:
            explore_duration = 6.0 + (trial_idx % 4) * 1.3
            print(f"  [1단계: 탐사 진입] AI 자율 원격 조종으로 터널 안쪽 진입 중 ({explore_duration:.1f}초)...")
            t_exp_start = time.time()
            while time.time() - t_exp_start < explore_duration:
                t_rel = time.time() - t_exp_start
                rclpy.spin_once(node, timeout_sec=0.03)
                outbound_path.append([float(node.cur_x), float(node.cur_y)])

                v_cmd = 0.16
                if trial_idx % 3 == 1:
                    w_cmd = 0.16 * math.sin(t_rel * 1.0)
                elif trial_idx % 3 == 2:
                    w_cmd = 0.12 * math.cos(t_rel * 0.8)
                else:
                    w_cmd = -0.15 * math.sin(t_rel * 0.9)

                if node.scan_front_min < 0.32:
                    v_cmd = 0.05
                    w_cmd = 0.38 if node.scan_left_min > node.scan_right_min else -0.38

                node.drive_raw(v_cmd, w_cmd)
                time.sleep(0.05)

        node.stop_drive()
        for _ in range(8):
            rclpy.spin_once(node, timeout_sec=0.05)
            outbound_path.append([float(node.cur_x), float(node.cur_y)])

        turnoff_x, turnoff_y = node.cur_x, node.cur_y
        explored_dist = math.hypot(turnoff_x - trial_start_x, turnoff_y - trial_start_y)
        dist_from_base = math.hypot(turnoff_x - base_camp_x, turnoff_y - base_camp_y)
        print(f"  [진입 완료] 탐사 전진 거리: {explored_dist:.2f} m (원점 이격: {dist_from_base:.2f} m) | 현재 좌표: ({turnoff_x:.2f}, {turnoff_y:.2f})")

        # Phase 2: Communication Blackout Trigger (Heartbeat DROP)
        print(f"  [2단계: 통신 단절] ⚠️ Heartbeat 송출 중단 (통신 두절 모사)!")
        node.heartbeat_enabled = False
        t_drop = time.time()

        # Wait for watchdog fail-safe trigger (3.0s timeout)
        while node.system_state == "NORMAL_TELEOP" and (time.time() - t_drop < 6.0):
            rclpy.spin_once(node, timeout_sec=0.1)
            outbound_path.append([float(node.cur_x), float(node.cur_y)])

        print(f"  [안전 워치독 발동] 제어기 상태: {node.system_state} -> 4계층 자율복귀 시작!")

        # Phase 3: Autonomous Return Tracking
        t_return_start = time.time()
        reached = False
        max_return_wait = max(90.0, target_depth * 13.0) if deep else 60.0

        while time.time() - t_return_start < max_return_wait:
            rclpy.spin_once(node, timeout_sec=0.1)
            cx = node.corr_x if node.corr_ready else node.cur_x
            cy = node.corr_y if node.corr_ready else node.cur_y
            return_path.append([float(cx), float(cy)])
            dist_to_base = math.hypot(cx - base_camp_x, cy - base_camp_y)

            # Check if mission complete or reached base camp radius after initiating return
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
            "explore_dist_m": explored_dist,
            "return_time_s": t_return_elapsed,
            "pos_err_cm": final_dist_err_cm,
            "success": success,
            "outbound_path": outbound_path,
            "return_path": return_path
        })

        # Inter-trial Cooldown: restore heartbeat and wait for recovery controller to reset
        node.heartbeat_enabled = True
        t_rst_wait = time.time()
        while node.system_state != "NORMAL_TELEOP" and (time.time() - t_rst_wait < 5.0):
            rclpy.spin_once(node, timeout_sec=0.1)
            time.sleep(0.05)
        time.sleep(1.0)

    # Summary Report
    print(f"\n{'='*95}")
    print(f"📊 [10회 자율복귀 라이브 관제 벤치마크 최종 정량 통계 보고서]")
    print(f"{'='*95}")
    print(f"{'회차':<6} | {'진입 탐사거리':<14} | {'복귀 소요시간':<14} | {'원점 종단오차 (cm)':<20} | {'최종 판정'}")
    print(f"{'-'*95}")

    for r in results:
        res_tag = "성공 (PASS)" if r['success'] else "실패 (FAIL)"
        print(f"#{r['trial']:02d}   | {r['explore_dist_m']:>6.2f} m       | {r['return_time_s']:>6.1f} 초       | {r['pos_err_cm']:>8.2f} cm         | {res_tag}")

    print(f"{'-'*95}")
    succ_count = sum(1 for r in results if r['success'])
    succ_rate = (succ_count / trials) * 100.0
    avg_err = (sum(r['pos_err_cm'] for r in results if r['success']) / succ_count) if succ_count > 0 else 0.0
    avg_time = (sum(r['return_time_s'] for r in results if r['success']) / succ_count) if succ_count > 0 else 0.0

    print(f"• 전체 복귀 성공률 : {succ_rate:.1f}% ({succ_count}/{trials} 회 전수 완료)")
    print(f"• 평균 종단 위치오차: {avg_err:.2f} cm")
    print(f"• 평균 복귀 소요시간: {avg_time:.1f} 초")
    print(f"{'='*95}\n")

    # Export trajectory data and generate graph
    dump_data = {
        "base_camp_x": float(base_camp_x),
        "base_camp_y": float(base_camp_y),
        "trials": results
    }
    json_path = "/tmp/tunnel_trials_result.json"
    with open(json_path, 'w') as f:
        json.dump(dump_data, f)
    print(f"[데이터 저장] 벤치마크 궤적 데이터 저장: {json_path}")

    # Plot using ros2_ai_venv python
    plot_script = "/home/lab/disaster_recovery_ws/scripts/host/plot_tunnel_trials.py"
    cmd_plot = f"/home/lab/ros2_ai_venv/bin/python3 {plot_script}"
    try:
        subprocess.call(cmd_plot, shell=True)
    except Exception as e:
        print(f"[경고] 플롯 생성 실패: {e}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Autonomous Disaster Trials")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials (default: 5)")
    parser.add_argument("--tunnel", action="store_true", help="Tunnel World mode (outbound yaw: 90 deg / 1.57 rad)")
    parser.add_argument("--house", action="store_true", help="House World mode (outbound yaw: 0 deg / 0.0 rad)")
    parser.add_argument("--deep", action="store_true", default=True, help="Deep corridor penetration mode (10m ~ 32m per trial)")
    parser.add_argument("--quick", action="store_true", help="Quick stress test mode (0.5m ~ 0.8m per trial)")
    parser.add_argument("--yaw", type=float, default=None, help="Custom outbound yaw in radians")
    args = parser.parse_args()

    if args.yaw is not None:
        target_yaw = args.yaw
    elif args.house:
        target_yaw = 0.0
    else:
        target_yaw = 1.57  # Default to tunnel

    is_deep = False if args.quick else args.deep
    run_benchmark(args.trials, target_yaw, deep=is_deep)
