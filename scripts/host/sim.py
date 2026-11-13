#!/usr/bin/env python3
"""
=================================================================================================
Project: Disaster Recovery TurtleBot3 Burger System
Script : sim.py (One-Touch Gazebo Simulation Cockpit, SLAM Mapping & Odometry Verification)
Author : Kyungpook National University / Jong Woo Ro
Usage  : 
    sim.py                     # Standard Mode (Collapsed Building World + SLAM Mapping + RViz + Teleop)
    sim.py --house             # Run in TurtleBot3 House World (Rooms & Corridors)
    sim.py --tunnel            # Run in Tunnel World (Legacy)
    sim.py --yolo              # Also run YOLO26n Vision AI (Spawns camera & detection window)
    sim.py --no-slam           # Disable Cartographer SLAM (Odometry only)
    sim.py --stop              # Stop all active simulation processes
=================================================================================================
"""

import os
import sys
import time
import signal
import select
import termios
import tty
import argparse
import subprocess
import threading

_HOST_DIR = os.path.dirname(os.path.abspath(__file__))
if _HOST_DIR not in sys.path:
    sys.path.insert(0, _HOST_DIR)

from debris_utils import (
    delete_debris_entities,
    select_random_debris_positions,
    spawn_random_debris_cluster,
)

# -----------------------------------------------------------------------------------------------
# CRITICAL GRAPHICS ENVIRONMENT FOR WSL2 / WSLg (HARDWARE ACCELERATION SUPPORT)
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# PREVENT WSL2 MIRRORED NETWORK COLLAPSE & DDS MULTICAST FLOODING (INTERNET DROP FIX)
# -----------------------------------------------------------------------------------------------
# In WSL2 mirrored networking, ROS 2 (FastDDS) and Gazebo broadcast massive multicast UDP packets
# across host Wi-Fi interfaces, causing Wi-Fi routers and Windows network stacks to drop DNS/TCP.
# Confining ROS 2 and Gazebo strictly to localhost (127.0.0.1) solves the internet drop completely!
os.environ["ROS_LOCALHOST_ONLY"] = "1"
os.environ["GAZEBO_IP"] = "127.0.0.1"
os.environ["GAZEBO_MASTER_URI"] = "http://127.0.0.1:11345"
os.environ["ROS_DOMAIN_ID"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"

# Windows 11 WSLg supports Direct3D12 hardware GPU acceleration (NVIDIA GeForce GTX 1050 Ti).
# By default, we enable native hardware acceleration. Software rendering (llvmpipe) is only
# activated when explicitly requested via --software.
if not os.environ.get("DISPLAY"):
    os.environ["DISPLAY"] = ":0"
if not os.environ.get("WAYLAND_DISPLAY"):
    os.environ["WAYLAND_DISPLAY"] = "wayland-0"
if not os.environ.get("XDG_RUNTIME_DIR"):
    os.environ["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"
if not os.environ.get("PULSE_SERVER"):
    os.environ["PULSE_SERVER"] = "unix:/mnt/wslg/PulseServer"

# Gazebo offline model paths to prevent network timeout freeze
os.environ["GAZEBO_MODEL_DATABASE_URI"] = ""
tb3_models = "/opt/ros/humble/share/turtlebot3_gazebo/models"
user_models = os.path.expanduser("~/.gazebo/models")
os.environ["GAZEBO_MODEL_PATH"] = f"{user_models}:{tb3_models}"


# ANSI Colors
class Colors:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"
    BG_BLUE = "\033[44m"
    BG_RED  = "\033[41m"
    BG_GREEN= "\033[42m"


def print_sim_banner(world_name, use_yolo, use_slam):
    banner = f"""
{Colors.CYAN}{Colors.BOLD}========================================================================================
  🎮  재난구조 로봇 가제보(Gazebo) 시뮬레이션 & 실시간 SLAM 맵핑 콘솔  🎮
========================================================================================{Colors.RESET}
  {Colors.BOLD}시뮬레이터{Colors.RESET}   : Gazebo 11 (ODE Physics) · WSLg 화면 렌더링 정상화 (LIBGL Software)
  {Colors.BOLD}시뮬레이션 맵{Colors.RESET} : {world_name}
  {Colors.BOLD}SLAM 맵핑 {Colors.RESET}   : {'Cartographer 2D SLAM 활성화 (실시간 /map 생성 & RViz2 시각화)' if use_slam else '비활성화 (오도메트리 전용)'}
  {Colors.BOLD}비전 AI   {Colors.RESET}   : {'YOLO26n 객체/조난자 인식 창 활성화 (/camera/image_raw)' if use_yolo else '비활성화 (라이다/오도메트리 전용 모드)'}
  {Colors.BOLD}오차 검증 {Colors.RESET}   : 2D 라이다 스캔 매칭(ICP) 휠 슬립 오차 보정 & 자율복귀
  {Colors.BOLD}통신 단절 {Colors.RESET}   : [ H ] 키 입력 시 Heartbeat 즉시 중단 -> 자율복귀 발동
{Colors.CYAN}========================================================================================{Colors.RESET}
"""
    print(banner)


def stop_all_sim():
    print(f"\n{Colors.YELLOW}[INFO] 모든 Gazebo 시뮬레이션, Cartographer SLAM, ROS 2 프로세스를 종료합니다...{Colors.RESET}")
    subprocess.call("killall -9 gzserver gzclient cartographer_node occupancy_grid_node recovery_controller host_ai_pipeline rviz2 2>/dev/null || true", shell=True)
    subprocess.call("pkill -9 -f 'turtlebot3_gazebo' 2>/dev/null || true", shell=True)
    subprocess.call("pkill -9 -f 'turtlebot3_cartographer' 2>/dev/null || true", shell=True)
    subprocess.call("pkill -9 -f 'recovery_controller' 2>/dev/null || true", shell=True)
    print(f"{Colors.GREEN}[ OK ] 모든 시뮬레이션 프로세스가 깨끗이 정리되었습니다.{Colors.RESET}")


class GazeboSimulationRunner:
    def __init__(self, world="world", use_slam=True, use_yolo=False, no_teleop=False, software=False, headless=False):
        self.world = world
        self.use_slam = use_slam
        self.use_yolo = use_yolo
        self.no_teleop = no_teleop
        self.software = software
        self.headless = headless
        self.processes = []
        self.heartbeat_active = True
        self.running = True

        # Robot Model selection: burger_cam if YOLO requested, otherwise burger
        if self.use_yolo:
            os.environ["TURTLEBOT3_MODEL"] = "burger_cam"
        else:
            os.environ["TURTLEBOT3_MODEL"] = "burger"

    def start(self):
        print_sim_banner(self.world, self.use_yolo, self.use_slam)

        # 1. Clean previous lingering processes
        subprocess.call("killall -9 gzserver gzclient cartographer_node occupancy_grid_node recovery_controller 2>/dev/null || true", shell=True)
        time.sleep(1.0)

        os.makedirs("/tmp/disaster_recovery", exist_ok=True)
        self.log_gz = open("/tmp/disaster_recovery/gazebo.log", "w")
        self.log_carto = open("/tmp/disaster_recovery/cartographer.log", "w")

        # 2. Start Gazebo World
        accel_mode = "소프트웨어 (llvmpipe)" if self.software else "하드웨어 GPU 가속 (NVIDIA GTX 1050 Ti)"
        view_mode = "헤드리스 물리서버 (초경량 CPU 모드)" if self.headless else "3D 뷰포트 GUI 활성화"
        print(f"{Colors.MAGENTA}{Colors.BOLD}▶ [1/5] Gazebo 시뮬레이션 환경 기동 중... (맵: {self.world} / 가속: {accel_mode} / 화면: {view_mode}){Colors.RESET}")
        
        gui_flag = "gui:=false" if self.headless else "gui:=true"
        disaster_world = "/home/lab/disaster_recovery_ws/src/disaster_recovery/worlds/disaster_world.world"
        disaster_house_world = "/home/lab/disaster_recovery_ws/src/disaster_recovery/worlds/disaster_house_world.world"
        disaster_maze_world = "/home/lab/disaster_recovery_ws/src/disaster_recovery/worlds/disaster_maze_world.world"
        disaster_tunnel_world = "/home/lab/disaster_recovery_ws/src/disaster_recovery/worlds/disaster_tunnel_world.world"
        disaster_collapsed_world = "/home/lab/disaster_recovery_ws/src/disaster_recovery/worlds/disaster_collapsed_building.world"
        if self.world in ["collapsed", "building", "collapsed_building"]:
            launch_cmd = f"ros2 launch disaster_recovery disaster_world.launch.py x_pose:=0.0 y_pose:=-4.0 yaw_pose:=1.5708 world:={disaster_collapsed_world} {gui_flag}"
        elif self.world in ["tunnel", "gazebo_tunnel", "tunnel_world"]:
            launch_cmd = f"ros2 launch disaster_recovery disaster_world.launch.py x_pose:=-0.5 y_pose:=-3.0 yaw_pose:=1.57 world:={disaster_tunnel_world} {gui_flag}"
        elif self.world in ["house", "turtlebot3_house"]:
            launch_cmd = f"ros2 launch disaster_recovery disaster_world.launch.py x_pose:=-2.0 y_pose:=-0.5 yaw_pose:=0.0 world:={disaster_house_world} {gui_flag}"
        elif self.world in ["maze", "turtlebot3_maze", "stage4"]:
            launch_cmd = f"ros2 launch disaster_recovery disaster_world.launch.py x_pose:=0.0 y_pose:=0.0 yaw_pose:=0.0 world:={disaster_maze_world} {gui_flag}"
        elif self.world in ["empty", "empty_world"]:
            launch_cmd = f"ros2 launch turtlebot3_gazebo empty_world.launch.py {gui_flag}"
        else:
            launch_cmd = f"ros2 launch disaster_recovery disaster_world.launch.py x_pose:=-2.0 y_pose:=-0.5 yaw_pose:=0.0 world:={disaster_world} {gui_flag}"

        net_env = "export ROS_LOCALHOST_ONLY=1 GAZEBO_IP=127.0.0.1 GAZEBO_MASTER_URI=http://127.0.0.1:11345 "
        if self.software:
            env_prefix = (
                net_env +
                "export GALLIUM_DRIVER=llvmpipe "
                "LIBGL_ALWAYS_SOFTWARE=1 "
                "MESA_LOADER_DRIVER_OVERRIDE=llvmpipe "
                "GAZEBO_MODEL_DATABASE_URI='' "
                f"GAZEBO_MODEL_PATH='{user_models}:{tb3_models}' && "
            )
        else:
            os.environ.pop("LIBGL_ALWAYS_SOFTWARE", None)
            os.environ.pop("GALLIUM_DRIVER", None)
            os.environ.pop("MESA_LOADER_DRIVER_OVERRIDE", None)
            os.environ["MESA_D3D12_DEFAULT_ADAPTER_NAME"] = "NVIDIA"
            env_prefix = (
                net_env +
                "unset LIBGL_ALWAYS_SOFTWARE GALLIUM_DRIVER MESA_LOADER_DRIVER_OVERRIDE && "
                "export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA GAZEBO_MODEL_DATABASE_URI='' "
                f"GAZEBO_MODEL_PATH='{user_models}:{tb3_models}' && "
            )

        p_gazebo = subprocess.Popen(
            f"bash -c '{env_prefix}source /opt/ros/humble/setup.bash && source /home/lab/disaster_recovery_ws/install/setup.bash && {launch_cmd}'",
            shell=True,
            stdout=self.log_gz,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid
        )
        self.processes.append(p_gazebo)

        print(f"{Colors.CYAN}[INFO] Gazebo 물리 엔진 및 로봇 스폰 대기 중 (5초)... (로그: /tmp/disaster_recovery/gazebo.log){Colors.RESET}")
        time.sleep(5)

        # 3. Start Cartographer SLAM (Mapping + RViz2)
        if self.use_slam:
            print(f"\n{Colors.MAGENTA}{Colors.BOLD}▶ [2/5] Cartographer SLAM 실시간 맵핑 및 RViz2 관제창 기동 중 (GPU 하드웨어 가속)...{Colors.RESET}")
            carto_cmd = (
                f"{env_prefix}source /opt/ros/humble/setup.bash && "
                "ros2 launch turtlebot3_cartographer cartographer.launch.py use_sim_time:=true use_rviz:=true"
            )
            p_carto = subprocess.Popen(
                f"bash -c '{carto_cmd}'",
                shell=True,
                stdout=self.log_carto,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid
            )
            self.processes.append(p_carto)
            print(f"{Colors.CYAN}[INFO] SLAM 맵핑 노드 및 RViz2 렌더링 대기 중 (3초)... (로그: /tmp/disaster_recovery/cartographer.log){Colors.RESET}")
            time.sleep(3)

        # 4. Start Recovery Controller Node
        use_aruco_str = "true" if self.use_yolo else "false"
        print(f"\n{Colors.MAGENTA}{Colors.BOLD}▶ [3/5] 재난 복귀 제어기 (recovery_controller) 기동 중... (ArUco 도킹: {use_aruco_str}){Colors.RESET}")
        recovery_cmd = (
            "source /opt/ros/humble/setup.bash && "
            "source /home/lab/disaster_recovery_ws/install/setup.bash && "
            "ros2 run disaster_recovery recovery_controller "
            "--ros-args --params-file /home/lab/disaster_recovery_ws/src/disaster_recovery/config/recovery_params.yaml "
            f"-p use_sim_time:=true -p use_aruco_docking:={use_aruco_str}"
        )
        p_recovery = subprocess.Popen(
            f"bash -c '{recovery_cmd}'",
            shell=True,
            preexec_fn=os.setsid
        )
        self.processes.append(p_recovery)
        time.sleep(1.5)

        # 5. Optional: Start YOLO26n Host AI Pipeline
        if self.use_yolo:
            print(f"\n{Colors.MAGENTA}{Colors.BOLD}▶ [4/5] YOLO26n 비전 AI 파이프라인 기동 중 (/camera/image_raw)...{Colors.RESET}")
            yolo_cmd = (
                "source /opt/ros/humble/setup.bash && "
                "source /home/lab/disaster_recovery_ws/install/setup.bash && "
                "source /home/lab/ros2_ai_venv/bin/activate && "
                "ros2 launch disaster_recovery host_station.launch.py "
                "image_topic:=/camera/image_raw use_compressed:=false show_window:=true rotation_angle:=0 use_rviz:=false"
            )
            p_yolo = subprocess.Popen(
                f"bash -c '{yolo_cmd}'",
                shell=True,
                preexec_fn=os.setsid
            )
            self.processes.append(p_yolo)
            time.sleep(2.0)
        elif not self.no_teleop:
            # Standalone Heartbeat Publisher Thread (10Hz)
            print(f"\n{Colors.MAGENTA}{Colors.BOLD}▶ [4/5] 관제국 Heartbeat (10Hz) 활성화...{Colors.RESET}")
            self.hb_thread = threading.Thread(target=self._heartbeat_worker, daemon=True)
            self.hb_thread.start()
        else:
            print(f"\n{Colors.MAGENTA}{Colors.BOLD}▶ [4/5] 관제국 Heartbeat 대기 (외부 제어기/자율 벤치마크 모드)...{Colors.RESET}")

        # 6. Interactive Cockpit & WASD Teleop
        print(f"\n{Colors.MAGENTA}{Colors.BOLD}▶ [5/5] 대화형 시뮬레이션 콘솔 준비 완료!{Colors.RESET}")
        print(f"""
{Colors.GREEN}{Colors.BOLD}[조작 안내]{Colors.RESET}
  • [ W / A / S / D ] : 터틀봇 전진 / 좌회전 / 후진 / 우회전
  • [ Space ]         : 긴급 제동 (정지)
  • {Colors.BG_RED}{Colors.WHITE}[ H ]{Colors.RESET}{Colors.BOLD}             : 통신 단절(Heartbeat DROP) 트리거 (오도메트리 복귀 테스트!){Colors.RESET}
  • {Colors.BG_GREEN}{Colors.WHITE}[ R ]{Colors.RESET}{Colors.BOLD}             : 통신 복구(Heartbeat RESUME){Colors.RESET}
  • {Colors.BG_BLUE}{Colors.WHITE}[ O ]{Colors.RESET}{Colors.BOLD}             : 랜덤 붕괴 암석(3~6개) 투입 / 제거 토글 (신규 장애물 테스트){Colors.RESET}
  • [ Q ] 또는 Ctrl+C : 시뮬레이션 정상 종료

{Colors.YELLOW}👉 [테스트 시나리오]:
   1. [W],[A],[S],[D]로 로봇을 12~20m 전진 운전합니다 (RViz2에서 실시간 지도가 그려집니다).
   2. [ O ] 키를 누르면 로봇이 지나온 복귀 경로에 3~6개의 붉은색 붕괴 암석(Debris)이 랜덤 투하됩니다.
   3. [ H ] 키를 누르면 통신 단절이 발생하고 3초 후 자율복귀가 발동합니다.
   4. 장애물 접근 시 TD3 국소 회피 제어기가 부드러운 우회 기동을 수행하고, 2D ICP가 슬립을 리셋하여 원점으로 복귀합니다.{Colors.RESET}
""")

        if not self.no_teleop:
            self._run_interactive_teleop()
        else:
            try:
                while self.running:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
            finally:
                self.shutdown()

    def _heartbeat_worker(self):
        """Publishes std_msgs/msg/Header to /heartbeat at 10Hz."""
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Header

        try:
            rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
        except Exception:
            pass

        node = Node('sim_heartbeat_publisher')
        pub = node.create_publisher(Header, '/heartbeat', 10)

        rate = node.create_rate(10.0)
        seq = 0
        while self.running and rclpy.ok():
            if self.heartbeat_active:
                msg = Header()
                msg.stamp = node.get_clock().now().to_msg()
                msg.frame_id = f"heartbeat_sim_{seq}"
                pub.publish(msg)
                seq += 1
            rate.sleep()

        node.destroy_node()

    def _run_interactive_teleop(self):
        """Embedded WASD teleoperation controller publishing to /cmd_vel_remote."""
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from std_msgs.msg import String
        import math

        try:
            rclpy.init()
        except Exception:
            pass

        teleop_node = Node('sim_teleop_controller')
        pub_cmd = teleop_node.create_publisher(Twist, '/cmd_vel_remote', 10)

        # Odometry and path tracking for obstacle placement
        cur_x = 0.0
        cur_y = 0.0
        cur_yaw = 0.0
        path_history = []
        spawned_debris_names = []
        last_known_state = ""

        def cb_odom(msg: Odometry):
            nonlocal cur_x, cur_y, cur_yaw
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            cur_x = pos.x
            cur_y = pos.y
            siny_cosp = 2.0 * (ori.w * ori.z + ori.x * ori.y)
            cosy_cosp = 1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z)
            cur_yaw = math.atan2(siny_cosp, cosy_cosp)
            if not path_history or math.hypot(cur_x - path_history[-1][0], cur_y - path_history[-1][1]) >= 0.35:
                path_history.append((cur_x, cur_y))

        def cb_status(msg: String):
            nonlocal last_known_state
            raw = msg.data
            parts = raw.split('|')
            st = parts[0].replace("STATE:SystemState.", "").replace("STATE:", "").strip()
            if st != last_known_state:
                arrow = f"{last_known_state} ➔ " if last_known_state else ""
                sys.stdout.write(f"\r\n>>> {Colors.CYAN}{Colors.BOLD}[상태 전이]{Colors.RESET} {arrow}{Colors.YELLOW}{Colors.BOLD}{st}{Colors.RESET}\r\n")
                sys.stdout.flush()
                last_known_state = st

        teleop_node.create_subscription(Odometry, '/odom', cb_odom, 5)
        teleop_node.create_subscription(String, '/recovery/status', cb_status, 5)

        speed_lin = 0.16
        speed_ang = 0.70

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setcbreak(sys.stdin.fileno())
            while self.running:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                if rlist:
                    ch = sys.stdin.read(1)
                    key = ch
                    if ch == '\x1b':
                        r2, _, _ = select.select([sys.stdin], [], [], 0.02)
                        if r2 and sys.stdin.read(1) == '[':
                            r3, _, _ = select.select([sys.stdin], [], [], 0.02)
                            if r3:
                                code = sys.stdin.read(1)
                                if code == 'A': key = 'w'
                                elif code == 'B': key = 's'
                                elif code == 'C': key = 'd'
                                elif code == 'D': key = 'a'

                    twist = Twist()

                    if key in ['w', 'W']:
                        twist.linear.x = speed_lin
                        pub_cmd.publish(twist)
                    elif key in ['s', 'S']:
                        twist.linear.x = -speed_lin
                        pub_cmd.publish(twist)
                    elif key in ['a', 'A']:
                        twist.angular.z = speed_ang
                        pub_cmd.publish(twist)
                    elif key in ['d', 'D']:
                        twist.angular.z = -speed_ang
                        pub_cmd.publish(twist)
                    elif key == ' ':
                        twist.linear.x = 0.0
                        twist.angular.z = 0.0
                        pub_cmd.publish(twist)
                    elif key in ['h', 'H']:
                        self.heartbeat_active = not self.heartbeat_active
                        status_str = f"{Colors.BG_GREEN} 통신 복구 (ACTIVE) {Colors.RESET}" if self.heartbeat_active else f"{Colors.BG_RED} 통신 단절 (CUT - 3초 후 자율복귀 발동) {Colors.RESET}"
                        sys.stdout.write(f"\r\n>>> [HEARTBEAT 상태 변경] {status_str} <<<\r\n")
                        sys.stdout.flush()
                    elif key in ['r', 'R']:
                        self.heartbeat_active = True
                        sys.stdout.write(f"\r\n>>> [HEARTBEAT] 통신 강제 복구 (ACTIVE) <<<\r\n")
                        sys.stdout.flush()
                    elif key in ['o', 'O']:
                        if not spawned_debris_names:
                            debris_pts = select_random_debris_positions(
                                path_history,
                                min_count=3,
                                max_count=6,
                                min_spacing=1.2,
                                end_margin=1.0,
                                fallback_pose=(cur_x, cur_y, cur_yaw),
                            )
                            spawned_debris_names = spawn_random_debris_cluster(
                                debris_pts, wait=False
                            )
                            coords_str = ", ".join(
                                f"({px:.2f}, {py:.2f})" for px, py in debris_pts
                            )
                            sys.stdout.write(
                                f"\r\n>>> {Colors.BG_RED}{Colors.WHITE}[OBSTACLE]{Colors.RESET} "
                                f"{Colors.RED}{Colors.BOLD}⚠️ 랜덤 붕괴 암석 {len(debris_pts)}개 투하! "
                                f"위치: [{coords_str}] - 복귀 경로 장애물 생성!{Colors.RESET}\r\n"
                            )
                            sys.stdout.flush()
                        else:
                            delete_debris_entities(spawned_debris_names, wait=False)
                            removed_cnt = len(spawned_debris_names)
                            spawned_debris_names = []
                            sys.stdout.write(
                                f"\r\n>>> {Colors.BG_GREEN}{Colors.WHITE}[OBSTACLE]{Colors.RESET} "
                                f"{Colors.GREEN}{Colors.BOLD}🧹 붕괴 암석 {removed_cnt}개 일괄 제거 완료 - 복귀 경로 개방!{Colors.RESET}\r\n"
                            )
                            sys.stdout.flush()
                    elif key in ['q', 'Q', '\x03']: # Ctrl+C
                        break
                rclpy.spin_once(teleop_node, timeout_sec=0.01)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            teleop_node.destroy_node()
            self.shutdown()

    def shutdown(self):
        self.running = False
        print(f"\n{Colors.YELLOW}[INFO] 시뮬레이션을 정리하고 모든 프로세스를 종료합니다...{Colors.RESET}")
        for p in self.processes:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass
        time.sleep(1.0)
        stop_all_sim()
        print(f"{Colors.GREEN}{Colors.BOLD}[완료] Gazebo 시뮬레이션 및 SLAM이 성공적으로 종료되었습니다.{Colors.RESET}")


def main():
    parser = argparse.ArgumentParser(description="Disaster Recovery TurtleBot3 Gazebo Simulator")
    parser.add_argument("--world", type=str, default="collapsed", help="World name (collapsed, world, house, maze, tunnel, empty)")
    parser.add_argument("--house", action="store_true", help="Shortcut for --world house (Corridors & Rooms)")
    parser.add_argument("--maze", action="store_true", help="Shortcut for --world maze (Complex Inner Walls & Obstacles)")
    parser.add_argument("--tunnel", action="store_true", help="Shortcut for --world tunnel (Subterranean Tunnel & Rubble)")
    parser.add_argument("--collapsed", action="store_true", help="Shortcut for --world collapsed (Collapsed Building Disaster Map)")
    parser.add_argument("--yolo", action="store_true", help="Launch YOLO26n Vision AI detection window")
    parser.add_argument("--no-slam", action="store_true", help="Disable Cartographer SLAM mapping & RViz")
    parser.add_argument("--no-teleop", action="store_true", help="Disable embedded keyboard teleop")
    parser.add_argument("--software", action="store_true", help="Force software rendering (llvmpipe) instead of GPU hardware acceleration")
    parser.add_argument("--headless", action="store_true", help="Run Gazebo in headless mode (No gzclient GUI, RViz2 only) for ultra-low CPU load")
    parser.add_argument("--stop", action="store_true", help="Stop all running simulation processes")

    args = parser.parse_args()

    if args.stop:
        stop_all_sim()
        return

    if args.collapsed:
        world = "collapsed"
    elif args.tunnel:
        world = "tunnel"
    elif args.maze:
        world = "maze"
    elif args.house:
        world = "house"
    else:
        world = args.world
    use_slam = not args.no_slam

    runner = GazeboSimulationRunner(
        world=world,
        use_slam=use_slam,
        use_yolo=args.yolo,
        no_teleop=args.no_teleop,
        software=args.software,
        headless=args.headless
    )
    try:
        runner.start()
    except KeyboardInterrupt:
        runner.shutdown()


if __name__ == "__main__":
    main()
