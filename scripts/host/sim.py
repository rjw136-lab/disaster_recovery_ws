#!/usr/bin/env python3
"""
=================================================================================================
Project: Disaster Recovery TurtleBot3 Burger System
Script : sim.py (One-Touch Gazebo Simulation Cockpit, SLAM Mapping & Odometry Verification)
Author : Autonomous Systems Laboratory / Jongwoo Noh
Usage  : 
    sim.py                     # Standard Mode (Gazebo World + SLAM Mapping + RViz + Teleop)
    sim.py --house             # Run in TurtleBot3 House World (Rooms & Corridors)
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

# -----------------------------------------------------------------------------------------------
# CRITICAL GRAPHICS ENVIRONMENT FOR WSL2 / WSLg
# -----------------------------------------------------------------------------------------------
# In WSL2/WSLg, Mesa Direct3D12 hardware rendering causes Gazebo OGRE 3D viewport to be pitch black.
# Forcing software rendering (LIBGL_ALWAYS_SOFTWARE=1) routes through llvmpipe, guaranteeing full
# OpenGL 3.3/4.5 compliance and rendering the 3D Gazebo world cleanly without black screen!
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["MESA_D3D12_DEFAULT_ADAPTER_NAME"] = "NVIDIA"
if not os.environ.get("DISPLAY"):
    os.environ["DISPLAY"] = ":0"
if not os.environ.get("WAYLAND_DISPLAY"):
    os.environ["WAYLAND_DISPLAY"] = "wayland-0"
if not os.environ.get("XDG_RUNTIME_DIR"):
    os.environ["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"
if not os.environ.get("PULSE_SERVER"):
    os.environ["PULSE_SERVER"] = "unix:/mnt/wslg/PulseServer"

os.environ["ROS_DOMAIN_ID"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"

# Gazebo model path to prevent offline timeout
tb3_models = "/opt/ros/humble/share/turtlebot3_gazebo/models"
cur_model_path = os.environ.get("GAZEBO_MODEL_PATH", "")
if tb3_models not in cur_model_path:
    os.environ["GAZEBO_MODEL_PATH"] = f"{tb3_models}:{cur_model_path}" if cur_model_path else tb3_models


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
    def __init__(self, world="world", use_slam=True, use_yolo=False, no_teleop=False):
        self.world = world
        self.use_slam = use_slam
        self.use_yolo = use_yolo
        self.no_teleop = no_teleop
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

        # 2. Start Gazebo World
        print(f"{Colors.MAGENTA}{Colors.BOLD}▶ [1/5] Gazebo 3D 시뮬레이션 환경 기동 중... (맵: {self.world}){Colors.RESET}")
        if self.world in ["house", "turtlebot3_house"]:
            launch_cmd = "ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py x_pose:=-2.0 y_pose:=-0.5"
        elif self.world in ["empty", "empty_world"]:
            launch_cmd = "ros2 launch turtlebot3_gazebo empty_world.launch.py"
        else:
            launch_cmd = "ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py x_pose:=-2.0 y_pose:=-0.5"

        p_gazebo = subprocess.Popen(
            f"bash -c 'source /opt/ros/humble/setup.bash && {launch_cmd}'",
            shell=True,
            preexec_fn=os.setsid
        )
        self.processes.append(p_gazebo)

        print(f"{Colors.CYAN}[INFO] Gazebo 물리 엔진 및 로봇 스폰 대기 중 (6초)...{Colors.RESET}")
        time.sleep(6)

        # 3. Start Cartographer SLAM (Mapping + RViz2)
        if self.use_slam:
            print(f"\n{Colors.MAGENTA}{Colors.BOLD}▶ [2/5] Cartographer SLAM 실시간 맵핑 및 RViz2 관제창 기동 중...{Colors.RESET}")
            carto_cmd = (
                "source /opt/ros/humble/setup.bash && "
                "ros2 launch turtlebot3_cartographer cartographer.launch.py use_sim_time:=true use_rviz:=true"
            )
            p_carto = subprocess.Popen(
                f"bash -c '{carto_cmd}'",
                shell=True,
                preexec_fn=os.setsid
            )
            self.processes.append(p_carto)
            print(f"{Colors.CYAN}[INFO] SLAM 맵핑 노드 및 RViz2 렌더링 대기 중 (3초)...{Colors.RESET}")
            time.sleep(3)

        # 4. Start Recovery Controller Node
        print(f"\n{Colors.MAGENTA}{Colors.BOLD}▶ [3/5] 재난 복귀 제어기 (recovery_controller) 기동 중...{Colors.RESET}")
        recovery_cmd = (
            "source /opt/ros/humble/setup.bash && "
            "source /home/lab/disaster_recovery_ws/install/setup.bash && "
            "ros2 run disaster_recovery recovery_controller "
            "--ros-args --params-file /home/lab/disaster_recovery_ws/src/disaster_recovery/config/recovery_params.yaml "
            "-p use_sim_time:=true"
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
        else:
            # Standalone Heartbeat Publisher Thread (10Hz)
            print(f"\n{Colors.MAGENTA}{Colors.BOLD}▶ [4/5] 관제국 Heartbeat (10Hz) 활성화...{Colors.RESET}")
            self.hb_thread = threading.Thread(target=self._heartbeat_worker, daemon=True)
            self.hb_thread.start()

        # 6. Interactive Cockpit & WASD Teleop
        print(f"\n{Colors.MAGENTA}{Colors.BOLD}▶ [5/5] 대화형 시뮬레이션 콘솔 준비 완료!{Colors.RESET}")
        print(f"""
{Colors.GREEN}{Colors.BOLD}[조작 안내]{Colors.RESET}
  • [ W / A / S / D ] : 터틀봇 전진 / 좌회전 / 후진 / 우회전
  • [ Space ]         : 긴급 제동 (정지)
  • {Colors.BG_RED}{Colors.WHITE}[ H ]{Colors.RESET}{Colors.BOLD}             : 통신 단절(Heartbeat DROP) 트리거 (오도메트리 복귀 테스트!){Colors.RESET}
  • {Colors.BG_GREEN}{Colors.WHITE}[ R ]{Colors.RESET}{Colors.BOLD}             : 통신 복구(Heartbeat RESUME){Colors.RESET}
  • [ Q ] 또는 Ctrl+C : 시뮬레이션 정상 종료

{Colors.YELLOW}👉 [테스트 방법]:
   1. [W],[A],[S],[D]로 로봇을 벽/장애물 사이로 5~10m 운전하세요 (RViz2에서 실시간 지도가 그려집니다).
   2. 원하는 위치에서 [ H ] 키를 누르면 통신 단절이 발생합니다.
   3. 3초 후 로봇이 자율 복귀 모드로 전환되어, 실시간 2D 라이다 스캔 매칭(ICP)으로 오도메트리 오차를 보정하며 원점 (0, 0)으로 복귀합니다.{Colors.RESET}
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

        try:
            rclpy.init()
        except Exception:
            pass

        teleop_node = Node('sim_teleop_controller')
        pub_cmd = teleop_node.create_publisher(Twist, '/cmd_vel_remote', 10)

        speed_lin = 0.16
        speed_ang = 0.70

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setraw(sys.stdin.fileno())
            while self.running:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                if rlist:
                    key = sys.stdin.read(1)
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
    parser.add_argument("--world", type=str, default="world", help="World name (world, house, empty)")
    parser.add_argument("--house", action="store_true", help="Shortcut for --world house (Corridors & Rooms)")
    parser.add_argument("--yolo", action="store_true", help="Launch YOLO26n Vision AI detection window")
    parser.add_argument("--no-slam", action="store_true", help="Disable Cartographer SLAM mapping & RViz")
    parser.add_argument("--no-teleop", action="store_true", help="Disable embedded keyboard teleop")
    parser.add_argument("--stop", action="store_true", help="Stop all running simulation processes")

    args = parser.parse_args()

    if args.stop:
        stop_all_sim()
        return

    world = "house" if args.house else args.world
    use_slam = not args.no_slam

    runner = GazeboSimulationRunner(
        world=world,
        use_slam=use_slam,
        use_yolo=args.yolo,
        no_teleop=args.no_teleop
    )
    try:
        runner.start()
    except KeyboardInterrupt:
        runner.shutdown()


if __name__ == "__main__":
    main()
