#!/usr/bin/env python3
"""
=================================================================================================
Project: Disaster Recovery TurtleBot3 Burger System
Script : start.py (One-Touch Master Launcher & Supervisor)
Author : Kyungpook National University / Jong Woo Ro
Usage  : 
    start.py                # Standard Mode (AI + SLAM/RViz background, Teleop in foreground)
    start.py --tmux         # Cockpit Mode (Tmux multi-pane dashboard)
    start.py --host-only    # Launch only Host PC nodes (skip robot remote launch)
    start.py --robot-only   # Launch only Robot onboard nodes via SSH
    start.py --stop         # Gracefully terminate all host and robot processes
    start.py --status       # Check system, network, and node status
=================================================================================================
"""

import os
import sys
import time
import signal
import socket
import getpass
import argparse
import subprocess
import threading
from pathlib import Path

# Optional paramiko import for remote robot SSH orchestration
try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False


# ===============================================================================================
# ANSI Styling & Colors
# ===============================================================================================
class Colors:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"
    BG_RED  = "\033[41m"
    BG_BLUE = "\033[44m"


def print_banner():
    banner = f"""
{Colors.CYAN}{Colors.BOLD}========================================================================================
  🚨  재난구조로봇 원터치 마스터 런처 (Disaster Recovery System)  🚨
========================================================================================{Colors.RESET}
  {Colors.BOLD}Host Station{Colors.RESET} : Windows 11 WSL2 · i7-8700 CPU · GTX 1050 Ti GPU · ROS 2 Humble
  {Colors.BOLD}AI Workload {Colors.RESET} : YOLO26n (GPU CUDA 가속) + Faster-Whisper base (GPU CUDA INT8 가속)
  {Colors.BOLD}VRAM Safety {Colors.RESET} : 4GB VRAM 통합 GPU 가속 (YOLO26n + Whisper INT8, 실패 시 CPU 폴백)
  {Colors.BOLD}Robot Target{Colors.RESET} : TurtleBot3 Burger (Raspberry Pi 4 · OpenCR · LDS-01 · MQ-2)
  {Colors.BOLD}Safety Rule {Colors.RESET} : Heartbeat 10Hz · Teleop Remap -> /cmd_vel_remote · Zero-Lag Vision
{Colors.CYAN}========================================================================================{Colors.RESET}
"""
    print(banner)


def log_info(msg):
    print(f"{Colors.CYAN}[INFO]{Colors.RESET} {msg}")

def log_ok(msg):
    print(f"{Colors.GREEN}{Colors.BOLD}[  OK  ]{Colors.RESET} {msg}")

def log_warn(msg):
    print(f"{Colors.YELLOW}[WARN]{Colors.RESET} {msg}")

def log_err(msg):
    print(f"{Colors.RED}{Colors.BOLD}[FAIL]{Colors.RESET} {msg}")

def log_step(step_idx, total_steps, title):
    print(f"\n{Colors.MAGENTA}{Colors.BOLD}▶ [{step_idx}/{total_steps}] {title}{Colors.RESET}")
    print("-" * 70)


# ===============================================================================================
# Configuration Constants
# ===============================================================================================
WORKSPACE_DIR    = "/home/lab/disaster_recovery_ws"
WS_SETUP_SCRIPT  = os.path.join(WORKSPACE_DIR, "install", "setup.bash")
ROS_SETUP_SCRIPT = "/opt/ros/humble/setup.bash"
VENV_DIR         = "/home/lab/ros2_ai_venv"
VENV_ACTIVATE    = os.path.join(VENV_DIR, "bin", "activate")
YOLO_MODEL_PATH  = "/home/lab/yolo26n.pt"
LOG_DIR          = "/tmp/disaster_recovery"

DEFAULT_ROBOT_CANDIDATE_IPS = [
    "192.168.0.135",   # Wi-Fi Router Local LAN (Recommended / Default)
    "100.91.141.24",   # Tailscale Mesh VPN
]
DEFAULT_ROBOT_USER = "lab"
SSH_KEY_PUB        = os.path.expanduser("~/.ssh/id_ed25519.pub")
SSH_KEY_PRIV       = os.path.expanduser("~/.ssh/id_ed25519")


# ===============================================================================================
# Pre-Flight System Checker
# ===============================================================================================
class PreFlightChecker:
    @staticmethod
    def check_all():
        log_step(1, 4, "호스트 PC 사전 환경 점검 (Pre-Flight Checks)")
        all_ok = True

        # 1. ROS 2 Installation Check
        if os.path.exists(ROS_SETUP_SCRIPT):
            log_ok(f"ROS 2 Humble 설치 확인: {ROS_SETUP_SCRIPT}")
        else:
            log_err(f"ROS 2 Humble setup.bash를 찾을 수 없습니다: {ROS_SETUP_SCRIPT}")
            all_ok = False

        # 2. Workspace Build Check
        if os.path.exists(WS_SETUP_SCRIPT):
            log_ok(f"재난구조 워크스페이스 빌드 확인: {WS_SETUP_SCRIPT}")
        else:
            log_warn(f"워크스페이스가 빌드되지 않았습니다 ({WS_SETUP_SCRIPT}). 빌드를 시도합니다...")
            ret = subprocess.call(
                f"bash -c 'source {ROS_SETUP_SCRIPT} && cd {WORKSPACE_DIR} && colcon build --symlink-install --packages-select disaster_recovery'",
                shell=True
            )
            if ret == 0:
                log_ok("워크스페이스 빌드 성공 완료.")
            else:
                log_err("워크스페이스 빌드 실패. disaster_recovery 패키지를 확인하세요.")
                all_ok = False

        # 3. Model Weights Check
        if os.path.exists(YOLO_MODEL_PATH):
            log_ok(f"YOLO26n 모델 가중치 파일 확인: {YOLO_MODEL_PATH}")
        else:
            log_warn(f"{YOLO_MODEL_PATH} 가 없습니다. 기본 YOLO 가중치를 다운로드/사용합니다.")

        # 4. Virtualenv Check
        if os.path.exists(VENV_ACTIVATE):
            log_ok(f"전용 가상환경 연동 확인: {VENV_DIR} (ROS 2 패키지 격리)")

        # 5. Hardware Compute Partitioning (GTX 1050 Ti VRAM Protection)
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                log_ok(f"GPU 가속: {gpu_name} ({vram_gb:.1f}GB VRAM) -> YOLO26n 비전 AI + Faster-Whisper base (CUDA INT8) 동시 가속")
            else:
                log_warn("CUDA 사용 불가. CPU 모드로 추론합니다 (FPS 저하 가능성 있음).")
        except ImportError:
            log_warn("PyTorch가 감지되지 않았습니다. CPU Fallback 모드로 동작합니다.")

        # 6. Environment Variables & WSLg GUI Display Protection
        os.environ["TURTLEBOT3_MODEL"] = "burger"
        os.environ["ROS_DOMAIN_ID"] = "0"
        os.environ["ROS_LOCALHOST_ONLY"] = "0"
        os.environ["PYTHONUNBUFFERED"] = "1"
        if not os.environ.get("DISPLAY"):
            os.environ["DISPLAY"] = ":0"
        if not os.environ.get("WAYLAND_DISPLAY"):
            os.environ["WAYLAND_DISPLAY"] = "wayland-0"
        if not os.environ.get("XDG_RUNTIME_DIR"):
            os.environ["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"
        if not os.environ.get("PULSE_SERVER"):
            os.environ["PULSE_SERVER"] = "unix:/mnt/wslg/PulseServer"
        log_ok(f"ROS 및 GUI 디스플레이 환경 준비 완료 (MODEL=burger, DOMAIN_ID=0, DISPLAY={os.environ['DISPLAY']})")

        # Create log dir
        os.makedirs(LOG_DIR, exist_ok=True)
        log_ok(f"실시간 로그 디렉토리 준비: {LOG_DIR}")

        return all_ok


# ===============================================================================================
# Network & Robot SSH Controller
# ===============================================================================================
class RobotManager:
    def __init__(self, target_ip=None, user=DEFAULT_ROBOT_USER):
        self.user = user
        self.ip = target_ip
        self.ssh_client = None
        self.use_password = None

    def detect_active_ip(self):
        log_step(2, 4, "로봇 네트워크 및 통신 상태 확인")
        if self.ip:
            candidates = [self.ip]
        else:
            candidates = DEFAULT_ROBOT_CANDIDATE_IPS

        active_ip = None
        for cand_ip in candidates:
            # Ping test with 1 second timeout
            res = subprocess.run(
                ["ping", "-c", "1", "-W", "1", cand_ip],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            if res.returncode == 0:
                # Also test port 22
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.5)
                port_open = False
                try:
                    sock.connect((cand_ip, 22))
                    port_open = True
                except Exception:
                    pass
                finally:
                    sock.close()

                if port_open:
                    log_ok(f"로봇 감지 성공: {cand_ip} (Ping OK & SSH Port 22 OPEN)")
                    active_ip = cand_ip
                    break
                else:
                    log_warn(f"IP {cand_ip} 응답은 있으나 SSH 포트(22)가 닫혀있습니다.")

        if not active_ip:
            log_warn("로봇 IP에 자동으로 연결할 수 없습니다.")
            return None

        self.ip = active_ip
        return active_ip

    def connect_ssh(self, password=None, allow_prompt=True):
        if not PARAMIKO_AVAILABLE:
            log_warn("paramiko 모듈이 설치되어 있지 않아 자동 SSH 원격 제어가 비활성화됩니다.")
            return False

        if not self.ip:
            return False

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # 1. Try public key authentication first
        if os.path.exists(SSH_KEY_PRIV) and not password:
            try:
                client.connect(
                    hostname=self.ip,
                    username=self.user,
                    key_filename=SSH_KEY_PRIV,
                    timeout=3
                )
                self.ssh_client = client
                log_ok("로봇 SSH 공개키 무암호 인증 성공!")
                return True
            except (paramiko.AuthenticationException, Exception):
                pass

        # If prompt is not allowed or stdin is not a tty, don't block
        if not password and (not allow_prompt or not sys.stdin.isatty()):
            return False

        # 2. Try password authentication
        current_password = password
        if not current_password:
            print(f"\n{Colors.YELLOW}[?] 로봇({self.user}@{self.ip})의 SSH 비밀번호 입력 (Enter 시 건너뛰고 수동 모드 진행):{Colors.RESET}")
            try:
                current_password = getpass.getpass("    Password: ")
            except (KeyboardInterrupt, EOFError):
                return False

        if not current_password:
            log_info("비밀번호 입력을 건너뜁니다.")
            return False

        try:
            client.connect(
                hostname=self.ip,
                username=self.user,
                password=current_password,
                timeout=4
            )
            self.ssh_client = client
            self.use_password = current_password
            log_ok("로봇 SSH 비밀번호 인증 성공!")

            # Auto-install SSH key to robot if not present so user never needs password again
            self._deploy_ssh_key_if_needed()
            return True
        except Exception as e:
            log_err(f"로봇 SSH 연결 실패: {e}")
            return False

    def _deploy_ssh_key_if_needed(self):
        """Automatically installs ~/.ssh/id_ed25519.pub onto robot for future passwordless login."""
        if not os.path.exists(SSH_KEY_PUB) or not self.ssh_client:
            return
        try:
            with open(SSH_KEY_PUB, "r") as f:
                pub_key = f.read().strip()
            setup_cmd = (
                f"mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
                f"touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && "
                f"grep -qxF '{pub_key}' ~/.ssh/authorized_keys || echo '{pub_key}' >> ~/.ssh/authorized_keys"
            )
            self.ssh_client.exec_command(setup_cmd)
            log_ok("✨ 호스트 PC SSH 공개키를 로봇에 자동 등록했습니다! (다음부터 암호 입력 불필요)")
        except Exception as e:
            log_warn(f"SSH 키 자동 등록 실패: {e}")

    def is_robot_running(self):
        if not self.ssh_client:
            return False
        try:
            # Use bracket regex trick to exclude SSH wrapper/pgrep process itself
            stdin, stdout, stderr = self.ssh_client.exec_command(
                "pgrep -f '[t]urtlebot3_ros|[t]urtlebot3_node|[r]ecovery_controller' || true"
            )
            out = stdout.read().decode().strip()
            pids = [p for p in out.split() if p.isdigit()]
            return len(pids) > 0
        except Exception:
            return False

    def start_robot_onboard(self):
        log_step(3, 4, "로봇 온보드(RPi 4) 노드 시작")
        if not self.ssh_client:
            log_warn("SSH 연결이 없어 로봇 온보드 노드를 원격 실행할 수 없습니다.")
            print(f"  {Colors.BOLD}👉 로봇 터미널에서 다음 명령을 직접 실행하세요:{Colors.RESET}")
            print(f"     bash ~/disaster_recovery_ws/scripts/rpi/start_all.sh\n")
            return False

        if self.is_robot_running():
            log_ok("로봇 온보드 노드가 이미 정상 동작 중입니다.")
            return True

        log_info("로봇에서 Bringup + camera_ros + 재난 복귀 노드를 통합 실행합니다...")
        start_cmd = (
            "nohup bash -c '"
            "export TURTLEBOT3_MODEL=burger; "
            "export LDS_MODEL=LDS-01; "
            "export ROS_DOMAIN_ID=0; "
            "export ROS_LOCALHOST_ONLY=0; "
            "source /opt/ros/humble/setup.bash; "
            "if [ -f ~/disaster_recovery_ws/install/setup.bash ]; then source ~/disaster_recovery_ws/install/setup.bash; fi; "
            "bash ~/disaster_recovery_ws/scripts/rpi/start_all.sh' > /tmp/robot_bringup.log 2>&1 &"
        )
        self.ssh_client.exec_command(start_cmd)

        # Wait for robot nodes to appear
        log_info("로봇 온보드 센서 및 노드 초기화 대기 중 (약 7초)...")
        for i in range(12):
            time.sleep(1)
            if self.is_robot_running():
                log_ok("로봇 온보드 통합 런칭 완료! (모터, LDS-01 라이다, 카메라, 워치독 활성화)")
                return True

        log_warn("로봇 프로세스 시작 신호를 전송했습니다. (/tmp/robot_bringup.log 확인 가능)")
        return True

    def stop_robot_onboard(self):
        if not self.ssh_client:
            return
        try:
            log_info("로봇 온보드 노드 정리 명령 전송 중...")
            stop_cmd = (
                "pkill -f 'camera_node|camera_ros|robot.launch|turtlebot3_node|turtlebot3_ros|hlds_laser|"
                "robot_state_publisher|recovery_controller|aruco_docking|audio_capture_node|"
                "gas_sensor_node|blackbox_recorder' 2>/dev/null || true"
            )
            self.ssh_client.exec_command(stop_cmd)
            log_ok("로봇 온보드 프로세스 종료 완료.")
        except Exception as e:
            log_warn(f"로봇 종료 명령 중 오류: {e}")

    def close(self):
        if self.ssh_client:
            self.ssh_client.close()


# ===============================================================================================
# Host PC Subprocess Orchestrator
# ===============================================================================================
class HostProcessManager:
    def __init__(self):
        self.processes = {}
        self.log_handles = {}
        self.running = True

    def launch(self, name, command, log_file, env=None):
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        disp = full_env.get("DISPLAY", ":0")
        wayl = full_env.get("WAYLAND_DISPLAY", "wayland-0")
        xdg = full_env.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
        pulse = full_env.get("PULSE_SERVER", "unix:/mnt/wslg/PulseServer")
        full_env["DISPLAY"] = disp
        full_env["WAYLAND_DISPLAY"] = wayl
        full_env["XDG_RUNTIME_DIR"] = xdg
        full_env["PULSE_SERVER"] = pulse

        venv_cmd = f"[ -f {VENV_ACTIVATE} ] && source {VENV_ACTIVATE} && " if os.path.exists(VENV_ACTIVATE) else ""
        wrapped_cmd = (
            f"bash -c 'source {ROS_SETUP_SCRIPT} && "
            f"{venv_cmd}"
            f"source {WS_SETUP_SCRIPT} && "
            f"export TURTLEBOT3_MODEL=burger && "
            f"export ROS_DOMAIN_ID=0 && "
            f"export DISPLAY={disp} && "
            f"export WAYLAND_DISPLAY={wayl} && "
            f"export XDG_RUNTIME_DIR={xdg} && "
            f"export PULSE_SERVER={pulse} && "
            f"exec {command}'"
        )

        log_path = os.path.join(LOG_DIR, log_file)
        f_log = open(log_path, "w", encoding="utf-8")
        self.log_handles[name] = f_log

        proc = subprocess.Popen(
            wrapped_cmd,
            shell=True,
            stdout=f_log,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            env=full_env
        )
        self.processes[name] = {
            "proc": proc,
            "cmd": command,
            "log": log_path,
        }
        log_ok(f"[{name}] 시작 완료 (PID: {proc.pid}) -> 로그: {log_path}")
        return proc

    def stop_all(self):
        if not self.processes:
            return

        print(f"\n{Colors.YELLOW}[STOP] Host PC 노드를 안전하게 종료합니다...{Colors.RESET}")
        for name, pinfo in reversed(list(self.processes.items())):
            proc = pinfo["proc"]
            if proc.poll() is None:
                try:
                    # Send SIGINT to process group
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                    log_info(f"[{name}] SIGINT 전송됨...")
                except Exception:
                    pass

        # Wait up to 3 seconds for graceful shutdown
        end_time = time.time() + 3.0
        for name, pinfo in self.processes.items():
            proc = pinfo["proc"]
            while time.time() < end_time and proc.poll() is None:
                time.sleep(0.1)

            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    log_warn(f"[{name}] 강제 종료(SIGKILL) 처리되었습니다.")
                except Exception:
                    pass

        # Close log handles
        for f in self.log_handles.values():
            try:
                f.close()
            except Exception:
                pass

        # Clean any remaining ROS lingering processes
        subprocess.call(
            "pkill -f 'host_ai_pipeline|cartographer_node|rviz2|teleop_twist_keyboard|wasd_teleop' 2>/dev/null || true",
            shell=True
        )
        log_ok("Host PC 노드 전체 정리 완료.")


# ===============================================================================================
# Execution Modes
# ===============================================================================================
def run_default_mode(robot_mgr, pm, args):
    """
    Standard Mode:
    - Runs Host AI Pipeline in background
    - Runs Cartographer SLAM & RViz in background
    - Runs Interactive Teleoperation directly in the foreground terminal
    - Captures Ctrl+C for unified graceful shutdown
    """
    log_step(4, 4, "Host PC 관제 및 텔레옵(Teleoperation) 시작")

    # 1. Launch Host AI Pipeline
    ai_cmd = (
        f"ros2 launch disaster_recovery host_station.launch.py "
        f"yolo_model_path:={YOLO_MODEL_PATH} "
        f"image_topic:=/camera/image_raw/compressed "
        f"use_compressed:=true "
        f"rotation_angle:=270 "
        f"show_window:=true "
        f"heartbeat_rate_hz:=10.0 "
        f"whisper_device:=cuda "
        f"whisper_model_size:=base "
        f"whisper_compute_type:=int8"
    )
    pm.launch("Host_AI_Pipeline", ai_cmd, "host_ai.log")
    time.sleep(1.5)

    # 2. Launch Cartographer SLAM (with RViz)
    if not args.no_cartographer:
        carto_cmd = "ros2 launch turtlebot3_cartographer cartographer.launch.py"
        pm.launch("Cartographer_SLAM", carto_cmd, "cartographer.log")
        time.sleep(1.5)

    # 3. Guide & Teleop Launch
    teleop_banner = f"""
{Colors.GREEN}{Colors.BOLD}========================================================================================
  🎮  WASD 직관 텔레옵(Teleoperation) 원격 조종 제어판 활성화!
========================================================================================{Colors.RESET}
  조종 키 안내:
        {Colors.BOLD}[ W ] / [ ↑ ]{Colors.RESET} : 직진 전진 (Linear X)
  {Colors.BOLD}[ A ] / [ ← ]{Colors.RESET}     {Colors.BOLD}[ D ] / [ → ]{Colors.RESET} : 좌회전 / 우회전 (제자리 선회)
        {Colors.BOLD}[ S ] / [ ↓ ]{Colors.RESET} : 직진 후진

  특수 조작:
      {Colors.BOLD}[ Q ] / [ E ]{Colors.RESET}         : 전진 좌회전 / 전진 우회전 (곡선 주행)
      {Colors.RED}{Colors.BOLD}[ Space ] / [ K ]{Colors.RESET}     : 즉시 긴급 정지 (Emergency Stop)
      {Colors.CYAN}[ + ] / [ - ]{Colors.RESET}         : 주행 선속도 ±0.02 m/s 조절
      {Colors.CYAN}[ ] ] / [ [ ]{Colors.RESET}         : 선회 각속도 ±0.10 rad/s 조절
      {Colors.YELLOW}[ C ]{Colors.RESET}                 : 크루즈 모드 (ON: 키 떼도 주행 유지 / OFF: 키 떼면 자동 감속)
      {Colors.YELLOW}Ctrl + C{Colors.RESET}              : 전체 재난구조 시스템 안전 종료

  {Colors.BOLD}💡 안전 및 시스템 상태:{Colors.RESET}
  - 텔레옵은 워치독 게이트웨이인 {Colors.CYAN}/cmd_vel_remote{Colors.RESET} 로 안전하게 송출됩니다.
  - Heartbeat(10Hz)가 지속 송출되는 동안만 수동 주행이 허용됩니다.
  - OpenCV 영상 창 및 Cartographer 2D SLAM 지도 창이 백그라운드 구동 중입니다.
{Colors.GREEN}========================================================================================{Colors.RESET}
"""
    print(teleop_banner)

    if args.no_teleop:
        log_info("--no-teleop 옵션이 지정되어 백그라운드 모니터링을 유지합니다. (종료: Ctrl+C)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        return

    # Run Teleop in foreground so terminal catches arrow/letter keys
    wasd_script = "/home/lab/wasd_teleop.py"
    if args.legacy_teleop or not os.path.exists(wasd_script):
        teleop_cmd = (
            f"bash -c 'source {ROS_SETUP_SCRIPT} && "
            f"ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel_remote'"
        )
    else:
        teleop_cmd = (
            f"bash -c 'source {ROS_SETUP_SCRIPT} && "
            f"python3 {wasd_script} topic:=/cmd_vel_remote'"
        )

    try:
        subprocess.run(teleop_cmd, shell=True)
    except KeyboardInterrupt:
        pass


def run_tmux_mode(robot_mgr, args):
    """
    Launches a dedicated tmux cockpit dashboard with split panes.
    """
    log_step(4, 4, "Tmux 콕핏 대시보드 모드 시작")
    session_name = "disaster_station"

    # Kill existing session if present
    subprocess.call(f"tmux kill-session -t {session_name} 2>/dev/null || true", shell=True)

    # Pane 0: Teleoperation (top)
    wasd_script = "/home/lab/wasd_teleop.py"
    if args.legacy_teleop or not os.path.exists(wasd_script):
        teleop_cmd = (
            f"source {ROS_SETUP_SCRIPT} && "
            f"ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel_remote"
        )
    else:
        teleop_cmd = (
            f"source {ROS_SETUP_SCRIPT} && "
            f"python3 {wasd_script} topic:=/cmd_vel_remote"
        )
    subprocess.call(f"tmux new-session -d -s {session_name} -n 'Mission_Control' '{teleop_cmd}'", shell=True)

    # Split horizontally for logs
    subprocess.call(f"tmux split-window -v -t {session_name}:0", shell=True)

    # Pane 1 (bottom left): Host AI Pipeline
    venv_cmd = f"[ -f {VENV_ACTIVATE} ] && source {VENV_ACTIVATE} && " if os.path.exists(VENV_ACTIVATE) else ""
    ai_cmd = (
        f"source {ROS_SETUP_SCRIPT} && {venv_cmd}source {WS_SETUP_SCRIPT} && "
        f"ros2 launch disaster_recovery host_station.launch.py "
        f"yolo_model_path:={YOLO_MODEL_PATH} "
        f"whisper_device:=cuda "
        f"whisper_model_size:=base "
        f"whisper_compute_type:=int8"
    )
    subprocess.call(f"tmux send-keys -t {session_name}:0.1 '{ai_cmd}' C-m", shell=True)

    # Split bottom vertically
    subprocess.call(f"tmux split-window -h -t {session_name}:0.1", shell=True)

    # Pane 2 (bottom right): Cartographer SLAM
    carto_cmd = (
        f"source {ROS_SETUP_SCRIPT} && source {WS_SETUP_SCRIPT} && "
        f"export TURTLEBOT3_MODEL=burger && "
        f"ros2 launch turtlebot3_cartographer cartographer.launch.py"
    )
    subprocess.call(f"tmux send-keys -t {session_name}:0.2 '{carto_cmd}' C-m", shell=True)

    # Select top pane for immediate keyboard control
    subprocess.call(f"tmux select-pane -t {session_name}:0.0", shell=True)

    log_ok(f"Tmux 세션 '{session_name}' 생성 완료. 대시보드에 연결합니다...")
    subprocess.call(f"tmux attach-session -t {session_name}", shell=True)


# ===============================================================================================
# Stop & Status Commands
# ===============================================================================================
def stop_all_system(robot_ip=None):
    print_banner()
    log_info("🛑 전체 재난구조로봇 시스템 종료 작업을 시작합니다...")

    # 1. Host cleanup
    subprocess.call(
        "pkill -f 'host_ai_pipeline|cartographer_node|rviz2|teleop_twist_keyboard|wasd_teleop' 2>/dev/null || true",
        shell=True
    )
    subprocess.call("tmux kill-session -t disaster_station 2>/dev/null || true", shell=True)
    log_ok("Host PC 모든 ROS 노드 및 대시보드 정리 완료.")

    # 2. Robot cleanup if reachable
    mgr = RobotManager(target_ip=robot_ip)
    ip = mgr.detect_active_ip()
    if ip and mgr.connect_ssh(allow_prompt=False):
        mgr.stop_robot_onboard()
        mgr.close()
    else:
        log_warn("로봇 무암호 SSH 연결이 설정되지 않아 Host 프로세스만 정리되었습니다.")

    print(f"\n{Colors.GREEN}{Colors.BOLD}✅ 전체 시스템이 안전하게 종료되었습니다.{Colors.RESET}\n")


def check_status(robot_ip=None):
    print_banner()
    PreFlightChecker.check_all()
    mgr = RobotManager(target_ip=robot_ip)
    ip = mgr.detect_active_ip()
    if ip:
        log_ok(f"로봇 감지: {ip}")
        if mgr.connect_ssh(allow_prompt=False):
            running = mgr.is_robot_running()
            if running:
                log_ok("로봇 온보드 노드 상태: [RUNNING] 정상 동작 중")
            else:
                log_warn("로봇 온보드 노드 상태: [STOPPED] 중지 상태")
            mgr.close()
        else:
            log_info("로봇 SSH 공개키 무암호 인증 미등록 상태 (암호 입력 필요)")
    else:
        log_warn("로봇 통신 불가.")


# ===============================================================================================
# Main Entry Point
# ===============================================================================================
def main():
    parser = argparse.ArgumentParser(description="Disaster Recovery TurtleBot3 Master Launcher")
    parser.add_argument("--tmux", action="store_true", help="Launch in tmux multi-pane dashboard mode")
    parser.add_argument("--host-only", action="store_true", help="Launch only Host PC nodes (skip robot remote SSH)")
    parser.add_argument("--robot-only", action="store_true", help="Launch only Robot onboard nodes via SSH")
    parser.add_argument("--no-cartographer", action="store_true", help="Disable Cartographer SLAM")
    parser.add_argument("--no-teleop", action="store_true", help="Disable foreground Teleop")
    parser.add_argument("--legacy-teleop", action="store_true", help="Use traditional teleop_twist_keyboard instead of WASD")
    parser.add_argument("--robot-ip", type=str, default=None, help="Target Robot IP address (e.g. 192.168.0.135)")
    parser.add_argument("--robot-user", type=str, default=DEFAULT_ROBOT_USER, help="Robot SSH username (default: lab)")
    parser.add_argument("--robot-pass", type=str, default=None, help="Robot SSH password")
    parser.add_argument("--sim", action="store_true", help="Launch Gazebo simulation cockpit for odometry verification")
    parser.add_argument("--rviz", action="store_true", help="Launch only RViz2 visualizer")
    parser.add_argument("--stop", action="store_true", help="Stop all running host and robot processes")
    parser.add_argument("--status", action="store_true", help="Check system and network status")

    args = parser.parse_args()

    # Handle --sim
    if args.sim:
        sim_script = "/home/lab/sim.py"
        if os.path.exists(sim_script):
            sys.exit(subprocess.call(["python3", sim_script]))
        else:
            log_err(f"{sim_script}를 찾을 수 없습니다.")
            return

    # Handle --rviz
    if args.rviz:
        rviz_script = "/home/lab/rviz.py"
        if os.path.exists(rviz_script):
            subprocess.call(f"python3 {rviz_script}", shell=True)
        else:
            subprocess.call(f"bash -c 'source {ROS_SETUP_SCRIPT} && ros2 run rviz2 rviz2'", shell=True)
        return

    # Handle --stop
    if args.stop:
        stop_all_system(args.robot_ip)
        return

    # Handle --status
    if args.status:
        check_status(args.robot_ip)
        return

    print_banner()

    # Step 1: Pre-Flight Checks
    if not PreFlightChecker.check_all():
        log_err("사전 환경 점검 실패. 문제를 해결한 후 다시 실행해주세요.")
        sys.exit(1)

    # Step 2 & 3: Robot Management
    robot_mgr = RobotManager(target_ip=args.robot_ip, user=args.robot_user)
    if not args.host_only:
        detected_ip = robot_mgr.detect_active_ip()
        if detected_ip:
            if robot_mgr.connect_ssh(password=args.robot_pass):
                robot_mgr.start_robot_onboard()
            else:
                log_warn("로봇 SSH 연결을 건너뛰고 Host 관제 스테이션을 계속 진행합니다.")
        else:
            log_warn("로봇이 감지되지 않았습니다. Host 관제 스테이션만 실행합니다.")

    if args.robot_only:
        log_ok("로봇 온보드 노드 시작 완료 (--robot-only). 프로그램을 종료합니다.")
        if robot_mgr.ssh_client:
            robot_mgr.close()
        return

    # Step 4: Host Process Orchestration
    pm = HostProcessManager()

    # Setup cleanup signals
    def cleanup_handler(signum, frame):
        log_warn("종료 신호(SIGINT/SIGTERM) 수신. 시스템을 정리합니다...")
        pm.stop_all()
        if robot_mgr.ssh_client:
            print(f"\n{Colors.YELLOW}[?] 로봇 온보드 노드도 함께 종료하시겠습니까? (y/N): {Colors.RESET}", end="")
            try:
                ans = input().strip().lower()
                if ans == "y":
                    robot_mgr.stop_robot_onboard()
            except Exception:
                pass
            robot_mgr.close()
        print(f"\n{Colors.GREEN}{Colors.BOLD}✅ 재난구조로봇 시스템이 정상 종료되었습니다.{Colors.RESET}\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

    try:
        if args.tmux:
            run_tmux_mode(robot_mgr, args)
        else:
            run_default_mode(robot_mgr, pm, args)
    finally:
        cleanup_handler(None, None)


if __name__ == "__main__":
    main()
