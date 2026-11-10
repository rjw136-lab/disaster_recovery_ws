#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================
Project: Disaster Recovery TurtleBot3 Burger System
File: start.py
Description:
  재난구조로봇 올인원(All-in-One) 통합 원클릭 시작 스크립트.
  터미널에서 `start.py` 한 번만 입력하면 하드웨어 진단부터 ROS 2 환경 로드,
  로봇 Bringup(모터/라이다), 카메라, 오디오 캡처, ArUco 감지 & 웹 스트리밍까지
  모든 세팅을 자동으로 완료하고 실시간 모니터링 대시보드를 제공합니다.
=================================================================================================
"""

import sys
import os
import time
import signal
import socket
import subprocess
import argparse
import re
from pathlib import Path

# --- ANSI 컬러 코드 정의 ---
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_RED = "\033[31m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_BLUE = "\033[34m"
COLOR_MAGENTA = "\033[35m"
COLOR_CYAN = "\033[36m"
COLOR_WHITE = "\033[37m"

LOG_DIR = Path("/tmp/disaster_robot_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 실행 관리용 전역 프로세스 리스트
MANAGED_PROCESSES = []
CLEANUP_CALLED = False


def find_script(filename):
    """스크립트 파일의 위치(동일 폴더 또는 /home/lab)를 자동으로 탐색합니다."""
    candidates = [
        Path(__file__).resolve().parent / filename,
        Path("/home/lab") / filename,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def print_banner():
    banner = f"""{COLOR_CYAN}{COLOR_BOLD}
╔══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                  ║
║   🚨  DISASTER RECOVERY TURTLEBOT3 BURGER - ALL-IN-ONE SYSTEM LAUNCHER  🚨       ║
║                       재난구조로봇 원클릭 통합 구동기                            ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝{COLOR_RESET}
"""
    print(banner)


def get_ros_env():
    """ROS 2 Humble 및 시스템 환경변수를 안전하게 로드합니다."""
    # 기본 환경변수 복사
    env = os.environ.copy()

    # bashrc 및 ros humble setup.bash 를 통하여 완전한 ROS 환경변수 맵 추출
    setup_cmd = (
        'source /opt/ros/humble/setup.bash 2>/dev/null; '
        '[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc" 2>/dev/null; '
        'env'
    )
    try:
        res = subprocess.run(
            ['bash', '-c', setup_cmd],
            capture_output=True,
            text=True,
            check=False
        )
        for line in res.stdout.splitlines():
            if '=' in line:
                k, v = line.split('=', 1)
                env[k] = v
    except Exception as e:
        print(f"{COLOR_YELLOW}[!] 환경변수 추출 중 경고: {e}{COLOR_RESET}")

    # 필수 환경변수 강제 보장
    env["TURTLEBOT3_MODEL"] = env.get("TURTLEBOT3_MODEL", "burger")
    env["LDS_MODEL"] = env.get("LDS_MODEL", "LDS-01")
    env["ROS_DOMAIN_ID"] = env.get("ROS_DOMAIN_ID", "0")
    env["PYTHONUNBUFFERED"] = "1"

    return env


def get_network_ips():
    """WLAN 및 Tailscale IP 주소를 조회합니다."""
    ips = {"wlan": None, "tailscale": None, "all": []}
    try:
        res = subprocess.run(['ip', '-4', 'addr', 'show'], capture_output=True, text=True, check=False)
        cur_iface = None
        for line in res.stdout.splitlines():
            m_iface = re.match(r'^\d+:\s+([^:@]+)', line)
            if m_iface:
                cur_iface = m_iface.group(1).strip()
            m_ip = re.search(r'inet\s+(\d+\.\d+\.\d+\.\d+)/\d+', line)
            if m_ip and cur_iface:
                ip_addr = m_ip.group(1)
                if cur_iface.startswith("wlan") or cur_iface.startswith("wl"):
                    ips["wlan"] = ip_addr
                elif "tailscale" in cur_iface:
                    ips["tailscale"] = ip_addr
                elif ip_addr != "127.0.0.1":
                    ips["all"].append((cur_iface, ip_addr))
    except Exception:
        pass
    return ips


def check_hardware():
    """하드웨어 장치 연결 상태를 점검합니다."""
    print(f"{COLOR_BOLD}🔍 [1단계] 재난구조로봇 하드웨어 사전 점검 (Pre-flight Check)...{COLOR_RESET}")

    checks = [
        ("OpenCR 1.0 (모터/센서 제어기)", "/dev/ttyACM0", "OpenCR 메인보드 USB 연결 필요"),
        ("LDS-01 (라이다 거리 센서)", "/dev/ttyUSB0", "LDS 라이다 USB 연결 필요"),
        ("Raspberry Pi Camera (IMX219)", "/dev/video0", "라즈베리파이 카메라 CSI 케이블 연결 필요"),
    ]

    all_ok = True
    for name, dev_path, hint in checks:
        if os.path.exists(dev_path):
            print(f"  {COLOR_GREEN}✔ [정상] {name:<30} -> {dev_path}{COLOR_RESET}")
        else:
            print(f"  {COLOR_RED}✘ [주의] {name:<30} -> {dev_path} 미발견 ({hint}){COLOR_RESET}")
            all_ok = False

    # USB 오디오 마이크 점검
    try:
        res = subprocess.run(['arecord', '-l'], capture_output=True, text=True, check=False)
        if "AB13X" in res.stdout or "USB Audio" in res.stdout:
            print(f"  {COLOR_GREEN}✔ [정상] USB Audio Microphone          -> ALSA Generic AB13X 감지됨{COLOR_RESET}")
        else:
            print(f"  {COLOR_YELLOW}▲ [주의] USB Audio Microphone          -> USB 마이크 장치 미감지 (오디오 비활성화 가능){COLOR_RESET}")
    except Exception:
        print(f"  {COLOR_YELLOW}▲ [주의] arecord 명령어 실행 불가 (오디오 장치 점검 건너뜀){COLOR_RESET}")

    if all_ok:
        print(f"  {COLOR_GREEN}{COLOR_BOLD}🚀 모든 핵심 하드웨어가 정상적으로 준비되었습니다.{COLOR_RESET}\n")
    else:
        print(f"  {COLOR_YELLOW}⚠️  일부 하드웨어가 감지되지 않았으나 실행을 계속 시도합니다.{COLOR_RESET}\n")


def kill_existing_processes():
    """이전 실행으로 남아있는 잔여 프로세스들을 깨끗이 정리합니다."""
    print(f"{COLOR_BOLD}🧹 [2단계] 기존 잔여 프로세스 정리 중...{COLOR_RESET}")
    pattern = 'aruco_detector.py|camera_node|robot.launch|turtlebot3_node|hlds_laser|robot_state_publisher|audio_capture_node.py'
    try:
        subprocess.run(['pkill', '-f', pattern], capture_output=True, check=False)
        time.sleep(1.0)
    except Exception:
        pass
    print(f"  {COLOR_GREEN}✔ 이전 프로세스 정리 완료.{COLOR_RESET}\n")


def launch_process(name, cmd, log_filename, env, wait_sec=2.0):
    """프로세스를 별도 프로세스 그룹으로 실행하고 로그를 저장합니다."""
    log_path = LOG_DIR / log_filename
    log_file = open(log_path, "w", encoding="utf-8")

    print(f"  ▶ {COLOR_BOLD}{name}{COLOR_RESET} 구동 중...")
    print(f"     명령어: {COLOR_CYAN}{' '.join(cmd)}{COLOR_RESET}")
    print(f"     로그 파일: {log_path}")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )

    MANAGED_PROCESSES.append({
        "name": name,
        "proc": proc,
        "log_file": log_file,
        "log_path": log_path,
        "cmd": cmd
    })

    time.sleep(wait_sec)

    # 프로세스 조기 사망 여부 확인
    ret = proc.poll()
    if ret is not None:
        print(f"  {COLOR_RED}✘ {name} 구동 실패! (종료 코드: {ret}){COLOR_RESET}")
        try:
            with open(log_path, "r", encoding="utf-8") as lf:
                lines = lf.readlines()[-10:]
                print(f"     {COLOR_RED}마지막 에러 로그 요약:\n     " + "     ".join(lines) + COLOR_RESET)
        except Exception:
            pass
        return False

    print(f"  {COLOR_GREEN}✔ {name} 정상 구동 완료 (PID: {proc.pid}){COLOR_RESET}\n")
    return True


def cleanup_all():
    """모든 실행된 프로세스를 안전하게 종료(Graceful Shutdown)합니다."""
    global CLEANUP_CALLED
    if CLEANUP_CALLED:
        return
    CLEANUP_CALLED = True

    print(f"\n{COLOR_YELLOW}{COLOR_BOLD}🛑 재난구조로봇 시스템 종료 신호 감지... 모든 프로세스를 안전하게 정리합니다.{COLOR_RESET}")

    # 1. 관리 중인 프로세스 그룹에 SIGTERM 전송
    for item in reversed(MANAGED_PROCESSES):
        proc = item["proc"]
        name = item["name"]
        if proc.poll() is None:
            try:
                print(f"  🔻 {name} (PID: {proc.pid}) 정중히 종료 중...")
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass

    # 종료 대기 (최대 3초)
    time.sleep(1.5)

    # 2. 아직 종료되지 않은 프로세스 SIGKILL 강제 종료
    for item in MANAGED_PROCESSES:
        proc = item["proc"]
        name = item["name"]
        if proc.poll() is None:
            try:
                print(f"  ⚠️  {name} 강제 종료 (SIGKILL)...")
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        try:
            item["log_file"].close()
        except Exception:
            pass

    # 3. 혹시 남은 고아 ROS 프로세스 pkill
    kill_existing_processes()
    print(f"{COLOR_GREEN}{COLOR_BOLD}✅ 재난구조로봇 시스템의 모든 센서와 노드가 안전하게 종료되었습니다.{COLOR_RESET}")


def signal_handler(signum, frame):
    cleanup_all()
    sys.exit(0)


def show_status():
    """현재 실행 중인 로봇 시스템 프로세스 상태를 표시합니다."""
    print(f"\n{COLOR_BOLD}📊 재난구조로봇 현재 구동 상태 점검{COLOR_RESET}")
    print("=" * 70)
    pattern = 'aruco_detector.py|camera_node|robot.launch|turtlebot3_node|hlds_laser|robot_state_publisher|audio_capture_node.py'
    res = subprocess.run(['pgrep', '-a', '-f', pattern], capture_output=True, text=True, check=False)

    lines = [l for l in res.stdout.splitlines() if 'pgrep' not in l and 'start.py --status' not in l]
    if not lines:
        print(f"  {COLOR_YELLOW}현재 실행 중인 재난구조로봇 프로세스가 없습니다.{COLOR_RESET}")
        print(f"  👉 실행하려면 터미널에 {COLOR_GREEN}start.py{COLOR_RESET} 를 입력하세요.")
    else:
        print(f"  {COLOR_GREEN}실행 중인 프로세스 ({len(lines)}개):{COLOR_RESET}")
        for l in lines:
            print(f"    • {l}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="TurtleBot3 Burger 재난구조로봇 원클릭 통합 시작 제어기",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--stop", action="store_true", help="실행 중인 모든 로봇 노드를 즉시 종료합니다.")
    parser.add_argument("--status", action="store_true", help="현재 로봇 노드의 실행 상태를 확인합니다.")
    parser.add_argument("--daemon", "-d", action="store_true", help="백그라운드 모드로 구동하고 쉘로 복귀합니다.")
    parser.add_argument("--no-audio", action="store_true", help="오디오 마이크 수집 노드를 제외하고 구동합니다.")
    parser.add_argument("--no-camera", action="store_true", help="카메라 노드를 제외하고 구동합니다.")
    parser.add_argument("--no-aruco", action="store_true", help="ArUco 마커 감지 및 웹 스트리밍 노드를 제외합니다.")
    parser.add_argument("--marker-size", type=float, default=0.05, help="ArUco 마커 실제 한 변 길이 (미터 단위, 기본값: 0.05 = 5.0cm)")
    parser.add_argument("--size-cm", type=float, default=None, help="ArUco 마커 실제 한 변 길이 (cm 단위, 예: 5.0)")
    args = parser.parse_args()

    if args.stop:
        kill_existing_processes()
        print(f"{COLOR_GREEN}✔ 모든 재난구조로봇 프로세스를 성공적으로 중지했습니다.{COLOR_RESET}")
        return

    if args.status:
        show_status()
        return

    # 인터럽트 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print_banner()

    # 1. 하드웨어 점검
    check_hardware()

    # 2. 이전 프로세스 클린업
    kill_existing_processes()

    # 3. 환경변수 준비
    env = get_ros_env()

    # 4. 단계별 노드 실행
    print(f"{COLOR_BOLD}🚀 [3단계] 재난구조로봇 서브시스템 순차 기동 시작...{COLOR_RESET}\n")

    # [1/4] TurtleBot3 Bringup
    bringup_cmd = ['ros2', 'launch', 'turtlebot3_bringup', 'robot.launch.py']
    launch_process("🐢 [1/4] TurtleBot3 Bringup (모터/라이다/센서/TF)", bringup_cmd, "bringup.log", env, wait_sec=3.5)

    # [2/4] Camera Node
    if not args.no_camera:
        cam_cmd = [
            'ros2', 'run', 'camera_ros', 'camera_node',
            '--ros-args',
            '-p', 'format:=BGR888',
            '-p', 'width:=640',
            '-p', 'height:=480'
        ]
        launch_process("📷 [2/4] 라즈베리파이 카메라 노드 (camera_ros)", cam_cmd, "camera.log", env, wait_sec=2.0)
    else:
        print(f"  {COLOR_YELLOW}⏩ [2/4] 카메라 노드 구동 건너뜀 (--no-camera){COLOR_RESET}\n")

    # [3/4] Audio Capture Node
    audio_script = find_script("audio_capture_node.py")
    if not args.no_audio and audio_script:
        audio_cmd = ['python3', '-u', audio_script]
        launch_process("🎙️ [3/4] 현장 오디오 수집 노드 (/audio/raw)", audio_cmd, "audio.log", env, wait_sec=1.5)
    else:
        print(f"  {COLOR_YELLOW}⏩ [3/4] 오디오 수집 노드 건너뜀{COLOR_RESET}\n")

    # [4/4] ArUco Detector & Web Streaming
    aruco_script = find_script("aruco_detector.py")
    if not args.no_aruco and aruco_script:
        marker_size = (args.size_cm / 100.0) if args.size_cm is not None else args.marker_size
        aruco_cmd = [
            'python3', '-u', aruco_script,
            '--rotate', '270',
            '--marker-size', str(marker_size)
        ]
        launch_process("🎯 [4/4] ArUco 마커 감지 & 3D 포즈 추정 웹 노드", aruco_cmd, "aruco.log", env, wait_sec=1.5)
    else:
        marker_size = 0.05
        print(f"  {COLOR_YELLOW}⏩ [4/4] ArUco 감지 노드 건너뜀{COLOR_RESET}\n")

    # 5. 네트워크 및 대시보드 출력
    ips = get_network_ips()
    wlan_ip = ips.get("wlan") or "192.168.0.135"
    tail_ip = ips.get("tailscale") or "100.91.141.24"

    dashboard = f"""{COLOR_GREEN}{COLOR_BOLD}
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    🎉 재난구조로봇 모든 세팅 완료! 🎉                            ║
╚══════════════════════════════════════════════════════════════════════════════════╝{COLOR_RESET}

{COLOR_BOLD}🌐 실시간 카메라 & ArUco 3D 포즈 추정 웹 주소:{COLOR_RESET}
    👉 {COLOR_CYAN}Tailscale VPN (PC 권장):  http://{tail_ip}:8080{COLOR_RESET}
    👉 {COLOR_CYAN}로컬 Wi-Fi 내부망:        http://{wlan_ip}:8080{COLOR_RESET}

{COLOR_BOLD}🎯 ArUco 3D 포즈 추정 모델 (Option B):{COLOR_RESET}
    • 기준 마커 크기 : {COLOR_CYAN}{marker_size * 100.0:.1f} cm{COLOR_RESET} (변경: --size-cm 5.0 또는 --marker-size 0.05)
    • 카메라 모델   : {COLOR_CYAN}Sony IMX219 (640x480, FOV 62.2°, f≈530px){COLOR_RESET}
    • 3D 측정 데이터 : {COLOR_WHITE}실제 직선거리(Dist), 전방깊이(Z), 좌우편차(X), 상하편차(Y), 3D 축 표시{COLOR_RESET}

{COLOR_BOLD}📡 활성화된 핵심 ROS 2 토픽:{COLOR_RESET}
    • {COLOR_WHITE}/cmd_vel{COLOR_RESET}            : 로봇 모터 주행 제어 명령
    • {COLOR_WHITE}/scan{COLOR_RESET}               : 360도 LDS-01 라이다 거리 센서 데이터
    • {COLOR_WHITE}/camera/image_raw{COLOR_RESET}   : 480x640 실시간 카메라 영상
    • {COLOR_WHITE}/audio/raw{COLOR_RESET}          : 재난 현장 USB 마이크 실시간 음성 스트림
    • {COLOR_WHITE}/odom, /tf{COLOR_RESET}          : 로봇 오도메트리 위치 및 좌표 변환

{COLOR_BOLD}📝 통합 로그 저장 위치:{COLOR_RESET}
    • Bringup 로그 : {LOG_DIR}/bringup.log
    • 카메라 로그  : {LOG_DIR}/camera.log
    • 오디오 로그  : {LOG_DIR}/audio.log
    • ArUco 로그   : {LOG_DIR}/aruco.log

{COLOR_MAGENTA}{COLOR_BOLD}💡 사용 팁:{COLOR_RESET}
    • 상태 확인 : 다른 터미널에서 {COLOR_CYAN}start.py --status{COLOR_RESET}
    • 일괄 종료 : 현재 화면에서 {COLOR_YELLOW}[Ctrl + C]{COLOR_RESET} 또는 다른 터미널에서 {COLOR_RED}stop.py{COLOR_RESET}
"""
    print(dashboard)

    if args.daemon:
        print(f"{COLOR_GREEN}✔ 백그라운드 모드(-d)로 실행되었습니다. 스크립트를 종료합니다.{COLOR_RESET}")
        return

    # 6. 실시간 프로세스 상태 모니터링 루프
    print(f"{COLOR_YELLOW} 로봇이 정상 작동 중입니다. 종료하려면 키보드 [Ctrl + C]를 누르세요.{COLOR_RESET}\n")

    try:
        while True:
            time.sleep(2.0)
            # 주기적으로 프로세스 생존 검사
            for item in MANAGED_PROCESSES:
                proc = item["proc"]
                name = item["name"]
                ret = proc.poll()
                if ret is not None:
                    print(f"\n{COLOR_RED}⚠️  경고: [{name}] 프로세스가 예기치 않게 종료되었습니다! (종료코드: {ret}){COLOR_RESET}")
                    print(f"👉 로그 확인: {item['log_path']}")
                    # 프로세스 리스트에서 재알림 방지
                    MANAGED_PROCESSES.remove(item)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup_all()


if __name__ == '__main__':
    main()
