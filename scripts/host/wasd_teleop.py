#!/usr/bin/env python3
"""
=================================================================================================
Project: Disaster Recovery TurtleBot3 Burger System
Script : wasd_teleop.py (Intuitive WASD & Arrow Keys Teleoperation Controller)
Author : Autonomous Systems Laboratory / Jong Woo Ro

Description:
  Intuitive, responsive, game-style keyboard teleoperation node for TurtleBot3 Burger.
  Publishes Twist commands to /cmd_vel_remote (for safety watchdog arbitration).

Key Bindings:
  [ W ] or [ ↑ ] : Forward
  [ S ] or [ ↓ ] : Backward
  [ A ] or [ ← ] : Turn Left (Counter-Clockwise)
  [ D ] or [ → ] : Turn Right (Clockwise)
  [ Q ]          : Forward Left
  [ E ]          : Forward Right
  [ X ]          : Backward (TurtleBot3 legacy)
  [ Space ] / [K]: Emergency Stop (v=0, w=0)
  [ + ] / [ = ]  : Increase linear speed by 0.02 m/s
  [ - ] / [ _ ]  : Decrease linear speed by 0.02 m/s
  [ ] ]          : Increase angular speed by 0.10 rad/s
  [ [ ]          : Decrease angular speed by 0.10 rad/s
  [ C ]          : Toggle Cruise Mode (Keep driving without holding key)
  [ Ctrl + C ]   : Quit teleop safely
=================================================================================================
"""

import os
import sys
import time
import select
import termios
import tty
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# ANSI Styling
C_RESET   = "\033[0m"
C_BOLD    = "\033[1m"
C_GREEN   = "\033[32m"
C_YELLOW  = "\033[33m"
C_RED     = "\033[31m"
C_CYAN    = "\033[36m"
C_MAGENTA = "\033[35m"
C_BG_BLUE = "\033[44m"

# TB3 Burger Physical Maximum Limits
BURGER_MAX_LIN = 0.22  # m/s
BURGER_MAX_ANG = 2.84  # rad/s

DEFAULT_LIN = 0.14     # m/s (Comfortable exploring speed)
DEFAULT_ANG = 0.70     # rad/s

LIN_STEP = 0.02        # m/s
ANG_STEP = 0.10        # rad/s
AUTO_STOP_TIMEOUT = 0.50  # sec (Auto stop after key release if not in cruise mode)


def print_ui(lin_speed, ang_speed, state_text, cruise_mode):
    # ANSI clear line and print status bar
    cruise_str = f"{C_GREEN}[ON (크루즈)]{C_RESET}" if cruise_mode else f"{C_YELLOW}[OFF (키를 떼면 정지)]{C_RESET}"
    sys.stdout.write(
        f"\r\033[K"
        f"  선속도: {C_BOLD}{lin_speed:.2f} m/s{C_RESET} | "
        f"각속도: {C_BOLD}{ang_speed:.2f} rad/s{C_RESET} | "
        f"크루즈: {cruise_str} | "
        f"상태: {C_CYAN}{C_BOLD}{state_text}{C_RESET} "
    )
    sys.stdout.flush()


def get_key(timeout=0.05):
    """Non-blocking keyboard read supporting multi-byte escape sequences (Arrow keys)."""
    fd = sys.stdin.fileno()
    rlist, _, _ = select.select([fd], [], [], timeout)
    if not rlist:
        return ''
    ch = sys.stdin.read(1)
    if ch == '\x1b':  # Escape sequence
        rlist2, _, _ = select.select([fd], [], [], 0.02)
        if rlist2:
            ch2 = sys.stdin.read(1)
            if ch2 == '[':
                rlist3, _, _ = select.select([fd], [], [], 0.02)
                if rlist3:
                    ch3 = sys.stdin.read(1)
                    if ch3 == 'A':
                        return 'UP'
                    elif ch3 == 'B':
                        return 'DOWN'
                    elif ch3 == 'C':
                        return 'RIGHT'
                    elif ch3 == 'D':
                        return 'LEFT'
    return ch


class WASDTeleopNode(Node):
    def __init__(self, target_topic='/cmd_vel_remote'):
        super().__init__('wasd_teleop')
        self.pub = self.create_publisher(Twist, target_topic, 10)
        self.target_topic = target_topic
        self.lin_speed = DEFAULT_LIN
        self.ang_speed = DEFAULT_ANG
        self.cruise = False
        self.last_key_time = time.time()
        self.curr_lin = 0.0
        self.curr_ang = 0.0
        self.state_text = "정지 (대기 중)"

    def set_vel(self, lin, ang, state_desc):
        self.curr_lin = max(-BURGER_MAX_LIN, min(BURGER_MAX_LIN, lin))
        self.curr_ang = max(-BURGER_MAX_ANG, min(BURGER_MAX_ANG, ang))
        self.state_text = state_desc
        self.last_key_time = time.time()
        self.publish_cmd()

    def stop(self, state_desc="긴급 정지 (STOP)"):
        self.curr_lin = 0.0
        self.curr_ang = 0.0
        self.state_text = state_desc
        self.publish_cmd()

    def publish_cmd(self):
        msg = Twist()
        msg.linear.x = float(self.curr_lin)
        msg.angular.z = float(self.curr_ang)
        self.pub.publish(msg)

    def check_auto_stop(self):
        """If user releases keys and cruise mode is OFF, gracefully stop robot."""
        if not self.cruise:
            if (self.curr_lin != 0.0 or self.curr_ang != 0.0):
                if (time.time() - self.last_key_time) > AUTO_STOP_TIMEOUT:
                    self.stop("감속 정지 (키 뗌)")
                    print_ui(self.lin_speed, self.ang_speed, self.state_text, self.cruise)


def main():
    topic_name = '/cmd_vel_remote'
    for arg in sys.argv[1:]:
        if arg.startswith('topic:='):
            topic_name = arg.split(':=')[1]

    # Terminal settings
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    rclpy.init()
    node = WASDTeleopNode(target_topic=topic_name)

    banner = f"""
{C_GREEN}{C_BOLD}========================================================================================
  🎮  WASD 직관 텔레옵 원격 조종 제어판 (TurtleBot3 Burger Disaster Recovery)
========================================================================================{C_RESET}
  조종 키 안내:
        {C_BOLD}[ W ] / [ ↑ ]{C_RESET} : 직진 전진
  {C_BOLD}[ A ] / [ ← ]{C_RESET}     {C_BOLD}[ D ] / [ → ]{C_RESET} : 좌회전 / 우회전 (제자리 선회)
        {C_BOLD}[ S ] / [ ↓ ]{C_RESET} : 직진 후진

  대각선:
      {C_BOLD}[ Q ]{C_RESET} : 전진 좌회전 (곡선 주행)
      {C_BOLD}[ E ]{C_RESET} : 전진 우회전 (곡선 주행)
      {C_BOLD}[ X ]{C_RESET} : 후진 (TB3 순정 키)

  속도 및 정지:
      {C_RED}{C_BOLD}[ Space ] 또는 [ K ]{C_RESET} : 즉시 긴급 정지 (Emergency Stop)
      {C_CYAN}[ + ] / [ - ]{C_RESET}         : 주행 선속도 ±{LIN_STEP} m/s 조절
      {C_CYAN}[ ] ] / [ [ ]{C_RESET}         : 선회 각속도 ±{ANG_STEP} rad/s 조절
      {C_YELLOW}[ C ]{C_RESET}                 : 크루즈 모드 ON/OFF (ON: 키 떼도 주행 유지, OFF: 키 떼면 정지)
      {C_BOLD}[ Ctrl + C ]{C_RESET}          : 조종 종료

  {C_MAGENTA}💡 안내:{C_RESET} 명령은 안전 워치독 게이트웨이인 {C_CYAN}{topic_name}{C_RESET} 로 자동 송출됩니다.
{C_GREEN}========================================================================================{C_RESET}
"""
    print(banner)
    print_ui(node.lin_speed, node.ang_speed, node.state_text, node.cruise)

    try:
        while rclpy.ok():
            key = get_key(timeout=0.05)
            node.check_auto_stop()

            if not key:
                continue

            # Directional inputs
            if key in ('w', 'W', 'UP', 'i'):
                node.set_vel(node.lin_speed, 0.0, "▲ 전진 주행 중 (Forward)")
            elif key in ('s', 'S', 'DOWN'):
                node.set_vel(-node.lin_speed, 0.0, "▼ 후진 주행 중 (Backward)")
            elif key in ('a', 'A', 'LEFT', 'j'):
                node.set_vel(0.0, node.ang_speed, "◀ 제자리 좌회전 중 (Turn Left)")
            elif key in ('d', 'D', 'RIGHT', 'l'):
                node.set_vel(0.0, -node.ang_speed, "▶ 제자리 우회전 중 (Turn Right)")
            elif key in ('q', 'Q', 'u'):
                node.set_vel(node.lin_speed, node.ang_speed * 0.7, "↖ 전진 좌회전 (Curve Left)")
            elif key in ('e', 'E', 'o'):
                node.set_vel(node.lin_speed, -node.ang_speed * 0.7, "↗ 전진 우회전 (Curve Right)")
            elif key in ('x', 'X', 'm', ','):
                node.set_vel(-node.lin_speed, 0.0, "▼ 후진 주행 중 (Backward)")
            elif key in (' ', 'k', 'K'):
                node.stop("■ 긴급 정지 (STOPPED)")
            # Speed modifiers
            elif key in ('+', '='):
                node.lin_speed = min(BURGER_MAX_LIN, node.lin_speed + LIN_STEP)
                node.state_text = f"선속도 증가 (+{LIN_STEP} m/s)"
            elif key in ('-', '_'):
                node.lin_speed = max(0.04, node.lin_speed - LIN_STEP)
                node.state_text = f"선속도 감소 (-{LIN_STEP} m/s)"
            elif key == ']':
                node.ang_speed = min(BURGER_MAX_ANG, node.ang_speed + ANG_STEP)
                node.state_text = f"각속도 증가 (+{ANG_STEP} rad/s)"
            elif key == '[':
                node.ang_speed = max(0.20, node.ang_speed - ANG_STEP)
                node.state_text = f"각속도 감소 (-{ANG_STEP} rad/s)"
            elif key in ('c', 'C'):
                node.cruise = not node.cruise
                node.state_text = f"크루즈 모드: {'ON' if node.cruise else 'OFF'}"
            elif key == '\x03':  # Ctrl+C
                break

            print_ui(node.lin_speed, node.ang_speed, node.state_text, node.cruise)

    except KeyboardInterrupt:
        pass
    finally:
        node.stop("종료 정지")
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        node.destroy_node()
        rclpy.shutdown()
        print(f"\n\n{C_GREEN}{C_BOLD}🎮 WASD 텔레옵 조종이 정상 종료되었습니다.{C_RESET}\n")


if __name__ == '__main__':
    main()
