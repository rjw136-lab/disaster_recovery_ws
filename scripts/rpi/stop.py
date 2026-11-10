#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================
Project: Disaster Recovery TurtleBot3 Burger System
File: stop.py
Description:
  재난구조로봇 올인원 원클릭 프로세스 종료 스크립트.
  터미널에서 `stop.py` 한 번만 입력하면 실행 중인 모든 로봇 노드, 카메라, 센서,
  오디오, ArUco 웹 스트리밍 프로세스를 일괄 안전 종료합니다.
=================================================================================================
"""

import sys
import subprocess
import time

COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_RED = "\033[31m"

def stop_all():
    print(f"{COLOR_YELLOW}{COLOR_BOLD}🛑 재난구조로봇 시스템의 모든 노드를 안전하게 종료합니다...{COLOR_RESET}")
    pattern = 'aruco_detector.py|camera_node|robot.launch|turtlebot3_node|hlds_laser|robot_state_publisher|audio_capture_node.py|start.py'
    
    # 1. 1차 정중한 종료 (SIGTERM)
    subprocess.run(['pkill', '-15', '-f', pattern], capture_output=True, check=False)
    time.sleep(1.2)

    # 2. 2차 확실한 강제 종료 (SIGKILL)
    subprocess.run(['pkill', '-9', '-f', pattern], capture_output=True, check=False)
    time.sleep(0.5)

    print(f"{COLOR_GREEN}{COLOR_BOLD}✅ 모터, 라이다, 카메라, 마이크, 웹 스트리밍 프로세스가 모두 완전히 종료되었습니다.{COLOR_RESET}")

if __name__ == '__main__':
    stop_all()
