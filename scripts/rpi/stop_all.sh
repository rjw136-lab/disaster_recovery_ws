#!/usr/bin/env bash
# 실행 중인 TurtleBot3 Bringup, 카메라, 오디오 노드 전체 정리
echo "🛑 실행 중인 TurtleBot3 Bringup, 카메라, 오디오 노드를 모두 종료합니다..."
pkill -f 'camera_node|camera_ros|robot.launch|turtlebot3_node|hlds_laser|robot_state_publisher|audio_capture_node' 2>/dev/null || true
sleep 1
echo "✅ 완전히 종료되었습니다."
