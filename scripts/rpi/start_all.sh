#!/usr/bin/env bash
# TurtleBot3 Bringup + camera_ros + 재난 복귀 전체 노드 통합 구동 스크립트

source /opt/ros/humble/setup.bash
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi

# 기존 프로세스 정리
pkill -f 'camera_node|camera_ros|robot.launch|turtlebot3_node|hlds_laser|robot_state_publisher|recovery_controller|aruco_docking' 2>/dev/null || true
sleep 1

echo "============================================================"
echo "  🐢 [1/3] TurtleBot3 로봇 Bringup 구동 중 (모터, 라이다, 센서)..."
ros2 launch turtlebot3_bringup robot.launch.py > /tmp/bringup.log 2>&1 &
BRINGUP_PID=$!
sleep 4

echo "  📷 [2/3] 라즈베리파이 카메라 노드 구동 중 (camera_ros)..."
ros2 run camera_ros camera_node --ros-args \
    -p width:=640 \
    -p height:=480 \
    -p format:=BGR888 \
    -p FrameDurationLimits:="[33333, 33333]" > /tmp/camera_node.log 2>&1 &
CAM_PID=$!
sleep 2

echo "  🚨 [3/3] 온보드 재난 복귀 패키지 구동 (robot_onboard.launch.py)..."
echo "============================================================"
echo "  🎮 로봇 모터(/cmd_vel), 라이다(/scan), 카메라, 오디오 모두 활성화됨!"
echo "  🛑 종료하려면 언제든지 키보드 [Ctrl + C]를 누르세요."
echo "============================================================"

cleanup() {
    echo -e "\n🛑 종료 신호 감지. 로봇과 센서 프로세스를 모두 정리합니다..."
    kill $CAM_PID $BRINGUP_PID 2>/dev/null || true
    pkill -f 'camera_node|camera_ros|robot.launch|turtlebot3_node|hlds_laser|robot_state_publisher|recovery_controller|aruco_docking|audio_capture_node' 2>/dev/null || true
    echo "✅ 모든 프로세스가 깔끔하게 종료되었습니다."
    exit 0
}
trap cleanup SIGINT SIGTERM

if [ -f "$HOME/disaster_recovery_ws/install/setup.bash" ]; then
    source "$HOME/disaster_recovery_ws/install/setup.bash"
    ros2 launch disaster_recovery robot_onboard.launch.py
else
    echo "⚠ disaster_recovery_ws 빌드 후 실행하세요."
fi
