#!/usr/bin/env bash
# Raspberry Pi 4 Camera Module 고속 모드 (camera_ros, 640x480 @ 30 FPS)
source /opt/ros/humble/setup.bash
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi

pkill -f camera_node 2>/dev/null || true
sleep 1

echo "📷 카메라 노드 고속 모드 시작 (640x480 @ 30 FPS)..."
nohup ros2 run camera_ros camera_node --ros-args \
    -p width:=640 \
    -p height:=480 \
    -p format:=BGR888 \
    -p FrameDurationLimits:="[33333, 33333]" \
    < /dev/null > /tmp/camera_node.log 2>&1 &
sleep 2
echo "✅ 카메라 프로세스 구동 확인:"
ps aux | grep camera_node | grep -v grep
