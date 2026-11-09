#!/usr/bin/env bash
# Generic AB13X USB 마이크 오디오 캡처 노드 시작 (/audio/raw)
source /opt/ros/humble/setup.bash
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi

pkill -f "audio_capture_node" 2>/dev/null || true
sleep 0.5

echo "🎙 Generic AB13X USB 마이크 오디오 캡처 노드 시작 (/audio/raw)..."
if [ -f "$HOME/disaster_recovery_ws/install/setup.bash" ]; then
    source "$HOME/disaster_recovery_ws/install/setup.bash"
    ros2 run disaster_recovery audio_capture_node
else
    python3 -u "$HOME/audio_capture_node.py"
fi
