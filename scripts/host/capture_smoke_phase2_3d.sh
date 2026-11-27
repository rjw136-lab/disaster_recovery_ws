#!/usr/bin/env bash
# Phase 2 smoke 3D evidence: re-run u_pocket_trap with Gazebo GUI (vglrun EGL) + RViz2
# on a private Xvfb display, record with ffmpeg x11grab, extract PNG frames.
# Display :9 isolated from any existing GUI; Gz port/domain match the smoke run so we
# can attach gzclient to the benchmark's own gzserver.
set -uo pipefail
DISP=:9
GZ_PORT=11467
ROS_DOMAIN=63
WS=/ws/disaster_recovery_ws
OUT=/tmp/novnc_gui
VID=$WS/docs/images/evidence/smoke_phase2_u_pocket_trap_3d.mp4
mkdir -p "$OUT" "$WS/docs/images/evidence"

pkill -f "Xvfb $DISP" 2>/dev/null || true
pkill -x fluxbox 2>/dev/null || true
pkill -x x11vnc 2>/dev/null || true
sleep 1

Xvfb "$DISP" -screen 0 1920x1080x24 -ac +extension GLX +render -noreset >"$OUT/xvfb9.log" 2>&1 &
sleep 1.5
DISPLAY=$DISP fluxbox >"$OUT/fluxbox9.log" 2>&1 &
sleep 0.8

cd "$WS"
set +u
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
set -u
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=$ROS_DOMAIN
export GAZEBO_MASTER_URI="http://127.0.0.1:$GZ_PORT"
export GAZEBO_IP=127.0.0.1 GAZEBO_MODEL_DATABASE_URI=''
export TURTLEBOT3_MODEL=burger
export GAZEBO_MODEL_PATH="$HOME/.gazebo/models:$WS/src/disaster_recovery/models:/opt/ros/humble/share/turtlebot3_gazebo/models"
export QT_X11_NO_MITSHM=1

python3 scripts/host/ab_disaster_benchmark.py \
  --scenario u_pocket_trap --trials 1 --seed 20261124 \
  --proposed-strategy phase4 --proposed-only \
  --out-dir benchmark_results/smoke_phase2 \
  --gz-port "$GZ_PORT" --ros-domain-id "$ROS_DOMAIN" \
  >"$OUT/bench_u_pocket_gui.log" 2>&1 &
BENCH_PID=$!
echo "benchmark pid $BENCH_PID"

# Wait for the benchmark's gzserver to come up on the port
for i in $(seq 1 40); do
  if ss -ltn 2>/dev/null | grep -q ":$GZ_PORT "; then break; fi
  sleep 0.5
done
sleep 1

DISPLAY=$DISP vglrun -d egl gzclient >"$OUT/gzclient9.log" 2>&1 &
sleep 8

ffmpeg -y -f x11grab -video_size 1920x1080 -framerate 20 -i "$DISP" \
  -c:v libx264 -preset ultrafast -crf 22 -pix_fmt yuv420p "$VID" >"$OUT/ffmpeg9.log" 2>&1 &
FFMPEG_PID=$!

wait "$BENCH_PID"
BENCH_RC=$?
sleep 3
kill "$FFMPEG_PID" 2>/dev/null || true
pkill -f "Xvfb $DISP" 2>/dev/null || true
pkill -x rviz2 2>/dev/null || true
pkill -x gzclient 2>/dev/null || true
pkill -x fluxbox 2>/dev/null || true
pkill -x x11vnc 2>/dev/null || true

# Extract evidence frames
ffmpeg -y -ss 00:01:00 -i "$VID" -frames:v 1 "$WS/docs/images/evidence/smoke_phase2_u_pocket_trap_3d_mid.png" >/dev/null 2>&1
ffmpeg -y -sseof -5 -i "$VID" -frames:v 1 "$WS/docs/images/evidence/smoke_phase2_u_pocket_trap_3d_end.png" >/dev/null 2>&1

echo "BENCH_RC=$BENCH_RC"
ls -l "$VID" "$WS/docs/images/evidence/smoke_phase2_u_pocket_trap_3d_*.png" 2>&1
