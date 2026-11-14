#!/usr/bin/env bash
set -euo pipefail

# Interactive GPU-accelerated (Xvfb + VirtualGL V100 EGL + fluxbox + x11vnc + noVNC) GUI Launcher
# Usage:
#   bash scripts/host/start_novnc_gui.sh [collapsed|collapsed_heavy|tunnel|house|maze|desktop|stop]

MODE="${1:-collapsed}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
VNC_PORT="${VNC_PORT:-5900}"
DISPLAY_NUM="${DISPLAY_NUM:-:1}"
ROS_DOMAIN_ID_GUI="${ROS_DOMAIN_ID_GUI:-43}"
GZ_PORT_GUI="${GZ_PORT_GUI:-11360}"

sg docker -c "docker exec jwroh_ros2_humble bash -c '
  mkdir -p /tmp/novnc_gui
  for pf in /tmp/novnc_gui/*.pid; do
    [ -f \"\$pf\" ] && kill \"\$(cat \"\$pf\")\" 2>/dev/null || true
    rm -f \"\$pf\"
  done
  pkill -9 -x gzserver 2>/dev/null || true
  pkill -9 -x gzclient 2>/dev/null || true
  pkill -9 -x rviz2 2>/dev/null || true
  pkill -9 -x async_slam_toolbox_node 2>/dev/null || true
  pkill -9 -x sync_slam_toolbox_node 2>/dev/null || true
'"

if [ "${MODE}" = "stop" ]; then
  echo "[noVNC GUI] Stopped display ${DISPLAY_NUM} and port ${NOVNC_PORT}."
  exit 0
fi

WORLD_DIR="/ws/disaster_recovery_ws/src/disaster_recovery/worlds"
case "${MODE}" in
  collapsed|collapsed_heavy)
    WORLD="${WORLD_DIR}/disaster_collapsed_building.world"; RX="0.0"; RY="-4.0"; RYAW="1.5708" ;;
  tunnel)
    WORLD="${WORLD_DIR}/disaster_tunnel_world.world"; RX="0.0"; RY="-8.0"; RYAW="1.5708" ;;
  house)
    WORLD="${WORLD_DIR}/disaster_house_world.world"; RX="0.0"; RY="-3.5"; RYAW="1.5708" ;;
  maze)
    WORLD="${WORLD_DIR}/disaster_maze_world.world"; RX="-3.2"; RY="-3.2"; RYAW="0.0" ;;
  desktop)
    WORLD=""; RX="0.0"; RY="0.0"; RYAW="0.0" ;;
  *)
    echo "Unknown mode: ${MODE}. Use: collapsed | collapsed_heavy | tunnel | house | maze | desktop | stop"
    exit 1 ;;
esac

sg docker -c "docker exec -d jwroh_ros2_humble bash -c '
  set -e
  mkdir -p /tmp/novnc_gui
  Xvfb ${DISPLAY_NUM} -screen 0 1920x1080x24 -ac +extension GLX +render -noreset >/tmp/novnc_gui/xvfb.log 2>&1 &
  echo \$! > /tmp/novnc_gui/xvfb.pid
  sleep 1.2
  export DISPLAY=${DISPLAY_NUM}

  fluxbox >/tmp/novnc_gui/fluxbox.log 2>&1 &
  echo \$! > /tmp/novnc_gui/fluxbox.pid
  sleep 0.6

  x11vnc -display ${DISPLAY_NUM} -forever -shared -nopw -rfbport ${VNC_PORT} -o /tmp/novnc_gui/x11vnc.log >/dev/null 2>&1 &
  echo \$! > /tmp/novnc_gui/x11vnc.pid

  websockify --web=/usr/share/novnc/ 0.0.0.0:${NOVNC_PORT} localhost:${VNC_PORT} >/tmp/novnc_gui/websockify.log 2>&1 &
  echo \$! > /tmp/novnc_gui/websockify.pid

  if [ -n \"${WORLD}\" ]; then
    set +u
    source /opt/ros/humble/setup.bash
    source /ws/disaster_recovery_ws/install/setup.bash
    set -u
    export ROS_LOCALHOST_ONLY=1
    export ROS_DOMAIN_ID=${ROS_DOMAIN_ID_GUI}
    export GAZEBO_MASTER_URI=http://127.0.0.1:${GZ_PORT_GUI}
    export GAZEBO_MODEL_DATABASE_URI=\"\"
    export QT_X11_NO_MITSHM=1

    python3 - << \"PYEOF\"
import re
src = \"/ws/disaster_recovery_ws/src/disaster_recovery/config/disaster_station.rviz\"
dst = \"/tmp/novnc_gui/live.rviz\"
with open(src, \"r\", encoding=\"utf-8\") as f:
    txt = f.read()
txt = txt.replace(\"Fixed Frame: base_link\", \"Fixed Frame: map\")
txt = txt.replace(\"Target Frame: <Fixed Frame>\", \"Target Frame: base_footprint\")
txt = txt.replace(\"Size (m): 0.03\", \"Size (m): 0.07\")
txt = txt.replace(\"Distance: 2.733222484588623\", \"Distance: 9.0\")
txt = txt.replace(\"Pitch: 0.785398006439209\", \"Pitch: 1.45\")
txt = txt.replace(\"Target Frame Rate: 15\", \"Target Frame Rate: 60\")
txt = re.sub(r\"(- Class: rviz_default_plugins/Image\\s+Enabled:\\s*)true(.*?Value:\\s*/camera/image_raw\\s+Value:\\s*)true\", r\"\\1false\\2false\", txt, flags=re.DOTALL)
with open(dst, \"w\", encoding=\"utf-8\") as f:
    f.write(txt)
PYEOF

    ros2 launch disaster_recovery disaster_world.launch.py \
      x_pose:=${RX} y_pose:=${RY} yaw_pose:=${RYAW} world:=${WORLD} gui:=false \
      >/tmp/novnc_gui/gzserver.log 2>&1 &
    echo \$! > /tmp/novnc_gui/gzserver.pid

    sleep 6
    ros2 launch slam_toolbox online_async_launch.py use_sim_time:=true \
      >/tmp/novnc_gui/slam.log 2>&1 &
    echo \$! > /tmp/novnc_gui/slam.pid

    vglrun -d egl gzclient >/tmp/novnc_gui/gzclient.log 2>&1 &
    echo \$! > /tmp/novnc_gui/gzclient.pid
    sleep 4
    vglrun -d egl rviz2 -d /tmp/novnc_gui/live.rviz --ros-args -p use_sim_time:=true \
      >/tmp/novnc_gui/rviz2.log 2>&1 &
    echo \$! > /tmp/novnc_gui/rviz2.pid

    sleep 10
    python3 - << \"PYEOF\"
import ctypes
x11 = ctypes.CDLL(\"libX11.so.6\")
x11.XOpenDisplay.argtypes = [ctypes.c_char_p]; x11.XOpenDisplay.restype = ctypes.c_void_p
x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]; x11.XDefaultRootWindow.restype = ctypes.c_ulong
x11.XQueryTree.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)), ctypes.POINTER(ctypes.c_uint)]
x11.XFetchName.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_char_p)]
x11.XMoveResizeWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint]
x11.XFlush.argtypes = [ctypes.c_void_p]; x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
dpy = x11.XOpenDisplay(b\"${DISPLAY_NUM}\")
if dpy:
    root = x11.XDefaultRootWindow(dpy)
    r, p, ch, n = ctypes.c_ulong(), ctypes.c_ulong(), ctypes.POINTER(ctypes.c_ulong)(), ctypes.c_uint()
    if x11.XQueryTree(dpy, root, ctypes.byref(r), ctypes.byref(p), ctypes.byref(ch), ctypes.byref(n)):
        for i in range(int(n.value)):
            np = ctypes.c_char_p()
            if x11.XFetchName(dpy, ch[i], ctypes.byref(np)) and np.value:
                t = np.value.decode(\"utf-8\", \"ignore\").lower()
                if \"gazebo\" in t: x11.XMoveResizeWindow(dpy, ch[i], 0, 0, 1220, 1040)
                elif \"rviz\" in t: x11.XMoveResizeWindow(dpy, ch[i], 1220, 0, 700, 1040)
    x11.XFlush(dpy); x11.XCloseDisplay(dpy)
PYEOF
  fi
'"

echo "[noVNC GUI] Starting ${MODE} (GPU: Tesla V100 EGL + slam_toolbox)..."
echo "  Open browser: http://localhost:${NOVNC_PORT}/vnc.html?autoconnect=true&resize=scale"
