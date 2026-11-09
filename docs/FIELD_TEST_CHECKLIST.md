# [공학 가이드 3] 실행 시나리오 및 통신 단절·가스 복귀 필드 테스트 체크리스트

---

## 1. 다중 터미널 실행 시퀀스 (Execution Sequence)

시스템 구동 전 Host PC와 Raspberry Pi 4가 동일한 사설 무선 공유기(192.168.0.x)에 연결되어 있고, `ROS_DOMAIN_ID=0`이 등록되었는지 확인한다.

### 1.1 Raspberry Pi 4 (온보드 로봇 본체) 실행

#### [RPi 4 - Terminal 1] 터틀봇3 버거 하드웨어 브링업 (모터 & LDS-01)
```bash
export TURTLEBOT3_MODEL=burger
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
ros2 launch turtlebot3_bringup robot.launch.py
```
*동작 검증*: `/odom`, `/scan`, `/tf` 토픽이 정상 발행되는지 확인한다.

#### [RPi 4 - Terminal 2] 카메라 센서 드라이버 구동

```bash
source /opt/ros/humble/setup.bash

# 방법 1 (권장: camera_ros - 640x480 @ 30 FPS 고속 구동)
ros2 run camera_ros camera_node --ros-args \
  -p width:=640 -p height:=480 -p format:=BGR888 \
  -p FrameDurationLimits:="[33333, 33333]"

# 또는 스크립트 실행:
bash ~/disaster_recovery_ws/scripts/rpi/start_camera.sh

# 방법 2 (대체: v4l2_camera 사용 시)
# ros2 run v4l2_camera v4l2_camera_node --ros-args \
#   -p video_device:="/dev/video0" \
#   -p image_size:="[640,480]" \
#   -p pixel_format:="YUYV" \
#   -p camera_info_url:=file:///home/ubuntu/camera_info.yaml \
#   -r image_raw:=/camera/image_raw \
#   -r camera_info:=/camera/camera_info
```
*동작 검증*: `ros2 topic hz /camera/image_raw` 명령으로 약 25~30 Hz 스트리밍을 확인한다.
`ros-humble-image-transport-plugins`가 설치되어 있으면 `/camera/image_raw/compressed`(JPEG)도 함께 발행되며, Host AI는 기본적으로 이 압축 토픽을 구독한다 (`ros2 topic hz /camera/image_raw/compressed`).

마이크 확인 (AB13X USB 마이크가 card 1, device 0으로 잡혀야 한다):
```bash
arecord -l                                              # card 1: AB13X USB Audio
arecord -D plughw:1,0 -f S16_LE -r 48000 -c 1 -d 3 test.wav && aplay test.wav
```

#### [RPi 4 - Terminal 3] 자율 복귀 및 안전 워치독 패키지 통합 구동
```bash
source /opt/ros/humble/setup.bash
source ~/disaster_recovery_ws/install/setup.bash
ros2 launch disaster_recovery robot_onboard.launch.py
```
*동작 검증*: `recovery_controller`, `aruco_docking`, `gas_sensor_node`, `blackbox_recorder`, `audio_capture_node` 노드가 일괄 초기화되며 대기 상태(`INIT`)로 진입한다. `gas_sensor_node`는 60초 예열 + 5초 기준 측정 후부터 `/gas/level`을 발행하므로, **기동 직후 1분간은 깨끗한 공기에서 대기**한다.

---

### 1.2 Host PC (Windows 11 WSL2 관제 스테이션) 실행

#### [Host PC - Terminal 1] Host AI 파이프라인 및 10Hz 하트비트 실행
```bash
source /opt/ros/humble/setup.bash
source ~/disaster_recovery_ws/install/setup.bash
ros2 launch disaster_recovery host_station.launch.py
```
*동작 검증*:
- 10Hz 주기로 `/heartbeat` 발행 시작.
- RPi 4의 `recovery_controller` 노드가 "First Heartbeat received" 로그와 함께 `NORMAL_TELEOP` 모드로 전이.
- GTX 1050 Ti CUDA 가속 기반 YOLO26n 및 Faster-Whisper(int8) 로드 완료.
- `Disaster Recovery - YOLO26n Host Station` OpenCV 창에 **정방향(270° 회전 보정)** 영상이 표시된다.
- 주요 launch argument (필요 시):
  ```bash
  ros2 launch disaster_recovery host_station.launch.py \
    image_topic:=/camera/image_raw use_compressed:=false \   # 원본 영상 사용 시
    rotation_angle:=0 show_window:=false use_rviz:=true        # 회전 없음 / 창 끄기 / RViz 동시 실행
  ```

#### [Host PC - Terminal 2] 원격 텔레옵(Teleoperation) 조종 노드 실행
```bash
source /opt/ros/humble/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/cmd_vel_remote
```
*주의*: 조종 명령은 모터에 직결되는 `/cmd_vel`이 아니라, 안전 게이트웨이인 `/cmd_vel_remote`로 리매핑하여 송출한다.

#### [Host PC - Terminal 3] 2D SLAM 및 RViz2 관제 화면 구동
```bash
source /opt/ros/humble/setup.bash
ros2 launch turtlebot3_cartographer cartographer.launch.py
```
*동작 검증*: 2D 격자 지도(Occupancy Grid)와 로봇의 현재 위치, 라이다 스캔 포인트가 실시간 렌더링된다.

---

## 2. 단계별 필드 테스트 및 성능 검증 시나리오

```
 [출발점: 베이스캠프]  ================ 탐색 주행 ===============>  [재난 탐사 구역]
 (ArUco ID:0 부착)                                                   (통신 음영/가스 누출)
        ▲                                                                     │
        │                                                                     │
        └───────────────── 자율 백트래킹 복귀 (LIFO 역추종) ──────────────────┘
```

### 2.1 [테스트 1] 정상 탐색 및 오도메트리 로깅 무결성 검증
1. Host PC Terminal 2에서 키보드로 로봇을 원격 조종하여 약 5~10m 전방으로 주행시킨다.
2. 회전과 직진을 번갈아 수행하며 RPi 4 Terminal 3의 로그를 관찰한다.
3. 이동 거리 $\Delta d \ge 0.30\,\text{m}$ 또는 헤딩각 $|\Delta \theta| \ge 25.0^\circ$ 마다 `[ODOM LOGGER] Recorded Waypoint #N` 로그가 출력되는지 확인한다.
4. LIFO 스택에 최소 15~20개 이상의 유효 경유지가 누적되었는지 확인한다.

### 2.2 [테스트 2] 통신 두절(Heartbeat Timeout) 시나리오 검증
1. 로봇이 복도 끝 모퉁이를 돌아 전진 주행 중일 때, Host PC Terminal 1에서 `Ctrl + C`를 눌러 `host_station.launch.py`를 강제 종료한다.
2. **동작 검증 항목**:
   - [ ] **정지 시점**: 마지막 하트비트 수신 시각으로부터 정확히 $3.0\,\text{s}$ 경과 즉시 로봇이 전진을 중단하고 비상 정지($v=0, \omega=0$)하는가?
   - [ ] **명령 차단**: Host PC Terminal 2에서 키보드를 입력해도 모터가 반응하지 않고 원격 제어가 물리적으로 차단되는가?
   - [ ] **블랙박스 기동**: RPi 4의 MicroSD 디렉터리(`/home/ubuntu/blackbox_logs/session_...`)에 `incident_video.mp4`와 `incident_telemetry.csv` 파일이 즉각 생성되는가?
   - [ ] **상태 전이**: 상태 머신이 `EMERGENCY_STOP`을 거쳐 `BACKTRACKING_ROTATE`로 자동 전이되는가?

### 2.3 [테스트 3] Rotate-then-Translate 백트래킹 역추종 검증
1. 로봇이 LIFO 스택에서 가장 최근에 저장된 경유지를 Pop하여 목표 헤딩각을 계산한다.
2. **동작 검증 항목**:
   - [ ] **제자리 회전**: 선속도를 0으로 유지한 채 제자리에서 회전하여 목표 방위각 오차를 $0.08\,\text{rad}$ ($4.6^\circ$) 이내로 정렬하는가?
   - [ ] **직진 이동**: 정렬 완료 후 선속도 $0.12\,\text{m/s}$ 이하로 목표 경유지 반경 $0.08\,\text{m}$ 이내까지 직진하는가?
   - [ ] **연속 팝핑**: 각 경유지 도달 시 순차적으로 스택을 비우며 출발지를 향해 왔던 길을 그대로 거슬러 올라가는가?

### 2.4 [테스트 4] 라이다 동적 장애물 긴급 정지 검증
1. 로봇이 백트래킹 복귀 주행을 수행하는 도중, 로봇 전방 약 15cm 거리에 테스트 상자나 발을 불시에 놓는다.
2. **동작 검증 항목**:
   - [ ] LDS-01 전방 $\pm 20^\circ$ 영역에서 거리 $0.12 \sim 0.25\,\text{m}$ 내 물체가 감지되는 즉시 로봇이 제동하여 `OBSTACLE_HOLD` 상태로 진입하는가?
   - [ ] 장애물이 전방에 머무는 동안 충돌 없이 정지 상태를 유지하는가?
   - [ ] 장애물을 치우는 즉시 `[LIDAR SAFETY] Obstacle cleared` 로그와 함께 백트래킹 주행이 안전하게 재개되는가?

### 2.5 [테스트 5] 베이스캠프 도달 및 ArUco 마커 15cm 정밀 도킹 검증
1. 로봇이 출발 원점 기준 반경 $0.50\,\text{m}$ 이내에 도달하거나 스택이 모두 소진되었을 때, `HANDOVER_TO_ARUCO` 신호가 발행되는지 확인한다.
2. 베이스캠프 벽면에 부착된 ArUco 마커(ID: 0, $10 \times 10\,\text{cm}$)를 카메라가 포착한다.
3. **동작 검증 항목**:
   - [ ] 카메라 캘리브레이션 파라미터 기반 `cv2.solvePnP`로 마커의 $X$ 편차와 $Z$ 거리가 실시간 계산되는가?
   - [ ] 횡방향 오차 $X \approx 0$ 및 전방 거리 $Z = 0.20\,\text{m}$를 향해 P-제어기로 미세 궤적 수렴을 수행하는가?
   - [ ] 바퀴 미끄러짐으로 인한 누적 오차를 상쇄하고 최종 정지 위치가 출발 원점 반경 **15cm 이내**에 안착하는가?
   - [ ] 정렬 허용 오차($\Delta Z \le 2\,\text{cm}, \Delta X \le 2\,\text{cm}, \Delta \text{yaw} \le 3^\circ$)가 10 프레임 연속 만족되어 `MISSION_COMPLETE`로 안전 착지하는가?

### 2.6 [테스트 6] 위험 가스(MQ-2) 누출 강제 복귀 검증
1. 정상 원격 탐색 주행 중, MQ-2 센서 근처에 가스 라이터 노즐을 대고 소량의 부탄가스를 분사한다 (**점화 금지**, 환기된 장소에서 수행).
2. **동작 검증 항목**:
   - [ ] `ros2 topic echo /gas/voltage` 값(분압 역변환된 MQ-2 출력)이 신속히 상승하는가?
   - [ ] `/gas/level` 토픽 값이 임계치 $0.40$을 초과하는 순간 통신 단절과 동일하게 비상 정지 및 자율 복귀 시퀀스가 즉각 작동하는가?
   - [ ] ADS1115 배선을 분리하면 `/gas/sensor_ok`가 `false`가 되고, `recovery_controller`가 `/gas/level stale` 경고를 출력하는가? (`gas_stale_triggers_recovery: true`로 설정 시에는 복귀가 시작되어야 한다)
