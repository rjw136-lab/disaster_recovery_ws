# 🚨 Disaster Recovery TurtleBot3

**통신 단절 적응형 실내 재난 탐사 및 자율 복귀 로봇 시스템 (TurtleBot3 Burger · ROS 2 Humble)**

![ROS 2](https://img.shields.io/badge/ROS%202-Humble-22314E?logo=ros)
![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-E95420?logo=ubuntu&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-GTX%201050%20Ti-76B900?logo=nvidia&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-ArUco%20%2F%20solvePnP-5C3EE8?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)
![YOLO](https://img.shields.io/badge/YOLO-26n-00FFFF)
![Status](https://img.shields.io/badge/Status-Prototype%20(Host%20AI%20verified)-orange)

> 원격 탐사 중 **Wi-Fi가 끊기거나(Heartbeat 3초 타임아웃)** **가연성 가스가 감지되면(MQ-2)**,
> 로봇이 Host PC 도움 없이 **온보드 연산만으로** 지나온 경로를 역순으로 되짚어 출발점 **반경 15 cm 이내**로 돌아온다.

---

## 📑 목차

1. [핵심 기능](#-핵심-기능)
2. [시스템 아키텍처](#-시스템-아키텍처)
3. [상태 머신](#-상태-머신-recovery_controller)
4. [하드웨어 구성](#-하드웨어-구성)
5. [저장소 구조](#-저장소-구조)
6. [설치](#-설치)
7. [실행](#-실행)
8. [ROS 2 인터페이스](#-ros-2-인터페이스)
9. [주요 파라미터](#-주요-파라미터)
10. [필드 테스트](#-필드-테스트)
11. [알려진 한계 및 TODO](#-알려진-한계-및-todo)
12. [참고 문헌](#-참고-문헌)

---

## ✨ 핵심 기능

| 기능 | 설명 | 실행 위치 |
| :--- | :--- | :--- |
| **Heartbeat 워치독** | `/heartbeat` 3.0 s 미수신 시 즉시 정지(v=0, ω=0) 후 원격 명령 차단 | RPi 4 |
| **가스 워치독** | MQ-2 → ADS1115(I2C) → `/gas/level` ≥ 0.40 이면 동일한 비상 복귀 시퀀스 진입 | RPi 4 |
| **오도메트리 로거** | 탐색 중 Δd ≥ 0.3 m 또는 \|Δθ\| ≥ 25° 마다 (x, y, θ)를 LIFO 스택에 push | RPi 4 |
| **LIFO 백트래킹** | 스택을 pop하며 *제자리 회전 → 직진* (Rotate-then-Translate) 추종 | RPi 4 |
| **LiDAR 긴급 정지** | 복귀 중 전방 ±20°, 0.12–0.25 m 장애물 감지 시 정지·대기, 해제 시 재개 | RPi 4 |
| **ArUco 정밀 도킹** | 원점 0.5 m 진입 시 ArUco(ID 0, 10 cm)를 `solvePnP`로 추정해 슬립 오차 보정 | RPi 4 |
| **로컬 블랙박스** | 비상 전환 시 MicroSD에 영상(mp4) + 텔레메트리(csv) 기록 | RPi 4 |
| **비전 AI** | Ultralytics **YOLO26n** (`yolo26n.pt`)으로 조난자(`person`) 탐지 · 압축 영상 수신 · 270° 회전 보정 · Zero-Lag 최신 프레임 워커 · 연속 2프레임 디바운스 | Host PC (GPU) |
| **오디오 캡처** | AB13X USB 마이크(`plughw:1,0`, 48 kHz S16_LE mono)를 `arecord`로 읽어 `/audio/raw`로 PCM 스트리밍 | RPi 4 |
| **음성 AI** | `/audio/raw` → 16 kHz 다운샘플 → RMS VAD → Faster-Whisper(int8, 한국어)로 "살려주세요 / 도와주세요 / help" 등 구조 요청 키워드 검출 | Host PC (GPU) |
| **RViz2 관제** | `disaster_station.rviz` — RobotModel + `/scan` + `/map` + `/detection/image` 원클릭 레이아웃 | Host PC |

> **설계 원칙:** 생존·복귀에 필요한 모든 제어 루프는 로봇 내부에서 닫힌다. Host PC는 관제·AI·Heartbeat만 담당하며, Host PC가 사라져도 복귀 기능은 영향을 받지 않는다 (Brooks, 1986의 계층형 제어 구조 개념 참고).

---

## 🏗 시스템 아키텍처

```mermaid
flowchart LR
    subgraph HOST["Host PC · Windows 11 + WSL2 (mirrored)"]
        HB["Heartbeat 10Hz"]
        TELEOP["teleop_twist_keyboard"]
        AI["host_ai_pipeline<br/>YOLO26n + Faster-Whisper"]
        SLAM["Cartographer + RViz2"]
    end

    subgraph RPI["Raspberry Pi 4 · Ubuntu 22.04"]
        RC["recovery_controller"]
        AD["aruco_docking"]
        BB["blackbox_recorder"]
        GB["gas_sensor_node"]
        CAM["v4l2_camera"]
        MIC["audio_capture_node<br/>AB13X USB mic"]
        LDS["LDS-01 driver"]
    end

    subgraph SENS["I2C 센서 보드"]
        MQ2["MQ-2 → 10k/20k 분압"]
        ADS["ADS1115 (0x48)"]
    end

    subgraph OCR["OpenCR 1.0 (순정 펌웨어)"]
        MOT["Dynamixel 구동 / odom"]
    end

    HB -- "/heartbeat" --> RC
    TELEOP -- "/cmd_vel_remote" --> RC
    CAM -- "/camera/image_raw/compressed" --> AI
    MIC -- "/audio/raw" --> AI
    CAM --> AD
    CAM --> BB
    LDS -- "/scan" --> RC
    MOT -- "/odom" --> RC
    MQ2 --> ADS
    ADS -- "I2C bus 1" --> GB
    GB -- "/gas/level" --> RC
    RC -- "/cmd_vel" --> MOT
    AD -- "/cmd_vel" --> MOT
    RC -- "/aruco/dock_trigger" --> AD
    AD -- "/aruco/dock_complete" --> RC
    RC -- "/blackbox/trigger" --> BB
```

### 네트워크 (Dual NIC)

```
 [학교 벽면 LAN] ── 유선 NIC ──┐
                              ├── Host PC (Windows 11) ── WSL2 (networkingMode=mirrored)
 [연구실 공유기 192.168.0.x] ── USB Wi-Fi 동글 ──┘
            │
            └── Wi-Fi ── Raspberry Pi 4 (TurtleBot3)

 공통: ROS_DOMAIN_ID=0, ROS_LOCALHOST_ONLY=0, DDS 멀티캐스트 디스커버리 (Fast DDS 기본)
```

- 학교 망 인증(고정 IP/MAC)은 유선 NIC에서 그대로 유지하고, ROS 2 트래픽은 사설 공유기로만 흐른다.
- WSL2 미러드 모드는 Windows의 NIC/IP를 Linux에 그대로 노출하므로 NAT 포트포워딩 없이 DDS 멀티캐스트가 동작한다. 단, Windows 방화벽(Hyper-V firewall)에서 UDP 수신을 허용해야 할 수 있다. 상세: [`docs/WSL_NETWORK_SETUP.md`](docs/WSL_NETWORK_SETUP.md)
- 공유기에 직접 붙을 수 없는 경우 **Tailscale + CycloneDDS 유니캐스트 피어**로도 운용 가능하지만, 지연(50–150 ms)·지터 때문에 Heartbeat 타임아웃 오작동 위험이 있어 현장 시험에는 사설 Wi-Fi 직결을 권장한다 (`WSL_NETWORK_SETUP.md` §6).

---

## 🔁 상태 머신 (`recovery_controller`)

```mermaid
stateDiagram-v2
    [*] --> INIT
    INIT --> NORMAL_TELEOP: odom 수신 + 첫 heartbeat
    NORMAL_TELEOP --> EMERGENCY_STOP: heartbeat > 3.0s 또는 gas >= 0.40
    EMERGENCY_STOP --> BACKTRACKING_ROTATE: 스택 pop
    BACKTRACKING_ROTATE --> BACKTRACKING_TRANSLATE: 헤딩 오차 <= 0.08 rad
    BACKTRACKING_TRANSLATE --> BACKTRACKING_ROTATE: 헤딩 발산 또는 다음 경유지
    BACKTRACKING_ROTATE --> OBSTACLE_HOLD: 전방 장애물
    BACKTRACKING_TRANSLATE --> OBSTACLE_HOLD: 전방 장애물
    OBSTACLE_HOLD --> BACKTRACKING_ROTATE: 장애물 해제
    BACKTRACKING_TRANSLATE --> HANDOVER_TO_ARUCO: 원점 0.5m 이내 또는 스택 소진
    HANDOVER_TO_ARUCO --> ARUCO_DOCKING
    ARUCO_DOCKING --> MISSION_COMPLETE: 오차 허용범위 10프레임 연속
    MISSION_COMPLETE --> [*]
```

**제어 법칙 요약**

- 회전 단계 (`|e_θ| > 0.08 rad`): `v = 0`, `ω = clamp(Kp_ω · e_θ, ±0.45)`
- 직진 단계: `v = clamp(Kp_v · d, 0.03, 0.12)`, `ω = clamp(0.5 · Kp_ω · e_θ, ±0.45)`
- 도킹 수렴 조건: `|Z − 0.20| ≤ 0.02 m`, `|X| ≤ 0.02 m`, `|yaw| ≤ 0.05 rad` 를 10프레임 연속 만족

---

## 🔧 하드웨어 구성

| 구분 | 사양 |
| :--- | :--- |
| 로봇 | TurtleBot3 Burger (OpenCR 1.0 + Raspberry Pi 4 4/8GB) |
| LiDAR | LDS-01 (USB2LDS) |
| 카메라 | Raspberry Pi Camera Module (CSI, 640×480) — **케이블 배선상 90° 옆으로 장착** → 소프트웨어에서 270° 회전 보정 |
| 가스 센서 | MQ-2 (5 V 구동) → 10k/20k 분압 → **ADS1115** 16-bit I2C ADC → RPi 4 |
| 마이크 | Generic **AB13X USB Audio** (card 1, device 0 = `plughw:1,0`, S16_LE · mono · 48 kHz) |
| Host PC | i7-8700 / GTX 1050 Ti 4GB / Windows 11 + WSL2 Ubuntu 22.04 / PyTorch 2.5.1 + CUDA 12.1 |

### MQ-2 + ADS1115 회로

OpenCR은 TurtleBot3 순정 펌웨어(모터·`/odom`)를 그대로 유지하고, 가스 센서는 RPi 4의 I2C로 읽는다. ADS1115는 **반드시 3.3 V로 급전**한다 (RPi GPIO는 5 V 비허용).

```
 RPi Pin 1 (3.3V) ──── ADS1115 VDD
 RPi Pin 3 (SDA)  ──── ADS1115 SDA
 RPi Pin 5 (SCL)  ──── ADS1115 SCL
 RPi Pin 6 (GND)  ──┬─ ADS1115 GND, ADDR (→ 0x48)
                    ├─ MQ-2 GND
 RPi Pin 2 (5V)   ──┼─ MQ-2 VCC
                    │  MQ-2 AOUT ── [R1 10kΩ] ──┬── ADS1115 AIN0
                    │                          [R2 20kΩ]
                    └──────────────────────────┘

 V_AIN0 = V_out × 20k / (10k + 20k) ≤ 3.33 V   (ADS1115 허용: VDD + 0.3 V = 3.6 V)
 PGA = ±4.096 V, 128 SPS, 연속 변환
```

```bash
sudo apt install -y i2c-tools python3-smbus && sudo usermod -aG i2c $USER
i2cdetect -y 1        # 48 이 보여야 정상
```

상세 결선 및 카메라 캘리브레이션: [`docs/CIRCUIT_AND_CALIBRATION.md`](docs/CIRCUIT_AND_CALIBRATION.md)

---

## 📁 저장소 구조

```
disaster_recovery_ws/
├── README.md
├── LICENSE
├── .gitignore
├── scripts/
│   └── rpi/                             # 라즈베리파이 온보드 편의 실행 스크립트
│       ├── start_camera.sh              # camera_ros 고속 모드 (640x480 @ 30 FPS)
│       ├── start_audio.sh               # AB13X USB 마이크 캡처 노드 시작
│       ├── start_all.sh                 # 로봇 Bringup + 카메라 + 복귀 패키지 원클릭 실행
│       └── stop_all.sh                  # 로봇 관련 프로세스 일괄 안전 종료
├── docs/
│   ├── DISASTER_RECOVERY_GUIDEBOOK.md   # 종합 설계 문서
│   ├── WSL_NETWORK_SETUP.md             # .wslconfig, ROS 2, CUDA 설정
│   ├── CIRCUIT_AND_CALIBRATION.md       # MQ-2 분압 회로, 카메라 캘리브레이션
│   ├── FIELD_TEST_CHECKLIST.md          # 터미널별 실행 순서, 검증 체크리스트
│   ├── Disaster_Recovery_TurtleBot3_Integration_Report.md   # YOLO26n·오디오 통합 구축 보고서
│   └── Disaster_Recovery_TurtleBot3_Integration_Report.docx # 동일 보고서 (Word)
└── src/disaster_recovery/
    ├── package.xml
    ├── setup.py / setup.cfg
    ├── config/
    │   ├── recovery_params.yaml         # 임계값, 속도, 게인
    │   ├── camera_info_sample.yaml      # 예시값 (반드시 직접 캘리브레이션한 값으로 교체)
    │   └── disaster_station.rviz        # RViz2 관제 레이아웃
    ├── launch/
    │   ├── robot_onboard.launch.py      # RPi 4용 (recovery/aruco/gas/blackbox/audio)
    │   └── host_station.launch.py       # Host PC용 (launch argument 지원, use_rviz)
    ├── disaster_recovery/
    │   ├── recovery_controller.py       # 워치독 + 로거 + 백트래킹 + LiDAR 정지 FSM
    │   ├── aruco_docking.py             # solvePnP 기반 도킹 P-제어 (카메라 장착 회전 보정)
    │   ├── host_ai_pipeline.py          # YOLO26n + Faster-Whisper + Heartbeat
    │   ├── gas_sensor_node.py           # ADS1115 → EMA → /gas/level
    │   ├── ads1115_mq2.py               # ADS1115 드라이버 + MQ-2 계산 (ROS 비의존)
    │   ├── blackbox_recorder.py         # MicroSD 영상·CSV 기록
    │   └── audio_capture_node.py        # arecord → /audio/raw PCM 스트리밍
    └── test/
        └── test_ads1115_mq2.py          # 단위 테스트 (pytest)
```

---

## ⚙ 설치

### 공통 (Host PC · RPi 4) — `~/.bashrc`

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export TURTLEBOT3_MODEL=burger
export LDS_MODEL=LDS-01          # RPi 4
```

### Host PC (WSL2)

`C:\Users\<user>\.wslconfig`

```ini
[wsl2]
networkingMode=mirrored
dnsTunneling=true
firewall=true
autoProxy=true
```

```bash
# PowerShell: wsl --shutdown 후 WSL 재진입
sudo apt install -y ros-humble-desktop ros-humble-cv-bridge \
                    ros-humble-teleop-twist-keyboard ros-humble-camera-calibration \
                    ros-humble-turtlebot3-cartographer
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip3 install ultralytics faster-whisper
pip3 install "numpy==1.26.4" "opencv-python==4.10.0.84"     # cv_bridge(NumPy 1.x) 호환 고정
python3 -c "import torch; print(torch.cuda.is_available())"   # True 확인
```

> WSL2 내부에 Linux용 NVIDIA 드라이버를 설치하지 않는다. Windows 드라이버가 WSL로 전달된다.
>
> `yolo26n.pt`는 첫 실행 시 Ultralytics가 자동 다운로드한다 (저장소에는 포함하지 않음). NumPy 2.x가 설치되면 `cv_bridge` import가 실패하므로 위 버전 고정을 유지한다.

### RPi 4

```bash
sudo apt install -y ros-humble-v4l2-camera ros-humble-cv-bridge python3-opencv \
                    ros-humble-image-transport-plugins \
                    python3-smbus i2c-tools alsa-utils
sudo usermod -aG i2c,audio $USER   # 재로그인 필요
arecord -l                          # card 1: AB13X USB Audio 확인 (번호가 다르면 alsa_device 수정)
```

### 단위 테스트 (ROS·하드웨어 불필요)

```bash
cd src/disaster_recovery && python3 -m pytest test -q   # ADS1115 레지스터/MQ-2 계산 8개 테스트
```

### 빌드 (양쪽 모두)

```bash
mkdir -p ~/disaster_recovery_ws && cd ~/disaster_recovery_ws
git clone https://github.com/rjw136-lab/disaster_recovery_ws.git .
colcon build --symlink-install --packages-select disaster_recovery
source install/setup.bash
```

---

## ▶ 실행

### ⚡ 원터치 통합 실행 (추천)

```bash
# Host PC 터미널에서 실행 (RPi 4 원격 브링업 + Host AI 파이프라인 + SLAM + 텔레옵 통합)
python3 scripts/host/start.py   # 또는 PATH 등록 시 start.py
```

### 단계별 수동 실행

| # | 기기 | 명령 |
| :-: | :--- | :--- |
| 1 | RPi 4 | `ros2 launch turtlebot3_bringup robot.launch.py` (또는 `bash scripts/rpi/start_all.sh`) |
| 2 | RPi 4 | `ros2 run camera_ros camera_node --ros-args -p width:=640 -p height:=480 -p format:=BGR888` (또는 `bash scripts/rpi/start_camera.sh`) |
| 3 | RPi 4 | `ros2 launch disaster_recovery robot_onboard.launch.py` |
| 4 | Host | `ros2 launch disaster_recovery host_station.launch.py` (RViz 동시 실행: `use_rviz:=true`) |
| 5 | Host | `ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel_remote` |
| 6 | Host | `ros2 launch turtlebot3_cartographer cartographer.launch.py` |

> ⚠ 텔레옵은 반드시 `/cmd_vel_remote`로 리매핑한다. `/cmd_vel`에 직접 발행하면 워치독의 원격 명령 차단이 무력화된다.

상태 모니터링:

```bash
ros2 topic echo /recovery/status
ros2 topic echo /disaster/victim_alert
ros2 topic echo /disaster/audio_alert
ros2 run rviz2 rviz2 -d $(ros2 pkg prefix disaster_recovery)/share/disaster_recovery/config/disaster_station.rviz
```

### `host_station.launch.py` 인자

| 인자 | 기본값 | 설명 |
| :--- | :-: | :--- |
| `yolo_model_path` | `yolo26n.pt` | YOLO 가중치 경로/이름 |
| `image_topic` | `/camera/image_raw/compressed` | 카메라 토픽 |
| `use_compressed` | `true` | `CompressedImage` 구독 (Wi-Fi 대역폭 절약, 권장) |
| `rotation_angle` | `270` | 영상 회전 (0/90/180/270, `cv2.rotate` 기준) |
| `show_window` | `true` | OpenCV 탐지 창 표시 |
| `heartbeat_rate_hz` | `10.0` | Heartbeat 주기 |
| `confidence_threshold` | `0.40` | YOLO 추론 임계값 |
| `person_conf_threshold` | `0.55` | 조난자 알림 임계값 |
| `only_person_detection` | `true` | COCO class 0(person)만 추론 |
| `min_consecutive_detections` | `2` | 알림 발행에 필요한 연속 탐지 프레임 수 |
| `cuda_device` | `0` | GPU 번호 |
| `whisper_model_size` | `base` | Faster-Whisper 모델 크기 (tiny/base/small) |
| `whisper_device` | `cpu` | Whisper 연산 디바이스 (`cpu`/`cuda`. GTX 1050 Ti VRAM 보호를 위해 CPU 권장) |
| `whisper_compute_type` | `int8` | Whisper CPU 양자화 가속 타입 (`int8` AVX2 권장) |
| `use_rviz` | `false` | `disaster_station.rviz`로 RViz2 동시 실행 |

예) 원본 영상 + 회전 없음 + 창 끄기:

```bash
ros2 launch disaster_recovery host_station.launch.py \
  image_topic:=/camera/image_raw use_compressed:=false rotation_angle:=0 show_window:=false
```

---

## 📡 ROS 2 인터페이스

| 토픽 | 타입 | 발행 → 구독 |
| :--- | :--- | :--- |
| `/heartbeat` | `std_msgs/Header` | host_ai_pipeline → recovery_controller (10 Hz) |
| `/cmd_vel_remote` | `geometry_msgs/Twist` | teleop → recovery_controller |
| `/cmd_vel` | `geometry_msgs/Twist` | recovery_controller, aruco_docking → turtlebot3_node |
| `/odom` | `nav_msgs/Odometry` | turtlebot3_node → recovery_controller, blackbox |
| `/scan` | `sensor_msgs/LaserScan` | LDS-01 → recovery_controller, blackbox |
| `/gas/level` | `std_msgs/Float32` | gas_sensor_node → recovery_controller, blackbox (예열·기준 측정 후 발행) |
| `/gas/voltage` | `std_msgs/Float32` | gas_sensor_node → (모니터링) MQ-2 출력 전압 |
| `/gas/rs_ro` | `std_msgs/Float32` | gas_sensor_node → (모니터링) 데이터시트 ppm 곡선용 Rs/Ro |
| `/gas/sensor_ok` | `std_msgs/Bool` | gas_sensor_node → (모니터링) I2C 읽기 실패 시 false |
| `/camera/image_raw` | `sensor_msgs/Image` | v4l2_camera → aruco_docking, blackbox, (host_ai: `use_compressed:=false`) |
| `/camera/image_raw/compressed` | `sensor_msgs/CompressedImage` | image_transport → host_ai_pipeline (기본, BEST_EFFORT depth 1) |
| `/detection/image` | `sensor_msgs/Image` | host_ai_pipeline → RViz2 (바운딩 박스 + SURVIVOR 배너) |
| `/audio/raw` | `std_msgs/UInt8MultiArray` | audio_capture_node → host_ai_pipeline (48 kHz S16_LE mono, 0.5 s/msg) |
| `/aruco/debug_image` | `sensor_msgs/Image` | aruco_docking → (모니터링, 정방향 회전) |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | v4l2_camera → aruco_docking |
| `/aruco/dock_trigger` | `std_msgs/Bool` | recovery_controller → aruco_docking |
| `/aruco/dock_complete` | `std_msgs/Bool` | aruco_docking → recovery_controller |
| `/blackbox/trigger` | `std_msgs/Bool` | recovery_controller → blackbox_recorder |
| `/recovery/status` | `std_msgs/String` | recovery_controller → (모니터링) |
| `/disaster/victim_alert` | `std_msgs/String` | host_ai_pipeline → (관제) |
| `/disaster/audio_alert` | `std_msgs/String` | host_ai_pipeline → (관제) |

---

## 🎛 주요 파라미터

[`config/recovery_params.yaml`](src/disaster_recovery/config/recovery_params.yaml)

| 파라미터 | 기본값 | 의미 |
| :--- | :-: | :--- |
| `heartbeat_timeout_sec` | 3.0 | 통신 단절 판정 시간 |
| `gas_threshold` | 0.40 | 가스 위험 판정값 (`1 − Rs/Rs_air`, ppm 아님 · 현장 보정 필요) |
| `gas_stale_timeout_sec` / `gas_stale_triggers_recovery` | 2.0 / false | `/gas/level` 끊김 경고 / 끊김 시 복귀 여부 |
| `warmup_sec` / `baseline_samples` | 60 / 100 | MQ-2 예열 시간 / 깨끗한 공기 기준 측정 샘플 수 |
| `rs_air_kohm` | 0.0 | 0보다 크면 저장된 기준값 재사용 (기준 측정 생략) |
| `log_dist_threshold_m` | 0.30 | 경유지 기록 거리 간격 |
| `log_angle_threshold_deg` | 25.0 | 경유지 기록 회전 간격 |
| `waypoint_reach_radius_m` | 0.08 | 경유지 도달 판정 반경 |
| `linear_speed_max` / `angular_speed_max` | 0.12 / 0.45 | 복귀 속도 상한 |
| `lidar_min_range_m` / `lidar_max_danger_range_m` | 0.12 / 0.25 | 정지 거리 구간 |
| `lidar_fov_half_deg` | 20.0 | 전방 감시 반각 |
| `docking_start_distance_m` | 0.50 | ArUco 도킹 전환 거리 |
| `marker_size_m` / `target_dock_dist_m` | 0.10 / 0.20 | 마커 크기 / 마커 앞 정지 거리 |
| `camera_rotation_deg` (aruco_docking) | 270 | 카메라 장착 회전. solvePnP 결과를 정방향 카메라 좌표계로 변환 |
| `alsa_device` / `sample_rate` (audio_capture_node) | `plughw:1,0` / 48000 | USB 마이크 장치 / 샘플레이트 |
| `chunk_duration_sec` (audio_capture_node) | 0.5 | `/audio/raw` 메시지 1개 길이 |

Host AI 오디오 파라미터 (`host_ai_pipeline`, 코드 기본값): `audio_sample_rate=48000`, `sample_rate_hz=16000`(Whisper 입력), `audio_buffer_sec=2.0`(50% 중첩), `voice_activity_threshold=0.012`(RMS), `whisper_model_size=base`.

---

## 🧪 필드 테스트

전체 체크리스트: [`docs/FIELD_TEST_CHECKLIST.md`](docs/FIELD_TEST_CHECKLIST.md)

- [ ] 텔레옵 주행 중 `[ODOM LOGGER] Recorded Waypoint` 로그가 0.3 m / 25° 간격으로 출력되는가
- [ ] Host PC의 `host_station.launch.py`를 `Ctrl+C`로 종료 → 약 3초 후 정지하는가
- [ ] 정지 이후 텔레옵 키 입력이 무시되는가
- [ ] `~/blackbox_logs/session_*/`에 mp4·csv가 생성되는가
- [ ] 제자리 회전 → 직진으로 경유지를 역순 추종하는가
- [ ] 복귀 중 전방 약 15 cm에 장애물을 두면 정지하고, 치우면 재개하는가
- [ ] 원점 0.5 m 이내에서 ArUco 도킹으로 전환되고 최종 위치가 출발점 15 cm 이내인가
- [ ] 라이터 가스(점화 금지)를 MQ-2에 근접시켰을 때 동일한 복귀 시퀀스가 시작되는가
- [x] Host 탐지 창/`/detection/image`가 정방향(270° 보정)으로 표시되고 사람 앞에서 `SURVIVOR LOCATED` 배너가 뜨는가
- [ ] 마이크 앞에서 "살려주세요"라고 말하면 `/disaster/audio_alert`가 발행되는가

---

## ⚠ 알려진 한계 및 TODO

본 저장소는 **설계·코드 프로토타입**이다. Host AI(YOLO26n 추론·영상 회전·압축 스트림 수신)는 실제 TurtleBot3 영상으로 확인했으나, 자율 복귀·도킹·가스·오디오 경로의 전체 하드웨어 검증은 아직 진행 중이다.

1. ~~**OpenCR 펌웨어 충돌**~~ → **해결됨.** MQ-2를 ADS1115(I2C)로 RPi 4에 직접 연결하는 방식으로 바꿨다. OpenCR은 순정 TurtleBot3 펌웨어를 유지한다. ADS1115 레지스터 설정과 MQ-2 계산은 단위 테스트로 검증했지만, 실제 하드웨어에서는 아직 검증하지 않았다.
2. ~~**`audio_capture_node`는 스텁**~~ → **구현됨.** `arecord`로 48 kHz PCM을 읽어 `/audio/raw`(`UInt8MultiArray`, 약 96 KB/s)로 보내고, Host가 16 kHz로 다운샘플해 Whisper에 넣는다. 단, 3:1 단순 decimation(저역통과 필터 없음)이라 앨리어싱이 생길 수 있고, 50% 중첩 윈도우 때문에 같은 키워드가 연속 2회 알림될 수 있다.
3. **블랙박스의 Wi-Fi 스트리밍 차단**은 구현되어 있지 않다. 현재는 로컬 기록만 시작한다.
4. **`OBSTACLE_HOLD`와 제자리 회전**: 회전 단계에서도 전방 장애물 검사를 하므로, 벽 근처에서 회전이 막혀 대기 상태가 지속될 수 있다. 회전 단계에서는 검사를 생략하거나 회전 방향 섹터만 검사하는 개선이 필요하다.
5. **도킹 기하**: 마커가 출발 위치 정면 벽에 있다는 가정이다. 출발 자세와 마커 위치의 상대 관계(오프셋)는 현장에서 실측해 `target_dock_dist_m`을 조정해야 한다.
6. **가스 지표**: `/gas/level`은 기준 공기 대비 센서 저항 감소율(`1 − Rs/Rs_air`)이며 ppm 값이 아니다. 기동 후 60초 예열 + 5초 기준 측정 동안은 가스 감시가 비활성 상태이고, 이 시간에는 **깨끗한 공기**에 두어야 한다. 정밀 측정에는 데이터시트상 장시간 burn-in이 권장된다.
7. `camera_info_sample.yaml`의 값은 **예시**이다. 반드시 직접 캘리브레이션한 값을 사용해야 한다. 캘리브레이션은 **회전하지 않은 원본 영상**으로 수행한다 (`aruco_docking`은 원본에서 solvePnP 후 `camera_rotation_deg`로 좌표만 변환).
8. **카메라 장착 회전 보정**: `camera_rotation_deg=270`의 축 변환(수평 오차 X, yaw)은 수식으로만 검증했다. 현장에서 마커를 좌/우에 두고 `/aruco/debug_image`의 X 부호가 맞는지 먼저 확인할 것. 블랙박스 영상은 회전 없이 원본으로 저장된다.
9. Host의 OpenCV 창(`show_window`)은 GUI가 없는 환경(SSH 등)에서는 `show_window:=false`로 꺼야 한다.

---

## 📚 참고 문헌

1. Borenstein, J., & Feng, L. (1996). Measurement and correction of systematic odometry errors in mobile robots. *IEEE Transactions on Robotics and Automation*, 12(6), 869–881.
2. Brooks, R. A. (1986). A robust layered control system for a mobile robot. *IEEE Journal on Robotics and Automation*, 2(1), 14–23.
3. Siegwart, R., Nourbakhsh, I. R., & Scaramuzza, D. (2011). *Introduction to Autonomous Mobile Robots* (2nd ed.). MIT Press.
4. Garrido-Jurado, S., Muñoz-Salinas, R., Madrid-Cuevas, F. J., & Marín-Jiménez, M. J. (2014). Automatic generation and detection of highly reliable fiducial markers under occlusion. *Pattern Recognition*, 47(6), 2280–2292.
5. Collins, T., & Bartoli, A. (2014). Infinitesimal plane-based pose estimation. *International Journal of Computer Vision*, 109(3), 252–286.
6. Zhang, Z. (2000). A flexible new technique for camera calibration. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 22(11), 1330–1334.

---

## 📄 License

[Apache License 2.0](LICENSE) © 2026 Jongwoo Ro (Kyungpook National University)
