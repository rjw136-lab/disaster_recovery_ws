# 🚨 Disaster Recovery TurtleBot3 Burger 종합 구축 보고서
**YOLO26n 비전 AI 연동, ArUco 정밀 도킹 및 통신 단절 적응형 자율 복귀 시스템**

- **작성 일자:** 2026년 11월 6일 (11월 9일 개정: USB 마이크 오디오 파이프라인, ArUco 회전 보정, launch 인자 추가)
- **시스템 환경:** Ubuntu 22.04 LTS (WSL2) · ROS 2 Humble · Python 3.10
- **하드웨어 구성:** TurtleBot3 Burger (Raspberry Pi 4 + OpenCR 1.0) · Host PC (NVIDIA GeForce GTX 1050 Ti)
- **핵심 인공지능:** Ultralytics YOLO26n (Nano 모델, 270도 회전 실시간 보정, CUDA 가속) · Faster-Whisper (base, int8)

---

## 1. 프로젝트 개요 및 시스템 목적

본 시스템은 원격 탐사 중 **Wi-Fi 통신 단절(Heartbeat 3.0초 타임아웃)** 또는 **유해 가스(MQ-2 센서 검출)**가 발생하는 가혹한 재난 환경에서, 터틀봇3 버거가 Host PC 도움 없이 **온보드 연산(RPi 4)만으로** 지나온 오도메트리 경로를 역추종(LIFO 백트래킹)하여 출발점 **반경 15 cm 이내**로 자율 복귀하고, **ArUco 마커 기반 P-Controller 시각 서보잉**을 통해 **2 cm 이내 정밀 도킹**을 수행하는 시스템입니다.

Host PC 관제국은 라즈베리파이의 480p 카메라 영상 스트림을 실시간 수신하여 최신 **Ultralytics YOLO26n** 모델로 조난자(`person`) 및 위험 요소를 고속 추론하고, **10Hz Heartbeat Watchdog** 신호를 로봇으로 발신하여 통신 상태를 지속적으로 유지·관리합니다.

---

## 2. 전체 구현 및 연동 작업 내역

### 2.1 Host PC GPU 가속 및 YOLO26n 나노 모델 탑재
- **PyTorch 2.5.1 + CUDA 12.1 환경 구축**: NVIDIA GeForce GTX 1050 Ti (4GB VRAM, `cuda:0`) 완벽 연동 및 Half-precision(FP16) 추론 지원
- **Ultralytics YOLO26 엔진 설치**: `yolo26n.pt` 가중치 자동 다운로드 및 480p 영상 추론 검증 완료 (단일 프레임 추론 속도 < 10ms)
- **ROS 2 Humble 의존성 정합**: NumPy 2.x와 `cv_bridge` 간 충돌 문제를 해결하기 위해 `numpy==1.26.4` 및 `opencv-python==4.10.0.84`로 버전 고정

### 2.2 disaster_recovery_ws 워크스페이스 구축 및 호스트 노드 개편
- GitHub 저장소 (`https://github.com/rjw136-lab/disaster_recovery_ws`) 클론 및 `colcon` 빌드 완료
- `host_ai_pipeline.py` 고도화:
  - 기본 모델을 `yolo26n.pt`로 전환
  - 원시 이미지(`/camera/image_raw`)와 압축 이미지(`/camera/image_raw/compressed`) 동시 지원
  - 10Hz Heartbeat Watchdog(`/heartbeat`) 발신으로 로봇의 비상 모드 오작동 차단
  - 조난자 검출 시 `/disaster/victim_alert` 발행 및 시각적 알림 배너 렌더링

### 2.3 270도 영상 회전 및 수직 보정 (`rotation_angle:=270`)
- **원인**: 터틀봇3 버거 차체에 라즈베리파이 카메라가 케이블 배선상 90도 측면으로 장착되어 영상이 누워있던 문제
- **해결**: 영상 수신 즉시 270도 반시계 방향 회전(`cv2.ROTATE_90_COUNTERCLOCKWISE`) 적용
- **효과**: 사람이 똑바로 서 있는 상태로 YOLO26n에 입력되어 객체 검출 정확도가 정상치로 회복되었으며, RViz2 및 모니터링 창에서도 정방향으로 출력

### 2.4 Zero-Lag 최신 프레임 워커 스레드 (네트워크 큐 지연 해소)
- **원인**: Wi-Fi 대역폭 한계로 영상 프레임이 뭉쳐 들어올 때, 큐에 쌓인 과거 프레임을 순차 처리하느라 1~3초의 렉(Latency) 발생
- **해결**: ROS 콜백은 최신 프레임 1장만 메모리에 보관하고 즉시 반환하며, 백그라운드 워커 스레드가 항상 **가장 최신 프레임**만 가져와 YOLO 추론을 수행하도록 아키텍처 개편

### 2.5 RViz2 전용 관제 설정 파일 (`disaster_station.rviz`) 구축
- 로봇 3D 모델(RobotModel) + 라이다 센서(`/scan`) + SLAM 지도(`/map`) + YOLO 감지 영상(`/detection/image`)을 원클릭으로 열 수 있는 RViz 설정 파일 배포
- `rqt_image_view` GUI 캐시 파일 충돌(`parent=__NoneType__` 분리 창 먹통 버그) 해결 및 클린 리셋
- `setup.py`에 `config/*.rviz` 설치 항목 추가 → `host_station.launch.py use_rviz:=true`로 관제 노드와 RViz2를 함께 실행

### 2.6 조난자 오탐 억제 (Person 전용 + 연속 프레임 디바운스)
- `only_person_detection:=true`: YOLO 추론을 COCO class 0(`person`)으로 제한하여 연산량과 오탐을 동시에 감소
- `person_conf_threshold:=0.55`: 추론 임계값(0.40)과 별도로 조난자 알림 전용 임계값 적용
- `min_consecutive_detections:=2`: 연속 2프레임 이상 탐지될 때만 `/disaster/victim_alert` 발행 및 `SURVIVOR LOCATED` 배너 표시

### 2.7 USB 마이크 실시간 오디오 파이프라인 (스텁 → 실구현)
- **하드웨어**: Generic AB13X USB Audio (RPi 4 card 1, device 0 → `plughw:1,0`), S16_LE · Mono · 48,000 Hz
- **로봇 측 (`audio_capture_node`)**: `arecord -D plughw:1,0 -f S16_LE -r 48000 -c 1 -t raw` 서브프로세스 출력을 0.5초(48,000 B) 단위로 잘라 `/audio/raw` (`std_msgs/UInt8MultiArray`)로 발행. 프로세스 종료 시 1초 후 자동 재시도, 10패킷마다 RMS 진단 로그
- **호스트 측 (`host_ai_pipeline`)**: 2초 슬라이딩 윈도우(50% 중첩) → 48 kHz→16 kHz 3:1 decimation → float32 정규화 → RMS ≥ 0.012 일 때만 Whisper 큐 투입 → Faster-Whisper(`language="ko"`, VAD 필터) → 구조 요청 키워드 매칭 시 `/disaster/audio_alert`
- GTX 1050 Ti(Pascal)는 FP16 가속 이점이 없어 Whisper `compute_type="int8"`로 통일

### 2.8 ArUco 도킹의 카메라 장착 회전 보정
- 카메라가 90° 옆으로 장착되어 있으므로, 원본 영상 그대로 solvePnP를 수행하면 수평 오차(X)와 yaw 축이 뒤바뀌어 도킹 제어가 발산할 수 있음
- 해결: 캘리브레이션 내부 파라미터가 유효한 **원본 영상**에서 solvePnP를 수행한 뒤, `camera_rotation_deg`(기본 270, Host `rotation_angle`과 동일 규약)에 따른 회전행렬로 `tvec`·`R`을 정방향 카메라 좌표계로 변환
- `/aruco/debug_image`도 정방향으로 회전하여 발행

---

## 3. 네트워크 최적화 분석 (Tailscale vs 무암호화 사설 Wi-Fi)

| 비교 항목 | 현재 (Tailscale VPN) | 개선 (사설 Wi-Fi 동글 직결) |
| :--- | :--- | :--- |
| **통신 경로** | 공용망 + 방화벽 우회 (DERP 릴레이 경유) | 로컬 사설 공유기 1:1 직결 (LAN) |
| **지연 시간 (Ping)** | 50 ~ 150 ms (패킷 지터 심함) | **1 ~ 3 ms (초저지연, 안정적)** |
| **암호화 유무** | 강제 암호화 (RPi CPU 발열 및 연산 병목) | **무암호화 순수 UDP (CPU 점유율 급감)** |
| **워치독 안정성** | 패킷 유실 시 3초 타임아웃 오작동 빈번 | **10Hz Heartbeat 누락 없이 완벽 유지** |

---

## 4. 주요 ROS 2 인터페이스 명세서

| 토픽 이름 | 메시지 타입 | 발행처 | 설명 |
| :--- | :--- | :--- | :--- |
| `/heartbeat` | `std_msgs/msg/Header` | Host PC | 10Hz 안전 워치독 (3초 이상 끊기면 비상 복귀 시작) |
| `/camera/image_raw` | `sensor_msgs/msg/Image` | RPi 4 | 480p 원본 카메라 영상 스트림 (Raw) |
| `/camera/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | RPi 4 | 480p JPEG 압축 스트림 (네트워크 렉 해소용, 권장) |
| `/detection/image` | `sensor_msgs/msg/Image` | Host PC | YOLO26n 객체 바운딩 박스가 렌더링된 실시간 영상 |
| `/disaster/victim_alert` | `std_msgs/msg/String` | Host PC | 조난자(`person`) 발견 시 알림 메시지 |
| `/disaster/audio_alert` | `std_msgs/msg/String` | Host PC | 구조 요청 음성 키워드 검출 알림 |
| `/audio/raw` | `std_msgs/msg/UInt8MultiArray` | RPi 4 | AB13X USB 마이크 48 kHz S16_LE mono PCM (0.5 s/메시지) |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | RPi 4 | 터틀봇3 버거 모터 구동 명령 (회전/직진 제어) |

---

## 5. 실행 및 운영 명령어 치트시트

### ① Host PC 관제국 실행 (기본 270도 회전 + 10Hz Heartbeat)
```bash
ros2 launch disaster_recovery host_station.launch.py
```

### ② 렉을 최소화하는 압축 이미지 모드 실행 (적극 권장)
```bash
ros2 launch disaster_recovery host_station.launch.py image_topic:=/camera/image_raw/compressed use_compressed:=true
```

### ③ RViz2 종합 관제 화면 원클릭 실행 (라이다 + 맵 + YOLO 카메라)
```bash
ros2 launch disaster_recovery host_station.launch.py use_rviz:=true
# 또는 단독 실행
ros2 run rviz2 rviz2 -d ~/disaster_recovery_ws/src/disaster_recovery/config/disaster_station.rviz
```

### ④ 라즈베리파이 로봇 온보드 노드 일괄 실행 (로봇 터미널)
```bash
ros2 launch disaster_recovery robot_onboard.launch.py
```

> `robot_onboard.launch.py`는 `audio_capture_node`를 포함하며, 마이크 장치·샘플레이트·ArUco 회전값은 `config/recovery_params.yaml`에서 변경한다.

### ⑤ 마이크 / 오디오 경로 점검
```bash
# RPi 4: 장치 확인 및 3초 녹음 테스트
arecord -l
arecord -D plughw:1,0 -f S16_LE -r 48000 -c 1 -d 3 test.wav && aplay test.wav
# Host PC: 오디오 패킷 수신 및 알림 확인
ros2 topic hz /audio/raw            # 약 2 Hz
ros2 topic echo /disaster/audio_alert
```
