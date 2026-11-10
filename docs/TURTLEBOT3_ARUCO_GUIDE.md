# TurtleBot3 Burger & ArUco 마커 인식 시스템 통합 가이드

이 문서는 라즈베리파이 기반 TurtleBot3 Burger 로봇에서 카메라 노드 구동, 270도 화면 각도 보정, 고속 웹 스트리밍, ArUco 마커 인식 및 전체 시스템 Bringup까지 구성한 모든 내용을 총정리한 공식 가이드입니다.

---

## 1. 시스템 환경 개요

- **로봇 하드웨어**: TurtleBot3 Burger
  - 메인 제어기: OpenCR 1.0 (`/dev/ttyACM0`)
  - 라이다 센서: LDS-01 (`/dev/ttyUSB0`)
  - 모터: Dynamixel X-series (2축 차동 구동)
- **카메라 사양**: Raspberry Pi Camera Module V2 (Sony IMX219 센서, CSI 인터페이스)
  - 해상도: 480 × 640 (세로 모드 보정)
  - 각도 보정: 270도 회전 (카메라가 눕혀진 상태를 고려하여 정상 방향으로 보정)
- **운영체제 및 미들웨어**:
  - Ubuntu 22.04 LTS
  - ROS 2 Humble Hawksbill
  - RMW 구현체: CycloneDDS (`rmw_cyclonedds_cpp`)
- **네트워크 주소**:
  - **Tailscale VPN IP (PC 접속용)**: `100.91.141.24`
  - 로컬 Wi-Fi IP: `192.168.0.135`
  - 웹 스트리밍 포트: `8080`

---

## 2. 생성 및 구성된 주요 파일

| 파일 경로 | 구분 | 역할 및 설명 |
| :--- | :--- | :--- |
| [`/home/lab/start.py`](file:///home/lab/start.py) | 올인원 시작 | 하드웨어 점검 + Bringup + 카메라 + 오디오 + ArUco 3D 추정 원클릭 시작 |
| [`/home/lab/stop.py`](file:///home/lab/stop.py) | 올인원 종료 | 실행 중인 모든 로봇 및 센서 노드 일괄 안전 종료 |
| [`/home/lab/start_all.sh`](file:///home/lab/start_all.sh) | 통합 스크립트 | 로봇 Bringup + 카메라 구동 + ArUco 감지 + 웹 스트리밍을 한 번에 실행 |
| [`/home/lab/stop_all.sh`](file:///home/lab/stop_all.sh) | 통합 스크립트 | 실행 중인 모든 로봇 및 카메라 노드 일괄 종료 |
| [`/home/lab/start_aruco.sh`](file:///home/lab/start_aruco.sh) | 단독 스크립트 | 카메라 노드 구동 + ArUco 감지 + 웹 스트리밍 실행 |
| [`/home/lab/stop_aruco.sh`](file:///home/lab/stop_aruco.sh) | 단독 스크립트 | 카메라 및 ArUco 관련 노드만 종료 |
| [`/home/lab/aruco_detector.py`](file:///home/lab/aruco_detector.py) | 핵심 노드 | OpenCV 기반 ArUco 감지, 3D 거리/포즈 추정(Option B), 270도 회전, 고속 웹 스트리밍 |
| [`/home/lab/generate_marker.py`](file:///home/lab/generate_marker.py) | 유틸리티 | 테스트용 ArUco 마커 이미지(PNG) 생성 도구 |
| [`/home/lab/detected_result.jpg`](file:///home/lab/detected_result.jpg) | 결과 사진 | 마커가 감지된 순간 자동으로 저장되는 최종 3D 좌표축 포함 캡처 이미지 |

---

## 3. 원클릭 복붙 단축어 (Cheat Sheet)

`~/.bashrc`에 단축어가 영구 등록되어 있으므로 터미널 어디서든 단어 하나만 입력하면 됩니다:

### 1) 전체 시스템 (추천)
- **실행**:
  ```bash
  tb3-all
  ```
  *(로봇 모터/라이다 Bringup + 카메라 + ArUco 감지 + 웹 화면이 한 번에 켜집니다)*
- **종료**: 실행 중인 터미널에서 `Ctrl + C` 또는 새 창에서 `tb3-stop`

### 2) 카메라 및 ArUco 마커만 실행
- **실행**:
  ```bash
  aruco
  ```
- **종료**: `Ctrl + C` 또는 `aruco-stop`

### 3) 로봇 Bringup(모터/라이다)만 단독 실행
- **실행**:
  ```bash
  bringup
  ```
- **종료**: `bringup-stop`

---

## 4. 실시간 웹 화면 모니터링 & 3D 포즈 추정

1. 웹 브라우저(크롬 또는 엣지) 주소창에 다음 주소를 입력합니다:
   ```text
   http://100.91.141.24:8080
   ```
   *(로컬 Wi-Fi 환경일 경우 `http://192.168.0.135:8080`)*
2. 화면에 실시간 480×640 세로 영상이 표시됩니다.
3. 카메라 앞에 ArUco 마커(기본: `DICT_4X4_50`, ID 0번)를 비추면:
   - **웹 화면**: 마커 위에 **3D 직교 좌표축**(빨강: X 좌우, 초록: Y 상하, 파랑: Z 전방깊이)과 실제 계산된 거리(예: `Dist: 45.2cm`, `Z: 45.1 (X:+2.1, Y:-1.0)`)가 실시간 오버레이됩니다.
   - **터미널**: 마커 감지 즉시 `[3D 포즈 측정 완료]` 로그와 함께 cm 단위 3차원 상대 좌표가 딱 1회 출력됩니다.
   - **스냅샷 저장**: 3D 축과 거리 정보가 렌더링된 캡처 사진이 [`/home/lab/detected_result.jpg`](file:///home/lab/detected_result.jpg) 파일로 자동 저장됩니다.

### 마커 실제 크기 설정 방법
기본값은 한 변의 길이가 **5.0 cm** (`0.05 m`)로 설정되어 있습니다. 소지하신 마커의 실제 크기에 맞춰 아래와 같이 지정할 수 있습니다:
```bash
# cm 단위로 지정 (예: 7.0 cm)
start.py --size-cm 7.0

# 또는 m 단위로 지정 (예: 10 cm = 0.1 m)
start.py --marker-size 0.1
```

---

## 5. 트러블슈팅 및 최적화 기록

1. **Option B 카메라 캘리브레이션 모델 적용**:
   - Sony IMX219 (라즈베리파이 카메라 V2, 640x480) 센서의 광학 사양(초점거리 3.04mm, 정방형 픽셀, 대각 화각 및 HFOV 62.2°)을 기반으로 이론적 카메라 매트릭스($f_x \approx f_y \approx 530\text{px}$, 광학중심 $c_x=240, c_y=320$)를 구현하여 체커보드 촬영 없이도 즉시 3D 위치 및 거리를 계산할 수 있도록 구축
   - 추후 정밀 캘리브레이션 YAML 파일 생성 시 `--calib` 옵션으로 언제든 교체 가능하도록 설계
2. **카메라 회전 (가로 눕힘 문제)**:
   - 물리적으로 카메라가 눕혀져 있어 초기 화면이 거꾸로 출력됨
   - `aruco_detector.py` 내부에서 OpenCV 270도 회전(`cv2.ROTATE_90_COUNTERCLOCKWISE`)을 적용하여 똑바로 선 480×640 세로 화면으로 보정
3. **반복 출력 및 중복 사진 저장 문제**:
   - 마커가 시야에 머무를 때 로그와 파일이 계속 덮어써지던 문제를 플래그 처리하여 **마커당 최초 1회만 출력 및 1장 저장**되도록 개선
4. **네트워크 접속 IP (`localhost` 문제)**:
   - 윈도우 PC와 로봇이 Tailscale VPN으로 연결되어 있어 `localhost`나 `192.168...` 대신 Tailscale IP(`100.91.141.24`)로 접속하도록 정리
5. **스트리밍 끊김 및 5 FPS 저하 문제**:
   - HTTP 1.0 순차 폴링 방식에서 발생하는 왕복 지연시간(RTT) 문제를 해결하기 위해 **고속 MJPEG 다이렉트 푸시 + `TCP_NODELAY` + 소켓 버퍼 즉시 플러시(`flush`)**로 전면 개편
   - 백그라운드에 남아있던 카메라 고아 프로세스를 완전히 정리하여 하드웨어 병목 해소
