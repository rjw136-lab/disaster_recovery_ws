# [공학 가이드 2] MQ-2 + ADS1115 결선 및 단 1회 파이 카메라 캘리브레이션 가이드

---

## 1. MQ-2 가스 센서 → ADS1115 → Raspberry Pi 4 결선

### 1.0 왜 OpenCR A0 대신 ADS1115인가
- TurtleBot3의 모터 구동과 `/odom`은 OpenCR에 올라간 **TurtleBot3 기본 펌웨어**가 담당한다. 별도 아두이노 스케치를 업로드하면 이 펌웨어가 덮어써져 모터가 멈춘다.
- OpenCR USB(`/dev/ttyACM0`)는 `turtlebot3_node`가 이미 점유하므로 별도 시리얼 프로토콜을 섞을 수 없다.
- 따라서 OpenCR은 **순정 상태로 두고**, MQ-2 아날로그 출력은 16-bit I2C ADC인 **ADS1115**를 통해 RPi 4가 직접 읽는다. 처리 노드: `gas_sensor_node.py`

### 1.1 전기적 제약
- **MQ-2**: 히터 구동 5.0 V 필요. 아날로그 출력(AOUT)은 최대 5.0 V까지 상승 가능.
- **ADS1115** (TI, SBAS444 데이터시트): 아날로그 입력 허용 범위는 GND−0.3 V ~ VDD+0.3 V. 측정 가능 범위는 PGA FSR와 무관하게 VDD를 넘을 수 없다.
- **RPi 4 GPIO/I2C**: 3.3 V 로직. 5 V 풀업이 걸리면 GPIO가 손상될 수 있으므로 **ADS1115는 반드시 3.3 V(RPi 핀 1)로 급전**한다. 시판 ADS1115 브레이크아웃의 I2C 풀업은 보드 VDD에 연결되어 있으므로 VDD=3.3 V이면 안전하다.

### 1.2 10 kΩ : 20 kΩ 분압
AOUT에 R1 = 10 kΩ 직렬, AIN0 노드에서 GND로 R2 = 20 kΩ:

$$V_{AIN0} = V_{out} \times \frac{R_2}{R_1 + R_2} = 5.0\,\text{V} \times \frac{2}{3} \approx 3.33\,\text{V}$$

- 3.33 V ≤ VDD + 0.3 V = 3.6 V 이므로 허용 범위 안이다. 다만 VDD(3.3 V)를 약간 넘는 구간(≈3.30~3.33 V, 센서 출력 4.95 V 이상)은 측정이 포화될 수 있으며, 이는 극고농도 영역이라 판정에 영향이 거의 없다.
- PGA는 ±4.096 V(LSB = 125 µV)로 설정한다. ±2.048 V로 두면 2.05 V 이상에서 포화된다.
- ADS1115 입력 임피던스(FSR ±4.096 V 기준 수 MΩ)는 20 kΩ보다 훨씬 커서 분압비에 영향을 주지 않는다.
- 노드에서 역변환: `V_out = V_AIN0 × (R1 + R2) / R2 = V_AIN0 × 1.5`

### 1.3 결선도

```
  RPi 4 GPIO 헤더                ADS1115 모듈                 MQ-2 모듈
  ───────────────               ─────────────               ──────────
  Pin 1  (3.3V) ─────────────── VDD
  Pin 3  (SDA1) ─────────────── SDA
  Pin 5  (SCL1) ─────────────── SCL
  Pin 6  (GND)  ──────┬──────── GND
                      │         ADDR ── GND  (I2C 주소 0x48)
                      │
  Pin 2  (5V)   ──────┼─────────────────────────────────────── VCC
                      ├─────────────────────────────────────── GND
                      │                                        DOUT (미사용)
                      │                                        AOUT
                      │                                          │
                      │                                     [R1 10 kΩ]
                      │                                          │
                      │         AIN0 ────────────────────────────┤
                      │                                     [R2 20 kΩ]
                      │                                          │
                      └──────────────────────────────────────────┘
```

- MQ-2 히터 전류는 약 150 mA 수준(데이터시트 히터 소비전력 ≤ 800 mW)이다. RPi 5 V 핀(핀 2/4)은 전원 레일에 직결되어 있어 공급 가능하다. 원래 계획대로 OpenCR 5 V 헤더에서 받아도 되지만, 이 경우 **OpenCR GND와 RPi GND가 공통**인지 반드시 확인한다 (TurtleBot3 기본 배선에서는 공통).

### 1.4 핀 매핑 표

| 신호 | 출발 | 도착 | 비고 |
| :--- | :--- | :--- | :--- |
| 3.3 V | RPi 핀 1 | ADS1115 VDD | **5 V 금지** |
| SDA | RPi 핀 3 (GPIO2) | ADS1115 SDA | I2C bus 1 |
| SCL | RPi 핀 5 (GPIO3) | ADS1115 SCL | |
| GND | RPi 핀 6 | ADS1115 GND, ADDR, MQ-2 GND, R2 | 공통 접지 |
| 5 V | RPi 핀 2 | MQ-2 VCC | 히터 전원 |
| AOUT | MQ-2 AOUT | R1(10 kΩ) → AIN0 | |
| 분압 | AIN0 노드 | R2(20 kΩ) → GND | |

### 1.5 RPi 4 소프트웨어 설정

```bash
# 1) I2C 활성화 확인 (Ubuntu 22.04 for RPi)
grep i2c_arm /boot/firmware/config.txt      # dtparam=i2c_arm=on 이 없으면 추가 후 재부팅

# 2) 도구 및 파이썬 라이브러리
sudo apt install -y i2c-tools python3-smbus
sudo usermod -aG i2c $USER                   # 재로그인 필요

# 3) 장치 확인: 0x48 위치에 '48'이 보여야 한다
i2cdetect -y 1

# 4) 단독 실행 테스트
ros2 run disaster_recovery gas_sensor_node
ros2 topic echo /gas/voltage                 # 즉시 출력
ros2 topic echo /gas/level                   # warmup(60s) + baseline(5s) 이후 출력
```

### 1.6 가스 지표 정의 (`/gas/level`)
- 노드 시작 후 `warmup_sec`(기본 60 s) 동안 히터를 예열하고, 이어서 깨끗한 공기에서 `baseline_samples`(기본 100개, 5 s)의 평균 Rs를 **Rs_air**로 저장한다.
- `gas_level = clamp(1 − Rs / Rs_air, 0, 1)` → 0.0은 기준 공기 수준, 0.4는 센서 저항이 40 % 떨어진 상태다.
- Rs = (VCC − V_out) · RL / V_out 이므로 **RL 값은 비율에서 상쇄**된다. 모듈의 RL(1 kΩ/10 kΩ 등)을 몰라도 지표는 정확하다.
- 이 값은 **ppm이 아니다.** 임계값 0.40은 출발점일 뿐이며, 현장에서 라이터 가스로 실측해 조정해야 한다. 데이터시트 곡선으로 ppm을 추정하려면 `/gas/rs_ro`(Rs/Ro, Ro = Rs_air / 9.83)를 사용한다.
- 제조사 데이터시트는 정밀 측정을 위해 24시간 이상의 초기 예열(burn-in)을 권장한다. 60 s 예열은 상대 변화 감지용 최소치다.
- 매번 기준 측정을 생략하려면 로그에 출력된 값을 `rs_air_kohm`에 넣는다.

---

## 2. 체커보드를 이용한 단 1회 파이 카메라 캘리브레이션

오도메트리 누적 슬립을 보정하고 반경 15cm 이내 정밀 복귀를 달성하기 위해서는 단일 카메라 기반 Perspective-n-Point(solvePnP) 알고리즘이 필수적이다. solvePnP의 정확도는 카메라 내부 파라미터($f_x, f_y, c_x, c_y$) 및 렌즈 왜곡 계수($k_1, k_2, p_1, p_2, k_3$)의 정밀도에 전적으로 의존한다 (Zhang, 2000, "A Flexible New Technique for Camera Calibration").

### 2.1 캘리브레이션 준비물
1. **체커보드 타겟**:
   - 8x6 내부 코너 (가로 9칸, 세로 7칸 격자 패턴)
   - 사각형 1칸의 한 변 크기: 정확히 **25mm** ($0.025\,\text{m}$)
   - 평평한 아크릴판 또는 하드보드지에 구김 없이 완벽히 수평 부착.
2. **Raspberry Pi Camera Module V2**:
   - 해상도: 640x480 (30 FPS)
   - 초점 링 고정 (캘리브레이션 이후 렌즈 초점 변경 금지).

### 2.2 ROS 2 카메라 드라이버 실행 (RPi 4)
RPi 4 터미널에서 v4l2_camera 노드를 구동하여 이미지를 발행한다:
```bash
ros2 run v4l2_camera v4l2_camera_node --ros-args \
  -p video_device:="/dev/video0" \
  -p image_size:="[640,480]" \
  -p pixel_format:="YUYV" \
  -r image_raw:=/camera/image_raw \
  -r camera_info:=/camera/camera_info \
  -r set_camera_info:=/camera/set_camera_info
```
> `v4l2_camera`는 기본적으로 `/image_raw`, `/camera_info`에 발행한다. 본 패키지의 노드는 `/camera/...`를 구독하므로 리매핑이 필요하다.

### 2.3 캘리브레이션 GUI 실행 (Host PC WSL2)
Host PC 터미널에서 ROS 2 공식 캘리브레이터 노드를 구동한다:
```bash
# ROS 2 카메라 캘리브레이션 툴 설치 (미설치 시)
sudo apt install -y ros-humble-camera-calibration

# 캘리브레이션 노드 실행
ros2 run camera_calibration cameracalibrator \
  --size 8x6 \
  --square 0.025 \
  --ros-args -r image:=/camera/image_raw -r camera:=/camera
```

### 2.4 캘리브레이션 측정 절차
1. GUI 창이 열리면 체커보드를 카메라 시야에 위치시킨다.
2. 우측 상단의 4개 프로그레스 바(X, Y, Size, Skew)가 모두 녹색으로 채워질 때까지 체커보드를 부드럽게 움직인다:
   - **X**: 화면의 좌측, 중앙, 우측 영역 골고루 노출
   - **Y**: 화면의 상단, 중앙, 하단 영역 골고루 노출
   - **Size**: 카메라에 가까이 다가갔다가(화면 전체 채움) 멀리 떨어짐
   - **Skew**: 체커보드를 상하/좌우로 15~30도 기울여 원근 왜곡 유도
3. 모든 막대가 녹색으로 전환되면 **[CALIBRATE]** 버튼이 활성화된다. 버튼을 클릭하고 연산이 완료될 때까지 약 10~30초간 대기한다.
4. 연산 완료 후 **[COMMIT]** 버튼을 클릭하여 파라미터를 저장한다.

### 2.5 캘리브레이션 결과 파일 확인 및 배포
캘리브레이션 결과는 `/tmp/calibrationdata.tar.gz`에 저장된다:
```bash
cd /tmp
tar -xvf calibrationdata.tar.gz
# 추출된 ost.yaml 확인
cat ost.yaml
```

이 `ost.yaml`의 수치를 추출하여 패키지 내 `camera_info.yaml`로 영구 보존한다:
```yaml
image_width: 640
image_height: 480
camera_name: rpi_camera_v2
camera_matrix:
  rows: 3
  cols: 3
  data: [530.1254, 0.0, 318.4521,
         0.0, 529.8971, 239.1124,
         0.0, 0.0, 1.0]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [0.12451, -0.21458, 0.00124, -0.00085, 0.00000]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0]
projection_matrix:
  rows: 3
  cols: 4
  data: [530.1254, 0.0, 318.4521, 0.0,
         0.0, 529.8971, 239.1124, 0.0,
         0.0, 0.0, 1.0, 0.0]
```

이제 `aruco_docking.py`는 기동 시 위 내부 행렬($K$)과 왜곡 계수($D$)를 주입받아 왜곡 없는 PnP 포즈 추정을 수행한다.
