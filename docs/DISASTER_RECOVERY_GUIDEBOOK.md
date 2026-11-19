# [종합 가이드북] 통신 단절 적응형 실내 재난 탐사 및 자율 복귀 터틀봇3 버거 시스템 구축 엔지니어링 리포트

> **최신 업데이트 (2026-11-20):** Phase 1(`RollingLocalGrid + 2D A*`), Phase 2-A(`Clearance-Gated RL`), **Phase 2-B(`v3_dynfix` · 15대 재난 시나리오 150회 `99.33%`)** 및 **Phase 3(논문 기반 2D 시맨틱 SLAM & 3채널 Context-RL, 150회 비퇴행 `96.00%`)** 구현·시뮬레이션 검증 결과를 통합 반영하였다.

---

<!-- TOC:BEGIN -->
## 📑 목차

- [1. 프로젝트 개요 및 핵심 설계 철학](#1-프로젝트-개요-및-핵심-설계-철학)
- [2. 학술적 배경 및 이론적 근거](#2-학술적-배경-및-이론적-근거)
  - [2.1 차동 구동 모바일 로봇의 사체 항법 누적 오차 및 백트래킹](#21-차동-구동-모바일-로봇의-사체-항법-누적-오차-및-백트래킹)
  - [2.2 비홀로노믹 구속 로봇의 제자리 회전 후 직진 (Rotate-then-Translate) 분리 제어](#22-비홀로노믹-구속-로봇의-제자리-회전-후-직진-rotate-then-translate-분리-제어)
  - [2.3 포섭 구조(Subsumption Architecture) 기반의 무정지 온보드 생존성](#23-포섭-구조subsumption-architecture-기반의-무정지-온보드-생존성)
  - [2.4 ArUco 마커 기반 원근 n-포인트(Perspective-n-Point) 정밀 도킹](#24-aruco-마커-기반-원근-n-포인트perspective-n-point-정밀-도킹)
  - [2.5 점유 격자 기반 국소 A* 탐색 및 Context-Adaptive RL 융합 (Phase 1 ~ Phase 2-B)](#25-점유-격자-기반-국소-a-탐색-및-context-adaptive-rl-융합-phase-1-phase-2-b)
- [3. 시스템 아키텍처 및 데이터 흐름](#3-시스템-아키텍처-및-데이터-흐름)
  - [3.1 하드웨어 및 네트워크 구성 다이어그램](#31-하드웨어-및-네트워크-구성-다이어그램)
- [4. 온보드 유한 상태 머신 (FSM) 전이도](#4-온보드-유한-상태-머신-fsm-전이도)
  - [각 상태(State)별 온보드 세부 실행 동작](#각-상태state별-온보드-세부-실행-동작)
- [5. 생성된 소스 코드 패키지 구조 및 파일 링크](#5-생성된-소스-코드-패키지-구조-및-파일-링크)
- [6. 결선도 및 파라미터 수식 검증 요약](#6-결선도-및-파라미터-수식-검증-요약)
  - [6.1 MQ-2 분압기 전압 무결성 검증 (ADS1115 경유)](#61-mq-2-분압기-전압-무결성-검증-ads1115-경유)
  - [6.2 오도메트리 이산화 스택 기준](#62-오도메트리-이산화-스택-기준)
  - [6.3 Rotate-then-Translate 제어기 수식](#63-rotate-then-translate-제어기-수식)
  - [6.4 ArUco solvePnP 정밀 도킹 수식](#64-aruco-solvepnp-정밀-도킹-수식)
- [7. Phase 3 구현·검증 현황 및 잔여 과제](#7-phase-3-구현검증-현황-및-잔여-과제)
- [8. 결론 및 종합 평가](#8-결론-및-종합-평가)

<!-- TOC:END -->

---

## 1. 프로젝트 개요 및 핵심 설계 철학

본 프로젝트는 2륜 차동 구동 로봇(TurtleBot3 Burger)을 활용하여 실내 재난 환경을 원격 탐색하다가, 통신 두절(Heartbeat 3.0초 타임아웃) 또는 위험 가스 누출(MQ-2 임계치 초과) 발생 시 외부 관제 PC의 개입 없이 로봇 자체 연산만으로 지나온 경로를 역추종(Backtracking)하여 시작 위치 반경 15cm 이내로 안전 복귀하는 시스템을 구현하는 것을 목표로 한다.

Host PC(Windows 11 WSL2)와 로봇 본체(Raspberry Pi 4 + OpenCR 1.0) 간의 역할을 엄격히 분리하여, 네트워크가 완전 단절되더라도 로봇의 생존 및 복귀 제어 루프(**Phase 2-B 4계층 하이브리드 자율복귀: Layer-0 연기/동적잔상 필터 · `RollingLocalGrid 2D A*` · `ContextRLStudent` ONNX · 2D SVD ICP · ArUco 정밀 도킹**)가 온보드 임베디드 컴퓨터에서 독립적으로 자율 작동하도록 구축한다.

---

## 2. 학술적 배경 및 이론적 근거

### 2.1 차동 구동 모바일 로봇의 사체 항법 누적 오차 및 백트래킹
- **주장**: 휠 엔코더 오도메트리 기반 주행 시 장시간 탐색 경로를 그대로 추종하면 누적 오차가 발산하므로, 이동 거리 $`\Delta d \ge 0.3\,\text{m}`$ 또는 회전각 $`\lvert \Delta \theta \rvert \ge 25^\circ`$ 기준의 이산화 경유지(Waypoint)를 LIFO 스택에 저장하고 역추종하는 기법이 필수적이다.
- **근거**: Borenstein & Feng (1996), "Measurement and Correction of Systematic Odometry Errors in Mobile Robots", *IEEE Transactions on Robotics and Automation*, 12(6), 869-881에 따르면, 차동 구동 모바일 로봇의 엔코더 사체 항법(Dead Reckoning)은 휠 슬립(Wheel slip), 지면 불균일성, 휠 반경의 미세 불일치로 인해 주행 거리가 증가함에 따라 방위각 오차($`\Delta \theta`$)가 적분되어 횡방향 위치 오차가 거리의 제곱에 비례하여 기하급수적으로 발산한다.
- **결론**: 따라서 무제한 오도메트리 추종을 지양하고, 탐색 단계에서 유의미한 공간 기하 변화 시점마다 노드를 추출하여 이산화된 LIFO 스택 구조로 관리함으로써 역추종 시 궤적 왜곡을 최소화한다.

### 2.2 비홀로노믹 구속 로봇의 제자리 회전 후 직진 (Rotate-then-Translate) 분리 제어
- **주장**: 2륜 차동 구동 로봇의 역추종 주행 제어 방식으로 제자리 회전(Rotate)과 직진(Translate)을 엄격히 분리한 방식 A를 채택한다.
- **근거**: Siegwart, Nourbakhsh, & Scaramuzza (2011), *Introduction to Autonomous Mobile Robots*, MIT Press에 따르면, 2륜 차동 로봇은 측면 이동이 불가능한 비홀로노믹(Non-holonomic) 시스템이다. 곡선 궤적을 연속적으로 생성하는 추종 제어는 슬립 환경에서 곡률 오차가 가중되는 반면, 목표점을 향해 제자리 회전으로 헤딩각 오차($`e_\theta`$)를 0으로 정렬한 뒤 순수 직진을 수행하는 2단계 분리 제어는 기구학적 불확실성을 배제하고 제어 안정성을 극대화한다.
- **결론**: 본 시스템에서는 헤딩 오차 $`\lvert \Delta \theta \rvert \gt 0.08\,\text{rad}`$ ($`4.6^\circ`$) 구간에서는 선속도를 0으로 제한하고 순수 각속도 $`P`$-제어를 수행한 후, 정렬 완료 시에만 선속도를 인가하는 Rotate-then-Translate 알고리즘을 구현한다.

### 2.3 포섭 구조(Subsumption Architecture) 기반의 무정지 온보드 생존성
- **주장**: 외부 통신(Host PC) 및 고수준 AI 연산과 로봇의 최하위 생존/복귀 루프는 완전히 물리적으로 격리되어야 한다.
- **근거**: Brooks (1986), "A Robust Layered Control System for a Mobile Robot", *IEEE Journal of Robotics and Automation*, 2(1), 14-23은 하위 행동 계층(충돌 회피 등)이 상위 계층과 독립적으로 동작하도록 설계하면 상위 계층이 실패해도 하위 계층의 기본 동작이 유지된다는 계층형(포섭) 제어 구조를 제시하였다. 본 설계는 이 원칙을 "네트워크 의존 계층 / 온보드 생존 계층" 분리에 적용한 것이다. (생존율을 정량적으로 보장한다는 근거는 아니며, 실제 신뢰성은 필드 테스트로 검증해야 한다.)
- **결론**: 10Hz 하트비트 감시, MQ-2 가스 전압 모니터링, LDS-01 전방 장애물 제동, LIFO 백트래킹 모듈은 모두 RPi 4 온보드 ROS 2 노드에 전속되어 네트워크 단절 시 외부 도움 없이 즉각 작동한다.

### 2.4 ArUco 마커 기반 원근 n-포인트(Perspective-n-Point) 정밀 도킹
- **주장**: 사체 항법의 누적 슬립 오차를 상쇄하고 최종 복귀 오차를 반경 15cm 이내로 수렴시키기 위해 ArUco 시각 마커 기반 PnP 정밀 정렬을 적용한다.
- **근거**: Garrido-Jurado et al. (2014), "Automatic generation and detection of highly reliable fiducial markers under occlusion", *Pattern Recognition*, 47(6), 2280-2292 및 Collins & Bartoli (2014), "Infinitesimal Plane-Based Pose Estimation", *International Journal of Computer Vision*에 따르면, 기하학적 형상이 기지인 평면 4변 마커의 코너점을 단일 카메라로 투영 해석하는 `solvePnP` (IPPE Square)는 서브센티미터(Sub-centimeter) 단위의 3차원 상대 위치 $`(X_c, Y_c, Z_c)`$ 추정이 가능하다.
- **결론**: 로봇이 출발 원점 50cm 거리에 진입하면 베이스캠프 벽면의 $`10 \times 10\,\text{cm}`$ 마커(ID 0)를 인식하여 횡방향 오차 $`X_c \approx 0`$, 정지 거리 $`Z_c = 0.20\,\text{m}`$를 달성함으로써 누적 엔코더 오차를 완전히 무효화하고 15cm 이내 정밀 복귀를 완성한다.

### 2.5 점유 격자 기반 국소 A\* 탐색 및 Context-Adaptive RL 융합 (Phase 1 ~ Phase 2-B)
- **주장**: 단순 반사형(Reactive) 회피만으로는 U자 막다른 함정·고밀도 낙석·동적 교차 장애물을 빠져나갈 수 없으므로, 단기 메트릭 점유 격자(`RollingLocalGrid` + `2D A*`)로 대우회·위상 지름길 경로를 산출하고, 협소 구간에서는 `A*`가 주도하며 열린 구간에서는 시공간 문맥 기반 경량 ONNX 정책(`ContextRLStudentPolicy`)이 부드러운 고속 회피를 담당하는 계층형 융합이 필수적이다.
- **근거**: Elfes (1989)의 점유 격자(Occupancy Grid) 및 Hart et al. (1968)의 `A*` 탐색을 바탕으로 `8 m × 8 m`(`0.04 m` 해상도) 롤링 격자에서 국소 경로를 계획하고, Schulman et al. (2017)의 PPO·Jiang et al. (2021)의 PLR 커리큘럼·Ross et al. (2011)의 DAgger 증류를 통해 `16스텝(1.6초)` 시계열 문맥과 `12×12` 자차 중심 코스트맵(`688차원`)을 입력받는 `< 0.04 ms` 초경량 ONNX 학생 정책을 구축하였다.
- **결론**: `Phase 2-B`에서는 여기에 Layer-0 연기/분진 스파이크 필터(`_despike_lidar_ranges`), Open-Ray Carving(`carve_decay=0.25`), `DYNAMIC_HOLD`, 비대칭 측면 게이팅을 결합해 15개 재난 시나리오 150회 벤치마크에서 **`149/150 (99.33%)`** 을 달성하였다.

---

## 3. 시스템 아키텍처 및 데이터 흐름

### 3.1 하드웨어 및 네트워크 구성 다이어그램

```mermaid
flowchart TD
    subgraph Host_PC ["Host PC (Windows 11 + WSL2)"]
        NIC1["유선 LAN<br/>(학교 고정 IP / 보안 인증)"]
        NIC2["USB 무선 Wi-Fi 동글<br/>(192.168.0.100)"]
        WSL["WSL2 Ubuntu 22.04 LTS<br/>(networkingMode=mirrored)"]
        GPU["NVIDIA GTX 1050 Ti<br/>(CUDA 12.x)"]
        AI1["YOLO26n Vision AI<br/>(Person 탐지, 15 FPS 캡, 270도 보정)"]
        AI2["Faster-Whisper STT<br/>(구조 요청 발화 분석)"]
        HB["Heartbeat Generator<br/>(10Hz /heartbeat)"]
        SLAM["Cartographer / slam_toolbox<br/>및 RViz2 관제"]
        SEM["semantic_map_node (Phase 3)<br/>(단안 RGB + /scan 퓨전 2D 마커 맵)"]
    end

    subgraph Network ["로컬 사설 Wi-Fi (192.168.0.x, DOMAIN_ID=0)"]
        UDP["초저지연 UDP 멀티캐스트<br/>(지연 시간 1~2ms)"]
    end

    subgraph Robot_RPi ["TurtleBot3 Burger - RPi 4 (Ubuntu 22.04)"]
        WIFI_RPI["RPi 4 내장 Wi-Fi<br/>(192.168.0.105)"]
        LDS["LDS-01 2D LiDAR<br/>(/scan 발행 + Layer-0 스파이크 필터)"]
        CAM["RPi Camera V2<br/>(camera_ros / v4l2 스트리밍)"]
        MIC["AB13X USB 마이크<br/>(48kHz -> /audio/raw)"]
        WLOG["Odometry Logger<br/>(LIFO 스택: d>=0.3m, th>=25deg)"]
        WDOG["Watchdog Manager<br/>(Heartbeat 3.0s 및 Gas 감시)"]
        BTRK["Phase 2-B 복귀 제어기<br/>(RollingGrid A* + ContextRL ONNX + 2D ICP)"]
        ARUCO["ArUco PnP Docking<br/>(15cm 이내 정밀 안착)"]
        BBOX["Blackbox Recorder<br/>(통신 단절 시 압축 기록)"]
        GAS["gas_sensor_node<br/>(ADS1115 I2C -> /gas/level)"]
    end

    subgraph Sensor_Board ["MQ-2 센서 회로 (RPi I2C bus 1)"]
        MQ2["MQ-2 가스 센서<br/>(5V, AOUT)"]
        DIV["10k : 20k 분압기<br/>(max 3.33V)"]
        ADS["ADS1115 16-bit ADC<br/>(VDD 3.3V, 0x48)"]
    end

    subgraph Robot_OpenCR ["OpenCR 1.0 (순정 펌웨어 유지)"]
        DYNAMIXEL["XL430-W250<br/>다이내믹셀 모터 제어"]
        ENC["휠 엔코더 오도메트리<br/>(/odom via turtlebot3_node)"]
    end

    NIC2 <--> UDP
    UDP <--> WIFI_RPI
    WSL --- GPU
    GPU --- AI1
    GPU --- AI2
    WSL --- HB
    WSL --- SLAM
    AI1 -.-> SEM
    SLAM -.-> SEM

    WIFI_RPI --- WLOG
    WIFI_RPI --- WDOG
    LDS --> WDOG
    LDS --> BTRK
    CAM --> ARUCO
    CAM --> BBOX
    MIC --> AI2
    WDOG --> BTRK
    BTRK --> ARUCO
    WDOG --> BBOX

    MQ2 --> DIV --> ADS --> GAS --> WDOG
    ENC -- "USB /dev/ttyACM0" --> WLOG
    BTRK --> DYNAMIXEL
    ARUCO --> DYNAMIXEL
```

---

## 4. 온보드 유한 상태 머신 (FSM) 전이도

로봇의 자율 생존 및 복귀는 온보드 `recovery_controller` 노드에 내장된 유한 상태 머신(Finite State Machine)을 통해 엄격하게 제어된다.

```mermaid
flowchart TD
    START((●)) -->|"시스템 부팅 및 초기화"| INIT["INIT<br/>(센서 및 첫 Heartbeat 대기)"]
    INIT -->|"센서 및 첫 Heartbeat 수신"| NORMAL["NORMAL_TELEOP<br/>(/cmd_vel_remote 중계 · 0.3m/25도 스택 푸시)"]
    NORMAL -->|"Heartbeat 타임아웃(3.0s) 또는 가스 초과(0.40)"| ESTOP["EMERGENCY_STOP<br/>(즉시 정지 · 원격 차단 · 블랙박스 · 스택 팝)"]
    ESTOP -->|"목표 경유지 헤딩 계산 완료"| ROT["BACKTRACKING_ROTATE<br/>(선속도 0 · 제자리 각속도 P-제어)"]
    ROT -->|"헤딩 오차 허용 범위 정렬"| HYB["BACKTRACKING_HYBRID<br/>(Regulated Pure Pursuit + 2D A* + Phase 2-B ContextRL ONNX)"]
    HYB -->|"큰 헤딩 발산 또는 다음 경유지 팝"| ROT
    HYB -->|"동적 장애물 급접근"| DHOLD["DYNAMIC_HOLD<br/>(1.2s 일시 정지 대기)"]
    DHOLD -->|"장애물 통과 또는 0.20m 근접"| HYB
    HYB -->|"정체 감시(Watchdog) 발동"| ESC["ESCAPE<br/>(안전 후진·회전 탈출 + 4.5s RL 쿨다운)"]
    ESC --> HYB
    ROT -->|"LDS-01 전방 20도 내 0.12~0.25m (비활성 시)"| HOLD["OBSTACLE_HOLD<br/>(일시 정지 및 장애물 대기)"]
    HOLD -->|"장애물 제거 및 경로 확보"| ROT
    HYB -->|"베이스캠프 0.50m 접근 (use_aruco=true)"| HAND["HANDOVER_TO_ARUCO<br/>(/aruco/dock_trigger 발행)"]
    HAND -->|"시각 서보 제어권 이양"| DOCK["ARUCO_DOCKING<br/>(solvePnP 3D 추정 · X/Z 비례 제어)"]
    DOCK -->|"거리 2cm 및 각도 3도 10회 연속 달성"| DONE["MISSION_COMPLETE<br/>(반경 15cm 이내 복귀 완료 · 모터 고정)"]
    DONE --> END(((●)))
```

### 각 상태(State)별 온보드 세부 실행 동작

| 상태 (State) | 세부 실행 동작 (Action Detail) | 비고 |
| :--- | :--- | :--- |
| **`INIT`** | 시스템 부팅, 하드웨어 점검, `/odom` 및 최초 `/heartbeat` 대기 | 초기화 단계 |
| **`NORMAL_TELEOP`** | `/cmd_vel_remote` 원격 조종 통과 중계<br/>이동 $`\Delta d \ge 0.3\,\text{m}`$ 또는 회전 $`\lvert \Delta \theta \rvert \ge 25^\circ`$마다 `Waypoint` 스택 푸시 및 선행 격자 관측 | 정상 탐색 및 궤적 로깅 |
| **`EMERGENCY_STOP`** | 선속도 0, 각속도 0 즉시 인가 (`HALT_MOTORS`)<br/>원격 명령 중계 차단 및 로컬 블랙박스 녹화 개시 (`LOCK_REMOTE`)<br/>LIFO 스택 최상단 경유지 추출 (`POP_WP`) | 비상 페일세이프 작동 |
| **`BACKTRACKING_ROTATE`** | 선속도 0 유지, 목표 경유지 헤딩 방향으로 제자리 회전 각속도 $`P`$-제어 인가 (`PURE_ROTATION`) | 제자리 회전 정렬 |
| **`BACKTRACKING_HYBRID`** | **Phase 2-B 기본 복귀 상태**: `RollingLocalGrid(0.04m) + 2D A*`·위상 지름길 추종, `Clearance-Gate` 조건에 따라 `ContextRLStudent` ONNX(`688차원`) 블렌딩 및 곡률 보존 클램프, 경유지 45 cm 이내 2D SVD ICP 슬립 보정 수행 | 계층형 하이브리드 복귀 |
| **`DYNAMIC_HOLD`** *(Phase 2-B)* | 직진 중 동적 장애물 급접근($`df/dt < -0.25\,\text{m/s}`$, $`0.20 \le d_{\text{front}} \le 0.55\,\text{m}`$) 감지 시 구석 후진 몰림을 피하기 위해 1.2초 정지 대기 (`observe()` 연속 갱신) | 교차 이동 장애물 대응 |
| **`ESCAPE`** *(Phase 1~2-B)* | 정체 감시(Watchdog) 발동 시 안전 후진·회전 탈출 및 4.5s RL 쿨다운(`repeat >= 1` 시 7.0s Lockout Anchor 부여) | 막다른 골목·저상 턱 탈출 |
| **`BACKTRACKING_TRANSLATE`** | `enable_hybrid_tracking=false` 레거시 직진 추종 모드 (`FORWARD_DRIVE`) | 레거시 호환 모드 |
| **`LOCAL_TD3_AVOIDANCE`** | Phase 1.5 레거시(`recovery_strategy=td3`) 복귀 시 28차원 Residual TD3 ONNX 우회 기동 수행 | 레거시 호환 모드 |
| **`OBSTACLE_HOLD`** | 회피 기능 비활성화 시 전방 $`\pm 20^\circ`$, $`0.12 \sim 0.25\,\text{m}`$ 장애물 앞에서 $`v=0, \omega=0`$ 정지 대기 | 전방 충돌 방지 세이프티 |
| **`HANDOVER_TO_ARUCO`** | 출발 원점 반경 0.50m 진입 시 `/aruco/dock_trigger` 발행 및 시각 서보 노드로 제어권 이양 (`ACTIVATE_DOCKING`) | 종말 도킹 핸드오버 |
| **`ARUCO_DOCKING`** | ArUco 코너 검출 및 `solvePnP` 기반 3D 상대 위치 추정 (`SOLVE_PNP`)<br/>$`X`$ 편차 및 $`Z`$ 거리 비례($`P`$) 제어로 정밀 접근 (`SERVO_CONTROL`) | 비주얼 서보잉 도킹 |
| **`MISSION_COMPLETE`** | 베이스캠프 반경 15cm 이내 최종 안착 확인, 모터 완전 정지 및 임무 완료 | 복귀 임무 완수 |

---

## 5. 생성된 소스 코드 패키지 구조 및 파일 링크

본 시스템을 위해 구축된 완전한 ROS 2 Humble 패키지 디렉터리 구조는 다음과 같다:

- [package.xml](../src/disaster_recovery/package.xml): ROS 2 Humble 의존성 및 패키지 메타데이터 정의
- [setup.py](../src/disaster_recovery/setup.py): 콘솔 스크립트 엔트리포인트 및 런치/파라미터 설치 설정
- [recovery_controller.py](../src/disaster_recovery/disaster_recovery/recovery_controller.py): 온보드 오도메트리 로거, 워치독 감시, Layer-0 `_despike_lidar_ranges`, Phase 2-B Clearance-Gated RL + 2D `A*` 블렌딩, `DYNAMIC_HOLD`, 2D ICP 스캔 매칭, LIFO 백트래킹 제어기
- [local_grid_planner.py](../src/disaster_recovery/disaster_recovery/local_grid_planner.py): `RollingLocalGrid(0.04m)` + `2D A*` + 비파괴 위상 지름길 + Open-Ray Carving(`carve_decay=0.25`) + `ESCAPE` + `_virtual_obstacles`
- [rl_context_policy.py](../src/disaster_recovery/disaster_recovery/rl_context_policy.py): `16스텝` 시계열 문맥 + `1ch(688차원)` / `3ch(976차원)` `12×12` Ego 코스트맵 기반 `ContextRLStudentPolicy` ONNX 추론 엔진 (`sim`/`real_tb3_policy_student[_semantic_v3].onnx`)
- [semantic_mapper_core.py](../src/disaster_recovery/disaster_recovery/semantic_mapper_core.py) · [semantic_map_node.py](../src/disaster_recovery/disaster_recovery/semantic_map_node.py): 단안 RGB + LDS-01 2D LiDAR 투영 퓨전(Eq.3 보정 · 270° CCW · `+0.25 m` 전경 클러스터링) 및 Alg. 1/2 시맨틱 마커·오버레이 저장 노드
- [td3_policy.py](../src/disaster_recovery/disaster_recovery/td3_policy.py): 28차원 상태 벡터 추출기, Residual TD3 Actor-Critic 신경망 및 ONNX Runtime 추론 엔진 (레거시 Baseline 비교용)
- [aruco_docking.py](../src/disaster_recovery/disaster_recovery/aruco_docking.py): OpenCV solvePnP 기반 ArUco 마커 3D 포즈 추정 및 15cm 정밀 도킹 제어기
- [host_ai_pipeline.py](../src/disaster_recovery/disaster_recovery/host_ai_pipeline.py): Host PC WSL2 GPU 가속 YOLO26n 비전(압축 영상·270° 회전·Zero-Lag 워커·15 FPS 캡·`/detection/objects` 발행) 및 Faster-Whisper 오디오 AI 파이프라인
- [gas_sensor_node.py](../src/disaster_recovery/disaster_recovery/gas_sensor_node.py): ADS1115(I2C)로 MQ-2 전압을 읽어 EMA·기준선 보정 후 `/gas/level` 발행
- [ads1115_mq2.py](../src/disaster_recovery/disaster_recovery/ads1115_mq2.py): ADS1115 드라이버 및 MQ-2 계산 함수 (ROS 비의존, 단위 테스트 대상)
- [blackbox_recorder.py](../src/disaster_recovery/disaster_recovery/blackbox_recorder.py): 통신 단절 트리거 시 로컬 MicroSD 영상 및 센서 CSV 무손실 기록 노드
- [audio_capture_node.py](../src/disaster_recovery/disaster_recovery/audio_capture_node.py): RPi 4 AB13X USB 마이크 `arecord` 기반 48kHz PCM 캡처 → `/audio/raw` 발행 노드
- [test_ads1115_mq2.py](../src/disaster_recovery/test/test_ads1115_mq2.py) · [test_hybrid_recovery.py](../src/disaster_recovery/test/test_hybrid_recovery.py) · [test_local_grid_planner.py](../src/disaster_recovery/test/test_local_grid_planner.py) · [test_semantic_controller.py](../src/disaster_recovery/test/test_semantic_controller.py) · [test_semantic_mapper.py](../src/disaster_recovery/test/test_semantic_mapper.py) · [test_semantic_rl_env.py](../src/disaster_recovery/test/test_semantic_rl_env.py): 전체 단위 테스트 (8 + 9 + 9 + 4 + 17 + 8 = **총 `55 passed`**)
- [recovery_params.yaml](../src/disaster_recovery/config/recovery_params.yaml) · [recovery_params_real.yaml](../src/disaster_recovery/config/recovery_params_real.yaml) · [semantic_map_params.yaml](../src/disaster_recovery/config/semantic_map_params.yaml): 시뮬레이션, 실물 TurtleBot3 Burger(Track B), 시맨틱 매핑 운영 파라미터 정의
- [robot_onboard.launch.py](../src/disaster_recovery/launch/robot_onboard.launch.py) · [robot_onboard_real.launch.py](../src/disaster_recovery/launch/robot_onboard_real.launch.py): RPi 4 온보드 생존 노드 일괄 기동 런치
- [disaster_cartographer.rviz](../src/disaster_recovery/config/disaster_cartographer.rviz) · [disaster_station.rviz](../src/disaster_recovery/config/disaster_station.rviz): RViz2 SLAM(`RobotModel` + `/map` Volatile) 및 AI 관제 레이아웃
- [host_station.launch.py](../src/disaster_recovery/launch/host_station.launch.py) · [disaster_world.launch.py](../src/disaster_recovery/launch/disaster_world.launch.py): Host PC AI·워치독 런치 및 다중 재난 Gazebo 월드 런치
- [sim.py](../scripts/host/sim.py) · [ab_disaster_benchmark.py](../scripts/host/ab_disaster_benchmark.py) · [train_phase2_v100.py](../scripts/host/train_phase2_v100.py) · [rl_procedural_env.py](../scripts/host/rl_procedural_env.py): Gazebo 통합 시뮬레이터, 15종 A/B 벤치마크 하네스, V100 Dual-Track RL 학습기

---

## 6. 결선도 및 파라미터 수식 검증 요약

### 6.1 MQ-2 분압기 전압 무결성 검증 (ADS1115 경유)
- 센서 $`V_{CC}`$: $`5.0\,\text{V}`$ / ADS1115 $`V_{DD}`$: $`3.3\,\text{V}`$
- 분압 저항: $`R_1 = 10\,\text{k}\Omega`$ (신호선 직렬), $`R_2 = 20\,\text{k}\Omega`$ (AIN0→GND)
- 최대 입력 전압 (ADS1115 입력 허용 범위 $`3.333\,\text{V} \le 3.6\,\text{V}`$ 충족):

```math
V_{AIN0} = 5.0\,\text{V} \times \frac{20\,\text{k}\Omega}{10\,\text{k}\Omega + 20\,\text{k}\Omega} = 3.333\,\text{V} \le V_{DD} + 0.3\,\text{V} = 3.6\,\text{V}
```

- 노이즈 제거: $`\alpha = 0.15`$ 기반 1차 지수이동평균(EMA) 필터 적용 (`gas_sensor_node`, 20 Hz).
- 가스 지표: `gas_level` $`= \text{clamp}(1 - R_s / R_{s,\text{air}}, 0, 1)`$ — ppm 아님, 현장 보정 필요.

### 6.2 오도메트리 이산화 스택 기준
- 주행 변위 조건: $`\Delta d = \sqrt{(x_k - x_{k-1})^2 + (y_k - y_{k-1})^2} \ge 0.30\,\text{m}`$
- 헤딩 회전 조건: $`\lvert \Delta \theta \rvert = \lvert \text{atan2}(\sin(\theta_k - \theta_{k-1}), \cos(\theta_k - \theta_{k-1})) \rvert \ge 25^\circ \approx 0.4363\,\text{rad}`$

### 6.3 Rotate-then-Translate 제어기 수식
- 회전 단계 ($`\lvert \Delta \theta_{\text{target}} \rvert \gt 0.08\,\text{rad}`$):

```math
v = 0.0\,\text{m/s}, \quad \omega = \text{clamp}(K_{p,\omega} \cdot \Delta \theta_{\text{target}}, -\omega_{\max}, \omega_{\max})
```

- 직진 단계 ($`\lvert \Delta \theta_{\text{target}} \rvert \le 0.08\,\text{rad}`$):

```math
v = \text{clamp}(K_{p,v} \cdot d_{\text{target}}, 0.03\,\text{m/s}, v_{\max}), \quad \omega = \text{clamp}(K_{p,\text{align}} \cdot \Delta \theta_{\text{target}}, -\omega_{\max}, \omega_{\max})
```

- 라이다 비상 정지: 전방 $`\pm 20^\circ`$ 내 유효 거리 $`0.12\,\text{m} \le d_{\text{lidar}} \le 0.25\,\text{m}`$ 검출 시 즉시 $`v=0, \omega=0`$.
- **Phase 2-B Layer-0 & Clearance-Gate 보완:** `_despike_lidar_ranges`($`\pm 3^\circ`$ 7-광선 창 내 유사 거리 이웃 `< 3`개인 `2.20 m` 이내 고립 스파이크 제거)로 연기 난반사를 차단하고, `compute_rl_blend_weight`로 협소 구간에서는 순수 `2D A*`로 양보하며, 곡률 보존 선속도 클램프($`v \le v_{\text{cmd}}`$)로 언더스티어를 방지한다.

### 6.4 ArUco solvePnP 정밀 도킹 수식
- 마커 코너 3D 좌표: $`[-s/2, s/2, 0], [s/2, s/2, 0], [s/2, -s/2, 0], [-s/2, -s/2, 0]`$ ($`s = 0.10\,\text{m}`$)
- PnP 해석: `cv2.solvePnP(obj_points, img_points, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)`
- 최종 오차 수렴 판정 ($`\ge 10\text{ frames}`$ 연속 만족):

```math
\lvert Z_c - 0.20\,\text{m} \rvert \le 0.02\,\text{m}, \quad \lvert X_c \rvert \le 0.02\,\text{m}, \quad \lvert \text{yaw} \rvert \le 0.05\,\text{rad}
```

출발 원점 대비 최종 안착 오차 반경 **15cm 이내** 달성.

---

## 7. Phase 3 구현·검증 현황 및 잔여 과제

> 📄 **상세 보고서 및 기법 비교:** [`차세대_고도화_시맨틱_SLAM_및_RL_재학습_계획.md`](./차세대_고도화_시맨틱_SLAM_및_RL_재학습_계획.md) · [`알고리즘_및_강화학습_기법_단계별_변천사.md`](./알고리즘_및_강화학습_기법_단계별_변천사.md)

1. **논문 기반 2D 재난 시맨틱 맵핑 (`semantic_mapper_core.py`, `semantic_map_node.py` 구현 완료):**
   - 선행 논문(이은원·김진현, *로봇학회 논문지*, 2025)을 단안 RGB + LDS-01 2D LiDAR 환경에 맞게 변형(수식 3 Y 오타 정정 · 270° CCW 역변환 · `+0.25 m` 전경 클러스터링)하여, 추가 Depth 카메라 없이 조난자·저상 장애물·가스 누출·음성 구조 요청 좌표를 `/map` 위 영구 마커(`MarkerArray`)로 통합하고 4종 아티팩트로 저장한다.
2. **3채널 Semantic-Aware RL 재학습 (`688 → 976차원`, `*_semantic_v3.onnx` 검증 완료):**
   - `12×12` 로컬 코스트맵을 **Ch 0(LiDAR 점유) + Ch 1(라이다 사각지대 저상 장애물, `z < 0.171 m`) + Ch 2(가스 반경 `0.55 m` · 조난자 보호 구역)** 3채널로 확장하고, Warm-Start 및 Modality Dropout(`28%`) 훈련으로 120개 Unseen Holdout `99.2%`, 15종 150회 비퇴행 `144/150 (96.00%)`, Gazebo A/B 저상 장애물 충돌 0회(`0.3799 m` 마진)·가스 중심 이격 `2.97×` 확대(`0.2577 m`)를 확인했다.
3. **잔여 과제:** 가스 위험 반경(`0.55 m`) 외곽 완전 이탈을 위한 Ch2 페널티 튜닝(`gas_hazard_free_rate = 46.7%`), 실물 RPi 4 Track B 필드 주행·추론 지연 실측, MQ-2+ADS1115 PCB 납땜 결선.

---

## 8. 결론 및 종합 평가

본 시스템은 Host PC와 TurtleBot3 Burger 온보드 컴퓨터 간의 철저한 계층적 역할 분담 및 포섭 구조 설계를 통해, 외부 네트워크 단절이라는 가혹한 재난 환경에서도 로봇이 단독으로 생존하고 지나온 경로를 안전하게 거슬러 복귀할 수 있는 높은 신뢰성(Phase 2-B 150회 **`99.33%`**, Phase 3 3채널 시맨틱 RL 150회 비퇴행 **`96.00%`**)과 공학적 완성도를 확보하였다. 향후 실물 TurtleBot3 필드 실증과 MQ-2 하드웨어 결선, 가스 회피 가중치 튜닝을 마무리해 현장 투입 완성도를 높일 예정이다.
