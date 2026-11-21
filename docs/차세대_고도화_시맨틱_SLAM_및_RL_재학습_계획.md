# 🗺 차세대 고도화 계획 및 구현 완료 보고: 논문 기반 시맨틱 SLAM & Phase 3 강화학습

> **한 줄 요약**: 참고 논문의 **"카메라로 찾은 장애물을 2D 지도 위에 표시하는 기술"**을 우리 프로젝트에 맞게 변형하여 **조난자·낮은 장애물·가스·구조요청 위치가 영구 보존되는 재난 지도(`semantic_map_node`)**를 구축하고, 이 정보를 3채널 미니맵($12\times 12\times 3$)으로 입력받아 **부딪히기 전에 미리 피해 가는 Phase 3 강화학습 모델(`sim`/`real_tb3_policy_student_semantic_v3.onnx`)**의 V100 Dual-Track 재학습 및 Gazebo E2E 회피 검증을 완주했다. (구현 및 V100·Gazebo 시뮬레이션 검증 완료 · 실물 TB3 캘리브레이션·필드 적용 예정)

---

<!-- TOC:BEGIN -->
## 📑 목차

- [1. 왜 기존 알고리즘을 바꾸려 했는가?](#1-왜-기존-알고리즘을-바꾸려-했는가)
- [2. 참고한 논문은 무엇이며, 핵심은 무엇인가?](#2-참고한-논문은-무엇이며-핵심은-무엇인가)
- [3. 우리 프로젝트에 어떻게 맞춰 적용했는가?](#3-우리-프로젝트에-어떻게-맞춰-적용했는가)
  - [3.1 핵심 차이: 우리 로봇에는 3D 깊이 카메라가 없다](#31-핵심-차이-우리-로봇에는-3d-깊이-카메라가-없다)
  - [3.2 구현 및 통합된 시맨틱 SLAM 파이프라인](#32-구현-및-통합된-시맨틱-slam-파이프라인)
- [4. 강화학습(Phase 3)은 어떻게 다시 학습했는가?](#4-강화학습phase-3은-어떻게-다시-학습했는가)
  - [4.1 무엇이 달라졌나?: 로봇에게 주는 '미니 지도'를 1장 → 3장(976차원)으로 확장](#41-무엇이-달라졌나-로봇에게-주는-미니-지도를-1장-3장976차원으로-확장)
  - [4.2 적용된 핵심 안전장치 3가지 & 검증 성과](#42-적용된-핵심-안전장치-3가지-검증-성과)
- [5. 단계별 진행 결과 및 달성 지표](#5-단계별-진행-결과-및-달성-지표)
  - [5.1 향후 잔여 과제: 실물 TurtleBot3 캘리브레이션 및 필드 실증](#51-향후-잔여-과제-실물-turtlebot3-캘리브레이션-가스-회피-튜닝-및-필드-실증)
- [6. 구현 시 해결한 주의사항 3가지](#6-구현-시-해결한-주의사항-3가지)

<!-- TOC:END -->

---

## 1. 왜 기존 알고리즘을 바꾸려 했는가?

기존 `disaster_recovery_ws`(Phase 2-B)는 15가지 재난 시나리오 150회 클린 테스트(`phase2b_strict_clean`)에서 **엄격 무충돌 `64.00%(96/150)` / 물리 도착 `93.33%(140/150)`** (과거 Legacy(충돌 허용) 기록 `149/150 = 99.33%`, 원본 JSON 미보관) 복귀를 달성했다. 하지만 실제 재난 환경 관제 및 회피 측면에서 다음 3가지 명확한 한계가 존재했다.

1. **카메라 AI의 결과가 지도에 남지 않는다**
   - 기존 `host_ai_pipeline.py`는 조난자(`person`)를 찾으면 화면에 네모 박스를 띄우고 경고 문자열만 보냈다.
   - 관제자가 **"그래서 조난자나 위험물이 지도상 정확히 어느 방, 몇 m 지점에 있는지"** 지도(`/map`) 위에서 한눈에 볼 수 없었다.
2. **라이다 높이(약 17 cm, `z = 0.171 m`) 아래 장애물은 부딪혀 봐야 안다**
   - TurtleBot3 Burger의 2D 라이다(LDS-01)는 바닥에서 약 17 cm(`0.171 m`) 높이만 수평으로 훑는다.
   - 그보다 낮은 잔해나 바닥 파이프·벽돌은 라이다에 안 찍히기 때문에, 기존에는 **일단 부딪혀서 멈춘 뒤("어? 안 나가네?") 뒤로 빠져나오는 방식**으로 대처해야 했다.
3. **가스 위험 구역이나 조난자 주변을 미리 돌아가지 못한다**
   - 물리적인 벽이 아닌 "가스 누출 지점"이나 "조난자 바로 앞"은 라이다로 막혀 보이지 않으므로, 강화학습 제어기가 알아서 조심스럽게 비켜 갈 근거가 없었다.

---

## 2. 참고한 논문은 무엇이며, 핵심은 무엇인가?

- **논문**: 이은원 · 김진현 (2025), *「장애물 정보를 포함한 SLAM 기반 실시간 환경 맵핑 알고리즘의 개발 및 구현」*, 로봇학회 논문지 20(2), 245–251.

이 논문이 하는 일은 생각보다 단순하다. **3D SLAM처럼 무거운 작업을 하지 않고, 가벼운 2D 지도 위에 카메라로 본 물체 이름과 위치만 콕콕 찍어주는 것**이다.

```text
[1단계] 카메라로 물체 찾기 (YOLO)
   └─▶ "화면 가운데 의자(Chair)가 보인다!"

[2단계] 거리·방향 계산해서 지도 좌표로 바꾸기
   └─▶ "로봇 위치에서 앞쪽 1.5 m 지점이니까, 지도에서는 (x=3.2, y=1.4)다!"

[3단계] 진짜 있는 물체만 걸러서 지도에 표시·저장하기
   ├─▶ 한 번 깜빡 잘못 본 건 버리고, 여러 번 연속 확인된 것만 '영구 마커'로 확정한다.
   └─▶ 최종 2D 지도 이미지 위에 동그라미와 이름(Chair, Person 등)을 그려 저장한다.
```

---

## 3. 우리 프로젝트에 어떻게 맞춰 적용했는가?

### 3.1 핵심 차이: 우리 로봇에는 3D 깊이 카메라가 없다
논문에서는 거리를 재기 위해 **3D Depth 카메라**를 썼다. 하지만 우리 TurtleBot3 Burger에는 **일반 단안 카메라(RGB)**와 **2D 라이다(LDS-01)**만 달려 있다.

대신 우리에게는 이미 **라이다가 매 순간 정확하게 재는 거리 정보(`/scan`)**가 있다.

| 구분 | 논문 원래 방식 | 우리 프로젝트 적용 방식 (하드웨어 추가 없음) |
| :--- | :--- | :--- |
| **거리 측정** | 3D Depth 카메라로 픽셀별 거리 측정 | **카메라 네모 박스 안에 들어온 2D 라이다 점들의 거리**를 투영·매칭한다. (바닥 잔해 등 라이다 높이 밖 물체는 지면 레이캐스팅 `GroundPlaneFallback`으로 추정) |
| **지도에 찍을 대상** | 실내 가구 10종 (의자, 테이블 등) | **조난자(`person`) + 저상 장애물 + 가스 위험 지점(`/gas/level`) + 구조 요청 위치(`/disaster/audio_alert`)** |
| **실행 위치** | 로봇/PC 통합 ROS 노드 | **Host PC 관제 레이어(`semantic_map_node`)** 에서만 실행한다. (통신이 끊겨도 RPi 4의 단독 자율 복귀 기능은 절대 건드리지 않는다) |

### 3.2 구현 및 통합된 시맨틱 SLAM 파이프라인

실제 구현은 다음 소스 파일 및 ROS 2 인터페이스로 완성되어 검증을 마쳤다:

1. **투영 퓨전 코어 엔진 ([`semantic_mapper_core.py`](../src/disaster_recovery/disaster_recovery/semantic_mapper_core.py))**:
   - `CameraIntrinsics.from_fov`: 핀홀 카메라 내부 행렬 구성 및 선행 논문 수식 (3)의 오타($X \cdot f_y \rightarrow Y \cdot f_y$)를 수학적으로 바로잡았다.
   - `unrotate_point`, `unrotate_bbox_xyxy`: TurtleBot3의 90° 시계방향 회전 장착 카메라를 소프트웨어적으로 정합하기 위해 270° CCW 역변환을 지원한다.
   - `LidarProjectionDepth`: 바운딩 박스 가로 중앙 70% 영역에 투영된 유효 LiDAR 광선 중 최소 깊이 $Z \le \min(Z) + 0.25\,\text{m}$ 전경 클러스터 중앙값을 추출해 원거리 배경 벽면 오인을 원천 차단했다.
   - `GroundPlaneFallback`: 지면 광선 교점 역추적으로 라이다 사각지대 저상 물체의 3차원 위치를 추정한다.
2. **시맨틱 매핑 ROS 2 노드 ([`semantic_map_node.py`](../src/disaster_recovery/disaster_recovery/semantic_map_node.py))**:
   - `/detection/objects` (JSON) 및 `/scan`, Cartographer `/map` TF를 수집하여 2D 지도 좌표계에 객체를 투영한다.
   - 논문 `Algorithm 1`을 확장한 수명 주기(TTL) 및 다중 관측 승격 필터로 일회성 오탐을 배제하고, RViz2 시각화용 `/semantic_map/markers`와 온보드 제어기용 `/semantic_map/cached_objects` 토픽을 발행한다.
   - `/semantic_map/save_map` (`std_srvs/srv/Trigger`) 서비스 호출 시 논문 `Algorithm 2`에 따라 4종 아티팩트(`raw_map.png`, `raw_map.yaml`, `semantic_objects.json`, `semantic_map_overlay.png`)를 영구 저장한다.
3. **호스트 AI 파이프라인 ([`host_ai_pipeline.py`](../src/disaster_recovery/disaster_recovery/host_ai_pipeline.py))**:
   - YOLO26n 추론 박스와 원본 이미지 타임스탬프(`stamp`), `frame_id`를 보존한 `/detection/objects` 토픽을 실시간으로 발행한다.

---

## 4. 강화학습(Phase 3)은 어떻게 다시 학습했는가?

지도에 장애물과 위험 지점을 찍는 것에서 끝내지 않고, **로봇의 복귀 주행 두뇌(RL 모델)가 이 정보를 보고 알아서 피해 가도록** Phase 3 V100 Dual-Track 재학습을 완료했다.

### 4.1 무엇이 달라졌나?: 로봇에게 주는 '미니 지도'를 1장 → 3장(976차원)으로 확장

기존 강화학습 모델(`ContextRLStudent`)은 로봇 주변 $1.8\,\text{m} \times 1.8\,\text{m}$ 영역을 $12 \times 12$ 격자로 쪼갠 **미니 지도 1장(라이다 장애물, 144셀 + 544스칼라 = 688차원)**만 보고 운전했다.
Phase 3에서는 같은 해상도의 미니 지도를 **3장(3채널: 432셀 + 544스칼라 = 976차원 Flat Input)**으로 확장했다.

| 미니 지도 | 담기는 정보 | 로봇이 학습한 실제 회피 행동 |
| :---: | :--- | :--- |
| **Ch 0** *(기존)* | **2D 라이다 벽·기둥 점유** | 벽, 붕괴 기둥, 이동 낙석 충돌 방지 |
| **Ch 1** *(신규)* | **라이다에 안 보이는 낮은 장애물** | 부딪히기 전에 미리 `0.380 m` 안전 마진을 두고 회피 (`low_obstacle_collision: False`) |
| **Ch 2** *(신규)* | **가스 위험 구역(`0.55 m`) · 조난자 보호 구역** | 초기 1회 `semantic_avoidance_ab.json`에서 유해 가스 중심점 이격을 Blind 대비 `2.97×`(`0.087 m → 0.258 m`) 확대했으나 `gas_zone_violated=True` 및 `final_error_m=0.2401` (`24.01 cm`) 한계가 있었으며, 최신 Option 2 10-Trial A/B(`semantic_avoidance_ab_10trials.json`)에서 **`9/10 (90.0%)`, 가스/저상 위반 `0회`** 달성 |

### 4.2 적용된 핵심 안전장치 3가지 & 검증 성과

1. **통신이 끊겼을 때도 100% 작동해야 한다 (`modality_dropout_prob = 0.28`, 28% Dropout)**
   - 재난 현장에서 통신이 끊기면 Host PC가 새 마커를 보내줄 수 없다.
   - 통신이 끊기기 직전까지 받은 마커는 온보드 제어기 메모리([`recovery_controller.py`](../src/disaster_recovery/disaster_recovery/recovery_controller.py), `SemanticObjectCache`)에 저장해 두고 계속 쓰되, **학습 중 28% 확률로 Ch 1·2를 강제로 0으로 마스킹**하여 훈련했다.
   - 그 결과, 시맨틱 입력이 완전히 두절된 조건에서도 120개 Unseen Holdout 기준(`phase3_option2_training_summary.json`) **Holdout 성공률 `TEST 94.2% / VAL 95.0%`** (초기 `phase3_semantic_training_summary.json` `sim 98.33% / real_tb3 96.67%`) 및 `Ch1=Ch2=0` 비퇴행 `max_abs_diff = 0.0`을 기록했다.
2. **가짜 정보(오탐)·위치 오차 노이즈 주입 훈련**
   - 시뮬레이터 훈련 단계에서 마커 위치에 $\pm 0.15\,\text{m}$ 가우시안 노이즈, 등록 지연, 미탐지, 가짜 마커를 무작위 주입하여 위치 오차에 흔들리지 않는 견고한 궤적을 학습시켰다.
3. **Option 2 (`frozen_residual`) 비퇴행 학습 & ONNX 분리 증류**
   - Phase 2-B 백본을 동결하고 Zero-Gated 잔차 시맨틱 어댑터와 후단 Safety Shield를 결합한 Option 2 아키텍처를 적용했다.
   - 기존 모델 파일은 그대로 보존하고 **`sim_policy_student_semantic_v3.onnx`** 및 **`real_tb3_policy_student_semantic_v3.onnx`** (`415.0 KB`, x86 단일 스레드 CPU 추론 평균 `0.121–0.124 ms`; 초기 `v3` `289.2 KB`, `0.089 ms`)로 안전하게 분리 증류했다.

---

## 5. 단계별 진행 결과 및 달성 지표

| 단계 | 계획했던 작업 내용 | 실제 달성 성과 및 지표 | 상태 |
| :---: | :--- | :--- | :---: |
| **1단계**<br>시맨틱 SLAM 코어 & 노드 | • 카메라 박스 + 라이다 거리 결합 로직 작성<br>• 논문 Alg. 1/2 마커 승격 및 2D 오버레이 지도 저장 구현<br>• Gazebo RViz2 마커 표출 | • `semantic_mapper_core.py` & `semantic_map_node.py` 구현<br>• `test_semantic_mapper.py` **17/17 PASS**<br>• `/semantic_map/save_map`으로 4종 아티팩트(`raw_map.png/yaml`, `semantic_objects.json`, `semantic_map_overlay.png`) 저장 확인 | ✅ 완료 |
| **2단계**<br>RL 3채널 환경 확장 | • `rl_procedural_env.py` 3채널 코스트맵 확장<br>• 저상 장애물 밭, 가스·조난자 복도 시나리오 추가<br>• Modality Dropout(28%) 및 시맨틱 노이즈 주입 | • 976차원 관측 공간 구성 완료<br>• `test_semantic_rl_env.py` **9/9 PASS**<br>• 256 병렬 환경 무오류·무결측(zero-NaN) 검증 완료 | ✅ 완료 |
| **3단계**<br>V100 대규모 재학습 & ONNX | • Phase 2-B 기반 Option 2 (`frozen_residual`) 재학습<br>• RPi 4용 초경량 3채널 ONNX 모델 증류 | • 120개 Unseen Holdout 시드 전수 평가(`phase3_option2_training_summary.json`): **Holdout SR `TEST 94.2% / VAL 95.0%`** (초기 `v3` `sim 98.33% / real_tb3 96.67%`)<br>• `Ch1=Ch2=0` 비퇴행 `max_abs_diff = 0.0`, `real_tb3` TEST Gas Hazard-Free `73.3%` (VAL `66.7%`)<br>• 배포 모델 크기 **`415.0 KB`**, x86 단일 스레드 CPU 추론 **`0.121–0.124 ms`** (`sim 0.1207 ms / real_tb3 0.1238 ms`) | ✅ 완료 |
| **4단계**<br>제어기 통합 & Gazebo 검증 | • `recovery_controller.py`에 `SemanticObjectCache` 및 `use_semantic_rl_policy` 연동<br>• Gazebo A/B 결정론적 회피 시험<br>• 15대 재난 시나리오 150회 비퇴행 벤치마크 | • `test_semantic_controller.py` **4/4 PASS** (전체 테스트 56/56 PASS)<br>• Gazebo 10-Trial A/B(`semantic_avoidance_ab_10trials.json`): Blind `0/10` 대비 **엄격 무위험 `9/10 (90.0%)`, 저상/가스 위반 `0회`** (초기 1회 `semantic_avoidance_ab.json`은 저상 충돌 0회·가스 이격 `0.2577 m`이나 `gas_zone_violated=True` 및 `final_error_m=0.2401` 한계 존재)<br>• 15대 150회 클린 벤치마크(`phase3_option2_clean`, `AUD-01~05` 이상 `0건`): **엄격 무충돌 `96/150 (64.00%)` / 물리 도착 `141/150 (94.00%)`** (`phase2b_strict_clean` 엄격 `96/150 = 64.00%`와 완전 비퇴행 동률; 초기 `phase3_semantic_v3` 엄격 `87/150 = 58.00%` / 도착 `143/150 = 95.33%`; 과거 Legacy(충돌 허용) 기록 `144/150 = 96.00%`, 원본 JSON 미보관) | ✅ 완료 |

### 5.1 향후 잔여 과제: 실물 TurtleBot3 캘리브레이션, 가스 회피 튜닝 및 필드 실증

시뮬레이션(Gazebo) 및 V100 GPU 가상 학습은 완료되었으며, 남은 최종 잔여 과제는 다음과 같다:

1. **실물 휠 오도메트리 전용 모델 필드 주행 및 RPi 4 추론 지연 실측**:
   - 실물 로봇의 휠 미끄러짐과 바닥 마찰 특성이 사전 학습된 Track B 모델([`real_tb3_policy_student_semantic_v3.onnx`](../src/disaster_recovery/config/real_tb3_policy_student_semantic_v3.onnx))을 Raspberry Pi 4 온보드에 탑재해 추론 지연 실측 및 실내 장애물 자율 복귀 주행을 검증한다.
2. **절차적 환경 잔여 가스 침범률(`real_tb3` TEST 26.7%) 및 동적 교차 고도화**:
   - Option 2 재학습으로 Gazebo `ab10` 가스 위반은 `0회`(`9/10`)로 해결되었고 절차적 홀드아웃 `real_tb3` `gas_hazard_free_rate`도 `73.3%`(TEST)까지 개선되었으나, 잔여 `26.7%` 가스 경계 진입(초기 1회 `semantic_avoidance_ab.json` `gas_zone_violated: True`, `final_error_m=0.2401`) 및 동적 교차 시나리오를 위해 Phase 4 Soft-CBF-QP 및 CVaR-RL로 확장한다.
3. **실물 MQ-2 가스 센서 + ADS1115 PCB 하드웨어 결선**:
   - 설계 완료된 $10\,\text{k}\Omega : 20\,\text{k}\Omega$ 분압 회로를 만능기판(PCB)에 납땜하고, 실제 가스 누출 시 온보드 `/gas/level` 경보 및 비상 복귀 진입을 최종 점검한다.

---

## 6. 구현 시 해결한 주의사항 3가지

1. **카메라 회전 방향 정합 (270° CCW 역변환)**:
   - 실물 TurtleBot3 카메라는 케이블 간섭을 피하기 위해 90° 시계방향으로 누워 장착되어 있다.
   - 단안-라이다 투영 퓨전 시 `unrotate_point` 및 `unrotate_bbox_xyxy`를 적용하여 270° 반시계방향 역변환을 수행함으로써 좌표 좌우 반전 오차를 차단했다.
2. **원거리 배경 벽면 오인 차단 (+0.25m 전경 클러스터링)**:
   - 2D LiDAR 수평 광선이 열린 공간을 지나 후방 벽면에 부딪힐 때, 박스 중심 영역에 벽면 광선이 투영되어 물체 거리가 수 미터로 튀는 현상을 방지했다.
   - 최소 깊이 기준 $+0.25\,\text{m}$ 이내의 전경 포인트 중앙값만 유효 깊이로 수용하도록 필터링했다.
3. **온보드 연산 부하 분리 및 통신 단절 내구성**:
   - 시맨틱 맵핑 및 YOLO AI 연산은 전적으로 Host PC에서 처리된다.
   - Raspberry Pi 4 온보드는 경량 3채널 ONNX 모델(`415.0 KB`, x86 CPU 기준 `0.121–0.124 ms`)과 로컬 좌표 캐시만 소비하므로, 통신이 단절되더라도 로봇의 자율 복귀 루프는 외부 의존 없이 독립 동작한다.
