# 📜 누적 변경 이력 (Changelog)

> **Disaster Recovery TurtleBot3 Burger** 저장소의 초기 업로드(`2026-11-05`)부터 최신 커밋까지의 전체 패치 내역을 마일스톤·일자별로 정리한 누적 변경 이력 문서입니다.
> 핵심 요약(최근 6건)은 [`README.md`](README.md#최근-변경사항-patch-notes--최근-6건)에서 확인할 수 있습니다.

---

## 🧭 개발 마일스톤 한눈에 보기

| 단계 (Phase) | 기간 | 핵심 달성 내역 | 대표 커밋 |
| :--- | :---: | :--- | :--- |
| **Phase 0. 코어 아키텍처 구축** | `2026-11-05` | Heartbeat/MQ-2 워치독, LIFO 백트래킹, ArUco 도킹, 블랙박스, Host AI 노드 초안 업로드 | `4792246`~`61f68f6` |
| **Phase 1. 실차 AI · 원터치 런처** | `2026-11-09` ~ `11-10` | YOLO26n(270° 보정), AB13X USB 마이크 + Faster-Whisper STT, RPi·Host `start.py`/`stop.py` 원터치 통합 | `06597a1`~`ee7af1e` |
| **Phase 1.5. 2D ICP & 특허·텔레옵** | `2026-11-11` | 2D LiDAR 키프레임 SVD ICP 슬립 보정, ArUco 10초 가드, WASD 텔레옵, 선행 특허 14선 분석 | `da2aa5b`~`557b9ab` |
| **Phase 2. TD3 DRL 하이브리드 복귀** | `2026-11-12` | `sim.py` 시뮬레이터, 99KB TD3 ONNX 엣지 추론, 50회 몬테카를로 100% 검증, 터널 20회 벤치마크 | `29a6695`~`07d151c` |
| **Phase 3. 붕괴 건물 & GPU 가속 고도화** | `2026-11-13` | `BACKTRACKING_HYBRID`(Pure Pursuit+TD3), 붕괴 건물 15회 벤치마크, WSLg D3D12·Whisper CUDA·YOLO 15FPS | `7e04510`~`73bb888` |
| **Phase 4. 로컬 2D A\* · V100 서버 완주 & Dual-Track RL** | `2026-11-14` | `RollingLocalGrid + 2D A*` **5대 환경 `49/50 (98.0%)`**, V100 **8,847만 스텝 Dual-Track RL & Gazebo Sim-to-Sim `47/50 (94.0%)`**, 3D `.mp4`/`.png` 증거, 트러블슈팅 `#23~#36` 완결 | `f1966bf`~`e455a74` |
| **Phase 4.5 (2-A). Clearance-Gated RL & 30 FPS `/map` 증거·문서 고도화** | `2026-11-15` | `compute_rl_blend_weight` 협소 구간 적응형 블렌딩(**Gazebo S2S `48/50 = 96.0%`**), `slam_toolbox` `/map` + VirtualGL 30 FPS 재녹화 & README GIF, 알고리즘 장단점·하드웨어 사양·선행특허/참고논문 27선 정비 (`24 passed`) | `0560e21`~`2b5dfce` |
| **Phase 4.8 (2-B). 15대 시나리오 적응형 인지 & 안전 포락선** | `2026-11-15` | 동적 장애물 잔상 Ray-Carving(`0.25`) & Layer-0 1D 각도 군집 필터, 비홀로노믹 곡률 클램프·`DYNAMIC_HOLD`로 **15대 150회 `149/150 (99.33%)`** 복귀 완주 | `phase2b` |
| **Phase 5 (3). 단안 RGB-2D LiDAR 시맨틱 SLAM & 3채널 Context-RL** | `2026-11-19` ~ `11-20` | 단안-LiDAR 투영 퓨전(수식 3 Y 오타 정정·270° CCW·전경 클러스터링), Cartographer 4종 영구 마커 오버레이, **Option 2(`frozen_residual`: Frozen Phase 2-B 백본 + Zero-Gated Residual 어댑터, `max_abs_diff = 0.0`, `414.7 KB`, `0.13 ms`)**, Gazebo 시맨틱 10-Trial A/B 엄격 무위험 **`80.0%` (`8/10`, 저상충돌 `0회`) vs Blind `0.0%`**, 15종 150회 엄격 무충돌 **`119/150 (79.3%)`** (Phase 2-B `69.3%` 대비 `+10.0%p`) | `3849fce`~`HEAD` |

---

## 📅 일자별 상세 누적 패치노트 (최신순)

### 2026-11-20 — Phase 3 Option 2 (`frozen_residual`) 비퇴행 아키텍처 채택 · V100 재학습 & 150회 엄격 무충돌 79.3% 달성

| 커밋 | 구분 | 변경 내용 및 주요 파일 |
| :--- | :---: | :--- |
| `HEAD` | `feat/rl/eval` | **Option 1(Hard Channel Gating) vs Option 2(Frozen Phase 2-B Backbone + Zero-Gated Residual Semantic Adapter) 아키텍처 비교 분석 및 Option 2 채택** — 시맨틱 공백(`Ch1=0, Ch2=0`) 구간에서 공유 백본 변형으로 발생하던 초기 3채널 모델(`v3`)의 비시맨틱 환경 엄격 무충돌 퇴행(`69.3% → 58.0%`)을 해결하기 위해 두 후보 아키텍처를 설계·비교했다. 로컬 코스트맵 경계에서 모드 스위칭 채터링이 발생하는 Option 1을 기각하고, 단일 ONNX 내에 동결된 Phase 2-B 백본과 Zero-Gated 잔차 시맨틱 어댑터($g(Ch_1,Ch_2)=\text{clamp}(\max(|Ch_1|,|Ch_2|)/10^{-4},0,1)$)를 결합한 **Option 2(`--arch frozen_residual`)** 를 구현했다. V100 Dual-Track(`10.62M` steps/track) 재학습 및 DAgger 증류 결과 `sim`/`real_tb3_policy_student_semantic_v3.onnx`(`414.7 KB`, CPU `0.13 ms`) 모두 `Ch1=0, Ch2=0`에서 `non_regression_max_abs_diff = 0.0`을 달성했고, Gazebo 시맨틱 10-Trial A/B 엄격 무위험 **`8/10 (80.0%)` vs Blind `0/10 (0.0%)` (저상 충돌 `0회`)**, 5단계 동일 맵 비교 **`11.76s` 최단시간 무충돌 복귀**, 15대 시나리오 150회 엄격 무충돌 **`119/150 (79.3%, same_debris=150/150)`** (Phase 2-B `104/150 = 69.3%` 대비 **`+10.0%p`**, 초기 `v3` `87/150 = 58.0%` 대비 **`+21.3%p`**) 및 단위 테스트 **`56 passed`** 를 검증했다 (`scripts/host/train_phase2_v100.py`, `src/disaster_recovery/test/test_semantic_rl_env.py`, `benchmark_results/phase3_option2/`, `docs/images/evidence_v4/`, `README.md`). |

### 2026-11-20 — Phase 3 2D LiDAR-단안 RGB 투영 시맨틱 SLAM E2E 및 3채널 Context-RL (144/150 = 96.0%) 검증 완료

| 커밋 | 구분 | 변경 내용 및 주요 파일 |
| :--- | :---: | :--- |
| `3849fce` | `feat/docs` | **단안 RGB-2D LiDAR 투영 퓨전 시맨틱 SLAM 코어 및 ROS 2 관제 노드 구축** — 선행 논문(이은원·김진현, *로봇학회 논문지*, 2025)을 기반으로 수식 3 투영 좌표계 오타($X \cdot f_y \rightarrow Y \cdot f_y$)를 바로잡고, 버거 카메라 90° 장착 왜곡을 해소하는 270° CCW 역변환과 배경 벽면 오인을 방지하는 +0.25m 전경 클러스터링 필터를 포함한 `semantic_mapper_core.py`를 순수 Python/NumPy로 신규 구현했다. 또한 Cartographer `/map` TF와 동기화하여 실시간 마커 및 4종 아티팩트(`raw_map.png`, `raw_map.yaml`, `semantic_objects.json`, `semantic_map_overlay.png`)를 영구 저장하는 `semantic_map_node.py`를 통합했다 (`src/disaster_recovery/`, 단위 테스트 17/17 PASS). |
| `3849fce` | `feat/rl` | **Tesla V100 Dual-Track 3채널 Context-RL 재학습 & 초경량 ONNX 증류** — 2D LiDAR LDS-01(Ch0)에 YOLO 저상 장애물(Ch1)과 가스/조난자 위험 구역(Ch2)을 결합한 3채널 미니 코스트맵($12\times 12\times 3$, 976차원) 환경(`rl_procedural_env.py`)을 구축하고, Phase 2-B 체크포인트로부터 Teacher Warm-Start 미세조정을 수행했다. 통신 단절 시 1채널 폴백을 보장하는 Modality Dropout (28%, `modality_dropout_prob = 0.28`) 하에서 120개 Unseen Holdout 평가 99.2%를 달성하고 경량 ONNX(`sim`/`real_tb3_policy_student_semantic_v3.onnx`, 289.2 KB, x86 단일 스레드 CPU 추론 `0.086~0.087 ms`)로 증류했다 (`scripts/host/train_phase2_v100.py`, 단위 테스트 8/8 PASS). |
| `3849fce` | `feat/controller` | **온보드 복귀 제어기 3채널 ONNX 배선 & 시맨틱 캐시 페일세이프** — `recovery_controller.py` 및 `rl_context_policy.py`에 `use_semantic_rl_policy` 파라미터(기본 `false`)와 `SemanticObjectCache`를 추가하여, 정상 시에는 시맨틱 정보를 즉각 반영하고 통신 단절 시에는 마지막 캐시 유지 및 제로-시맨틱 안전 폴백(Ch1/Ch2=0)으로 기존 99.33% 복귀 능력을 보존하도록 연결했다 (`src/disaster_recovery/`, 단위 테스트 4/4 PASS · 전체 55/55 PASS). |
| `3849fce` | `eval/docs` | **Gazebo E2E 시맨틱 회피 A/B 실증 & 15대 시나리오 150회 비퇴행 벤치마크 (`144/150 = 96.00%`)** — Gazebo A/B 결정론적 시험(`semantic_avoidance_ab.json`)에서 Blind 주행(저상 장애물 `0.2485 m` 충돌, 가스 중심 `0.0869 m` 침범, 타임아웃 FAIL) 대비 Proposed RL v3가 저상 장애물 충돌 0회(`0.3799 m` 안전 마진, PASS) 및 가스 중심 이격 `2.97×` 확대(`0.2577 m`, 반경 `0.55 m` 경계 외곽 완전 이탈은 추후 튜닝 과제)로 복귀를 완주함을 입증했다. 또한 기존 15대 재난 시나리오 150회 몬테카를로 벤치마크에서 **`144/150 (96.00%)`** (성공 평균 `100.67 s`, 전체 평균 `107.06 s`, Baseline 15종 `80/150 = 53.33%` / 14종 `80/140 = 57.14%`)를 기록하고 1080p 영상 3종 및 검증 스냅샷 6종을 영구 기록했다 (`benchmark_results/phase3_semantic_v3/`, `docs/images/evidence/`). |

### 2026-11-19 — 논문 기반 2D 재난 시맨틱 맵핑 및 Phase 3 RL 재학습 고도화 계획 수립

| 커밋 | 구분 | 변경 내용 및 주요 파일 |
| :--- | :---: | :--- |
| `e4e211d` | `docs` | **논문 기반 2D 재난 시맨틱 맵핑(단안 RGB + 2D LiDAR 퓨전) & Phase 3 Semantic-Aware RL 재학습 계획 수립** — 선행 논문(이은원·김진현, *로봇학회 논문지*, 2025)을 검토하여 추가 3D Depth 카메라 하드웨어 없이 기존 LDS-01 2D LiDAR와 YOLO26n 단안 카메라 투영 퓨전을 통해 조난자·낮은 장애물·가스·음성 경보를 2D 점유 지도 위에 실시간 마커로 영구 보존하는 Host 관제 레이어(`semantic_map_node`) 아키텍처를 설계했다. 아울러 라이다 사각지대 저상 장애물 선제 회피 및 가스/조난자 보호 구역 우회를 위해 로컬 코스트맵 패치를 3채널(`976`차원)로 확장하고, 통신 단절 시 기존 99.33% 복귀 성능을 보장하는 Modality Dropout 훈련 및 Warm-Start 전이 계획을 수립했다 (`README.md`, `docs/차세대_고도화_시맨틱_SLAM_및_RL_재학습_계획.md`). |

### 2026-11-15 — Phase 2-A Clearance-Gated RL(`48/50 = 96.0%`) · `/map` 30 FPS 재녹화 · 문서 및 참고문헌 정비

| 커밋 | 구분 | 변경 내용 및 주요 파일 |
| :--- | :---: | :--- |
| `HEAD` | `docs` | **선행 특허 중복 제거 · 참고 논문 27선 대폭 확충 · 내부 마크다운 문서 일괄 정비** — `README.md` 하단 선행 특허 2회 중복 표기를 단일 통합 분석표(14건, 동일 패밀리 `#02/#07` 병기 · 독립 문헌 13건)로 통합하고, 프로젝트에 적용된 전 핵심 알고리즘(SVD ICP, Occupancy Grid, `A*`, Pure Pursuit, Regulated Pure Pursuit, Stanley, TD3, PPO, PLR, RMA, Domain Randomization, DAgger, Policy Distillation, Cartographer, `slam_toolbox`, ROS 2, Gazebo, YOLO, Whisper, ArUco, IPPE `solvePnP`, Camera Calibration) 학술 논문 27편 추가 및 내부 문서(`docs/*.md`, `CHANGELOG.md`) 링크·수치·오탈자 일괄 다듬기 |
| `16e7948`~`2b5dfce` | `docs` | **3대 연산 노드 하드웨어 사양 정밀화** — Onboard Edge(`Raspberry Pi 4 Model B 4GB`), Host PC(`Intel Core i7-8700 6C/12T` · `DDR4 16GB 2666MHz` · `GTX 1050 Ti 4GB`), V100 Training Server(`Intel Core i9-10900X 10C/20T` · `DDR4 256GB 2666MHz Non-ECC` · `Tesla V100-PCIE-32GB`) 실측 반영 (`README.md`) |
| `49ade0b`~`2984627` | `docs` | **Phase 1 / Phase 2 / Phase 2-A / Zero-Shot 용어 해설 및 알고리즘별 작동 방식·장단점 비교표 신설** — `Baseline` vs `Phase 1 (Proposed A*)` vs `Phase 2 (v1.0)` vs `Phase 2-A (Clearance-Gated RL)` 작동 원리와 유리한 점·한계점을 체계적으로 정리 (`README.md`) |
| `0560e21` | `feat/docs` | **Phase 2-A Clearance-Gated RL 블렌딩(`48/50 = 96.0%`) · `/map` 30 FPS 전수 재녹화 · README GIF 내장** — `local_grid_planner.py`(`last_clearances`, `last_used_tight_pass`), `rl_context_policy.py`(`observe()` 전용 갱신 분리), `recovery_controller.py`(`compute_rl_blend_weight` 협소 문틀 `c_path < 0.30 m` 시 2D `A*` 100% 이양 + 에피소드 리셋 시 `reset()`), Gazebo 50회 Sim-to-Sim **`48/50 (96.0%)`** 달성(`house` `8/10 → 9/10`, `98.22 s`), `slam_toolbox` 실시간 `/map` 포함 30 FPS NVENC `.mp4` 10종 재녹화 및 `docs/images/evidence/*.gif` 9종 생성, 단위 테스트 `24 passed` |

---

### 2026-11-14 — V100 서버 Phase 1(`49/50`) & Phase 2 Dual-Track RL(`88.47M` 스텝 · S2S `47/50`) 완주

| 커밋 | 구분 | 변경 내용 및 주요 파일 |
| :--- | :---: | :--- |
| `e455a74` | `feat/docs` | **30 FPS VirtualGL/noVNC HUD 증거 고도화 · `slam_toolbox` 연동 · 보고서 `§6.1` 심층 분석 추가** — `record_3d_evidence.py` VirtualGL V100 30 FPS + `h264_nvenc` + 한글 HUD 파이프라인 구축, `start_novnc_gui.sh` 추가, `train_phase2_v100.py` PASS 롤아웃 범례 보강, 5대 환경 보고서 `§6.1` 원인 분석 추가 |
| `4dd00f1` | `feat/docs` | **V100 서버 Phase 1·2 완주 및 최종 보고서 · 트러블슈팅 `#30~#36` 통합** — `_virtual_obstacles` `±0.03m`/9s 튜닝으로 **5대 환경 `49/50 (98.0%)`** 확정(`collapsed`·`collapsed_heavy`·`tunnel`·`maze` 40/40), FastDDS/`debris_utils` 격리(`#31`), 3D Gazebo+RViz2 지연 `gzclient` 캡처(`record_3d_evidence.py`, `#32`), Tesla V100 32GB **`88,473,600` 스텝(392,119 에피소드) Dual-Track RL**(`train_phase2_v100.py`, `rl_procedural_env.py`, `rl_context_policy.py`) 및 **`sim`/`real_tb3_policy_student.onnx` (`231.6 KB`, `0.036~0.039 ms`)**, `proposed_rl` 공식 Gazebo Zero-Shot Sim-to-Sim **`47/50 (94.0%)`**, `sync_to_gdrive.sh`, `test_hybrid_recovery.py`(`23 passed`), `README.md`, 5대 A/B 보고서, 트러블슈팅 `#30~#36` 최종 현행화 |
| `03e8a22`~`917f324` | `docs` | **문서 제목·용어 정리 & Tesla V100 배지 반영** — 15회 보고서 제목 정비, V100 32GB 사양 표기, 타 플랫폼 무관 언급 제거 |
| `b151a4f` | `docs` | **상시 프로젝트 규칙(`AGENTS.md`) & 서버 트러블슈팅 기록 의무화** — 모든 에이전트 공용 `AGENTS.md` 신설 및 `SERVER_MISSION.md`에 `#30+` 트러블슈팅 누적 기록 규칙 추가 |
| `d9168e8` | `docs` | **트러블슈팅 총정리 백서 `#23~#29` 통합 & 상대경로 정정** — Whisper CUDA, 15회 교착 해결, TD3→로컬 2D A\* 전환, 저상 `cafe_table` 보완, `.partial.json`/병렬 격리까지 전수 반영 및 `file://` 링크 제거 (`docs/2026-11-10_통신단절_재난대응_터틀봇3_프로젝트_시도내역_및_트러블슈팅_총정리.md`) |
| `65fce0d`~`a115dc7` | `docs` | **V100 32GB Phase 2 대규모 RL · Dual-Track · 3D 증거 규칙 확정** — `100k~500k+` 에피소드 과적합 방지 프로토콜, RMA/GRU+PLR 능동 적응, `sim_policy` + 실물 휠오돔용 `real_tb3_policy` 이원화 및 3D 사진·`.mp4` 증거 의무화 (`SERVER_MISSION.md`, `docs/2026-11-14_중간상황_및_서버이관_체크포인트.md`) |
| `bbe1d5a` | `feat` | **병렬 Gazebo 포트/도메인 격리 & 하드코딩 경로 제거** — `--gz-port`, `--ros-domain-id`, `_kill_matching_gazebo_procs`, `--kill-stale` 추가 및 7개 파일의 `/home/lab/...` 절대경로 동적 탐색 전환 (`ab_disaster_benchmark.py`, `debris_utils.py`, `plot_ab_comparison.py`, `world_map_geometry.py`, `disaster_world.launch.py`, `recovery_controller.py`, `recovery_params.yaml`) |
| `7de5747` | `chore/docs` | **`house` 7/10 재검증 기록 & 서버 이관 체크포인트 작성** — `_virtual_obstacles`로 `#07` `cafe_table` 받침대 탈출 확인 및 `~0.28m` 틈새 튜닝 포인트 정리 (`SERVER_MISSION.md`, `ab_summary_house_10trials.json`, `docs/2026-11-14_중간상황_및_서버이관_체크포인트.md`) |
| `f1966bf` | `feat` | **`RollingLocalGrid` (`0.04m`) + 2D `A*` + 5대 재난 A/B 벤치마크 & YOLO 병목 제거** — `local_grid_planner.py`(위상 지름길·선행 관측·벽 너머 단축 차단·전방 unpin·`_virtual_obstacles`), `/recovery/reset_episode`, `ab_disaster_benchmark.py`(`.partial.json`, `--resume`, `--reuse-baseline`), `plot_ab_comparison.py`, `world_map_geometry.py`, `host_ai_pipeline.py` 15 FPS 안정화 및 `test_local_grid_planner.py` 확장(`20 passed`) |

---

### 2026-11-13 — 하이브리드 추종 · 붕괴 건물 벤치마크 · GPU 통합 가속

| 커밋 | 구분 | 변경 내용 및 주요 파일 |
| :--- | :---: | :--- |
| `73bb888` | `docs` | **README 현행화 및 패치노트 신설** — 최신 코드(`recovery_params.yaml`, `sim.py`, `host_station.launch.py`) 기준으로 상태 머신·파라미터·저장소 구조·WSLg D3D12 GPU 렌더링 설명 동기화 (`README.md`) |
| `b212688` | `feat` | **RViz2 `RobotModel` 분리 & YOLO 15 FPS 상향** — `disaster_cartographer.rviz` 신규 추가(`/robot_description` + Map/Scan), `start.py`/`sim.py`에서 `use_rviz:=false` 연동, `host_ai_pipeline.py`에 `yolo_max_fps`(기본 `15.0`) 도입 |
| `0871c58` | `perf` | **Faster-Whisper CUDA INT8 GPU 가속** — `_preload_nvidia_cuda_libs()`로 pip cuBLAS/cuDNN 프리로드 지원, `host_station.launch.py`·`start.py` 기본값을 `whisper_device:=cuda`, `compute_type:=int8`로 전환 (실패 시 CPU 자동 폴백) |
| `dde535c` | `docs/feat` | **WSLg D3D12 GPU 기본화 & 붕괴 건물 15회 벤치마크 분석** — `sim.py`에서 `LIBGL_ALWAYS_SOFTWARE` 해제 + `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` 기본 적용, 기본 맵 `--collapsed` 전환, `[O]` 랜덤 3~6개 낙석 투하(`debris_utils.py`), 15회 붕괴 건물 분석 보고서(11/15 PASS · 실패 4건 원인 · CPU/GPU 프로파일링) 추가 (`docs/붕괴건물_15회_자율복귀_벤치마크_분석보고서.md`, `auto_disaster_trials.py`) |
| `bdc4bfe` | `feat` | **붕괴 건물 재난 월드 & 커스텀 버거 모델 통합** — `disaster_collapsed_building.world` 추가, LiDAR `<visualize>true</visualize>` 및 카메라 포함 커스텀 `turtlebot3_burger` SDF 추가, `disaster_world.launch.py`·`host_ai_pipeline.py`·`recovery_controller.py`·`disaster_station.rviz` 개선 |
| `8e1e538` | `chore` | **구형 시뮬레이션 아티팩트 정리** — 이전 터널 벤치마크 임시 영상·이미지·보고서 제거 및 `.gitignore`/`README.md` 정리 |
| `7e04510` | `feat` | **Regulated Pure Pursuit + Residual TD3 하이브리드 추종 구현** — `BACKTRACKING_HYBRID` 연속 곡률 추종(`lookahead_dist_m=0.45`, `v_max=0.20 m/s`, `w_max=0.65 rad/s`) 및 잔차 RL 블렌딩 적용, `recovery_params.yaml` 고도화 및 `test_hybrid_recovery.py` 통합 테스트 5개(총 13개) 확장 |

---

### 2026-11-12 — TD3 DRL 엣지 추론 · Gazebo 시뮬레이터 · 터널 20회 벤치마크

| 커밋 | 구분 | 변경 내용 및 주요 파일 |
| :--- | :---: | :--- |
| `07d151c` | `docs` | **20회 심층 침투 및 동적 낙석 벤치마크 수치 추가** — 지하 터널 기술 보고서에 20회 연속 시험 데이터 보강 |
| `6b05543` | `feat` | **지하 터널 20회 동적 낙석 벤치마크 실행기 고도화** — `auto_disaster_trials.py`·`plot_tunnel_trials.py`·`recovery_controller.py`·`recovery_params.yaml`에 복귀로 상 불시 낙석 투하 및 TD3 회피 검증 루프 통합 |
| `674e6e9` | `docs` | **소속 표기 단순화** — `PATENT_ANALYSIS_AND_PRIOR_ART.md` 대학 소속 표기 간소화 |
| `3563af6` | `docs` | **기술 보고서 작성자 직함 표기 통일** — 엔지니어링 보고서 및 트러블슈팅 총정리 내 작성자 표기 통일 |
| `ada701e` | `docs` | **README 터널 벤치마크·YOLO 시뮬레이션 가이드 추가** — 20회 터널 시험 요약 및 트러블슈팅 아카이브 반영 |
| `496660f` | `docs` | **터널 보고서 시각화 및 자동 시험 스크립트 보강** — `auto_disaster_trials.py`, `plot_tunnel_trials.py` 개선 및 그래프 삽입 |
| `a744526` | `feat` | **지하 터널 월드 통합 & 극한 스트레스 검증기 추가** — `disaster_tunnel_world.world`, `disaster_maze_world.world`, `disaster_world.launch.py`, `auto_disaster_trials.py`, `plot_tunnel_trials.py`, `verify_extreme_stress.py` 신규 구축 |
| `2daa1ff` | `fix` | **텔레옵 응답성 개선 & 내장 Heartbeat/렌더링 옵션 추가** — `sim.py`에 `--software`/`--headless` 플래그와 내장 10 Hz 하트비트 스레드 적용, `wasd_teleop.py` 이중 발행 보강 |
| `58685f6` | `docs` | **벤치마크 서술 정량화 및 실험 한계 명시** — `README.md`, `DISASTER_RECOVERY_GUIDEBOOK.md`, 제어기 주석을 객관적 지표 중심으로 정비 |
| `aba8d8e` | `docs` | **3D 프린팅 쉘 캡션 수정** — `README.md` 외피 구상도 캡션 문구 정리 |
| `0989670` | `feat` | **4계층 하이브리드 자율복귀 50회 몬테카를로 검증 완료** — `verify_hybrid_recovery.py`·`test_hybrid_recovery.py` 신규 작성, `td3_policy.py`·`train_td3.py`·`recovery_controller.py` 고도화, `recovery_td3.onnx` 갱신 (`101,450 B`, 100% PASS) |
| `4f49841` | `docs` | **GFM 수식/표/Mermaid 모바일 호환성 수정** — `DISASTER_RECOVERY_GUIDEBOOK.md` 및 트러블슈팅 문서 마크다운 문법 보정 |
| `e610661` | `docs` | **README 상단 System Overview 문체 정비** — 시스템 요약 문구 정리 |
| `5ff868e` | `docs` | **공학 표현 정교화** — `README.md`, `DISASTER_RECOVERY_GUIDEBOOK.md`, ICP 보고서 문구 다듬기 |
| `e6971ab` | `fix` | **README 히어로 배너 캐시 갱신** — `docs/images/disaster_recovery_banner_latest.png` 신규 에셋 적용 |
| `8d98bc6` | `docs` | **README 히어로 레이아웃 및 하드웨어 쉘 섹션 재구성** — 3D 프린팅 쉘 이미지를 하드웨어 섹션으로 배치 |
| `ca74f0e` | `feat` | **초경량 TD3 ONNX 엣지 추론 모델 추가** — `src/disaster_recovery/config/recovery_td3.onnx` 최초 커밋 |
| `a4b6570` | `docs` | **소셜 프리뷰 배너 이미지 교정** — 겹치는 크로스헤어 수정 및 소속 배지 정리 (`disaster_recovery_social_preview.png`) |
| `142e916` | `docs/feat` | **TD3 학습기/추론기 초안 & 히어로 배너·이미지 에셋 추가** — `train_td3.py`, `td3_policy.py`, `debris_obstacle.sdf`, `recovery_td3.pth` 및 `recovery_controller.py` TD3 회피 연동 |
| `9d4d4dd` | `fix` | **MathJax 블록 구분자 및 Mermaid 플로우차트 복원** — `DISASTER_RECOVERY_GUIDEBOOK.md` 렌더링 수정 |
| `c6916d7` | `fix` | **수식 블록 들여쓰기 및 모바일 Mermaid 오류 수정** — `DISASTER_RECOVERY_GUIDEBOOK.md` |
| `c830225` | `style` | **가이드북 모바일 렌더링 최적화** — `DISASTER_RECOVERY_GUIDEBOOK.md` 표·수식 레이아웃 정리 |
| `2892261` | `docs` | **Phase 1(#01~#11) 및 Phase 1.5(#12~#22) 트러블슈팅 백서 통합** — `2026-11-10_통신단절_재난대응_터틀봇3_프로젝트_시도내역_및_트러블슈팅_총정리.md` |
| `3d01bc6` | `fix` | **작성자/메인테이너 영문 성명 통일** — 패키지·노드·보고서 전역 `Jong Woo Ro`로 통일 (12개 파일) |
| `f785431` | `fix` | **시뮬레이터 로그 리다이렉트 & 화살표 키 지원** — `sim.py`에서 Gazebo/Cartographer 로그를 `/tmp/disaster_recovery/`로 분리 |
| `1729cae` | `docs` | **README 가독성 전면 재구조화** — 목차·다이어그램·표 중심 문서 개편 |
| `29a6695` | `feat/docs` | **원터치 Gazebo 시뮬레이터 `sim.py` 추가 & 2D ICP 보정 버그 수정** — `scripts/host/sim.py` 최초 도입, `recovery_controller.py` 지속형 SE(2) 오프셋 보정 적용, `README.md` TD3 로드맵 추가 |

---

### 2026-11-11 — 2D ICP 스캔 매칭 · WASD 텔레옵 · 선행 특허 분석

| 커밋 | 구분 | 변경 내용 및 주요 파일 |
| :--- | :---: | :--- |
| `557b9ab` | `feat` | **WASD 텔레옵 및 온보드 프로세스 생명주기 개선** — `scripts/host/wasd_teleop.py` 신규 추가, `start.py`·`scripts/rpi/start_all.sh`·`stop_all.sh`·`gas_sensor_node.py` 개선 |
| `38bcfa3` | `docs` | **선행 특허 14선 검색성 개선 및 영문 표기 동기화** — `PATENT_ANALYSIS_AND_PRIOR_ART.md`, `README.md` |
| `d6566a9` | `docs` | **선행 특허(Prior Art 14선) 및 IP 회피설계 분석 보고서 추가** — `docs/PATENT_ANALYSIS_AND_PRIOR_ART.md` 신규 작성 및 `README.md` 반영 |
| `3f4d0bf` | `docs` | **개발 목표·현황 및 해결된 한계 README 업데이트** — `README.md` 정비 |
| `da2aa5b` | `feat` | **2D LiDAR 키프레임 스캔 매칭(ICP) & ArUco 10초 세이프티 가드 구현** — `recovery_controller.py`에 SVD 2D ICP 정합 추가, `aruco_docking.py`에 마커 미감지 타임아웃 가드 적용, ICP 엔지니어링 보고서·트러블슈팅 문서 작성 |

---

### 2026-11-10 — Host/RPi 원터치 통합 런처 · 온보드 스크립트

| 커밋 | 구분 | 변경 내용 및 주요 파일 |
| :--- | :---: | :--- |
| `ee7af1e` | `feat` | **RPi 4 올인원 런처 & ArUco 3D 포즈 검출기 추가** — `scripts/rpi/start.py`, `stop.py`, `aruco_detector.py`, `audio_capture_node.py`, `generate_marker.py`, `docs/TURTLEBOT3_ARUCO_GUIDE.md` 추가 |
| `00f374b` | `feat` | **Host PC 원터치 런처 `start.py`/`stop.py` & Whisper CPU 가속 연동** — SSH 원격 브링업·tmux 대시보드·사전 점검을 통합한 `scripts/host/start.py`(715줄) 및 `stop.py` 도입, `host_ai_pipeline.py` Whisper CPU int8 연동 |
| `801f986` | `feat` | **RPi 4 쉘 헬퍼 스크립트 & `camera_ros` 가이드 추가** — `scripts/rpi/start_all.sh`, `stop_all.sh`, `start_camera.sh`, `start_audio.sh` 추가 및 `FIELD_TEST_CHECKLIST.md` 갱신 |

---

### 2026-11-09 — 비전/음성 AI 파이프라인 & 카메라 회전 보정

| 커밋 | 구분 | 변경 내용 및 주요 파일 |
| :--- | :---: | :--- |
| `06597a1` | `feat` | **YOLO26n Host 관제 · USB 마이크 오디오 스트리밍 · ArUco 270° 회전 보정** — `host_ai_pipeline.py` Zero-Lag 워커 및 음성 키워드 스포터 구현, `audio_capture_node.py` `arecord` 연동, `aruco_docking.py` 회전 보정, `disaster_station.rviz` 및 `Disaster_Recovery_TurtleBot3_Integration_Report.{md,docx}` 추가 |
| `e7ddce8` | `docs` | **README `ROS_DOMAIN_ID` 설정 수정** — `ROS_DOMAIN_ID=0` 통일 (`README.md`) |

---

### 2026-11-05 — 프로젝트 최초 업로드 및 코어 ROS 2 패키지 구축

| 커밋 | 구분 | 변경 내용 및 주요 파일 |
| :--- | :---: | :--- |
| `61f68f6` | `docs` | **LICENSE 저작권 소유자 표기 수정** (`LICENSE`) |
| `f4aed66` | `docs` | **README 저작권 소유자 표기 수정** (`README.md`) |
| `4792246` | `feat` | **프로젝트 최초 업로드 (23개 파일, +3,200줄)** — `recovery_controller.py`(Heartbeat/가스 워치독, LIFO 백트래킹, LiDAR 긴급 정지), `aruco_docking.py`(`solvePnP` 도킹), `ads1115_mq2.py`·`gas_sensor_node.py`(I2C 가스 센서 + 8개 단위 테스트), `blackbox_recorder.py`, `audio_capture_node.py`, `host_ai_pipeline.py`, `robot_onboard.launch.py`, `host_station.launch.py`, 핵심 설계/회로/네트워크/체크리스트 문서 4종 |
