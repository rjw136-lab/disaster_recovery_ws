# Server Agent Mission Brief (`disaster_recovery_ws`)

이 저장소를 클론하거나 풀 받은 연구실 서버 Docker(Ubuntu 22.04 + ROS 2 Humble + Tesla V100 32GB + `/ws` NVMe SSD + `/hdd` 1TB HDD) 에이전트는 **가장 먼저 [docs/2026-11-14_중간상황_및_서버이관_체크포인트.md](docs/2026-11-14_중간상황_및_서버이관_체크포인트.md) 전문을 읽고** 거기에 정의된 순서대로 작업을 수행하라.

## 핵심 요약
1. **저장공간 규칙:** `/` 루트 대용량 저장 금지 · 고속 I/O/실시간 학습 체크포인트는 `/ws` · 장기 아카이브/로그/대용량 데이터는 `/hdd`.
2. **범위 제한:** 현재 범위는 TurtleBot3 Burger (Gazebo ROS 2 Humble 시뮬 검증 Phase 1 및 V100 대규모 강화학습 Phase 2)까지로 한정하며, **별도 지시 전까지 타 로봇 플랫폼 관련 SDK 설치·구현은 미리 시작하지 않는다.**
3. **Phase 1:** `_virtual_obstacles` 좁은 복도 틈새(`0.28m`) 폐색 튜닝 및 `colcon build` 1회 완료 후, `--gz-port`와 `--ros-domain-id`를 분리해 시나리오들을 **병렬 실행**(`house` 재검증 · 업그레이드 `maze` 10회 A/B · `collapsed` / `collapsed_heavy` / `tunnel` `--reuse-baseline`) → `plot_ab_comparison.py`(`--img-dir docs/images`) 그래프 검수 및 보고서 최종 업데이트.
4. **Phase 2 (V100 32GB 대규모 · 다중 상황 능동 적응 학습 · 과적합 엄격 방지):**
   - Gazebo 병목을 피하기 위해 고속 벡터/GPU 병렬 환경에서 **100,000~500,000+ 에피소드(5천만~2억+ 스텝)** 규모로 학습하되, **과적합과 파국적 망각(Catastrophic Forgetting)을 절대 경계**한다.
   - **능동적 상황 적응 구조:** 최근 `1~2초` `(LiDAR 시계열 + 직전 cmd vs 실제 변위 오차 + 로컬 코스트맵)` 기반 시계열 문맥 추정(GRU/Transformer · RMA 방식)으로 미끄럼·동적 장애물·협소 틈새·막다른 길을 스스로 구분해 주행 스타일을 동적 전환하고, 다중 상황 혼합 배치 + PLR 기반 난이도 자동 복습으로 전 환경을 고르게 학습한다.
   - **Train/Test 완전 분리:** 학습은 매 에피소드 새로 생성되는 절차적(Procedural) 재난 맵·무작위 시드로만 수행하고, **미학습 50+ 시드** 및 **공식 Gazebo 5종(`ab_disaster_benchmark.py`)은 학습에 절대 사용하지 않는 순수 Sim-to-Sim 평가용**으로 격리한다.
   - **최종 산출물 이원화 (시뮬용 + 실물 TurtleBot3 배포용 2트랙):**
     - **Track A (시뮬레이션용 `sim_policy`):** 미학습 홀드아웃 50+ 시드 및 공식 Gazebo 5종 교차 검증 최고 점수(Early Stopping) 체크포인트 채택 + 경량 `Student ONNX` 추출.
     - **Track B (실물 TurtleBot3용 `real_tb3_policy`):** 실물 **휠 오도메트리 드리프트·좌우 슬립·회전 바이어스**, **모터 데드존/지연**, **실물 LiDAR(LDS-01/LD08) 노이즈·빔 누락·난반사**를 극강 도메인 무작위화로 학습해, `/odom`이 틀어져도 LiDAR 기하·`enable_scan_matching`(ICP)·시계열 문맥으로 강인하게 복귀하는 라즈베리파이 온보드용(`10~20 Hz` CPU) 경량 `ONNX` 별도 추출·검증.
   - **학습 전(Phase 1) vs 학습 후(Phase 2) 동일 조건 비교 평가(필수):** Phase 2 완료 후 공식 5대 환경(`ab_disaster_benchmark.py`, 동일 시드 `--seed 2026` · 동일 탐사 경로 · 동일 `debris_positions`)에서 **A(구형 Baseline) vs B(Phase 1 학습 전 규칙 기반 `2D A*`) vs C(Phase 2 학습 후 `sim_policy` + `A*`)** 의 **복귀 성공률, 복귀 소요 시간(초), 종단 오차(cm)** 를 비교 표·그래프로 산출하고, Track B(`real_tb3_policy`)는 휠 슬립/드리프트 주입 조건에서도 학습 전 대비 개선 폭을 함께 비교한다.
5. **사진 · 3D 동영상 증거 의무 기록 (필수 — 누락 금지):**
   - **벤치마크 중 3D 증거:** 5대 환경(`collapsed`, `collapsed_heavy`, `tunnel`, `house`, `maze`)의 실제 **Gazebo 3D(`gzclient` / Follow-Cam) 및 RViz2 3D 뷰**에서 Baseline 실패 vs Proposed/RL 성공 주행 장면을 `Xvfb` + `ffmpeg` 등으로 **반드시 고화질 `.mp4` 동영상 및 핵심 프레임 `.png` 사진** 으로 녹화·캡처한다.
   - **학습 중 증거:** V100 학습 중 **초기/중기/후기 롤아웃 시각화 영상(`.mp4`) 및 스냅샷(`.png`)**, **학습 곡선(보상·성공률·Generalization Gap PNG)**, **V100 GPU(`nvidia-smi`) 실제 가동 기록**을 남긴다.
   - 모든 원본 영상·대용량 캡처는 `/hdd/evidence/`에 보관하고, 대표 스냅샷은 육안 검수(검은 화면 여부 확인) 후 보고서에 링크할 수 있게 `docs/images/evidence/`에도 정리한다.
6. **트러블슈팅 문서 누적 업데이트 및 Phase 1·2 완료 시 전체 문서 최종 동기화 의무:**
   - Phase 1·2 수행 중 새로운 버그, 튜닝 이슈, Sim-to-Sim/학습 난제 등을 해결할 때마다 [docs/2026-11-10_통신단절_재난대응_터틀봇3_프로젝트_시도내역_및_트러블슈팅_총정리.md](docs/2026-11-10_통신단절_재난대응_터틀봇3_프로젝트_시도내역_및_트러블슈팅_총정리.md)에 **`#30` 이후 번호로 반드시 누적 기록**하고 현황표·상대경로 링크까지 함께 갱신한다. (`AGENTS.md` 상시 규칙 참조)
   - Phase 1·2 완료 후 최종 수치와 3D 증거 자료가 확정되면 [`README.md`](README.md), [`CHANGELOG.md`](CHANGELOG.md), [docs/2026-11-13_5대재난환경_자율복귀_알고리즘_AB비교_및_원리해설_보고서.md](docs/2026-11-13_5대재난환경_자율복귀_알고리즘_AB비교_및_원리해설_보고서.md)(남아 있는 `file://` 링크 상대경로 정리 포함)를 최종 결과로 일괄 업데이트한다.
