# Server Agent Mission Brief (`disaster_recovery_ws`)

이 저장소를 클론하거나 풀 받은 연구실 서버 Docker(Ubuntu 22.04 + ROS 2 Humble + Tesla V100 32GB + `/ws` NVMe SSD + `/hdd` 1TB HDD) 에이전트는 **가장 먼저 [docs/2026-11-14_중간상황_및_서버이관_체크포인트.md](docs/2026-11-14_중간상황_및_서버이관_체크포인트.md) 전문을 읽고** 거기에 정의된 순서대로 작업을 수행하라.

## 핵심 요약
1. **저장공간 규칙:** `/` 루트 대용량 저장 금지 · 고속 I/O/실시간 학습 체크포인트는 `/ws` · 장기 아카이브/로그/대용량 데이터는 `/hdd`.
2. **범위 제한:** 현재 범위는 TurtleBot3 Gazebo ROS 2 Humble 시뮬 검증(Phase 1) 및 V100 대규모 강화학습(Phase 2)까지이며, **별도 지시 전까지 Unitree Go2-X 관련 설치·구현은 미리 시작하지 않는다.**
3. **Phase 1:** `_virtual_obstacles` 좁은 복도 틈새(`0.28m`) 폐색 튜닝 및 `colcon build` 1회 완료 후, `--gz-port`와 `--ros-domain-id`를 분리해 시나리오들을 **병렬 실행**(`house` 재검증 · 업그레이드 `maze` 10회 A/B · `collapsed` / `collapsed_heavy` / `tunnel` `--reuse-baseline`) → `plot_ab_comparison.py`(`--img-dir docs/images`) 그래프 검수 및 보고서 최종 업데이트.
4. **Phase 2 (V100 32GB 대규모 · 다중 상황 능동 적응 학습 · 과적합 엄격 방지):**
   - Gazebo 병목을 피하기 위해 고속 벡터/GPU 병렬 환경에서 **100,000~500,000+ 에피소드(5천만~2억+ 스텝)** 규모로 학습하되, **과적합과 파국적 망각(Catastrophic Forgetting)을 절대 경계**한다.
   - **능동적 상황 적응 구조:** 최근 `1~2초` `(LiDAR 시계열 + 직전 cmd vs 실제 변위 오차 + 로컬 코스트맵)` 기반 시계열 문맥 추정(GRU/Transformer · RMA 방식)으로 미끄럼·동적 장애물·협소 틈새·막다른 길을 스스로 구분해 주행 스타일을 동적 전환하고, 다중 상황 혼합 배치 + PLR 기반 난이도 자동 복습으로 전 환경을 고르게 학습한다.
   - **Train/Test 완전 분리:** 학습은 매 에피소드 새로 생성되는 절차적(Procedural) 재난 맵·무작위 시드로만 수행하고, **미학습 50+ 시드** 및 **공식 Gazebo 5종(`ab_disaster_benchmark.py`)은 학습에 절대 사용하지 않는 순수 Sim-to-Sim 평가용**으로 격리한다.
   - **최종 산출물 이원화 (시뮬용 + 실물 TurtleBot3 배포용 2트랙):**
     - **Track A (시뮬레이션용 `sim_policy`):** 미학습 홀드아웃 50+ 시드 및 공식 Gazebo 5종 교차 검증 최고 점수(Early Stopping) 체크포인트 채택 + 경량 `Student ONNX` 추출.
     - **Track B (실물 TurtleBot3용 `real_tb3_policy`):** 실물 **휠 오도메트리 드리프트·좌우 슬립·회전 바이어스**, **모터 데드존/지연**, **실물 LiDAR(LDS-01/LD08) 노이즈·빔 누락·난반사**를 극강 도메인 무작위화로 학습해, `/odom`이 틀어져도 LiDAR 기하·`enable_scan_matching`(ICP)·시계열 문맥으로 강인하게 복귀하는 라즈베리파이 온보드용(`10~20 Hz` CPU) 경량 `ONNX` 별도 추출·검증.
5. **사진 · 3D 동영상 증거 의무 기록 (필수 — 누락 금지):**
   - **벤치마크 중 3D 증거:** 5대 환경(`collapsed`, `collapsed_heavy`, `tunnel`, `house`, `maze`)의 실제 **Gazebo 3D(`gzclient` / Follow-Cam) 및 RViz2 3D 뷰**에서 Baseline 실패 vs Proposed/RL 성공 주행 장면을 `Xvfb` + `ffmpeg` 등으로 **반드시 고화질 `.mp4` 동영상 및 핵심 프레임 `.png` 사진** 으로 녹화·캡처한다.
   - **학습 중 증거:** V100 학습 중 **초기/중기/후기 롤아웃 시각화 영상(`.mp4`) 및 스냅샷(`.png`)**, **학습 곡선(보상·성공률·Generalization Gap PNG)**, **V100 GPU(`nvidia-smi`) 실제 가동 기록**을 남긴다.
   - 모든 원본 영상·대용량 캡처는 `/hdd/evidence/`에 보관하고, 대표 스냅샷은 육안 검수(검은 화면 여부 확인) 후 보고서에 링크할 수 있게 `docs/images/evidence/`에도 정리한다.
