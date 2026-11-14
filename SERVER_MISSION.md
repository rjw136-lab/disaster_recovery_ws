# Server Agent Mission Brief (`disaster_recovery_ws`)

이 저장소를 클론하거나 풀 받은 연구실 서버 Docker(Ubuntu 22.04 + ROS 2 Humble + Tesla V100 32GB + `/ws` NVMe SSD + `/hdd` 1TB HDD) 에이전트는 **가장 먼저 [docs/2026-11-14_중간상황_및_서버이관_체크포인트.md](docs/2026-11-14_중간상황_및_서버이관_체크포인트.md) 전문을 읽고** 거기에 정의된 순서대로 작업을 수행하라.

## 핵심 요약
1. **저장공간 규칙:** `/` 루트 대용량 저장 금지 · 고속 I/O/실시간 학습 체크포인트는 `/ws` · 장기 아카이브/로그/대용량 데이터는 `/hdd`.
2. **범위 제한:** 현재 범위는 TurtleBot3 Gazebo ROS 2 Humble 시뮬 검증(Phase 1) 및 V100 대규모 강화학습(Phase 2)까지이며, **별도 지시 전까지 Unitree Go2-X 관련 설치·구현은 미리 시작하지 않는다.**
3. **Phase 1:** `_virtual_obstacles` 좁은 복도 틈새(`0.28m`) 폐색 튜닝 및 `colcon build` 1회 완료 후, `--gz-port`와 `--ros-domain-id`를 분리해 시나리오들을 **병렬 실행**(`house` 재검증 · 업그레이드 `maze` 10회 A/B · `collapsed` / `collapsed_heavy` / `tunnel` `--reuse-baseline`) → `plot_ab_comparison.py`(`--img-dir docs/images`) 그래프 검수 및 보고서 최종 업데이트.
4. **Phase 2 (V100 32GB 대규모 학습 · 과적합 엄격 방지):**
   - Gazebo 병목을 피하기 위해 고속 벡터/GPU 병렬 환경에서 **100,000~500,000+ 에피소드(5천만~2억+ 스텝)** 규모로 학습하되, **과적합을 절대 경계**한다.
   - **Train/Test 완전 분리:** 학습은 매 에피소드 새로 생성되는 절차적(Procedural) 재난 맵·무작위 시드로만 수행하고, **미학습 50+ 시드** 및 **공식 Gazebo 5종(`ab_disaster_benchmark.py`)은 학습에 절대 사용하지 않는 순수 Sim-to-Sim 평가용**으로 격리한다.
   - **최종 모델 선정 기준:** 마지막 스텝이 아니라 **미학습 홀드아웃 + Gazebo Sim-to-Sim 교차 검증 최고 점수 체크포인트(Early Stopping)** 를 채택하고, 다중 시드 앙상블 Teacher → 경량 `Student ONNX` 증류 후 하이브리드 최종 검증한다.
