# Server Agent Mission Brief (`disaster_recovery_ws`)

이 저장소를 클론하거나 풀 받은 연구실 서버 Docker(Ubuntu 22.04 + ROS 2 Humble + Tesla V100 32GB + `/ws` NVMe SSD + `/hdd` 1TB HDD) 에이전트는 **가장 먼저 [docs/2026-11-14_중간상황_및_서버이관_체크포인트.md](docs/2026-11-14_중간상황_및_서버이관_체크포인트.md) 전문을 읽고** 거기에 정의된 순서대로 작업을 수행하라.

## 핵심 요약
1. **저장공간 규칙:** `/` 루트 대용량 저장 금지 · 고속 I/O/실시간 학습 체크포인트는 `/ws` · 장기 아카이브/로그/대용량 데이터는 `/hdd`.
2. **범위 제한:** 현재 범위는 TurtleBot3 Gazebo ROS 2 Humble 시뮬 검증(Phase 1) 및 V100 대규모 강화학습(Phase 2)까지이며, **별도 지시 전까지 Unitree Go2-X 관련 설치·구현은 미리 시작하지 않는다.**
3. **Phase 1:** `_virtual_obstacles` 좁은 복도 틈새(`0.28m`) 폐색 튜닝 및 `colcon build` 1회 완료 후, `--gz-port`와 `--ros-domain-id`를 분리해 시나리오들을 **병렬 실행**(`house` 재검증 · 업그레이드 `maze` 10회 A/B · `collapsed` / `collapsed_heavy` / `tunnel` `--reuse-baseline`) → `plot_ab_comparison.py`(`--img-dir docs/images`) 그래프 검수 및 보고서 최종 업데이트.
4. **Phase 2:** V100 32GB로 **최소 10,000~50,000+ 에피소드** 규모의 커리큘럼 + 도메인 무작위화 + 시계열(Frame-Stack/GRU)/코스트맵 RL(PPO/SAC) 학습 → 미학습 50+ 시드 검증(`≥ 95%`) → 경량 `Student ONNX` 증류 및 하이브리드 검증.
