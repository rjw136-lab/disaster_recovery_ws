#!/usr/bin/env bash
set -euo pipefail

sg docker -c 'docker exec jwroh_ros2_humble bash -lc "
  set -euo pipefail
  cd /ws/disaster_recovery_ws
  set +u
  source /opt/ros/humble/setup.bash
  colcon build --packages-select disaster_recovery --symlink-install
  source install/setup.bash
  set -u

  COMMON=(--fps 30 --snapshot-dir /hdd/evidence/benchmarks)

  python3 scripts/host/record_3d_evidence.py \"\${COMMON[@]}\" \
    --scenario collapsed --strategy baseline --trial 3 --max-sim-sec 65 \
    --tag p1_collapsed_baseline_t03 \
    --out-mp4 /hdd/evidence/benchmarks/p1_collapsed_baseline_t03.mp4

  python3 scripts/host/record_3d_evidence.py \"\${COMMON[@]}\" \
    --scenario collapsed --strategy proposed --trial 3 --max-sim-sec 135 \
    --tag p1_collapsed_proposed_t03 \
    --out-mp4 /hdd/evidence/benchmarks/p1_collapsed_proposed_t03.mp4

  python3 scripts/host/record_3d_evidence.py \"\${COMMON[@]}\" \
    --scenario collapsed --strategy proposed_rl --trial 3 --max-sim-sec 135 \
    --tag p2_sim2sim_collapsed_t03 \
    --out-mp4 /hdd/evidence/benchmarks/p2_sim2sim_collapsed_t03.mp4

  python3 scripts/host/record_3d_evidence.py \"\${COMMON[@]}\" \
    --scenario collapsed_heavy --strategy baseline --trial 2 --max-sim-sec 65 \
    --tag p1_collapsed_heavy_baseline_t02 \
    --out-mp4 /hdd/evidence/benchmarks/p1_collapsed_heavy_baseline_t02.mp4

  python3 scripts/host/record_3d_evidence.py \"\${COMMON[@]}\" \
    --scenario collapsed_heavy --strategy proposed --trial 2 --max-sim-sec 150 \
    --tag p1_collapsed_heavy_proposed_t02 \
    --out-mp4 /hdd/evidence/benchmarks/p1_collapsed_heavy_proposed_t02.mp4

  python3 scripts/host/record_3d_evidence.py \"\${COMMON[@]}\" \
    --scenario house --strategy proposed --trial 4 --max-sim-sec 150 \
    --tag p1_house_proposed_t04 \
    --out-mp4 /hdd/evidence/benchmarks/p1_house_proposed_t04.mp4

  python3 scripts/host/record_3d_evidence.py \"\${COMMON[@]}\" \
    --scenario house --strategy proposed_rl --trial 3 --max-sim-sec 120 \
    --tag p2_sim2sim_house_t03 \
    --out-mp4 /hdd/evidence/benchmarks/p2_sim2sim_house_t03.mp4

  python3 scripts/host/record_3d_evidence.py \"\${COMMON[@]}\" \
    --scenario tunnel --strategy proposed --trial 5 --max-sim-sec 120 \
    --tag p1_tunnel_proposed_t05 \
    --out-mp4 /hdd/evidence/benchmarks/p1_tunnel_proposed_t05.mp4

  python3 scripts/host/record_3d_evidence.py \"\${COMMON[@]}\" \
    --scenario maze --strategy proposed --trial 7 --max-sim-sec 60 \
    --tag p1_maze_proposed_t07 \
    --out-mp4 /hdd/evidence/benchmarks/p1_maze_proposed_t07.mp4

  python3 scripts/host/record_3d_evidence.py \"\${COMMON[@]}\" \
    --scenario maze --strategy proposed_rl --trial 7 --max-sim-sec 60 \
    --tag p2_sim2sim_maze_t07 \
    --out-mp4 /hdd/evidence/benchmarks/p2_sim2sim_maze_t07.mp4

  cp -f /hdd/evidence/benchmarks/*_encounter.png /hdd/evidence/benchmarks/*_outcome.png \
    /ws/disaster_recovery_ws/docs/images/evidence/
"'
