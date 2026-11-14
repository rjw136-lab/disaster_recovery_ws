#!/usr/bin/env bash
set -euo pipefail

docker exec jwroh_ros2_humble bash -lc '
  set -euo pipefail
  cd /ws/disaster_recovery_ws
  set +u
  source /opt/ros/humble/setup.bash
  colcon build --packages-select disaster_recovery
  source install/setup.bash
  set -u

  python3 scripts/host/record_3d_evidence.py \
    --scenario collapsed --strategy baseline --trial 3 --fps 30 --max-sim-sec 65 \
    --tag p1_collapsed_baseline_t03 \
    --out-mp4 /hdd/evidence/benchmarks/p1_collapsed_baseline_t03.mp4

  python3 scripts/host/record_3d_evidence.py \
    --scenario collapsed --strategy proposed --trial 3 --fps 30 --max-sim-sec 135 \
    --tag p1_collapsed_proposed_t03 \
    --out-mp4 /hdd/evidence/benchmarks/p1_collapsed_proposed_t03.mp4

  python3 scripts/host/record_3d_evidence.py \
    --scenario collapsed --strategy proposed_rl --trial 3 --fps 30 --max-sim-sec 135 \
    --tag p2_sim2sim_collapsed_t03 \
    --out-mp4 /hdd/evidence/benchmarks/p2_sim2sim_collapsed_t03.mp4
'
