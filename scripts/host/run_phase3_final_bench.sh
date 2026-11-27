#!/usr/bin/env bash
# Phase 3 final benchmark: 5 scenarios x 10 trials, proposed-only (phase4 w/ 12-line CBF guard).
# Runs inside container jwroh_ros2_humble (ROS 2 Humble). Concurrency=2 to protect
# criterion 4 (ctrl_hz_min >= 8.0 Hz). Each job has an isolated --gz-port and --ros-domain-id.
# NOTE: `set -u` is intentionally NOT used here. /opt/ros/humble/setup.bash (line 8)
# references AMENT_TRACE_SETUP_FILES, which is unset; under `set -u` sourcing aborts
# immediately and the whole benchmark dies before any job starts.
source /opt/ros/humble/setup.bash
cd /ws/disaster_recovery_ws
source install/setup.bash

OUT=/ws/disaster_recovery_ws/benchmark_results/phase3_final
mkdir -p "$OUT"
: > "$OUT/_run_status.log"

run_job() {
  local name=$1 dom=$2 port=$3
  echo "[START] $name dom=$dom port=$port $(date)" >> "$OUT/_run_status.log"
  python3 scripts/host/ab_disaster_benchmark.py \
    --scenario "$name" --trials 10 \
    --proposed-strategy phase4 --proposed-only \
    --out-dir "$OUT" \
    --ros-domain-id "$dom" --gz-port "$port" \
    > "$OUT/${name}.log" 2>&1
  echo "[DONE ] $name exit=$? $(date)" >> "$OUT/_run_status.log"
}
export -f run_job
export OUT

# Concurrency 2 (two at a time, three waves)
run_job dynamic_crossfire_alley 101 11811 &
run_job dynamic_gauntlet        102 11812 &
wait
run_job smoke_lidar_corruption  103 11813 &
run_job collapsed               104 11814 &
wait
run_job u_pocket_trap           105 11815 &
wait

echo "ALL_JOBS_COMPLETE $(date)" >> "$OUT/_run_status.log"
