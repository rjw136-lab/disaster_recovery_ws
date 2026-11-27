#!/bin/bash
set -u

LOG_FILE="/tmp/disaster_recovery/ab_queue.log"
mkdir -p /tmp/disaster_recovery
exec >> "$LOG_FILE" 2>&1

echo "=================================================="
echo "ab_queue supervisor started at $(date)."
echo "=================================================="

cd /ws/disaster_recovery_ws
set +u
source /opt/ros/humble/setup.bash
source install/setup.bash
set -u

cleanup_env() {
    echo "Cleaning up residual processes..."
    pkill -9 -f "[g]zserver|[g]zclient|[r]ecovery_controller|[h]ost_ai_pipeline|[s]lam_toolbox" 2>/dev/null || true
    pkill -9 -x gz 2>/dev/null || true
    sleep 5
}

# (0) Finish ab_phase4_150 first if not yet completed
if [ ! -f "/ws/disaster_recovery_ws/benchmark_results/ab_phase4_150/ab_summary_all15_10trials.json" ]; then
    cleanup_env
    echo "Resuming Batch 0: ab_phase4_150..."
    python3 scripts/host/ab_disaster_benchmark.py \
        --scenario all15 \
        --trials 10 \
        --proposed-strategy phase4 \
        --out-dir /ws/disaster_recovery_ws/benchmark_results/ab_phase4_150 \
        --resume
fi

# (1) ab_phase2b_150
cleanup_env
echo "Starting Batch 1: ab_phase2b_150..."
python3 scripts/host/ab_disaster_benchmark.py \
    --scenario all15 \
    --trials 10 \
    --proposed-strategy proposed_rl \
    --context-rl-onnx /ws/disaster_recovery_ws/src/disaster_recovery/config/sim_policy_student.onnx \
    --reuse-baseline \
    --baseline-dir /ws/disaster_recovery_ws/benchmark_results/ab_phase4_150 \
    --out-dir /ws/disaster_recovery_ws/benchmark_results/ab_phase2b_150 \
    --resume

# (2) ab_phase3_150
cleanup_env
echo "Starting Batch 2: ab_phase3_150..."
python3 scripts/host/ab_disaster_benchmark.py \
    --scenario all15 \
    --trials 10 \
    --proposed-strategy proposed_rl \
    --controller-param use_semantic_rl_policy:=true \
    --controller-param enable_dynamic_tracker:=false \
    --controller-param enable_cbf_safety_shield:=false \
    --controller-param semantic_rl_onnx_path:=/ws/disaster_recovery_ws/src/disaster_recovery/config/sim_policy_student_semantic_v3.onnx \
    --reuse-baseline \
    --baseline-dir /ws/disaster_recovery_ws/benchmark_results/ab_phase4_150 \
    --out-dir /ws/disaster_recovery_ws/benchmark_results/ab_phase3_150 \
    --resume

# (3) ab_phase4_real_150
cleanup_env
echo "Starting Batch 3: ab_phase4_real_150..."
python3 scripts/host/ab_disaster_benchmark.py \
    --scenario all15 \
    --trials 10 \
    --proposed-strategy phase4_real \
    --reuse-baseline \
    --baseline-dir /ws/disaster_recovery_ws/benchmark_results/ab_phase4_150 \
    --out-dir /ws/disaster_recovery_ws/benchmark_results/ab_phase4_real_150 \
    --resume

cleanup_env
echo "All 3 queue batches completed."
