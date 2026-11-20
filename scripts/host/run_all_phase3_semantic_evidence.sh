#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash
cd /ws/disaster_recovery_ws
colcon build --packages-select disaster_recovery
source install/setup.bash

echo "=== [1/4] Quick YOLO26n Gazebo Survivor Check ==="
python3 -u scripts/host/run_phase3_semantic_suite.py --mode check-yolo --gz-port 11360 --ros-domain-id 60

echo "=== [2/4] 10-Trial Semantic Rescue & Avoidance A/B Benchmark ==="
python3 -u scripts/host/run_phase3_semantic_suite.py --mode ab10 --trials 10 --gz-port 11361 --ros-domain-id 61 --rtf 3.5

echo "=== [3/4] 3-Pane Semantic SLAM E2E + Victim Map Coord Recording ==="
python3 -u scripts/host/run_phase3_semantic_suite.py --mode record-e2e --gz-port 11362 --ros-domain-id 62 --display :98

echo "=== [4/4] Blind vs Semantic-Aware A/B (/map active!) + Dynamic Gauntlet Non-Reg 3D ==="
python3 -u scripts/host/run_phase3_semantic_suite.py --mode record-ab --gz-port 11363 --ros-domain-id 63 --display :98

python3 -u scripts/host/record_3d_evidence.py \
  --scenario dynamic_gauntlet --strategy proposed_rl --trial 1 \
  --ab-dir /ws/disaster_recovery_ws/benchmark_results/phase3_semantic_v3 \
  --onnx-path /ws/disaster_recovery_ws/src/disaster_recovery/config/sim_policy_student_semantic_v3.onnx \
  --out-mp4 /hdd/evidence/phase3_semantic/phase3_nonreg_dynamic_gauntlet_t01.mp4 \
  --snapshot-dir /ws/disaster_recovery_ws/docs/images/evidence \
  --tag phase3_nonreg_dynamic_gauntlet_t01 \
  --display :98 --gz-port 11365 --ros-domain-id 65 --rtf 2.0 --fps 20

ffmpeg -y -i /hdd/evidence/phase3_semantic/phase3_nonreg_dynamic_gauntlet_t01.mp4 \
  -vf "fps=10,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer" \
  /ws/disaster_recovery_ws/docs/images/evidence/phase3_nonreg_dynamic_gauntlet_t01.gif >/dev/null 2>&1

cp -f /ws/disaster_recovery_ws/docs/images/evidence/phase3_nonreg_dynamic_gauntlet_t01_encounter.png /hdd/evidence/phase3_semantic/
cp -f /ws/disaster_recovery_ws/docs/images/evidence/phase3_nonreg_dynamic_gauntlet_t01_outcome.png /hdd/evidence/phase3_semantic/
echo "=== ALL PHASE 3 SEMANTIC EVIDENCE COMPLETE ==="
