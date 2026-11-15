#!/usr/bin/env bash
set -eo pipefail

WS="/ws/disaster_recovery_ws"
ONNX="${WS}/src/disaster_recovery/config/sim_policy_student.onnx"
AB_DIR="/hdd/rl_v2_exp/bench_phase2b_v3_dynfix"
SNAP_DIR="${WS}/docs/images/evidence"
VID_HDD="/hdd/rl_v2_exp/videos"
VID_REPO="${WS}/docs/videos"

mkdir -p "${SNAP_DIR}" "${VID_HDD}" "${VID_REPO}"

source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"

encode_gif_under_limit() {
  local in_mp4="$1"
  local out_gif="$2"
  local pal="/tmp/pal_$$.png"
  local max_bytes=7800000

  for spec in "760:10:1.0:256" "700:9:1.0:256" "640:8:1.0:256" "580:7:1.0:128" "560:7:0.5:128"; do
    IFS=":" read -r w fps pts maxc <<< "${spec}"
    ffmpeg -y -v error -i "${in_mp4}" \
      -vf "setpts=${pts}*PTS,fps=${fps},scale=${w}:-1:flags=lanczos,palettegen=max_colors=${maxc}:stats_mode=diff" "${pal}"
    ffmpeg -y -v error -i "${in_mp4}" -i "${pal}" \
      -lavfi "setpts=${pts}*PTS,fps=${fps},scale=${w}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4" \
      "${out_gif}"
    local sz
    sz=$(stat -c%s "${out_gif}")
    echo "[GIF] ${out_gif} (${w}px @ ${fps}fps pts=${pts} c=${maxc}) -> ${sz} B"
    if [ "${sz}" -le "${max_bytes}" ]; then
      rm -f "${pal}"
      return 0
    fi
  done
  rm -f "${pal}"
}

record_and_gif() {
  local scenario="$1"
  local trial="$2"
  local out_mp4="$3"
  local out_gif="$4"
  local rtf="${5:-2.5}"
  local max_sim="${6:-135}"

  echo "=== RECORDING ${scenario} trial #${trial} (rtf=${rtf}, max_sim=${max_sim}) ==="
  python3 "${WS}/scripts/host/record_3d_evidence.py" \
    --scenario "${scenario}" \
    --strategy proposed_rl \
    --onnx-path "${ONNX}" \
    --ab-dir "${AB_DIR}" \
    --trial "${trial}" \
    --out-mp4 "${out_mp4}" \
    --snapshot-dir "${SNAP_DIR}" \
    --display :98 \
    --gz-port 11398 \
    --ros-domain-id 98 \
    --rtf "${rtf}" \
    --max-sim-sec "${max_sim}" \
    --fps 30

  encode_gif_under_limit "${out_mp4}" "${out_gif}"
}

encode_gif_under_limit "${VID_REPO}/phase2b_v3_smoke_lidar_corruption_t02.mp4" "${SNAP_DIR}/p2b_smoke_lidar_corruption.gif"
record_and_gif "dynamic_crossfire_alley" 2 "${VID_REPO}/phase2b_v3_dynamic_crossfire_alley_t02.mp4" "${SNAP_DIR}/p2b_dynamic_crossfire_alley.gif" 2.5 135
record_and_gif "u_pocket_trap" 10 "${VID_HDD}/phase2b_v3_u_pocket_trap_t10.mp4" "${SNAP_DIR}/phase2b_u_pocket_trap_t10.gif" 2.5 135
record_and_gif "slip_chicane" 1 "${VID_HDD}/phase2b_v3_slip_chicane_t01.mp4" "${SNAP_DIR}/phase2b_slip_chicane_t01.gif" 2.5 145
record_and_gif "house_hard" 8 "${VID_HDD}/phase2b_v3_house_hard_t08.mp4" "${SNAP_DIR}/phase2b_house_hard_t08.gif" 1.0 150
record_and_gif "dynamic_gauntlet" 1 "${VID_HDD}/phase2b_v3_dynamic_gauntlet_t01.mp4" "${SNAP_DIR}/phase2b_dynamic_gauntlet_t01.gif" 2.5 135
record_and_gif "dynamic_gauntlet" 2 "${VID_HDD}/phase2b_v3_dynamic_gauntlet_t02.mp4" "${SNAP_DIR}/phase2b_dynamic_gauntlet_t02.gif" 2.5 135
python3 "${WS}/scripts/host/capture_3d_world_atlas.py"

echo "=== ALL 7 PHASE 2-B 3D CLIPS & GIFS COMPLETE ==="
