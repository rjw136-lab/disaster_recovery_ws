#!/usr/bin/env bash
# ==============================================================================
# Upload /hdd/evidence, benchmark plots, and selected Phase 2 summaries to the
# user's Google Drive root folder: "scis server".
#
# Usage:
#   ./scripts/host/sync_to_gdrive.sh                 # upload evidence + docs/images
#   ./scripts/host/sync_to_gdrive.sh --with-archives # also upload /hdd RL archives
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HDD_DIR="${HDD_DIR:-/home/jwroh/hdd}"
REMOTE_NAME="${GDRIVE_REMOTE:-gdrive}"
DRIVE_ROOT="${GDRIVE_ROOT:-scis server}"
PROJECT_SUB="${DRIVE_ROOT}/disaster_recovery_ws"

WITH_ARCHIVES=0
if [[ "${1:-}" == "--with-archives" ]]; then
  WITH_ARCHIVES=1
fi

RCLONE_BIN="${RCLONE_BIN:-$(command -v /usr/local/bin/rclone || command -v rclone || true)}"
if [[ -z "${RCLONE_BIN}" ]]; then
  echo "[ERROR] rclone is not installed on this host."
  exit 1
fi

if ! "${RCLONE_BIN}" listremotes | grep -qx "${REMOTE_NAME}:"; then
  echo "[INFO] rclone remote '${REMOTE_NAME}:' is not configured yet."
  echo "       Run 'rclone config' (New remote -> name: ${REMOTE_NAME} -> storage: drive)"
  echo "       and paste the OAuth token from 'rclone authorize \"drive\"' on a machine with a browser."
  exit 2
fi

echo "[GDRIVE] Ensuring target folder '${REMOTE_NAME}:${PROJECT_SUB}' exists..."
"${RCLONE_BIN}" mkdir "${REMOTE_NAME}:${DRIVE_ROOT}"
"${RCLONE_BIN}" mkdir "${REMOTE_NAME}:${PROJECT_SUB}"

if [[ -d "${HDD_DIR}/evidence" ]]; then
  echo "[GDRIVE] Uploading 3D evidence (${HDD_DIR}/evidence -> ${REMOTE_NAME}:${PROJECT_SUB}/evidence)..."
  "${RCLONE_BIN}" copy "${HDD_DIR}/evidence" "${REMOTE_NAME}:${PROJECT_SUB}/evidence" \
    --transfers 4 --checkers 8 --progress
fi

if [[ -d "${WS_DIR}/docs/images" ]]; then
  echo "[GDRIVE] Uploading report figures (${WS_DIR}/docs/images -> ${REMOTE_NAME}:${PROJECT_SUB}/docs_images)..."
  "${RCLONE_BIN}" copy "${WS_DIR}/docs/images" "${REMOTE_NAME}:${PROJECT_SUB}/docs_images" \
    --transfers 4 --checkers 8 --progress
fi

if [[ -d "${WS_DIR}/benchmark_results/ab_comparison" ]]; then
  echo "[GDRIVE] Uploading Phase 1 benchmark JSONs..."
  "${RCLONE_BIN}" copy "${WS_DIR}/benchmark_results/ab_comparison" "${REMOTE_NAME}:${PROJECT_SUB}/benchmark_results/ab_comparison" \
    --filter "- *.partial.json" --filter "- *.tmp" --filter "+ *.json" --filter "- *" \
    --transfers 4 --checkers 8 --progress
fi

if [[ -d "${WS_DIR}/benchmark_results/phase2_sim2sim" ]]; then
  echo "[GDRIVE] Uploading Phase 2 Sim-to-Sim JSONs..."
  "${RCLONE_BIN}" copy "${WS_DIR}/benchmark_results/phase2_sim2sim" "${REMOTE_NAME}:${PROJECT_SUB}/benchmark_results/phase2_sim2sim" \
    --filter "- *.partial.json" --filter "- *.tmp" --filter "+ *.json" --filter "- *" \
    --transfers 4 --checkers 8 --progress
fi

if [[ -f "${WS_DIR}/benchmark_results/phase2_training_summary.json" ]]; then
  echo "[GDRIVE] Uploading Phase 2 training summary JSON..."
  "${RCLONE_BIN}" copyto "${WS_DIR}/benchmark_results/phase2_training_summary.json" \
    "${REMOTE_NAME}:${PROJECT_SUB}/benchmark_results/phase2_training_summary.json" --progress
fi

if [[ "${WITH_ARCHIVES}" -eq 1 && -d "${HDD_DIR}/rl_archives" ]]; then
  echo "[GDRIVE] Uploading RL model archives (${HDD_DIR}/rl_archives)..."
  "${RCLONE_BIN}" copy "${HDD_DIR}/rl_archives" "${REMOTE_NAME}:${PROJECT_SUB}/rl_archives" \
    --transfers 2 --checkers 4 --progress
fi

echo "[GDRIVE] Done. Remote contents:"
"${RCLONE_BIN}" lsd "${REMOTE_NAME}:${PROJECT_SUB}"
