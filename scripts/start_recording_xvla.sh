#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/record_towel_fold_xvla.json}"
SKIP_BRINGUP="${SKIP_BRINGUP:-false}"
SKIP_RESET="${SKIP_RESET:-true}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [[ "${SKIP_BRINGUP}" != "true" ]]; then
  "${SCRIPT_DIR}/bringup_can.sh"
fi

if [[ "${SKIP_RESET}" != "true" ]]; then
  "${SCRIPT_DIR}/reset_arms.sh"
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo
echo "Starting X-VLA recording (LeRobot format, same recorder as pi05/ACT)."
echo "  Config: ${CONFIG_PATH}"
echo "  Press Ctrl+C once after each episode to save."
echo

python -m piper_towel_fold.start_recording --config "${CONFIG_PATH}"
