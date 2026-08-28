#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/record_pick_cube.json}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m piper_towel_fold.start_recording --config "$CONFIG_PATH"
