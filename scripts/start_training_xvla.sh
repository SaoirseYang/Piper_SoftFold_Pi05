#!/usr/bin/env bash
set -euo pipefail

# X-VLA 训练入口；默认会自动 *_ojag 改写 action（与 pi05/ACT 相同）。
#   bash scripts/start_training_xvla.sh configs/record_towel_fold_xvla.json
#   bash scripts/start_training_xvla.sh configs/record_towel_fold_xvla.json --dry-run
#   bash scripts/start_training_xvla.sh configs/record_towel_fold_xvla.json --no-action-compose

CONFIG_PATH="${1:-configs/record_towel_fold_xvla.json}"
if [[ $# -gt 0 ]]; then
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m piper_towel_fold.start_training --config "${CONFIG_PATH}" "$@"
