#!/usr/bin/env bash
# PI05 训练入口（本机或 GPU 服务器均可）。默认会自动 *_ojag 改写 action。
#   bash scripts/start_training_pi05.sh configs/softfold_piper_pi05.json
#   bash scripts/start_training_pi05.sh configs/foo.json --no-action-compose
set -euo pipefail

CONFIG_PATH="${1:-configs/softfold_piper_pi05.json}"
if [[ $# -gt 0 ]]; then
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=lib/activate_conda_piper.sh
source "${SCRIPT_DIR}/lib/activate_conda_piper.sh"
activate_conda_piper

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m piper_towel_fold.start_training --config "${CONFIG_PATH}" "$@"
