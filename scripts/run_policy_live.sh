#!/usr/bin/env bash
set -euo pipefail

# 真机推理入口：先按同一 config 回到数据集 episode0 初始位姿，再跑 policy live。
#
# 用法：
#   bash scripts/run_policy_live.sh configs/fyx.json
#   SKIP_MOVE_TO_START=true bash scripts/run_policy_live.sh configs/fyx.json   # 跳过归位
#
# 环境变量：
#   SKIP_MOVE_TO_START  默认 false；true 时跳过 move_to_episode_start

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

CONFIG_PATH="${1:-configs/record_pick_cube.json}"
if [[ $# -gt 0 ]]; then
  shift
fi

SKIP_MOVE_TO_START="${SKIP_MOVE_TO_START:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${SKIP_MOVE_TO_START}" != "true" ]]; then
  echo "=== [1/2] Move to episode start (config=${CONFIG_PATH}) ==="
  bash "${SCRIPT_DIR}/run_move_to_episode_start.sh" "${CONFIG_PATH}"
  echo
  echo "=== [2/2] Start policy live ==="
else
  echo "SKIP_MOVE_TO_START=true：跳过归位，直接推理。"
fi

python -m piper_towel_fold.start_policy_live --config "${CONFIG_PATH}" "$@"
