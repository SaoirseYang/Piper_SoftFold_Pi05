#!/usr/bin/env bash

set -euo pipefail

# 将机械臂移动到配置指向数据集的 episode 初始位姿（默认第 0 条）。
# 复用 replay 的 move-to 阶段，不做轨迹回放。
#
# 用法：
#   bash scripts/run_move_to_episode_start.sh configs/fyx.json
#   bash scripts/run_move_to_episode_start.sh configs/fyx.json --dry-run
#   bash scripts/run_move_to_episode_start.sh configs/fyx.json --episode-index 0

CONFIG_PATH="${1:-configs/fyx.json}"
if [[ $# -gt 0 ]]; then
  shift
fi

LEFT_CAN="${LEFT_CAN:-}"
RIGHT_CAN="${RIGHT_CAN:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "Move arms to episode start pose from config..."
echo "  config  -> ${CONFIG_PATH}"
echo "  extra   -> $*"
echo

EXTRA=()
if [[ -n "${LEFT_CAN}" ]]; then
  EXTRA+=(--left-can "${LEFT_CAN}")
fi
if [[ -n "${RIGHT_CAN}" ]]; then
  EXTRA+=(--right-can "${RIGHT_CAN}")
fi

python tools/move_to_episode_start.py \
  --config "${CONFIG_PATH}" \
  "${EXTRA[@]}" \
  "$@"
