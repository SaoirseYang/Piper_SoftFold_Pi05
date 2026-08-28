#!/usr/bin/env bash
set -euo pipefail

# 训练入口。默认流水线（start_training.py）：
#   1) 读取配置里的 repo_id（原始录制数据，action 仍是 leader）
#   2) 自动生成/复用 sibling *_ojag：关节←observation.state，夹爪←action
#   3) 在 *_ojag 上调用 lerobot-train
#
# 用法：
#   bash scripts/start_training.sh configs/cxn/record_pick_cube_cxn_act.json
#   bash scripts/start_training.sh configs/record_pick_cube_act.json --dry-run
#   bash scripts/start_training.sh configs/foo.json --no-action-compose   # 关闭改写
#   bash scripts/start_training.sh configs/foo.json --action-compose-overwrite
#
# 新录制数据集：配置里只需写原始 repo_id，不必手写 action_compose。

CONFIG_PATH="${1:-configs/record_pick_cube.json}"
if [[ $# -gt 0 ]]; then
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"


export HF_ENDPOINT=https://hf-mirror.com

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m piper_towel_fold.start_training --config "$CONFIG_PATH" "$@"
