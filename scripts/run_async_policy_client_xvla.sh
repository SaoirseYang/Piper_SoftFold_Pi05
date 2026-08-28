#!/usr/bin/env bash
# 运行在：工控机
# 作用：连接 Piper 机械臂 + 相机，通过 SSH 隧道调用远程 X-VLA 推理
set -euo pipefail
CONFIG_PATH="${1:-configs/softfold_piper.json}"
SKIP_BRINGUP="${SKIP_BRINGUP:-true}"
SKIP_RESET="${SKIP_RESET:-true}"
if [[ $# -gt 0 ]]; then
  shift
fi

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
python -m piper_towel_fold.start_async_policy_client --config "${CONFIG_PATH}" "$@"
