#!/usr/bin/env bash
# 运行在：工控机
# 作用：启动前按同一 config 回到数据集 episode0 初始位姿，再运行 Robot Client（远程 X-VLA 推理）
#
# 用法：
#   bash scripts/run_async_policy_client_xvla_remote.sh configs/softfold_piper.json
#   SKIP_MOVE_TO_START=true bash scripts/run_async_policy_client_xvla_remote.sh configs/softfold_piper.json
set -euo pipefail
export HF_HUB_OFFLINE=1
CONFIG_PATH="${1:-configs/softfold_piper.json}"
if [[ $# -gt 0 ]]; then
  shift
fi

SKIP_MOVE_TO_START="${SKIP_MOVE_TO_START:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=lib/remote_gpu_config.sh
source "${SCRIPT_DIR}/lib/remote_gpu_config.sh"
remote_gpu_load_config "${CONFIG_PATH}" "${REPO_ROOT}"

echo "Piper async client (remote X-VLA mode)."
echo "  machine: robot IPC"
echo "  policy server: ${ASYNC_SERVER_ADDRESS}"
echo "  ssh host: ${REMOTE_SSH_HOST}"
echo
if [[ "${REMOTE_USE_SSH_TUNNEL}" == "true" ]]; then
  echo "Prerequisites:"
  echo "  1) On ${REMOTE_SSH_HOST}: bash scripts/start_policy_server_xvla.sh ${CONFIG_PATH}"
  echo "  2) On robot IPC: bash scripts/ssh_tunnel_policy_server.sh ${CONFIG_PATH}  (keep open)"
  echo "  3) Then run this script"
  echo
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${SKIP_MOVE_TO_START}" != "true" ]]; then
  echo "=== [1/2] Move to episode start (config=${CONFIG_PATH}) ==="
  bash "${SCRIPT_DIR}/run_move_to_episode_start.sh" "${CONFIG_PATH}"
  echo
  echo "=== [2/2] Start async policy client ==="
else
  echo "SKIP_MOVE_TO_START=true：跳过归位，直接启动 client。"
fi

exec "${SCRIPT_DIR}/run_async_policy_client_xvla.sh" "${CONFIG_PATH}" "$@"
