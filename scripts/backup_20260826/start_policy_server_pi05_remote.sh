#!/usr/bin/env bash
# 可选：运行在工控机，通过 SSH 远程启动 A600 上的 Policy Server
# 推荐做法：登录 A600 后直接运行 start_policy_server_pi05.sh
set -euo pipefail
CONFIG_PATH="${1:-configs/softfold_piper_pi05.json}"
if [[ $# -gt 0 ]]; then
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=lib/remote_gpu_config.sh
source "${SCRIPT_DIR}/lib/remote_gpu_config.sh"
remote_gpu_load_config "${CONFIG_PATH}" "${REPO_ROOT}"

echo "Starting remote policy server via SSH."
echo "  run from: robot IPC"
echo "  target: ${REMOTE_SSH_HOST}"
echo "  gpu repo: ${REMOTE_REPO_ROOT}"
echo "  conda env: ${REMOTE_CONDA_ENV}"
echo
echo "Preferred: ssh ${REMOTE_SSH_HOST} then run scripts/start_policy_server_pi05.sh directly."
echo

REMOTE_CMD="cd $(printf '%q' "${REMOTE_REPO_ROOT}") && PIPER_CONDA_ENV=$(printf '%q' "${REMOTE_CONDA_ENV}") bash scripts/start_policy_server_pi05.sh $(printf '%q' "${CONFIG_PATH}")"
if [[ $# -gt 0 ]]; then
  REMOTE_CMD+=" $(printf '%q ' "$@")"
fi
ssh -t "${REMOTE_SSH_HOST}" "${REMOTE_CMD}"
