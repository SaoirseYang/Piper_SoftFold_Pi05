#!/usr/bin/env bash
# 运行在：工控机
# 作用：同步代码/数据到 A600，并 SSH 启动远端训练。
#
# 用法：
#   bash scripts/start_training_remote.sh configs/softfold_piper_pi05.json
#   SYNC_CODE=false bash scripts/start_training_remote.sh configs/softfold_piper_pi05.json --dry-run
#   DETACHED=true bash scripts/start_training_remote.sh configs/softfold_piper_pi05.json
set -euo pipefail

CONFIG_PATH="${1:-configs/softfold_piper_pi05.json}"
if [[ $# -gt 0 ]]; then
  shift
fi

SYNC_CODE="${SYNC_CODE:-true}"
SYNC_DATA="${SYNC_DATA:-true}"
DETACHED="${DETACHED:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=lib/remote_gpu_config.sh
source "${SCRIPT_DIR}/lib/remote_gpu_config.sh"
remote_gpu_load_config "${CONFIG_PATH}" "${REPO_ROOT}"

if [[ "${SYNC_CODE}" == "true" ]]; then
  bash "${SCRIPT_DIR}/sync_code_to_remote.sh" "${CONFIG_PATH}"
fi
if [[ "${SYNC_DATA}" == "true" ]]; then
  bash "${SCRIPT_DIR}/upload_dataset_to_remote.sh" "${CONFIG_PATH}"
fi

echo
echo "Start remote training on ${REMOTE_SSH_HOST}"
echo "  repo: ${REMOTE_REPO_ROOT}"
echo "  config: ${CONFIG_PATH}"
echo "  extra: $*"
echo "  detached: ${DETACHED}"
echo

TRAIN_CMD="cd $(printf '%q' "${REMOTE_REPO_ROOT}") && PIPER_CONDA_ENV=$(printf '%q' "${REMOTE_CONDA_ENV}") bash scripts/start_training_pi05.sh $(printf '%q' "${CONFIG_PATH}")"
if [[ $# -gt 0 ]]; then
  TRAIN_CMD+=" $(printf '%q ' "$@")"
fi

if [[ "${DETACHED}" == "true" ]]; then
  ssh "${REMOTE_SSH_HOST}" "mkdir -p $(printf '%q' "${REMOTE_REPO_ROOT}/logs") && nohup bash -lc $(printf '%q' "${TRAIN_CMD}") >$(printf '%q' "${REMOTE_REPO_ROOT}/logs/train_remote.log") 2>&1 & echo \"[remote] training started, log=${REMOTE_REPO_ROOT}/logs/train_remote.log pid=\$!\""
else
  ssh -t "${REMOTE_SSH_HOST}" "${TRAIN_CMD}"
fi
