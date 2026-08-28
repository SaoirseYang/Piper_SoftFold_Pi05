#!/usr/bin/env bash
# 运行在：工控机
# 作用：把本机 SoftFold 代码同步到 remote_gpu.gpu_repo_root（不含 data/outputs）
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

echo "Sync SoftFold code to remote GPU."
echo "  config: ${REMOTE_CONFIG_PATH}"
echo "  local:  ${REPO_ROOT}/"
echo "  ssh:    ${REMOTE_SSH_HOST}"
echo "  remote: ${REMOTE_REPO_ROOT}/"
echo

ssh "${REMOTE_SSH_HOST}" "mkdir -p $(printf '%q' "${REMOTE_REPO_ROOT}")"

rsync -aH --info=stats2 \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache/' \
  --exclude '*.egg-info/' \
  --exclude 'outputs/' \
  --exclude 'data/' \
  --exclude '.cursor/' \
  "${REPO_ROOT}/" \
  "${REMOTE_SSH_HOST}:${REMOTE_REPO_ROOT}/"

echo
echo "Code sync done: ${REMOTE_SSH_HOST}:${REMOTE_REPO_ROOT}"
