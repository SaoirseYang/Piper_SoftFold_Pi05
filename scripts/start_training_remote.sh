#!/usr/bin/env bash
# 运行在：工控机
# 作用：同步代码/数据到 A600，并 SSH 启动远端训练。
#
# 用法：
#   bash scripts/start_training_remote.sh configs/softfold_piper_pi05.json
#   SYNC_CODE=false bash scripts/start_training_remote.sh configs/softfold_piper_pi05.json --dry-run
#   DETACHED=true bash scripts/start_training_remote.sh configs/softfold_piper_pi05.json
#   DETACH_MODE=tmux bash scripts/start_training_remote.sh configs/softfold_piper_pi05.json
#
# DETACH_MODE:
#   foreground  — 交互式 ssh -t（默认；与旧行为一致）
#   nohup       — DETACHED=true 的旧行为
#   tmux        — 远端 named tmux session（推荐）
# DETACHED=true 仍等价于 DETACH_MODE=nohup（兼容旧用法）。
set -euo pipefail

CONFIG_PATH="${1:-configs/softfold_piper_pi05.json}"
if [[ $# -gt 0 ]]; then
  shift
fi

SYNC_CODE="${SYNC_CODE:-true}"
SYNC_DATA="${SYNC_DATA:-true}"
DETACHED="${DETACHED:-false}"
DETACH_MODE="${DETACH_MODE:-}"
GPU_CHECK="${GPU_CHECK:-true}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-16000}"

if [[ -z "${DETACH_MODE}" ]]; then
  if [[ "${DETACHED}" == "true" ]]; then
    DETACH_MODE="nohup"
  else
    DETACH_MODE="foreground"
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=lib/remote_gpu_config.sh
source "${SCRIPT_DIR}/lib/remote_gpu_config.sh"
# shellcheck source=lib/remote_resource_check.sh
source "${SCRIPT_DIR}/lib/remote_resource_check.sh"
remote_gpu_load_config "${CONFIG_PATH}" "${REPO_ROOT}"

eval "$(
  CONFIG_PATH="${REMOTE_CONFIG_PATH}" python3 - <<'PY'
import json
import os
import re
import shlex

path = os.environ["CONFIG_PATH"]
with open(path, encoding="utf-8") as f:
    cfg = json.load(f)
training = cfg.get("training") or {}
job = training.get("job_name") or "pi05_train"
safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(job))[:48]
print(f"export TRAIN_JOB_NAME={shlex.quote(str(job))}")
print(f"export TRAIN_TMUX_SESSION={shlex.quote('sf-train-' + safe)}")
PY
)"

if [[ "${SYNC_CODE}" == "true" ]]; then
  bash "${SCRIPT_DIR}/sync_code_to_remote.sh" "${CONFIG_PATH}"
fi
if [[ "${SYNC_DATA}" == "true" ]]; then
  bash "${SCRIPT_DIR}/upload_dataset_to_remote.sh" "${CONFIG_PATH}"
fi

echo
remote_resource_print_gpu
remote_resource_require_gpu "${CUDA_VISIBLE_DEVICES:-0}"
echo
echo "Start remote training on ${REMOTE_SSH_HOST}"
echo "  repo: ${REMOTE_REPO_ROOT}"
echo "  config: ${CONFIG_PATH}"
echo "  extra: $*"
echo "  detach_mode: ${DETACH_MODE}"
echo "  tmux session: ${TRAIN_TMUX_SESSION}"
echo

TRAIN_CMD="cd $(printf '%q' "${REMOTE_REPO_ROOT}") && PIPER_CONDA_ENV=$(printf '%q' "${REMOTE_CONDA_ENV}") bash scripts/start_training_pi05.sh $(printf '%q' "${CONFIG_PATH}")"
if [[ $# -gt 0 ]]; then
  TRAIN_CMD+=" $(printf '%q ' "$@")"
fi

LOG_DIR="${REMOTE_REPO_ROOT}/logs"
LOG_FILE="${LOG_DIR}/train_${TRAIN_TMUX_SESSION}.log"

case "${DETACH_MODE}" in
  foreground)
    ssh -t "${REMOTE_SSH_HOST}" "${TRAIN_CMD}"
    ;;
  nohup)
    ssh "${REMOTE_SSH_HOST}" "mkdir -p $(printf '%q' "${LOG_DIR}") && nohup bash -lc $(printf '%q' "${TRAIN_CMD}") >$(printf '%q' "${LOG_FILE}") 2>&1 & echo \"[remote] training started (nohup), log=${LOG_FILE} pid=\$!\""
    ;;
  tmux)
    REMOTE_WRAP=$(cat <<EOF
set -euo pipefail
mkdir -p $(printf '%q' "${LOG_DIR}")
if tmux has-session -t $(printf '%q' "${TRAIN_TMUX_SESSION}") 2>/dev/null; then
  echo "[remote] tmux session already exists: ${TRAIN_TMUX_SESSION}"
  echo "  attach: ssh ${REMOTE_SSH_HOST} -t tmux attach -t ${TRAIN_TMUX_SESSION}"
  echo "  kill:   ssh ${REMOTE_SSH_HOST} tmux kill-session -t ${TRAIN_TMUX_SESSION}"
  exit 1
fi
tmux new-session -d -s $(printf '%q' "${TRAIN_TMUX_SESSION}") \
  "bash -lc $(printf '%q' "${TRAIN_CMD} 2>&1 | tee -a ${LOG_FILE}; echo [train exited \\\$?]; sleep 2")"
echo "[remote] tmux session started: ${TRAIN_TMUX_SESSION}"
echo "  log:    ${LOG_FILE}"
echo "  attach: ssh ${REMOTE_SSH_HOST} -t tmux attach -t ${TRAIN_TMUX_SESSION}"
echo "  capture: ssh ${REMOTE_SSH_HOST} tmux capture-pane -pt ${TRAIN_TMUX_SESSION}"
EOF
)
    ssh "${REMOTE_SSH_HOST}" "bash -s" <<<"${REMOTE_WRAP}"
    ;;
  *)
    echo "Unknown DETACH_MODE=${DETACH_MODE} (use foreground|nohup|tmux)" >&2
    exit 1
    ;;
esac
