#!/usr/bin/env bash
# SoftFold pi05 工控机一键流水线（upload / train / deploy / status）。
# 保留旧三终端流程；本脚本只是薄封装。
#
# 用法：
#   bash scripts/pi05_pipeline.sh upload configs/softfold_piper_pi05_rgrasp.json
#   bash scripts/pi05_pipeline.sh train  configs/softfold_piper_pi05_rgrasp.json
#   bash scripts/pi05_pipeline.sh train-fg configs/...   # 旧：交互式前台
#   bash scripts/pi05_pipeline.sh train-nohup configs/... # 旧：nohup 后台
#   bash scripts/pi05_pipeline.sh deploy --yaml configs/deploy_async_pi05_rgrasp.yaml
#   bash scripts/pi05_pipeline.sh deploy --ckpt last,050000 --steps 5,10 --alpha 0.2,0.3
#   bash scripts/pi05_pipeline.sh select
#   bash scripts/pi05_pipeline.sh stop
#   bash scripts/pi05_pipeline.sh stop-all
#   bash scripts/pi05_pipeline.sh status
#   bash scripts/pi05_pipeline.sh resources configs/...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_CONFIG="configs/softfold_piper_pi05_rgrasp.json"

# shellcheck source=lib/remote_gpu_config.sh
source "${SCRIPT_DIR}/lib/remote_gpu_config.sh"
# shellcheck source=lib/remote_resource_check.sh
source "${SCRIPT_DIR}/lib/remote_resource_check.sh"

usage() {
  sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

cmd="${1:-}"
if [[ -z "${cmd}" ]]; then
  usage 1
fi
shift || true

case "${cmd}" in
  upload)
    CONFIG_PATH="${1:-${DEFAULT_CONFIG}}"
    bash "${SCRIPT_DIR}/upload_dataset_to_remote.sh" "${CONFIG_PATH}"
    ;;
  train)
    CONFIG_PATH="${1:-${DEFAULT_CONFIG}}"
    if [[ $# -gt 0 ]]; then shift; fi
    DETACH_MODE=tmux bash "${SCRIPT_DIR}/start_training_remote.sh" "${CONFIG_PATH}" "$@"
    ;;
  train-fg|train-foreground)
    CONFIG_PATH="${1:-${DEFAULT_CONFIG}}"
    if [[ $# -gt 0 ]]; then shift; fi
    DETACH_MODE=foreground bash "${SCRIPT_DIR}/start_training_remote.sh" "${CONFIG_PATH}" "$@"
    ;;
  train-nohup)
    CONFIG_PATH="${1:-${DEFAULT_CONFIG}}"
    if [[ $# -gt 0 ]]; then shift; fi
    DETACH_MODE=nohup bash "${SCRIPT_DIR}/start_training_remote.sh" "${CONFIG_PATH}" "$@"
    ;;
  deploy|deploy-up)
    has_yaml=false
    has_config=false
    for a in "$@"; do
      case "${a}" in
        --yaml|-f|--deploy-yaml) has_yaml=true ;;
        --config) has_config=true ;;
      esac
    done
    if [[ "${has_yaml}" == "true" ]]; then
      bash "${SCRIPT_DIR}/deploy_async_inference.sh" up "$@"
    elif [[ "${has_config}" == "false" ]]; then
      bash "${SCRIPT_DIR}/deploy_async_inference.sh" up --config "${DEFAULT_CONFIG}" "$@"
    else
      bash "${SCRIPT_DIR}/deploy_async_inference.sh" up "$@"
    fi
    ;;
  select|list|stop|stop-all|stopall|status)
    if [[ "${cmd}" == "stopall" ]]; then
      cmd="stop-all"
    fi
    bash "${SCRIPT_DIR}/deploy_async_inference.sh" "${cmd}" "$@"
    ;;
  resources|res)
    CONFIG_PATH="${1:-${DEFAULT_CONFIG}}"
    remote_gpu_load_config "${CONFIG_PATH}" "${REPO_ROOT}"
    remote_resource_status
    ;;
  -h|--help|help)
    usage 0
    ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    usage 1
    ;;
esac
