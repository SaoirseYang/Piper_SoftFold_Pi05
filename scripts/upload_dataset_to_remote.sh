#!/usr/bin/env bash
# 运行在：工控机（本地已录制数据集的机器）
# 作用：按 config 的 root/repo_id，rsync 数据集到 A600 仓库的相同相对路径。
# 若存在 sibling *_ojag，也会一并同步。
# 默认同时上传当前 config JSON（可用 SYNC_CONFIG=false 关闭，行为与备份版一致）。
#
# 用法：
#   bash scripts/upload_dataset_to_remote.sh configs/softfold_piper_pi05_rgrasp.json
#   SYNC_CONFIG=false bash scripts/upload_dataset_to_remote.sh ...   # 仅数据集（旧行为）
#   DISK_CHECK=false bash scripts/upload_dataset_to_remote.sh ...
set -euo pipefail

CONFIG_PATH="${1:-configs/softfold_piper_pi05.json}"
if [[ $# -gt 0 ]]; then
  shift
fi

SYNC_CONFIG="${SYNC_CONFIG:-true}"

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
import shlex
from pathlib import Path

path = os.environ["CONFIG_PATH"]
with open(path, encoding="utf-8") as f:
    cfg = json.load(f)

root = cfg["root"]
repo_id = cfg["repo_id"]
dataset_rel = f"{root}/{repo_id}"

def emit(name, value):
    print(f"export {name}={shlex.quote(str(value))}")

emit("DATASET_ROOT_REL", dataset_rel)
emit("DATASET_REPO_ID", repo_id)
emit("DATASET_ROOT_PARENT_REL", str(Path(dataset_rel).parent))
PY
)"

LOCAL_DATASET="${REPO_ROOT}/${DATASET_ROOT_REL}"
REMOTE_DATASET="${REMOTE_REPO_ROOT}/${DATASET_ROOT_REL}"

if [[ ! -f "${LOCAL_DATASET}/meta/info.json" ]]; then
  echo "本地数据集不存在: ${LOCAL_DATASET}/meta/info.json" >&2
  echo "请先完成录制，或检查 config 中的 root / repo_id。" >&2
  exit 1
fi

sync_one() {
  local local_path="$1"
  local remote_path="$2"
  if [[ ! -d "${local_path}" ]]; then
    return 0
  fi
  echo "rsync ${local_path}/ -> ${REMOTE_SSH_HOST}:${remote_path}/"
  ssh "${REMOTE_SSH_HOST}" "mkdir -p $(printf '%q' "$(dirname "${remote_path}")")"
  rsync -aH --info=stats2 \
    "${local_path}/" \
    "${REMOTE_SSH_HOST}:${remote_path}/"
}

NEED_BYTES="$(remote_resource_local_dataset_bytes "${LOCAL_DATASET}")"
# config is tiny; still reserve headroom via DISK_MIN_FREE_GIB
remote_resource_require_disk "${REMOTE_REPO_ROOT}" "${NEED_BYTES}"
remote_resource_print_disk "${REMOTE_REPO_ROOT}"
echo

echo "同步数据集（保持仓库内相对路径一致）"
echo "  config:       ${REMOTE_CONFIG_PATH}"
echo "  相对路径:     ${DATASET_ROOT_REL}"
echo "  本地仓库:     ${REPO_ROOT}"
echo "  本地数据集:   ${LOCAL_DATASET}"
echo "  服务器仓库:   ${REMOTE_REPO_ROOT}  (config remote_gpu.gpu_repo_root)"
echo "  服务器数据集: ${REMOTE_DATASET}"
echo "  ssh:          ${REMOTE_SSH_HOST}"
echo "  sync config:  ${SYNC_CONFIG}"
echo

if [[ "${SYNC_CONFIG}" == "true" ]]; then
  # Prefer relative path under repo so remote layout matches local.
  if [[ "${REMOTE_CONFIG_PATH}" == "${REPO_ROOT}/"* ]]; then
    CONFIG_REL="${REMOTE_CONFIG_PATH#"${REPO_ROOT}/"}"
  else
    CONFIG_REL="${CONFIG_PATH}"
  fi
  REMOTE_CONFIG_FILE="${REMOTE_REPO_ROOT}/${CONFIG_REL}"
  echo "rsync config -> ${REMOTE_SSH_HOST}:${REMOTE_CONFIG_FILE}"
  ssh "${REMOTE_SSH_HOST}" "mkdir -p $(printf '%q' "$(dirname "${REMOTE_CONFIG_FILE}")")"
  rsync -aH --info=stats2 \
    "${REMOTE_CONFIG_PATH}" \
    "${REMOTE_SSH_HOST}:${REMOTE_CONFIG_FILE}"
  echo
fi

sync_one "${LOCAL_DATASET}" "${REMOTE_DATASET}"

OJAG_LOCAL="${LOCAL_DATASET}_ojag"
if [[ -d "${OJAG_LOCAL}" ]]; then
  echo
  sync_one "${OJAG_LOCAL}" "${REMOTE_DATASET}_ojag"
fi

echo
echo "Done. Train on the server with:"
echo "  bash scripts/start_training_remote.sh ${CONFIG_PATH}"
echo "  # tmux (recommended): DETACH_MODE=tmux bash scripts/start_training_remote.sh ${CONFIG_PATH}"
echo "or legacy one-shot:"
echo "  ssh ${REMOTE_SSH_HOST}"
echo "  cd ${REMOTE_REPO_ROOT}"
echo "  bash scripts/start_training_pi05.sh ${CONFIG_PATH}"
