#!/usr/bin/env bash
# 运行在：工控机（本地已录制数据集的机器）
# 作用：按 config 的 root/repo_id，rsync 数据集到 A600 仓库的相同相对路径。
# 若存在 sibling *_ojag，也会一并同步。
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

echo "同步数据集（保持仓库内相对路径一致）"
echo "  config:       ${REMOTE_CONFIG_PATH}"
echo "  相对路径:     ${DATASET_ROOT_REL}"
echo "  本地仓库:     ${REPO_ROOT}"
echo "  本地数据集:   ${LOCAL_DATASET}"
echo "  服务器仓库:   ${REMOTE_REPO_ROOT}  (config remote_gpu.gpu_repo_root)"
echo "  服务器数据集: ${REMOTE_DATASET}"
echo "  ssh:          ${REMOTE_SSH_HOST}"
echo

sync_one "${LOCAL_DATASET}" "${REMOTE_DATASET}"

OJAG_LOCAL="${LOCAL_DATASET}_ojag"
if [[ -d "${OJAG_LOCAL}" ]]; then
  echo
  sync_one "${OJAG_LOCAL}" "${REMOTE_DATASET}_ojag"
fi

echo
echo "Done. Train on the server with:"
echo "  bash scripts/start_training_remote.sh ${CONFIG_PATH}"
echo "or:"
echo "  ssh ${REMOTE_SSH_HOST}"
echo "  cd ${REMOTE_REPO_ROOT}"
echo "  bash scripts/start_training_pi05.sh ${CONFIG_PATH}"
