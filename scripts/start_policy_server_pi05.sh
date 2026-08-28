#!/usr/bin/env bash
# 运行在：GPU 服务器 (A600)
# 作用：在本机启动 pi05 Policy Server，等待工控机通过 SSH 隧道连接
set -euo pipefail
CONFIG_PATH="${1:-configs/softfold_piper_pi05.json}"
if [[ $# -gt 0 ]]; then
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=lib/activate_conda_piper.sh
source "${SCRIPT_DIR}/lib/activate_conda_piper.sh"
activate_conda_piper

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "Starting Piper async policy server."
echo "  machine: GPU server (run this script ON A600)"
echo "  python: $(command -v python)"
echo "  config: ${CONFIG_PATH}"
echo "  bind: see policy_server.host / policy_server.port in config"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  IPC connects via: scripts/ssh_tunnel_policy_server.sh"
echo
python -m piper_towel_fold.start_async_policy_server --config "${CONFIG_PATH}" "$@"
