#!/usr/bin/env bash
# 运行在：工控机
# 作用：建立到 A600 的 SSH 隧道，把本地 8080 转发到 GPU 服务器的 Policy Server
set -euo pipefail
CONFIG_PATH="${1:-configs/softfold_piper_pi05.json}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=lib/remote_gpu_config.sh
source "${SCRIPT_DIR}/lib/remote_gpu_config.sh"
remote_gpu_load_config "${CONFIG_PATH}" "${REPO_ROOT}"

if [[ "${REMOTE_USE_SSH_TUNNEL}" != "true" ]]; then
  echo "remote_gpu.use_ssh_tunnel is false."
  echo "Connect the client directly to: ${ASYNC_SERVER_ADDRESS}"
  exit 0
fi

echo "Opening SSH tunnel for policy server gRPC."
echo "  machine: robot IPC (run this script ON the industrial PC)"
echo "  ssh host: ${REMOTE_SSH_HOST}"
echo "  local:  127.0.0.1:${REMOTE_TUNNEL_LOCAL_PORT}"
echo "  remote: ${REMOTE_TUNNEL_REMOTE_HOST}:${REMOTE_TUNNEL_REMOTE_PORT}"
echo "  client server_address: ${ASYNC_SERVER_ADDRESS}"
echo
echo "Keep this terminal open while the robot client is running."
echo "Press Ctrl+C to close the tunnel."
echo

ssh -N \
  -L "${REMOTE_TUNNEL_LOCAL_PORT}:${REMOTE_TUNNEL_REMOTE_HOST}:${REMOTE_TUNNEL_REMOTE_PORT}" \
  "${REMOTE_SSH_HOST}"
