#!/usr/bin/env bash
# shellcheck shell=bash
# Read remote_gpu / async_inference settings from a JSON config file.

remote_gpu_load_config() {
  local config_path="${1:-configs/softfold_piper_pi05.json}"
  local repo_root="${2:-}"

  if [[ -z "${repo_root}" ]]; then
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
    repo_root="$(cd "${script_dir}/../.." && pwd)"
  fi

  if [[ ! -f "${repo_root}/${config_path}" && -f "${config_path}" ]]; then
    config_path="${config_path}"
  else
    config_path="${repo_root}/${config_path}"
  fi

  if [[ ! -f "${config_path}" ]]; then
    echo "Config not found: ${config_path}" >&2
    return 1
  fi

  eval "$(
    CONFIG_PATH="${config_path}" python3 - <<'PY'
import json
import os
import shlex

path = os.environ["CONFIG_PATH"]
with open(path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

remote = cfg.get("remote_gpu", {}) or {}
deployment = cfg.get("deployment", {}) or {}
gpu_server = deployment.get("gpu_server", {}) or {}
async_cfg = cfg.get("async_inference", {}) or {}
policy_server = cfg.get("policy_server", {}) or {}

def emit(name, value):
    print(f"export {name}={shlex.quote(str(value))}")

emit("REMOTE_CONFIG_PATH", path)
emit("REMOTE_SSH_HOST", remote.get("ssh_host", gpu_server.get("ssh_host", "A600")))
emit(
    "REMOTE_REPO_ROOT",
    remote.get(
        "gpu_repo_root",
        remote.get("remote_repo_root", gpu_server.get("repo_root", "/data/yangjingwen/code/SoftFold")),
    ),
)
emit("REMOTE_USE_SSH_TUNNEL", str(remote.get("use_ssh_tunnel", True)).lower())
emit("REMOTE_TUNNEL_LOCAL_PORT", remote.get("tunnel_local_port", 8080))
emit("REMOTE_TUNNEL_REMOTE_HOST", remote.get("tunnel_remote_host", "127.0.0.1"))
emit("REMOTE_TUNNEL_REMOTE_PORT", remote.get("tunnel_remote_port", policy_server.get("port", 8080)))
emit("REMOTE_POLICY_SERVER_PORT", policy_server.get("port", 8080))
emit("ASYNC_SERVER_ADDRESS", async_cfg.get("server_address", "127.0.0.1:8080"))
emit("REMOTE_CONDA_ENV", remote.get("conda_env", "piper"))
PY
  )"
}
