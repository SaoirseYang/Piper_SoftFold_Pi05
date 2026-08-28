#!/usr/bin/env bash
# 运行在：工控机
# 作用：一键部署多个 pi05 Policy Server（A600 tmux）+ 本地 SSH tunnel，再交互选择跑 client。
# 不替代旧的三终端流程；旧方式仍可用 start_policy_server_pi05_remote / ssh_tunnel / client。
#
# 用法（推荐 YAML）：
#   bash scripts/deploy_async_inference.sh up --yaml configs/deploy_async_pi05_rgrasp.yaml
# 用法（CLI 等价）：
#   bash scripts/deploy_async_inference.sh up \
#     --config configs/softfold_piper_pi05_rgrasp.json \
#     --ckpt last,050000 --steps 5,10 --chunk 0.5 --alpha 0.2,0.3
#   bash scripts/deploy_async_inference.sh list|select|stop|stop-all|status [run_id]
#
# 环境变量：
#   MAX_INSTANCES=4          笛卡尔积上限
#   GPU_MIN_FREE_MIB=8000    每个实例启动前检查
#   GPU_CHECK / DISK_CHECK   true|false
#   CUDA_VISIBLE_DEVICES     传给远端 server（同卡多实例需够显存）
#   SKIP_RESOURCE_CHECK=true 跳过磁盘/GPU 检查
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNS_ROOT="${REPO_ROOT}/runs/async_deploy"
MAX_INSTANCES="${MAX_INSTANCES:-4}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-8000}"

# shellcheck source=lib/remote_gpu_config.sh
source "${SCRIPT_DIR}/lib/remote_gpu_config.sh"
# shellcheck source=lib/remote_resource_check.sh
source "${SCRIPT_DIR}/lib/remote_resource_check.sh"

usage() {
  sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

latest_run_id() {
  if [[ ! -d "${RUNS_ROOT}" ]]; then
    return 1
  fi
  ls -1dt "${RUNS_ROOT}"/*/ 2>/dev/null | head -n1 | xargs -I{} basename {}
}

resolve_run_dir() {
  local run_id="${1:-}"
  if [[ -z "${run_id}" ]]; then
    run_id="$(latest_run_id)" || {
      echo "No deploy runs under ${RUNS_ROOT}" >&2
      exit 1
    }
  fi
  local dir="${RUNS_ROOT}/${run_id}"
  if [[ ! -f "${dir}/manifest.json" ]]; then
    echo "manifest not found: ${dir}/manifest.json" >&2
    exit 1
  fi
  echo "${dir}"
}

cmd_up() {
  local config_path=""
  local deploy_yaml=""
  local ckpts="last"
  local steps=""
  local chunk=""
  local alpha=""
  local base_port=0
  local run_id=""
  local sync_code="false"
  local cuda_override=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --yaml|-f|--deploy-yaml) deploy_yaml="$2"; shift 2 ;;
      --config) config_path="$2"; shift 2 ;;
      --ckpt|--ckpts) ckpts="$2"; shift 2 ;;
      --steps) steps="$2"; shift 2 ;;
      --chunk) chunk="$2"; shift 2 ;;
      --alpha) alpha="$2"; shift 2 ;;
      --base-port) base_port="$2"; shift 2 ;;
      --run-id) run_id="$2"; shift 2 ;;
      --sync-code) sync_code="true"; shift ;;
      -h|--help) usage 0 ;;
      *) echo "Unknown arg: $1" >&2; usage 1 ;;
    esac
  done

  if [[ -n "${deploy_yaml}" ]]; then
    if [[ ! -f "${deploy_yaml}" && -f "${REPO_ROOT}/${deploy_yaml}" ]]; then
      deploy_yaml="${REPO_ROOT}/${deploy_yaml}"
    fi
    if [[ ! -f "${deploy_yaml}" ]]; then
      echo "Deploy YAML not found: ${deploy_yaml}" >&2
      exit 1
    fi
    if [[ -z "${config_path}" ]]; then
      config_path="$(
        python3 - <<PY
import yaml
from pathlib import Path
d = yaml.safe_load(Path("${deploy_yaml}").read_text(encoding="utf-8")) or {}
print(d.get("base_config") or "")
PY
      )"
    fi
    cuda_override="$(
      python3 - <<PY
import yaml
from pathlib import Path
d = yaml.safe_load(Path("${deploy_yaml}").read_text(encoding="utf-8")) or {}
print(d.get("cuda_visible_devices") or "")
PY
    )"
  fi

  if [[ -z "${config_path}" ]]; then
    config_path="configs/softfold_piper_pi05_rgrasp.json"
  fi

  remote_gpu_load_config "${config_path}" "${REPO_ROOT}"
  if [[ -n "${cuda_override}" ]]; then
    export CUDA_VISIBLE_DEVICES="${cuda_override}"
  fi

  if [[ -z "${run_id}" ]]; then
    run_id="$(date +%Y%m%d_%H%M%S)"
  fi
  local out_dir="${RUNS_ROOT}/${run_id}"
  mkdir -p "${out_dir}/pids" "${out_dir}/cfgs"

  echo "=== resource status (${REMOTE_SSH_HOST}) ==="
  if [[ "${SKIP_RESOURCE_CHECK:-false}" != "true" ]]; then
    remote_resource_status
    echo
    # Deploy configs are small; enforce free-space reserve only.
    remote_resource_require_disk "${REMOTE_REPO_ROOT}" 0
  fi

  local gen_args=(
    --base-config "${REMOTE_CONFIG_PATH}"
    --out-dir "${out_dir}"
    --run-id "${run_id}"
    --max-instances "${MAX_INSTANCES}"
  )
  if [[ -n "${deploy_yaml}" ]]; then
    echo "Using deploy YAML: ${deploy_yaml}"
    gen_args+=(--deploy-yaml "${deploy_yaml}")
  else
    gen_args+=(--ckpts "${ckpts}")
    [[ -n "${steps}" ]] && gen_args+=(--steps "${steps}")
    [[ -n "${chunk}" ]] && gen_args+=(--chunk "${chunk}")
    [[ -n "${alpha}" ]] && gen_args+=(--alpha "${alpha}")
  fi
  [[ "${base_port}" -gt 0 ]] && gen_args+=(--base-port "${base_port}")

  python3 "${SCRIPT_DIR}/lib/gen_deploy_configs.py" "${gen_args[@]}"
  echo "manifest: ${out_dir}/manifest.json"

  # Preflight: every policy_path must exist on A600 before starting servers.
  echo
  echo "=== preflight: verify checkpoints on ${REMOTE_SSH_HOST} ==="
  if ! python3 "${SCRIPT_DIR}/lib/deploy_status.py" \
      --run-dir "${out_dir}" \
      --ssh-host "${REMOTE_SSH_HOST}" \
      --repo-root "${REMOTE_REPO_ROOT}" \
      --preflight; then
    exit 1
  fi

  if [[ "${sync_code}" == "true" ]]; then
    bash "${SCRIPT_DIR}/sync_code_to_remote.sh" "${config_path}"
  fi

  # Sync derived configs to the same relative path under remote runs/
  local remote_run_dir="${REMOTE_REPO_ROOT}/runs/async_deploy/${run_id}"
  echo "rsync deploy cfgs -> ${REMOTE_SSH_HOST}:${remote_run_dir}/"
  ssh "${REMOTE_SSH_HOST}" "mkdir -p $(printf '%q' "${remote_run_dir}")"
  rsync -aH "${out_dir}/cfgs/" "${REMOTE_SSH_HOST}:${remote_run_dir}/cfgs/"
  rsync -aH "${out_dir}/manifest.json" "${REMOTE_SSH_HOST}:${remote_run_dir}/manifest.json"

  local n
  n="$(python3 -c "import json;print(len(json.load(open('${out_dir}/manifest.json'))['instances']))")"
  echo
  echo "Starting ${n} remote policy server(s) + local tunnel(s)..."

  local i=0
  while (( i < n )); do
    eval "$(
      python3 - <<PY
import json
inst = json.load(open("${out_dir}/manifest.json"))["instances"][${i}]
for k, v in inst.items():
    if isinstance(v, (int, float)):
        print(f'export I_{k.upper()}={v}')
    else:
        import shlex
        print(f'export I_{k.upper()}={shlex.quote(str(v))}')
PY
    )"

    if [[ "${SKIP_RESOURCE_CHECK:-false}" != "true" ]]; then
      GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB}" remote_resource_require_gpu "${CUDA_VISIBLE_DEVICES:-0}" \
        || {
          echo "Stopping further starts after GPU check failure (already started 0..$((i - 1)))." >&2
          echo "Partial run kept at ${out_dir}. Use: bash scripts/deploy_async_inference.sh stop ${run_id}" >&2
          exit 1
        }
    fi

    local remote_cfg="${remote_run_dir}/${I_CONFIG_REL}"
    local remote_log="${remote_run_dir}/server_${I_ID}.log"
    local srv_cmd="cd $(printf '%q' "${REMOTE_REPO_ROOT}") && CUDA_VISIBLE_DEVICES=$(printf '%q' "${CUDA_VISIBLE_DEVICES:-0}") PIPER_CONDA_ENV=$(printf '%q' "${REMOTE_CONDA_ENV}") bash scripts/start_policy_server_pi05.sh $(printf '%q' "${remote_cfg}")"

    echo
    echo "[${I_ID}] server tmux=${I_TMUX_SESSION} port=${I_PORT} ckpt=${I_POLICY_PATH}"
    ssh "${REMOTE_SSH_HOST}" bash -s <<EOF
set -euo pipefail
mkdir -p $(printf '%q' "${remote_run_dir}")
if tmux has-session -t $(printf '%q' "${I_TMUX_SESSION}") 2>/dev/null; then
  echo "  tmux already exists: ${I_TMUX_SESSION} (reuse)"
else
  tmux new-session -d -s $(printf '%q' "${I_TMUX_SESSION}") \
    "bash -lc $(printf '%q' "${srv_cmd} 2>&1 | tee -a ${remote_log}; echo [server exited \\\$?]; sleep 2")"
  echo "  started tmux ${I_TMUX_SESSION}"
  echo "  log ${remote_log}"
fi
EOF

    # Local tunnel (background). Kill existing if pid file present.
    local pid_file="${out_dir}/${I_TUNNEL_PID_FILE}"
    mkdir -p "$(dirname "${pid_file}")"
    if [[ -f "${pid_file}" ]]; then
      local old_pid
      old_pid="$(cat "${pid_file}" || true)"
      if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
        kill "${old_pid}" 2>/dev/null || true
        sleep 0.3
      fi
      rm -f "${pid_file}"
    fi
    # Free local port if something else holds it
    if command -v fuser >/dev/null 2>&1; then
      fuser -k "${I_PORT}/tcp" 2>/dev/null || true
    fi
    ssh -fN \
      -o ExitOnForwardFailure=yes \
      -L "${I_PORT}:127.0.0.1:${I_PORT}" \
      "${REMOTE_SSH_HOST}"
    # Find the ssh tunnel pid (best-effort)
    local tun_pid
    tun_pid="$(pgrep -f "ssh -fN .*${I_PORT}:127.0.0.1:${I_PORT}.*${REMOTE_SSH_HOST}" | tail -n1 || true)"
    if [[ -n "${tun_pid}" ]]; then
      echo "${tun_pid}" >"${pid_file}"
      echo "  tunnel pid=${tun_pid}  127.0.0.1:${I_PORT} -> ${REMOTE_SSH_HOST}:${I_PORT}"
    else
      echo "  WARN: tunnel started but pid not found; check manually" >&2
    fi

    i=$((i + 1))
  done

  echo
  echo "Deploy ready: run_id=${run_id}"
  echo "  list:   bash scripts/deploy_async_inference.sh list ${run_id}"
  echo "  select: bash scripts/deploy_async_inference.sh select ${run_id}"
  echo "  stop:   bash scripts/deploy_async_inference.sh stop ${run_id}"
  echo
  echo "Legacy single-instance flow still available:"
  echo "  bash scripts/start_policy_server_pi05_remote.sh ${config_path}"
  echo "  bash scripts/ssh_tunnel_policy_server.sh ${config_path}"
  echo "  bash scripts/run_async_policy_client_pi05_remote.sh ${config_path}"
}

print_table() {
  local run_dir="$1"
  remote_gpu_load_config "$(python3 -c "import json;print(json.load(open('${run_dir}/manifest.json'))['base_config'])")" "${REPO_ROOT}"
  python3 "${SCRIPT_DIR}/lib/deploy_status.py" \
    --run-dir "${run_dir}" \
    --ssh-host "${REMOTE_SSH_HOST}" \
    --repo-root "${REMOTE_REPO_ROOT}"
}

cmd_list() {
  local run_dir
  run_dir="$(resolve_run_dir "${1:-}")"
  print_table "${run_dir}"
}

cmd_status() {
  local run_dir
  run_dir="$(resolve_run_dir "${1:-}")"
  remote_gpu_load_config "$(python3 -c "import json;print(json.load(open('${run_dir}/manifest.json'))['base_config'])")" "${REPO_ROOT}"
  print_table "${run_dir}"
  echo
  remote_resource_status
}

cmd_select() {
  local run_dir
  run_dir="$(resolve_run_dir "${1:-}")"
  remote_gpu_load_config "$(python3 -c "import json;print(json.load(open('${run_dir}/manifest.json'))['base_config'])")" "${REPO_ROOT}"
  print_table "${run_dir}"
  echo
  local choice
  if [[ -n "${SELECT_ID:-}" ]]; then
    choice="${SELECT_ID}"
  else
    read -r -p "Select instance id (or q): " choice
  fi
  if [[ "${choice}" == "q" || "${choice}" == "Q" ]]; then
    exit 0
  fi

  # Refuse missing ckpt / dead server before starting client
  if ! python3 "${SCRIPT_DIR}/lib/deploy_status.py" \
      --run-dir "${run_dir}" \
      --ssh-host "${REMOTE_SSH_HOST}" \
      --repo-root "${REMOTE_REPO_ROOT}" \
      --check-id "${choice}"; then
    echo >&2
    echo "未启动 client。请换一个 status=READY 的 id，或先修 YAML/重新 deploy。" >&2
    echo "  远端可对照: ssh ${REMOTE_SSH_HOST} 'ls ${REMOTE_REPO_ROOT}/outputs/train/*/checkpoints'" >&2
    exit 1
  fi

  eval "$(
    python3 - <<PY
import json, shlex
m = json.load(open("${run_dir}/manifest.json"))
inst = None
for x in m["instances"]:
    if int(x["id"]) == int("${choice}"):
        inst = x
        break
if inst is None:
    raise SystemExit(f"unknown id: ${choice}")
print(f"export SEL_CONFIG={shlex.quote(inst['config_path'])}")
print(f"export SEL_NAME={shlex.quote(inst['name'])}")
print(f"export SEL_PORT={inst['port']}")
print(f"export SEL_POLICY={shlex.quote(inst['policy_path'])}")
print(f"export SEL_PID_FILE={shlex.quote('${run_dir}/' + inst['tunnel_pid_file'])}")
PY
  )"

  if [[ -f "${SEL_PID_FILE}" ]]; then
    local pid
    pid="$(cat "${SEL_PID_FILE}")"
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "Tunnel dead; recreating for port ${SEL_PORT}..."
      remote_gpu_load_config "${SEL_CONFIG}" "${REPO_ROOT}"
      ssh -fN -o ExitOnForwardFailure=yes \
        -L "${SEL_PORT}:127.0.0.1:${SEL_PORT}" \
        "${REMOTE_SSH_HOST}"
      pgrep -f "ssh -fN .*${SEL_PORT}:127.0.0.1:${SEL_PORT}.*${REMOTE_SSH_HOST}" | tail -n1 >"${SEL_PID_FILE}" || true
    fi
  fi

  echo "Running client: ${SEL_NAME}"
  echo "  config: ${SEL_CONFIG}"
  echo "  policy: ${SEL_POLICY}"
  local cfg_arg="${SEL_CONFIG}"
  if [[ "${SEL_CONFIG}" == "${REPO_ROOT}/"* ]]; then
    cfg_arg="${SEL_CONFIG#"${REPO_ROOT}/"}"
  fi
  exec bash "${SCRIPT_DIR}/run_async_policy_client_pi05_remote.sh" "${cfg_arg}"
}

cmd_stop() {
  local run_dir
  run_dir="$(resolve_run_dir "${1:-}")"
  remote_gpu_load_config "$(python3 -c "import json;print(json.load(open('${run_dir}/manifest.json'))['base_config'])")" "${REPO_ROOT}"

  echo "Stopping deploy $(basename "${run_dir}") ..."
  python3 - <<PY | while read -r sess; do
import json
from pathlib import Path
m = json.loads(Path("${run_dir}/manifest.json").read_text())
for inst in m["instances"]:
    print(inst["tmux_session"])
PY
    [[ -z "${sess}" ]] && continue
    ssh "${REMOTE_SSH_HOST}" "tmux kill-session -t $(printf '%q' "${sess}") 2>/dev/null || true"
    echo "  killed remote tmux ${sess}"
  done

  local pid_file
  for pid_file in "${run_dir}"/pids/tunnel_*.pid; do
    [[ -f "${pid_file}" ]] || continue
    local pid
    pid="$(cat "${pid_file}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      echo "  killed local tunnel pid=${pid}"
    fi
    rm -f "${pid_file}"
  done
  echo "Done."
}

cmd_stop_all() {
  local config_path="${1:-configs/softfold_piper_pi05_rgrasp.json}"
  if [[ $# -gt 0 ]]; then
    shift
  fi
  remote_gpu_load_config "${config_path}" "${REPO_ROOT}"

  echo "Stopping ALL deploy instances on ${REMOTE_SSH_HOST} (tmux sf-srv-*) ..."
  ssh "${REMOTE_SSH_HOST}" bash -s <<'EOF'
set +e
count=0
while IFS= read -r sess; do
  [[ -z "${sess}" ]] && continue
  tmux kill-session -t "${sess}" 2>/dev/null && echo "  killed remote tmux ${sess}" && count=$((count + 1))
done < <(tmux ls -F '#{session_name}' 2>/dev/null | grep '^sf-srv-' || true)
if [[ "${count}" -eq 0 ]]; then
  echo "  (no sf-srv-* tmux sessions on remote)"
fi
EOF

  echo
  echo "Stopping all local deploy SSH tunnels under ${RUNS_ROOT} ..."
  local killed_local=0
  if [[ -d "${RUNS_ROOT}" ]]; then
    local pid_file
    while IFS= read -r -d '' pid_file; do
      [[ -f "${pid_file}" ]] || continue
      local pid
      pid="$(cat "${pid_file}" || true)"
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        kill "${pid}" 2>/dev/null || true
        echo "  killed local tunnel pid=${pid} ($(basename "${pid_file}"))"
        killed_local=$((killed_local + 1))
      fi
      rm -f "${pid_file}"
    done < <(find "${RUNS_ROOT}" -path '*/pids/tunnel_*.pid' -print0 2>/dev/null || true)
  fi
  if [[ "${killed_local}" -eq 0 ]]; then
    echo "  (no local tunnel pid files)"
  fi
  echo "Done. All deploy servers/tunnels cleared."
  echo "  single run: bash scripts/deploy_async_inference.sh stop [run_id]"
}

main() {
  local cmd="${1:-}"
  if [[ -z "${cmd}" ]]; then
    usage 1
  fi
  shift || true
  case "${cmd}" in
    up) cmd_up "$@" ;;
    list) cmd_list "$@" ;;
    select) cmd_select "$@" ;;
    stop) cmd_stop "$@" ;;
    stop-all|stopall) cmd_stop_all "$@" ;;
    status) cmd_status "$@" ;;
    -h|--help|help) usage 0 ;;
    *) echo "Unknown command: ${cmd}" >&2; usage 1 ;;
  esac
}

main "$@"
