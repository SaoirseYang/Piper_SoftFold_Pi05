#!/usr/bin/env bash
# shellcheck shell=bash
# A600 disk / GPU probes. Requires remote_gpu_load_config already sourced.

remote_resource_print_disk() {
  local path="${1:-${REMOTE_REPO_ROOT}}"
  echo "[remote disk] ${REMOTE_SSH_HOST}:${path}"
  ssh "${REMOTE_SSH_HOST}" "df -h $(printf '%q' "${path}") | tail -n +1" || true
}

remote_resource_print_gpu() {
  echo "[remote gpu] ${REMOTE_SSH_HOST}"
  ssh "${REMOTE_SSH_HOST}" \
    "nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader" \
    || echo "  (nvidia-smi unavailable)" >&2
}

# Estimate bytes needed locally (dataset + optional ojag). Returns size via stdout.
remote_resource_local_dataset_bytes() {
  local local_dataset="$1"
  local total=0
  if [[ -d "${local_dataset}" ]]; then
    total=$((total + $(du -sb "${local_dataset}" | awk '{print $1}')))
  fi
  if [[ -d "${local_dataset}_ojag" ]]; then
    total=$((total + $(du -sb "${local_dataset}_ojag" | awk '{print $1}')))
  fi
  echo "${total}"
}

# Fail if remote free space on path is below need_bytes * margin (default 1.2).
# Env: DISK_CHECK=true|false  DISK_MARGIN=1.2  DISK_MIN_FREE_GIB=20
remote_resource_require_disk() {
  local path="$1"
  local need_bytes="${2:-0}"
  local margin="${DISK_MARGIN:-1.2}"
  local min_free_gib="${DISK_MIN_FREE_GIB:-20}"

  if [[ "${DISK_CHECK:-true}" != "true" ]]; then
    echo "[disk check] skipped (DISK_CHECK=${DISK_CHECK:-})"
    return 0
  fi

  local free_kb
  free_kb="$(ssh "${REMOTE_SSH_HOST}" "df -Pk $(printf '%q' "${path}") | awk 'NR==2{print \$4}'")"
  if [[ -z "${free_kb}" || ! "${free_kb}" =~ ^[0-9]+$ ]]; then
    echo "[disk check] WARN: could not read free space for ${path}; continuing" >&2
    return 0
  fi
  local free_bytes=$((free_kb * 1024))
  local min_free_bytes
  min_free_bytes="$(python3 -c "print(int(float('${min_free_gib}') * 1024**3))")"
  local need_with_margin
  need_with_margin="$(python3 -c "print(int(float('${need_bytes}') * float('${margin}')))")"

  python3 - <<PY
need = int("${need_with_margin}")
free = int("${free_bytes}")
min_free = int("${min_free_bytes}")
print(f"[disk check] free={free/1024**3:.1f} GiB  need(with margin)={need/1024**3:.1f} GiB  min_reserve={min_free/1024**3:.1f} GiB")
if free < min_free:
    raise SystemExit(f"ERROR: remote free space {free/1024**3:.1f} GiB < DISK_MIN_FREE_GIB={min_free/1024**3:.1f} GiB")
if need > 0 and free < need:
    raise SystemExit(f"ERROR: remote free space {free/1024**3:.1f} GiB < required {need/1024**3:.1f} GiB")
PY
}

# Fail if GPU index has less free MiB than required.
# Env: GPU_CHECK=true|false  GPU_MIN_FREE_MIB=8000  CUDA_VISIBLE_DEVICES / GPU_INDEX
remote_resource_require_gpu() {
  local gpu_index="${1:-${GPU_INDEX:-${CUDA_VISIBLE_DEVICES:-0}}}"
  local min_free_mib="${GPU_MIN_FREE_MIB:-8000}"

  # If CUDA_VISIBLE_DEVICES is a list, take first.
  gpu_index="${gpu_index%%,*}"

  if [[ "${GPU_CHECK:-true}" != "true" ]]; then
    echo "[gpu check] skipped (GPU_CHECK=${GPU_CHECK:-})"
    return 0
  fi

  local free_mib
  free_mib="$(ssh "${REMOTE_SSH_HOST}" \
    "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $(printf '%q' "${gpu_index}")" \
    | tr -d ' ' | head -n1)" || true

  if [[ -z "${free_mib}" || ! "${free_mib}" =~ ^[0-9]+$ ]]; then
    echo "[gpu check] WARN: could not read GPU ${gpu_index} free memory; continuing" >&2
    return 0
  fi

  echo "[gpu check] GPU ${gpu_index} free=${free_mib} MiB  required>=${min_free_mib} MiB"
  if (( free_mib < min_free_mib )); then
    echo "ERROR: GPU ${gpu_index} free ${free_mib} MiB < GPU_MIN_FREE_MIB=${min_free_mib}" >&2
    remote_resource_print_gpu
    return 1
  fi
}

remote_resource_status() {
  remote_resource_print_disk "${REMOTE_REPO_ROOT}"
  echo
  remote_resource_print_gpu
}
