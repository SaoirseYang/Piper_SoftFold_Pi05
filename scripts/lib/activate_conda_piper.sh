#!/usr/bin/env bash
# shellcheck shell=bash
# Activate the conda env used for training / policy server.
# Safe to source when the env is already active.

activate_conda_piper() {
  local env_name="${PIPER_CONDA_ENV:-piper}"
  if command -v conda >/dev/null 2>&1 \
      && [[ "${CONDA_DEFAULT_ENV:-}" == "${env_name}" ]]; then
    return 0
  fi

  local candidates=(
    "${CONDA_SH:-}"
    "${HOME}/miniconda3/etc/profile.d/conda.sh"
    "${HOME}/anaconda3/etc/profile.d/conda.sh"
  )
  local sh
  for sh in "${candidates[@]}"; do
    [[ -n "${sh}" && -f "${sh}" ]] || continue
    # shellcheck disable=SC1090
    source "${sh}"
    conda activate "${env_name}"
    return 0
  done

  echo "warning: conda env '${env_name}' not activated; using $(command -v python || echo python)" >&2
}
