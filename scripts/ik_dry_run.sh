#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${SRC:-${ROOT}/../piper/data/lerobot/xvla-soft-fold}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python -u data/eef_to_joint_converter.py \
  --src-root "${SRC}" \
  --out-root /tmp/_ik_dry_unused \
  --dry-run \
  --max-episodes "${MAX_EPISODES:-10}" \
  "$@"
