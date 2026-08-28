#!/usr/bin/env bash
set -euo pipefail
# Smoke: convert first N Soft-Fold episodes → Piper joint virtual set
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${SRC:-${ROOT}/../piper/data/lerobot/xvla-soft-fold}"
OUT="${OUT:-${ROOT}/data/lerobot/piper-softfold-virtual-smoke}"
MAX="${MAX_EPISODES:-5}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

python data/eef_to_joint_converter.py \
  --src-root "${SRC}" \
  --out-root "${OUT}" \
  --repo-id piper-softfold-virtual-smoke \
  --max-episodes "${MAX}" \
  --mask-mode band \
  --smooth-window 5 \
  --frame-stride "${FRAME_STRIDE:-2}" \
  --orient-mode seed \
  --min-ik-success-ratio 0.7 \
  --overwrite \
  "$@"
