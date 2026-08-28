#!/usr/bin/env bash

set -euo pipefail

LEFT_CAN="${LEFT_CAN:-can2}"
RIGHT_CAN="${RIGHT_CAN:-can0}"
PERIOD="${PERIOD:-1.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "Running joint-state reader..."
echo "  left  -> ${LEFT_CAN}"
echo "  right -> ${RIGHT_CAN}"
echo "  period -> ${PERIOD}s"
echo

python tools/test_read_state.py \
    --left-can "${LEFT_CAN}" \
    --right-can "${RIGHT_CAN}" \
    --period "${PERIOD}"
