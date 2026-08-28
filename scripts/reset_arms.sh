#!/usr/bin/env bash
set -euo pipefail

LEFT_CAN="${LEFT_CAN:-can2}"
RIGHT_CAN="${RIGHT_CAN:-can0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "Resetting Piper arms to exit teach/drag mode..."
echo "  left  -> ${LEFT_CAN}"
echo "  right -> ${RIGHT_CAN}"
echo

python -m piper_towel_fold.reset --can "${LEFT_CAN},${RIGHT_CAN}" "$@"
