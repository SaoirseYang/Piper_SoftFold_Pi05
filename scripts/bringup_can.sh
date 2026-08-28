#!/usr/bin/env bash

set -euo pipefail

LEFT_CAN="${LEFT_CAN:-can2}"
RIGHT_CAN="${RIGHT_CAN:-can0}"
BITRATE="${BITRATE:-1000000}"

bringup_can() {
    local iface="$1"

    echo "Configuring ${iface} with bitrate ${BITRATE}..."
    sudo ip link set "${iface}" down || true
    sudo ip link set "${iface}" type can bitrate "${BITRATE}"
    sudo ip link set "${iface}" up
    ip -details link show "${iface}"
    echo
}

echo "Bringing up Piper follower CAN interfaces..."
echo "  left  -> ${LEFT_CAN}"
echo "  right -> ${RIGHT_CAN}"
echo

bringup_can "${LEFT_CAN}"
bringup_can "${RIGHT_CAN}"

echo "CAN bring-up complete."
