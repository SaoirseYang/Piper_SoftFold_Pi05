import argparse
import time
from typing import Any


def import_piper_interface() -> type:
    try:
        from piper_sdk import C_PiperInterface_V2

        return C_PiperInterface_V2
    except ImportError:
        from piper_sdk import C_PiperInterface

        return C_PiperInterface


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def call_connect_port(arm: Any, piper_init: bool) -> None:
    connect_port = getattr(arm, "ConnectPort")
    try:
        connect_port(piper_init=piper_init)
    except TypeError:
        connect_port()


def call_optional(arm: Any, method_name: str, *args: Any) -> bool:
    method = getattr(arm, method_name, None)
    if not callable(method):
        return False

    try:
        method(*args)
    except TypeError:
        method()
    return True


def call_motion_reset(arm: Any) -> None:
    motion_ctrl_1 = getattr(arm, "MotionCtrl_1", None)
    if callable(motion_ctrl_1):
        motion_ctrl_1(0x02, 0, 0)
        return

    reset_piper = getattr(arm, "ResetPiper", None)
    if callable(reset_piper):
        reset_piper()
        return

    raise RuntimeError(
        "This piper_sdk version exposes neither MotionCtrl_1() nor ResetPiper(). "
        "Please update piper_sdk or run the official demo piper_ctrl_reset.py."
    )


def call_enable_arm(arm: Any) -> bool:
    enable_arm = getattr(arm, "EnableArm", None)
    if not callable(enable_arm):
        return False

    try:
        enable_arm(7)
    except TypeError:
        enable_arm()
    return True


def call_sdk_mode(arm: Any, control_speed: int) -> bool:
    motion_ctrl_2 = getattr(arm, "MotionCtrl_2", None)
    if not callable(motion_ctrl_2):
        return False

    try:
        motion_ctrl_2(0x01, 0x01, int(control_speed), 0x00)
    except TypeError:
        motion_ctrl_2(0x01, 0x01, int(control_speed))
    return True


def reset_arm(interface_cls: type, can_name: str, args: argparse.Namespace) -> None:
    print(f"[{can_name}] connecting")
    try:
        arm = interface_cls(can_name=can_name, judge_flag=args.judge_flag)
    except TypeError:
        arm = interface_cls(can_name=can_name)
    call_connect_port(arm, piper_init=args.piper_init)
    time.sleep(args.connect_wait)

    for attempt in range(1, args.repeat + 1):
        print(f"[{can_name}] sending MotionCtrl_1(0x02, 0, 0) reset ({attempt}/{args.repeat})")
        call_motion_reset(arm)
        time.sleep(args.command_wait)

    time.sleep(args.post_reset_wait)

    if args.resume_emergency_stop:
        print(f"[{can_name}] sending EmergencyStop resume")
        if not call_optional(arm, "EmergencyStop", 0x02):
            print(f"[{can_name}] EmergencyStop is not available in this piper_sdk version")
        time.sleep(args.command_wait)

    if not args.skip_sdk_recover:
        print(f"[{can_name}] re-enabling arm for SDK control")
        if not call_enable_arm(arm):
            print(f"[{can_name}] EnableArm is not available in this piper_sdk version")
        time.sleep(args.command_wait)

        print(f"[{can_name}] switching back to SDK control mode")
        if not call_sdk_mode(arm, args.control_speed):
            print(f"[{can_name}] MotionCtrl_2 is not available in this piper_sdk version")
        time.sleep(args.command_wait)

    status = getattr(arm, "GetArmStatus", None)
    if callable(status):
        print(f"[{can_name}] status: {status()}")

    disconnect = getattr(arm, "DisconnectPort", None)
    if callable(disconnect):
        disconnect()

    print(f"[{can_name}] reset command attempted; check piper_sdk logs above for send failures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reset Piper arms after using teach/drag mode so SDK control can be enabled again."
    )
    parser.add_argument(
        "--can",
        default="can2,can0",
        help="Comma-separated CAN interfaces to reset, e.g. can2,can0.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Require typing RESET before running the reset sequence.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="How many times to send MotionCtrl_1(0x02, 0, 0).",
    )
    parser.add_argument(
        "--piper-init",
        action="store_true",
        help="Let ConnectPort run PiperInit. Default is off to avoid limit-query failures after teach mode.",
    )
    parser.add_argument(
        "--judge-flag",
        action="store_true",
        help="Enable piper_sdk CAN-port judging in the interface constructor.",
    )
    parser.add_argument(
        "--resume-emergency-stop",
        action="store_true",
        help="Send EmergencyStop(0x02) after reset if available.",
    )
    parser.add_argument(
        "--skip-sdk-recover",
        action="store_true",
        help="Only send reset; do not re-enable the arm or switch back to SDK control mode.",
    )
    parser.add_argument(
        "--control-speed",
        type=int,
        default=15,
        help="Control speed passed to MotionCtrl_2 when restoring SDK control.",
    )
    parser.add_argument("--connect-wait", type=float, default=0.5)
    parser.add_argument("--command-wait", type=float, default=1.0)
    parser.add_argument(
        "--post-reset-wait",
        type=float,
        default=1.5,
        help="Extra wait after reset before re-enabling SDK control.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    can_names = parse_csv(args.can)
    if not can_names:
        raise ValueError("--can must contain at least one CAN interface.")
    if args.repeat <= 0:
        raise ValueError("--repeat must be greater than 0.")

    print("WARNING: Piper reset will make the arm lose power immediately.")
    print("Make sure the arm is supported, clear of people/objects, and you are ready to re-enable it afterward.")
    print(f"CAN interfaces: {', '.join(can_names)}")
    if args.confirm:
        answer = input("Type RESET to continue: ").strip()
        if answer != "RESET":
            print("Canceled.")
            return

    interface_cls = import_piper_interface()
    for can_name in can_names:
        reset_arm(interface_cls, can_name, args)

    print("Done. If piper_sdk logged SEND_MESSAGE_FAILED, the reset command did not reach the arm.")
    if args.skip_sdk_recover:
        print("SDK recover was skipped, so you still need to enable the arm and switch mode manually.")
    else:
        print("Reset plus SDK-mode recovery was sent. If the arm is still in teach mode, try increasing --post-reset-wait.")


if __name__ == "__main__":
    main()
