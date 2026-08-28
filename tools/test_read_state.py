import argparse
import math
import time

from piper_towel_fold.config import PiperRobotConfig
from piper_towel_fold.piper import PiperRobot


def format_arm_state(obs: dict, side: str) -> str:
    joint_values = []
    for index in range(1, 7):
        radians_value = obs[f"{side}_joint_{index}.pos"]
        degrees_value = math.degrees(radians_value)
        joint_values.append(f"j{index}={degrees_value:8.3f} deg")

    gripper_value = obs[f"{side}_gripper.pos"]
    return f"{side}: " + ", ".join(joint_values) + f", gripper={gripper_value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Read Piper dual-arm joint states once per second.")
    parser.add_argument("--left-can", required=True, help="Left follower CAN interface, e.g. can0")
    parser.add_argument("--right-can", required=True, help="Right follower CAN interface, e.g. can1")
    parser.add_argument("--period", type=float, default=1.0, help="Print period in seconds")
    args = parser.parse_args()

    config = PiperRobotConfig(
        follower_left_port=args.left_can,
        follower_right_port=args.right_can,
        cameras={},
    )
    robot = PiperRobot(config)

    try:
        robot.connect()
        print("Connected. Press Ctrl+C to stop.")
        while True:
            obs = robot.get_observation()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(timestamp)
            print(format_arm_state(obs, "left"))
            print(format_arm_state(obs, "right"))
            print()
            time.sleep(args.period)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
