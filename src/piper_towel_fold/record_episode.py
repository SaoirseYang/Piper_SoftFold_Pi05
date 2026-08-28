import argparse
import json
import signal
import time
from pathlib import Path

from lerobot.cameras.opencv import OpenCVCameraConfig

from .config import PiperRobotConfig
from .piper import PiperRobot
from .preprocessing import (
    FramePreprocessor,
    load_preprocessing_config,
    resolve_augmented_repo_id,
)
from .recorder import DualLeRobotEpisodeRecorder, LeRobotEpisodeRecorder, PiperEpisodeRecorder


def resolve_action_source(args: argparse.Namespace) -> str:
    if args.action_source != "auto":
        return args.action_source

    if args.leader_left_can and args.leader_right_can:
        return "leader"
    return "follower"


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_camera_ref(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def make_camera_configs(args: argparse.Namespace) -> dict[str, OpenCVCameraConfig]:
    camera_refs = parse_csv(args.camera_indices)
    if not camera_refs:
        return {}

    camera_names = parse_csv(args.camera_names)
    if camera_names and len(camera_names) != len(camera_refs):
        raise ValueError("--camera-names must have the same number of items as --camera-indices.")

    if not camera_names:
        default_names = ["cam_top", "cam_left", "cam_right"]
        camera_names = default_names[: len(camera_refs)]
        if len(camera_names) < len(camera_refs):
            camera_names.extend(f"cam_{index}" for index in range(len(camera_names), len(camera_refs)))

    return {
        camera_name: OpenCVCameraConfig(
            index_or_path=parse_camera_ref(camera_ref),
            fps=args.camera_fps,
            width=args.camera_width,
            height=args.camera_height,
        )
        for camera_name, camera_ref in zip(camera_names, camera_refs, strict=True)
    }


def write_episode_outcome(
    dataset_path: Path,
    task: str,
    outcome: str,
    stop_reason: str,
    episode_index: int | None = None,
) -> None:
    if outcome == "skip":
        return

    dataset_path.mkdir(parents=True, exist_ok=True)
    outcome_path = dataset_path / "episode_outcomes.jsonl"
    outcome_record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task": task,
        "outcome": outcome,
        "stop_reason": stop_reason,
    }
    if episode_index is not None:
        outcome_record["episode_index"] = episode_index
    with outcome_path.open("a", encoding="utf-8") as outcome_file:
        outcome_file.write(json.dumps(outcome_record, ensure_ascii=False) + "\n")


def prompt_episode_outcome() -> str:
    options = {
        "s": "success",
        "success": "success",
        "f": "failure",
        "fail": "failure",
        "failure": "failure",
        "u": "unknown",
        "unknown": "unknown",
        "": "unknown",
    }
    while True:
        value = input("Episode outcome [s=success, f=failure, u=unknown]: ").strip().lower()
        outcome = options.get(value)
        if outcome is not None:
            return outcome
        print("Please enter s, f, or u.")


def install_stop_handler() -> tuple[dict[str, bool], object]:
    stop_requested = {"value": False}
    previous_handler = signal.getsignal(signal.SIGINT)

    def request_stop(signum: int, frame: object) -> None:
        del signum, frame
        if stop_requested["value"]:
            print("\nStop already requested. Waiting for the current frame to finish.")
            return
        stop_requested["value"] = True
        print("\nStop requested. Finishing the current frame, then saving the episode.")

    signal.signal(signal.SIGINT, request_stop)
    return stop_requested, previous_handler


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record a minimal Piper teleop episode.")
    parser.add_argument("--task", default="pick_cube", help="Task name stored in metadata.")
    parser.add_argument(
        "--dataset-format",
        choices=("lerobot", "jsonl"),
        default="lerobot",
        help="Storage format for the recorded episode.",
    )
    parser.add_argument("--repo-id", default="local/piper_pick_cube", help="LeRobot dataset repo id.")
    parser.add_argument("--root", default="data/lerobot", help="LeRobot dataset root directory.")
    parser.add_argument("--robot-type", default="piper", help="Robot type stored in LeRobot metadata.")
    parser.add_argument("--no-videos", action="store_true", help="Store image frames instead of videos.")
    parser.add_argument(
        "--prompt-outcome",
        action="store_true",
        help="Ask whether the manually stopped episode succeeded or failed.",
    )
    parser.add_argument(
        "--episode-outcome",
        choices=("success", "failure", "unknown", "skip"),
        default="skip",
        help="Outcome to write after recording. Use --prompt-outcome for manual labeling.",
    )
    parser.add_argument("--output-dir", default="data/raw", help="Directory for JSONL debug episodes.")
    parser.add_argument("--fps", type=float, default=10.0, help="Recording frequency.")
    parser.add_argument("--duration", type=float, default=None, help="Optional duration in seconds.")
    parser.add_argument("--follower-left-can", default="can2", help="Left follower CAN interface.")
    parser.add_argument("--follower-right-can", default="can0", help="Right follower CAN interface.")
    parser.add_argument("--leader-left-can", default=None, help="Left leader CAN interface.")
    parser.add_argument("--leader-right-can", default=None, help="Right leader CAN interface.")
    parser.add_argument(
        "--camera-indices",
        default="",
        help="Comma-separated OpenCV camera indices or paths, e.g. 0,2,4.",
    )
    parser.add_argument(
        "--camera-names",
        default="",
        help="Comma-separated camera names, e.g. cam_top,cam_left,cam_right.",
    )
    parser.add_argument("--camera-width", type=int, default=640, help="Camera capture width.")
    parser.add_argument("--camera-height", type=int, default=480, help="Camera capture height.")
    parser.add_argument("--camera-fps", type=int, default=30, help="Camera capture fps.")
    parser.add_argument(
        "--image-format",
        choices=("jpg", "jpeg", "png"),
        default="jpg",
        help="Image format for saved camera frames.",
    )
    parser.add_argument("--image-quality", type=int, default=95, help="JPEG image quality.")
    parser.add_argument(
        "--action-source",
        choices=("auto", "leader", "follower"),
        default="auto",
        help="Use leader joints as actions when available, otherwise follower joints.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise ValueError("--fps must be greater than 0.")


def make_lerobot_recorder(
    args: argparse.Namespace,
    repo_id: str,
    camera_names: list[str],
    camera_shape: tuple[int, int, int],
) -> LeRobotEpisodeRecorder:
    return LeRobotEpisodeRecorder(
        root=args.root,
        repo_id=repo_id,
        task=args.task,
        fps=int(args.fps),
        camera_names=camera_names,
        camera_shape=camera_shape,
        robot_type=args.robot_type,
        use_videos=not args.no_videos,
    )


def write_episode_outcomes(
    dataset_paths: list[Path],
    task: str,
    outcome: str,
    stop_reason: str,
    dataset_format: str,
) -> None:
    for dataset_path in dataset_paths:
        episode_index: int | None = None
        if dataset_format == "lerobot":
            try:
                from .episode_outcomes import read_episode_count

                episode_index = read_episode_count(dataset_path) - 1
            except (FileNotFoundError, ValueError, json.JSONDecodeError):
                episode_index = None
        write_episode_outcome(
            dataset_path=dataset_path,
            task=task,
            outcome=outcome,
            stop_reason=stop_reason,
            episode_index=episode_index,
        )


def run_recording(args: argparse.Namespace) -> None:
    validate_args(args)

    camera_configs = make_camera_configs(args)
    action_source = resolve_action_source(args)
    if action_source == "leader":
        # 共 CAN 时可只配 follower：从同一总线 sniff GetArm*Ctrl 作为主臂 action。
        has_leader = bool(args.leader_left_can or args.leader_right_can)
        has_follower = bool(args.follower_left_can or args.follower_right_can)
        if not (has_leader or has_follower):
            raise ValueError(
                "Leader action recording needs at least one CAN "
                "(--leader-*-can and/or --follower-*-can) to sniff GetArm*Ctrl."
            )

    config = PiperRobotConfig(
        leader_left_port=args.leader_left_can,
        leader_right_port=args.leader_right_can,
        follower_left_port=args.follower_left_can,
        follower_right_port=args.follower_right_can,
        cameras=camera_configs,
    )
    robot = PiperRobot(config)
    period = 1.0 / args.fps
    started_at = time.monotonic()
    camera_shape = (args.camera_height, args.camera_width, 3)
    camera_names = list(camera_configs.keys())
    preprocessing_config = load_preprocessing_config(getattr(args, "preprocessing", None))
    frame_preprocessor = (
        FramePreprocessor(preprocessing_config, mode="online")
        if preprocessing_config.active
        else None
    )
    use_dual_record = (
        args.dataset_format == "lerobot"
        and preprocessing_config.dual_record_active
    )
    augmented_repo_id = resolve_augmented_repo_id(
        args.repo_id,
        preprocessing_config.dual_record.augmented_repo_id,
    )
    if use_dual_record:
        print("Dual recording enabled.")
        print(f"  raw dataset: {args.root}/{args.repo_id}")
        print(f"  augmented dataset: {args.root}/{augmented_repo_id}")
    elif frame_preprocessor is not None:
        print("Frame preprocessing enabled for recording.")
    if action_source == "leader":
        print("Leader-follower teleop recording enabled.")
        print(f"  observation: follower feedback GetArm*Msgs ({args.follower_left_can}, {args.follower_right_can})")
        print(
            "  action: leader command GetArm*Ctrl "
            f"({args.leader_left_can or args.follower_left_can}, "
            f"{args.leader_right_can or args.follower_right_can})"
        )
        print("  Keep leader aviation plugs connected; do NOT unplug masters or run reset while recording.")
        print("  Move the leader arms; followers should track via Piper teleop.")
    else:
        print("Follower recording enabled (observation and action from follower CAN).")
        print(f"  follower CAN: left={args.follower_left_can}, right={args.follower_right_can}")
        print("  Both observation and action come from follower feedback (GetArm*Msgs).")
    stop_reason = "completed"
    dataset_paths: list[Path] = []
    stop_requested, previous_sigint_handler = install_stop_handler()

    try:
        robot.connect()
        if args.dataset_format == "lerobot":
            if use_dual_record:
                recorder_context = DualLeRobotEpisodeRecorder(
                    raw_recorder=make_lerobot_recorder(
                        args,
                        repo_id=args.repo_id,
                        camera_names=camera_names,
                        camera_shape=camera_shape,
                    ),
                    augmented_recorder=make_lerobot_recorder(
                        args,
                        repo_id=augmented_repo_id,
                        camera_names=camera_names,
                        camera_shape=camera_shape,
                    ),
                )
            else:
                recorder_context = make_lerobot_recorder(
                    args,
                    repo_id=args.repo_id,
                    camera_names=camera_names,
                    camera_shape=camera_shape,
                )
        else:
            recorder_context = PiperEpisodeRecorder(
                output_dir=args.output_dir,
                task=args.task,
                fps=args.fps,
                action_source=action_source,
                camera_names=camera_names,
                image_format=args.image_format,
                image_quality=args.image_quality,
            )

        with recorder_context as recorder:
            if use_dual_record:
                dataset_paths = recorder.episode_dirs
                print(f"Recording raw to {dataset_paths[0]}")
                print(f"Recording augmented to {dataset_paths[1]}")
            else:
                dataset_paths = [Path(recorder.episode_dir)]
                print(f"Recording to {dataset_paths[0]}")
            print("Press Ctrl+C once to stop after the current frame and save the episode.")
            if frame_preprocessor is not None:
                frame_preprocessor.reset()
            while True:
                loop_started_at = time.monotonic()
                observation = robot.get_observation()
                if action_source == "leader":
                    action = robot.get_leader_action()
                else:
                    action = {
                        key: value
                        for key, value in observation.items()
                        if key.endswith(".pos")
                    }

                if use_dual_record:
                    raw_observation = observation
                    raw_action = action
                    augmented_observation, augmented_action = frame_preprocessor.process_frame(
                        observation,
                        action,
                        camera_names,
                    )
                    recorder.record_frame(
                        raw_observation=raw_observation,
                        raw_action=raw_action,
                        augmented_observation=augmented_observation,
                        augmented_action=augmented_action,
                    )
                else:
                    if frame_preprocessor is not None and frame_preprocessor.enabled:
                        observation, action = frame_preprocessor.process_frame(
                            observation,
                            action,
                            camera_names,
                        )
                    recorder.record_frame(observation=observation, action=action)

                if stop_requested["value"]:
                    stop_reason = "manual"
                    break

                if args.duration is not None and time.monotonic() - started_at >= args.duration:
                    stop_reason = "duration"
                    break

                elapsed = time.monotonic() - loop_started_at
                time.sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        stop_reason = "manual"
        print("\nStopping recording.")
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
        if robot.is_connected:
            robot.disconnect()

    if dataset_paths:
        outcome = prompt_episode_outcome() if args.prompt_outcome else args.episode_outcome
        write_episode_outcomes(
            dataset_paths=dataset_paths,
            task=args.task,
            outcome=outcome,
            stop_reason=stop_reason,
            dataset_format=args.dataset_format,
        )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_recording(args)


if __name__ == "__main__":
    main()
