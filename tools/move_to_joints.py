"""Move Piper dual-arm followers to given joint angles (degrees).

Starting pose reference (from read_state):
  left:  j1=0.287, j2=0.000, j3=0.000, j4=-0.634, j5=25.124, j6=3.356, gripper=0.000280
  right: j1=0.000, j2=0.000, j3=0.000, j4=1.580, j5=26.510, j6=2.634, gripper=0.000140

Examples:
  # 先回到上述起始位姿，再在真实当前位姿上 j5 +5°
  PYTHONPATH=src python tools/move_to_joints.py --preset home
  PYTHONPATH=src python tools/move_to_joints.py --preset demo

  # 连续平滑扫动约 10s（默认左右 j5 正弦），用于观察抖动
  PYTHONPATH=src python tools/move_to_joints.py --preset sweep
  PYTHONPATH=src python tools/move_to_joints.py --preset sweep --duration 10 --amp-deg 5 --freq-hz 0.2

  # 多关节同时扫动（测耦合/整臂是否抖）
  PYTHONPATH=src python tools/move_to_joints.py --preset sweep --sweep-joints 2,3,5
  PYTHONPATH=src python tools/move_to_joints.py --preset sweep --sweep-joints 1,2,3,4,5,6 \\
    --amp-deg 8 --phase-mode stagger --duration 10

  # 自定义目标（度）
  PYTHONPATH=src python tools/move_to_joints.py \\
    --left-joints 0.287,0,0,-0.634,30.124,3.356 \\
    --right-joints 0,0,0,1.580,31.510,2.634
"""

from __future__ import annotations

import argparse
import math
import time

from piper_towel_fold.config import PiperRobotConfig
from piper_towel_fold.piper import PiperRobot

# 起始 / home 位姿（度 / 米）——--preset home 使用
HOME_LEFT_DEG = [0.287, 0.000, 0.000, -0.634, 25.124, 3.356]
HOME_RIGHT_DEG = [0.000, 0.000, 0.000, 1.580, 26.510, 2.634]
HOME_LEFT_GRIPPER_M = 0.000280
HOME_RIGHT_GRIPPER_M = 0.000140

DEMO_J5_OFFSET_DEG = 5.0
# sweep 默认：在 j5 上做正弦往返，其它关节保持当前角
SWEEP_DEFAULT_JOINTS = (5,)  # 1-based
SWEEP_DEFAULT_DURATION_S = 10.0
SWEEP_DEFAULT_AMP_DEG = 5.0
SWEEP_DEFAULT_FREQ_HZ = 0.2


def parse_joints_deg(text: str) -> list[float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("需要恰好 6 个关节角（度），用逗号分隔")
    return [float(p) for p in parts]


def parse_joint_indices(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("至少指定 1 个关节，如 5 或 2,3,5")
    indices: list[int] = []
    for part in parts:
        value = int(part)
        if value < 1 or value > 6:
            raise argparse.ArgumentTypeError(f"关节序号必须在 1..6，收到 {value}")
        if value not in indices:
            indices.append(value)
    return indices


def parse_amp_list(text: str, n_joints: int) -> list[float]:
    """单个幅值，或与关节数等长的逗号分隔幅值列表。"""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) == 1:
        amp = float(parts[0])
        if amp < 0:
            raise argparse.ArgumentTypeError("幅值必须 >= 0")
        return [amp] * n_joints
    if len(parts) != n_joints:
        raise argparse.ArgumentTypeError(
            f"amp-deg 需 1 个数或与关节数相同（{n_joints}），收到 {len(parts)} 个"
        )
    amps = [float(p) for p in parts]
    if any(a < 0 for a in amps):
        raise argparse.ArgumentTypeError("幅值必须 >= 0")
    return amps


def phase_offsets_rad(n_joints: int, phase_mode: str) -> list[float]:
    if phase_mode == "sync":
        return [0.0] * n_joints
    if phase_mode == "stagger":
        # 各关节错相，末端轨迹更复杂，更能暴露耦合抖动
        return [2.0 * math.pi * i / max(n_joints, 1) for i in range(n_joints)]
    raise ValueError(f"未知 phase-mode: {phase_mode}")

def joints_to_action(side: str, joints_deg: list[float], gripper_m: float) -> dict[str, float]:
    action = {
        f"{side}_joint_{index}.pos": math.radians(value)
        for index, value in enumerate(joints_deg, start=1)
    }
    action[f"{side}_gripper.pos"] = float(gripper_m)
    return action


def arm_joints_deg(obs: dict, side: str) -> list[float]:
    return [math.degrees(obs[f"{side}_joint_{index}.pos"]) for index in range(1, 7)]


def max_joint_error_rad(obs: dict, side: str, joints_deg: list[float]) -> float:
    return max(
        abs(obs[f"{side}_joint_{index}.pos"] - math.radians(value))
        for index, value in enumerate(joints_deg, start=1)
    )


def joint_errors_deg(obs: dict, side: str, joints_deg: list[float]) -> list[float]:
    return [
        abs(math.degrees(obs[f"{side}_joint_{index}.pos"]) - value)
        for index, value in enumerate(joints_deg, start=1)
    ]


def format_arm(obs: dict, side: str) -> str:
    joints = ", ".join(
        f"j{index}={math.degrees(obs[f'{side}_joint_{index}.pos']):8.3f} deg"
        for index in range(1, 7)
    )
    gripper = obs[f"{side}_gripper.pos"]
    return f"{side}: {joints}, gripper={gripper:.6f}"


def is_all_joints_near_zero(obs: dict, *, eps_deg: float = 0.05) -> bool:
    for side in ("left", "right"):
        for index in range(1, 7):
            if abs(math.degrees(obs[f"{side}_joint_{index}.pos"])) > eps_deg:
                return False
    return True


def wait_for_valid_state(
    robot: PiperRobot,
    *,
    timeout_s: float = 5.0,
    period_s: float = 0.05,
) -> dict:
    """ConnectPort 后首帧常为全 0，需等到 CAN 关节反馈到达再控。"""
    deadline = time.monotonic() + timeout_s
    last = robot.get_observation()
    while time.monotonic() < deadline:
        if not is_all_joints_near_zero(last):
            return last
        time.sleep(period_s)
        last = robot.get_observation()

    raise RuntimeError(
        "连接后关节反馈仍为全 0。请确认已 bringup_can、从臂已使能，"
        "并先用 scripts/run_read_state.sh 能读到非零角度。"
    )


def move_to(
    robot: PiperRobot,
    left_deg: list[float],
    right_deg: list[float],
    left_gripper_m: float,
    right_gripper_m: float,
    *,
    tol_deg: float,
    period_s: float,
    timeout_s: float,
) -> None:
    tol_rad = math.radians(tol_deg)
    deadline = time.monotonic() + timeout_s
    step = 0

    while True:
        action = {}
        action.update(joints_to_action("left", left_deg, left_gripper_m))
        action.update(joints_to_action("right", right_deg, right_gripper_m))
        robot.send_action(action)

        obs = robot.get_observation()
        err_left = max_joint_error_rad(obs, "left", left_deg)
        err_right = max_joint_error_rad(obs, "right", right_deg)
        err = max(err_left, err_right)
        step += 1

        if step == 1 or step % 25 == 0:
            left_errs = joint_errors_deg(obs, "left", left_deg)
            right_errs = joint_errors_deg(obs, "right", right_deg)
            print(
                f"step={step}  max_err={math.degrees(err):.3f} deg  "
                f"(left={math.degrees(err_left):.3f}, right={math.degrees(err_right):.3f})"
            )
            print(
                "  left errs : "
                + ", ".join(f"j{i}={e:.3f}" for i, e in enumerate(left_errs, start=1))
            )
            print(
                "  right errs: "
                + ", ".join(f"j{i}={e:.3f}" for i, e in enumerate(right_errs, start=1))
            )

        if err < tol_rad:
            print("到位。")
            print(format_arm(obs, "left"))
            print(format_arm(obs, "right"))
            return

        if time.monotonic() > deadline:
            print("超时，未完全到位。当前状态：")
            print(format_arm(obs, "left"))
            print(format_arm(obs, "right"))
            raise TimeoutError(f"未能在 {timeout_s:.1f}s 内到达目标（容差 {tol_deg}°）")

        time.sleep(period_s)


def sweep_joints(
    robot: PiperRobot,
    base_left_deg: list[float],
    base_right_deg: list[float],
    left_gripper_m: float,
    right_gripper_m: float,
    *,
    duration_s: float,
    amps_deg: list[float],
    freq_hz: float,
    joint_indices: list[int],
    phase_mode: str,
    period_s: float,
) -> None:
    """以当前位姿为中心，对一个或多个关节做正弦连续运动，便于目视/统计抖动。"""
    if not joint_indices:
        raise ValueError("至少指定 1 个关节")
    if len(amps_deg) != len(joint_indices):
        raise ValueError("amps_deg 长度必须与 joint_indices 一致")
    if duration_s <= 0:
        raise ValueError("duration 必须 > 0")
    if freq_hz < 0:
        raise ValueError("freq-hz 必须 >= 0")

    phases = phase_offsets_rad(len(joint_indices), phase_mode)
    joint_label = ",".join(f"j{j}" for j in joint_indices)
    amp_label = ",".join(f"±{a:.1f}°" for a in amps_deg)
    print(
        f"sweep：关节 [{joint_label}]，幅值 [{amp_label}]，频率 {freq_hz:.3f} Hz，"
        f"相位={phase_mode}，时长 {duration_s:.1f}s，周期 {period_s:.3f}s"
    )
    print("未列入的关节保持起始角。Ctrl+C 可中断。")

    t0 = time.monotonic()
    next_tick = t0
    step = 0
    # 按侧、按关节统计
    max_track_err: dict[str, dict[int, float]] = {
        "left": {j: 0.0 for j in joint_indices},
        "right": {j: 0.0 for j in joint_indices},
    }
    max_jerk_proxy: dict[str, dict[int, float]] = {
        "left": {j: 0.0 for j in joint_indices},
        "right": {j: 0.0 for j in joint_indices},
    }
    prev_fb: dict[str, dict[int, float | None]] = {
        "left": {j: None for j in joint_indices},
        "right": {j: None for j in joint_indices},
    }
    prev_dfb: dict[str, dict[int, float | None]] = {
        "left": {j: None for j in joint_indices},
        "right": {j: None for j in joint_indices},
    }

    while True:
        now = time.monotonic()
        t = now - t0
        if t >= duration_s:
            break

        left_deg = list(base_left_deg)
        right_deg = list(base_right_deg)
        offsets: list[float] = []
        for amp, phase, joint_index in zip(amps_deg, phases, joint_indices, strict=True):
            offset = amp * math.sin(2.0 * math.pi * freq_hz * t + phase)
            offsets.append(offset)
            ji = joint_index - 1
            left_deg[ji] = base_left_deg[ji] + offset
            right_deg[ji] = base_right_deg[ji] + offset

        action = {}
        action.update(joints_to_action("left", left_deg, left_gripper_m))
        action.update(joints_to_action("right", right_deg, right_gripper_m))
        robot.send_action(action)
        obs = robot.get_observation()

        for side, cmd_deg in (("left", left_deg), ("right", right_deg)):
            for joint_index in joint_indices:
                ji = joint_index - 1
                fb = math.degrees(obs[f"{side}_joint_{joint_index}.pos"])
                track_err = abs(fb - cmd_deg[ji])
                max_track_err[side][joint_index] = max(
                    max_track_err[side][joint_index], track_err
                )

                prev = prev_fb[side][joint_index]
                if prev is not None:
                    dfb = fb - prev
                    prev_d = prev_dfb[side][joint_index]
                    if prev_d is not None:
                        max_jerk_proxy[side][joint_index] = max(
                            max_jerk_proxy[side][joint_index], abs(dfb - prev_d)
                        )
                    prev_dfb[side][joint_index] = dfb
                prev_fb[side][joint_index] = fb

        step += 1
        if step == 1 or step % 25 == 0:
            off_str = ", ".join(
                f"j{j}={o:+5.1f}" for j, o in zip(joint_indices, offsets, strict=True)
            )
            print(f"t={t:5.2f}s  offsets[{off_str}]")

        next_tick += period_s
        sleep_s = next_tick - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            next_tick = time.monotonic()

    print("sweep 结束，回到起始中心角…")
    move_to(
        robot,
        list(base_left_deg),
        list(base_right_deg),
        left_gripper_m,
        right_gripper_m,
        tol_deg=1.0,
        period_s=period_s,
        timeout_s=max(10.0, duration_s),
    )

    def fmt_joint_stats(stats: dict[str, dict[int, float]]) -> str:
        lines = []
        for side in ("left", "right"):
            parts = [f"j{j}={stats[side][j]:.3f}" for j in joint_indices]
            lines.append(f"  {side}: " + ", ".join(parts))
        return "\n".join(lines)

    print("抖动粗测（越大越不平）：")
    print("max |cmd-fb| (°)：")
    print(fmt_joint_stats(max_track_err))
    print("max |ΔΔfb| (°)：")
    print(fmt_joint_stats(max_jerk_proxy))
    worst_track = max(max_track_err[s][j] for s in ("left", "right") for j in joint_indices)
    worst_jerk = max(max_jerk_proxy[s][j] for s in ("left", "right") for j in joint_indices)
    print(f"最差：max|cmd-fb|={worst_track:.3f}°  max|ΔΔfb|={worst_jerk:.3f}°")
    print("目视：若运动过程有明显顿挫/颤动（尤其末端），即存在较大抖动。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 Piper 双臂从臂移动到给定关节角目标。")
    parser.add_argument("--left-can", default="can2")
    parser.add_argument("--right-can", default="can0")
    parser.add_argument(
        "--preset",
        choices=("home", "demo", "sweep"),
        default=None,
        help="home=回到记录位姿；demo=当前位姿 j5 +5°；sweep=连续正弦扫动测抖动",
    )
    parser.add_argument(
        "--left-joints",
        type=parse_joints_deg,
        default=None,
        help="左臂 6 关节目标角（度）",
    )
    parser.add_argument(
        "--right-joints",
        type=parse_joints_deg,
        default=None,
        help="右臂 6 关节目标角（度）",
    )
    parser.add_argument("--left-gripper", type=float, default=None)
    parser.add_argument("--right-gripper", type=float, default=None)
    parser.add_argument("--control-speed", type=int, default=15)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.04)
    parser.add_argument("--tol-deg", type=float, default=1.0, help="到位容差（度）")
    parser.add_argument("--period", type=float, default=0.02, help="控制周期（秒）")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--state-wait",
        type=float,
        default=5.0,
        help="连接后等待非零关节反馈的最长时间（秒）",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=SWEEP_DEFAULT_DURATION_S,
        help="sweep 时长（秒），默认 10",
    )
    parser.add_argument(
        "--amp-deg",
        type=str,
        default=str(SWEEP_DEFAULT_AMP_DEG),
        help="sweep 正弦幅值（度）：单个值，或与 --sweep-joints 等长的列表，如 5 或 5,8,5",
    )
    parser.add_argument(
        "--freq-hz",
        type=float,
        default=SWEEP_DEFAULT_FREQ_HZ,
        help="sweep 正弦频率（Hz），默认 0.2",
    )
    parser.add_argument(
        "--sweep-joints",
        type=parse_joint_indices,
        default=list(SWEEP_DEFAULT_JOINTS),
        help="参与扫动的关节，如 5 或 2,3,5 或 1,2,3,4,5,6",
    )
    parser.add_argument(
        "--sweep-joint",
        type=int,
        default=None,
        help="兼容旧参数：单个关节序号；若设置则覆盖 --sweep-joints",
    )
    parser.add_argument(
        "--phase-mode",
        choices=("sync", "stagger"),
        default="stagger",
        help="多关节相位：sync=同相；stagger=错相（默认，更能测耦合）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印目标，不连接/不下发",
    )
    return parser.parse_args()

def resolve_static_targets(
    args: argparse.Namespace,
) -> tuple[list[float] | None, list[float] | None, float | None, float | None]:
    """demo/sweep 需要读真实状态后再算，这里先返回 None。"""
    if args.preset in ("demo", "sweep"):
        return None, None, None, None
    if args.preset == "home":
        return (
            list(HOME_LEFT_DEG),
            list(HOME_RIGHT_DEG),
            HOME_LEFT_GRIPPER_M,
            HOME_RIGHT_GRIPPER_M,
        )
    if args.left_joints is not None and args.right_joints is not None:
        lg = HOME_LEFT_GRIPPER_M if args.left_gripper is None else args.left_gripper
        rg = HOME_RIGHT_GRIPPER_M if args.right_gripper is None else args.right_gripper
        return args.left_joints, args.right_joints, lg, rg
    raise SystemExit(
        "请指定 --preset home|demo|sweep，或同时提供 --left-joints 与 --right-joints"
    )


def targets_from_demo(obs: dict) -> tuple[list[float], list[float], float, float]:
    left = arm_joints_deg(obs, "left")
    right = arm_joints_deg(obs, "right")
    left[4] += DEMO_J5_OFFSET_DEG
    right[4] += DEMO_J5_OFFSET_DEG
    return left, right, obs["left_gripper.pos"], obs["right_gripper.pos"]


def print_targets(
    left_deg: list[float],
    right_deg: list[float],
    left_g: float,
    right_g: float,
) -> None:
    print("目标位姿（度）：")
    print(
        "left:  "
        + ", ".join(f"j{i}={v:8.3f}" for i, v in enumerate(left_deg, start=1))
        + f", gripper={left_g:.6f}"
    )
    print(
        "right: "
        + ", ".join(f"j{i}={v:8.3f}" for i, v in enumerate(right_deg, start=1))
        + f", gripper={right_g:.6f}"
    )


def main() -> None:
    args = parse_args()
    left_deg, right_deg, left_g, right_g = resolve_static_targets(args)

    if args.dry_run and args.preset in ("demo", "sweep"):
        print(f"{args.preset} 需连接后读取真实位姿；dry-run 跳过。")
        return

    if args.dry_run:
        assert left_deg is not None and right_deg is not None
        assert left_g is not None and right_g is not None
        if args.left_gripper is not None:
            left_g = args.left_gripper
        if args.right_gripper is not None:
            right_g = args.right_gripper
        print_targets(left_deg, right_deg, left_g, right_g)
        print("dry-run：未连接机械臂。")
        return

    # sweep 需要限速能跟上正弦；默认 0.04rad/step 对慢扫足够，仍允许用户加大
    max_step = args.max_joint_step_rad
    if args.preset == "sweep" and args.max_joint_step_rad <= 0.04:
        # 略放宽，避免限速把正弦切成折线（看起来像抖）
        max_step = 0.08

    config = PiperRobotConfig(
        follower_left_port=args.left_can,
        follower_right_port=args.right_can,
        enable_control=True,
        control_speed=args.control_speed,
        max_joint_step_rad=max_step,
        cameras={},
    )
    robot = PiperRobot(config)

    try:
        robot.connect()
        print(f"等待关节反馈（最多 {args.state_wait:.1f}s）…")
        obs = wait_for_valid_state(robot, timeout_s=args.state_wait)
        print("当前位姿：")
        print(format_arm(obs, "left"))
        print(format_arm(obs, "right"))

        if args.preset == "sweep":
            base_left = arm_joints_deg(obs, "left")
            base_right = arm_joints_deg(obs, "right")
            left_g = obs["left_gripper.pos"]
            right_g = obs["right_gripper.pos"]
            if args.left_gripper is not None:
                left_g = args.left_gripper
            if args.right_gripper is not None:
                right_g = args.right_gripper

            joint_indices = args.sweep_joints
            if args.sweep_joint is not None:
                joint_indices = [args.sweep_joint]
            amps_deg = parse_amp_list(args.amp_deg, len(joint_indices))

            sweep_joints(
                robot,
                base_left,
                base_right,
                left_g,
                right_g,
                duration_s=args.duration,
                amps_deg=amps_deg,
                freq_hz=args.freq_hz,
                joint_indices=joint_indices,
                phase_mode=args.phase_mode,
                period_s=args.period,
            )
            return

        if args.preset == "demo":
            left_deg, right_deg, left_g, right_g = targets_from_demo(obs)
            print(f"demo：在当前位姿上左右 j5 +{DEMO_J5_OFFSET_DEG}°")
        else:
            assert left_deg is not None and right_deg is not None
            assert left_g is not None and right_g is not None
            if args.left_gripper is not None:
                left_g = args.left_gripper
            if args.right_gripper is not None:
                right_g = args.right_gripper

        print_targets(left_deg, right_deg, left_g, right_g)
        print("开始移动…")
        move_to(
            robot,
            left_deg,
            right_deg,
            left_g,
            right_g,
            tol_deg=args.tol_deg,
            period_s=args.period,
            timeout_s=args.timeout,
        )
    except KeyboardInterrupt:
        print("\n已中断。")
    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
