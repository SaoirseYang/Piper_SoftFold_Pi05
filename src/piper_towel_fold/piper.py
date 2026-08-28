from functools import cached_property
import math

import numpy as np
from lerobot.cameras import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from lerobot.robots import Robot

from .config import PiperRobotConfig

try:
    from piper_sdk import C_PiperInterface_V2
except ImportError:
    try:
        from piper_sdk import C_PiperInterface as C_PiperInterface_V2
    except ImportError:
        C_PiperInterface_V2 = None


class PiperRobot(Robot):
    config_class = PiperRobotConfig
    name = "piper"

    def __init__(self, config: PiperRobotConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._connected = False
        self._leaders: dict[str, object | None] = {"left": None, "right": None}
        self._followers: dict[str, object | None] = {"left": None, "right": None}

    @property
    def _arm_joint_names(self) -> list[str]:
        return [f"joint_{i}" for i in range(1, 7)]

    @property
    def _motors_ft(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for side in ("left", "right"):
            for joint_name in self._arm_joint_names:
                features[f"{side}_{joint_name}.pos"] = float
            features[f"{side}_gripper.pos"] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        return {
            name: (camera_cfg.height, camera_cfg.width, 3)
            for name, camera_cfg in self.config.cameras.items()
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        cameras_connected = all(cam.is_connected for cam in self.cameras.values())
        leaders_connected = all(
            arm is None or self._sdk_arm_is_connected(arm) for arm in self._leaders.values()
        )
        arms_connected = all(
            arm is None or self._sdk_arm_is_connected(arm) for arm in self._followers.values()
        )
        return self._connected and cameras_connected and leaders_connected and arms_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate  # Calibration behavior is not implemented yet.
        self._leaders["left"] = self._make_follower(self.config.leader_left_port)
        self._leaders["right"] = self._make_follower(self.config.leader_right_port)
        self._followers["left"] = self._make_follower(self.config.follower_left_port)
        self._followers["right"] = self._make_follower(self.config.follower_right_port)

        for cam in self.cameras.values():
            cam.connect()
        self._connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def _make_follower(self, can_name: str | None) -> object | None:
        if can_name is None or str(can_name).strip() == "":
            return None

        if C_PiperInterface_V2 is None:
            raise ImportError("piper_sdk is not installed in the current Python environment.")

        # Default SDK level is WARNING, which still prints [ERROR] on CAN send failures
        # (e.g. disconnected left arm). Keep the console clean during live runs.
        create_kwargs: dict[str, object] = {"can_name": can_name}
        try:
            from piper_sdk import LogLevel

            create_kwargs["logger_level"] = LogLevel.SILENT
        except Exception:
            pass

        arm = C_PiperInterface_V2(**create_kwargs)
        arm.ConnectPort()
        if self.config.enable_control:
            self._enable_arm_control(arm)
        return arm

    def _enable_arm_control(self, arm: object) -> None:
        enable_arm = getattr(arm, "EnableArm", None)
        if callable(enable_arm):
            try:
                enable_arm(7)
            except TypeError:
                enable_arm()

        mode_ctrl = getattr(arm, "MotionCtrl_2", None)
        if callable(mode_ctrl):
            try:
                mode_ctrl(0x01, 0x01, int(self.config.control_speed), 0x00)
            except TypeError:
                mode_ctrl(0x01, 0x01, int(self.config.control_speed))

    def _sdk_arm_is_connected(self, arm: object) -> bool:
        get_status = getattr(arm, "get_connect_status", None)
        if callable(get_status):
            return bool(get_status())
        return True

    @staticmethod
    def _joints_mdeg_to_rad(joints_mdeg: list[float | int]) -> list[float]:
        return [math.radians(float(value) / 1000.0) for value in joints_mdeg]

    def _read_joint_positions(self, arm: object | None) -> list[float]:
        """从臂反馈（CAN feedback 2Ax）：实际关节位置。"""
        if arm is None:
            return [0.0] * 6

        joint_state = arm.GetArmJointMsgs().joint_state
        joints_mdeg = [
            joint_state.joint_1,
            joint_state.joint_2,
            joint_state.joint_3,
            joint_state.joint_4,
            joint_state.joint_5,
            joint_state.joint_6,
        ]
        return self._joints_mdeg_to_rad(joints_mdeg)

    def _read_joint_ctrl_positions(self, arm: object | None) -> list[float]:
        """主臂指令（CAN control 15x）：发给从臂的目标关节角。

        共 CAN 示教时，SDK 用 GetArmJointCtrl 承接主臂下发的目标值。
        """
        if arm is None:
            return [0.0] * 6

        joint_ctrl = arm.GetArmJointCtrl().joint_ctrl
        joints_mdeg = [
            joint_ctrl.joint_1,
            joint_ctrl.joint_2,
            joint_ctrl.joint_3,
            joint_ctrl.joint_4,
            joint_ctrl.joint_5,
            joint_ctrl.joint_6,
        ]
        return self._joints_mdeg_to_rad(joints_mdeg)

    def _read_gripper_position(self, arm: object | None) -> float:
        """从臂夹爪反馈，单位米。"""
        if arm is None:
            return 0.0

        gripper_state = arm.GetArmGripperMsgs().gripper_state
        return gripper_state.grippers_angle / 1_000_000.0

    def _read_gripper_ctrl_position(self, arm: object | None) -> float:
        """主臂夹爪指令，单位米。"""
        if arm is None:
            return 0.0

        gripper_ctrl = arm.GetArmGripperCtrl().gripper_ctrl
        return gripper_ctrl.grippers_angle / 1_000_000.0

    def _read_arm_state(self, arm: object | None, side: str) -> dict[str, float]:
        state: dict[str, float] = {}
        joints = self._read_joint_positions(arm)
        for index, position in enumerate(joints, start=1):
            state[f"{side}_joint_{index}.pos"] = position
        state[f"{side}_gripper.pos"] = self._read_gripper_position(arm)
        return state

    def _read_arm_ctrl(self, arm: object | None, side: str) -> dict[str, float]:
        """读取主臂示教指令（观测用反馈请用 _read_arm_state）。"""
        state: dict[str, float] = {}
        joints = self._read_joint_ctrl_positions(arm)
        for index, position in enumerate(joints, start=1):
            state[f"{side}_joint_{index}.pos"] = position
        state[f"{side}_gripper.pos"] = self._read_gripper_ctrl_position(arm)
        return state

    def _bus_arm_for_leader(self, side: str) -> object | None:
        """共 CAN 时 leader/follower 可能指向同一口；优先 leader，否则 sniff follower。"""
        return self._leaders[side] or self._followers[side]

    def _limited_value(self, target: float, current: float, max_step: float) -> float:
        # <=0 / inf / 超大步长 = 不限速，直接下发目标（推理夹爪、replay 1:1 等）
        if (not math.isfinite(max_step)) or max_step <= 0.0 or max_step >= 1e6:
            return float(target)
        return float(np.clip(target, current - max_step, current + max_step))

    def set_control_speed(self, control_speed: int) -> None:
        """更新 SDK MotionCtrl_2 速度档（0-100），立即作用于已连接从臂。"""
        self.config.control_speed = int(control_speed)
        if not self.is_connected:
            return
        for arm in self._followers.values():
            if arm is not None:
                self._enable_arm_control(arm)

    def _arm_command_from_action(
        self,
        action: RobotAction,
        current: dict[str, float],
        side: str,
    ) -> tuple[list[float], float]:
        joints = []
        for index in range(1, 7):
            key = f"{side}_joint_{index}.pos"
            target = float(action.get(key, current[key]))
            joints.append(self._limited_value(target, current[key], self.config.max_joint_step_rad))

        gripper_key = f"{side}_gripper.pos"
        gripper_target = float(action.get(gripper_key, current[gripper_key]))
        gripper = self._limited_value(
            gripper_target,
            current[gripper_key],
            self.config.max_gripper_step_m,
        )
        return joints, gripper

    def _send_arm_command(self, arm: object | None, joints_rad: list[float], gripper_m: float) -> None:
        if arm is None:
            return

        joint_ctrl = getattr(arm, "JointCtrl", None)
        if callable(joint_ctrl):
            joints_mdeg = [int(round(math.degrees(value) * 1000.0)) for value in joints_rad]
            joint_ctrl(*joints_mdeg)

        gripper_ctrl = getattr(arm, "GripperCtrl", None)
        if callable(gripper_ctrl):
            gripper_um = int(round(gripper_m * 1_000_000.0))
            gripper_ctrl(gripper_um, int(self.config.gripper_effort), 0x01, 0)

    def _command_dict(self, side: str, joints_rad: list[float], gripper_m: float) -> dict[str, float]:
        command: dict[str, float] = {}
        for index, value in enumerate(joints_rad, start=1):
            command[f"{side}_joint_{index}.pos"] = value
        command[f"{side}_gripper.pos"] = gripper_m
        return command

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        observation: RobotObservation = {}

        for side in ("left", "right"):
            observation.update(self._read_arm_state(self._followers[side], side))

        for camera_name, (height, width, channels) in self._cameras_ft.items():
            camera = self.cameras[camera_name]
            try:
                observation[camera_name] = camera.read_latest()
            except Exception:
                observation[camera_name] = np.zeros((height, width, channels), dtype=np.uint8)

        return observation

    @check_if_not_connected
    def get_leader_action(self) -> RobotAction:
        """主臂 action：读总线控制帧 GetArm*Ctrl，而非从臂反馈 GetArm*Msgs。"""
        action: RobotAction = {}
        for side in ("left", "right"):
            action.update(self._read_arm_ctrl(self._bus_arm_for_leader(side), side))
        return action

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.config.enable_control:
            raise RuntimeError("PiperRobotConfig.enable_control must be True before sending actions.")

        sanitized_action: RobotAction = {}
        for side in ("left", "right"):
            current = self._read_arm_state(self._followers[side], side)
            joints, gripper = self._arm_command_from_action(action, current, side)
            self._send_arm_command(self._followers[side], joints, gripper)
            sanitized_action.update(self._command_dict(side, joints, gripper))

        return sanitized_action

    @check_if_not_connected
    def disconnect(self) -> None:
        for cam in self.cameras.values():
            cam.disconnect()
        for arm in (*self._leaders.values(), *self._followers.values()):
            disconnect = getattr(arm, "DisconnectPort", None)
            if callable(disconnect):
                disconnect()
        self._connected = False
