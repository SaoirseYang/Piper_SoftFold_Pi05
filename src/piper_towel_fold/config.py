from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("piper")
@dataclass
class PiperRobotConfig(RobotConfig):
    leader_left_port: str | None = None
    leader_right_port: str | None = None
    follower_left_port: str | None = None
    follower_right_port: str | None = None
    enable_control: bool = False
    control_speed: int = 15
    max_joint_step_rad: float = 0.04
    max_gripper_step_m: float = 0.002
    gripper_effort: int = 1000
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_top": OpenCVCameraConfig(
                index_or_path=2,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )
