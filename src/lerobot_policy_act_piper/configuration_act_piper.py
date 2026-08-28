from dataclasses import dataclass, field

from lerobot.policies.act.configuration_act import ACTConfig

try:
    from lerobot.configs import PreTrainedConfig
except ImportError:
    from lerobot.configs.policies import PreTrainedConfig


DEFAULT_CAMERA_SCALES: dict[str, float] = {
    "cam_right": 2.0,
    "cam_side": 0.9,
    "cam_top": 0.8,
}


@PreTrainedConfig.register_subclass("act_piper")
@dataclass
class ACTPiperConfig(ACTConfig):
    """ACT config with per-camera feature scaling and optional camera ID embeddings.

    Camera keys in ``camera_scales`` use the short camera name (e.g. ``cam_right``),
    not the full dataset feature key (``observation.images.cam_right``).
    """

    camera_scales: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CAMERA_SCALES))
    learnable_camera_scales: bool = False
    use_camera_id_embed: bool = True
