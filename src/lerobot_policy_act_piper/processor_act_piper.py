from typing import Any

import torch

from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from .configuration_act_piper import ACTPiperConfig


def make_act_piper_pre_post_processors(
    config: ACTPiperConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Reuse ACT normalization processors; Piper-specific logic lives in the model."""
    return make_act_pre_post_processors(config, dataset_stats=dataset_stats)
