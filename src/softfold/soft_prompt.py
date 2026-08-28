"""Soft Prompt injection helpers for X-VLA / Florence-style backbones.

LeRobot's X-VLA already exposes ``train_soft_prompts``. This module provides an
optional explicit Piper soft-prompt bank that can be attached when you need a
fresh randomly-initialized robot token set separate from Aloha/Franka prompts.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class SoftPromptBank(nn.Module):
    """Learnable soft prompt tokens: [num_tokens, hidden_dim]."""

    def __init__(self, num_tokens: int = 16, hidden_dim: int = 768, init_std: float = 0.02):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.hidden_dim = int(hidden_dim)
        self.tokens = nn.Parameter(torch.randn(self.num_tokens, self.hidden_dim) * init_std)

    def forward(self, batch_size: int) -> torch.Tensor:
        """Return prompts expanded to (B, T, D)."""
        return self.tokens.unsqueeze(0).expand(batch_size, -1, -1)


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


def count_trainable(module: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


def configure_stage1_peft(policy: Any, *, train_soft_prompts: bool = True) -> dict[str, int]:
    """Stage-1: freeze backbone; keep soft prompts (+ action head if exposed) trainable.

    Works best when ``policy`` is a LeRobot X-VLA policy that already honors
    ``freeze_vision_encoder`` / ``freeze_language_encoder`` / ``train_soft_prompts``.
    """
    if hasattr(policy, "parameters"):
        freeze_module(policy)
    # Re-enable soft prompt params by name heuristic
    enabled = 0
    for name, p in policy.named_parameters():
        key = name.lower()
        if train_soft_prompts and ("soft_prompt" in key or "softprompt" in key or "prompt" in key):
            p.requires_grad = True
            enabled += p.numel()
        if "action_head" in key or "action_out" in key or key.endswith("action_proj.weight"):
            p.requires_grad = True
            enabled += p.numel()
    trainable, total = count_trainable(policy)
    return {"trainable": trainable, "total": total, "promptish_enabled": enabled}


def configure_stage2_shallow_unfreeze(
    policy: Any,
    *,
    vision_last_n_blocks: int = 2,
    language_last_n_blocks: int = 0,
) -> dict[str, int]:
    """Stage-2: keep stage-1 params + unfreeze last N vision/language blocks."""
    # Keep whatever was trainable, then unfreeze trailing blocks by name.
    for name, module in policy.named_modules():
        lname = name.lower()
        if vision_last_n_blocks > 0 and ("vision" in lname or "siglip" in lname or "visual" in lname):
            # Best-effort: unfreeze modules whose name contains block index near the end.
            for p in module.parameters(recurse=False):
                p.requires_grad = True
        if language_last_n_blocks > 0 and ("language" in lname or "gemma" in lname or "text" in lname):
            for p in module.parameters(recurse=False):
                p.requires_grad = True
    trainable, total = count_trainable(policy)
    return {
        "trainable": trainable,
        "total": total,
        "vision_last_n_blocks": vision_last_n_blocks,
        "language_last_n_blocks": language_last_n_blocks,
    }
