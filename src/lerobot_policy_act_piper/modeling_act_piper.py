"""ACT Piper: per-camera feature scaling and camera ID embeddings on top of LeRobot ACT."""

from __future__ import annotations

import einops
import torch
from torch import Tensor, nn

from lerobot.policies.act.modeling_act import ACT, ACTPolicy, ACTTemporalEnsembler
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from .configuration_act_piper import ACTPiperConfig


def _camera_name_from_feature_key(feature_key: str) -> str:
    prefix = "observation.images."
    if feature_key.startswith(prefix):
        return feature_key[len(prefix) :]
    return feature_key


class ACTPiper(ACT):
    """ACT backbone with optional per-camera scaling and camera identity embeddings."""

    def __init__(self, config: ACTPiperConfig):
        super().__init__(config)
        self._setup_camera_modulations()

    def _resolve_camera_scale_values(self) -> list[float]:
        scales: list[float] = []
        for feature_key in self.config.image_features:
            camera_name = _camera_name_from_feature_key(feature_key)
            scales.append(self.config.camera_scales.get(camera_name, 1.0))
        return scales

    def _setup_camera_modulations(self) -> None:
        if not self.config.image_features:
            self.camera_scales = None
            self.camera_id_embed = None
            return

        scale_tensor = torch.tensor(self._resolve_camera_scale_values(), dtype=torch.float32)
        if self.config.learnable_camera_scales:
            self.camera_scales = nn.Parameter(scale_tensor)
        else:
            self.register_buffer("camera_scales", scale_tensor)

        if self.config.use_camera_id_embed:
            self.camera_id_embed = nn.Embedding(len(self.config.image_features), self.config.dim_model)
        else:
            self.camera_id_embed = None

    def _append_camera_tokens(
        self,
        batch: dict[str, Tensor],
        encoder_in_tokens: list[Tensor],
        encoder_in_pos_embed: list[Tensor],
    ) -> None:
        for cam_idx, img in enumerate(batch[OBS_IMAGES]):
            cam_features = self.backbone(img)["feature_map"]
            cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
            cam_features = self.encoder_img_feat_input_proj(cam_features)

            if self.camera_scales is not None:
                scale = self.camera_scales[cam_idx].to(dtype=cam_features.dtype, device=cam_features.device)
                cam_features = cam_features * scale

            cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
            cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

            if self.camera_id_embed is not None:
                cam_id = self.camera_id_embed.weight[cam_idx].to(
                    dtype=cam_features.dtype, device=cam_features.device
                )
                cam_features = cam_features + cam_id.view(1, 1, -1)

            encoder_in_tokens.extend(list(cam_features))
            encoder_in_pos_embed.extend(list(cam_pos_embed))

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            pos_embed = self.vae_encoder_pos_enc.clone().detach()
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)

            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch[OBS_STATE].device
            )

        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.image_features:
            self._append_camera_tokens(batch, encoder_in_tokens, encoder_in_pos_embed)

        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )
        decoder_out = decoder_out.transpose(0, 1)
        actions = self.action_head(decoder_out)
        return actions, (mu, log_sigma_x2)


class ACTPiperPolicy(ACTPolicy):
    config_class = ACTPiperConfig
    name = "act_piper"

    def __init__(self, config: ACTPiperConfig, **kwargs):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.model = ACTPiper(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()
