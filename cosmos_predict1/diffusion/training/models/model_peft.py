# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Tuple, Type, TypeVar, Union

import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from torch import Tensor

from cosmos_predict1.diffusion.training.context_parallel import split_inputs_cp
from cosmos_predict1.diffusion.training.models.extend_model import ExtendDiffusionModel
from cosmos_predict1.diffusion.training.models.model import DiffusionModel as VideoDiffusionModel
from cosmos_predict1.diffusion.training.models.model_image import diffusion_fsdp_class_decorator
from cosmos_predict1.diffusion.training.utils.layer_control.peft_control_config_parser import LayerControlConfigParser
from cosmos_predict1.diffusion.training.utils.peft.peft import add_lora_layers, setup_lora_requires_grad
from cosmos_predict1.diffusion.utils.customization.customization_manager import CustomizationType
from cosmos_predict1.utils import log, misc
from cosmos_predict1.utils.lazy_config import instantiate as lazy_instantiate

T = TypeVar("T")


def video_peft_decorator(base_class: Type[T]) -> Type[T]:
    class PEFTVideoDiffusionModel(base_class):
        def __init__(self, config: dict, fsdp_checkpointer=None):
            super().__init__(config)

        @misc.timer("PEFTVideoDiffusionModel: set_up_model")
        def set_up_model(self):
            config = self.config
            peft_control_config_parser = LayerControlConfigParser(config=config.peft_control)
            peft_control_config = peft_control_config_parser.parse()
            self.model = self.build_model()
            if peft_control_config and peft_control_config["customization_type"] == CustomizationType.LORA:
                add_lora_layers(self.model, peft_control_config)
                num_lora_params = setup_lora_requires_grad(self.model)
                if num_lora_params == 0:
                    raise ValueError("No LoRA parameters found. Please check the model configuration.")
            if config.ema.enabled:
                with misc.timer("PEFTDiffusionModel: instantiate ema"):
                    config.ema.model = self.model
                    self.model_ema = lazy_instantiate(config.ema)
                    config.ema.model = None
            else:
                self.model_ema = None

        def state_dict_model(self) -> Dict:
            return {
                "model": self.model.state_dict(),
                "ema": self.model_ema.state_dict() if self.model_ema else None,
            }

    return PEFTVideoDiffusionModel


@video_peft_decorator
class PEFTVideoDiffusionModel(VideoDiffusionModel):
    pass


@video_peft_decorator
class PEFTExtendDiffusionModel(ExtendDiffusionModel):
    pass


class _LoRAExtendDiffusionModel(ExtendDiffusionModel):
    """ExtendDiffusionModel that injects LoRA layers during build_model.

    Designed to be wrapped by diffusion_fsdp_class_decorator so that FSDP
    sharding happens *after* LoRA injection.
    """

    def build_model(self) -> torch.nn.ModuleDict:
        model = super().build_model()
        config = self.config
        if config.peft_control is not None:
            parser = LayerControlConfigParser(config=config.peft_control)
            peft_cfg = parser.parse()
            if peft_cfg and peft_cfg["customization_type"] == CustomizationType.LORA:
                add_lora_layers(model, peft_cfg)
                n = setup_lora_requires_grad(model)
                log.critical(f"LoRA injected: {n:,} trainable parameters")
                if n == 0:
                    raise ValueError("No LoRA parameters found.")
        return model


@diffusion_fsdp_class_decorator
class FSDPPEFTExtendDiffusionModel(_LoRAExtendDiffusionModel):
    pass


class _LoRAGen3CExtendDiffusionModel(ExtendDiffusionModel):
    """ExtendDiffusionModel adapted for Gen3C warp-conditioned training with LoRA.

    Overrides get_data_and_condition to encode video frames as synthetic warp
    tensors and set them as condition_video_pose. During training, randomly
    marks some warp frames as fully uncovered in pixel space before the VAE
    encode so "missing frame" is represented the same way partial holes are:
    warp pixels at -1 and mask pixels at 0.
    """

    brightness_stability_weight = 0.02

    def build_model(self) -> torch.nn.ModuleDict:
        model = super().build_model()
        config = self.config
        if config.peft_control is not None:
            parser = LayerControlConfigParser(config=config.peft_control)
            peft_cfg = parser.parse()
            if peft_cfg and peft_cfg["customization_type"] == CustomizationType.LORA:
                add_lora_layers(model, peft_cfg)
                n = setup_lora_requires_grad(model)
                log.critical(f"LoRA injected (Gen3C): {n:,} trainable parameters")
                if n == 0:
                    raise ValueError("No LoRA parameters found.")
        return model

    def _sample_lowres_keep_mask(
        self,
        height: int,
        width: int,
        keep_ratio_min: float,
        keep_ratio_max: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Sample a blocky spatial keep mask in [0, 1]."""
        min_grid_h = max(4, height // 96)
        max_grid_h = max(min_grid_h, height // 24)
        min_grid_w = max(4, width // 96)
        max_grid_w = max(min_grid_w, width // 24)

        for _ in range(8):
            grid_h = int(torch.randint(min_grid_h, max_grid_h + 1, (1,), device=device).item())
            grid_w = int(torch.randint(min_grid_w, max_grid_w + 1, (1,), device=device).item())
            keep_ratio = torch.rand((), device=device).item() * (keep_ratio_max - keep_ratio_min) + keep_ratio_min
            lowres = (torch.rand((1, 1, grid_h, grid_w), device=device) < keep_ratio).to(dtype)
            keep_mask = F.interpolate(lowres, size=(height, width), mode="nearest")
            keep_mean = keep_mask.mean().item()
            if 0.0 < keep_mean < 1.0:
                return keep_mask

        # Fallback to a centered visible box if random sampling degenerates.
        keep_mask = torch.zeros((1, 1, height, width), device=device, dtype=dtype)
        box_h = max(1, int(height * keep_ratio_max**0.5))
        box_w = max(1, int(width * keep_ratio_max**0.5))
        top = max(0, (height - box_h) // 2)
        left = max(0, (width - box_w) // 2)
        keep_mask[:, :, top : top + box_h, left : left + box_w] = 1.0
        return keep_mask

    def _sample_warp_keep_mask(
        self, num_frames: int, height: int, width: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor | None:
        """Sample per-frame spatial keep masks for warp conditioning.

        Returns a tensor of shape [1, 1, T, H, W] in [0, 1], or ``None`` when no
        warp corruption is applied for this sample.
        """
        cfg = self.config.conditioner.video_cond_bool
        rate = getattr(cfg, "condition_zero_out_rate", 0.0)
        if rate <= 0.0 or not self.training:
            return None
        if torch.rand((), device=device).item() > rate:
            return None

        use_contiguous = torch.rand((), device=device).item() < 0.5
        keep = torch.ones(num_frames, dtype=torch.bool, device=device)

        if use_contiguous:
            block_len = max(1, int(torch.randint(num_frames // 4, num_frames + 1, (1,), device=device).item()))
            start = torch.randint(0, num_frames - block_len + 1, (1,), device=device).item()
            keep[start : start + block_len] = False
        else:
            keep_prob = torch.rand((), device=device).item() * 0.7 + 0.1
            keep = torch.rand(num_frames, device=device) < keep_prob

        spatial_keep = torch.ones((1, 1, num_frames, height, width), device=device, dtype=dtype)
        for t in range(num_frames):
            if keep[t]:
                continue

            severity = torch.rand((), device=device).item()
            if severity < 0.10:
                frame_keep = torch.zeros((1, 1, height, width), device=device, dtype=dtype)
            elif severity < 0.30:
                frame_keep = self._sample_lowres_keep_mask(
                    height, width, keep_ratio_min=0.08, keep_ratio_max=0.30, device=device, dtype=dtype
                )
            else:
                frame_keep = self._sample_lowres_keep_mask(
                    height, width, keep_ratio_min=0.40, keep_ratio_max=0.75, device=device, dtype=dtype
                )

            spatial_keep[:, :, t] = frame_keep

        return spatial_keep

    def _encode_warp_from_video(self, raw_state: Tensor, latent_state: Tensor, keep: Tensor | None = None) -> Tensor:
        """Create synthetic warp conditioning from video pixels.

        We build a pixel-space warp clip and a pixel-space mask clip, then VAE
        encode both to match inference. Kept frames use the current video frame
        with full coverage (+1 mask after scaling). Dropped frames represent a
        fully missing warp frame: warp pixels at -1 and mask pixels at -1
        (equivalent to a 0-valued mask before scaling).

        Returns:
            Tensor of shape [B, 16 * frame_buffer_max * 2, T, H, W]
        """
        frame_buffer_max = getattr(self.config, "frame_buffer_max", 2)
        _, _, num_frames, height, width = raw_state.shape

        if keep is None:
            warp_pixel = raw_state
            mask_pixel = torch.ones_like(raw_state)
        else:
            neg_ones = torch.full_like(raw_state, -1.0)
            warp_pixel = torch.where(keep > 0.5, raw_state, neg_ones)
            mask_pixel = torch.where(keep > 0.5, torch.ones_like(raw_state), neg_ones)

        warp_latent = self.encode(warp_pixel)
        mask_latent = self.encode(mask_pixel)

        parts = [warp_latent, mask_latent]
        for _ in range(frame_buffer_max - 1):
            parts.append(torch.zeros_like(latent_state))
            parts.append(torch.zeros_like(latent_state))

        return torch.cat(parts, dim=1)

    def _compute_brightness_stability_loss(self, pred_x0: Tensor, target_x0: Tensor, keep_mask: Tensor) -> Tensor:
        """Penalize per-frame mean brightness drift on missing regions.

        This nudges masked frames to match the latent mean of the ground truth
        in the corrupted areas, without changing the main diffusion objective.
        """
        missing_mask = 1.0 - keep_mask
        mask_sum = missing_mask.sum(dim=(3, 4))
        valid_frames = mask_sum > 0
        if not valid_frames.any():
            return pred_x0.new_zeros(pred_x0.shape[0])

        pred_mean = pred_x0.mean(dim=1, keepdim=True)
        target_mean = target_x0.mean(dim=1, keepdim=True)

        pred_frame_mean = (pred_mean * missing_mask).sum(dim=(3, 4)) / mask_sum.clamp_min(1.0)
        target_frame_mean = (target_mean * missing_mask).sum(dim=(3, 4)) / mask_sum.clamp_min(1.0)
        frame_loss = (pred_frame_mean - target_frame_mean) ** 2
        frame_loss = frame_loss * valid_frames.to(frame_loss.dtype)
        return frame_loss.sum(dim=(1, 2)) / valid_frames.sum(dim=(1, 2)).clamp_min(1)

    def compute_loss_with_epsilon_and_sigma(
        self,
        data_batch: dict[str, torch.Tensor],
        x0_from_data_batch: torch.Tensor,
        x0: torch.Tensor,
        condition,
        epsilon: torch.Tensor,
        sigma: torch.Tensor,
    ):
        output_batch, kendall_loss, pred_mse, edm_loss = super().compute_loss_with_epsilon_and_sigma(
            data_batch, x0_from_data_batch, x0, condition, epsilon, sigma
        )

        keep_mask = getattr(condition, "warp_keep_mask", None)
        if keep_mask is None or condition.data_type is None or not str(condition.data_type).endswith("VIDEO"):
            return output_batch, kendall_loss, pred_mse, edm_loss

        pred_x0 = output_batch["model_pred"].x0
        target_x0 = output_batch["x0"]
        if parallel_state.is_initialized() and parallel_state.get_context_parallel_world_size() > 1:
            keep_mask = split_inputs_cp(keep_mask, seq_dim=2, cp_group=parallel_state.get_context_parallel_group())

        brightness_loss = self._compute_brightness_stability_loss(pred_x0, target_x0, keep_mask)
        brightness_weight = getattr(self.config, "brightness_stability_weight", self.brightness_stability_weight)
        kendall_loss = kendall_loss + brightness_weight * brightness_loss.view(-1, 1)
        output_batch["brightness_stability_loss"] = brightness_loss.mean()
        output_batch["brightness_stability_weight"] = torch.tensor(
            brightness_weight, device=pred_x0.device, dtype=pred_x0.dtype
        )
        return output_batch, kendall_loss, pred_mse, edm_loss

    def get_data_and_condition(
        self, data_batch: dict, num_condition_t: Union[int, None] = None
    ) -> Tuple[Tensor, Tensor, Any]:
        raw_state, latent_state, condition = super().get_data_and_condition(data_batch, num_condition_t)

        if condition.data_type is not None and str(condition.data_type).endswith("VIDEO"):
            pixel_keep_mask = self._sample_warp_keep_mask(
                raw_state.shape[2], raw_state.shape[3], raw_state.shape[4], raw_state.device, raw_state.dtype
            )
            if pixel_keep_mask is None:
                keep_mask = torch.ones(
                    (1, 1, latent_state.shape[2], latent_state.shape[3], latent_state.shape[4]),
                    device=latent_state.device,
                    dtype=latent_state.dtype,
                )
            else:
                keep_mask = F.interpolate(pixel_keep_mask, size=latent_state.shape[2:], mode="nearest").to(
                    latent_state.dtype
                )
            warp_pose = self._encode_warp_from_video(raw_state, latent_state, keep=pixel_keep_mask)
            condition.condition_video_pose = warp_pose.contiguous()
            condition.warp_keep_mask = keep_mask.contiguous()

        return raw_state, latent_state, condition


@diffusion_fsdp_class_decorator
class FSDPPEFTGen3CExtendDiffusionModel(_LoRAGen3CExtendDiffusionModel):
    pass
