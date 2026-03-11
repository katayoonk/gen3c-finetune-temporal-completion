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


def get_fa_ca_qv_lora_config(first_nblocks=28, rank=8, scale=1):
    """
    Get a LoRA configuration for the Self-Attention (FA) and Cross-Attention (CA) blocks in the model.
    This LoRA configuration is used to inject LoRA parameters into the model.

    Args:
        first_nblocks (int): The number of blocks to apply LoRA to.
        rank (int): The rank of the LoRA matrices.
    """
    blocks_regex = r"\b(" + "|".join([str(i) for i in range(first_nblocks)]) + r")\b"
    return dict(
        enabled=True,
        customization_type="LoRA",
        rank=rank,
        scale=scale,
        edits=[
            dict(
                blocks=blocks_regex,
                customization_type="LoRA",
                rank=rank,
                scale=scale,
                block_edit=[
                    "FA[to_q, to_v]",
                    "CA[to_q, to_v]",
                ],
            )
        ],
    )


def get_fa_ca_qkvo_lora_config(first_nblocks=28, rank=32, scale=1):
    """LoRA on all attention projections (Q/K/V/Out) in both FA and CA.

    ~4x more parameters than QV-only at the same rank, targeting the attention
    mechanism where the model must learn to remap zero-warp activations.
    """
    blocks_regex = r"\b(" + "|".join([str(i) for i in range(first_nblocks)]) + r")\b"
    return dict(
        enabled=True,
        customization_type="LoRA",
        rank=rank,
        scale=scale,
        edits=[
            dict(
                blocks=blocks_regex,
                customization_type="LoRA",
                rank=rank,
                scale=scale,
                block_edit=[
                    "FA[to_q, to_k, to_v, to_out]",
                    "CA[to_q, to_k, to_v, to_out]",
                ],
            )
        ],
    )


def get_full_lora_config(first_nblocks=28, rank=16, scale=1):
    """LoRA on all attention projections (Q/K/V/Out) plus MLP layers.

    Provides ~3x more trainable parameters than QV-only, giving the model
    enough capacity to compensate for x_embedder bias from zero warp input.
    """
    blocks_regex = r"\b(" + "|".join([str(i) for i in range(first_nblocks)]) + r")\b"
    return dict(
        enabled=True,
        customization_type="LoRA",
        rank=rank,
        scale=scale,
        edits=[
            dict(
                blocks=blocks_regex,
                customization_type="LoRA",
                rank=rank,
                scale=scale,
                block_edit=[
                    "FA[to_q, to_k, to_v, to_out]",
                    "CA[to_q, to_k, to_v, to_out]",
                    "MLP[l1, l2]",
                ],
            )
        ],
    )
