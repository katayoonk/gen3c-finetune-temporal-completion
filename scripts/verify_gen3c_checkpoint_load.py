#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Verify that GEN3C weights can be loaded by the training stack.

This script:
- Composes a training config (video2world LoRA experiment as a base)
- Overrides it to point at the GEN3C checkpoint
- Overrides the video-conditioned net input channel count to match GEN3C
- Instantiates trainer + model (with LoRA injected)
- Initializes optimizer/scheduler (LoRA params only)
- Loads the checkpoint through the checkpointer

It DOES NOT run any training steps or require any dataset to exist.
Run under torchrun (even for 1 GPU) because the trainer initializes NCCL.
"""

from __future__ import annotations

import argparse

import torch
import torch.distributed as dist

from cosmos_predict1.utils import log
from cosmos_predict1.utils.config_helper import get_config_module, override


def destroy_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except ValueError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/Gen3C-Cosmos-7B/model.pt",
        help="Path to GEN3C model checkpoint (.pt)",
    )
    parser.add_argument(
        "--net_in_channels",
        type=int,
        default=81,
        help=(
            "Base `model.net.in_channels` (excludes the optional concat_padding_mask channel). "
            "For GEN3C-Cosmos-7B weights, 81 works with the default net config where "
            "`concat_padding_mask=True` (effective channels become 82)."
        ),
    )
    args = parser.parse_args()

    # Base config entrypoint for diffusion training.
    config_file = "cosmos_predict1/diffusion/training/config/config.py"
    config_module = get_config_module(config_file)
    config = __import__(config_module, fromlist=["make_config"]).make_config()

    # Use the built-in video2world LoRA experiment as a starting point, but load GEN3C weights
    # and match GEN3C's expected input channel count.
    config = override(
        config,
        [
            "--",
            "experiment=video2world_7b_lora_example_cosmos_nemo_assets",
            f"checkpoint.load_path={args.checkpoint}",
            "model_parallel.context_parallel_size=1",
            f"model.net.in_channels={args.net_in_channels}",
            # Keep this lightweight; we are not training here.
            "checkpoint.save_iter=999999999",
            "checkpoint.async_saving=false",
        ],
    )

    config.validate()
    config.freeze()  # type: ignore

    # Import here so we reuse the same instantiation path as train.py
    from cosmos_predict1.diffusion.training.train import instantiate_model

    trainer = config.trainer.type(config)
    model = instantiate_model(config, trainer)
    model.on_model_init_end()

    # Mirror trainer.train() init order, but stop after checkpoint load.
    model = model.to("cuda", memory_format=config.trainer.memory_format)  # type: ignore
    model.on_train_start(config.trainer.memory_format)

    optimizer, scheduler = model.init_optimizer_scheduler(config.optimizer, config.scheduler)
    grad_scaler = torch.amp.GradScaler("cuda", **config.trainer.grad_scaler_args)

    log.info(f"Loading checkpoint via checkpointer: {args.checkpoint}")
    iteration = trainer.checkpointer.load(model, optimizer, scheduler, grad_scaler)
    log.success(f"✅ Checkpoint load completed (iteration={iteration}).")
    destroy_distributed()


if __name__ == "__main__":
    main()

