#!/usr/bin/env bash
set -euo pipefail

# LoRA fine-tuning for temporal completion on 8xA100-40GB.
#
# Base: video2world_7b_lora_example_cosmos_nemo_assets
#   - PEFTExtendDiffusionModel (LoRA rank=8 on first 28 DiT blocks)
#   - Loads Cosmos-Predict1-7B-Video2World checkpoint
#   - DDP + context_parallel_size for sequence splitting across GPUs
#
# Hardware: 8x NVIDIA A100-40GB
#   - context_parallel_size=8 → splits 121-frame sequence across all GPUs
#   - Each GPU holds the full model (~14GB bf16) + 1/8 of activations

NUM_GPUS="${NUM_GPUS:-8}"
MAX_ITER="${MAX_ITER:-1}"
SAVE_ITER="${SAVE_ITER:-1}"
DATASET_DIR="${DATASET_DIR:-datasets/temporal_completion_pilot}"

PYTHONPATH="$(pwd)" torchrun --nproc_per_node="${NUM_GPUS}" \
  -m cosmos_predict1.diffusion.training.train \
  --config cosmos_predict1/diffusion/training/config/config.py \
  -- \
  experiment=video2world_7b_lora_example_cosmos_nemo_assets \
  model_parallel.context_parallel_size="${NUM_GPUS}" \
  dataloader_train.dataset.dataset_dir="${DATASET_DIR}" \
  dataloader_train.sampler.dataset.dataset_dir="${DATASET_DIR}" \
  dataloader_val.dataset.dataset_dir="${DATASET_DIR}" \
  dataloader_val.sampler.dataset.dataset_dir="${DATASET_DIR}" \
  trainer.max_iter="${MAX_ITER}" \
  trainer.logging_iter=1 \
  checkpoint.save_iter="${SAVE_ITER}" \
  checkpoint.async_saving=false \
  scheduler.warm_up_steps=[0]
