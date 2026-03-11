#!/usr/bin/env bash
set -euo pipefail

# Gen3C LoRA fine-tuning for temporal completion on 8xA100-40GB.
#
# Trains on the Gen3C checkpoint (27GB) with synthetic warp conditioning.
# During training, warp frames are randomly zeroed-out (scattered + contiguous
# block patterns) to teach the model that zero warp = missing info, not dark.
# LoRA rank=16 on first 28 DiT blocks, condition_zero_out_rate=0.8.
# 384x384 resolution, 25 frames to fit in 40GB VRAM.

NUM_GPUS="${NUM_GPUS:-8}"
MAX_ITER="${MAX_ITER:-5000}"
SAVE_ITER="${SAVE_ITER:-1000}"
DATASET_DIR="${DATASET_DIR:-datasets/temporal_completion}"
CKPT_PATH="${CKPT_PATH:-checkpoints/Gen3C-Cosmos-7B/model.pt}"

PYTHONPATH="$(pwd)" torchrun --nproc_per_node="${NUM_GPUS}" \
  -m cosmos_predict1.diffusion.training.train \
  --config cosmos_predict1/diffusion/training/config/config.py \
  -- \
  experiment=video2world_7b_lora_8gpu_40gb \
  checkpoint.load_path="${CKPT_PATH}" \
  dataloader_train.dataset.dataset_dir="${DATASET_DIR}" \
  dataloader_train.sampler.dataset.dataset_dir="${DATASET_DIR}" \
  trainer.max_iter="${MAX_ITER}" \
  trainer.logging_iter=1 \
  checkpoint.save_iter="${SAVE_ITER}" \
  checkpoint.async_saving=false
