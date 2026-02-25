#!/usr/bin/env bash
set -euo pipefail

# Tiny LoRA pilot for temporal completion behavior.
# - Uses Video2World LoRA baseline
# - Loads GEN3C checkpoint
# - Switches conditioning pattern to interpolation-like first_and_last_1
# - Runs a minimal number of iterations to verify the stack end-to-end

PYTHONPATH="$(pwd)" torchrun --nproc_per_node=1 -m cosmos_predict1.diffusion.training.train \
  --config cosmos_predict1/diffusion/training/config/config.py \
  -- \
  experiment=video2world_7b_lora_example_cosmos_nemo_assets \
  model_parallel.context_parallel_size=1 \
  dataloader_train.dataset.dataset_dir=datasets/temporal_completion_pilot \
  dataloader_train.sampler.dataset.dataset_dir=datasets/temporal_completion_pilot \
  dataloader_val.dataset.dataset_dir=datasets/temporal_completion_pilot \
  dataloader_val.sampler.dataset.dataset_dir=datasets/temporal_completion_pilot \
  trainer.max_iter=1 \
  trainer.logging_iter=1 \
  checkpoint.save_iter=1 \
  checkpoint.async_saving=false \
  scheduler.warm_up_steps=[0]
