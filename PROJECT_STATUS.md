# Temporal Completion Project Status

This document summarizes the GEN3C temporal completion project from the first fine-tuning attempt through the current Fix 3 conditioning-repair direction. It is meant as a lightweight record of what was tried, what worked, what failed, and what should happen next.

## Project Goal

GEN3C is a 3D-aware video generation model that conditions diffusion on rendered warp images and masks from a 3D cache. The project studies temporal video completion when some conditioning frames are fully missing, such as repeated 20-frame gaps in a 121-frame sequence.

The original failure mode was severe dimming or contrast collapse inside missing spans. Over time, the evidence suggested that this was not only a lack of fine-tuning capacity. The stronger explanation is a conditioning mismatch: empty, black, or incorrectly encoded warp conditioning is out of distribution for GEN3C's native rendering pipeline.

## High-Level Timeline

1. Reproduced the dimming artifact with controlled real/black conditioning clips.
2. Built a LoRA fine-tuning pipeline for GEN3C temporal completion.
3. Created training data from Pexels/web videos and generated T5 embeddings.
4. Ran multiple LoRA experiments to test whether the model could learn to fill missing spans.
5. Evaluated the best LoRA checkpoints on held-out drone and Pexels videos.
6. Investigated why LoRA did not generalize cleanly.
7. Shifted to Fix 3: repair missing conditioning at inference time before it reaches GEN3C.

## Fine-Tuning Setup

The fine-tuning direction used parameter-efficient LoRA adaptation instead of full model fine-tuning. The goal was to teach GEN3C to handle repeated missing conditioning spans while preserving the model's original 3D-aware generation behavior.

The training pipeline included scripts for downloading Pexels clips, preprocessing videos, building a temporal completion dataset, generating T5 embeddings, running LoRA training, and merging LoRA checkpoints back into GEN3C weights for inference.

The important design question was how to represent missing frames. Early attempts treated missing spans too much like zeroed or masked latent inputs, but GEN3C expects rendered 3D-cache conditioning with meaningful image and mask semantics. This mismatch became the central lesson of the project.

## LoRA Experiments Tried

- Initial sanity and pilot runs verified that the GEN3C checkpoint could load in the training stack and that LoRA training could run.
- Early runs tested small training sets and different LoRA settings, including warp-only or cross-embedding variants.
- Larger runs used more Pexels clips and repeated `gap20` style missing spans.
- `run15` became the strongest completed fine-tuning result because it represented missing frames in pixel space before VAE encoding, which better matched GEN3C's original conditioning semantics.
- `run17` started from the stronger `run15` direction and added a brightness-stability loss to reduce pulsing or over-brightening, but it did not improve the results.

## Results From Fine-Tuning

- The base model often dimmed, lost contrast, or collapsed when conditioning frames were fully missing.
- Early LoRA runs helped in some cases but did not reliably generalize to new videos.
- `run15` was the best fine-tuned checkpoint. It reduced the catastrophic dimming artifact better than earlier attempts and was visually preferred on the main drone and web-video comparisons.
- `run15` still had artifacts, especially brightness overshoot or pulsing in some missing spans.
- `run17` was a useful negative result. The brightness-stability loss did not fix the pulsing and often made over-brightening worse on held-out examples.

## Main Interpretation

Fine-tuning GEN3C with LoRA for frame filling is hard because the missing-frame problem conflicts with how GEN3C expects to use its 3D cache. GEN3C is not a standard video inpainting model; it expects rendered geometry-aware conditioning, not arbitrary masked or empty frames.

The project suggests that the most important issue is preserving the semantics of the structured conditioning interface. A larger LoRA rank, longer training, or stronger auxiliary loss may help somewhat, but they do not fully solve the problem if the input conditioning itself is unnatural for the model.

## Fix 3: Conditioning Repair

Fix 3 changes the inference input rather than immediately fine-tuning again. The idea is to repair missing rendered warp tensors before they are passed into GEN3C, so the model does not receive completely empty conditioning frames.

The current implementation is in `cosmos_predict1/diffusion/inference/diffusion_only.py` and adds:

```bash
--conditioning_fill_mode none
--conditioning_fill_mode copy_forward
--conditioning_fill_mode interpolate
```

The code detects missing conditioning frames using `rendered_warp_masks`, then repairs both `rendered_warp_images` and `rendered_warp_masks`.

## Fix 3 Results So Far

- `copy_forward` fills a missing frame with the nearest previous valid conditioning frame. It removed obvious dimming, but it caused a pause/freeze artifact because the camera or scene conditioning stopped moving during the missing span.
- `interpolate` linearly blends between the previous and next valid conditioning frames. This is the current preferred direction because it tries to avoid both empty conditioning and copy-forward motion pauses.
- `pexels_13831205` with `run15 + copy_forward` reduced dimming but showed pauses.
- `pexels_14488533` city street was tested with base and LoRA variants. Copy-forward again reduced dimming but introduced motion pauses; interpolation was tested as a better alternative.
- `pexels_34762049` was first tested incorrectly using a single-frame rendering pipeline, which made the result mostly static.
- The corrected `pexels_34762049` video-input pipeline extracted all frames, generated video-based normal tensors, and created `gap20` tensors by zeroing frames `20:40`, `60:80`, and `100:120`.
- As of the latest check, the corrected `pexels_34762049` base + interpolate inference was no longer running and the final `.mp4` output was not present, so that run should be rerun or debugged before reporting it as a completed result.

## Current GitHub Push Plan

Good candidates to commit:

- `PROJECT_STATUS.md`
- `cosmos_predict1/diffusion/inference/diffusion_only.py`
- Small source scripts needed to reproduce the experiments, if present and intentionally changed
- `.gitignore`, but only after restoring it because it currently appears deleted in the working tree

Do not commit:

- `checkpoints/`
- `checkpoints_eval_*`
- `outputs/`
- `report/`
- `datasets/temporal_completion/` raw videos and generated tensors
- `logs/`
- `*.log`
- `__pycache__/`
- local IDE files such as `.vscode/`
- downloaded external repos such as `apex/`

## Current Git Warning

Several tracked files currently appear deleted in the working tree:

- `.gitignore`
- `scripts/build_training_dataset.py`
- `scripts/download_pexels_videos.py`
- `scripts/get_t5_embeddings.py`
- `scripts/merge_lora_checkpoint.py`
- `scripts/preprocess_dataset_videos.py`
- `scripts/run_temporal_completion_lora_pilot.sh`

These files were already part of the GitHub-tracked project. Unless the goal is to remove them from the repository, they should be restored before committing the new status doc and inference change.

## To Do

- Restore the accidentally deleted tracked files before making the next commit.
- Rerun or debug the corrected `pexels_34762049` video-input base + interpolate inference and save the output video.
- Add a small reproducible script or documented command for creating `gap20` tensors from normal rendered tensors.
- Test interpolation on at least 3-4 held-out videos, including the drone clip and Pexels clips not used during training.
- Compare base, `run15`, and `run17` under the same conditioning-fill settings.
- Add a stronger Fix 3 method after linear interpolation, such as bidirectional blending, optical-flow propagation, or camera-aware reprojection.
- If Fix 3 consistently helps, consider fine-tuning with repaired conditioning rather than empty conditioning so training and inference match.
