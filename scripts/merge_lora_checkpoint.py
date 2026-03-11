#!/usr/bin/env python3
"""
Merge LoRA delta weights into the base model weights so the resulting
checkpoint can be loaded by the standard Gen3C inference pipeline
(which does not know about LoRA layers).

LoRA structure used in this codebase:
  attn.to_q        → base linear weight at  `{prefix}.to_q.0.weight`
  attn.to_q_lora   → LoRA net stored at
                       `{prefix}.to_q_lora.net.0.weight`  (down, [rank, in])
                       `{prefix}.to_q_lora.net.1.weight`  (up,   [out, rank])

Merge formula: base_weight  +=  scale * (up @ down)
"""

import argparse
import re
import torch
from collections import OrderedDict


LORA_SCALE = 1.0  # matches lora_scale=1 set in the training experiment config


def find_lora_groups(state_dict):
    """
    Return a dict mapping  lora_prefix → base_prefix  for every LoRA group.

    e.g.  "net.blocks.block0...attn.to_q_lora"
        → "net.blocks.block0...attn.to_q"
    """
    down_pattern = re.compile(r"^(.+_lora)\.net\.0\.weight$")
    groups = {}
    for key in state_dict:
        m = down_pattern.match(key)
        if m:
            lora_prefix = m.group(1)                     # e.g. ...to_q_lora
            base_prefix = lora_prefix[: -len("_lora")]   # e.g. ...to_q
            groups[lora_prefix] = base_prefix
    return groups


def extract_null_warp(state_dict):
    """Extract null_warp_embedding from state dict if present."""
    key = "null_warp.embedding"
    if key in state_dict:
        return state_dict[key].clone()
    return None


def merge(state_dict, scale=LORA_SCALE):
    merged = OrderedDict()

    # Collect all LoRA keys to skip later
    lora_prefixes = find_lora_groups(state_dict)
    lora_keys_to_skip = set()
    for lp in lora_prefixes:
        lora_keys_to_skip.add(f"{lp}.net.0.weight")
        lora_keys_to_skip.add(f"{lp}.net.1.weight")

    # Keys to strip from merged output (training-only parameters)
    training_only_keys = {"null_warp_embedding", "null_warp.embedding"}

    # Copy non-LoRA keys as-is
    for key, value in state_dict.items():
        if key not in lora_keys_to_skip and key not in training_only_keys:
            merged[key] = value.clone() if isinstance(value, torch.Tensor) else value

    # Apply LoRA deltas
    n_merged = 0
    for lora_prefix, base_prefix in lora_prefixes.items():
        down_key = f"{lora_prefix}.net.0.weight"   # [rank, in_features]
        up_key   = f"{lora_prefix}.net.1.weight"   # [out_features, rank]
        base_key = f"{base_prefix}.0.weight"        # [out_features, in_features]

        if base_key not in merged:
            print(f"  [WARN] base key not found: {base_key}  (skipping)")
            continue

        down = state_dict[down_key].float()   # [rank, in]
        up   = state_dict[up_key].float()     # [out, rank]
        delta = scale * (up @ down)           # [out, in]

        merged[base_key] = (merged[base_key].float() + delta).to(merged[base_key].dtype)
        n_merged += 1

    print(f"Merged {n_merged} LoRA groups into base weights.")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base Gen3C checkpoint")
    parser.add_argument(
        "--lora_checkpoint",
        required=True,
        help="Path to the *_reg_model.pt file saved during LoRA training",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the merged checkpoint (will be named model.pt or similar)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=LORA_SCALE,
        help=f"LoRA scale factor (default: {LORA_SCALE})",
    )
    args = parser.parse_args()

    print(f"Loading LoRA checkpoint from: {args.lora_checkpoint}")
    raw = torch.load(args.lora_checkpoint, map_location="cpu", weights_only=False)

    # Handle both bare state-dicts and nested {'model': ...} wrappers
    if isinstance(raw, dict) and "model" in raw:
        state_dict = raw["model"]
        print("  Detected nested state_dict under 'model' key.")
    else:
        state_dict = raw

    lora_count = sum(1 for k in state_dict if "lora" in k.lower())
    base_count  = len(state_dict) - lora_count
    print(f"  Base keys: {base_count},  LoRA keys: {lora_count}")

    null_warp = extract_null_warp(state_dict)
    if null_warp is not None:
        null_warp_path = args.output.replace(".pt", "_null_warp.pt")
        print(f"Extracted null_warp_embedding: shape {list(null_warp.shape)}, "
              f"mean={null_warp.float().mean().item():.6f}, std={null_warp.float().std().item():.6f}")
        torch.save(null_warp, null_warp_path)
        print(f"Saved null_warp_embedding to: {null_warp_path}")

    merged = merge(state_dict, scale=args.scale)

    remaining_lora = [k for k in merged if "lora" in k.lower()]
    if remaining_lora:
        print(f"[WARN] {len(remaining_lora)} LoRA keys still in merged dict — check patterns.")
    else:
        print("All LoRA keys successfully merged and removed.")

    print(f"Saving merged checkpoint to: {args.output}")
    torch.save(merged, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
