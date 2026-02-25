#!/usr/bin/env python3
"""
Create conditioning clips with temporal black-frame masking patterns.

Default behavior repeats a pattern across the entire source clip:
  real_block frames kept as-is, then black_block frames set to black.
This repeats until the video ends, preserving original length.

Examples:
  # Repeat 20 real / 20 black over full clip (same length as input)
  conda run -n cosmos-predict1 python scripts/make_black_conditioning_clip.py \
    --input_path assets/doer_video.mp4 \
    --output_path datasets/test_inputs/doer_same_len_repeat20real20black.mp4 \
    --num_real 20 --num_black 20 --mode repeat

  # Legacy behavior: take first N real frames then append M black frames
  conda run -n cosmos-predict1 python scripts/make_black_conditioning_clip.py \
    --input_path assets/demo_dynamic.gif \
    --output_path datasets/test_inputs/cond_3real_6black.mp4 \
    --num_real 3 --num_black 6 --start 0 --mode append_tail
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    # Backwards compatible flags (kept so earlier commands still work).
    parser.add_argument("--input_gif", default=None, help="(deprecated) Use --input_path instead.")
    parser.add_argument("--output_gif", default=None, help="(deprecated) Use --output_path instead.")
    parser.add_argument("--input_path", default=None, help="Path to input clip (.gif/.mp4)")
    parser.add_argument("--output_path", default=None, help="Path to output clip (.gif/.mp4)")
    parser.add_argument(
        "--mode",
        choices=["repeat", "append_tail", "single_gap_keep_tail"],
        default="repeat",
        help=(
            "repeat: real/black pattern across full clip; "
            "append_tail: first N real then M black; "
            "single_gap_keep_tail: keep full clip length but set one interval to black."
        ),
    )
    parser.add_argument(
        "--num_real",
        type=int,
        default=20,
        help="In repeat mode: number of real frames per cycle. In append_tail: number of real frames to copy.",
    )
    parser.add_argument(
        "--num_black",
        type=int,
        default=20,
        help="In repeat mode: number of black frames per cycle. In append_tail: number of black frames to append.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start frame index for append_tail mode")
    parser.add_argument("--gap_start", type=int, default=20, help="Start frame index for single_gap_keep_tail mode")
    parser.add_argument("--gap_len", type=int, default=4, help="Gap length for single_gap_keep_tail mode")
    parser.add_argument("--fps", type=float, default=24.0, help="FPS for mp4 output (and gif timing metadata)")
    parser.add_argument(
        "--make_black_unique",
        action="store_true",
        help="If set, add 1-pixel noise so repeated black frames don't get collapsed by GIF encoders.",
    )
    args = parser.parse_args()

    input_path = args.input_path or args.input_gif
    output_path = args.output_path or args.output_gif
    if not input_path or not output_path:
        raise SystemExit("Provide --input_path and --output_path (or legacy --input_gif/--output_gif).")

    import imageio.v3 as iio
    import imageio.v2 as iio2

    frames = iio.imread(input_path)  # (T,H,W,C) or (T,H,W)
    if frames.ndim == 3:
        frames = frames[..., None]
    if frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)

    if args.num_real <= 0 or args.num_black <= 0:
        raise ValueError("--num_real and --num_black must be > 0")

    if args.mode == "append_tail":
        start = args.start
        end = start + args.num_real
        if end > frames.shape[0]:
            raise ValueError(f"Input only has {frames.shape[0]} frames; need at least {end}.")
        real = frames[start:end]
        h, w = real.shape[1], real.shape[2]
        black = np.zeros((args.num_black, h, w, 3), dtype=real.dtype)
        if args.make_black_unique and args.num_black > 1:
            # Make each black frame slightly different so GIF writers don't collapse them.
            # Keep the perturbation invisible (one pixel set to 1).
            for i in range(args.num_black):
                black[i, i % h, i % w, :] = 1
        out = np.concatenate([real, black], axis=0)
    elif args.mode == "repeat":
        # Repeat pattern across full clip length: [real...][black...] cyclically.
        out = frames.copy()
        total = out.shape[0]
        cycle = args.num_real + args.num_black
        for s in range(0, total, cycle):
            b0 = s + args.num_real
            b1 = min(s + cycle, total)
            if b0 < total:
                out[b0:b1] = 0
        if args.make_black_unique:
            # Optional helper for GIF encoding stability.
            black_idx = np.where(out.mean(axis=(1, 2, 3)) < 1)[0]
            h, w = out.shape[1], out.shape[2]
            for i, t in enumerate(black_idx):
                out[t, i % h, i % w, :] = 1
    else:
        out = frames.copy()
        start = max(0, int(args.gap_start))
        end = min(out.shape[0], start + int(args.gap_len))
        if start >= out.shape[0] or end <= start:
            raise ValueError(f"Invalid gap range [{start}, {end}) for total frames {out.shape[0]}.")
        out[start:end] = 0
        if args.make_black_unique:
            h, w = out.shape[1], out.shape[2]
            for i, t in enumerate(range(start, end)):
                out[t, i % h, i % w, :] = 1

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    output_lower = output_path.lower()
    if output_lower.endswith(".mp4"):
        iio2.mimwrite(output_path, out, fps=float(args.fps), quality=8)
    elif output_lower.endswith(".gif"):
        duration = 1.0 / max(float(args.fps), 1e-6)
        # subrectangles=False helps avoid some encoder optimizations that can drop frames.
        iio2.mimsave(output_path, out, duration=duration, loop=0, subrectangles=False)
    else:
        raise ValueError("output_path must end with .mp4 or .gif")

    # Re-read to verify the saved frame count is what downstream inference will see.
    saved = iio.imread(output_path)
    saved_T = int(saved.shape[0]) if saved.ndim >= 4 else 1
    if args.mode == "append_tail":
        mode_msg = f"{args.num_real} real + {args.num_black} black (append_tail)"
    elif args.mode == "single_gap_keep_tail":
        mode_msg = f"single black gap [{args.gap_start}, {args.gap_start + args.gap_len}) keep tail"
    else:
        mode_msg = f"repeat pattern {args.num_real} real / {args.num_black} black"
    print(f"Wrote {output_path}: {mode_msg}. Frames written: {out.shape[0]}; on re-read: {saved_T}.")


if __name__ == "__main__":
    main()

