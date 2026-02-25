#!/usr/bin/env python3
"""
Prepare a tiny training dataset for temporal-completion pilot runs.

Layout created:
  datasets/temporal_completion_pilot/
    videos/<name>.mp4
    t5_xxl/<name>.pickle

The dataset loader expects one .pickle per video file containing a list where
item 0 is a [n_tokens, 1024] float32 array.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_video",
        default="assets/doer_video.mp4",
        help="Source video path.",
    )
    parser.add_argument(
        "--dataset_dir",
        default="datasets/temporal_completion_pilot",
        help="Output dataset root directory.",
    )
    parser.add_argument(
        "--video_name",
        default="doer_pilot",
        help="Basename for output .mp4 and .pickle.",
    )
    parser.add_argument(
        "--copy_video",
        action="store_true",
        help="Copy video instead of symlinking.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.input_video).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Input video not found: {src}")

    root = Path(args.dataset_dir).resolve()
    video_dir = root / "videos"
    t5_dir = root / "t5_xxl"
    video_dir.mkdir(parents=True, exist_ok=True)
    t5_dir.mkdir(parents=True, exist_ok=True)

    dst_video = video_dir / f"{args.video_name}.mp4"
    if dst_video.exists() or dst_video.is_symlink():
        dst_video.unlink()
    if args.copy_video:
        shutil.copy2(src, dst_video)
    else:
        os.symlink(src, dst_video)

    # Minimal valid embedding payload consumed by dataset_video.py:
    # pickle.load(...)[0] -> np.ndarray([n_tokens, 1024], float32)
    embedding = np.zeros((1, 1024), dtype=np.float32)
    dst_t5 = t5_dir / f"{args.video_name}.pickle"
    with open(dst_t5, "wb") as f:
        pickle.dump([embedding], f)

    print(f"Prepared pilot dataset at: {root}")
    print(f"Video: {dst_video}")
    print(f"T5 embedding: {dst_t5}")


if __name__ == "__main__":
    main()

