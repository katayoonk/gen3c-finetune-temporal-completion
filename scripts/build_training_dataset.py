#!/usr/bin/env python3
"""Orchestrate building the full temporal completion training dataset.

Runs the pipeline: download -> preprocess -> T5 embeddings -> validate.

Usage:
    PEXELS_API_KEY=<key> python scripts/build_training_dataset.py \
        --output_dir datasets/temporal_completion \
        --num_pexels 300
"""

import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Build training dataset end-to-end")
    parser.add_argument("--output_dir", type=str, default="datasets/temporal_completion")
    parser.add_argument("--num_pexels", type=int, default=300, help="Number of Pexels videos to download")
    parser.add_argument("--pexels_api_key", type=str, default=None)
    parser.add_argument(
        "--drone_dir", type=str,
        default="/home/ubuntu/world-models/GEN3C-Project/video_dataset",
        help="Path to DVC drone videos",
    )
    parser.add_argument("--skip_download", action="store_true", help="Skip Pexels download step")
    parser.add_argument("--skip_preprocess", action="store_true", help="Skip preprocessing step")
    parser.add_argument("--skip_t5", action="store_true", help="Skip T5 embedding generation")
    return parser.parse_args()


def run_cmd(cmd: list[str], desc: str):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    print(f"  > {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=os.getcwd())
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        return False
    return True


def validate_dataset(output_dir: str):
    """Validate the final dataset structure."""
    out = Path(output_dir)
    vid_dir = out / "videos"
    t5_dir = out / "t5_xxl"
    meta_dir = out / "metas"

    videos = sorted(vid_dir.glob("*.mp4")) if vid_dir.exists() else []
    t5_files = sorted(t5_dir.glob("*.pickle")) if t5_dir.exists() else []
    meta_files = sorted(meta_dir.glob("*.txt")) if meta_dir.exists() else []

    vid_stems = {v.stem for v in videos}
    t5_stems = {t.stem for t in t5_files}
    meta_stems = {m.stem for m in meta_files}

    matched = vid_stems & t5_stems
    missing_t5 = vid_stems - t5_stems
    missing_vid = t5_stems - vid_stems

    print(f"\n{'='*60}")
    print("  Dataset Validation")
    print(f"{'='*60}")
    print(f"  Videos:          {len(videos)}")
    print(f"  T5 embeddings:   {len(t5_files)}")
    print(f"  Caption files:   {len(meta_files)}")
    print(f"  Matched pairs:   {len(matched)}")

    if missing_t5:
        print(f"  Missing T5:      {len(missing_t5)}")
        for name in sorted(missing_t5)[:5]:
            print(f"    - {name}")
        if len(missing_t5) > 5:
            print(f"    ... and {len(missing_t5) - 5} more")

    if missing_vid:
        print(f"  Orphan T5:       {len(missing_vid)}")

    # Spot-check a few T5 files
    errors = 0
    for t5_path in list(t5_files)[:10]:
        try:
            with open(t5_path, "rb") as f:
                data = pickle.load(f)
            arr = data[0]
            assert isinstance(arr, np.ndarray), f"Expected ndarray, got {type(arr)}"
            assert arr.ndim == 2 and arr.shape[1] == 1024, f"Bad shape: {arr.shape}"
        except Exception as e:
            print(f"  Bad T5 file {t5_path.name}: {e}")
            errors += 1

    if errors == 0 and len(matched) > 0:
        print(f"\n  PASS — {len(matched)} video-embedding pairs ready for training")
    elif len(matched) > 0:
        print(f"\n  WARNING — {len(matched)} pairs found but {errors} T5 files have issues")
    else:
        print(f"\n  FAIL — no matched video-embedding pairs")

    return len(matched)


def main():
    args = parse_args()
    out = Path(args.output_dir)
    raw_pexels = out / "raw_pexels"
    api_key = args.pexels_api_key or os.environ.get("PEXELS_API_KEY")

    # Step 1: Download Pexels videos
    if not args.skip_download:
        if not api_key:
            print("WARNING: No Pexels API key provided. Skipping Pexels download.")
            print("  Set PEXELS_API_KEY env var or --pexels_api_key to enable.")
        else:
            run_cmd([
                sys.executable, "scripts/download_pexels_videos.py",
                "--output_dir", str(raw_pexels),
                "--num_videos", str(args.num_pexels),
                "--api_key", api_key,
            ], "Step 1: Download Pexels videos")

    # Step 2: Preprocess all videos
    if not args.skip_preprocess:
        preprocess_cmd = [
            sys.executable, "scripts/preprocess_dataset_videos.py",
            "--output_dir", str(out),
            "--min_frames", "25",
        ]
        if raw_pexels.exists():
            preprocess_cmd.extend(["--pexels_dir", str(raw_pexels)])
        drone = Path(args.drone_dir)
        if drone.exists() and any(drone.rglob("*.mp4")):
            preprocess_cmd.extend(["--drone_dir", str(drone)])
        run_cmd(preprocess_cmd, "Step 2: Preprocess videos")

    # Step 3: Generate T5 embeddings
    if not args.skip_t5:
        run_cmd([
            sys.executable, "scripts/get_t5_embeddings.py",
            "--dataset_path", str(out),
            "--cache_dir", "checkpoints",
        ], "Step 3: Generate T5-XXL embeddings")

    # Step 4: Validate
    n = validate_dataset(args.output_dir)
    if n > 0:
        print(f"\nDataset ready at: {args.output_dir}")
        print(f"To train: update DATASET_DIR in scripts/run_temporal_completion_lora_pilot.sh")


if __name__ == "__main__":
    main()
