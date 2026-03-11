#!/usr/bin/env python3
"""Preprocess raw videos into the training dataset format.

Takes raw video directories (Pexels downloads, DVC drone clips, etc.) and
produces the standardized layout expected by the Cosmos training dataloader:

    datasets/temporal_completion/
    ├── videos/      *.mp4  (H.264, 30fps)
    ├── metas/       *.txt  (one-line caption per video)
    └── t5_xxl/      *.pickle  (generated separately)

Usage:
    python scripts/preprocess_dataset_videos.py \
        --pexels_dir datasets/temporal_completion/raw_pexels \
        --drone_dir /home/ubuntu/world-models/GEN3C-Project/video_dataset \
        --output_dir datasets/temporal_completion \
        --min_frames 25
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess videos for training")
    parser.add_argument("--pexels_dir", type=str, default=None, help="Directory with Pexels downloads")
    parser.add_argument("--drone_dir", type=str, default=None, help="Directory with drone video clips from DVC")
    parser.add_argument("--extra_dirs", type=str, nargs="*", default=[], help="Additional video directories")
    parser.add_argument("--output_dir", type=str, default="datasets/temporal_completion")
    parser.add_argument("--min_frames", type=int, default=25, help="Minimum frames required")
    parser.add_argument("--target_fps", type=int, default=30)
    parser.add_argument("--max_duration", type=float, default=30.0, help="Max duration in seconds (clip longer videos)")
    return parser.parse_args()


def _find_ffmpeg_bin(name: str) -> str:
    """Return path to ffprobe/ffmpeg, searching conda envs if not on PATH."""
    import shutil
    if shutil.which(name):
        return name
    # Common conda env location
    candidates = [
        f"/home/ubuntu/miniconda3/envs/cosmos-predict1/bin/{name}",
        f"/opt/conda/bin/{name}",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return name  # fall back, will fail with a clear error


def get_video_info(path: str) -> dict | None:
    """Get video metadata via ffprobe."""
    try:
        cmd = [
            _find_ffmpeg_bin("ffprobe"), "-v", "quiet", "-print_format", "json",
            "-show_streams", "-show_format", path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        info = json.loads(result.stdout)
        for s in info.get("streams", []):
            if s.get("codec_type") == "video":
                nb = int(s.get("nb_frames", 0) or 0)
                if nb == 0:
                    dur = float(info.get("format", {}).get("duration", 0))
                    fps_parts = s.get("r_frame_rate", "30/1").split("/")
                    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30
                    nb = int(dur * fps)
                return {
                    "frames": nb,
                    "width": int(s.get("width", 0)),
                    "height": int(s.get("height", 0)),
                    "fps": s.get("r_frame_rate", "30/1"),
                    "duration": float(info.get("format", {}).get("duration", 0)),
                    "codec": s.get("codec_name", ""),
                }
    except Exception:
        pass
    return None


def reencode_video(src: str, dst: str, target_fps: int = 30, max_duration: float | None = None) -> bool:
    """Re-encode video to H.264, target fps."""
    cmd = [_find_ffmpeg_bin("ffmpeg"), "-y", "-i", src, "-c:v", "libx264", "-preset", "fast", "-crf", "18",
           "-r", str(target_fps), "-an", "-pix_fmt", "yuv420p"]
    if max_duration:
        cmd.extend(["-t", str(max_duration)])
    cmd.append(dst)
    try:
        subprocess.run(cmd, capture_output=True, timeout=300, check=True)
        return True
    except Exception:
        return False


def collect_pexels(pexels_dir: Path) -> list[dict]:
    """Collect Pexels videos with their captions."""
    items = []
    if not pexels_dir or not pexels_dir.exists():
        return items

    meta_file = pexels_dir / "metadata" / "all_videos.json"
    caption_map = {}
    if meta_file.exists():
        with open(meta_file) as f:
            for entry in json.load(f):
                caption_map[entry["filename"]] = entry.get("caption", "a video clip")

    for mp4 in sorted(pexels_dir.glob("*.mp4")):
        caption = caption_map.get(mp4.name, "a video clip")
        items.append({"path": str(mp4), "caption": caption, "source": "pexels"})
    return items


def collect_drone(drone_dir: Path) -> list[dict]:
    """Collect drone videos from DVC dataset."""
    items = []
    if not drone_dir or not drone_dir.exists():
        return items

    for mp4 in sorted(drone_dir.rglob("*.mp4")):
        name = mp4.stem.replace("_", " ")
        caption = f"aerial drone footage of {name}"
        items.append({"path": str(mp4), "caption": caption, "source": "drone"})

    # Also look for image sequences that could be assembled
    # (drone_samples often has frames as JPGs — skip those, we need videos)
    return items


def collect_extra(dirs: list[str]) -> list[dict]:
    items = []
    for d in dirs:
        p = Path(d)
        if not p.exists():
            continue
        for mp4 in sorted(p.rglob("*.mp4")):
            items.append({"path": str(mp4), "caption": "a video clip", "source": "extra"})
    return items


def main():
    args = parse_args()
    out = Path(args.output_dir)
    vid_dir = out / "videos"
    meta_dir = out / "metas"
    vid_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Collect from all sources
    all_items = []
    if args.pexels_dir:
        pexels = collect_pexels(Path(args.pexels_dir))
        print(f"Pexels: {len(pexels)} videos found")
        all_items.extend(pexels)
    if args.drone_dir:
        drone = collect_drone(Path(args.drone_dir))
        print(f"Drone: {len(drone)} videos found")
        all_items.extend(drone)
    if args.extra_dirs:
        extra = collect_extra(args.extra_dirs)
        print(f"Extra: {len(extra)} videos found")
        all_items.extend(extra)

    print(f"Total raw: {len(all_items)} videos")

    accepted = 0
    skipped_short = 0
    skipped_err = 0

    for item in tqdm(all_items, desc="Processing"):
        src = item["path"]
        info = get_video_info(src)
        if info is None:
            skipped_err += 1
            continue

        # Check if long enough after re-encoding at target fps
        estimated_frames = int(info["duration"] * args.target_fps) if info["duration"] > 0 else info["frames"]
        if estimated_frames < args.min_frames:
            skipped_short += 1
            continue

        # Determine output name
        src_stem = Path(src).stem
        # Prefix with source to avoid name collisions
        if item["source"] == "pexels":
            out_name = src_stem  # already prefixed pexels_XXXX
        elif item["source"] == "drone":
            out_name = f"drone_{src_stem}"
        else:
            out_name = f"extra_{src_stem}"

        dst_video = vid_dir / f"{out_name}.mp4"
        dst_meta = meta_dir / f"{out_name}.txt"

        if dst_video.exists():
            # Already processed
            if not dst_meta.exists():
                with open(dst_meta, "w") as f:
                    f.write(item["caption"])
            accepted += 1
            continue

        clip_dur = min(args.max_duration, info["duration"]) if info["duration"] > 0 else None
        ok = reencode_video(src, str(dst_video), args.target_fps, clip_dur)
        if not ok:
            skipped_err += 1
            continue

        # Verify the output
        out_info = get_video_info(str(dst_video))
        if out_info is None or out_info["frames"] < args.min_frames:
            dst_video.unlink(missing_ok=True)
            skipped_short += 1
            continue

        with open(dst_meta, "w") as f:
            f.write(item["caption"])

        accepted += 1

    print(f"\nDone:")
    print(f"  Accepted: {accepted}")
    print(f"  Skipped (too short): {skipped_short}")
    print(f"  Skipped (errors): {skipped_err}")
    print(f"  Videos: {vid_dir}")
    print(f"  Captions: {meta_dir}")


if __name__ == "__main__":
    main()
