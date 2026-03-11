#!/usr/bin/env python3
"""Download diverse videos for training data from multiple sources.

Supports:
  1. Pexels API (requires PEXELS_API_KEY env var or --api_key)
  2. YouTube search via yt-dlp (no key needed, slower)

Usage:
    # With Pexels API key:
    PEXELS_API_KEY=<key> python scripts/download_pexels_videos.py \
        --output_dir datasets/temporal_completion/raw_pexels --num_videos 300

    # Without API key (uses yt-dlp):
    python scripts/download_pexels_videos.py \
        --output_dir datasets/temporal_completion/raw_pexels --num_videos 300 --source youtube
"""

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

SEARCH_QUERIES = [
    "nature landscape",
    "city street",
    "ocean waves",
    "forest trees",
    "mountain scenery",
    "sunset sky",
    "river flowing",
    "snow winter",
    "desert sand",
    "beach tropical",
    "traffic cars",
    "people walking",
    "rain storm",
    "clouds timelapse",
    "flowers garden",
    "birds flying",
    "underwater fish",
    "waterfall",
    "field grass",
    "night city lights",
    "drone aerial view",
    "highway road",
    "building architecture",
    "park outdoor",
    "lake reflection",
    "autumn leaves",
    "sports action",
    "cooking food",
    "dog animal",
    "cat pet",
    "factory industrial",
    "train railway",
    "boat sailing",
    "horse running",
    "fireworks celebration",
    "campfire flame",
    "bridge infrastructure",
    "market crowd",
    "airport plane",
    "sunrise morning",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Download videos for training data")
    parser.add_argument("--output_dir", type=str, default="datasets/temporal_completion/raw_pexels")
    parser.add_argument("--num_videos", type=int, default=300)
    parser.add_argument("--api_key", type=str, default=None, help="Pexels API key")
    parser.add_argument("--source", type=str, default="auto", choices=["auto", "pexels", "youtube"],
                        help="Video source. 'auto' uses Pexels if key available, else YouTube.")
    parser.add_argument("--min_duration", type=int, default=3, help="Min video duration in seconds")
    parser.add_argument("--max_duration", type=int, default=30, help="Max video duration in seconds")
    parser.add_argument("--workers", type=int, default=4, help="Parallel download threads")
    return parser.parse_args()


# ──────────────────── Pexels API ────────────────────

def fetch_pexels_videos(api_key: str, query: str, per_page: int = 40, page: int = 1) -> list:
    url = "https://api.pexels.com/videos/search"
    headers = {"Authorization": api_key}
    params = {"query": query, "per_page": per_page, "page": page, "orientation": "landscape"}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("videos", [])


def pick_best_pexels_file(video_info: dict) -> dict | None:
    files = video_info.get("video_files", [])
    if not files:
        return None
    candidates = sorted(files, key=lambda f: abs((f.get("height") or 0) - 720))
    for c in candidates:
        if c.get("link") and c.get("file_type") == "video/mp4":
            return c
    return candidates[0] if candidates else None


def download_file(url: str, dest: Path, timeout: int = 120) -> bool:
    if dest.exists():
        return True
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        tmp.rename(dest)
        return True
    except Exception as e:
        print(f"  Failed {dest.name}: {e}")
        if dest.with_suffix(".tmp").exists():
            dest.with_suffix(".tmp").unlink()
        return False


def download_pexels(args, api_key: str, out_dir: Path, meta_dir: Path):
    videos_per_query = max(1, args.num_videos // len(SEARCH_QUERIES)) + 2
    seen_ids = set()
    download_queue = []

    print(f"Searching Pexels for {args.num_videos} videos across {len(SEARCH_QUERIES)} queries...")

    for query in tqdm(SEARCH_QUERIES, desc="Querying Pexels"):
        pages_needed = (videos_per_query + 39) // 40
        for page in range(1, pages_needed + 1):
            try:
                results = fetch_pexels_videos(api_key, query, per_page=40, page=page)
            except Exception as e:
                print(f"  Query '{query}' page {page} failed: {e}")
                time.sleep(1)
                continue

            for v in results:
                vid = v["id"]
                dur = v.get("duration", 0)
                if vid in seen_ids or dur < args.min_duration or dur > args.max_duration:
                    continue
                best = pick_best_pexels_file(v)
                if not best or not best.get("link"):
                    continue
                seen_ids.add(vid)
                caption = v.get("url", "").split("/")[-1].replace("-", " ").rsplit(" ", 1)[0]
                if not caption:
                    caption = query
                download_queue.append({
                    "id": vid, "url": best["link"],
                    "filename": f"pexels_{vid}.mp4", "caption": caption,
                    "query": query, "duration": dur,
                    "width": best.get("width"), "height": best.get("height"),
                })
                if len(download_queue) >= args.num_videos:
                    break
            if len(download_queue) >= args.num_videos:
                break
            time.sleep(0.2)
        if len(download_queue) >= args.num_videos:
            break

    print(f"Found {len(download_queue)} videos to download")
    with open(meta_dir / "all_videos.json", "w") as f:
        json.dump(download_queue, f, indent=2)

    success, failed = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for item in download_queue:
            dest = out_dir / item["filename"]
            fut = pool.submit(download_file, item["url"], dest)
            futures[fut] = item
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            item = futures[fut]
            if fut.result():
                success += 1
                with open(meta_dir / item["filename"].replace(".mp4", ".txt"), "w") as f:
                    f.write(item["caption"])
            else:
                failed += 1

    print(f"Pexels: {success} downloaded, {failed} failed")
    return success


# ──────────────────── YouTube via yt-dlp ────────────────────

def download_youtube(args, out_dir: Path, meta_dir: Path):
    """Download short Creative Commons videos from YouTube using yt-dlp."""
    videos_per_query = max(1, args.num_videos // len(SEARCH_QUERIES)) + 3
    total_downloaded = 0

    print(f"Downloading from YouTube via yt-dlp ({args.num_videos} target)...")

    for query in tqdm(SEARCH_QUERIES, desc="Searching YouTube"):
        if total_downloaded >= args.num_videos:
            break

        search_term = f"ytsearch{videos_per_query}:{query} short video"
        try:
            cmd = [
                "yt-dlp", "--flat-playlist", "--print", "%(id)s\t%(title)s\t%(duration)s",
                "--no-warnings", search_term,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
        except Exception as e:
            print(f"  Search failed for '{query}': {e}")
            continue

        for line in lines:
            if total_downloaded >= args.num_videos:
                break
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            vid_id, title, dur_str = parts[0], parts[1], parts[2]
            try:
                dur = float(dur_str)
            except (ValueError, TypeError):
                continue
            if dur < args.min_duration or dur > args.max_duration:
                continue

            outfile = out_dir / f"yt_{vid_id}.mp4"
            caption_file = meta_dir / f"yt_{vid_id}.txt"
            if outfile.exists():
                total_downloaded += 1
                continue

            try:
                dl_cmd = [
                    "yt-dlp", "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
                    "--merge-output-format", "mp4",
                    "-o", str(outfile),
                    "--no-playlist", "--no-warnings", "--quiet",
                    "--socket-timeout", "30",
                    f"https://www.youtube.com/watch?v={vid_id}",
                ]
                subprocess.run(dl_cmd, timeout=120, check=True, capture_output=True)
                if outfile.exists():
                    with open(caption_file, "w") as f:
                        f.write(title.strip() or query)
                    total_downloaded += 1
            except Exception:
                outfile.unlink(missing_ok=True)
                continue

    print(f"YouTube: {total_downloaded} downloaded")
    return total_downloaded


# ──────────────────── Main ────────────────────

def main():
    args = parse_args()
    api_key = args.api_key or os.environ.get("PEXELS_API_KEY")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = out_dir / "metadata"
    meta_dir.mkdir(exist_ok=True)

    source = args.source
    if source == "auto":
        source = "pexels" if api_key else "youtube"
        print(f"Auto-selected source: {source}")

    if source == "pexels":
        if not api_key:
            raise ValueError("Pexels API key required. Set --api_key or PEXELS_API_KEY env var.")
        n = download_pexels(args, api_key, out_dir, meta_dir)
    else:
        n = download_youtube(args, out_dir, meta_dir)

    print(f"\nTotal: {n} videos saved to {out_dir}")


if __name__ == "__main__":
    main()
