#!/usr/bin/env python3
"""Preprocess KTH Actions dataset: extract subsampled frames from subsequences.

For each video file, reads the subsequence boundaries from 00sequences.txt,
extracts evenly spaced frames per subsequence, and horizontally flips
right-to-left sequences (subsequences 1 and 3, i.e., indices 0 and 2) so
all motion is left-to-right.

Usage:
    python datasets/preprocess_kth.py
    python datasets/preprocess_kth.py --n-frames 10 --out-dir /path/to/output
    python datasets/preprocess_kth.py --limit 1   # process only 1 video (debug)
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

ACTIONS_OF_INTEREST = {"jogging", "running", "walking"}


def parse_sequences_file(seq_file: Path) -> dict[str, list[tuple[int, int]]]:
    """Parse 00sequences.txt -> {video_name: [(start, end), ...]} with 1-indexed frames."""
    log.info("Opening sequences file: %s", seq_file)
    sequences: dict[str, list[tuple[int, int]]] = {}
    with open(seq_file, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or "frames" not in line:
                continue
            match = re.match(r"^(\S+)\s+frames\s+(.+)$", line)
            if not match:
                raise ValueError(
                    f"Could not parse line {line_no} of {seq_file}: {line!r}"
                )
            video_name = match.group(1)
            ranges_str = match.group(2)
            ranges: list[tuple[int, int]] = []
            for r in ranges_str.split(","):
                r = r.strip()
                parts = r.split("-")
                if len(parts) != 2:
                    raise ValueError(
                        f"Could not parse range {r!r} on line {line_no}: {line!r}"
                    )
                start, end = int(parts[0]), int(parts[1])
                if start > end:
                    raise ValueError(
                        f"start > end in range {r!r} on line {line_no}: {line!r}"
                    )
                ranges.append((start, end))
            log.debug("  parsed: %s -> %s", video_name, ranges)
            sequences[video_name] = ranges
    log.info("Parsed %d total sequence entries", len(sequences))
    return sequences


def extract_all_frames(video_path: Path) -> np.ndarray:
    """Extract all frames from a video using ffmpeg -> (N, H, W, 3) uint8 array."""
    log.debug("  ffprobe: %s", video_path)
    probe_cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-show_entries",
        "stream=width,height,nb_frames",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    parts = result.stdout.strip().split(",")
    if len(parts) != 3:
        raise RuntimeError(
            f"Unexpected ffprobe output for {video_path}: {result.stdout!r}"
        )
    width, height, nb_frames = int(parts[0]), int(parts[1]), int(parts[2])
    log.debug("  video info: %dx%d, %d frames", width, height, nb_frames)

    # Decode all frames as raw RGB24
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-v",
        "quiet",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    raw = np.frombuffer(result.stdout, dtype=np.uint8)
    log.debug("  ffmpeg decoded %d bytes", raw.size)

    expected_size = nb_frames * height * width * 3
    if raw.size != expected_size:
        raise RuntimeError(
            f"Frame data size mismatch for {video_path}: "
            f"got {raw.size} bytes, expected {expected_size} "
            f"({nb_frames} frames x {height}x{width}x3)"
        )
    return raw.reshape(nb_frames, height, width, 3)


def subsample_frame_indices(start: int, end: int, n_frames: int) -> list[int]:
    """Return n_frames evenly spaced 1-indexed frame numbers from start to end (inclusive)."""
    if n_frames < 2:
        raise ValueError(f"n_frames must be >= 2, got {n_frames}")
    if end - start + 1 < n_frames:
        raise ValueError(
            f"Subsequence {start}-{end} has only {end - start + 1} frames, "
            f"cannot pick {n_frames} distinct frames"
        )
    indices = np.linspace(start, end, n_frames, dtype=int)
    return indices.tolist()


def parse_directions_file(directions_file: Path) -> dict[str, str]:
    """Parse start_direction.txt -> {video_name: "left" or "right"}.

    The file only lists videos that start going *left*; all others default
    to "right".  Lines look like ``person01_running_d4_uncomp.jpg: left``.
    The ``_uncomp.jpg`` suffix is stripped so keys match 00sequences.txt names.
    """
    directions: dict[str, str] = {}
    with open(directions_file, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(
                    f"Expected 'name: left|right' on line {line_no}, got: {line!r}"
                )
            raw_name, direction = line.split(":", 1)
            raw_name = raw_name.strip()
            direction = direction.strip().lower()
            if direction not in ("left", "right"):
                raise ValueError(
                    f"Direction must be 'left' or 'right' on line {line_no}, "
                    f"got {direction!r}"
                )
            # Strip _uncomp.jpg / _uncomp.avi / _uncomp suffixes
            name = re.sub(r"_uncomp(\.[a-zA-Z0-9]+)?$", "", raw_name)
            if name in directions:
                raise ValueError(f"Duplicate entry for {name!r} on line {line_no}")
            directions[name] = direction
    log.info("Parsed %d direction entries from %s", len(directions), directions_file)
    return directions


def mirror_indices_for_video(start_direction: str) -> set[int]:
    """Return the set of subsequence indices (0-3) that need horizontal flipping.

    We want everyone to go left-to-right.  Directions alternate each subseq.
    If start_direction=="right", the person already goes L->R in seq0,2
    so we mirror seq 1,3.  If "left", seq0,2 go R->L and need mirroring.
    """
    if start_direction == "left":
        return {0, 2}
    else:  # "right"
        return {1, 3}


def process_one_video(
    video_name: str,
    video_path: Path,
    subseqs: list[tuple[int, int]],
    out_dir: Path,
    n_frames: int,
    mirror_subseqs: set[int],
) -> int:
    """Process a single video: extract & save subsampled frames. Returns count saved."""
    if len(subseqs) != 4:
        raise ValueError(
            f"Expected exactly 4 subsequences for {video_name}, "
            f"got {len(subseqs)}: {subseqs}"
        )

    frames = extract_all_frames(video_path)
    n_video_frames = frames.shape[0]
    log.debug("  loaded %d frames, shape %s", n_video_frames, frames.shape)
    saved = 0

    for seq_idx, (start, end) in enumerate(subseqs):
        if start < 1:
            raise ValueError(f"{video_name} seq{seq_idx}: start frame {start} < 1")
        if end > n_video_frames:
            raise ValueError(
                f"{video_name} seq{seq_idx}: end frame {end} > "
                f"total frames {n_video_frames}"
            )

        sample_indices = subsample_frame_indices(start, end, n_frames)
        mirror = seq_idx in mirror_subseqs
        log.debug(
            "  seq%d: frames %d-%d, sampled %s, mirror=%s",
            seq_idx,
            start,
            end,
            sample_indices,
            mirror,
        )

        for frame_1idx in sample_indices:
            frame = frames[frame_1idx - 1]  # Convert to 0-indexed
            if mirror:
                frame = frame[:, ::-1, :].copy()  # Horizontal flip

            img = Image.fromarray(frame)
            out_name = f"{video_name}_seq{seq_idx}_frame{frame_1idx:04d}.png"
            img.save(out_dir / out_name)
            log.debug("    saved %s", out_name)
            saved += 1

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess KTH Actions dataset into per-subsequence frames"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("/projects/static2dynamic/datasets/KTH_Actions/raw_data"),
        help="Directory containing raw *_uncomp.avi files",
    )
    parser.add_argument(
        "--seq-file",
        type=Path,
        default=Path("/projects/static2dynamic/datasets/KTH_Actions/00sequences.txt"),
        help="Path to 00sequences.txt",
    )
    parser.add_argument(
        "--directions-file",
        type=Path,
        default=Path(
            "/projects/static2dynamic/datasets/KTH_Actions/start_direction.txt"
        ),
        help="Per-video start direction file (format: 'video_name: left|right')",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/projects/static2dynamic/datasets/KTH_Actions/processed"),
        help="Output directory for extracted frames",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=10,
        help="Number of frames to extract per subsequence (default: 10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only this many videos (0 = all). Useful for debugging.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Max parallel workers (0 = one per video, default: 8)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    # ------------------------------------------------------------------
    # 1. Parse sequence boundaries
    # ------------------------------------------------------------------
    log.info("Parsing sequences from %s ...", args.seq_file)
    all_sequences = parse_sequences_file(args.seq_file)

    # Keep only the actions we care about
    sequences = {
        name: ranges
        for name, ranges in all_sequences.items()
        if any(action in name for action in ACTIONS_OF_INTEREST)
    }
    log.info("%d entries for actions %s", len(sequences), ACTIONS_OF_INTEREST)

    # ------------------------------------------------------------------
    # 1b. Parse per-video start directions
    # ------------------------------------------------------------------
    log.info("Parsing directions from %s ...", args.directions_file)
    all_directions = parse_directions_file(args.directions_file)

    # Videos not listed default to "right"
    for name in sequences:
        if name not in all_directions:
            all_directions[name] = "right"
    n_left = sum(1 for d in all_directions.values() if d == "left")
    log.info(
        "Directions: %d left, %d right (of %d videos)",
        n_left,
        len(sequences) - n_left,
        len(sequences),
    )

    # ------------------------------------------------------------------
    # 2. Map sequence names to actual video files
    # ------------------------------------------------------------------
    # Files are named e.g. person01_jogging_d1_uncomp.avi
    # Sequence keys are   e.g. person01_jogging_d1
    video_files = sorted(args.raw_dir.glob("*.avi"))
    if not video_files:
        raise FileNotFoundError(f"No .avi files found in {args.raw_dir}")

    name_to_file: dict[str, Path] = {}
    for vf in video_files:
        name = vf.stem.replace("_uncomp", "")
        if name in name_to_file:
            raise RuntimeError(f"Duplicate video name after stripping _uncomp: {name}")
        name_to_file[name] = vf

    missing = sorted(set(sequences) - set(name_to_file))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} sequence entries have no matching video file:\n"
            + "\n".join(f"  {m}" for m in missing)
        )

    # ------------------------------------------------------------------
    # 3. Process each video (parallelised across videos)
    # ------------------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sorted_names = sorted(sequences)
    if args.limit > 0:
        sorted_names = sorted_names[: args.limit]
        log.info("--limit %d: processing only %s", args.limit, sorted_names)
    n_total = len(sorted_names)
    total_saved = 0

    n_workers = min(args.workers, n_total) if args.workers > 0 else n_total
    log.info("Processing %d videos with %d workers ...", n_total, n_workers)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                process_one_video,
                video_name,
                name_to_file[video_name],
                sequences[video_name],
                args.out_dir,
                args.n_frames,
                mirror_indices_for_video(all_directions[video_name]),
            ): video_name
            for video_name in sorted_names
        }
        for future in as_completed(futures):
            video_name = futures[future]
            saved = future.result()  # will raise if the worker raised
            total_saved += saved
            log.info("  %s -> %d frames", video_name, saved)

    log.info(
        "Done. Processed %d videos, saved %d frames to %s",
        n_total,
        total_saved,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
