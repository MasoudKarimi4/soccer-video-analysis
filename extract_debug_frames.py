#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract selected frames from a video.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resize-width", type=int, default=0)
    args = parser.parse_args()
    if any(frame < 0 for frame in args.frames) or args.resize_width < 0:
        parser.error("Frame indices and resize width must be nonnegative.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wanted = sorted(set(args.frames))
    if not wanted:
        return

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open {args.video}")

    wanted_set = set(wanted)
    frame_idx = 0
    while wanted_set:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in wanted_set:
            if args.resize_width > 0 and frame.shape[1] != args.resize_width:
                scale = args.resize_width / float(frame.shape[1])
                resized_h = int(round(frame.shape[0] * scale))
                frame = cv2.resize(frame, (args.resize_width, resized_h), interpolation=cv2.INTER_AREA)
            out_path = args.output_dir / f"frame_{frame_idx:04d}.png"
            if not cv2.imwrite(str(out_path), frame):
                cap.release()
                raise SystemExit(f"Could not write {out_path}")
            print(out_path)
            wanted_set.remove(frame_idx)
        frame_idx += 1

    cap.release()
    if wanted_set:
        raise SystemExit(f"Requested frames unavailable: {sorted(wanted_set)}")


if __name__ == "__main__":
    main()
