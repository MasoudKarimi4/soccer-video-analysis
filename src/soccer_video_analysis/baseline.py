#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time

from .runtime import validate_run, run_metadata, configure_ultralytics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Set

import cv2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a direct Ultralytics tracking baseline on a short video window.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--resize-width", type=int, default=960)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")
    parser.add_argument("--include-ball", action="store_true")
    parser.add_argument("--random-seed", type=int, default=0)
    return parser


def resize_by_width(frame, width: int):
    if width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w == width:
        return frame
    nh = int(round(h * (width / float(w))))
    return cv2.resize(frame, (width, nh), interpolation=cv2.INTER_AREA)


def main() -> None:
    args = build_parser().parse_args()
    validate_run(args)
    started = time.perf_counter()
    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    configure_ultralytics()
    from ultralytics import YOLO

    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    start_frame = max(0, int(round(args.start_sec * fps)))
    limit_frames = int(round(args.duration_sec * fps)) if args.duration_sec > 0 else 10**9
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = None
    processed = 0
    frame_idx = start_frame
    class_filter = [0, 32] if args.include_ball else [0]

    det_by_name: Counter[str] = Counter()
    unique_tracks: Dict[str, Set[int]] = defaultdict(set)

    while processed < limit_frames:
        ok, frame = cap.read()
        if not ok:
            break

        frame = resize_by_width(frame, args.resize_width)
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(args.output / "annotated_tracking.mp4"), fourcc, fps, (frame.shape[1], frame.shape[0]))
            if not writer.isOpened():
                cap.release()
                raise SystemExit("Could not create MP4 output; check codec and output permissions.")

        result = model.track(
            source=frame,
            persist=True,
            tracker=args.tracker,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            classes=class_filter,
            device=args.device,
            verbose=False,
        )[0]

        boxes = getattr(result, "boxes", None)
        if boxes is not None and boxes.cls is not None:
            cls_ids = boxes.cls.cpu().numpy().astype(int).tolist()
            track_ids = boxes.id.cpu().numpy().astype(int).tolist() if boxes.id is not None else []
            for idx, cls_id in enumerate(cls_ids):
                cls_name = result.names.get(cls_id, str(cls_id))
                det_by_name[cls_name] += 1
                if idx < len(track_ids):
                    unique_tracks[cls_name].add(track_ids[idx])

        plotted = result.plot()
        cv2.putText(
            plotted,
            f"Frame {frame_idx} | Model {args.model.name}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(plotted)

        processed += 1
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    if processed == 0:
        raise SystemExit("No frames processed; check the start time.")
    summary = {
        "input_video": str(args.video),
        "model": str(args.model),
        "start_sec": args.start_sec,
        "duration_sec": args.duration_sec,
        "frames_processed": processed,
        "metadata": run_metadata(args, fps, time.perf_counter() - started),
        "detections_by_class": {k: int(v) for k, v in det_by_name.items()},
        "unique_tracks_by_class": {k: len(v) for k, v in unique_tracks.items()},
        "tracker": args.tracker,
        "output_video": str(args.output / "annotated_tracking.mp4"),
    }
    (args.output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Done")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
