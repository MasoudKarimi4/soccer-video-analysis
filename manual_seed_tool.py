#!/usr/bin/env python3
"""
Interactive tool for creating one manual calibration seed frame.

Controls:
- r: draw red team boxes
- w: draw light team boxes
- b: draw ball box
- u: undo last annotation
- c: clear all annotations
- s: save and exit
- q or Esc: quit without saving
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2


BBox = Tuple[int, int, int, int]


def infer_default_video() -> Path:
    mp4s = sorted(Path(".").glob("*.mp4"))
    mp4s = [p for p in mp4s if "annotated" not in p.name.lower()]
    return mp4s[0] if mp4s else Path("soccer_footage.mp4")


def resize_by_width(frame, width: int):
    if width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w == width:
        return frame
    new_h = int(round(h * (width / float(w))))
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)


def box_from_points(p0: Tuple[int, int], p1: Tuple[int, int], frame_w: int, frame_h: int) -> BBox:
    x1 = max(0, min(frame_w - 1, min(p0[0], p1[0])))
    y1 = max(0, min(frame_h - 1, min(p0[1], p1[1])))
    x2 = max(0, min(frame_w - 1, max(p0[0], p1[0])))
    y2 = max(0, min(frame_h - 1, max(p0[1], p1[1])))
    w = max(1, x2 - x1 + 1)
    h = max(1, y2 - y1 + 1)
    return (x1, y1, w, h)


def read_frame(video_path: Path, frame_index: int, resize_width: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read frame {frame_index} from {video_path}")
    return resize_by_width(frame, resize_width), fps


@dataclass
class SeedState:
    mode: str = "red"
    red_boxes: List[BBox] = field(default_factory=list)
    light_boxes: List[BBox] = field(default_factory=list)
    ball_box: Optional[BBox] = None
    actions: List[Tuple[str, Optional[BBox]]] = field(default_factory=list)
    drawing: bool = False
    drag_start: Optional[Tuple[int, int]] = None
    preview_box: Optional[BBox] = None

    def add_box(self, kind: str, box: BBox) -> None:
        if kind == "red":
            self.red_boxes.append(box)
        elif kind == "light":
            self.light_boxes.append(box)
        elif kind == "ball":
            prev = self.ball_box
            self.ball_box = box
            self.actions.append(("ball_replace", prev))
            return
        self.actions.append((kind, box))

    def undo(self) -> None:
        if not self.actions:
            return
        kind, payload = self.actions.pop()
        if kind == "red" and self.red_boxes:
            self.red_boxes.pop()
        elif kind == "light" and self.light_boxes:
            self.light_boxes.pop()
        elif kind == "ball_replace":
            self.ball_box = payload

    def clear(self) -> None:
        self.red_boxes.clear()
        self.light_boxes.clear()
        self.ball_box = None
        self.actions.clear()


def draw_annotations(base_frame, state: SeedState):
    frame = base_frame.copy()

    for box in state.red_boxes:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (30, 30, 230), 2)
        cv2.putText(frame, "RED", (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 230), 1, cv2.LINE_AA)

    for box in state.light_boxes:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (240, 240, 240), 2)
        cv2.putText(frame, "LIGHT", (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

    if state.ball_box is not None:
        x, y, w, h = state.ball_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, "BALL", (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    if state.preview_box is not None:
        x, y, w, h = state.preview_box
        color = (0, 255, 255) if state.mode == "ball" else (30, 30, 230) if state.mode == "red" else (240, 240, 240)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)

    instructions = [
        f"Mode: {state.mode.upper()}",
        "r=red  w=light  b=ball",
        "u=undo  c=clear  s=save  q=quit",
        "Drag boxes around a few players and one ball box",
    ]
    y = 24
    for line in instructions:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        y += 24

    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a manual seed JSON for soccer tracking calibration.")
    parser.add_argument("--video", type=Path, default=infer_default_video())
    parser.add_argument("--output", type=Path, default=Path("manual_seed.json"))
    parser.add_argument("--frame-sec", type=float, default=3.7)
    parser.add_argument("--frame-index", type=int, default=-1)
    parser.add_argument("--resize-width", type=int, default=960)
    args = parser.parse_args()

    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    cap.release()

    frame_index = args.frame_index if args.frame_index >= 0 else int(round(args.frame_sec * fps))
    frame, fps = read_frame(args.video, frame_index, args.resize_width)
    frame_h, frame_w = frame.shape[:2]

    state = SeedState()
    window_name = "Manual Seed Tool"

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state.drawing = True
            state.drag_start = (x, y)
            state.preview_box = (x, y, 1, 1)
        elif event == cv2.EVENT_MOUSEMOVE and state.drawing and state.drag_start is not None:
            state.preview_box = box_from_points(state.drag_start, (x, y), frame_w, frame_h)
        elif event == cv2.EVENT_LBUTTONUP and state.drawing and state.drag_start is not None:
            state.drawing = False
            box = box_from_points(state.drag_start, (x, y), frame_w, frame_h)
            if state.mode == "ball" and box[2] <= 3 and box[3] <= 3:
                cx, cy = x, y
                box = box_from_points((cx - 6, cy - 6), (cx + 6, cy + 6), frame_w, frame_h)
            state.add_box(state.mode, box)
            state.drag_start = None
            state.preview_box = None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        display = draw_annotations(frame, state)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(16) & 0xFF

        if key == ord("r"):
            state.mode = "red"
        elif key == ord("w"):
            state.mode = "light"
        elif key == ord("b"):
            state.mode = "ball"
        elif key == ord("u"):
            state.undo()
        elif key == ord("c"):
            state.clear()
        elif key == ord("s"):
            output = {
                "video": str(args.video),
                "frame_index": frame_index,
                "frame_sec": frame_index / fps,
                "image_size": [frame_w, frame_h],
                "red_boxes": state.red_boxes,
                "light_boxes": state.light_boxes,
                "ball_box": state.ball_box,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
            print(f"Saved seed file to {args.output}")
            break
        elif key == ord("q") or key == 27:
            print("Exited without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
