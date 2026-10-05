#!/usr/bin/env python3
"""
Hybrid soccer pipeline:
- field segmentation + crowd suppression
- pluggable player detector stage
- CV filtering + jersey-color team classification
- Kalman/Hungarian tracking
- optional classical ball tracker

Recommended backend:
- ultralytics with YOLO11s and separate 960/1280 player/ball inference

Fallback backends:
- hog
- motion
"""

from __future__ import annotations

import argparse
import json
import csv
import time

from runtime_support import validate_run, run_metadata, configure_ultralytics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import soccer_opencv_pipeline as base


BBox = base.BBox


@dataclass
class ScoredBox:
    bbox: BBox
    score: float


def clamp_bbox(box: BBox, frame_w: int, frame_h: int) -> Optional[BBox]:
    return base.normalize_box(box, frame_w, frame_h)


def letterbox(frame: np.ndarray, size: int) -> Tuple[np.ndarray, float, int, int]:
    h, w = frame.shape[:2]
    scale = min(size / float(max(w, 1)), size / float(max(h, 1)))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - nw) // 2
    pad_y = (size - nh) // 2
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


class OpenCVYoloPersonDetector:
    def __init__(
        self,
        model_path: Path,
        input_size: int,
        score_thresh: float,
        nms_thresh: float,
        person_class: int,
        onnx_has_objectness: bool,
    ) -> None:
        if not model_path.exists():
            raise SystemExit(f"Detector model not found: {model_path}")
        self.model_path = model_path
        self.input_size = max(128, int(input_size))
        self.score_thresh = float(score_thresh)
        self.nms_thresh = float(nms_thresh)
        self.person_class = int(person_class)
        self.onnx_has_objectness = onnx_has_objectness
        self.net = cv2.dnn.readNet(str(model_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.output_names = self.net.getUnconnectedOutLayersNames()

    def _parse_candidates(
        self,
        output: np.ndarray,
        frame_w: int,
        frame_h: int,
        scale: float,
        pad_x: int,
        pad_y: int,
    ) -> List[ScoredBox]:
        arr = np.asarray(output)
        if arr.ndim == 4:
            arr = arr.reshape(-1, arr.shape[-1])
        elif arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            else:
                arr = arr.reshape(-1, arr.shape[-1])
        elif arr.ndim != 2:
            return []

        if arr.shape[0] <= 128 and arr.shape[0] < arr.shape[1]:
            arr = arr.T

        boxes_xywh: List[List[int]] = []
        scores: List[float] = []

        for row in arr:
            if row.shape[0] < 6:
                continue

            cx, cy, bw, bh = [float(v) for v in row[:4]]
            if bw <= 1.0 or bh <= 1.0:
                continue

            if self.onnx_has_objectness:
                cls_idx = 5 + self.person_class
                if row.shape[0] <= cls_idx:
                    continue
                score = float(row[4]) * float(row[cls_idx])
            else:
                cls_idx = 4 + self.person_class
                if row.shape[0] <= cls_idx:
                    continue
                score = float(row[cls_idx])

            if score < self.score_thresh:
                continue

            x1 = int(round((cx - 0.5 * bw - pad_x) / max(scale, 1e-6)))
            y1 = int(round((cy - 0.5 * bh - pad_y) / max(scale, 1e-6)))
            x2 = int(round((cx + 0.5 * bw - pad_x) / max(scale, 1e-6)))
            y2 = int(round((cy + 0.5 * bh - pad_y) / max(scale, 1e-6)))

            x1 = max(0, min(frame_w - 1, x1))
            y1 = max(0, min(frame_h - 1, y1))
            x2 = max(0, min(frame_w, x2))
            y2 = max(0, min(frame_h, y2))
            w = x2 - x1
            h = y2 - y1
            if w < 4 or h < 8:
                continue

            boxes_xywh.append([x1, y1, w, h])
            scores.append(score)

        if not boxes_xywh:
            return []

        keep = cv2.dnn.NMSBoxes(boxes_xywh, scores, self.score_thresh, self.nms_thresh)
        if keep is None or len(keep) == 0:
            return []

        kept: List[ScoredBox] = []
        for idx in np.array(keep).reshape(-1).tolist():
            x, y, w, h = boxes_xywh[idx]
            kept.append(ScoredBox((x, y, w, h), float(scores[idx])))
        return kept

    def detect(self, frame: np.ndarray) -> List[ScoredBox]:
        inp, scale, pad_x, pad_y = letterbox(frame, self.input_size)
        blob = cv2.dnn.blobFromImage(inp, scalefactor=1.0 / 255.0, size=(self.input_size, self.input_size), swapRB=True)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_names)
        if isinstance(outputs, tuple):
            outputs = list(outputs)
        elif not isinstance(outputs, list):
            outputs = [outputs]

        all_boxes: List[ScoredBox] = []
        frame_h, frame_w = frame.shape[:2]
        for out in outputs:
            all_boxes.extend(self._parse_candidates(out, frame_w, frame_h, scale, pad_x, pad_y))

        # Extra NMS across output tensors if the export returns more than one output array.
        if not all_boxes:
            return []

        boxes_xywh = [[b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3]] for b in all_boxes]
        scores = [b.score for b in all_boxes]
        keep = cv2.dnn.NMSBoxes(boxes_xywh, scores, self.score_thresh, self.nms_thresh)
        if keep is None or len(keep) == 0:
            return []
        return [all_boxes[idx] for idx in np.array(keep).reshape(-1).tolist()]


class HOGPersonDetector:
    def __init__(self, score_thresh: float) -> None:
        self.score_thresh = float(score_thresh)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> List[ScoredBox]:
        rects, weights = self.hog.detectMultiScale(
            frame,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.03,
        )
        out: List[ScoredBox] = []
        for rect, weight in zip(rects, weights):
            if float(weight) < self.score_thresh:
                continue
            x, y, w, h = [int(v) for v in rect]
            # HOG boxes tend to be slightly oversized.
            x += int(round(w * 0.10))
            w = int(round(w * 0.80))
            y += int(round(h * 0.06))
            h = int(round(h * 0.88))
            if w < 4 or h < 8:
                continue
            out.append(ScoredBox((x, y, w, h), float(weight)))
        return out


class UltralyticsPersonDetector:
    def __init__(
        self,
        model_path: Path,
        input_size: int,
        score_thresh: float,
        nms_thresh: float,
        person_class: int,
        ball_class: Optional[int],
        ball_score_thresh: float,
        ball_input_size: int,
        device: str,
    ) -> None:
        if not model_path.exists():
            raise SystemExit(f"Detector model not found: {model_path}")
        try:
            configure_ultralytics()
            from ultralytics import YOLO
        except Exception as exc:
            raise SystemExit(
                f"Could not import Ultralytics: {exc}. Check requirements and writable YOLO_CONFIG_DIR."
            ) from exc

        self.model = YOLO(str(model_path))
        self.input_size = max(128, int(input_size))
        self.score_thresh = float(score_thresh)
        self.nms_thresh = float(nms_thresh)
        self.person_class = int(person_class)
        self.ball_class = None if ball_class is None or int(ball_class) < 0 else int(ball_class)
        self.ball_score_thresh = float(ball_score_thresh)
        self.ball_input_size = max(self.input_size, int(ball_input_size))
        self.device = device
        self._people_cache_frame: Optional[np.ndarray] = None
        self._ball_cache_frame: Optional[np.ndarray] = None
        self._cached_people: List[ScoredBox] = []
        self._cached_balls: List[ScoredBox] = []

    def _run_predict(
        self,
        frame: np.ndarray,
        classes: List[int],
        imgsz: int,
        conf_thresh: float,
        max_det: int,
    ) -> List[ScoredBox]:
        results = self.model.predict(
            source=frame,
            imgsz=imgsz,
            conf=conf_thresh,
            iou=self.nms_thresh,
            classes=classes,
            device=self.device,
            verbose=False,
            max_det=max_det,
        )
        if not results:
            return []

        result = results[0]
        boxes_obj = getattr(result, "boxes", None)
        if boxes_obj is None or boxes_obj.xyxy is None:
            return []

        xyxy = boxes_obj.xyxy.cpu().numpy()
        conf = boxes_obj.conf.cpu().numpy() if boxes_obj.conf is not None else np.ones((len(xyxy),), dtype=np.float32)
        out: List[ScoredBox] = []
        for coords, score in zip(xyxy, conf):
            x1, y1, x2, y2 = [int(round(float(v))) for v in coords.tolist()]
            w = x2 - x1
            h = y2 - y1
            if w < 2 or h < 2:
                continue
            out.append(ScoredBox((x1, y1, w, h), float(score)))
        return out

    def _infer_people(self, frame: np.ndarray) -> None:
        if self._people_cache_frame is frame:
            return

        people: List[ScoredBox] = []
        for box in self._run_predict(
            frame=frame,
            classes=[self.person_class],
            imgsz=self.input_size,
            conf_thresh=self.score_thresh,
            max_det=150,
        ):
            x, y, w, h = box.bbox
            if w < 4 or h < 8:
                continue
            people.append(box)

        self._people_cache_frame = frame
        self._cached_people = people

    def _infer_ball(self, frame: np.ndarray) -> None:
        if self.ball_class is None:
            self._ball_cache_frame = frame
            self._cached_balls = []
            return
        if self._ball_cache_frame is frame:
            return

        balls: List[ScoredBox] = []
        for box in self._run_predict(
            frame=frame,
            classes=[self.ball_class],
            imgsz=self.ball_input_size,
            conf_thresh=self.ball_score_thresh,
            max_det=60,
        ):
            x, y, w, h = box.bbox
            if w < 2 or h < 2:
                continue
            balls.append(box)

        self._ball_cache_frame = frame
        self._cached_balls = balls

    def detect(self, frame: np.ndarray) -> List[ScoredBox]:
        self._infer_people(frame)
        return list(self._cached_people)

    def detect_ball(self, frame: np.ndarray) -> List[ScoredBox]:
        self._infer_ball(frame)
        return list(self._cached_balls)


def extract_torso_roi(roi_hsv: np.ndarray) -> np.ndarray:
    if roi_hsv.size == 0:
        return roi_hsv
    h, w = roi_hsv.shape[:2]
    x1 = max(0, int(round(w * 0.18)))
    x2 = min(w, max(x1 + 2, int(round(w * 0.82))))
    y1 = max(0, int(round(h * 0.04)))
    y2 = min(h, max(y1 + 3, int(round(h * 0.62))))
    return roi_hsv[y1:y2, x1:x2]


def classify_team_from_bbox(
    hsv: np.ndarray,
    bbox: BBox,
    field_lower: np.ndarray,
    field_upper: np.ndarray,
    color_profile: Optional[base.TeamColorProfile],
) -> str:
    x, y, w, h = bbox
    roi = hsv[y : y + h, x : x + w]
    if roi.size == 0:
        return base.TEAM_OTHER
    torso = extract_torso_roi(roi)
    label = base.classify_team(torso, field_lower, field_upper, color_profile=color_profile)
    if label == base.TEAM_OTHER:
        label = base.classify_team(roi, field_lower, field_upper, color_profile=color_profile)
    return label


def filter_detector_boxes(
    raw_boxes: Sequence[ScoredBox],
    hsv: np.ndarray,
    playable: np.ndarray,
    field_lower: np.ndarray,
    field_upper: np.ndarray,
    top_y: int,
    min_area: float,
    max_area: float,
    min_aspect: float,
    max_aspect: float,
    min_nongreen: float,
    min_playable_overlap: float,
    color_profile: Optional[base.TeamColorProfile],
) -> List[base.Detection]:
    frame_h, frame_w = hsv.shape[:2]
    depth_denom = float(max(1, frame_h - top_y))
    dets: List[base.Detection] = []

    for raw in raw_boxes:
        normalized = clamp_bbox(raw.bbox, frame_w, frame_h)
        if normalized is None:
            continue
        x, y, w, h = normalized
        if y < top_y or w < 4 or h < 8:
            continue

        aspect = h / float(max(w, 1))
        if aspect < min_aspect or aspect > max_aspect:
            continue

        depth = (y + h - top_y) / depth_denom
        depth = min(max(depth, 0.0), 1.0)
        adaptive_min_area = min_area * (0.24 + 0.56 * depth)
        area = float(w * h)
        if area < adaptive_min_area or area > (max_area * 1.35):
            continue

        roi_play = playable[y : y + h, x : x + w]
        playable_ratio = cv2.countNonZero(roi_play) / float(max(w * h, 1))
        if playable_ratio < min_playable_overlap:
            continue

        roi_hsv = hsv[y : y + h, x : x + w]
        green = cv2.inRange(roi_hsv, field_lower, field_upper)
        nongreen = 1.0 - (cv2.countNonZero(green) / float(max(w * h, 1)))
        adaptive_nongreen = min_nongreen * (0.55 + 0.45 * depth)
        if nongreen < adaptive_nongreen:
            continue

        team_label = classify_team_from_bbox(
            hsv=hsv,
            bbox=(x, y, w, h),
            field_lower=field_lower,
            field_upper=field_upper,
            color_profile=color_profile,
        )
        centroid = (x + w / 2.0, y + h / 2.0)
        dets.append(base.Detection((x, y, w, h), centroid, team_label))

    return dets


def build_detector(args: argparse.Namespace) -> Optional[Any]:
    if args.detector_backend == "motion":
        return None
    if args.detector_backend == "hog":
        return HOGPersonDetector(score_thresh=args.detector_score_thresh)
    if args.detector_backend == "ultralytics":
        if args.detector_model is None:
            raise SystemExit("--detector-model is required for --detector-backend ultralytics")
        return UltralyticsPersonDetector(
            model_path=args.detector_model,
            input_size=args.detector_input_size,
            score_thresh=args.detector_score_thresh,
            nms_thresh=args.detector_nms_thresh,
            person_class=args.detector_person_class,
            ball_class=args.detector_ball_class if args.ball_backend in ("detector", "hybrid") else None,
            ball_score_thresh=args.detector_ball_score_thresh,
            ball_input_size=args.detector_ball_input_size,
            device=args.detector_device,
        )
    if args.detector_backend == "onnx-yolo":
        if args.detector_model is None:
            raise SystemExit("--detector-model is required for --detector-backend onnx-yolo")
        return OpenCVYoloPersonDetector(
            model_path=args.detector_model,
            input_size=args.detector_input_size,
            score_thresh=args.detector_score_thresh,
            nms_thresh=args.detector_nms_thresh,
            person_class=args.detector_person_class,
            onnx_has_objectness=args.onnx_has_objectness,
        )
    raise SystemExit(f"Unsupported detector backend: {args.detector_backend}")


def detect_players_with_backend(
    args: argparse.Namespace,
    detector: Optional[object],
    frame: np.ndarray,
    hsv: np.ndarray,
    mot: np.ndarray,
    playable: np.ndarray,
    field_lower: np.ndarray,
    field_upper: np.ndarray,
    top_y: int,
    color_profile: Optional[base.TeamColorProfile],
) -> Tuple[List[base.Detection], int]:
    if args.detector_backend == "motion":
        players = base.detect_players(
            hsv=hsv,
            mot=mot,
            field_lower=field_lower,
            field_upper=field_upper,
            top_y=top_y,
            min_area=args.min_area,
            max_area=args.max_area,
            min_aspect=args.min_aspect,
            max_aspect=args.max_aspect,
            min_nongreen=args.min_nongreen,
            color_profile=color_profile,
        )
        return players, len(players)

    assert detector is not None
    raw_boxes = detector.detect(frame)
    players = filter_detector_boxes(
        raw_boxes=raw_boxes,
        hsv=hsv,
        playable=playable,
        field_lower=field_lower,
        field_upper=field_upper,
        top_y=top_y,
        min_area=args.min_area,
        max_area=args.max_area,
        min_aspect=args.min_aspect,
        max_aspect=args.max_aspect,
        min_nongreen=args.min_nongreen,
        min_playable_overlap=args.min_playable_overlap,
        color_profile=color_profile,
    )

    if args.fuse_motion_proposals:
        motion_players = base.detect_players(
            hsv=hsv,
            mot=mot,
            field_lower=field_lower,
            field_upper=field_upper,
            top_y=top_y,
            min_area=args.min_area,
            max_area=args.max_area,
            min_aspect=args.min_aspect,
            max_aspect=args.max_aspect,
            min_nongreen=args.min_nongreen,
            color_profile=color_profile,
        )
        players = base.merge_detections(players, motion_players)

    return players, len(raw_boxes)


def detector_ball_candidates(
    detector: Optional[Any],
    frame: np.ndarray,
    hsv: np.ndarray,
    playable: np.ndarray,
    top_y: int,
    field_lower: np.ndarray,
    field_upper: np.ndarray,
    color_profile: Optional[base.TeamColorProfile],
    player_boxes: Sequence[BBox],
    prior_center: Optional[Tuple[float, float]],
) -> List[base.BallCandidate]:
    if detector is None or not hasattr(detector, "detect_ball"):
        return []

    frame_h, frame_w = hsv.shape[:2]
    boxes = detector.detect_ball(frame)
    out: List[base.BallCandidate] = []
    for box in boxes:
        normalized = base.normalize_box(box.bbox, frame_w, frame_h)
        if normalized is None:
            continue
        x, y, w, h = normalized
        if y < top_y or w < 2 or h < 2 or w > 36 or h > 36:
            continue

        aspect = w / float(max(h, 1))
        if aspect < 0.45 or aspect > 2.2:
            continue

        roi_play = playable[y : y + h, x : x + w]
        playable_ratio = cv2.countNonZero(roi_play) / float(max(w * h, 1))
        if playable_ratio < 0.35:
            continue

        center = (x + w / 2.0, y + h / 2.0)
        score_size = max(3, int(round(max(w, h) * 0.82)))
        score_x = int(round(center[0] - score_size / 2.0))
        score_y = int(round(center[1] - score_size / 2.0))
        score_box = base.normalize_box((score_x, score_y, score_size, score_size), frame_w, frame_h)
        if score_box is None:
            continue

        temporal_bonus = 0.0
        if prior_center is not None:
            temporal_dist = base.math.dist(center, prior_center)
            temporal_bonus = max(0.0, 0.24 - (temporal_dist / 180.0))

        visual = base.score_ball_candidate(
            hsv=hsv,
            bbox=score_box,
            area=max(3.0, 0.72 * float(score_box[2] * score_box[3])),
            color_profile=color_profile,
            player_boxes=player_boxes,
            min_white_ratio=0.16,
            min_isolation=0.01,
        )
        foot_bonus = base.player_foot_zone_bonus(center, player_boxes)
        if visual is None:
            if float(box.score) < 0.32 and foot_bonus < 0.06 and temporal_bonus < 0.10:
                continue
            radius = max(2.0, max(w, h) / 2.0)
            score = 0.28 + (0.55 * float(box.score)) + foot_bonus + temporal_bonus
            out.append(base.BallCandidate(center=center, radius=radius, score=score))
            continue

        visual.score += 0.30 + (0.70 * float(box.score)) + foot_bonus + temporal_bonus
        out.append(visual)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = "Hybrid soccer tracking with pluggable player detection backend"
    parser.set_defaults(output=Path("outputs_hybrid"))
    parser.add_argument("--detector-backend", choices=["motion", "hog", "onnx-yolo", "ultralytics"], default="motion")
    parser.add_argument("--detector-model", type=Path, default=None)
    parser.add_argument("--detector-input-size", type=int, default=640)
    parser.add_argument("--detector-score-thresh", type=float, default=0.25)
    parser.add_argument("--detector-nms-thresh", type=float, default=0.45)
    parser.add_argument("--detector-person-class", type=int, default=0)
    parser.add_argument("--detector-ball-class", type=int, default=32)
    parser.add_argument("--detector-ball-score-thresh", type=float, default=0.08)
    parser.add_argument("--detector-ball-input-size", type=int, default=1280)
    parser.add_argument("--detector-device", type=str, default="cpu")
    parser.add_argument("--onnx-has-objectness", action="store_true")
    parser.add_argument("--min-playable-overlap", type=float, default=0.35)
    parser.add_argument("--fuse-motion-proposals", action="store_true")
    parser.add_argument("--ball-backend", choices=["classical", "detector", "hybrid"], default="hybrid")
    parser.add_argument("--disable-ball", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_run(args)
    started = time.perf_counter()
    if not args.disable_ball and args.ball_backend == "detector" and args.detector_backend != "ultralytics":
        raise SystemExit("Detector-only ball mode requires the ultralytics backend.")
    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    manual_seed = base.load_manual_seed(args.seed_file)
    detector = build_detector(args)

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    masks_dir = out / "masks"
    if args.save_masks:
        masks_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    start = max(0, int(round(args.start_sec * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    limit = int(round(args.duration_sec * fps)) if args.duration_sec > 0 else 10**9

    bg = cv2.createBackgroundSubtractorKNN(history=700, dist2Threshold=600.0, detectShadows=False)
    ptracker = base.PlayerTracker(
        fps=fps,
        max_missed=args.max_missed,
        max_match_distance=args.max_match_distance,
        trail_length=args.trail_length,
        class_mismatch_penalty=args.class_mismatch_penalty,
    )
    btracker = base.BallTracker(args.ball_max_missed, args.ball_max_dist, max(20, args.trail_length))

    writer: Optional[cv2.VideoWriter] = None
    processed = 0
    frame_idx = start

    det_total = 0
    det_team = base.Counter()
    recovered_total = 0
    raw_detector_total = 0
    detector_ball_total = 0
    ball_visible = 0
    ball_measured = 0
    frame_records = []

    color_profile = base.TeamColorProfile()
    calibration_red_samples: List[np.ndarray] = []
    calibration_white_samples: List[np.ndarray] = []
    calibration_complete = False

    if manual_seed is not None:
        seeded_profile = base.build_profile_from_manual_seed(
            video_path=args.video,
            resize_width=args.resize_width,
            seed_data=manual_seed,
            field_lower=args.field_hsv_lower,
            field_upper=args.field_hsv_upper,
            row_green_threshold=args.row_green_threshold,
            row_run=args.row_green_run,
            manual_top_cut=args.crowd_top_cut,
            min_pixels=args.manual_seed_min_pixels,
        )
        if seeded_profile is not None:
            color_profile = seeded_profile
            calibration_complete = True

    while processed < limit:
        ok, frame = cap.read()
        if not ok:
            break

        frame = base.resize_by_width(frame, args.resize_width)
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out / "annotated_tracking.mp4"), fourcc, fps, (frame.shape[1], frame.shape[0]))
            if not writer.isOpened():
                cap.release()
                raise SystemExit("Could not create MP4 output; check codec and output permissions.")

        hsv, pitch, playable, top_y = base.segment_field(
            frame=frame,
            lower_hsv=args.field_hsv_lower,
            upper_hsv=args.field_hsv_upper,
            row_green_threshold=args.row_green_threshold,
            row_run=args.row_green_run,
            manual_top_cut=args.crowd_top_cut,
        )
        mot = base.motion_mask(frame, bg, playable)

        if processed < args.warmup_frames and args.detector_backend == "motion":
            players: List[base.Detection] = []
            raw_detector_count = 0
        else:
            players, raw_detector_count = detect_players_with_backend(
                args=args,
                detector=detector,
                frame=frame,
                hsv=hsv,
                mot=mot,
                playable=playable,
                field_lower=args.field_hsv_lower,
                field_upper=args.field_hsv_upper,
                top_y=top_y,
                color_profile=color_profile,
            )

        raw_detector_total += raw_detector_count

        if processed >= args.warmup_frames:
            calibration_window_open = processed < (args.warmup_frames + args.calibration_frames)
            red_mask, white_mask = base.build_team_color_masks(
                hsv=hsv,
                playable=playable,
                field_lower=args.field_hsv_lower,
                field_upper=args.field_hsv_upper,
                color_profile=color_profile,
            )

            if calibration_window_open and not calibration_complete and players:
                sample_red, sample_white = base.collect_calibration_pixels(
                    detections=players,
                    hsv=hsv,
                    field_lower=args.field_hsv_lower,
                    field_upper=args.field_hsv_upper,
                )
                calibration_red_samples.extend(sample_red)
                calibration_white_samples.extend(sample_white)
                calibrated = base.fit_team_color_profile(
                    red_samples=calibration_red_samples,
                    white_samples=calibration_white_samples,
                    min_pixels=args.calibration_min_pixels,
                )
                if calibrated is not None:
                    color_profile = calibrated
                    calibration_complete = True
                    if args.detector_backend == "motion":
                        players, _ = detect_players_with_backend(
                            args=args,
                            detector=detector,
                            frame=frame,
                            hsv=hsv,
                            mot=mot,
                            playable=playable,
                            field_lower=args.field_hsv_lower,
                            field_upper=args.field_hsv_upper,
                            top_y=top_y,
                            color_profile=color_profile,
                        )
                    else:
                        players = filter_detector_boxes(
                            raw_boxes=detector.detect(frame) if detector is not None else (),
                            hsv=hsv,
                            playable=playable,
                            field_lower=args.field_hsv_lower,
                            field_upper=args.field_hsv_upper,
                            top_y=top_y,
                            min_area=args.min_area,
                            max_area=args.max_area,
                            min_aspect=args.min_aspect,
                            max_aspect=args.max_aspect,
                            min_nongreen=args.min_nongreen,
                            min_playable_overlap=args.min_playable_overlap,
                            color_profile=color_profile,
                        )

            recovered_players = base.recover_players_from_tracks(
                tracks=list(ptracker.active.values()),
                detections=players,
                red_mask=red_mask,
                white_mask=white_mask,
                top_y=top_y,
                local_window=args.team_local_window,
                max_recover_missed=args.team_recover_max_missed,
            )
            recovered_total += len(recovered_players)
            players = base.merge_detections(players, recovered_players)

            if args.enable_global_team_mask_recovery:
                color_mask_players = base.detect_players_from_team_masks(red_mask, white_mask, mot, top_y)
                players = base.merge_detections(players, color_mask_players)

        ball_cands: List[base.BallCandidate] = []
        if not args.disable_ball and processed >= args.warmup_frames:
            ball_keepout_boxes = [d.bbox for d in players]
            if args.ball_backend in ("detector", "hybrid"):
                detector_balls = detector_ball_candidates(
                    detector=detector,
                    frame=frame,
                    hsv=hsv,
                    playable=playable,
                    top_y=top_y,
                    field_lower=args.field_hsv_lower,
                    field_upper=args.field_hsv_upper,
                    color_profile=color_profile,
                    player_boxes=ball_keepout_boxes,
                    prior_center=btracker.track.position if btracker.track is not None else None,
                )
                detector_ball_total += len(detector_balls)
                ball_cands.extend(detector_balls)

            if args.ball_backend in ("classical", "hybrid"):
                ball_cands.extend(
                    base.detect_ball(
                        hsv=hsv,
                        mot=mot,
                        top_y=top_y,
                        min_area=args.ball_min_area,
                        max_area=args.ball_max_area,
                        color_profile=color_profile,
                        player_boxes=ball_keepout_boxes,
                    )
                )
                if btracker.track is not None:
                    ball_cands.extend(
                        base.detect_ball_local_near(
                            hsv=hsv,
                            playable=playable,
                            center=btracker.track.position,
                            field_lower=args.field_hsv_lower,
                            field_upper=args.field_hsv_upper,
                            window=args.ball_local_window,
                            min_area=args.ball_min_area,
                            max_area=args.ball_max_area,
                            color_profile=color_profile,
                            player_boxes=ball_keepout_boxes,
                        )
                    )
                ball_cands.extend(
                    base.detect_ball_near_players(
                        hsv=hsv,
                        playable=playable,
                        player_boxes=ball_keepout_boxes,
                        top_y=top_y,
                        field_lower=args.field_hsv_lower,
                        field_upper=args.field_hsv_upper,
                        min_area=args.ball_min_area,
                        max_area=args.ball_max_area,
                        color_profile=color_profile,
                    )
                )
            if ball_cands:
                ball_cands = sorted(ball_cands, key=lambda z: z.score, reverse=True)
                deduped: List[base.BallCandidate] = []
                for cand in ball_cands:
                    if all(base.math.dist(cand.center, d.center) > 4.0 for d in deduped):
                        deduped.append(cand)
                    if len(deduped) >= 12:
                        break
                ball_cands = deduped

        seeded_ball = base.manual_seed_ball_candidate(
            seed_data=manual_seed,
            frame_idx=frame_idx,
            frame_w=frame.shape[1],
            frame_h=frame.shape[0],
        )
        if seeded_ball is not None:
            ball_cands = [seeded_ball] + ball_cands

        tracks = ptracker.update(players, frame_idx)
        ball = None if args.disable_ball else btracker.update(ball_cands)

        det_total += len(players)
        for det in players:
            det_team[det.team_label] += 1

        ann = frame.copy()
        base.shade_crowd(ann, top_y)
        for tr in tracks:
            base.draw_player(ann, tr)

        if ball is not None and ball.missed <= 6:
            base.draw_ball(ann, ball)
            ball_visible += 1

        active = base.Counter(t.team_label for t in tracks)
        status = (
            f"Frame {frame_idx} | "
            f"RED {active[base.TEAM_RED]} "
            f"LIGHT {active[base.TEAM_LIGHT]} "
            f"OTHER {active[base.TEAM_OTHER]}"
        )
        cv2.putText(ann, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            ann,
            f"Detector {args.detector_backend.upper()}",
            (10, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        visible = ball is not None and ball.missed <= 6
        measured = visible and ball.missed == 0
        ball_measured += int(measured)
        frame_records.append([frame_idx, round(frame_idx / fps, 6), int(processed < args.warmup_frames),
                              len(players), len(tracks), sum(t.missed == 0 for t in tracks),
                              "measured" if measured else "predicted" if visible else "absent",
                              ball.position[0] if visible else "", ball.position[1] if visible else ""])
        writer.write(ann)

        if args.save_masks:
            cv2.imwrite(str(masks_dir / f"pitch_{frame_idx:06d}.png"), pitch)
            cv2.imwrite(str(masks_dir / f"playable_{frame_idx:06d}.png"), playable)
            cv2.imwrite(str(masks_dir / f"motion_{frame_idx:06d}.png"), mot)

        processed += 1
        frame_idx += 1

        if args.preview:
            cv2.imshow("Soccer Hybrid Tracking", ann)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


    cap.release()
    if writer is not None:
        writer.release()
    if args.preview:
        cv2.destroyAllWindows()

    if processed == 0:
        raise SystemExit("No frames processed; the start time may be beyond the video duration.")
    with (out / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        rows = csv.writer(stream)
        rows.writerow(["frame_index", "time_sec", "warmup", "player_detections", "active_tracks", "measured_player_tracks", "ball_state", "ball_x", "ball_y"])
        rows.writerows(frame_records)
    tracks_all = ptracker.all_tracks()
    base.write_metrics(out / "track_metrics.csv", tracks_all)

    summary = {
        "input_video": str(args.video),
        "manual_seed_file": str(args.seed_file) if args.seed_file is not None else "",
        "detector_backend": args.detector_backend,
        "detector_model": str(args.detector_model) if args.detector_model is not None else "",
        "ball_backend": args.ball_backend,
        "frames_processed": processed,
        "raw_detector_candidates_total": raw_detector_total,
        "detector_ball_candidates_total": detector_ball_total,
        "player_detections_total": det_total,
        "recovered_detections_total": recovered_total,
        "color_profile": {
            "calibrated": color_profile.calibrated,
            "red_low_max": color_profile.red_low_max,
            "red_high_min": color_profile.red_high_min,
            "red_sat_min": color_profile.red_sat_min,
            "red_val_min": color_profile.red_val_min,
            "white_sat_max": color_profile.white_sat_max,
            "white_val_min": color_profile.white_val_min,
        },
        "player_detections_by_team": {
            base.TEAM_RED: int(det_team[base.TEAM_RED]),
            base.TEAM_LIGHT: int(det_team[base.TEAM_LIGHT]),
            base.TEAM_OTHER: int(det_team[base.TEAM_OTHER]),
        },
        "tracks_total": len(tracks_all),
        "ball_visible_frames": ball_visible,
        "ball_measured_frames": ball_measured,
        "ball_predicted_visible_frames": ball_visible - ball_measured,
        "metadata": run_metadata(args, fps, time.perf_counter() - started),
        "matching": "hungarian_scipy" if base.SCIPY_AVAILABLE else "greedy_fallback",
        "output_video": str(out / "annotated_tracking.mp4"),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Done")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
