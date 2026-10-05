#!/usr/bin/env python3
"""
Improved soccer pipeline:
- ignores crowd rows above the field
- groups players into red vs light kits
- tracks the ball separately
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time

from .runtime import validate_run, run_metadata
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment

    SCIPY_AVAILABLE = True
except Exception:
    linear_sum_assignment = None
    SCIPY_AVAILABLE = False


Point = Tuple[float, float]
BBox = Tuple[int, int, int, int]

TEAM_RED = "team_red"
TEAM_LIGHT = "team_light"
TEAM_OTHER = "team_other"
TEAM_NAME = {TEAM_RED: "RED", TEAM_LIGHT: "LIGHT", TEAM_OTHER: "OTHER"}
TEAM_COLOR = {TEAM_RED: (30, 30, 230), TEAM_LIGHT: (245, 245, 245), TEAM_OTHER: (0, 215, 255)}


@dataclass
class Detection:
    bbox: BBox
    centroid: Point
    team_label: str


@dataclass
class Track:
    track_id: int
    kalman: cv2.KalmanFilter
    bbox: BBox
    last_position: Point
    last_update_frame: int
    team_label: str = TEAM_OTHER
    label_votes: Dict[str, int] = field(default_factory=dict)
    missed: int = 0
    trail: Deque[Point] = field(default_factory=deque)
    speeds_px_per_s: List[float] = field(default_factory=list)
    confirmed_points: int = 1
    last_measured_position: Optional[Point] = None


@dataclass
class BallCandidate:
    center: Point
    radius: float
    score: float


@dataclass
class BallTrack:
    kalman: cv2.KalmanFilter
    position: Point
    radius: float
    missed: int = 0
    trail: Deque[Point] = field(default_factory=deque)


@dataclass
class TeamColorProfile:
    red_low_max: int = 12
    red_high_min: int = 165
    red_sat_min: int = 65
    red_val_min: int = 38
    white_sat_max: int = 105
    white_val_min: int = 132
    calibrated: bool = False


def parse_hsv_triplet(value: str) -> np.ndarray:
    try:
        vals = [int(p.strip()) for p in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("HSV must contain integer H,S,V values") from exc
    if len(vals) != 3 or not (0 <= vals[0] <= 179) or any(v < 0 or v > 255 for v in vals[1:]):
        raise argparse.ArgumentTypeError("OpenCV HSV requires H in [0,179], S,V in [0,255]")
    return np.array(vals, dtype=np.uint8)


def infer_default_video() -> Path:
    mp4s = sorted(Path(".").glob("*.mp4"))
    mp4s = [p for p in mp4s if "annotated" not in p.name.lower()]
    return mp4s[0] if mp4s else Path("soccer_footage.mp4")


def read_video_frame(video_path: Path, frame_index: int, resize_width: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return resize_by_width(frame, resize_width)


def normalize_box(box: Sequence[Any], frame_w: int, frame_h: int) -> Optional[BBox]:
    if len(box) != 4 or frame_w <= 0 or frame_h <= 0:
        return None
    x, y, w, h = [int(round(float(v))) for v in box]
    if w <= 0 or h <= 0:
        return None
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame_w, x + w), min(frame_h, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def load_manual_seed(seed_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if seed_path is None:
        return None
    if not seed_path.exists():
        raise SystemExit(f"Manual seed file not found: {seed_path}")

    data = json.loads(seed_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("Manual seed file must contain a JSON object.")

    data.setdefault("red_boxes", [])
    data.setdefault("light_boxes", [])
    data.setdefault("ball_box", None)
    if "frame_index" not in data:
        raise SystemExit("Manual seed file must include 'frame_index'.")
    data["frame_index"] = int(data["frame_index"])
    return data


def create_kalman(initial_point: Point, process_noise: float, meas_noise: float) -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * meas_noise
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    x, y = initial_point
    kf.statePost = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
    return kf


def resize_by_width(frame: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w == width:
        return frame
    nh = int(round(h * (width / float(w))))
    return cv2.resize(frame, (width, nh), interpolation=cv2.INTER_AREA)


def segment_field(
    frame: np.ndarray,
    lower_hsv: np.ndarray,
    upper_hsv: np.ndarray,
    row_green_threshold: float,
    row_run: int,
    manual_top_cut: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pitch = cv2.inRange(hsv, lower_hsv, upper_hsv)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pitch = cv2.morphologyEx(pitch, cv2.MORPH_OPEN, k, iterations=1)
    pitch = cv2.morphologyEx(pitch, cv2.MORPH_CLOSE, k, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(pitch, connectivity=8)
    if n > 1:
        lid = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        pitch = (labels == lid).astype(np.uint8) * 255

    row_ratio = np.mean(pitch > 0, axis=1)
    top_y = int(round(frame.shape[0] * 0.18))
    for i in range(0, max(1, len(row_ratio) - row_run)):
        if np.all(row_ratio[i : i + row_run] >= row_green_threshold):
            top_y = max(0, i - 2)
            break
    top_y = max(top_y, max(0, manual_top_cut))

    playable = pitch.copy()
    playable[:top_y, :] = 0
    return hsv, pitch, playable, top_y


def motion_mask(frame: np.ndarray, bg: cv2.BackgroundSubtractor, playable: np.ndarray) -> np.ndarray:
    fg = bg.apply(cv2.GaussianBlur(frame, (5, 5), 0))
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=1)
    fg = cv2.dilate(fg, k, iterations=1)
    return cv2.bitwise_and(fg, playable)


def make_red_mask(hsv: np.ndarray, profile: TeamColorProfile) -> np.ndarray:
    low = cv2.inRange(
        hsv,
        np.array([0, profile.red_sat_min, profile.red_val_min], np.uint8),
        np.array([profile.red_low_max, 255, 255], np.uint8),
    )
    high = cv2.inRange(
        hsv,
        np.array([profile.red_high_min, profile.red_sat_min, profile.red_val_min], np.uint8),
        np.array([179, 255, 255], np.uint8),
    )
    return cv2.bitwise_or(low, high)


def make_white_mask(hsv: np.ndarray, profile: TeamColorProfile) -> np.ndarray:
    return cv2.inRange(
        hsv,
        np.array([0, 0, profile.white_val_min], np.uint8),
        np.array([179, profile.white_sat_max, 255], np.uint8),
    )


def make_ball_white_mask(hsv: np.ndarray, profile: TeamColorProfile) -> np.ndarray:
    sat_cap = int(np.clip(max(96, profile.white_sat_max + 30), 88, 170))
    val_floor = int(np.clip(min(172, max(112, profile.white_val_min - 34)), 100, 188))
    return cv2.inRange(
        hsv,
        np.array([0, 0, val_floor], np.uint8),
        np.array([179, sat_cap, 255], np.uint8),
    )


def sample_upper_player_pixels(
    roi_hsv: np.ndarray,
    field_lower: np.ndarray,
    field_upper: np.ndarray,
    max_samples: int = 140,
) -> np.ndarray:
    if roi_hsv.size == 0:
        return np.empty((0, 3), dtype=np.uint8)

    upper = roi_hsv[: max(1, int(round(roi_hsv.shape[0] * 0.60))), :]
    field_mask = cv2.inRange(upper, field_lower, field_upper)
    pixels = upper[field_mask == 0]
    if pixels.size == 0:
        return np.empty((0, 3), dtype=np.uint8)

    stride = max(1, len(pixels) // max_samples)
    return pixels[::stride][:max_samples]


def fit_team_color_profile(
    red_samples: Sequence[np.ndarray],
    white_samples: Sequence[np.ndarray],
    min_pixels: int,
) -> Optional[TeamColorProfile]:
    red_count = sum(len(sample) for sample in red_samples)
    white_count = sum(len(sample) for sample in white_samples)
    if red_count < min_pixels or white_count < min_pixels:
        return None

    red_pixels = np.vstack([sample for sample in red_samples if len(sample) > 0])
    white_pixels = np.vstack([sample for sample in white_samples if len(sample) > 0])

    low_red = red_pixels[red_pixels[:, 0] <= 40]
    high_red = red_pixels[red_pixels[:, 0] >= 140]

    profile = TeamColorProfile(calibrated=True)
    base_profile = TeamColorProfile()
    if len(low_red) > 0:
        fitted = int(np.clip(np.percentile(low_red[:, 0], 94) + 3, 8, 22))
        profile.red_low_max = int(round(0.7 * fitted + 0.3 * base_profile.red_low_max))
    if len(high_red) > 0:
        fitted = int(np.clip(np.percentile(high_red[:, 0], 6) - 3, 148, 172))
        profile.red_high_min = int(round(0.7 * fitted + 0.3 * base_profile.red_high_min))

    fitted_red_sat = int(np.clip(np.percentile(red_pixels[:, 1], 14) - 12, 42, 125))
    fitted_red_val = int(np.clip(np.percentile(red_pixels[:, 2], 10) - 20, 24, 132))
    fitted_white_sat = int(np.clip(np.percentile(white_pixels[:, 1], 94) + 18, 65, 165))
    fitted_white_val = int(np.clip(np.percentile(white_pixels[:, 2], 10) - 24, 92, 210))

    profile.red_sat_min = int(round(0.42 * fitted_red_sat + 0.58 * base_profile.red_sat_min))
    profile.red_val_min = int(round(0.34 * fitted_red_val + 0.66 * base_profile.red_val_min))
    profile.white_sat_max = int(round(0.58 * fitted_white_sat + 0.42 * base_profile.white_sat_max))
    profile.white_val_min = int(round(0.52 * fitted_white_val + 0.48 * base_profile.white_val_min))

    # Guard rails prevent a few bright seed boxes from making "white" too clean to match shadows.
    profile.white_sat_max = max(profile.white_sat_max, 82)
    profile.white_val_min = min(profile.white_val_min, 156)
    return profile


def collect_calibration_pixels(
    detections: Sequence[Detection],
    hsv: np.ndarray,
    field_lower: np.ndarray,
    field_upper: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    red_samples: List[np.ndarray] = []
    white_samples: List[np.ndarray] = []

    for det in detections:
        if det.team_label not in (TEAM_RED, TEAM_LIGHT):
            continue

        x, y, w, h = det.bbox
        roi_hsv = hsv[y : y + h, x : x + w]
        pixels = sample_upper_player_pixels(roi_hsv, field_lower, field_upper)
        if len(pixels) == 0:
            continue

        if det.team_label == TEAM_RED:
            red_samples.append(pixels)
        else:
            white_samples.append(pixels)

    return red_samples, white_samples


def collect_seed_box_samples(
    hsv: np.ndarray,
    boxes: Sequence[Sequence[Any]],
    field_lower: np.ndarray,
    field_upper: np.ndarray,
) -> List[np.ndarray]:
    frame_h, frame_w = hsv.shape[:2]
    out: List[np.ndarray] = []
    for box in boxes:
        normalized = normalize_box(box, frame_w, frame_h)
        if normalized is None:
            continue
        x, y, w, h = normalized
        roi = hsv[y : y + h, x : x + w]
        pixels = sample_upper_player_pixels(roi, field_lower, field_upper, max_samples=180)
        if len(pixels) > 0:
            out.append(pixels)
    return out


def seed_box_for_frame(seed_data: Dict[str, Any], box: Sequence[Any], frame_w: int, frame_h: int) -> Optional[BBox]:
    """Transform boxes from the annotation resolution into the analysis frame."""
    size = seed_data.get("image_size")
    if size is not None:
        if len(size) != 2 or float(size[0]) <= 0 or float(size[1]) <= 0:
            raise SystemExit("Seed image_size must be [positive width, positive height].")
        sx, sy = frame_w / float(size[0]), frame_h / float(size[1])
        box = [float(box[0]) * sx, float(box[1]) * sy, float(box[2]) * sx, float(box[3]) * sy]
    return normalize_box(box, frame_w, frame_h)


def build_profile_from_manual_seed(
    video_path: Path,
    resize_width: int,
    seed_data: Dict[str, Any],
    field_lower: np.ndarray,
    field_upper: np.ndarray,
    row_green_threshold: float,
    row_run: int,
    manual_top_cut: int,
    min_pixels: int,
) -> Optional[TeamColorProfile]:
    frame = read_video_frame(video_path, int(seed_data["frame_index"]), resize_width)
    if frame is None:
        return None

    hsv, _, _, _ = segment_field(
        frame=frame,
        lower_hsv=field_lower,
        upper_hsv=field_upper,
        row_green_threshold=row_green_threshold,
        row_run=row_run,
        manual_top_cut=manual_top_cut,
    )

    height, width = hsv.shape[:2]
    red_boxes = [seed_box_for_frame(seed_data, box, width, height) for box in seed_data.get("red_boxes", [])]
    white_boxes = [seed_box_for_frame(seed_data, box, width, height) for box in seed_data.get("light_boxes", [])]
    red_samples = collect_seed_box_samples(hsv, [box for box in red_boxes if box is not None], field_lower, field_upper)
    white_samples = collect_seed_box_samples(hsv, [box for box in white_boxes if box is not None], field_lower, field_upper)
    return fit_team_color_profile(red_samples, white_samples, min_pixels=min_pixels)


def manual_seed_ball_candidate(
    seed_data: Optional[Dict[str, Any]],
    frame_idx: int,
    frame_w: int,
    frame_h: int,
) -> Optional[BallCandidate]:
    if seed_data is None:
        return None
    if int(seed_data.get("frame_index", -1)) != frame_idx:
        return None

    ball_box = seed_data.get("ball_box")
    if ball_box is None:
        return None
    normalized = seed_box_for_frame(seed_data, ball_box, frame_w, frame_h)
    if normalized is None:
        return None

    x, y, w, h = normalized
    center = (x + w / 2.0, y + h / 2.0)
    radius = max(w, h) / 2.0
    return BallCandidate(center=center, radius=radius, score=10.0)


def center_inside_player_upper_body(
    center: Point,
    player_boxes: Sequence[BBox],
    upper_ratio: float = 0.68,
) -> bool:
    cx, cy = center
    for x, y, w, h in player_boxes:
        upper_y2 = y + max(2, int(round(h * upper_ratio)))
        if (x - 1) <= cx <= (x + w + 1) and (y - 1) <= cy <= upper_y2:
            return True
    return False


def center_inside_player_box(
    center: Point,
    player_boxes: Sequence[BBox],
    lower_ratio: float = 1.0,
    pad: int = 1,
) -> bool:
    cx, cy = center
    for x, y, w, h in player_boxes:
        y2 = y + max(2, int(round(h * lower_ratio)))
        if (x - pad) <= cx <= (x + w + pad) and (y - pad) <= cy <= (y2 + pad):
            return True
    return False


def player_foot_zone_bonus(center: Point, player_boxes: Sequence[BBox]) -> float:
    cx, cy = center
    best = 0.0
    for x, y, w, h in player_boxes:
        pad_x = max(3, int(round(w * 0.45)))
        pad_bottom = max(4, int(round(h * 0.24)))
        zone_x1 = x - pad_x
        zone_x2 = x + w + pad_x
        zone_y1 = y + max(1, int(round(h * 0.54)))
        zone_y2 = y + h + pad_bottom
        if not (zone_x1 <= cx <= zone_x2 and zone_y1 <= cy <= zone_y2):
            continue

        foot_x = x + 0.5 * w
        foot_y = y + 0.94 * h
        norm_dx = abs(cx - foot_x) / float(max(zone_x2 - zone_x1, 1))
        norm_dy = abs(cy - foot_y) / float(max(zone_y2 - zone_y1, 1))
        zone_score = max(0.0, 1.0 - (0.90 * norm_dx + 1.15 * norm_dy))
        if y <= cy <= y + int(round(h * 0.78)):
            zone_score *= 0.45
        best = max(best, zone_score)
    return 0.28 * best


def expand_player_color_box(
    abs_x: int,
    abs_y: int,
    wb: int,
    hb: int,
    frame_w: int,
    frame_h: int,
    top_y: int,
) -> Optional[BBox]:
    pad_x = max(2, int(round(wb * 0.45)))
    pad_top = max(1, int(round(hb * 0.10)))
    pad_bottom = max(4, int(round(hb * 1.00)))

    rx = max(0, abs_x - pad_x)
    ry = max(top_y, abs_y - pad_top)
    rx2 = min(frame_w, abs_x + wb + pad_x)
    ry2 = min(frame_h, abs_y + hb + pad_bottom)
    rw = rx2 - rx
    rh = ry2 - ry
    if rw < 4 or rh < 7:
        return None
    return (rx, ry, rw, rh)


def score_ball_candidate(
    hsv: np.ndarray,
    bbox: BBox,
    area: float,
    color_profile: Optional[TeamColorProfile] = None,
    player_boxes: Sequence[BBox] = (),
    min_white_ratio: float = 0.24,
    min_isolation: float = 0.05,
) -> Optional[BallCandidate]:
    x, y, w, h = bbox
    roi = hsv[y : y + h, x : x + w]
    if roi.size == 0:
        return None

    profile = color_profile or TeamColorProfile()
    center = (x + w / 2.0, y + h / 2.0)
    inside_upper_body = center_inside_player_upper_body(center, player_boxes)
    inside_player_body = center_inside_player_box(center, player_boxes, lower_ratio=0.84)
    inside_player_any = center_inside_player_box(center, player_boxes)
    foot_bonus = player_foot_zone_bonus(center, player_boxes)

    white = make_ball_white_mask(roi, profile)
    white_ratio = cv2.countNonZero(white) / float(max(w * h, 1))
    if white_ratio < min_white_ratio:
        return None

    radius = max(w, h) / 2.0
    circ = area / (math.pi * (radius * radius + 1e-6))
    circ_score = 1.0 - min(abs(1.0 - circ), 1.0)
    density = area / float(max(w * h, 1))
    density_score = 1.0 - min(abs(0.72 - density) / 0.72, 1.0)
    mean_v = float(np.mean(roi[:, :, 2]))
    bright_score = max(0.0, min(1.0, (mean_v - 132.0) / 108.0))

    pad = max(2, int(round(max(w, h) * 0.9)))
    frame_h, frame_w = hsv.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame_w, x + w + pad)
    y2 = min(frame_h, y + h + pad)
    outer = hsv[y1:y2, x1:x2]
    outer_white = make_ball_white_mask(outer, profile)
    ring = np.full(outer_white.shape, 255, dtype=np.uint8)
    ring[(y - y1) : (y - y1 + h), (x - x1) : (x - x1 + w)] = 0
    ring_px = cv2.countNonZero(ring)
    ring_white = 0.0
    if ring_px > 0:
        ring_white = cv2.countNonZero(cv2.bitwise_and(outer_white, ring)) / float(ring_px)

    isolation = max(0.0, white_ratio - 0.72 * ring_white)
    if isolation < min_isolation and ring_white > 0.26:
        return None
    if inside_upper_body and white_ratio < 0.34 and isolation < max(0.08, min_isolation + 0.02):
        return None
    if inside_player_body and foot_bonus < 0.05 and isolation < max(0.07, min_isolation + 0.02):
        return None
    if inside_player_any and foot_bonus < 0.04 and white_ratio < max(0.28, min_white_ratio + 0.04) and circ_score < 0.36:
        return None

    score = (
        (1.18 * white_ratio)
        + (0.72 * isolation)
        + (0.42 * circ_score)
        + (0.18 * density_score)
        + (0.16 * bright_score)
        + foot_bonus
        - (0.25 * ring_white)
        - (0.28 if inside_upper_body else 0.0)
        - (0.18 if inside_player_body and foot_bonus < 0.08 else 0.0)
    )
    if score <= 0.18:
        return None
    return BallCandidate(center=center, radius=radius, score=score)


def classify_team(
    roi_hsv: np.ndarray,
    field_lower: np.ndarray,
    field_upper: np.ndarray,
    color_profile: Optional[TeamColorProfile] = None,
) -> str:
    if roi_hsv.size == 0:
        return TEAM_OTHER

    profile = color_profile or TeamColorProfile()
    upper = roi_hsv[: max(1, int(roi_hsv.shape[0] * 0.6)), :]
    field = cv2.inRange(upper, field_lower, field_upper)
    valid = cv2.bitwise_not(field)
    valid_px = cv2.countNonZero(valid)
    if valid_px < 10:
        return TEAM_OTHER

    red = make_red_mask(upper, profile)
    white = make_white_mask(upper, profile)
    r = cv2.countNonZero(cv2.bitwise_and(red, valid)) / float(valid_px)
    w = cv2.countNonZero(cv2.bitwise_and(white, valid)) / float(valid_px)

    sat_vals = upper[:, :, 1][valid > 0]
    val_vals = upper[:, :, 2][valid > 0]
    mean_sat = float(np.mean(sat_vals)) if sat_vals.size > 0 else 255.0
    mean_val = float(np.mean(val_vals)) if val_vals.size > 0 else 0.0

    tiny = (upper.shape[0] * upper.shape[1]) <= 140
    if (r > 0.085 and r > 0.95 * w) or (tiny and r > 0.055 and mean_sat > max(78.0, profile.red_sat_min + 8.0)):
        return TEAM_RED
    if w > 0.075 or (
        mean_sat < max(92.0, profile.white_sat_max + 20.0)
        and mean_val > max(108.0, profile.white_val_min - 18.0)
    ):
        return TEAM_LIGHT
    return TEAM_OTHER


def build_team_color_masks(
    hsv: np.ndarray,
    playable: np.ndarray,
    field_lower: np.ndarray,
    field_upper: np.ndarray,
    color_profile: Optional[TeamColorProfile] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    profile = color_profile or TeamColorProfile()
    non_green = cv2.bitwise_not(cv2.inRange(hsv, field_lower, field_upper))

    red = make_red_mask(hsv, profile)
    white = make_white_mask(hsv, profile)

    red = cv2.bitwise_and(red, playable)
    red = cv2.bitwise_and(red, non_green)
    white = cv2.bitwise_and(white, playable)
    white = cv2.bitwise_and(white, non_green)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel, iterations=1)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel, iterations=1)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel, iterations=1)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel, iterations=1)

    return red, white


def recover_local_team_detection(
    team_mask: np.ndarray,
    track: Track,
    top_y: int,
    local_window: int,
) -> Optional[Detection]:
    frame_h, frame_w = team_mask.shape[:2]
    cx, cy = int(round(track.last_position[0])), int(round(track.last_position[1]))

    half_w = max(local_window, int(round(track.bbox[2] * 2.2)))
    half_h = max(local_window, int(round(track.bbox[3] * 1.7)))

    x1 = max(0, cx - half_w)
    y1 = max(top_y, cy - half_h)
    x2 = min(frame_w, cx + half_w + 1)
    y2 = min(frame_h, cy + half_h + 1)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None

    roi_mask = team_mask[y1:y2, x1:x2]
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    prev_box_area = float(max(track.bbox[2] * track.bbox[3], 1))
    best_detection: Optional[Detection] = None
    best_score = float("-inf")

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < max(6.0, prev_box_area * 0.035):
            continue
        if area > max(900.0, prev_box_area * 1.35):
            continue

        xb, yb, wb, hb = cv2.boundingRect(contour)
        if wb < 2 or hb < 3:
            continue

        # Expand color blobs downward so the recovered box better matches full-player shape.
        abs_x = x1 + xb
        abs_y = y1 + yb
        expanded = expand_player_color_box(abs_x, abs_y, wb, hb, frame_w, frame_h, top_y)
        if expanded is None:
            continue
        rx, ry, rw, rh = expanded

        aspect = rh / float(max(rw, 1))
        if aspect < 1.1 or aspect > 7.0:
            continue

        center = (rx + rw / 2.0, ry + rh / 2.0)
        dist = math.dist(center, track.last_position)
        area_ratio = area / prev_box_area
        aspect_penalty = abs(aspect - 2.2)
        ratio_bonus = 1.0 if 0.03 <= area_ratio <= 0.75 else 0.5
        score = ratio_bonus - (0.028 * dist) - (0.18 * aspect_penalty)

        if score > best_score:
            best_score = score
            best_detection = Detection((rx, ry, rw, rh), center, track.team_label)

    return best_detection


def recover_players_from_tracks(
    tracks: Sequence[Track],
    detections: Sequence[Detection],
    red_mask: np.ndarray,
    white_mask: np.ndarray,
    top_y: int,
    local_window: int,
    max_recover_missed: int,
) -> List[Detection]:
    recovered: List[Detection] = []

    for track in tracks:
        if track.team_label not in (TEAM_RED, TEAM_LIGHT):
            continue
        if track.missed > max_recover_missed:
            continue

        near_existing = False
        min_keepout = max(18.0, track.bbox[2] * 0.9, track.bbox[3] * 0.45)
        for det in list(detections) + recovered:
            if math.dist(det.centroid, track.last_position) <= min_keepout:
                near_existing = True
                break
        if near_existing:
            continue

        team_mask = red_mask if track.team_label == TEAM_RED else white_mask
        recovered_det = recover_local_team_detection(team_mask, track, top_y, local_window)
        if recovered_det is not None:
            recovered.append(recovered_det)

    return recovered


def detect_players_from_team_masks(
    red_mask: np.ndarray,
    white_mask: np.ndarray,
    mot: np.ndarray,
    top_y: int,
) -> List[Detection]:
    def detect_one(team_mask: np.ndarray, team_label: str) -> List[Tuple[float, Detection]]:
        frame_h, frame_w = team_mask.shape[:2]
        contours, _ = cv2.findContours(team_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scored: List[Tuple[float, Detection]] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 9.0 or area > 720.0:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if y < top_y or w < 2 or h < 3 or w > 62 or h > 92:
                continue

            fill_ratio = area / float(max(w * h, 1))
            if fill_ratio < 0.18:
                continue

            expanded = expand_player_color_box(x, y, w, h, frame_w, frame_h, top_y)
            if expanded is None:
                continue
            rx, ry, rw, rh = expanded

            aspect = rh / float(max(rw, 1))
            if aspect < 1.1 or aspect > 7.0:
                continue

            motion_roi = mot[ry : ry + rh, rx : rx + rw]
            motion_pixels = cv2.countNonZero(motion_roi)
            motion_ratio = motion_pixels / float(max(rw * rh, 1))
            if motion_pixels < 6 or motion_ratio < 0.02:
                continue

            center = (rx + rw / 2.0, ry + rh / 2.0)
            score = fill_ratio + min(0.9, area / 80.0) + (1.6 * min(0.18, motion_ratio)) - (0.06 * abs(aspect - 2.3))
            scored.append((score, Detection((rx, ry, rw, rh), center, team_label)))

        return scored

    scored = detect_one(red_mask, TEAM_RED) + detect_one(white_mask, TEAM_LIGHT)
    scored.sort(key=lambda item: item[0], reverse=True)
    return [det for _, det in scored]


def merge_detections(primary: Sequence[Detection], extras: Sequence[Detection]) -> List[Detection]:
    merged = list(primary)
    for det in extras:
        keepout = max(14.0, det.bbox[2] * 0.55, det.bbox[3] * 0.28)
        overlap = False
        for existing in merged:
            existing_keepout = max(14.0, existing.bbox[2] * 0.55, existing.bbox[3] * 0.28)
            if math.dist(det.centroid, existing.centroid) <= max(keepout, existing_keepout):
                overlap = True
                break
        if not overlap:
            merged.append(det)
    return merged


def detect_players(
    hsv: np.ndarray,
    mot: np.ndarray,
    field_lower: np.ndarray,
    field_upper: np.ndarray,
    top_y: int,
    min_area: float,
    max_area: float,
    min_aspect: float,
    max_aspect: float,
    min_nongreen: float,
    color_profile: Optional[TeamColorProfile] = None,
) -> List[Detection]:
    contours, _ = cv2.findContours(mot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets: List[Detection] = []
    frame_h = mot.shape[0]
    depth_denom = float(max(1, frame_h - top_y))

    for c in contours:
        area = float(cv2.contourArea(c))
        if area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(c)
        if y < top_y or w < 3 or h < 7:
            continue

        asp = h / float(max(w, 1))
        if asp < min_aspect or asp > max_aspect:
            continue

        depth = (y + h - top_y) / depth_denom
        depth = min(max(depth, 0.0), 1.0)
        adaptive_min_area = min_area * (0.42 + 0.58 * depth)
        if area < adaptive_min_area:
            continue

        roi = hsv[y : y + h, x : x + w]
        green = cv2.inRange(roi, field_lower, field_upper)
        nongreen = 1.0 - cv2.countNonZero(green) / float(max(w * h, 1))
        adaptive_nongreen = min_nongreen * (0.72 + 0.28 * depth)
        if nongreen < adaptive_nongreen:
            continue

        team = classify_team(roi, field_lower, field_upper, color_profile=color_profile)
        dets.append(Detection((x, y, w, h), (x + w / 2.0, y + h / 2.0), team))
    return dets


def detect_ball(
    hsv: np.ndarray,
    mot: np.ndarray,
    top_y: int,
    min_area: float,
    max_area: float,
    color_profile: Optional[TeamColorProfile] = None,
    player_boxes: Sequence[BBox] = (),
) -> List[BallCandidate]:
    contours, _ = cv2.findContours(mot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands: List[BallCandidate] = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(c)
        if y < top_y or w < 2 or h < 2 or w > 26 or h > 26:
            continue

        asp = w / float(max(h, 1))
        if asp < 0.45 or asp > 2.2:
            continue

        candidate = score_ball_candidate(
            hsv=hsv,
            bbox=(x, y, w, h),
            area=area,
            color_profile=color_profile,
            player_boxes=player_boxes,
            min_white_ratio=0.24,
            min_isolation=0.05,
        )
        if candidate is not None:
            cands.append(candidate)

    cands.sort(key=lambda z: z.score, reverse=True)
    return cands[:10]


def detect_ball_local_near(
    hsv: np.ndarray,
    playable: np.ndarray,
    center: Point,
    field_lower: np.ndarray,
    field_upper: np.ndarray,
    window: int,
    min_area: float,
    max_area: float,
    color_profile: Optional[TeamColorProfile] = None,
    player_boxes: Sequence[BBox] = (),
) -> List[BallCandidate]:
    h, w = hsv.shape[:2]
    cx, cy = int(round(center[0])), int(round(center[1]))
    x1 = max(0, cx - window)
    y1 = max(0, cy - window)
    x2 = min(w, cx + window + 1)
    y2 = min(h, cy + window + 1)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return []

    roi_hsv = hsv[y1:y2, x1:x2]
    roi_play = playable[y1:y2, x1:x2]

    profile = color_profile or TeamColorProfile()
    bright = make_ball_white_mask(roi_hsv, profile)
    non_green = cv2.bitwise_not(cv2.inRange(roi_hsv, field_lower, field_upper))
    mask = cv2.bitwise_and(bright, roi_play)
    mask = cv2.bitwise_and(mask, non_green)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[BallCandidate] = []
    min_a = max(3.0, min_area * 0.5)
    max_a = max_area * 1.4

    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_a or area > max_a:
            continue

        xb, yb, wb, hb = cv2.boundingRect(c)
        if wb < 2 or hb < 2 or wb > 26 or hb > 26:
            continue

        asp = wb / float(max(hb, 1))
        if asp < 0.45 or asp > 2.2:
            continue

        ax = x1 + xb + wb / 2.0
        ay = y1 + yb + hb / 2.0
        dist = math.dist((ax, ay), center)
        prox_bonus = max(0.0, 0.35 - (dist / float(max(window, 1))))
        candidate = score_ball_candidate(
            hsv=hsv,
            bbox=(x1 + xb, y1 + yb, wb, hb),
            area=area,
            color_profile=color_profile,
            player_boxes=player_boxes,
            min_white_ratio=0.20,
            min_isolation=0.03,
        )
        if candidate is not None:
            candidate.score += prox_bonus
            out.append(candidate)

    out.sort(key=lambda z: z.score, reverse=True)
    return out[:6]


def detect_ball_near_players(
    hsv: np.ndarray,
    playable: np.ndarray,
    player_boxes: Sequence[BBox],
    top_y: int,
    field_lower: np.ndarray,
    field_upper: np.ndarray,
    min_area: float,
    max_area: float,
    color_profile: Optional[TeamColorProfile] = None,
) -> List[BallCandidate]:
    if not player_boxes:
        return []

    profile = color_profile or TeamColorProfile()
    frame_h, frame_w = hsv.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out: List[BallCandidate] = []
    min_a = max(2.5, min_area * 0.35)
    max_a = min(max_area * 0.95, 145.0)

    for px, py, pw, ph in player_boxes:
        if py < top_y or pw < 4 or ph < 8:
            continue

        pad_x = max(5, int(round(pw * 0.55)))
        pad_bottom = max(4, int(round(ph * 0.24)))
        x1 = max(0, px - pad_x)
        y1 = max(top_y, py + int(round(ph * 0.48)))
        x2 = min(frame_w, px + pw + pad_x)
        y2 = min(frame_h, py + ph + pad_bottom)
        if x2 - x1 < 5 or y2 - y1 < 4:
            continue

        roi_hsv = hsv[y1:y2, x1:x2]
        roi_play = playable[y1:y2, x1:x2]
        bright = make_ball_white_mask(roi_hsv, profile)
        non_green = cv2.bitwise_not(cv2.inRange(roi_hsv, field_lower, field_upper))
        mask = cv2.bitwise_and(bright, roi_play)
        mask = cv2.bitwise_and(mask, non_green)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_a or area > max_a:
                continue

            xb, yb, wb, hb = cv2.boundingRect(contour)
            if wb < 2 or hb < 2 or wb > 20 or hb > 20:
                continue

            aspect = wb / float(max(hb, 1))
            if aspect < 0.45 or aspect > 2.2:
                continue

            abs_box = (x1 + xb, y1 + yb, wb, hb)
            center = (abs_box[0] + wb / 2.0, abs_box[1] + hb / 2.0)
            if center_inside_player_upper_body(center, [(px, py, pw, ph)], upper_ratio=0.82):
                continue
            if center_inside_player_box(center, [(px, py, pw, ph)], lower_ratio=0.90) and center[1] < (py + 0.82 * ph):
                continue

            candidate = score_ball_candidate(
                hsv=hsv,
                bbox=abs_box,
                area=area,
                color_profile=color_profile,
                player_boxes=player_boxes,
                min_white_ratio=0.16,
                min_isolation=0.02,
            )
            if candidate is None:
                continue

            foot_bonus = player_foot_zone_bonus(candidate.center, [(px, py, pw, ph)])
            if foot_bonus <= 0.01:
                continue
            candidate.score += 0.16 + (1.2 * foot_bonus)
            out.append(candidate)

    if not out:
        return []

    out.sort(key=lambda z: z.score, reverse=True)
    deduped: List[BallCandidate] = []
    for cand in out:
        if all(math.dist(cand.center, other.center) > 4.0 for other in deduped):
            deduped.append(cand)
        if len(deduped) >= 8:
            break
    return deduped


def assign(cost: np.ndarray, max_cost: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    nt, nd = cost.shape
    if nt == 0 or nd == 0:
        return [], list(range(nt)), list(range(nd))

    matched: List[Tuple[int, int]] = []
    used_t = set()
    used_d = set()

    if SCIPY_AVAILABLE and linear_sum_assignment is not None:
        # Gate before optimization so an invalid edge cannot steal a valid pairing.
        allowed = np.isfinite(cost) & (cost <= max_cost)
        if not np.any(allowed):
            return [], list(range(nt)), list(range(nd))
        offset = min(0.0, float(np.min(cost[allowed])))
        valid_cost = cost.astype(np.float64) - offset
        forbidden = (float(np.max(valid_cost[allowed])) + 1.0) * (min(nt, nd) + 1)
        rr, cc = linear_sum_assignment(np.where(allowed, valid_cost, forbidden))
        for r, c in zip(rr.tolist(), cc.tolist()):
            if cost[r, c] <= max_cost:
                matched.append((r, c))
                used_t.add(r)
                used_d.add(c)
    else:
        tmp = cost.copy()
        while True:
            idx = int(np.argmin(tmp))
            r, c = np.unravel_index(idx, tmp.shape)
            v = float(tmp[r, c])
            if not np.isfinite(v) or v > max_cost:
                break
            matched.append((r, c))
            used_t.add(r)
            used_d.add(c)
            tmp[r, :] = np.inf
            tmp[:, c] = np.inf

    um_t = [i for i in range(nt) if i not in used_t]
    um_d = [j for j in range(nd) if j not in used_d]
    return matched, um_t, um_d

class PlayerTracker:
    def __init__(
        self,
        fps: float,
        max_missed: int,
        max_match_distance: float,
        trail_length: int,
        class_mismatch_penalty: float,
    ) -> None:
        self.fps = max(fps, 1.0)
        self.max_missed = max_missed
        self.max_match_distance = max_match_distance
        self.trail_length = trail_length
        self.class_mismatch_penalty = class_mismatch_penalty
        self.next_id = 1
        self.active: Dict[int, Track] = {}
        self.finished: Dict[int, Track] = {}

    def _new_track(self, det: Detection, frame_idx: int) -> None:
        votes: Dict[str, int] = {}
        if det.team_label != TEAM_OTHER:
            votes[det.team_label] = 1
        tr = Track(
            track_id=self.next_id,
            kalman=create_kalman(det.centroid, process_noise=1e-2, meas_noise=1e-1),
            bbox=det.bbox,
            last_position=det.centroid,
            last_update_frame=frame_idx,
            last_measured_position=det.centroid,
            team_label=det.team_label,
            label_votes=votes,
            trail=deque([det.centroid], maxlen=self.trail_length),
        )
        self.active[self.next_id] = tr
        self.next_id += 1

    def _penalty(self, tr_label: str, det_label: str) -> float:
        if tr_label != TEAM_OTHER and det_label != TEAM_OTHER and tr_label != det_label:
            return self.class_mismatch_penalty
        if tr_label == TEAM_OTHER and det_label != TEAM_OTHER:
            return self.class_mismatch_penalty * 0.35
        return 0.0

    def update(self, detections: Sequence[Detection], frame_idx: int) -> List[Track]:
        if not self.active:
            for d in detections:
                self._new_track(d, frame_idx)
            return list(self.active.values())

        tids = list(self.active.keys())
        pred: Dict[int, Point] = {}
        for tid in tids:
            s = self.active[tid].kalman.predict()
            pred[tid] = (float(s[0, 0]), float(s[1, 0]))

        cost = np.full((len(tids), len(detections)), np.inf, dtype=np.float32)
        for i, tid in enumerate(tids):
            tlabel = self.active[tid].team_label
            px, py = pred[tid]
            for j, d in enumerate(detections):
                dist = math.hypot(px - d.centroid[0], py - d.centroid[1])
                cost[i, j] = dist + self._penalty(tlabel, d.team_label)

        matched, um_t_rows, um_d_cols = assign(cost, self.max_match_distance)

        for r, c in matched:
            tid = tids[r]
            d = detections[c]
            tr = self.active[tid]

            m = np.array([[np.float32(d.centroid[0])], [np.float32(d.centroid[1])]], dtype=np.float32)
            corr = tr.kalman.correct(m)
            pos = (float(corr[0, 0]), float(corr[1, 0]))

            dt = max(1, frame_idx - tr.last_update_frame)
            speed = math.dist(tr.last_measured_position or tr.last_position, pos) * (self.fps / dt)
            tr.last_measured_position = pos
            tr.confirmed_points += 1
            tr.speeds_px_per_s.append(speed)
            tr.last_position = pos
            tr.last_update_frame = frame_idx
            tr.bbox = d.bbox
            tr.missed = 0
            tr.trail.append(pos)

            if d.team_label != TEAM_OTHER:
                tr.label_votes[d.team_label] = tr.label_votes.get(d.team_label, 0) + 1
                tr.team_label = max(tr.label_votes, key=tr.label_votes.get)

        for r in um_t_rows:
            tid = tids[r]
            if tid not in self.active:
                continue
            tr = self.active[tid]
            tr.missed += 1
            tr.last_position = pred[tid]

        dead = [tid for tid, tr in self.active.items() if tr.missed > self.max_missed]
        for tid in dead:
            self.finished[tid] = self.active.pop(tid)

        for c in um_d_cols:
            self._new_track(detections[c], frame_idx)

        return list(self.active.values())

    def all_tracks(self) -> Dict[int, Track]:
        out = dict(self.finished)
        out.update(self.active)
        return out


class BallTracker:
    def __init__(self, max_missed: int, max_dist: float, trail_len: int) -> None:
        self.max_missed = max_missed
        self.max_dist = max_dist
        self.trail_len = trail_len
        self.track: Optional[BallTrack] = None

    def update(self, candidates: Sequence[BallCandidate]) -> Optional[BallTrack]:
        if self.track is None:
            if candidates:
                c = candidates[0]
                self.track = BallTrack(
                    kalman=create_kalman(c.center, process_noise=5e-2, meas_noise=4e-1),
                    position=c.center,
                    radius=max(2.0, c.radius),
                    trail=deque([c.center], maxlen=self.trail_len),
                )
            return self.track

        s = self.track.kalman.predict()
        pred = (float(s[0, 0]), float(s[1, 0]))

        best: Optional[BallCandidate] = None
        best_adj = float("inf")
        best_dist = float("inf")
        best_near: Optional[BallCandidate] = None
        best_near_adj = float("inf")
        best_near_dist = float("inf")
        strongest: Optional[BallCandidate] = None
        strongest_score = float("-inf")
        for c in candidates:
            dist = math.dist(pred, c.center)
            score_bonus = min(max(c.score, 0.0), 1.65) * 14.0
            adj = dist - score_bonus
            if c.score > strongest_score:
                strongest = c
                strongest_score = c.score
            if dist <= self.max_dist and adj < best_near_adj:
                best_near = c
                best_near_adj = adj
                best_near_dist = dist
            if dist > (self.max_dist * 0.82) and c.score < 0.85:
                continue
            if adj < best_adj:
                best = c
                best_adj = adj
                best_dist = dist

        if strongest is not None and best_near is not None:
            strong_dist = math.dist(pred, strongest.center)
            if (
                strongest.score >= 1.18
                and strong_dist <= max(self.max_dist * 4.0, 220.0)
                and strongest.score >= (best_near.score + 0.42)
            ):
                best = strongest
                best_dist = strong_dist

        if strongest is not None and best_near is None:
            strong_dist = math.dist(pred, strongest.center)
            if strongest.score >= 1.05 and strong_dist <= max(self.max_dist * 5.0, 260.0):
                best = strongest
                best_dist = strong_dist

        if best is not None and best_dist <= self.max_dist:
            m = np.array([[np.float32(best.center[0])], [np.float32(best.center[1])]], dtype=np.float32)
            corr = self.track.kalman.correct(m)
            pos = (float(corr[0, 0]), float(corr[1, 0]))
            self.track.position = pos
            self.track.radius = max(2.0, best.radius)
            self.track.missed = 0
            self.track.trail.append(pos)
        elif best is not None and best.score >= 1.05 and best_dist <= max(self.max_dist * 5.0, 260.0):
            self.track = BallTrack(
                kalman=create_kalman(best.center, process_noise=5e-2, meas_noise=4e-1),
                position=best.center,
                radius=max(2.0, best.radius),
                trail=deque([best.center], maxlen=self.trail_len),
            )
        else:
            self.track.position = pred
            self.track.missed += 1
            self.track.trail.append(pred)

        if self.track.missed > self.max_missed:
            self.track = None
        return self.track


def draw_player(img: np.ndarray, tr: Track) -> None:
    x, y, w, h = tr.bbox
    color = TEAM_COLOR.get(tr.team_label, TEAM_COLOR[TEAM_OTHER])
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    label = f"{TEAM_NAME.get(tr.team_label, 'OTHER')} #{tr.track_id}"
    cv2.putText(img, label, (x, max(16, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    pts = [(int(round(a)), int(round(b))) for (a, b) in tr.trail]
    for i in range(1, len(pts)):
        cv2.line(img, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)


def draw_ball(img: np.ndarray, ball: BallTrack) -> None:
    cx, cy = int(round(ball.position[0])), int(round(ball.position[1]))
    r = max(6, int(round(ball.radius * 1.6)))
    cv2.circle(img, (cx, cy), r + 2, (0, 0, 0), 2)
    cv2.circle(img, (cx, cy), r, (0, 255, 255), 2)
    cv2.circle(img, (cx, cy), max(2, r // 3), (0, 255, 255), -1)

    label_pos = (cx + r + 6, max(18, cy - r - 2))
    (tw, th), baseline = cv2.getTextSize("BALL", cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    bx1 = label_pos[0] - 3
    by1 = label_pos[1] - th - 3
    bx2 = label_pos[0] + tw + 4
    by2 = label_pos[1] + baseline + 3
    cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
    cv2.putText(img, "BALL", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)


def shade_crowd(img: np.ndarray, top_y: int) -> None:
    if top_y <= 0:
        return
    h, w = img.shape[:2]
    top_y = min(top_y, h - 1)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w - 1, top_y), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.28, img, 0.72, 0.0, dst=img)
    cv2.line(img, (0, top_y), (w - 1, top_y), (255, 0, 255), 2)


def write_metrics(path: Path, tracks: Dict[int, Track]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "team_label", "points", "trail_points", "avg_speed_px_s", "max_speed_px_s"])
        for tid in sorted(tracks.keys()):
            tr = tracks[tid]
            avg = float(np.mean(tr.speeds_px_per_s)) if tr.speeds_px_per_s else 0.0
            mx = float(np.max(tr.speeds_px_per_s)) if tr.speeds_px_per_s else 0.0
            w.writerow([tid, tr.team_label, tr.confirmed_points, len(tr.trail), f"{avg:.3f}", f"{mx:.3f}"])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Soccer tracking with team grouping + ball + crowd suppression")
    p.add_argument("--video", type=Path, default=infer_default_video())
    p.add_argument("--output", type=Path, default=Path("outputs/classical"))
    p.add_argument("--seed-file", type=Path, default=None)
    p.add_argument("--start-sec", type=float, default=0.0)
    p.add_argument("--duration-sec", type=float, default=15.0)
    p.add_argument("--resize-width", type=int, default=960)

    p.add_argument("--field-hsv-lower", type=parse_hsv_triplet, default=parse_hsv_triplet("28,25,25"))
    p.add_argument("--field-hsv-upper", type=parse_hsv_triplet, default=parse_hsv_triplet("95,255,255"))
    p.add_argument("--row-green-threshold", type=float, default=0.30)
    p.add_argument("--row-green-run", type=int, default=8)
    p.add_argument("--crowd-top-cut", type=int, default=0)

    p.add_argument("--min-area", type=float, default=80.0)
    p.add_argument("--max-area", type=float, default=3600.0)
    p.add_argument("--min-aspect", type=float, default=1.1)
    p.add_argument("--max-aspect", type=float, default=6.0)
    p.add_argument("--min-nongreen", type=float, default=0.10)

    p.add_argument("--max-match-distance", type=float, default=78.0)
    p.add_argument("--class-mismatch-penalty", type=float, default=55.0)
    p.add_argument("--max-missed", type=int, default=24)
    p.add_argument("--trail-length", type=int, default=60)

    p.add_argument("--ball-min-area", type=float, default=6.0)
    p.add_argument("--ball-max-area", type=float, default=180.0)
    p.add_argument("--ball-max-dist", type=float, default=68.0)
    p.add_argument("--ball-max-missed", type=int, default=18)
    p.add_argument("--ball-local-window", type=int, default=85)
    p.add_argument("--team-local-window", type=int, default=58)
    p.add_argument("--team-recover-max-missed", type=int, default=4)
    p.add_argument("--enable-global-team-mask-recovery", action="store_true")
    p.add_argument("--calibration-frames", type=int, default=45)
    p.add_argument("--calibration-min-pixels", type=int, default=260)
    p.add_argument("--manual-seed-min-pixels", type=int, default=110)

    p.add_argument("--warmup-frames", type=int, default=30)
    p.add_argument("--save-masks", action="store_true")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--random-seed", type=int, default=0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    validate_run(args)
    started = time.perf_counter()
    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")
    manual_seed = load_manual_seed(args.seed_file)

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
    ptracker = PlayerTracker(
        fps=fps,
        max_missed=args.max_missed,
        max_match_distance=args.max_match_distance,
        trail_length=args.trail_length,
        class_mismatch_penalty=args.class_mismatch_penalty,
    )
    btracker = BallTracker(args.ball_max_missed, args.ball_max_dist, max(20, args.trail_length))

    writer: Optional[cv2.VideoWriter] = None
    processed = 0
    frame_idx = start
    det_total = 0
    det_team = Counter()
    ball_visible = 0
    ball_measured = 0
    frame_records = []
    color_profile = TeamColorProfile()
    calibration_red_samples: List[np.ndarray] = []
    calibration_white_samples: List[np.ndarray] = []
    calibration_complete = False

    if manual_seed is not None:
        seeded_profile = build_profile_from_manual_seed(
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

        frame = resize_by_width(frame, args.resize_width)
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out / "annotated_tracking.mp4"), fourcc, fps, (frame.shape[1], frame.shape[0]))
            if not writer.isOpened():
                cap.release()
                raise SystemExit("Could not create MP4 output; check codec and output permissions.")

        hsv, pitch, playable, top_y = segment_field(
            frame,
            args.field_hsv_lower,
            args.field_hsv_upper,
            args.row_green_threshold,
            args.row_green_run,
            args.crowd_top_cut,
        )
        mot = motion_mask(frame, bg, playable)

        if processed < args.warmup_frames:
            players = []
            ball_cands = []
        else:
            calibration_window_open = processed < (args.warmup_frames + args.calibration_frames)
            red_mask, white_mask = build_team_color_masks(
                hsv=hsv,
                playable=playable,
                field_lower=args.field_hsv_lower,
                field_upper=args.field_hsv_upper,
                color_profile=color_profile,
            )
            players = detect_players(
                hsv,
                mot,
                args.field_hsv_lower,
                args.field_hsv_upper,
                top_y,
                args.min_area,
                args.max_area,
                args.min_aspect,
                args.max_aspect,
                args.min_nongreen,
                color_profile=color_profile,
            )
            if args.enable_global_team_mask_recovery:
                color_mask_players = detect_players_from_team_masks(red_mask, white_mask, mot, top_y)
                players = merge_detections(players, color_mask_players)
            if calibration_window_open and not calibration_complete:
                sample_red, sample_white = collect_calibration_pixels(
                    detections=players,
                    hsv=hsv,
                    field_lower=args.field_hsv_lower,
                    field_upper=args.field_hsv_upper,
                )
                calibration_red_samples.extend(sample_red)
                calibration_white_samples.extend(sample_white)

                calibrated = fit_team_color_profile(
                    red_samples=calibration_red_samples,
                    white_samples=calibration_white_samples,
                    min_pixels=args.calibration_min_pixels,
                )
                if calibrated is not None:
                    color_profile = calibrated
                    calibration_complete = True
                    red_mask, white_mask = build_team_color_masks(
                        hsv=hsv,
                        playable=playable,
                        field_lower=args.field_hsv_lower,
                        field_upper=args.field_hsv_upper,
                        color_profile=color_profile,
                    )
                    players = detect_players(
                        hsv,
                        mot,
                        args.field_hsv_lower,
                        args.field_hsv_upper,
                        top_y,
                        args.min_area,
                        args.max_area,
                        args.min_aspect,
                        args.max_aspect,
                        args.min_nongreen,
                        color_profile=color_profile,
                    )
                    if args.enable_global_team_mask_recovery:
                        color_mask_players = detect_players_from_team_masks(red_mask, white_mask, mot, top_y)
                        players = merge_detections(players, color_mask_players)

            recovered_players = recover_players_from_tracks(
                tracks=list(ptracker.active.values()),
                detections=players,
                red_mask=red_mask,
                white_mask=white_mask,
                top_y=top_y,
                local_window=args.team_local_window,
                max_recover_missed=args.team_recover_max_missed,
            )
            players = merge_detections(players, recovered_players)
            ball_keepout_boxes = [d.bbox for d in players]
            ball_cands = detect_ball(
                hsv,
                mot,
                top_y,
                args.ball_min_area,
                args.ball_max_area,
                color_profile=color_profile,
                player_boxes=ball_keepout_boxes,
            )
            if btracker.track is not None:
                local_ball = detect_ball_local_near(
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
                ball_cands.extend(local_ball)
            ball_cands.extend(
                detect_ball_near_players(
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
                ball_cands = sorted(ball_cands, key=lambda c: c.score, reverse=True)
                deduped: List[BallCandidate] = []
                for cand in ball_cands:
                    if all(math.dist(cand.center, d.center) > 4.0 for d in deduped):
                        deduped.append(cand)
                    if len(deduped) >= 12:
                        break
                ball_cands = deduped

        seeded_ball = manual_seed_ball_candidate(
            seed_data=manual_seed,
            frame_idx=frame_idx,
            frame_w=frame.shape[1],
            frame_h=frame.shape[0],
        )
        if seeded_ball is not None:
            ball_cands = [seeded_ball] + ball_cands

        tracks = ptracker.update(players, frame_idx)
        ball = btracker.update(ball_cands)

        det_total += len(players)
        for d in players:
            det_team[d.team_label] += 1

        ann = frame.copy()
        shade_crowd(ann, top_y)
        for tr in tracks:
            draw_player(ann, tr)

        if ball is not None and ball.missed <= 6:
            draw_ball(ann, ball)
            ball_visible += 1

        active = Counter(t.team_label for t in tracks)
        txt = f"Frame {frame_idx} | RED {active[TEAM_RED]} LIGHT {active[TEAM_LIGHT]} OTHER {active[TEAM_OTHER]}"
        cv2.putText(ann, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

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
            cv2.imshow("Soccer Classical Tracking", ann)
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
    write_metrics(out / "track_metrics.csv", tracks_all)

    summary = {
        "input_video": str(args.video),
        "manual_seed_file": str(args.seed_file) if args.seed_file is not None else "",
        "frames_processed": processed,
        "player_detections_total": det_total,
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
            TEAM_RED: int(det_team[TEAM_RED]),
            TEAM_LIGHT: int(det_team[TEAM_LIGHT]),
            TEAM_OTHER: int(det_team[TEAM_OTHER]),
        },
        "tracks_total": len(tracks_all),
        "ball_visible_frames": ball_visible,
        "ball_measured_frames": ball_measured,
        "ball_predicted_visible_frames": ball_visible - ball_measured,
        "metadata": run_metadata(args, fps, time.perf_counter() - started),
        "matching": "hungarian_scipy" if SCIPY_AVAILABLE else "greedy_fallback",
        "output_video": str(out / "annotated_tracking.mp4"),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Done")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
