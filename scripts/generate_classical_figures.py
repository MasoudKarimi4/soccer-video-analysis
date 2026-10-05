#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from soccer_video_analysis import classical as base


def add_title(img: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    banner_h = 58 if subtitle else 40
    cv2.rectangle(out, (0, 0), (w - 1, banner_h), (18, 18, 18), -1)
    cv2.putText(out, title, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(out, subtitle, (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 210, 210), 1, cv2.LINE_AA)
    return out


def save_image(path: Path, img: np.ndarray, title: str, subtitle: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), add_title(img, title, subtitle))


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def colorize_gray(mask: np.ndarray, cmap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if mask.ndim == 3 else mask
    return cv2.applyColorMap(gray, cmap)


def overlay_mask(frame: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.48) -> np.ndarray:
    out = frame.copy()
    solid = np.zeros_like(frame)
    solid[:] = color
    active = mask > 0
    if np.any(active):
        blended = cv2.addWeighted(frame, 1.0 - alpha, solid, alpha, 0.0)
        out[active] = blended[active]
    return out


def label_panel(img: np.ndarray, label: str) -> np.ndarray:
    out = ensure_bgr(img).copy()
    cv2.rectangle(out, (0, 0), (230, 28), (18, 18, 18), -1)
    cv2.putText(out, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def stack_h(images: Sequence[np.ndarray]) -> np.ndarray:
    imgs = [ensure_bgr(img) for img in images]
    target_h = max(img.shape[0] for img in imgs)
    resized: List[np.ndarray] = []
    for img in imgs:
        h, w = img.shape[:2]
        if h != target_h:
            img = cv2.resize(img, (int(round(w * (target_h / float(max(h, 1))))), target_h), interpolation=cv2.INTER_AREA)
        resized.append(img)
    return cv2.hconcat(resized)


def stack_v(images: Sequence[np.ndarray]) -> np.ndarray:
    imgs = [ensure_bgr(img) for img in images]
    target_w = max(img.shape[1] for img in imgs)
    resized: List[np.ndarray] = []
    for img in imgs:
        h, w = img.shape[:2]
        if w != target_w:
            img = cv2.resize(img, (target_w, int(round(h * (target_w / float(max(w, 1)))))), interpolation=cv2.INTER_AREA)
        resized.append(img)
    return cv2.vconcat(resized)


def hsv_panel(frame: np.ndarray, hsv: np.ndarray) -> np.ndarray:
    h, s, v = cv2.split(hsv)
    return stack_v(
        [
            stack_h([label_panel(frame, "Input BGR"), label_panel(colorize_gray(h), "Hue")]),
            stack_h([label_panel(colorize_gray(s), "Saturation"), label_panel(colorize_gray(v), "Value")]),
        ]
    )


def draw_contours(frame: np.ndarray, mask: np.ndarray, top_y: int) -> np.ndarray:
    out = frame.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for idx, contour in enumerate(contours, start=1):
        area = float(cv2.contourArea(contour))
        if area < 10.0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        color = (0, 255, 255) if y >= top_y else (90, 90, 90)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 1)
        if y >= top_y:
            cv2.putText(out, f"C{idx}", (x, max(18, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return out


def draw_detections(frame: np.ndarray, detections: Sequence[base.Detection], show_team: bool = True) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        x, y, w, h = det.bbox
        color = base.TEAM_COLOR.get(det.team_label, base.TEAM_COLOR[base.TEAM_OTHER])
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        if show_team:
            label = base.TEAM_NAME.get(det.team_label, "OTHER")
            cv2.putText(out, label, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
    return out


def draw_tracks(frame: np.ndarray, tracks: Sequence[base.Track], recovered: Sequence[base.Detection]) -> np.ndarray:
    out = frame.copy()
    for tr in tracks:
        base.draw_player(out, tr)
    for det in recovered:
        x, y, w, h = det.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(out, "RECOVERED", (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_ball_candidates(frame: np.ndarray, candidates: Sequence[base.BallCandidate], color: Tuple[int, int, int], prefix: str) -> np.ndarray:
    out = frame.copy()
    for idx, cand in enumerate(candidates, start=1):
        cx, cy = int(round(cand.center[0])), int(round(cand.center[1]))
        radius = max(4, int(round(cand.radius * 1.35)))
        cv2.circle(out, (cx, cy), radius, color, 2)
        cv2.putText(out, f"{prefix}{idx} {cand.score:.2f}", (cx + radius + 4, max(18, cy - radius - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return out


def seeded_profile_from_file(video: Path, seed_data: Optional[Dict[str, object]], resize_width: int, min_pixels: int) -> Optional[base.TeamColorProfile]:
    if seed_data is None:
        return None
    return base.build_profile_from_manual_seed(
        video_path=video,
        resize_width=resize_width,
        seed_data=seed_data,
        field_lower=base.parse_hsv_triplet("28,25,25"),
        field_upper=base.parse_hsv_triplet("95,255,255"),
        row_green_threshold=0.30,
        row_run=8,
        manual_top_cut=0,
        min_pixels=min_pixels,
    )


def manual_seed_overlay(video: Path, seed_data: Dict[str, object], resize_width: int) -> Optional[np.ndarray]:
    frame = base.read_video_frame(video, int(seed_data["frame_index"]), resize_width)
    if frame is None:
        return None
    out = frame.copy()
    for box in seed_data.get("red_boxes", []):
        normalized = base.seed_box_for_frame(seed_data, box, frame.shape[1], frame.shape[0])
        if normalized is None:
            continue
        x, y, w, h = normalized
        cv2.rectangle(out, (x, y), (x + w, y + h), base.TEAM_COLOR[base.TEAM_RED], 2)
    for box in seed_data.get("light_boxes", []):
        normalized = base.seed_box_for_frame(seed_data, box, frame.shape[1], frame.shape[0])
        if normalized is None:
            continue
        x, y, w, h = normalized
        cv2.rectangle(out, (x, y), (x + w, y + h), base.TEAM_COLOR[base.TEAM_LIGHT], 2)
    ball_box = seed_data.get("ball_box")
    if ball_box is not None:
        normalized = base.seed_box_for_frame(seed_data, ball_box, frame.shape[1], frame.shape[0])
        if normalized is not None:
            x, y, w, h = normalized
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)
    return out


def dedupe_ball_candidates(candidates: Sequence[base.BallCandidate]) -> List[base.BallCandidate]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
    deduped: List[base.BallCandidate] = []
    for cand in ordered:
        if all(base.math.dist(cand.center, prev.center) > 4.0 for prev in deduped):
            deduped.append(cand)
        if len(deduped) >= 12:
            break
    return deduped


def write_captions(path: Path, target_frame: int, recovery_frame: Optional[int]) -> None:
    text = f"""# Classical V7 Presentation Step Captions

Main example frame: `{target_frame}`

1. `01_input_frame.png`: Original resized frame used by the final classical pipeline.
2. `02_hsv_channels.png`: HSV preprocessing view used for threshold-based segmentation and color reasoning.
3. `03_pitch_segmentation.png`: Pitch segmentation from HSV green thresholding, morphology, and connected components.
4. `04_playable_region_and_crowd.png`: Final playable mask after crowd suppression.
5. `05_motion_mask.png`: Motion foreground from background subtraction and morphology.
6. `06_motion_contours.png`: Raw contour proposals extracted from the motion mask.
7. `07_classical_player_detections.png`: Motion-based player detections after geometric and non-green filtering.
8. `08_team_masks_and_calibration.png`: Red and light team masks using the calibrated/seeded color profile.
9. `09_team_classified_players.png`: Player detections labeled by explicit HSV team reasoning.
10. `10_tracking_and_recovery_main_frame.png`: Track-guided local recovery around active players.
11. `11_classical_tracking.png`: Kalman + Hungarian tracking result for the stronger classical pipeline.
12. `12_ball_candidate_sources.png`: Global, local, and near-player ball candidates from the classical ball subsystem.
13. `13_final_v7_classical_output.png`: Final fully classical v7-style output.
14. `14_manual_seed_calibration.png`: Manual seed frame used to learn the classical red/light profile and seed the ball.
15. `15_recovery_example.png`: Separate frame showing local recovery clearly.
"""
    if recovery_frame is not None:
        text += f"\nRecovery example frame: `{recovery_frame}`\n"
    path.write_text(text, encoding="utf-8")


def find_recovery_example(args: argparse.Namespace, seed_data: Optional[Dict[str, object]], color_profile: base.TeamColorProfile, target_frame: int, search_limit: int) -> Optional[Dict[str, object]]:
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        return None
    bg = cv2.createBackgroundSubtractorKNN(history=700, dist2Threshold=600.0, detectShadows=False)
    ptracker = base.PlayerTracker(
        fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
        max_missed=args.max_missed,
        max_match_distance=args.max_match_distance,
        trail_length=args.trail_length,
        class_mismatch_penalty=args.class_mismatch_penalty,
    )
    btracker = base.BallTracker(args.ball_max_missed, args.ball_max_dist, max(20, args.trail_length))
    calibration_red_samples: List[np.ndarray] = []
    calibration_white_samples: List[np.ndarray] = []
    calibration_complete = seed_data is not None
    processed = 0
    out: Optional[Dict[str, object]] = None
    while processed <= search_limit:
        ok, frame = cap.read()
        if not ok:
            break
        frame = base.resize_by_width(frame, args.resize_width)
        hsv, _, playable, top_y = base.segment_field(frame, args.field_hsv_lower, args.field_hsv_upper, args.row_green_threshold, args.row_green_run, args.crowd_top_cut)
        mot = base.motion_mask(frame, bg, playable)
        if processed < args.warmup_frames:
            players: List[base.Detection] = []
            recovered: List[base.Detection] = []
            ball_cands: List[base.BallCandidate] = []
        else:
            calibration_window_open = processed < (args.warmup_frames + args.calibration_frames)
            red_mask, white_mask = base.build_team_color_masks(hsv, playable, args.field_hsv_lower, args.field_hsv_upper, color_profile)
            players = base.detect_players(hsv, mot, args.field_hsv_lower, args.field_hsv_upper, top_y, args.min_area, args.max_area, args.min_aspect, args.max_aspect, args.min_nongreen, color_profile)
            if calibration_window_open and not calibration_complete:
                sample_red, sample_white = base.collect_calibration_pixels(players, hsv, args.field_hsv_lower, args.field_hsv_upper)
                calibration_red_samples.extend(sample_red)
                calibration_white_samples.extend(sample_white)
                calibrated = base.fit_team_color_profile(calibration_red_samples, calibration_white_samples, args.calibration_min_pixels)
                if calibrated is not None:
                    color_profile = calibrated
                    calibration_complete = True
                    red_mask, white_mask = base.build_team_color_masks(hsv, playable, args.field_hsv_lower, args.field_hsv_upper, color_profile)
                    players = base.detect_players(hsv, mot, args.field_hsv_lower, args.field_hsv_upper, top_y, args.min_area, args.max_area, args.min_aspect, args.max_aspect, args.min_nongreen, color_profile)
            recovered = base.recover_players_from_tracks(list(ptracker.active.values()), players, red_mask, white_mask, top_y, args.team_local_window, args.team_recover_max_missed)
            merged = base.merge_detections(players, recovered)
            keepout = [d.bbox for d in merged]
            ball_cands = list(base.detect_ball(hsv, mot, top_y, args.ball_min_area, args.ball_max_area, color_profile, keepout))
            if btracker.track is not None:
                ball_cands.extend(base.detect_ball_local_near(hsv, playable, btracker.track.position, args.field_hsv_lower, args.field_hsv_upper, args.ball_local_window, args.ball_min_area, args.ball_max_area, color_profile, keepout))
            ball_cands.extend(base.detect_ball_near_players(hsv, playable, keepout, top_y, args.field_hsv_lower, args.field_hsv_upper, args.ball_min_area, args.ball_max_area, color_profile))
            ball_cands = dedupe_ball_candidates(ball_cands)
            seeded_ball = base.manual_seed_ball_candidate(seed_data, processed, frame.shape[1], frame.shape[0])
            if seeded_ball is not None:
                ball_cands = [seeded_ball] + ball_cands
            tracks = ptracker.update(merged, processed)
            _ = btracker.update(ball_cands)
            if recovered and processed != target_frame:
                out = {"frame_idx": processed, "frame": frame.copy(), "tracks": list(tracks), "recovered": list(recovered)}
                break
        processed += 1
    cap.release()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate presentation-ready pipeline-step images for the best classical v7-style soccer pipeline.")
    parser.add_argument("--video", type=Path, default=Path("soccer_footage.mp4"))
    parser.add_argument("--frame", type=int, default=159)
    parser.add_argument("--seed-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("presentation_pipeline_steps_classical_v7_frame159"))
    parser.add_argument("--resize-width", type=int, default=960)
    parser.add_argument("--recovery-search-limit", type=int, default=260)
    cli = parser.parse_args()
    cv2.setRNGSeed(0)

    args = base.build_parser().parse_args([])
    args.video = cli.video
    args.seed_file = cli.seed_file
    args.resize_width = cli.resize_width

    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    out_dir = cli.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_data = base.load_manual_seed(args.seed_file)
    color_profile = base.TeamColorProfile()
    calibration_complete = False
    seeded_profile = seeded_profile_from_file(args.video, seed_data, args.resize_width, args.manual_seed_min_pixels)
    if seeded_profile is not None:
        color_profile = seeded_profile
        calibration_complete = True

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    bg = cv2.createBackgroundSubtractorKNN(history=700, dist2Threshold=600.0, detectShadows=False)
    ptracker = base.PlayerTracker(fps=fps, max_missed=args.max_missed, max_match_distance=args.max_match_distance, trail_length=args.trail_length, class_mismatch_penalty=args.class_mismatch_penalty)
    btracker = base.BallTracker(args.ball_max_missed, args.ball_max_dist, max(20, args.trail_length))
    calibration_red_samples: List[np.ndarray] = []
    calibration_white_samples: List[np.ndarray] = []
    snapshot: Optional[Dict[str, object]] = None

    processed = 0
    while processed <= cli.frame:
        ok, frame = cap.read()
        if not ok:
            break
        frame = base.resize_by_width(frame, args.resize_width)
        hsv, pitch, playable, top_y = base.segment_field(frame, args.field_hsv_lower, args.field_hsv_upper, args.row_green_threshold, args.row_green_run, args.crowd_top_cut)
        mot = base.motion_mask(frame, bg, playable)

        if processed < args.warmup_frames:
            players: List[base.Detection] = []
            recovered: List[base.Detection] = []
            red_mask = np.zeros(playable.shape, dtype=np.uint8)
            white_mask = np.zeros(playable.shape, dtype=np.uint8)
            global_ball: List[base.BallCandidate] = []
            local_ball: List[base.BallCandidate] = []
            near_ball: List[base.BallCandidate] = []
        else:
            calibration_window_open = processed < (args.warmup_frames + args.calibration_frames)
            red_mask, white_mask = base.build_team_color_masks(hsv, playable, args.field_hsv_lower, args.field_hsv_upper, color_profile)
            players = base.detect_players(hsv, mot, args.field_hsv_lower, args.field_hsv_upper, top_y, args.min_area, args.max_area, args.min_aspect, args.max_aspect, args.min_nongreen, color_profile)
            if calibration_window_open and not calibration_complete:
                sample_red, sample_white = base.collect_calibration_pixels(players, hsv, args.field_hsv_lower, args.field_hsv_upper)
                calibration_red_samples.extend(sample_red)
                calibration_white_samples.extend(sample_white)
                calibrated = base.fit_team_color_profile(calibration_red_samples, calibration_white_samples, args.calibration_min_pixels)
                if calibrated is not None:
                    color_profile = calibrated
                    calibration_complete = True
                    red_mask, white_mask = base.build_team_color_masks(hsv, playable, args.field_hsv_lower, args.field_hsv_upper, color_profile)
                    players = base.detect_players(hsv, mot, args.field_hsv_lower, args.field_hsv_upper, top_y, args.min_area, args.max_area, args.min_aspect, args.max_aspect, args.min_nongreen, color_profile)
            recovered = base.recover_players_from_tracks(list(ptracker.active.values()), players, red_mask, white_mask, top_y, args.team_local_window, args.team_recover_max_missed)
            players = base.merge_detections(players, recovered)
            keepout = [d.bbox for d in players]
            global_ball = list(base.detect_ball(hsv, mot, top_y, args.ball_min_area, args.ball_max_area, color_profile, keepout))
            local_ball = list(base.detect_ball_local_near(hsv, playable, btracker.track.position, args.field_hsv_lower, args.field_hsv_upper, args.ball_local_window, args.ball_min_area, args.ball_max_area, color_profile, keepout)) if btracker.track is not None else []
            near_ball = list(base.detect_ball_near_players(hsv, playable, keepout, top_y, args.field_hsv_lower, args.field_hsv_upper, args.ball_min_area, args.ball_max_area, color_profile))

        ball_cands = dedupe_ball_candidates([*global_ball, *local_ball, *near_ball])
        seeded_ball = base.manual_seed_ball_candidate(seed_data, processed, frame.shape[1], frame.shape[0])
        if seeded_ball is not None:
            ball_cands = [seeded_ball] + ball_cands

        tracks = ptracker.update(players, processed)
        ball = btracker.update(ball_cands)

        if processed == cli.frame:
            snapshot = {
                "frame": frame.copy(),
                "hsv": hsv.copy(),
                "pitch": pitch.copy(),
                "playable": playable.copy(),
                "top_y": top_y,
                "motion": mot.copy(),
                "players": list(players),
                "recovered": list(recovered),
                "tracks": list(tracks),
                "global_ball": list(global_ball),
                "local_ball": list(local_ball),
                "near_ball": list(near_ball),
                "ball": ball,
                "red_mask": red_mask.copy(),
                "white_mask": white_mask.copy(),
                "profile": color_profile,
            }
            break
        processed += 1

    cap.release()
    if snapshot is None:
        raise SystemExit(f"Could not reach frame {cli.frame}.")

    frame = snapshot["frame"]
    hsv = snapshot["hsv"]
    pitch = snapshot["pitch"]
    playable = snapshot["playable"]
    top_y = int(snapshot["top_y"])
    motion = snapshot["motion"]
    players = snapshot["players"]
    recovered = snapshot["recovered"]
    tracks = snapshot["tracks"]
    global_ball = snapshot["global_ball"]
    local_ball = snapshot["local_ball"]
    near_ball = snapshot["near_ball"]
    ball = snapshot["ball"]
    red_mask = snapshot["red_mask"]
    white_mask = snapshot["white_mask"]
    profile = snapshot["profile"]

    crowd_overlay = frame.copy()
    base.shade_crowd(crowd_overlay, top_y)
    pitch_overlay = overlay_mask(frame, pitch, (40, 220, 80))
    playable_overlay = overlay_mask(crowd_overlay, playable, (0, 220, 120))
    contour_overlay = draw_contours(frame, motion, top_y)
    team_masks = stack_h([label_panel(overlay_mask(frame, red_mask, base.TEAM_COLOR[base.TEAM_RED]), "Red mask"), label_panel(overlay_mask(frame, white_mask, (200, 200, 200)), "Light mask")])
    tracking_overlay = draw_tracks(frame, tracks, [])
    recovery_overlay = draw_tracks(frame, tracks, recovered)
    ball_panel = stack_v(
        [
            stack_h([label_panel(draw_ball_candidates(frame, global_ball, (0, 255, 255), "G"), "Global ball search"), label_panel(draw_ball_candidates(frame, local_ball, (255, 200, 0), "L"), "Local near-track search")]),
            stack_h([label_panel(draw_ball_candidates(frame, near_ball, (255, 120, 0), "P"), "Near-player search"), label_panel(frame.copy(), "Reference frame")]),
        ]
    )
    final_overlay = tracking_overlay.copy()
    if ball is not None and ball.missed <= 6:
        base.draw_ball(final_overlay, ball)

    save_image(out_dir / "01_input_frame.png", frame, "1. Input Frame", f"Frame {cli.frame} resized to width {args.resize_width}")
    save_image(out_dir / "02_hsv_channels.png", hsv_panel(frame, hsv), "2. HSV Preprocessing", "Hue, saturation, and value channels used by the final classical pipeline")
    save_image(out_dir / "03_pitch_segmentation.png", pitch_overlay, "3. Pitch Segmentation", "HSV green threshold + morphology + largest connected component")
    save_image(out_dir / "04_playable_region_and_crowd.png", playable_overlay, "4. Playable Region And Crowd Suppression", f"Playable top row: {top_y}")
    save_image(out_dir / "05_motion_mask.png", colorize_gray(motion), "5. Motion Mask", "Background subtraction + thresholding + morphology")
    save_image(out_dir / "06_motion_contours.png", contour_overlay, "6. Motion Contours", "Raw contour proposals from the motion mask")
    save_image(out_dir / "07_classical_player_detections.png", draw_detections(frame, players, show_team=False), "7. Classical Player Detection", "Motion, geometry, and non-green filtering")
    save_image(out_dir / "08_team_masks_and_calibration.png", team_masks, "8. Team Masks And Calibration", f"Calibrated profile: red_low_max={profile.red_low_max}, white_val_min={profile.white_val_min}")
    save_image(out_dir / "09_team_classified_players.png", draw_detections(frame, players, show_team=True), "9. Team Classification", "Explicit HSV team labels on classical detections")
    save_image(out_dir / "10_tracking_and_recovery_main_frame.png", recovery_overlay, "10. Local Recovery", f"Recovered detections on this frame: {len(recovered)}")
    save_image(out_dir / "11_classical_tracking.png", tracking_overlay, "11. Classical Tracking", "Kalman + Hungarian player tracking")
    save_image(out_dir / "12_ball_candidate_sources.png", ball_panel, "12. Classical Ball Candidate Sources", "Global, local, and near-player ball proposals")
    save_image(out_dir / "13_final_v7_classical_output.png", final_overlay, "13. Final V7 Classical Output", "Best fully classical pipeline output for this frame")

    if seed_data is not None:
        seed_overlay = manual_seed_overlay(args.video, seed_data, args.resize_width)
        if seed_overlay is not None:
            save_image(out_dir / "14_manual_seed_calibration.png", seed_overlay, "14. Manual Seed Calibration", f"Frame {seed_data['frame_index']} used to learn the red/light profile and seed the ball")

    recovery_example = find_recovery_example(args, seed_data, color_profile, cli.frame, cli.recovery_search_limit)
    recovery_frame_idx: Optional[int] = None
    if recovery_example is not None:
        recovery_frame_idx = int(recovery_example["frame_idx"])
        save_image(out_dir / "15_recovery_example.png", draw_tracks(recovery_example["frame"], recovery_example["tracks"], recovery_example["recovered"]), "15. Recovery Example", f"Frame {recovery_frame_idx} with recovered detections: {len(recovery_example['recovered'])}")

    write_captions(out_dir / "captions.md", cli.frame, recovery_frame_idx)
    manifest = {
        "video": str(args.video),
        "main_frame": cli.frame,
        "seed_frame": int(seed_data["frame_index"]) if seed_data is not None else None,
        "recovery_example_frame": recovery_frame_idx,
        "output_dir": str(out_dir),
        "files": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
