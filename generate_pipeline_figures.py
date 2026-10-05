#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import soccer_hybrid_pipeline as hybrid
import soccer_opencv_pipeline as base


BBox = base.BBox


FIELD_LOWER = base.parse_hsv_triplet("28,25,25")
FIELD_UPPER = base.parse_hsv_triplet("95,255,255")


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
    if mask.ndim == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask
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


def stack_h(images: Sequence[np.ndarray]) -> np.ndarray:
    if not images:
        raise ValueError("No images to stack.")
    imgs = [ensure_bgr(img) for img in images]
    target_h = max(img.shape[0] for img in imgs)
    resized: List[np.ndarray] = []
    for img in imgs:
        h, w = img.shape[:2]
        if h != target_h:
            new_w = int(round(w * (target_h / float(max(h, 1)))))
            img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
        resized.append(img)
    return cv2.hconcat(resized)


def stack_v(images: Sequence[np.ndarray]) -> np.ndarray:
    if not images:
        raise ValueError("No images to stack.")
    imgs = [ensure_bgr(img) for img in images]
    target_w = max(img.shape[1] for img in imgs)
    resized: List[np.ndarray] = []
    for img in imgs:
        h, w = img.shape[:2]
        if w != target_w:
            new_h = int(round(h * (target_w / float(max(w, 1)))))
            img = cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA)
        resized.append(img)
    return cv2.vconcat(resized)


def label_panel(img: np.ndarray, label: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (180, 28), (18, 18, 18), -1)
    cv2.putText(out, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def hsv_panel(frame: np.ndarray, hsv: np.ndarray) -> np.ndarray:
    h, s, v = cv2.split(hsv)
    hue = colorize_gray(h)
    sat = colorize_gray(s)
    val = colorize_gray(v)
    return stack_v(
        [
            stack_h(
                [
                    label_panel(frame, "Input BGR"),
                    label_panel(hue, "Hue"),
                ]
            ),
            stack_h(
                [
                    label_panel(sat, "Saturation"),
                    label_panel(val, "Value"),
                ]
            ),
        ]
    )


def draw_boxes(
    frame: np.ndarray,
    boxes: Sequence[BBox],
    color: Tuple[int, int, int],
    label: str = "",
    thickness: int = 2,
) -> np.ndarray:
    out = frame.copy()
    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        if label:
            cv2.putText(
                out,
                f"{label} {idx}",
                (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                color,
                1,
                cv2.LINE_AA,
            )
    return out


def draw_scored_boxes(
    frame: np.ndarray,
    scored: Sequence[hybrid.ScoredBox],
    color: Tuple[int, int, int],
    prefix: str,
) -> np.ndarray:
    out = frame.copy()
    for idx, item in enumerate(scored, start=1):
        x, y, w, h = item.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            out,
            f"{prefix}{idx} {item.score:.2f}",
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
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


def draw_ball_candidates(
    frame: np.ndarray,
    candidates: Sequence[base.BallCandidate],
    color: Tuple[int, int, int],
    prefix: str,
) -> np.ndarray:
    out = frame.copy()
    for idx, cand in enumerate(candidates, start=1):
        cx, cy = int(round(cand.center[0])), int(round(cand.center[1]))
        radius = max(4, int(round(cand.radius * 1.35)))
        cv2.circle(out, (cx, cy), radius, color, 2)
        cv2.putText(
            out,
            f"{prefix}{idx} {cand.score:.2f}",
            (cx + radius + 4, max(18, cy - radius - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def make_args(video: Path, seed_file: Optional[Path], detector_model: Path, resize_width: int) -> SimpleNamespace:
    return SimpleNamespace(
        video=video,
        output=Path("unused"),
        seed_file=seed_file,
        start_sec=0.0,
        duration_sec=10.0,
        resize_width=resize_width,
        field_hsv_lower=FIELD_LOWER,
        field_hsv_upper=FIELD_UPPER,
        row_green_threshold=0.30,
        row_green_run=8,
        crowd_top_cut=0,
        min_area=80.0,
        max_area=3600.0,
        min_aspect=1.1,
        max_aspect=6.0,
        min_nongreen=0.10,
        max_match_distance=78.0,
        class_mismatch_penalty=55.0,
        max_missed=24,
        trail_length=60,
        ball_min_area=6.0,
        ball_max_area=180.0,
        ball_max_dist=68.0,
        ball_max_missed=18,
        ball_local_window=85,
        team_local_window=58,
        team_recover_max_missed=4,
        enable_global_team_mask_recovery=False,
        calibration_frames=45,
        calibration_min_pixels=260,
        manual_seed_min_pixels=110,
        warmup_frames=30,
        save_masks=False,
        preview=False,
        detector_backend="ultralytics",
        detector_model=detector_model,
        detector_input_size=960,
        detector_score_thresh=0.15,
        detector_nms_thresh=0.45,
        detector_person_class=0,
        detector_ball_class=32,
        detector_ball_score_thresh=0.05,
        detector_ball_input_size=1280,
        detector_device="cpu",
        onnx_has_objectness=False,
        min_playable_overlap=0.25,
        fuse_motion_proposals=False,
        ball_backend="hybrid",
        disable_ball=False,
    )


def seeded_profile_from_file(video: Path, seed_data: Optional[Dict[str, object]], resize_width: int, min_pixels: int) -> Optional[base.TeamColorProfile]:
    if seed_data is None:
        return None
    return base.build_profile_from_manual_seed(
        video_path=video,
        resize_width=resize_width,
        seed_data=seed_data,
        field_lower=FIELD_LOWER,
        field_upper=FIELD_UPPER,
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
        cv2.putText(out, "SEED RED", (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, base.TEAM_COLOR[base.TEAM_RED], 1, cv2.LINE_AA)
    for box in seed_data.get("light_boxes", []):
        normalized = base.seed_box_for_frame(seed_data, box, frame.shape[1], frame.shape[0])
        if normalized is None:
            continue
        x, y, w, h = normalized
        cv2.rectangle(out, (x, y), (x + w, y + h), base.TEAM_COLOR[base.TEAM_LIGHT], 2)
        cv2.putText(out, "SEED LIGHT", (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, base.TEAM_COLOR[base.TEAM_LIGHT], 1, cv2.LINE_AA)
    ball_box = seed_data.get("ball_box")
    if ball_box is not None:
        normalized = base.seed_box_for_frame(seed_data, ball_box, frame.shape[1], frame.shape[0])
        if normalized is not None:
            x, y, w, h = normalized
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(out, "SEED BALL", (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def write_captions(path: Path, target_frame: int, recovery_frame: Optional[int]) -> None:
    text = f"""# Presentation Step Captions

Main example frame: `{target_frame}`

1. `01_input_frame.png`
   Original resized frame used by the hybrid pipeline.
2. `02_hsv_channels.png`
   Preprocessing view showing the hue, saturation, and value channels used for later thresholding.
3. `03_pitch_segmentation.png`
   Green-pitch segmentation after HSV thresholding and cleanup.
4. `04_playable_region_and_crowd.png`
   Final playable-region mask and crowd cutoff prior.
5. `05_motion_mask.png`
   Background-subtraction foreground mask after morphology.
6. `06_raw_yolo_player_detections.png`
   Raw learned player proposals before classical soccer-scene filtering.
7. `07_detector_filtering_comparison.png`
   Left: raw YOLO detections. Right: detections kept after geometric, playable-region, and non-green filtering.
8. `08_team_masks.png`
   Red and light jersey masks created from HSV thresholds and calibration.
9. `09_team_classified_players.png`
   Player detections after team labeling with classical color analysis.
10. `10_tracking_and_recovery_main_frame.png`
    Active tracks and any recovered detections on the main frame.
11. `11_ball_detector_candidates.png`
    Ball proposals from the high-resolution YOLO ball pass.
12. `12_ball_classical_candidates.png`
    Ball proposals from classical visual logic.
13. `13_ball_fusion_final.png`
    Final fused player+ball output for the main frame.
14. `14_manual_seed_calibration.png`
    Manual calibration seed frame used to learn the red/light profile.
15. `15_recovery_example.png`
    Example frame showing local recovery around active tracks."""
    if recovery_frame is not None:
        text += f"\n\nRecovery example frame: `{recovery_frame}`\n"
    path.write_text(text, encoding="utf-8")


def find_recovery_example(
    args: SimpleNamespace,
    detector: object,
    seed_data: Optional[Dict[str, object]],
    color_profile: base.TeamColorProfile,
    target_frame: int,
    search_limit: int,
) -> Optional[Dict[str, object]]:
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
    processed = 0
    recovery_snapshot: Optional[Dict[str, object]] = None
    while processed <= search_limit:
        ok, frame = cap.read()
        if not ok:
            break
        frame = base.resize_by_width(frame, args.resize_width)
        hsv, pitch, playable, top_y = base.segment_field(
            frame, args.field_hsv_lower, args.field_hsv_upper, args.row_green_threshold, args.row_green_run, args.crowd_top_cut
        )
        mot = base.motion_mask(frame, bg, playable)
        if processed < args.warmup_frames:
            players: List[base.Detection] = []
        else:
            players, _ = hybrid.detect_players_with_backend(
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
        red_mask, white_mask = base.build_team_color_masks(
            hsv=hsv,
            playable=playable,
            field_lower=args.field_hsv_lower,
            field_upper=args.field_hsv_upper,
            color_profile=color_profile,
        )
        active_before = list(ptracker.active.values())
        recovered = base.recover_players_from_tracks(
            tracks=active_before,
            detections=players,
            red_mask=red_mask,
            white_mask=white_mask,
            top_y=top_y,
            local_window=args.team_local_window,
            max_recover_missed=args.team_recover_max_missed,
        )
        merged = base.merge_detections(players, recovered)
        tracks = ptracker.update(merged, processed)
        if recovered and processed != target_frame:
            recovery_snapshot = {
                "frame_idx": processed,
                "frame": frame.copy(),
                "tracks": list(tracks),
                "recovered": list(recovered),
            }
            break
        _ = seed_data
        _ = pitch
        processed += 1
    cap.release()
    return recovery_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate presentation-ready pipeline-step images for the hybrid soccer tracker.")
    parser.add_argument("--video", type=Path, default=Path("soccer_footage.mp4"))
    parser.add_argument("--frame", type=int, default=159)
    parser.add_argument("--seed-file", type=Path, default=None)
    parser.add_argument("--detector-model", type=Path, default=Path("yolo11s.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("presentation_pipeline_steps"))
    parser.add_argument("--resize-width", type=int, default=960)
    parser.add_argument("--recovery-search-limit", type=int, default=260)
    args_ns = parser.parse_args()
    cv2.setRNGSeed(0)

    if not args_ns.video.exists():
        raise SystemExit(f"Video not found: {args_ns.video}")
    if not args_ns.detector_model.exists():
        raise SystemExit(f"Detector model not found: {args_ns.detector_model}")

    args = make_args(args_ns.video, args_ns.seed_file, args_ns.detector_model, args_ns.resize_width)
    out_dir = args_ns.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_data = base.load_manual_seed(args.seed_file)
    color_profile = base.TeamColorProfile()
    seeded_profile = seeded_profile_from_file(args.video, seed_data, args.resize_width, args.manual_seed_min_pixels)
    if seeded_profile is not None:
        color_profile = seeded_profile

    detector = hybrid.build_detector(args)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    bg = cv2.createBackgroundSubtractorKNN(history=700, dist2Threshold=600.0, detectShadows=False)
    ptracker = base.PlayerTracker(
        fps=fps,
        max_missed=args.max_missed,
        max_match_distance=args.max_match_distance,
        trail_length=args.trail_length,
        class_mismatch_penalty=args.class_mismatch_penalty,
    )
    btracker = base.BallTracker(args.ball_max_missed, args.ball_max_dist, max(20, args.trail_length))

    target_snapshot: Optional[Dict[str, object]] = None
    processed = 0
    while processed <= args_ns.frame:
        ok, frame = cap.read()
        if not ok:
            break
        frame = base.resize_by_width(frame, args.resize_width)
        hsv, pitch, playable, top_y = base.segment_field(
            frame=frame,
            lower_hsv=args.field_hsv_lower,
            upper_hsv=args.field_hsv_upper,
            row_green_threshold=args.row_green_threshold,
            row_run=args.row_green_run,
            manual_top_cut=args.crowd_top_cut,
        )
        mot = base.motion_mask(frame, bg, playable)

        raw_boxes = detector.detect(frame)
        if processed < args.warmup_frames:
            filtered_players: List[base.Detection] = []
        else:
            filtered_players = hybrid.filter_detector_boxes(
                raw_boxes=raw_boxes,
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

        red_mask, white_mask = base.build_team_color_masks(
            hsv=hsv,
            playable=playable,
            field_lower=args.field_hsv_lower,
            field_upper=args.field_hsv_upper,
            color_profile=color_profile,
        )

        active_before = list(ptracker.active.values())
        if processed >= args.warmup_frames:
            recovered_players = base.recover_players_from_tracks(
                tracks=active_before,
                detections=filtered_players,
                red_mask=red_mask,
                white_mask=white_mask,
                top_y=top_y,
                local_window=args.team_local_window,
                max_recover_missed=args.team_recover_max_missed,
            )
        else:
            recovered_players = []
        merged_players = base.merge_detections(filtered_players, recovered_players)

        detector_ball_candidates: List[base.BallCandidate] = []
        classical_ball_candidates: List[base.BallCandidate] = []
        if processed >= args.warmup_frames:
            player_boxes = [d.bbox for d in merged_players]
            detector_ball_candidates = hybrid.detector_ball_candidates(
                detector=detector,
                frame=frame,
                hsv=hsv,
                playable=playable,
                top_y=top_y,
                field_lower=args.field_hsv_lower,
                field_upper=args.field_hsv_upper,
                color_profile=color_profile,
                player_boxes=player_boxes,
                prior_center=btracker.track.position if btracker.track is not None else None,
            )
            classical_ball_candidates.extend(
                base.detect_ball(
                    hsv=hsv,
                    mot=mot,
                    top_y=top_y,
                    min_area=args.ball_min_area,
                    max_area=args.ball_max_area,
                    color_profile=color_profile,
                    player_boxes=player_boxes,
                )
            )
            if btracker.track is not None:
                classical_ball_candidates.extend(
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
                        player_boxes=player_boxes,
                    )
                )
            classical_ball_candidates.extend(
                base.detect_ball_near_players(
                    hsv=hsv,
                    playable=playable,
                    player_boxes=player_boxes,
                    top_y=top_y,
                    field_lower=args.field_hsv_lower,
                    field_upper=args.field_hsv_upper,
                    min_area=args.ball_min_area,
                    max_area=args.ball_max_area,
                    color_profile=color_profile,
                )
            )

        all_ball_candidates = list(detector_ball_candidates) + list(classical_ball_candidates)
        if all_ball_candidates:
            all_ball_candidates = sorted(all_ball_candidates, key=lambda z: z.score, reverse=True)
            deduped: List[base.BallCandidate] = []
            for cand in all_ball_candidates:
                if all(base.math.dist(cand.center, prev.center) > 4.0 for prev in deduped):
                    deduped.append(cand)
                if len(deduped) >= 12:
                    break
            all_ball_candidates = deduped

        seeded_ball = base.manual_seed_ball_candidate(
            seed_data=seed_data,
            frame_idx=processed,
            frame_w=frame.shape[1],
            frame_h=frame.shape[0],
        )
        if seeded_ball is not None:
            all_ball_candidates = [seeded_ball] + all_ball_candidates

        tracks = ptracker.update(merged_players, processed)
        ball = btracker.update(all_ball_candidates)

        if processed == args_ns.frame:
            target_snapshot = {
                "frame_idx": processed,
                "frame": frame.copy(),
                "hsv": hsv.copy(),
                "pitch": pitch.copy(),
                "playable": playable.copy(),
                "top_y": top_y,
                "motion": mot.copy(),
                "raw_boxes": list(raw_boxes),
                "filtered_players": list(filtered_players),
                "recovered_players": list(recovered_players),
                "merged_players": list(merged_players),
                "tracks": list(tracks),
                "detector_ball_candidates": list(detector_ball_candidates),
                "classical_ball_candidates": list(classical_ball_candidates),
                "ball": ball,
                "red_mask": red_mask.copy(),
                "white_mask": white_mask.copy(),
                "color_profile": color_profile,
            }
            break

        processed += 1

    cap.release()

    if target_snapshot is None:
        raise SystemExit(f"Could not reach frame {args_ns.frame}.")

    frame = target_snapshot["frame"]
    hsv = target_snapshot["hsv"]
    pitch = target_snapshot["pitch"]
    playable = target_snapshot["playable"]
    top_y = int(target_snapshot["top_y"])
    motion = target_snapshot["motion"]
    raw_boxes = target_snapshot["raw_boxes"]
    filtered_players = target_snapshot["filtered_players"]
    recovered_players = target_snapshot["recovered_players"]
    merged_players = target_snapshot["merged_players"]
    tracks = target_snapshot["tracks"]
    detector_ball_candidates = target_snapshot["detector_ball_candidates"]
    classical_ball_candidates = target_snapshot["classical_ball_candidates"]
    ball = target_snapshot["ball"]
    red_mask = target_snapshot["red_mask"]
    white_mask = target_snapshot["white_mask"]
    profile = target_snapshot["color_profile"]

    crowd_overlay = frame.copy()
    base.shade_crowd(crowd_overlay, top_y)

    pitch_overlay = overlay_mask(frame, pitch, (40, 220, 80))
    cv2.putText(pitch_overlay, "Largest connected green region", (14, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)
    playable_overlay = overlay_mask(crowd_overlay, playable, (0, 220, 120))
    cv2.putText(playable_overlay, f"Playable top row: {top_y}", (14, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)

    raw_overlay = draw_scored_boxes(frame, raw_boxes, (0, 255, 255), "YOLO ")
    filtered_overlay = draw_detections(frame, filtered_players, show_team=False)
    compare_filter = stack_h([label_panel(raw_overlay, "Raw YOLO proposals"), label_panel(filtered_overlay, "After CV filtering")])

    team_mask_panel = stack_h(
        [
            label_panel(overlay_mask(frame, red_mask, base.TEAM_COLOR[base.TEAM_RED]), "Red mask"),
            label_panel(overlay_mask(frame, white_mask, (200, 200, 200)), "Light mask"),
        ]
    )
    classified_overlay = draw_detections(frame, merged_players, show_team=True)

    tracking_overlay = draw_tracks(frame, tracks, recovered_players)
    detector_ball_overlay = draw_ball_candidates(frame, detector_ball_candidates, (0, 255, 255), "D")
    classical_ball_overlay = draw_ball_candidates(frame, classical_ball_candidates, (255, 255, 0), "C")
    final_overlay = tracking_overlay.copy()
    if ball is not None and ball.missed <= 6:
        base.draw_ball(final_overlay, ball)

    save_image(out_dir / "01_input_frame.png", frame, "1. Input Frame", f"Frame {args_ns.frame} resized to width {args.resize_width}")
    save_image(out_dir / "02_hsv_channels.png", hsv_panel(frame, hsv), "2. HSV Preprocessing", "Hue, saturation, and value channels for later thresholding")
    save_image(out_dir / "03_pitch_segmentation.png", pitch_overlay, "3. Field Segmentation", "HSV green threshold + morphology + largest connected component")
    save_image(out_dir / "04_playable_region_and_crowd.png", playable_overlay, "4. Playable Region And Crowd Suppression", "Crowd area removed before later reasoning")
    save_image(out_dir / "05_motion_mask.png", colorize_gray(motion), "5. Motion Mask", "Background subtraction + thresholding + morphology")
    save_image(out_dir / "06_raw_yolo_player_detections.png", raw_overlay, "6. Learned Player Detection", "Raw Ultralytics player proposals before soccer-scene filtering")
    save_image(out_dir / "07_detector_filtering_comparison.png", compare_filter, "7. Detector Filtering With Classical CV", "Raw detector boxes versus boxes kept after classical scene rules")
    save_image(out_dir / "08_team_masks.png", team_mask_panel, "8. Team Color Masks", f"Calibrated profile: red_low_max={profile.red_low_max}, white_val_min={profile.white_val_min}")
    save_image(out_dir / "09_team_classified_players.png", classified_overlay, "9. Team Classification", "HSV-based team labels on the kept detections")
    save_image(out_dir / "10_tracking_and_recovery_main_frame.png", tracking_overlay, "10. Player Tracking And Recovery", f"Recovered detections on this frame: {len(recovered_players)}")
    save_image(out_dir / "11_ball_detector_candidates.png", detector_ball_overlay, "11. Ball Detector Candidates", "High-resolution YOLO ball proposals")
    save_image(out_dir / "12_ball_classical_candidates.png", classical_ball_overlay, "12. Classical Ball Candidates", "Motion, whiteness, circularity, and player-foot context")
    save_image(out_dir / "13_ball_fusion_final.png", final_overlay, "13. Final Hybrid Output", "Final fused player tracks and ball result")

    if seed_data is not None:
        seed_overlay = manual_seed_overlay(args.video, seed_data, args.resize_width)
        if seed_overlay is not None:
            save_image(
                out_dir / "14_manual_seed_calibration.png",
                seed_overlay,
                "14. Manual Seed Calibration",
                f"Frame {seed_data['frame_index']} used to learn the red/light profile and seed the ball",
            )

    recovery_example = find_recovery_example(
        args=args,
        detector=detector,
        seed_data=seed_data,
        color_profile=color_profile,
        target_frame=args_ns.frame,
        search_limit=args_ns.recovery_search_limit,
    )
    recovery_frame_idx: Optional[int] = None
    if recovery_example is not None:
        recovery_frame_idx = int(recovery_example["frame_idx"])
        recovery_overlay = draw_tracks(
            recovery_example["frame"],
            recovery_example["tracks"],
            recovery_example["recovered"],
        )
        save_image(
            out_dir / "15_recovery_example.png",
            recovery_overlay,
            "15. Recovery Example",
            f"Frame {recovery_frame_idx} with recovered detections: {len(recovery_example['recovered'])}",
        )

    write_captions(out_dir / "captions.md", args_ns.frame, recovery_frame_idx)

    manifest = {
        "video": str(args.video),
        "main_frame": args_ns.frame,
        "seed_frame": int(seed_data["frame_index"]) if seed_data is not None else None,
        "recovery_example_frame": recovery_frame_idx,
        "output_dir": str(out_dir),
        "files": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
