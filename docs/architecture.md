# Architecture and code map

The repository retains two mature branches sharing the same CV and tracking primitives. The older `soccer_opencv_pipeline_v2.py` prototype is omitted because its functionality is superseded by the current classical implementation. The detector-only baseline remains as a useful comparison.

## Shared stages

| Stage | Function/class | Main signal |
|---|---|---|
| Resize | `resize_by_width` | Preserved aspect ratio; default width 960 |
| Pitch and crowd | `segment_field` | HSV green threshold, 5×5 morphology, largest component, row occupancy |
| Foreground | `motion_mask` | Gaussian smoothing, KNN background subtraction, 3×3 morphology |
| Calibration | `fit_team_color_profile`, `build_profile_from_manual_seed` | Pixel percentiles blended with generic defaults |
| Team labels | `classify_team`, `build_team_color_masks` | Red/white pixel ratios excluding green |
| Local recovery | `recover_players_from_tracks` | Team masks near recent track positions |
| Association | `assign`, `PlayerTracker` | Centroid distance plus team mismatch penalty |
| Ball search | `detect_ball`, `detect_ball_local_near`, `detect_ball_near_players` | Brightness, compactness, isolation and soccer context |
| Ball state | `BallTracker` | Kalman correction, temporary prediction and strong-candidate reacquisition |
| Rendering | `draw_player`, `draw_ball`, `shade_crowd` | Team boxes, trails, ball marker and scene boundary |

These functions are in [soccer_opencv_pipeline.py](../soccer_opencv_pipeline.py). Motion is the primary player proposal source only in the classical branch. Hybrid motion remains an auxiliary ball/recovery signal.

## Hybrid branch

[soccer_hybrid_pipeline.py](../soccer_hybrid_pipeline.py) wraps these stages with `UltralyticsPersonDetector`, `filter_detector_boxes`, `classify_team_from_bbox`, and `detector_ball_candidates`.

Player boxes are clipped to the image and filtered by area, aspect ratio, perspective-dependent constraints, non-green ratio, and playable overlap. Jersey classification emphasizes a central torso crop, with a full-box fallback when that crop is ambiguous. The inherited classifier then further emphasizes its crop's upper portion; this is a heuristic, not anatomical pose estimation.

The high-resolution ball pass is independent of person inference. Each pass has separate cached results for the same frame object. The cache now retains the actual NumPy frame reference: temporary Python object IDs can be reused after objects are released and must not serve as persistent frame identity.

Ball proposals are gated by box size/aspect and playable overlap, then rescored with visual/contextual evidence. Classical global, local and foot-region candidates join the learned candidates. Nearby hypotheses are deduplicated and the highest-ranked candidates reach the ball tracker. Scores are hand-designed ranking values, not calibrated probabilities.

The ball branch starts after the shared warmup; YOLO player proposals can be processed during warmup. Optional manual seed injection occurs on its absolute source frame. Color calibration from the manual seed reads that frame before processing, making this an offline, user-assisted mode.

## Alternative backends

The hybrid script also retains `motion`, `hog` and `onnx-yolo`. HOG is a generic people detector with limited suitability for small broadcast players. ONNX expects raw YOLO-style box/class tensors (with optional objectness); arbitrary end-to-end or custom exports are not automatically supported. Detector-based ball proposals are implemented only for Ultralytics. These alternatives are experimental and are not used for the reported recommended results.

## Outputs

| File | Meaning |
|---|---|
| `annotated_tracking.mp4` | One annotated frame per processed source frame, at source FPS |
| `run_summary.json` | Accumulated counts, final color profile, parameters, versions, elapsed time, seed/model hashes |
| `track_metrics.csv` | One row per created player track, confirmed observation count, retained trail length, image-plane speed summaries |
| `frame_metrics.csv` | Classical/hybrid: source index/time, warmup flag, detections, active/measured player tracks, ball state/coordinates |
| `masks/` | Optional pitch/playable/motion masks from `--save-masks` |

`measured` means a candidate was accepted and used to update the tracker. It does not mean the object was manually verified. Predicted track boxes can remain visible briefly; the active-track status is not the same as the number of current detections.
