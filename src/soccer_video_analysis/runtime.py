"""Shared validation and reproducibility metadata for command-line runs."""
from __future__ import annotations

import hashlib
import json
import os
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cv2
import numpy as np


def validate_run(args) -> None:
    if args.start_sec < 0 or args.duration_sec < 0:
        raise SystemExit("Start and duration must be nonnegative; duration 0 means to end of video.")
    if args.resize_width < 0:
        raise SystemExit("Resize width must be nonnegative; 0 preserves source size.")
    for name in ("warmup_frames", "max_missed", "ball_max_missed", "calibration_frames"):
        if hasattr(args, name) and getattr(args, name) < 0:
            raise SystemExit(f"{name} must be nonnegative.")
    if hasattr(args, "trail_length") and args.trail_length < 1:
        raise SystemExit("Trail length must be at least 1.")
    if not 0 <= args.random_seed <= 2147483647:
        raise SystemExit("Random seed must be in [0,2147483647].")
    cv2.setRNGSeed(args.random_seed)
    np.random.seed(args.random_seed)


def configure_ultralytics() -> None:
    """Use workspace-local settings unless the caller supplied another location."""
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / "outputs" / ".ultralytics"))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_metadata(args, fps: float, elapsed: float) -> dict:
    parameters = {}
    for name, value in vars(args).items():
        parameters[name] = value.tolist() if isinstance(value, np.ndarray) else str(value) if isinstance(value, Path) else value
    packages = {}
    for package in ("numpy", "opencv-python", "scipy", "ultralytics", "torch", "lap"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            continue
    fingerprints = {}
    for name in ("seed_file", "detector_model", "model"):
        path = getattr(args, name, None)
        if path is not None and path.is_file():
            fingerprints[name] = sha256(path)
            if name == "seed_file":
                content = json.loads(path.read_text(encoding="utf-8"))
                canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
                fingerprints["seed_file_canonical_json"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "schema_version": 2,
        "parameters": parameters,
        "source_fps": fps,
        "warmup_frames": getattr(args, "warmup_frames", 0),
        "elapsed_seconds": round(elapsed, 3),
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "packages": packages},
        "asset_sha256": fingerprints,
    }
