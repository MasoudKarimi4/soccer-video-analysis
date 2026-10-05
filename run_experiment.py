"""Run a checked-in experiment preset without shell-specific quoting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = {"classical": "soccer_opencv_pipeline.py", "hybrid": "soccer_hybrid_pipeline.py", "baseline": "run_ultralytics_baseline.py"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline", choices=SCRIPTS)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--seed-file", type=Path)
    parser.add_argument("--duration-sec", type=float)
    parser.add_argument("--device")
    args, extra = parser.parse_known_args()
    config = args.config or ROOT / "configs" / f"{args.pipeline}.json"
    preset = json.loads(config.read_text(encoding="utf-8"))
    if preset.get("pipeline") != args.pipeline:
        parser.error("Configuration pipeline does not match the selected pipeline.")
    params = dict(preset["parameters"])
    params.update({"video": str(args.video), "output": str(args.output)})
    if args.duration_sec is not None:
        params["duration-sec"] = args.duration_sec
    if args.pipeline != "classical":
        if args.model is None:
            parser.error("--model is required for the hybrid and baseline presets.")
        params["model" if args.pipeline == "baseline" else "detector-model"] = str(args.model)
        if args.device:
            params["device" if args.pipeline == "baseline" else "detector-device"] = args.device
    if args.seed_file:
        if args.pipeline == "baseline":
            parser.error("The detector-only baseline does not use a seed file.")
        params["seed-file"] = str(args.seed_file)
    command = [sys.executable, str(ROOT / SCRIPTS[args.pipeline])]
    for key, value in params.items():
        if value is True:
            command.append(f"--{key}")
        elif value is not False and value is not None:
            command.extend([f"--{key}", str(value)])
    subprocess.run(command + extra, check=True)


if __name__ == "__main__":
    main()
