# Reproducing the experiments

## Environment

The repository pins core numerical/CV libraries and the main learned-inference dependencies. Python 3.10–3.12 is supported; fresh verification used Python 3.12 on Windows, OpenCV 4.11.0, NumPy 1.26.4, SciPy 1.13.1, Ultralytics 8.3.107, and CPU PyTorch 2.6.0. A full Windows verification environment is recorded in [results/verified](../results/verified/README.md).

Create and activate a virtual environment, then install the package from the checkout. For classical processing, run `python -m pip install -e .`. For CPU-only learned inference, install PyTorch first:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[ml]"
```

For a GPU, install a compatible PyTorch build for your hardware; the checked-in preset defaults to CPU. The full requirements file pins the tested PyTorch version. The base installation includes classical dependencies. Use `python -m pip install -e ".[visualization]"` for frame contact sheets and result charts; combine extras with `python -m pip install -e ".[ml,visualization]"`.

## Data and model placement

Put the original video at `data/soccer_footage.mp4` and YOLO11s weights at `models/yolo11s.pt`, or pass other paths explicitly. Both directories ignore binary assets. The reference clip is broadcast footage. Its original download URL is unavailable; the repository records a local source fingerprint.

The example seed is tied to **source frame 111**, **3.7 seconds**, and a **960×540 analysis image**. Its boxes are `[x, y, width, height]` at that resolution. Seed coordinates are scaled when processing another resolution. Do not apply this seed to a different video.

The first 10 seconds of the original clip contain **300 frames at 30 FPS**. Source indices are zero-based. `--start-sec` seeks to the corresponding source frame and restarts the background model and trackers for that segment. The frame telemetry reports absolute source indices. Manual color calibration can read a future seed frame before segment processing; this is not a strictly causal online evaluation.

## Canonical commands

```bash
soccer-analyze classical --video data/soccer_footage.mp4 --seed-file configs/manual_seed.example.json --output outputs/classical_seeded
soccer-analyze baseline --video data/soccer_footage.mp4 --model models/yolo11s.pt --output outputs/baseline
soccer-analyze hybrid --video data/soccer_footage.mp4 --model models/yolo11s.pt --seed-file configs/manual_seed.example.json --output outputs/hybrid_seeded
```

Use `--duration-sec 30` for a longer run. The recorded 30-second result belongs to the saved development implementation; a fresh 30-second run of this repository was not performed. Other CLI flags can follow the preset command and override parameters, for example `--save-masks`, `--start-sec 10`, or `--warmup-frames 15`.

The direct equivalent of the recommended hybrid preset is:

```bash
soccer-hybrid --video data/soccer_footage.mp4 --duration-sec 10 --resize-width 960 --output outputs/hybrid_direct --seed-file configs/manual_seed.example.json --detector-backend ultralytics --detector-model models/yolo11s.pt --detector-input-size 960 --detector-ball-input-size 1280 --detector-score-thresh 0.15 --detector-ball-score-thresh 0.05 --detector-device cpu --min-playable-overlap 0.25 --ball-backend hybrid --random-seed 0
```

`soccer-hybrid` defaults to `motion` detection; use the hybrid preset or specify the learned backend to obtain the recommended system. `--duration-sec 0` runs to EOF. Scripts overwrite same-named output files, so give each experiment a distinct output directory.

## Calibration for your own video

Start with automatic calibration by omitting `--seed-file`. If the teams are red/light and classification needs improvement, create representative samples:

```bash
python scripts/manual_seed_tool.py --video data/my_match.mp4 --frame-sec 3.7 --resize-width 960 --output configs/my_match_seed.json
```

The GUI supports `r` red jersey boxes, `w` light jersey boxes, `b` a ball box, `u` undo, `c` clear, `s` save, and `q`/Escape exit without saving. Draw boxes with the mouse. Select jerseys in multiple lighting/depth conditions and avoid large grass/line regions. Run the classical/hybrid preset with your saved seed. This implementation's red/light masks must be changed for differently colored teams; a seed cannot turn the current two-color design into a universal kit classifier.

## Capture visual evidence

```bash
python scripts/capture_evidence.py --input-video data/soccer_footage.mp4 --annotated-video outputs/hybrid_seeded/annotated_tracking.mp4 --classical-video outputs/classical_seeded/annotated_tracking.mp4 --frames 60 159 240 --output-dir outputs/evidence
```

This writes unmodified decoded PNGs, a labeled contact sheet and a manifest. For an annotated segment beginning at source frame 300, supply `--annotated-start-frame 300` and absolute source frame indices such as `--frames 360 459 540`.

To explain intermediate stages:

```bash
python scripts/generate_classical_figures.py --video data/soccer_footage.mp4 --seed-file configs/manual_seed.example.json --frame 159 --output-dir outputs/classical_figures
python scripts/generate_pipeline_figures.py --video data/soccer_footage.mp4 --detector-model models/yolo11s.pt --seed-file configs/manual_seed.example.json --frame 159 --output-dir outputs/hybrid_figures
```

Generators replay the video from frame zero to warm up temporal state. Hybrid figure generation also searches for a local recovery example, so it can take longer than extracting a single frame. Stage figures are illustrations; the saved final output frames and CSV telemetry are the canonical evidence of an experiment.

## Inspect metrics and verify consistency

```bash
python -m unittest discover -s tests -v
python scripts/validate_repository.py
python scripts/summarize_results.py
```

The last command builds a comparison CSV and chart from the selected checked-in summaries. New schema-version-2 summaries include all resolved parameters, source FPS, elapsed seconds, versions and seed/model SHA-256 hashes. OpenCV RNG is seeded with `--random-seed 0`; cross-platform codec differences and ML arithmetic can still affect exact outputs. A complete verification environment freeze is evidence of one installation, not a universal cross-platform lock file.

Raw seed hashes in the recorded runs identify the input bytes used at that time. Git normalizes JSON line endings, which can change a raw hash without changing any box values. [Asset metadata](../results/verified/assets.json) also records a canonical JSON hash for comparison independent of formatting; future runs include this canonical hash alongside the raw hash.

## Troubleshooting

| Symptom | Action |
|---|---|
| Video/model not found | Pass the actual asset path; binaries are deliberately external |
| Command not found | Activate the environment, install the package, or use `python -m soccer_video_analysis` |
| No frames processed | Check the start time against the video duration |
| MP4 writer cannot open | Check writable output path and OpenCV/codec installation |
| Hybrid runs motion detection | Use `run_experiment.py hybrid` or specify `--detector-backend ultralytics` |
| Team labels fail on other kits | Adapt the color masks; this implementation assumes red/light jerseys |
| Ball count looks perfect but target is wrong | Inspect frame evidence; state availability is not localization accuracy |
| CPU inference is slow | Try smaller inference sizes for a new experiment, or a compatible GPU; record the configuration change |

To keep Ultralytics settings inside the workspace, set `YOLO_CONFIG_DIR` to a local directory before running. Fresh CPU verification used `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=4`; elapsed values are local observations with concurrent workloads, not a controlled performance benchmark.
