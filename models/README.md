# Detector weights

The recommended preset expects a local `yolo11s.pt` file. Weights are excluded from Git. Obtain them using the official [Ultralytics YOLO11 workflow](https://docs.ultralytics.com/models/yolo11/), for example after installing the full requirements:

```bash
python -c "from ultralytics import YOLO; YOLO('models/yolo11s.pt')"
```

That command may download weights from Ultralytics when the file does not exist. The pipeline itself checks for a local model before inference. You can also supply an existing file with `--model`.

YOLO11s, YOLO11m and YOLO11n files were present in the original workspace. The curated results use the final YOLO11s hybrid plus a supplied YOLO11m baseline/comparison; no soccer-specific trained checkpoint was supplied. The exact verified YOLO11s fingerprint is recorded in [assets.json](../results/verified/assets.json).
