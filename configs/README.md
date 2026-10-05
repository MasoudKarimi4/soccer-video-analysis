# Custom configurations

Built-in classical, hybrid, and baseline presets live in [the package's presets directory](../src/soccer_video_analysis/presets/) and are included in installed wheels. The runner reads them automatically.

To customize a preset, copy one of those JSON files here, keep its `pipeline` field, and edit the `parameters` mapping. Keys use the pipeline's CLI option names without the leading `--`. Explicit runner options and trailing pipeline flags override the preset.

```bash
soccer-analyze hybrid --config configs/my_hybrid.json --video data/my_match.mp4 --model models/yolo11s.pt --output outputs/my_hybrid
```

[manual_seed.example.json](manual_seed.example.json) is a separate calibration asset for the reference clip. Pass it through `--seed-file` only for that clip. Create a new seed with `python scripts/manual_seed_tool.py` for another match.
