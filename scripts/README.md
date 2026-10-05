# Supporting tools

Install the package first with `python -m pip install -e .`. Run these tools from the repository root. Add `.[visualization]` for contact sheets and charts, or `.[ml,visualization]` for the learned pipeline and all figures.

| Tool | Purpose |
|---|---|
| `manual_seed_tool.py` | Select jersey and ball calibration boxes interactively |
| `extract_debug_frames.py` | Extract video frames for inspection |
| `capture_evidence.py` | Capture aligned input/output PNGs and a provenance manifest |
| `generate_classical_figures.py` | Visualize classical pipeline stages |
| `generate_pipeline_figures.py` | Visualize hybrid pipeline stages |
| `summarize_results.py` | Build a comparison CSV and chart from checked-in summaries |
| `validate_repository.py` | Validate local documentation links, image hashes, and metric totals |

Example: `python scripts/capture_evidence.py --help`. Calibration and figure tools support `--help`; the summary and validation scripts run directly. See [reproduction](../docs/reproduction.md) for complete examples.
