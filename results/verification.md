# Verification record

Repository preparation was completed on **October 4, 2026** (America/Toronto). Verification was performed locally on Windows with Python 3.12; the checked-in GitHub Actions matrix is configured for Windows/Linux and Python 3.10/3.12 but has not run on a remote GitHub repository yet.

## Checks passed

- Twelve `unittest` tests covering gated association, empty/forbidden assignments, greedy fallback, box intersection, HSV limits, detector frame caching, track retirement/lifetime counts, ball measured/predicted state, seed-coordinate scaling, speed after a measurement gap, preview-exit accounting, and generated-video I/O/metric consistency.
- All Markdown file/image links resolve locally.
- Imported and captured PNG hashes match their manifests.
- Checked-in team counts sum to player totals; ball visibility does not exceed processed frames.
- Python sources compile and the pinned verification environment passes `pip check`.
- Full 300-frame seeded classical and hybrid experiments complete on the original footage; all output MP4s decode at 30 FPS and expected frame counts.
- A 60-frame direct YOLO11s/ByteTrack smoke run completes on the original footage.
- Classical stage generation completes at frame 159; hybrid stage generation completes at frame 35 and finds a recovery example at frame 31.
- Fresh contact sheets and the counts chart were visually inspected for alignment, legibility and label overlap.

## Interpretation limits

These checks verify implementation and artifact consistency. They do not measure ground-truth detection/localization accuracy, real-time performance, cross-match generalization, or remote CI success. See [results](../docs/results.md) for the distinction between raw historical evidence, report-only claims and fresh runs.
