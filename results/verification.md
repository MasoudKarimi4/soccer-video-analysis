# Verification record

Repository preparation was completed on **October 4, 2026** (America/Toronto). Verification was performed locally on Windows with Python 3.12. The published repository also runs a GitHub Actions matrix for Windows/Linux and Python 3.10/3.12; check the [workflow history](https://github.com/MasoudKarimi4/soccer-video-analysis/actions/workflows/ci.yml) for results on each commit.

## Checks passed

- Fourteen `unittest` tests covering gated association, empty/forbidden assignments, greedy fallback, box intersection, HSV limits, detector frame caching, track retirement/lifetime counts, ball measured/predicted state, seed-coordinate scaling, speed after a measurement gap, preview-exit accounting, generated-video I/O/metric consistency, packaged presets, and installed CLI execution outside the checkout.
- All Markdown file/image links resolve locally.
- Imported and captured PNG hashes match their manifests.
- Checked-in team counts sum to player totals; ball visibility does not exceed processed frames.
- Python sources compile and the pinned verification environment passes `pip check`.
- Full 300-frame seeded classical and hybrid experiments complete on the original footage; all output MP4s decode at 30 FPS and expected frame counts.
- A 60-frame direct YOLO11s/ByteTrack smoke run completes on the original footage.
- Classical stage generation completes at frame 159; hybrid stage generation completes at frame 35 and finds a recovery example at frame 31.
- Fresh contact sheets and the counts chart were visually inspected for alignment, legibility and label overlap.
- The `src` package builds as an editable installation and a distributable wheel. The wheel contains all pipeline modules and three built-in JSON presets; all fourteen tests also pass with that wheel imported from a separate installation directory.
- The installed package processes short reference-video smoke segments through classical (15 frames), hybrid (15 frames), and baseline (6 frames) entry points. A custom preset and duration override work outside the checkout. All five calibration/capture/stage tool help commands load with package imports.

## Interpretation limits

These local checks verify implementation and artifact consistency. They do not measure ground-truth detection/localization accuracy, real-time performance, or cross-match generalization. Remote CI results are recorded separately in the linked workflow history. See [results](../docs/results.md) for the distinction between saved development evidence and verified runs.
