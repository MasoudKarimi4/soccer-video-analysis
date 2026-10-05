# Repository preparation and publishing

## Curated contents

The root workspace and laptop bundle contained identical core scripts. The extra laptop work supplied classical experimental videos/metrics, intermediate-stage images, comparison notes and multiple versions of the report. The full [source inventory](source-inventory.csv) records the reviewed root/bundle files and curation decisions.

Retained:

- Current classical/hybrid pipelines, direct learned baseline, interactive calibration, frame extraction and the two mature stage generators.
- The final recommended YOLO11s configuration plus classical/baseline presets and the clip-specific seed example.
- Seven representative historical summaries, their available track CSVs, original-summary hashes and fresh verification data.
- Selected stage illustrations, aligned original/output screenshots, a comparison chart, theory/reproduction/results documentation, and the adapted analysis paper.
- Meaningful regression tests, a generated-video smoke test and GitHub CI.

Excluded:

- Copied virtual environments and bytecode caches, binary weights, full source/annotated videos, download ZIPs and document-build archives.
- Superseded `soccer_opencv_pipeline_v2.py`, one-frame debugging utilities, the older v2-only figure generator and duplicate high-resolution results.
- Duplicate PDF/DOCX drafts, repeated presentation notes and long conversational development context. Their relevant findings are distilled into the documentation.

The original source directories are preserved. A malformed two-column report figure layout was observed during PDF review; the GitHub paper adaptation uses Markdown and repaired image links to remain readable, without replacing the user's original report.

## Implementation enhancements

| Change | Reason | Verification |
|---|---|---|
| Cache actual frame references | Python can reuse released object IDs and return stale detector results | Detector-cache regression; fresh full hybrid run |
| Gate invalid assignment edges before optimization | Invalid edges can steal feasible pairings | Crafted assignment counterexample and forbidden/empty cases |
| Clip boxes by actual intersection | Clamping an entirely external box can fabricate edge pixels | Partial/outside-box regression |
| Store total confirmed points separately | Deque length caps lifetime counts | Trail-limit and missed-frame regression |
| Retain last measured position for speed | Predictions should not replace the prior measurement for interval displacement | Gap-speed regression |
| Scale seed boxes using annotation image size | Jersey and ball calibration should follow the chosen analysis resolution | Seed-coordinate scaling regression |
| Count written frames before preview exit | A user quitting preview should not lose the final frame from summary counts | Preview-exit regression with mocked GUI |
| Separate measured/predicted ball states | Availability is not detector success or accuracy | State-transition tests and frame-summary consistency |
| Save resolved parameters/environment/hashes | Historic summaries lack reproducibility detail | Synthetic and real-run metadata checks |
| Seed OpenCV/NumPy RNG and isolate YOLO settings | Support repeatable runs and portable workspace configuration | Pinned-environment real runs |
| Check failed writers, empty segments and missing extracted frames | Prevent apparently successful unusable output | Synthetic video/CLI tests and real output decode checks |
| Repair generator defaults/FPS and calibration output paths | Remove implicit clip dependency and hardcoded tracker FPS | Both retained figure generators executed |

These changes preserve the project design while improving reproducibility and inspectability. Fresh results remain separate from original summaries. Main dependency versions are pinned; the exact local installation freeze is recorded with verification artifacts.

## Local Git repository

The prepared folder is a standalone Git repository on branch `main`, with a clean initial commit. It is ready to copy, inspect or publish. Raw assets and generated runtime outputs are ignored. The delivery archive is made from tracked files, excluding `.git`, local videos/weights and build environments.

The repository has not been uploaded to GitHub: no authenticated GitHub creation tool/CLI was available in the preparation session, and no remote destination or visibility was specified. No repository URL is invented.

## Publish when a destination is available

Create an empty GitHub repository named `soccer-video-analysis`, choosing the desired visibility. Do not initialize an additional README, since this folder already has a commit. From this repository, replacing `YOUR_ACCOUNT` with the destination account:

```bash
git remote add origin https://github.com/YOUR_ACCOUNT/soccer-video-analysis.git
git push -u origin main
```

Alternatively, an authenticated GitHub CLI can create and push the repository; choose visibility explicitly. Before public publication, the contributors should select their intended code license and resolve the broadcast-frame asset policy. Existing dependency/model licenses remain applicable. These choices were not inferred during preparation.

Suggested description: **Hybrid soccer video analysis with OpenCV, YOLO11, HSV team classification, Kalman tracking and ball fusion.** Suggested topics: `computer-vision`, `opencv`, `soccer`, `object-tracking`, `yolo`, `image-processing`.
