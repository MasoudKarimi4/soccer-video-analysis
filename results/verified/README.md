# Fresh verification of the enhanced repository

These runs execute this repository's retained/enhanced code against the original local footage. They are separate from the saved development experiments in `results/historical/`. Source videos remain in the local ignored `outputs/` directory; decoded evidence is checked in under [docs/images/verified](../../docs/images/verified/capture_manifest.json).

| Run | Duration / frames | Player/person detections | Team red / light / other | Created player/person IDs | Ball evidence |
|---|---|---:|---|---:|---|
| [Classical, seeded](classical_10sec/run_summary.json) | 10 s / 300 | 2,249 | 1,108 / 1,116 / 25 | 45 | 270 measured states; 0 predicted visible states |
| [Hybrid YOLO11s, seeded](hybrid_10sec/run_summary.json) | 10 s / 300 | 4,590 | 2,334 / 2,229 / 27 | 32 | 221 filtered detector candidates; 269 measured states; 1 predicted visible state |
| [Direct YOLO11s smoke run](baseline_2sec/run_summary.json) | 2 s / 60 | 1,114 | Not classified | 21 | 2 sports-ball detections and 2 ball IDs |

The shorter direct baseline run verifies the command line and learned tracker on real footage; it is not a 10-second comparison and should not be extrapolated. No fresh 30-second run or labeled accuracy benchmark was performed.

Classical/hybrid folders include `frame_metrics.csv` and `track_metrics.csv`. Their summary metadata contains resolved parameters, runtime versions, asset fingerprints and local elapsed seconds. Thread settings during learned inference were `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=4`. Other jobs overlapped, so elapsed times are not controlled runtime benchmarks.

The hybrid reproduces the supplied final player/team/candidate totals while creating 32 rather than 35 IDs after association/clipping/reproducibility fixes. This difference alone does not establish better identity accuracy. The classical fresh run is balanced and resembles the reported restored classical checkpoint, but is a new experiment: the missing original checkpoint is not being retroactively reconstructed as a saved artifact.

The ball's `measured` state means the tracker accepted a candidate. Correct ball localization still requires a labeled reference. [Assets](assets.json) identify the supplied video/weights/seed; the [environment freeze](environment-windows-py312.txt) records the tested Windows installation and is not a cross-platform lock.
