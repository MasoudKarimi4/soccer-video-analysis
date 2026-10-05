# References and attribution

## Project sources

- **Masoud Karimi and Zachary Sikka, Group 5.** *Soccer Player Detection and Multi-Object Tracking*, CSI4133 Picture Processing, April 2026. The latest supplied Markdown paper is adapted in [analysis-paper.md](paper/analysis-paper.md). It supplies the project framing and historical experimental claims.
- The supplied development context, architecture comparison and progress report describe earlier iterations. Their overlapping claims are reconciled in [Results](results.md), rather than copied as unqualified current results.
- Selected saved `run_summary.json` and `track_metrics.csv` files are under [historical results](../results/historical/provenance.json). Numerical values are retained; path strings are normalized to basenames.

The source bundle and original drafts remain outside this curated repository. Duplicate reports, document build archives, presentation notes, old prototypes, model weights and full videos are excluded.

## Implementation references

- [OpenCV color conversions](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html)
- [OpenCV HSV thresholding with inRange](https://docs.opencv.org/4.x/da/d97/tutorial_threshold_inRange.html)
- [OpenCV morphological operations](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)
- [OpenCV contour processing](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)
- [OpenCV background subtraction](https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html)
- [OpenCV KalmanFilter](https://docs.opencv.org/4.x/dd/d6a/classcv_1_1KalmanFilter.html)
- [SciPy linear_sum_assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [Ultralytics tracking](https://docs.ultralytics.com/modes/track/)

The morphology, assignment, YOLO11 and SORT references were consulted during repository preparation. Library documentation can change independently of the pinned runtime versions.

## Research context

- R. E. Kalman, *A New Approach to Linear Filtering and Prediction Problems*, 1960. [DOI](https://doi.org/10.1115/1.3662552).
- H. W. Kuhn, *The Hungarian Method for the Assignment Problem*, 1955. [DOI](https://doi.org/10.1002/nav.3800020109).
- A. Bewley et al., *Simple Online and Realtime Tracking*, 2016. [Paper](https://arxiv.org/abs/1602.00763).
- J. Redmon et al., *You Only Look Once: Unified, Real-Time Object Detection*, 2016. [Paper](https://arxiv.org/abs/1506.02640).
- Z. Zivkovic and F. van der Heijden, *Efficient Adaptive Density Estimation per Image Pixel for the Task of Background Subtraction*, 2006. [DOI](https://doi.org/10.1016/j.patrec.2005.11.005).

The tracker is inspired by classical tracking-by-detection, not a faithful implementation of SORT. The original YOLO paper is background for one-stage detection, not an exact architectural specification for YOLO11. The report's MathWorks references are comparative background; no MATLAB implementation or MATLAB dependency is included.
