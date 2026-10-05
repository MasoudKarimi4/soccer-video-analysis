# Limitations and next experiments

## What the current evidence establishes

The code runs end to end on the original clip and produces interpretable masks, labels, trajectories and ball hypotheses. Fresh regression and video tests verify runtime behavior, output consistency and specific implementation fixes. Saved frame examples make the output inspectable.

There are no supplied ground-truth annotations. More detections do not establish better recall, fewer track IDs do not establish fewer identity switches, and a balanced team split does not establish team accuracy. A manually seeded configuration also uses additional information that an unseeded detector-only baseline does not receive.

## Known failure modes

- **Camera motion and cuts:** KNN foreground modeling and image-coordinate Kalman states are not compensated for pans/zooms. There is no scene-cut reset.
- **Small objects:** Far players and the ball occupy few pixels; analysis resizing can remove detail before the high-resolution ball pass upsamples the frame.
- **White-object ambiguity:** Lines, socks and light jerseys can resemble the ball. Candidate acceptance and prediction can maintain the wrong hypothesis.
- **Team assumptions:** Explicit masks target red/light kits. Other teams, unusual goalkeeper colors and referees require different handling.
- **Crowd/field overlap:** Largest-green-component segmentation and a row cutoff are approximate scene priors and can discard valid objects near boundaries.
- **Association:** Centroid distance/team votes are weak under close crossings. No appearance embeddings or re-identification recover long-lost IDs.
- **Local recovery:** Color searches can attach clutter or a neighboring player. Recovered observations are heuristics, not independent detections.
- **Prediction/rendering:** Active players may be drawn with stale bounding boxes during misses. Ball trails can include predicted points. Visual continuity is not proof of tracking correctness.
- **Calibration leakage:** Manual color calibration reads the seed frame ahead of processing. The method is offline and user-assisted.
- **Motion statistics:** Pixel speeds include camera movement and perspective effects; they are not physical player speeds.
- **Generalization:** Most tuning and evaluation use one broadcast clip. No held-out match or multi-camera benchmark was supplied.
- **Provenance:** Some historical checkpoints lack raw artifacts/configurations, and the original broadcast source URL is unknown.

## Next useful validation

1. Annotate player boxes, team labels, track IDs, and ball centers on sampled frames plus difficult sequences. Declare labeling conventions and split calibration/evaluation segments.
2. Evaluate player precision/recall and IoU thresholds; report ball pixel error and failures/occlusions, not just rendered-state availability.
3. Evaluate identities with IDF1/HOTA or related MOT measures; avoid interpreting track-count reduction as an accuracy improvement.
4. Run controlled ablations with the same detector, footage, seed policy and environment: masks on/off, recovery on/off, 960/1280 ball inference, and classical/detector/fused ball streams.
5. Test unseen matches, jersey colors, lighting and scene cuts. Add cut detection and camera compensation before claiming transferable motion analytics.
6. Add pitch homography for metric coordinates before reporting physical speeds or spatial heatmaps.

These are proposed extensions, not implemented or measured results. Heatmaps and physical movement analysis are not implemented.
