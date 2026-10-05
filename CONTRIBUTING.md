# Contributing

Start with the core dependencies and run `python -m unittest discover -s tests -v`. Check local links, image hashes and saved metric totals with `python validate_repository.py`.

For detection/tracking changes, record the configuration, runtime versions and seed/model fingerprints. Save new results separately from historical runs. Include examples of a difficult case and any regressions, and describe output counts as counts unless labeled ground truth supports an accuracy claim.

Add regression tests for state, assignment or I/O bugs. The test suite should remain runnable without the original footage, internet access or model weights. Use generated footage for basic end-to-end tests and local private assets for optional learned-inference validation.

Keep large binaries, full match footage, weights, virtual environments and generated videos out of commits. Visual evidence should identify its source run, frame index and any transformations. Do not fabricate detections or edit screenshots to hide failures.
