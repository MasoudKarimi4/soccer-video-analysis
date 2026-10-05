# Contributing

Create a virtual environment and install the package with `python -m pip install -e .`. Run `python -m unittest discover -s tests -v`. Check local links, image hashes and saved metric totals with `python scripts/validate_repository.py`.

Keep reusable implementation in `src/soccer_video_analysis/` and checkout-specific tools in `scripts/`. Update imports, installed CLI entry points and examples together when changing modules. Built-in presets are package data; tests run the installed CLI from a temporary directory to verify that they remain available outside the checkout. CI installs a wheel rather than importing uninstalled source files.

For detection/tracking changes, record the configuration, runtime versions and seed/model fingerprints. Save new results separately from historical runs. Include examples of a difficult case and any regressions, and describe output counts as counts unless labeled ground truth supports an accuracy claim.

Add regression tests for state, assignment or I/O bugs. The test suite should remain runnable without the original footage, internet access or model weights. Use generated footage for basic end-to-end tests and local private assets for optional learned-inference validation.

Keep large binaries, full match footage, weights, virtual environments and generated videos out of commits. Visual evidence should identify its source run, frame index and any transformations. Do not fabricate detections or edit screenshots to hide failures.
