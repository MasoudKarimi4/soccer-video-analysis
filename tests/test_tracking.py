"""Regression coverage for association, temporal state, clipping and detector caching."""
import argparse
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soccer_opencv_pipeline as base
import soccer_hybrid_pipeline as hybrid


class TrackingTests(unittest.TestCase):
    def test_assignment_gates_before_optimization(self):
        matches, tracks, detections = base.assign(np.array([[1., 4.], [4., 6.]]), 5.)
        self.assertEqual(set(matches), {(0, 1), (1, 0)})
        self.assertEqual((tracks, detections), ([], []))

    def test_assignment_all_forbidden_and_empty(self):
        self.assertEqual(base.assign(np.array([[np.inf, 9.]]), 5.), ([], [0], [0, 1]))
        self.assertEqual(base.assign(np.empty((0, 2)), 5.), ([], [], [0, 1]))

    def test_greedy_fallback_handles_forbidden(self):
        with patch.object(base, 'SCIPY_AVAILABLE', False):
            self.assertEqual(base.assign(np.array([[1., 9.], [9., 2.]]), 5.)[0], [(0, 0), (1, 1)])

    def test_clipping_uses_box_intersection(self):
        self.assertEqual(base.normalize_box((-5, -3, 10, 10), 100, 100), (0, 0, 5, 7))
        self.assertIsNone(base.normalize_box((105, 5, 8, 8), 100, 100))
        self.assertIsNone(base.normalize_box((-20, 5, 10, 8), 100, 100))

    def test_hsv_limits(self):
        self.assertEqual(base.parse_hsv_triplet('179,255,255').tolist(), [179, 255, 255])
        for invalid in ('180,0,0', 'a,0,0', '0,0', '0,-1,0'):
            with self.assertRaises(argparse.ArgumentTypeError):
                base.parse_hsv_triplet(invalid)

    def test_seed_coordinates_scale_with_analysis_resolution(self):
        seed = {'image_size': [960, 540], 'frame_index': 111, 'ball_box': [200, 100, 20, 20]}
        self.assertEqual(base.seed_box_for_frame(seed, [100, 100, 40, 60], 480, 270), (50, 50, 20, 30))
        ball = base.manual_seed_ball_candidate(seed, 111, 480, 270)
        self.assertEqual(ball.center, (105., 55.))
        self.assertIsNone(base.manual_seed_ball_candidate(seed, 112, 480, 270))

    def test_total_points_survives_trail_limit_and_occlusion(self):
        tracker = base.PlayerTracker(30, 2, 78, 3, 55)
        for frame in range(8):
            tracker.update([base.Detection((10, 10, 12, 24), (16., 22.), base.TEAM_RED)], frame)
        tr = tracker.active[1]
        self.assertEqual(tr.confirmed_points, 8)
        self.assertEqual(len(tr.trail), 3)
        tracker.update([], 8)
        self.assertEqual(tr.missed, 1)
        self.assertEqual(tr.confirmed_points, 8)
        tracker.update([], 9)
        tracker.update([], 10)
        self.assertNotIn(1, tracker.active)
        self.assertIn(1, tracker.finished)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'tracks.csv'
            base.write_metrics(dest, tracker.all_tracks())
            with dest.open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row['points'], '8')
            self.assertEqual(row['trail_points'], '3')

    def test_ball_measured_prediction_and_retirement(self):
        tracker = base.BallTracker(2, 68, 10)
        self.assertIsNone(tracker.update([]))
        self.assertEqual(tracker.update([base.BallCandidate((30., 30.), 3., 1.)]).missed, 0)
        self.assertEqual(tracker.update([]).missed, 1)
        self.assertEqual(tracker.update([]).missed, 2)
        self.assertIsNone(tracker.update([]))

    def test_speed_uses_previous_measurement_after_gap(self):
        tracker = base.PlayerTracker(30, 4, 100, 10, 55)
        tracker.update([base.Detection((10, 10, 12, 24), (16., 22.), base.TEAM_RED)], 0)
        tracker.update([base.Detection((12, 10, 12, 24), (18., 22.), base.TEAM_RED)], 1)
        previous = tracker.active[1].last_measured_position
        tracker.update([], 2)
        tracker.update([], 3)
        tracker.update([base.Detection((18, 10, 12, 24), (24., 22.), base.TEAM_RED)], 4)
        current = tracker.active[1]
        expected = base.math.dist(previous, current.last_measured_position) * 30 / 3
        self.assertAlmostEqual(current.speeds_px_per_s[-1], expected)

    def test_detector_cache_holds_actual_frame(self):
        detector = object.__new__(hybrid.UltralyticsPersonDetector)
        detector._people_cache_frame = None
        detector._cached_people = []
        detector.person_class = 0
        detector.input_size = 960
        detector.score_thresh = .15
        calls = []
        detector._run_predict = lambda **kwargs: calls.append(kwargs['frame']) or []
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = a.copy()
        detector.detect(a)
        detector.detect(a)
        detector.detect(b)
        self.assertEqual(len(calls), 2)
        self.assertIs(detector._people_cache_frame, b)


if __name__ == '__main__':
    unittest.main()
