"""End-to-end classical smoke test using generated footage (no external data)."""
import csv
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import soccer_opencv_pipeline as base

ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def test_preview_quit_counts_the_written_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            video = tmp/'input.mp4'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 30., (320, 180))
            self.assertTrue(writer.isOpened())
            writer.write(np.full((180, 320, 3), (35, 130, 35), np.uint8))
            writer.release()
            out = tmp/'out'
            argv = ['soccer_opencv_pipeline.py', '--video', str(video), '--output', str(out), '--preview']
            with patch.object(sys, 'argv', argv), patch.object(cv2, 'imshow'), patch.object(cv2, 'waitKey', return_value=ord('q')), patch.object(cv2, 'destroyAllWindows'), redirect_stdout(io.StringIO()):
                base.main()
            summary = json.loads((out/'run_summary.json').read_text())
            self.assertEqual(summary['frames_processed'], 1)
            with (out/'frame_metrics.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)

    def test_synthetic_video_produces_consistent_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            video = tmp / 'synthetic.mp4'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 30., (320, 180))
            self.assertTrue(writer.isOpened(), 'MP4 codec must be available')
            for i in range(45):
                frame = np.full((180, 320, 3), (35, 130, 35), np.uint8)
                frame[:35] = (40, 40, 40)
                cv2.rectangle(frame, (30+i, 90), (44+i, 128), (30, 30, 220), -1)
                cv2.rectangle(frame, (180-i, 80), (192-i, 116), (235, 235, 235), -1)
                cv2.circle(frame, (90+i, 140), 3, (255, 255, 255), -1)
                writer.write(frame)
            writer.release()
            out = tmp / 'output'
            subprocess.run([sys.executable, str(ROOT/'run_experiment.py'), 'classical', '--video', str(video), '--output', str(out), '--duration-sec', '1.5', '--resize-width', '320', '--warmup-frames', '5'], check=True, capture_output=True, text=True)
            summary = json.loads((out/'run_summary.json').read_text())
            self.assertEqual(summary['frames_processed'], 45)
            self.assertEqual(summary['player_detections_total'], sum(summary['player_detections_by_team'].values()))
            self.assertEqual(summary['ball_visible_frames'], summary['ball_measured_frames'] + summary['ball_predicted_visible_frames'])
            self.assertEqual(summary['metadata']['schema_version'], 2)
            with (out/'frame_metrics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 45)
            self.assertEqual(sum(int(row['player_detections']) for row in rows), summary['player_detections_total'])
            self.assertEqual(sum(row['ball_state'] == 'measured' for row in rows), summary['ball_measured_frames'])
            cap = cv2.VideoCapture(str(out/'annotated_tracking.mp4'))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 45)
            self.assertEqual(cap.get(cv2.CAP_PROP_FPS), 30.)
            cap.release()


if __name__ == '__main__':
    unittest.main()
