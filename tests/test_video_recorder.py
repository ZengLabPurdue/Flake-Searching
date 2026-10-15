from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "App"))
from imaging.video_recorder import VideoRecorder


class VideoRecorderTests(unittest.TestCase):
    def test_auto_mode_accepts_frames_above_resolution_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = VideoRecorder(directory)
            recorder.start(resolution="HIGH", auto_fps=True, camera_fps=50.0)
            for index, value in enumerate((20, 100, 220)):
                with patch("imaging.video_recorder.time.monotonic", return_value=100.0 + index * 0.02):
                    recorder.submit(np.full((48, 64, 3), value, np.uint8))
            recorder.close()
            self.assertIsNone(recorder.error)
            video = cv2.VideoCapture(str(recorder.video_path))
            try:
                self.assertAlmostEqual(video.get(cv2.CAP_PROP_FPS), 50.0, places=2)
                for value in (20, 100, 220):
                    success, frame = video.read()
                    self.assertTrue(success)
                    self.assertAlmostEqual(float(frame.mean()), value, delta=3)
                self.assertFalse(video.read()[0])
            finally:
                video.release()
            recorder.start(resolution="HIGH")
            self.assertFalse(recorder.auto_fps)
            self.assertEqual(recorder.max_fps, 10.0)
            recorder.close()

    def test_auto_mode_requires_valid_camera_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = VideoRecorder(directory)
            for fps in (None, 0, -1, float("nan"), float("inf")):
                with self.subTest(fps=fps), self.assertRaisesRegex(RuntimeError, "live camera feed"):
                    recorder.start(auto_fps=True, camera_fps=fps)
                self.assertTrue(recorder.done.is_set())
                self.assertFalse(recorder.recording)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_resolution_selects_recording_rate(self):
        for resolution, fps in (("HIGH", 10), ("MED", 30), ("LOW", 60)):
            with self.subTest(resolution=resolution), tempfile.TemporaryDirectory() as directory:
                recorder = VideoRecorder(directory)
                recorder.start(resolution=resolution)
                frame = np.zeros((48, 64, 3), np.uint8)
                with patch("imaging.video_recorder.time.monotonic", return_value=100.0):
                    recorder.submit(frame)
                with patch("imaging.video_recorder.time.monotonic", return_value=100.0 + 0.9 / fps):
                    recorder.submit(None)  # Too early: must skip before copying.
                with patch("imaging.video_recorder.time.monotonic", return_value=100.0 + 1.0 / fps):
                    recorder.submit(frame)
                recorder.stop()
                self.assertTrue(recorder.done.wait(10))
                self.assertIsNone(recorder.error)
                self.assertEqual(len(list(recorder.folder.glob("*.png"))), 0)
                video = cv2.VideoCapture(str(recorder.video_path))
                try:
                    self.assertAlmostEqual(video.get(cv2.CAP_PROP_FPS), fps, places=2)
                finally:
                    video.release()

    def test_direct_video_is_playable_in_order_without_intermediate_images(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = VideoRecorder(directory)
            recorder.start()
            for index, value in enumerate((20, 100, 220)):
                with patch("imaging.video_recorder.time.monotonic", return_value=100.0 + index * 0.101):
                    recorder.submit(np.full((48, 64, 3), value, np.uint8))
            recorder.stop()
            recorder.submit(np.zeros((48, 64, 3), np.uint8))
            self.assertTrue(recorder.done.wait(10))
            self.assertIsNone(recorder.error)
            self.assertEqual(len(list(recorder.folder.glob("*.png"))), 0)
            video = cv2.VideoCapture(str(recorder.video_path))
            try:
                for value in (20, 100, 220):
                    success, frame = video.read()
                    self.assertTrue(success)
                    self.assertAlmostEqual(float(frame.mean()), value, delta=3)
                self.assertFalse(video.read()[0])
            finally:
                video.release()
            first_folder = recorder.folder
            recorder.start()
            recorder.close()
            self.assertNotEqual(first_folder, recorder.folder)
            self.assertIn("No camera frames", recorder.error)
            self.assertTrue((first_folder / "recording.avi").exists())

    def test_write_failure_is_reported_and_recording_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = VideoRecorder(directory)
            with patch("imaging.video_recorder.cv2.VideoWriter") as writer:
                writer.return_value.write.side_effect = OSError("Could not write video")
                recorder.start()
                recorder.submit(np.zeros((48, 64, 3), np.uint8))
                self.assertTrue(recorder.done.wait(5))
            self.assertFalse(recorder.recording)
            self.assertIn("Could not write", recorder.error)
            writer.return_value.release.assert_called_once()

    def test_encoder_failure_is_reported_and_writer_released(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = VideoRecorder(directory)
            with patch("imaging.video_recorder.cv2.VideoWriter") as writer:
                writer.return_value.isOpened.return_value = False
                recorder.start()
                recorder.submit(np.zeros((48, 64, 3), np.uint8))
                recorder.stop()
                self.assertTrue(recorder.done.wait(5))
            self.assertIn("encoder", recorder.error)
            self.assertIsNone(recorder.video_path)
            writer.return_value.release.assert_called_once()

    def test_slow_disk_stops_recording_and_finishes_accepted_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = VideoRecorder(directory)
            writing = threading.Event()
            resume = threading.Event()
            real_writer = cv2.VideoWriter

            class SlowWriter:
                def __init__(self, *args):
                    self.writer = real_writer(*args)

                def isOpened(self):
                    return self.writer.isOpened()

                def write(self, frame):
                    writing.set()
                    resume.wait(5)
                    self.writer.write(frame)

                def release(self):
                    self.writer.release()

            with patch("imaging.video_recorder.cv2.VideoWriter", side_effect=SlowWriter):
                recorder.start()
                frame = np.zeros((48, 64, 3), np.uint8)
                with patch("imaging.video_recorder.time.monotonic", return_value=100.0):
                    recorder.submit(frame)
                self.assertTrue(writing.wait(5))
                try:
                    for index in range(9):
                        with patch(
                            "imaging.video_recorder.time.monotonic",
                            return_value=101.0 + index,
                        ):
                            recorder.submit(frame)
                    self.assertFalse(recorder.recording)
                    self.assertIn("could not keep up", recorder.error)
                finally:
                    resume.set()
                self.assertTrue(recorder.done.wait(10))
            self.assertEqual(len((recorder.folder / "timestamps.csv").read_text().splitlines()), 10)
            self.assertTrue(recorder.video_path.exists())

    def test_fast_feed_is_sampled_before_copying_and_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = VideoRecorder(directory)
            recorder.start()
            frame = np.zeros((48, 64, 3), np.uint8)
            with patch("imaging.video_recorder.time.monotonic", return_value=100.0):
                recorder.submit(frame)
            # A frame inside the 100 ms interval must never even be copied.
            with patch("imaging.video_recorder.time.monotonic", return_value=100.09):
                recorder.submit(None)
            with patch("imaging.video_recorder.time.monotonic", return_value=100.1):
                recorder.submit(frame)
            recorder.stop()
            self.assertTrue(recorder.done.wait(10))
            self.assertIsNone(recorder.error)
            self.assertEqual(len(list(recorder.folder.glob("*.png"))), 0)
            video = cv2.VideoCapture(str(recorder.video_path))
            try:
                self.assertAlmostEqual(video.get(cv2.CAP_PROP_FPS), 10.0, places=2)
            finally:
                video.release()

    def test_camera_gap_preserves_playback_timing_and_close_drains_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = VideoRecorder(directory)
            recorder.start()
            for timestamp, value in ((100.0, 20), (101.0, 220)):
                with patch("imaging.video_recorder.time.monotonic", return_value=timestamp):
                    recorder.submit(np.full((48, 64, 3), value, np.uint8))
            recorder.close()
            self.assertTrue(recorder.done.is_set())
            self.assertIsNone(recorder.error)
            video = cv2.VideoCapture(str(recorder.video_path))
            try:
                self.assertEqual(video.get(cv2.CAP_PROP_FRAME_COUNT), 11)
                for value in [20] * 10 + [220]:
                    success, frame = video.read()
                    self.assertTrue(success)
                    self.assertAlmostEqual(float(frame.mean()), value, delta=3)
            finally:
                video.release()

    def test_frames_are_encoded_before_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = VideoRecorder(directory)
            written = threading.Event()
            with patch("imaging.video_recorder.cv2.VideoWriter") as writer:
                writer.return_value.write.side_effect = lambda frame: written.set()
                recorder.start()
                try:
                    recorder.submit(np.zeros((48, 64, 3), np.uint8))
                    self.assertTrue(written.wait(5))
                    self.assertTrue(recorder.recording)
                    writer.return_value.release.assert_not_called()
                finally:
                    recorder.close()
                writer.return_value.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
