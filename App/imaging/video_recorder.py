"""Encode camera frames directly to video without blocking Tk."""

from datetime import datetime
from pathlib import Path
import queue
import math
import threading
import time
import uuid

import cv2


class VideoRecorder:
    FPS_BY_RESOLUTION = {"HIGH": 10.0, "MED": 30.0, "LOW": 60.0}

    def __init__(self, saved_images):
        self.saved_images = Path(saved_images)
        self._lock = threading.Lock()
        self.recording = False
        self.done = threading.Event()
        self.done.set()
        self.error = None
        self.video_path = None
        self.folder = None

    def start(self, resolution="HIGH", auto_fps=False, camera_fps=None):
        with self._lock:
            if not self.done.is_set():
                raise RuntimeError("The previous video is still being saved.")
            if auto_fps and (camera_fps is None or not math.isfinite(camera_fps) or camera_fps <= 0):
                raise RuntimeError("Wait for the live camera feed before using Auto / max FPS.")
            self.auto_fps = auto_fps
            self.max_fps = camera_fps if auto_fps else self.FPS_BY_RESOLUTION[resolution]
            name = datetime.now().strftime("Video_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
            self.folder = self.saved_images / name
            self.folder.mkdir(parents=True)
            self.video_path = None
            self.error = None
            self._next_frame_time = 0.0
            # Bound memory use; stop explicitly if storage cannot keep up.
            self._frames = queue.Queue(maxsize=8)
            self._stop = threading.Event()
            self.done.clear()
            self.recording = True
            self._thread = threading.Thread(target=self._run, name="video-recorder")
            self._thread.start()

    def submit(self, frame):
        with self._lock:
            if not self.recording:
                return
            timestamp = time.monotonic()
            if not self.auto_fps and timestamp < self._next_frame_time:
                return
            try:
                self._frames.put_nowait((timestamp, frame.copy()))
                self._next_frame_time = timestamp + 1.0 / self.max_fps
            except queue.Full:
                self.error = "Recording stopped because video encoding or disk writing could not keep up."
                self.recording = False
                self._stop.set()

    def stop(self):
        with self._lock:
            if self.recording:
                self.recording = False
                self._stop.set()

    def close(self):
        self.stop()
        self.done.wait()

    def _run(self):
        count = 0
        first_time = None
        size = None
        writer = None
        previous_frame = None
        output_count = 0
        try:
            with (self.folder / "timestamps.csv").open("w", encoding="utf-8") as manifest:
                manifest.write("capture_index,video_frame_index,elapsed_seconds\n")
                while not self._stop.is_set() or not self._frames.empty():
                    try:
                        timestamp, frame = self._frames.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    frame_size = (frame.shape[1], frame.shape[0])
                    if size is not None and frame_size != size:
                        raise RuntimeError("Camera resolution changed during recording.")
                    size = frame_size
                    if writer is None:
                        output = self.folder / "recording.avi"
                        writer = cv2.VideoWriter(
                            str(output), cv2.VideoWriter_fourcc(*"MJPG"),
                            self.max_fps, size,
                        )
                        if not writer.isOpened():
                            raise RuntimeError("OpenCV could not open the MJPEG video encoder.")
                        self.video_path = output
                        first_time = timestamp
                    # VideoWriter uses constant FPS. Repeat the preceding frame
                    # across camera gaps so a slower feed does not play sped up.
                    video_index = max(output_count, round((timestamp - first_time) * self.max_fps))
                    while output_count < video_index:
                        writer.write(previous_frame)
                        output_count += 1
                    writer.write(frame)
                    output_count += 1
                    previous_frame = frame
                    manifest.write(f"{count},{video_index},{timestamp - first_time:.9f}\n")
                    count += 1
            if not count:
                raise RuntimeError("No camera frames were received during recording.")
        except Exception as exc:
            self.error = str(exc)
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception as exc:
                    self.error = f"Could not finalize the video: {exc}"
            with self._lock:
                self.recording = False
                while not self._frames.empty():
                    self._frames.get_nowait()
            self.done.set()
