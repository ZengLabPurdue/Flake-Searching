import threading
import time

import cv2
import numpy as np


class FocusCancelled(RuntimeError):
    pass


class FocusController:
    """Magnification-aware autofocus using one live sweep and settled refinement."""

    # focus_range is a radius: the live sweep covers current Z +/- this value.
    # The coarse spacing limits Z travel between camera frames.  The settled
    # search removes camera/exposure latency from the final focus position.
    FOCUS_PROFILES = {
        "2X": {
            "initial_range": 800.0,
            "initial_velocity": 500,
            "initial_acceleration": 10000,
            "maximum_live_spacing": 30.0,
            "refinement_radius": 120.0,
            "refinement_velocity": 500,
            "target_precision": 5.0,
            "refinement_frame_count": 1,
            "final_frame_count": 1,
            "metric_max_dimension": 640,
            "metric_roi_fraction": 0.70,
            "shape_metric": False,
            "minimum_prominence": 0.01,
        },
        "10X": {
            "initial_range": 300.0,
            "initial_velocity": 250,
            "initial_acceleration": 10000,
            "maximum_live_spacing": 6.0,
            "refinement_radius": 35.0,
            "refinement_velocity": 100,
            "target_precision": 1.0,
            "refinement_frame_count": 1,
            "final_frame_count": 2,
            "metric_max_dimension": 720,
            "metric_roi_fraction": 0.70,
            "shape_metric": False,
            "minimum_prominence": 0.01,
        },
        "20X": {
            "initial_range": 200.0,
            "initial_velocity": 150,
            "initial_acceleration": 8000,
            "maximum_live_spacing": 3.0,
            "refinement_radius": 15.0,
            "refinement_velocity": 50,
            "target_precision": 0.5,
            "refinement_frame_count": 2,
            "final_frame_count": 2,
            "metric_max_dimension": 800,
            "metric_roi_fraction": 0.80,
            "shape_metric": False,
            "minimum_prominence": 0.01,
        },
        "100X": {
            "initial_range": 60.0,
            "initial_velocity": 50,
            "initial_acceleration": 6000,
            "maximum_live_spacing": 0.75,
            "refinement_radius": 3.0,
            "refinement_velocity": 10,
            "target_precision": 0.1,
            "refinement_frame_count": 2,
            "final_frame_count": 3,
            "metric_max_dimension": 896,
            "metric_roi_fraction": 0.70,
            "shape_metric": True,
            "minimum_prominence": 0.015,
        },
    }
    DEFAULT_FOCUS_PROFILE = dict(FOCUS_PROFILES["10X"])

    FOCUS_FRAME_TIMEOUT_SECONDS = 5.0
    AUTO_EXPOSURE_SETTLE_SECONDS = 0.75
    MINIMUM_LIVE_SAMPLES = 5
    FALLBACK_SAMPLE_COUNT = 7
    REFINEMENT_POINT_COUNT = 5
    MAX_REFINEMENT_ROUNDS = 8
    MAX_REFINEMENT_EXPANSIONS = 6
    STAGE_Z_INCREMENT = 0.1

    def __init__(
        self,
        app,
        stage,
        frame_processor,
        sharpness_callback=None,
    ):
        self.app = app
        self.stage = stage
        self.frame_processor = frame_processor
        self.sharpness_callback = sharpness_callback or (lambda sharpness: None)

        self.focus_thread = None
        self.stop_focus_event = threading.Event()
        self.focus_running = False
        self._operation_active = threading.Event()
        self._operation_lock = threading.Lock()
        self._focus_thread_lock = threading.Lock()
        self._last_sharpness_publish_time = 0.0

    def _run_on_ui_thread(self, callback, *args):
        dispatcher = getattr(self.app, "call_on_ui_thread", None)
        if dispatcher is not None:
            dispatcher(callback, *args)
            return
        if threading.current_thread() is threading.main_thread():
            callback(*args)
        else:
            self.app.root.after(0, callback, *args)

    def _wait_for_focus_thread(self, focus_thread):
        if threading.current_thread() is threading.main_thread():
            while focus_thread.is_alive():
                self.app.root.update()
                focus_thread.join(timeout=0.05)
            self.app.root.update()
        else:
            focus_thread.join()

    def is_focus_active(self):
        return self._operation_active.is_set()

    def autofocus_defaults(self, magnification=None):
        magnification = str(
            magnification or self.app.get_magnification()
        ).upper()
        profile = self.FOCUS_PROFILES.get(
            magnification,
            self.DEFAULT_FOCUS_PROFILE,
        )
        return {
            "focus_range": profile["initial_range"],
            "z_velo": profile["initial_velocity"],
            "z_accel": profile["initial_acceleration"],
        }

    def _validated_autofocus_settings(
        self,
        focus_range=None,
        z_velo=None,
        z_accel=None,
        magnification=None,
    ):
        magnification = str(
            magnification or self.app.get_magnification()
        ).upper()
        base_profile = self.FOCUS_PROFILES.get(
            magnification,
            self.DEFAULT_FOCUS_PROFILE,
        )
        focus_range = (
            base_profile["initial_range"]
            if focus_range is None
            else focus_range
        )
        z_velo = (
            base_profile["initial_velocity"]
            if z_velo is None
            else z_velo
        )
        z_accel = (
            base_profile["initial_acceleration"]
            if z_accel is None
            else z_accel
        )
        settings = {
            "Autofocus range": (
                self._positive_finite("Autofocus range", focus_range),
                float(base_profile["initial_range"]),
            ),
            "Autofocus velocity": (
                self._positive_finite("Autofocus velocity", z_velo),
                float(base_profile["initial_velocity"]),
            ),
            "Autofocus acceleration": (
                self._positive_finite(
                    "Autofocus acceleration",
                    z_accel,
                ),
                float(base_profile["initial_acceleration"]),
            ),
        }
        for name, (value, maximum) in settings.items():
            if value > maximum:
                raise ValueError(
                    f"{name} must not exceed {maximum:g} for "
                    f"{magnification}."
                )
        return (
            magnification,
            settings["Autofocus range"][0],
            settings["Autofocus velocity"][0],
            settings["Autofocus acceleration"][0],
        )

    def start_auto_focus_thread(
        self,
        focus_range=None,
        z_velo=None,
        z_accel=None,
        wait=True,
    ):
        with self._focus_thread_lock:
            focus_thread = self.focus_thread
            if focus_thread is not None and focus_thread.is_alive():
                already_running = True
            else:
                already_running = False
                (
                    _magnification,
                    focus_range,
                    z_velo,
                    z_accel,
                ) = self._validated_autofocus_settings(
                    focus_range=focus_range,
                    z_velo=z_velo,
                    z_accel=z_accel,
                )

                self.stop_focus_event.clear()
                self.focus_running = True
                self._run_on_ui_thread(self.app.disable_buttons)

                focus_thread = threading.Thread(
                    target=self._auto_focus_worker,
                    args=(focus_range, z_velo, z_accel),
                    daemon=True,
                )
                focus_thread._autofocus_error = None
                self.focus_thread = focus_thread
                focus_thread.start()

        if already_running:
            print("Autofocus already running")
        if wait:
            self._wait_for_focus_thread(focus_thread)
            focus_error = getattr(
                focus_thread,
                "_autofocus_error",
                None,
            )
            if focus_error is not None:
                raise focus_error
        return focus_thread

    def _auto_focus_worker(self, focus_range, z_velo, z_accel):
        focus_thread = threading.current_thread()
        try:
            self.auto_focus(
                focus_range=focus_range,
                z_velo=z_velo,
                z_accel=z_accel,
            )
        except Exception as exc:
            import traceback

            focus_thread._autofocus_error = exc
            print("Autofocus error:", exc)
            traceback.print_exc()
        finally:
            with self._focus_thread_lock:
                if threading.current_thread() is self.focus_thread:
                    self.focus_running = False
            self._run_on_ui_thread(self.app.enable_buttons)

    def stop_auto_focus(self):
        self.stop_focus_event.set()

    def _check_focus_cancelled(self):
        if self.stop_focus_event.is_set():
            raise FocusCancelled("Autofocus was stopped.")

    def get_latest_frame(self):
        data = self._latest_frame_data()
        if data is None:
            return None, None, None
        return data["frame"], data["frame_id"], data["z"]

    def _latest_frame_data(self):
        condition = getattr(self.frame_processor, "_frame_condition", None)
        if condition is None:
            if len(self.frame_processor.frame_buffer) == 0:
                return None
            return self.frame_processor.frame_buffer[-1]
        with condition:
            if len(self.frame_processor.frame_buffer) == 0:
                return None
            return self.frame_processor.frame_buffer[-1]

    def _wait_for_new_frame_data(self, after_frame_id, timeout):
        deadline = time.monotonic() + max(0.0, timeout)
        condition = getattr(self.frame_processor, "_frame_condition", None)
        while True:
            self._check_focus_cancelled()
            data = self._latest_frame_data()
            if data is not None and data["frame_id"] > after_frame_id:
                return data

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            if condition is None:
                time.sleep(min(0.005, remaining))
                continue
            with condition:
                condition.wait(timeout=min(0.05, remaining))

    @staticmethod
    def _gradient_energy(image):
        gradient_x = cv2.Scharr(image, cv2.CV_32F, 1, 0)
        gradient_y = cv2.Scharr(image, cv2.CV_32F, 0, 1)
        cv2.multiply(gradient_x, gradient_x, gradient_x)
        cv2.multiply(gradient_y, gradient_y, gradient_y)
        return cv2.add(gradient_x, gradient_y)

    @staticmethod
    def _tile_slices(height, width, grid_size=3):
        for row in range(grid_size):
            y_start = row * height // grid_size
            y_end = (row + 1) * height // grid_size
            for column in range(grid_size):
                x_start = column * width // grid_size
                x_end = (column + 1) * width // grid_size
                yield np.s_[y_start:y_end, x_start:x_end]

    @classmethod
    def _tiled_tenengrad_score(
        cls,
        energy,
        grid_size=3,
        strongest_tile_count=3,
    ):
        tile_scores = [
            float(cv2.mean(energy[tile_slice])[0])
            for tile_slice in cls._tile_slices(
                *energy.shape[:2],
                grid_size=grid_size,
            )
        ]
        strongest = sorted(tile_scores)[
            -max(1, int(strongest_tile_count)):
        ]
        return float(np.mean(strongest))

    @classmethod
    def _color_shape_score(cls, image, gray):
        """Preserve chromatic flake boundaries and emphasize five shape tiles."""
        if image.ndim == 2:
            color_planes = (image,)
        else:
            color_planes = cv2.split(image[:, :, :3])
        energy = np.zeros(gray.shape, dtype=np.float32)
        for color_plane in color_planes:
            filtered = cv2.GaussianBlur(
                color_plane,
                (0, 0),
                sigmaX=1.0,
            )
            cv2.add(
                energy,
                cls._gradient_energy(filtered),
                energy,
            )
        strongest_tiles = cls._tiled_tenengrad_score(
            energy,
            grid_size=5,
            strongest_tile_count=5,
        )
        brightness_scale = float(cv2.mean(gray)[0]) ** 2
        return strongest_tiles / max(1.0, brightness_scale)

    def find_sharpness(
        self,
        image,
        magnification=None,
        profile=None,
    ):
        """Return a fast, noise-resistant focus score and its runtime."""
        start_time = time.perf_counter()
        if image is None or np.size(image) == 0:
            return 0.0, 0.0

        if profile is None:
            magnification = str(
                magnification or self.app.get_magnification()
            ).upper()
            profile = self.FOCUS_PROFILES.get(
                magnification,
                self.DEFAULT_FOCUS_PROFILE,
            )

        height, width = image.shape[:2]
        roi_fraction = min(
            1.0,
            max(0.2, float(profile["metric_roi_fraction"])),
        )
        roi_width = max(8, min(width, int(round(width * roi_fraction))))
        roi_height = max(8, min(height, int(round(height * roi_fraction))))
        x_start = max(0, (width - roi_width) // 2)
        y_start = max(0, (height - roi_height) // 2)
        roi = image[
            y_start:y_start + roi_height,
            x_start:x_start + roi_width,
        ]

        maximum_dimension = max(64, int(profile["metric_max_dimension"]))
        resize_scale = min(
            1.0,
            maximum_dimension / max(roi.shape[:2]),
        )
        if resize_scale < 1.0:
            roi = cv2.resize(
                roi,
                None,
                fx=resize_scale,
                fy=resize_scale,
                interpolation=cv2.INTER_AREA,
            )

        if roi.ndim == 2:
            gray = roi
        elif roi.shape[2] == 4:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        if profile.get("shape_metric"):
            sharpness = self._color_shape_score(roi, gray)
        else:
            # A small prefilter removes sensor-pixel noise without erasing
            # object boundaries at the capped working resolution.
            fine = cv2.GaussianBlur(gray, (0, 0), sigmaX=0.8)
            fine_energy = self._gradient_energy(fine)
            sharpness = self._tiled_tenengrad_score(fine_energy)

        if not np.isfinite(sharpness):
            sharpness = 0.0
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return float(sharpness), float(elapsed_ms)

    def _score_focus_frame(self, frame, profile):
        sharpness, elapsed_ms = self.find_sharpness(
            frame,
            profile=profile,
        )
        now = time.monotonic()
        if now - self._last_sharpness_publish_time >= 0.05:
            self._run_on_ui_thread(self.sharpness_callback, sharpness)
            self._last_sharpness_publish_time = now
        return sharpness, elapsed_ms

    def _focus_profile(self, focus_range=None, magnification=None):
        magnification = str(
            magnification or self.app.get_magnification()
        ).upper()
        profile = dict(
            self.FOCUS_PROFILES.get(
                magnification,
                self.DEFAULT_FOCUS_PROFILE,
            )
        )
        if focus_range is None:
            focus_range = profile["initial_range"]
        profile["target_precision"] = min(
            float(profile["target_precision"]),
            max(self.STAGE_Z_INCREMENT, float(focus_range) / 4),
        )
        profile["refinement_radius"] = min(
            float(profile["refinement_radius"]),
            float(focus_range),
        )
        return magnification, profile

    @staticmethod
    def _positive_finite(name, value):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value)
            or float(value) <= 0
        ):
            raise ValueError(f"{name} must be a positive finite number.")
        return float(value)

    def _move_to_focus_z(self, target_z, wait=True):
        self._check_focus_cancelled()
        self.stage.wait_until_not_busy(
            cancel_check=self._check_focus_cancelled
        )
        moved = self.stage.move_to_z(
            float(target_z),
            wait=wait,
            cancel_check=self._check_focus_cancelled,
        )
        if not moved:
            raise RuntimeError(
                f"The stage could not start the autofocus Z move to {target_z:g}."
            )

    def _measure_focus_at(self, target_z, frame_count=1, profile=None):
        self._move_to_focus_z(target_z, wait=True)
        self._check_focus_cancelled()

        # Moving and settled phases must score the same raw sensor field.
        capture = getattr(self.frame_processor, "capture_frame_raw", None)
        if capture is None:
            capture = self.frame_processor.capture_frame
        frame = capture(
            num_images=max(1, int(frame_count)),
            cancel_check=self._check_focus_cancelled,
            timeout_seconds=self.FOCUS_FRAME_TIMEOUT_SECONDS,
        )
        if profile is None:
            _magnification, profile = self._focus_profile()
        sharpness, elapsed_ms = self._score_focus_frame(frame, profile)
        actual_z = float(self.stage.get_z_position(strict=True))
        print(
            f"Settled focus | Z: {actual_z:>10.2f} | "
            f"Sharpness: {sharpness:>12.3f} | "
            f"Metric: {elapsed_ms:>7.2f}ms"
        )
        return float(sharpness), actual_z

    def _camera(self):
        getter = getattr(self.frame_processor, "get_camera", None)
        if getter is not None:
            try:
                return getter()
            except Exception:
                return None
        return getattr(self.frame_processor, "hcam", None)

    def _stabilize_and_lock_auto_exposure(self, magnification):
        """Freeze auto exposure so every focus score is comparable."""
        camera = self._camera()
        required_methods = (
            "get_AutoExpoEnable",
            "put_AutoExpoEnable",
        )
        if camera is None or not all(
            hasattr(camera, method) for method in required_methods
        ):
            return None

        try:
            was_enabled = bool(camera.get_AutoExpoEnable())
        except Exception as exc:
            print(f"Could not read auto exposure state: {exc}")
            return None
        if not was_enabled:
            return None

        can_measure_exposure = all(
            hasattr(camera, method)
            for method in ("get_ExpoTime", "get_ExpoAGain")
        )
        if can_measure_exposure:
            settle_seconds = (
                self.AUTO_EXPOSURE_SETTLE_SECONDS
                if magnification == "100X"
                else min(0.25, self.AUTO_EXPOSURE_SETTLE_SECONDS)
            )
            deadline = time.monotonic() + settle_seconds
            latest = self._latest_frame_data()
            last_frame_id = latest["frame_id"] if latest is not None else -1
            previous_state = None
            stable_transitions = 0
            while time.monotonic() < deadline and stable_transitions < 2:
                data = self._wait_for_new_frame_data(
                    last_frame_id,
                    timeout=0.12,
                )
                if data is None:
                    continue
                last_frame_id = data["frame_id"]
                try:
                    state = (
                        float(camera.get_ExpoTime()),
                        float(camera.get_ExpoAGain()),
                    )
                except Exception as exc:
                    print(
                        "Could not read exposure while stabilizing "
                        f"autofocus: {exc}"
                    )
                    break
                if previous_state is not None:
                    time_scale = max(1.0, abs(previous_state[0]))
                    gain_scale = max(1.0, abs(previous_state[1]))
                    stable = (
                        abs(state[0] - previous_state[0]) / time_scale
                        <= 0.02
                        and abs(state[1] - previous_state[1]) / gain_scale
                        <= 0.02
                    )
                    stable_transitions = (
                        stable_transitions + 1 if stable else 0
                    )
                previous_state = state

        try:
            camera.put_AutoExpoEnable(False)
        except Exception as exc:
            print(f"Could not lock auto exposure for autofocus: {exc}")
            return None
        print(f"{magnification} exposure locked for autofocus")
        return camera

    @staticmethod
    def _restore_auto_exposure(camera):
        if camera is None:
            return
        try:
            camera.put_AutoExpoEnable(True)
        except Exception as exc:
            print(f"Could not restore auto exposure: {exc}")

    def _fresh_live_frame_barrier(self):
        flush = getattr(self.frame_processor, "_flush_camera_queue", None)
        if flush is not None:
            try:
                minimum_frame_id = int(flush(discard_frames=1))
                return minimum_frame_id - 1
            except Exception as exc:
                print(f"Could not flush camera before focus sweep: {exc}")
        latest = self._latest_frame_data()
        return latest["frame_id"] if latest is not None else -1

    def _is_z_busy(self):
        if hasattr(self.stage, "is_z_busy"):
            return bool(self.stage.is_z_busy(strict=True))
        return bool(self.stage.is_busy())

    def _z_limit_state(self):
        getter = getattr(self.stage, "get_z_limit_state", None)
        if getter is None:
            return False, False
        negative_limit, positive_limit = getter(strict=True)
        return bool(negative_limit), bool(positive_limit)

    def _raise_if_z_limit_blocks(self, target_z=None):
        negative_limit, positive_limit = self._z_limit_state()
        if not negative_limit and not positive_limit:
            return
        if target_z is None:
            raise RuntimeError(
                "Autofocus cannot start while a Z-axis physical limit "
                "switch is active."
            )
        current_z = float(self.stage.get_z_position(strict=True))
        tolerance = max(
            self.STAGE_Z_INCREMENT,
            float(
                getattr(
                    self.stage,
                    "Z_POSITION_TOLERANCE_UM",
                    self.STAGE_Z_INCREMENT,
                )
            ),
        )
        blocked_negative = (
            negative_limit and float(target_z) < current_z - tolerance
        )
        blocked_positive = (
            positive_limit and float(target_z) > current_z + tolerance
        )
        if blocked_negative or blocked_positive:
            direction = "negative" if blocked_negative else "positive"
            raise RuntimeError(
                f"Autofocus reached the {direction} Z-axis physical limit."
            )

    def _effective_sweep_velocity(self, requested_velocity, profile):
        frame_rate = getattr(self.frame_processor, "camera_fps", 0)
        if (
            not isinstance(frame_rate, (int, float, np.number))
            or not np.isfinite(frame_rate)
            or not 5 <= float(frame_rate) <= 120
        ):
            frame_rate = 30.0
        frame_limited_velocity = (
            float(frame_rate) * float(profile["maximum_live_spacing"])
        )
        return max(
            1,
            int(round(min(float(requested_velocity), frame_limited_velocity))),
        )

    def _continuous_focus_sweep(
        self,
        z_start,
        z_end,
        z_velocity,
        z_acceleration,
        profile,
        positioning_velocity=None,
    ):
        """Sweep low-to-high once while scoring the newest live camera frame."""
        positioning_velocity = max(
            int(z_velocity),
            int(positioning_velocity or z_velocity),
        )
        self.stage.set_z_velocity(positioning_velocity)
        self.stage.set_z_acceleration(z_acceleration)
        self._move_to_focus_z(z_start, wait=True)
        self.stage.set_z_velocity(z_velocity)
        last_frame_id = self._fresh_live_frame_barrier()
        self._check_focus_cancelled()

        if not self.stage.move_to_z(
            float(z_end),
            wait=False,
            cancel_check=self._check_focus_cancelled,
        ):
            raise RuntimeError("The stage could not start the autofocus sweep.")

        distance = abs(float(z_end) - float(z_start))
        expected_seconds = distance / max(1.0, float(z_velocity))
        deadline = time.monotonic() + min(
            180.0,
            max(5.0, expected_seconds * 2 + 5.0),
        )
        samples = []
        idle_at_target = 0
        last_log_time = 0.0

        try:
            while True:
                self._check_focus_cancelled()
                self._raise_if_z_limit_blocks(z_end)
                data = self._wait_for_new_frame_data(
                    last_frame_id,
                    timeout=0.08,
                )
                if data is not None:
                    last_frame_id = data["frame_id"]
                    frame_z = float(data["z"])
                    if (
                        min(z_start, z_end) - 1.0
                        <= frame_z
                        <= max(z_start, z_end) + 1.0
                    ):
                        sharpness, elapsed_ms = self._score_focus_frame(
                            data["frame"],
                            profile,
                        )
                        samples.append((frame_z, float(sharpness)))
                        now = time.monotonic()
                        if now - last_log_time >= 0.25:
                            print(
                                f"Live focus | Z: {frame_z:>10.2f} | "
                                f"Sharpness: {sharpness:>12.3f} | "
                                f"Metric: {elapsed_ms:>7.2f}ms"
                            )
                            last_log_time = now

                if self._is_z_busy():
                    idle_at_target = 0
                else:
                    actual_z = float(
                        self.stage.get_z_position(strict=True)
                    )
                    tolerance = max(
                        self.STAGE_Z_INCREMENT,
                        float(
                            getattr(
                                self.stage,
                                "Z_POSITION_TOLERANCE_UM",
                                self.STAGE_Z_INCREMENT,
                            )
                        ),
                    )
                    if abs(actual_z - float(z_end)) <= tolerance:
                        idle_at_target += 1
                        if idle_at_target >= 2:
                            break
                    else:
                        idle_at_target = 0

                if time.monotonic() >= deadline:
                    raise TimeoutError("The autofocus sweep timed out.")
        except Exception:
            self._stop_z_safely()
            raise

        finite_samples = self._finite_unique_samples(samples)
        if len(finite_samples) >= self.MINIMUM_LIVE_SAMPLES:
            best_live_z, best_live_score = max(
                finite_samples,
                key=lambda sample: sample[1],
            )
            print(
                f"Live sweep complete | Samples: {len(finite_samples)} | "
                f"Best observed Z: {best_live_z:.2f} | "
                f"Sharpness: {best_live_score:.3f}"
            )
            return finite_samples

        print(
            "Too few live frames were received; "
            "using settled range samples"
        )
        fallback = []
        for target_z in np.linspace(
            float(z_start),
            float(z_end),
            self.FALLBACK_SAMPLE_COUNT,
        ):
            score, actual_z = self._measure_focus_at(
                target_z,
                frame_count=1,
                profile=profile,
            )
            fallback.append((actual_z, score))
        return self._finite_unique_samples(fallback)

    @staticmethod
    def _finite_unique_samples(samples):
        scores_by_z = {}
        for z, score in samples:
            if not np.isfinite(z) or not np.isfinite(score):
                continue
            key = round(float(z), 1)
            scores_by_z.setdefault(key, []).append(float(score))
        return sorted(
            (z, float(np.median(scores)))
            for z, scores in scores_by_z.items()
        )

    @staticmethod
    def _quadratic_peak(samples):
        if len(samples) != 3:
            return None
        ordered = sorted((float(z), float(score)) for z, score in samples)
        center = ordered[1][0]
        x_values = np.asarray(
            [sample[0] - center for sample in ordered],
            dtype=np.float64,
        )
        y_values = np.asarray(
            [sample[1] for sample in ordered],
            dtype=np.float64,
        )
        if len(np.unique(x_values)) != 3:
            return None
        matrix = np.column_stack((
            x_values * x_values,
            x_values,
            np.ones(3),
        ))
        try:
            coefficient, slope, _offset = np.linalg.solve(matrix, y_values)
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite(coefficient) or coefficient >= 0:
            return None
        vertex = center - slope / (2 * coefficient)
        if not ordered[0][0] <= vertex <= ordered[-1][0]:
            return None
        return float(vertex)

    def _estimate_sweep_peak(self, samples):
        measured = self._finite_unique_samples(samples)
        if not measured:
            raise RuntimeError("No valid focus frames were measured.")
        if len(measured) < 3:
            best_z, _score = max(measured, key=lambda sample: sample[1])
            return float(best_z), 0.0

        z_values = np.asarray(
            [sample[0] for sample in measured],
            dtype=np.float64,
        )
        scores = np.asarray(
            [sample[1] for sample in measured],
            dtype=np.float64,
        )
        median_filtered = scores.copy()
        if len(scores) >= 5:
            for index in range(1, len(scores) - 1):
                median_filtered[index] = np.median(
                    scores[index - 1:index + 2]
                )
        padded = np.pad(median_filtered, (1, 1), mode="edge")
        smoothed = (
            0.25 * padded[:-2]
            + 0.50 * padded[1:-1]
            + 0.25 * padded[2:]
        )
        best_index = int(np.argmax(smoothed))
        estimate = float(z_values[best_index])
        if 0 < best_index < len(measured) - 1:
            vertex = self._quadratic_peak(
                measured[best_index - 1:best_index + 2]
            )
            if vertex is not None:
                estimate = vertex

        spacings = np.diff(z_values)
        positive_spacings = spacings[spacings > 0]
        observed_spacing = (
            float(np.median(positive_spacings))
            if positive_spacings.size
            else 0.0
        )
        return estimate, observed_spacing

    @staticmethod
    def _bounded_interval(center, radius, outer_start, outer_end):
        outer_start, outer_end = sorted((
            float(outer_start),
            float(outer_end),
        ))
        width = min(
            outer_end - outer_start,
            max(0.0, float(radius) * 2),
        )
        lower = float(center) - width / 2
        upper = float(center) + width / 2
        if lower < outer_start:
            upper += outer_start - lower
            lower = outer_start
        if upper > outer_end:
            lower -= upper - outer_end
            upper = outer_end
        return max(outer_start, lower), min(outer_end, upper)

    @staticmethod
    def _relative_prominence(samples):
        scores = np.asarray(
            [
                float(score)
                for _z, score in samples
                if np.isfinite(score)
            ],
            dtype=np.float64,
        )
        if scores.size < 3:
            return 0.0
        peak = float(np.max(scores))
        baseline = float(np.percentile(scores, 10))
        scale = max(abs(peak), abs(baseline), 1e-9)
        return max(0.0, (peak - baseline) / scale)

    def _settled_refinement(
        self,
        coarse_samples,
        z_start,
        z_end,
        fallback_z,
        profile,
    ):
        coarse_estimate, observed_spacing = self._estimate_sweep_peak(
            coarse_samples
        )
        initial_radius = max(
            float(profile["refinement_radius"]),
            observed_spacing * 2.5,
            float(profile["target_precision"]) * 4,
        )
        initial_radius = min(
            initial_radius,
            (float(z_end) - float(z_start)) / 2,
        )
        lower, upper = self._bounded_interval(
            coarse_estimate,
            initial_radius,
            z_start,
            z_end,
        )
        positioning_velocity = int(
            profile.get(
                "active_positioning_velocity",
                profile["refinement_velocity"],
            )
        )
        refinement_velocity = int(
            profile.get(
                "active_refinement_velocity",
                profile["refinement_velocity"],
            )
        )
        self.stage.set_z_velocity(max(1, positioning_velocity))
        self._move_to_focus_z(lower, wait=True)
        self.stage.set_z_velocity(max(1, refinement_velocity))
        target_precision = float(profile["target_precision"])
        frame_count = int(profile["refinement_frame_count"])
        cache = {}
        measured_samples = []
        last_round = []
        peak_bracketed = False
        unbracketed = False
        expansion_count = 0
        refinement_rounds = 0
        converged = False

        def evaluate(target_z, requested_frame_count=frame_count, use_cache=True):
            target_z = min(upper, max(lower, float(target_z)))
            target_z = round(target_z, 1)
            if use_cache and target_z in cache:
                return cache[target_z]
            score, actual_z = self._measure_focus_at(
                target_z,
                frame_count=requested_frame_count,
                profile=profile,
            )
            sample = (float(actual_z), float(score))
            if use_cache:
                cache[target_z] = sample
                measured_samples.append(sample)
            return sample

        maximum_search_rounds = (
            self.MAX_REFINEMENT_ROUNDS
            + self.MAX_REFINEMENT_EXPANSIONS
        )
        for round_index in range(maximum_search_rounds):
            self._check_focus_cancelled()
            targets = np.linspace(
                lower,
                upper,
                self.REFINEMENT_POINT_COUNT,
            )
            last_round = []
            seen_z = set()
            for target_z in targets:
                sample = evaluate(target_z)
                if sample[0] not in seen_z:
                    last_round.append(sample)
                    seen_z.add(sample[0])
            last_round.sort()
            if len(last_round) < 3:
                unbracketed = True
                break

            best_index = max(
                range(len(last_round)),
                key=lambda index: last_round[index][1],
            )
            if best_index == 0:
                peak_bracketed = False
                if (
                    lower <= float(z_start) + self.STAGE_Z_INCREMENT
                    or expansion_count >= self.MAX_REFINEMENT_EXPANSIONS
                ):
                    unbracketed = True
                    break
                width = max(
                    upper - lower,
                    target_precision * 4,
                    self.STAGE_Z_INCREMENT * 4,
                )
                anchor = lower
                new_lower = max(float(z_start), anchor - 1.5 * width)
                new_upper = min(float(z_end), anchor + 0.25 * width)
            elif best_index == len(last_round) - 1:
                peak_bracketed = False
                if (
                    upper >= float(z_end) - self.STAGE_Z_INCREMENT
                    or expansion_count >= self.MAX_REFINEMENT_EXPANSIONS
                ):
                    unbracketed = True
                    break
                width = max(
                    upper - lower,
                    target_precision * 4,
                    self.STAGE_Z_INCREMENT * 4,
                )
                anchor = upper
                new_lower = max(float(z_start), anchor - 0.25 * width)
                new_upper = min(float(z_end), anchor + 1.5 * width)
            else:
                peak_bracketed = True
                refinement_rounds += 1
                new_lower, new_upper = (
                    last_round[best_index - 1][0],
                    last_round[best_index + 1][0],
                )
            lower = max(float(z_start), float(new_lower))
            upper = min(float(z_end), float(new_upper))
            if best_index in (0, len(last_round) - 1):
                expansion_count += 1
                if upper - lower < self.STAGE_Z_INCREMENT:
                    unbracketed = True
                    break
                print(
                    f"Focus refinement {round_index + 1} | "
                    f"Expanding bracket: {lower:.2f} to {upper:.2f}"
                )
                self.stage.set_z_velocity(max(1, positioning_velocity))
                self._move_to_focus_z(lower, wait=True)
                self.stage.set_z_velocity(max(1, refinement_velocity))
                continue
            print(
                f"Focus refinement {round_index + 1} | "
                f"Bracket: {lower:.2f} to {upper:.2f}"
            )
            if (
                upper - lower
                <= target_precision * 2 + self.STAGE_Z_INCREMENT * 1e-6
            ):
                converged = True
                break
            if refinement_rounds >= self.MAX_REFINEMENT_ROUNDS:
                break

        prominence = self._relative_prominence(measured_samples)
        if prominence < float(profile["minimum_prominence"]):
            print(
                "Autofocus found too little structure for a reliable peak "
                f"(prominence {prominence:.3f}); retaining the starting Z"
            )
            self._move_to_focus_z(fallback_z, wait=True)
            best_score = max(
                (sample[1] for sample in measured_samples),
                default=0.0,
            )
            return float(best_score), float(fallback_z), prominence

        if not last_round:
            raise RuntimeError("The settled focus refinement produced no samples.")
        best_index = max(
            range(len(last_round)),
            key=lambda index: last_round[index][1],
        )
        if (
            unbracketed
            or not peak_bracketed
            or not converged
            or best_index in (0, len(last_round) - 1)
        ):
            print(
                "Autofocus could not bracket a settled focus peak inside "
                "the configured range; retaining the starting Z"
            )
            self._move_to_focus_z(fallback_z, wait=True)
            best_score = max(
                (sample[1] for sample in measured_samples),
                default=0.0,
            )
            return float(best_score), float(fallback_z), 0.0
        best_discrete_z = float(last_round[best_index][0])
        boundary_margin = max(
            self.STAGE_Z_INCREMENT,
            target_precision * 1.5,
        )
        if (
            abs(best_discrete_z - float(z_start)) <= boundary_margin
            or abs(best_discrete_z - float(z_end)) <= boundary_margin
        ):
            print(
                "Autofocus did not bracket a peak inside the configured "
                "range; retaining the starting Z"
            )
            self._move_to_focus_z(fallback_z, wait=True)
            best_score = max(
                (sample[1] for sample in measured_samples),
                default=0.0,
            )
            return float(best_score), float(fallback_z), 0.0

        quadratic_z = None
        if 0 < best_index < len(last_round) - 1:
            quadratic_z = self._quadratic_peak(
                last_round[best_index - 1:best_index + 2]
            )

        final_targets = [best_discrete_z]
        if (
            quadratic_z is not None
            and abs(quadratic_z - best_discrete_z)
            >= self.STAGE_Z_INCREMENT / 2
        ):
            final_targets.append(quadratic_z)

        final_samples = [
            evaluate(
                target_z,
                requested_frame_count=int(profile["final_frame_count"]),
                use_cache=False,
            )
            for target_z in final_targets
        ]
        best_z, best_score = max(
            final_samples,
            key=lambda sample: sample[1],
        )
        self._move_to_focus_z(best_z, wait=True)
        return float(best_score), float(best_z), prominence

    def _stop_z_safely(self):
        try:
            self.stage.stop_z()
        except Exception:
            pass
        try:
            self.stage.wait_until_not_busy()
        except Exception:
            pass

    def _restore_stage_setting(self, setter, value, description):
        try:
            setter(value)
        except Exception as exc:
            print(f"Could not restore {description}: {exc}")

    def auto_focus(
        self,
        focus_range=None,
        z_velo=None,
        z_accel=None,
    ):
        (
            magnification,
            focus_range,
            z_velo,
            z_accel,
        ) = self._validated_autofocus_settings(
            focus_range=focus_range,
            z_velo=z_velo,
            z_accel=z_accel,
        )

        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError("Autofocus is already active.")
        self._operation_active.set()

        start_time = time.perf_counter()
        default_z_velocity = None
        default_z_acceleration = None
        exposure_camera = None

        try:
            default_z_velocity = self.stage.get_z_velocity()
            default_z_acceleration = self.stage.get_z_acceleration()
            starting_z = float(self.stage.get_z_position(strict=True))
            self._raise_if_z_limit_blocks()
            magnification, profile = self._focus_profile(
                focus_range,
                magnification=magnification,
            )
            z_start = starting_z - focus_range
            z_end = starting_z + focus_range
            sweep_velocity = self._effective_sweep_velocity(z_velo, profile)

            print(
                "Starting range-scan autofocus | "
                f"Magnification: {magnification} | "
                f"Range: {z_start:.1f} to {z_end:.1f} | "
                f"Live velocity: {sweep_velocity} | "
                f"Target precision: {profile['target_precision']:g}"
            )

            self._check_focus_cancelled()
            exposure_camera = self._stabilize_and_lock_auto_exposure(
                magnification
            )
            coarse_samples = self._continuous_focus_sweep(
                z_start,
                z_end,
                sweep_velocity,
                int(round(z_accel)),
                profile,
                positioning_velocity=int(round(z_velo)),
            )
            self._check_focus_cancelled()

            refinement_velocity = max(
                1,
                int(round(min(
                    z_velo,
                    float(profile["refinement_velocity"]),
                ))),
            )
            profile["active_positioning_velocity"] = int(round(z_velo))
            profile["active_refinement_velocity"] = refinement_velocity
            self.stage.set_z_velocity(refinement_velocity)
            self.stage.set_z_acceleration(int(round(z_accel)))
            best_sharpness, best_z, prominence = self._settled_refinement(
                coarse_samples,
                z_start,
                z_end,
                starting_z,
                profile,
            )
            final_z = float(self.stage.get_z_position(strict=True))
            elapsed = time.perf_counter() - start_time
            print(
                f"Autofocus complete | Best Z: {best_z:.2f} | "
                f"Stage Z: {final_z:.2f} | "
                f"Sharpness: {best_sharpness:.3f} | "
                f"Prominence: {prominence:.3f} | "
                f"Time: {elapsed:.2f}s"
            )
            return best_z
        except FocusCancelled:
            self._stop_z_safely()
            actual_z = float(self.stage.get_z_position(strict=False))
            print(f"Autofocus stopped at Z {actual_z:.2f}")
            return actual_z
        except Exception:
            self._stop_z_safely()
            raise
        finally:
            self._restore_auto_exposure(exposure_camera)
            if default_z_velocity is not None:
                self._restore_stage_setting(
                    self.stage.set_z_velocity,
                    default_z_velocity,
                    "Z velocity",
                )
            if default_z_acceleration is not None:
                self._restore_stage_setting(
                    self.stage.set_z_acceleration,
                    default_z_acceleration,
                    "Z acceleration",
                )
            self._operation_active.clear()
            self._operation_lock.release()
