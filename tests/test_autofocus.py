from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np


APP_DIR = Path(__file__).resolve().parents[1] / "App"
sys.path.insert(0, str(APP_DIR))

from Hardware.turret_controller import TurretController  # noqa: E402
from Imaging.focus import FocusController  # noqa: E402


class _FakeApp:
    def __init__(self, magnification="10X"):
        self.magnification = magnification

    def get_magnification(self):
        return self.magnification


class _FakeStage:
    POSITION_TOLERANCE_UM = 1.0
    Z_POSITION_TOLERANCE_UM = 0.05

    def __init__(self, z=0.0):
        self.z = float(z)
        self.z_velocity = 500
        self.z_acceleration = 10000
        self.moves = []
        self.velocity_changes = []
        self.acceleration_changes = []
        self.stop_calls = 0

    def get_z_velocity(self):
        return self.z_velocity

    def get_z_acceleration(self):
        return self.z_acceleration

    def set_z_velocity(self, value):
        self.z_velocity = int(value)
        self.velocity_changes.append(self.z_velocity)
        return self.z_velocity

    def set_z_acceleration(self, value):
        self.z_acceleration = int(value)
        self.acceleration_changes.append(self.z_acceleration)

    def get_z_position(self, strict=False):
        return self.z

    @staticmethod
    def wait_until_not_busy(**_kwargs):
        return 0.0

    @staticmethod
    def is_z_busy(strict=False):
        return False

    @staticmethod
    def is_busy():
        return False

    def move_to_z(self, z, wait=True, **_kwargs):
        self.z = round(float(z), 1)
        self.moves.append((self.z, wait))
        return True

    def stop_z(self):
        self.stop_calls += 1


class _FakeFrameProcessor:
    def __init__(self, stage):
        self.stage = stage
        self.camera_fps = 30.0
        self.frame_buffer = []
        self._frame_condition = threading.Condition()
        self.raw_capture_calls = []
        self.cropped_capture_calls = []
        self.camera = None

    def capture_frame_raw(self, **kwargs):
        self.raw_capture_calls.append(kwargs)
        return np.full((40, 60, 3), self.stage.z, dtype=np.float32)

    def capture_frame(self, **kwargs):
        self.cropped_capture_calls.append(kwargs)
        return np.zeros((10, 10, 3), dtype=np.float32)

    def get_camera(self):
        return self.camera


class AutofocusTests(unittest.TestCase):
    @staticmethod
    def _controller(magnification="10X", z=0.0):
        stage = _FakeStage(z)
        frame_processor = _FakeFrameProcessor(stage)
        controller = FocusController(
            app=_FakeApp(magnification),
            stage=stage,
            frame_processor=frame_processor,
        )
        return controller, stage, frame_processor

    @staticmethod
    def _shape_image():
        image = np.full((720, 960, 3), 170, dtype=np.uint8)
        cv2.rectangle(image, (80, 100), (610, 570), (45, 95, 220), -1)
        cv2.circle(image, (790, 170), 85, (220, 60, 70), -1)
        cv2.line(image, (30, 680), (900, 40), (20, 215, 45), 10)
        return image

    def test_profiles_shrink_range_and_precision_with_magnification(self):
        expected = {
            "2X": (800.0, 500, 10000, 5.0),
            "10X": (300.0, 250, 10000, 1.0),
            "20X": (200.0, 150, 8000, 0.5),
            "100X": (60.0, 50, 6000, 0.1),
        }

        for magnification, values in expected.items():
            with self.subTest(magnification=magnification):
                controller, _stage, _processor = self._controller(
                    magnification
                )
                defaults = controller.autofocus_defaults()
                _name, profile = controller._focus_profile(
                    defaults["focus_range"]
                )
                self.assertEqual(
                    (
                        defaults["focus_range"],
                        defaults["z_velo"],
                        defaults["z_accel"],
                        profile["target_precision"],
                    ),
                    values,
                )

    def test_live_velocity_is_capped_by_camera_sampling(self):
        controller, _stage, processor = self._controller("100X")
        processor.camera_fps = 20.0
        _name, profile = controller._focus_profile(60)

        self.assertEqual(
            controller._effective_sweep_velocity(100, profile),
            15,
        )

        controller.app.magnification = "2X"
        _name, profile = controller._focus_profile(800)
        self.assertEqual(
            controller._effective_sweep_velocity(500, profile),
            500,
        )

    def test_focus_metrics_decrease_monotonically_with_blur(self):
        image = self._shape_image()
        blur_sigmas = (0.0, 0.8, 1.5, 3.0, 6.0)

        for magnification in ("2X", "10X", "20X", "100X"):
            with self.subTest(magnification=magnification):
                controller, _stage, _processor = self._controller(
                    magnification
                )
                scores = []
                for sigma in blur_sigmas:
                    frame = (
                        image
                        if sigma == 0
                        else cv2.GaussianBlur(image, (0, 0), sigma)
                    )
                    scores.append(controller.find_sharpness(frame)[0])
                self.assertTrue(all(
                    left > right
                    for left, right in zip(scores, scores[1:])
                ))

    def test_100x_shape_metric_rejects_blurred_sensor_noise(self):
        controller, _stage, _processor = self._controller("100X")
        image = self._shape_image()
        blurred = cv2.GaussianBlur(image, (0, 0), 6.0)
        random = np.random.default_rng(23)
        blurred_noisy = np.clip(
            blurred.astype(np.float32)
            + random.normal(0, 10, blurred.shape),
            0,
            255,
        ).astype(np.uint8)
        blank_noisy = np.clip(
            np.full_like(image, 170).astype(np.float32)
            + random.normal(0, 10, image.shape),
            0,
            255,
        ).astype(np.uint8)

        sharp_score = controller.find_sharpness(image)[0]
        blurred_score = controller.find_sharpness(blurred_noisy)[0]
        blank_score = controller.find_sharpness(blank_noisy)[0]

        self.assertGreater(sharp_score, blurred_score * 3)
        self.assertGreater(blurred_score, blank_score * 2)

    def test_live_sweep_consumes_asynchronous_frames_across_full_range(self):
        class FrameProcessor:
            def __init__(self):
                self.camera_fps = 60.0
                self.frame_buffer = []
                self._frame_condition = threading.Condition()
                self.frame_id = 0

            def _flush_camera_queue(self, discard_frames=1):
                with self._frame_condition:
                    self.frame_buffer.clear()
                    return self.frame_id + int(discard_frames)

            def push(self, z, score):
                with self._frame_condition:
                    self.frame_buffer.append({
                        "frame": np.full(
                            (8, 8, 3),
                            score,
                            dtype=np.float32,
                        ),
                        "frame_id": self.frame_id,
                        "z": float(z),
                    })
                    self.frame_id += 1
                    self._frame_condition.notify_all()

        class SweepingStage(_FakeStage):
            def __init__(self, processor):
                super().__init__(0)
                self.processor = processor
                self.busy = False
                self.worker = None

            def is_z_busy(self, strict=False):
                return self.busy

            def move_to_z(self, z, wait=True, **_kwargs):
                target = float(z)
                if wait:
                    self.z = target
                    self.moves.append((target, True))
                    return True

                start = self.z
                self.busy = True

                def produce():
                    for frame_z in np.linspace(start, target, 41):
                        self.z = float(frame_z)
                        score = 1000.0 - 8.0 * (frame_z - 2.0) ** 2
                        self.processor.push(frame_z, score)
                        time.sleep(0.002)
                    self.z = target
                    self.busy = False

                self.worker = threading.Thread(target=produce, daemon=True)
                self.worker.start()
                return True

        processor = FrameProcessor()
        stage = SweepingStage(processor)
        controller = FocusController(_FakeApp("20X"), stage, processor)
        controller._score_focus_frame = (
            lambda frame, _profile: (float(frame[0, 0, 0]), 0.0)
        )
        _name, profile = controller._focus_profile(10)

        samples = controller._continuous_focus_sweep(
            -10,
            10,
            z_velocity=60,
            z_acceleration=8000,
            profile=profile,
        )
        stage.worker.join(timeout=1)
        peak_z, _spacing = controller._estimate_sweep_peak(samples)

        self.assertGreaterEqual(len(samples), 10)
        self.assertAlmostEqual(samples[0][0], -9.5, delta=1.0)
        self.assertAlmostEqual(samples[-1][0], 10.0, delta=0.2)
        self.assertAlmostEqual(peak_z, 2.0, delta=1.0)

    def test_sweep_peak_estimator_rejects_an_isolated_outlier(self):
        controller, _stage, _processor = self._controller("20X")
        samples = [
            (float(z), 1000.0 - 10.0 * (z - 2.0) ** 2)
            for z in range(-10, 11)
        ]
        samples[3] = (samples[3][0], 1_000_000.0)

        estimate, spacing = controller._estimate_sweep_peak(samples)

        self.assertAlmostEqual(estimate, 2.0, delta=1.0)
        self.assertEqual(spacing, 1.0)

    def test_settled_refinement_converges_for_every_objective(self):
        peak_z = 3.7
        coarse = [
            (float(z), 1000.0 - 5.0 * (z - peak_z) ** 2)
            for z in range(-20, 21, 2)
        ]

        for magnification in ("2X", "10X", "20X", "100X"):
            with self.subTest(magnification=magnification):
                controller, stage, _processor = self._controller(
                    magnification
                )
                _name, profile = controller._focus_profile(20)
                measured = []

                def measure(target_z, frame_count=1, profile=None):
                    actual_z = round(float(target_z), 1)
                    stage.z = actual_z
                    measured.append((actual_z, frame_count))
                    score = 2000.0 - 30.0 * (actual_z - peak_z) ** 2
                    return score, actual_z

                with patch.object(
                    controller,
                    "_measure_focus_at",
                    side_effect=measure,
                ):
                    _score, best_z, prominence = (
                        controller._settled_refinement(
                            coarse,
                            -20,
                            20,
                            fallback_z=0,
                            profile=profile,
                        )
                    )

                self.assertLessEqual(
                    abs(best_z - peak_z),
                    max(profile["target_precision"], 0.2),
                )
                self.assertGreater(prominence, profile["minimum_prominence"])
                self.assertLessEqual(len(measured), 30)
                self.assertEqual(stage.z, best_z)

    def test_low_texture_refinement_keeps_starting_z(self):
        controller, stage, _processor = self._controller("100X", z=5.0)
        _name, profile = controller._focus_profile(10)
        coarse = [(float(z), 1.0) for z in range(-5, 16)]

        def measure(target_z, frame_count=1, profile=None):
            stage.z = round(float(target_z), 1)
            return 100.0, stage.z

        with patch.object(
            controller,
            "_measure_focus_at",
            side_effect=measure,
        ):
            _score, best_z, prominence = controller._settled_refinement(
                coarse,
                -5,
                15,
                fallback_z=5,
                profile=profile,
            )

        self.assertEqual(best_z, 5.0)
        self.assertEqual(stage.z, 5.0)
        self.assertEqual(prominence, 0.0)

    def test_refinement_expands_when_live_peak_estimate_is_wrong(self):
        controller, stage, _processor = self._controller("100X", z=0.0)
        _name, profile = controller._focus_profile(20)
        coarse = [
            (float(z), 1000.0 - 20.0 * float(z) ** 2)
            for z in range(-20, 21)
        ]
        # The settled peak is far enough outside the initial +/-3 um bracket
        # that expansion and precision convergence require separate budgets.
        settled_peak_z = 18.0

        def measure(target_z, frame_count=1, profile=None):
            stage.z = round(float(target_z), 1)
            score = 2000.0 - 30.0 * (stage.z - settled_peak_z) ** 2
            return score, stage.z

        with patch.object(
            controller,
            "_measure_focus_at",
            side_effect=measure,
        ):
            _score, best_z, prominence = controller._settled_refinement(
                coarse,
                -20,
                20,
                fallback_z=0,
                profile=profile,
            )

        self.assertAlmostEqual(best_z, settled_peak_z, delta=0.2)
        self.assertEqual(stage.z, best_z)
        self.assertGreater(prominence, profile["minimum_prominence"])

    def test_monotonic_curve_does_not_park_stage_at_range_boundary(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                controller, stage, _processor = self._controller(
                    "2X",
                    z=100.0,
                )
                _name, profile = controller._focus_profile(20)
                coarse = [
                    (float(z), direction * float(z))
                    for z in range(80, 121)
                ]

                def measure(target_z, frame_count=1, profile=None):
                    stage.z = round(float(target_z), 1)
                    return direction * stage.z, stage.z

                with patch.object(
                    controller,
                    "_measure_focus_at",
                    side_effect=measure,
                ):
                    _score, best_z, prominence = (
                        controller._settled_refinement(
                            coarse,
                            80,
                            120,
                            fallback_z=100,
                            profile=profile,
                        )
                    )

                self.assertEqual(best_z, 100.0)
                self.assertEqual(stage.z, 100.0)
                self.assertEqual(prominence, 0.0)

    def test_auto_focus_uses_one_sweep_then_restores_stage_settings(self):
        controller, stage, _processor = self._controller("20X", z=50.0)
        peak_z = 57.3
        coarse = [
            (float(z), 1000.0 - 5.0 * (z - peak_z) ** 2)
            for z in range(30, 71, 2)
        ]

        def measure(target_z, frame_count=1, profile=None):
            actual_z = round(float(target_z), 1)
            stage.z = actual_z
            score = 2000.0 - 30.0 * (actual_z - peak_z) ** 2
            return score, actual_z

        with (
            patch.object(
                controller,
                "_continuous_focus_sweep",
                return_value=coarse,
            ) as live_sweep,
            patch.object(
                controller,
                "_measure_focus_at",
                side_effect=measure,
            ),
        ):
            best_z = controller.auto_focus(
                focus_range=20,
                z_velo=75,
                z_accel=8000,
            )

        self.assertLessEqual(abs(best_z - peak_z), 0.5)
        self.assertEqual(stage.z, best_z)
        self.assertEqual(live_sweep.call_count, 1)
        self.assertEqual(stage.z_velocity, 500)
        self.assertEqual(stage.z_acceleration, 10000)
        self.assertEqual(stage.stop_calls, 0)
        self.assertFalse(controller.is_focus_active())

    def test_failure_stops_z_and_restores_controller_settings(self):
        controller, stage, _processor = self._controller("10X", z=25.0)

        with patch.object(
            controller,
            "_continuous_focus_sweep",
            side_effect=RuntimeError("camera failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "camera failed"):
                controller.auto_focus(
                    focus_range=20,
                    z_velo=50,
                    z_accel=9000,
                )

        self.assertEqual(stage.stop_calls, 1)
        self.assertEqual(stage.z_velocity, 500)
        self.assertEqual(stage.z_acceleration, 10000)
        self.assertFalse(controller.is_focus_active())

    def test_setup_failure_releases_autofocus_operation_lock(self):
        controller, stage, _processor = self._controller("10X", z=25.0)
        stage.get_z_acceleration = (
            lambda: (_ for _ in ()).throw(RuntimeError("controller read failed"))
        )

        with self.assertRaisesRegex(RuntimeError, "controller read failed"):
            controller.auto_focus()

        self.assertFalse(controller.is_focus_active())
        self.assertTrue(controller._operation_lock.acquire(blocking=False))
        controller._operation_lock.release()

    def test_cancel_stops_without_starting_another_move(self):
        controller, stage, _processor = self._controller("10X", z=25.0)
        controller.stop_focus_event.set()

        result = controller.auto_focus(
            focus_range=20,
            z_velo=50,
            z_accel=9000,
        )

        self.assertEqual(result, 25.0)
        self.assertEqual(stage.stop_calls, 1)
        self.assertEqual(stage.moves, [])
        self.assertEqual(stage.z_velocity, 500)
        self.assertEqual(stage.z_acceleration, 10000)

    def test_settled_measurement_uses_uncropped_fresh_average(self):
        controller, stage, processor = self._controller("10X")
        controller.find_sharpness = (
            lambda frame, **_kwargs: (float(np.mean(frame)), 0.0)
        )
        _name, profile = controller._focus_profile(20)

        score, actual_z = controller._measure_focus_at(
            12.4,
            frame_count=2,
            profile=profile,
        )

        self.assertEqual(actual_z, 12.4)
        self.assertAlmostEqual(score, 12.4, places=4)
        self.assertEqual(stage.moves[-1], (12.4, True))
        self.assertEqual(len(processor.raw_capture_calls), 1)
        self.assertEqual(processor.cropped_capture_calls, [])
        self.assertEqual(
            processor.raw_capture_calls[-1]["num_images"],
            2,
        )
        self.assertEqual(
            processor.raw_capture_calls[-1]["timeout_seconds"],
            controller.FOCUS_FRAME_TIMEOUT_SECONDS,
        )

    def test_auto_exposure_is_locked_and_restored_for_every_objective(self):
        class Camera:
            def __init__(self):
                self.auto = True
                self.changes = []

            def get_AutoExpoEnable(self):
                return self.auto

            def put_AutoExpoEnable(self, active):
                self.auto = bool(active)
                self.changes.append(self.auto)

            @staticmethod
            def get_ExpoTime():
                return 1500

            @staticmethod
            def get_ExpoAGain():
                return 100

        for magnification in ("10X", "100X"):
            with self.subTest(magnification=magnification):
                controller, _stage, processor = self._controller(
                    magnification
                )
                processor.camera = Camera()
                controller.AUTO_EXPOSURE_SETTLE_SECONDS = 0

                camera = controller._stabilize_and_lock_auto_exposure(
                    magnification
                )
                controller._restore_auto_exposure(camera)

                self.assertIs(camera, processor.camera)
                self.assertEqual(processor.camera.changes, [False, True])
                self.assertTrue(processor.camera.auto)

    def test_concurrent_thread_requests_share_one_autofocus_worker(self):
        controller, _stage, _processor = self._controller("10X")
        controller.app.disable_buttons = lambda: None
        controller.app.enable_buttons = lambda: None
        controller._run_on_ui_thread = lambda *_args: None
        callers_ready = threading.Barrier(3)
        focus_entered = threading.Event()
        release_focus = threading.Event()
        settings_calls = []
        autofocus_calls = []
        returned_threads = []
        original_settings = controller._validated_autofocus_settings

        def slow_settings(**kwargs):
            settings_calls.append(True)
            time.sleep(0.05)
            return original_settings(**kwargs)

        def blocked_autofocus(**_kwargs):
            autofocus_calls.append(True)
            focus_entered.set()
            release_focus.wait(timeout=2)

        def request_focus():
            callers_ready.wait(timeout=2)
            returned_threads.append(
                controller.start_auto_focus_thread(wait=False)
            )

        controller._validated_autofocus_settings = slow_settings
        with patch.object(
            controller,
            "auto_focus",
            side_effect=blocked_autofocus,
        ):
            callers = [
                threading.Thread(target=request_focus)
                for _index in range(2)
            ]
            for caller in callers:
                caller.start()
            callers_ready.wait(timeout=2)
            for caller in callers:
                caller.join(timeout=2)

            self.assertTrue(focus_entered.wait(timeout=1))
            self.assertEqual(len(settings_calls), 1)
            self.assertEqual(len(autofocus_calls), 1)
            self.assertIs(returned_threads[0], returned_threads[1])

            release_focus.set()
            returned_threads[0].join(timeout=2)

        self.assertFalse(returned_threads[0].is_alive())
        self.assertFalse(controller.focus_running)

    def test_waiting_thread_caller_receives_autofocus_failure(self):
        controller, _stage, _processor = self._controller("10X")
        controller.app.disable_buttons = lambda: None
        controller.app.enable_buttons = lambda: None
        controller._run_on_ui_thread = lambda *_args: None
        controller._wait_for_focus_thread = (
            lambda focus_thread: focus_thread.join(timeout=2)
        )

        with (
            patch.object(
                controller,
                "auto_focus",
                side_effect=RuntimeError("simulated autofocus failure"),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "simulated autofocus failure",
            ),
        ):
            controller.start_auto_focus_thread(wait=True)

        self.assertFalse(controller.focus_thread.is_alive())
        self.assertFalse(controller.focus_running)

    def test_invalid_nonfinite_motion_parameters_are_rejected(self):
        controller, _stage, _processor = self._controller("10X")
        invalid_calls = (
            {"focus_range": float("nan")},
            {"focus_range": float("inf")},
            {"focus_range": True},
            {"z_velo": 0},
            {"z_accel": -1},
            {"focus_range": 301},
            {"z_velo": 251},
            {"z_accel": 10001},
        )

        for arguments in invalid_calls:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    controller.auto_focus(**arguments)

    def test_active_physical_z_limit_rejects_focus_before_motion(self):
        controller, stage, _processor = self._controller("100X", z=5.0)
        stage.get_z_limit_state = (
            lambda strict=False: (False, True)
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "physical limit switch is active",
        ):
            controller.auto_focus()

        self.assertEqual(stage.moves, [])
        self.assertFalse(controller.is_focus_active())

    def test_quadratic_peak_is_stable_at_large_stage_coordinates(self):
        peak_z = 60814.35
        samples = [
            (z, 1000.0 - 100.0 * (z - peak_z) ** 2)
            for z in (60814.1, 60814.4, 60814.7)
        ]

        estimate = FocusController._quadratic_peak(samples)

        self.assertAlmostEqual(estimate, peak_z, places=6)

    def test_objective_change_sets_magnification_before_autofocus(self):
        class FrameProcessor:
            def __init__(self):
                self.auto_exposure_changes = []

            @staticmethod
            def set_default_exposure():
                return None

            def set_auto_exposure(self, active):
                self.auto_exposure_changes.append(active)

        class App:
            def __init__(self):
                self.magnification = "2X"
                self.frame_processor = FrameProcessor()

            @staticmethod
            def disable_buttons():
                return None

            @staticmethod
            def enable_buttons():
                return None

            def set_magnification(self, magnification):
                self.magnification = magnification

        class Stage:
            @staticmethod
            def get_position():
                return 100.0, 200.0, 300.0

            @staticmethod
            def move_to_xy(_x, _y, **_kwargs):
                return True

            @staticmethod
            def move_to_z(_z, **_kwargs):
                return True

        class Turret:
            @staticmethod
            def check_position():
                return 1

            @staticmethod
            def turn_to_position(_position, **_kwargs):
                return None

        for position, expected_magnification in ((3, "20X"), (5, "100X")):
            with self.subTest(magnification=expected_magnification):
                app = App()
                autofocus_magnifications = []
                controller = TurretController.__new__(TurretController)
                controller.app = app
                controller.stage = Stage()
                controller.turret = Turret()
                controller.start_auto_focus_thread = lambda: (
                    autofocus_magnifications.append(app.magnification)
                )

                controller.change_objective(position)

                self.assertEqual(
                    autofocus_magnifications,
                    [expected_magnification],
                )
                self.assertEqual(app.magnification, expected_magnification)
                self.assertEqual(
                    app.frame_processor.auto_exposure_changes,
                    [True] if position == 5 else [],
                )

    def test_shutdown_helper_focuses_when_already_on_objective_one(self):
        class FrameProcessor:
            def __init__(self):
                self.default_exposure_calls = 0

            def set_default_exposure(self):
                self.default_exposure_calls += 1

        class App:
            def __init__(self):
                self.magnification = "100X"
                self.frame_processor = FrameProcessor()

            def set_magnification(self, magnification):
                self.magnification = magnification

        class Turret:
            @staticmethod
            def check_position():
                return 1

        app = App()
        autofocus_magnifications = []
        controller = TurretController.__new__(TurretController)
        controller.app = app
        controller.turret = Turret()
        controller.start_auto_focus_thread = lambda: (
            autofocus_magnifications.append(app.magnification)
        )

        controller.return_to_first_objective_and_focus()

        self.assertEqual(app.magnification, "2X")
        self.assertEqual(app.frame_processor.default_exposure_calls, 1)
        self.assertEqual(autofocus_magnifications, ["2X"])


if __name__ == "__main__":
    unittest.main()
