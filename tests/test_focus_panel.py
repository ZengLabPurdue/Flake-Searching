from pathlib import Path
import sys
import unittest


APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

from ui.panels.focus_panel import FocusPanel  # noqa: E402


class _FakeVariable:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _FakeApp:
    def __init__(self, hardware_available=True):
        self.hardware_available = hardware_available

    def hardware_controls_available(self):
        return self.hardware_available


class _FakeFocusController:
    def __init__(self):
        self.calls = []

    def start_auto_focus_thread(self, **kwargs):
        self.calls.append(kwargs)


class FocusPanelTests(unittest.TestCase):
    @staticmethod
    def _panel(
        range_value="Auto",
        velocity_value="Auto",
        acceleration_value="Auto",
        hardware_available=True,
    ):
        panel = FocusPanel.__new__(FocusPanel)
        panel.app = _FakeApp(hardware_available)
        panel.focus_controller = _FakeFocusController()
        panel.range_var = _FakeVariable(range_value)
        panel.velocity_var = _FakeVariable(velocity_value)
        panel.acceleration_var = _FakeVariable(acceleration_value)
        panel.sharpness_var = _FakeVariable("Sharpness: Unknown")
        return panel

    def test_default_entry_value_uses_auto_only_for_none(self):
        self.assertEqual(FocusPanel._default_entry_value(None), "Auto")
        self.assertEqual(FocusPanel._default_entry_value(60), "60")
        self.assertEqual(FocusPanel._default_entry_value(12.5), "12.5")

    def test_optional_positive_value_accepts_auto_and_positive_overrides(self):
        for value in ("", "  ", "Auto", "AUTO", "default", " Default "):
            with self.subTest(value=value):
                self.assertIsNone(
                    FocusPanel._optional_positive_value(value)
                )

        self.assertEqual(
            FocusPanel._optional_positive_value("12.5"),
            12.5,
        )
        self.assertEqual(
            FocusPanel._optional_positive_value("75", integer=True),
            75,
        )
        self.assertEqual(
            FocusPanel._optional_positive_value("75.0", integer=True),
            75,
        )

    def test_optional_positive_value_rejects_invalid_values(self):
        invalid_values = ("not-a-number", "0", "-1", "nan", "inf")
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    FocusPanel._optional_positive_value(value)

        with self.assertRaises(ValueError):
            FocusPanel._optional_positive_value("2.5", integer=True)

    def test_run_auto_focus_passes_none_for_objective_defaults(self):
        panel = self._panel(
            range_value="Auto",
            velocity_value=" default ",
            acceleration_value="",
        )

        panel.run_auto_focus()

        self.assertEqual(
            panel.focus_controller.calls,
            [{
                "focus_range": None,
                "z_velo": None,
                "z_accel": None,
            }],
        )
        self.assertEqual(panel.sharpness_var.value, "Sharpness: Unknown")

    def test_run_auto_focus_parses_custom_values(self):
        panel = self._panel(
            range_value="25.5",
            velocity_value="75.0",
            acceleration_value="6000",
        )

        panel.run_auto_focus()

        self.assertEqual(
            panel.focus_controller.calls,
            [{
                "focus_range": 25.5,
                "z_velo": 75,
                "z_accel": 6000,
            }],
        )

    def test_run_auto_focus_rejects_invalid_input_without_starting(self):
        panel = self._panel(velocity_value="2.5")

        panel.run_auto_focus()

        self.assertEqual(panel.focus_controller.calls, [])
        self.assertEqual(panel.sharpness_var.value, "Invalid focus values")

    def test_run_auto_focus_does_nothing_when_hardware_is_unavailable(self):
        panel = self._panel(
            range_value="not-a-number",
            hardware_available=False,
        )

        panel.run_auto_focus()

        self.assertEqual(panel.focus_controller.calls, [])
        self.assertEqual(panel.sharpness_var.value, "Sharpness: Unknown")


if __name__ == "__main__":
    unittest.main()
