from pathlib import Path
import sys
import unittest
from unittest.mock import patch


APP_DIR = Path(__file__).resolve().parents[1] / "App"
sys.path.insert(0, str(APP_DIR))

from UI.panels import objective_control_panel  # noqa: E402


class _FakeVariable:
    def __init__(self, value=None):
        self.value = value

    def set(self, value):
        self.value = value


class _FakeWidget:
    def __init__(self, *_args, **kwargs):
        self.command = kwargs.get("command")
        self.bindings = []

    def place(self, *_args, **_kwargs):
        return None

    def pack(self, *_args, **_kwargs):
        return None

    def grid(self, *_args, **_kwargs):
        return None

    def pack_propagate(self, *_args, **_kwargs):
        return None

    def rowconfigure(self, *_args, **_kwargs):
        return None

    def columnconfigure(self, *_args, **_kwargs):
        return None

    def bind(self, event, callback):
        self.bindings.append((event, callback))


class _FakeStyle:
    def configure(self, *_args, **_kwargs):
        return None


class ObjectiveControlPanelTests(unittest.TestCase):
    def test_objective_buttons_use_release_commands_without_press_bindings(self):
        class App:
            def __init__(self):
                self.buttons = []
                self.magnification = None

            def register_button(self, button):
                self.buttons.append(button)

            @staticmethod
            def hardware_controls_available():
                return True

            def set_magnification(self, magnification):
                self.magnification = magnification

        class TurretController:
            def __init__(self):
                self.turn_calls = []
                self.change_calls = []

            def turn_to_position(self, position):
                self.turn_calls.append(position)

            def change_objective(self, position):
                self.change_calls.append(position)

        app = App()
        turret = TurretController()
        with (
            patch.object(objective_control_panel.tk, "Frame", _FakeWidget),
            patch.object(objective_control_panel.tk, "Label", _FakeWidget),
            patch.object(
                objective_control_panel.tk,
                "StringVar",
                _FakeVariable,
            ),
            patch.object(objective_control_panel.ttk, "Style", _FakeStyle),
            patch.object(objective_control_panel.ttk, "Button", _FakeWidget),
        ):
            panel = objective_control_panel.ObjectiveControlPanel(
                parent=object(),
                app=app,
                stage=object(),
                turret_controller=turret,
            )

        self.assertEqual(turret.turn_calls, [1])
        self.assertEqual(len(app.buttons), 5)
        self.assertTrue(all(button.command for button in app.buttons))
        self.assertTrue(all(not button.bindings for button in app.buttons))

        panel.btn3.command()

        self.assertEqual(turret.change_calls, [3])
        self.assertEqual(panel.objective_var.value, "Objective: 3")


if __name__ == "__main__":
    unittest.main()
